require("dotenv").config();
const express = require("express");

const app = express();

// Log every inbound request immediately — before body parsing
app.use((req, res, next) => {
  process.stdout.write(`>> ${req.method} ${req.path} [${new Date().toISOString()}]\n`);
  next();
});

app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));

// ─── Session booking guard ─────────────────────────────────────────────────────
// Tracks call IDs that have already had a successful booking.
// Prevents GPT-4o from retrying book_appointment after BOOKING_SUCCESS.
const bookedCallIds = new Set();

const { analyzeIntent, detectLanguage } = require("./intent");
const {
  applyContactTags,
  findContactByPhone,
  cancelDemo,
  rescheduleDemo,
  getUpcomingAppointment,
  createDemo,
  formatDateTime,
} = require("./index");
const axios = require("axios");

const ghl = axios.create({
  baseURL: "https://services.leadconnectorhq.com",
  headers: {
    Authorization: `Bearer ${process.env.GHL_API_KEY}`,
    "Content-Type": "application/json",
    Version: "2021-07-28",
  },
});

const LOCATION_ID = process.env.GHL_LOCATION_ID;

const applyTagToContact = (contactId, tag) => applyContactTags(contactId, [tag]);

const PORT = process.env.PORT || 3000;

// ─── POST /analyze-intent ─────────────────────────────────────────────────────

app.post("/analyze-intent", async (req, res) => {
  console.log("🔥 WEBHOOK HIT");

  // GHL sends SMS text in `body`, contact text via `message` may be an object
  const message = req.body.body || req.body.message;
  const contactId = req.body.contactId;

  if (!message || typeof message !== "string") {
    console.log("⚠️  No valid message string found in payload");
    return res.json({ success: true });
  }

  console.log("Message:", message);

  const { intent } = await analyzeIntent(message);

  let tag = "intent_general";

  if (intent === "book_job") tag = "intent_book_job";
  if (intent === "book_estimate") tag = "intent_book_estimate";
  if (intent === "reschedule") tag = "intent_reschedule";
  if (intent === "cancel") tag = "intent_cancel";

  console.log("Intent:", intent);
  console.log("Tag:", tag);

  // 👇 THIS IS THE KEY LINE
  await applyTagToContact(contactId, tag);

  res.json({ success: true });
});

// ─── GET /audit-workflows ─────────────────────────────────────────────────────

app.get("/audit-workflows", async (req, res) => {
  console.log("\n[/audit-workflows] Fetching workflows from GHL...");

  const response = await ghl.get("/workflows/", {
    params: { locationId: LOCATION_ID },
  });

  const workflows = response.data.workflows || [];
  console.log(`  Found ${workflows.length} workflows`);

  const report = workflows.map((wf) => {
    const steps = (wf.actions || []).map((action, i) => ({
      step: i + 1,
      type: action.type,
      name: action.name || action.type,
    }));

    return {
      id:      wf.id,
      name:    wf.name,
      status:  wf.status,
      trigger: wf.triggers?.[0]?.type || "unknown",
      steps,
      stepCount: steps.length,
    };
  });

  // Console audit dump
  console.log("\n══════════════════════════════════════════");
  console.log("  GHL WORKFLOW AUDIT");
  console.log("══════════════════════════════════════════");
  for (const wf of report) {
    console.log(`\n  📋 ${wf.name}`);
    console.log(`     Status  : ${wf.status}`);
    console.log(`     Trigger : ${wf.trigger}`);
    console.log(`     Steps   : ${wf.stepCount}`);
    for (const s of wf.steps) {
      console.log(`       ${s.step}. [${s.type}] ${s.name}`);
    }
  }
  console.log("\n══════════════════════════════════════════\n");

  res.json({ total: report.length, workflows: report });
});

// ─── POST /cancel-appointment ─────────────────────────────────────────────────
// Called by GHL voice agent when caller says they want to cancel.
// Body: { phone: "+15141234567" }

app.post("/cancel-appointment", async (req, res) => {
  console.log("\n[/cancel-appointment] BODY:", JSON.stringify(req.body, null, 2));

  const rawPhone = req.query.phone || req.body.phone;

  if (!rawPhone) {
    console.log("⚠️  No phone number provided");
    return res.json({ message: "I'm sorry, I wasn't able to find your phone number. Can you repeat it for me?" });
  }

  let phone = rawPhone.replace(/\D/g, "");
  if (phone.length === 10) phone = "1" + phone;
  if (phone.length > 11) phone = phone.slice(-11);
  console.log("Incoming phone:", phone);

  const contact = await findContactByPhone(phone, "client1");
  if (!contact) {
    console.log("⚠️  No contact found for phone:", phone);
    return res.json({ message: "I wasn't able to find an appointment linked to that number. Could you double-check the phone number you used to book?" });
  }

  const appointment = await getUpcomingAppointment(contact.id, "client1");
  if (!appointment) {
    console.log("⚠️  No upcoming appointment for contact:", contact.id);
    return res.json({ message: "I don't see any upcoming appointments for your number. If you think this is a mistake, please call us back and we'll sort it out." });
  }

  const appointmentTime = formatDateTime(appointment.startTime);
  await cancelDemo(contact.id, "client1");

  console.log(`✅ Cancelled appointment for ${contact.firstName} at ${appointmentTime}`);
  return res.json({
    message: `Done! Your appointment on ${appointmentTime} has been cancelled. If you'd like to rebook, we're happy to help. Is there anything else I can do for you?`,
  });
});

// ─── POST /reschedule-appointment ─────────────────────────────────────────────
// Called by GHL voice agent when caller wants to reschedule.
// Body: { phone: "+15141234567", newDateTime: "2026-04-20T14:00:00-04:00" }

app.post("/reschedule-appointment", async (req, res) => {
  try {
    console.log("=== RESCHEDULE HIT ===");

    // GHL sends variables as query params, not body
    let phone = (req.query.phone || req.body.phone || "").replace(/\D/g, "");
    if (phone.length === 10) phone = "1" + phone;
    if (phone.length > 11) phone = phone.slice(-11);
    const newDateTime = decodeURIComponent(req.query.newDateTime || req.body.newDateTime || "");

    if (!phone || !newDateTime) {
      console.log("⚠️  Missing phone or newDateTime:", { phone, newDateTime });
      return res.status(400).json({
        message: "I'm missing the phone number or new appointment time."
      });
    }

    const contact = await findContactByPhone(phone, "client1");

    if (!contact || !contact.id) {
      return res.status(404).json({
        message: "I couldn't find a contact with that phone number."
      });
    }

    const contactId = contact.id;

    console.log("Contact ID:", contactId);

    const parsedDate = new Date(newDateTime);
    if (isNaN(parsedDate.getTime())) {
      console.log("⚠️  Invalid datetime value:", newDateTime);
      return res.status(400).json({
        message: "The date and time I received wasn't valid. Please try again."
      });
    }

    const isoTime = parsedDate.toISOString();

    const result = await rescheduleDemo(contactId, isoTime, "client1");

    return res.json({
      message: "You're all set. Your appointment has been rescheduled."
    });

  } catch (err) {
    console.error("ERROR:", err);

    return res.json({
      message: "Something went wrong while rescheduling."
    });
  }
});

// ─── POST /sms-reschedule ─────────────────────────────────────────────────────
// Called by GHL workflow when a contact replies with their new preferred time.
// GHL sends the raw SMS text + phone number.
// Claude parses the date/time from natural language, then reschedules.
// Body/query: { phone, message }
// Returns: { message } — the SMS text GHL should send back to the customer.

app.post("/sms-reschedule", async (req, res) => {
  try {
    console.log("\n=== SMS RESCHEDULE HIT ===");
    console.log("QUERY:", JSON.stringify(req.query, null, 2));

    const rawPhone = req.query.phone || req.body.phone || "";
    const message  = req.query.message || req.body.body || req.body.message || "";

    let phone = rawPhone.replace(/\D/g, "");
    if (phone.length === 10) phone = "1" + phone;
    if (phone.length > 11)   phone = phone.slice(-11);

    if (!phone || !message) {
      console.log("⚠️  Missing phone or message");
      return res.json({ message: "Sorry, I didn't get enough info to reschedule. Please reply with your preferred date and time." });
    }

    console.log("Phone  :", phone);
    console.log("Message:", message);

    // Ask Claude to extract a date/time from the customer's natural language reply
    const Anthropic = require("@anthropic-ai/sdk");
    const claude = new Anthropic();

    const today = new Date().toISOString().split("T")[0];

    const extraction = await claude.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 128,
      system: `You extract appointment date/time from a customer SMS message.
Today's date is ${today}. The timezone is America/Toronto (EDT, UTC-4).
Return ONLY a JSON object: { "isoDateTime": "YYYY-MM-DDTHH:mm:ss-04:00" } or { "isoDateTime": null } if no clear date/time found.
No explanation. No extra text.`,
      messages: [{ role: "user", content: message }],
    });

    const rawText = extraction.content[0].text;
    console.log("Claude raw response:", rawText);

    let isoDateTime = null;
    try {
      const cleaned = rawText.replace(/```json|```/g, "").trim();
      const parsed = JSON.parse(cleaned);
      isoDateTime = parsed.isoDateTime || null;
    } catch {
      isoDateTime = null;
    }

    console.log("Extracted dateTime:", isoDateTime);

    if (!isoDateTime || isNaN(new Date(isoDateTime).getTime())) {
      return res.json({
        message: "I couldn't quite understand the date and time. Could you reply with something like \"April 20 at 2pm\"?"
      });
    }

    const contact = await findContactByPhone(phone, "client1");
    if (!contact) {
      console.log("⚠️  No contact found for phone:", phone);
      return res.json({ message: "I couldn't find your appointment. Please call us and we'll sort it out." });
    }

    const appointment = await getUpcomingAppointment(contact.id, "client1");
    if (!appointment) {
      console.log("⚠️  No upcoming appointment for contact:", contact.id);
      return res.json({ message: "I don't see any upcoming appointments to reschedule. Feel free to call us to book a new one." });
    }

    await rescheduleDemo(contact.id, isoDateTime, "client1");

    const newTime = formatDateTime(isoDateTime);
    console.log(`✅ Rescheduled for ${contact.firstName} to ${newTime}`);

    return res.json({
      message: `Done! Your appointment has been rescheduled to ${newTime}. See you then!`
    });

  } catch (err) {
    console.error("SMS RESCHEDULE ERROR:", err.message);
    return res.json({ message: "Something went wrong while rescheduling. Please call us and we'll get it sorted." });
  }
});

// ─── POST /sms-book-appointment ───────────────────────────────────────────────
// Called by GHL workflow when a contact wants to book via SMS.
// Claude parses date/time from natural language, checks for duplicates, creates appointment.
// Body/query: { phone, message }
// Returns: { message } — SMS text GHL sends back to the customer.

app.post("/sms-book-appointment", async (req, res) => {
  try {
    console.log("\n=== SMS BOOK APPOINTMENT HIT ===");
    console.log("QUERY:", JSON.stringify(req.query, null, 2));

    const rawPhone = req.query.phone || req.body.phone || "";
    const message  = req.query.message || req.body.body || req.body.message || "";

    let phone = rawPhone.replace(/\D/g, "");
    if (phone.length === 10) phone = "1" + phone;
    if (phone.length > 11)   phone = phone.slice(-11);

    if (!phone || !message) {
      return res.json({ message: "Sorry, I didn't get enough info to book. Please reply with your preferred date and time." });
    }

    console.log("Phone  :", phone);
    console.log("Message:", message);

    const lang = detectLanguage(message);

    // Use Claude to extract date/time from natural language
    const Anthropic = require("@anthropic-ai/sdk");
    const claude = new Anthropic();
    const today = new Date().toISOString().split("T")[0];

    const extraction = await claude.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 128,
      system: `You extract appointment date/time from a customer SMS message.
Today's date is ${today}. Timezone is America/Toronto (EDT, UTC-4).
Return ONLY a JSON object: { "isoDateTime": "YYYY-MM-DDTHH:mm:ss-04:00" } or { "isoDateTime": null } if no clear date/time found.
No explanation. No extra text.`,
      messages: [{ role: "user", content: message }],
    });

    const rawText = extraction.content[0].text;
    console.log("Claude raw response:", rawText);

    let isoDateTime = null;
    try {
      const cleaned = rawText.replace(/```json|```/g, "").trim();
      isoDateTime = JSON.parse(cleaned).isoDateTime || null;
    } catch { isoDateTime = null; }

    console.log("Extracted dateTime:", isoDateTime);

    if (!isoDateTime || isNaN(new Date(isoDateTime).getTime())) {
      return res.json({
        message: lang === "french"
          ? "Je n'ai pas bien compris la date et l'heure. Pouvez-vous répondre avec quelque chose comme \"20 avril à 14h\"?"
          : "I couldn't understand the date and time. Could you reply with something like \"April 20 at 2pm\"?"
      });
    }

    // Find contact
    const contact = await findContactByPhone(phone, "client1");
    if (!contact) {
      console.log("⚠️  No contact found for phone:", phone);
      return res.json({
        message: lang === "french"
          ? "Je n'ai pas trouvé votre profil. Appelez-nous et on va arranger ça."
          : "I couldn't find your profile. Please call us and we'll get it sorted."
      });
    }

    // Check for duplicate — block only if existing appointment is within 1 hour of requested time
    const existing = await getUpcomingAppointment(contact.id, "client1");
    if (existing) {
      const existingTime = formatDateTime(existing.startTime);
      const diffMs = Math.abs(new Date(existing.startTime) - new Date(isoDateTime));
      const diffHours = diffMs / (1000 * 60 * 60);
      console.log(`Existing appointment at ${existingTime}, diff: ${diffHours.toFixed(2)}h`);
      if (diffHours < 1) {
        return res.json({
          message: lang === "french"
            ? `Vous avez déjà un rendez-vous prévu le ${existingTime}. Voulez-vous le reprogrammer à la place?`
            : `You already have an appointment on ${existingTime}. Would you like to reschedule that instead?`
        });
      }
    }

    // Create the appointment
    const contactName = [contact.firstName, contact.lastName].filter(Boolean).join(" ") || "Client";
    await createDemo(contact.id, isoDateTime, contactName, "client1");

    const bookedTime = formatDateTime(isoDateTime);
    console.log(`✅ Booked appointment for ${contactName} at ${bookedTime}`);

    return res.json({
      message: lang === "french"
        ? `Parfait! Votre rendez-vous est confirmé pour le ${bookedTime}. À bientôt!`
        : `Done! Your appointment is confirmed for ${bookedTime}. See you then!`
    });

  } catch (err) {
    console.error("SMS BOOK ERROR:", err.message);
    return res.json({ message: "Something went wrong. Please call us to book your appointment." });
  }
});

// ─── POST /log-caller-info ────────────────────────────────────────────────────
// Called by VAPI tool at end of call to save caller info to GHL.
// Body: { name, phone, service, language, address, urgency }

app.post("/log-caller-info", async (req, res) => {
  const t0 = Date.now();
  try {
    console.log("\n=== LOG CALLER INFO ===");

    // VAPI wraps tool call args in message.toolCalls or functionCall
    const vapiArgs = req.body?.message?.toolCalls?.[0]?.function?.arguments
      || req.body?.message?.functionCall?.parameters
      || req.body;

    const args = typeof vapiArgs === "string" ? JSON.parse(vapiArgs) : vapiArgs;
    console.log("Parsed args:", JSON.stringify(args));

    const { name, phone, service, language, address, urgency } = args;

    if (!name || !phone) {
      return res.json({ result: "Missing name or phone number." });
    }

    // Upsert contact
    const ghlClient = axios.create({
      baseURL: "https://services.leadconnectorhq.com",
      headers: {
        Authorization: `Bearer ${process.env.GHL_API_KEY_CLIENT1}`,
        "Content-Type": "application/json",
        Version: "2021-07-28",
      },
    });

    const locationId = process.env.GHL_LOCATION_ID_CLIENT1;

    const [firstName, ...rest] = name.trim().split(" ");
    const lastName = rest.join(" ");

    const customFields = [
      language ? { id: "58kmzQPUxUjlJcKQhlJd", value: language } : null,
      service  ? { id: "GDQeXTgRHwRr3lVfg8aO", value: service  } : null,
      urgency  ? { id: "ylisHM7fN0nBk99sOsD3", value: urgency  } : null,
    ].filter(Boolean);

    const upsertPayload = {
      locationId,
      firstName,
      lastName,
      phone,
      ...(address ? { address1: address } : {}),
      ...(customFields.length ? { customFields } : {}),
    };

    console.log(`[log-caller] +${Date.now()-t0}ms — upserting`);
    const response = await ghlClient.post("/contacts/upsert", upsertPayload);
    const contact = response.data.contact;

    console.log(`✅ Contact saved: ${contact.id} (+${Date.now()-t0}ms)`);

    const isEmergency = (urgency || "").toLowerCase().includes("emergency");
    const isBooked    = args.booked === true || args.booked === "true";
    const tag        = isEmergency ? "emergency" : isBooked ? "job_booked" : "new_lead";
    const workflowId = isEmergency
      ? "97d59ad9-9e52-43bf-8cd1-0f7bae2ee2f4"
      : isBooked ? null
      : "30d64865-45df-4722-96da-b5dd739a0560";

    const responseMsg = isEmergency
      ? `Got it, ${firstName}. I'm flagging this as an emergency — our team will reach out right away.`
      : `Got it. I've saved your information, ${firstName}. Someone from the team will be in touch shortly.`;

    // Respond immediately — don't block on tag/workflow
    console.log(`[log-caller] +${Date.now()-t0}ms — sending response`);
    res.json({ result: responseMsg });

    // Fire-and-forget tag + workflow in background
    console.log(`[log-caller-info] tag: "${tag}", workflow: ${workflowId}`);
    ghlClient.post(`/contacts/${contact.id}/tags`, { tags: [tag] })
      .then(() => console.log(`✅ Tag applied: ${tag}`))
      .catch(e => console.error("Tag error:", e.response?.data || e.message));

    if (workflowId) {
      const now = new Date();
      const pad = (n) => String(n).padStart(2, "0");
      const localISO = `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())}T${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}-04:00`;
      ghlClient.post(`/contacts/${contact.id}/workflow/${workflowId}`, { eventStartTime: localISO })
        .then(() => console.log(`✅ Enrolled in workflow: ${workflowId}`))
        .catch(e => console.error("Workflow error:", e.response?.data || e.message));
    }

  } catch (err) {
    console.error("LOG CALLER INFO ERROR:", err.response?.data || err.message);
    return res.json({ result: "I had trouble saving your info, but someone from the team will follow up shortly." });
  }
});

// ─── POST /get-appointment ────────────────────────────────────────────────────
// Called by VAPI tool when caller asks about their existing appointment.
// Args: { phone }

app.post("/get-appointment", async (req, res) => {
  try {
    const vapiArgs = req.body?.message?.toolCalls?.[0]?.function?.arguments
      || req.body?.message?.functionCall?.parameters
      || req.body;
    const args = typeof vapiArgs === "string" ? JSON.parse(vapiArgs) : vapiArgs;

    const FR_DIGITS = {"zéro":"0","zero":"0","un":"1","une":"1","deux":"2","trois":"3","quatre":"4","cinq":"5","six":"6","sept":"7","huit":"8","neuf":"9"};
    let phone = (args.phone || "").toLowerCase()
      .replace(/\b(zéro|zero|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf)\b/g, m => FR_DIGITS[m])
      .replace(/\D/g, "");
    if (phone.length === 10) phone = "1" + phone;
    if (phone.length > 11)   phone = phone.slice(-11);

    if (!phone) return res.json({ result: "I need your phone number to look up the appointment." });

    const contact = await findContactByPhone(phone, "client1");
    if (!contact) return res.json({ result: "I couldn't find any account linked to that number. Could you double-check the phone number you used to book?" });

    const appt = await getUpcomingAppointment(contact.id, "client1");
    if (!appt) return res.json({ result: "I don't see any upcoming appointments for that number. Feel free to book a new one." });

    const apptTime = formatDateTime(appt.startTime);
    return res.json({ result: `Your appointment is scheduled for ${apptTime}.` });

  } catch (err) {
    console.error("GET APPOINTMENT ERROR:", err.message);
    return res.json({ result: "I had trouble looking that up. The team will be able to confirm your appointment time." });
  }
});

// ─── POST /book-appointment ───────────────────────────────────────────────────
// Called by VAPI tool when caller wants to book directly during a call.
// Body (via VAPI toolCalls): { phone, dateTime, service }
// dateTime: ISO 8601 string like "2026-04-30T10:00:00-04:00"

app.post("/book-appointment", async (req, res) => {
  const t0 = Date.now();
  try {
    console.log("\n=== BOOK APPOINTMENT (VAPI) ===");

    // Parse VAPI tool call args (same structure as log-caller-info)
    const vapiArgs = req.body?.message?.toolCalls?.[0]?.function?.arguments
      || req.body?.message?.functionCall?.parameters
      || req.body;

    const args = typeof vapiArgs === "string" ? JSON.parse(vapiArgs) : vapiArgs;
    console.log("Parsed args:", JSON.stringify(args, null, 2));

    // Session guard — if this call already had a successful booking, stop immediately
    const callId = req.body?.message?.call?.id;
    if (callId && bookedCallIds.has(callId)) {
      console.log(`[book-appointment] ⚠️  Call ${callId} already booked — blocking retry`);
      return res.json({ result: "BOOKING_ALREADY_CONFIRMED. The appointment for this caller was already booked earlier in this call. Do NOT call book_appointment again. Tell the caller their appointment is confirmed and end the call." });
    }

    let { phone, dateTime, service, name, language, address } = args;

    if (!phone || !dateTime) {
      return res.json({ result: "I need both a phone number and a date and time to book the appointment." });
    }

    // Normalize phone — convert French number words to digits as a safety net
    const FR_DIGITS = {
      "zéro":"0","zero":"0","un":"1","une":"1","deux":"2","trois":"3",
      "quatre":"4","cinq":"5","six":"6","sept":"7","huit":"8","neuf":"9"
    };
    phone = phone.toLowerCase().replace(/\b(zéro|zero|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf)\b/g, m => FR_DIGITS[m]);
    phone = phone.replace(/\D/g, "");
    if (phone.length === 10) phone = "1" + phone;
    if (phone.length > 11)   phone = phone.slice(-11);

    // Validate dateTime
    const parsedDate = new Date(dateTime);
    if (isNaN(parsedDate.getTime())) {
      return res.json({ result: "The date and time I received wasn't valid. Can you repeat it?" });
    }

    const bookingGhlClient = axios.create({
      baseURL: "https://services.leadconnectorhq.com",
      headers: {
        Authorization: `Bearer ${process.env.GHL_API_KEY_CLIENT1}`,
        "Content-Type": "application/json",
        Version: "2021-07-28",
      },
    });

    // Upsert contact — one call, always works for new or existing
    let contact = null;
    const [firstName, ...rest] = (name || "Client").trim().split(" ");
    const customFields = [
      language ? { id: "58kmzQPUxUjlJcKQhlJd", value: language } : null,
      service  ? { id: "GDQeXTgRHwRr3lVfg8aO", value: service  } : null,
    ].filter(Boolean);

    console.log(`[book-appt] +${Date.now()-t0}ms — starting upsert`);
    const upsertRes = await bookingGhlClient.post("/contacts/upsert", {
      locationId: process.env.GHL_LOCATION_ID_CLIENT1,
      firstName,
      lastName: rest.join(" "),
      phone: `+${phone}`,
      ...(address ? { address1: address } : {}),
      ...(customFields.length ? { customFields } : {}),
    });
    contact = upsertRes.data.contact;
    console.log(`✅ Contact upserted: ${contact.id} (+${Date.now()-t0}ms)`);

    // Book — GHL will reject if slot is unavailable
    const contactName = [contact.firstName, contact.lastName].filter(Boolean).join(" ") || "Client";
    console.log(`[book-appt] +${Date.now()-t0}ms — starting createDemo`);
    await createDemo(contact.id, parsedDate.toISOString(), contactName, "client1");

    const bookedTime = formatDateTime(parsedDate.toISOString());
    console.log(`✅ Booked appointment for ${contactName} at ${bookedTime} (+${Date.now()-t0}ms)`);

    // Mark this call as booked — fire-and-forget tag, respond immediately
    if (callId) bookedCallIds.add(callId);
    bookingGhlClient.post(`/contacts/${contact.id}/tags`, { tags: ["job_booked"] })
      .then(() => console.log(`✅ Tag applied: job_booked`))
      .catch(e => console.error("Tag error:", e.response?.data || e.message));

    console.log(`[book-appt] +${Date.now()-t0}ms — sending BOOKING_SUCCESS`);
    return res.json({ result: `BOOKING_SUCCESS. Appointment confirmed for ${bookedTime}. Tell the caller their appointment is booked for ${bookedTime}. Then call log_caller_info with booked: true. Then end the call. Do NOT call book_appointment again.` });

  } catch (err) {
    const errMsg = err.response?.data?.message || err.message || "";
    console.error("BOOK APPOINTMENT ERROR:", err.response?.data || err.message);

    // If slot is "unavailable" AND we already have the contact, check if they
    // already have an upcoming appointment (means previous attempt booked it but
    // VAPI never received the BOOKING_SUCCESS response).
    if (contact && (errMsg.toLowerCase().includes("no longer available") || errMsg.toLowerCase().includes("slot"))) {
      try {
        const existing = await getUpcomingAppointment(contact.id, "client1");
        if (existing) {
          const bookedTime = formatDateTime(existing.startTime);
          console.log(`[book-appt] Slot unavailable but contact already has appt at ${bookedTime} — ALREADY_CONFIRMED`);
          if (callId) bookedCallIds.add(callId);
          return res.json({ result: `BOOKING_ALREADY_CONFIRMED. The appointment is already booked for ${bookedTime}. Tell the caller their appointment is confirmed for ${bookedTime}. Call log_caller_info with booked: true, then end the call.` });
        }
      } catch (checkErr) {
        console.error("[book-appt] Check existing appt error:", checkErr.message);
      }
    }

    return res.json({ result: "BOOKING_FAILED_SLOT_UNAVAILABLE. That time slot is not available. Ask the caller for a different date or time and try again." });
  }
});

// ─── POST /call-ended ─────────────────────────────────────────────────────────
// VAPI end-of-call webhook. Saves whatever was collected even if caller hung up
// before log_caller_info was triggered.

app.post("/call-ended", async (req, res) => {
  try {
    const body = req.body;
    const type = body?.message?.type;
    if (type !== "end-of-call-report") return res.json({ received: true });

    const transcript = body?.message?.transcript || "";
    const toolCallResults = body?.message?.toolCallResults || [];
    const endedReason = body?.message?.endedReason || "unknown";

    console.log("\n=== CALL ENDED WEBHOOK ===");
    console.log("Ended reason:", endedReason);
    console.log("Tool calls made:", toolCallResults.length);

    // If log_caller_info already ran successfully, nothing to do
    const alreadyLogged = toolCallResults.some(
      (r) => r?.name === "log_caller_info" && !r?.error
    );
    if (alreadyLogged) {
      console.log("[call-ended] log_caller_info already ran — skipping");
      return res.json({ received: true });
    }

    // Try to extract phone from transcript using regex
    const phoneMatch = transcript.replace(/\s/g, "").match(/(\d{10})/);
    const phone = phoneMatch ? phoneMatch[1] : null;

    if (!phone) {
      console.log("[call-ended] No phone found in transcript — skipping");
      return res.json({ received: true });
    }

    console.log("[call-ended] Saving fallback contact for phone:", phone);

    const axios = require("axios");
    const ghlClient = axios.create({
      baseURL: "https://services.leadconnectorhq.com",
      headers: {
        Authorization: `Bearer ${process.env.GHL_API_KEY_CLIENT1}`,
        "Content-Type": "application/json",
        Version: "2021-07-28",
      },
    });

    const upsertRes = await ghlClient.post("/contacts/upsert", {
      locationId: process.env.GHL_LOCATION_ID_CLIENT1,
      phone: `+1${phone}`,
    });
    const contact = upsertRes.data.contact;

    await ghlClient.post(`/contacts/${contact.id}/tags`, { tags: ["incomplete_call"] });
    console.log(`✅ Fallback contact saved: ${contact.id} — tagged incomplete_call`);

    return res.json({ received: true });
  } catch (err) {
    console.error("[call-ended] Error:", err.response?.data || err.message);
    return res.json({ received: true });
  }
});

// ─── Start ────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`\n  GHL Intent Server running on port ${PORT}`);
  console.log(`  POST http://localhost:${PORT}/analyze-intent\n`);
});
