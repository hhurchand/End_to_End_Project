export type SplashPhase =
  | 'intro'       // envelope visible, idle
  | 'opening'     // flap animating
  | 'quiz'        // questionnaire active
  | 'submitting'  // form being sent
  | 'done'        // success screen
  | 'exiting';    // transitioning to main site

export interface Answer {
  questionId: string;
  value: string | string[];
}

export interface QuizAnswers {
  websiteStatus?: string;
  businessType?: string;
  businessTypeOther?: string;
  goal?: string;
  timeline?: string;
  name?: string;
  businessName?: string;
  email?: string;
  phone?: string;
  currentUrl?: string;
  details?: string;
  consent?: boolean;
}

export type SubmitPayload = QuizAnswers;
