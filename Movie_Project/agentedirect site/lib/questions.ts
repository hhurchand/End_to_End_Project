export interface Option {
  id: string;
  label: string;
  hasInput?: boolean;
}

export interface Question {
  id: keyof import('./types').QuizAnswers;
  number: number;
  text: string;
  subtext?: string;
  options: Option[];
}

export const QUESTIONS: Question[] = [
  {
    id: 'websiteStatus',
    number: 1,
    text: 'What best describes your current website situation?',
    options: [
      { id: 'none',       label: "I don't have a website yet." },
      { id: 'outdated',   label: 'I have a website, but it looks outdated.' },
      { id: 'no-leads',   label: "I have a website, but it doesn't bring enough inquiries." },
      { id: 'new-look',   label: 'I want a completely new look for my business.' },
      { id: 'exploring',  label: "I'm exploring my options." },
    ],
  },
  {
    id: 'businessType',
    number: 2,
    text: 'What type of business do you run?',
    options: [
      { id: 'home-services',  label: 'Home services or contracting.' },
      { id: 'automotive',     label: 'Automotive.' },
      { id: 'restaurant',     label: 'Restaurant or hospitality.' },
      { id: 'professional',   label: 'Professional services.' },
      { id: 'retail',         label: 'Retail or e-commerce.' },
      { id: 'other',          label: 'Other.', hasInput: true },
    ],
  },
  {
    id: 'goal',
    number: 3,
    text: 'What would you most like your new website to achieve?',
    options: [
      { id: 'professional',   label: 'Make my business look more professional.' },
      { id: 'leads',          label: 'Get more calls and quote requests.' },
      { id: 'local',          label: 'Help local customers find my business online.' },
      { id: 'showcase',       label: 'Showcase my services and previous work.' },
      { id: 'replace',        label: 'Replace an existing website that no longer works.' },
      { id: 'combo',          label: 'A combination of these.' },
    ],
  },
  {
    id: 'timeline',
    number: 4,
    text: 'When are you looking to make a change?',
    options: [
      { id: 'asap',    label: 'As soon as possible.' },
      { id: '1mo',     label: 'Within the next month.' },
      { id: '3mo',     label: 'Within the next three months.' },
      { id: 'research',label: 'Just researching for now.' },
    ],
  },
];

export const TOTAL_STEPS = QUESTIONS.length + 1; // +1 for contact step
