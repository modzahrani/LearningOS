export type QuizQuestion = {
  id: string;
  text: string;
  options: string[];
};

export type QuizStartResponse = {
  quiz_id: string;
  question: QuizQuestion;
  question_number: number;
  total_questions: number;
  progress_percent: number;
};

export type QuizAnswerResponse = {
  quiz_id: string;
  correct: boolean;
  finished: boolean;
  score: number;
  question_number: number;
  total_questions: number;
  progress_percent: number;
  question?: QuizQuestion | null;
};
