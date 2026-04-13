import apiClient from './clientProvider';
import { ENDPOINTS } from '@/constants/endpoints';

// Types
export interface QuizQuestion {
  text: string;
  options: string[];
}

export interface QuizStartResponse {
  quiz_id: string;
  question: QuizQuestion;
  question_number: number;
  total_questions: number;
  progress_percent: number;
}

export interface QuizAnswerRequest {
  quiz_id: string;
  answer_index: number;
}

export interface QuizAnswerResponse {
  quiz_id: string;
  correct: boolean;
  finished: boolean;
  score: number;
  question_number: number;
  total_questions: number;
  progress_percent: number;
  explanation?: string | null;
  question?: QuizQuestion | null;
}

// ==================== QUIZ OPERATIONS ====================

// Start quiz
export const startQuiz = async () => {
  const res = await apiClient.post<QuizStartResponse>(ENDPOINTS.QUIZ.START);
  return res;
};

// Resume active quiz
export const resumeQuiz = async () => {
  const res = await apiClient.get<QuizStartResponse>(ENDPOINTS.QUIZ.RESUME);
  return res;
};

// Submit answer
export const answerQuiz = async (payload: QuizAnswerRequest) => {
  const res = await apiClient.post<QuizAnswerResponse>(ENDPOINTS.QUIZ.ANSWER, payload);
  return res;
};
