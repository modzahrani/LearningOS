import apiClient from "./clientProvider";
import { ENDPOINTS } from "@/constants/endpoints";
import type { AssignedLesson } from "@/api/quizProvider";

export interface AssignedLessonsResponse {
  role: string;
  level: number;
  lessons: AssignedLesson[];
}

export interface CompleteLessonRequest {
  source: string;
  status?: "assigned" | "in_progress" | "completed";
  quiz_score?: number;
  total_questions?: number;
}

export interface LessonStoryboardSlide {
  title: string;
  narrative: string;
  bullets: string[];
  scene_caption?: string | null;
  dialogue_line?: string | null;
  illustration_prompt: string;
  speaker_note?: string | null;
  checkpoint_question?: string | null;
}

export interface LessonQuizQuestion {
  question: string;
  options: string[];
  correct_index: number;
  explanation: string;
}

export interface LessonNextRecommendation {
  source: string;
  title: string;
  reason: string;
  level: number;
}

export interface LessonStateResponse {
  source: string;
  active_slide: number;
  phase: "story" | "quiz" | "next";
  quiz_answers: Record<string, number>;
  quiz_submitted: boolean;
}

export interface SaveLessonStateRequest {
  source: string;
  active_slide: number;
  phase: "story" | "quiz" | "next";
  quiz_answers: Record<string, number>;
  quiz_submitted: boolean;
}

export interface LessonDetailResponse {
  source: string;
  topic: string;
  role: string;
  level: number;
  status: "assigned" | "in_progress" | "completed";
  estimated_minutes: number;
  progress: number;
  overview: string;
  learning_objectives: string[];
  slides: LessonStoryboardSlide[];
  end_quiz: LessonQuizQuestion[];
  next_lesson?: LessonNextRecommendation | null;
}

export const getAssignedLessons = async () => {
  const res = await apiClient.get<AssignedLessonsResponse>(ENDPOINTS.LESSONS.ASSIGNED);
  return res;
};

export const getLessonDetail = async (source: string) => {
  const res = await apiClient.get<LessonDetailResponse>(ENDPOINTS.LESSONS.DETAIL, {
    params: { source },
  });
  return res;
};

export const getLessonState = async (source: string) => {
  const res = await apiClient.get<LessonStateResponse>(ENDPOINTS.LESSONS.STATE, {
    params: { source },
  });
  return res;
};

export const saveLessonState = async (payload: SaveLessonStateRequest) => {
  const res = await apiClient.post<LessonStateResponse>(
    ENDPOINTS.LESSONS.STATE,
    payload
  );
  return res;
};

export const completeLesson = async (payload: CompleteLessonRequest) => {
  const res = await apiClient.post<AssignedLessonsResponse>(
    ENDPOINTS.LESSONS.COMPLETE,
    payload
  );
  return res;
};
