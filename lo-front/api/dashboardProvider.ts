import apiClient from './clientProvider';
import { ENDPOINTS } from '@/constants/endpoints';

export interface DashboardStat {
  title: string;
  value: string;
}

export interface DashboardCurrentLesson {
  source?: string | null;
  title: string;
  description: string;
  progress: number;
  completed_label: string;
  modules_label: string;
  status: string;
}

export interface DashboardDailyGoal {
  progress: number;
  message: string;
}

export interface DashboardRecommendation {
  source?: string | null;
  title: string;
  subtitle: string;
  duration: string;
  level: string;
}

export interface DashboardActivityItem {
  type: "quiz_completed" | "module_started" | "module_completed";
  title: string;
  subtitle: string;
}

export interface DashboardResponse {
  user: {
    name: string;
    learned_minutes_today: number;
  };
  stats: DashboardStat[];
  current_lesson: DashboardCurrentLesson;
  daily_goal: DashboardDailyGoal;
  recommendations: DashboardRecommendation[];
  recent_activity: DashboardActivityItem[];
}

export const getDashboard = async () => {
  const res = await apiClient.get<DashboardResponse>(ENDPOINTS.DASHBOARD.GET);
  return res;
};
