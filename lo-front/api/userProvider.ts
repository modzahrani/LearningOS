import apiClient from './clientProvider';
import { ENDPOINTS } from '@/constants/endpoints';

// Types
export interface RegisterData {
  email: string;
  password: string;
  first_name: string;
  last_name: string;
  agree_terms: boolean;
}

export interface LoginData {
  email: string;
  password: string;
}

export type OAuthProvider = "google";

export interface ForgotPasswordData {
  email: string;
  redirect_to: string;
}

export interface ResetPasswordData {
  access_token: string;
  new_password: string;
}

export interface UpdateProfileData {
  name: string;
}

export interface LoginResponse {
  detail: string;
  user: {
    id: string;
    email: string;
    first_name?: string | null;
    last_name?: string | null;
  };
}

export interface OAuthStartResponse {
  url: string;
}

export interface MessageResponse {
  detail: string;
}

export interface ProfileStats {
  lessons_completed: number;
  lessons_in_progress: number;
  lessons_assigned: number;
  average_quiz_score: number;
  last_quiz_score: number;
}

export interface ProfileLessonSummary {
  source?: string | null;
  title: string;
  status: "assigned" | "in_progress" | "completed";
  progress: number;
}

export interface ProfileResponse {
  id: string;
  email: string;
  first_name: string;
  last_name: string;
  full_name: string;
  role: "student" | "individual" | "enterprise" | string;
  difficulty: string;
  current_level: number;
  joined_at?: string | null;
  stats: ProfileStats;
  current_lesson?: ProfileLessonSummary | null;
}

// ==================== AUTH OPERATIONS ====================

// Register a new user
export const register = async (userData: RegisterData) => {
  const res = await apiClient.post(ENDPOINTS.AUTH.REGISTER, userData);
  return res;
};

// Login user
export const login = async (credentials: LoginData) => {
  const res = await apiClient.post<LoginResponse>(ENDPOINTS.AUTH.LOGIN, credentials);
  return res;
};

export const logout = async () => {
  const res = await apiClient.post<MessageResponse>(ENDPOINTS.AUTH.LOGOUT);
  return res;
};

export const startOAuthLogin = async (
  provider: OAuthProvider,
  redirect_to: string
) => {
  const res = await apiClient.post<OAuthStartResponse>(
    ENDPOINTS.AUTH.OAUTH_START,
    { provider, redirect_to }
  );
  return res;
};

export const createOAuthSession = async (access_token: string) => {
  const res = await apiClient.post<LoginResponse>(
    ENDPOINTS.AUTH.OAUTH_SESSION,
    { access_token }
  );
  return res;
};

export const forgotPassword = async (payload: ForgotPasswordData) => {
  const res = await apiClient.post<MessageResponse>(
    ENDPOINTS.AUTH.FORGOT_PASSWORD,
    payload
  );
  return res;
};

export const resetPassword = async (payload: ResetPasswordData) => {
  const res = await apiClient.post<MessageResponse>(
    ENDPOINTS.AUTH.RESET_PASSWORD,
    payload
  );
  return res;
};

export const getProfile = async () => {
  const res = await apiClient.get<ProfileResponse>(ENDPOINTS.PROFILE.GET);
  return res;
};
