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

export interface ForgotPasswordData {
  email: string;
}

export interface ResetPasswordData {
  new_password: string;
  token: string;
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

// ==================== AUTH OPERATIONS ====================

// Register a new user
export const register = async (userData: RegisterData) => {
  const res = await apiClient.post(ENDPOINTS.AUTH.REGISTER, userData);
  return res;
};

// Login user
export const login = async (credentials: LoginData) => {
  const res = await apiClient.post(ENDPOINTS.AUTH.LOGIN, credentials);
  return res;
};
