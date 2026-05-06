import { API_BASE } from '../constants/endpoints';
import axios from 'axios';

export const apiClient = axios.create({
  baseURL: API_BASE,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error)
);

// Response interceptor - handle errors globally
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    const requestUrl = (error.config?.url || "").toString();
    const isAuthRequest =
      requestUrl.includes("/login") ||
      requestUrl.endsWith("login") ||
      requestUrl.includes("/register") ||
      requestUrl.endsWith("register") ||
      requestUrl.includes("/auth/logout") ||
      requestUrl.endsWith("auth/logout") ||
      requestUrl.includes("/auth/oauth/start") ||
      requestUrl.endsWith("auth/oauth/start") ||
      requestUrl.includes("/auth/oauth/session") ||
      requestUrl.endsWith("auth/oauth/session") ||
      requestUrl.includes("/auth/forgot-password") ||
      requestUrl.endsWith("auth/forgot-password") ||
      requestUrl.includes("/auth/reset-password") ||
      requestUrl.endsWith("auth/reset-password");

    // Handle 401 Unauthorized - token expired or invalid
    if (error.response?.status === 401 && !isAuthRequest) {
      window.location.href = '/login';
    }
    
    // Handle 403 Forbidden
    if (error.response?.status === 403) {
      console.error('Access forbidden');
    }
    
    return Promise.reject(error);
  }
);

export default apiClient;
