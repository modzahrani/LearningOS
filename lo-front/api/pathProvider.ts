import apiClient from './clientProvider';
import { ENDPOINTS } from '@/constants/endpoints';

// Types
export interface PathOption {
  id: "student" | "individual" | "enterprise";
  title: string;
  description: string;
  image: string;
};

export interface SelectedPathResponse {
  id: string;
  role: PathOption["id"];
  difficulty: string;
}

// ==================== PATH OPERATIONS ====================

// Save selected path
export const saveSelectedPath = async (pathId: PathOption["id"]) => {
  const res = await apiClient.post(ENDPOINTS.PATH.SELECT_PATH, { role: pathId });
  return res;
};

// Get selected path
export const getSelectedPath = async () => {
  const res = await apiClient.get<SelectedPathResponse>(ENDPOINTS.PATH.GET_SELECTED_PATH);
  return res;
};

