import apiClient from './clientProvider';
import { ENDPOINTS } from '@/constants/endpoints';

// Types
export interface PathOption {
  id: "student" | "individual" | "enterprise";
  title: string;
  description: string;
  image: string;
};

// ==================== PATH OPERATIONS ====================

// Save selected path
export const saveSelectedPath = async (pathId: PathOption["id"]) => {
  const res = await apiClient.post(ENDPOINTS.PATH.SELECT_PATH, { role: pathId });
  return res;
};


