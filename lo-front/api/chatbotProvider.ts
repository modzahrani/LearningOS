import apiClient from "./clientProvider";
import { ENDPOINTS } from "@/constants/endpoints";

export interface ChatSource {
  source: string;
  topic: string;
  role: string;
  level: number;
}

export interface ChatMessageResponse {
  answer: string;
  sources: ChatSource[];
}

export const sendChatMessage = async (message: string, source?: string) => {
  const res = await apiClient.post<ChatMessageResponse>(ENDPOINTS.CHATBOT.MESSAGE, {
    message,
    source,
  });
  return res;
};
