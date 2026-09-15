import { api, type ApiEnvelope } from "../lib/api";
import type { AuthMeData } from "../types";

async function fetchAuthMe() {
  const response = await api.get<ApiEnvelope<AuthMeData>>("/api/v1/auth/me");
  return response.data.data;
}

export const authService = { fetchAuthMe };
