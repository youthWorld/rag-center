import axios from "axios";

export type ApiEnvelope<T> = {
  code: number;
  msg: string;
  data: T;
};

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "",
  headers: {
    "Content-Type": "application/json",
  },
});

export function getApiErrorMessage(error: unknown) {
  if (axios.isAxiosError(error)) {
    const data = error.response?.data as { msg?: string; detail?: string } | undefined;
    return data?.msg || data?.detail || "接口请求失败，请检查后端服务。";
  }
  if (error instanceof Error) return error.message;
  return "操作失败，请稍后重试。";
}
