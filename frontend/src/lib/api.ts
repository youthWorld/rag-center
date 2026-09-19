import axios from "axios";

export type ApiEnvelope<T> = {
  code: number;
  msg: string;
  data: T;
};

export const AUTH_ERROR_MESSAGE = "API Key 无效或未配置，请检查 frontend/.env 中的 API_KEY";

export class ApiResponseError extends Error {
  readonly code: number;

  constructor(code: number, message: string) {
    super(message);
    this.name = "ApiResponseError";
    this.code = code;
  }
}

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "",
  headers: {
    "Content-Type": "application/json",
  },
});

function isUnauthorizedEnvelope(value: unknown): value is { code: number } {
  return Boolean(value && typeof value === "object" && "code" in value && value.code === 20010);
}

api.interceptors.request.use((config) => {
  const apiKey = import.meta.env.API_KEY?.trim();
  if (apiKey) config.headers.Authorization = `Bearer ${apiKey}`;
  if (typeof FormData !== "undefined" && config.data instanceof FormData) {
    delete config.headers["Content-Type"];
  }
  return config;
});

api.interceptors.response.use(
  (response) => {
    if (isUnauthorizedEnvelope(response.data)) {
      return Promise.reject(new ApiResponseError(20010, AUTH_ERROR_MESSAGE));
    }
    return response;
  },
  (error) => {
    if (axios.isAxiosError(error) && isUnauthorizedEnvelope(error.response?.data)) {
      return Promise.reject(new ApiResponseError(20010, AUTH_ERROR_MESSAGE));
    }
    return Promise.reject(error);
  },
);

export function getApiErrorMessage(error: unknown) {
  if (error instanceof ApiResponseError) return error.message;
  if (axios.isAxiosError(error)) {
    const data = error.response?.data as { code?: number; msg?: string; detail?: string } | undefined;
    if (data?.code === 20010) return AUTH_ERROR_MESSAGE;
    return data?.msg || data?.detail || "接口请求失败，请检查后端服务。";
  }
  if (error instanceof Error) return error.message;
  return "操作失败，请稍后重试。";
}

export function getApiErrorCode(error: unknown): number | null {
  if (error instanceof ApiResponseError) return error.code;
  if (axios.isAxiosError(error)) {
    const data = error.response?.data as { code?: unknown } | undefined;
    return typeof data?.code === "number" ? data.code : null;
  }
  return null;
}
