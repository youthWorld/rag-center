import { api, type ApiEnvelope } from "../lib/api";
import type {
  DocumentUploadResponse,
  KnowledgeBase,
  KnowledgeBaseTenantTree,
} from "../types";

export type CreateKnowledgeBasePayload = {
  name: string;
  description?: string;
};

export type UploadDocumentPayload = {
  kb_id: string;
  title: string;
  content: string;
};

export async function fetchKnowledgeBaseTree(keyword?: string) {
  const response = await api.get<ApiEnvelope<KnowledgeBaseTenantTree[]>>(
    "/api/v1/knowledge-bases/tree",
    { params: keyword?.trim() ? { keyword: keyword.trim() } : undefined },
  );
  return response.data.data;
}

export const knowledgeBaseService = {
  fetchTree: fetchKnowledgeBaseTree,
};

export async function createKnowledgeBase(payload: CreateKnowledgeBasePayload) {
  const response = await api.post<ApiEnvelope<KnowledgeBase>>(
    "/api/v1/knowledge-bases/create",
    payload,
  );
  return response.data.data;
}

export async function uploadDocument(payload: UploadDocumentPayload) {
  const response = await api.post<ApiEnvelope<DocumentUploadResponse>>(
    "/api/v1/documents/upload",
    payload,
  );
  return response.data.data;
}
