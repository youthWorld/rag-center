import { api, type ApiEnvelope } from "../lib/api";
import type { DocumentUploadResponse, KnowledgeBase } from "../types";

export type CreateKnowledgeBasePayload = {
  name: string;
  tenant_id: string;
  description?: string;
};

export type UploadDocumentPayload = {
  tenant_id: string;
  kb_id: string;
  title: string;
  content: string;
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
