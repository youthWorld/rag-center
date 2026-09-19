import { api, type ApiEnvelope } from "../lib/api";
import type { DocumentDeleteResponse, DocumentUploadResponse } from "../types";

export async function deleteDocument(documentId: string) {
  const response = await api.delete<ApiEnvelope<DocumentDeleteResponse>>(
    `/api/v1/documents/${encodeURIComponent(documentId)}`,
  );
  return response.data.data;
}

export async function reindexDocument(documentId: string) {
  const response = await api.post<ApiEnvelope<DocumentUploadResponse>>(
    `/api/v1/documents/${encodeURIComponent(documentId)}/reindex`,
  );
  return response.data.data;
}

export async function uploadDocumentFile(formData: FormData) {
  const response = await api.post<ApiEnvelope<DocumentUploadResponse>>(
    "/api/v1/documents/upload",
    formData,
    { headers: { "Content-Type": undefined } },
  );
  return response.data.data;
}

export const documentService = {
  deleteDocument,
  reindexDocument,
  uploadDocumentFile,
};
