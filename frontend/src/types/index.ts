export type KnowledgeBase = {
  kb_id: string;
  name: string;
  tenant_id: string;
  created_at: string;
};

export type DocumentUploadResponse = {
  document_id: string;
  kb_id: string;
  status: number;
  chunk_count: number;
};

export type UploadState = "queued" | "uploading" | "success" | "error";

export type UploadItem = {
  id: string;
  file: File;
  relativePath: string;
  state: UploadState;
  message?: string;
  documentId?: string;
  chunkCount?: number;
};
