export type KnowledgeBase = {
  kb_id: string;
  name: string;
  tenant_id: string;
  created_at: string;
};

export type KnowledgeBaseTreeDocument = {
  document_id: string;
  title: string;
  status: "SUCCESS";
  chunk_count: number;
  created_at: string;
};

export type KnowledgeBaseTreeItem = {
  kb_id: string;
  name: string;
  description?: string | null;
  created_at: string;
  documents: KnowledgeBaseTreeDocument[];
};

export type KnowledgeBaseTenantTree = {
  tenant_id: string;
  knowledge_bases: KnowledgeBaseTreeItem[];
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

export type RetrievalMode = "vector" | "bm25" | "hybrid";

export type RetrievalOptions = {
  mode: RetrievalMode;
  vector_top_k?: number;
  bm25_top_k?: number;
  rrf_k?: number;
};

export type RerankOptions = {
  enabled: true;
  top_n: number;
};

export type RetrievedChunk = {
  document_id: string;
  chunk_id: string;
  title: string;
  content: string;
  score: number;
  vector_score?: number | null;
  bm25_score?: number | null;
  vector_rank?: number | null;
  bm25_rank?: number | null;
  retrieval_source: "vector" | "bm25" | "hybrid";
  rerank_score?: number | null;
};

export type RetrievalMetadata = {
  top_k?: number;
  latency_ms?: number;
  vector_store?: string;
  retrieval?: {
    mode?: RetrievalMode;
    vector_top_k?: number;
    bm25_top_k?: number;
    vector_count?: number;
    bm25_count?: number;
    fused_count?: number;
    degraded?: boolean;
    degraded_reason?: string;
    [key: string]: unknown;
  };
  rerank?: {
    enabled?: boolean;
    provider?: string;
    llm_provider?: string | null;
    model?: string | null;
    top_n?: number;
    candidate_count?: number;
    degraded?: boolean;
    error?: string;
    [key: string]: unknown;
  };
  [key: string]: unknown;
};

export type RagRetrieveResponse = {
  query: string;
  kb_id: string;
  retrieved_chunks: RetrievedChunk[];
  metadata: RetrievalMetadata;
};
