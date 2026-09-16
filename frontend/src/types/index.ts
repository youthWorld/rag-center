export type KnowledgeBase = {
  kb_id: string;
  name: string;
  tenant_id: string;
  created_at: string;
};

export type DocumentStatusCode = 1 | 2 | 3;

export type DocumentStatus = "SUCCESS" | "FAILED" | "PROCESSING";

export type KnowledgeBaseTreeDocument = {
  document_id: string;
  title: string;
  status: DocumentStatus;
  error_message: string | null;
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

export type KnowledgeBaseDetailData = {
  kb_id: string;
  name: string;
  description: string | null;
  settings: Record<string, unknown>;
  document_count: number;
  created_at: string;
  updated_at: string;
};

export type UpdateKnowledgeBaseRequest = {
  name?: string;
  description?: string | null;
  settings?: Record<string, unknown>;
};

export type KnowledgeBaseTenantTree = {
  tenant_id: string;
  knowledge_bases: KnowledgeBaseTreeItem[];
};

export type AuthMeData = {
  tenant_id: string;
  tenant_name: string;
  key_prefix?: string | null;
  key_name?: string | null;
};

export type DocumentUploadResponse = {
  document_id: string;
  kb_id: string;
  status: DocumentStatusCode;
  chunk_count: number;
};

export type KnowledgeBaseDeleteResponse = {
  kb_id: string;
};

export type DocumentDeleteResponse = {
  document_id: string;
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

export type QueryRewriteStrategy = "noop" | "rewrite";

export type QueryOptions = {
  enabled?: boolean;
  strategy?: QueryRewriteStrategy;
};

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

export type QueryProcessingMetadata = {
  raw_query: string;
  effective_query: string;
  search_query: string;
  strategy: QueryRewriteStrategy;
  rewrite_latency_ms: number;
  degraded: boolean;
  degraded_reason?: string | null;
  synonym_applied: boolean;
  synonym_expansions: string[];
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
  query_processing?: QueryProcessingMetadata | null;
  [key: string]: unknown;
};

export type RagRetrieveResponse = {
  query: string;
  kb_id: string;
  retrieved_chunks: RetrievedChunk[];
  metadata: RetrievalMetadata;
};
