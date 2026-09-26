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
  active_index_version?: string;
};

export type KnowledgeBaseDetailData = {
  kb_id: string;
  name: string;
  description: string | null;
  settings: Record<string, unknown>;
  document_count: number;
  created_at: string;
  updated_at: string;
  active_index_version?: string;
};

export type IndexVersion = {
  version: string;
  status: "building" | "ready" | "active" | "failed" | "archived";
  document_count: number;
  chunk_count: number;
  error_message?: string | null;
  created_at: string;
};

export type IndexVersionList = {
  kb_id: string;
  active_index_version: string;
  versions: IndexVersion[];
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

export type TenantPlan = "free" | "standard" | "pro";

export type AuthMeData = {
  tenant_id: string;
  tenant_name: string;
  key_prefix?: string | null;
  key_name?: string | null;
  plan: TenantPlan;
  features: {
    allowed_profiles: RetrieveProfile[];
    hybrid_allowed: boolean;
    rerank_allowed: boolean;
    query_rewrite_allowed: boolean;
    evidence_allowed: boolean;
    research_allowed: boolean;
  };
  limits: {
    retrieve_qps: number;
    retrieve_daily: number;
    max_kb: number;
    max_kb_per_retrieve: number;
    max_documents_per_kb: number;
    max_processing_documents: number;
  };
  usage: {
    kb_count: number;
    retrieve_daily_count: number;
  };
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
export type RetrieveProfile = "speed" | "balanced" | "quality" | "custom";

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
  enabled?: boolean;
  top_n?: number;
};

export type EvidenceOptions = {
  enabled?: boolean;
  max_items?: number;
};

export type EvidenceRole = "core" | "supporting";
export type EvidenceStatus = "complete" | "partial" | "missing";

export type EvidenceItem = {
  evidence_id: string;
  chunk_id: string;
  kb_id: string;
  document_id: string;
  title: string;
  heading_path?: string | null;
  index_version: string;
  content: string;
  source: string;
  retrieved_rank?: number | null;
  rounds?: number[];
  query_ids?: string[];
};

export type EvidencePack = {
  status: EvidenceStatus;
  missing_aspects: string[];
  groups: Array<{
    aspect: string;
    covered: boolean;
    evidence: Array<{ evidence_id: string; role: EvidenceRole }>;
  }>;
  items: EvidenceItem[];
};

export type EvidenceMetadata = {
  enabled?: boolean;
  executed?: boolean;
  candidate_count?: number;
  evidence_count?: number;
  output_chars?: number;
  budget_exceeded?: boolean;
  context_sources_included?: boolean;
  status?: EvidenceStatus | null;
  latency_ms?: number;
  degraded?: boolean;
  error_code?: string | null;
  error?: string | null;
  model_call?: {
    provider?: string | null;
    request_model?: string | null;
    response_model?: string | null;
    input_tokens?: number | null;
    output_tokens?: number | null;
    total_tokens?: number | null;
    [key: string]: unknown;
  } | null;
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
  kb_id?: string | null;
  kb_name?: string | null;
  title: string;
  content: string;
  score: number;
  vector_score?: number | null;
  bm25_score?: number | null;
  vector_rank?: number | null;
  bm25_rank?: number | null;
  retrieval_source: "vector" | "bm25" | "hybrid" | "graph";
  index_version?: string | null;
  section_id?: string | null;
  parent_section_id?: string | null;
  order_index?: number | null;
  context?: {
    content: string;
    chunk_ids: string[];
    sources: Array<{chunk_id: string; relation: "anchor" | "previous" | "next" | "same_section" | "parent_section" | "reference"; content?: string}>;
  } | null;
  rerank_score?: number | null;
  metadata?: {
    heading_path?: string | null;
    chunk_type?: string | null;
    [key: string]: unknown;
  };
};

export type RetrievalMetadata = {
  log_id?: string;
  trace_id?: string | null;
  top_k?: number;
  latency_ms?: number;
  vector_store?: string;
  index_version?: string;
  index_versions?: Record<string, string>;
  graph_injection?: {executed?: boolean; graph_injected_count?: number; graph_selected_count?: number; latency_ms?: number; degraded?: boolean; error?: string | null};
  context_expansion?: {executed?: boolean; supplemental_chunk_count?: number; latency_ms?: number; degraded?: boolean; error?: string | null};
  evidence?: EvidenceMetadata;
  retrieval?: {
    mode?: RetrievalMode;
    multi_kb?: boolean;
    kb_count?: number;
    per_kb_top_k?: number;
    fusion?: string | null;
    rrf_k?: number | null;
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
  tenant_policy?: {
    plan?: TenantPlan;
    retrieve_profile?: RetrieveProfile;
    effective_mode?: RetrievalMode;
    effective_rerank?: boolean;
    effective_query_rewrite?: boolean;
    effective_evidence?: boolean;
  };
  [key: string]: unknown;
};

export type RagRetrieveResponse = {
  query: string;
  kb_id: string;
  kb_ids?: string[];
  retrieved_chunks: RetrievedChunk[];
  evidence_pack?: EvidencePack | null;
  metadata: RetrievalMetadata;
};

export type FeedbackRequest = {
  trace_id: string;
  log_id: string;
  score: number;
  comment?: string;
};

export type FeedbackData = {
  feedback_id: string;
  trace_id: string;
  log_id: string;
  score: number;
};

export type ResearchAspect = {
  aspect_id: string;
  description: string;
};

export type PlannedQuery = {
  query_id: string;
  query: string;
  aspect_ids: string[];
};

export type ResearchPlanData = {
  aspects: ResearchAspect[];
  initial_queries: PlannedQuery[];
  degraded: boolean;
  error?: string | null;
};

export type ResearchTaskResult = {
  query_id: string;
  query: string;
  aspect_ids: string[];
  round: number;
  success: boolean;
  latency_ms: number;
  chunk_count: number;
  new_chunk_count: number;
  degraded: boolean;
  error?: string | null;
};

export type ResearchRound = {
  round: number;
  purpose: "initial" | "supplement";
  tasks: ResearchTaskResult[];
  candidate_count: number;
  new_chunk_count: number;
  latency_ms: number;
};

export type ResearchStageUsage = {
  stage: "planner" | "decider" | "finalizer";
  role: "fast" | "strong";
  model: string;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  degraded: boolean;
  error?: string | null;
};

export type ResearchMetadata = {
  research_id: string;
  log_id: string;
  trace_id?: string | null;
  profile: "research_fixed";
  tenant_plan: TenantPlan;
  index_versions: Record<string, string>;
  round_count: number;
  retrieval_task_count: number;
  llm_call_count: number;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
    stop_reason: "evidence_complete" | "no_new_evidence" | "max_rounds" | "duplicate_query" | "timeout" | "budget_exhausted" | "degraded";
    first_round_missing_aspect_ids: string[];
  degraded: boolean;
  errors: string[];
  stages: ResearchStageUsage[];
  planner: Record<string, unknown>;
  decider: Record<string, unknown>;
  finalizer: Record<string, unknown>;
};

export type ResearchData = {
  query: string;
  kb_id: string;
  kb_ids: string[];
  answer: string | null;
  evidence_pack: EvidencePack;
  plan: ResearchPlanData;
  rounds: ResearchRound[];
  metadata: ResearchMetadata;
};
