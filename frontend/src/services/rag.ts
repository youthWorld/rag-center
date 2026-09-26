import { api, type ApiEnvelope } from "../lib/api";
import type {
  FeedbackData,
  FeedbackRequest,
  RagRetrieveResponse,
  ResearchData,
  EvidenceOptions,
  QueryOptions,
  RetrieveProfile,
  RetrievalOptions,
  RerankOptions,
} from "../types";

export type RetrievePayload = {
  kb_id?: string;
  kb_ids?: string[];
  user_id: string;
  query: string;
  profile: RetrieveProfile;
  top_k?: number;
  retrieval_options?: RetrievalOptions;
  rerank_options?: RerankOptions;
  query_options?: QueryOptions;
  evidence_options?: EvidenceOptions;
  index_version?: string;
};

export type ResearchPayload = {
  kb_id?: string;
  kb_ids?: string[];
  user_id: string;
  query: string;
};

async function retrieve(payload: RetrievePayload) {
  const response = await api.post<ApiEnvelope<RagRetrieveResponse>>(
    "/api/v1/rag/retrieve",
    payload,
  );
  return response.data.data;
}

async function submitFeedback(payload: FeedbackRequest) {
  const response = await api.post<ApiEnvelope<FeedbackData>>(
    "/api/v1/rag/feedback",
    payload,
  );
  return response.data.data;
}

async function research(payload: ResearchPayload) {
  const response = await api.post<ApiEnvelope<ResearchData>>(
    "/api/v1/rag/research",
    payload,
  );
  return response.data.data;
}

export const ragService = { retrieve, research, submitFeedback };
