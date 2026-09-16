import { api, type ApiEnvelope } from "../lib/api";
import type {
  RagRetrieveResponse,
  QueryOptions,
  RetrieveProfile,
  RetrievalOptions,
  RerankOptions,
} from "../types";

export type RetrievePayload = {
  kb_id: string;
  user_id: string;
  query: string;
  profile: RetrieveProfile;
  top_k?: number;
  retrieval_options?: RetrievalOptions;
  rerank_options?: RerankOptions;
  query_options?: QueryOptions;
};

async function retrieve(payload: RetrievePayload) {
  const response = await api.post<ApiEnvelope<RagRetrieveResponse>>(
    "/api/v1/rag/retrieve",
    payload,
  );
  return response.data.data;
}

export const ragService = { retrieve };
