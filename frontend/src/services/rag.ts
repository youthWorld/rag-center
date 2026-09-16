import { api, type ApiEnvelope } from "../lib/api";
import type {
  RagRetrieveResponse,
  QueryOptions,
  RetrievalMode,
  RerankOptions,
} from "../types";

export type RetrievePayload = {
  kb_id: string;
  user_id: string;
  query: string;
  top_k: number;
  retrieval_options: {
    mode: RetrievalMode;
    vector_top_k?: number;
    bm25_top_k?: number;
    rrf_k?: number;
  };
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
