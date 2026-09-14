import { useMutation } from "@tanstack/react-query";
import {
  createKnowledgeBase,
  type CreateKnowledgeBasePayload,
} from "../services/knowledge-base";

export function useCreateKnowledgeBase() {
  return useMutation({
    mutationFn: (payload: CreateKnowledgeBasePayload) => createKnowledgeBase(payload),
  });
}
