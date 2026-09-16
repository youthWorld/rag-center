import { AlertCircle, LoaderCircle, Pencil, X } from "lucide-react";
import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import { getApiErrorMessage } from "../../lib/api";
import { knowledgeBaseService } from "../../services/knowledge-base";
import type { KnowledgeBaseDetailData, UpdateKnowledgeBaseRequest } from "../../types";

type EditKnowledgeBaseDialogProps = {
  open: boolean;
  kbId: string | null;
  onClose: () => void;
  onSaved: () => void;
};

export function EditKnowledgeBaseDialog({
  open,
  kbId,
  onClose,
  onSaved,
}: EditKnowledgeBaseDialogProps) {
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [settingsText, setSettingsText] = useState("{}");
  const [validationError, setValidationError] = useState<string | null>(null);

  const detailQuery = useQuery<KnowledgeBaseDetailData>({
    queryKey: ["knowledge-base-detail", kbId],
    queryFn: () => knowledgeBaseService.fetchDetail(kbId as string),
    enabled: open && Boolean(kbId),
  });

  const mutation = useMutation({
    mutationFn: (payload: UpdateKnowledgeBaseRequest) =>
      knowledgeBaseService.updateKnowledgeBase(kbId as string, payload),
  });

  useEffect(() => {
    if (!open) return;
    mutation.reset();
    setValidationError(null);
    setName("");
    setDescription("");
    setSettingsText("{}");
  }, [kbId, open]);

  useEffect(() => {
    if (!detailQuery.data) return;
    setName(detailQuery.data.name);
    setDescription(detailQuery.data.description ?? "");
    setSettingsText(JSON.stringify(detailQuery.data.settings ?? {}, null, 2));
  }, [detailQuery.data]);

  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !mutation.isPending) onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [mutation.isPending, onClose, open]);

  const submit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmedName = name.trim();
    if (!trimmedName) {
      setValidationError("请输入知识库名称");
      return;
    }

    let settings: unknown;
    try {
      settings = JSON.parse(settingsText);
    } catch {
      setValidationError("settings 不是有效的 JSON，请检查格式");
      return;
    }
    if (!settings || typeof settings !== "object" || Array.isArray(settings)) {
      setValidationError("settings 必须是一个 JSON 对象");
      return;
    }

    setValidationError(null);
    mutation.mutate(
      {
        name: trimmedName,
        description: description.trim() || null,
        settings: settings as Record<string, unknown>,
      },
      {
        onSuccess: async (detail) => {
          await queryClient.invalidateQueries({ queryKey: ["knowledge-base-tree"] });
          queryClient.setQueryData(["knowledge-base-detail", kbId], detail);
          onSaved();
          onClose();
        },
      },
    );
  };

  if (!open) return null;

  const errorMessage = validationError || (mutation.isError ? getApiErrorMessage(mutation.error) : null);

  return (
    <div
      className="fixed inset-0 z-40 grid place-items-center bg-ink/35 px-4 py-6 backdrop-blur-[2px]"
      role="presentation"
      onMouseDown={(event) => {
        if (event.currentTarget === event.target && !mutation.isPending) onClose();
      }}
    >
      <div
        className="max-h-[min(760px,calc(100vh-48px))] w-full max-w-[620px] overflow-y-auto rounded-2xl border border-white/60 bg-paper shadow-popover"
        role="dialog"
        aria-modal="true"
        aria-labelledby="edit-kb-title"
      >
        <div className="flex items-start justify-between border-b border-line px-6 py-5">
          <div className="flex items-start gap-3">
            <div className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-moss/10 text-moss">
              <Pencil size={16} />
            </div>
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-moss">Knowledge base settings</p>
              <h2 id="edit-kb-title" className="mt-2 text-xl font-bold tracking-[-0.03em]">编辑知识库</h2>
              <p className="mt-1.5 text-xs leading-5 text-muted">更新名称、描述或检索设置，保存后立即用于后续检索。</p>
            </div>
          </div>
          <button
            type="button"
            title="关闭"
            aria-label="关闭"
            onClick={onClose}
            disabled={mutation.isPending}
            className="grid h-9 w-9 shrink-0 place-items-center rounded-lg text-muted hover:bg-ink/5 hover:text-ink disabled:opacity-40"
          >
            <X size={17} />
          </button>
        </div>

        {detailQuery.isLoading ? (
          <div className="grid min-h-[280px] place-items-center px-6 py-12 text-center">
            <div>
              <LoaderCircle size={24} className="mx-auto animate-spin text-moss" />
              <p className="mt-4 text-sm font-semibold text-ink">正在读取知识库详情</p>
            </div>
          </div>
        ) : detailQuery.isError ? (
          <div className="px-6 py-8">
            <div className="flex items-start gap-2.5 rounded-xl border border-red-200 bg-red-50 px-3.5 py-3 text-xs leading-5 text-danger">
              <AlertCircle size={15} className="mt-0.5 shrink-0" />
              <span>{getApiErrorMessage(detailQuery.error)}</span>
            </div>
            <div className="mt-5 flex justify-end">
              <Button type="button" variant="ghost" onClick={onClose}>关闭</Button>
            </div>
          </div>
        ) : (
          <form onSubmit={submit} className="space-y-5 px-6 py-6">
            <label className="block">
              <span className="text-sm font-bold text-ink">知识库名称</span>
              <Input value={name} onChange={(event) => setName(event.target.value)} className="mt-2" maxLength={255} />
            </label>
            <label className="block">
              <span className="text-sm font-bold text-ink">描述</span>
              <Textarea value={description} onChange={(event) => setDescription(event.target.value)} className="mt-2" maxLength={5000} />
            </label>
            <label className="block">
              <span className="flex items-center gap-2 text-sm font-bold text-ink">
                settings JSON
                <span className="text-xs font-normal text-muted">对象格式</span>
              </span>
              <Textarea
                value={settingsText}
                onChange={(event) => setSettingsText(event.target.value)}
                className="mt-2 min-h-[220px] font-mono text-xs leading-5"
                spellCheck={false}
              />
            </label>
            {errorMessage && (
              <div className="flex items-start gap-2.5 rounded-xl border border-red-200 bg-red-50 px-3.5 py-3 text-xs leading-5 text-danger" role="alert">
                <AlertCircle size={15} className="mt-0.5 shrink-0" />
                <span>{errorMessage}</span>
              </div>
            )}
            <div className="flex justify-end gap-2.5 border-t border-line pt-5">
              <Button type="button" variant="ghost" onClick={onClose} disabled={mutation.isPending}>取消</Button>
              <Button type="submit" disabled={mutation.isPending}>
                {mutation.isPending && <LoaderCircle size={15} className="animate-spin" />}
                {mutation.isPending ? "保存中..." : "保存修改"}
              </Button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
