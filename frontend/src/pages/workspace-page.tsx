import {
  AlertCircle,
  Check,
  CheckCircle2,
  Database,
  FileText,
  FolderOpen,
  Info,
  LoaderCircle,
  Plus,
  RotateCcw,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Progress } from "../components/ui/progress";
import { Textarea } from "../components/ui/textarea";
import { getApiErrorMessage } from "../lib/api";
import { formatBytes, formatDate, createId } from "../lib/utils";
import { useCreateKnowledgeBase } from "../hooks/use-knowledge-base";
import { uploadDocument } from "../services/knowledge-base";
import type { KnowledgeBase, UploadItem, UploadState } from "../types";

const LAST_KB_KEY = "rag-center:last-knowledge-base";

const knowledgeBaseSchema = z.object({
  name: z.string().trim().min(1, "请输入知识库名称").max(255, "名称不能超过 255 个字符"),
  tenantId: z.string().trim().min(1, "请输入租户标识").max(128, "租户标识不能超过 128 个字符"),
  description: z.string().max(5000, "描述不能超过 5000 个字符").optional(),
});

type KnowledgeBaseForm = z.infer<typeof knowledgeBaseSchema>;

function getStoredKnowledgeBase(): KnowledgeBase | null {
  try {
    const raw = localStorage.getItem(LAST_KB_KEY);
    return raw ? (JSON.parse(raw) as KnowledgeBase) : null;
  } catch {
    return null;
  }
}

function statusLabel(state: UploadState) {
  switch (state) {
    case "uploading":
      return "处理中";
    case "success":
      return "已完成";
    case "error":
      return "失败";
    default:
      return "待上传";
  }
}

function statusClass(state: UploadState) {
  switch (state) {
    case "uploading":
      return "border-amber-200 bg-amber-50 text-amber-700";
    case "success":
      return "border-emerald-200 bg-emerald-50 text-emerald-700";
    case "error":
      return "border-red-200 bg-red-50 text-danger";
    default:
      return "border-line bg-paper text-muted";
  }
}

function StatusIcon({ state }: { state: UploadState }) {
  if (state === "uploading") return <LoaderCircle size={16} className="animate-spin text-amber-600" />;
  if (state === "success") return <CheckCircle2 size={16} className="text-emerald-600" />;
  if (state === "error") return <AlertCircle size={16} className="text-danger" />;
  return <FileText size={16} className="text-muted" />;
}

export function WorkspacePage() {
  const [knowledgeBase, setKnowledgeBase] = useState<KnowledgeBase | null>(getStoredKnowledgeBase);
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [uploadItems, setUploadItems] = useState<UploadItem[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 3800);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const counts = useMemo(() => {
    const total = uploadItems.length;
    const success = uploadItems.filter((item) => item.state === "success").length;
    const failed = uploadItems.filter((item) => item.state === "error").length;
    const pending = uploadItems.filter((item) => item.state === "queued" || item.state === "uploading").length;
    return { total, success, failed, pending };
  }, [uploadItems]);

  const progress = counts.total ? ((counts.success + counts.failed) / counts.total) * 100 : 0;

  const addFiles = (files: FileList | File[]) => {
    const candidates = Array.from(files).map((file) => ({
      id: `${file.name}-${file.size}-${file.lastModified}-${file.webkitRelativePath}`,
      file,
      relativePath: file.webkitRelativePath || file.name,
      state: "queued" as const,
    }));

    setUploadItems((current) => {
      const existingIds = new Set(current.map((item) => item.id));
      return [...current, ...candidates.filter((item) => !existingIds.has(item.id))];
    });
    setToast(`${candidates.length} 个文件已加入上传队列`);
  };

  const updateUploadItem = (id: string, patch: Partial<UploadItem>) => {
    setUploadItems((current) =>
      current.map((item) => (item.id === id ? { ...item, ...patch } : item)),
    );
  };

  const uploadQueued = async (retryOnly = false) => {
    if (!knowledgeBase || isUploading) return;
    const queue = uploadItems.filter((item) =>
      retryOnly ? item.state === "error" : item.state === "queued",
    );
    if (!queue.length) return;

    setIsUploading(true);
    for (const item of queue) {
      updateUploadItem(item.id, { state: "uploading", message: undefined });
      try {
        const content = await item.file.text();
        if (!content.trim()) throw new Error("文件内容为空");
        const response = await uploadDocument({
          tenant_id: knowledgeBase.tenant_id,
          kb_id: knowledgeBase.kb_id,
          title: item.file.name,
          content,
        });
        updateUploadItem(item.id, {
          state: "success",
          documentId: response.document_id,
          chunkCount: response.chunk_count,
          message: `已切分 ${response.chunk_count} 个片段`,
        });
      } catch (error) {
        updateUploadItem(item.id, {
          state: "error",
          message: getApiErrorMessage(error),
        });
      }
    }
    setIsUploading(false);
    setToast(retryOnly ? "失败文件已重新处理" : "上传批次处理完成");
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    if (event.dataTransfer.files.length) addFiles(event.dataTransfer.files);
  };

  const handleCreated = (created: KnowledgeBase) => {
    setKnowledgeBase(created);
    setUploadItems([]);
    localStorage.setItem(LAST_KB_KEY, JSON.stringify(created));
    setIsCreateOpen(false);
    setToast(`知识库“${created.name}”已创建`);
  };

  const removeItem = (id: string) => {
    if (isUploading) return;
    setUploadItems((current) => current.filter((item) => item.id !== id));
  };

  const clearCompleted = () => {
    if (isUploading) return;
    setUploadItems((current) => current.filter((item) => item.state !== "success"));
  };

  return (
    <>
      <section className="flex flex-col justify-between gap-6 border-b border-line pb-7 lg:flex-row lg:items-end">
        <div>
          <div className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-[0.18em] text-moss">
            <span className="h-1.5 w-1.5 rounded-full bg-ember" />
            Content ingestion
          </div>
          <h1 className="text-balance text-3xl font-bold tracking-[-0.04em] text-ink md:text-[40px]">知识库工作台</h1>
          <p className="mt-3 max-w-[620px] text-sm leading-6 text-muted md:text-[15px]">
            创建一个干净的知识库，按顺序上传文档，确认每个文件的索引结果，为后续重排调优准备证据。
          </p>
        </div>
        <Button onClick={() => setIsCreateOpen(true)}>
          <Plus size={17} />
          新建知识库
        </Button>
      </section>

      {!knowledgeBase ? (
        <EmptyKnowledgeBase onCreate={() => setIsCreateOpen(true)} />
      ) : (
        <>
          <section className="flex flex-col justify-between gap-4 border-b border-line py-6 sm:flex-row sm:items-center">
            <div className="flex min-w-0 items-center gap-3.5">
              <div className="grid h-11 w-11 place-items-center rounded-2xl bg-moss/10 text-moss">
                <Database size={20} />
              </div>
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2.5">
                  <h2 className="text-lg font-bold tracking-[-0.02em]">{knowledgeBase.name}</h2>
                  <Badge className="border-moss/15 bg-moss/8 text-moss">
                    <span className="h-1.5 w-1.5 rounded-full bg-moss" />
                    当前工作区
                  </Badge>
                </div>
                <p className="mt-1 break-all text-xs text-muted">
                  {knowledgeBase.kb_id} · {knowledgeBase.tenant_id} · 创建于 {formatDate(knowledgeBase.created_at)}
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => setIsCreateOpen(true)}
              className="flex items-center gap-1.5 text-xs font-bold text-moss hover:text-moss-dark"
            >
              切换到新知识库
              <Plus size={14} />
            </button>
          </section>

          <div className="grid gap-6 pt-6 xl:grid-cols-[minmax(0,1fr)_320px]">
            <section className="min-w-0 rounded-2xl border border-line bg-white shadow-soft">
              <div className="flex flex-col justify-between gap-3 border-b border-line px-5 py-5 sm:flex-row sm:items-center sm:px-6">
                <div>
                  <h2 className="text-base font-bold">上传文档</h2>
                  <p className="mt-1 text-xs text-muted">文件会逐个提交，单个失败不会阻塞其他文件。</p>
                </div>
                <div className="flex items-center gap-2 text-xs font-semibold text-muted">
                  <span className="h-2 w-2 rounded-full bg-ember" />
                  {counts.total ? `${counts.total} 个文件在本批次` : "等待添加文件"}
                </div>
              </div>

              <div className="px-5 py-5 sm:px-6">
                <div
                  onDragEnter={(event) => {
                    event.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragOver={(event) => event.preventDefault()}
                  onDragLeave={(event) => {
                    if (event.currentTarget === event.target) setIsDragging(false);
                  }}
                  onDrop={handleDrop}
                  className={`rounded-2xl border border-dashed px-5 py-9 text-center transition-colors sm:py-12 ${
                    isDragging ? "border-moss bg-moss/5" : "border-ink/15 bg-paper/60 hover:border-moss/40"
                  }`}
                >
                  <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-white text-moss shadow-sm">
                    <Upload size={22} />
                  </div>
                  <h3 className="mt-4 text-sm font-bold">拖拽文件到这里</h3>
                  <p className="mx-auto mt-2 max-w-[430px] text-xs leading-5 text-muted">
                    支持一次选择多个文件，也可以直接选择文件夹。文件会在队列中逐个处理。
                  </p>
                  <div className="mt-5 flex flex-wrap justify-center gap-2.5">
                    <Button type="button" variant="secondary" size="sm" onClick={() => fileInputRef.current?.click()}>
                      <FileText size={15} />
                      选择文件
                    </Button>
                    <Button type="button" variant="secondary" size="sm" onClick={() => folderInputRef.current?.click()}>
                      <FolderOpen size={15} />
                      选择文件夹
                    </Button>
                  </div>
                  <input
                    ref={fileInputRef}
                    className="hidden"
                    type="file"
                    multiple
                    onChange={(event) => {
                      if (event.target.files) addFiles(event.target.files);
                      event.target.value = "";
                    }}
                  />
                  <input
                    ref={folderInputRef}
                    className="hidden"
                    type="file"
                    multiple
                    onChange={(event) => {
                      if (event.target.files) addFiles(event.target.files);
                      event.target.value = "";
                    }}
                    {...({ webkitdirectory: "", directory: "" } as React.InputHTMLAttributes<HTMLInputElement>)}
                  />
                </div>

                <div className="mt-5 flex flex-col justify-between gap-3 border-b border-line pb-4 sm:flex-row sm:items-center">
                  <div className="flex items-center gap-3 text-xs text-muted">
                    <Info size={14} className="text-moss" />
                    <span>当前知识库：{knowledgeBase.name}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {counts.failed > 0 && (
                      <Button type="button" variant="danger" size="sm" disabled={isUploading} onClick={() => uploadQueued(true)}>
                        <RotateCcw size={14} />
                        重试失败项
                      </Button>
                    )}
                    <Button
                      type="button"
                      size="sm"
                      disabled={isUploading || counts.pending === 0}
                      onClick={() => uploadQueued()}
                    >
                      {isUploading ? <LoaderCircle size={14} className="animate-spin" /> : <Upload size={14} />}
                      {isUploading ? "处理中..." : "开始上传"}
                    </Button>
                  </div>
                </div>

                <div className="mt-4 flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-bold">本批次文件</p>
                    <p className="mt-1 text-xs text-muted">
                      {counts.total ? `${counts.success} 个成功 · ${counts.failed} 个失败 · ${counts.pending} 个待处理` : "添加文件后将在这里查看逐项结果"}
                    </p>
                  </div>
                  {counts.success > 0 && (
                    <button
                      type="button"
                      onClick={clearCompleted}
                      disabled={isUploading}
                      className="flex items-center gap-1.5 text-xs font-semibold text-muted hover:text-danger disabled:opacity-50"
                    >
                      <Trash2 size={13} />
                      清除已完成
                    </button>
                  )}
                </div>

                <div className="mt-4 overflow-hidden rounded-xl border border-line">
                  {uploadItems.length === 0 ? (
                    <div className="grid min-h-[170px] place-items-center px-6 py-10 text-center">
                      <div>
                        <FileText size={22} className="mx-auto text-muted/50" />
                        <p className="mt-3 text-sm font-semibold text-muted">上传队列还是空的</p>
                        <p className="mt-1 text-xs text-muted/70">可以选择多个文件，或选择一个文件夹开始。</p>
                      </div>
                    </div>
                  ) : (
                    <div className="divide-y divide-line">
                      {uploadItems.map((item) => (
                        <div key={item.id} className="flex items-start gap-3 px-4 py-4 sm:items-center sm:px-5">
                          <div className="mt-0.5 grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-paper sm:mt-0">
                            <StatusIcon state={item.state} />
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:justify-between sm:gap-3">
                              <div className="min-w-0">
                                <p className="truncate text-sm font-semibold" title={item.relativePath}>{item.file.name}</p>
                                <p className="mt-1 truncate text-xs text-muted" title={item.relativePath}>
                                  {item.relativePath} · {formatBytes(item.file.size)}
                                </p>
                              </div>
                              <Badge className={`w-fit ${statusClass(item.state)}`}>
                                {statusLabel(item.state)}
                              </Badge>
                            </div>
                            {item.message && (
                              <p className={`mt-2 text-xs ${item.state === "error" ? "text-danger" : "text-muted"}`}>
                                {item.message}
                              </p>
                            )}
                          </div>
                          <button
                            type="button"
                            title="移除文件"
                            aria-label={`移除 ${item.file.name}`}
                            disabled={isUploading}
                            onClick={() => removeItem(item.id)}
                            className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-lg text-muted hover:bg-danger/8 hover:text-danger disabled:opacity-40 sm:mt-0"
                          >
                            <X size={15} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </section>

            <aside className="space-y-6">
              <section className="rounded-2xl border border-line bg-white p-5 shadow-soft">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-muted">Batch status</p>
                    <h2 className="mt-2 text-base font-bold">处理概览</h2>
                  </div>
                  <div className="grid h-9 w-9 place-items-center rounded-xl bg-[#fff3df] text-ember">
                    <Check size={18} />
                  </div>
                </div>
                <div className="mt-6 grid grid-cols-3 divide-x divide-line border-y border-line py-4">
                  <Stat label="总计" value={counts.total} />
                  <Stat label="成功" value={counts.success} tone="success" />
                  <Stat label="失败" value={counts.failed} tone="danger" />
                </div>
                <div className="mt-5 flex items-center justify-between text-xs font-semibold text-muted">
                  <span>完成度</span>
                  <span className="text-ink">{Math.round(progress)}%</span>
                </div>
                <Progress className="mt-2" value={progress} />
                <p className="mt-4 text-xs leading-5 text-muted">
                  每个文件完成索引后会单独反馈结果。失败项可以在本批次内重新尝试。
                </p>
              </section>

              <section className="rounded-2xl border border-line bg-[#edf5f0] p-5">
                <div className="flex items-center gap-2 text-moss">
                  <Info size={16} />
                  <p className="text-sm font-bold">验证提示</p>
                </div>
                <p className="mt-3 text-xs leading-5 text-muted">
                  建议先用 2–5 个内容明确的文本文件建立样本集，再进入检索接口观察向量分数与重排分数的变化。
                </p>
              </section>
            </aside>
          </div>
        </>
      )}

      <CreateKnowledgeBaseDialog
        open={isCreateOpen}
        onClose={() => setIsCreateOpen(false)}
        onCreated={handleCreated}
        onError={setToast}
      />

      {toast && (
        <div className="fixed bottom-5 right-5 z-50 flex max-w-[min(360px,calc(100vw-40px))] items-start gap-3 rounded-xl border border-line bg-white px-4 py-3.5 shadow-popover" role="status">
          <div className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-full bg-moss/10 text-moss">
            <Check size={15} />
          </div>
          <p className="pt-1 text-sm font-semibold leading-5 text-ink">{toast}</p>
          <button type="button" aria-label="关闭提示" onClick={() => setToast(null)} className="ml-auto text-muted hover:text-ink">
            <X size={15} />
          </button>
        </div>
      )}
    </>
  );
}

function EmptyKnowledgeBase({ onCreate }: { onCreate: () => void }) {
  return (
    <section className="mt-8 grid min-h-[420px] place-items-center rounded-2xl border border-line bg-white px-6 py-16 shadow-soft">
      <div className="max-w-[470px] text-center">
        <div className="mx-auto grid h-16 w-16 place-items-center rounded-3xl bg-moss/10 text-moss">
          <Database size={28} strokeWidth={1.8} />
        </div>
        <h2 className="mt-6 text-xl font-bold tracking-[-0.02em]">先创建一个知识库</h2>
        <p className="mt-3 text-sm leading-6 text-muted">
          创建完成后，这里会变成你的文档上传工作区。每个文件都会单独返回索引结果，方便定位调优问题。
        </p>
        <Button className="mt-6" onClick={onCreate}>
          <Plus size={17} />
          创建知识库
        </Button>
      </div>
    </section>
  );
}

function Stat({ label, value, tone = "default" }: { label: string; value: number; tone?: "default" | "success" | "danger" }) {
  const toneClass = tone === "success" ? "text-emerald-600" : tone === "danger" ? "text-danger" : "text-ink";
  return (
    <div className="px-3 text-center first:pl-0 last:pr-0">
      <p className={`text-xl font-bold tracking-[-0.04em] ${toneClass}`}>{value}</p>
      <p className="mt-1 text-[11px] font-semibold text-muted">{label}</p>
    </div>
  );
}

function CreateKnowledgeBaseDialog({
  open,
  onClose,
  onCreated,
  onError,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: (knowledgeBase: KnowledgeBase) => void;
  onError: (message: string) => void;
}) {
  const mutation = useCreateKnowledgeBase();
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<KnowledgeBaseForm>({
    resolver: zodResolver(knowledgeBaseSchema),
    defaultValues: {
      tenantId: "tenant_demo",
      name: "",
      description: "",
    },
  });

  useEffect(() => {
    if (open) reset({ tenantId: "tenant_demo", name: "", description: "" });
  }, [open, reset]);

  useEffect(() => {
    if (!open) return;
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !mutation.isPending) onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [mutation.isPending, onClose, open]);

  const submit = (values: KnowledgeBaseForm) => {
    mutation.mutate(
      {
        name: values.name.trim(),
        tenant_id: values.tenantId.trim(),
        description: values.description?.trim() || undefined,
      },
      {
        onSuccess: onCreated,
        onError: (error) => onError(getApiErrorMessage(error)),
      },
    );
  };

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-40 grid place-items-center bg-ink/35 px-4 py-6 backdrop-blur-[2px]"
      role="presentation"
      onMouseDown={(event) => {
        if (event.currentTarget === event.target && !mutation.isPending) onClose();
      }}
    >
      <div className="w-full max-w-[520px] rounded-2xl border border-white/60 bg-paper shadow-popover" role="dialog" aria-modal="true" aria-labelledby="create-kb-title">
        <div className="flex items-start justify-between border-b border-line px-6 py-5">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-moss">New workspace</p>
            <h2 id="create-kb-title" className="mt-2 text-xl font-bold tracking-[-0.03em]">创建知识库</h2>
            <p className="mt-1.5 text-xs leading-5 text-muted">先设定一个清晰的样本边界，再开始添加文档。</p>
          </div>
          <button type="button" title="关闭" aria-label="关闭" onClick={onClose} disabled={mutation.isPending} className="grid h-9 w-9 place-items-center rounded-lg text-muted hover:bg-ink/5 hover:text-ink disabled:opacity-40">
            <X size={17} />
          </button>
        </div>
        <form onSubmit={handleSubmit(submit)} className="space-y-5 px-6 py-6">
          <Field label="知识库名称" required error={errors.name?.message}>
            <Input placeholder="例如：退款政策样本集" {...register("name")} />
          </Field>
          <Field label="租户标识" required hint="用于调用现有后端接口，默认使用 tenant_demo。" error={errors.tenantId?.message}>
            <Input placeholder="tenant_demo" {...register("tenantId")} />
          </Field>
          <Field label="描述" hint="可选，用来记录这批数据的用途。" error={errors.description?.message}>
            <Textarea placeholder="例如：用于比较不同 rerank prompt 的退款问答样本" {...register("description")} />
          </Field>
          {mutation.isError && (
            <div className="flex items-start gap-2.5 rounded-xl border border-red-200 bg-red-50 px-3.5 py-3 text-xs leading-5 text-danger">
              <AlertCircle size={15} className="mt-0.5 shrink-0" />
              <span>{getApiErrorMessage(mutation.error)}</span>
            </div>
          )}
          <div className="flex justify-end gap-2.5 border-t border-line pt-5">
            <Button type="button" variant="ghost" onClick={onClose} disabled={mutation.isPending}>取消</Button>
            <Button type="submit" disabled={mutation.isPending}>
              {mutation.isPending && <LoaderCircle size={15} className="animate-spin" />}
              {mutation.isPending ? "创建中..." : "创建并进入工作区"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

function Field({
  label,
  required,
  hint,
  error,
  children,
}: {
  label: string;
  required?: boolean;
  hint?: string;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="flex items-center gap-1.5 text-sm font-bold text-ink">
        {label}
        {required && <span className="text-ember">*</span>}
      </span>
      {hint && <span className="mt-1 block text-xs leading-5 text-muted">{hint}</span>}
      <span className="mt-2 block">{children}</span>
      {error && <span className="mt-1.5 block text-xs font-medium text-danger">{error}</span>}
    </label>
  );
}
