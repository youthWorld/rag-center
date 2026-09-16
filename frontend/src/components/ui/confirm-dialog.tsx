import { AlertTriangle, LoaderCircle, X } from "lucide-react";
import { useEffect } from "react";
import { Button } from "./button";

type ConfirmDialogProps = {
  open: boolean;
  title: string;
  description: string;
  confirmLabel?: string;
  isPending?: boolean;
  onClose: () => void;
  onConfirm: () => void;
};

export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = "确认删除",
  isPending = false,
  onClose,
  onConfirm,
}: ConfirmDialogProps) {
  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !isPending) onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isPending, onClose, open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-ink/35 px-4 py-6 backdrop-blur-[2px]"
      role="presentation"
      onMouseDown={(event) => {
        if (event.currentTarget === event.target && !isPending) onClose();
      }}
    >
      <div
        className="w-full max-w-[440px] rounded-2xl border border-white/60 bg-paper shadow-popover"
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
      >
        <div className="flex items-start justify-between border-b border-line px-6 py-5">
          <div className="flex items-start gap-3">
            <div className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-danger/10 text-danger">
              <AlertTriangle size={17} />
            </div>
            <div>
              <h2 id="confirm-dialog-title" className="text-base font-bold text-ink">
                {title}
              </h2>
              <p className="mt-2 text-sm leading-6 text-muted">{description}</p>
            </div>
          </div>
          <button
            type="button"
            title="关闭"
            aria-label="关闭"
            onClick={onClose}
            disabled={isPending}
            className="grid h-8 w-8 shrink-0 place-items-center rounded-lg text-muted hover:bg-ink/5 hover:text-ink disabled:opacity-40"
          >
            <X size={16} />
          </button>
        </div>
        <div className="flex justify-end gap-2.5 px-6 py-5">
          <Button type="button" variant="ghost" onClick={onClose} disabled={isPending}>
            取消
          </Button>
          <Button type="button" variant="danger" onClick={onConfirm} disabled={isPending}>
            {isPending && <LoaderCircle size={15} className="animate-spin" />}
            {isPending ? "删除中..." : confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}
