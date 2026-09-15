import type { ReactNode } from "react";
import { CircleHelp } from "lucide-react";

export type HelpTooltipPlacement = "top" | "bottom" | "left" | "right";

export function HelpTooltip({
  content,
  label = "查看参数说明",
  placement = "top",
}: {
  content: ReactNode;
  label?: string;
  placement?: HelpTooltipPlacement;
}) {
  return (
    <span className="help-tooltip relative inline-flex shrink-0 align-middle" data-placement={placement}>
      <button
        type="button"
        className="inline-flex h-5 w-5 items-center justify-center rounded-full text-muted transition-colors hover:text-moss"
        aria-label={label}
      >
        <CircleHelp size={14} strokeWidth={1.8} />
      </button>
      <span
        role="tooltip"
        className="help-tooltip-panel pointer-events-none z-30 w-64 max-w-[calc(100vw-32px)] rounded-lg border border-line bg-ink px-3 py-2 text-left text-xs font-medium leading-5 text-white shadow-popover"
      >
        {content}
      </span>
    </span>
  );
}
