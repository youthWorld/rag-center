import type { ReactNode } from "react";
import { createPortal } from "react-dom";
import { CircleHelp } from "lucide-react";
import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from "react";

export type HelpTooltipPlacement = "top" | "bottom" | "left" | "right";

const TOOLTIP_GAP = 8;
const VIEWPORT_MARGIN = 16;

type TooltipPosition = {
  top: number;
  left: number;
};

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), Math.max(min, max));
}

export function HelpTooltip({
  content,
  label = "查看参数说明",
  placement = "top",
}: {
  content: ReactNode;
  label?: string;
  placement?: HelpTooltipPlacement;
}) {
  const anchorRef = useRef<HTMLSpanElement>(null);
  const panelRef = useRef<HTMLSpanElement>(null);
  const tooltipId = useId();
  const [isHovered, setIsHovered] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const [position, setPosition] = useState<TooltipPosition | null>(null);
  const isOpen = isHovered || isFocused;

  const updatePosition = useCallback(() => {
    const anchor = anchorRef.current;
    const panel = panelRef.current;
    if (!anchor || !panel) return;

    const anchorRect = anchor.getBoundingClientRect();
    const panelRect = panel.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const centeredLeft = anchorRect.left + (anchorRect.width - panelRect.width) / 2;
    const centeredTop = anchorRect.top + (anchorRect.height - panelRect.height) / 2;
    let nextPlacement = placement;
    let left = centeredLeft;
    let top = centeredTop;

    if (placement === "top") {
      top = anchorRect.top - panelRect.height - TOOLTIP_GAP;
      if (top < VIEWPORT_MARGIN && anchorRect.bottom + TOOLTIP_GAP + panelRect.height <= viewportHeight - VIEWPORT_MARGIN) {
        nextPlacement = "bottom";
      }
    } else if (placement === "bottom") {
      top = anchorRect.bottom + TOOLTIP_GAP;
      if (top + panelRect.height > viewportHeight - VIEWPORT_MARGIN && anchorRect.top - TOOLTIP_GAP - panelRect.height >= VIEWPORT_MARGIN) {
        nextPlacement = "top";
      }
    } else if (placement === "left") {
      left = anchorRect.left - panelRect.width - TOOLTIP_GAP;
      if (left < VIEWPORT_MARGIN && anchorRect.right + TOOLTIP_GAP + panelRect.width <= viewportWidth - VIEWPORT_MARGIN) {
        nextPlacement = "right";
      }
    } else {
      left = anchorRect.right + TOOLTIP_GAP;
      if (left + panelRect.width > viewportWidth - VIEWPORT_MARGIN && anchorRect.left - TOOLTIP_GAP - panelRect.width >= VIEWPORT_MARGIN) {
        nextPlacement = "left";
      }
    }

    if (nextPlacement === "top") {
      left = centeredLeft;
      top = anchorRect.top - panelRect.height - TOOLTIP_GAP;
    } else if (nextPlacement === "bottom") {
      left = centeredLeft;
      top = anchorRect.bottom + TOOLTIP_GAP;
    } else if (nextPlacement === "left") {
      left = anchorRect.left - panelRect.width - TOOLTIP_GAP;
      top = centeredTop;
    } else {
      left = anchorRect.right + TOOLTIP_GAP;
      top = centeredTop;
    }

    setPosition({
      left: clamp(left, VIEWPORT_MARGIN, viewportWidth - panelRect.width - VIEWPORT_MARGIN),
      top: clamp(top, VIEWPORT_MARGIN, viewportHeight - panelRect.height - VIEWPORT_MARGIN),
    });
  }, [placement]);

  useLayoutEffect(() => {
    if (!isOpen) {
      setPosition(null);
      return;
    }
    updatePosition();
  }, [isOpen, updatePosition]);

  useEffect(() => {
    if (!isOpen) return;
    const handleViewportChange = () => updatePosition();
    window.addEventListener("resize", handleViewportChange);
    window.addEventListener("scroll", handleViewportChange, true);
    return () => {
      window.removeEventListener("resize", handleViewportChange);
      window.removeEventListener("scroll", handleViewportChange, true);
    };
  }, [isOpen, updatePosition]);

  return (
    <>
      <span
        ref={anchorRef}
        className="help-tooltip relative inline-flex shrink-0 align-middle"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
      <button
        type="button"
        className="inline-flex h-5 w-5 items-center justify-center rounded-full text-muted transition-colors hover:text-moss"
        aria-label={label}
        aria-describedby={isOpen ? tooltipId : undefined}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
      >
        <CircleHelp size={14} strokeWidth={1.8} />
      </button>
      </span>
      {isOpen && typeof document !== "undefined" &&
        createPortal(
          <span
            ref={panelRef}
            id={tooltipId}
            role="tooltip"
            className="help-tooltip-panel pointer-events-none z-50 w-64 max-w-[calc(100vw-32px)] rounded-lg border border-line bg-ink px-3 py-2 text-left text-xs font-medium leading-5 text-white shadow-popover"
            style={{
              left: position?.left ?? 0,
              opacity: position ? 1 : 0,
              top: position?.top ?? 0,
              visibility: position ? "visible" : "hidden",
            }}
          >
            {content}
          </span>,
          document.body,
        )}
    </>
  );
}
