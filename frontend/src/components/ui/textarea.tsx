import { forwardRef, type TextareaHTMLAttributes } from "react";
import { cn } from "../../lib/utils";

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cn(
        "min-h-28 w-full resize-y rounded-xl border border-line bg-white px-3.5 py-3 text-sm text-ink placeholder:text-muted/70 focus:border-moss focus:ring-4 focus:ring-moss/10",
        className,
      )}
      {...props}
    />
  ),
);
Textarea.displayName = "Textarea";
