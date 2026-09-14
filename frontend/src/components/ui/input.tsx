import { forwardRef, type InputHTMLAttributes } from "react";
import { cn } from "../../lib/utils";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "h-11 w-full rounded-xl border border-line bg-white px-3.5 text-sm text-ink placeholder:text-muted/70 focus:border-moss focus:ring-4 focus:ring-moss/10",
        className,
      )}
      {...props}
    />
  ),
);
Input.displayName = "Input";
