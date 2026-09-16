import { AlertCircle, CheckCircle2, LoaderCircle } from "lucide-react";
import { Badge } from "../ui/badge";
import type { DocumentStatus, DocumentStatusCode } from "../../types";

type StatusBadgeProps = {
  status: DocumentStatus | DocumentStatusCode;
};

function getStatusMeta(status: StatusBadgeProps["status"]) {
  if (status === "SUCCESS" || status === 1) {
    return {
      label: "成功",
      icon: <CheckCircle2 size={13} />,
      className: "border-emerald-200 bg-emerald-50 text-emerald-700",
    };
  }
  if (status === "FAILED" || status === 2) {
    return {
      label: "失败",
      icon: <AlertCircle size={13} />,
      className: "border-red-200 bg-red-50 text-danger",
    };
  }
  return {
    label: "索引中",
    icon: <LoaderCircle size={13} className="animate-spin" />,
    className: "border-amber-200 bg-amber-50 text-amber-700",
  };
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const meta = getStatusMeta(status);
  return (
    <Badge className={meta.className}>
      {meta.icon}
      {meta.label}
    </Badge>
  );
}
