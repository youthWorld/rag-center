import type { RetrieveProfile, TenantPlan } from "../types";

export const tenantPlanMeta: Record<TenantPlan, { label: string; className: string }> = {
  free: {
    label: "免费",
    className: "border-slate-200 bg-slate-50 text-slate-700",
  },
  standard: {
    label: "标准",
    className: "border-sky-200 bg-sky-50 text-sky-700",
  },
  pro: {
    label: "专业",
    className: "border-violet-200 bg-violet-50 text-violet-700",
  },
};

const planRank: Record<TenantPlan, number> = {
  free: 0,
  standard: 1,
  pro: 2,
};

const profileMinimumPlan: Record<RetrieveProfile, TenantPlan> = {
  speed: "free",
  balanced: "standard",
  quality: "pro",
  custom: "standard",
};

export function getProfileUpgradeMessage(
  plan: TenantPlan,
  profile: RetrieveProfile,
  profileLabel: string,
) {
  const requiredPlan = profileMinimumPlan[profile];
  if (planRank[plan] >= planRank[requiredPlan]) return "";

  return `当前为${tenantPlanMeta[plan].label}套餐，不支持${profileLabel}。请升级到${tenantPlanMeta[requiredPlan].label}套餐后使用。`;
}

export function formatPlanQuota(current: number, limit: number) {
  return `${new Intl.NumberFormat("zh-CN").format(current)} / ${new Intl.NumberFormat("zh-CN").format(limit)}`;
}
