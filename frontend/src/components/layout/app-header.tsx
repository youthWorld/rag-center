import { Database, FileSearch, LayoutGrid, Sparkles } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { cn } from "../../lib/utils";

type HeaderLink = {
  label: string;
  to: string;
  icon: typeof Database;
};

const pageLinks: Record<string, HeaderLink[]> = {
  "/": [
    { label: "知识库列表", to: "/knowledge-bases", icon: Database },
    { label: "检索调试", to: "/retrieve", icon: FileSearch },
  ],
  "/knowledge-bases": [
    { label: "知识库上传", to: "/", icon: LayoutGrid },
    { label: "检索调试", to: "/retrieve", icon: FileSearch },
  ],
  "/retrieve": [
    { label: "知识库上传", to: "/", icon: LayoutGrid },
    { label: "知识库列表", to: "/knowledge-bases", icon: Database },
  ],
};

export function AppHeader() {
  const { pathname } = useLocation();
  const links = pageLinks[pathname] ?? pageLinks["/"];

  return (
    <header className="sticky top-0 z-10 flex min-h-[76px] items-center justify-between gap-4 border-b border-line bg-paper/90 px-5 py-3 backdrop-blur md:px-10">
      <div className="flex min-w-0 items-center gap-3">
        <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-ink text-white md:h-9 md:w-9 md:rounded-xl">
          <Sparkles size={15} strokeWidth={2.2} />
        </div>
        <div className="min-w-0">
          <p className="truncate text-sm font-bold tracking-[-0.02em] text-ink md:text-[15px]">RAG 知识文件管理</p>
          <p className="mt-0.5 hidden text-[10px] font-medium uppercase tracking-[0.16em] text-muted sm:block">Debug workspace</p>
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-2 md:gap-3">
        <nav aria-label="页面导航" className="flex items-center gap-1 rounded-xl border border-line bg-white/75 p-1">
          {links.map(({ label, to, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              aria-label={label}
              title={label}
              className={cn(
                "inline-flex h-9 items-center gap-1.5 rounded-lg px-2.5 text-xs font-semibold text-muted transition-colors hover:bg-moss/8 hover:text-moss sm:px-3",
              )}
            >
              <Icon size={15} />
              <span className="hidden sm:inline">{label}</span>
            </Link>
          ))}
        </nav>
        <div className="hidden items-center gap-2 text-xs font-semibold text-muted xl:flex">
          <span className="h-2 w-2 rounded-full bg-moss shadow-[0_0_0_4px_rgba(30,114,92,0.10)]" />
          API proxy ready
        </div>
      </div>
    </header>
  );
}
