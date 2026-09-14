import {
  Boxes,
  ChevronDown,
  FileCheck2,
  HelpCircle,
  LayoutGrid,
  Settings2,
  Sparkles,
} from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import { cn } from "../../lib/utils";

const navItems = [
  { label: "知识库工作台", icon: LayoutGrid, to: "/" },
];

export function AppShell() {
  return (
    <div className="min-h-screen bg-paper text-ink">
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-[248px] flex-col border-r border-line bg-[#eef3ef] md:flex">
        <div className="flex h-[76px] items-center gap-3 border-b border-line px-6">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-ink text-white shadow-sm">
            <Sparkles size={17} strokeWidth={2.2} />
          </div>
          <div>
            <p className="text-[15px] font-bold tracking-[-0.02em]">RAG Center</p>
            <p className="mt-0.5 text-[11px] font-medium uppercase tracking-[0.16em] text-muted">Lab workspace</p>
          </div>
        </div>

        <div className="flex-1 px-3 py-6">
          <p className="px-3 text-[10px] font-bold uppercase tracking-[0.2em] text-muted/80">Workspace</p>
          <nav className="mt-3 space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    cn(
                      "flex h-11 items-center gap-3 rounded-xl px-3 text-sm font-semibold transition-colors",
                      isActive ? "bg-white text-ink shadow-sm" : "text-muted hover:bg-white/70 hover:text-ink",
                    )
                  }
                >
                  <Icon size={17} />
                  {item.label}
                </NavLink>
              );
            })}
          </nav>

          <div className="mt-8 border-t border-line pt-6">
            <p className="px-3 text-[10px] font-bold uppercase tracking-[0.2em] text-muted/80">Validation</p>
            <div className="mt-3 space-y-1">
              <div className="flex h-10 items-center gap-3 rounded-xl px-3 text-sm font-medium text-muted">
                <FileCheck2 size={16} />
                Upload results
              </div>
              <div className="flex h-10 items-center gap-3 rounded-xl px-3 text-sm font-medium text-muted">
                <Boxes size={16} />
                Evidence flow
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-1 border-t border-line p-3">
          <button className="flex h-10 w-full items-center gap-3 rounded-xl px-3 text-sm font-medium text-muted hover:bg-white/70 hover:text-ink">
            <HelpCircle size={16} />
            使用帮助
          </button>
          <button className="flex h-10 w-full items-center gap-3 rounded-xl px-3 text-sm font-medium text-muted hover:bg-white/70 hover:text-ink">
            <Settings2 size={16} />
            工作区设置
          </button>
          <div className="mt-2 flex items-center justify-between rounded-xl bg-white/70 px-3 py-2.5">
            <div className="flex items-center gap-2.5">
              <div className="grid h-7 w-7 place-items-center rounded-full bg-[#f0c36a] text-xs font-bold text-ink">YL</div>
              <div>
                <p className="text-xs font-bold">本地调优环境</p>
                <p className="text-[10px] text-muted">developer</p>
              </div>
            </div>
            <ChevronDown size={14} className="text-muted" />
          </div>
        </div>
      </aside>

      <div className="md:pl-[248px]">
        <header className="sticky top-0 z-10 flex h-[76px] items-center justify-between border-b border-line bg-paper/90 px-5 backdrop-blur md:px-10">
          <div className="flex items-center gap-3 md:hidden">
            <div className="grid h-8 w-8 place-items-center rounded-lg bg-ink text-white">
              <Sparkles size={15} />
            </div>
            <span className="text-sm font-bold">RAG Center</span>
          </div>
          <div className="hidden items-center gap-2 text-xs font-medium text-muted md:flex">
            <span>RAG Center</span>
            <span className="text-muted/50">/</span>
            <span className="text-ink">知识库工作台</span>
          </div>
          <div className="flex items-center gap-2.5 text-xs font-semibold text-muted">
            <span className="h-2 w-2 rounded-full bg-moss shadow-[0_0_0_4px_rgba(30,114,92,0.10)]" />
            API proxy ready
          </div>
        </header>
        <main className="mx-auto max-w-[1440px] px-5 py-8 md:px-10 md:py-10">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
