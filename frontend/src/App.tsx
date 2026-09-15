import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/layout/app-shell";
import { KnowledgeFileTreePage } from "./pages/knowledge-file-tree-page";
import { RetrievePage } from "./pages/retrieve-page";
import { WorkspacePage } from "./pages/workspace-page";

export function App() {
  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<WorkspacePage />} />
          <Route path="/knowledge-bases" element={<KnowledgeFileTreePage />} />
          <Route path="/retrieve" element={<RetrievePage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
