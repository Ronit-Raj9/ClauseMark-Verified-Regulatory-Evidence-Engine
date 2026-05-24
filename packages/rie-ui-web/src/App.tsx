import { Link, NavLink, Route, Routes } from "react-router-dom";

import Dashboard from "@/pages/Dashboard";
import Queue from "@/pages/Queue";
import Review from "@/pages/Review";

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  [
    "px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
    isActive
      ? "bg-slate-900 text-white"
      : "text-slate-600 hover:text-slate-900 hover:bg-slate-200",
  ].join(" ");

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-200 bg-white">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center gap-6">
          <Link to="/" className="font-semibold tracking-tight text-slate-900">
            RIE Audit Console
          </Link>
          <nav className="flex items-center gap-1">
            <NavLink to="/" end className={navLinkClass}>
              Coverage
            </NavLink>
            <NavLink to="/queue" className={navLinkClass}>
              Review Queue
            </NavLink>
          </nav>
          <div className="ml-auto text-xs text-slate-500 font-mono">
            Layer-1 verified · Layer-2 advisory
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-6">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/queue" element={<Queue />} />
          <Route path="/review/:claimId" element={<Review />} />
          <Route
            path="*"
            element={
              <div className="text-center text-slate-500 py-24">
                Not found.{" "}
                <Link to="/" className="text-slate-900 underline">
                  Back to coverage
                </Link>
              </div>
            }
          />
        </Routes>
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="max-w-7xl mx-auto px-6 py-3 text-xs text-slate-500 font-mono">
          rie-ui-web · talks only to rie-api · no in-process orchestration imports
        </div>
      </footer>
    </div>
  );
}
