import { Link, NavLink, Route, Routes } from "react-router-dom";

import Dashboard from "@/pages/Dashboard";
import Queue from "@/pages/Queue";
import Review from "@/pages/Review";

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  [
    "rounded-lg px-3.5 py-2 text-sm font-medium transition-all duration-200",
    isActive
      ? "bg-ink text-white shadow-float"
      : "text-muted hover:bg-surface hover:text-ink",
  ].join(" ");

export default function App() {
  return (
    <div className="min-h-screen rie-canvas-grid">
      <div className="pointer-events-none fixed inset-x-0 top-0 h-72 bg-gradient-to-b from-white/80 to-transparent" />

      <header className="sticky top-0 z-50 border-b border-line/80 bg-canvas/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center gap-6 px-6 py-4">
          <Link to="/" className="group flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-ink text-sm font-serif font-semibold text-white transition-transform group-hover:scale-[0.98]">
              CM
            </span>
            <span className="hidden sm:block">
              <span className="block font-serif text-lg font-medium leading-none tracking-tight text-ink">
                ClauseMark
              </span>
              <span className="mt-0.5 block text-[11px] text-muted">
                Regulatory evidence audit
              </span>
            </span>
          </Link>

          <nav className="flex items-center gap-1 rounded-xl border border-line bg-surface/70 p-1">
            <NavLink to="/" end className={navLinkClass}>
              Coverage
            </NavLink>
            <NavLink to="/queue" className={navLinkClass}>
              Review queue
            </NavLink>
          </nav>

          <div className="ml-auto hidden items-center gap-2 md:flex">
            <span className="rie-badge bg-layer-advisory-bg text-layer-advisory-text">
              Layer 1 · verified
            </span>
            <span className="rie-badge border border-line bg-surface text-muted">
              Layer 2 · advisory
            </span>
          </div>
        </div>
      </header>

      <main className="relative mx-auto w-full max-w-7xl flex-1 px-6 py-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/queue" element={<Queue />} />
          <Route path="/review/:claimId" element={<Review />} />
          <Route
            path="*"
            element={
              <div className="rie-panel py-24 text-center">
                <p className="font-serif text-2xl text-ink">Page not found</p>
                <p className="mt-2 text-sm text-muted">
                  <Link to="/" className="font-medium text-ink underline underline-offset-4">
                    Return to coverage
                  </Link>
                </p>
              </div>
            }
          />
        </Routes>
      </main>

      <footer className="border-t border-line bg-surface/60">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-2 px-6 py-4 text-[11px] text-muted">
          <span className="font-mono">ClauseMark · rie-ui-web → rie-api</span>
          <span>Span-verified citations · human-confirmed scoring</span>
        </div>
      </footer>
    </div>
  );
}
