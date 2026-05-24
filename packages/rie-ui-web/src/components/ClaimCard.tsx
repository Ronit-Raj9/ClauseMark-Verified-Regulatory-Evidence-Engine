import { Link } from "react-router-dom";

import type { ClaimDTO } from "@/types";

// Compact claim summary used by the review queue.

interface Props {
  claim: ClaimDTO;
  badge?: string | null;
  tone?: "neutral" | "warn" | "danger";
}

const TONES = {
  neutral: "border-slate-200",
  warn: "border-coverage-absent",
  danger: "border-gate-fail",
} as const;

const STATUS_STYLES: Record<string, string> = {
  verified: "bg-gate-pass text-white",
  flagged: "bg-coverage-absent text-white",
  rejected: "bg-gate-fail text-white",
  draft: "bg-slate-300 text-slate-800",
};

export default function ClaimCard({ claim, badge, tone = "neutral" }: Props) {
  const statusClass = STATUS_STYLES[claim.layer1_status] ?? "bg-slate-300 text-slate-800";
  return (
    <Link
      to={`/review/${encodeURIComponent(claim.claim_id)}`}
      className={[
        "block bg-white border-l-4 border border-slate-200 rounded-md p-3 hover:shadow-sm transition-shadow",
        TONES[tone],
      ].join(" ")}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[11px] font-mono text-slate-500 mb-1 flex flex-wrap items-center gap-2">
            <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700">
              {claim.jurisdiction}
            </span>
            <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700">
              pillar {claim.pillar_id}
            </span>
            <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700">
              ind {claim.indicator_id}
            </span>
            <span className={["px-1.5 py-0.5 rounded font-semibold", statusClass].join(" ")}>
              {claim.layer1_status}
            </span>
          </div>
          <div className="text-sm font-medium text-slate-900 truncate">
            {claim.claim_id}
          </div>
          <div className="text-xs text-slate-500 truncate">
            clause {claim.clause_id} · pattern {claim.clause_pattern} · {claim.evidence_spans.length}
            {" "}span(s)
          </div>
        </div>
        {badge && (
          <span className="shrink-0 text-[11px] font-mono px-2 py-0.5 rounded bg-slate-900 text-white">
            {badge}
          </span>
        )}
      </div>
    </Link>
  );
}
