import { Link } from "react-router-dom";

import type { ClaimDTO } from "@/types";

interface Props {
  claim: ClaimDTO;
  badge?: string | null;
  tone?: "neutral" | "warn" | "danger";
}

const BORDER = {
  neutral: "border-l-ink/20",
  warn: "border-l-coverage-absent-solid",
  danger: "border-l-gate-fail-solid",
} as const;

const STATUS_STYLES: Record<string, string> = {
  verified: "bg-gate-pass-bg text-gate-pass-text",
  flagged: "bg-gate-flag-bg text-gate-flag-text",
  rejected: "bg-gate-fail-bg text-gate-fail-text",
  draft: "bg-canvas text-muted border border-line",
};

export default function ClaimCard({ claim, badge, tone = "neutral" }: Props) {
  const statusClass = STATUS_STYLES[claim.layer1_status] ?? "bg-canvas text-muted border border-line";

  return (
    <Link
      to={`/review/${encodeURIComponent(claim.claim_id)}`}
      className={[
        "rie-panel block border-l-4 p-4 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-float",
        BORDER[tone],
      ].join(" ")}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <span className="rie-badge border border-line bg-canvas text-muted">
              {claim.jurisdiction}
            </span>
            <span className="rie-badge border border-line bg-canvas text-muted">
              pillar {claim.pillar_id}
            </span>
            <span className="rie-badge border border-line bg-canvas text-muted">
              {claim.indicator_id}
            </span>
            <span className={["rie-badge", statusClass].join(" ")}>{claim.layer1_status}</span>
          </div>
          <div className="truncate font-mono text-sm font-medium text-ink">{claim.claim_id}</div>
          <div className="mt-1 truncate text-xs text-muted">
            clause {claim.clause_id} · {claim.clause_pattern} · {claim.evidence_spans.length} span
            {claim.evidence_spans.length === 1 ? "" : "s"}
          </div>
        </div>
        {badge && (
          <span className="shrink-0 rie-badge bg-ink text-white">{badge}</span>
        )}
      </div>
    </Link>
  );
}
