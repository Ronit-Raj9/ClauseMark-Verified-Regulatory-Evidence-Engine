import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { claims as claimsApi, coverage as coverageApi } from "@/api/client";
import ClaimCard from "@/components/ClaimCard";
import type { ClaimDTO, CoverageDTO } from "@/types";

// Review queue per systemArchitecture §7:
//   1. Coverage gaps (no_evidence_in_searched_corpus, insufficient_coverage) at TOP.
//   2. Flagged claims next (any failing gate / unstable self-consistency).
//   3. Verified claims last (eligible for spot-check sampling).
//
// All ordering driven by API data — no pillar-specific bias.

interface CoverageItem {
  kind: "coverage";
  record: CoverageDTO;
}

interface ClaimItem {
  kind: "claim";
  claim: ClaimDTO;
  tone: "warn" | "danger" | "neutral";
  badge: string;
}

type QueueItem = CoverageItem | ClaimItem;

function rankCoverage(state: string): number {
  if (state === "no_evidence_in_searched_corpus") return 0;
  if (state === "insufficient_coverage") return 1;
  return 99;
}

export default function Queue() {
  const [params, setParams] = useSearchParams();
  const jurisdictionFilter = params.get("jurisdiction") ?? "";
  const [pillarFilter, setPillarFilter] = useState("");

  const claimsQuery = useQuery({
    queryKey: ["claims", jurisdictionFilter || null, pillarFilter || null],
    queryFn: () =>
      claimsApi.list({
        jurisdiction: jurisdictionFilter || undefined,
        pillar_id: pillarFilter || undefined,
      }),
  });

  const coverageQuery = useQuery({
    queryKey: ["coverage-queue", jurisdictionFilter || null],
    queryFn: () => coverageApi.list(jurisdictionFilter || undefined),
  });

  const items: QueueItem[] = useMemo(() => {
    const out: QueueItem[] = [];

    // Coverage gaps first.
    const gaps = (coverageQuery.data?.items ?? [])
      .filter((c) => c.state !== "evidence_found")
      .sort((a, b) => rankCoverage(a.state) - rankCoverage(b.state));
    for (const g of gaps) out.push({ kind: "coverage", record: g });

    // Flagged, then verified — preserve API ordering within each bucket.
    const all = claimsQuery.data?.items ?? [];
    const flagged = all.filter((c) => c.layer1_status === "flagged");
    const rejected = all.filter((c) => c.layer1_status === "rejected");
    const verified = all.filter((c) => c.layer1_status === "verified");
    const other = all.filter(
      (c) =>
        !["flagged", "rejected", "verified"].includes(c.layer1_status),
    );

    for (const c of flagged) out.push({ kind: "claim", claim: c, tone: "warn", badge: "FLAGGED" });
    for (const c of rejected) out.push({ kind: "claim", claim: c, tone: "danger", badge: "REJECTED" });
    for (const c of other) out.push({ kind: "claim", claim: c, tone: "neutral", badge: c.layer1_status.toUpperCase() });
    for (const c of verified) out.push({ kind: "claim", claim: c, tone: "neutral", badge: "VERIFIED" });

    return out;
  }, [claimsQuery.data, coverageQuery.data]);

  return (
    <section className="space-y-4">
      <header className="flex items-center justify-between gap-3 flex-wrap">
        <div>
          <h1 className="text-xl font-semibold text-slate-900">Review queue</h1>
          <p className="text-sm text-slate-500">
            Coverage gaps first, then flagged claims, then verified for spot-check.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <input
            value={jurisdictionFilter}
            onChange={(e) => {
              const v = e.target.value;
              const next = new URLSearchParams(params);
              if (v) next.set("jurisdiction", v);
              else next.delete("jurisdiction");
              setParams(next, { replace: true });
            }}
            placeholder="jurisdiction"
            className="border border-slate-300 rounded px-2 py-1.5 text-sm w-44"
          />
          <input
            value={pillarFilter}
            onChange={(e) => setPillarFilter(e.target.value)}
            placeholder="pillar id"
            className="border border-slate-300 rounded px-2 py-1.5 text-sm w-32"
          />
        </div>
      </header>

      {(claimsQuery.isLoading || coverageQuery.isLoading) && (
        <div className="text-sm text-slate-500">Loading queue…</div>
      )}
      {claimsQuery.isError && (
        <div className="text-sm text-red-600 font-mono">
          {(claimsQuery.error as Error).message}
        </div>
      )}

      <ul className="space-y-2">
        {items.map((item) =>
          item.kind === "coverage" ? (
            <li key={`cov-${item.record.jurisdiction}-${item.record.indicator_id}`}>
              <CoverageRow record={item.record} />
            </li>
          ) : (
            <li key={item.claim.claim_id}>
              <ClaimCard claim={item.claim} tone={item.tone} badge={item.badge} />
            </li>
          ),
        )}
        {items.length === 0 && !claimsQuery.isLoading && (
          <li className="text-sm text-slate-500 border border-dashed border-slate-300 rounded p-6 text-center">
            Queue empty. Trigger a run or relax filters.
          </li>
        )}
      </ul>
    </section>
  );
}

function CoverageRow({ record }: { record: CoverageDTO }) {
  const tone =
    record.state === "no_evidence_in_searched_corpus"
      ? "bg-coverage-absent"
      : "bg-coverage-insufficient";
  const recallPct =
    record.measured_recall != null
      ? `${Math.round(record.measured_recall * 100)}%`
      : "n/a";
  return (
    <div className="bg-white border-l-4 border border-slate-200 rounded-md p-3 flex items-center gap-3 border-l-coverage-absent">
      <span
        className={[
          "px-2 py-0.5 rounded text-[11px] font-mono font-semibold text-white",
          tone,
        ].join(" ")}
      >
        {record.state}
      </span>
      <div className="text-sm">
        <span className="font-medium">{record.jurisdiction}</span>{" "}
        <span className="font-mono text-slate-500">· indicator {record.indicator_id}</span>
      </div>
      <div className="ml-auto text-xs text-slate-500 font-mono">
        recall {recallPct}
        {record.reason ? ` · ${record.reason}` : ""}
      </div>
    </div>
  );
}
