import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { claims as claimsApi, coverage as coverageApi } from "@/api/client";
import ClaimCard from "@/components/ClaimCard";
import PageHeader from "@/components/PageHeader";
import type { ClaimDTO, CoverageDTO } from "@/types";

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

    const gaps = (coverageQuery.data?.items ?? [])
      .filter((c) => c.state !== "evidence_found")
      .sort((a, b) => rankCoverage(a.state) - rankCoverage(b.state));
    for (const g of gaps) out.push({ kind: "coverage", record: g });

    const all = claimsQuery.data?.items ?? [];
    const flagged = all.filter((c) => c.layer1_status === "flagged");
    const rejected = all.filter((c) => c.layer1_status === "rejected");
    const verified = all.filter((c) => c.layer1_status === "verified");
    const other = all.filter(
      (c) => !["flagged", "rejected", "verified"].includes(c.layer1_status),
    );

    for (const c of flagged) out.push({ kind: "claim", claim: c, tone: "warn", badge: "FLAGGED" });
    for (const c of rejected) out.push({ kind: "claim", claim: c, tone: "danger", badge: "REJECTED" });
    for (const c of other) out.push({ kind: "claim", claim: c, tone: "neutral", badge: c.layer1_status.toUpperCase() });
    for (const c of verified) out.push({ kind: "claim", claim: c, tone: "neutral", badge: "VERIFIED" });

    return out;
  }, [claimsQuery.data, coverageQuery.data]);

  return (
    <section className="space-y-6">
      <PageHeader
        eyebrow="Human review"
        title="Review queue"
        subtitle="Coverage gaps surface first, then flagged claims, then verified items for spot-check. Ordering is driven entirely by API data."
        actions={
          <>
            <input
              value={jurisdictionFilter}
              onChange={(e) => {
                const v = e.target.value;
                const next = new URLSearchParams(params);
                if (v) next.set("jurisdiction", v);
                else next.delete("jurisdiction");
                setParams(next, { replace: true });
              }}
              placeholder="Jurisdiction"
              className="rie-input w-40"
            />
            <input
              value={pillarFilter}
              onChange={(e) => setPillarFilter(e.target.value)}
              placeholder="Pillar id"
              className="rie-input w-28"
            />
          </>
        }
      />

      {(claimsQuery.isLoading || coverageQuery.isLoading) && (
        <div className="rie-panel px-6 py-12 text-center text-sm text-muted">Loading queue…</div>
      )}

      {claimsQuery.isError && (
        <div className="rounded-xl border border-gate-fail-border bg-gate-fail-bg px-4 py-3 text-sm text-gate-fail-text font-mono">
          {(claimsQuery.error as Error).message}
        </div>
      )}

      <ul className="space-y-3">
        {items.map((item, idx) =>
          item.kind === "coverage" ? (
            <li key={`cov-${item.record.jurisdiction}-${item.record.indicator_id}`} style={{ animationDelay: `${idx * 30}ms` }} className="animate-fade-up">
              <CoverageRow record={item.record} />
            </li>
          ) : (
            <li key={item.claim.claim_id} style={{ animationDelay: `${idx * 30}ms` }} className="animate-fade-up">
              <ClaimCard claim={item.claim} tone={item.tone} badge={item.badge} />
            </li>
          ),
        )}
        {items.length === 0 && !claimsQuery.isLoading && (
          <li className="rie-panel px-6 py-16 text-center">
            <p className="font-serif text-xl text-ink">Queue is empty</p>
            <p className="mt-2 text-sm text-muted">Trigger a run or relax your filters.</p>
          </li>
        )}
      </ul>
    </section>
  );
}

function CoverageRow({ record }: { record: CoverageDTO }) {
  const isAbsent = record.state === "no_evidence_in_searched_corpus";
  const recallPct =
    record.measured_recall != null ? `${Math.round(record.measured_recall * 100)}%` : "n/a";

  return (
    <div
      className={[
        "rie-panel flex flex-wrap items-center gap-3 border-l-4 p-4",
        isAbsent ? "border-l-coverage-absent-solid" : "border-l-coverage-insufficient-solid",
      ].join(" ")}
    >
      <span
        className={[
          "rie-badge",
          isAbsent
            ? "bg-coverage-absent-bg text-coverage-absent-text"
            : "bg-coverage-insufficient-bg text-coverage-insufficient-text",
        ].join(" ")}
      >
        {record.state}
      </span>
      <div className="text-sm">
        <span className="font-medium text-ink">{record.jurisdiction}</span>
        <span className="font-mono text-muted"> · indicator {record.indicator_id}</span>
      </div>
      <div className="ml-auto font-mono text-xs text-muted">
        recall {recallPct}
        {record.reason ? ` · ${record.reason}` : ""}
      </div>
    </div>
  );
}
