import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { coverage as coverageApi } from "@/api/client";
import CoverageGrid from "@/components/CoverageGrid";
import PageHeader from "@/components/PageHeader";
import type { CoverageDTO, CoverageState } from "@/types";

function countByState(records: CoverageDTO[], state: CoverageState) {
  return records.filter((r) => r.state === state).length;
}

function StatCard({
  label,
  value,
  hint,
  accent,
}: {
  label: string;
  value: number | string;
  hint: string;
  accent: string;
}) {
  return (
    <div className="rie-panel p-5 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-float">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted">
            {label}
          </p>
          <p className="mt-2 font-serif text-3xl font-medium tabular-nums text-ink">{value}</p>
          <p className="mt-1 text-xs text-muted">{hint}</p>
        </div>
        <span className={["mt-1 h-2 w-2 shrink-0 rounded-full", accent].join(" ")} />
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [filter, setFilter] = useState("");
  const navigate = useNavigate();

  const query = useQuery({
    queryKey: ["coverage", filter || null],
    queryFn: () => coverageApi.list(filter || undefined),
  });

  const records = useMemo(() => query.data?.items ?? [], [query.data]);

  const stats = useMemo(() => {
    const jurisdictions = new Set(records.map((r) => r.jurisdiction.toLowerCase())).size;
    return {
      jurisdictions,
      evidence: countByState(records, "evidence_found"),
      absent: countByState(records, "no_evidence_in_searched_corpus"),
      insufficient: countByState(records, "insufficient_coverage"),
    };
  }, [records]);

  return (
    <section className="space-y-8">
      <PageHeader
        eyebrow="Audit console"
        title="Coverage matrix"
        subtitle="Three-state absence reasoning per jurisdiction and indicator. The system never emits a bare zero — every gap carries measured recall context."
        actions={
          <>
            <input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter jurisdiction…"
              className="rie-input w-52"
            />
            <button
              type="button"
              onClick={() => query.refetch()}
              disabled={query.isFetching}
              className="rie-btn-primary min-w-[96px]"
            >
              {query.isFetching ? "Refreshing…" : "Refresh"}
            </button>
          </>
        }
      />

      {query.isLoading && (
        <div className="rie-panel px-6 py-12 text-center text-sm text-muted">
          Loading coverage records…
        </div>
      )}

      {query.isError && (
        <div className="rounded-xl border border-gate-fail-border bg-gate-fail-bg px-4 py-3 text-sm text-gate-fail-text">
          <span className="font-semibold">Could not load coverage.</span>{" "}
          <span className="font-mono">{(query.error as Error).message}</span>
        </div>
      )}

      {query.data && (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatCard
              label="Jurisdictions"
              value={stats.jurisdictions}
              hint="Distinct rows in the matrix"
              accent="bg-ink"
            />
            <StatCard
              label="Evidence found"
              value={stats.evidence}
              hint="Verified claims attached"
              accent="bg-coverage-evidence-solid"
            />
            <StatCard
              label="Absent in corpus"
              value={stats.absent}
              hint="No evidence — recall scoped"
              accent="bg-coverage-absent-solid"
            />
            <StatCard
              label="Insufficient"
              value={stats.insufficient}
              hint="Corpus too thin to conclude"
              accent="bg-coverage-insufficient-solid"
            />
          </div>

          <CoverageGrid
            records={records}
            onCellClick={(rec: CoverageDTO) => {
              const firstClaim = rec.verified_claim_ids[0];
              if (firstClaim) {
                navigate(`/review/${encodeURIComponent(firstClaim)}`);
              } else {
                navigate(`/queue?jurisdiction=${encodeURIComponent(rec.jurisdiction)}`);
              }
            }}
          />
        </>
      )}
    </section>
  );
}
