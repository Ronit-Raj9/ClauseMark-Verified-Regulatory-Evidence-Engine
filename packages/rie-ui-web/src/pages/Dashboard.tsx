import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { coverage as coverageApi } from "@/api/client";
import CoverageGrid from "@/components/CoverageGrid";
import type { CoverageDTO } from "@/types";

export default function Dashboard() {
  const [filter, setFilter] = useState("");
  const navigate = useNavigate();

  const query = useQuery({
    queryKey: ["coverage", filter || null],
    queryFn: () => coverageApi.list(filter || undefined),
  });

  const records = useMemo(() => query.data?.items ?? [], [query.data]);

  return (
    <section className="space-y-6">
      <header className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900">Coverage</h1>
          <p className="text-sm text-slate-500">
            3-state absence reasoning per jurisdiction × indicator. Never a bare zero.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="filter jurisdiction (e.g. EU)"
            className="border border-slate-300 rounded px-2 py-1.5 text-sm w-56"
          />
          <button
            type="button"
            onClick={() => query.refetch()}
            className="px-3 py-1.5 rounded-md bg-slate-900 text-white text-sm"
          >
            Refresh
          </button>
        </div>
      </header>

      {query.isLoading && (
        <div className="text-sm text-slate-500">Loading coverage…</div>
      )}
      {query.isError && (
        <div className="text-sm text-red-600 font-mono">
          {(query.error as Error).message}
        </div>
      )}

      {query.data && (
        <CoverageGrid
          records={records}
          onCellClick={(rec: CoverageDTO) => {
            // Drill-in: jump to first verified claim if any; else queue.
            const firstClaim = rec.verified_claim_ids[0];
            if (firstClaim) {
              navigate(`/review/${encodeURIComponent(firstClaim)}`);
            } else {
              navigate(`/queue?jurisdiction=${encodeURIComponent(rec.jurisdiction)}`);
            }
          }}
        />
      )}
    </section>
  );
}
