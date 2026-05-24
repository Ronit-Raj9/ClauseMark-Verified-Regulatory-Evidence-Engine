import { useMemo } from "react";

import type { CoverageDTO, CoverageState } from "@/types";

// Renders the jurisdiction x indicator grid. The component is fully data-
// driven — no pillar id, indicator id, or jurisdiction is hard-coded here.
// We derive the row + column axes from the records themselves.

interface Props {
  records: CoverageDTO[];
  onCellClick?: (record: CoverageDTO) => void;
}

const STATE_STYLES: Record<CoverageState, { bg: string; label: string; fg: string }> = {
  evidence_found: {
    bg: "bg-coverage-evidence",
    fg: "text-white",
    label: "Evidence",
  },
  no_evidence_in_searched_corpus: {
    bg: "bg-coverage-absent",
    fg: "text-white",
    label: "Absent",
  },
  insufficient_coverage: {
    bg: "bg-coverage-insufficient",
    fg: "text-white",
    label: "Insufficient",
  },
};

function styleFor(state: string) {
  return (STATE_STYLES as Record<string, (typeof STATE_STYLES)[CoverageState]>)[state] ?? {
    bg: "bg-slate-300",
    fg: "text-slate-900",
    label: state,
  };
}

export default function CoverageGrid({ records, onCellClick }: Props) {
  const { jurisdictions, indicators, lookup } = useMemo(() => {
    const jset = new Set<string>();
    const iset = new Set<string>();
    const map = new Map<string, CoverageDTO>();
    for (const r of records) {
      jset.add(r.jurisdiction);
      iset.add(r.indicator_id);
      map.set(`${r.jurisdiction}::${r.indicator_id}`, r);
    }
    return {
      jurisdictions: [...jset].sort(),
      indicators: [...iset].sort(),
      lookup: map,
    };
  }, [records]);

  if (records.length === 0) {
    return (
      <div className="border border-dashed border-slate-300 rounded-lg p-12 text-center text-slate-500">
        No coverage records yet. Trigger a pipeline run from the API to populate.
      </div>
    );
  }

  return (
    <div className="overflow-auto border border-slate-200 rounded-lg bg-white">
      <table className="min-w-full text-sm">
        <thead>
          <tr>
            <th className="sticky left-0 z-10 bg-slate-50 border-b border-r border-slate-200 px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
              Jurisdiction \ Indicator
            </th>
            {indicators.map((i) => (
              <th
                key={i}
                className="border-b border-slate-200 px-3 py-2 text-left text-xs font-mono font-semibold text-slate-600"
              >
                {i}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {jurisdictions.map((j) => (
            <tr key={j} className="hover:bg-slate-50">
              <th className="sticky left-0 z-10 bg-white border-b border-r border-slate-200 px-3 py-2 text-left text-xs font-semibold text-slate-700">
                {j}
              </th>
              {indicators.map((i) => {
                const rec = lookup.get(`${j}::${i}`);
                if (!rec) {
                  return (
                    <td
                      key={i}
                      className="border-b border-slate-200 px-2 py-2 text-center text-slate-300"
                    >
                      —
                    </td>
                  );
                }
                const style = styleFor(rec.state);
                const recallPct =
                  rec.measured_recall != null
                    ? `${Math.round(rec.measured_recall * 100)}%`
                    : null;
                return (
                  <td key={i} className="border-b border-slate-200 px-1.5 py-1.5">
                    <button
                      type="button"
                      onClick={() => onCellClick?.(rec)}
                      className={[
                        "w-full rounded-md px-2 py-1.5 text-xs font-medium",
                        "flex flex-col items-start gap-0.5 text-left",
                        "transition-transform hover:scale-[1.02]",
                        style.bg,
                        style.fg,
                      ].join(" ")}
                      title={rec.reason ?? rec.state}
                    >
                      <span>{style.label}</span>
                      {recallPct && (
                        <span className="font-mono text-[10px] opacity-90">
                          recall {recallPct}
                        </span>
                      )}
                      {rec.verified_claim_ids.length > 0 && (
                        <span className="font-mono text-[10px] opacity-90">
                          {rec.verified_claim_ids.length} claim
                          {rec.verified_claim_ids.length === 1 ? "" : "s"}
                        </span>
                      )}
                    </button>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>

      <div className="border-t border-slate-200 px-3 py-2 flex flex-wrap gap-3 text-xs text-slate-600">
        {(Object.keys(STATE_STYLES) as CoverageState[]).map((s) => (
          <span key={s} className="inline-flex items-center gap-1.5">
            <span className={["w-3 h-3 rounded-sm", STATE_STYLES[s].bg].join(" ")} />
            {STATE_STYLES[s].label} <span className="font-mono opacity-70">({s})</span>
          </span>
        ))}
      </div>
    </div>
  );
}
