import { useMemo } from "react";

import type { CoverageDTO, CoverageState } from "@/types";

interface Props {
  records: CoverageDTO[];
  onCellClick?: (record: CoverageDTO) => void;
}

type StateStyle = {
  bg: string;
  text: string;
  border: string;
  dot: string;
  label: string;
};

const STATE_STYLES: Record<CoverageState, StateStyle> = {
  evidence_found: {
    bg: "bg-coverage-evidence-bg",
    text: "text-coverage-evidence-text",
    border: "border-coverage-evidence-border",
    dot: "bg-coverage-evidence-solid",
    label: "Evidence",
  },
  no_evidence_in_searched_corpus: {
    bg: "bg-coverage-absent-bg",
    text: "text-coverage-absent-text",
    border: "border-coverage-absent-border",
    dot: "bg-coverage-absent-solid",
    label: "Absent",
  },
  insufficient_coverage: {
    bg: "bg-coverage-insufficient-bg",
    text: "text-coverage-insufficient-text",
    border: "border-coverage-insufficient-border",
    dot: "bg-coverage-insufficient-solid",
    label: "Insufficient",
  },
};

function styleFor(state: string): StateStyle {
  return (STATE_STYLES as Record<string, StateStyle>)[state] ?? {
    bg: "bg-canvas",
    text: "text-muted",
    border: "border-line",
    dot: "bg-muted",
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
      jurisdictions: [...jset].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" })),
      indicators: [...iset].sort((a, b) => a.localeCompare(b, undefined, { numeric: true })),
      lookup: map,
    };
  }, [records]);

  if (records.length === 0) {
    return (
      <div className="rie-panel flex flex-col items-center justify-center gap-3 px-8 py-20 text-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-line bg-canvas font-serif text-xl text-muted">
          —
        </div>
        <p className="font-serif text-xl text-ink">No coverage records yet</p>
        <p className="max-w-md text-sm text-muted">
          Trigger a pipeline run from the API to populate the jurisdiction × indicator matrix.
        </p>
      </div>
    );
  }

  return (
    <div className="rie-panel overflow-hidden animate-fade-up">
      <div className="overflow-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b border-line bg-canvas/80">
              <th className="sticky left-0 z-20 min-w-[140px] border-r border-line bg-canvas/95 px-4 py-3 text-left">
                <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted">
                  Jurisdiction
                </span>
                <span className="mt-0.5 block font-mono text-[10px] text-muted/70">\ indicator</span>
              </th>
              {indicators.map((i) => (
                <th
                  key={i}
                  className="min-w-[108px] border-r border-line/60 px-3 py-3 text-center last:border-r-0"
                >
                  <span className="inline-flex rounded-md border border-line bg-surface px-2 py-1 font-mono text-xs font-semibold text-ink">
                    {i}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {jurisdictions.map((j, rowIdx) => (
              <tr
                key={j}
                className="border-b border-line/70 transition-colors last:border-b-0 hover:bg-canvas/40"
                style={{ animationDelay: `${rowIdx * 40}ms` }}
              >
                <th className="sticky left-0 z-10 border-r border-line bg-surface px-4 py-3 text-left">
                  <span className="font-medium text-ink">{j}</span>
                </th>
                {indicators.map((i) => {
                  const rec = lookup.get(`${j}::${i}`);
                  if (!rec) {
                    return (
                      <td key={i} className="border-r border-line/40 px-2 py-2 text-center last:border-r-0">
                        <span className="font-mono text-xs text-muted/40">—</span>
                      </td>
                    );
                  }
                  const style = styleFor(rec.state);
                  const recallPct =
                    rec.measured_recall != null
                      ? `${Math.round(rec.measured_recall * 100)}%`
                      : null;
                  return (
                    <td key={i} className="border-r border-line/40 p-2 last:border-r-0">
                      <button
                        type="button"
                        onClick={() => onCellClick?.(rec)}
                        title={rec.reason ?? rec.state}
                        className={[
                          "group w-full rounded-lg border px-2.5 py-2 text-left transition-all duration-200",
                          "hover:-translate-y-0.5 hover:shadow-float",
                          style.bg,
                          style.text,
                          style.border,
                        ].join(" ")}
                      >
                        <div className="flex items-center gap-1.5">
                          <span className={["h-1.5 w-1.5 shrink-0 rounded-full", style.dot].join(" ")} />
                          <span className="text-[11px] font-semibold uppercase tracking-wide">
                            {style.label}
                          </span>
                        </div>
                        {recallPct && (
                          <div className="mt-1 font-mono text-[10px] opacity-80">
                            recall {recallPct}
                          </div>
                        )}
                        {rec.verified_claim_ids.length > 0 && (
                          <div className="mt-0.5 font-mono text-[10px] opacity-70">
                            {rec.verified_claim_ids.length} claim
                            {rec.verified_claim_ids.length === 1 ? "" : "s"}
                          </div>
                        )}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex flex-wrap items-center gap-4 border-t border-line bg-canvas/50 px-4 py-3">
        <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted">
          Legend
        </span>
        {(Object.keys(STATE_STYLES) as CoverageState[]).map((s) => (
          <span key={s} className="inline-flex items-center gap-2 text-xs text-muted">
            <span className={["h-2.5 w-2.5 rounded-sm", STATE_STYLES[s].dot].join(" ")} />
            <span className="font-medium text-ink">{STATE_STYLES[s].label}</span>
            <span className="font-mono text-[10px] opacity-70">({s})</span>
          </span>
        ))}
      </div>
    </div>
  );
}
