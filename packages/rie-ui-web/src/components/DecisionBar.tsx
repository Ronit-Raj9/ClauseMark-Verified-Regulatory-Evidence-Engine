import { useEffect, useState } from "react";

import type { ReviewDecision, ScoreBand } from "@/types";

interface Props {
  reviewer: string;
  onReviewerChange: (v: string) => void;
  onSubmit: (input: {
    decision: ReviewDecision;
    corrected_score?: ScoreBand | null;
    note: string;
  }) => Promise<void> | void;
  disabled?: boolean;
  lastError?: string | null;
  suggestedBand?: ScoreBand | string | null;
}

const SCORE_BANDS: ScoreBand[] = ["0", "0.5", "1", "no_evidence", "insufficient_coverage"];

function toScoreBand(value: ScoreBand | string | null | undefined): ScoreBand | null {
  if (value == null) return null;
  return SCORE_BANDS.includes(value as ScoreBand) ? (value as ScoreBand) : null;
}

export default function DecisionBar({
  reviewer,
  onReviewerChange,
  onSubmit,
  disabled,
  lastError,
  suggestedBand,
}: Props) {
  const [note, setNote] = useState("");
  const [correctedScore, setCorrectedScore] = useState<ScoreBand>(
    () => toScoreBand(suggestedBand) ?? "0.5",
  );
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const band = toScoreBand(suggestedBand);
    if (band) setCorrectedScore(band);
  }, [suggestedBand]);

  async function fire(decision: ReviewDecision) {
    if (!reviewer.trim()) return;
    setBusy(true);
    try {
      await onSubmit({
        decision,
        corrected_score: decision === "correct" ? correctedScore : null,
        note,
      });
      setNote("");
    } finally {
      setBusy(false);
    }
  }

  const isDisabled = disabled || busy || !reviewer.trim();

  return (
    <div className="sticky bottom-4 z-40 rie-panel border-ink/10 p-5 shadow-float backdrop-blur-sm">
      <p className="mb-4 text-[10px] font-semibold uppercase tracking-[0.12em] text-muted">
        Reviewer decision
      </p>

      <div className="flex flex-wrap items-end gap-4">
        <label className="flex min-w-[140px] flex-col gap-1.5 text-xs font-medium text-muted">
          Reviewer handle
          <input
            value={reviewer}
            onChange={(e) => onReviewerChange(e.target.value)}
            placeholder="your.handle"
            className="rie-input"
          />
        </label>

        <label className="flex min-w-[160px] flex-col gap-1.5 text-xs font-medium text-muted">
          Corrected score
          {suggestedBand && (
            <span className="font-mono text-[10px] font-normal text-coverage-absent-text">
              Layer-2 recommends {suggestedBand}
            </span>
          )}
          <select
            value={correctedScore}
            onChange={(e) => setCorrectedScore(e.target.value as ScoreBand)}
            className="rie-input font-mono"
          >
            {SCORE_BANDS.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </select>
        </label>

        <label className="flex min-w-[220px] flex-1 flex-col gap-1.5 text-xs font-medium text-muted">
          Note (optional)
          <input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Reason or rationale"
            className="rie-input"
          />
        </label>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => fire("accept")}
          disabled={isDisabled}
          className="rie-btn bg-gate-pass-solid text-white hover:opacity-90"
        >
          Accept
        </button>
        <button
          type="button"
          onClick={() => fire("correct")}
          disabled={isDisabled}
          className="rie-btn bg-coverage-absent-solid text-white hover:opacity-90"
        >
          Correct
        </button>
        <button
          type="button"
          onClick={() => fire("reject")}
          disabled={isDisabled}
          className="rie-btn bg-gate-fail-solid text-white hover:opacity-90"
        >
          Reject
        </button>
        {!reviewer.trim() && (
          <span className="text-xs text-muted">Enter a reviewer handle to enable actions</span>
        )}
      </div>

      {lastError && (
        <div className="mt-3 rounded-lg border border-gate-fail-border bg-gate-fail-bg px-3 py-2 font-mono text-xs text-gate-fail-text">
          {lastError}
        </div>
      )}
    </div>
  );
}
