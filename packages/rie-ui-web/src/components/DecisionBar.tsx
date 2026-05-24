import { useState } from "react";

import type { ReviewDecision, ScoreBand } from "@/types";

// HITL decision bar — emits Accept / Correct (with corrected_score) / Reject.
// All persistence is the parent's responsibility (calls reviews.submit).
//
// The corrected-score select uses the ScoreBand enum string values exactly
// (mirrors rie_contracts.ScoreBand). We DO NOT invent any score — Layer-2 is
// reviewer-authored at decision time per the two-layer rule.

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
}

const SCORE_BANDS: ScoreBand[] = ["0", "0.5", "1", "no_evidence", "insufficient_coverage"];

export default function DecisionBar({
  reviewer,
  onReviewerChange,
  onSubmit,
  disabled,
  lastError,
}: Props) {
  const [note, setNote] = useState("");
  const [correctedScore, setCorrectedScore] = useState<ScoreBand>("0.5");
  const [busy, setBusy] = useState(false);

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
    <div className="sticky bottom-0 border-t border-slate-200 bg-white p-4 rounded-md shadow-sm">
      <div className="flex flex-wrap gap-3 items-end">
        <label className="flex flex-col text-xs text-slate-600">
          Reviewer
          <input
            value={reviewer}
            onChange={(e) => onReviewerChange(e.target.value)}
            placeholder="your.handle"
            className="mt-1 border border-slate-300 rounded px-2 py-1 text-sm w-40"
          />
        </label>

        <label className="flex flex-col text-xs text-slate-600">
          Corrected score (for Correct)
          <select
            value={correctedScore}
            onChange={(e) => setCorrectedScore(e.target.value as ScoreBand)}
            className="mt-1 border border-slate-300 rounded px-2 py-1 text-sm font-mono"
          >
            {SCORE_BANDS.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </select>
        </label>

        <label className="flex-1 min-w-[200px] flex flex-col text-xs text-slate-600">
          Note (optional)
          <input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="reason / rationale"
            className="mt-1 border border-slate-300 rounded px-2 py-1 text-sm"
          />
        </label>
      </div>

      <div className="flex flex-wrap gap-2 mt-3">
        <button
          type="button"
          onClick={() => fire("accept")}
          disabled={isDisabled}
          className="px-3 py-1.5 rounded-md bg-gate-pass text-white text-sm font-medium disabled:opacity-50"
        >
          Accept
        </button>
        <button
          type="button"
          onClick={() => fire("correct")}
          disabled={isDisabled}
          className="px-3 py-1.5 rounded-md bg-coverage-absent text-white text-sm font-medium disabled:opacity-50"
        >
          Correct
        </button>
        <button
          type="button"
          onClick={() => fire("reject")}
          disabled={isDisabled}
          className="px-3 py-1.5 rounded-md bg-gate-fail text-white text-sm font-medium disabled:opacity-50"
        >
          Reject
        </button>
        {!reviewer.trim() && (
          <span className="text-xs text-slate-500 self-center">
            enter reviewer handle to enable actions
          </span>
        )}
      </div>

      {lastError && (
        <div className="mt-2 text-xs text-red-600 font-mono">{lastError}</div>
      )}
    </div>
  );
}
