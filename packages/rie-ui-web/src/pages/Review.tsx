import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { claims as claimsApi, reviews as reviewsApi } from "@/api/client";
import DecisionBar from "@/components/DecisionBar";
import GateBadge from "@/components/GateBadge";
import SpanViewer from "@/components/SpanViewer";
import type {
  CitationDTO,
  ClaimDetailResponse,
  Decomposition,
  ReviewDecision,
  ReviewDTO,
  ScoreBand,
} from "@/types";

// Audit viewer route /review/:claimId.
//
// Two-pane layout per systemArchitecture §7:
//   left  — source document with the exact evidence span(s) highlighted yellow.
//   right — claim decomposition (subject / condition / constraint / context),
//           indicator + clause pattern badges, per-gate verification rows, and
//           the overall layer1 status.
//
// Bottom: Accept / Correct / Reject -> POST /v1/reviews via reviews.submit.
//
// No pillar id or indicator id is hard-coded — every label is rendered from
// the API payload. Layer-2 recommendation is shown above the decision bar;
// the Correct score select is pre-filled from the recommended band.

const LAYER1_TONES: Record<string, string> = {
  verified: "bg-gate-pass text-white",
  flagged: "bg-coverage-absent text-white",
  rejected: "bg-gate-fail text-white",
  draft: "bg-slate-300 text-slate-800",
};

// Decomposition is open-ended — these are the canonical field names rie-extract
// commits to in rie_contracts.Decomposition. We render them in this order if
// present, then any extra keys after, so nothing is hidden from the auditor.
const PRIMARY_FIELDS: readonly string[] = [
  "subject",
  "condition",
  "constraint",
  "context",
];

function renderValue(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function decompositionEntries(d: Decomposition): Array<[string, unknown]> {
  const out: Array<[string, unknown]> = [];
  const seen = new Set<string>();
  for (const key of PRIMARY_FIELDS) {
    if (key in d) {
      out.push([key, (d as Record<string, unknown>)[key]]);
      seen.add(key);
    }
  }
  for (const [k, v] of Object.entries(d)) {
    if (!seen.has(k)) out.push([k, v]);
  }
  return out;
}

export default function Review() {
  const { claimId } = useParams<{ claimId: string }>();
  const queryClient = useQueryClient();
  const [reviewer, setReviewer] = useState("");
  const [activeSpanId, setActiveSpanId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [lastReview, setLastReview] = useState<ReviewDTO | null>(null);

  const detailQuery = useQuery<ClaimDetailResponse>({
    queryKey: ["claim", claimId],
    queryFn: () => claimsApi.get(claimId!),
    enabled: !!claimId,
  });

  const reviewsQuery = useQuery({
    queryKey: ["claim-reviews", claimId],
    queryFn: () => claimsApi.reviews(claimId!),
    enabled: !!claimId,
  });

  const submitMutation = useMutation({
    mutationFn: async (input: {
      decision: ReviewDecision;
      corrected_score?: ScoreBand | null;
      note: string;
    }) => {
      if (!claimId) throw new Error("missing claim id");
      return reviewsApi.submit({
        claim_id: claimId,
        reviewer: reviewer.trim(),
        decision: input.decision,
        corrected_score: input.corrected_score ?? null,
        note: input.note,
      });
    },
    onSuccess: (data) => {
      setSubmitError(null);
      setLastReview(data);
      void queryClient.invalidateQueries({ queryKey: ["claim-reviews", claimId] });
      void queryClient.invalidateQueries({ queryKey: ["claim", claimId] });
      void queryClient.invalidateQueries({ queryKey: ["claims"] });
    },
    onError: (err) => {
      setSubmitError(err instanceof Error ? err.message : String(err));
    },
  });

  if (!claimId) {
    return <div className="text-sm text-red-600">Missing claim id in URL.</div>;
  }

  if (detailQuery.isLoading) {
    return <div className="text-sm text-slate-500">Loading claim {claimId}…</div>;
  }

  if (detailQuery.isError) {
    return (
      <div className="text-sm text-red-600 font-mono">
        {(detailQuery.error as Error).message}
      </div>
    );
  }

  const detail = detailQuery.data;
  if (!detail) {
    return <div className="text-sm text-slate-500">No claim returned.</div>;
  }

  const { claim, verification, citations, layer2 } = detail;
  const layer1Tone = LAYER1_TONES[claim.layer1_status] ?? "bg-slate-300 text-slate-800";
  const decompEntries = decompositionEntries(claim.decomposition);

  return (
    <section className="space-y-4">
      {/* Header strip */}
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-xs text-slate-500 font-mono mb-1">
            <Link to="/queue" className="hover:underline">
              ← back to queue
            </Link>
          </div>
          <h1 className="text-xl font-semibold text-slate-900 break-all">
            {claim.claim_id}
          </h1>
          <p className="text-xs text-slate-500 font-mono mt-0.5">
            jurisdiction {claim.jurisdiction} · pillar {claim.pillar_id} ·
            clause {claim.clause_id} · regime {claim.regime}
          </p>
        </div>
        <span
          className={[
            "px-2.5 py-1 rounded-md text-xs font-mono font-semibold",
            layer1Tone,
          ].join(" ")}
          title={`layer1_status = ${claim.layer1_status}`}
        >
          layer1 {claim.layer1_status}
        </span>
      </header>

      {/* Two-pane grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* LEFT: source spans */}
        <div className="space-y-2">
          <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">
            Source spans
          </h2>
          <p className="text-xs text-slate-500">
            Highlighted in yellow at the exact evidence offsets. Text materialised
            deterministically by rie-api — never authored by the LLM.
          </p>
          <SpanViewer
            citations={citations}
            activeSpanId={activeSpanId}
            onSelect={(c: CitationDTO) => setActiveSpanId(c.span_id)}
          />

          {claim.evidence_spans.length > 0 && (
            <div className="text-[11px] font-mono text-slate-500 border border-slate-200 rounded-md p-2 bg-slate-50">
              <div className="font-semibold mb-1 text-slate-700">
                evidence_spans ({claim.evidence_spans.length})
              </div>
              <ul className="space-y-0.5">
                {claim.evidence_spans.map((s) => (
                  <li key={s.span_id}>
                    <button
                      type="button"
                      onClick={() => setActiveSpanId(s.span_id)}
                      className={[
                        "text-left hover:underline",
                        s.span_id === activeSpanId ? "text-slate-900 font-semibold" : "",
                      ].join(" ")}
                    >
                      {s.span_id} · {s.role} · doc {s.doc_id} · {s.char_start}–{s.char_end}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* RIGHT: decomposition + gates */}
        <div className="space-y-4">
          {/* Indicator / pattern badges */}
          <div className="flex flex-wrap gap-2">
            <span
              className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-slate-900 text-white text-xs font-mono"
              title="indicator_id"
            >
              <span className="opacity-70">indicator</span>
              <span className="font-semibold">{claim.indicator_id}</span>
            </span>
            <span
              className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-slate-700 text-white text-xs font-mono"
              title="clause_pattern"
            >
              <span className="opacity-70">pattern</span>
              <span className="font-semibold">{claim.clause_pattern}</span>
            </span>
            {claim.model_confidence != null && (
              <span
                className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-slate-200 text-slate-800 text-xs font-mono"
                title="model_confidence (Layer-1 selector confidence, not a score)"
              >
                <span className="opacity-70">conf</span>
                <span className="font-semibold">
                  {claim.model_confidence.toFixed(2)}
                </span>
              </span>
            )}
          </div>

          {/* Decomposition */}
          <div className="bg-white border border-slate-200 rounded-md">
            <header className="px-3 py-2 border-b border-slate-200 bg-slate-50 text-xs font-semibold uppercase tracking-wide text-slate-600">
              Decomposition
            </header>
            <dl className="divide-y divide-slate-100">
              {decompEntries.length === 0 && (
                <div className="px-3 py-3 text-xs text-slate-500">
                  (empty decomposition)
                </div>
              )}
              {decompEntries.map(([k, v]) => (
                <div key={k} className="px-3 py-2 grid grid-cols-[8rem_1fr] gap-3">
                  <dt className="text-xs font-mono uppercase text-slate-500 pt-0.5">
                    {k}
                  </dt>
                  <dd className="text-sm text-slate-900 whitespace-pre-wrap break-words">
                    {renderValue(v)}
                  </dd>
                </div>
              ))}
            </dl>
          </div>

          {/* Layer-2 recommendation (human confirmation required) */}
          {layer2 && (
            <div className="bg-amber-50 border border-amber-200 rounded-md">
              <header className="px-3 py-2 border-b border-amber-200 bg-amber-100/60 text-xs font-semibold uppercase tracking-wide text-amber-900 flex items-center justify-between gap-2">
                <span>Layer-2 recommendation</span>
                {layer2.human_confirmation_required && (
                  <span className="font-mono normal-case text-[10px] px-1.5 py-0.5 rounded bg-amber-200 shrink-0">
                    human confirmation required
                  </span>
                )}
              </header>
              <div className="px-3 py-3 space-y-2 text-sm">
                <div className="flex flex-wrap gap-2 items-center">
                  <span className="text-xs font-mono text-amber-800">indicator</span>
                  <span className="px-2 py-0.5 rounded-md bg-amber-800/90 text-white font-mono font-semibold text-xs">
                    {layer2.indicator_id}
                  </span>
                  <span className="text-xs font-mono text-amber-800">recommended band</span>
                  <span className="px-2 py-0.5 rounded-md bg-amber-900 text-white font-mono font-semibold text-xs">
                    {layer2.recommended_band}
                  </span>
                </div>
                <p className="text-slate-800 whitespace-pre-wrap">{layer2.rationale}</p>
                {layer2.open_questions.length > 0 && (
                  <div>
                    <div className="text-xs font-semibold uppercase text-amber-900 mb-1">
                      Open questions
                    </div>
                    <ul className="list-disc list-inside text-slate-700 space-y-0.5">
                      {layer2.open_questions.map((q) => (
                        <li key={q}>{q}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Verification gates */}
          <div className="bg-white border border-slate-200 rounded-md">
            <header className="px-3 py-2 border-b border-slate-200 bg-slate-50 text-xs font-semibold uppercase tracking-wide text-slate-600 flex items-center justify-between">
              <span>Verification gates</span>
              {verification && (
                <span
                  className={[
                    "px-2 py-0.5 rounded text-[11px] font-mono font-semibold",
                    verification.status === "verified"
                      ? "bg-gate-pass text-white"
                      : verification.status === "flagged"
                        ? "bg-coverage-absent text-white"
                        : "bg-gate-fail text-white",
                  ].join(" ")}
                >
                  {verification.status}
                </span>
              )}
            </header>
            {!verification && (
              <div className="px-3 py-3 text-xs text-slate-500">
                No verification report attached yet.
              </div>
            )}
            {verification && (
              <ul className="divide-y divide-slate-100">
                {verification.gates.map((g, idx) => (
                  <li
                    key={`${g.gate}-${idx}`}
                    className="px-3 py-2 flex items-start gap-3"
                  >
                    <div className="shrink-0">
                      <GateBadge gate={g} />
                    </div>
                    <div className="text-xs text-slate-600 flex-1 min-w-0">
                      <div className="font-mono break-words">
                        {g.detail || (g.passed ? "passed" : "failed")}
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5">
                        ran_at {g.ran_at}
                      </div>
                    </div>
                  </li>
                ))}
                {verification.failure_reasons.length > 0 && (
                  <li className="px-3 py-2 text-xs text-red-700 bg-red-50">
                    <span className="font-semibold">failure_reasons:</span>{" "}
                    {verification.failure_reasons.join("; ")}
                  </li>
                )}
              </ul>
            )}
          </div>

          {/* Prior reviews (audit trail) */}
          {reviewsQuery.data && reviewsQuery.data.items.length > 0 && (
            <div className="bg-white border border-slate-200 rounded-md">
              <header className="px-3 py-2 border-b border-slate-200 bg-slate-50 text-xs font-semibold uppercase tracking-wide text-slate-600">
                Prior reviews ({reviewsQuery.data.total})
              </header>
              <ul className="divide-y divide-slate-100">
                {reviewsQuery.data.items.map((r, idx) => (
                  <li
                    key={`${r.reviewer}-${r.decided_at}-${idx}`}
                    className="px-3 py-2 text-xs"
                  >
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-mono font-semibold">{r.reviewer}</span>
                      <span
                        className={[
                          "px-1.5 py-0.5 rounded font-mono",
                          r.decision === "accept"
                            ? "bg-gate-pass text-white"
                            : r.decision === "correct"
                              ? "bg-coverage-absent text-white"
                              : "bg-gate-fail text-white",
                        ].join(" ")}
                      >
                        {r.decision}
                      </span>
                      {r.corrected_score && (
                        <span className="px-1.5 py-0.5 rounded bg-slate-200 text-slate-800 font-mono">
                          score {r.corrected_score}
                        </span>
                      )}
                      <span className="text-slate-400 font-mono ml-auto">
                        {r.decided_at}
                      </span>
                    </div>
                    {r.note && (
                      <div className="mt-1 text-slate-600 whitespace-pre-wrap">
                        {r.note}
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {lastReview && (
            <div className="border border-gate-pass/50 bg-green-50 text-xs text-green-800 rounded-md px-3 py-2 font-mono">
              recorded {lastReview.decision} by {lastReview.reviewer} at {lastReview.decided_at}
            </div>
          )}
        </div>
      </div>

      {/* Decision bar — Accept / Correct / Reject */}
      <DecisionBar
        reviewer={reviewer}
        onReviewerChange={setReviewer}
        disabled={submitMutation.isPending}
        lastError={submitError}
        suggestedBand={layer2?.recommended_band ?? null}
        onSubmit={async (input) => {
          await submitMutation.mutateAsync(input);
        }}
      />
    </section>
  );
}
