import { useState, type ReactNode } from "react";
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

const LAYER1_TONES: Record<string, string> = {
  verified: "bg-gate-pass-bg text-gate-pass-text",
  flagged: "bg-gate-flag-bg text-gate-flag-text",
  rejected: "bg-gate-fail-bg text-gate-fail-text",
  draft: "bg-canvas text-muted border border-line",
};

const PRIMARY_FIELDS: readonly string[] = ["subject", "condition", "constraint", "context"];

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

function Panel({
  title,
  children,
  action,
}: {
  title: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="rie-panel overflow-hidden">
      <header className="flex items-center justify-between border-b border-line bg-canvas/60 px-4 py-3">
        <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted">
          {title}
        </span>
        {action}
      </header>
      {children}
    </div>
  );
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
    return <div className="text-sm text-gate-fail-text">Missing claim id in URL.</div>;
  }

  if (detailQuery.isLoading) {
    return <div className="rie-panel px-6 py-12 text-center text-sm text-muted">Loading claim…</div>;
  }

  if (detailQuery.isError) {
    return (
      <div className="rounded-xl border border-gate-fail-border bg-gate-fail-bg px-4 py-3 font-mono text-sm text-gate-fail-text">
        {(detailQuery.error as Error).message}
      </div>
    );
  }

  const detail = detailQuery.data;
  if (!detail) {
    return <div className="text-sm text-muted">No claim returned.</div>;
  }

  const { claim, verification, citations, layer2 } = detail;
  const layer1Tone = LAYER1_TONES[claim.layer1_status] ?? "bg-canvas text-muted border border-line";
  const decompEntries = decompositionEntries(claim.decomposition);

  return (
    <section className="space-y-6 animate-fade-up">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <Link
            to="/queue"
            className="text-xs font-medium text-muted transition-colors hover:text-ink"
          >
            ← Back to queue
          </Link>
          <h1 className="mt-2 break-all font-mono text-lg font-semibold text-ink">{claim.claim_id}</h1>
          <p className="mt-1 font-mono text-xs text-muted">
            {claim.jurisdiction} · pillar {claim.pillar_id} · {claim.clause_id} · {claim.regime}
          </p>
        </div>
        <span className={["rie-badge", layer1Tone].join(" ")}>layer1 · {claim.layer1_status}</span>
      </header>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <div>
            <h2 className="font-serif text-xl text-ink">Source spans</h2>
            <p className="mt-1 text-xs leading-relaxed text-muted">
              Evidence highlighted at exact offsets. Text materialised deterministically — never authored by the model.
            </p>
          </div>
          <SpanViewer
            citations={citations}
            activeSpanId={activeSpanId}
            onSelect={(c: CitationDTO) => setActiveSpanId(c.span_id)}
          />
          {claim.evidence_spans.length > 0 && (
            <Panel title={`Evidence spans (${claim.evidence_spans.length})`}>
              <ul className="divide-y divide-line">
                {claim.evidence_spans.map((s) => (
                  <li key={s.span_id}>
                    <button
                      type="button"
                      onClick={() => setActiveSpanId(s.span_id)}
                      className={[
                        "block w-full px-4 py-2.5 text-left font-mono text-[11px] transition-colors hover:bg-canvas",
                        s.span_id === activeSpanId ? "bg-coverage-absent-bg text-ink" : "text-muted",
                      ].join(" ")}
                    >
                      {s.span_id} · {s.role} · {s.char_start}–{s.char_end}
                    </button>
                  </li>
                ))}
              </ul>
            </Panel>
          )}
        </div>

        <div className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <span className="rie-badge bg-ink text-white">ind {claim.indicator_id}</span>
            <span className="rie-badge border border-line bg-canvas text-muted">
              {claim.clause_pattern}
            </span>
            {claim.model_confidence != null && (
              <span className="rie-badge border border-line bg-canvas font-mono text-muted">
                conf {claim.model_confidence.toFixed(2)}
              </span>
            )}
          </div>

          <Panel title="Decomposition">
            <dl className="divide-y divide-line">
              {decompEntries.length === 0 && (
                <div className="px-4 py-4 text-xs text-muted">Empty decomposition</div>
              )}
              {decompEntries.map(([k, v]) => (
                <div key={k} className="grid grid-cols-[7rem_1fr] gap-3 px-4 py-3">
                  <dt className="font-mono text-[10px] uppercase tracking-wide text-muted">{k}</dt>
                  <dd className="text-sm leading-relaxed text-ink whitespace-pre-wrap break-words">
                    {renderValue(v)}
                  </dd>
                </div>
              ))}
            </dl>
          </Panel>

          {layer2 && (
            <div className="overflow-hidden rounded-xl border border-coverage-absent-border bg-coverage-absent-bg">
              <header className="flex items-center justify-between gap-2 border-b border-coverage-absent-border px-4 py-3">
                <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-coverage-absent-text">
                  Layer-2 recommendation
                </span>
                {layer2.human_confirmation_required && (
                  <span className="rie-badge bg-coverage-absent-solid text-white">
                    Human confirmation required
                  </span>
                )}
              </header>
              <div className="space-y-3 px-4 py-4 text-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rie-badge bg-ink text-white">{layer2.recommended_band}</span>
                  <span className="font-mono text-xs text-coverage-absent-text">
                    indicator {layer2.indicator_id}
                  </span>
                </div>
                <p className="leading-relaxed text-ink whitespace-pre-wrap">{layer2.rationale}</p>
                {layer2.open_questions.length > 0 && (
                  <ul className="list-inside list-disc space-y-1 text-sm text-muted">
                    {layer2.open_questions.map((q) => (
                      <li key={q}>{q}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}

          <Panel
            title="Verification gates"
            action={
              verification ? (
                <span
                  className={[
                    "rie-badge",
                    verification.status === "verified"
                      ? "bg-gate-pass-bg text-gate-pass-text"
                      : verification.status === "flagged"
                        ? "bg-gate-flag-bg text-gate-flag-text"
                        : "bg-gate-fail-bg text-gate-fail-text",
                  ].join(" ")}
                >
                  {verification.status}
                </span>
              ) : undefined
            }
          >
            {!verification && (
              <div className="px-4 py-4 text-xs text-muted">No verification report attached yet.</div>
            )}
            {verification && (
              <ul className="divide-y divide-line">
                {verification.gates.map((g, idx) => (
                  <li key={`${g.gate}-${idx}`} className="flex items-start gap-3 px-4 py-3">
                    <GateBadge gate={g} />
                    <div className="min-w-0 flex-1 text-xs text-muted">
                      <div className="break-words font-mono">{g.detail || (g.passed ? "passed" : "failed")}</div>
                      <div className="mt-0.5 text-[10px] opacity-70">{g.ran_at}</div>
                    </div>
                  </li>
                ))}
                {verification.failure_reasons.length > 0 && (
                  <li className="bg-gate-fail-bg px-4 py-3 text-xs text-gate-fail-text">
                    {verification.failure_reasons.join("; ")}
                  </li>
                )}
              </ul>
            )}
          </Panel>

          {reviewsQuery.data && reviewsQuery.data.items.length > 0 && (
            <Panel title={`Prior reviews (${reviewsQuery.data.total})`}>
              <ul className="divide-y divide-line">
                {reviewsQuery.data.items.map((r, idx) => (
                  <li key={`${r.reviewer}-${r.decided_at}-${idx}`} className="px-4 py-3 text-xs">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono font-semibold text-ink">{r.reviewer}</span>
                      <span
                        className={[
                          "rie-badge",
                          r.decision === "accept"
                            ? "bg-gate-pass-bg text-gate-pass-text"
                            : r.decision === "correct"
                              ? "bg-gate-flag-bg text-gate-flag-text"
                              : "bg-gate-fail-bg text-gate-fail-text",
                        ].join(" ")}
                      >
                        {r.decision}
                      </span>
                      {r.corrected_score && (
                        <span className="rie-badge border border-line bg-canvas font-mono text-muted">
                          score {r.corrected_score}
                        </span>
                      )}
                      <span className="ml-auto font-mono text-muted">{r.decided_at}</span>
                    </div>
                    {r.note && <p className="mt-2 whitespace-pre-wrap text-muted">{r.note}</p>}
                  </li>
                ))}
              </ul>
            </Panel>
          )}

          {lastReview && (
            <div className="rounded-xl border border-gate-pass-bg bg-gate-pass-bg px-4 py-3 font-mono text-xs text-gate-pass-text">
              Recorded {lastReview.decision} by {lastReview.reviewer}
            </div>
          )}
        </div>
      </div>

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
