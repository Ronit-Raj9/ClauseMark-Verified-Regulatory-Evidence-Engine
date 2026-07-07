import { useMemo } from "react";

import type { CitationDTO } from "@/types";

interface Props {
  citations: CitationDTO[];
  activeSpanId?: string | null;
  onSelect?: (citation: CitationDTO) => void;
}

export default function SpanViewer({ citations, activeSpanId, onSelect }: Props) {
  const grouped = useMemo(() => {
    const map = new Map<string, CitationDTO[]>();
    for (const c of citations) {
      const key = `${c.doc_id}::${c.element_id}`;
      const bucket = map.get(key) ?? [];
      bucket.push(c);
      map.set(key, bucket);
    }
    return [...map.entries()];
  }, [citations]);

  if (citations.length === 0) {
    return (
      <div className="rie-panel border-dashed px-6 py-12 text-center text-sm text-muted">
        No citations attached to this claim.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {grouped.map(([key, items]) => {
        const first = items[0];
        if (!first) return null;
        return (
          <article key={key} className="rie-panel overflow-hidden">
            <header className="flex items-center justify-between border-b border-line bg-canvas/60 px-4 py-3 font-mono text-xs text-muted">
              <span>
                doc <span className="text-ink">{first.doc_id}</span> · element{" "}
                <span className="text-ink">{first.element_id}</span>
              </span>
              <span>{items.length} span{items.length === 1 ? "" : "s"}</span>
            </header>
            <div className="space-y-3 p-4">
              {items.map((c) => {
                const active = c.span_id === activeSpanId;
                return (
                  <button
                    key={c.span_id}
                    type="button"
                    onClick={() => onSelect?.(c)}
                    className={[
                      "block w-full rounded-lg border p-4 text-left transition-all duration-200",
                      active
                        ? "border-coverage-absent-border bg-coverage-absent-bg shadow-float"
                        : "border-line bg-surface hover:border-ink/15 hover:shadow-float",
                    ].join(" ")}
                  >
                    <div className="mb-2 flex flex-wrap items-center gap-2 font-mono text-[11px] text-muted">
                      <span className="rie-badge bg-canvas text-muted">{c.role}</span>
                      <span>{c.span_id}</span>
                      <span>
                        {c.char_start}–{c.char_end}
                      </span>
                    </div>
                    <p className="text-sm leading-relaxed text-ink whitespace-pre-wrap">
                      <span className="rie-span-highlight">{c.text || "(empty span)"}</span>
                    </p>
                  </button>
                );
              })}
            </div>
          </article>
        );
      })}
    </div>
  );
}
