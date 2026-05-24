import { useMemo } from "react";

import type { CitationDTO } from "@/types";

// Renders the source text for a claim with the verified span highlighted.
//
// Backend invariant: rie-api materialises the citation text deterministically
// (rie_api.routes.claims._materialise_citations). The UI NEVER fabricates
// surrounding context — we render exactly what was returned.

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
      <div className="border border-dashed border-slate-300 rounded-md p-6 text-sm text-slate-500">
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
          <article key={key} className="bg-white border border-slate-200 rounded-md">
            <header className="px-4 py-2 border-b border-slate-200 bg-slate-50 text-xs font-mono text-slate-600 flex items-center justify-between">
              <span>
                doc <span className="text-slate-900">{first.doc_id}</span> ·
                element <span className="text-slate-900">{first.element_id}</span>
              </span>
              <span className="text-slate-400">{items.length} span(s)</span>
            </header>
            <div className="px-4 py-3 space-y-3">
              {items.map((c) => {
                const active = c.span_id === activeSpanId;
                return (
                  <button
                    key={c.span_id}
                    type="button"
                    onClick={() => onSelect?.(c)}
                    className={[
                      "block w-full text-left rounded-md border p-3 transition-colors",
                      active
                        ? "border-yellow-500 bg-yellow-50"
                        : "border-slate-200 hover:border-slate-400",
                    ].join(" ")}
                  >
                    <div className="text-[11px] font-mono text-slate-500 mb-1 flex items-center gap-2">
                      <span className="px-1.5 py-0.5 rounded bg-slate-200 text-slate-700">
                        {c.role}
                      </span>
                      <span>span {c.span_id}</span>
                      <span>
                        offsets {c.char_start}–{c.char_end}
                      </span>
                    </div>
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">
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
