// Pure, dependency-free highlight engine.
//
// `renderHighlight(text, spans)` slices a document string into ordered,
// non-overlapping segments, each flagged as highlighted or not. Overlapping
// or adjacent highlighted spans are merged so the caller never has to reason
// about nesting. This module is deliberately framework-agnostic and is the
// unit-tested keystone behind <SpanHighlight /> / <SpanViewer />.
//
// It NEVER produces HTML — escaping is the renderer's concern. It only returns
// data. (Keeping markup out of here is what makes XSS-by-offset impossible:
// the React layer always renders segment.text as a text node.)

/** A region of the source document to highlight, by half-open char offsets. */
export interface HighlightSpan {
  /** Stable identifier for the span (e.g. "doc12#4120-4215"). */
  readonly id: string;
  /** Inclusive start offset into `text`. */
  readonly start: number;
  /** Exclusive end offset into `text`. */
  readonly end: number;
}

/** One contiguous slice of the source document. */
export interface HighlightSegment {
  /** Verbatim substring of the source `text`. */
  readonly text: string;
  /** Inclusive start offset into the source `text`. */
  readonly start: number;
  /** Exclusive end offset into the source `text`. */
  readonly end: number;
  /** Whether this slice falls inside (the union of) the input spans. */
  readonly highlighted: boolean;
  /** IDs of every input span that overlaps this slice (highlighted only). */
  readonly spanIds: readonly string[];
}

interface NormalizedSpan {
  start: number;
  end: number;
  ids: string[];
}

/**
 * Clamp a raw span to the bounds of `text` and drop empties / inverted ranges.
 * Out-of-range offsets are clamped rather than dropped so a span that extends
 * one char past the end still highlights what it legitimately covers.
 */
function clampSpans(textLength: number, spans: readonly HighlightSpan[]): NormalizedSpan[] {
  const out: NormalizedSpan[] = [];
  for (const s of spans) {
    const start = Math.max(0, Math.min(s.start, textLength));
    const end = Math.max(0, Math.min(s.end, textLength));
    if (end <= start) continue; // empty or inverted → nothing to highlight
    out.push({ start, end, ids: [s.id] });
  }
  return out;
}

/**
 * Merge overlapping or touching spans into disjoint runs, unioning their IDs.
 * Input need not be sorted.
 */
function mergeSpans(spans: NormalizedSpan[]): NormalizedSpan[] {
  if (spans.length === 0) return [];
  const sorted = [...spans].sort((a, b) => a.start - b.start || a.end - b.end);
  const merged: NormalizedSpan[] = [];
  for (const span of sorted) {
    const last = merged[merged.length - 1];
    // `<=` so abutting spans (prev.end === next.start) coalesce into one run.
    if (last && span.start <= last.end) {
      last.end = Math.max(last.end, span.end);
      for (const id of span.ids) if (!last.ids.includes(id)) last.ids.push(id);
    } else {
      merged.push({ start: span.start, end: span.end, ids: [...span.ids] });
    }
  }
  return merged;
}

/**
 * Split `text` into an ordered list of segments. Highlighted segments
 * correspond to the merged union of `spans`; the gaps between them are emitted
 * as plain (un-highlighted) segments. Concatenating every `segment.text`
 * reproduces `text` exactly.
 */
export function renderHighlight(
  text: string,
  spans: readonly HighlightSpan[],
): HighlightSegment[] {
  if (text.length === 0) return [];
  const merged = mergeSpans(clampSpans(text.length, spans));

  if (merged.length === 0) {
    return [{ text, start: 0, end: text.length, highlighted: false, spanIds: [] }];
  }

  const segments: HighlightSegment[] = [];
  let cursor = 0;
  for (const run of merged) {
    if (run.start > cursor) {
      segments.push({
        text: text.slice(cursor, run.start),
        start: cursor,
        end: run.start,
        highlighted: false,
        spanIds: [],
      });
    }
    segments.push({
      text: text.slice(run.start, run.end),
      start: run.start,
      end: run.end,
      highlighted: true,
      spanIds: run.ids,
    });
    cursor = run.end;
  }
  if (cursor < text.length) {
    segments.push({
      text: text.slice(cursor),
      start: cursor,
      end: text.length,
      highlighted: false,
      spanIds: [],
    });
  }
  return segments;
}
