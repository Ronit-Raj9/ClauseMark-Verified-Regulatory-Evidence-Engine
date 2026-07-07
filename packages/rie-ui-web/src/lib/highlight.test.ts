import { describe, expect, it } from "vitest";

import { renderHighlight, type HighlightSpan } from "./highlight";

const text = "The quick brown fox jumps over the lazy dog.";

describe("renderHighlight", () => {
  it("returns nothing for empty text", () => {
    expect(renderHighlight("", [{ id: "a", start: 0, end: 3 }])).toEqual([]);
  });

  it("returns a single plain segment when there are no spans", () => {
    const segs = renderHighlight(text, []);
    expect(segs).toHaveLength(1);
    expect(segs[0]).toMatchObject({ text, highlighted: false, spanIds: [] });
  });

  it("reconstructs the original text from concatenated segments", () => {
    const spans: HighlightSpan[] = [
      { id: "a", start: 4, end: 9 }, // "quick"
      { id: "b", start: 16, end: 19 }, // "fox"
    ];
    const segs = renderHighlight(text, spans);
    expect(segs.map((s) => s.text).join("")).toBe(text);
  });

  it("highlights a single span with correct offsets and id", () => {
    const segs = renderHighlight(text, [{ id: "a", start: 4, end: 9 }]);
    const hl = segs.filter((s) => s.highlighted);
    expect(hl).toHaveLength(1);
    expect(hl[0]).toMatchObject({ text: "quick", start: 4, end: 9, spanIds: ["a"] });
  });

  it("emits highlighted and plain segments in document order", () => {
    const segs = renderHighlight(text, [
      { id: "b", start: 16, end: 19 }, // "fox" — given out of order
      { id: "a", start: 4, end: 9 }, // "quick"
    ]);
    expect(segs.map((s) => s.highlighted)).toEqual([false, true, false, true, false]);
    expect(segs.filter((s) => s.highlighted).map((s) => s.text)).toEqual(["quick", "fox"]);
  });

  it("merges overlapping spans into one run, unioning ids", () => {
    const segs = renderHighlight(text, [
      { id: "a", start: 4, end: 12 },
      { id: "b", start: 9, end: 15 },
    ]);
    const hl = segs.filter((s) => s.highlighted);
    expect(hl).toHaveLength(1);
    expect(hl[0]).toMatchObject({ start: 4, end: 15 });
    expect([...hl[0]!.spanIds].sort()).toEqual(["a", "b"]);
  });

  it("merges abutting spans (prev.end === next.start)", () => {
    const segs = renderHighlight(text, [
      { id: "a", start: 4, end: 9 },
      { id: "b", start: 9, end: 15 },
    ]);
    const hl = segs.filter((s) => s.highlighted);
    expect(hl).toHaveLength(1);
    expect(hl[0]).toMatchObject({ start: 4, end: 15, text: "quick brown" });
  });

  it("clamps out-of-range offsets to the text bounds", () => {
    const segs = renderHighlight("hi", [{ id: "a", start: 1, end: 999 }]);
    const hl = segs.filter((s) => s.highlighted);
    expect(hl).toHaveLength(1);
    expect(hl[0]).toMatchObject({ text: "i", start: 1, end: 2 });
  });

  it("drops empty and inverted spans", () => {
    const segs = renderHighlight(text, [
      { id: "empty", start: 5, end: 5 },
      { id: "inverted", start: 10, end: 4 },
    ]);
    expect(segs).toHaveLength(1);
    expect(segs[0]!.highlighted).toBe(false);
  });

  it("does not produce HTML — segment text is verbatim, including unsafe chars", () => {
    const unsafe = 'a <script>"&" </script> b';
    const segs = renderHighlight(unsafe, [{ id: "x", start: 2, end: 23 }]);
    // The library returns raw text; escaping is the renderer's job. Verify the
    // dangerous substring is passed through untouched (so React can text-node it).
    expect(segs.map((s) => s.text).join("")).toBe(unsafe);
    expect(segs.some((s) => s.text.includes("<script>"))).toBe(true);
  });

  it("highlights a span that covers the entire text", () => {
    const segs = renderHighlight("abc", [{ id: "a", start: 0, end: 3 }]);
    expect(segs).toEqual([
      { text: "abc", start: 0, end: 3, highlighted: true, spanIds: ["a"] },
    ]);
  });
});
