import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import type { ReviewSubmit, RunRequest } from "@/types";

// The client reads `window.location.origin` to resolve relative paths. We run
// under the default (node) test environment to stay dependency-light, so we
// install a minimal `window` shim instead of pulling in jsdom/happy-dom.
const hadWindow = "window" in globalThis;
beforeAll(() => {
  if (!hadWindow) {
    (globalThis as unknown as { window: unknown }).window = {
      location: { origin: "http://localhost" },
    };
  }
});
afterAll(() => {
  if (!hadWindow) {
    delete (globalThis as unknown as { window?: unknown }).window;
  }
});

// Imported after the window shim is declared (module init reads no window, but
// keep the import here so hoisting order is unambiguous to readers).
import { audit, claims, coverage, health, pillars, reviews, runs } from "./client";

// The client builds URLs with `new URL(path, window.location.origin)` then
// strips the origin so the final fetch arg is the API-relative path (BASE is
// "" under the test env). We capture every fetch call and assert the mapping.

interface Captured {
  url: string;
  method: string;
  body: unknown;
}

const calls: Captured[] = [];

function mockOk(json: unknown = {}): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push({
        url: String(input),
        method: (init?.method ?? "GET").toUpperCase(),
        body: init?.body ? JSON.parse(String(init.body)) : undefined,
      });
      return new Response(JSON.stringify(json), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }),
  );
}

beforeEach(() => {
  calls.length = 0;
  mockOk();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function last(): Captured {
  const c = calls[calls.length - 1];
  if (!c) throw new Error("no fetch call captured");
  return c;
}

describe("api client endpoint mapping", () => {
  it("health.live → GET /healthz", async () => {
    await health.live();
    expect(last().url).toBe("/healthz");
    expect(last().method).toBe("GET");
  });

  it("health.ready → GET /readyz", async () => {
    await health.ready();
    expect(last().url).toBe("/readyz");
  });

  it("pillars.list → GET /v1/pillars", async () => {
    await pillars.list();
    expect(last().url).toBe("/v1/pillars");
  });

  it("pillars.get encodes the id → GET /v1/pillars/:id", async () => {
    await pillars.get("6.4");
    expect(last().url).toBe("/v1/pillars/6.4");
  });

  it("claims.list with no params → GET /v1/claims", async () => {
    await claims.list();
    expect(last().url).toBe("/v1/claims");
  });

  it("claims.list forwards filters as query params", async () => {
    await claims.list({ jurisdiction: "SG", pillar_id: "6", status: "flagged" });
    const u = new URL(last().url, "http://x");
    expect(u.pathname).toBe("/v1/claims");
    expect(u.searchParams.get("jurisdiction")).toBe("SG");
    expect(u.searchParams.get("pillar_id")).toBe("6");
    expect(u.searchParams.get("status")).toBe("flagged");
  });

  it("claims.get → GET /v1/claims/:id", async () => {
    await claims.get("claim-1");
    expect(last().url).toBe("/v1/claims/claim-1");
  });

  it("claims.reviews → GET /v1/claims/:id/reviews", async () => {
    await claims.reviews("claim-1");
    expect(last().url).toBe("/v1/claims/claim-1/reviews");
  });

  it("coverage.list forwards jurisdiction", async () => {
    await coverage.list("SG");
    const u = new URL(last().url, "http://x");
    expect(u.pathname).toBe("/v1/coverage");
    expect(u.searchParams.get("jurisdiction")).toBe("SG");
  });

  it("coverage.list omits empty jurisdiction", async () => {
    await coverage.list();
    expect(last().url).toBe("/v1/coverage");
  });

  it("audit.package → GET /v1/audit/:jurisdiction", async () => {
    await audit.package("SG");
    expect(last().url).toBe("/v1/audit/SG");
  });

  it("reviews.submit → POST /v1/reviews with JSON body", async () => {
    const body: ReviewSubmit = {
      claim_id: "claim-1",
      reviewer: "alice",
      decision: "accept",
      note: "",
    };
    await reviews.submit(body);
    expect(last().url).toBe("/v1/reviews");
    expect(last().method).toBe("POST");
    expect(last().body).toMatchObject({ claim_id: "claim-1", decision: "accept" });
  });

  it("runs.trigger → POST /v1/runs with JSON body", async () => {
    const body: RunRequest = { jurisdiction: "SG", pillar_ids: ["6"] };
    await runs.trigger(body);
    expect(last().url).toBe("/v1/runs");
    expect(last().method).toBe("POST");
    expect(last().body).toMatchObject({ jurisdiction: "SG", pillar_ids: ["6"] });
  });

  it("runs.status → GET /v1/runs/:id with jurisdiction query", async () => {
    await runs.status("run-1", "SG");
    const u = new URL(last().url, "http://x");
    expect(u.pathname).toBe("/v1/runs/run-1");
    expect(u.searchParams.get("jurisdiction")).toBe("SG");
  });

  it("runs.resume → POST /v1/runs/:id/resume with decisions body", async () => {
    await runs.resume("run-1", { decisions: { "claim-1": "accept" } });
    expect(last().url).toBe("/v1/runs/run-1/resume");
    expect(last().method).toBe("POST");
    expect(last().body).toEqual({ decisions: { "claim-1": "accept" } });
  });
});

describe("api client error handling", () => {
  it("throws ApiError carrying status + detail on non-2xx", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(JSON.stringify({ detail: "nope" }), {
            status: 404,
            headers: { "content-type": "application/json" },
          }),
      ),
    );
    await expect(claims.get("missing")).rejects.toMatchObject({
      status: 404,
      message: expect.stringContaining("nope"),
    });
  });
});
