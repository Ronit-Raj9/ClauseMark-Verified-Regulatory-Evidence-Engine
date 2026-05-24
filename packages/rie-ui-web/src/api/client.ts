// Typed thin wrapper over fetch — never imports orchestration / persistence.
// Base URL comes from VITE_API_BASE; empty means "same origin" which works
// in dev (vite proxy) and in prod (when the SPA is served by rie-api itself
// or behind a reverse proxy).

import type {
  ClaimDetailResponse,
  ClaimListResponse,
  CoverageListResponse,
  EvidencePackage,
  HealthResponse,
  Layer1Status,
  PillarDetailResponse,
  PillarsListResponse,
  ReviewDTO,
  ReviewListResponse,
  ReviewSubmit,
  RunRequest,
  RunResponse,
  RunStatusResponse,
} from "@/types";

const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/+$/, "");

export class ApiError extends Error {
  public readonly status: number;
  public readonly payload: unknown;

  constructor(status: number, message: string, payload: unknown) {
    super(message);
    this.status = status;
    this.payload = payload;
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  init?: RequestInit & { query?: Record<string, string | number | boolean | undefined | null> },
): Promise<T> {
  const { query, ...rest } = init ?? {};
  const url = new URL(`${BASE}${path}`, window.location.origin);
  if (query) {
    for (const [k, v] of Object.entries(query)) {
      if (v === undefined || v === null || v === "") continue;
      url.searchParams.set(k, String(v));
    }
  }

  const res = await fetch(url.toString().replace(window.location.origin, BASE || ""), {
    headers: {
      "content-type": "application/json",
      accept: "application/json",
      ...(rest.headers ?? {}),
    },
    ...rest,
  });

  if (!res.ok) {
    let payload: unknown = null;
    try {
      payload = await res.json();
    } catch {
      payload = await res.text().catch(() => null);
    }
    const detail =
      payload && typeof payload === "object" && "detail" in payload
        ? String((payload as { detail: unknown }).detail)
        : res.statusText;
    throw new ApiError(res.status, `${res.status} ${detail}`, payload);
  }

  // 204 No Content
  if (res.status === 204) return undefined as unknown as T;
  return (await res.json()) as T;
}

// ── Health ─────────────────────────────────────────────────────────────────

export const health = {
  live: () => request<HealthResponse>("/healthz"),
  ready: () => request<HealthResponse>("/readyz"),
};

// ── Pillars ────────────────────────────────────────────────────────────────

export const pillars = {
  list: () => request<PillarsListResponse>("/v1/pillars"),
  get: (pillarId: string) => request<PillarDetailResponse>(`/v1/pillars/${encodeURIComponent(pillarId)}`),
};

// ── Claims ─────────────────────────────────────────────────────────────────

export const claims = {
  list: (params?: {
    jurisdiction?: string;
    pillar_id?: string;
    status?: Layer1Status;
  }) =>
    request<ClaimListResponse>("/v1/claims", {
      query: params,
    }),
  get: (claimId: string) =>
    request<ClaimDetailResponse>(`/v1/claims/${encodeURIComponent(claimId)}`),
  reviews: (claimId: string) =>
    request<ReviewListResponse>(`/v1/claims/${encodeURIComponent(claimId)}/reviews`),
};

// ── Coverage ───────────────────────────────────────────────────────────────

export const coverage = {
  list: (jurisdiction?: string) =>
    request<CoverageListResponse>("/v1/coverage", {
      query: { jurisdiction },
    }),
};

// ── Audit ──────────────────────────────────────────────────────────────────

export const audit = {
  package: (jurisdiction: string) =>
    request<EvidencePackage>(`/v1/audit/${encodeURIComponent(jurisdiction)}`),
};

// ── Reviews ────────────────────────────────────────────────────────────────

export const reviews = {
  submit: (body: ReviewSubmit) =>
    request<ReviewDTO>("/v1/reviews", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};

// ── Runs ───────────────────────────────────────────────────────────────────

export const runs = {
  trigger: (body: RunRequest) =>
    request<RunResponse>("/v1/runs", { method: "POST", body: JSON.stringify(body) }),
  status: (runId: string, jurisdiction?: string) =>
    request<RunStatusResponse>(`/v1/runs/${encodeURIComponent(runId)}`, {
      query: { jurisdiction },
    }),
};

export const apiBase = BASE;
