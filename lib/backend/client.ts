import { backendConfig } from "./config";
import type { ApiResponse, StateSnapshot, TransitionRequest } from "./types";

// Thin HTTP client for the AstroFlow-AI backend API.
// Wraps fetch so callers never deal with URL construction, JSON parsing, or
// error mapping directly.

function buildUrl(path: string): string {
  const base = `${backendConfig.api.host}:${backendConfig.api.port}${backendConfig.api.basePath}`;
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

function newRequestId(): string {
  return `req_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<ApiResponse<T>> {
  const response = await fetch(buildUrl(path), {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!response.ok) {
    return {
      ok: false,
      data: null,
      error: {
        code: `HTTP_${response.status}`,
        message: response.statusText || "Request failed",
      },
      meta: { requestId: newRequestId(), timestamp: new Date().toISOString() },
    };
  }

  return (await response.json()) as ApiResponse<T>;
}

export const apiClient = {
  getState: () => request<StateSnapshot>("state", { method: "GET" }),
  transition: (body: TransitionRequest) =>
    request<StateSnapshot>("state/transition", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};

export type { ApiResponse, StateSnapshot, TransitionRequest };