// Backend service configuration for AstroFlow-AI.
// Centralizes environment-driven settings so the API layer can read a single
// source of truth at runtime without reaching for process.env everywhere.

export const backendConfig = {
  api: {
    host: process.env.API_HOST ?? "0.0.0.0",
    port: Number(process.env.API_PORT ?? 8080),
    basePath: process.env.API_BASE_PATH ?? "/api/v1",
  },
  stream: {
    reconnectDelayMs: Number(process.env.STREAM_RECONNECT_DELAY_MS ?? 2000),
    maxReconnectAttempts: Number(process.env.STREAM_MAX_RECONNECT_ATTEMPTS ?? 10),
  },
  storage: {
    logDir: process.env.LOG_DIR ?? "./logs",
    recordingDir: process.env.RECORDING_DIR ?? "./recordings",
  },
} as const;

export type BackendConfig = typeof backendConfig;