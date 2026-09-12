#!/usr/bin/env bun
/**
 * scripts/dev-all.mjs - Unified Launcher for AstroFlow AI
 * 
 * Boots both:
 *   1. Python Edge Vision & Telemetry Hub (Port 8080)
 *   2. Next.js Avionics Console (Port 3000)
 * Handles graceful shutdown on SIGINT / CTRL+C.
 */

import { spawn, execSync } from "child_process";
import fs from "fs";
import path from "path";

const isWin = process.platform === "win32";

// Locate Python inside .venv
let pythonPath = isWin
  ? path.resolve(process.cwd(), ".venv", "Scripts", "python.exe")
  : path.resolve(process.cwd(), ".venv", "bin", "python");

if (!fs.existsSync(pythonPath)) {
  pythonPath = isWin ? "python" : "python3";
}

console.log("\x1b[36m%s\x1b[0m", "=======================================================");
console.log("\x1b[32m%s\x1b[0m", "🚀  ASTROFLOW AI — UNIFIED EDGE LAUNCHER");
console.log("\x1b[36m%s\x1b[0m", "=======================================================");
console.log(`[SYS] Python Runtime: ${pythonPath}`);
console.log(`[SYS] Next.js Console: Port 3000`);
console.log(`[SYS] AI Edge Hub:     Port 8080\n`);

// 1. Spawn Python Edge Server
const aiProcess = spawn(
  pythonPath,
  ["-m", "uvicorn", "ai_engine.server:app", "--host", "0.0.0.0", "--port", "8080"],
  {
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, PYTHONPATH: "." },
  }
);

aiProcess.stdout.on("data", (data) => {
  const text = data.toString().trim();
  if (text) {
    console.log(`\x1b[32m[AI-HUB]\x1b[0m ${text}`);
  }
});

aiProcess.stderr.on("data", (data) => {
  const text = data.toString().trim();
  // Filter noisy absl/tflite delegate warnings
  if (text && !text.includes("inference_feedback_manager") && !text.includes("InitializeLog")) {
    console.log(`\x1b[33m[AI-LOG]\x1b[0m ${text}`);
  }
});

// 2. Spawn Next.js Dev Server
const nextProcess = spawn(
  isWin ? "bun.exe" : "bun",
  ["run", "dev"],
  {
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, API_PROXY_TARGET: "http://localhost:8080" },
  }
);

nextProcess.stdout.on("data", (data) => {
  const text = data.toString().trim();
  if (text) {
    console.log(`\x1b[34m[CONSOLE]\x1b[0m ${text}`);
  }
});

nextProcess.stderr.on("data", (data) => {
  const text = data.toString().trim();
  if (text) {
    console.log(`\x1b[35m[CONSOLE-ERR]\x1b[0m ${text}`);
  }
});

// Robust process tree termination (prevents lingering ports on Windows & Linux)
function killTree(proc) {
  if (!proc || !proc.pid) return;
  try {
    if (isWin) {
      execSync(`taskkill /F /T /PID ${proc.pid}`, { stdio: "ignore" });
    } else {
      process.kill(-proc.pid, "SIGTERM");
    }
  } catch {
    try {
      proc.kill("SIGKILL");
    } catch {}
  }
}

function cleanup() {
  console.log("\n\x1b[31m[SYS] Shutting down AstroFlow AI services...\x1b[0m");
  killTree(aiProcess);
  killTree(nextProcess);
  process.exit(0);
}

process.on("SIGINT", cleanup);
process.on("SIGTERM", cleanup);
