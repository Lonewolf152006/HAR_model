#!/usr/bin/env bun
import { spawn } from "child_process";
import fs from "fs";
import path from "path";

const isWin = process.platform === "win32";
let py = isWin
  ? path.resolve(process.cwd(), ".venv", "Scripts", "python.exe")
  : path.resolve(process.cwd(), ".venv", "bin", "python");

if (!fs.existsSync(py)) {
  py = isWin ? "python" : "python3";
}

const child = spawn(py, ["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], {
  stdio: "inherit",
  env: { ...process.env, PYTHONPATH: "." },
});

child.on("exit", (code) => {
  process.exit(code || 0);
});
