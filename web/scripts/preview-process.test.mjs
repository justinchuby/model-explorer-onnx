import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import test from "node:test";
import { stopPreview } from "./preview-process.mjs";

const child = (code) => spawn(process.execPath, ["-e", code], { stdio: "ignore" });

test("cleanup of an exited preview preserves the original failure", { timeout: 2000 }, async () => {
  const preview = child("process.exit(7)");
  await once(preview, "exit");
  const original = new Error("Preview exited: 7");
  await assert.rejects(async () => {
    try {
      throw original;
    } finally {
      await stopPreview(preview);
    }
  }, error => error === original);
});

test("cleanup terminates a running preview", { timeout: 2000 }, async () => {
  const preview = child("setInterval(() => {}, 1000)");
  await once(preview, "spawn");
  await stopPreview(preview);
  assert.equal(preview.signalCode, "SIGTERM");
});

test("cleanup returns for a preview already terminated by a signal", { timeout: 2000 }, async () => {
  const preview = child("setInterval(() => {}, 1000)");
  await once(preview, "spawn");
  const exited = once(preview, "exit");
  preview.kill();
  await exited;
  assert.equal(preview.exitCode, null);
  await stopPreview(preview);
});

test("cleanup returns when spawning the preview failed", { timeout: 2000 }, async () => {
  const preview = spawn("/nonexistent/onnx-preview-test", [], { stdio: "ignore" });
  await once(preview, "error");
  await stopPreview(preview);
});
