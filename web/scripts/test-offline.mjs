import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";
import { stopPreview } from "./preview-process.mjs";

// A one-node Identity ONNX model. No Python installation is needed by the test.
const model = Buffer.from(
  "CA46QwoQCgF4EgF5IghJZGVudGl0eRINb2ZmbGluZS1zbW9rZVoPCgF4EgoKCAgBEgQKAggBYg8KAXkSCgoICAESBAoCCAFCAhAc",
  "base64",
);
const preview = spawn(process.execPath, ["node_modules/vite/bin/vite.js", "preview", "--host", "127.0.0.1", "--port", "4175", "--strictPort"], {
  stdio: ["ignore", "pipe", "inherit"],
});
const profile = await mkdtemp(join(tmpdir(), "onnx-offline-test-"));
let browser;
const launchOptions = {
  headless: true,
  executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH || undefined,
};
try {
  await new Promise((resolve, reject) => {
    preview.stdout.on("data", (data) => {
      if (data.toString().includes("http://127.0.0.1:4175")) resolve();
    });
    preview.once("error", reject);
    preview.once("exit", (code) => reject(new Error(`Preview exited: ${code}`)));
  });
  browser = await chromium.launchPersistentContext(profile, launchOptions);
  const context = browser;
  const recordWorkers = () => {
    window.visualizerRequests = [];
    window.visualizerResults = [];
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        this.addEventListener("message", ({ data }) => {
          if (data.eventType === 1) window.visualizerResults.push(data);
        });
      }
      postMessage(data, ...args) {
        if (data.eventType === 0) window.visualizerRequests.push(data);
        return super.postMessage(data, ...args);
      }
    };
  };
  await context.addInitScript(recordWorkers);
  const page = await context.newPage();
  await page.goto("http://127.0.0.1:4175/");
  // No model is supplied: opening the page alone must warm all dependencies.
  await page.waitForFunction(() => document.querySelector("#ui-status")?.textContent.includes("Offline resources cached"), null, { timeout: 180_000 });
  assert.equal(await page.locator(".model-item").count(), 0);
  const urls = await page.evaluate(async () => {
    const entries = await Promise.all((await caches.keys()).map(async key => (await (await caches.open(key)).keys()).map(request => request.url)));
    return entries.flat();
  });
  for (const resource of ["pyodide.asm.wasm", "python_stdlib.zip", "onnx_weekly-", "onnx_ir-", "typing_extensions-", "pypi.org/simple/onnx-ir/", "FontRegular.png", "FontRegular.json", "/worker.js"]) {
    assert(urls.some(url => url.includes(resource)), `Missing offline resource: ${resource}`);
  }
  console.log("PASS: fresh visit eagerly initializes Python and caches runtime, wheels, metadata, and visualizer assets.");
  await context.close();

  // Restart the browser offline: HTTP memory caches cannot satisfy this test.
  browser = await chromium.launchPersistentContext(profile, { ...launchOptions, offline: true });
  await browser.addInitScript(recordWorkers);
  const offlinePage = await browser.newPage();
  const errors = [];
  offlinePage.on("pageerror", error => errors.push(error.message));
  await offlinePage.goto("http://127.0.0.1:4175/");
  // Submit while Python starts; the file must wait rather than be discarded.
  await offlinePage.locator("#model-file").setInputFiles({ name: "smoke.onnx", mimeType: "application/octet-stream", buffer: model });
  await offlinePage.waitForFunction(() => document.querySelector(".model-name")?.textContent === "smoke.onnx", null, { timeout: 120_000 }).catch(async error => {
    throw new Error(`Offline conversion failed: ${await offlinePage.locator("#ui-status").textContent()}`, { cause: error });
  });
  await offlinePage.waitForFunction(() => window.visualizerRequests.length > 0, null, { timeout: 30_000 });
  assert.equal(await offlinePage.evaluate(() => window.visualizerRequests.at(-1).groupNodeChildrenCountThreshold), 1_000_000);
  console.log("PASS: offline new page initializes and converts a model selected during startup; theme preserves the 1,000,000 threshold.");

  // Exercise the actual graph-processing worker beyond its old 1,000-node limit.
  await offlinePage.locator("model-explorer-visualizer").evaluate(element => {
    window.visualizerResults.length = 0;
    const nodes = Array.from({ length: 1001 }, (_, i) => ({
      id: `node-${i}`, label: "Identity", namespace: "",
      incomingEdges: i ? [{ sourceNodeId: `node-${i - 1}`, sourceNodeOutputId: "0", targetNodeInputId: "0" }] : [],
    }));
    element.graphCollections = [{ label: "large", graphs: [{ id: "large", nodes }] }];
  });
  await offlinePage.waitForFunction(() => window.visualizerResults.some(result => result.modelGraph.id === "large"), null, { timeout: 60_000 });
  const labels = await offlinePage.evaluate(() => window.visualizerResults.find(result => result.modelGraph.id === "large").modelGraph.nodes.map(node => node.label));
  assert(!labels.some(label => /^section_/.test(label)));
  assert.deepEqual(errors, []);
  console.log("PASS: a 1,001-node graph processes offline without artificial sections or page errors.");

  // A hard reload may have an active SW without page control. Even if control
  // never arrives, the app must stop waiting and initialize Python normally.
  const fallbackPage = await browser.newPage();
  await fallbackPage.addInitScript(() => {
    Object.defineProperty(navigator.serviceWorker, "controller", { get: () => null });
  });
  await fallbackPage.goto("http://127.0.0.1:4175/");
  await fallbackPage.locator("#model-file").setInputFiles({ name: "fallback.onnx", mimeType: "application/octet-stream", buffer: model });
  await fallbackPage.waitForFunction(() => document.querySelector(".model-name")?.textContent === "fallback.onnx", null, { timeout: 60_000 });
  assert.match(await fallbackPage.locator("#debug-log").textContent(), /Offline cache setup timed out/);
  console.log("PASS: missing service-worker control times out and queued conversion still completes.");
} finally {
  await browser?.close();
  await rm(profile, { recursive: true, force: true });
  await stopPreview(preview);
}
