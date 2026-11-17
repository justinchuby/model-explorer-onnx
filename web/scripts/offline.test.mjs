import assert from "node:assert/strict";
import { getEventListeners } from "node:events";
import { readFile } from "node:fs/promises";
import { setImmediate } from "node:timers/promises";
import test from "node:test";
import ts from "typescript";

// Exercise the browser-independent lifecycle code without a bundler or network.
const source = await readFile(new URL("../src/offline.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
});
const { waitForOfflineControl } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
const fixture = () => {
  const active = Object.assign(new EventTarget(), { state: "activated" });
  const registration = { active, installing: null, waiting: null };
  const container = Object.assign(new EventTarget(), {
    controller: active,
    register: async () => registration,
  });
  // The lifecycle must not rely on ready, which can remain pending indefinitely.
  Object.defineProperty(container, "ready", { get() { throw new Error("Do not wait for ready"); } });
  return { container, registration };
};
const prepare = (container) => waitForOfflineControl(container, "/sw.js", "/", 40);

test("an already controlled page succeeds without serviceWorker.ready", async () => {
  const { container } = fixture();
  await prepare(container);
  assert.equal(getEventListeners(container, "controllerchange").length, 0);
});

test("an upgrade waits for both new activation and control", async () => {
  const { container, registration } = fixture();
  const installing = Object.assign(new EventTarget(), { state: "installing" });
  registration.installing = installing;
  let done = false;
  const pending = waitForOfflineControl(container, "/sw.js", "/", 1000).then(() => { done = true; });
  await setImmediate();
  assert.equal(done, false);
  installing.state = "activated";
  registration.active = installing;
  installing.dispatchEvent(new Event("statechange"));
  await setImmediate();
  assert.equal(done, false);
  container.controller = installing;
  container.dispatchEvent(new Event("controllerchange"));
  await pending;
  assert.equal(getEventListeners(installing, "statechange").length, 0);
  assert.equal(getEventListeners(container, "controllerchange").length, 0);
});

test("failed precaching rejects immediately and removes its listener", async () => {
  const { container, registration } = fixture();
  const installing = Object.assign(new EventTarget(), { state: "installing" });
  registration.active = null;
  registration.installing = installing;
  const pending = prepare(container);
  await setImmediate();
  installing.state = "redundant";
  installing.dispatchEvent(new Event("statechange"));
  await assert.rejects(pending, /installation failed/);
  assert.equal(getEventListeners(installing, "statechange").length, 0);
});

test("registration timeout allows startup to fall back", async () => {
  const { container } = fixture();
  container.register = () => new Promise(() => {});
  await assert.rejects(prepare(container), /timed out/);
});

test("activation timeout removes its listener", async () => {
  const { container, registration } = fixture();
  registration.installing = Object.assign(new EventTarget(), { state: "installing" });
  await assert.rejects(prepare(container), /timed out/);
  assert.equal(getEventListeners(registration.installing, "statechange").length, 0);
});

test("hard reload with an active worker but no controller times out", async () => {
  const { container } = fixture();
  container.controller = null;
  await assert.rejects(prepare(container), /timed out/);
  assert.equal(getEventListeners(container, "controllerchange").length, 0);
});

test("no active or installing worker rejects instead of waiting for ready", async () => {
  const { container, registration } = fixture();
  registration.active = null;
  await assert.rejects(prepare(container), /No active/);
});
