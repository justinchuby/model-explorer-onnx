import "ai-edge-model-explorer-visualizer";
import { createIcons, icons } from "lucide";
import "./style.css";

type WorkerStatusMessage = { type: "status"; message: string };
type WorkerReadyMessage = { type: "ready" };
type WorkerResultMessage = {
  type: "result";
  requestId: string;
  graphCollections: unknown[];
};
type WorkerErrorMessage = {
  type: "error";
  requestId?: string;
  message: string;
  traceback?: string;
};
type WorkerMessage =
  | WorkerStatusMessage
  | WorkerReadyMessage
  | WorkerResultMessage
  | WorkerErrorMessage;

type Graph = { id?: string };
type GraphCollection = { label: string; graphs: Graph[]; sourceUrl?: string };
type VisualizerElement = HTMLElement & {
  graphCollections: GraphCollection[];
  config?: Record<string, unknown>;
};

declare global {
  interface Window {
    modelExplorer?: { assetFilesBaseUrl?: string; workerScriptPath?: string };
  }
}

const appEl = document.getElementById("app");
const sidebarToggleBtn = document.getElementById("sidebar-toggle") as HTMLButtonElement;
const sidebarFabBtn = document.getElementById("sidebar-fab") as HTMLButtonElement;
const fileInput = document.getElementById("model-file") as HTMLInputElement;
const modelUrlInput = document.getElementById("model-url-input") as HTMLInputElement;
const loadUrlBtn = document.getElementById("load-url-btn") as HTMLButtonElement;
const copyShareBtn = document.getElementById("copy-share-btn") as HTMLButtonElement;
const debugToggleBtn = document.getElementById("debug-toggle") as HTMLButtonElement;
const sideHeaderEl = document.querySelector(".side-header") as HTMLElement;
const leftPanelEl = document.getElementById("left-panel") as HTMLElement;
const modelListEl = document.getElementById("model-list");
const debugLogEl = document.getElementById("debug-log");
const viewerShell = document.getElementById("viewer-shell");
const dropOverlay = document.getElementById("drop-overlay");

if (
  !appEl ||
  !sidebarToggleBtn ||
  !sidebarFabBtn ||
  !fileInput ||
  !modelUrlInput ||
  !loadUrlBtn ||
  !copyShareBtn ||
  !debugToggleBtn ||
  !sideHeaderEl ||
  !leftPanelEl ||
  !modelListEl ||
  !debugLogEl ||
  !viewerShell ||
  !dropOverlay
) {
  throw new Error("Missing required DOM elements.");
}

const meGlobal = window.modelExplorer;
if (meGlobal) {
  meGlobal.assetFilesBaseUrl = `${import.meta.env.BASE_URL}static_files`;
  meGlobal.workerScriptPath = `${import.meta.env.BASE_URL}worker.js`;
}

const visualizer = document.createElement("model-explorer-visualizer") as VisualizerElement;
const worker = new Worker(new URL("./pyodide-worker.ts", import.meta.url), { type: "module" });

let runtimeReady = false;
let loadedCollections: GraphCollection[] = [];
let activeCollectionLabel: string | null = null;
const debugLines: string[] = [];
let debugVisible = false;
let dragDepth = 0;
let draggingPanel = false;
let dragOffsetX = 0;
let dragOffsetY = 0;
let draggingFromFab = false;
let dragStartClientX = 0;
let dragStartClientY = 0;
let dragMoved = false;
let suppressNextFabClick = false;

const pendingRequests = new Map<
  string,
  { resolve: (value: unknown[]) => void; reject: (reason: Error) => void }
>();

const renderIcons = () =>
  createIcons({
    icons,
    attrs: { width: "15", height: "15", strokeWidth: "2" },
  });

const nextRequestId = () => `req-${Date.now()}-${Math.random().toString(16).slice(2)}`;

const isOnnxLike = (file: File) =>
  /\.(onnx|onnxtxt|onnxtext|textproto|json|onnxjson)$/i.test(file.name);

const isOnnxLikeName = (name: string) =>
  /\.(onnx|onnxtxt|onnxtext|textproto|json|onnxjson)$/i.test(name);

const setStatus = (message: string) => {
  addDebugLine(`[status] ${message}`);
};

const addDebugLine = (line: string) => {
  debugLines.push(line);
  if (debugLines.length > 200) {
    debugLines.shift();
  }
  debugLogEl.textContent = debugLines.join("\n");
};

const renderVisualizerCollections = () => {
  if (!activeCollectionLabel) {
    visualizer.graphCollections = loadedCollections;
    return;
  }
  const active = loadedCollections.find((c) => c.label === activeCollectionLabel);
  if (!active) {
    visualizer.graphCollections = loadedCollections;
    return;
  }
  visualizer.graphCollections = [active, ...loadedCollections.filter((c) => c.label !== activeCollectionLabel)];
};

const updateShareQueryParams = () => {
  const currentUrl = new URL(window.location.href);
  currentUrl.searchParams.delete("url");
  for (const collection of loadedCollections) {
    if (collection.sourceUrl) {
      currentUrl.searchParams.append("url", collection.sourceUrl);
    }
  }
  history.replaceState({}, "", currentUrl.toString());
};

const renderModelList = () => {
  modelListEl.replaceChildren();
  if (loadedCollections.length === 0) {
    const emptyState = document.createElement("div");
    emptyState.className = "model-list-empty";
    emptyState.innerHTML = `
      <div class="model-list-empty-title">No models loaded</div>
      <div class="model-list-empty-hint">Paste a model URL and click the link button</div>
      <div class="model-list-empty-hint">Or click + to open local files</div>
      <div class="model-list-empty-hint">Or drag ONNX files anywhere on this page</div>
    `;
    modelListEl.appendChild(emptyState);
    return;
  }

  for (const collection of loadedCollections) {
    const row = document.createElement("div");
    row.className = "model-item";
    if (collection.label === activeCollectionLabel) {
      row.classList.add("active");
    }
    row.addEventListener("click", () => {
      activeCollectionLabel = collection.label;
      renderModelList();
      renderVisualizerCollections();
    });

    const name = document.createElement("span");
    name.className = "model-name";
    name.textContent = collection.label;
    name.title = collection.label;

    if (collection.sourceUrl) {
      const sourceBadge = document.createElement("span");
      sourceBadge.className = "model-url-badge";
      sourceBadge.textContent = "URL";
      sourceBadge.title = collection.sourceUrl;
      row.appendChild(sourceBadge);
    }

    const removeBtn = document.createElement("button");
    removeBtn.className = "ghost-btn model-remove-btn";
    removeBtn.title = "Remove model";
    removeBtn.innerHTML = `<i data-lucide="trash-2"></i>`;
    removeBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      loadedCollections = loadedCollections.filter((c) => c.label !== collection.label);
      if (activeCollectionLabel === collection.label) {
        activeCollectionLabel =
          loadedCollections.length > 0 ? loadedCollections[loadedCollections.length - 1].label : null;
      }
      renderModelList();
      renderVisualizerCollections();
      updateShareQueryParams();
      setStatus(`Removed ${collection.label}`);
      renderIcons();
    });

    row.append(name, removeBtn);
    modelListEl.appendChild(row);
  }
  renderIcons();
};

const getUniqueLabel = (baseName: string): string => {
  const existing = new Set(loadedCollections.map((c) => c.label));
  if (!existing.has(baseName)) {
    return baseName;
  }
  let i = 2;
  while (existing.has(`${baseName} (${i})`)) {
    i += 1;
  }
  return `${baseName} (${i})`;
};

const convertModel = async (modelName: string, bytes: ArrayBuffer): Promise<unknown[]> => {
  const requestId = nextRequestId();
  return new Promise((resolve, reject) => {
    pendingRequests.set(requestId, { resolve, reject });
    worker.postMessage(
      {
        type: "convert",
        requestId,
        modelName,
        bytes,
        settings: { const_element_count_limit: 1024 },
      },
      [bytes],
    );
  });
};

const appendLoadedCollection = (
  labelBase: string,
  graphCollections: unknown[],
  sourceUrl?: string,
) => {
  const firstCollection = graphCollections[0] as { graphs?: Graph[] } | undefined;
  if (!firstCollection?.graphs) {
    throw new Error(`Converter returned no graph for ${labelBase}`);
  }
  const label = getUniqueLabel(labelBase);
  loadedCollections.push({ label, graphs: firstCollection.graphs, sourceUrl });
  activeCollectionLabel = label;
  renderModelList();
  renderVisualizerCollections();
  updateShareQueryParams();
};

const loadFiles = async (files: File[]) => {
  if (files.length === 0) {
    setStatus("No ONNX file selected.");
    return;
  }
  if (!runtimeReady) {
    setStatus("Runtime is still initializing...");
    return;
  }
  for (const [index, file] of files.entries()) {
    setStatus(`Converting ${file.name} (${index + 1}/${files.length})...`);
    try {
      const bytes = await file.arrayBuffer();
      const graphCollections = await convertModel(file.name, bytes);
      appendLoadedCollection(file.name, graphCollections);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : `Failed to convert ${file.name}.`);
      return;
    }
  }
  setStatus(`Loaded ${files.length} model(s).`);
};

const guessModelNameFromUrl = (urlText: string): string => {
  try {
    const url = new URL(urlText);
    const name = decodeURIComponent(url.pathname.split("/").pop() ?? "").trim();
    if (name && isOnnxLikeName(name)) {
      return name;
    }
  } catch {
    // handled by fetch/URL parser later
  }
  return "model.onnx";
};

const loadModelFromUrl = async (rawUrl: string) => {
  const modelUrl = rawUrl.trim();
  if (!modelUrl) {
    setStatus("Please enter a model URL.");
    return;
  }
  if (!runtimeReady) {
    setStatus("Runtime is still initializing...");
    return;
  }
  setStatus(`Fetching model from URL: ${modelUrl}`);
  try {
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`URL request failed with status ${response.status}`);
    }
    const bytes = await response.arrayBuffer();
    const modelName = guessModelNameFromUrl(modelUrl);
    const graphCollections = await convertModel(modelName, bytes);
    appendLoadedCollection(modelName, graphCollections, modelUrl);
    setStatus(`Loaded model from URL: ${modelName}`);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to load model URL.";
    setStatus(`Failed to load URL model. ${message} (Check CORS and URL accessibility.)`);
  }
};

const parseUrlModelsFromQuery = (): string[] => {
  const params = new URL(window.location.href).searchParams;
  return params
    .getAll("url")
    .map((u) => u.trim())
    .filter((u) => u.length > 0);
};

worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const data = event.data;
  if (data.type === "status") {
    setStatus(data.message);
    return;
  }
  if (data.type === "ready") {
    runtimeReady = true;
    setStatus("Runtime ready.");
    const queryUrls = parseUrlModelsFromQuery();
    if (queryUrls.length > 0) {
      void (async () => {
        for (const [index, url] of queryUrls.entries()) {
          setStatus(`Loading query URL ${index + 1}/${queryUrls.length}...`);
          await loadModelFromUrl(url);
        }
      })();
    }
    return;
  }
  if (data.type === "result") {
    const pending = pendingRequests.get(data.requestId);
    if (!pending) {
      return;
    }
    pendingRequests.delete(data.requestId);
    pending.resolve(data.graphCollections);
    return;
  }
  if (data.type === "error") {
    const msg = data.traceback ? `${data.message}\n${data.traceback}` : data.message;
    addDebugLine(`[error] ${msg}`);
    if (data.requestId) {
      const pending = pendingRequests.get(data.requestId);
      if (pending) {
        pendingRequests.delete(data.requestId);
        pending.reject(new Error(data.message));
      }
    }
    setStatus(data.message);
  }
};

const applyDefaultTheme = async (): Promise<void> => {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}themes/netron.json`);
    if (!response.ok) {
      throw new Error(`Theme request failed with status ${response.status}`);
    }
    const nodeStylerRules = (await response.json()) as unknown[];
    visualizer.config = { nodeStylerRules };
  } catch (error) {
    addDebugLine(`[warn] Failed to load default theme: ${String(error)}`);
  }
};

fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files ?? []).filter(isOnnxLike);
  await loadFiles(files);
  fileInput.value = "";
});

loadUrlBtn.addEventListener("click", async () => {
  await loadModelFromUrl(modelUrlInput.value);
});

modelUrlInput.addEventListener("keydown", async (event) => {
  if (event.key !== "Enter") {
    return;
  }
  event.preventDefault();
  await loadModelFromUrl(modelUrlInput.value);
});

copyShareBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(window.location.href);
    setStatus("Share URL copied.");
  } catch (error) {
    setStatus(
      error instanceof Error ? `Failed to copy URL: ${error.message}` : "Failed to copy URL.",
    );
  }
});

sidebarToggleBtn.addEventListener("click", () => {
  appEl.classList.toggle("sidebar-collapsed");
});

sidebarFabBtn.addEventListener("click", () => {
  if (suppressNextFabClick) {
    suppressNextFabClick = false;
    return;
  }
  appEl.classList.remove("sidebar-collapsed");
});

debugToggleBtn.addEventListener("click", () => {
  debugVisible = !debugVisible;
  debugLogEl.classList.toggle("hidden", !debugVisible);
});

window.addEventListener("dragenter", (event) => {
  event.preventDefault();
  dragDepth += 1;
  dropOverlay.classList.remove("hidden");
});

window.addEventListener("dragover", (event) => {
  event.preventDefault();
});

window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) {
    dropOverlay.classList.add("hidden");
  }
});

window.addEventListener("drop", async (event) => {
  event.preventDefault();
  dragDepth = 0;
  dropOverlay.classList.add("hidden");
  const files = Array.from(event.dataTransfer?.files ?? []).filter(isOnnxLike);
  await loadFiles(files);
});

const setPanelPosition = (left: number, top: number) => {
  const appRect = appEl.getBoundingClientRect();
  const panelRect = leftPanelEl.getBoundingClientRect();
  const minLeft = 8;
  const minTop = 8;
  const maxLeft = Math.max(minLeft, appRect.width - panelRect.width - 8);
  const maxTop = Math.max(minTop, appRect.height - panelRect.height - 8);
  const clampedLeft = Math.min(maxLeft, Math.max(minLeft, left));
  const clampedTop = Math.min(maxTop, Math.max(minTop, top));
  appEl.style.setProperty("--panel-left", `${clampedLeft}px`);
  appEl.style.setProperty("--panel-top", `${clampedTop}px`);
};

const clampPanelPositionToViewport = () => {
  const style = getComputedStyle(appEl);
  const currentLeft = Number.parseFloat(style.getPropertyValue("--panel-left")) || 14;
  const currentTop = Number.parseFloat(style.getPropertyValue("--panel-top")) || 56;
  setPanelPosition(currentLeft, currentTop);
};

sideHeaderEl.addEventListener("pointerdown", (event) => {
  const target = event.target as HTMLElement | null;
  if (target && target.closest("button,input,label")) {
    return;
  }
  draggingPanel = true;
  const panelRect = leftPanelEl.getBoundingClientRect();
  const appRect = appEl.getBoundingClientRect();
  dragOffsetX = event.clientX - panelRect.left;
  dragOffsetY = event.clientY - panelRect.top;
  draggingFromFab = false;
  dragStartClientX = event.clientX;
  dragStartClientY = event.clientY;
  dragMoved = false;
  sideHeaderEl.setPointerCapture(event.pointerId);
  setPanelPosition(panelRect.left - appRect.left, panelRect.top - appRect.top);
});

sidebarFabBtn.addEventListener("pointerdown", (event) => {
  if (!appEl.classList.contains("sidebar-collapsed")) {
    return;
  }
  draggingPanel = true;
  draggingFromFab = true;
  dragStartClientX = event.clientX;
  dragStartClientY = event.clientY;
  dragMoved = false;
  const panelRect = leftPanelEl.getBoundingClientRect();
  dragOffsetX = event.clientX - panelRect.left;
  dragOffsetY = event.clientY - panelRect.top;
  sidebarFabBtn.setPointerCapture(event.pointerId);
});

window.addEventListener("pointermove", (event) => {
  if (!draggingPanel) {
    return;
  }
  if (appEl.classList.contains("sidebar-collapsed") && !draggingFromFab) {
    return;
  }
  if (
    Math.hypot(event.clientX - dragStartClientX, event.clientY - dragStartClientY) > 4
  ) {
    dragMoved = true;
  }
  const appRect = appEl.getBoundingClientRect();
  const nextLeft = event.clientX - appRect.left - dragOffsetX;
  const nextTop = event.clientY - appRect.top - dragOffsetY;
  setPanelPosition(nextLeft, nextTop);
});

window.addEventListener("pointerup", () => {
  if (draggingFromFab && dragMoved) {
    suppressNextFabClick = true;
  }
  draggingPanel = false;
  draggingFromFab = false;
  dragMoved = false;
});

window.addEventListener("resize", () => {
  if (appEl.classList.contains("sidebar-collapsed")) {
    return;
  }
  clampPanelPositionToViewport();
});

void applyDefaultTheme().finally(() => {
  viewerShell.appendChild(visualizer);
  renderModelList();
  renderVisualizerCollections();
  renderIcons();
  clampPanelPositionToViewport();
});
