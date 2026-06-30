import "ai-edge-model-explorer-visualizer";
import { createIcons } from "lucide";
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
type GraphCollection = { label: string; graphs: Graph[] };
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
const addFilesBtn = document.getElementById("add-files-btn") as HTMLButtonElement;
const fileInput = document.getElementById("model-file") as HTMLInputElement;
const debugToggleBtn = document.getElementById("debug-toggle") as HTMLButtonElement;
const modelListEl = document.getElementById("model-list");
const debugLogEl = document.getElementById("debug-log");
const viewerShell = document.getElementById("viewer-shell");
const dropOverlay = document.getElementById("drop-overlay");

if (
  !appEl ||
  !sidebarToggleBtn ||
  !addFilesBtn ||
  !fileInput ||
  !debugToggleBtn ||
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

const pendingRequests = new Map<
  string,
  { resolve: (value: unknown[]) => void; reject: (reason: Error) => void }
>();

const nextRequestId = () => `req-${Date.now()}-${Math.random().toString(16).slice(2)}`;

const isOnnxLike = (file: File) =>
  /\.(onnx|onnxtxt|onnxtext|textproto|json|onnxjson)$/i.test(file.name);

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

const renderModelList = () => {
  const header = document.createElement("div");
  header.className = "model-list-header";
  header.textContent = `Loaded models (${loadedCollections.length})`;
  modelListEl.replaceChildren(header);

  for (const collection of loadedCollections) {
    const row = document.createElement("div");
    row.className = "model-item";
    if (collection.label === activeCollectionLabel) {
      row.classList.add("active");
    }

    const nameBtn = document.createElement("button");
    nameBtn.className = "ghost-btn";
    nameBtn.title = "Set active model";
    nameBtn.innerHTML = `<i data-lucide="eye"></i>`;
    nameBtn.addEventListener("click", () => {
      activeCollectionLabel = collection.label;
      renderModelList();
      renderVisualizerCollections();
    });

    const name = document.createElement("span");
    name.className = "model-name";
    name.textContent = collection.label;
    name.title = collection.label;

    const removeBtn = document.createElement("button");
    removeBtn.className = "ghost-btn model-remove-btn";
    removeBtn.title = "Remove model";
    removeBtn.innerHTML = `<i data-lucide="trash-2"></i>`;
    removeBtn.addEventListener("click", () => {
      loadedCollections = loadedCollections.filter((c) => c.label !== collection.label);
      if (activeCollectionLabel === collection.label) {
        activeCollectionLabel =
          loadedCollections.length > 0 ? loadedCollections[loadedCollections.length - 1].label : null;
      }
      renderModelList();
      renderVisualizerCollections();
      setStatus(`Removed ${collection.label}`);
      createIcons({ attrs: { width: "14", height: "14", strokeWidth: "2" } });
    });

    row.append(nameBtn, name, removeBtn);
    modelListEl.appendChild(row);
  }
  createIcons({ attrs: { width: "14", height: "14", strokeWidth: "2" } });
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

const convertModel = async (file: File): Promise<unknown[]> => {
  const requestId = nextRequestId();
  const bytes = await file.arrayBuffer();
  return new Promise((resolve, reject) => {
    pendingRequests.set(requestId, { resolve, reject });
    worker.postMessage(
      {
        type: "convert",
        requestId,
        modelName: file.name,
        bytes,
        settings: { const_element_count_limit: 1024 },
      },
      [bytes],
    );
  });
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
      const graphCollections = await convertModel(file);
      const firstCollection = graphCollections[0] as { graphs?: Graph[] } | undefined;
      if (!firstCollection?.graphs) {
        throw new Error(`Converter returned no graph for ${file.name}`);
      }
      const label = getUniqueLabel(file.name);
      loadedCollections.push({ label, graphs: firstCollection.graphs });
      activeCollectionLabel = label;
      renderModelList();
      renderVisualizerCollections();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : `Failed to convert ${file.name}.`);
      return;
    }
  }
  setStatus(`Loaded ${files.length} model(s).`);
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

addFilesBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files ?? []).filter(isOnnxLike);
  await loadFiles(files);
  fileInput.value = "";
});

sidebarToggleBtn.addEventListener("click", () => {
  appEl.classList.toggle("sidebar-collapsed");
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

void applyDefaultTheme().finally(() => {
  viewerShell.appendChild(visualizer);
  renderModelList();
  renderVisualizerCollections();
  createIcons({ attrs: { width: "15", height: "15", strokeWidth: "2" } });
});
