import "ai-edge-model-explorer-visualizer";
import "./style.css";

type WorkerStatusMessage = {
  type: "status";
  message: string;
};

type WorkerReadyMessage = {
  type: "ready";
};

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

declare global {
  interface Window {
    modelExplorer?: {
      assetFilesBaseUrl?: string;
      workerScriptPath?: string;
    };
  }
}

const statusEl = document.getElementById("status");
const fileInput = document.getElementById("model-file") as HTMLInputElement;
const loadBtn = document.getElementById("load-btn") as HTMLButtonElement;
const appEl = document.getElementById("app");

if (!statusEl || !fileInput || !loadBtn || !appEl) {
  throw new Error("Missing required DOM elements.");
}

const meGlobal = window.modelExplorer;
if (meGlobal) {
  meGlobal.assetFilesBaseUrl = `${import.meta.env.BASE_URL}static_files`;
  meGlobal.workerScriptPath = `${import.meta.env.BASE_URL}worker.js`;
}

const visualizer = document.createElement("model-explorer-visualizer");
appEl.appendChild(visualizer);

const worker = new Worker(new URL("./pyodide-worker.ts", import.meta.url), {
  type: "module",
});

const pendingRequests = new Map<
  string,
  { resolve: (value: unknown[]) => void; reject: (reason: Error) => void }
>();

worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const data = event.data;
  if (data.type === "status") {
    statusEl.textContent = data.message;
    return;
  }
  if (data.type === "ready") {
    loadBtn.disabled = false;
    statusEl.textContent = "Runtime ready. Choose an ONNX model and click Load model.";
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
    if (data.requestId) {
      const pending = pendingRequests.get(data.requestId);
      if (pending) {
        pendingRequests.delete(data.requestId);
        pending.reject(new Error(data.message));
      }
    }
    statusEl.textContent = data.traceback
      ? `${data.message}\n${data.traceback}`
      : data.message;
  }
};

const nextRequestId = () => `req-${Date.now()}-${Math.random().toString(16).slice(2)}`;

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
        settings: {
          const_element_count_limit: 1024,
        },
      },
      [bytes],
    );
  });
};

loadBtn.addEventListener("click", async () => {
  const file = fileInput.files?.[0];
  if (!file) {
    statusEl.textContent = "Please select a model file.";
    return;
  }
  loadBtn.disabled = true;
  statusEl.textContent = `Converting ${file.name}...`;
  try {
    const graphCollections = await convertModel(file);
    (visualizer as { graphCollections: unknown[] }).graphCollections = graphCollections;
    statusEl.textContent = `Loaded ${file.name}`;
  } catch (error) {
    statusEl.textContent =
      error instanceof Error ? error.message : "Failed to convert model.";
  } finally {
    loadBtn.disabled = false;
  }
});
