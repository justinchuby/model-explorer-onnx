import converterCode from "./python/convert.py?raw";

type ConvertMessage = {
  type: "convert";
  requestId: string;
  modelName: string;
  bytes: ArrayBuffer;
  settings?: Record<string, unknown>;
};

type IncomingMessage = ConvertMessage;

const ONNX_WHEEL_URL =
  "https://files.pythonhosted.org/packages/82/f3/7996e76defeddd96aed47c7b8f54091672c23943ebaa2ddba3d1d3de5855/onnx_weekly-1.22.0.dev20260511-cp312-abi3-pyemscripten_2026_0_wasm32.whl";
const PYODIDE_INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.2/full/";

let pyodide: PyodideInterface | null = null;
let initPromise: Promise<void> | null = null;

type PyodideInterface = {
  loadPackage: (packages: string[]) => Promise<void>;
  runPythonAsync: (code: string) => Promise<unknown>;
  globals: { set: (name: string, value: unknown) => void };
  FS: { writeFile: (path: string, data: Uint8Array) => void };
};

const postStatus = (message: string) => {
  self.postMessage({ type: "status", message });
};

const ensureInitialized = async (): Promise<void> => {
  if (initPromise) {
    await initPromise;
    return;
  }
  initPromise = (async () => {
    postStatus("Loading Pyodide...");
    const pyodideModule = (await import(
      /* @vite-ignore */ `${PYODIDE_INDEX_URL}pyodide.mjs`
    )) as {
      loadPyodide: (opts: { indexURL: string }) => Promise<PyodideInterface>;
    };
    pyodide = await pyodideModule.loadPyodide({
      indexURL: PYODIDE_INDEX_URL,
    });
    postStatus("Loading Python package manager...");
    await pyodide.loadPackage([
      "micropip",
      "numpy",
      "sympy",
      "protobuf",
      "ml-dtypes",
      "packaging",
    ]);

    postStatus("Installing ONNX runtime packages...");
    pyodide.globals.set("onnx_wheel_url_js", ONNX_WHEEL_URL);
    await pyodide.runPythonAsync(`
import micropip
onnx_wheel_url = (
    onnx_wheel_url_js.to_py()
    if hasattr(onnx_wheel_url_js, "to_py")
    else onnx_wheel_url_js
)
# Install ONNX implementation first with matching pyemscripten platform tag.
await micropip.install(onnx_wheel_url, deps=False)
# onnx-ir depends on onnx; skip deps to avoid pulling a conflicting onnx wheel.
await micropip.install("typing_extensions>=4.10", deps=False)
await micropip.install("onnx-ir==0.2.1", deps=False)
`);

    postStatus("Loading converter...");
    await pyodide.runPythonAsync(converterCode);
    self.postMessage({ type: "ready" });
  })();
  await initPromise;
};

const toJs = (value: unknown): unknown => {
  if (
    value &&
    typeof value === "object" &&
    "toJs" in value &&
    typeof (value as { toJs: unknown }).toJs === "function"
  ) {
    return (
      value as {
        toJs: (opts?: { dict_converter?: (entries: [string, unknown][]) => unknown }) => unknown;
      }
    ).toJs({
      dict_converter: Object.fromEntries,
    });
  }
  return value;
};

self.onmessage = async (event: MessageEvent<IncomingMessage>) => {
  const data = event.data;
  try {
    await ensureInitialized();
    if (!pyodide) {
      throw new Error("Pyodide is not initialized.");
    }
    postStatus(`Converting ${data.modelName}...`);
    const filePath = "/tmp/input.onnx";
    pyodide.FS.writeFile(filePath, new Uint8Array(data.bytes));
    pyodide.globals.set("model_settings_js", data.settings ?? {});
    pyodide.globals.set("model_path_js", filePath);
    const pyResult = await pyodide.runPythonAsync(
      "convert_onnx_file(model_path_js.to_py(), dict(model_settings_js.to_py()))",
    );
    const result = toJs(pyResult) as { graphs: unknown[] };
    self.postMessage({
      type: "result",
      requestId: data.requestId,
      graphCollections: [
        {
          label: data.modelName,
          graphs: result.graphs,
        },
      ],
    });
    postStatus(`Converted ${data.modelName}.`);
  } catch (error) {
    let message = "Failed to convert model.";
    let traceback: string | undefined;
    if (error instanceof Error) {
      message = error.message;
      traceback = error.stack;
    }
    self.postMessage({
      type: "error",
      requestId: data.requestId,
      message,
      traceback,
    });
  }
};

void ensureInitialized().catch((error) => {
  self.postMessage({
    type: "error",
    message: error instanceof Error ? error.message : "Failed to initialize runtime.",
  });
});
