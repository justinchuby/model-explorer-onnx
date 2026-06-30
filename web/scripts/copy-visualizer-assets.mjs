import { cpSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDir, "..");
const packageDist = resolve(
  projectRoot,
  "node_modules",
  "ai-edge-model-explorer-visualizer",
  "dist",
);
const netronThemePath = resolve(projectRoot, "..", "themes", "netron.json");
const publicDir = resolve(projectRoot, "public");

if (!existsSync(packageDist)) {
  console.warn(
    "Visualizer package dist directory not found; skipping asset copy.",
  );
  process.exit(0);
}

mkdirSync(publicDir, { recursive: true });

cpSync(resolve(packageDist, "worker.js"), resolve(publicDir, "worker.js"));
cpSync(resolve(packageDist, "static_files"), resolve(publicDir, "static_files"), {
  recursive: true,
  force: true,
});
mkdirSync(resolve(publicDir, "themes"), { recursive: true });
cpSync(netronThemePath, resolve(publicDir, "themes", "netron.json"));

console.log(
  "Copied visualizer assets and themes/netron.json into web/public/.",
);
