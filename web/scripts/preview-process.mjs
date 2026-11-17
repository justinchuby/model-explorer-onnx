import { once } from "node:events";

export const stopPreview = async (preview) => {
  // exit is not replayed for a process that already exited (or failed to spawn).
  if (preview.exitCode !== null || preview.signalCode !== null || !preview.pid) return;
  const exited = once(preview, "exit");
  preview.kill();
  await exited;
};
