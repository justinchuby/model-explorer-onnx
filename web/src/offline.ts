// Wait for control, not just activation: otherwise the first Python downloads
// bypass runtime caching. Create the dedicated worker only after this resolves.
export const prepareOfflineCache = async (): Promise<boolean> => {
  if (!import.meta.env.PROD || !("serviceWorker" in navigator)) {
    return false;
  }
  const registration = await navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`, {
    scope: import.meta.env.BASE_URL,
  });
  // On an upgrade, ready may refer to the previous active worker. Wait for the
  // new caching rules before downloading Python packages through that worker.
  const installing = registration.installing ?? registration.waiting;
  if (installing && installing.state !== "activated") {
    await new Promise<void>((resolve, reject) => {
      const onStateChange = () => {
        if (installing.state === "activated" || installing.state === "redundant") {
          installing.removeEventListener("statechange", onStateChange);
          if (installing.state === "activated") resolve();
          else reject(new Error("Offline cache installation failed."));
        }
      };
      installing.addEventListener("statechange", onStateChange);
      onStateChange();
    });
  }
  await navigator.serviceWorker.ready;
  const hasCurrentController = () =>
    navigator.serviceWorker.controller &&
    navigator.serviceWorker.controller === registration.active;
  if (!hasCurrentController()) {
    await new Promise<void>((resolve) => {
      const onControllerChange = () => {
        if (hasCurrentController()) {
          navigator.serviceWorker.removeEventListener("controllerchange", onControllerChange);
          resolve();
        }
      };
      navigator.serviceWorker.addEventListener("controllerchange", onControllerChange);
      onControllerChange();
    });
  }
  // Best effort: browsers may decline persistence, but normal caching still works.
  void navigator.storage?.persist?.().catch(() => false);
  return true;
};
