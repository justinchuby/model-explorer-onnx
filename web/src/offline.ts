// Wait for control, not just activation: otherwise the first Python downloads
// bypass runtime caching. Create the dedicated worker only after this resolves.
export const prepareOfflineCache = async (): Promise<boolean> => {
  if (!import.meta.env.PROD || !("serviceWorker" in navigator)) {
    return false;
  }
  await navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`, {
    scope: import.meta.env.BASE_URL,
  });
  await navigator.serviceWorker.ready;
  if (!navigator.serviceWorker.controller) {
    await new Promise<void>((resolve) => {
      const onControllerChange = () => {
        if (navigator.serviceWorker.controller) {
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
