const OFFLINE_CACHE_TIMEOUT_MS = 15_000;

const withTimeout = async <T>(promise: Promise<T>, timeoutMs: number): Promise<T> => {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error("Offline cache setup timed out.")), timeoutMs);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
};

const waitForCondition = async (
  target: EventTarget,
  eventName: string,
  isReady: () => boolean,
  timeoutMs: number,
): Promise<void> => {
  let onChange = () => {};
  try {
    await withTimeout(new Promise<void>((resolve, reject) => {
      onChange = () => {
        try {
          if (isReady()) resolve();
        } catch (error) {
          reject(error);
        }
      };
      target.addEventListener(eventName, onChange);
      onChange();
    }), timeoutMs);
  } finally {
    target.removeEventListener(eventName, onChange);
  }
};

// Wait for control, not just activation: otherwise the first Python downloads
// bypass runtime caching. Every wait is bounded so online conversion can recover.
export const waitForOfflineControl = async (
  serviceWorker: ServiceWorkerContainer,
  scriptURL: string,
  scope: string,
  timeoutMs = OFFLINE_CACHE_TIMEOUT_MS,
): Promise<void> => {
  const registration = await withTimeout(serviceWorker.register(scriptURL, { scope }), timeoutMs);
  // An existing active worker may still use the previous caching rules.
  const installing = registration.installing ?? registration.waiting;
  if (installing) {
    await waitForCondition(installing, "statechange", () => {
      if (installing.state === "redundant") {
        throw new Error("Offline cache installation failed.");
      }
      return installing.state === "activated";
    }, timeoutMs);
  }
  // Avoid serviceWorker.ready: it can stay pending after a failed precache.
  if (!registration.active) {
    throw new Error("No active offline cache service worker.");
  }
  await waitForCondition(
    serviceWorker,
    "controllerchange",
    () => serviceWorker.controller !== null && serviceWorker.controller === registration.active,
    timeoutMs,
  );
};

export const prepareOfflineCache = async (): Promise<boolean> => {
  if (!import.meta.env.PROD || !("serviceWorker" in navigator)) {
    return false;
  }
  await waitForOfflineControl(
    navigator.serviceWorker,
    `${import.meta.env.BASE_URL}sw.js`,
    import.meta.env.BASE_URL,
  );
  // Best effort: browsers may decline persistence, but normal caching still works.
  void navigator.storage?.persist?.().catch(() => false);
  return true;
};
