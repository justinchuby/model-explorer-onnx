import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  base: "./",
  plugins: [
    VitePWA({
      registerType: "autoUpdate",
      // main.ts registers before creating the Python worker so the first visit
      // caches runtime downloads too.
      injectRegister: false,
      includeAssets: ["themes/netron.json"],
      manifest: {
        name: "Model Explorer ONNX",
        short_name: "ONNX Explorer",
        description: "Static ONNX model explorer powered by Pyodide.",
        theme_color: "#2563eb",
        background_color: "#eef2ff",
        display: "standalone",
        start_url: "./",
        scope: "./",
        icons: [
          {
            src: "pwa-192.png",
            sizes: "192x192",
            type: "image/png",
          },
          {
            src: "pwa-512.png",
            sizes: "512x512",
            type: "image/png",
          },
          {
            src: "pwa-512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "maskable",
          },
        ],
      },
      workbox: {
        globPatterns: ["**/*.{js,css,html,json,png,svg,ico,wasm,woff,woff2}"],
        maximumFileSizeToCacheInBytes: 12 * 1024 * 1024,
        clientsClaim: true,
        skipWaiting: true,
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/.*/i,
            handler: "CacheFirst",
            options: {
              cacheName: "cdn-jsdelivr",
              cacheableResponse: { statuses: [200] },
            },
          },
          {
            urlPattern: /^https:\/\/(files\.pythonhosted\.org|pypi\.org)\/.*/i,
            handler: "CacheFirst",
            options: {
              cacheName: "python-wheels",
              cacheableResponse: { statuses: [200] },
            },
          },
        ],
      },
    }),
  ],
  worker: {
    format: "es",
  },
});
