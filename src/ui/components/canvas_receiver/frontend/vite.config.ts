import { defineConfig } from "vite";

const isProduction = process.env.NODE_ENV === "production";

export default defineConfig({
  base: "./",
  build: {
    outDir: "build",
    lib: {
      entry: "./src/index.ts",
      name: "CanvasReceiver",
      formats: ["es"],
      fileName: "index-[hash]",
    },
    minify: isProduction ? "esbuild" : false,
    sourcemap: !isProduction,
    rollupOptions: {
      output: {
        entryFileNames: "index-[hash].js",
      },
    },
  },
  esbuild: isProduction
    ? {
        drop: ["console", "debugger"],
        minifyIdentifiers: true,
        minifySyntax: true,
      }
    : {},
  define: {
    "process.env.NODE_ENV": JSON.stringify(process.env.NODE_ENV),
  },
});
