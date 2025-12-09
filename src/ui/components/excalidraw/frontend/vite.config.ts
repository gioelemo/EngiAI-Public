import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const isProduction = process.env.NODE_ENV === "production";

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "build",
    minify: isProduction ? "esbuild" : false,
    sourcemap: !isProduction,
    rollupOptions: {
      output: {
        // Disable code splitting - create a single bundle
        inlineDynamicImports: true,
      },
    },
  },
  esbuild: isProduction
    ? {
        drop: ["debugger"],
        minifyIdentifiers: true,
        minifySyntax: true,
      }
    : {},
  define: {
    "process.env.NODE_ENV": JSON.stringify(process.env.NODE_ENV),
  },
});
