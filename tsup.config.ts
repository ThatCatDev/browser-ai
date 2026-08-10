import { defineConfig } from "tsup";

/**
 * ESM and types, and nothing else.
 *
 * The one dependency this has — `@huggingface/transformers` — is a peer and
 * stays external: bundling it would ship a second copy of a large library and a
 * second WASM runtime into any app that already has one. It is reached through
 * a dynamic import at the point of use, which survives bundling and keeps the
 * weights and the library off the critical path of whatever loads this.
 */
export default defineConfig({
  entry: { index: "src/index.ts", worker: "src/worker/worker.ts" },
  format: ["esm"],
  dts: true,
  clean: true,
  sourcemap: true,
  target: "es2020",
  external: ["@huggingface/transformers"]
});
