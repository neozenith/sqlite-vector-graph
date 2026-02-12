import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    coverage: {
      provider: "v8",
      include: ["src/**"],
      thresholds: {
        lines: 60, // Needs to get up above 90%
      },
    },
  },
});
