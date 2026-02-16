import { describe, expect, it } from "vitest";
import { version, getLoadablePath } from "./index";

describe("sqlite-muninn npm package", () => {
  it("exports a version string", () => {
    expect(version).toMatch(/^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$/);
  });

  it("getLoadablePath returns a string ending with the platform extension", () => {
    const path = getLoadablePath();
    expect(typeof path).toBe("string");
    expect(path).toMatch(/muninn\.(dylib|so|dll)$/);
  });
});
