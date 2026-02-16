/**
 * sqlite-muninn — HNSW vector search + graph traversal + Node2Vec for SQLite
 *
 * Single TypeScript source compiled to CJS + ESM via tsup.
 * Works with better-sqlite3, node:sqlite (Node 22.5+), and bun:sqlite.
 */
import { readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { arch, platform } from "node:process";

// tsup injects __dirname for both CJS and ESM builds.
// In CJS it's the native __dirname; in ESM tsup generates a shim.
declare const __dirname: string;

// __dirname points to dist/ (built) or src/ (test), ROOT is the npm/ package directory
const ROOT = join(__dirname, "..");
const REPO_ROOT = join(ROOT, "..");

const EXT_MAP: Record<string, string> = { darwin: "dylib", linux: "so", win32: "dll" };

/** Read VERSION from npm/ package dir first, then repo root */
function readVersion(): string {
  for (const dir of [ROOT, REPO_ROOT]) {
    const p = join(dir, "VERSION");
    if (statSync(p, { throwIfNoEntry: false })) {
      return readFileSync(p, "utf8").trim();
    }
  }
  return "0.0.0";
}

/** Package version from the VERSION file */
export const version: string = readVersion();

/**
 * Get the absolute path to the muninn loadable extension binary.
 *
 * Resolution order:
 * 1. Local dev — build/ directory at repo root
 * 2. Platform-specific optional dependency (@neozenith/sqlite-muninn-<platform>-<arch>)
 *    Uses sibling resolution: __dirname/../<pkg>/muninn.<ext>
 *    This works because npm installs optionalDependencies as siblings
 *    under node_modules/.
 */
export function getLoadablePath(): string {
  const ext = EXT_MAP[platform];
  if (!ext) {
    throw new Error(`Unsupported platform: ${platform}-${arch}. Supported: ${Object.keys(EXT_MAP).join(", ")}`);
  }

  // Try build output dir first (git install / local dev)
  const buildPath = join(REPO_ROOT, "build", `muninn.${ext}`);
  if (statSync(buildPath, { throwIfNoEntry: false })) {
    return buildPath;
  }

  // Try platform-specific package (npm registry install)
  // Sibling resolution: node_modules/sqlite-muninn/dist/../.. = node_modules/
  // Then into @neozenith/sqlite-muninn-<platform>-<arch>/muninn.<ext>
  const archName = arch === "arm64" ? "arm64" : "x64";
  const pkgName = `@neozenith/sqlite-muninn-${platform}-${archName}`;
  const pkgPath = join(ROOT, "..", pkgName, `muninn.${ext}`);
  if (statSync(pkgPath, { throwIfNoEntry: false })) {
    return pkgPath;
  }

  throw new Error(
    `muninn binary not found for ${platform}-${arch}. Build with 'make all' or install from npm: npm install sqlite-muninn`,
  );
}

/** A SQLite database connection that supports loading extensions */
interface Db {
  loadExtension(file: string, entrypoint?: string): void;
}

/**
 * Load the muninn extension into a SQLite database connection.
 */
export function load(db: Db): void {
  db.loadExtension(getLoadablePath());
}
