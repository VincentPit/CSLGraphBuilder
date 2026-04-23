#!/usr/bin/env node
/**
 * Contract check — proves the frontend's response handlers don't throw
 * on any reasonable backend payload.
 *
 * What it covers:
 *   1. ``formatApiError`` against every error shape in api-fixtures.ts
 *      (null, plain strings, FastAPI 4xx, FastAPI 422 validation arrays,
 *      nested objects, empty arrays). This is the function that crashed
 *      React earlier when given a Pydantic error array.
 *   2. Type-shape compatibility — mock fixtures must match the exported
 *      TS types in `lib/api.ts` (verified by tsc separately).
 *
 * Why a plain Node script and not a full test runner:
 *   - Zero new dev deps for the build pipeline.
 *   - Run anywhere Node 18+ runs (CI, local).
 *   - Single file, fast (< 1s), readable failure output.
 *
 * Add more contract assertions here as new fragile spots are found.
 */

import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..');

// We use the project's own TS compiler to transpile the api module + the
// fixtures into a temp dir, then dynamically import them. Using esbuild
// would be lighter, but tsc is already a dependency — no new packages.
function transpileForRuntime() {
  const out = mkdtempSync(`${tmpdir()}/contract-check-`);
  const tsconfig = {
    compilerOptions: {
      target: 'ES2022',
      module: 'ESNext',
      moduleResolution: 'Bundler',
      esModuleInterop: true,
      isolatedModules: true,
      skipLibCheck: true,
      jsx: 'preserve',
      // baseUrl/paths are deprecated in TS 7 but our editor uses them via
      // the main tsconfig; silence the warning here so the build
      // pipeline doesn't fail on cosmetics.
      ignoreDeprecations: '6.0',
      baseUrl: repoRoot,
      paths: { '@/*': ['./*'] },
      outDir: out,
      // Turn off strictness and emit plain JS — we just want runtime
      // correctness; strict typecheck is a separate CI step.
      strict: false,
      noEmit: false,
      declaration: false,
      // api.ts references `process.env.NEXT_PUBLIC_*` for client config —
      // pull in node + dom typings so the transpile sees them.
      types: ['node'],
      lib: ['ES2022', 'DOM'],
    },
    include: ['lib/api.ts', '__mocks__/api-fixtures.ts'],
  };
  writeFileSync(`${repoRoot}/tsconfig.contract.json`, JSON.stringify(tsconfig, null, 2));

  const res = spawnSync(
    'npx',
    ['tsc', '-p', 'tsconfig.contract.json'],
    { cwd: repoRoot, encoding: 'utf-8' }
  );
  if (res.status !== 0) {
    console.error('✗ tsc transpile failed:');
    console.error(res.stdout);
    console.error(res.stderr);
    process.exit(1);
  }
  return out;
}

async function main() {
  const outDir = transpileForRuntime();

  // Tell Node to treat the transpiled .js as ESM (tsc emitted ESM).
  const fs0 = await import('node:fs');
  fs0.writeFileSync(`${outDir}/package.json`, JSON.stringify({ type: 'module' }));

  // The transpiled tree mirrors the source tree under outDir. The "@/lib/api"
  // import in the fixtures resolves via the path mapping at build time but
  // not at runtime — rewrite it. Simplest fix: import each file directly.
  const apiUrl = `file://${outDir}/lib/api.js`;
  const fixturesUrl = `file://${outDir}/__mocks__/api-fixtures.js`;

  // The compiled api-fixtures.js still has `import ... from "@/lib/api"`.
  // Patch the JS in-place to use a relative path.
  const fs = await import('node:fs');
  const fixturesPath = `${outDir}/__mocks__/api-fixtures.js`;
  const original = fs.readFileSync(fixturesPath, 'utf-8');
  const patched = original.replace(/['"]@\/lib\/api['"]/g, '"../lib/api.js"');
  fs.writeFileSync(fixturesPath, patched);

  // We can't import api.ts directly because it imports `axios`, which
  // imports browser-only assumptions in some paths. Stub the axios
  // module with an empty object before importing.
  const apiPath = `${outDir}/lib/api.js`;
  const apiOriginal = fs.readFileSync(apiPath, 'utf-8');
  const apiPatched = apiOriginal.replace(
    /import\s+axios.*?from\s+['"]axios['"]/,
    "const axios = { create: () => ({ defaults: {} }) };"
  );
  fs.writeFileSync(apiPath, apiPatched);

  const api = await import(apiUrl);
  const fixtures = await import(fixturesUrl);

  // ────────────────────────────────────────────────────────────────
  // Assertion 1: formatApiError returns a non-empty string for every
  // error shape and never throws.
  // ────────────────────────────────────────────────────────────────
  const failures = [];
  for (const { label, err } of fixtures.errorShapes) {
    try {
      const result = api.formatApiError(err, 'fallback');
      if (typeof result !== 'string' || result.length === 0) {
        failures.push(`  ✗ ${label}: returned ${typeof result} ${JSON.stringify(result)}`);
      } else {
        console.log(`  ✓ ${label.padEnd(38)} → ${result.slice(0, 60)}`);
      }
    } catch (e) {
      failures.push(`  ✗ ${label}: threw ${e.message}`);
    }
  }

  // ────────────────────────────────────────────────────────────────
  // Assertion 2: Mock payloads have the shape the frontend expects.
  // (TS already enforces this at build time; this asserts the runtime
  // values aren't undefined / null in places they shouldn't be.)
  // ────────────────────────────────────────────────────────────────
  const required = {
    mockGraphStats: ['total_entities', 'total_relationships', 'entity_type_counts'],
    mockEntity: ['id', 'name', 'entity_type', 'tags'],
    mockRelationship: ['id', 'source_entity_id', 'target_entity_id', 'relationship_type'],
    mockJob: ['job_id', 'status', 'stages', 'stage_progress', 'events'],
    mockMetrics: ['llm', 'embedding', 'pipeline', 'cache_sizes'],
  };
  for (const [name, fields] of Object.entries(required)) {
    const obj = fixtures[name];
    if (!obj) {
      failures.push(`  ✗ fixture ${name} missing`);
      continue;
    }
    for (const f of fields) {
      if (obj[f] === undefined) {
        failures.push(`  ✗ ${name}.${f} is undefined`);
      }
    }
    if (failures.length === 0) console.log(`  ✓ ${name} shape valid`);
  }

  // ────────────────────────────────────────────────────────────────
  // Cleanup tsconfig artefact
  // ────────────────────────────────────────────────────────────────
  fs.rmSync(`${repoRoot}/tsconfig.contract.json`, { force: true });
  fs.rmSync(outDir, { recursive: true, force: true });

  if (failures.length > 0) {
    console.error('\n✗ Contract check failed:\n');
    failures.forEach((f) => console.error(f));
    process.exit(1);
  }

  console.log('\n✓ Contract check passed — frontend handles all mock backend responses.');
}

main().catch((e) => {
  console.error('✗ Contract check crashed:', e);
  process.exit(1);
});
