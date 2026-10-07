#!/usr/bin/env node
// build.mjs: poopdeck-earthquakes. Every USGS ComCat magnitude 4+ event of 2020-2024 (77,231 rows),
// exported from the poopdeck `earthquakes-v2` archive and re-ordered by time.
//
// Usage:
//   node build.mjs --out <public/data/poopdeck-earthquakes> --cache <fresh empty dir for the raw export>
//
// Steps: run scripts/data/poopdeck/stt-export.mjs over the whole archive and 2020-01-01 .. 2025-01-01
// (zoom 0, the lowest zoom, so there are no tile seams or replicated features), then sort every column
// by time so the rows are a chronological catalog. Columns: position (lon, lat float32), timestamp
// (uint32 seconds since properties.timeOriginMs), depth (km, float32), magnitude (float32), mag_band and
// type (uint8 categories). The `place` and `title` strings are dropped (44k distinct values).
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const args = {};
for (let i = 2; i < process.argv.length; i++) {
  if (process.argv[i].startsWith('--')) args[process.argv[i].slice(2)] = process.argv[++i];
}
if (!args.out || !args.cache) {
  console.error('usage: node build.mjs --out <dir> --cache <dir>');
  process.exit(2);
}
const here = path.dirname(fileURLToPath(import.meta.url));
const exporter = path.join(here, '..', 'poopdeck', 'stt-export.mjs');
const run = spawnSync(
  process.execPath,
  [
    exporter,
    'https://tiles.poopdeck.gl/data/earthquakes-v2/manifest.json',
    '--out', args.cache,
    '--id', 'poopdeck-earthquakes',
    '--time', '2020-01-01,2025-01-01',
    '--max-features', '80000',
    '--max-tiles', '20000'
  ],
  {stdio: 'inherit'}
);
if (run.status !== 0) process.exit(run.status ?? 1);

const manifest = JSON.parse(fs.readFileSync(path.join(args.cache, 'manifest.json'), 'utf8'));
const count = manifest.count;
const read = (name, Type) => {
  const bytes = fs.readFileSync(path.join(args.cache, manifest.columns[name].file));
  return new Type(bytes.buffer, bytes.byteOffset, bytes.byteLength / Type.BYTES_PER_ELEMENT);
};
const columns = {
  position: read('position', Float32Array),
  timestamp: read('timestamp', Uint32Array),
  depth: read('depth', Float32Array),
  magnitude: read('magnitude', Float32Array),
  mag_band: read('mag_band', Uint8Array),
  type: read('type', Uint8Array)
};
const order = Uint32Array.from({length: count}, (_, i) => i).sort(
  (a, b) => columns.timestamp[a] - columns.timestamp[b] || a - b
);
fs.mkdirSync(args.out, {recursive: true});
for (const [name, source] of Object.entries(columns)) {
  const components = name === 'position' ? 2 : 1;
  const target = new source.constructor(source.length);
  for (let row = 0; row < count; row++) {
    for (let c = 0; c < components; c++) target[row * components + c] = source[order[row] * components + c];
  }
  fs.writeFileSync(path.join(args.out, manifest.columns[name].file), Buffer.from(target.buffer));
}
manifest.properties.note =
  'USGS ComCat events of magnitude 4 and above, 2020-01-01 to 2024-12-30, sorted by time. type is the USGS event type (earthquake, mining explosion, ...). Depths are in km below the datum, some are negative or fixed at 10 km by the contributing network.';
fs.writeFileSync(path.join(args.out, 'manifest.json'), JSON.stringify(manifest, null, 1));
console.error(`wrote ${count} rows to ${args.out}`);
