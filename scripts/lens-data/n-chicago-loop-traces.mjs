// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Extracts SIMULATED GPS traces inside the Chicago Loop from the showcase `chicago-gps-traces`
// dataset (read-only). The traces are synthetic: random shortest paths on the OSM road network,
// sampled every 5-15 s with 10-25 m Gaussian noise. For each of the three longest runs of
// consecutive fixes that stay inside the Loop bounding box, the fixes keep their original order.

import {readFileSync, writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

const root = new URL('../../', import.meta.url);
const source = new URL('examples/deck/spatial-analysis-showcase/public/data/', root);
const readColumn = (file, Type) => {
  const bytes = readFileSync(new URL(`chicago-gps-traces/${file}.bin`, source));
  return new Type(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};
const pathOffsets = readColumn('pathOffsets', Uint32Array);
const vertices = readColumn('vertices', Float32Array);
const timestamps = readColumn('timestamp', Uint32Array);
const noiseSigma = readColumn('noiseSigma', Float32Array);
const bbox = [-87.655, 41.865, -87.615, 41.902];
const inside = (x, y) => x >= bbox[0] && x <= bbox[2] && y >= bbox[1] && y <= bbox[3];
const runs = [];
for (let trace = 0; trace < pathOffsets.length - 1; trace++) {
  let start = -1;
  for (let fix = pathOffsets[trace]; fix <= pathOffsets[trace + 1]; fix++) {
    const isInside = fix < pathOffsets[trace + 1] && inside(vertices[2 * fix], vertices[2 * fix + 1]);
    if (isInside && start < 0) start = fix;
    if (!isInside && start >= 0) {
      runs.push({trace, start, length: fix - start});
      start = -1;
    }
  }
}
runs.sort((a, b) => b.length - a.length || a.trace - b.trace || a.start - b.start);
const selected = runs.slice(0, 3);
const trackOffsets = [0];
const positions = [];
const secondsSinceStart = [];
for (const run of selected) {
  for (let fix = run.start; fix < run.start + run.length; fix++) {
    positions.push(Number(vertices[2 * fix].toFixed(6)), Number(vertices[2 * fix + 1].toFixed(6)));
    secondsSinceStart.push(timestamps[fix] - timestamps[run.start]);
  }
  trackOffsets.push(positions.length / 2);
}
const data = {
  simulated: true,
  note: 'SIMULATED GPS fixes (random shortest paths on OSM roads with Gaussian noise); not real vehicles',
  bbox,
  trackCount: selected.length,
  sampleCount: positions.length / 2,
  sourceTraces: selected.map(run => run.trace),
  noiseSigmaMeters: selected.map(run => Number(noiseSigma[run.trace].toFixed(1))),
  trackOffsets,
  positions,
  secondsSinceStart
};
const bytes = `${JSON.stringify(data)}\n`;
if (data.sampleCount < 100 || data.sampleCount > 300 || Buffer.byteLength(bytes) > 150000)
  throw new Error('Trace budget failed');
const destination = new URL('modules/experimental/src/computation-lenses/data/n-chicago-loop-traces.json', root);
writeFileSync(destination, bytes);
// biome-ignore lint/suspicious/noConsole: Extraction reports its deterministic sizes and counts.
console.log(`${fileURLToPath(destination)}: ${Buffer.byteLength(bytes)} bytes; ${data.trackCount} tracks; ${data.sampleCount} samples; source traces ${data.sourceTraces}; sigma ${data.noiseSigmaMeters}`);
