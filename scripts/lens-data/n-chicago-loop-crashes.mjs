// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Extracts a deterministic sample of 2023 traffic crashes inside the Chicago Loop from the showcase
// `chicago-crashes` dataset (read-only; City of Chicago Data Portal, CPD-reported crashes). Only
// crashes within 25 m of the showcase road network are kept, then every k-th row (in source order)
// so that 64 events remain. The positions are plain longitude/latitude; GPUNetworkKFunction snaps
// them to the roads on the GPU.

import {readFileSync, writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

const root = new URL('../../', import.meta.url);
const source = new URL('examples/deck/spatial-analysis-showcase/public/data/', root);
const readColumn = (file, Type) => {
  const bytes = readFileSync(new URL(`chicago-crashes/${file}.bin`, source));
  return new Type(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};
const position = readColumn('position', Float32Array);
const snapDistance = readColumn('snapDistance', Uint8Array);
const bbox = [-87.655, 41.865, -87.615, 41.902];
const candidates = [];
for (let crash = 0; crash < snapDistance.length; crash++) {
  const [x, y] = [position[2 * crash], position[2 * crash + 1]];
  if (x >= bbox[0] && x <= bbox[2] && y >= bbox[1] && y <= bbox[3] && snapDistance[crash] <= 25)
    candidates.push(crash);
}
const eventCount = 64;
const step = Math.floor(candidates.length / eventCount);
const selected = Array.from({length: eventCount}, (_, index) => candidates[index * step]);
const data = {
  bbox,
  year: 2023,
  eligibleCount: candidates.length,
  sampleStep: step,
  eventCount,
  positions: selected.flatMap(crash => [
    Number(position[2 * crash].toFixed(5)),
    Number(position[2 * crash + 1].toFixed(5))
  ])
};
const bytes = `${JSON.stringify(data)}\n`;
const destination = new URL('modules/experimental/src/computation-lenses/data/n-chicago-loop-crashes.json', root);
writeFileSync(destination, bytes);
// biome-ignore lint/suspicious/noConsole: Extraction reports its deterministic sizes and counts.
console.log(`${fileURLToPath(destination)}: ${Buffer.byteLength(bytes)} bytes; ${eventCount} of ${candidates.length} eligible crashes (every ${step}th)`);
