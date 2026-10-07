#!/usr/bin/env node
// simplify.mjs <rawDir> <outDir> [toleranceMeters]
// Reads an stt-export.mjs "trajectories" output (bixi-points window), simplifies every ride with
// Douglas-Peucker in local meters (vertex times of kept vertices are unchanged), and writes the
// showcase layout: pathOffsets, vertices, timestamp, startTime, endTime, durationMin, tripId.
import fs from 'node:fs';
import path from 'node:path';

const [rawDir, outDir, tolArg] = process.argv.slice(2);
// The archive manifest leaves its attribution empty; the showcase manifest and the dataset
// descriptor (src/data/datasets/poopdeck-bixi-rides.dataset.ts) carry this one.
const ATTRIBUTION =
  'BIXI Montréal open data (CC BY); routes via OSRM on © OpenStreetMap contributors (ODbL); poopdeck.gl';
const tolerance = Number(tolArg || 4);
const manifest = JSON.parse(fs.readFileSync(path.join(rawDir, 'manifest.json'), 'utf8'));
const read = (file, Type) => {
  const b = fs.readFileSync(path.join(rawDir, file));
  return new Type(b.buffer, b.byteOffset, b.byteLength / Type.BYTES_PER_ELEMENT);
};
const offsets = read('pathOffsets.bin', Uint32Array);
const vertices = read('vertices.bin', Float32Array);
const times = read('time.bin', Uint32Array);
const duration = read('duration_min.bin', Float32Array);
const tripId = read('trip_id.bin', Float32Array);
const n = manifest.count;

const lat0 = (manifest.bbox[1] + manifest.bbox[3]) / 2;
const kx = 111320 * Math.cos((lat0 * Math.PI) / 180);
const ky = 110574;

function simplifyRange(from, to, keep) {
  // iterative Douglas-Peucker over vertex indices [from, to)
  const stack = [[from, to - 1]];
  keep[from] = 1; keep[to - 1] = 1;
  while (stack.length) {
    const [a, b] = stack.pop();
    if (b - a < 2) continue;
    const ax = vertices[a * 2] * kx, ay = vertices[a * 2 + 1] * ky;
    const bx = vertices[b * 2] * kx, by = vertices[b * 2 + 1] * ky;
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    let worst = -1, worstDist = tolerance;
    for (let i = a + 1; i < b; i++) {
      const px = vertices[i * 2] * kx - ax, py = vertices[i * 2 + 1] * ky - ay;
      let d;
      if (len2 === 0) d = Math.hypot(px, py);
      else {
        const t = Math.max(0, Math.min(1, (px * dx + py * dy) / len2));
        d = Math.hypot(px - t * dx, py - t * dy);
      }
      if (d > worstDist) { worstDist = d; worst = i; }
    }
    if (worst >= 0) { keep[worst] = 1; stack.push([a, worst], [worst, b]); }
  }
}

const keep = new Uint8Array(vertices.length / 2);
for (let i = 0; i < n; i++) simplifyRange(offsets[i], offsets[i + 1], keep);

const outOffsets = [0], outVerts = [], outTimes = [], start = [], end = [], dur = [], ids = [];
for (let i = 0; i < n; i++) {
  const count0 = outTimes.length;
  for (let v = offsets[i]; v < offsets[i + 1]; v++) {
    if (!keep[v]) continue;
    // drop repeated times: a vertex must advance the clock
    if (outTimes.length > count0 && times[v] === outTimes[outTimes.length - 1]) continue;
    outVerts.push(vertices[v * 2], vertices[v * 2 + 1]);
    outTimes.push(times[v]);
  }
  if (outTimes.length - count0 < 2) {
    outVerts.length = count0 * 2; outTimes.length = count0; continue;
  }
  outOffsets.push(outTimes.length);
  start.push(outTimes[count0]); end.push(outTimes[outTimes.length - 1]);
  dur.push(duration[i]); ids.push(tripId[i]);
}
fs.mkdirSync(outDir, {recursive: true});
const originIso = new Date(manifest.properties.timeOriginMs).toISOString();
const columns = {};
let bytes = 0;
function put(name, file, typed, extra = {}) {
  fs.writeFileSync(path.join(outDir, file), new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength));
  bytes += typed.byteLength;
  const dtype = typed instanceof Float32Array ? 'float32' : 'uint32';
  columns[name] = {file, dtype, components: extra.components || 1, length: typed.length / (extra.components || 1), ...extra.meta};
}
const count = start.length;
put('pathOffsets', 'pathOffsets.bin', Uint32Array.from(outOffsets));
put('vertices', 'vertices.bin', Float32Array.from(outVerts), {components: 2});
put('timestamp', 'timestamp.bin', Uint32Array.from(outTimes), {meta: {unit: `seconds since ${originIso}`}});
put('startTime', 'startTime.bin', Uint32Array.from(start), {meta: {unit: `seconds since ${originIso}`}});
put('endTime', 'endTime.bin', Uint32Array.from(end), {meta: {unit: `seconds since ${originIso}`}});
put('durationMin', 'durationMin.bin', Float32Array.from(dur), {meta: {unit: 'minutes, whole ride (before the window clip)'}});
put('tripId', 'tripId.bin', Float32Array.from(ids), {meta: {description: 'BIXI trip id in the poopdeck archive'}});
const out = {
  id: 'poopdeck-bixi-rides',
  version: 1,
  kind: 'trajectories',
  count,
  bbox: manifest.bbox,
  crs: 'EPSG:4326',
  columns,
  properties: {
    ...manifest.properties,
    source: {...manifest.properties.source, attribution: ATTRIBUTION},
    vertexCount: outTimes.length,
    rawVertexCount: vertices.length / 2,
    simplifyToleranceMeters: tolerance,
    window: 'rides on 2024-08-15, 11:30-14:00 UTC (07:30-10:00 America/Montreal), clipped to the window',
    notes: 'Routes are OSRM bicycle-profile paths between the start and end station, with times interpolated along the route. They are not GPS traces.'
  }
};
fs.writeFileSync(path.join(outDir, 'manifest.json'), JSON.stringify(out, null, 2));
console.log(JSON.stringify({rides: count, vertices: outTimes.length, rawVertices: vertices.length / 2, bytes}));
