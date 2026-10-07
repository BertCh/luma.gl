#!/usr/bin/env node
// Builds `poopdeck-gtfs-nl`: every scheduled trip in the Randstad between 07:00 and 09:00 local time
// (05:00-07:00 UTC) on Friday 3 July 2026, from the poopdeck.gl `gtfs-nl` archive.
//
//   node build.mjs [--work <scratch dir>] [--out <dir>] [--tolerance <meters>]
//
// 1. runs ../poopdeck/stt-export.mjs (clip to the window; the archive tiles at zoom 6 are read with
//    HTTP range requests, ~30 MB). Trips arrive in pieces cut at every hour bucket,
// 1b. stitches the pieces here: the exporter's `--stitch trip_id` cannot be used because the
//    archive stores trip_id as a lossy float, so pieces of one trip do not compare equal. A piece
//    continues another when its first vertex has the same time as the other's last vertex (the cut
//    vertex is duplicated), lies within 60 m of it, and route type and name agree,
// 2. clips each trip to the bounding box, keeping one vertex beyond each side so the line still
//    leaves the box,
// 3. simplifies each trip with a time-synchronous (SED) Douglas-Peucker pass: a vertex is dropped
//    only if the vehicle's position interpolated in time stays within `tolerance` meters, so dwell
//    at stops and speed changes survive while straight shape points do not,
// 4. rewrites the columns with the time origin pinned to the window start (uint32 seconds).
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, token, index, all) => {
    if (token.startsWith('--')) pairs.push([token.slice(2), all[index + 1]]);
    return pairs;
  }, [])
);
const work = args.work || path.join(os.tmpdir(), 'poopdeck-gtfs-nl-work');
const out = args.out || path.join(here, '../../../public/data/poopdeck-gtfs-nl');
const tolerance = Number(args.tolerance || 20);

const BBOX = [4.2, 51.85, 5.2, 52.45];
const WINDOW = ['2026-07-03T05:00:00Z', '2026-07-03T07:00:00Z'];
const ORIGIN_MS = Date.parse(WINDOW[0]);
const MANIFEST_URL = 'https://tiles.poopdeck.gl/data/gtfs-nl/manifest.json';

fs.rmSync(work, {recursive: true, force: true});
const run = spawnSync(
  'node',
  [
    path.join(here, '../poopdeck/stt-export.mjs'),
    MANIFEST_URL,
    '--out', work,
    '--id', 'poopdeck-gtfs-nl',
    '--bbox', BBOX.join(','),
    '--time', WINDOW.join(','),
    '--clip-time',
    '--attrs', 'route_type,route_short_name',
    '--max-features', '100000'
  ],
  {stdio: ['ignore', 'inherit', 'inherit']}
);
if (run.status !== 0) throw new Error('stt-export failed');

const read = (file, Type) => {
  const bytes = fs.readFileSync(path.join(work, file));
  return new Type(bytes.buffer, bytes.byteOffset, bytes.byteLength / Type.BYTES_PER_ELEMENT);
};
const source = JSON.parse(fs.readFileSync(path.join(work, 'manifest.json'), 'utf8'));
const pieceOffsets = read('pathOffsets.bin', Uint32Array);
const pieceVertices = read('vertices.bin', Float32Array);
const pieceTimes = read('time.bin', Uint32Array);
const pieceType = read('route_type.bin', Uint8Array);
const nameColumn = source.columns.route_short_name;
const pieceName = read('route_short_name.bin', nameColumn.dtype === 'uint8' ? Uint8Array : Uint16Array);
const exportOrigin = source.properties.timeOriginMs;
const shift = Math.round((exportOrigin - ORIGIN_MS) / 1000);

// ---- stitch pieces into trips ------------------------------------------------------------------
const pieceCount = pieceType.length;
let currentPiece = 0;
const vertexKey = (v) => `${pieceTimes[v]},${pieceType[currentPiece]},${pieceName[currentPiece]}`;
const firstOf = new Map();
for (let piece = 0; piece < pieceCount; piece++) {
  currentPiece = piece;
  const key = vertexKey(pieceOffsets[piece]);
  if (!firstOf.has(key)) firstOf.set(key, []);
  firstOf.get(key).push(piece);
}
const nextOf = new Int32Array(pieceCount).fill(-1);
const hasPrevious = new Uint8Array(pieceCount);
const JOIN_METERS = 60;
const latScale = 110574;
const lonScale = 111320 * Math.cos((52.15 * Math.PI) / 180);
for (let piece = 0; piece < pieceCount; piece++) {
  currentPiece = piece;
  const last = pieceOffsets[piece + 1] - 1;
  const candidates = firstOf.get(vertexKey(last));
  let best = -1;
  let bestDistance = JOIN_METERS;
  for (const other of candidates ?? []) {
    if (other === piece || hasPrevious[other]) continue;
    const first = pieceOffsets[other];
    const distance = Math.hypot(
      (pieceVertices[first * 2] - pieceVertices[last * 2]) * lonScale,
      (pieceVertices[first * 2 + 1] - pieceVertices[last * 2 + 1]) * latScale
    );
    if (distance < bestDistance) { bestDistance = distance; best = other; }
  }
  if (best >= 0) {
    nextOf[piece] = best;
    hasPrevious[best] = 1;
  }
}
const chains = [];
for (let piece = 0; piece < pieceCount; piece++) {
  if (hasPrevious[piece]) continue;
  const chain = [];
  for (let at = piece; at >= 0; at = nextOf[at]) chain.push(at);
  chains.push(chain);
}
let mergedVertexTotal = 0;
for (const chain of chains) for (const piece of chain) mergedVertexTotal += pieceOffsets[piece + 1] - pieceOffsets[piece];
const offsets = new Uint32Array(chains.length + 1);
const vertices = new Float32Array(mergedVertexTotal * 2);
const times = new Uint32Array(mergedVertexTotal);
const routeType = new Uint8Array(chains.length);
const routeName = new Uint16Array(chains.length);
let mergedCursor = 0;
chains.forEach((chain, trip) => {
  offsets[trip] = mergedCursor;
  routeType[trip] = pieceType[chain[0]];
  routeName[trip] = pieceName[chain[0]];
  chain.forEach((piece, index) => {
    // The cut vertex is duplicated at the join: skip the first vertex of every continuation.
    for (let v = pieceOffsets[piece] + (index > 0 ? 1 : 0); v < pieceOffsets[piece + 1]; v++) {
      vertices[mergedCursor * 2] = pieceVertices[v * 2];
      vertices[mergedCursor * 2 + 1] = pieceVertices[v * 2 + 1];
      times[mergedCursor++] = pieceTimes[v];
    }
  });
});
offsets[chains.length] = mergedCursor;
console.error(`stitched ${pieceCount} pieces into ${chains.length} trips`);
const tripCount = chains.length;

// Local metric scale at the box center.
const latMid = (BBOX[1] + BBOX[3]) / 2;
const mx = 111320 * Math.cos((latMid * Math.PI) / 180);
const my = 110574;

const kept = [];
let dropped = 0;
for (let trip = 0; trip < tripCount; trip++) {
  const from = offsets[trip];
  const to = offsets[trip + 1];
  // clip to bbox: first and last inside vertex, plus one neighbour each
  let first = -1;
  let last = -1;
  for (let v = from; v < to; v++) {
    const x = vertices[v * 2];
    const y = vertices[v * 2 + 1];
    if (x >= BBOX[0] && x <= BBOX[2] && y >= BBOX[1] && y <= BBOX[3]) {
      if (first < 0) first = v;
      last = v;
    }
  }
  if (first < 0) { dropped++; continue; }
  first = Math.max(from, first - 1);
  last = Math.min(to - 1, last + 1);
  // SED Douglas-Peucker over [first, last]
  const keep = new Uint8Array(last - first + 1);
  keep[0] = 1;
  keep[last - first] = 1;
  const stack = [[first, last]];
  while (stack.length) {
    const [a, b] = stack.pop();
    if (b - a < 2) continue;
    const ta = times[a];
    const span = times[b] - ta;
    let worst = -1;
    let worstDistance = tolerance;
    for (let v = a + 1; v < b; v++) {
      const f = span > 0 ? (times[v] - ta) / span : 0;
      const ex = vertices[a * 2] + (vertices[b * 2] - vertices[a * 2]) * f;
      const ey = vertices[a * 2 + 1] + (vertices[b * 2 + 1] - vertices[a * 2 + 1]) * f;
      const distance = Math.hypot((vertices[v * 2] - ex) * mx, (vertices[v * 2 + 1] - ey) * my);
      if (distance > worstDistance) { worstDistance = distance; worst = v; }
    }
    if (worst >= 0) {
      keep[worst - first] = 1;
      stack.push([a, worst], [worst, b]);
    }
  }
  const indices = [];
  for (let v = first; v <= last; v++) if (keep[v - first]) indices.push(v);
  if (indices.length < 2) { dropped++; continue; }
  kept.push({trip, indices});
}
kept.sort((p, q) => times[p.indices[0]] - times[q.indices[0]]);

const count = kept.length;
const vertexCount = kept.reduce((sum, t) => sum + t.indices.length, 0);
const outOffsets = new Uint32Array(count + 1);
const outVertices = new Float32Array(vertexCount * 2);
const outTimes = new Uint32Array(vertexCount);
const outStart = new Uint32Array(count);
const outEnd = new Uint32Array(count);
const outType = new Uint8Array(count);
const outName = new Uint16Array(count);
// Dense route name dictionary from the codes actually kept.
const nameCategories = source.columns.route_short_name.categories;
const nameRemap = new Map();
const names = [];
let cursor = 0;
kept.forEach((t, i) => {
  outOffsets[i] = cursor;
  for (const v of t.indices) {
    outVertices[cursor * 2] = vertices[v * 2];
    outVertices[cursor * 2 + 1] = vertices[v * 2 + 1];
    outTimes[cursor] = Math.max(0, times[v] + shift);
    cursor++;
  }
  outStart[i] = outTimes[outOffsets[i]];
  outEnd[i] = outTimes[cursor - 1];
  outType[i] = routeType[t.trip];
  const label = nameCategories[routeName[t.trip]] || '';
  if (!nameRemap.has(label)) { nameRemap.set(label, names.length); names.push(label); }
  outName[i] = nameRemap.get(label);
});
outOffsets[count] = cursor;

fs.mkdirSync(out, {recursive: true});
const write = (file, typed) => fs.writeFileSync(path.join(out, file), new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength));
write('pathOffsets.bin', outOffsets);
write('vertices.bin', outVertices);
write('time.bin', outTimes);
write('startTime.bin', outStart);
write('endTime.bin', outEnd);
write('route_type.bin', outType);
write('route_short_name.bin', outName);

const unit = `seconds since ${new Date(ORIGIN_MS).toISOString()}`;
const manifest = {
  id: 'poopdeck-gtfs-nl',
  version: 1,
  kind: 'trajectories',
  count,
  bbox: BBOX,
  crs: 'EPSG:4326',
  columns: {
    pathOffsets: {file: 'pathOffsets.bin', dtype: 'uint32', length: count + 1},
    vertices: {file: 'vertices.bin', dtype: 'float32', components: 2, length: vertexCount},
    timestamp: {file: 'time.bin', dtype: 'uint32', length: vertexCount, unit},
    startTime: {file: 'startTime.bin', dtype: 'uint32', length: count, unit},
    endTime: {file: 'endTime.bin', dtype: 'uint32', length: count, unit},
    route_type: {file: 'route_type.bin', dtype: 'uint8', length: count, categories: source.columns.route_type.categories},
    route_short_name: {file: 'route_short_name.bin', dtype: 'uint16', length: count, categories: names}
  },
  properties: {
    timeOriginMs: ORIGIN_MS,
    timeRangeMs: [ORIGIN_MS, Date.parse(WINDOW[1])],
    vertexCount,
    description: 'Scheduled GTFS trips, positions interpolated between stops along the shape. Not real-time positions.',
    window: {bbox: BBOX, utc: WINDOW, local: 'Friday 3 July 2026, 07:00-09:00 CEST'},
    simplification: `time-synchronous Douglas-Peucker, ${tolerance} m`,
    droppedOutsideBox: dropped,
    source: {
      manifestUrl: MANIFEST_URL,
      upstream: 'https://gtfs.ovapi.nl/nl/gtfs-nl.zip',
      attribution: 'OVapi / NDOV, CC0'
    }
  }
};
fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 2));
const bytes = outOffsets.byteLength + outVertices.byteLength + outTimes.byteLength + outStart.byteLength * 2 + count * 3;
console.log(JSON.stringify({count, vertexCount, droppedOutsideBox: dropped, approxBytes: bytes, routeTypes: source.columns.route_type.categories}));
