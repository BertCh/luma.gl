#!/usr/bin/env node
// build.mjs: poopdeck-adsb-paths. One day of continental-US flights (OpenSky ADS-B, Monday 2020-01-06)
// as stitched, simplified 3-D trajectories with per-vertex time, altitude and reported ground speed.
//
// The poopdeck `adsb-paths` archive has no altitude (2-D positions only), so this reads the `flights`
// state-vector archive (points: altitude ft, speed kt, heading, vertical_rate, icao24), groups the pings by
// icao24 into flights, drops ground pings, splits at coverage gaps, and simplifies each flight with a
// synchronised-Euclidean-distance (space-time) Douglas-Peucker so cruise speed stays measurable.
//
// Usage:
//   node build.mjs --out <public/data/poopdeck-adsb-paths> --cache <fresh dir for the raw download> [options]
//     --tolerance <m>     SED tolerance in metres (default 10000)
//     --max-step <s>      longest allowed gap between kept vertices (default 600)
//     --split-gap <s>     split a flight at ping gaps longer than this (default 600)
//     --min-duration <s>  drop flights seen airborne for less than this (default 1200)
//     --min-peak-altitude <m>  keep flights whose highest ping reaches this (default 7500, about FL246)
//     --min-vertices <n>  drop flights with fewer kept vertices (default 5)
//     --core <path>       @poopdeck.gl/core dist/index.js (default ~/Documents/GitHub/poopdeck.gl/...)
//
// Output (little-endian, headerless): pathOffsets (u32), vertices (f32 lon,lat), timestamp (u32 s since
// timeOrigin), altitude (u16 metres), groundSpeed (u8, 3-knot units, as reported by the aircraft), plus manifest.json (flight start and end times are the first and last vertex times).
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {pathToFileURL} from 'node:url';

const args = {};
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i];
  if (a.startsWith('--')) args[a.slice(2)] = process.argv[++i];
}
if (!args.out || !args.cache) {
  console.error('usage: node build.mjs --out <dir> --cache <dir>');
  process.exit(2);
}
const TOLERANCE = Number(args.tolerance || 10000);
const MAX_STEP = Number(args['max-step'] || 600);
const SPLIT_GAP = Number(args['split-gap'] || 600);
const MIN_DURATION = Number(args['min-duration'] || 1200); // seconds of airborne coverage
const MIN_PEAK_ALTITUDE = Number(args['min-peak-altitude'] || 7500); // metres: keeps jets, drops light aircraft and helicopters
const SPEED_STEP = 3; // knots per stored unit
const MIN_VERTICES = Number(args['min-vertices'] || 5);
const URL_FLIGHTS = 'https://tiles.poopdeck.gl/data/flights/manifest.json';
const core = await import(pathToFileURL(args.core || path.join(os.homedir(), 'Documents/GitHub/poopdeck.gl/packages/core/dist/index.js')).href);
const {STTArchive, InlineTileDecoder} = core;

const archive = new STTArchive({url: URL_FLIGHTS, decoder: new InlineTileDecoder(), opfsCache: false});
const meta = await archive.getMetadata();
const bbox = {minLon: -130, minLat: 20, maxLon: -60, maxLat: 55};
const [tStart, tEnd] = [meta.timeRange.start, meta.timeRange.end];
const ids = await archive.getTileIdsInBounds(bbox, 0, {start: tStart, end: tEnd});
console.error(`z0 tiles: ${ids.length}`);

// ---- collect pings (de-duplicated: tiles are replicated across temporal buckets) ------------------
const raw = {icao: [], t: [], lon: [], lat: [], alt: [], spd: []};
const dict = new Map();
const seen = new Set();
const FT = 0.3048;
for (let b = 0; b < ids.length; b += 4) {
  for (const tile of await archive.getTiles(ids.slice(b, b + 4))) {
    if (!tile) continue;
    for (const layer of tile.layers) {
      if (layer.name === 'summary') continue;
      const f = layer.features;
      if (!f.featureCount || f.geometryType !== 0) continue;
      const cat = f.categoricalProps.icao24;
      for (let i = 0; i < f.featureCount; i++) {
        const t = f.startTimes[i] + f.timeOffset;
        const id64 = f.featureIds64 ? f.featureIds64[i] : null;
        if (id64 !== null) {
          const key = `${id64}:${t}`;
          if (seen.has(key)) continue;
          seen.add(key);
        }
        const name = cat.categories[cat.indices[i]];
        let code = dict.get(name);
        if (code === undefined) { code = dict.size; dict.set(name, code); }
        raw.icao.push(code); raw.t.push(t);
        raw.lon.push(f.positions[i * 2]); raw.lat.push(f.positions[i * 2 + 1]);
        raw.alt.push(f.numericProps.altitude[i] * FT); raw.spd.push(f.numericProps.speed[i]);
      }
    }
  }
  if ((b / 4) % 10 === 0) console.error(`tiles ${Math.min(b + 4, ids.length)}/${ids.length}, ${raw.t.length} pings`);
}
console.error(`${raw.t.length} unique pings, ${dict.size} aircraft`);
fs.mkdirSync(args.cache, {recursive: true});
fs.writeFileSync(path.join(args.cache, 'pings.json'), JSON.stringify({n: raw.t.length}));

// ---- group by aircraft, split into flights ---------------------------------------------------------
const order = Uint32Array.from({length: raw.t.length}, (_, i) => i);
order.sort((a, b) => raw.icao[a] - raw.icao[b] || raw.t[a] - raw.t[b]);
const GROUND_M = 60; // pings below this are treated as on the ground
const flights = [];
let cur = null;
let prev = -1;
const flush = () => { if (cur && cur.length >= 2) flights.push(cur); cur = null; };
for (const i of order) {
  if (prev >= 0 && raw.icao[i] !== raw.icao[prev]) flush();
  prev = i;
  if (raw.alt[i] < GROUND_M || !Number.isFinite(raw.alt[i])) { flush(); continue; }
  if (cur && (raw.t[i] - raw.t[cur[cur.length - 1]]) / 1000 > SPLIT_GAP) flush();
  // ADS-B repeats the last decoded position until the next position message arrives; drop those stale pings
  if (cur) {
    const last = cur[cur.length - 1];
    if (Math.abs(raw.lon[i] - raw.lon[last]) < 1e-4 && Math.abs(raw.lat[i] - raw.lat[last]) < 1e-4) continue;
  }
  (cur ??= []).push(i);
}
flush();
console.error(`${flights.length} airborne flights before simplification`);

// ---- space-time simplification ---------------------------------------------------------------------
const DEG = Math.PI / 180;
function simplify(idx) {
  const n = idx.length;
  const keep = new Uint8Array(n);
  keep[0] = keep[n - 1] = 1;
  const lat0 = raw.lat[idx[0]] * DEG;
  const kx = 6371000 * Math.cos(lat0) * DEG, ky = 6371000 * DEG;
  const stack = [[0, n - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    if (b - a < 2) continue;
    const ia = idx[a], ib = idx[b];
    const dt = raw.t[ib] - raw.t[ia];
    let worst = -1, worstErr = 0;
    for (let k = a + 1; k < b; k++) {
      const ik = idx[k];
      const f = dt > 0 ? (raw.t[ik] - raw.t[ia]) / dt : 0;
      const ex = (raw.lon[ik] - (raw.lon[ia] + (raw.lon[ib] - raw.lon[ia]) * f)) * kx;
      const ey = (raw.lat[ik] - (raw.lat[ia] + (raw.lat[ib] - raw.lat[ia]) * f)) * ky;
      const ez = (raw.alt[ik] - (raw.alt[ia] + (raw.alt[ib] - raw.alt[ia]) * f)) * 4;
      const err = Math.hypot(ex, ey, ez);
      if (err > worstErr) { worstErr = err; worst = k; }
    }
    if (dt > MAX_STEP * 1000 && (worst < 0 || worstErr < TOLERANCE)) {
      worst = a + ((b - a) >> 1); worstErr = Infinity;
    }
    if (worst >= 0 && worstErr > TOLERANCE) {
      keep[worst] = 1;
      stack.push([a, worst], [worst, b]);
    }
  }
  const out = [];
  for (let k = 0; k < n; k++) if (keep[k]) out.push(idx[k]);
  return out;
}
const kept = [];
let pingsIn = 0;
for (const f of flights) {
  pingsIn += f.length;
  const s = simplify(f);
  const peak = Math.max(...s.map((i) => raw.alt[i]));
  if (peak >= MIN_PEAK_ALTITUDE && s.length >= MIN_VERTICES && (raw.t[s[s.length - 1]] - raw.t[s[0]]) / 1000 >= MIN_DURATION) kept.push(s);
}
kept.sort((a, b) => raw.t[a[0]] - raw.t[b[0]]);
const vertexCount = kept.reduce((s, f) => s + f.length, 0);
console.error(`${kept.length} flights, ${vertexCount} vertices (from ${pingsIn} pings)`);

// ---- write -----------------------------------------------------------------------------------------
const originMs = Math.floor(raw.t[kept[0][0]] / 1000) * 1000;
const sec = (ms) => Math.max(0, Math.round((ms - originMs) / 1000));
const n = kept.length;
const offsets = new Uint32Array(n + 1);
const verts = new Float32Array(vertexCount * 2);
const time = new Uint32Array(vertexCount);
const altitude = new Uint16Array(vertexCount);
const speed = new Uint8Array(vertexCount);
const startT = new Uint32Array(n), endT = new Uint32Array(n);
let o = 0, minX = 180, minY = 90, maxX = -180, maxY = -90, maxEnd = 0;
kept.forEach((f, fi) => {
  offsets[fi] = o;
  for (const i of f) {
    verts[o * 2] = raw.lon[i]; verts[o * 2 + 1] = raw.lat[i];
    time[o] = sec(raw.t[i]);
    altitude[o] = Math.min(65535, Math.round(raw.alt[i]));
    speed[o] = Math.min(255, Math.max(0, Math.round(raw.spd[i] / SPEED_STEP)));
    minX = Math.min(minX, raw.lon[i]); maxX = Math.max(maxX, raw.lon[i]);
    minY = Math.min(minY, raw.lat[i]); maxY = Math.max(maxY, raw.lat[i]);
    o++;
  }
  startT[fi] = time[offsets[fi]]; endT[fi] = time[o - 1];
  maxEnd = Math.max(maxEnd, endT[fi]);
});
offsets[n] = o;
fs.mkdirSync(args.out, {recursive: true});
const unit = `seconds since ${new Date(originMs).toISOString()}`;
const files = {};
const col = (name, file, typed, dtype, extra = {}) => {
  fs.writeFileSync(path.join(args.out, file), new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength));
  files[file] = typed.byteLength;
  return [name, {file, dtype, ...extra, length: extra.components ? typed.length / extra.components : typed.length}];
};
const columns = Object.fromEntries([
  col('pathOffsets', 'pathOffsets.bin', offsets, 'uint32'),
  col('vertices', 'vertices.bin', verts, 'float32', {components: 2}),
  col('timestamp', 'time.bin', time, 'uint32', {unit}),
  col('altitude', 'altitude.bin', altitude, 'uint16', {unit: 'metres above mean sea level (barometric, from feet)'}),
  col('groundSpeed', 'groundSpeed.bin', speed, 'uint8', {unit: 'units of 3 knots (value x 3 = knots), ground speed as reported by the aircraft (ADS-B velocity message)'})
]);
const manifest = {
  id: 'poopdeck-adsb-paths', version: 1, kind: 'trajectories', count: n,
  bbox: [minX, minY, maxX, maxY], crs: 'EPSG:4326', columns,
  properties: {
    timeOriginMs: originMs, timeRangeMs: [originMs, originMs + maxEnd * 1000], vertexCount,
    day: '2020-01-06 (Monday)', toleranceMeters: TOLERANCE, maxStepSeconds: MAX_STEP,
    pingsBeforeSimplification: pingsIn,
    source: {manifestUrl: URL_FLIGHTS, name: meta.name, note: 'state vectors grouped by icao24; ground pings dropped; flights split at ping gaps > splitGap seconds'}
  }
};
fs.writeFileSync(path.join(args.out, 'manifest.json'), JSON.stringify(manifest, null, 2));
console.log(JSON.stringify({flights: n, vertices: vertexCount, bytes: Object.values(files).reduce((a, b) => a + b, 0), files}, null, 1));
