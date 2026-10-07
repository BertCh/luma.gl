#!/usr/bin/env node
// stt-export.mjs: export a window of a poopdeck.gl STT archive as showcase-format columns.
//
// Usage:
//   node stt-export.mjs <manifestUrl> --out <dir> [options]
//
// Options:
//   --id <id>            dataset id written to manifest.json (default: archive folder name)
//   --bbox w,s,e,n       lon/lat window (default: whole archive)
//   --time a,b           time window, ISO dates or Unix ms (default: whole archive)
//   --zoom <z>           tile zoom to read (default: archive minZoom, which has the fewest tile seams)
//   --max-features <n>   cap, seeded reservoir sample (default 200000)
//   --max-tiles <n>      refuse to fetch more tiles than this (default 400)
//   --attrs a,b | none   property columns to keep (default: all numeric, categorical <= 65535 values)
//   --stitch a,b         join path pieces that share these attribute values (tiles split tracks at every
//                        temporal bucket; e.g. drifters need --stitch drifter_id,segment). Sampling then
//                        picks whole tracks. Collects all pieces in the window first, so keep windows small.
//                        Stitch only on exact keys: gtfs-nl stores trip_id as a float32 attribute, so distinct
//                        trips collide (scripts/data/poopdeck-gtfs-nl stitches on its own key instead).
//   --clip-time          trim path vertices to the time window (needs per-vertex times)
//   --seed <n>           sampling seed (default 1)
//
// Point archives to trajectories (AIS vessels, flights, any "one row per report" archive):
//   --group-by <attr>    group point features by this attribute (e.g. mmsi, icao24), sort each group by time and
//                        emit one trajectory per group, with per-vertex timestamps. The attribute is written as a
//                        per-track column (uint32 when every value is an integer, else category codes), plus a
//                        dense `vesselIndex` (uint32) so tracks of one entity can be counted. --max-features then
//                        caps whole tracks (seeded sample). Points are read in one pass into columnar arrays, so a
//                        whole-archive z0 read of about 1.3 M points needs about 1 GB of heap at most.
//   --max-gap <min>      split a track where two consecutive raw reports are further apart than this (default 30).
//                        The test uses the previous report, not the last kept one, so thinning never splits.
//   --thin <s>           keep a report only if at least this many seconds after the last kept one (default 0).
//   --still-thin <s>     while an entity stays within --still-radius meters of its last kept report, require this
//                        many seconds between kept reports (default 0 = off). Thins moored vessels hard.
//   --still-radius <m>   radius of the "has not moved" test above (default 30).
//   --max-speed <kn>     drop a report whose implied speed from the last kept report exceeds this (default 0 = off);
//                        removes GPS teleports. Units are knots (for aircraft pass a large value or leave it off).
//   --min-points <n>     drop tracks with fewer kept reports (default 2).
//   --min-travel <m>     drop tracks whose first-to-last spread (bounding-box diagonal) is under this (default 0).
//   --keep-dwell <min>   keep a track that fails --min-travel when it lasts at least this many minutes (parked,
//                        anchored or moored entities; default 0 = off).
//   --max-vertices <n>   after --max-features, stop adding (seeded random) tracks once this many vertices are kept.
//   --vertex-values a,b  per-vertex numeric columns kept from the point attributes, written as <name>.bin with one
//                        value per vertex. Each entry is name[:dtype[*scale]]: dtype is float32 (default), uint8 or
//                        uint16; with a scale the stored integer is round(value * scale) and the manifest column
//                        gets "scale": 1/scale (real = stored * scale) and "noData" (255 / 65535 for NaN).
//                        Example: speed:uint8*2,heading:uint16
//   --vertex-range r     name:min:max[,name:min:max...] values outside the range become NaN (AIS: speed:0:102,
//                        heading:0:359.9; 102.3 kn and 511 deg mean "not available").
//                        Point archives with a z coordinate (flights: altitude) keep it (vertices has 3 components).
//   Other numeric attributes become per-track medians, categorical ones the first non-empty value.
//   --core <path>        path to @poopdeck.gl/core dist/index.js (default: sibling poopdeck.gl checkout)
//   --probe              print per-zoom tile and feature counts for the window, write nothing
//
// Output (little-endian, headerless typed arrays, see showcase src/data/README.md):
//   points:       position.bin (float32 x2|3), timestamp.bin (uint32, seconds since timeOrigin)
//   lines:        pathOffsets.bin, vertices.bin, startTime.bin, endTime.bin, timestamp.bin (per vertex, if present),
//                 vertexValue.bin (float32 per vertex, if present), <name>.bin per --vertex-values entry
//   polygons:     ringOffsets.bin, polygonRingOffsets.bin, vertices.bin, startTime.bin, endTime.bin
//   attributes:   <name>.bin (float32, or uint8/uint16 category codes with manifest categories)
//
// The decoder runs inline (no workers), over plain fetch with HTTP Range requests, so it works in Node 18+.

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {pathToFileURL} from 'node:url';

const DEFAULT_CORE = path.join(os.homedir(), 'Documents/GitHub/poopdeck.gl/packages/core/dist/index.js');

function parseArgs(argv) {
  const args = {_: []};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith('--')) args[key] = true;
      else { args[key] = next; i++; }
    } else args._.push(a);
  }
  return args;
}

function parseTime(s) {
  if (/^-?\d+$/.test(s) && s.length >= 10) return Number(s);
  const t = Date.parse(s);
  if (Number.isNaN(t)) throw new Error(`bad time: ${s}`);
  return t;
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function writeBin(dir, file, typed) {
  fs.writeFileSync(path.join(dir, file), new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength));
  return typed.byteLength;
}

const args = parseArgs(process.argv.slice(2));
if (args.help) {
  // print the header comment block of this file
  const header = fs.readFileSync(new URL(import.meta.url), 'utf8').split('\n');
  console.log(header.slice(1, header.findIndex((line, i) => i > 0 && !line.startsWith('//'))).map((line) => line.replace(/^\/\/ ?/, '')).join('\n'));
  process.exit(0);
}
const manifestUrl = args._[0];
if (!manifestUrl || (!args.out && !args.probe)) {
  console.error('usage: node stt-export.mjs <manifestUrl> --out <dir> [--bbox w,s,e,n] [--time a,b] [--max-features n] [--zoom z]');
  process.exit(2);
}

const core = await import(pathToFileURL(args.core || DEFAULT_CORE).href);
const {STTArchive, InlineTileDecoder} = core;

let bytesFetched = 0;
let requests = 0;
const countingFetch = async (input, init) => {
  const res = await fetch(input, init);
  requests++;
  const len = Number(res.headers.get('content-length'));
  if (Number.isFinite(len)) bytesFetched += len;
  return res;
};

const archive = new STTArchive({url: manifestUrl, decoder: new InlineTileDecoder(), opfsCache: false, fetch: countingFetch});
const meta = await archive.getMetadata();

const bbox = args.bbox
  ? (([w, s, e, n]) => ({minLon: w, minLat: s, maxLon: e, maxLat: n}))(args.bbox.split(',').map(Number))
  : {minLon: -180, minLat: -90, maxLon: 180, maxLat: 90};
const [tStart, tEnd] = args.time
  ? args.time.split(',').map(parseTime)
  : [meta.timeRange.start, meta.timeRange.end];
const zoom = args.zoom !== undefined ? Number(args.zoom) : meta.minZoom;
const maxFeatures = Number(args['max-features'] || 200000);
const maxTiles = Number(args['max-tiles'] || 400);
const clipTime = Boolean(args['clip-time']);
const stitchKeys = typeof args.stitch === 'string' ? args.stitch.split(',') : null;
const pieceCap = stitchKeys ? 4_000_000 : maxFeatures;
const rng = mulberry32(Number(args.seed || 1));
const groupBy = typeof args['group-by'] === 'string' ? args['group-by'] : null;
const maxGapMs = Number(args['max-gap'] ?? 30) * 60000;
const thinMs = Number(args.thin || 0) * 1000;
const stillThinMs = Number(args['still-thin'] || 0) * 1000;
const stillRadius = Number(args['still-radius'] || 30);
const maxSpeedKnots = Number(args['max-speed'] || 0);
const minPoints = Number(args['min-points'] || 2);
const minTravel = Number(args['min-travel'] || 0);
const keepDwellMs = Number(args['keep-dwell'] || 0) * 60000;
const maxVertices = Number(args['max-vertices'] || 0);
const vertexSpecs = (typeof args['vertex-values'] === 'string' ? args['vertex-values'].split(',') : []).map((entry) => {
  const m = /^([^:*]+)(?::(float32|uint8|uint16))?(?:\*([\d.]+))?$/.exec(entry);
  if (!m) throw new Error(`bad --vertex-values entry: ${entry}`);
  return {name: m[1], dtype: m[2] || 'float32', scale: m[3] ? Number(m[3]) : 1};
});
const vertexRanges = new Map(
  (typeof args['vertex-range'] === 'string' ? args['vertex-range'].split(',') : []).map((entry) => {
    const [name, lo, hi] = entry.split(':');
    return [name, [Number(lo), Number(hi)]];
  })
);

const log = (...m) => console.error('[stt-export]', ...m);
log(`archive "${meta.name}" z${meta.minZoom}-${meta.maxZoom}, bucket ${meta.temporalBucketMs} ms, partition ${meta.partition ?? 'replicated'}`);

const ids = await archive.getTileIdsInBounds(bbox, zoom, {start: tStart, end: tEnd});
log(`z${zoom}: ${ids.length} tiles overlap bbox and window`);

if (args.probe) {
  for (let z = meta.minZoom; z <= meta.maxZoom; z++) {
    const zIds = await archive.getTileIdsInBounds(bbox, z, {start: tStart, end: tEnd});
    let n = 0;
    if (zIds.length > 0 && zIds.length <= 60) {
      for (const t of await archive.getTiles(zIds)) if (t) for (const l of t.layers) n += l.features.featureCount;
    }
    console.log(`z${z}: ${zIds.length} tiles, ${zIds.length <= 60 ? n + ' features' : 'features not counted (>60 tiles)'}`);
  }
  console.log(`fetched ${bytesFetched} bytes in ${requests} requests`);
  process.exit(0);
}
if (ids.length === 0) { log('no tiles; widen the bbox or window'); process.exit(1); }
if (ids.length > maxTiles) { log(`${ids.length} tiles > --max-tiles ${maxTiles}; narrow the window or raise the limit`); process.exit(1); }

const METERS_PER_DEGREE = 111194.9;
function buildTracks() {
  const tracks = [];
  const wantsNumber = [...groups.keys()].every((k) => /^\d{1,10}$/.test(k) && Number(k) < 4294967296);
  let entityIndex = 0;
  const median = (a) => { a.sort((u, w) => u - w); return a[a.length >> 1]; };
  for (const [key, g] of groups) {
    const n = g.t.length;
    const order = Array.from({length: n}, (_, i) => i).sort((a, b) => g.t[a] - g.t[b]);
    const staticNum = {};
    for (const [k, a] of Object.entries(g.num)) staticNum[k] = median(a);
    const entity = entityIndex++;
    let cur = null;
    let previousReport = -Infinity;
    const flush = () => {
      if (!cur) return;
      const m = cur.t.length;
      let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
      for (let v = 0; v < m; v++) {
        xmin = Math.min(xmin, cur.p[v * dims]); xmax = Math.max(xmax, cur.p[v * dims]);
        ymin = Math.min(ymin, cur.p[v * dims + 1]); ymax = Math.max(ymax, cur.p[v * dims + 1]);
      }
      const midLat = ((ymin + ymax) / 2) * Math.PI / 180;
      const spread = Math.hypot((xmax - xmin) * Math.cos(midLat), ymax - ymin) * METERS_PER_DEGREE;
      if (m >= minPoints && (spread >= minTravel || (keepDwellMs && cur.t[m - 1] - cur.t[0] >= keepDwellMs))) {
        const rec = {
          start: cur.t[0], end: cur.t[m - 1], id: null,
          pos: Float32Array.from(cur.p), vt: Float64Array.from(cur.t), vvs: {},
          num: {...staticNum, [`${groupBy}`]: wantsNumber ? Number(key) : undefined, vesselIndex: entity},
          cat: {...g.cat}
        };
        if (!wantsNumber) { rec.cat[groupBy] = key; delete rec.num[groupBy]; } else rec.num[groupBy] = Number(key);
        for (const spec of vertexSpecs) rec.vvs[spec.name] = Float32Array.from(cur.v[spec.name]);
        tracks.push(rec);
      }
      cur = null;
    };
    const start = () => {
      cur = {t: [], p: [], v: Object.fromEntries(vertexSpecs.map((s) => [s.name, []]))};
    };
    for (let oi = 0; oi < n; oi++) {
      const i = order[oi];
      const t = g.t[i];
      const x = g.p[i * dims];
      const y = g.p[i * dims + 1];
      const previousT = previousReport;
      previousReport = t;
      if (cur) {
        const m = cur.t.length;
        const dt = t - cur.t[m - 1];
        if (dt <= 0) continue; // duplicate timestamp
        const dx = (x - cur.p[(m - 1) * dims]) * Math.cos((y * Math.PI) / 180);
        const dy = y - cur.p[(m - 1) * dims + 1];
        const moved = Math.hypot(dx, dy) * METERS_PER_DEGREE;
        // the gap test uses the previous report, not the last kept one, so thinning never splits a track
        if (t - previousT > maxGapMs) flush();
        else {
          if (maxSpeedKnots && moved / (dt / 1000) > maxSpeedKnots * 0.514444) continue;
          if (dt < thinMs) continue;
          if (stillThinMs && moved < stillRadius && dt < stillThinMs) continue;
        }
      }
      if (!cur) start();
      cur.t.push(t);
      for (let d = 0; d < dims; d++) cur.p.push(g.p[i * dims + d]);
      for (const spec of vertexSpecs) cur.v[spec.name].push(g.v[spec.name][i]);
    }
    flush();
  }
  return tracks;
}

// ---- collect features (reservoir sample) -------------------------------------------------
const keep = [];
let seen = 0;
let passing = 0;
let geometryType = null;
let dims = 2;
let hasVertexTimes = false;
let hasVertexValues = false;
const dedupe = new Set();
const numericNames = new Set();
const categoricalNames = new Set();
let layerName = null;

// ---- --group-by: columnar point collection ------------------------------------------------
const groups = new Map(); // key -> {t: number[], p: number[], v: {name: number[]}, num: {name: number[]}, cat: {}}
let pointsRead = 0;
let pointsInWindow = 0;
function collectPoints(f) {
  const wantedAttrs = typeof args.attrs === 'string' && args.attrs !== 'none' ? new Set(args.attrs.split(',')) : null;
  const vertexNames = new Set(vertexSpecs.map((v) => v.name));
  for (let i = 0; i < f.featureCount; i++) {
    pointsRead++;
    const t = f.startTimes[i] + f.timeOffset;
    if (t < tStart || t > tEnd) continue;
    const x = f.positions[i * dims];
    const y = f.positions[i * dims + 1];
    if (!(x >= bbox.minLon && x <= bbox.maxLon && y >= bbox.minLat && y <= bbox.maxLat)) continue;
    let key;
    if (groupBy in f.categoricalProps) {
      const c = f.categoricalProps[groupBy];
      const code = c.indices[i];
      key = code === 0xffff ? '' : c.categories[code];
    } else if (groupBy in f.numericProps) key = String(f.numericProps[groupBy][i]);
    else throw new Error(`--group-by ${groupBy}: no such attribute (have ${[...Object.keys(f.numericProps), ...Object.keys(f.categoricalProps)].join(', ')})`);
    if (key === '') continue;
    pointsInWindow++;
    let g = groups.get(key);
    if (!g) { g = {t: [], p: [], v: {}, num: {}, cat: {}}; groups.set(key, g); }
    g.t.push(t);
    for (let d = 0; d < dims; d++) g.p.push(f.positions[i * dims + d]);
    for (const spec of vertexSpecs) {
      let value = f.numericProps[spec.name] ? f.numericProps[spec.name][i] : NaN;
      const range = vertexRanges.get(spec.name);
      if (range && !(value >= range[0] && value <= range[1])) value = NaN;
      (g.v[spec.name] ??= []).push(value);
    }
    for (const k of Object.keys(f.numericProps)) {
      if (vertexNames.has(k) || (wantedAttrs && !wantedAttrs.has(k)) || k === groupBy) continue;
      const value = f.numericProps[k][i];
      if (Number.isFinite(value)) (g.num[k] ??= []).push(value);
    }
    for (const [k, c] of Object.entries(f.categoricalProps)) {
      if ((wantedAttrs && !wantedAttrs.has(k)) || k === groupBy || g.cat[k]) continue;
      const code = c.indices[i];
      if (code !== 0xffff && c.categories[code] !== '') g.cat[k] = c.categories[code];
    }
  }
}

const BATCH = 8;
for (let b = 0; b < ids.length; b += BATCH) {
  const tiles = await archive.getTiles(ids.slice(b, b + BATCH));
  for (const tile of tiles) {
    if (!tile) continue;
    for (const layer of tile.layers) {
      if (layer.name === 'summary') continue; // summary-tier aggregates, not raw features
      const f = layer.features;
      if (!f.featureCount) continue;
      layerName ??= layer.name;
      geometryType ??= f.geometryType;
      if (f.geometryType !== geometryType) continue;
      dims = f.positionDimensions || 2;
      hasVertexTimes ||= Boolean(f.vertexTimestamps);
      hasVertexValues ||= Boolean(f.vertexValues);
      for (const k of Object.keys(f.numericProps)) numericNames.add(k);
      for (const k of Object.keys(f.categoricalProps)) categoricalNames.add(k);
      const isPoint = f.geometryType === 0;
      const starts = f.startIndices;
      if (groupBy) {
        if (!isPoint) throw new Error('--group-by needs a point archive');
        collectPoints(f);
        continue;
      }
      for (let i = 0; i < f.featureCount; i++) {
        const from = isPoint ? i : starts[i];
        const to = isPoint ? i + 1 : starts[i + 1];
        const start = f.startTimes[i] + f.timeOffset;
        const end = f.endTimes[i] + f.timeOffset;
        if (isPoint ? start < tStart || start > tEnd : end < tStart || start > tEnd) continue;
        // tile start/end can be the bucket bounds, so test the real vertex times when present
        if (!isPoint && f.vertexTimestamps && (f.vertexTimestamps[to - 1] + f.timeOffset < tStart || f.vertexTimestamps[from] + f.timeOffset > tEnd)) continue;
        // bbox: any vertex inside keeps the whole feature
        let inside = false;
        for (let v = from; v < to && !inside; v++) {
          const x = f.positions[v * dims];
          const y = f.positions[v * dims + 1];
          inside = x >= bbox.minLon && x <= bbox.maxLon && y >= bbox.minLat && y <= bbox.maxLat;
        }
        if (!inside) continue;
        const key = f.featureIds64 ? `${f.featureIds64[i]}:${start}:${to - from}` : null;
        if (key) { if (dedupe.has(key)) continue; dedupe.add(key); }
        passing++;
        seen++;
        let slot = keep.length;
        if (keep.length >= pieceCap) {
          slot = Math.floor(rng() * seen);
          if (slot >= pieceCap) continue;
        }
        const rec = {start, end, id: f.featureIds64 ? f.featureIds64[i] : null, num: {}, cat: {}};
        let lo = from, hi = to;
        if (clipTime && f.vertexTimestamps) {
          while (hi - lo > 0 && f.vertexTimestamps[lo] + f.timeOffset < tStart) lo++;
          while (hi - lo > 0 && f.vertexTimestamps[hi - 1] + f.timeOffset > tEnd) hi--;
        }
        if (!isPoint && hi - lo < 2) continue; // clipped away
        rec.pos = Float32Array.from(f.positions.subarray(lo * dims, hi * dims));
        if (!isPoint) {
          if (f.vertexTimestamps) {
            rec.vt = new Float64Array(hi - lo);
            for (let v = lo; v < hi; v++) rec.vt[v - lo] = f.vertexTimestamps[v] + f.timeOffset;
            rec.start = rec.vt[0];
            rec.end = rec.vt[hi - lo - 1];
          }
          if (f.vertexValues) rec.vv = Float32Array.from(f.vertexValues.subarray(lo, hi));
          if (f.geometryType === 2) {
            // ring boundaries (vertex indices) that fall inside this polygon
            const rings = [];
            const ri = f.ringIndices;
            if (ri) for (let r = 0; r < ri.length; r++) if (ri[r] >= from && ri[r] <= to) rings.push(ri[r] - from);
            rec.rings = rings.length ? rings : [0, to - from];
          }
        }
        for (const k of Object.keys(f.numericProps)) rec.num[k] = f.numericProps[k][i];
        for (const [k, c] of Object.entries(f.categoricalProps)) {
          const code = c.indices[i];
          rec.cat[k] = code === 0xffff ? '' : c.categories[code];
        }
        if (slot === keep.length) keep.push(rec); else keep[slot] = rec;
      }
    }
  }
  if ((b / BATCH) % 5 === 0) log(`tiles ${Math.min(b + BATCH, ids.length)}/${ids.length}, ${passing} features in window, kept ${keep.length}`);
}

if (groupBy) {
  if (groups.size === 0) { log('no points in window'); process.exit(1); }
  log(`read ${pointsRead} points, ${pointsInWindow} in window, ${groups.size} groups by ${groupBy}`);
  const toTracks = buildTracks();
  geometryType = 1;
  hasVertexTimes = true;
  hasVertexValues = false;
  numericNames.clear();
  categoricalNames.clear();
  for (const r of toTracks) {
    for (const k of Object.keys(r.num)) numericNames.add(k);
    for (const k of Object.keys(r.cat)) categoricalNames.add(k);
  }
  // seeded shuffle, then cap whole tracks
  for (let i = toTracks.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [toTracks[i], toTracks[j]] = [toTracks[j], toTracks[i]];
  }
  passing = toTracks.length;
  let vertexBudget = 0;
  for (const r of toTracks.slice(0, maxFeatures)) {
    const m = r.pos.length / dims;
    if (maxVertices && vertexBudget + m > maxVertices) continue;
    vertexBudget += m;
    keep.push(r);
  }
  log(`built ${toTracks.length} tracks, kept ${keep.length}`);
}
if (keep.length === 0) { log('no features in window'); process.exit(1); }
if (stitchKeys) {
  const groups = new Map();
  for (const r of keep) {
    const key = stitchKeys.map((k) => (k in r.cat ? r.cat[k] : r.num[k])).join('|');
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  const merged = [];
  for (const pieces of groups.values()) {
    pieces.sort((a, b) => a.start - b.start);
    const first = pieces[0];
    const parts = [];
    for (const p of pieces) {
      const prev = parts[parts.length - 1];
      let skip = 0;
      if (prev) {
        const m = prev.pos.length / dims;
        const same = prev.pos[(m - 1) * dims] === p.pos[0] && prev.pos[(m - 1) * dims + 1] === p.pos[1];
        if (same) skip = 1;
      }
      parts.push({pos: p.pos, vt: p.vt, vv: p.vv, skip});
    }
    const total = parts.reduce((s, q) => s + q.pos.length / dims - q.skip, 0);
    const rec = {...first, end: pieces[pieces.length - 1].end, pos: new Float32Array(total * dims)};
    if (hasVertexTimes) rec.vt = new Float64Array(total);
    if (hasVertexValues) rec.vv = new Float32Array(total).fill(NaN);
    let o = 0;
    for (const q of parts) {
      const m = q.pos.length / dims - q.skip;
      rec.pos.set(q.pos.subarray(q.skip * dims), o * dims);
      if (q.vt && rec.vt) rec.vt.set(q.vt.subarray(q.skip), o);
      if (q.vv && rec.vv) rec.vv.set(q.vv.subarray(q.skip), o);
      o += m;
    }
    merged.push(rec);
  }
  keep.length = 0;
  // seeded shuffle, then cap whole tracks
  for (let i = merged.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [merged[i], merged[j]] = [merged[j], merged[i]];
  }
  log(`stitched ${passing} pieces into ${merged.length} tracks`);
  passing = merged.length;
  keep.push(...merged.slice(0, maxFeatures));
}
keep.sort((a, b) => a.start - b.start);
const originMs = Math.floor(keep[0].start / 1000) * 1000;
const toSec = (ms) => Math.max(0, Math.round((ms - originMs) / 1000));

// ---- write -------------------------------------------------------------------------------
const out = args.out;
fs.mkdirSync(out, {recursive: true});
const id = args.id || meta.name || manifestUrl.split('/').slice(-2, -1)[0];
const columns = {};
const sizes = {};
const n = keep.length;
const kind = geometryType === 0 ? 'points' : geometryType === 1 ? (hasVertexTimes ? 'trajectories' : 'lines') : 'polygons';

function addColumn(name, file, typed, extra = {}) {
  sizes[file] = writeBin(out, file, typed);
  const dtype = typed instanceof Float32Array ? 'float32' : typed instanceof Float64Array ? 'float64'
    : typed instanceof Uint32Array ? 'uint32' : typed instanceof Uint16Array ? 'uint16' : 'uint8';
  columns[name] = {file, dtype, ...extra, length: extra.components ? typed.length / extra.components : typed.length};
}

let vertexCount = n;
if (geometryType === 0) {
  const pos = new Float32Array(n * dims);
  const ts = new Uint32Array(n);
  keep.forEach((r, i) => { pos.set(r.pos, i * dims); ts[i] = toSec(r.start); });
  addColumn('position', 'position.bin', pos, {components: dims});
  addColumn('timestamp', 'time.bin', ts, {unit: `seconds since ${new Date(originMs).toISOString()}`});
} else {
  vertexCount = keep.reduce((s, r) => s + r.pos.length / dims, 0);
  const offsets = new Uint32Array(n + 1);
  const verts = new Float32Array(vertexCount * dims);
  const startT = new Uint32Array(n);
  const endT = new Uint32Array(n);
  const vtime = hasVertexTimes ? new Uint32Array(vertexCount) : null;
  const vval = hasVertexValues ? new Float32Array(vertexCount).fill(NaN) : null;
  let o = 0;
  keep.forEach((r, i) => {
    offsets[i] = o;
    verts.set(r.pos, o * dims);
    const m = r.pos.length / dims;
    if (vtime && r.vt) for (let v = 0; v < m; v++) vtime[o + v] = toSec(r.vt[v]);
    if (vval && r.vv) vval.set(r.vv, o);
    startT[i] = toSec(r.start);
    endT[i] = toSec(r.end);
    o += m;
  });
  offsets[n] = o;
  if (geometryType === 1) addColumn('pathOffsets', 'pathOffsets.bin', offsets);
  addColumn('vertices', 'vertices.bin', verts, {components: dims});
  addColumn('startTime', 'startTime.bin', startT, {unit: `seconds since ${new Date(originMs).toISOString()}`});
  addColumn('endTime', 'endTime.bin', endT, {unit: `seconds since ${new Date(originMs).toISOString()}`});
  if (vtime) addColumn('timestamp', 'time.bin', vtime, {unit: `seconds since ${new Date(originMs).toISOString()}`});
  if (vval) addColumn('vertexValue', 'vertexValue.bin', vval, {note: 'per-vertex scalar, NaN = none'});
  for (const spec of vertexSpecs) {
    const total = new Float32Array(vertexCount);
    let q = 0;
    for (const r of keep) { total.set(r.vvs[spec.name], q); q += r.pos.length / dims; }
    if (spec.dtype === 'float32') { addColumn(spec.name, `${spec.name}.bin`, total, {note: 'per-vertex, NaN = none'}); continue; }
    const max = spec.dtype === 'uint8' ? 255 : 65535;
    const q8 = spec.dtype === 'uint8' ? new Uint8Array(vertexCount) : new Uint16Array(vertexCount);
    for (let v = 0; v < vertexCount; v++) q8[v] = Number.isFinite(total[v]) ? Math.min(max - 1, Math.max(0, Math.round(total[v] * spec.scale))) : max;
    addColumn(spec.name, `${spec.name}.bin`, q8, {note: 'per-vertex, quantised', scale: 1 / spec.scale, noData: max});
  }
  if (geometryType === 2) {
    // ringOffsets over all vertices (global), polygonRingOffsets over rings
    const ringOffsets = [];
    const polygonRingOffsets = [0];
    let base = 0;
    for (const r of keep) {
      for (let k = 0; k < r.rings.length - 1; k++) ringOffsets.push(base + r.rings[k]);
      polygonRingOffsets.push(ringOffsets.length);
      base += r.pos.length / dims;
    }
    ringOffsets.push(base);
    addColumn('ringOffsets', 'ringOffsets.bin', Uint32Array.from(ringOffsets));
    addColumn('polygonRingOffsets', 'polygonRingOffsets.bin', Uint32Array.from(polygonRingOffsets));
  }
}

// attributes
const wanted = args.attrs === 'none' ? [] : typeof args.attrs === 'string' ? args.attrs.split(',') : null;
const droppedAttrs = [];
for (const name of numericNames) {
  if (name !== 'vesselIndex' && wanted && !wanted.includes(name) && name !== groupBy) continue;
  if (groupBy && (name === groupBy || name === 'vesselIndex')) {
    addColumn(name, `${name}.bin`, Uint32Array.from(keep, (r) => r.num[name] ?? 0));
    continue;
  }
  addColumn(name, `${name}.bin`, Float32Array.from(keep, (r) => r.num[name] ?? NaN));
}
for (const name of categoricalNames) {
  if (wanted && !wanted.includes(name) && name !== groupBy) continue;
  const dict = new Map();
  for (const r of keep) { const v = r.cat[name] ?? ''; if (!dict.has(v)) dict.set(v, dict.size); }
  if (!wanted && dict.size > 4096) { droppedAttrs.push(`${name} (${dict.size} distinct strings)`); continue; }
  if (dict.size > 65535) { droppedAttrs.push(`${name} (${dict.size} distinct strings)`); continue; }
  const categories = [...dict.keys()];
  const codes = dict.size <= 256 ? new Uint8Array(n) : new Uint16Array(n);
  keep.forEach((r, i) => { codes[i] = dict.get(r.cat[name] ?? ''); });
  addColumn(name, `${name}.bin`, codes, {categories});
}

let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
for (const r of keep) for (let v = 0; v < r.pos.length; v += dims) {
  minX = Math.min(minX, r.pos[v]); maxX = Math.max(maxX, r.pos[v]);
  minY = Math.min(minY, r.pos[v + 1]); maxY = Math.max(maxY, r.pos[v + 1]);
}
const maxEnd = keep.reduce((m, r) => Math.max(m, r.end), 0);

const manifest = {
  id,
  version: 1,
  kind,
  count: n,
  bbox: [minX, minY, maxX, maxY],
  crs: 'EPSG:4326',
  columns,
  properties: {
    timeOriginMs: originMs,
    timeRangeMs: [originMs, maxEnd],
    vertexCount,
    source: {
      manifestUrl,
      name: meta.name,
      description: meta.description,
      attribution: meta.attribution,
      layer: layerName,
      exportedFrom: 'stt-export.mjs',
      zoom,
      window: {bbox: [bbox.minLon, bbox.minLat, bbox.maxLon, bbox.maxLat], time: [tStart, tEnd]},
      featuresInWindow: passing,
      sampled: passing > n
    },
    droppedAttributes: droppedAttrs
  }
};
fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 2));

const total = Object.values(sizes).reduce((a, b) => a + b, 0);
console.log(JSON.stringify({
  id, kind, features: n, featuresInWindow: passing, vertices: vertexCount, tiles: ids.length,
  downloadBytes: bytesFetched, httpRequests: requests, outputBytes: total, files: sizes, dropped: droppedAttrs
}, null, 2));
process.exit(0);
