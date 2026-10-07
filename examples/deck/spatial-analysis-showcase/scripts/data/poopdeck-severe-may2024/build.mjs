#!/usr/bin/env node
// build.mjs: rebuild the five poopdeck severe-weather datasets (21-22 May 2024) for the showcase.
//
//   node build.mjs [--out <public/data dir>] [--only tracks,lightning,reports,warnings,outages]
//
// Reads poopdeck.gl STT archives (tiles.poopdeck.gl) through @poopdeck.gl/core (sibling checkout,
// prebuilt dist), keeps the window ORIGIN .. END, and writes showcase-format columns. Every dataset
// uses one shared time origin so the scenes can combine them: uint32 seconds since 2024-05-21T12:00Z.
//
//   poopdeck-mrms-precip-tracks    MRMS storm-cell tracks, hourly pieces chained back into tracks
//   poopdeck-goes-glm-lightning    GOES-16 GLM flashes (z5 raw layer), seeded sample
//   poopdeck-mrms-storm3d-reports  NOAA SPC preliminary storm reports
//   poopdeck-mrms-storm3d-warnings NWS warning polygons with issue/valid-until times
//   poopdeck-mrms-storm3d-outages  DOE EAGLE-I county outage snapshots (table keyed by county FIPS)

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(
  process.argv
    .slice(2)
    .reduce((pairs, value, index, all) => {
      if (value.startsWith('--')) pairs.push([value.slice(2), all[index + 1]?.startsWith('--') ? 'true' : (all[index + 1] ?? 'true')]);
      return pairs;
    }, [])
);
const OUT = path.resolve(args.out ?? path.join(HERE, '../../../public/data'));
const ONLY = new Set((args.only ?? 'tracks,lightning,reports,warnings,outages').split(','));
const CORE = path.join(os.homedir(), 'Documents/GitHub/poopdeck.gl/packages/core/dist/index.js');
const {STTArchive, InlineTileDecoder} = await import(pathToFileURL(CORE).href);

const ORIGIN_MS = Date.parse('2024-05-21T12:00:00Z');
const END_MS = Date.parse('2024-05-22T06:00:00Z');
const ORIGIN_TEXT = 'seconds since 2024-05-21T12:00:00Z';
const BASE = 'https://tiles.poopdeck.gl/data';
const WORLD = {minLon: -180, minLat: -90, maxLon: 180, maxLat: 90};
const LIGHTNING_CAP = 150000;
const log = (...message) => console.error('[severe-build]', ...message);

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Streams the raw (non-summary) features of an archive window; `visit(f, i)` gets one feature. */
async function readFeatures(name, zoom, startMs, endMs, visit) {
  const archive = new STTArchive({url: `${BASE}/${name}/manifest.json`, decoder: new InlineTileDecoder(), opfsCache: false});
  const ids = await archive.getTileIdsInBounds(WORLD, zoom, {start: startMs, end: endMs});
  log(`${name} z${zoom}: ${ids.length} tiles`);
  for (let b = 0; b < ids.length; b += 16) {
    for (const tile of await archive.getTiles(ids.slice(b, b + 16))) {
      if (!tile) continue;
      for (const layer of tile.layers) {
        if (layer.name === 'summary') continue;
        const f = layer.features;
        for (let i = 0; i < f.featureCount; i++) visit(f, i);
      }
    }
  }
}

const seconds = ms => Math.max(0, Math.round((ms - ORIGIN_MS) / 1000));
const category = (f, name, i) => {
  const c = f.categoricalProps[name];
  const code = c.indices[i];
  return code === 0xffff ? '' : c.categories[code];
};

function writeBin(dir, file, typed) {
  fs.writeFileSync(path.join(dir, file), new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength));
  return typed.byteLength;
}
function writeDataset(id, manifest, files) {
  const dir = path.join(OUT, id);
  fs.rmSync(dir, {recursive: true, force: true});
  fs.mkdirSync(dir, {recursive: true});
  let bytes = 0;
  for (const [file, typed] of Object.entries(files)) bytes += writeBin(dir, file, typed);
  fs.writeFileSync(path.join(dir, 'manifest.json'), `${JSON.stringify({id, version: 1, crs: 'EPSG:4326', ...manifest}, null, 1)}\n`);
  log(`${id}: ${(bytes / 1e6).toFixed(2)} MB`);
}
function bboxOf(positions) {
  const box = [Infinity, Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 2) {
    box[0] = Math.min(box[0], positions[i]);
    box[1] = Math.min(box[1], positions[i + 1]);
    box[2] = Math.max(box[2], positions[i]);
    box[3] = Math.max(box[3], positions[i + 1]);
  }
  return box.map(value => Number(value.toFixed(4)));
}
const source = (name, extra = {}) => ({manifestUrl: `${BASE}/${name}/manifest.json`, name, exportedFrom: 'poopdeck-severe-may2024/build.mjs', ...extra});
const properties = (name, extra) => ({timeOriginMs: ORIGIN_MS, timeRangeMs: [ORIGIN_MS, END_MS], source: source(name), ...extra});

// ---- tracks ----------------------------------------------------------------------------------
async function buildTracks() {
  const pieces = new Map();
  await readFeatures('mrms-precip-tracks', 0, ORIGIN_MS, END_MS, (f, i) => {
    const from = f.startIndices[i];
    const to = f.startIndices[i + 1];
    if (!f.vertexTimestamps || to - from < 2) return;
    const id = String(f.featureIds64[i]);
    if (pieces.has(id)) return;
    const t = [];
    const p = [];
    const v = [];
    for (let k = from; k < to; k++) {
      t.push(f.vertexTimestamps[k] + f.timeOffset);
      p.push(f.positions[k * 2], f.positions[k * 2 + 1]);
      v.push(f.vertexValues ? f.vertexValues[k] : NaN);
    }
    pieces.set(id, {id, t, p, v});
  });
  const list = [...pieces.values()];
  const key = (x, y) => `${x},${y}`;
  const starts = new Map();
  for (const piece of list) {
    const k = key(piece.p[0], piece.p[1]);
    if (!starts.has(k)) starts.set(k, []);
    starts.get(k).push(piece);
  }
  const successor = new Map();
  const hasPredecessor = new Set();
  for (const piece of list.sort((a, b) => a.t[0] - b.t[0])) {
    const n = piece.t.length;
    const candidates = (starts.get(key(piece.p[(n - 1) * 2], piece.p[(n - 1) * 2 + 1])) ?? []).filter(
      other => other !== piece && !hasPredecessor.has(other.id) && other.t[0] >= piece.t[n - 1] - 1000 && other.t[0] - piece.t[n - 1] <= 15 * 60000
    );
    if (candidates.length) {
      candidates.sort((a, b) => a.t[0] - b.t[0]);
      successor.set(piece.id, candidates[0]);
      hasPredecessor.add(candidates[0].id);
    }
  }
  const tracks = [];
  for (const piece of list) {
    if (hasPredecessor.has(piece.id)) continue;
    const t = [];
    const p = [];
    const v = [];
    for (let cur = piece; cur; cur = successor.get(cur.id)) {
      for (let k = 0; k < cur.t.length; k++) {
        // The archive cuts tracks at hourly buckets and repeats the boundary vertex a second later.
        if (t.length && cur.t[k] - t[t.length - 1] < 90000) continue;
        t.push(cur.t[k]);
        p.push(cur.p[k * 2], cur.p[k * 2 + 1]);
        v.push(cur.v[k]);
      }
    }
    tracks.push({t, p, v});
  }
  log(`tracks: ${list.length} pieces -> ${tracks.length} chains`);
  const kept = tracks.filter(track => track.t.length >= 3 && track.t[track.t.length - 1] - track.t[0] >= 15 * 60000 && track.t[track.t.length - 1] >= ORIGIN_MS);
  kept.sort((a, b) => a.t[0] - b.t[0]);
  const histogram = {};
  for (const track of kept) histogram[track.t.length] = (histogram[track.t.length] ?? 0) + 1;
  log('chain vertex-count histogram', JSON.stringify(histogram));
  let vertexCount = 0;
  for (const track of kept) vertexCount += track.t.length;
  const offsets = new Uint32Array(kept.length + 1);
  const vertices = new Float32Array(vertexCount * 2);
  const times = new Uint32Array(vertexCount);
  const reflectivity = new Float32Array(vertexCount);
  const peak = new Float32Array(kept.length);
  const startTime = new Uint32Array(kept.length);
  const endTime = new Uint32Array(kept.length);
  let row = 0;
  kept.forEach((track, index) => {
    offsets[index] = row;
    startTime[index] = seconds(track.t[0]);
    endTime[index] = seconds(track.t[track.t.length - 1]);
    peak[index] = Math.max(...track.v.filter(Number.isFinite));
    for (let k = 0; k < track.t.length; k++, row++) {
      vertices[row * 2] = track.p[k * 2];
      vertices[row * 2 + 1] = track.p[k * 2 + 1];
      times[row] = seconds(track.t[k]);
      reflectivity[row] = track.v[k];
    }
  });
  offsets[kept.length] = row;
  const n = kept.length;
  writeDataset(
    'poopdeck-mrms-precip-tracks',
    {
      kind: 'trajectories',
      count: n,
      bbox: bboxOf(vertices),
      columns: {
        pathOffsets: {file: 'pathOffsets.bin', dtype: 'uint32', length: n + 1},
        vertices: {file: 'vertices.bin', dtype: 'float32', components: 2, length: vertexCount},
        timestamp: {file: 'time.bin', dtype: 'uint32', length: vertexCount, unit: ORIGIN_TEXT},
        reflectivity: {file: 'reflectivity.bin', dtype: 'float32', length: vertexCount, unit: 'dBZ at the cell'},
        startTime: {file: 'startTime.bin', dtype: 'uint32', length: n, unit: ORIGIN_TEXT},
        endTime: {file: 'endTime.bin', dtype: 'uint32', length: n, unit: ORIGIN_TEXT},
        peakDbz: {file: 'peakDbz.bin', dtype: 'float32', length: n, unit: 'dBZ'}
      },
      properties: properties('mrms-precip-tracks', {vertexCount, note: 'Hourly pieces of the archive chained by matching end and start vertices; chains under 3 vertices or 15 minutes dropped.'})
    },
    {'pathOffsets.bin': offsets, 'vertices.bin': vertices, 'time.bin': times, 'reflectivity.bin': reflectivity, 'startTime.bin': startTime, 'endTime.bin': endTime, 'peakDbz.bin': peak}
  );
}

// ---- lightning -------------------------------------------------------------------------------
async function buildLightning() {
  const rng = mulberry32(7);
  const keep = [];
  let seen = 0;
  const dedupe = new Set();
  await readFeatures('goes-glm-lightning', 5, ORIGIN_MS, END_MS, (f, i) => {
    const t = f.startTimes[i] + f.timeOffset;
    if (t < ORIGIN_MS || t > END_MS) return;
    const x = f.positions[i * 2];
    const y = f.positions[i * 2 + 1];
    const k = `${t}:${x}:${y}:${f.numericProps.energy_fj[i]}`;
    if (dedupe.has(k)) return;
    dedupe.add(k);
    seen++;
    const record = [x, y, seconds(t), f.numericProps.energy_fj[i], f.numericProps.area_km2[i]];
    if (keep.length < LIGHTNING_CAP) keep.push(record);
    else {
      const slot = Math.floor(rng() * seen);
      if (slot < LIGHTNING_CAP) keep[slot] = record;
    }
  });
  log(`lightning: ${seen} flashes in window, kept ${keep.length}`);
  keep.sort((a, b) => a[2] - b[2]);
  const n = keep.length;
  const positions = new Float32Array(n * 2);
  const times = new Uint32Array(n);
  const energy = new Float32Array(n);
  const area = new Float32Array(n);
  keep.forEach((record, index) => {
    positions.set([record[0], record[1]], index * 2);
    times[index] = record[2];
    energy[index] = record[3];
    area[index] = record[4];
  });
  writeDataset(
    'poopdeck-goes-glm-lightning',
    {
      kind: 'points',
      count: n,
      bbox: bboxOf(positions),
      columns: {
        position: {file: 'position.bin', dtype: 'float32', components: 2, length: n},
        timestamp: {file: 'time.bin', dtype: 'uint32', length: n, unit: ORIGIN_TEXT},
        energy: {file: 'energy.bin', dtype: 'float32', length: n, unit: 'femtojoules (optical energy of the flash)'},
        area: {file: 'area.bin', dtype: 'float32', length: n, unit: 'km2 (flash footprint)'}
      },
      properties: properties('goes-glm-lightning', {flashesInWindow: seen, sampleFraction: n / seen, sampled: seen > n})
    },
    {'position.bin': positions, 'time.bin': times, 'energy.bin': energy, 'area.bin': area}
  );
}

// ---- reports ---------------------------------------------------------------------------------
async function buildReports() {
  const rows = [];
  const dedupe = new Set();
  const kinds = [];
  await readFeatures('mrms-storm3d-reports', 3, ORIGIN_MS, END_MS, (f, i) => {
    const t = f.startTimes[i] + f.timeOffset;
    if (t < ORIGIN_MS || t > END_MS) return;
    const x = f.positions[i * 2];
    const y = f.positions[i * 2 + 1];
    const kind = category(f, 'kind', i);
    const k = `${t}:${x}:${y}:${kind}:${f.numericProps.magnitude?.[i]}`;
    if (dedupe.has(k)) return;
    dedupe.add(k);
    if (!kinds.includes(kind)) kinds.push(kind);
    rows.push({x, y, t: seconds(t), magnitude: f.numericProps.magnitude ? f.numericProps.magnitude[i] : NaN, kind});
  });
  rows.sort((a, b) => a.t - b.t);
  const order = ['tornado', 'wind', 'hail', 'flood', 'damage', 'other'].filter(kind => kinds.includes(kind));
  for (const kind of kinds) if (!order.includes(kind)) order.push(kind);
  const n = rows.length;
  const positions = new Float32Array(n * 2);
  const times = new Uint32Array(n);
  const magnitude = new Float32Array(n);
  const kind = new Uint8Array(n);
  rows.forEach((row, index) => {
    positions.set([row.x, row.y], index * 2);
    times[index] = row.t;
    magnitude[index] = row.magnitude;
    kind[index] = order.indexOf(row.kind);
  });
  writeDataset(
    'poopdeck-mrms-storm3d-reports',
    {
      kind: 'points',
      count: n,
      bbox: bboxOf(positions),
      columns: {
        position: {file: 'position.bin', dtype: 'float32', components: 2, length: n},
        timestamp: {file: 'time.bin', dtype: 'uint32', length: n, unit: ORIGIN_TEXT},
        magnitude: {file: 'magnitude.bin', dtype: 'float32', length: n, unit: 'hail inches, wind mph, tornado EF/NaN, 0 or NaN when not reported'},
        kind: {file: 'kind.bin', dtype: 'uint8', length: n, categories: order}
      },
      properties: properties('mrms-storm3d-reports')
    },
    {'position.bin': positions, 'time.bin': times, 'magnitude.bin': magnitude, 'kind.bin': kind}
  );
}

// ---- warnings --------------------------------------------------------------------------------
async function buildWarnings() {
  const items = [];
  const dedupe = new Set();
  await readFeatures('mrms-storm3d-warnings', 3, ORIGIN_MS, END_MS, (f, i) => {
    const from = f.startIndices[i];
    const to = f.startIndices[i + 1];
    const id = String(f.featureIds64[i]);
    if (dedupe.has(id)) return;
    dedupe.add(id);
    const rings = [];
    const ringIndices = f.ringIndices;
    if (ringIndices) for (let r = 0; r < ringIndices.length; r++) if (ringIndices[r] >= from && ringIndices[r] <= to) rings.push(ringIndices[r] - from);
    if (rings.length < 2) rings.splice(0, rings.length, 0, to - from);
    if (rings[0] !== 0) rings.unshift(0);
    if (rings[rings.length - 1] !== to - from) rings.push(to - from);
    const ring = [];
    for (let k = from; k < to; k++) ring.push(f.positions[k * 2], f.positions[k * 2 + 1]);
    items.push({
      start: f.startTimes[i] + f.timeOffset,
      end: f.endTimes[i] + f.timeOffset,
      phenom: category(f, 'phenom', i),
      etn: category(f, 'etn', i),
      ring,
      rings
    });
  });
  // Warning polygons are storm-based and re-issued every few minutes. Chain the versions of one
  // warning (same event number and phenomenon, consecutive intervals, nearby centroid) so each
  // version knows when the warning was first issued.
  const centroid = item => {
    let x = 0;
    let y = 0;
    for (let k = 0; k < item.ring.length; k += 2) {
      x += item.ring[k];
      y += item.ring[k + 1];
    }
    const count = item.ring.length / 2;
    return [x / count, y / count];
  };
  items.forEach(item => (item.centroid = centroid(item)));
  const groups = new Map();
  for (const item of items) {
    const k = `${item.phenom}:${item.etn}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(item);
  }
  for (const group of groups.values()) {
    group.sort((a, b) => a.start - b.start);
    const chains = [];
    for (const item of group) {
      let best = null;
      let bestDistance = Infinity;
      for (const chain of chains) {
        const last = chain[chain.length - 1];
        if (item.start < last.end - 60000 || item.start - last.end > 120000) continue;
        const distance = Math.hypot((item.centroid[0] - last.centroid[0]) * Math.cos((item.centroid[1] * Math.PI) / 180), item.centroid[1] - last.centroid[1]) * 111.2;
        if (distance < 150 && distance < bestDistance) {
          best = chain;
          bestDistance = distance;
        }
      }
      if (best) best.push(item);
      else chains.push([item]);
    }
    chains.forEach((chain, index) => {
      for (const item of chain) {
        item.issue = chain[0].start;
        item.chain = `${chain[0].phenom}-${chain[0].etn}-${index}`;
      }
    });
  }
  items.sort((a, b) => a.start - b.start);
  const phenomena = ['TO', 'SV', 'FF'];
  for (const item of items) if (!phenomena.includes(item.phenom)) phenomena.push(item.phenom);
  const n = items.length;
  const ringOffsets = [0];
  const polygonRingOffsets = [0];
  const vertices = [];
  for (const item of items) {
    // Rings close implicitly: drop an explicit closing vertex.
    for (let r = 0; r + 1 < item.rings.length; r++) {
      let a = item.rings[r];
      let b = item.rings[r + 1];
      if (b - a > 1 && item.ring[a * 2] === item.ring[(b - 1) * 2] && item.ring[a * 2 + 1] === item.ring[(b - 1) * 2 + 1]) b--;
      for (let k = a; k < b; k++) vertices.push(item.ring[k * 2], item.ring[k * 2 + 1]);
      ringOffsets.push(vertices.length / 2);
    }
    polygonRingOffsets.push(ringOffsets.length - 1);
  }
  const vertexArray = Float32Array.from(vertices);
  const chainIds = new Map();
  writeDataset(
    'poopdeck-mrms-storm3d-warnings',
    {
      kind: 'polygons',
      count: n,
      bbox: bboxOf(vertexArray),
      columns: {
        vertices: {file: 'vertices.bin', dtype: 'float32', components: 2, length: vertexArray.length / 2},
        ringOffsets: {file: 'ringOffsets.bin', dtype: 'uint32', length: ringOffsets.length},
        polygonRingOffsets: {file: 'polygonRingOffsets.bin', dtype: 'uint32', length: polygonRingOffsets.length},
        startTime: {file: 'startTime.bin', dtype: 'uint32', length: n, unit: `${ORIGIN_TEXT} (this polygon version becomes valid)`},
        endTime: {file: 'endTime.bin', dtype: 'uint32', length: n, unit: `${ORIGIN_TEXT} (this polygon version is replaced or expires)`},
        issueTime: {file: 'issueTime.bin', dtype: 'uint32', length: n, unit: `${ORIGIN_TEXT} (first version of the same warning)`},
        warning: {file: 'warning.bin', dtype: 'uint32', length: n, description: 'dense id of the chain of versions of one warning'},
        partFeature: {file: 'partFeature.bin', dtype: 'uint32', length: n, description: 'feature of each polygon part (one part per feature)'},
        phenomenon: {file: 'phenomenon.bin', dtype: 'uint8', length: n, categories: phenomena}
      },
      properties: properties('mrms-storm3d-warnings', {warningCount: new Set(items.map(item => item.chain)).size, note: 'TO tornado, SV severe thunderstorm, FF flash flood.'})
    },
    {
      'vertices.bin': vertexArray,
      'ringOffsets.bin': Uint32Array.from(ringOffsets),
      'polygonRingOffsets.bin': Uint32Array.from(polygonRingOffsets),
      'startTime.bin': Uint32Array.from(items.map(item => seconds(item.start))),
      'endTime.bin': Uint32Array.from(items.map(item => seconds(item.end))),
      'issueTime.bin': Uint32Array.from(items.map(item => seconds(item.issue ?? item.start))),
      'warning.bin': Uint32Array.from(items.map(item => {
        if (!chainIds.has(item.chain)) chainIds.set(item.chain, chainIds.size);
        return chainIds.get(item.chain);
      })),
      'partFeature.bin': Uint32Array.from(items.map((_, index) => index)),
      'phenomenon.bin': Uint8Array.from(items.map(item => phenomena.indexOf(item.phenom)))
    }
  );
}

// ---- outages ---------------------------------------------------------------------------------
async function buildOutages() {
  const rows = [];
  await readFeatures('mrms-storm3d-outages', 3, ORIGIN_MS, END_MS, (f, i) => {
    const fips = Number(category(f, 'fips', i));
    const t = f.startTimes[i] + f.timeOffset;
    if (!Number.isFinite(fips) || t < ORIGIN_MS || t > END_MS) return;
    rows.push([fips, seconds(t), f.numericProps.customers_out[i]]);
  });
  const seen = new Set();
  const unique = rows.filter(row => {
    const k = `${row[0]}:${row[1]}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
  unique.sort((a, b) => a[1] - b[1] || a[0] - b[0]);
  const n = unique.length;
  log(`outages: ${rows.length} rows, ${n} unique county snapshots, ${new Set(unique.map(row => row[0])).size} counties`);
  writeDataset(
    'poopdeck-mrms-storm3d-outages',
    {
      kind: 'table',
      count: n,
      bbox: [-124.7258, 24.4981, -66.9499, 49.3844],
      columns: {
        fips: {file: 'fips.bin', dtype: 'uint32', length: n, description: 'five-digit county FIPS as an integer; join to us-counties'},
        timestamp: {file: 'time.bin', dtype: 'uint32', length: n, unit: ORIGIN_TEXT},
        customersOut: {file: 'customersOut.bin', dtype: 'uint32', length: n, description: 'customers without power at the snapshot'}
      },
      properties: properties('mrms-storm3d-outages', {note: 'One row per county and 15-minute snapshot with at least one customer out. Geometry comes from us-counties.'})
    },
    {
      'fips.bin': Uint32Array.from(unique.map(row => row[0])),
      'time.bin': Uint32Array.from(unique.map(row => row[1])),
      'customersOut.bin': Uint32Array.from(unique.map(row => Math.round(row[2])))
    }
  );
}

if (ONLY.has('tracks')) await buildTracks();
if (ONLY.has('lightning')) await buildLightning();
if (ONLY.has('reports')) await buildReports();
if (ONLY.has('warnings')) await buildWarnings();
if (ONLY.has('outages')) await buildOutages();
