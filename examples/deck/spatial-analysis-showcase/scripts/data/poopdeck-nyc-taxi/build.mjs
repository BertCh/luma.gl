#!/usr/bin/env node
// Builds public/data/poopdeck-nyc-taxi from the poopdeck.gl `nyc-taxi-paths` archive.
//
// The archive stores 500k OSRM-routed yellow-taxi trips (NYC TLC, 1-2 Jan 2015) as paths split at
// 1-minute buckets. A trip's first vertex is its pickup, its last vertex its dropoff. This script
// streams every z10 tile, keeps only the earliest and latest vertex per trip_id, and packs the
// trips as quantized columns (see README.md).
//
// usage: node build.mjs --out <public/data/poopdeck-nyc-taxi> [--max-trips 440000] [--raw <cache.json>]
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), [])
);
const OUT = args.out;
const MAX_TRIPS = Number(args['max-trips'] || 440000);
const CACHE = args.raw;
const CORE = args.core || path.join(os.homedir(), 'Documents/GitHub/poopdeck.gl/packages/core/dist/index.js');
const MANIFEST = 'https://tiles.poopdeck.gl/data/nyc-taxi-paths/manifest.json';
const BBOX = {minLon: -74.05, minLat: 40.6, maxLon: -73.7, maxLat: 40.9};
if (!OUT) throw new Error('--out required');

// ---- 1. collect first/last vertex per trip -------------------------------------------------
let trips;
if (CACHE && fs.existsSync(CACHE)) {
  trips = new Map(JSON.parse(fs.readFileSync(CACHE, 'utf8')));
} else {
  const {STTArchive, InlineTileDecoder} = await import(pathToFileURL(CORE).href);
  const archive = new STTArchive({url: MANIFEST, decoder: new InlineTileDecoder(), opfsCache: false});
  const meta = await archive.getMetadata();
  const ids = await archive.getTileIdsInBounds(
    {minLon: -74.3, minLat: 40.45, maxLon: -73.65, maxLat: 41.0},
    meta.minZoom,
    {start: meta.timeRange.start, end: meta.timeRange.end}
  );
  console.error(`z${meta.minZoom}: ${ids.length} tiles`);
  trips = new Map();
  const BATCH = 24;
  for (let b = 0; b < ids.length; b += BATCH) {
    const tiles = await archive.getTiles(ids.slice(b, b + BATCH));
    for (const tile of tiles) {
      if (!tile) continue;
      for (const layer of tile.layers) {
        const f = layer.features;
        if (layer.name === 'summary' || !f.featureCount || !f.vertexTimestamps) continue;
        const dims = f.positionDimensions || 2;
        for (let i = 0; i < f.featureCount; i++) {
          const id = Math.round(f.numericProps.trip_id[i]);
          let rec = trips.get(id);
          if (!rec) {
            rec = {
              t0: Infinity, x0: 0, y0: 0, t1: -Infinity, x1: 0, y1: 0,
              d: f.numericProps.trip_distance[i], f: f.numericProps.fare_amount[i], p: f.numericProps.passenger_count[i]
            };
            trips.set(id, rec);
          }
          const from = f.startIndices[i];
          const to = f.startIndices[i + 1];
          const tFirst = f.vertexTimestamps[from] + f.timeOffset;
          const tLast = f.vertexTimestamps[to - 1] + f.timeOffset;
          if (tFirst < rec.t0) { rec.t0 = tFirst; rec.x0 = f.positions[from * dims]; rec.y0 = f.positions[from * dims + 1]; }
          if (tLast > rec.t1) { rec.t1 = tLast; rec.x1 = f.positions[(to - 1) * dims]; rec.y1 = f.positions[(to - 1) * dims + 1]; }
        }
      }
    }
    if ((b / BATCH) % 20 === 0) console.error(`tiles ${Math.min(b + BATCH, ids.length)}/${ids.length}, trips ${trips.size}`);
  }
  if (CACHE) fs.writeFileSync(CACHE, JSON.stringify([...trips]));
}
console.error(`trips collected: ${trips.size}`);

// ---- 2. filter and sample -------------------------------------------------------------------
let rows = [...trips.entries()].map(([id, r]) => ({id, ...r}));
rows = rows.filter(
  (r) =>
    r.x0 > BBOX.minLon && r.x0 < BBOX.maxLon && r.y0 > BBOX.minLat && r.y0 < BBOX.maxLat &&
    r.x1 > BBOX.minLon && r.x1 < BBOX.maxLon && r.y1 > BBOX.minLat && r.y1 < BBOX.maxLat &&
    r.t1 > r.t0 && r.t1 - r.t0 < 3.5 * 3600e3 && r.d > 0.05 && r.d < 40 && r.f > 0 && r.f < 250
);
console.error(`after filter: ${rows.length}`);
// seeded shuffle, then cap
let seed = 7;
const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
if (rows.length > MAX_TRIPS) {
  for (let i = rows.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }
  rows.length = MAX_TRIPS;
}
rows.sort((a, b) => a.t0 - b.t0);
const n = rows.length;

// ---- 3. pack ---------------------------------------------------------------------------------
const originMs = Math.floor(rows[0].t0 / 3600e3) * 3600e3; // whole hour
const GRID = [BBOX.minLon, BBOX.minLat, BBOX.maxLon, BBOX.maxLat];
const quant = (v, lo, hi) => Math.max(0, Math.min(65535, Math.round(((v - lo) / (hi - lo)) * 65535)));
const originXY = new Uint16Array(n * 2), destXY = new Uint16Array(n * 2);
const pickupTime = new Uint16Array(n); // 4 s units since originMs
const duration = new Uint8Array(n); // 15 s units, capped 255 (63.75 min)
const distance = new Uint8Array(n); // 0.1 mile, capped 255
const fare = new Uint8Array(n); // 0.5 USD, capped 255
const passengers = new Uint8Array(n);
rows.forEach((r, i) => {
  originXY[2 * i] = quant(r.x0, GRID[0], GRID[2]); originXY[2 * i + 1] = quant(r.y0, GRID[1], GRID[3]);
  destXY[2 * i] = quant(r.x1, GRID[0], GRID[2]); destXY[2 * i + 1] = quant(r.y1, GRID[1], GRID[3]);
  pickupTime[i] = Math.min(65535, Math.round((r.t0 - originMs) / 4000));
  duration[i] = Math.min(255, Math.round((r.t1 - r.t0) / 15000));
  distance[i] = Math.min(255, Math.round(r.d * 10));
  fare[i] = Math.min(255, Math.round(r.f * 2));
  passengers[i] = Math.max(1, Math.min(6, Math.round(r.p)));
});
fs.mkdirSync(OUT, {recursive: true});
const columns = {};
const write = (name, file, typed, extra) => {
  fs.writeFileSync(path.join(OUT, file), new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength));
  const dtype = typed instanceof Uint16Array ? 'uint16' : 'uint8';
  columns[name] = {file, dtype, length: extra.components ? typed.length / extra.components : typed.length, ...extra};
};
write('origin', 'origin.bin', originXY, {components: 2, unit: 'uint16 pairs spanning properties.quantBbox (lon, lat)'});
write('destination', 'destination.bin', destXY, {components: 2, unit: 'uint16 pairs spanning properties.quantBbox (lon, lat)'});
write('pickupTime', 'pickupTime.bin', pickupTime, {unit: `4-second steps since properties.timeOriginMs`});
write('duration', 'duration.bin', duration, {unit: '15-second steps (derived from the OSRM routed path timestamps)'});
write('distance', 'distance.bin', distance, {unit: '0.1 mile steps (TLC trip_distance)'});
write('fare', 'fare.bin', fare, {unit: '0.5 USD steps (TLC fare_amount, no tips)'});
write('passengers', 'passengers.bin', passengers, {unit: 'passengers (1-6)'});
const manifest = {
  id: 'poopdeck-nyc-taxi',
  version: 1,
  kind: 'flows',
  count: n,
  bbox: GRID,
  crs: 'EPSG:4326',
  columns,
  properties: {
    timeOriginMs: originMs,
    quantBbox: GRID,
    description: 'Yellow-taxi trips as origin/destination pairs. Positions are uint16 across quantBbox; decode lon = w + q / 65535 * (e - w).',
    source: {manifestUrl: MANIFEST, derivedFrom: 'first and last vertex of each OSRM-routed path, grouped by trip_id', exportedFrom: 'build.mjs', tripsInArchive: trips.size, tripsAfterFilter: rows.length}
  }
};
fs.writeFileSync(path.join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 2));
const bytes = Object.values(columns).reduce((s, c) => s + fs.statSync(path.join(OUT, c.file)).size, 0);
console.log(JSON.stringify({trips: n, bytes, originIso: new Date(originMs).toISOString()}));
