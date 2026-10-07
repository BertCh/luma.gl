#!/usr/bin/env node
// Builds public/data/poopdeck-drifters from the poopdeck.gl `drifters` archive (NOAA Global Drifter
// Program, 6-hourly interpolated fixes with per-vertex SST).
//
//   node build.mjs --raw <empty-download-dir> --out <app>/public/data/poopdeck-drifters
//
// Steps:
//   1. stt-export.mjs reads 2017 (stitched, clipped to the window; whole tracks, not 7-day pieces).
//   2. Every track is clipped to its first 60 days (release = first fix inside the window), thinned to
//      about one fix per 12 hours, and cut at the antimeridian (planar line drawing cannot cross it).
//   3. `daily` holds the real position at lead day 0..30 after release (interpolated from the 6-hourly
//      fixes, NaN where the record has a gap longer than 36 h or has ended): the reference the
//      virtual particles are compared with.
//
// Columns written (little-endian, headerless):
//   pathOffsets (uint32, pieces + 1), vertices (float32 lon,lat), timestamp (uint32 s since timeOrigin),
//   sst (uint8, deg C = value * 0.2 - 5, 255 = none), trackId (uint16, piece -> track),
//   releaseTime (uint32 per track), deployed (uint8 per track), daily (float32 lon,lat x tracks x 31).

import {execFileSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(
  process.argv
    .slice(2)
    .reduce((pairs, token, index, all) => {
      if (token.startsWith('--')) pairs.push([token.slice(2), all[index + 1]]);
      return pairs;
    }, [])
);
if (!args.raw || !args.out) {
  console.error('usage: node build.mjs --raw <download dir> --out <output dir>');
  process.exit(2);
}
const WINDOW = ['2017-01-01', '2017-12-31'];
const CLIP_DAYS = 60;
const LEAD_DAYS = 30;
const DAY = 86400;
const KEEP_SECONDS = 40000; // thin 6-hourly fixes to about 12-hourly
const SST_OFFSET = -5;
const SST_SCALE = 0.2;

fs.mkdirSync(args.raw, {recursive: true});
fs.mkdirSync(args.out, {recursive: true});
if (!fs.existsSync(path.join(args.raw, 'manifest.json'))) {
  execFileSync(
    'node',
    [
      path.join(here, '../poopdeck/stt-export.mjs'),
      'https://tiles.poopdeck.gl/data/drifters/manifest.json',
      '--id',
      'poopdeck-drifters-raw',
      '--time',
      WINDOW.join(','),
      '--stitch',
      'drifter_id,segment',
      '--clip-time',
      '--max-features',
      '6000',
      '--attrs',
      'none',
      '--out',
      args.raw
    ],
    {stdio: 'inherit'}
  );
}

const read = (name, Type) => {
  const bytes = fs.readFileSync(path.join(args.raw, name));
  return new Type(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};
const rawManifest = JSON.parse(fs.readFileSync(path.join(args.raw, 'manifest.json'), 'utf8'));
const offsets = read('pathOffsets.bin', Uint32Array);
const vertices = read('vertices.bin', Float32Array);
const times = read('time.bin', Uint32Array);
const sstValues = read('vertexValue.bin', Float32Array);
const timeOriginMs = rawManifest.properties.timeOriginMs;
const rawTracks = offsets.length - 1;

const wrapLongitude = longitude => ((((longitude + 180) % 360) + 360) % 360) - 180;

const pieceOffsets = [0];
const outVertices = [];
const outTimes = [];
const outSst = [];
const pieceTrack = [];
const releaseTimes = [];
const deployed = [];
const dailyRows = [];
let validDaily = 0;

for (let track = 0; track < rawTracks; track++) {
  const first = offsets[track];
  const last = offsets[track + 1];
  if (last - first < 4) continue;
  const trackIndex = releaseTimes.length;
  const release = times[first];
  releaseTimes.push(release);
  deployed.push(release > 5 * DAY ? 1 : 0);

  // Unwrapped longitudes so interpolation across the antimeridian is continuous.
  const unwrapped = new Float64Array(last - first);
  for (let k = first; k < last; k++) {
    const longitude = vertices[2 * k];
    unwrapped[k - first] =
      k === first
        ? longitude
        : unwrapped[k - first - 1] +
          (((longitude - vertices[2 * (k - 1)] + 540) % 360) - 180);
  }

  // Real position at each lead day.
  let cursor = first;
  for (let day = 0; day <= LEAD_DAYS; day++) {
    const target = release + day * DAY;
    while (cursor < last - 1 && times[cursor + 1] < target) cursor++;
    let longitude = Number.NaN;
    let latitude = Number.NaN;
    if (target <= times[last - 1]) {
      let a = cursor;
      while (a < last - 1 && times[a + 1] <= target) a++;
      const b = Math.min(a + 1, last - 1);
      const span = times[b] - times[a];
      if (span <= 36 * 3600) {
        const w = span > 0 ? (target - times[a]) / span : 0;
        longitude = wrapLongitude(unwrapped[a - first] + (unwrapped[b - first] - unwrapped[a - first]) * w);
        latitude = vertices[2 * a + 1] + (vertices[2 * b + 1] - vertices[2 * a + 1]) * w;
        validDaily++;
      }
    }
    dailyRows.push(longitude, latitude);
  }

  // Clipped, thinned, antimeridian-split path pieces.
  let lastKept = -1;
  let pieceStartVertex = outTimes.length;
  let previousLongitude = Number.NaN;
  const keep = [];
  for (let k = first; k < last; k++) {
    if (times[k] - release > CLIP_DAYS * DAY) break;
    if (lastKept < 0 || times[k] - times[lastKept] >= KEEP_SECONDS) {
      keep.push(k);
      lastKept = k;
    }
  }
  if (keep.length > 0) {
    // The final clipped fix always closes the path.
    const lastIndex = keep[keep.length - 1];
    let tailIndex = lastIndex;
    for (let k = lastIndex + 1; k < last && times[k] - release <= CLIP_DAYS * DAY; k++) tailIndex = k;
    if (tailIndex !== lastIndex) keep.push(tailIndex);
  }
  for (const k of keep) {
    const longitude = vertices[2 * k];
    if (!Number.isNaN(previousLongitude) && Math.abs(longitude - previousLongitude) > 180) {
      if (outTimes.length - pieceStartVertex >= 2) {
        pieceOffsets.push(outTimes.length);
        pieceTrack.push(trackIndex);
      } else {
        // Drop a one-vertex sliver and restart the piece.
        outVertices.length = pieceStartVertex * 2;
        outTimes.length = pieceStartVertex;
        outSst.length = pieceStartVertex;
      }
      pieceStartVertex = outTimes.length;
    }
    outVertices.push(longitude, vertices[2 * k + 1]);
    outTimes.push(times[k]);
    const sst = sstValues[k];
    outSst.push(Number.isFinite(sst) ? Math.max(0, Math.min(254, Math.round((sst - SST_OFFSET) / SST_SCALE))) : 255);
    previousLongitude = longitude;
  }
  if (outTimes.length - pieceStartVertex >= 2) {
    pieceOffsets.push(outTimes.length);
    pieceTrack.push(trackIndex);
  } else {
    outVertices.length = pieceStartVertex * 2;
    outTimes.length = pieceStartVertex;
    outSst.length = pieceStartVertex;
  }
}

const trackCount = releaseTimes.length;
const pieceCount = pieceTrack.length;
const files = {
  'pathOffsets.bin': Uint32Array.from(pieceOffsets),
  'vertices.bin': Float32Array.from(outVertices),
  'time.bin': Uint32Array.from(outTimes),
  'sst.bin': Uint8Array.from(outSst),
  'trackId.bin': Uint16Array.from(pieceTrack),
  'releaseTime.bin': Uint32Array.from(releaseTimes),
  'deployed.bin': Uint8Array.from(deployed),
  'daily.bin': Float32Array.from(dailyRows)
};
let total = 0;
for (const [name, array] of Object.entries(files)) {
  fs.writeFileSync(path.join(args.out, name), new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
  total += array.byteLength;
}

let west = 180;
let east = -180;
let south = 90;
let north = -90;
for (let i = 0; i < outVertices.length; i += 2) {
  west = Math.min(west, outVertices[i]);
  east = Math.max(east, outVertices[i]);
  south = Math.min(south, outVertices[i + 1]);
  north = Math.max(north, outVertices[i + 1]);
}
const timeUnit = `seconds since ${new Date(timeOriginMs).toISOString()}`;
const vertexCount = outTimes.length;
const manifest = {
  id: 'poopdeck-drifters',
  version: 1,
  kind: 'trajectories',
  count: pieceCount,
  bbox: [west, south, east, north],
  crs: 'EPSG:4326',
  columns: {
    pathOffsets: {file: 'pathOffsets.bin', dtype: 'uint32', length: pieceCount + 1},
    vertices: {file: 'vertices.bin', dtype: 'float32', components: 2, length: vertexCount},
    timestamp: {file: 'time.bin', dtype: 'uint32', unit: timeUnit, length: vertexCount},
    sst: {file: 'sst.bin', dtype: 'uint8', unit: 'deg C = value * 0.2 - 5; 255 = none', length: vertexCount},
    trackId: {file: 'trackId.bin', dtype: 'uint16', length: pieceCount},
    releaseTime: {file: 'releaseTime.bin', dtype: 'uint32', unit: timeUnit, length: trackCount},
    deployed: {file: 'deployed.bin', dtype: 'uint8', unit: '1 = first fix after 5 Jan (new deployment or new record)', length: trackCount},
    daily: {
      file: 'daily.bin',
      dtype: 'float32',
      components: 2,
      length: trackCount * (LEAD_DAYS + 1),
      unit: 'lon, lat at release + 0..30 days; NaN = no fix'
    }
  },
  properties: {
    timeOriginMs,
    leadDays: LEAD_DAYS,
    clipDays: CLIP_DAYS,
    window: WINDOW,
    trackCount,
    pieceCount,
    sstOffset: SST_OFFSET,
    sstScale: SST_SCALE,
    thinning: 'about 12-hourly fixes (the archive is 6-hourly)',
    validDailyPositions: validDaily,
    source: {archive: 'https://tiles.poopdeck.gl/data/drifters/manifest.json', exporter: 'stt-export.mjs --stitch drifter_id,segment --clip-time'}
  }
};
fs.writeFileSync(path.join(args.out, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(
  `tracks ${trackCount}, pieces ${pieceCount}, vertices ${vertexCount}, bytes ${total}, valid daily ${validDaily}/${trackCount * (LEAD_DAYS + 1)}, bbox ${[west, south, east, north].map(v => v.toFixed(1))}`
);
