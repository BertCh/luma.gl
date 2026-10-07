#!/usr/bin/env node
// Builds public/data/poopdeck-ecco-currents from the poopdeck.gl `ecco-currents` archive
// (NASA/JPL ECCO V4r4 modelled surface particles, 2016-12 to 2017-12, per-vertex speed).
//
//   node build.mjs --raw <empty-download-dir> --out <app>/public/data/poopdeck-ecco-currents
//
// Two outputs from one pass over the whole archive:
//   1. A capped sample of the particle pieces (`--sample`, default 30000) for drawing.
//   2. `currents.bin`, a gridded annual-mean surface current field derived from ALL pieces. Every
//      archive piece is a 2-vertex displacement over about 3.5 days; its velocity is binned into
//      0.5 degree cells with a Gaussian weight (sigma 0.5 degree), so the field is an
//      annual MEAN with ECCO's coarse (about 1 degree) effective resolution. Cells with too little
//      weight stay NaN (land, ice, shelf).
//
// currents.bin is float32, depth 3, laid out [plane][row][col] with row 0 at the NORTH edge:
//   plane 0: u, degrees of longitude per day   plane 1: v, degrees of latitude per day
//   plane 2: kernel weight (about the number of 3.5-day samples that contributed)
// Degrees per day are the native unit of GPUParticleAdvection on a lon/lat field.

import {execFileSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const exporter = path.join(here, '../poopdeck/stt-export.mjs');
const archive = 'https://tiles.poopdeck.gl/data/ecco-currents/manifest.json';
const args = Object.fromEntries(
  process.argv
    .slice(2)
    .reduce((pairs, token, index, all) => {
      if (token.startsWith('--')) pairs.push([token.slice(2), all[index + 1]]);
      return pairs;
    }, [])
);
if (!args.raw || !args.out) {
  console.error('usage: node build.mjs --raw <download dir> --out <output dir> [--sample 30000]');
  process.exit(2);
}
const SAMPLE = Number(args.sample ?? 30000);
const CELL = 0.5;
const WEST = -180;
const NORTH = 80;
const SOUTH = -80;
const WIDTH = 360 / CELL;
const HEIGHT = (NORTH - SOUTH) / CELL;
const SIGMA = 0.5;
const REACH = 2;
const MIN_WEIGHT = 2.5;
const MAX_SPEED = 3.5; // m/s, drops obvious jumps
const METERS_PER_DEGREE = 111194.9;
const DAY = 86400;

fs.mkdirSync(args.raw, {recursive: true});
fs.mkdirSync(args.out, {recursive: true});
const full = path.join(args.raw, 'full');
if (!fs.existsSync(path.join(full, 'manifest.json'))) {
  execFileSync(
    'node',
    [exporter, archive, '--id', 'ecco-full', '--max-features', '1500000', '--attrs', 'speed', '--out', full],
    {stdio: 'inherit'}
  );
}
execFileSync(
  'node',
  [exporter, archive, '--id', 'poopdeck-ecco-currents', '--max-features', String(SAMPLE), '--attrs', 'speed,basin', '--seed', '7', '--out', args.out],
  {stdio: 'inherit'}
);

const read = (name, Type) => {
  const bytes = fs.readFileSync(path.join(full, name));
  return new Type(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};
const offsets = read('pathOffsets.bin', Uint32Array);
const vertices = read('vertices.bin', Float32Array);
const times = read('time.bin', Uint32Array);

const sumU = new Float64Array(WIDTH * HEIGHT);
const sumV = new Float64Array(WIDTH * HEIGHT);
const sumW = new Float64Array(WIDTH * HEIGHT);
let segments = 0;
let rejected = 0;
for (let piece = 0; piece < offsets.length - 1; piece++) {
  for (let k = offsets[piece]; k < offsets[piece + 1] - 1; k++) {
    const dt = times[k + 1] - times[k];
    if (dt <= 0) continue;
    let dlon = vertices[2 * k + 2] - vertices[2 * k];
    dlon = ((dlon + 540) % 360) - 180;
    const dlat = vertices[2 * k + 3] - vertices[2 * k + 1];
    const lat = (vertices[2 * k + 1] + vertices[2 * k + 3]) / 2;
    const speed =
      Math.hypot(dlon * Math.cos((lat * Math.PI) / 180), dlat) * (METERS_PER_DEGREE / dt);
    if (speed > MAX_SPEED) {
      rejected++;
      continue;
    }
    let lon = vertices[2 * k] + dlon / 2;
    lon = ((lon + 540) % 360) - 180;
    const u = (dlon / dt) * DAY;
    const v = (dlat / dt) * DAY;
    const column = Math.floor((lon - WEST) / CELL);
    const row = Math.floor((NORTH - lat) / CELL);
    if (row < 0 || row >= HEIGHT) continue;
    segments++;
    for (let dr = -REACH; dr <= REACH; dr++) {
      for (let dc = -REACH; dc <= REACH; dc++) {
        const r = row + dr;
        if (r < 0 || r >= HEIGHT) continue;
        const c = (column + dc + WIDTH) % WIDTH;
        // Distance from the sample to the cell center, in degrees.
        const cellLon = WEST + (c + 0.5) * CELL;
        const cellLat = NORTH - (r + 0.5) * CELL;
        let offsetLon = cellLon - lon;
        offsetLon = ((offsetLon + 540) % 360) - 180;
        const distance2 =
          (offsetLon * Math.cos((cellLat * Math.PI) / 180)) ** 2 + (cellLat - lat) ** 2;
        const weight = Math.exp(-distance2 / (2 * SIGMA * SIGMA));
        const index = r * WIDTH + c;
        sumU[index] += weight * u;
        sumV[index] += weight * v;
        sumW[index] += weight;
      }
    }
  }
}

const field = new Float32Array(WIDTH * HEIGHT * 3);
let valid = 0;
let peak = 0;
let peakAt = [0, 0];
for (let index = 0; index < WIDTH * HEIGHT; index++) {
  const weight = sumW[index];
  field[2 * WIDTH * HEIGHT + index] = weight;
  if (weight >= MIN_WEIGHT) {
    const u = sumU[index] / weight;
    const v = sumV[index] / weight;
    field[index] = u;
    field[WIDTH * HEIGHT + index] = v;
    valid++;
    const row = Math.floor(index / WIDTH);
    const lat = NORTH - (row + 0.5) * CELL;
    const speed = Math.hypot(u * Math.cos((lat * Math.PI) / 180), v) * METERS_PER_DEGREE / DAY;
    if (speed > peak) {
      peak = speed;
      peakAt = [WEST + ((index % WIDTH) + 0.5) * CELL, lat];
    }
  } else {
    field[index] = Number.NaN;
    field[WIDTH * HEIGHT + index] = Number.NaN;
  }
}
fs.writeFileSync(path.join(args.out, 'currents.bin'), new Uint8Array(field.buffer));

const manifestPath = path.join(args.out, 'manifest.json');
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
manifest.id = 'poopdeck-ecco-currents';
manifest.raster = {
  file: 'currents.bin',
  encoding: 'float32-bin',
  width: WIDTH,
  height: HEIGHT,
  depth: 3,
  bounds: [WEST, SOUTH, WEST + 360, NORTH],
  noData: null,
  unit: 'degrees per day (u: longitude, v: latitude), plane 2 = kernel weight',
  rowOrigin: 'north'
};
manifest.properties = {
  ...manifest.properties,
  field: {
    cellDegrees: CELL,
    kernelSigmaDegrees: SIGMA,
    minimumWeight: MIN_WEIGHT,
    segmentsBinned: segments,
    segmentsRejected: rejected,
    validCells: valid,
    cells: WIDTH * HEIGHT,
    peakSpeedMetersPerSecond: peak,
    peakAt,
    planes: ['u (deg lon/day)', 'v (deg lat/day)', 'weight']
  }
};
fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(
  `segments ${segments} (rejected ${rejected}), valid cells ${valid}/${WIDTH * HEIGHT}, peak ${peak.toFixed(2)} m/s at ${peakAt.map(v => v.toFixed(1))}`
);
