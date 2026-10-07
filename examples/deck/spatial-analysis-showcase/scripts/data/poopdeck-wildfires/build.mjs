#!/usr/bin/env node
// poopdeck-wildfires: regroup the poopdeck.gl `wildfires` archive (one STT feature per polygon part)
// into one feature per fire, with shells counter-clockwise and holes clockwise.
//
// Usage: node build.mjs [--raw <dir>] [--out <dir>]
//   --raw  directory for the raw stt-export output (default: $RAW or <tmp>/poopdeck-wildfires-raw).
//          Filled by ../poopdeck/stt-export.mjs when it has no manifest.json yet.
//   --out  default ../../../public/data/poopdeck-wildfires
//
// Output columns (little-endian, headerless):
//   vertices (float32 x2, lon/lat), ringOffsets, polygonRingOffsets (part -> rings),
//   featurePolygonOffsets (fire -> parts), acres, year, perimeterTime (uint32 s since 2020-01-01Z),
//   name / sizeClass / agency (uint8 categories), objectId, partCount.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : fallback;
};
const raw = arg('raw', process.env.RAW || path.join(os.tmpdir(), 'poopdeck-wildfires-raw'));
const out = arg('out', path.join(here, '../../../public/data/poopdeck-wildfires'));
const TIME_ORIGIN_MS = Date.UTC(2020, 0, 1);

if (!fs.existsSync(path.join(raw, 'manifest.json'))) {
  fs.mkdirSync(raw, {recursive: true});
  execFileSync(
    'node',
    [
      path.join(here, '../poopdeck/stt-export.mjs'),
      'https://tiles.poopdeck.gl/data/wildfires/manifest.json',
      '--id', 'poopdeck-wildfires',
      '--time', '2020-01-01,2024-01-01',
      '--max-features', '1000',
      '--out', raw
    ],
    {stdio: 'inherit'}
  );
}

const source = JSON.parse(fs.readFileSync(path.join(raw, 'manifest.json'), 'utf8'));
const read = (file, Type) => {
  const bytes = fs.readFileSync(path.join(raw, file));
  return new Type(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};
const vertices = read('vertices.bin', Float32Array);
const ringOffsets = read('ringOffsets.bin', Uint32Array);
const partRings = read('polygonRingOffsets.bin', Uint32Array);
const acres = read('acres.bin', Float32Array);
const years = read('year.bin', Float32Array);
const startTime = read('startTime.bin', Uint32Array);
const objectIds = read('object_id.bin', Float32Array);
const nameCodes = read('name.bin', Uint8Array);
const severityCodes = read('severity.bin', Uint8Array);
const agencyCodes = read('agency.bin', Uint8Array);
const partCount = source.count;
const sourceOriginMs = source.properties.timeOriginMs;

// Group parts by NIFC object id, in order of first appearance (the archive is sorted by date).
const groups = new Map();
for (let part = 0; part < partCount; part++) {
  const key = objectIds[part];
  if (!groups.has(key)) groups.set(key, []);
  groups.get(key).push(part);
}

const SIZE_CLASSES = ['moderate', 'high', 'extreme', 'catastrophic'];
const sizeClassOf = code => SIZE_CLASSES.indexOf(source.columns.severity.categories[code]);
const outVertices = [];
const outRingOffsets = [0];
const outPartRings = [0];
const outFeatureParts = [0];
const columns = {acres: [], year: [], perimeterTime: [], name: [], sizeClass: [], agency: [], objectId: [], partCount: []};
const ringSignedArea = (start, end) => {
  let sum = 0;
  for (let v = start; v < end; v++) {
    const w = v + 1 < end ? v + 1 : start;
    sum += vertices[2 * v] * vertices[2 * w + 1] - vertices[2 * w] * vertices[2 * v + 1];
  }
  return sum / 2;
};

let droppedEast = 0;
let reversed = 0;
for (const parts of groups.values()) {
  const first = parts[0];
  // Western US only: the archive also holds a few prairie and Gulf fires.
  let sumLongitude = 0;
  let count = 0;
  for (let v = ringOffsets[partRings[first]]; v < ringOffsets[partRings[first + 1]]; v++) {
    sumLongitude += vertices[2 * v];
    count++;
  }
  if (sumLongitude / count > -100) {
    droppedEast++;
    continue;
  }
  for (const part of parts) {
    for (let ring = partRings[part]; ring < partRings[part + 1]; ring++) {
      let start = ringOffsets[ring];
      let end = ringOffsets[ring + 1];
      if (end - start > 1 && vertices[2 * start] === vertices[2 * (end - 1)] && vertices[2 * start + 1] === vertices[2 * (end - 1) + 1]) {
        end--;
      }
      if (end - start < 3) {
        continue;
      }
      const isShell = ring === partRings[part];
      const area = ringSignedArea(start, end);
      const wantPositive = isShell;
      const flip = (area > 0) !== wantPositive;
      if (flip) reversed++;
      for (let k = 0; k < end - start; k++) {
        const v = flip ? end - 1 - k : start + k;
        outVertices.push(vertices[2 * v], vertices[2 * v + 1]);
      }
      outRingOffsets.push(outVertices.length / 2);
    }
    outPartRings.push(outRingOffsets.length - 1);
  }
  outFeatureParts.push(outPartRings.length - 1);
  columns.acres.push(acres[first]);
  columns.year.push(years[first]);
  columns.perimeterTime.push(startTime[first] + Math.round((sourceOriginMs - TIME_ORIGIN_MS) / 1000));
  columns.name.push(nameCodes[first]);
  columns.sizeClass.push(sizeClassOf(severityCodes[first]));
  columns.agency.push(agencyCodes[first]);
  columns.objectId.push(objectIds[first]);
  columns.partCount.push(parts.length);
}

const fireCount = outFeatureParts.length - 1;
let west = Infinity;
let south = Infinity;
let east = -Infinity;
let north = -Infinity;
for (let v = 0; v < outVertices.length / 2; v++) {
  west = Math.min(west, outVertices[2 * v]);
  east = Math.max(east, outVertices[2 * v]);
  south = Math.min(south, outVertices[2 * v + 1]);
  north = Math.max(north, outVertices[2 * v + 1]);
}

fs.mkdirSync(out, {recursive: true});
const write = (file, array) => fs.writeFileSync(path.join(out, file), Buffer.from(array.buffer, array.byteOffset, array.byteLength));
const files = {
  vertices: ['vertices.bin', Float32Array.from(outVertices), 'float32', 2],
  ringOffsets: ['ringOffsets.bin', Uint32Array.from(outRingOffsets), 'uint32'],
  polygonRingOffsets: ['polygonRingOffsets.bin', Uint32Array.from(outPartRings), 'uint32'],
  featurePolygonOffsets: ['featurePolygonOffsets.bin', Uint32Array.from(outFeatureParts), 'uint32'],
  acres: ['acres.bin', Float32Array.from(columns.acres), 'float32'],
  year: ['year.bin', Uint16Array.from(columns.year), 'uint16'],
  perimeterTime: ['perimeterTime.bin', Uint32Array.from(columns.perimeterTime), 'uint32'],
  name: ['name.bin', Uint8Array.from(columns.name), 'uint8'],
  sizeClass: ['sizeClass.bin', Uint8Array.from(columns.sizeClass), 'uint8'],
  agency: ['agency.bin', Uint8Array.from(columns.agency), 'uint8'],
  objectId: ['objectId.bin', Uint32Array.from(columns.objectId), 'uint32'],
  partCount: ['partCount.bin', Uint16Array.from(columns.partCount), 'uint16']
};
const manifestColumns = {};
for (const [key, [file, array, dtype, components]] of Object.entries(files)) {
  write(file, array);
  manifestColumns[key] = {file, dtype, length: components ? array.length / components : array.length};
  if (components) manifestColumns[key].components = components;
}
manifestColumns.perimeterTime.unit = 'seconds since 2020-01-01T00:00:00Z (uint32; use days in float32 on the GPU)';
manifestColumns.name.categories = source.columns.name.categories;
manifestColumns.sizeClass.categories = SIZE_CLASSES;
manifestColumns.agency.categories = source.columns.agency.categories;
manifestColumns.acres.unit = 'acres (NIFC GIS acres, repeated on every part of a fire)';

const manifest = {
  id: 'poopdeck-wildfires',
  version: 1,
  kind: 'polygons',
  count: fireCount,
  bbox: [west, south, east, north],
  crs: 'EPSG:4326',
  columns: manifestColumns,
  properties: {
    timeOrigin: '2020-01-01T00:00:00Z',
    polygonPartCount: outPartRings.length - 1,
    ringCount: outRingOffsets.length - 1,
    vertexCount: outVertices.length / 2,
    ringOrientation: 'shells counter-clockwise, holes clockwise (lon/lat), so GPUGeometryMeasures holeRule "winding" gives area with holes subtracted',
    sizeClassNote: 'poopdeck calls this attribute severity; it is an acreage class (moderate < 10k, high 11k-33k, extreme 50k-97k, catastrophic > 300k acres), not burn severity',
    perimeterTimeNote: 'the perimeter date of the NIFC record (usually the last mapping), not the ignition date',
    source: source.properties.source,
    exportedFrom: 'poopdeck.gl wildfires archive via stt-export.mjs, regrouped by object id; fires with mean longitude east of -100 dropped',
    droppedEasternFires: droppedEast,
    reversedRings: reversed
  }
};
fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 1));
const bytes = Object.values(files).reduce((sum, [, array]) => sum + array.byteLength, 0);
console.log(`fires ${fireCount} (dropped ${droppedEast} eastern), parts ${manifest.properties.polygonPartCount}, rings ${manifest.properties.ringCount}, vertices ${manifest.properties.vertexCount}, ${bytes} bytes, reversed ${reversed} rings`);
console.log('bbox', manifest.bbox);
