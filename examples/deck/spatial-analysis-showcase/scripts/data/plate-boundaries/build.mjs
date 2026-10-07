#!/usr/bin/env node
// build.mjs: plate-boundaries. The 5,824 boundary steps of Bird's PB2002 plate model, one two-vertex line per
// step, with the step class and the relative plate motion.
//
// Usage:
//   node build.mjs <PB2002_steps.json> <out dir>
//
// Input: GeoJSON/PB2002_steps.json from https://github.com/fraxen/tectonicplates (ODC-BY 1.0). Each step is a
// roughly 100 km great-circle arc, stored densified; this keeps its start and end points only (a straight
// segment in the planar frame of the scenes differs from the arc by metres).
//
// Output (little-endian, headerless): pathOffsets (u32, n + 1), vertices (f32 lon, lat), stepClass (u8 codes),
// convergence (f32 mm/a, positive when the plates approach, from the divergent component), rightLateral (f32
// mm/a), plus manifest.json.
import fs from 'node:fs';
import path from 'node:path';

const [input, out] = process.argv.slice(2);
if (!input || !out) {
  console.error('usage: node build.mjs <PB2002_steps.json> <out dir>');
  process.exit(2);
}
const CLASSES = [
  'SUB: subduction zone',
  'OCB: oceanic convergent boundary',
  'CCB: continental convergent boundary',
  'OSR: oceanic spreading ridge',
  'CRB: continental rift boundary',
  'OTF: oceanic transform fault',
  'CTF: continental transform fault'
];
const codeOf = new Map(CLASSES.map((label, code) => [label.slice(0, 3), code]));
const steps = JSON.parse(fs.readFileSync(input, 'utf8')).features;
const n = steps.length;
const pathOffsets = new Uint32Array(n + 1);
const vertices = new Float32Array(n * 4);
const stepClass = new Uint8Array(n);
const convergence = new Float32Array(n);
const rightLateral = new Float32Array(n);
let west = 180;
let south = 90;
let east = -180;
let north = -90;
for (let i = 0; i < n; i++) {
  const {properties, geometry} = steps[i];
  const line = geometry.coordinates;
  const first = line[0];
  const last = line[line.length - 1];
  pathOffsets[i + 1] = (i + 1) * 2;
  vertices.set([first[0], first[1], last[0], last[1]], i * 4);
  const code = codeOf.get(properties.STEPCLASS);
  if (code === undefined) throw new Error(`unknown step class ${properties.STEPCLASS}`);
  stepClass[i] = code;
  convergence[i] = -properties.VELOCITYDI;
  rightLateral[i] = properties.VELOCITYRI;
  west = Math.min(west, first[0], last[0]);
  east = Math.max(east, first[0], last[0]);
  south = Math.min(south, first[1], last[1]);
  north = Math.max(north, first[1], last[1]);
}
fs.mkdirSync(out, {recursive: true});
const write = (name, array) => fs.writeFileSync(path.join(out, name), Buffer.from(array.buffer));
write('pathOffsets.bin', pathOffsets);
write('vertices.bin', vertices);
write('stepClass.bin', stepClass);
write('convergence.bin', convergence);
write('rightLateral.bin', rightLateral);
const manifest = {
  id: 'plate-boundaries',
  version: 1,
  kind: 'lines',
  count: n,
  bbox: [west, south, east, north],
  crs: 'EPSG:4326',
  columns: {
    pathOffsets: {file: 'pathOffsets.bin', dtype: 'uint32', components: 1, length: n + 1},
    vertices: {file: 'vertices.bin', dtype: 'float32', components: 2, length: n * 2},
    stepClass: {file: 'stepClass.bin', dtype: 'uint8', length: n, categories: CLASSES},
    convergence: {
      file: 'convergence.bin',
      dtype: 'float32',
      length: n,
      unit: 'mm/a',
      note: 'rate at which the two plates approach (negative = they separate)'
    },
    rightLateral: {file: 'rightLateral.bin', dtype: 'float32', length: n, unit: 'mm/a'}
  },
  properties: {
    source: 'Bird (2003), PB2002, via github.com/fraxen/tectonicplates (ODC-BY 1.0)',
    note: 'One line per ~100 km boundary step. Boundaries that crossed the antimeridian were split in the source.'
  }
};
fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 1));
console.error(`wrote ${n} steps to ${out}`);
