// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync, writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

const root = new URL('../../', import.meta.url);
const source = new URL('examples/deck/spatial-analysis-showcase/public/data/', root);
const readColumn = (dataset, file, Type) => {
  const bytes = readFileSync(new URL(`${dataset}/${file}.bin`, source));
  return new Type(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};
const positions = readColumn('chicago-roads', 'nodes', Float32Array);
const sources = readColumn('chicago-roads', 'edgeSource', Uint32Array);
const targets = readColumn('chicago-roads', 'edgeTarget', Uint32Array);
const times = readColumn('chicago-roads', 'edgeTravelTime', Float32Array);
const lengths = readColumn('chicago-roads', 'edgeLength', Float32Array);
const classes = readColumn('chicago-roads', 'edgeClass', Uint8Array);
const bbox = [-87.655, 41.865, -87.615, 41.902];
const inside = (longitude, latitude) =>
  longitude >= bbox[0] && longitude <= bbox[2] && latitude >= bbox[1] && latitude <= bbox[3];
const selected = new Set(
  Array.from({length: positions.length / 2}, (_, index) => index).filter(index =>
    inside(positions[2 * index], positions[2 * index + 1])
  )
);
const forward = new Map([...selected].map(index => [index, []]));
const reverse = new Map([...selected].map(index => [index, []]));
for (let edge = 0; edge < sources.length; edge++) {
  if (selected.has(sources[edge]) && selected.has(targets[edge])) {
    forward.get(sources[edge]).push(targets[edge]);
    reverse.get(targets[edge]).push(sources[edge]);
  }
}
// Iterative Kosaraju: sorted source ids and deterministic adjacency order break every tie.
const visited = new Set();
const finish = [];
for (const node of selected) {
  if (visited.has(node)) continue;
  const stack = [[node, false]];
  while (stack.length) {
    const [current, leaving] = stack.pop();
    if (leaving) {
      finish.push(current);
      continue;
    }
    if (visited.has(current)) continue;
    visited.add(current);
    stack.push([current, true]);
    for (const neighbor of forward.get(current))
      if (!visited.has(neighbor)) stack.push([neighbor, false]);
  }
}
visited.clear();
const components = [];
for (const node of finish.reverse()) {
  if (visited.has(node)) continue;
  const component = [];
  const stack = [node];
  visited.add(node);
  while (stack.length) {
    const current = stack.pop();
    component.push(current);
    for (const neighbor of reverse.get(current))
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        stack.push(neighbor);
      }
  }
  components.push(component.sort((left, right) => left - right));
}
components.sort((left, right) => right.length - left.length || left[0] - right[0]);
const originalIds = components[0];
const nodeIds = new Map(originalIds.map((original, index) => [original, index]));
const round = (value, digits) => Number(value.toFixed(digits));
const nodes = originalIds.flatMap(index => [
  round(positions[2 * index], 5),
  round(positions[2 * index + 1], 5)
]);
const edges = Array.from({length: sources.length}, (_, index) => index).filter(
  index => nodeIds.has(sources[index]) && nodeIds.has(targets[index])
);
edges.sort(
  (left, right) => sources[left] - sources[right] || targets[left] - targets[right] || left - right
);
const offsets = new Array(originalIds.length + 1).fill(0);
for (const edge of edges) offsets[nodeIds.get(sources[edge]) + 1]++;
for (let node = 1; node < offsets.length; node++) offsets[node] += offsets[node - 1];
const longitudeScale = Math.cos(((bbox[1] + bbox[3]) * Math.PI) / 360);
const nearestNode = (longitude, latitude) => {
  let nearest = 0;
  let minimum = Infinity;
  for (let node = 0; node < originalIds.length; node++) {
    const distance =
      ((nodes[2 * node] - longitude) * longitudeScale) ** 2 + (nodes[2 * node + 1] - latitude) ** 2;
    if (distance < minimum) {
      minimum = distance;
      nearest = node;
    }
  }
  return nearest;
};
// Source CSV has quoted commas and doubled quotes; parse cells without losing names.
const csv = readFileSync(new URL('chicago-facilities/names.csv', source), 'utf8');
const rows = [];
let row = [],
  cell = '',
  quoted = false;
for (let index = 0; index < csv.length; index++) {
  const character = csv[index];
  if (character === '"') {
    if (quoted && csv[index + 1] === '"') {
      cell += '"';
      index++;
    } else quoted = !quoted;
  } else if (!quoted && (character === ',' || character === '\n')) {
    row.push(cell.replace(/\r$/, ''));
    cell = '';
    if (character === '\n') {
      rows.push(row);
      row = [];
    }
  } else cell += character;
}
if (cell || row.length) {
  row.push(cell.replace(/\r$/, ''));
  rows.push(row);
}
const facilityPositions = readColumn('chicago-facilities', 'position', Float32Array);
const facilityCategories = readColumn('chicago-facilities', 'category', Uint8Array);
const categories = ['hospital', 'library', 'cps_school', 'fire_station'];
const facilities = [];
for (let index = 0; index < facilityCategories.length; index++) {
  const longitude = facilityPositions[2 * index],
    latitude = facilityPositions[2 * index + 1];
  if (inside(longitude, latitude))
    facilities.push({
      name: rows[index + 1][1],
      category: categories[facilityCategories[index]],
      lon: round(longitude, 5),
      lat: round(latitude, 5),
      nearestNode: nearestNode(longitude, latitude)
    });
}
const data = {
  bbox,
  nodeCount: originalIds.length,
  edgeCount: edges.length,
  startNode: nearestNode((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
  nodes,
  offsets,
  neighbors: edges.map(edge => nodeIds.get(targets[edge])),
  travelTimeSeconds: edges.map(edge => round(times[edge], 1)),
  lengthMeters: edges.map(edge => round(lengths[edge], 1)),
  roadClass: edges.map(edge => classes[edge]),
  facilities
};
const bytes = `${JSON.stringify(data)}\n`;
if (
  data.nodeCount < 900 ||
  data.nodeCount > 1400 ||
  Buffer.byteLength(bytes) > 150000 ||
  facilities.length < 4 ||
  new Set(facilities.map(facility => facility.category)).size < 2
)
  throw new Error('Dataset budget or facility coverage failed');
const destination = new URL(
  'modules/experimental/src/computation-lenses/data/n-chicago-loop-roads.json',
  root
);
writeFileSync(destination, bytes);
// biome-ignore lint/suspicious/noConsole: Extraction reports its deterministic sizes and counts.
console.log(
  `${fileURLToPath(destination)}: ${Buffer.byteLength(bytes)} bytes; ${data.nodeCount} nodes; ${data.edgeCount} edges; ${facilities.length} facilities; startNode ${data.startNode}; minimum weight ${Math.min(...data.travelTimeSeconds)} s`
);
