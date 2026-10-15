// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Extracts raw, un-noded road centrelines of the Chicago Loop from the showcase `chicago-roads`
// dataset (read-only) for GPUNetworkNoding. The showcase edges are already split at every
// intersection, so streets are rebuilt by chaining edges straight through each junction (turn of at
// most 20 degrees), and the vertices where one selected street meets another are dropped when the
// street is locally straight. The result is what a raw centreline file looks like: continuous
// lines whose crossings and T junctions have to be found geometrically.

import {readFileSync, writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

const root = new URL('../../', import.meta.url);
const source = new URL('examples/deck/spatial-analysis-showcase/public/data/', root);
const readColumn = (file, Type) => {
  const bytes = readFileSync(new URL(`chicago-roads/${file}.bin`, source));
  return new Type(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};
const nodes = readColumn('nodes', Float32Array);
const edgeSource = readColumn('edgeSource', Uint32Array);
const edgeTarget = readColumn('edgeTarget', Uint32Array);
const edgeClass = readColumn('edgeClass', Uint8Array);
const edgeLength = readColumn('edgeLength', Float32Array);
const pathOffsets = readColumn('edgePathOffsets', Uint32Array);
const vertices = readColumn('edgeVertices', Float32Array);
const bbox = [-87.655, 41.865, -87.615, 41.902];
const inside = (x, y) => x >= bbox[0] && x <= bbox[2] && y >= bbox[1] && y <= bbox[3];
const longitudeScale = Math.cos(((bbox[1] + bbox[3]) * Math.PI) / 360);

// One undirected edge per street segment, oriented from the lower to the higher node id. Motorways
// and trunk roads are left out: the Loop's grade-separated crossings are expressways.
const segments = [];
const seen = new Set();
for (let edge = 0; edge < edgeSource.length; edge++) {
  if (edgeClass[edge] < 2) continue;
  const [from, to] = [edgeSource[edge], edgeTarget[edge]];
  const key = from < to ? `${from}:${to}` : `${to}:${from}`;
  const first = pathOffsets[edge];
  const last = pathOffsets[edge + 1];
  let allInside = true;
  for (let vertex = first; vertex < last; vertex++)
    if (!inside(vertices[2 * vertex], vertices[2 * vertex + 1])) allInside = false;
  if (!allInside || seen.has(key)) continue;
  seen.add(key);
  const path = [];
  for (let vertex = first; vertex < last; vertex++)
    path.push([vertices[2 * vertex], vertices[2 * vertex + 1]]);
  // Orient from the lower node id.
  const oriented = from < to ? path : path.slice().reverse();
  segments.push({from: Math.min(from, to), to: Math.max(from, to), path: oriented, cls: edgeClass[edge], length: edgeLength[edge]});
}
const incident = new Map();
segments.forEach((segment, index) => {
  for (const node of [segment.from, segment.to]) {
    if (!incident.has(node)) incident.set(node, []);
    incident.get(node).push(index);
  }
});
// Unit direction leaving `node` along a segment.
const direction = (segment, node) => {
  const path = node === segment.from ? segment.path : segment.path.slice().reverse();
  const [x0, y0] = path[0];
  const [x1, y1] = path[1];
  const dx = (x1 - x0) * longitudeScale;
  const dy = y1 - y0;
  const norm = Math.hypot(dx, dy);
  return [dx / norm, dy / norm];
};
// Pair the two most collinear segments at each junction (turn <= 20 degrees).
const partner = new Map();
for (const [node, list] of [...incident].sort((a, b) => a[0] - b[0])) {
  const candidates = [];
  for (let i = 0; i < list.length; i++)
    for (let j = i + 1; j < list.length; j++) {
      const [ax, ay] = direction(segments[list[i]], node);
      const [bx, by] = direction(segments[list[j]], node);
      const cosine = ax * bx + ay * by;
      if (cosine <= -Math.cos((20 * Math.PI) / 180)) candidates.push({cosine, i: list[i], j: list[j]});
    }
  candidates.sort((a, b) => a.cosine - b.cosine || a.i - b.i || a.j - b.j);
  const used = new Set();
  for (const {i, j} of candidates) {
    if (used.has(i) || used.has(j)) continue;
    used.add(i);
    used.add(j);
    partner.set(`${node}:${i}`, j);
    partner.set(`${node}:${j}`, i);
  }
}
const visited = new Set();
const streets = [];
for (let start = 0; start < segments.length; start++) {
  if (visited.has(start)) continue;
  // Walk to one end of the chain, then collect it.
  let head = start;
  let headNode = segments[start].from;
  const guard = new Set([start]);
  while (partner.has(`${headNode}:${head}`)) {
    const next = partner.get(`${headNode}:${head}`);
    if (guard.has(next)) break;
    guard.add(next);
    headNode = segments[next].from === headNode ? segments[next].to : segments[next].from;
    head = next;
  }
  const chain = [];
  let current = head;
  let node = headNode;
  const walked = new Set();
  for (;;) {
    walked.add(current);
    visited.add(current);
    const segment = segments[current];
    const path = segment.from === node ? segment.path : segment.path.slice().reverse();
    const farNode = segment.from === node ? segment.to : segment.from;
    chain.push({path, farNode, cls: segment.cls, length: segment.length});
    const next = partner.get(`${farNode}:${current}`);
    if (next === undefined || walked.has(next)) break;
    current = next;
    node = farNode;
  }
  const points = [];
  const junctions = [];
  let metres = 0;
  chain.forEach(({path, farNode, length}, index) => {
    for (let vertex = index === 0 ? 0 : 1; vertex < path.length; vertex++) points.push(path[vertex]);
    junctions.push(points.length - 1);
    metres += length;
  });
  streets.push({points, junctions: junctions.slice(0, -1), metres, cls: Math.min(...chain.map(part => part.cls))});
}
streets.sort((a, b) => b.metres - a.metres);
// The eight longest streets of each orientation, so the selection forms a crossing grid.
const isEastWest = street => {
  const [first, last] = [street.points[0], street.points.at(-1)];
  return Math.abs((last[0] - first[0]) * longitudeScale) > Math.abs(last[1] - first[1]);
};
const selected = [
  ...streets.filter(isEastWest).slice(0, 8),
  ...streets.filter(street => !isEastWest(street)).slice(0, 8)
];
// Drop junction vertices that are locally straight, so crossings are interior crossings.
const keyOf = point => `${point[0].toFixed(6)},${point[1].toFixed(6)}`;
const counts = new Map();
for (const street of selected) for (const point of street.points) counts.set(keyOf(point), (counts.get(keyOf(point)) ?? 0) + 1);
let dropped = 0;
const lines = selected.map(street => {
  const kept = street.points.filter((point, index) => {
    if (index === 0 || index === street.points.length - 1) return true;
    if (counts.get(keyOf(point)) < 2) return true;
    const [px, py] = street.points[index - 1];
    const [nx, ny] = street.points[index + 1];
    const ax = (point[0] - px) * longitudeScale;
    const ay = point[1] - py;
    const bx = (nx - point[0]) * longitudeScale;
    const by = ny - point[1];
    const cosine = (ax * bx + ay * by) / (Math.hypot(ax, ay) * Math.hypot(bx, by));
    if (cosine >= Math.cos((5 * Math.PI) / 180)) {
      dropped++;
      return false;
    }
    return true;
  });
  return {kept, cls: street.cls};
});
const lineOffsets = [0];
const positions = [];
for (const {kept} of lines) {
  for (const [x, y] of kept) positions.push(Number(x.toFixed(6)), Number(y.toFixed(6)));
  lineOffsets.push(positions.length / 2);
}
const data = {
  bbox,
  lineCount: lines.length,
  vertexCount: positions.length / 2,
  lineOffsets,
  roadClass: lines.map(line => line.cls),
  positions
};
const bytes = `${JSON.stringify(data)}\n`;
if (data.lineCount < 10 || data.lineCount > 40 || Buffer.byteLength(bytes) > 150000)
  throw new Error('Polyline budget failed');
const destination = new URL('modules/experimental/src/computation-lenses/data/n-chicago-loop-polylines.json', root);
writeFileSync(destination, bytes);
// biome-ignore lint/suspicious/noConsole: Extraction reports its deterministic sizes and counts.
console.log(`${fileURLToPath(destination)}: ${Buffer.byteLength(bytes)} bytes; ${data.lineCount} lines; ${data.vertexCount} vertices; ${dropped} straight junction vertices dropped; lengths ${selected.map(street => Math.round(street.metres)).join(',')}`);
