// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPULinearReferencing} from '../../../src/gpu-spatial-analysis/linear-referencing';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

let viewSerial = 0;

function getSphericalNodes(positionCount: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'sint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${viewSerial++}`, format, length);
  return new GPULinearReferencing({
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
    queryPositions: view('float32x2', 16),
    positions: view('float32x2', positionCount),
    lineOffsets: view('uint32', 4),
    radius: view('float32', 1),
    output: {
      lineIndices: view('uint32', 16),
      distances: view('float32', 16),
      footPoints: view('float32x2', 16),
      status: {overflow: view('uint32', 1), candidateCount: view('uint32', 1)}
    }
  }).getCommandNodes(graph);
}

it('GPULinearReferencing great-circle builds an arc BVH instead of scanning every arc', () => {
  const ids = getSphericalNodes(300).map(node => node.id);
  const order = [
    'linear-referencing-segment-table',
    'linear-referencing-arc-bounds',
    'linear-referencing-arcs',
    'linear-referencing-node-data',
    'linear-referencing-vertex-measures',
    'linear-referencing-project',
    'linear-referencing-output'
  ];
  let previous = -1;
  for (const prefix of order) {
    const index = ids.findIndex(id => id === prefix || id.startsWith(`${prefix}-`));
    expect(index, prefix).toBeGreaterThan(previous);
    previous = index;
  }
  // 300 positions: Hilbert spatial sort on by default.
  expect(ids.some(id => id.startsWith('linear-referencing-arcs-sort'))).toBe(true);
});

it('GPULinearReferencing great-circle kernels parse and fit the core limits', () => {
  const nodes = getSphericalNodes(300);
  const sources = getParsedKernelSources(nodes);
  const project = [...sources].find(([id]) => id.includes('linear-referencing-project'));
  expect(project).toBeDefined();
  // The traversal replaced the two full scans over every row.
  expect(project?.[1]).not.toMatch(/for \(var row = 0u; row \+ 1u < ROW_COUNT/);
  expect(project?.[1]).toMatch(/boxLowerAngle/);
  for (const [kernelId, source] of sources) {
    expect(getStorageBindingCount(source), kernelId).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(source), kernelId).toBeLessThanOrEqual(8192);
  }
  // Sizes may be baked, but no kernel family repeats with a size-dependent variant (rule 12). Both
  // sizes use 512 BVH leaves, so the upstream GPUBVH picks the same build strategy.
  expect(getGPUCommandNodePipelineKeys(getSphericalNodes(500)).size).toBe(
    getGPUCommandNodePipelineKeys(nodes).size
  );
});

/** f64 replica of the arc box rule of `createSphericalArcBoundsNode`, checked by dense sampling. */
function getArcBox(start: [number, number], end: [number, number]) {
  const toRadians = Math.PI / 180;
  const unit = ([longitude, latitude]: [number, number]) => [
    Math.cos(latitude * toRadians) * Math.cos(longitude * toRadians),
    Math.cos(latitude * toRadians) * Math.sin(longitude * toRadians),
    Math.sin(latitude * toRadians)
  ];
  const a = unit(start);
  const b = unit(end);
  const cosine = Math.min(1, Math.max(-1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]));
  const angle = Math.acos(cosine);
  // Tangent at a toward b: the component of b orthogonal to a, normalized.
  const orthogonal = b.map((value, axis) => value - cosine * a[axis]);
  const norm = Math.hypot(...orthogonal);
  const tangent = orthogonal.map(value => value / norm);
  const reach = Math.asin(Math.min(1, Math.hypot(a[2], tangent[2]))) / toRadians;
  const peak = Math.atan2(tangent[2], a[2]);
  const trough = peak > 0 ? peak - Math.PI : peak + Math.PI;
  let minimumLatitude = Math.min(start[1], end[1]);
  let maximumLatitude = Math.max(start[1], end[1]);
  if (peak >= 0 && peak <= angle) {
    maximumLatitude = Math.max(maximumLatitude, reach);
  }
  if (trough >= 0 && trough <= angle) {
    minimumLatitude = Math.min(minimumLatitude, -reach);
  }
  const padding = 1e-4;
  let minimumLongitude = Math.min(start[0], end[0]) - padding;
  let maximumLongitude = Math.max(start[0], end[0]) + padding;
  minimumLatitude -= padding;
  maximumLatitude += padding;
  if (Math.abs(end[0] - start[0]) >= 179 || maximumLatitude >= 89.9 || minimumLatitude <= -89.9) {
    minimumLongitude = -180;
    maximumLongitude = 180;
  }
  return {minimumLongitude, maximumLongitude, minimumLatitude, maximumLatitude, a, tangent, angle};
}

it('arc boxes contain every point of the great-circle arc (latitude peak, antimeridian, poles)', () => {
  let seed = 3;
  const random = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  const cases: [[number, number], [number, number]][] = [
    // A long east-west arc at 60 degrees bulges north of both endpoints.
    [
      [-60, 60],
      [60, 60]
    ],
    // Across the antimeridian.
    [
      [175, 10],
      [-175, 12]
    ],
    // Near the pole.
    [
      [0, 85],
      [150, 86]
    ]
  ];
  for (let trial = 0; trial < 400; trial++) {
    const start: [number, number] = [random() * 360 - 180, random() * 170 - 85];
    // Wrapped into [-180, 180): an arc whose short path crosses the antimeridian then has a raw
    // longitude difference above 180 degrees and gets every longitude.
    const end: [number, number] = [
      ((((start[0] + (random() - 0.5) * 120 + 180) % 360) + 360) % 360) - 180,
      Math.max(-89, Math.min(89, start[1] + (random() - 0.5) * 80))
    ];
    cases.push([start, end]);
  }
  for (const [start, end] of cases) {
    const box = getArcBox(start, end);
    for (let step = 0; step <= 200; step++) {
      const s = (box.angle * step) / 200;
      const point = box.a.map(
        (value, axis) => Math.cos(s) * value + Math.sin(s) * box.tangent[axis]
      );
      const latitude = (Math.asin(point[2]) * 180) / Math.PI;
      let longitude = (Math.atan2(point[1], point[0]) * 180) / Math.PI;
      // +-180 is one meridian; atan2 may return either sign there.
      if (Math.abs(Math.abs(longitude) - 180) < 1e-9 && box.minimumLongitude <= -180 + 1e-3) {
        longitude = box.minimumLongitude;
      }
      expect(latitude, `${start} ${end} latitude`).toBeGreaterThanOrEqual(box.minimumLatitude);
      expect(latitude, `${start} ${end} latitude`).toBeLessThanOrEqual(box.maximumLatitude);
      expect(longitude, `${start} ${end} longitude`).toBeGreaterThanOrEqual(box.minimumLongitude);
      expect(longitude, `${start} ${end} longitude`).toBeLessThanOrEqual(box.maximumLongitude);
    }
  }
});
