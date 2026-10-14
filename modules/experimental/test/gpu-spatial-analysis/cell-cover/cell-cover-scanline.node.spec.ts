// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUCellCover, type GPUCellCoverProps} from '../../../src/gpu-spatial-analysis/cell-cover';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {
  coverH3OnCPU,
  coverQuadbinOnCPU,
  createStarPolygon,
  flattenCoverFeatures,
  roundFeatures
} from './cell-cover-oracle';
import type {CoverFeature} from './cell-cover-oracle';
import {
  createScanlineLattice,
  getNecessaryCellsOnCPU,
  getScanlineCandidatesOnCPU
} from './cell-cover-scanline-oracle';

type Vertex = [number, number];

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUCellCoverProps> = {},
  vertexCount: number = 4
): GPUCellCoverProps {
  const view = <Format extends 'uint32' | 'float32x2' | 'uint32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `scanline-view-${serial++}`, format, length);
  return {
    family: 'quadbin',
    resolution: 5,
    positions: view('float32x2', vertexCount),
    featureOffsets: view('uint32', 3),
    polygonOffsets: view('uint32', 3),
    ringOffsets: view('uint32', 3),
    candidateCapacity: 100_000,
    output: {
      featureIds: view('uint32', 16),
      cells: view('uint32x2', 16),
      status: {
        count: view('uint32', 1),
        overflow: view('uint32', 1),
        candidateOverflow: view('uint32', 1),
        candidateCount: view('uint32', 1)
      }
    },
    ...overrides
  };
}

function getNodes(overrides: Partial<GPUCellCoverProps> = {}, id: string = 'scanline-graph') {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `${id}-${serial++}`});
  const cover = new GPUCellCover(createProps(graph, overrides));
  return {cover, nodes: cover.getCommandNodes(graph)};
}

it('GPUCellCover candidateEnumeration resolves and validates', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'scanline-validation'});
  const make = (overrides: Partial<GPUCellCoverProps>) =>
    new GPUCellCover(createProps(graph, overrides));
  expect(make({}).candidateEnumeration).toBe('auto');
  // 'auto' builds the stage from 65536 candidates.
  expect(make({candidateCapacity: 65535}).scanline).toBe(false);
  expect(make({candidateCapacity: 65536}).scanline).toBe(true);
  expect(make({candidateEnumeration: 'bounds'}).scanline).toBe(false);
  expect(make({candidateEnumeration: 'scanline', candidateCapacity: 100}).scanline).toBe(true);
  expect(() => make({candidateEnumeration: 'sparse' as 'bounds'})).toThrow(/candidateEnumeration/);
  expect(() => make({candidateEnumeration: 'scanline', scanlineSlotCapacity: 0})).toThrow(
    /scanlineSlotCapacity/
  );
  expect(() =>
    make({candidateEnumeration: 'scanline', candidateCapacity: 100, scanlineSlotCapacity: 101})
  ).toThrow(/scanlineSlotCapacity/);
  expect(() => make({candidateEnumeration: 'scanline', scanlineSlotCapacity: 2 ** 20 + 1})).toThrow(
    /scanlineSlotCapacity/
  );
  // The default capacity is clamped to the candidate capacity.
  expect(
    make({candidateEnumeration: 'scanline', candidateCapacity: 100}).scanlineSlotCapacity
  ).toBe(100);
  expect(
    make({candidateEnumeration: 'scanline', candidateCapacity: 10_000_000}).scanlineSlotCapacity
  ).toBe(16384);
});

const SCANLINE_NODE_NAMES = [
  'edges',
  'clear-demand',
  'rows',
  'choose',
  'cost-scan',
  'select',
  'slots',
  'edge-scan',
  'events',
  'iota',
  'sort',
  'flags',
  'depth-scan',
  'crossing-scan',
  'cover-state',
  'gaps',
  'gap-scan',
  'compact',
  'runs',
  'run-scan',
  'finish',
  'clamp'
];

it('GPUCellCover wires the scanline stage between the counts and the candidate scan', () => {
  for (const family of ['quadbin', 'h3'] as const) {
    for (const edgeSlabs of [false, true]) {
      const bounds = getNodes({family, candidateEnumeration: 'bounds', edgeSlabs});
      const boundsIds = bounds.nodes.map(node => node.id);
      expect(boundsIds.some(id => id.includes('-scanline-'))).toBe(false);
      for (const enumeration of ['scanline', 'auto'] as const) {
        const {cover, nodes} = getNodes({family, candidateEnumeration: enumeration, edgeSlabs});
        expect(cover.scanline).toBe(true);
        const ids = nodes.map(node => node.id);
        expect(new Set(ids).size).toBe(ids.length);
        for (const name of SCANLINE_NODE_NAMES) {
          expect(
            ids.some(id => id.startsWith(`cell-cover-scanline-${name}`)),
            `${family} ${name}`
          ).toBe(true);
        }
        // The stage replaces the bounding-box counts, so it sits after the count kernel and before
        // the candidate scan, the slab gate (which reads the counts) and the test.
        const first = ids.findIndex(id => id.startsWith('cell-cover-scanline-'));
        const last = ids.map(id => id.startsWith('cell-cover-scanline-')).lastIndexOf(true);
        expect(ids.indexOf('cell-cover-count')).toBeLessThan(first);
        expect(last).toBeLessThan(ids.findIndex(id => id.startsWith('cell-cover-candidate-scan')));
        expect(last).toBeLessThan(ids.indexOf('cell-cover-test'));
        // Every other node of the bounding-box pipeline is unchanged and in the same order.
        expect(ids.filter(id => !id.startsWith('cell-cover-scanline-'))).toEqual(boundsIds);
        // Event sort: two stable passes (position, then row).
        expect(ids.filter(id => /scanline-sort-pass-\d+-sort/.test(id)).length).toBeGreaterThan(0);
      }
    }
  }
});

it('GPUCellCover scanline kernels parse and bind at most 8 storage buffers', () => {
  for (const family of ['quadbin', 'h3'] as const) {
    for (const withCore of [false, true]) {
      const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
        id: `scanline-wgsl-${serial++}`
      });
      const base = createProps(graph, {
        family,
        resolution: 6,
        candidateEnumeration: 'scanline',
        edgeSlabs: true
      });
      const core = createTransientView(graph, `scanline-core-${serial++}`, 'uint32', 16);
      const cover = new GPUCellCover(withCore ? {...base, output: {...base.output, core}} : base);
      const nodes = cover.getCommandNodes(graph);
      const sources = getParsedKernelSources(nodes);
      const scanline = [...sources].filter(([id]) => id.includes('-scanline-'));
      expect(scanline.length).toBeGreaterThanOrEqual(SCANLINE_NODE_NAMES.length);
      for (const [id, source] of sources) {
        expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
      }
      // No loop index, round or chunk reaches the WGSL (rule 12): the shared module does not
      // depend on the id or on anything but the shapes, so renaming the contributor only renames
      // the pipelines' labels.
      const other = new GPUCommandGraph(createNullWebGPUDevice(), {
        id: `scanline-keys-${serial++}`
      });
      const otherBase = createProps(other, {
        id: 'renamed',
        family,
        resolution: 6,
        candidateEnumeration: 'scanline',
        edgeSlabs: true
      });
      const otherNodes = new GPUCellCover(
        withCore
          ? {
              ...otherBase,
              output: {
                ...otherBase.output,
                core: createTransientView(other, `scanline-core-${serial++}`, 'uint32', 16)
              }
            }
          : otherBase
      ).getCommandNodes(other);
      expect(getGPUCommandNodePipelineKeys(otherNodes).size).toBe(
        getGPUCommandNodePipelineKeys(nodes).size
      );
      expect(getGPUCommandNodeKernelSources(otherNodes).length).toBe(
        getGPUCommandNodeKernelSources(nodes).length
      );
    }
  }
});

it('GPUCellCover scanline test kernel decodes runs and keeps the bounding-box decode', () => {
  for (const family of ['quadbin', 'h3'] as const) {
    const bounds = getParsedKernelSources(
      getNodes({family, candidateEnumeration: 'bounds'}).nodes
    ).get('cell-cover-test');
    const scanline = getParsedKernelSources(
      getNodes({family, candidateEnumeration: 'scanline'}).nodes
    ).get('cell-cover-test');
    expect(bounds).toBeDefined();
    expect(scanline).toBeDefined();
    expect(bounds).not.toContain('firstRun');
    expect(scanline).toContain('firstRun');
    expect(scanline).toContain('candidateLocal % width');
    // Range table stride: lattice + flag words + run words with the stage, the lattice only without.
    expect(bounds).toContain('const RANGE_STRIDE: u32 = 5u;');
    expect(scanline).toContain('const RANGE_STRIDE: u32 = 8u;');
  }
});

// ---------------------------------------------------------------------------------------------
// CPU model of the scanline enumeration on adversarial polygons.
// ---------------------------------------------------------------------------------------------

function createStrip(from: Vertex, to: Vertex, width: number): number[][] {
  const length = Math.hypot(to[0] - from[0], to[1] - from[1]);
  const normal: Vertex = [
    (-(to[1] - from[1]) / length) * (width / 2),
    ((to[0] - from[0]) / length) * (width / 2)
  ];
  return [
    [from[0] + normal[0], from[1] + normal[1]],
    [to[0] + normal[0], to[1] + normal[1]],
    [to[0] - normal[0], to[1] - normal[1]],
    [from[0] - normal[0], from[1] - normal[1]]
  ];
}

function createSpiral(
  center: Vertex,
  turns: number,
  outerRadius: number,
  width: number
): number[][] {
  const steps = 24 * turns;
  const side = (offset: number) =>
    Array.from({length: steps + 1}, (_, step) => {
      const angle = (2 * Math.PI * step) / 24;
      const radius = (outerRadius * (step + 1)) / (steps + 1) + offset;
      return [center[0] + radius * Math.cos(angle), center[1] + radius * Math.sin(angle)];
    });
  return [...side(width / 2), ...side(-width / 2).reverse()];
}

function createComb(
  origin: Vertex,
  teeth: number,
  pitch: number,
  toothWidth: number,
  height: number
): number[][] {
  const barHeight = height / 8;
  const width = (teeth - 1) * pitch + toothWidth;
  const ring: number[][] = [
    [origin[0], origin[1]],
    [origin[0] + width, origin[1]],
    [origin[0] + width, origin[1] + barHeight]
  ];
  for (let tooth = teeth - 1; tooth >= 0; tooth--) {
    const left = origin[0] + tooth * pitch;
    ring.push(
      [left + toothWidth, origin[1] + barHeight],
      [left + toothWidth, origin[1] + height],
      [left, origin[1] + height],
      [left, origin[1] + barHeight]
    );
  }
  ring.push([origin[0], origin[1] + barHeight]);
  return ring;
}

function createRectangle([west, south, east, north]: number[]): number[][] {
  return [
    [west, south],
    [east, south],
    [east, north],
    [west, north]
  ];
}

/** Adds vertices to a ring at the row line of a lattice row (vertex on the line, horizontal edge). */
function createLineTouchingFeature(
  family: 'quadbin' | 'h3',
  resolution: number,
  base: number[][],
  mode: 'vertex' | 'horizontal'
): CoverFeature {
  const lattice = createScanlineLattice([[base]], family, resolution);
  if (!lattice) {
    throw new Error('base polygon has no lattice');
  }
  const line = lattice.getRowLine(Math.floor(lattice.rows / 2));
  const [a, b] = [base[0], base[1]];
  const midX = (a[0] + b[0]) / 2;
  // A notch whose vertices sit exactly on the row line, inserted into the first (south) edge so the
  // bounding box is unchanged (the line is above the south edge; the notch goes up to the line).
  const ring =
    mode === 'vertex'
      ? [a, [midX - 0.01, a[1]], [midX, line], [midX + 0.01, a[1]], ...base.slice(1)]
      : [
          a,
          [midX - 0.02, a[1]],
          [midX - 0.02, line],
          [midX + 0.02, line],
          [midX + 0.02, a[1]],
          ...base.slice(1)
        ];
  return roundFeatures([[[ring]]])[0];
}

type Case = {
  name: string;
  feature: CoverFeature;
  /** Candidates must be at most `boundsCount / minimumPruning`. */
  minimumPruning: number;
};

function getCases(family: 'quadbin' | 'h3', resolution: number): Case[] {
  const rounded = (feature: CoverFeature) => roundFeatures([feature])[0];
  const strip = rounded([[createStrip([0.3, 0.2], [7.1, 6.9], 1e-4)]]);
  // A strip wide enough to contain cell centers at the coarse H3 resolutions.
  const wideStrip = rounded([[createStrip([0.3, 0.2], [7.1, 6.9], family === 'h3' ? 0.12 : 0.02)]]);
  const northBase = createRectangle([2, 40, 3.5, 41.5]);
  return [
    {name: 'thin diagonal strip', feature: strip, minimumPruning: 10},
    {name: 'wide diagonal strip', feature: wideStrip, minimumPruning: 8},
    {
      name: 'spiral',
      feature: rounded([[createSpiral([10, 20], 3, 2.4, family === 'h3' ? 0.1 : 0.03)]]),
      minimumPruning: 2
    },
    {
      name: 'comb',
      feature: rounded([[createComb([20, 30], 12, 1.2, 0.05, 6)]]),
      minimumPruning: 2
    },
    {
      name: 'rectangle with two holes',
      feature: rounded([
        [
          createRectangle([30, 10, 33, 13]),
          createRectangle([30.4, 10.4, 31.2, 11.2]).reverse(),
          createRectangle([31.8, 11.8, 32.6, 12.6]).reverse()
        ]
      ]),
      minimumPruning: 1
    },
    {
      name: 'vertex exactly on a row line',
      feature: createLineTouchingFeature(family, resolution, northBase, 'vertex'),
      minimumPruning: 1
    },
    {
      name: 'horizontal edge exactly on a row line',
      feature: createLineTouchingFeature(family, resolution, northBase, 'horizontal'),
      minimumPruning: 1
    },
    {
      name: 'multipolygon with far-apart parts',
      feature: rounded([
        [createRectangle([-9, -7, -8.5, -6.5])],
        [createRectangle([4, 11, 4.6, 11.4])],
        [createStrip([-3, 8], [-2.5, 8.6], 0.05)],
        [createRectangle([9, -6, 9.4, -5.6])]
      ]),
      minimumPruning: 10
    },
    {
      name: 'tiny polygon inside one cell',
      feature: rounded([[createRectangle([5.0001, 5.0001, 5.0003, 5.0003])]]),
      minimumPruning: 1
    }
  ];
}

const FAMILIES: {family: 'quadbin' | 'h3'; resolutions: number[]}[] = [
  {family: 'quadbin', resolutions: [10, 12]},
  {family: 'h3', resolutions: [4, 5]}
];

it('scanline model: candidates contain every cell the exact test could accept', () => {
  for (const {family, resolutions} of FAMILIES) {
    for (const resolution of resolutions) {
      for (const {name, feature, minimumPruning} of getCases(family, resolution)) {
        for (const latitudeJitter of [0, 6e-6]) {
          const label = `${family} ${resolution} ${name} jitter ${latitudeJitter}`;
          const result = getScanlineCandidatesOnCPU(feature, family, resolution, {latitudeJitter});
          expect(result, label).toBeDefined();
          const {lattice, boundsCount, candidates} = result!;
          const present = new Set(
            candidates.map(([row, column]) => row * lattice.columns + column)
          );
          // Row-major, once each.
          expect(present.size, label).toBe(candidates.length);
          expect(
            candidates.every(
              ([row, column], index) =>
                index === 0 ||
                row > candidates[index - 1][0] ||
                (row === candidates[index - 1][0] && column > candidates[index - 1][1])
            ),
            label
          ).toBe(true);
          expect(
            candidates.every(([row, column]) => row < lattice.rows && column < lattice.columns)
          ).toBe(true);
          const necessary = getNecessaryCellsOnCPU(feature, lattice);
          const missing = [...necessary].filter(cell => !present.has(cell));
          expect(missing.length, `${label}: ${missing.length} of ${necessary.size} missing`).toBe(
            0
          );
          // Pruning is only meaningful with enough cells; coarse lattices are a handful of rows.
          if (boundsCount >= 2000) {
            expect(candidates.length * minimumPruning, label).toBeLessThanOrEqual(boundsCount);
          } else {
            expect(candidates.length, label).toBeLessThanOrEqual(boundsCount);
          }
        }
      }
    }
  }
});

it('scanline model: cells accepted by the cover oracles are candidates', () => {
  for (const {family, resolutions} of FAMILIES) {
    for (const resolution of resolutions) {
      for (const {name, feature} of getCases(family, resolution)) {
        const label = `${family} ${resolution} ${name}`;
        const result = getScanlineCandidatesOnCPU(feature, family, resolution);
        const {lattice, candidates} = result!;
        const present = new Set(candidates.map(([row, column]) => row * lattice.columns + column));
        const arrays = flattenCoverFeatures([feature]);
        const accepted =
          family === 'quadbin'
            ? (['center', 'full', 'intersects'] as const).flatMap(
                containment => coverQuadbinOnCPU(arrays, resolution, containment).cells
              )
            : coverH3OnCPU([feature], resolution).cells;
        for (const cell of accepted) {
          const [row, column] = lattice.getCellPosition(cell);
          expect(present.has(row * lattice.columns + column), `${label}: cell ${cell}`).toBe(true);
        }
      }
    }
  }
});

it('scanline model: thin and sparse polygons are enumerated far below the bounding box', () => {
  // The 1e-4 degree wide, 9.6 degrees long diagonal strip at Quadbin z12 and H3 res 5.
  for (const {family, resolution} of [
    {family: 'quadbin', resolution: 12},
    {family: 'h3', resolution: 5}
  ] as const) {
    const strip = getCases(family, resolution)[0].feature;
    const result = getScanlineCandidatesOnCPU(strip, family, resolution, {cheaperOnly: true})!;
    expect(result.selected).toBe(true);
    expect(result.boundsCount).toBeGreaterThan(5000);
    // About three columns of candidates per row against hundreds of columns.
    expect(result.candidates.length).toBeLessThan(result.lattice.rows * 6);
  }
});

it('scanline model: the per-feature choice and capacity fallback follow the documented rule', () => {
  const tiny = getCases('quadbin', 12).find(item => item.name === 'tiny polygon inside one cell')!;
  const strip = getCases('quadbin', 12)[0];
  // A feature whose box is no larger than its scanline work keeps the bounding box ('auto').
  expect(getScanlineCandidatesOnCPU(tiny.feature, 'quadbin', 12)!.selected).toBe(false);
  expect(getScanlineCandidatesOnCPU(tiny.feature, 'quadbin', 12, {force: true})!.selected).toBe(
    true
  );
  expect(getScanlineCandidatesOnCPU(strip.feature, 'quadbin', 12)!.selected).toBe(true);
  // Too few slots: the feature falls back to the bounding box, even when forced.
  const result = getScanlineCandidatesOnCPU(strip.feature, 'quadbin', 12, {
    force: true,
    slotCapacity: 10
  })!;
  expect(result.selected).toBe(false);
  expect(result.slots).toBeGreaterThan(10);
});

it('scanline model: polygon at the H3 latitude limit', () => {
  const feature = roundFeatures([[[createStrip([0, 87.2], [12, 88.95], 0.4)]]])[0];
  for (const resolution of [3, 4]) {
    const result = getScanlineCandidatesOnCPU(feature, 'h3', resolution, {latitudeJitter: 6e-6})!;
    const necessary = getNecessaryCellsOnCPU(feature, result.lattice);
    const present = new Set(
      result.candidates.map(([row, column]) => row * result.lattice.columns + column)
    );
    expect([...necessary].filter(cell => !present.has(cell))).toEqual([]);
    expect(result.candidates.length).toBeGreaterThan(0);
  }
  // Reaching past 89 degrees: the lattice is clamped, the necessary cells are the lattice cells.
  const polar = roundFeatures([[[createRectangle([0, 88, 10, 89.9])]]])[0];
  const result = getScanlineCandidatesOnCPU(polar, 'h3', 4)!;
  const present = new Set(
    result.candidates.map(([row, column]) => row * result.lattice.columns + column)
  );
  expect(
    [...getNecessaryCellsOnCPU(polar, result.lattice)].filter(cell => !present.has(cell))
  ).toEqual([]);
});

it('scanline model: the f32 slack is not overly generous', () => {
  // Without the slack contract a runaway enumeration would also pass the superset tests: the
  // thin strip stays within a handful of columns of the true strip.
  const strip = getCases('quadbin', 12)[0].feature;
  const result = getScanlineCandidatesOnCPU(strip, 'quadbin', 12)!;
  const perRow = new Map<number, number>();
  for (const [row] of result.candidates) {
    perRow.set(row, (perRow.get(row) ?? 0) + 1);
  }
  expect(Math.max(...perRow.values())).toBeLessThanOrEqual(8);
});

it('scanline model: negative control, the boundary trace is needed', () => {
  // A small polygon in the upper part of one z12 tile row crosses no row centre line, so only the
  // boundary trace finds the tile its edges enter.
  const resolution = 12;
  const tileLatitude = (row: number) =>
    (Math.atan(Math.sinh(Math.PI * (1 - (2 * row) / 2 ** resolution))) * 180) / Math.PI;
  const row = 1500;
  const north = tileLatitude(row);
  const centre = tileLatitude(row + 0.5);
  const west = (2100 * 360) / 2 ** resolution - 180 + 0.01;
  const feature = roundFeatures([
    [
      [
        createRectangle([
          west,
          centre + 0.2 * (north - centre),
          west + 0.02,
          centre + 0.6 * (north - centre)
        ])
      ]
    ]
  ])[0];
  const withSpans = getScanlineCandidatesOnCPU(feature, 'quadbin', resolution)!;
  const withoutSpans = getScanlineCandidatesOnCPU(feature, 'quadbin', resolution, {
    omitSpans: true
  })!;
  const necessary = getNecessaryCellsOnCPU(feature, withSpans.lattice);
  expect(necessary.size).toBeGreaterThan(0);
  const toKeys = (result: typeof withSpans) =>
    new Set(result.candidates.map(([r, c]) => r * result.lattice.columns + c));
  expect([...necessary].every(cell => toKeys(withSpans).has(cell))).toBe(true);
  expect([...necessary].some(cell => !toKeys(withoutSpans).has(cell))).toBe(true);
});

it('scanline model: random polygons with holes keep the superset property', () => {
  let state = 20261014;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  for (const {family, resolution} of [
    {family: 'quadbin', resolution: 11},
    {family: 'h3', resolution: 5}
  ] as const) {
    for (let trial = 0; trial < 12; trial++) {
      const center: Vertex = [random() * 60 - 30, random() * 100 - 50];
      const radius = 0.5 + random() * 3;
      const star = createStarPolygon(trial + 1, center, radius, 5 + Math.floor(random() * 30));
      const hole = createStarPolygon(trial + 77, center, radius * 0.25, 7).reverse();
      const feature = roundFeatures([[[star, hole]]])[0];
      const result = getScanlineCandidatesOnCPU(feature, family, resolution, {
        latitudeJitter: trial % 2 === 0 ? 0 : 5e-6
      })!;
      const present = new Set(
        result.candidates.map(([row, column]) => row * result.lattice.columns + column)
      );
      const missing = [...getNecessaryCellsOnCPU(feature, result.lattice)].filter(
        cell => !present.has(cell)
      );
      expect(missing.length, `${family} trial ${trial}`).toBe(0);
    }
  }
});
