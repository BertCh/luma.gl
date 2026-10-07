// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUCoverageDissolve,
  GPUCoverageValidity,
  GPU_COVERAGE_VALIDITY_FLAG,
  getGPUCoverageValidityParameterValues
} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {flattenPolygons, type OraclePolygons} from '../spatial-weights/spatial-weights-oracle';
import {
  COVERAGE_DISSOLVE_SCENES,
  COVERAGE_VALIDITY_SCENES,
  type CoverageDissolveScene,
  type CoverageValidityScene
} from './coverage-topology-scenes';

type TestDevice = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

function normalizeSegment(x0: number, y0: number, x1: number, y1: number): string {
  const first = [x0, y0];
  const second = [x1, y1];
  const swap = x0 > x1 || (x0 === x1 && y0 > y1);
  return (swap ? [...second, ...first] : [...first, ...second]).join(',');
}

type ValidityRun = {
  /** `polygon,x0,y0,x1,y1` of every flagged segment. */
  flagged: Set<string>;
  flags: number[];
  isValid: number;
  invalidSegmentCount: number;
  polygonInvalidCounts: number[];
};

/** Reads flagged segments for the gap widths in order, re-encoding one compiled graph. */
async function runValidity(
  device: TestDevice,
  scene: CoverageValidityScene,
  gapWidths: number[],
  spatialSort?: boolean
): Promise<ValidityRun[]> {
  const polygons = scene.polygons as OraclePolygons;
  const layout = flattenPolygons(polygons);
  const vertexCount = layout.positions.length / 2;
  const polygonCount = polygons.length;
  const rig = new WeightsRig(device);
  const segmentFlags = rig.output('uint32', vertexCount);
  const polygonInvalidCounts = rig.output('uint32', polygonCount);
  const invalidSegmentCount = rig.output('uint32', 1);
  const isValid = rig.output('uint32', 1);
  const parameters = new GPUParameterBuffer(device, {
    id: 'coverage-validity-parameters',
    format: 'float32',
    length: 4,
    values: getGPUCoverageValidityParameterValues({gapWidth: gapWidths[0]})
  });
  rig.graph.add(
    new GPUCoverageValidity({
      positions: rig.input(layout.positions, 'float32x2', vertexCount),
      ringOffsets: rig.input(layout.ringOffsets, 'uint32', layout.ringOffsets.length),
      polygonOffsets: rig.input(layout.polygonOffsets, 'uint32', layout.polygonOffsets.length),
      parameters: parameters.importToGraph(rig.graph),
      spatialSort,
      output: {
        segmentFlags: segmentFlags.view,
        polygonInvalidCounts: polygonInvalidCounts.view,
        invalidSegmentCount: invalidSegmentCount.view,
        isValid: isValid.view
      }
    })
  );
  const compiled = rig.graph.compile();
  const runs: ValidityRun[] = [];
  for (const gapWidth of gapWidths) {
    parameters.write(getGPUCoverageValidityParameterValues({gapWidth}));
    submitGraph(device, compiled, undefined);
    const flags = await readUint32(segmentFlags.buffer, vertexCount);
    const flagged = new Set<string>();
    let vertex = 0;
    polygons.forEach((rings, polygon) => {
      for (const ring of rings) {
        ring.forEach(([x0, y0], index) => {
          if (flags[vertex + index] !== 0) {
            const [x1, y1] = ring[(index + 1) % ring.length];
            flagged.add(`${polygon},${normalizeSegment(x0, y0, x1, y1)}`);
          }
        });
        vertex += ring.length;
      }
    });
    runs.push({
      flagged,
      flags,
      isValid: (await readUint32(isValid.buffer, 1))[0],
      invalidSegmentCount: (await readUint32(invalidSegmentCount.buffer, 1))[0],
      polygonInvalidCounts: await readUint32(polygonInvalidCounts.buffer, polygonCount)
    });
  }
  compiled.destroy?.();
  parameters.destroy();
  rig.destroy();
  return runs;
}

it('GPUCoverageValidity matches Shapely coverage_invalid_edges on every scene', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const mismatches: string[] = [];
  let checkedSegments = 0;
  for (const scene of COVERAGE_VALIDITY_SCENES) {
    const gapWidths = scene.cases.map(testCase => testCase.gapWidth);
    const runs = await runValidity(device, scene, gapWidths);
    scene.cases.forEach((testCase, caseIndex) => {
      const run = runs[caseIndex];
      const expected = new Set(
        testCase.invalid.map(
          ([polygon, x0, y0, x1, y1]) => `${polygon},${normalizeSegment(x0, y0, x1, y1)}`
        )
      );
      checkedSegments += expected.size;
      const missing = [...expected].filter(segment => !run.flagged.has(segment));
      const extra = [...run.flagged].filter(segment => !expected.has(segment));
      if (missing.length || extra.length) {
        mismatches.push(
          `${scene.name} gap ${testCase.gapWidth}: missing ${JSON.stringify(missing)} extra ${JSON.stringify(extra)}`
        );
      }
      expect(run.isValid, `${scene.name} gap ${testCase.gapWidth} isValid`).toBe(
        testCase.isValid ? 1 : 0
      );
      expect(run.invalidSegmentCount).toBe(run.flagged.size);
      expect(run.polygonInvalidCounts.reduce((sum, count) => sum + count, 0)).toBe(
        run.flagged.size
      );
    });
  }
  expect(mismatches).toEqual([]);
  expect(checkedSegments).toBeGreaterThan(100);
});

it('GPUCoverageValidity reports flag kinds and re-encodes a new gap width without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const byName = (name: string): CoverageValidityScene =>
    COVERAGE_VALIDITY_SCENES.find(scene => scene.name === name)!;
  // One compiled graph, three gap widths: the gap appears only once the width reaches it.
  const gap = await runValidity(device, byName('gap'), [0, 0.125, 0.03125, 0.0625]);
  expect(gap.map(run => run.isValid)).toEqual([1, 0, 1, 0]);
  const gapFlags = gap[1].flags.filter(flag => flag !== 0);
  expect(gapFlags).toEqual([GPU_COVERAGE_VALIDITY_FLAG.gap, GPU_COVERAGE_VALIDITY_FLAG.gap]);
  const kinds = async (name: string) => {
    const [run] = await runValidity(device, byName(name), [0]);
    return [...new Set(run.flags.filter(flag => flag !== 0))].sort();
  };
  expect(await kinds('crossing')).toEqual([GPU_COVERAGE_VALIDITY_FLAG.crossing]);
  expect(await kinds('mismatch')).toEqual([GPU_COVERAGE_VALIDITY_FLAG.unmatched]);
  expect(await kinds('duplicate')).toEqual([GPU_COVERAGE_VALIDITY_FLAG.overlap]);
  expect(await kinds('island')).toEqual([GPU_COVERAGE_VALIDITY_FLAG.overlap]);
  // Morton sorting changes cost only.
  const jittered = byName('delaunay-jitter-3');
  const [sorted] = await runValidity(device, jittered, [0.125], true);
  const [unsorted] = await runValidity(device, jittered, [0.125], false);
  expect(sorted.flags).toEqual(unsorted.flags);
});

type DissolveRun = {
  count: number;
  overflow: number;
  boundarySegmentCount: number;
  /** Per label: summed signed area, shells, holes and undirected edges. */
  labels: Map<number, {area: number; parts: number; holes: number; edges: Set<string>}>;
};

async function runDissolve(
  device: TestDevice,
  scene: Pick<CoverageDissolveScene, 'polygons' | 'labels'>,
  options: {snapTolerance?: number; ringCapacity?: number; vertexCapacity?: number} = {}
): Promise<DissolveRun> {
  const polygons = scene.polygons as OraclePolygons;
  const layout = flattenPolygons(polygons);
  const vertexCount = layout.positions.length / 2;
  const ringCapacity = options.ringCapacity ?? layout.ringOffsets.length;
  const vertexCapacity = options.vertexCapacity ?? vertexCount + ringCapacity;
  const rig = new WeightsRig(device);
  const out = {
    ringOffsets: rig.output('uint32', ringCapacity + 1),
    positions: rig.output('float32x2', vertexCapacity),
    ringAreas: rig.output('float32', ringCapacity),
    ringIsHole: rig.output('uint32', ringCapacity),
    ringGroups: rig.output('uint32', ringCapacity),
    count: rig.output('uint32', 1),
    overflow: rig.output('uint32', 1),
    boundarySegmentCount: rig.output('uint32', 1)
  };
  rig.run(
    new GPUCoverageDissolve({
      positions: rig.input(layout.positions, 'float32x2', vertexCount),
      ringOffsets: rig.input(layout.ringOffsets, 'uint32', layout.ringOffsets.length),
      polygonOffsets: rig.input(layout.polygonOffsets, 'uint32', layout.polygonOffsets.length),
      labels: rig.input(Uint32Array.from(scene.labels), 'uint32', scene.labels.length),
      snapTolerance: options.snapTolerance,
      output: {
        ringOffsets: out.ringOffsets.view,
        positions: out.positions.view,
        ringAreas: out.ringAreas.view,
        ringIsHole: out.ringIsHole.view,
        ringGroups: out.ringGroups.view,
        count: out.count.view,
        overflow: out.overflow.view,
        boundarySegmentCount: out.boundarySegmentCount.view
      }
    })
  );
  const [count] = await readUint32(out.count.buffer, 1);
  const offsets = await readUint32(out.ringOffsets.buffer, ringCapacity + 1);
  const coordinates = await readFloat32(out.positions.buffer, 2 * vertexCapacity);
  const areas = await readFloat32(out.ringAreas.buffer, ringCapacity);
  const isHole = await readUint32(out.ringIsHole.buffer, ringCapacity);
  const groups = await readUint32(out.ringGroups.buffer, ringCapacity);
  const labels: DissolveRun['labels'] = new Map();
  for (let ring = 0; ring < count; ring++) {
    const entry = labels.get(groups[ring]) ?? {area: 0, parts: 0, holes: 0, edges: new Set()};
    labels.set(groups[ring], entry);
    entry.area += areas[ring];
    if (isHole[ring]) {
      entry.holes++;
    } else {
      entry.parts++;
    }
    for (let vertex = offsets[ring]; vertex + 1 < offsets[ring + 1]; vertex++) {
      entry.edges.add(
        normalizeSegment(
          coordinates[2 * vertex],
          coordinates[2 * vertex + 1],
          coordinates[2 * vertex + 2],
          coordinates[2 * vertex + 3]
        )
      );
    }
  }
  const result = {
    count,
    overflow: (await readUint32(out.overflow.buffer, 1))[0],
    boundarySegmentCount: (await readUint32(out.boundarySegmentCount.buffer, 1))[0],
    labels
  };
  rig.destroy();
  return result;
}

it('GPUCoverageDissolve matches GeoPandas dissolve(method="coverage") per label', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const scene of COVERAGE_DISSOLVE_SCENES) {
    const run = await runDissolve(device, scene);
    expect(run.overflow, scene.name).toBe(0);
    expect([...run.labels.keys()].sort(), scene.name).toEqual(
      scene.expected.map(entry => entry.label).sort()
    );
    let boundarySegments = 0;
    for (const expected of scene.expected) {
      const actual = run.labels.get(expected.label)!;
      const expectedEdges = new Set(
        expected.edges.map(([x0, y0, x1, y1]) => normalizeSegment(x0, y0, x1, y1))
      );
      const message = `${scene.name} label ${expected.label}`;
      expect([...actual.edges].sort(), message).toEqual([...expectedEdges].sort());
      expect(actual.parts, message).toBe(expected.parts);
      expect(actual.holes, message).toBe(expected.holes);
      expect(Math.abs(actual.area - expected.area), message).toBeLessThan(1e-4 * expected.area);
      boundarySegments += expected.edges.length;
    }
    // Every boundary edge is one directed segment; shared edges between labels count twice.
    expect(run.boundarySegmentCount, scene.name).toBeGreaterThanOrEqual(boundarySegments);
  }
});

it('GPUCoverageDissolve snaps near-equal vertices, flags overflow and skips internal edges', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const grid = COVERAGE_DISSOLVE_SCENES.find(scene => scene.name === 'grid-4x4')!;
  const clean = await runDissolve(device, grid);
  // Move every vertex of odd polygons by far less than the snap tolerance.
  const nudged = {
    labels: grid.labels,
    polygons: grid.polygons.map((rings, polygon) =>
      polygon % 2
        ? rings.map(ring =>
            ring.map(([x, y]) => [x + 0.00048828125, y - 0.0009765625] as [number, number])
          )
        : rings
    )
  };
  const snapped = await runDissolve(device, nudged, {snapTolerance: 0.01});
  expect(snapped.overflow).toBe(0);
  for (const [label, expected] of clean.labels) {
    const actual = snapped.labels.get(label)!;
    expect(actual.parts).toBe(expected.parts);
    expect(actual.holes).toBe(expected.holes);
    expect(actual.edges.size).toBe(expected.edges.size);
    expect(Math.abs(actual.area - expected.area)).toBeLessThan(1e-3 * expected.area);
  }
  // Without snapping the nudged polygons are no coverage: far more boundary survives.
  const unsnapped = await runDissolve(device, nudged);
  expect(unsnapped.boundarySegmentCount).toBeGreaterThan(clean.boundarySegmentCount);
  // Capacity limits flag overflow instead of writing out of bounds.
  const small = await runDissolve(device, grid, {ringCapacity: 3, vertexCapacity: 20});
  expect(small.overflow).toBe(1);
  expect(small.count).toBeLessThanOrEqual(3);
});

it('GPUCoverageValidity and GPUCoverageDissolve scale to a 60 x 60 grid', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const size = 60;
  const block = 10;
  const polygons: OraclePolygons = [];
  const labels: number[] = [];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      polygons.push([
        [
          [x, y],
          [x + 1, y],
          [x + 1, y + 1],
          [x, y + 1]
        ]
      ]);
      labels.push(Math.floor(y / block) * (size / block) + Math.floor(x / block));
    }
  }
  const startValidity = performance.now();
  const [valid] = await runValidity(
    device,
    {name: 'grid', polygons: polygons as never, cases: []},
    [0.25]
  );
  const validityMilliseconds = performance.now() - startValidity;
  expect(valid.isValid).toBe(1);
  expect(valid.invalidSegmentCount).toBe(0);
  const startDissolve = performance.now();
  const run = await runDissolve(device, {polygons: polygons as never, labels});
  const dissolveMilliseconds = performance.now() - startDissolve;
  expect(run.overflow).toBe(0);
  expect(run.labels.size).toBe((size / block) ** 2);
  for (const entry of run.labels.values()) {
    expect(entry.parts).toBe(1);
    expect(entry.holes).toBe(0);
    expect(entry.edges.size).toBe(4 * block);
    expect(entry.area).toBeCloseTo(block * block, 3);
  }
  console.log(
    `coverage-topology ${size * size} polygons: validity ${validityMilliseconds.toFixed(0)} ms, dissolve ${dissolveMilliseconds.toFixed(0)} ms (includes compile)`
  );
});
