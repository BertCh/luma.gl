// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPU_REGION_STATISTICS_FLAGS} from '../../../src/map-graphs/region-statistics';
import {
  createRandom,
  GridStatisticsHarness,
  type GridStatisticsHarnessOptions,
  type RecipeReadback
} from './gpu-region-statistics-grid-harness';
import {
  CIRCLE,
  computeRegionStatistics,
  isInsidePolygon,
  isInsideRectangle,
  LASSO,
  POSITIONS,
  RECTANGLE,
  RECTANGLE_2,
  SOURCE_IDS,
  selectRows,
  VALUES
} from './region-statistics-fixtures';

const CANDIDATES_TRUNCATED = GPU_REGION_STATISTICS_FLAGS.candidatesTruncated;
const DOMAIN_SIZE = 1000;
const WORD = {
  selectedCount: 0,
  valueCount: 1,
  sum: 2,
  mean: 3,
  minimum: 4,
  maximum: 5,
  outside: 6,
  flags: 7
};

type RandomData = {positions: Float32Array; values: Float32Array; sourceIds: Uint32Array};

function createRandomData(seed: number, count: number, integerValues: boolean): RandomData {
  const random = createRandom(seed);
  const positions = new Float32Array(count * 2);
  const values = new Float32Array(count);
  const sourceIds = new Uint32Array(count);
  for (let row = 0; row < count; row++) {
    positions[row * 2] = random() * DOMAIN_SIZE;
    positions[row * 2 + 1] = random() * DOMAIN_SIZE;
    values[row] =
      random() < 0.05
        ? Number.NaN
        : integerValues
          ? Math.floor(random() * 101) - 50
          : random() * 200 - 100;
    sourceIds[row] = 5000 + row * 3;
  }
  return {positions, values, sourceIds};
}

/** Asserts the documented identity guarantee between brute-force and grid-index readbacks. */
function expectIdentical(
  brute: RecipeReadback,
  grid: RecipeReadback,
  exactSum: boolean,
  label: string
): void {
  const message = (what: string) => `${label}: ${what}`;
  expect(grid.result.candidatesTruncated, message('candidatesTruncated')).toBe(false);
  expect(brute.result.candidatesTruncated).toBe(false);
  for (const [name, word] of Object.entries(WORD)) {
    if (name === 'sum' || name === 'mean') continue;
    const bruteWord =
      word === WORD.flags ? brute.words[word] & ~CANDIDATES_TRUNCATED : brute.words[word];
    expect(grid.words[word], message(name)).toBe(bruteWord);
  }
  expect(Array.from(grid.result.histogram), message('histogram')).toEqual(
    Array.from(brute.result.histogram)
  );
  if (exactSum) {
    expect(grid.result.sum, message('sum')).toBe(brute.result.sum);
    expect(grid.result.mean, message('mean')).toBe(brute.result.mean);
  } else {
    const tolerance = 1e-4 * Math.max(1, Math.abs(brute.result.sum));
    expect(Math.abs(grid.result.sum - brute.result.sum), message('sum')).toBeLessThanOrEqual(
      tolerance
    );
    expect(Math.abs(grid.result.mean - brute.result.mean), message('mean')).toBeLessThanOrEqual(
      1e-4 * Math.max(1, Math.abs(brute.result.mean))
    );
  }
  expect(grid.ids, message('ids')).toEqual(brute.ids);
  expect(grid.outputOverflow, message('output overflow')).toBe(brute.outputOverflow);
  expect(grid.outputTotal, message('output total')).toBe(brute.outputTotal);
  expect(grid.mask, message('mask')).toEqual(brute.mask);
}

function createStarPolygon(random: () => number, vertexCount: number): number[] {
  const centerX = 150 + random() * 700;
  const centerY = 150 + random() * 700;
  const vertices: number[] = [];
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const angle = (vertex / vertexCount) * Math.PI * 2;
    const radius = 40 + random() * 200;
    vertices.push(centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius);
  }
  return vertices;
}

type SelectionCase = {label: string; apply: (harness: GridStatisticsHarness) => void};

function createSelectionCases(
  kind: 'rectangle' | 'radius' | 'polygon',
  seed: number
): SelectionCase[] {
  const random = createRandom(seed);
  const cases: SelectionCase[] = [];
  if (kind === 'rectangle') {
    for (let index = 0; index < 8; index++) {
      const x = random() * 900;
      const y = random() * 900;
      const width = random() * (index < 4 ? 40 : 400);
      const height = random() * (index < 4 ? 40 : 400);
      cases.push({
        label: `rectangle ${index}`,
        apply: harness => harness.setRectangle([x, y, x + width, y + height])
      });
    }
    cases.push(
      {label: 'whole domain', apply: harness => harness.setRectangle([0, 0, 1000, 1000])},
      {label: 'beyond domain', apply: harness => harness.setRectangle([-500, -500, 2000, 300])},
      {label: 'cell aligned', apply: harness => harness.setRectangle([31.25, 62.5, 156.25, 250])},
      {label: 'inverted', apply: harness => harness.setRectangle([500, 500, 100, 100])},
      {label: 'empty corner', apply: harness => harness.setRectangle([2000, 2000, 3000, 3000])},
      {label: 'nan bounds', apply: harness => harness.setRectangle([Number.NaN, 0, 500, 500])}
    );
  } else if (kind === 'radius') {
    for (let index = 0; index < 8; index++) {
      const x = random() * 1000;
      const y = random() * 1000;
      const radius = random() * (index < 4 ? 30 : 300);
      cases.push({label: `circle ${index}`, apply: harness => harness.setCircle([x, y, radius])});
    }
    cases.push(
      {label: 'zero radius', apply: harness => harness.setCircle([500, 500, 0])},
      {label: 'negative radius', apply: harness => harness.setCircle([500, 500, -5])},
      {label: 'beyond domain', apply: harness => harness.setCircle([1050, 500, 200])},
      {label: 'whole domain', apply: harness => harness.setCircle([500, 500, 800])}
    );
  } else {
    for (let index = 0; index < 8; index++) {
      const vertices = createStarPolygon(random, 8 + index * 3);
      cases.push({label: `lasso ${index}`, apply: harness => harness.setPolygon(vertices)});
    }
    const nanLasso = createStarPolygon(random, 12);
    nanLasso[4] = Number.NaN;
    nanLasso[9] = Number.POSITIVE_INFINITY;
    cases.push(
      {label: 'lasso with non-finite vertices', apply: harness => harness.setPolygon(nanLasso)},
      {label: 'two vertices', apply: harness => harness.setPolygon(createStarPolygon(random, 2))},
      {
        label: 'vertex count above capacity',
        apply: harness => harness.setPolygon(createStarPolygon(random, 64), 200)
      },
      {
        label: 'concave',
        apply: harness => harness.setPolygon([100, 100, 900, 100, 900, 900, 500, 500, 100, 900])
      }
    );
  }
  return cases;
}

type RunOptions = Pick<
  GridStatisticsHarnessOptions,
  'binCount' | 'histogramDomain' | 'withOutput' | 'withOutputMask' | 'indexMode'
> & {integerValues: boolean; useSourceIds?: boolean};

async function runRandomComparison(
  id: string,
  kind: 'rectangle' | 'radius' | 'polygon',
  count: number,
  options: RunOptions
): Promise<boolean> {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return false;
  }
  const data = createRandomData(count + kind.length, count, options.integerValues);
  const harness = new GridStatisticsHarness({
    device,
    id,
    selectionKind: kind,
    positions: data.positions,
    values: data.values,
    sourceIds: options.useSourceIds === false ? undefined : data.sourceIds,
    candidateCapacity: count,
    binCount: options.binCount,
    histogramDomain: options.histogramDomain,
    withOutput: options.withOutput,
    withOutputMask: options.withOutputMask,
    indexMode: options.indexMode
  });
  harness.buildIndex();
  let nonEmptyCases = 0;
  // One compiled pair of graphs serves every selection: per-frame parameters never recompile.
  for (const selectionCase of createSelectionCases(kind, count)) {
    selectionCase.apply(harness);
    harness.encodeBoth();
    const brute = await harness.read('brute');
    const grid = await harness.read('grid');
    expectIdentical(brute, grid, options.integerValues, `${id} ${selectionCase.label}`);
    if (brute.result.selectedCount > 0) nonEmptyCases++;
  }
  expect(nonEmptyCases).toBeGreaterThan(3);
  harness.destroy();
  return true;
}

for (const kind of ['rectangle', 'radius', 'polygon'] as const) {
  it(`GPURegionStatistics grid index matches brute force for ${kind} with integer values`, async () => {
    await runRandomComparison(`grid-int-${kind}`, kind, 30000, {
      integerValues: true,
      binCount: 8,
      withOutput: true,
      withOutputMask: true,
      indexMode: 'same-graph'
    });
  });

  it(`GPURegionStatistics grid index matches brute force for ${kind} with float values`, async () => {
    await runRandomComparison(`grid-float-${kind}`, kind, 20000, {
      integerValues: false,
      binCount: 16,
      histogramDomain: [-50, 50],
      withOutput: true,
      indexMode: 'separate-graph'
    });
  });
}

it('GPURegionStatistics grid index matches brute force without output and with row-index IDs', async () => {
  await runRandomComparison('grid-no-output', 'rectangle', 25000, {
    integerValues: true,
    binCount: 5,
    histogramDomain: 'selection',
    indexMode: 'same-graph'
  });
  await runRandomComparison('grid-row-ids', 'radius', 25000, {
    integerValues: true,
    withOutput: true,
    useSourceIds: false,
    indexMode: 'separate-graph'
  });
});

it('GPURegionStatistics grid index matches the CPU fixtures', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rowCount = POSITIONS.length / 2;
  const cases = [
    {
      kind: 'rectangle' as const,
      apply: (harness: GridStatisticsHarness, which: number) =>
        harness.setRectangle(which ? RECTANGLE_2 : RECTANGLE),
      rows: (which: number) =>
        selectRows((x, y) => isInsideRectangle(x, y, which ? RECTANGLE_2 : RECTANGLE))
    },
    {
      kind: 'radius' as const,
      apply: (harness: GridStatisticsHarness) => harness.setCircle(CIRCLE),
      rows: () => selectRows((x, y) => Math.hypot(x - CIRCLE[0], y - CIRCLE[1]) <= CIRCLE[2])
    },
    {
      kind: 'polygon' as const,
      apply: (harness: GridStatisticsHarness) => harness.setPolygon(LASSO),
      rows: () => selectRows((x, y) => isInsidePolygon(x, y, LASSO, 5))
    }
  ];
  for (const fixtureCase of cases) {
    const harness = new GridStatisticsHarness({
      device,
      id: `grid-fixture-${fixtureCase.kind}`,
      selectionKind: fixtureCase.kind,
      positions: POSITIONS,
      values: VALUES,
      sourceIds: SOURCE_IDS,
      candidateCapacity: rowCount,
      binCount: 4,
      gridSize: [4, 4],
      domain: [0, 0, 10, 10],
      withOutput: true,
      indexMode: 'same-graph'
    });
    for (const which of fixtureCase.kind === 'rectangle' ? [0, 1] : [0]) {
      fixtureCase.apply(harness, which);
      harness.encodeGrid();
      const grid = await harness.read('grid');
      const rows = fixtureCase.rows(which);
      const expected = computeRegionStatistics(rows, VALUES, 4);
      expect(grid.result.selectedCount).toBe(expected.selectedCount);
      expect(grid.result.valueCount).toBe(expected.valueCount);
      expect(grid.result.sum).toBeCloseTo(expected.sum, 4);
      expect(grid.result.minimum).toBeCloseTo(expected.minimum, 4);
      expect(grid.result.maximum).toBeCloseTo(expected.maximum, 4);
      expect(Array.from(grid.result.histogram)).toEqual(expected.histogram);
      expect(grid.result.candidatesTruncated).toBe(false);
      expect(grid.ids).toEqual(rows.map(row => SOURCE_IDS[row]));
    }
    harness.destroy();
  }
});

it('GPURegionStatistics grid index reports empty selections', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const data = createRandomData(7, 5000, true);
  const harness = new GridStatisticsHarness({
    device,
    id: 'grid-empty',
    selectionKind: 'rectangle',
    positions: data.positions,
    values: data.values,
    candidateCapacity: 64,
    binCount: 4,
    withOutput: true,
    withOutputMask: true,
    indexMode: 'same-graph'
  });
  harness.setRectangle([2000, 2000, 3000, 3000]);
  harness.encodeGrid();
  const grid = await harness.read('grid');
  expect(grid.result.selectedCount).toBe(0);
  expect(grid.result.valueCount).toBe(0);
  expect(grid.result.candidatesTruncated).toBe(false);
  expect(grid.ids).toEqual([]);
  expect(grid.mask?.every(word => word === 0)).toBe(true);
  expect(Array.from(grid.result.histogram)).toEqual([0, 0, 0, 0]);
  harness.destroy();
});

it('GPURegionStatistics grid index flags candidate overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const data = createRandomData(11, 20000, true);
  const harness = new GridStatisticsHarness({
    device,
    id: 'grid-overflow',
    selectionKind: 'rectangle',
    positions: data.positions,
    values: data.values,
    candidateCapacity: 100,
    binCount: 4,
    withOutput: true,
    indexMode: 'same-graph'
  });
  harness.setRectangle([0, 0, 1000, 1000]);
  harness.encodeBoth();
  const brute = await harness.read('brute');
  const grid = await harness.read('grid');
  expect(grid.result.candidatesTruncated).toBe(true);
  expect(grid.words[WORD.flags] & CANDIDATES_TRUNCATED).toBe(CANDIDATES_TRUNCATED);
  expect(grid.result.selectedCount).toBeLessThanOrEqual(100);
  expect(grid.result.selectedCount).toBeLessThan(brute.result.selectedCount);
  expect(grid.outputOverflow).toBe(1);
  // A small region fits in the same capacity (a few cells plus margin) and clears the flag on the next encoding.
  harness.setRectangle([100, 100, 105, 105]);
  harness.encodeBoth();
  const smallBrute = await harness.read('brute');
  const smallGrid = await harness.read('grid');
  expect(smallBrute.result.selectedCount).toBeLessThanOrEqual(100);
  expectIdentical(smallBrute, smallGrid, true, 'small region after overflow');
  harness.destroy();
});

it('GPURegionStatistics grid index flags rows missing from the index', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const data = createRandomData(13, 8000, true);
  // Three rows outside the index domain, one with a NaN position.
  data.positions.set([2500, 100], 10);
  data.positions.set([-300, 40], 20);
  data.positions.set([4000, 4000], 30);
  data.positions.set([Number.NaN, 5], 40);
  const harness = new GridStatisticsHarness({
    device,
    id: 'grid-missing',
    selectionKind: 'rectangle',
    positions: data.positions,
    values: data.values,
    candidateCapacity: 8000,
    binCount: 4,
    withOutput: true,
    indexMode: 'separate-graph'
  });
  harness.buildIndex();
  expect(await harness.readIndexCount()).toBe(8000 - 4);
  // The region contains the out-of-domain rows, which only the brute-force path can find.
  harness.setRectangle([-1000, -1000, 5000, 5000]);
  harness.encodeBoth();
  const brute = await harness.read('brute');
  const grid = await harness.read('grid');
  expect(grid.result.candidatesTruncated).toBe(true);
  expect(brute.result.candidatesTruncated).toBe(false);
  expect(brute.result.selectedCount).toBe(8000 - 1);
  expect(grid.result.selectedCount).toBe(8000 - 4);
  harness.destroy();
});

it('GPURegionStatistics grid index keeps the same row order for IDs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const data = createRandomData(17, 10000, true);
  const harness = new GridStatisticsHarness({
    device,
    id: 'grid-order',
    selectionKind: 'radius',
    positions: data.positions,
    values: data.values,
    sourceIds: data.sourceIds,
    candidateCapacity: 10000,
    withOutput: true,
    indexMode: 'same-graph'
  });
  harness.setCircle([500, 500, 250]);
  harness.encodeGrid();
  const grid = await harness.read('grid');
  expect(grid.ids?.length).toBeGreaterThan(100);
  expect(grid.ids).toEqual([...(grid.ids ?? [])].sort((left, right) => left - right));
  harness.destroy();
});
