// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {validateGPUComputationRecord, validateGPUComputationSidecars} from '@luma.gl/gpgpu';
import {expect, test} from 'vitest';
import {getLensScenario} from '../../../src/computation-lenses/registry';
import data from '../../../src/computation-lenses/data/k-core.json';
import '../../../src/computation-lenses/scenarios/gpu-core/index';

const scenarioIds = [
  'scan',
  'scan.segmented',
  'scan.anatomy',
  'flag-offsets',
  'compaction',
  'mask'
];

test.each(scenarioIds)('gpu-core.%s matches a CPU oracle after one real GPU run', async name => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const scenario = await getLensScenario(`gpu-core.${name}`)!.load();
  const record = await scenario.run(device);
  expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
  expect(
    validateGPUComputationSidecars(record, {captures: record.captures, metrics: record.metrics})
  ).toMatchObject({valid: true});
  const snapshots = record.captures!.snapshots;
  const final = (captureName: string) =>
    snapshots.filter(capture => capture.name === captureName).at(-1)!;
  const output = (captureName: string) => Array.from(final(captureName).values);
  const rainfall = data.stations.prcpMm.map(Math.round);
  const wet = data.stations.prcpMm.map(value => Number(value >= 100));
  const high = data.stations.elevationM.map(value => Number(value >= 500));
  const selected = wet.map((flag, index) => flag & high[index]);
  if (name === 'scan' || name === 'scan.segmented') {
    let total = 0;
    const starts = new Set(data.stations.stateOffsets.slice(0, -1));
    const prefix = rainfall.map((value, index) => {
      if (name === 'scan.segmented' && starts.has(index)) total = 0;
      total = (total + value) >>> 0;
      return total;
    });
    const first = Number(/first:(\d+)/.exec(final('rainTotals').meaning ?? '')?.[1] ?? 0);
    expect(output('rainTotals')).toEqual(prefix.slice(first, first + 16));
    expect(output('rainfall')).toEqual(rainfall.slice(first, first + 16));
    const frames = snapshots.filter(capture => capture.name === 'rainTotals');
    expect(frames.map(frame => frame.iteration)).toEqual([0, 1, 2]);
    expect(frames.slice(0, -1).every(frame => frame.role === undefined)).toBe(true);
    if (name === 'scan') expect(final('rainTotals').highlight?.label).toContain(String(prefix[9]));
    else {
      expect(output('stateStarts')).toEqual(
        Array.from({length: 16}, (_, index) => Number(starts.has(first + index)))
      );
      expect(snapshots.every(capture => capture.meaning?.includes(`first:${first}`))).toBe(true);
    }
  } else if (name === 'flag-offsets') {
    let count = 0;
    const offsets = wet.map(flag => {
      const offset = count;
      count += flag;
      return offset;
    });
    expect(output('wetOffsets')).toEqual(offsets.slice(0, 16));
    expect(output('wetCount')).toEqual([count]);
  } else if (name === 'compaction') {
    const stationRows = data.stations.row.filter((_, index) => wet[index]);
    expect(output('wetStations')).toEqual(stationRows.slice(0, 16));
    expect(output('wetCount')).toEqual([stationRows.length]);
  } else if (name === 'mask') {
    expect(output('heavyRainHighStations')).toEqual(selected.slice(0, 16));
  } else {
    const stationRows = data.stations.row.filter((_, index) => selected[index]);
    expect(output('ids')).toEqual(stationRows.slice(0, 16));
    expect(output('count')).toEqual([stationRows.length]);
    const total = data.stations.prcpMm.reduce(
      (sum, value, index) => sum + (selected[index] ? value : 0),
      0
    );
    expect(output('totalRain')[0]).toBeCloseTo(total, 2);
    expect(record.metrics![0].series?.map(series => series.name)).toEqual([
      'slices@512',
      'slices@2048',
      'slices@8192'
    ]);
    for (const series of record.metrics![0].series ?? []) {
      expect(series.values).toHaveLength(record.graphs[0].nodes.length);
      expect(series.values.every(Number.isInteger)).toBe(true);
    }
    expect(record.metrics![0].encoding.retained).toEqual([]);
  }
  expect(snapshots.every(capture => capture.provenance === 'gpu-readback')).toBe(true);
  expect(JSON.stringify(record).length).toBeLessThan(60 * 1024);
});

test('gpu-core K-3 scenarios match CPU oracles after one real GPU run', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const stations = data.stations;
  const run = async (name: string) => {
    const record = await (await getLensScenario(`gpu-core.${name}`)!.load()).run(device);
    expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
    expect(
      validateGPUComputationSidecars(record, {captures: record.captures, metrics: record.metrics})
    ).toMatchObject({valid: true});
    const snapshots = record.captures!.snapshots;
    expect(snapshots.every(capture => capture.provenance === 'gpu-readback')).toBe(true);
    return (captureName: string) =>
      Array.from(snapshots.filter(capture => capture.name === captureName).at(-1)!.values);
  };
  const first = 62;
  const lastOfState = (state: number) => stations.stateOffsets[state + 1];

  expect((await run('reduction'))('maxRainfall')[0]).toBeCloseTo(Math.max(...stations.prcpMm), 2);

  const segmentedTotals = (await run('segmented-reduction'))('stateTotals');
  const totals = stations.stateOffsets
    .slice(0, -1)
    .map((start, state) =>
      stations.prcpMm.slice(start, lastOfState(state)).reduce((a, b) => a + b, 0)
    );
  segmentedTotals.forEach((value, index) => expect(value).toBeCloseTo(totals[2 + index], 1));

  const tenths = stations.prcpMm.map(value => Math.round(value * 10));
  const ranked = Array.from(tenths).sort((a, b) => b - a);
  expect((await run('sort'))('rankedRainfall')).toEqual(ranked.slice(0, 16));

  const segmentedSort = (await run('segmented-sort'))('sortedRainfall');
  // The captured window starts at state 2 and may run into later states; each sorts on its own.
  const sortedWithinStates = stations.stateOffsets
    .slice(0, -1)
    .flatMap((start, state) => tenths.slice(start, lastOfState(state)).sort((a, b) => b - a));
  expect(segmentedSort).toEqual(sortedWithinStates.slice(first, first + segmentedSort.length));

  const gathered = (await run('gather'))('statePrevDayRainfall');
  // The gather scenario picks West Virginia, the last state.
  const westVirginia = stations.states.indexOf('WV');
  expect(gathered).toEqual(
    Array.from(
      stations.prcpPrevDayMm.slice(stations.stateOffsets[westVirginia], lastOfState(westVirginia)),
      Math.fround
    )
  );

  const bins = Array.from({length: 16}, () => 0);
  for (const value of stations.prcpMm) {
    if (value >= 0 && value <= 320) bins[Math.min(15, Math.floor(value / 20))]++;
  }
  expect((await run('histogram'))('stationsPerBin')).toEqual(bins);

  for (const name of ['elementwise', 'elementwise.id-probe']) {
    const sums = (await run(name))('twoDayRainfall');
    sums.forEach((value, index) =>
      expect(value).toBeCloseTo(stations.prcpMm[index] + stations.prcpPrevDayMm[index], 2)
    );
  }
});

test('gpu-core K-4 grid, group and hash scenarios match CPU oracles after one real GPU run', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const stations = data.stations;
  const run = async (name: string) => {
    const record = await (await getLensScenario(`gpu-core.${name}`)!.load()).run(device);
    expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
    expect(
      validateGPUComputationSidecars(record, {captures: record.captures, metrics: record.metrics})
    ).toMatchObject({valid: true});
    const snapshots = record.captures!.snapshots;
    expect(snapshots.every(capture => capture.provenance === 'gpu-readback')).toBe(true);
    expect(JSON.stringify(record).length).toBeLessThan(512 * 1024);
    return (captureName: string) =>
      Array.from(snapshots.filter(capture => capture.name === captureName).at(-1)!.values);
  };
  // Mirrors getGridCoordinate for non-crossing bounds in float32 arithmetic.
  const coordinate = (value: number, minimum: number, maximum: number, size: number) =>
    Math.min(
      Math.trunc(
        Math.fround(
          Math.fround(Math.fround(Math.fround(value) - minimum) / (maximum - minimum)) * size
        )
      ),
      size - 1
    );
  const cells = stations.lon.map(
    (lon, index) => coordinate(lon, -92, -74, 18) + 18 * coordinate(stations.lat[index], 24, 40, 16)
  );
  const counts = Array.from({length: 288}, () => 0);
  const totals = Array.from({length: 288}, () => 0);
  cells.forEach((cell, index) => {
    counts[cell]++;
    totals[cell] += stations.prcpMm[index];
  });

  const offsets = (await run('grid-index'))('cellOffsets');
  expect(offsets).toHaveLength(289);
  expect(offsets.slice(1).map((end, cell) => end - offsets[cell])).toEqual(counts);

  expect((await run('grid-binning'))('stationsPerCell')).toEqual(counts);

  (await run('grid-aggregation'))('rainfallPerCell').forEach((value, cell) =>
    expect(value).toBeCloseTo(totals[cell], 1)
  );

  const perState = await run('group-aggregation');
  const stateCount = stations.states.length;
  const expectedCounts = Array.from({length: stateCount}, () => 0);
  const expectedTotals = Array.from({length: stateCount}, () => 0);
  stations.state.forEach((state, index) => {
    expectedCounts[state]++;
    expectedTotals[state] += stations.prcpMm[index];
  });
  expect(perState('stationCounts')).toEqual(expectedCounts);
  perState('rainTotals').forEach((value, state) =>
    expect(value).toBeCloseTo(expectedTotals[state], 1)
  );
  perState('rainMeans').forEach((value, state) =>
    expect(value).toBeCloseTo(expectedTotals[state] / expectedCounts[state], 2)
  );

  const hashed = await run('hash-index');
  const empty = 0xffffffff;
  const window = hashed('tableKeys');
  const slots = hashed('slotIds');
  const values = hashed('tableValues');
  expect(window).toHaveLength(16);
  window.forEach((key, index) => {
    if (key === empty) return;
    expect(stations.row[values[index]]).toBe(key);
    expect(slots[index]).toBe(slots[0] + index);
  });
  expect(hashed('statistics').slice(0, 4)).toEqual([1024, 0, 0, 0]);
  expect(hashed('maxProbes')[0]).toBeGreaterThanOrEqual(1);

  const joined = await run('hash-join');
  const wetIndices = stations.prcpMm.flatMap((value, index) => (value >= 100 ? [index] : []));
  expect(joined('pairCount')).toEqual([wetIndices.length]);
  expect(joined('wetPositions')).toEqual(Array.from({length: 16}, (_, index) => index));
  expect(joined('wetStationIndices')).toEqual(wetIndices.slice(0, 16));
});

test('gpu-core K-5 scenarios match CPU oracles after one real GPU run', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const {dem} = data;
  const width = dem.width;
  const elevation = (row: number, column: number) => dem.elevationM[row * width + column];
  const run = async (name: string) => {
    const record = await (await getLensScenario(`gpu-core.${name}`)!.load()).run(device);
    expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
    expect(
      validateGPUComputationSidecars(record, {captures: record.captures, metrics: record.metrics})
    ).toMatchObject({valid: true});
    // Every record stays within the 512 KB budget.
    expect(JSON.stringify(record).length).toBeLessThan(512 * 1024);
    const snapshots = record.captures!.snapshots;
    expect(snapshots.every(capture => capture.provenance === 'gpu-readback')).toBe(true);
    return {
      record,
      final: (captureName: string) =>
        Array.from(snapshots.filter(capture => capture.name === captureName).at(-1)!.values)
    };
  };

  const block = (row: number, column: number) => elevation(24 + row, 24 + column);
  const matVec = await run('mat-vec');
  matVec
    .final('rowMeans')
    .forEach((mean, row) =>
      expect(mean).toBeCloseTo(
        Array.from({length: 8}, (_, column) => block(row, column)).reduce((a, b) => a + b) / 8,
        2
      )
    );

  const matMul = await run('mat-mul');
  matMul.final('eastWestSteps').forEach((step, index) => {
    const row = Math.floor(index / 8);
    const column = index % 8;
    expect(step).toBeCloseTo(column < 7 ? block(row, column + 1) - block(row, column) : 0, 2);
  });

  const convolution = await run('convolution');
  const curvature = convolution.final('curvature');
  expect(curvature).toHaveLength(width * dem.height);
  for (let row = 0; row < dem.height; row++) {
    for (let column = 0; column < width; column++) {
      // Zero boundary: neighbours outside the tile count as 0 m.
      const at = (r: number, c: number) =>
        r < 0 || r >= dem.height || c < 0 || c >= width ? 0 : elevation(r, c);
      const expected =
        at(row - 1, column) +
        at(row + 1, column) +
        at(row, column - 1) +
        at(row, column + 1) -
        4 * at(row, column);
      expect(curvature[row * width + column]).toBeCloseTo(expected, 1);
    }
  }

  const spectrum1d = await run('fft1d');
  const profile = Array.from({length: width}, (_, column) => elevation(32, column));
  const mean = profile.reduce((a, b) => a + b) / width;
  const output1d = spectrum1d.final('spectrum');
  expect(output1d).toHaveLength(width);
  for (let k = 0; k < width / 2; k++) {
    let real = 0;
    let imaginary = 0;
    profile.forEach((value, j) => {
      real += (value - mean) * Math.cos((2 * Math.PI * j * k) / width);
      imaginary -= (value - mean) * Math.sin((2 * Math.PI * j * k) / width);
    });
    expect(output1d[k * 2]).toBeCloseTo(real, -1);
    expect(output1d[k * 2 + 1]).toBeCloseTo(imaginary, -1);
  }

  const spectrum2d = await run('fft2d');
  const output2d = spectrum2d.final('spectrum');
  expect(output2d).toHaveLength(width * dem.height * 2);
  // Separable DFT: rows first, then columns, in float64.
  const rowReal = new Float64Array(width * dem.height);
  const rowImaginary = new Float64Array(width * dem.height);
  for (let row = 0; row < dem.height; row++) {
    for (let u = 0; u < width; u++) {
      for (let column = 0; column < width; column++) {
        const angle = (2 * Math.PI * u * column) / width;
        rowReal[row * width + u] += elevation(row, column) * Math.cos(angle);
        rowImaginary[row * width + u] -= elevation(row, column) * Math.sin(angle);
      }
    }
  }
  const tolerance = 2e-6 * Math.hypot(output2d[0], output2d[1]);
  for (let u = 0; u < width; u += 7) {
    for (let v = 0; v < dem.height; v += 5) {
      let real = 0;
      let imaginary = 0;
      for (let row = 0; row < dem.height; row++) {
        const angle = (2 * Math.PI * v * row) / dem.height;
        const cosine = Math.cos(angle);
        const sine = -Math.sin(angle);
        real += rowReal[row * width + u] * cosine - rowImaginary[row * width + u] * sine;
        imaginary += rowReal[row * width + u] * sine + rowImaginary[row * width + u] * cosine;
      }
      const index = (v * width + u) * 2;
      expect(Math.abs(output2d[index] - real)).toBeLessThan(tolerance);
      expect(Math.abs(output2d[index + 1] - imaginary)).toBeLessThan(tolerance);
    }
  }

  const solver = await run('conjugate-gradient-program');
  const side = 16;
  const rhs = Array.from({length: side * side}, (_, index) =>
    elevation(24 + Math.floor(index / side), 24 + (index % side))
  );
  const neighbours = (index: number) => {
    const row = Math.floor(index / side);
    const column = index % side;
    return [
      ...(row > 0 ? [index - side] : []),
      ...(column > 0 ? [index - 1] : []),
      ...(column < side - 1 ? [index + 1] : []),
      ...(row < side - 1 ? [index + side] : [])
    ];
  };
  const apply = (x: number[]) =>
    x.map(
      (value, index) =>
        (1 + 4 * neighbours(index).length) * value -
        4 * neighbours(index).reduce((sum, other) => sum + x[other], 0)
    );
  const dot = (a: number[], b: number[]) =>
    a.reduce((sum, value, index) => sum + value * b[index], 0);
  const series = solver.record.metrics![0].series!.find(item => item.name === 'residual')!;
  expect(series.axis).toBe('iteration');
  // CPU oracle: the same bounded CG in float64, stopping at the same squared-residual tolerance.
  let x = rhs.map(() => 0);
  let residual = [...rhs];
  let search = [...rhs];
  let squared = dot(residual, residual);
  const oracleTrace = [Math.sqrt(squared)];
  for (let iteration = 0; iteration < 12 && squared > 35000; iteration++) {
    const q = apply(search);
    const alpha = squared / dot(search, q);
    x = x.map((value, index) => value + alpha * search[index]);
    residual = residual.map((value, index) => value - alpha * q[index]);
    const next = dot(residual, residual);
    search = residual.map((value, index) => value + (next / squared) * search[index]);
    squared = next;
    oracleTrace.push(Math.sqrt(squared));
  }
  // float32 may stop one iteration away from float64 at the tolerance boundary.
  expect(Math.abs(series.values.length - oracleTrace.length)).toBeLessThanOrEqual(1);
  expect(series.values[0]).toBeCloseTo(oracleTrace[0], -1);
  series.values
    .slice(0, 4)
    .forEach((value, index) => expect(value).toBeCloseTo(oracleTrace[index], -1));
  solver
    .final('smoothed')
    .forEach((value, index) => expect(Math.abs(value - x[index])).toBeLessThan(3));
  expect(solver.final('elevation')).toEqual(rhs);
  const frames = solver.record.captures!.snapshots.filter(capture => capture.name === 'smoothed');
  expect(frames.map(frame => frame.iteration)).toEqual(
    [...frames.map(frame => frame.iteration)].sort((a, b) => a! - b!)
  );
  expect(frames[0].iteration).toBe(0);
  expect(frames.at(-1)!.iteration).toBe(series.values.length - 1);
  // The receipt reads the scalar arena, residual, and search direction across iterations.
  expect(solver.record.metrics![0].encoding.retained).toEqual([
    'gpu-value-arena-buffer',
    'smooth-r',
    'smooth-p'
  ]);
});
