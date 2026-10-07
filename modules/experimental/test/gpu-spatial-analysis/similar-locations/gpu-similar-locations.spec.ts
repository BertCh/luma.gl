// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUSimilarLocationsParameterLength,
  getGPUSimilarLocationsParameterValues,
  GPUSimilarLocations,
  GPU_SIMILAR_LOCATIONS_NO_RANK,
  type GPUSimilarLocationsStandardization
} from '../../../src/gpu-spatial-analysis/similar-locations/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {SIMILAR_LOCATIONS_REFERENCE as REFERENCE} from './similar-locations-reference';

/** f32 statistics and distances against float64: relative 1e-4 of the largest distance. */
const DISTANCE_TOLERANCE = 1e-4;
const RESULT_COUNT = 8;

it('GPUSimilarLocations matches the NumPy/SciPy reference and changes parameters without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {rowCount, attributeCount} = REFERENCE;
  for (const standardization of ['zscore', 'rank'] as GPUSimilarLocationsStandardization[]) {
    const attributes = createInputBuffer(device, REFERENCE.attributes);
    const mask = createInputBuffer(device, REFERENCE.mask);
    const selection = createOutputBuffer(device, rowCount);
    const ranksOut = createOutputBuffer(device, rowCount);
    const distancesOut = createOutputBuffer(device, rowCount);
    const topOut = createOutputBuffer(device, RESULT_COUNT);
    const countOut = createOutputBuffer(device, 1);
    const parameterBuffer = new GPUParameterBuffer(device, {
      id: 'similar-parameters',
      format: 'float32',
      length: getGPUSimilarLocationsParameterLength(attributeCount)
    });
    const graph = new GPUCommandGraph(device, {id: 'similar-graph'});
    graph.add(
      new GPUSimilarLocations({
        id: 'similar',
        attributes: importGraphBuffer(
          graph,
          'attributes',
          attributes,
          'float32',
          rowCount * attributeCount
        ),
        attributeCount,
        mask: importGraphBuffer(graph, 'mask', mask, 'uint32', rowCount),
        selection: importGraphBuffer(graph, 'selection', selection, 'uint32', rowCount),
        parameters: parameterBuffer.importToGraph(graph),
        standardization,
        maximumResultCount: RESULT_COUNT,
        output: {
          ranks: importGraphBuffer(graph, 'ranks', ranksOut, 'uint32', rowCount),
          distances: importGraphBuffer(graph, 'distances', distancesOut, 'float32', rowCount),
          topIds: importGraphBuffer(graph, 'top', topOut, 'uint32', RESULT_COUNT),
          count: importGraphBuffer(graph, 'count', countOut, 'uint32', 1)
        }
      })
    );
    const compiled = graph.compile();
    for (const reference of REFERENCE.cases.filter(c => c.standardization === standardization)) {
      selection.write(reference.selection);
      parameterBuffer.write(
        getGPUSimilarLocationsParameterValues(
          {resultCount: RESULT_COUNT, direction: reference.direction, weights: reference.weights},
          attributeCount
        )
      );
      submitGraph(device, compiled, undefined);
      const distances = await readFloat32(distancesOut, rowCount);
      const ranks = await readUint32(ranksOut, rowCount);
      const top = await readUint32(topOut, RESULT_COUNT);
      const count = (await readUint32(countOut, 1))[0];
      const label = `${standardization}: ${reference.name}`;
      let scale = 1e-9;
      for (const distance of reference.distances) {
        if (Number.isFinite(distance)) {
          scale = Math.max(scale, distance);
        }
      }
      for (let row = 0; row < rowCount; row++) {
        const expected = reference.distances[row];
        if (Number.isNaN(expected)) {
          expect(Number.isNaN(distances[row]), `${label} row ${row} unranked`).toBe(true);
          expect(ranks[row]).toBe(GPU_SIMILAR_LOCATIONS_NO_RANK);
        } else {
          expect(Math.abs(distances[row] - expected), `${label} row ${row}`).toBeLessThanOrEqual(
            DISTANCE_TOLERANCE * scale
          );
        }
      }
      // Ranks are a permutation ordered by (GPU distance, id): ties keep the lowest ID, and the
      // order agrees with the reference except for pairs inside f32 rounding of each other.
      const gpuOrder = Array.from({length: rowCount}, (_, row) => row)
        .filter(row => ranks[row] !== GPU_SIMILAR_LOCATIONS_NO_RANK)
        .sort((left, right) => ranks[left] - ranks[right]);
      expect(gpuOrder.length, label).toBe(reference.order.length);
      expect(gpuOrder.map((_, position) => ranks[gpuOrder[position]])).toEqual(
        gpuOrder.map((_, position) => position)
      );
      const sign = reference.direction === 'most' ? 1 : -1;
      for (let position = 1; position < gpuOrder.length; position++) {
        const previous = gpuOrder[position - 1];
        const current = gpuOrder[position];
        const gap = sign * (distances[current] - distances[previous]);
        expect(gap >= 0, `${label} monotone at ${position}`).toBe(true);
        if (gap === 0) {
          expect(current > previous, `${label} tie order at ${position}`).toBe(true);
        }
      }
      const referenceTop = reference.order.slice(0, RESULT_COUNT);
      expect(count, label).toBe(Math.min(RESULT_COUNT, reference.order.length));
      for (let position = 0; position < count; position++) {
        const gpuRow = top[position];
        expect(gpuRow, `${label} top ${position}`).toBe(gpuOrder[position]);
        // Rank swaps against float64 only happen between distances closer than the tolerance.
        expect(
          Math.abs(reference.distances[gpuRow] - reference.distances[referenceTop[position]]),
          `${label} top ${position} vs reference`
        ).toBeLessThanOrEqual(DISTANCE_TOLERANCE * scale);
      }
      expect(top.slice(count).every(id => id === GPU_SIMILAR_LOCATIONS_NO_RANK)).toBe(true);
    }
    compiled.destroy();
    parameterBuffer.destroy();
    for (const buffer of [attributes, mask, selection, ranksOut, distancesOut, topOut, countOut]) {
      buffer.destroy();
    }
  }
});

it('GPUSimilarLocations standardizes large tied inputs with the column reductions and the rank sort', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // 5000 rows: not a multiple of the 256-lane reduction, above the rank sort threshold, with heavy
  // ties, a signed zero and invalid rows.
  const rowCount = 5000;
  const attributeCount = 3;
  const attributes = new Float32Array(rowCount * attributeCount);
  for (let row = 0; row < rowCount; row++) {
    attributes[row * 3] = (row * 7) % 11;
    attributes[row * 3 + 1] = row % 2 === 0 ? 0 : -0;
    attributes[row * 3 + 2] = Math.sin(row * 0.37) * 5 + (row % 5);
  }
  attributes[3 * 17 + 2] = Number.NaN;
  attributes[3 * 4096 + 0] = Number.POSITIVE_INFINITY;
  const selectedRow = 123;
  const selectionValues = new Uint32Array(rowCount);
  selectionValues[selectedRow] = 1;
  const valid = Array.from({length: rowCount}, (_, row) =>
    [0, 1, 2].every(channel => Number.isFinite(attributes[row * 3 + channel]))
  );
  for (const standardization of ['zscore', 'rank'] as GPUSimilarLocationsStandardization[]) {
    const standardized = new Float64Array(rowCount * attributeCount);
    for (let channel = 0; channel < attributeCount; channel++) {
      const column = Array.from({length: rowCount}, (_, row) => attributes[row * 3 + channel]);
      const validColumn = column.filter((_, row) => valid[row]);
      const mean = validColumn.reduce((sum, value) => sum + value, 0) / validColumn.length;
      const deviation = Math.sqrt(
        validColumn.reduce((sum, value) => sum + (value - mean) ** 2, 0) / validColumn.length
      );
      const sorted = [...validColumn].sort((left, right) => left - right);
      for (let row = 0; row < rowCount; row++) {
        if (standardization === 'zscore') {
          standardized[row * 3 + channel] = deviation > 0 ? (column[row] - mean) / deviation : 0;
        } else {
          // Average-tie percentile rank by binary search on the sorted column.
          let low = 0;
          let high = sorted.length;
          while (low < high) {
            const middle = (low + high) >> 1;
            if (sorted[middle] < column[row]) low = middle + 1;
            else high = middle;
          }
          const below = low;
          high = sorted.length;
          while (low < high) {
            const middle = (low + high) >> 1;
            if (sorted[middle] <= column[row]) low = middle + 1;
            else high = middle;
          }
          standardized[row * 3 + channel] =
            (below + 0.5 * (low - below - 1)) / Math.max(sorted.length - 1, 1);
        }
      }
    }
    const expected = new Float64Array(rowCount);
    for (let row = 0; row < rowCount; row++) {
      let sum = 0;
      for (let channel = 0; channel < attributeCount; channel++) {
        sum += (standardized[row * 3 + channel] - standardized[selectedRow * 3 + channel]) ** 2;
      }
      expected[row] = Math.sqrt(sum);
    }
    const attributeBuffer = createInputBuffer(device, attributes);
    const selection = createInputBuffer(device, selectionValues);
    const ranksOut = createOutputBuffer(device, rowCount);
    const distancesOut = createOutputBuffer(device, rowCount);
    const topOut = createOutputBuffer(device, RESULT_COUNT);
    const countOut = createOutputBuffer(device, 1);
    const parameterBuffer = new GPUParameterBuffer(device, {
      id: 'similar-large-parameters',
      format: 'float32',
      length: getGPUSimilarLocationsParameterLength(attributeCount)
    });
    parameterBuffer.write(
      getGPUSimilarLocationsParameterValues({resultCount: RESULT_COUNT}, attributeCount)
    );
    const graph = new GPUCommandGraph(device, {id: 'similar-large-graph'});
    graph.add(
      new GPUSimilarLocations({
        id: 'similar-large',
        attributes: importGraphBuffer(
          graph,
          'attributes',
          attributeBuffer,
          'float32',
          rowCount * attributeCount
        ),
        attributeCount,
        selection: importGraphBuffer(graph, 'selection', selection, 'uint32', rowCount),
        parameters: parameterBuffer.importToGraph(graph),
        standardization,
        maximumResultCount: RESULT_COUNT,
        output: {
          ranks: importGraphBuffer(graph, 'ranks', ranksOut, 'uint32', rowCount),
          distances: importGraphBuffer(graph, 'distances', distancesOut, 'float32', rowCount),
          topIds: importGraphBuffer(graph, 'top', topOut, 'uint32', RESULT_COUNT),
          count: importGraphBuffer(graph, 'count', countOut, 'uint32', 1)
        }
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const distances = await readFloat32(distancesOut, rowCount);
    const ranks = await readUint32(ranksOut, rowCount);
    for (let row = 0; row < rowCount; row++) {
      if (!valid[row] || row === selectedRow) {
        expect(ranks[row], `${standardization} row ${row} unranked`).toBe(
          GPU_SIMILAR_LOCATIONS_NO_RANK
        );
      } else {
        expect(
          Math.abs(distances[row] - expected[row]),
          `${standardization} row ${row}`
        ).toBeLessThanOrEqual(1e-4 * Math.max(1, expected[row]));
      }
    }
    compiled.destroy();
    parameterBuffer.destroy();
    for (const buffer of [attributeBuffer, selection, ranksOut, distancesOut, topOut, countOut]) {
      buffer.destroy();
    }
  }
});
