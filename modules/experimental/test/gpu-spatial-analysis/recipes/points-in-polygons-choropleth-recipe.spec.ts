// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues,
  packGPUColor
} from '../../../src/gpu-dataframe/column-classification/index';
import {
  addPointsInPolygonsChoroplethRecipe,
  type GPUChoroplethStatistic
} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe';
import {createSeededRandom} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {RecipeTestFixture, isClose} from './recipe-harness';

type Ring = [number, number][];

/** Six rectangles in a 3 x 2 grid; the last one has a square hole. */
function createPolygons() {
  const features: Ring[][] = [];
  for (let row = 0; row < 2; row++) {
    for (let column = 0; column < 3; column++) {
      const x = column * 10;
      const y = row * 10;
      const shell: Ring = [
        [x, y],
        [x + 10, y],
        [x + 10, y + 10],
        [x, y + 10]
      ];
      features.push(
        row === 1 && column === 2
          ? [
              shell,
              [
                [x + 3, y + 3],
                [x + 7, y + 3],
                [x + 7, y + 7],
                [x + 3, y + 7]
              ]
            ]
          : [shell]
      );
    }
  }
  const positions: number[] = [];
  const ringOffsets = [0];
  const polygonOffsets = [0];
  const featureOffsets = [0];
  for (const rings of features) {
    for (const ring of rings) {
      for (const [x, y] of ring) {
        positions.push(x, y);
      }
      ringOffsets.push(positions.length / 2);
    }
    polygonOffsets.push(ringOffsets.length - 1);
    featureOffsets.push(polygonOffsets.length - 1);
  }
  return {
    features,
    positions: Float32Array.from(positions),
    ringOffsets: Uint32Array.from(ringOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    featureOffsets: Uint32Array.from(featureOffsets)
  };
}

function isInside(rings: Ring[], x: number, y: number): boolean {
  let inside = false;
  for (const ring of rings) {
    for (let index = 0; index < ring.length; index++) {
      const [ax, ay] = ring[index];
      const [bx, by] = ring[(index + 1) % ring.length];
      if (ay > y !== by > y && x < ax + ((y - ay) / (by - ay)) * (bx - ax)) {
        inside = !inside;
      }
    }
  }
  return inside;
}

function createScene() {
  const random = createSeededRandom(21);
  const points = new Float32Array(1200);
  const values = new Float32Array(600);
  for (let index = 0; index < 600; index++) {
    // Slightly beyond the polygon extent, so some points are in no feature.
    points[2 * index] = -2 + random() * 34;
    points[2 * index + 1] = -2 + random() * 24;
    values[index] = 1 + random() * 9 + (points[2 * index] > 20 ? 20 : 0);
  }
  return {points, values};
}

for (const backend of ['zonal', 'group'] as const) {
  for (const statistic of ['count', 'mean', 'sum', 'maximum'] as GPUChoroplethStatistic[]) {
    if (backend === 'group' && statistic === 'sum') {
      // Covered by the zonal run; the group sum is fixed point and compared below by mean.
    }
    it(`addPointsInPolygonsChoroplethRecipe ${backend} ${statistic} matches the CPU join`, async () => {
      const device = await getWebGPUTestDevice();
      if (!device) {
        return;
      }
      const polygons = createPolygons();
      const {points, values} = createScene();
      const featureCount = polygons.features.length;
      // CPU oracle.
      const expectedCounts = new Array<number>(featureCount).fill(0);
      const sums = new Array<number>(featureCount).fill(0);
      const maxima = new Array<number>(featureCount).fill(NaN);
      for (let index = 0; index < values.length; index++) {
        const feature = polygons.features.findIndex(rings =>
          isInside(rings, points[2 * index], points[2 * index + 1])
        );
        if (feature >= 0) {
          expectedCounts[feature]++;
          sums[feature] += values[index];
          maxima[feature] = Number.isNaN(maxima[feature])
            ? values[index]
            : Math.max(maxima[feature], values[index]);
        }
      }
      const expectedValues = expectedCounts.map((count, feature) =>
        statistic === 'count'
          ? count
          : statistic === 'sum'
            ? sums[feature]
            : statistic === 'mean'
              ? sums[feature] / count
              : maxima[feature]
      );

      const fixture = new RecipeTestFixture(device, `choropleth-${backend}-${statistic}`);
      try {
        const counts = fixture.output('counts', 'uint32', featureCount);
        const featureValues = fixture.output('feature-values', 'float32', featureCount);
        const overflow = fixture.output('overflow', 'uint32', 1);
        const breaks = fixture.output('breaks', 'float32', 4);
        const classCount = fixture.output('class-count', 'uint32', 1);
        const colors = fixture.output('colors', 'uint32', featureCount);
        const classIndices = fixture.output('class-indices', 'uint32', featureCount);
        const palette = [packGPUColor(0, 0, 255), packGPUColor(0, 255, 0), packGPUColor(255, 0, 0)];
        const recipe = addPointsInPolygonsChoroplethRecipe(fixture.graph, {
          backend,
          statistic,
          points: fixture.input('points', points, 'float32x2', 600),
          values: fixture.input('values', values, 'float32', 600),
          polygons: {
            polygonPositions: fixture.input(
              'polygon-positions',
              polygons.positions,
              'float32x2',
              polygons.positions.length / 2
            ),
            featureOffsets: fixture.input(
              'feature-offsets',
              polygons.featureOffsets,
              'uint32',
              polygons.featureOffsets.length
            ),
            polygonOffsets: fixture.input(
              'polygon-offsets',
              polygons.polygonOffsets,
              'uint32',
              polygons.polygonOffsets.length
            ),
            ringOffsets: fixture.input(
              'ring-offsets',
              polygons.ringOffsets,
              'uint32',
              polygons.ringOffsets.length
            ),
            candidateCapacity: 4096
          },
          outputs: {
            counts: counts.view,
            ...(statistic === 'count' ? {} : {featureValues: featureValues.view}),
            overflow: overflow.view,
            color: {
              breaks: breaks.view,
              classCount: classCount.view,
              colors: colors.view,
              classIndices: classIndices.view
            }
          },
          color: {
            classBreaksParameters: fixture.parameters(
              'class-breaks-parameters',
              'float32',
              getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 3}, 3)
            ),
            maximumClassCount: 3,
            methods: ['equal-interval'],
            colorScaleParameters: fixture.parameters(
              'color-scale-parameters',
              'float32',
              getGPUColorScaleParameterValues({scale: 'quantile', domainCount: 4, paletteCount: 3})
            ),
            palette: fixture.input('palette', Uint32Array.from(palette), 'uint32', 3),
            maximumPaletteCount: 3
          }
        });
        expect(recipe.contributors.length).toBe(backend === 'zonal' ? 3 : 4);
        expect(recipe.outputs.counts).toBe(recipe.counts);
        expect(recipe.status.stages[0].status.overflow).toBe(recipe.overflow);
        fixture.run();

        expect((await fixture.readUint32(overflow, 1))[0]).toBe(0);
        const gpuCounts = await fixture.readUint32(counts, featureCount);
        expect(gpuCounts).toEqual(expectedCounts);
        expect(Math.min(...gpuCounts)).toBeGreaterThan(0);
        // For 'count' the value column is the uint32 counts view itself, with no cast.
        const gpuValues =
          statistic === 'count'
            ? Array.from(gpuCounts)
            : await fixture.readFloat32(featureValues, featureCount);
        for (let feature = 0; feature < featureCount; feature++) {
          expect(
            isClose(gpuValues[feature], expectedValues[feature], 1e-3, 1e-4),
            `${statistic} feature ${feature}: ${gpuValues[feature]} vs ${expectedValues[feature]}`
          ).toBe(true);
        }
        const gpuBreaks = await fixture.readFloat32(breaks, 4);
        const minimum = Math.min(...expectedValues);
        const maximum = Math.max(...expectedValues);
        expect(maximum).toBeGreaterThan(minimum);
        expect(isClose(gpuBreaks[0], minimum, 1e-2, 1e-4)).toBe(true);
        expect(isClose(gpuBreaks[3], maximum, 1e-2, 1e-4)).toBe(true);
        const gpuClasses = await fixture.readUint32(classIndices, featureCount);
        const gpuColors = await fixture.readUint32(colors, featureCount);
        for (let feature = 0; feature < featureCount; feature++) {
          const expectedClass =
            (gpuBreaks[1] <= gpuValues[feature] ? 1 : 0) +
            (gpuBreaks[2] <= gpuValues[feature] ? 1 : 0);
          expect(gpuClasses[feature]).toBe(expectedClass);
          expect(gpuColors[feature]).toBe(palette[expectedClass]);
        }
        expect(new Set(gpuClasses).size).toBeGreaterThan(1);
      } finally {
        fixture.destroy();
      }
    }, 120000);
  }
}
