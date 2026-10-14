// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import derivativesFixture from '../oracles/fixtures/terrain-derivatives.json';
import ruggednessFixture from '../oracles/fixtures/terrain-ruggedness.json';
import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {
  GPUTerrainDerivatives,
  getGPUTerrainDerivativesParameterValues
} from '../../src/gpu-terrain/terrain-analysis';
import {GPUTerrainRuggedness} from '../../src/gpu-terrain/topographic-position/gpu-terrain-ruggedness';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

/**
 * Runs the GPU contributors on the `gdaldem` fixtures written by
 * `test/oracles/terrain-derivatives.py` and compares them with the reference within the recorded
 * tolerances (float32 kernel against GDAL). The same fixtures cross-validate the CPU oracles in
 * `terrain-reference-oracle.node.spec.ts`; the convention mapping is in each fixture's
 * `conventionMap`.
 */

type Nullable = (number | null)[];

type DerivativeFixture = {
  cases: {
    name: string;
    width: number;
    height: number;
    cellSize: [number, number];
    elevation: Nullable;
    tolerances: Record<string, {absolute: number; minimumGradient?: number}>;
    variants: {
      name: string;
      output: 'slope' | 'aspect' | 'hillshade';
      ours: {
        slopeUnits?: 'degrees' | 'percent';
        zFactor: number;
        cellSizeOverride?: [number, number];
        azimuthDegrees?: number;
        altitudeDegrees?: number;
      };
      toleranceKey: string;
      values: Nullable;
    }[];
  }[];
};

type RuggednessFixture = {
  cases: {
    name: string;
    width: number;
    height: number;
    elevation: Nullable;
    tolerance: {absolute: number};
    variants: {
      name: string;
      output: 'tpi' | 'tri-riley' | 'tri-wilson' | 'roughness';
      edgeMode: 'nodata' | 'extrapolate';
      values: Nullable;
    }[];
  }[];
};

// Imported, not read with node:fs, so the browser-based headless project can load them.
const FIXTURES: Record<string, unknown> = {
  'terrain-derivatives.json': derivativesFixture,
  'terrain-ruggedness.json': ruggednessFixture
};

function loadFixture<T>(name: string): T {
  return FIXTURES[name] as T;
}

function toFloat32(values: Nullable): Float32Array {
  return Float32Array.from(values, value => value ?? Number.NaN);
}

/** Asserts nodata agreement everywhere and returns the compared (defined) pixel count. */
function compareToReference(
  actual: ArrayLike<number>,
  reference: Nullable,
  tolerance: number,
  label: string,
  isSkipped: (index: number) => boolean = () => false,
  isAngle = false
): number {
  let compared = 0;
  for (const [index, expected] of reference.entries()) {
    if (expected === null) {
      continue;
    }
    expect(Number.isNaN(actual[index]), `${label} pixel ${index} should be defined`).toBe(false);
    if (isSkipped(index)) {
      continue;
    }
    let difference = Math.abs(actual[index] - expected);
    if (isAngle) {
      difference = Math.min(difference, 360 - difference);
    }
    expect(difference, `${label} pixel ${index}`).toBeLessThan(tolerance);
    compared++;
  }
  return compared;
}

async function runDerivatives(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  settingsValues: Float32Array,
  slopeUnits: 'degrees' | 'percent'
): Promise<{slope: number[]; aspect: number[]; hillshade: number[]}> {
  const pixelCount = width * height;
  const graph = new GPUCommandGraph(device, {id: 'terrain-reference-derivatives'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const outputs: Record<'slope' | 'aspect' | 'hillshade', Buffer> = {
    slope: createOutputBuffer(device, pixelCount),
    aspect: createOutputBuffer(device, pixelCount),
    hillshade: createOutputBuffer(device, pixelCount)
  };
  const settings = new GPUParameterBuffer(device, {
    id: 'reference-derivative-settings',
    format: 'float32',
    length: 8,
    values: settingsValues
  });
  graph.add(
    new GPUTerrainDerivatives({
      grid: makeGPURasterMetadata({width, height}),
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
        }
      },
      parameters: settings.importToGraph(graph),
      output: {
        slope: importGraphBuffer(graph, 'slope', outputs.slope, 'float32', pixelCount),
        aspect: importGraphBuffer(graph, 'aspect', outputs.aspect, 'float32', pixelCount),
        hillshade: importGraphBuffer(graph, 'hillshade', outputs.hillshade, 'float32', pixelCount)
      },
      slopeUnits,
      borderMode: 'nodata'
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    slope: await readFloat32(outputs.slope, pixelCount),
    aspect: await readFloat32(outputs.aspect, pixelCount),
    hillshade: await readFloat32(outputs.hillshade, pixelCount)
  };
  compiled.destroy();
  settings.destroy();
  elevationBuffer.destroy();
  for (const buffer of Object.values(outputs)) buffer.destroy();
  return result;
}

async function runRuggedness(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  edgeMode: 'nodata' | 'extrapolate',
  algorithm: 'riley' | 'wilson'
): Promise<{tpi: number[]; tri: number[]; roughness: number[]}> {
  const pixelCount = width * height;
  const graph = new GPUCommandGraph(device, {id: 'terrain-reference-ruggedness'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const outputs: Record<'tpi' | 'tri' | 'roughness', Buffer> = {
    tpi: createOutputBuffer(device, pixelCount),
    tri: createOutputBuffer(device, pixelCount),
    roughness: createOutputBuffer(device, pixelCount)
  };
  graph.add(
    new GPUTerrainRuggedness({
      grid: makeGPURasterMetadata({width, height}),
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
        }
      },
      output: {
        topographicPositionIndex: importGraphBuffer(
          graph,
          'tpi',
          outputs.tpi,
          'float32',
          pixelCount
        ),
        terrainRuggednessIndex: importGraphBuffer(graph, 'tri', outputs.tri, 'float32', pixelCount),
        roughness: importGraphBuffer(graph, 'roughness', outputs.roughness, 'float32', pixelCount)
      },
      terrainRuggednessAlgorithm: algorithm,
      edgeMode
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    tpi: await readFloat32(outputs.tpi, pixelCount),
    tri: await readFloat32(outputs.tri, pixelCount),
    roughness: await readFloat32(outputs.roughness, pixelCount)
  };
  compiled.destroy();
  elevationBuffer.destroy();
  for (const buffer of Object.values(outputs)) buffer.destroy();
  return result;
}

it('GPUTerrainDerivatives matches gdaldem slope, aspect and hillshade fixtures', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = loadFixture<DerivativeFixture>('terrain-derivatives.json');
  for (const testCase of fixture.cases) {
    const elevation = toFloat32(testCase.elevation);
    for (const variant of testCase.variants) {
      const label = `${testCase.name}/${variant.name}`;
      const cellSize = variant.ours.cellSizeOverride ?? testCase.cellSize;
      const settingsValues = getGPUTerrainDerivativesParameterValues(
        {
          zFactor: variant.ours.zFactor,
          azimuthDegrees: variant.ours.azimuthDegrees,
          altitudeDegrees: variant.ours.altitudeDegrees
        },
        makeGPURasterMetadata({width: testCase.width, height: testCase.height, cellSize})
      );
      const result = await runDerivatives(
        device,
        elevation,
        testCase.width,
        testCase.height,
        settingsValues,
        variant.ours.slopeUnits ?? 'degrees'
      );
      const tolerance = testCase.tolerances[variant.toleranceKey];
      if (variant.output === 'aspect') {
        // Skip ill-conditioned aspects: tan(slope) below the recorded minimum gradient.
        const slope = (
          await runDerivatives(
            device,
            elevation,
            testCase.width,
            testCase.height,
            settingsValues,
            'degrees'
          )
        ).slope;
        const minimumGradient = tolerance.minimumGradient ?? 0;
        const compared = compareToReference(
          result.aspect,
          variant.values,
          tolerance.absolute,
          label,
          index => Math.tan((slope[index] * Math.PI) / 180) < minimumGradient,
          true
        );
        expect(compared, `${label} compared pixels`).toBeGreaterThan(30);
        // Flat cells are nodata in GDAL and -1 in ours; nodata cells are NaN.
        for (const [index, expected] of variant.values.entries()) {
          if (expected === null) {
            expect(result.aspect[index] === -1 || Number.isNaN(result.aspect[index])).toBe(true);
          }
        }
        continue;
      }
      const compared = compareToReference(
        result[variant.output],
        variant.values,
        tolerance.absolute,
        label
      );
      expect(compared, `${label} compared pixels`).toBeGreaterThan(30);
      for (const [index, expected] of variant.values.entries()) {
        if (expected === null) {
          expect(Number.isNaN(result[variant.output][index]), `${label} nodata ${index}`).toBe(
            true
          );
        }
      }
    }
  }
});

it('GPUTerrainRuggedness matches gdaldem TPI, TRI and roughness fixtures', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = loadFixture<RuggednessFixture>('terrain-ruggedness.json');
  for (const testCase of fixture.cases) {
    const elevation = toFloat32(testCase.elevation);
    for (const edgeMode of ['nodata', 'extrapolate'] as const) {
      for (const algorithm of ['riley', 'wilson'] as const) {
        const result = await runRuggedness(
          device,
          elevation,
          testCase.width,
          testCase.height,
          edgeMode,
          algorithm
        );
        const outputs = {
          tpi: result.tpi,
          [`tri-${algorithm}`]: result.tri,
          roughness: result.roughness
        } as Record<string, number[]>;
        for (const variant of testCase.variants.filter(
          entry => entry.edgeMode === edgeMode && entry.output in outputs
        )) {
          const label = `${testCase.name}/${variant.name}`;
          const compared = compareToReference(
            outputs[variant.output],
            variant.values,
            testCase.tolerance.absolute,
            label
          );
          expect(compared, `${label} compared pixels`).toBeGreaterThan(30);
          for (const [index, expected] of variant.values.entries()) {
            if (expected === null) {
              expect(Number.isNaN(outputs[variant.output][index]), `${label} nodata ${index}`).toBe(
                true
              );
            }
          }
        }
      }
    }
  }
});
