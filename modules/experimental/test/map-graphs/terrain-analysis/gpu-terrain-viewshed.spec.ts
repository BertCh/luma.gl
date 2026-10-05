// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPURasterTextureToBuffer} from '../../../src/gpu-raster';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  GPU_TERRAIN_VISIBILITY as V,
  GPUTerrainViewshed,
  getGPUTerrainViewshedParameterValues,
  getGPUTerrainVisibilityToleranceParameterValues,
  type GPUTerrainSightLineTraversal,
  type GPUTerrainViewshedSettings,
  type GPUTerrainVisibilityToleranceSettings
} from '../../../src/map-graphs/terrain-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  isSoftwareDevice,
  readUint32
} from '../map-graph-test-utils';
import {
  computeTerrainViewshed,
  computeTerrainViewshedWithTolerance,
  createFractalTerrain
} from './terrain-analysis-oracle';

const WIDTH = 9;
const HEIGHT = 7;
const PIXEL_COUNT = WIDTH * HEIGHT;
const WALL = Float32Array.from({length: PIXEL_COUNT}, (_, index) => (index % WIDTH === 5 ? 10 : 0));
const BASE_SETTINGS: GPUTerrainViewshedSettings = {
  observer: [2, 3],
  observerHeight: 2,
  targetHeight: 0,
  cellSize: [1, 1]
};

function createViewshedFixture(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  settings: GPUTerrainViewshedSettings,
  mask?: Uint32Array
) {
  const pixelCount = width * height;
  const elevationBuffer = createInputBuffer(device, elevation);
  const maskBuffer = mask ? createInputBuffer(device, mask) : undefined;
  const visibilityBuffer = createOutputBuffer(device, pixelCount);
  const settingsBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'viewshed-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainViewshedParameterValues(settings)
  });
  const graph = new GPUCommandGraph(device, {id: 'terrain-viewshed-test'});
  graph.add(
    new GPUTerrainViewshed({
      width,
      height,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
        },
        validity: maskBuffer
          ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', pixelCount)
          : undefined
      },
      settings: settingsBuffer.importToGraph(graph),
      visibility: importGraphBuffer(graph, 'visibility', visibilityBuffer, 'uint32', pixelCount)
    })
  );
  const compiled = graph.compile();
  return {
    compiled,
    settingsBuffer,
    visibilityBuffer,
    destroy: () => {
      compiled.destroy();
      settingsBuffer.destroy();
      for (const buffer of [
        elevationBuffer,
        visibilityBuffer,
        ...(maskBuffer ? [maskBuffer] : [])
      ]) {
        buffer.destroy();
      }
    }
  };
}

it('GPUTerrainViewshed hides cells behind a wall and follows a per-frame observer', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createViewshedFixture(device, WALL, WIDTH, HEIGHT, BASE_SETTINGS);
  const run = async (settings: GPUTerrainViewshedSettings) => {
    fixture.settingsBuffer.write(getGPUTerrainViewshedParameterValues(settings));
    submitGraph(device, fixture.compiled, undefined);
    return readUint32(fixture.visibilityBuffer, PIXEL_COUNT);
  };

  let visibility = await run(BASE_SETTINGS);
  if (!isSoftwareDevice(device)) {
    expect(visibility).toEqual(
      computeTerrainViewshed(
        WALL,
        undefined,
        WIDTH,
        HEIGHT,
        getGPUTerrainViewshedParameterValues(BASE_SETTINGS)
      )
    );
  }
  const row3 = visibility.slice(3 * WIDTH, 4 * WIDTH);
  expect(row3.slice(0, 6)).toEqual(new Array(6).fill(V.visible));
  expect(row3.slice(6)).toEqual(new Array(3).fill(V.hidden));

  visibility = await run({...BASE_SETTINGS, maxDistance: 3.5});
  expect(visibility[3 * WIDTH + 6]).toBe(V.outOfRange);
  expect(visibility[3 * WIDTH + 5]).toBe(V.visible);

  visibility = await run({...BASE_SETTINGS, observer: [7, 3]});
  expect(visibility.slice(3 * WIDTH, 3 * WIDTH + 5)).toEqual(new Array(5).fill(V.hidden));
  expect(visibility[3 * WIDTH + 5]).toBe(V.visible);

  visibility = await run({...BASE_SETTINGS, observerHeight: 1000});
  expect(visibility.every(code => code === V.visible)).toBe(true);

  visibility = await run({...BASE_SETTINGS, observer: [-1, 3]});
  expect(visibility.every(code => code === V.noData)).toBe(true);
  fixture.destroy();
});

it('GPUTerrainViewshed skips invalid cells and applies curvature', async () => {
  const device = await getWebGPUTestDevice();
  if (!device || isSoftwareDevice(device)) {
    return;
  }
  const mask = new Uint32Array(PIXEL_COUNT).fill(1);
  mask[3 * WIDTH + 5] = 0;
  const masked = createViewshedFixture(device, WALL, WIDTH, HEIGHT, BASE_SETTINGS, mask);
  submitGraph(device, masked.compiled, undefined);
  const visibility = await readUint32(masked.visibilityBuffer, PIXEL_COUNT);
  expect(visibility[3 * WIDTH + 5]).toBe(V.noData);
  expect(visibility).toEqual(
    computeTerrainViewshed(
      WALL,
      mask,
      WIDTH,
      HEIGHT,
      getGPUTerrainViewshedParameterValues(BASE_SETTINGS)
    )
  );
  masked.destroy();

  const curvatureSettings: GPUTerrainViewshedSettings = {
    observer: [0, 0],
    observerHeight: 1,
    cellSize: [1, 1],
    curvatureCoefficient: 0.25
  };
  const flat = new Float32Array(8);
  const curved = createViewshedFixture(device, flat, 8, 1, curvatureSettings);
  submitGraph(device, curved.compiled, undefined);
  const curvedVisibility = await readUint32(curved.visibilityBuffer, 8);
  expect(curvedVisibility).toEqual(
    computeTerrainViewshed(
      flat,
      undefined,
      8,
      1,
      getGPUTerrainViewshedParameterValues(curvatureSettings)
    )
  );
  expect(curvedVisibility.slice(1, 3)).toEqual([V.visible, V.visible]);
  expect(curvedVisibility.slice(5)).toEqual([V.hidden, V.hidden, V.hidden]);
  curved.destroy();
});

it('GPUTerrainViewshed calibrates uint32 elevation and writes a visibility texture', async () => {
  const device = await getWebGPUTestDevice();
  if (!device || !device.getTextureFormatCapabilities('r32uint').store) {
    return;
  }
  const elevationBuffer = createInputBuffer(
    device,
    Uint32Array.from(WALL, value => value * 2)
  );
  const texture = device.createTexture({
    format: 'r32uint',
    width: WIDTH,
    height: HEIGHT,
    usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_DST
  });
  const readbackBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const readbackValidityBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const settings = new GPUMapGraphParameterBuffer(device, {
    id: 'texture-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainViewshedParameterValues(BASE_SETTINGS)
  });
  const graph = new GPUCommandGraph(device, {id: 'terrain-viewshed-texture'});
  const textureView = graph.createTextureView(
    graph.importTexture(
      {
        id: 'visibility-texture',
        format: 'r32uint',
        width: WIDTH,
        height: HEIGHT,
        usage: texture.props.usage
      },
      texture
    ),
    {mipLevelCount: 1}
  ) as never;
  graph.add(
    new GPUTerrainViewshed({
      width: WIDTH,
      height: HEIGHT,
      elevation: {
        id: 'elevation',
        format: 'uint32',
        scale: 0.5,
        offset: 0,
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'uint32', PIXEL_COUNT)
        }
      },
      settings: settings.importToGraph(graph),
      visibilityTexture: textureView
    })
  );
  new GPURasterTextureToBuffer({
    id: 'visibility-readback',
    input: {id: 'visibility-band', format: 'uint32', storage: {kind: 'texture', view: textureView}},
    output: importGraphBuffer(graph, 'readback', readbackBuffer, 'uint32', PIXEL_COUNT),
    outputValidity: importGraphBuffer(
      graph,
      'readback-validity',
      readbackValidityBuffer,
      'uint32',
      PIXEL_COUNT
    )
  }).addToGraph(graph);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const visibility = await readUint32(readbackBuffer, PIXEL_COUNT);
  const row3 = visibility.slice(3 * WIDTH, 4 * WIDTH);
  expect(row3.slice(0, 6)).toEqual(new Array(6).fill(V.visible));
  expect(row3.slice(6)).toEqual(new Array(3).fill(V.hidden));
  compiled.destroy();
  settings.destroy();
  for (const resource of [elevationBuffer, texture, readbackBuffer, readbackValidityBuffer]) {
    resource.destroy();
  }
});

type ViewshedOptions = {
  traversal?: GPUTerrainSightLineTraversal;
  tolerance?: GPUTerrainVisibilityToleranceSettings;
};

/** Runs one viewshed and returns its codes; the graph is rebuilt per call (topology varies). */
async function runViewshed(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  settings: GPUTerrainViewshedSettings,
  options: ViewshedOptions = {}
): Promise<number[]> {
  const pixelCount = width * height;
  const elevationBuffer = createInputBuffer(device, elevation);
  const visibilityBuffer = createOutputBuffer(device, pixelCount);
  const settingsBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'viewshed-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainViewshedParameterValues(settings)
  });
  const toleranceBuffer = options.tolerance
    ? new GPUMapGraphParameterBuffer(device, {
        id: 'viewshed-tolerance',
        format: 'float32',
        length: 4,
        values: getGPUTerrainVisibilityToleranceParameterValues(options.tolerance)
      })
    : undefined;
  const graph = new GPUCommandGraph(device, {id: 'terrain-viewshed-variant'});
  graph.add(
    new GPUTerrainViewshed({
      width,
      height,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
        }
      },
      settings: settingsBuffer.importToGraph(graph),
      traversal: options.traversal,
      tolerance: toleranceBuffer?.importToGraph(graph),
      visibility: importGraphBuffer(graph, 'visibility', visibilityBuffer, 'uint32', pixelCount)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const codes = await readUint32(visibilityBuffer, pixelCount);
  compiled.destroy();
  settingsBuffer.destroy();
  toleranceBuffer?.destroy();
  elevationBuffer.destroy();
  visibilityBuffer.destroy();
  return codes;
}

const FRACTAL_WIDTH = 96;
const FRACTAL_HEIGHT = 80;

it('GPUTerrainViewshed pyramid traversal is bit-identical to march', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const terrain = createFractalTerrain(FRACTAL_WIDTH, FRACTAL_HEIGHT, 7, 55);
  const observers: [number, number][] = [
    [48, 40],
    [10.5, 70.25],
    [90.75, 5.5],
    [0, 0]
  ];
  const toleranceCases: (GPUTerrainVisibilityToleranceSettings | undefined)[] = [
    undefined,
    {toleranceMeters: 2, tolerancePerKilometer: 1, targetIgnoreDistance: 60, targetIgnoreFraction: 0.02}
  ];
  let hiddenShare = 0;
  let marginalCount = 0;
  for (const observer of observers) {
    for (const curvatureCoefficient of [0, 3e-5]) {
      for (const tolerance of toleranceCases) {
        const settings: GPUTerrainViewshedSettings = {
          observer,
          observerHeight: 1.7,
          cellSize: [30, 30],
          curvatureCoefficient
        };
        const march = await runViewshed(device, terrain, FRACTAL_WIDTH, FRACTAL_HEIGHT, settings, {
          traversal: 'march',
          tolerance
        });
        const pyramid = await runViewshed(device, terrain, FRACTAL_WIDTH, FRACTAL_HEIGHT, settings, {
          traversal: 'pyramid',
          tolerance
        });
        expect(pyramid).toEqual(march);
        hiddenShare += march.filter(code => code === V.hidden).length / march.length;
        marginalCount += march.filter(code => code === V.marginal).length;
        if (!tolerance && curvatureCoefficient === 0 && !isSoftwareDevice(device)) {
          expect(pyramid).toEqual(
            computeTerrainViewshed(
              terrain,
              undefined,
              FRACTAL_WIDTH,
              FRACTAL_HEIGHT,
              getGPUTerrainViewshedParameterValues(settings)
            )
          );
          // The legacy kernel (no new props) agrees with the new march kernel.
          expect(await runViewshed(device, terrain, FRACTAL_WIDTH, FRACTAL_HEIGHT, settings)).toEqual(
            march
          );
        }
      }
    }
  }
  // Non-trivial structure: a mix of hidden and visible cells, and some marginal cells.
  const meanHidden = hiddenShare / (observers.length * 4);
  expect(meanHidden).toBeGreaterThan(0.1);
  expect(meanHidden).toBeLessThan(0.9);
  expect(marginalCount).toBeGreaterThan(0);
});

it('GPUTerrainViewshed tolerance bands classify marginal cells and ignore the last stretch', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // 12 x 1 flat strip; observer at column 0 (eye 1). A wall of 0.5 at column 6 sticks 0.0167 slope
  // units (0.167 m at the target) above the sight line to column 10.
  const strip = new Float32Array(12);
  strip[6] = 0.5;
  const settings: GPUTerrainViewshedSettings = {observer: [0, 0], observerHeight: 1, cellSize: [1, 1]};
  for (const traversal of ['march', 'pyramid'] as const) {
    const exact = await runViewshed(device, strip, 12, 1, settings, {traversal});
    expect(exact[10]).toBe(V.hidden);
    const wide = await runViewshed(device, strip, 12, 1, settings, {
      traversal,
      tolerance: {toleranceMeters: 0.5}
    });
    expect(wide[10]).toBe(V.marginal);
    const narrow = await runViewshed(device, strip, 12, 1, settings, {
      traversal,
      tolerance: {toleranceMeters: 0.05}
    });
    expect(narrow[10]).toBe(V.hidden);
    // Cells clearly in front of the wall stay visible, and cells behind it are not visible.
    expect(wide[3]).toBe(V.visible);
    expect(wide[11]).not.toBe(V.visible);

    // A target just behind its own lip: the lip at column 9 is ignored by the last stretch.
    const lip = new Float32Array(12);
    lip[9] = 5;
    expect((await runViewshed(device, lip, 12, 1, settings, {traversal}))[10]).toBe(V.hidden);
    const ignored = await runViewshed(device, lip, 12, 1, settings, {
      traversal,
      tolerance: {targetIgnoreDistance: 1.5}
    });
    expect(ignored[10]).toBe(V.visible);
    const fractionIgnored = await runViewshed(device, lip, 12, 1, settings, {
      traversal,
      tolerance: {targetIgnoreFraction: 0.15}
    });
    expect(fractionIgnored[10]).toBe(V.visible);
  }
});

it('GPUTerrainViewshed with tolerance matches the float64 oracle away from knife edges', async () => {
  const device = await getWebGPUTestDevice();
  if (!device || isSoftwareDevice(device)) {
    return;
  }
  const terrain = createFractalTerrain(FRACTAL_WIDTH, FRACTAL_HEIGHT, 11, 90);
  const settings: GPUTerrainViewshedSettings = {
    observer: [30.5, 44.25],
    observerHeight: 1.7,
    cellSize: [30, 30],
    curvatureCoefficient: 3e-5
  };
  const tolerance = {toleranceMeters: 2, tolerancePerKilometer: 1, targetIgnoreDistance: 90};
  const expected = computeTerrainViewshedWithTolerance(
    terrain,
    undefined,
    FRACTAL_WIDTH,
    FRACTAL_HEIGHT,
    getGPUTerrainViewshedParameterValues(settings),
    getGPUTerrainVisibilityToleranceParameterValues(tolerance)
  );
  for (const traversal of ['march', 'pyramid'] as const) {
    const actual = await runViewshed(device, terrain, FRACTAL_WIDTH, FRACTAL_HEIGHT, settings, {
      traversal,
      tolerance
    });
    let mismatches = 0;
    for (const [index, code] of actual.entries()) {
      if (code !== expected[index].code) {
        mismatches++;
        const {maxSlope, targetSlope, band} = expected[index];
        const edge = Math.min(
          Math.abs(maxSlope - (targetSlope + band)),
          Math.abs(maxSlope - (targetSlope - band))
        );
        expect(edge).toBeLessThan(1e-4);
      }
    }
    expect(mismatches / actual.length).toBeLessThan(0.005);
    expect(actual.some(code => code === V.hidden)).toBe(true);
    expect(actual.some(code => code === V.visible)).toBe(true);
  }
});
