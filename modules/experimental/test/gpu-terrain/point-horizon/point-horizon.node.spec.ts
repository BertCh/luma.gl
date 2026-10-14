// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPU_POINT_HORIZON_PROFILE_PARAMETER_LENGTH,
  GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH,
  getGPUPointHorizonDistanceLattice,
  getGPUPointHorizonProfileParameterValues,
  getGPUPointHorizonSegments,
  getGPUPointHorizonVisibilityParameterValues,
  GPUPointHorizonProfile,
  GPUPointHorizonVisibility,
  type GPUPointHorizonProfileProps,
  type GPUPointHorizonVisibilityProps
} from '../../../src/gpu-terrain/point-horizon';
import {
  getPointHorizonMarchWGSL,
  resolvePointHorizonModel
} from '../../../src/gpu-terrain/point-horizon/point-horizon-march';
import {
  POINT_HORIZON_PROFILE_PROJECTION_SLOT,
  POINT_HORIZON_VISIBILITY_PROJECTION_SLOT
} from '../../../src/gpu-terrain/point-horizon/point-horizon-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {makeMercatorTestGrid} from './point-horizon-oracle';

function createBand(graph: GPUCommandGraph, id: string, length: number) {
  return {
    id,
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: createTransientView(graph, id, 'float32', length)}
  };
}

function checkPointHorizonVerticalDatums(
  create: (
    overrides: Partial<
      Pick<
        GPUPointHorizonProfileProps,
        'id' | 'elevation' | 'heightReference' | 'observerVerticalDatum'
      >
    >
  ) => unknown,
  elevation: GPUPointHorizonProfileProps['elevation']
) {
  const id = 'datum-horizon';
  const declaredElevation = {...elevation, verticalDatum: 'EPSG:5773'};
  expect(() =>
    create({
      id,
      elevation: declaredElevation,
      heightReference: 'absolute',
      observerVerticalDatum: 'ellipsoidal'
    })
  ).toThrow(
    new Error(
      `${id} observer heights are ellipsoidal but elevation is EPSG:5773; convert to one vertical datum first`
    )
  );
  for (const heightReference of [undefined, 'ground'] as const) {
    expect(() =>
      create({
        id,
        elevation: declaredElevation,
        heightReference,
        observerVerticalDatum: 'EPSG:5773'
      })
    ).toThrow(new Error(`${id} observerVerticalDatum needs heightReference absolute`));
  }
  for (const [observerVerticalDatum, verticalDatum] of [
    [undefined, undefined],
    [undefined, 'EPSG:5773'],
    ['ellipsoidal', undefined],
    ['epsg:5773', 'EPSG:5773'],
    [' EPSG:5773 ', 'EPSG:5773']
  ]) {
    expect(() =>
      create({
        id,
        elevation: {...elevation, verticalDatum},
        heightReference: 'absolute',
        observerVerticalDatum
      })
    ).not.toThrow();
  }
}

it('getGPUPointHorizonDistanceLattice builds the exact power-of-two lattice', () => {
  const lattice = getGPUPointHorizonDistanceLattice({maximumDistance: 150_000, cellSize: 30});
  // 20 m is exactly on the lattice (octave 4, step 0.25: j = 16).
  expect(lattice.firstDistance).toBe(20);
  expect(lattice.octaves[0]).toEqual({
    exponent: 4,
    base: 16,
    step: 0.25,
    firstJ: 16,
    endJ: 64,
    firstIndex: 0
  });
  const steps = Object.fromEntries(lattice.octaves.map(octave => [octave.exponent, octave.step]));
  // near field: clamp(0.01 d, 0.25, 15); far field: 3.5e-4 d, quantized down to a power of two.
  expect(steps[4]).toBe(0.25);
  expect(steps[10]).toBe(8); // 10.24 -> 8
  expect(steps[11]).toBe(8); // min(20.48, 15) = 15 -> 8
  expect(steps[17]).toBe(32); // 3.5e-4 * 131072 = 45.9 -> 32
  expect(lattice.octaves.at(-1)?.exponent).toBe(17);
  // Every distance is exact in float32, strictly increasing, and equals base + j * step.
  let previous = 0;
  for (let index = 0; index < lattice.sampleCount; index++) {
    const distance = lattice.getDistance(index);
    expect(Math.fround(distance)).toBe(distance);
    expect(distance).toBeGreaterThan(previous);
    previous = distance;
  }
  expect(lattice.lastDistance).toBeLessThanOrEqual(150_000);
  expect(lattice.sampleCount).toBe(
    lattice.octaves.reduce((sum, octave) => sum + octave.endJ - octave.firstJ, 0)
  );
  // Ceil / floor index agree with the distances.
  expect(lattice.getCeilIndex(20)).toBe(0);
  expect(lattice.getCeilIndex(20.01)).toBe(1);
  expect(lattice.getFloorIndex(20.01)).toBe(0);
  expect(lattice.getFloorIndex(19.9)).toBe(-1);
  expect(lattice.getCeilIndex(1e9)).toBe(lattice.sampleCount);
  expect(lattice.getFloorIndex(1e9)).toBe(lattice.sampleCount - 1);
  expect(lattice.getDistance(lattice.getCeilIndex(5000.1))).toBeGreaterThanOrEqual(5000.1);
  expect(() => lattice.getDistance(lattice.sampleCount)).toThrow(RangeError);

  const short = getGPUPointHorizonDistanceLattice({
    minimumDistance: 20,
    maximumDistance: 40,
    cellSize: 10
  });
  expect(short.lastDistance).toBe(40);
  expect(short.octaves.map(octave => octave.exponent)).toEqual([4, 5]);
  expect(() => getGPUPointHorizonDistanceLattice({maximumDistance: 10, cellSize: 10})).toThrow(
    /maximumDistance/
  );
  expect(() => getGPUPointHorizonDistanceLattice({maximumDistance: 100, cellSize: 0})).toThrow(
    /cellSize/
  );
  expect(() =>
    getGPUPointHorizonDistanceLattice({minimumDistance: 0.5, maximumDistance: 100, cellSize: 10})
  ).toThrow(/minimumDistance/);
});

it('getGPUPointHorizonSegments breaks the great circle at observer-independent distances', () => {
  const lattice = getGPUPointHorizonDistanceLattice({maximumDistance: 30_000, cellSize: 20});
  const segments = getGPUPointHorizonSegments(lattice);
  expect(segments.distances[0]).toBe(20);
  expect(segments.distances.at(-1)).toBe(30_000);
  expect(segments.endIndices.length).toBe(segments.distances.length - 1);
  expect(segments.endIndices.at(-1)).toBe(lattice.sampleCount);
  for (let index = 1; index < segments.endIndices.length; index++) {
    expect(segments.endIndices[index]).toBeGreaterThanOrEqual(segments.endIndices[index - 1]);
  }
  const radius = 6371008.8;
  const last = segments.distances.length - 1;
  expect(segments.sines[last]).toBe(Math.fround(Math.sin(30_000 / radius)));
  expect(segments.oneMinusCosines[last]).toBeCloseTo(1 - Math.cos(30_000 / radius), 12);
  // A lower latitude bound gives a longer chord and fewer segments.
  expect(getGPUPointHorizonSegments(lattice, {maximumLatitude: 10}).distances.length).toBeLessThan(
    segments.distances.length
  );
  expect(() => getGPUPointHorizonSegments(lattice, {maximumLatitude: 100})).toThrow(
    /maximumLatitude/
  );
});

it('point horizon packers lay out settings', () => {
  const planarGrid = (cellSize: number | readonly [number, number]) =>
    makeGPURasterMetadata({width: 16, height: 16, cellSize});
  expect(GPU_POINT_HORIZON_PROFILE_PARAMETER_LENGTH).toBe(8);
  expect(GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH).toBe(12);
  expect(
    Array.from(
      getGPUPointHorizonProfileParameterValues({curvatureCoefficient: 0.5}, planarGrid([2, 3]))
    )
  ).toEqual([0.5, 2, 3, 0, 0, 0, 0, 1]); // slot 7: planar projection tag
  const originY = 1000;
  const worldPixelSize = 512 * 2 ** 10;
  // The Web Mercator words are derived from the EPSG:3857 grid: pixels per 360 degrees from the
  // cell size, the Mercator ordinate of the top edge from the north edge.
  const mercator = getGPUPointHorizonProfileParameterValues(
    {projection: 'web-mercator', maximumDistance: 5000},
    makeMercatorTestGrid(16, 16, worldPixelSize, originY)
  );
  expect(mercator[0]).toBe(0);
  expect(mercator[1]).toBe(Math.fround(worldPixelSize));
  expect(mercator[2]).toBe(Math.fround(Math.PI * (1 - (2 * originY) / worldPixelSize)));
  expect(mercator[3]).toBe(5000);
  expect(mercator[4]).toBe(0); // the opaque-zero guard
  expect(mercator[7]).toBe(2); // Web Mercator projection tag
  const visibility = getGPUPointHorizonVisibilityParameterValues(
    {sigmaZ: 7, targetIgnoreFraction: 0.1},
    planarGrid(10)
  );
  expect(Array.from(visibility)).toEqual([
    0,
    10,
    10,
    0,
    0,
    Math.fround(0.02),
    7,
    Math.fround(0.05),
    -1, // derived default: one ground pixel along the ray at the target (review C-05)
    Math.fround(0.1),
    0,
    1 // planar projection tag
  ]);
  const reused = new Float32Array(12).fill(9);
  getGPUPointHorizonProfileParameterValues({}, planarGrid(1), reused);
  expect(Array.from(reused.slice(0, 8))).toEqual([0, 1, 1, 0, 0, 0, 0, 1]);
  expect(() => getGPUPointHorizonProfileParameterValues({}, planarGrid([0, 1]))).toThrow(
    /cell size/
  );
  // A Web Mercator grid needs square pixels and rows increasing south.
  expect(() =>
    getGPUPointHorizonProfileParameterValues({projection: 'web-mercator'}, planarGrid([10, 20]))
  ).toThrow(/square pixels/);
  expect(() =>
    getGPUPointHorizonProfileParameterValues(
      {projection: 'web-mercator'},
      makeGPURasterMetadata({width: 16, height: 16, cellSize: 10, rowDirection: 'north'})
    )
  ).toThrow(/rows increasing south/);
  expect(() =>
    getGPUPointHorizonProfileParameterValues({}, planarGrid(1), new Float32Array(7))
  ).toThrow(/must hold 8 values/);
  expect(() =>
    getGPUPointHorizonVisibilityParameterValues({}, planarGrid(1), new Float32Array(11))
  ).toThrow(/must hold 12 values/);
});

it('GPUPointHorizonProfile schedules canonicalization, pyramid levels and march chunks', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUPointHorizonProfileProps> = {}) => {
    instance++;
    return new GPUPointHorizonProfile({
      grid: makeGPURasterMetadata({width: 64, height: 48, cellSize: 10}),
      elevation: createBand(graph, `terrain-${instance}`, 64 * 48),
      observers: createTransientView(graph, `observers-${instance}`, 'float32x4', 3),
      parameters: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      output: {
        tangent: createTransientView(graph, `tangent-${instance}`, 'float32', 3 * 720),
        skylineAngle: createTransientView(graph, `skyline-angle-${instance}`, 'float32', 3 * 720),
        distance: createTransientView(graph, `distance-${instance}`, 'float32', 3 * 720)
      },
      maximumDistance: 1000,
      cellSize: 10,
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.id).toBe('point-horizon-profile');
  checkPointHorizonVerticalDatums(create, createBand(graph, 'datum-elevation', 64 * 48));
  const nodes = contributor.getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  const levelCount = contributor.model.layout?.levels.length ?? 0;
  expect(levelCount).toBe(5); // 64 x 48 with 4 px blocks: 16x12, 8x6, 4x3, 2x2, 1x1
  // The pyramid builder may fuse its coarse levels into fewer nodes; only its presence matters.
  expect(ids[0]).toBe('point-horizon-profile-elevation');
  // With tangent and skyline angle both wired, a per-ray pass derives the angle from the tangent.
  expect(ids.at(-1)).toBe('point-horizon-profile-skyline-angle');
  expect(ids.at(-2)).toBe('point-horizon-profile-march-0');
  expect(ids.at(-3)).toBe('point-horizon-profile-chunks');
  expect(ids.slice(1, -3).length).toBeGreaterThanOrEqual(1);
  expect(ids.slice(1, -3).every(id => id.includes('-pyramid-level'))).toBe(true);
  // Every kernel binds at most 8 storage buffers. This pyramid march binds 7: values, pyramid,
  // settings, observers, tangent, distance and the chunk start word (review C-11 dropped the
  // never-read validity binding); with `samples` too it binds exactly 8.
  for (const node of nodes) {
    expect(node.resources?.length ?? 0).toBeLessThanOrEqual(8);
  }
  const march = nodes.find(node => node.id.endsWith('march-0'));
  expect(march?.resources?.length).toBe(7);

  const marchOnly = create({id: 'ring', traversal: 'march'});
  expect(marchOnly.getCommandNodes(graph).map(node => node.id)).toEqual([
    'ring-elevation',
    'ring-chunks',
    'ring-march-0',
    'ring-skyline-angle'
  ]);
  const chunked = create({id: 'chunk', raysPerDispatch: 1000, traversal: 'march'});
  expect(chunked.getCommandNodes(graph).map(node => node.id)).toEqual([
    'chunk-elevation',
    'chunk-chunks',
    'chunk-march-0',
    'chunk-march-1',
    'chunk-march-2',
    'chunk-skyline-angle'
  ]);
  const tangentOnly = (name: string, length = 3 * 720) => ({
    tangent: createTransientView(graph, name, 'float32', length)
  });
  const single = create({id: 'single', output: tangentOnly('single-tangent')});
  const singleMarch = single.getCommandNodes(graph).at(-1);
  // values, pyramid, settings, observers, tangent and the chunk start word.
  expect(singleMarch?.resources?.length).toBe(6);
  const sector = create({
    id: 'sector',
    azimuthSpan: 100,
    firstAzimuth: 20,
    output: tangentOnly('sector-tangent', 300)
  });
  expect(sector.model.azimuthSpan).toBe(100);

  expect(() => create({output: {}})).toThrow(/at least one output/);
  expect(() => create({output: tangentOnly('short-tangent', 10)})).toThrow(
    /observers \* azimuthSpan/
  );
  expect(() =>
    create({parameters: createTransientView(graph, 'settings-short', 'float32', 7)})
  ).toThrow(/parameters/);
  expect(() =>
    create({observers: createTransientView(graph, 'observers-int', 'uint32x4' as never, 3)})
  ).toThrow(/observers/);
  expect(() => create({grid: makeGPURasterMetadata({width: 1, height: 48})})).toThrow(/dimensions/);
  expect(() => create({projection: 'polar' as never})).toThrow(/projection/);
  expect(() => create({traversal: 'fast' as never})).toThrow(/traversal/);
  expect(() => create({heightReference: 'relative' as never})).toThrow(/heightReference/);
  expect(() => create({rowDirection: 'up' as never})).toThrow(/rowDirection/);
  expect(() => create({azimuthCount: 0})).toThrow(/azimuthCount/);
  expect(() => create({firstAzimuth: 720})).toThrow(/firstAzimuth/);
  expect(() => create({azimuthSpan: 721})).toThrow(/azimuthSpan/);
  expect(() => create({maximumDistance: 5})).toThrow(/maximumDistance/);
  expect(() => create({raysPerDispatch: 0})).toThrow(/raysPerDispatch/);
  const shared = createBand(graph, 'shared', 3 * 720);
  expect(() =>
    create({
      elevation: {...shared, storage: {kind: 'buffer', values: shared.storage.values}},
      output: {tangent: shared.storage.values}
    })
  ).toThrow(/share buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUPointHorizonProfile({
      grid: makeGPURasterMetadata({width: 4, height: 4}),
      elevation: createBand(otherGraph, 'foreign', 16),
      observers: createTransientView(otherGraph, 'foreign-observers', 'float32x4', 1),
      parameters: createTransientView(otherGraph, 'foreign-settings', 'float32', 8),
      output: {tangent: createTransientView(otherGraph, 'foreign-tangent', 'float32', 720)},
      maximumDistance: 100,
      cellSize: 10
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('GPUPointHorizonVisibility schedules classification chunks within 8 bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUPointHorizonVisibilityProps> = {}) => {
    instance++;
    return new GPUPointHorizonVisibility({
      grid: makeGPURasterMetadata({width: 64, height: 48, cellSize: 10}),
      elevation: createBand(graph, `terrain-${instance}`, 64 * 48),
      observers: createTransientView(graph, `observers-${instance}`, 'float32x4', 2),
      targets: createTransientView(graph, `targets-${instance}`, 'float32x4', 5),
      parameters: createTransientView(graph, `settings-${instance}`, 'float32', 12),
      output: {
        visibility: createTransientView(graph, `visibility-${instance}`, 'uint32', 5),
        details: createTransientView(graph, `details-${instance}`, 'float32x4', 5)
      },
      maximumDistance: 1000,
      cellSize: 10,
      projection: 'web-mercator',
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.id).toBe('point-horizon-visibility');
  checkPointHorizonVerticalDatums(create, createBand(graph, 'datum-elevation', 64 * 48));
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.at(-1)?.id).toBe('point-horizon-visibility-classify-0');
  expect(nodes.at(-1)?.resources?.length).toBe(8);
  expect(nodes.some(node => node.id === 'point-horizon-visibility-pyramid-level-0')).toBe(true);
  expect(
    create({
      id: 'chunked',
      targetsPerDispatch: 2,
      traversal: 'march',
      output: {visibility: createTransientView(graph, 'chunked-visibility', 'uint32', 5)}
    })
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toEqual([
    'chunked-elevation',
    'chunked-chunks',
    'chunked-classify-0',
    'chunked-classify-1',
    'chunked-classify-2'
  ]);
  expect(() =>
    create({output: {visibility: createTransientView(graph, 'short-visibility', 'uint32', 4)}})
  ).toThrow(/one value per target/);
  expect(() =>
    create({
      output: {
        visibility: createTransientView(graph, 'details-visibility', 'uint32', 5),
        details: createTransientView(graph, 'short-details', 'float32x4', 4)
      }
    })
  ).toThrow(/one row per target/);
  expect(() =>
    create({parameters: createTransientView(graph, 'short-settings', 'float32', 11)})
  ).toThrow(/parameters/);
  expect(() => create({targetsPerDispatch: 0})).toThrow(/targetsPerDispatch/);
  expect(() =>
    create({targets: createTransientView(graph, 'bad-targets', 'float32x2' as never, 5)})
  ).toThrow(/targets/);
  device.destroy();
});

it('point horizon kernels reject settings packed for the other projection', () => {
  // settings.projection selects the tag the packers write; the kernels compare it and the kernels compare the tag
  // with their compile-time projection, so a mismatch marks every eye invalid (NaN / no-data)
  // instead of reading worldPixelSize as a cell size. Tag 0 (hand-written arrays) is accepted.
  expect(POINT_HORIZON_PROFILE_PROJECTION_SLOT).toBe(7);
  expect(POINT_HORIZON_VISIBILITY_PROJECTION_SLOT).toBe(11);
  const options = {maximumDistance: 2000, cellSize: 30, traversal: 'march' as const};
  const grid = makeGPURasterMetadata({width: 16, height: 16});
  const planar = resolvePointHorizonModel('planar', grid, options);
  const mercator = resolvePointHorizonModel('mercator', grid, {
    ...options,
    projection: 'web-mercator'
  });
  const planarWGSL = getPointHorizonMarchWGSL(planar, POINT_HORIZON_PROFILE_PROJECTION_SLOT);
  expect(planarWGSL).toContain('let tag = settings[settingsOffset + 7u];');
  expect(planarWGSL).toContain('return tag == 0.0 || tag == 1.0;');
  expect(planarWGSL).toContain('eye.valid = isInsideGrid(column, row) && isProjectionTagValid();');
  const mercatorWGSL = getPointHorizonMarchWGSL(mercator, POINT_HORIZON_VISIBILITY_PROJECTION_SLOT);
  expect(mercatorWGSL).toContain('let tag = settings[settingsOffset + 11u];');
  expect(mercatorWGSL).toContain('return tag == 0.0 || tag == 2.0;');
  // The packers tag by settings.projection in both layouts.
  expect(
    getGPUPointHorizonVisibilityParameterValues(
      {projection: 'web-mercator'},
      makeMercatorTestGrid(16, 16, 1024, 0)
    )[11]
  ).toBe(2);
  expect(
    getGPUPointHorizonProfileParameterValues(
      {},
      makeGPURasterMetadata({width: 16, height: 16, cellSize: 5})
    )[7]
  ).toBe(1);
});
