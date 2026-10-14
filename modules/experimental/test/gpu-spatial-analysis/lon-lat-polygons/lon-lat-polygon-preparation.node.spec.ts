// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPULonLatPolygonPreparation,
  getGPULonLatPolygonPreparationParameterValues,
  GPU_LON_LAT_POLYGON_PREPARATION_PARAMETER_LENGTH,
  type GPULonLatPolygonPreparationProps
} from '../../../src/gpu-spatial-analysis/lon-lat-polygons';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  prepareLonLatPolygons,
  getLonLatFeatures,
  flattenLonLatFeatures,
  densifyRing,
  classifyRing,
  capPoleRing,
  normalizePolygon,
  splitPolygonAtAntimeridian
} from './lon-lat-polygon-oracle';
import {
  LON_LAT_POLYGON_FIXTURES,
  FIJI_TRANSMERIDIAN,
  ANTARCTICA_NATURAL_EARTH,
  ARCTIC_RING,
  SOUTHERN_POLE_RING,
  AMBIGUOUS_180_EDGE,
  SPIRAL_WIDER_THAN_360
} from './fixtures';

let serial = 0;
function createProps(graph: GPUCommandGraph): GPULonLatPolygonPreparationProps {
  const words = (length: number) =>
    createTransientView(graph, `words-${serial++}`, 'uint32', length);
  return {
    id: 'prepare',
    positions: createTransientView(graph, `positions-${serial++}`, 'float32x2', 12),
    featureOffsets: words(3),
    polygonOffsets: words(4),
    ringOffsets: words(5),
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
    parameters: createTransientView(graph, `parameters-${serial++}`, 'float32', 4),
    maximumPieceCount: 4,
    output: {
      positions: createTransientView(graph, `output-${serial++}`, 'float32x2', 64),
      featureOffsets: words(3),
      polygonOffsets: words(7),
      ringOffsets: words(9),
      ringFlags: words(4),
      status: {
        count: words(1),
        overflow: words(1),
        requiredCount: words(1),
        invalidCount: words(1),
        approximation: words(1)
      }
    }
  };
}

it('preparation parameters use edge-model defaults and validate degree steps and target length', () => {
  expect(GPU_LON_LAT_POLYGON_PREPARATION_PARAMETER_LENGTH).toBe(4);
  expect(Array.from(getGPULonLatPolygonPreparationParameterValues('linear'))).toEqual([0, 0, 0, 0]);
  expect(getGPULonLatPolygonPreparationParameterValues('great-circle')[0]).toBe(Math.fround(0.1));
  for (const maximumSegmentDegrees of [-1, NaN]) {
    expect(() =>
      getGPULonLatPolygonPreparationParameterValues('linear', {maximumSegmentDegrees})
    ).toThrow('getGPULonLatPolygonPreparationParameterValues maximumSegmentDegrees must be >= 0');
  }
  expect(() =>
    getGPULonLatPolygonPreparationParameterValues('linear', {}, new Float32Array(3))
  ).toThrow('getGPULonLatPolygonPreparationParameterValues target must hold 4 elements');
  const target = new Float32Array(6).fill(7);
  expect(
    getGPULonLatPolygonPreparationParameterValues(
      'great-circle',
      {maximumSegmentDegrees: 0},
      target
    )
  ).toBe(target);
  expect(Array.from(target)).toEqual([0, 0, 0, 0, 7, 7]);
});

it('preparation validates every fixed topology length with the contract messages', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  for (const [name, expected] of [
    ['featureOffsets', 3],
    ['polygonOffsets', 7],
    ['ringOffsets', 9],
    ['ringFlags', 4]
  ] as const) {
    const props = createProps(graph);
    props.output[name] = createTransientView(graph, `short-${serial++}`, 'uint32', expected - 1);
    expect(() => new GPULonLatPolygonPreparation(props)).toThrow(
      `prepare output.${name} must hold ${expected} rows`
    );
  }
  for (const maximumPieceCount of [0, 1.5, 65537]) {
    expect(
      () => new GPULonLatPolygonPreparation({...createProps(graph), maximumPieceCount})
    ).toThrow('prepare maximumPieceCount must be an integer in [1, 65536]');
  }
  for (const maximumPieceCount of [1, 65536]) {
    expect(
      new GPULonLatPolygonPreparation({...createProps(graph), maximumPieceCount}).props
        .maximumPieceCount
    ).toBe(maximumPieceCount);
  }
  const props = createProps(graph);
  expect(() => new GPULonLatPolygonPreparation({...props, spatialContext: undefined})).toThrow(
    'prepare does not support planar coordinates'
  );
  expect(
    () =>
      new GPULonLatPolygonPreparation({
        ...props,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'ellipsoidal',
          edgeModel: 'great-circle'
        }
      })
  ).toThrow('prepare ellipsoidal contexts take ellipsoidalEdgeModel, not edgeModel');
  expect(
    () =>
      new GPULonLatPolygonPreparation({
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'}
      })
  ).toThrow('prepare does not support the ellipsoidal metric');
  expect(() => new GPULonLatPolygonPreparation({...props, antimeridian: 'other' as never})).toThrow(
    'prepare antimeridian must be split or reject'
  );
  expect(() => new GPULonLatPolygonPreparation({...props, poleRings: 'other' as never})).toThrow(
    'prepare poleRings must be cap or reject'
  );
  expect(
    () =>
      new GPULonLatPolygonPreparation({
        ...props,
        parameters: createTransientView(graph, `short-${serial++}`, 'float32', 3)
      })
  ).toThrow('prepare parameters must hold 4 float32 values');
  for (const name of ['featureOffsets', 'polygonOffsets', 'ringOffsets'] as const) {
    expect(
      () =>
        new GPULonLatPolygonPreparation({
          ...props,
          [name]: createTransientView(graph, `short-${serial++}`, 'uint32', 1)
        })
    ).toThrow(`prepare ${name} needs two rows`);
  }
  expect(
    () =>
      new GPULonLatPolygonPreparation({
        ...props,
        positions: createTransientView(graph, `empty-${serial++}`, 'float32x2', 0)
      })
  ).toThrow('prepare needs at least one position');
  expect(
    () =>
      new GPULonLatPolygonPreparation({
        ...props,
        output: {...props.output, positions: props.positions}
      })
  ).toThrow(/share buffers/);
});

it('preparation preserves metric, units and datum while declaring linear output edges', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const spatialContext = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'kilometers',
    edgeModel: 'great-circle',
    datum: 'WGS84'
  } as const;
  const preparation = new GPULonLatPolygonPreparation({
    ...createProps(graph),
    spatialContext,
    id: undefined
  });
  expect(preparation.id).toBe('lon-lat-polygon-preparation');
  expect(preparation.spatialContext).toEqual({...spatialContext, edgeModel: 'linear'});
});

it('preparation emits valid WGSL for both edge models and every kernel has at most eight storage bindings', () => {
  for (const edgeModel of ['linear', 'great-circle'] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const preparation = new GPULonLatPolygonPreparation({
      ...createProps(graph),
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none', edgeModel}
    });
    const nodes = preparation.getCommandNodes(graph);
    expect(nodes.at(-1)?.id).toBe('prepare-publish');
    for (const source of getGPUCommandNodeKernelSources(nodes)) {
      const reflection = new WgslReflect(source.source);
      expect(reflection.storage.length).toBeLessThanOrEqual(8);
    }
  }
});

it('binary64 preparation preserves Antarctica, caps either pole and rejects complete invalid features', () => {
  const antarctica = prepareLonLatPolygons(ANTARCTICA_NATURAL_EARTH);
  expect(antarctica.positions).toEqual(ANTARCTICA_NATURAL_EARTH.positions);
  expect(Array.from(antarctica.ringFlags)).toEqual([0]);
  for (const [fixture, pole] of [
    [ARCTIC_RING, 90],
    [SOUTHERN_POLE_RING, -90]
  ] as const) {
    const prepared = prepareLonLatPolygons(fixture);
    expect(prepared.ringFlags[0] & 2).toBe(2);
    expect(Array.from(prepared.positions).filter((_, index) => index % 2)).toContain(pole);
  }
  for (const fixture of [AMBIGUOUS_180_EDGE, SPIRAL_WIDER_THAN_360]) {
    const feature = getLonLatFeatures(FIJI_TRANSMERIDIAN)[0].concat(getLonLatFeatures(fixture)[0]);
    const prepared = prepareLonLatPolygons(flattenLonLatFeatures([feature]));
    expect(prepared.status.invalidCount).toBe(1);
    expect(prepared.positions.length).toBe(0);
  }
  const bounded = prepareLonLatPolygons(FIJI_TRANSMERIDIAN, {positionCapacity: 3});
  expect(bounded.status).toMatchObject({count: 3, requiredCount: 8, overflow: 1});
  expect(Array.from(bounded.ringOffsets)).toEqual([0, 3, 3]);
});

it('bounds transient streams by output capacity independently of input size and maximumPieceCount', () => {
  for (const capacity of [0, 3, 64]) {
    for (const maximumPieceCount of [undefined, 65536]) {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      const props = createProps(graph);
      props.positions = createTransientView(graph, `large-input-${serial++}`, 'float32x2', 10000);
      props.output.positions = createTransientView(
        graph,
        `capacity-${serial++}`,
        'float32x2',
        capacity
      );
      props.maximumPieceCount = maximumPieceCount;
      const createView = vi.spyOn(graph, 'createDataView');
      try {
        const nodes = new GPULonLatPolygonPreparation(props).getCommandNodes(graph);
        const streams = createView.mock.calls.filter(([buffer]) =>
          /^prepare-(dense|cap|clip-\d+)-(positions|turns)/.test(buffer.id)
        );
        expect(streams).toHaveLength(12);
        for (const [, view] of streams) {
          expect(view.length).toBe(capacity);
        }
        const stageScans = createView.mock.calls.filter(([buffer]) =>
          /^prepare-clip-\d+-\d+-(counts|starts)$/.test(buffer.id)
        );
        expect(stageScans).toHaveLength(capacity === 0 ? 0 : 8);
        for (const [, view] of stageScans) {
          expect(view.length).toBe(capacity);
        }
        // Six position/turn stream pairs (72 bytes per capacity row), plus four count/scan pairs.
        expect(
          streams.reduce(
            (bytes, [, view]) => bytes + view.length * (view.format === 'float32x2' ? 8 : 4),
            0
          )
        ).toBe(72 * capacity);
        const clipCounts = nodes.filter(node => /-clip-\d+-\d+-count$/.test(node.id));
        expect(clipCounts).toHaveLength(capacity === 0 ? 0 : 4);
        for (const source of getGPUCommandNodeKernelSources(nodes)) {
          expect(new WgslReflect(source.source).storage.length).toBeLessThanOrEqual(8);
        }
      } finally {
        createView.mockRestore();
      }
    }
  }
});

it('the binary64 resize bound covers dense, cap, both clipped copies and the exact final count', () => {
  for (const input of Object.values(LON_LAT_POLYGON_FIXTURES)) {
    for (const edgeModel of ['linear', 'great-circle'] as const) {
      const polygons = getLonLatFeatures(input).flat();
      let denseCount = 0;
      let capCount = 0;
      let clipCount = 0;
      for (const polygon of polygons) {
        const rings = polygon.map(ring => {
          const dense = densifyRing(ring, edgeModel).unwrapped;
          denseCount += dense.length;
          const capped = classifyRing(ring, edgeModel).poleRing ? capPoleRing(dense) : dense;
          capCount += capped.length;
          return capped;
        });
        clipCount += splitPolygonAtAntimeridian(normalizePolygon(rings)).flat(2).length;
      }
      const bound = 8 * (denseCount + 3 * (input.ringOffsets.length - 1));
      expect(bound).toBeGreaterThanOrEqual(denseCount);
      expect(bound).toBeGreaterThanOrEqual(capCount);
      expect(bound).toBeGreaterThanOrEqual(clipCount);
      expect(bound).toBeGreaterThanOrEqual(
        prepareLonLatPolygons(input, {edgeModel}).status.requiredCount
      );
    }
  }
});
