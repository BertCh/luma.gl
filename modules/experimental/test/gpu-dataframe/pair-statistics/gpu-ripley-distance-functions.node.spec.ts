// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPURipleyDistanceFunctions,
  type GPURipleyDistanceFunctionsOutput,
  type GPURipleyDistanceFunctionsProps
} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley-distance-functions';
import {
  getGPURipleyDistanceFunctionsParameterValues,
  GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH
} from '../../../src/gpu-dataframe/pair-statistics/ripley-distance-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

type Overrides = Omit<Partial<GPURipleyDistanceFunctionsProps>, 'output'> & {
  output?: Partial<GPURipleyDistanceFunctionsOutput>;
};

function createProps(
  graph: GPUCommandGraph,
  overrides: Overrides = {}
): GPURipleyDistanceFunctionsProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    positions: view('float32x2', 10),
    parameters: view('float32', GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH),
    gridSize: [4, 4],
    referenceGrid: [5, 5],
    radiusCount: 8,
    ...overrides,
    output: {g: view('float32', 8), ...overrides.output}
  };
}

it('getGPURipleyDistanceFunctionsParameterValues packs and validates the layout', () => {
  expect(
    Array.from(
      getGPURipleyDistanceFunctionsParameterValues({bounds: [0, 1, 10, 11], maximumDistance: 3})
    )
  ).toEqual([0, 1, 10, 11, 3, 1, 0, 0]);
  expect(
    getGPURipleyDistanceFunctionsParameterValues({
      bounds: [0, 1, 10, 11],
      maximumDistance: 3,
      edgeCorrection: 'none'
    })[5]
  ).toBe(0);
  expect(
    ['kaplan-meier', 'hanisch'].map(
      edgeCorrection =>
        getGPURipleyDistanceFunctionsParameterValues({
          bounds: [0, 1, 10, 11],
          maximumDistance: 3,
          edgeCorrection: edgeCorrection as 'hanisch'
        })[5]
    )
  ).toEqual([2, 3]);
  expect(() =>
    getGPURipleyDistanceFunctionsParameterValues({
      bounds: [0, 0, 1, 1],
      maximumDistance: 1,
      edgeCorrection: 'isotropic' as 'none'
    })
  ).toThrow(/edgeCorrection/);
  expect(() =>
    getGPURipleyDistanceFunctionsParameterValues({bounds: [0, 0, 1, 1], maximumDistance: 0})
  ).toThrow(/maximumDistance/);
});

it('GPURipleyDistanceFunctions validates props and declares nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'ripley-distance-nodes'});
  expect(
    () => new GPURipleyDistanceFunctions(createProps(graph, {output: {g: undefined}}))
  ).toThrow(/at least one/);
  expect(() => new GPURipleyDistanceFunctions(createProps(graph, {radiusCount: 0}))).toThrow(
    /radiusCount/
  );
  expect(() => new GPURipleyDistanceFunctions(createProps(graph, {referenceGrid: [0, 4]}))).toThrow(
    /referenceGrid/
  );
  const ids = new GPURipleyDistanceFunctions(createProps(graph))
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids).toContain('ripley-distance-functions-f-reference');
  expect(ids.at(-1)).toBe('ripley-distance-functions-finish');
  device.destroy();
});

it('GPURipleyDistanceFunctions declares planar-only spatial support', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(
    () =>
      new GPURipleyDistanceFunctions(
        createProps(graph, {
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const
        })
      )
  ).toThrow(/project positions to a planar frame first/);
  expect(new GPURipleyDistanceFunctions(createProps(graph)).spatialContext.coordinateSpace).toBe(
    'planar'
  );
  device.destroy();
});

import {WgslReflect} from 'wgsl_reflect';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';
import {createRipleyTestWindow} from './ripley-envelope-fixture';
import {
  computeRipleyDistanceFunctionsOnCPU,
  getReferenceLatticeErosion
} from './ripley-distance-functions-oracle';
import {
  ORACLE_RECTANGLE,
  ORACLE_L_SHAPE,
  getOracleErodedArea,
  getOracleWindowArea
} from '../../analysis-core/study-window-oracle';

it('GPURipleyDistanceFunctions packs study-window bounds and builds polygon Hanisch even for G only', () => {
  const packed = getGPURipleyDistanceFunctionsParameterValues({
    bounds: 'study-window',
    maximumDistance: 100,
    edgeCorrection: 'hanisch'
  });
  expect(Array.from(packed.slice(0, 4)).every(Number.isNaN)).toBe(true);
  expect(packed[5]).toBe(3);
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const studyWindow = createRipleyTestWindow(graph);
  const nodes = new GPURipleyDistanceFunctions(createProps(graph, {studyWindow})).getCommandNodes(
    graph
  );
  for (const suffix of ['window-parameters', 'hanisch-reference', 'hanisch-suffix']) {
    expect(nodes.some(node => node.id === `ripley-distance-functions-${suffix}`)).toBe(true);
  }
  const sources = getGPUCommandNodeKernelSources(nodes);
  for (const {id, source} of sources) {
    expect(() => new WgslReflect(source), id).not.toThrow();
    expect(new WgslReflect(source).storage.length, id).toBeLessThanOrEqual(8);
  }
  const reference = sources.find(kernel => kernel.id.endsWith('-f-reference'))!.source;
  expect(reference.match(/isInsideStudyWindow\(/g)).toHaveLength(2);
  expect(reference).toContain(
    'getStudyWindowBoundaryDistance(vec2f(x, y), lattice.maximumDistance)'
  );
  expect(reference).toContain('mix(f32(erosionCounts');
  const finish = sources.find(kernel => kernel.id.endsWith('-finish'))!.source;
  expect(finish).toContain('let REFERENCE_COUNT = f32(getReferencePrefix(3u, RADIUS_COUNT))');
  device.destroy();
});

it('GPURipleyDistanceFunctions explicit rectangle keeps every generated shader unchanged', () => {
  const device = createNullWebGPUDevice();
  const build = (studyWindow?: 'rectangle') => {
    const graph = new GPUCommandGraph(device);
    return getGPUCommandNodeKernelSources(
      new GPURipleyDistanceFunctions(createProps(graph, {studyWindow})).getCommandNodes(graph)
    ).map(kernel => kernel.source);
  };
  expect(build('rectangle')).toEqual(build());
  expect(build().join('\n')).not.toContain('studyWindow');
  expect(build().join('\n')).not.toContain('erosionCounts');
  device.destroy();
});

it('polygon rectangle oracle matches border and Kaplan-Meier and estimates rectangle erosion', () => {
  const scene = {
    positions: new Float32Array([110, 140, 170, 180, 420, 350, 800, 510, 1330, 790, 1700, 300])
  };
  for (const edgeCorrection of ['border', 'kaplan-meier'] as const) {
    const parameters = {bounds: [0, 0, 2000, 1000] as const, maximumDistance: 400, edgeCorrection};
    const rectangle = computeRipleyDistanceFunctionsOnCPU(scene, parameters, 8, [31, 29]);
    const polygon = computeRipleyDistanceFunctionsOnCPU(
      scene,
      {...parameters, bounds: 'study-window'},
      8,
      [31, 29],
      ORACLE_RECTANGLE
    );
    expect(polygon).toEqual(rectangle);
  }
  const erosion = getReferenceLatticeErosion(ORACLE_RECTANGLE, [400, 200], 400, 16);
  for (const radius of [25, 87.5, 150, 275]) {
    expect(
      Math.abs(erosion.getArea(radius) - (2000 - 2 * radius) * (1000 - 2 * radius))
    ).toBeLessThan(16000);
  }
});

it('Hanisch reference-lattice erosion approaches the independent raster oracle', () => {
  const radius = 123;
  const expected = getOracleErodedArea(ORACLE_L_SHAPE, radius, 600);
  const coarse = getReferenceLatticeErosion(ORACLE_L_SHAPE, [12, 12], 400, 64).getArea(radius);
  const fine = getReferenceLatticeErosion(ORACLE_L_SHAPE, [320, 320], 400, 64).getArea(radius);
  expect(Math.abs(fine - expected)).toBeLessThan(Math.abs(coarse - expected));
  expect(Math.abs(fine - expected) / getOracleWindowArea(ORACLE_L_SHAPE)).toBeLessThan(0.01);
  const result = computeRipleyDistanceFunctionsOnCPU(
    {positions: new Float32Array([100, 100, 300, 100, 900, 200, 1500, 1500])},
    {bounds: 'study-window', maximumDistance: 400, edgeCorrection: 'none'},
    4,
    [20, 20],
    ORACLE_L_SHAPE
  );
  expect(result.gDenominators).toEqual([3, 3, 3, 3]);
  expect(result.fDenominators).toEqual([231, 231, 231, 231]);
});
