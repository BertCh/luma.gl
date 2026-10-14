// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUCatchmentAccessibility} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility';
import {GPUHuffTradeAreas} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas';
import {GPUDotDensity} from '../../../src/gpu-spatial-analysis/dot-density/gpu-dot-density';
import {GPURandomPointsInPolygon} from '../../../src/gpu-spatial-analysis/dot-density/gpu-random-points-in-polygon';
import {GPURandomPointsOnLine} from '../../../src/gpu-spatial-analysis/dot-density/gpu-random-points-on-line';
import {GPUGeographicDistribution} from '../../../src/gpu-spatial-analysis/geographic-distribution/gpu-geographic-distribution';
import {GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/geographic-distribution/geographic-distribution-parameters';
import {GPUShapeGenerator} from '../../../src/gpu-spatial-analysis/grid-generators/gpu-shape-generator';
import {GPUGroupGeometry} from '../../../src/gpu-spatial-analysis/group-geometry/gpu-group-geometry';
import {GPULabelPoint} from '../../../src/gpu-spatial-analysis/label-point/gpu-label-point';
import {GPUGreatCircleArcs} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-great-circle-arcs';
import {GPULineChunk} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-chunk';
import {GPULineSegmentize} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-segmentize';
import {GPULineSimplification} from '../../../src/gpu-spatial-analysis/line-simplification/gpu-line-simplification';
import {getLineGeodesicMetricWGSL} from '../../../src/gpu-spatial-analysis/line-simplification/line-simplification-kernels';
import {GPULineLocate} from '../../../src/gpu-spatial-analysis/linear-referencing/gpu-line-locate';
import {GPULinearReferencing} from '../../../src/gpu-spatial-analysis/linear-referencing/gpu-linear-referencing';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

const EQUIDISTANT: GPUSpatialContext = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  frame: {id: 'f', kind: 'local-azimuthal-equidistant'}
};
const EQUAL_AREA: GPUSpatialContext = {
  ...EQUIDISTANT,
  frame: {id: 'f', kind: 'local-lambert-equal-area'}
};
const KILOMETERS: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'kilometers'
};

function createViews() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'measure-contract'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const createStatus = () => ({count: view('uint32', 1), overflow: view('uint32', 1)});
  const createLine = () => ({positions: view('float32x2', 3), lineOffsets: view('uint32', 2)});
  const createPolygon = () => ({
    positions: view('float32x2', 3),
    ringOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    featureOffsets: view('uint32', 2)
  });
  const createLineOutput = () => ({...createLine(), status: createStatus()});
  const createPointsOutput = () => ({
    positions: view('float32x2', 1),
    dots: {ids: view('uint32', 1), status: createStatus()}
  });
  const createWeights = (spatialContext?: GPUSpatialContext) => ({
    spatialContext,
    offsets: view('uint32', 2),
    neighbors: view('uint32', 1),
    weights: view('float32', 1)
  });
  return {
    graph,
    view,
    createStatus,
    createLine,
    createPolygon,
    createLineOutput,
    createPointsOutput,
    createWeights
  };
}

type Views = ReturnType<typeof createViews>;
type MeasureCase = {
  name: string;
  measure: 'distance' | 'area';
  create: (views: Views, spatialContext: GPUSpatialContext) => {spatialContext: GPUSpatialContext};
};

const MEASURE_CASES: MeasureCase[] = [
  {
    name: 'GPUGeographicDistribution',
    measure: 'distance',
    create: ({view}, spatialContext) =>
      new GPUGeographicDistribution({
        spatialContext,
        positions: view('float32x2', 1),
        parameters: view('float32', GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH),
        output: {standardDistances: view('float32', 1)}
      })
  },
  {
    name: 'GPUGroupGeometry',
    measure: 'distance',
    create: ({view}, spatialContext) =>
      new GPUGroupGeometry({
        spatialContext,
        positions: view('float32x2', 1),
        labels: view('uint32', 1),
        groupCount: 1,
        parameters: view('float32', GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH),
        output: {standardDistances: view('float32', 1)}
      })
  },
  {
    name: 'GPULabelPoint',
    measure: 'distance',
    create: ({view, createPolygon}, spatialContext) =>
      new GPULabelPoint({
        spatialContext,
        ...createPolygon(),
        output: {points: view('float32x2', 1), distances: view('float32', 1)}
      })
  },
  {
    name: 'GPULineSegmentize',
    measure: 'distance',
    create: ({view, createLine, createLineOutput}, spatialContext) =>
      new GPULineSegmentize({
        spatialContext,
        ...createLine(),
        parameters: view('float32', 4),
        output: createLineOutput()
      })
  },
  {
    name: 'GPULineChunk',
    measure: 'distance',
    create: ({view, createLine, createLineOutput}, spatialContext) =>
      new GPULineChunk({
        spatialContext,
        ...createLine(),
        mode: 'substring',
        parameters: view('float32', 4),
        output: createLineOutput()
      })
  },
  {
    name: 'GPULinearReferencing',
    measure: 'distance',
    create: ({view, createLine}, spatialContext) =>
      new GPULinearReferencing({
        spatialContext,
        ...createLine(),
        queryPositions: view('float32x2', 1),
        radius: view('float32', 1),
        candidateCapacity: 1,
        output: {measures: view('float32', 1), status: {overflow: view('uint32', 1)}}
      })
  },
  {
    name: 'GPULineLocate',
    measure: 'distance',
    create: ({view, createLine}, spatialContext) =>
      new GPULineLocate({
        spatialContext,
        ...createLine(),
        eventLineIndices: view('uint32', 1),
        eventMeasures: view('float32', 1),
        output: {positions: view('float32x2', 1)}
      })
  },
  {
    name: 'GPUShapeGenerator',
    measure: 'distance',
    create: ({view}, spatialContext) =>
      new GPUShapeGenerator({
        spatialContext,
        shape: 'circle',
        maximumSegments: 3,
        centers: view('float32x2', 1),
        radii: view('float32', 1),
        parameters: view('float32', 2),
        output: {positions: view('float32x2', 4), offsets: view('uint32', 2)}
      })
  },
  {
    name: 'GPURandomPointsOnLine',
    measure: 'distance',
    create: ({view, createLine, createStatus}, spatialContext) =>
      new GPURandomPointsOnLine({
        spatialContext,
        ...createLine(),
        counts: view('uint32', 1),
        parameters: view('uint32', 8),
        output: {
          positions: view('float32x2', 1),
          points: {ids: view('uint32', 1), status: createStatus()}
        }
      })
  },
  {
    name: 'GPUDotDensity',
    measure: 'area',
    create: ({view, createPolygon, createPointsOutput}, spatialContext) =>
      new GPUDotDensity({
        spatialContext,
        ...createPolygon(),
        values: view('float32', 1),
        parameters: view('uint32', 8),
        output: createPointsOutput()
      })
  },
  {
    name: 'GPURandomPointsInPolygon',
    measure: 'area',
    create: ({view, createPolygon, createPointsOutput}, spatialContext) =>
      new GPURandomPointsInPolygon({
        spatialContext,
        ...createPolygon(),
        counts: view('uint32', 1),
        parameters: view('uint32', 8),
        output: createPointsOutput()
      })
  },
  ...(['douglas-peucker', 'visvalingam'] as const).map(
    (method): MeasureCase => ({
      name: `GPULineSimplification ${method}`,
      measure: method === 'visvalingam' ? 'area' : 'distance',
      create: ({view, createLine}, spatialContext) =>
        new GPULineSimplification({
          spatialContext,
          method,
          ...createLine(),
          importance: view('float32', 3)
        })
    })
  )
];

for (const {name, measure, create} of MEASURE_CASES) {
  it(`${name} accepts the frame for ${measure} and rejects the other frame`, () => {
    const views = createViews();
    const accepted = measure === 'distance' ? EQUIDISTANT : EQUAL_AREA;
    const rejected = measure === 'distance' ? EQUAL_AREA : EQUIDISTANT;
    expect(() => create(views, rejected)).toThrow(new RegExp(`measures ${measure}`));
    expect(create(views, accepted).spatialContext).toEqual(accepted);
  });
}

it('GPUGreatCircleArcs remains longitude/latitude only and honors kilometers', () => {
  const {graph, view, createLineOutput} = createViews();
  const create = (spatialContext: GPUSpatialContext) =>
    new GPUGreatCircleArcs({
      spatialContext,
      sources: view('float32x2', 1),
      targets: view('float32x2', 1),
      parameters: view('float32', 4),
      output: createLineOutput()
    });
  for (const spatialContext of [EQUIDISTANT, EQUAL_AREA]) {
    expect(() => create(spatialContext)).toThrow(/does not support planar coordinates/);
  }
  const arcs = create(KILOMETERS);
  expect(arcs.radius).toBeCloseTo(6371.0088, 8);
  expect(arcs.getCommandNodes(graph).length).toBeGreaterThan(0);
});

it('Douglas-Peucker uses kilometer radii in its generated distance kernels', () => {
  const {graph, view, createLine} = createViews();
  const line = new GPULineSimplification({
    spatialContext: KILOMETERS,
    method: 'douglas-peucker',
    ...createLine(),
    importance: view('float32', 3),
    maximumIterations: 1
  });
  expect(line.spatialContext.units).toBe('kilometers');
  expect(line.sphereRadius).toBeCloseTo(6371.0088, 8);
  const radiusDeclaration = `const LINE_SPHERE_RADIUS: f32 = ${line.sphereRadius};`;
  expect(getLineGeodesicMetricWGSL(line.sphereRadius!)).toContain(radiusDeclaration);
  const sources = getGPUCommandNodeKernelSources(line.getCommandNodes(graph));
  expect(sources.some(({source}) => source.includes(radiusDeclaration))).toBe(true);
});

for (const name of ['GPUCatchmentAccessibility', 'GPUHuffTradeAreas'] as const) {
  it(`${name} checks both declared weight contexts and accepts omitted metadata`, () => {
    const {view, createWeights} = createViews();
    const create = (demandContext?: GPUSpatialContext, facilityContext?: GPUSpatialContext) => {
      const demandWeights = createWeights(demandContext);
      const facilityWeights = createWeights(facilityContext);
      return name === 'GPUCatchmentAccessibility'
        ? new GPUCatchmentAccessibility({
            method: '2sfca',
            supply: view('float32', 1),
            demand: view('float32', 1),
            demandWeights,
            facilityWeights,
            output: {accessibility: view('float32', 1)}
          })
        : new GPUHuffTradeAreas({
            attractiveness: view('float32', 1),
            demandWeights,
            facilityWeights,
            output: {tradeArea: view('uint32', 1)}
          });
    };
    expect(() =>
      create(EQUIDISTANT, {
        ...EQUIDISTANT,
        frame: {id: 'other-frame', kind: 'local-azimuthal-equidistant'}
      })
    ).toThrow(/incompatible spatial contexts/);
    expect(() => create(EQUIDISTANT, EQUIDISTANT)).not.toThrow();
    expect(() => create(EQUIDISTANT)).not.toThrow();
    expect(() => create(undefined, EQUIDISTANT)).not.toThrow();
    expect(() => create()).not.toThrow();
  });
}
