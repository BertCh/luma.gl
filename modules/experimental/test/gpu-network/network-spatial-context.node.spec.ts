// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  getGPUGreatCircleRadius,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  GPU_PLANAR_SPATIAL_CONTEXT,
  type GPUSpatialContext
} from '../../src/analysis-core/contracts/spatial-context';
import {createGPULocalFrame} from '../../src/analysis-core/contracts/local-frame';
import {getNetworkMetricWGSL} from '../../src/gpu-network/network-spatial-context';
import {GPUEdgeBundling} from '../../src/gpu-network/edge-bundling';
import {
  GPU_MAP_MATCHING_PARAMETER_LENGTH,
  GPUMapMatching
} from '../../src/gpu-network/map-matching/index';
import {GPUNetworkSnapping} from '../../src/gpu-network/network-accessibility/index';
import {
  GPUNetworkIsochrones,
  GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH,
  type GPUNetworkIsochronesProps
} from '../../src/gpu-network/network-isochrones';
import {GPUNetworkKFunction} from '../../src/gpu-network/network-k-function/index';
import {
  GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH,
  GPUNetworkLineGraph
} from '../../src/gpu-network/network-line-graph/index';
import {GPUNetworkNoding} from '../../src/gpu-network/network-noding/index';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {
  getLineGraphEdgeDirections,
  getLineGraphTurnAngle
} from './network-line-graph/network-line-graph-oracle';
import {createGreatCircleNetworkMetric, getHaversineDistance} from './network-metric-oracle';

const GREAT_CIRCLE: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle'
};
const PLANAR: GPUSpatialContext = {coordinateSpace: 'planar', metric: 'native'};

const KILOMETERS: GPUSpatialContext = {...GREAT_CIRCLE, units: 'kilometers'};
const EQUIDISTANT = createGPULocalFrame({origin: [10, 60]}).spatialContext;
const EQUAL_AREA = createGPULocalFrame({
  origin: [10, 60],
  kind: 'lambert-equal-area'
}).spatialContext;

function expectDistanceFrames(
  id: string,
  createContributor: (spatialContext: GPUSpatialContext) => {spatialContext: GPUSpatialContext}
): void {
  expect(EQUIDISTANT.units).toBe('meters');
  expect(createContributor(EQUIDISTANT).spatialContext).toBe(EQUIDISTANT);
  expect(() => createContributor(EQUAL_AREA)).toThrow(
    `${id} measures distance; project into an azimuthal-equidistant frame, not lambert-equal-area`
  );
}

let serial = 0;

function createView<Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4' | 'uint32x2'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${serial++}`, format, length);
}

// Reference values from python3 -I with the math module, R = 6371008.8 m:
//   haversine(a, b) = 2 R asin(sqrt(sin^2(dphi / 2) + cos(phi1) cos(phi2) sin^2(dlambda / 2)))
//   exact sphere point-to-arc distance = cross-track |asin(sin(d13) sin(t13 - t12))| R, with the
//   along-track fraction acos(cos(d13) / cos(dxt)) R / length(a, b)
//   turn angle = atan2(cross, dot) of the arrival direction at v (initial bearing v -> u plus pi)
//   and the departure direction at v (initial bearing v -> w), directions (sin t, cos t)
describe('great-circle network metric oracle', () => {
  it('matches python haversine distances, including across the antimeridian', () => {
    expect(getHaversineDistance([0, 0], [1, 0])).toBeCloseTo(111195.0802335329, 6);
    expect(getHaversineDistance([179.999, 10], [-179.999, 10])).toBeCloseTo(219.01155422501236, 6);
    expect(getHaversineDistance([10, 60], [10.01, 60])).toBeCloseTo(555.9754006384692, 6);
  });

  it('snaps in the local frame within centimeters of the exact sphere distance', () => {
    const metric = createGreatCircleNetworkMetric();
    const cases = [
      // 1.24 km edge at 60 N, point 50 m off: local frame (python mirror) 49.72726005937693 m at
      // 0.4200024186305257; exact great-circle arc 49.672330880686815 m at 0.4200181387181484.
      // The 5.5 cm gap is the bow between a straight degree segment and the great circle,
      // L^2 tan(lat) / (8 R), negligible for road-length edges against GPS noise.
      {
        point: [10.008, 60.0025],
        start: [10, 60],
        end: [10.02, 60.005],
        mirror: [49.72726005937693, 0.4200024186305257],
        exact: [49.672330880686815, 0.4200181387181484]
      },
      // Edge across the antimeridian at 20 S: mirror 55.59754011662153 m at 0.4000000000005684;
      // exact 55.60502494079836 m at 0.3999996872109345.
      {
        point: [179.999, -19.9995],
        start: [179.995, -20],
        end: [-179.995, -20],
        mirror: [55.59754011662153, 0.4000000000005684],
        exact: [55.60502494079836, 0.3999996872109345]
      }
    ];
    for (const {point, start, end, mirror, exact} of cases) {
      const fraction = metric.getSegmentFraction(point, start, end);
      const distance = metric.getDistance(point, metric.interpolate(start, end, fraction));
      expect(distance).toBeCloseTo(mirror[0], 6);
      expect(fraction).toBeCloseTo(mirror[1], 9);
      expect(Math.abs(distance - exact[0])).toBeLessThan(0.1);
      expect(Math.abs(fraction - exact[1])).toBeLessThan(1e-4);
    }
  });

  it('measures line graph turns between great-circle bearings', () => {
    // u = (10, 60) -> v = (10.01, 60) -> w = (10.02, 60.005). Python: 0.7855871032120485 rad
    // (about 45 degrees); the same turn read in raw degrees is atan2(0.005, 0.01) =
    // 0.4636476090008061 rad, the east-west stretch of degrees at 60 N.
    const base = {
      offsets: Uint32Array.of(0, 1, 2, 2),
      neighbors: Uint32Array.of(1, 2),
      weights: Float32Array.of(1, 1)
    };
    const positions = [10, 60, 10.01, 60, 10.02, 60.005];
    const geodesic = getLineGraphEdgeDirections(base, positions, true);
    expect(getLineGraphTurnAngle(geodesic.arrivals[0], geodesic.departures[1])).toBeCloseTo(
      0.7855871032120485,
      9
    );
    const planar = getLineGraphEdgeDirections(base, positions);
    expect(getLineGraphTurnAngle(planar.arrivals[0], planar.departures[1])).toBeCloseTo(
      0.4636476090008061,
      6
    );
  });
});

describe('network contributor spatial contexts', () => {
  it('emits the context radius in kilometers without changing the default meter radius', () => {
    expect(getNetworkMetricWGSL(getGPUGreatCircleRadius(KILOMETERS))).toContain(
      `const NETWORK_EARTH_RADIUS: f32 = ${Math.fround(6371.0088)};`
    );
    expect(getNetworkMetricWGSL(getGPUGreatCircleRadius(GREAT_CIRCLE))).toContain(
      `const NETWORK_EARTH_RADIUS: f32 = ${Math.fround(6371008.8)}.0;`
    );
    expect(getNetworkMetricWGSL(undefined)).not.toContain('NETWORK_EARTH_RADIUS');
  });

  it('GPUNetworkSnapping accepts great-circle lon/lat on both candidate paths', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const props = {
      positions: createView(graph, 'float32x2', 3),
      topology: {
        offsets: createView(graph, 'uint32', 5),
        neighbors: createView(graph, 'uint32', 5),
        nodePositions: createView(graph, 'float32x2', 4)
      },
      output: {
        snappedEdges: createView(graph, 'uint32', 3),
        snapDistances: createView(graph, 'float32', 3),
        snappedPositions: createView(graph, 'float32x2', 3),
        sourceCosts: createView(graph, 'float32', 3),
        targetCosts: createView(graph, 'float32', 3)
      }
    };
    expect(new GPUNetworkSnapping(props).spatialContext.coordinateSpace).toBe('planar');
    const snapping = new GPUNetworkSnapping({...props, spatialContext: GREAT_CIRCLE});
    expect(snapping.spatialContext).toEqual(GREAT_CIRCLE);
    // The edge sources are expanded from `topology.offsets` before the scan.
    expect(snapping.getCommandNodes(graph).map(node => node.id)).toEqual([
      'network-snapping-edge-sources-clear',
      'network-snapping-edge-sources',
      'network-snapping-scan',
      'network-snapping-costs',
      'network-snapping-positions'
    ]);
    expect(
      () =>
        new GPUNetworkSnapping({
          ...props,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
        })
    ).toThrow(/great-circle metric/);
    expectDistanceFrames(
      'network-snapping',
      spatialContext => new GPUNetworkSnapping({...props, spatialContext})
    );
    const kilometers = new GPUNetworkSnapping({
      ...props,
      id: 'kilometers',
      spatialContext: KILOMETERS
    });
    expect(kilometers.spatialContext).toBe(KILOMETERS);
    expect(kilometers.getCommandNodes(graph).map(node => node.id)).toContain('kilometers-scan');
    // The BVH candidate path passes the context to GPUNearestFeatureJoin (great-circle search).
    const joined = new GPUNetworkSnapping({
      ...props,
      id: 'joined',
      spatialContext: GREAT_CIRCLE,
      candidateCapacity: 4,
      maximumSnapDistance: createView(graph, 'float32', 1),
      output: {...props.output, status: {overflow: createView(graph, 'uint32', 1)}}
    }).getCommandNodes(graph);
    const kilometerCandidates = new GPUNetworkSnapping({
      ...props,
      id: 'kilometer-candidates',
      spatialContext: KILOMETERS,
      candidateCapacity: 4,
      maximumSnapDistance: createView(graph, 'float32', 1),
      output: {...props.output, status: {overflow: createView(graph, 'uint32', 1)}}
    }).getCommandNodes(graph);
    // Kilometer contexts take the same join path (the join honours the context's units).
    expect(kilometerCandidates.some(node => node.id.startsWith('kilometer-candidates-join'))).toBe(
      true
    );
    const joinedIds = joined.map(node => node.id);
    expect(joinedIds.some(id => id.startsWith('joined-join'))).toBe(true);
    expect(joinedIds).toContain('joined-project');
    expect(
      () =>
        new GPUNetworkSnapping({
          ...props,
          spatialContext: {coordinateSpace: 'topology-only', metric: 'none'}
        })
    ).toThrow(/does not support topology-only/);
  });

  it('GPUMapMatching accepts great-circle lon/lat and rejects degrees without it', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const props = {
      positions: createView(graph, 'float32x2', 6),
      trackOffsets: createView(graph, 'uint32', 2),
      topology: {
        offsets: createView(graph, 'uint32', 5),
        neighbors: createView(graph, 'uint32', 6),
        nodePositions: createView(graph, 'float32x2', 4)
      },
      parameters: createView(graph, 'float32', GPU_MAP_MATCHING_PARAMETER_LENGTH),
      cellSize: 0.001,
      bounds: {minimum: [179.99, 59.99], maximum: [180.01, 60.01]} as const,
      output: {
        matchedEdges: createView(graph, 'uint32', 6),
        snapDistances: createView(graph, 'float32', 6),
        snappedPositions: createView(graph, 'float32x2', 6)
      }
    };
    expectDistanceFrames(
      'map-matching',
      spatialContext => new GPUMapMatching({...props, spatialContext})
    );
    const kilometers = new GPUMapMatching({...props, id: 'kilometers', spatialContext: KILOMETERS});
    expect(kilometers.spatialContext).toBe(KILOMETERS);
    expect(kilometers.getCommandNodes(graph).length).toBeGreaterThan(0);
    const matching = new GPUMapMatching({...props, spatialContext: GREAT_CIRCLE});
    expect(matching.spatialContext).toEqual(GREAT_CIRCLE);
    expect(matching.getCommandNodes(graph).map(node => node.id)).toContain(
      'map-matching-positions'
    );
    expect(new GPUMapMatching(props).spatialContext.coordinateSpace).toBe('planar');
    expect(
      () =>
        new GPUMapMatching({
          ...props,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'rhumb'}
        })
    ).toThrow(/great-circle metric/);
  });

  it('GPUNetworkKFunction passes the context to snapping, including BVH candidates', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const props = {
      positions: createView(graph, 'float32x2', 10),
      topology: {
        offsets: createView(graph, 'uint32', 7),
        neighbors: createView(graph, 'uint32', 12),
        weights: createView(graph, 'float32', 12),
        nodePositions: createView(graph, 'float32x2', 6)
      },
      maximumDistance: createView(graph, 'float32', 1),
      networkLength: createView(graph, 'float32', 1),
      parameters: createView(graph, 'uint32', 4),
      bandCount: 8,
      rowsPerBlock: 10,
      output: {kValues: createView(graph, 'float32', 8)}
    };
    expectDistanceFrames(
      'network-k-function',
      spatialContext => new GPUNetworkKFunction({...props, spatialContext})
    );
    const kilometers = new GPUNetworkKFunction({
      ...props,
      id: 'kilometers',
      spatialContext: KILOMETERS
    });
    expect(kilometers.spatialContext).toBe(KILOMETERS);
    expect(kilometers.getCommandNodes(graph).length).toBeGreaterThan(0);
    const kFunction = new GPUNetworkKFunction({...props, spatialContext: GREAT_CIRCLE});
    expect(kFunction.spatialContext).toEqual(GREAT_CIRCLE);
    expect(kFunction.getCommandNodes(graph).map(node => node.id)).toContain(
      'network-k-function-snap-scan'
    );
    const joined = new GPUNetworkKFunction({
      ...props,
      id: 'joined-k',
      spatialContext: GREAT_CIRCLE,
      candidateCapacity: 4,
      maximumSnapDistance: createView(graph, 'float32', 1),
      output: {...props.output, status: {overflow: createView(graph, 'uint32', 1)}}
    }).getCommandNodes(graph);
    expect(joined.map(node => node.id)).toContain('joined-k-snap-project');
    const kilometerCandidates = new GPUNetworkKFunction({
      ...props,
      id: 'kilometer-k',
      spatialContext: KILOMETERS,
      candidateCapacity: 4,
      maximumSnapDistance: createView(graph, 'float32', 1),
      output: {...props.output, status: {overflow: createView(graph, 'uint32', 1)}}
    }).getCommandNodes(graph);
    expect(kilometerCandidates.map(node => node.id)).toContain('kilometer-k-snap-project');
  });

  it('GPUNetworkLineGraph accepts great-circle lon/lat', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const props = {
      topology: {
        offsets: createView(graph, 'uint32', 4),
        neighbors: createView(graph, 'uint32', 4),
        weights: createView(graph, 'float32', 4),
        nodePositions: createView(graph, 'float32x2', 3)
      },
      parameters: createView(graph, 'float32', GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH),
      output: {
        topology: {
          offsets: createView(graph, 'uint32', 5),
          neighbors: createView(graph, 'uint32', 8),
          weights: createView(graph, 'float32', 8)
        },
        status: {overflow: createView(graph, 'uint32', 1)}
      }
    };
    expectDistanceFrames(
      'network-line-graph',
      spatialContext => new GPUNetworkLineGraph({...props, spatialContext})
    );
    const kilometers = new GPUNetworkLineGraph({
      ...props,
      id: 'kilometers',
      spatialContext: KILOMETERS
    });
    expect(kilometers.spatialContext).toBe(KILOMETERS);
    expect(kilometers.getCommandNodes(graph).length).toBeGreaterThan(0);
    const lineGraph = new GPUNetworkLineGraph({...props, spatialContext: GREAT_CIRCLE});
    expect(lineGraph.spatialContext).toEqual(GREAT_CIRCLE);
    expect(lineGraph.getCommandNodes(graph)[0].id).toBe('network-line-graph-edge-vectors');
    expect(
      () =>
        new GPUNetworkLineGraph({
          ...props,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
        })
    ).toThrow(/great-circle metric/);
  });

  it('GPUNetworkNoding is planar only', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    expect(
      () =>
        new GPUNetworkNoding({
          spatialContext: GREAT_CIRCLE,
          tolerance: createView(graph, 'float32', 1)
        } as never)
    ).toThrow(/does not support longitude-latitude coordinates; project positions to a planar/);
  });

  it('GPUNetworkNoding reduces the snap origin before keying end points', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const pieceCapacity = 8;
    const nodeCapacity = 8;
    const noding = new GPUNetworkNoding({
      id: 'noding',
      lines: {
        kind: 'lines',
        positions: createView(graph, 'float32x2', 8),
        lineOffsets: createView(graph, 'uint32', 4)
      },
      intersectionCapacity: 16,
      tolerance: createView(graph, 'float32', 1),
      output: {
        pieces: {
          geometry: {
            kind: 'lines',
            positions: createView(graph, 'float32x2', 32),
            lineOffsets: createView(graph, 'uint32', pieceCapacity + 1)
          },
          sourceIds: createView(graph, 'uint32', pieceCapacity),
          status: {count: createView(graph, 'uint32', 1), overflow: createView(graph, 'uint32', 1)}
        },
        nodes: {
          positions: createView(graph, 'float32x2', nodeCapacity),
          status: {count: createView(graph, 'uint32', 1)}
        },
        edges: {
          fromNodes: createView(graph, 'uint32', pieceCapacity),
          toNodes: createView(graph, 'uint32', pieceCapacity),
          lengths: createView(graph, 'float32', pieceCapacity)
        },
        topology: {
          offsets: createView(graph, 'uint32', nodeCapacity + 1),
          neighbors: createView(graph, 'uint32', pieceCapacity * 2),
          weights: createView(graph, 'float32', pieceCapacity * 2)
        },
        status: {overflow: createView(graph, 'uint32', 1)}
      }
    });
    expectDistanceFrames(
      'noding',
      spatialContext => new GPUNetworkNoding({...noding.props, spatialContext})
    );
    expect(() => new GPUNetworkNoding({...noding.props, spatialContext: KILOMETERS})).toThrow(
      /does not support longitude-latitude/
    );
    const ids = noding.getCommandNodes(graph).map(node => node.id);
    const order = [
      'noding-clear-snap-origin',
      'noding-clear-snap-overflow',
      'noding-endpoints',
      'noding-endpoint-keys',
      'noding-scalars'
    ].map(id => ids.indexOf(id));
    expect(order.every(position => position >= 0)).toBe(true);
    expect([...order].sort((a, b) => a - b)).toEqual(order);
  });

  it('GPUEdgeBundling maps a lon/lat context to its degree-aware work box', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const props = {
      nodePositions: createView(graph, 'float32x2', 10),
      sourceNodes: createView(graph, 'uint32', 6),
      targetNodes: createView(graph, 'uint32', 6),
      output: {paths: createView(graph, 'float32x2', 6 * 16)}
    };
    for (const spatialContext of [EQUIDISTANT, EQUAL_AREA, KILOMETERS]) {
      expect(new GPUEdgeBundling({...props, spatialContext}).spatialContext).toBe(spatialContext);
    }
    const getCoordinateSpace = (bundling: GPUEdgeBundling) =>
      bundling.spatialContext.coordinateSpace;
    expect(getCoordinateSpace(new GPUEdgeBundling(props))).toBe('planar');
    expect(getCoordinateSpace(new GPUEdgeBundling({...props, spatialContext: GREAT_CIRCLE}))).toBe(
      'longitude-latitude'
    );
    expect(
      getCoordinateSpace(
        new GPUEdgeBundling({
          ...props,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
        })
      )
    ).toBe('longitude-latitude');
    expect(
      () =>
        new GPUEdgeBundling({
          ...props,
          spatialContext: {coordinateSpace: 'topology-only', metric: 'none'}
        })
    ).toThrow(/does not support topology-only/);
  });

  it('GPUNetworkIsochrones keeps the raster planar and the cell outline lon/lat', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const raster: GPUNetworkIsochronesProps['raster'] = {
      width: 8,
      height: 8,
      output: {
        triangles: createView(graph, 'float32x2', 3 * 64),
        triangleBands: createView(graph, 'uint32', 64),
        status: {
          count: createView(graph, 'uint32', 1),
          overflow: createView(graph, 'uint32', 1)
        }
      }
    };
    const cellOutline: GPUNetworkIsochronesProps['cellOutline'] = {
      family: 'h3',
      resolution: 8,
      output: {
        rows: createView(graph, 'uint32', 16),
        cells: createView(graph, 'uint32x2', 16),
        edgeIndices: createView(graph, 'uint32', 16),
        endpoints: createView(graph, 'float32x4', 16),
        status: {count: createView(graph, 'uint32', 1), overflow: createView(graph, 'uint32', 1)}
      }
    };
    const props: GPUNetworkIsochronesProps = {
      topology: {
        offsets: createView(graph, 'uint32', 5),
        neighbors: createView(graph, 'uint32', 6),
        weights: createView(graph, 'float32', 6),
        nodePositions: createView(graph, 'float32x2', 4)
      },
      costs: createView(graph, 'float32', 4),
      costBreaks: createView(graph, 'float32', 3),
      parameters: createView(graph, 'float32', GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH)
    };
    expectDistanceFrames(
      'network-isochrones',
      spatialContext => new GPUNetworkIsochrones({...props, raster, spatialContext})
    );
    expect(new GPUNetworkIsochrones({...props, raster}).spatialContext).toBe(
      GPU_PLANAR_SPATIAL_CONTEXT
    );
    expect(new GPUNetworkIsochrones({...props, cellOutline}).spatialContext).toBe(
      GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT
    );
    const combined = new GPUNetworkIsochrones({...props, raster, cellOutline});
    expect(combined.spatialContext).toBe(GPU_PLANAR_SPATIAL_CONTEXT);
    expect(combined.getCommandNodes(graph).length).toBeGreaterThan(0);
    expect(
      () => new GPUNetworkIsochrones({...props, raster, cellOutline, spatialContext: PLANAR})
    ).toThrow(/cellOutline needs longitude-latitude/);
    expect(
      () => new GPUNetworkIsochrones({...props, raster, cellOutline, spatialContext: GREAT_CIRCLE})
    ).toThrow(/raster needs planar positions/);
    expect(
      new GPUNetworkIsochrones({...props, cellOutline, spatialContext: KILOMETERS}).spatialContext
    ).toBe(KILOMETERS);
    expect(
      new GPUNetworkIsochrones({...props, raster, spatialContext: PLANAR}).spatialContext
    ).toEqual(PLANAR);
    expect(
      () => new GPUNetworkIsochrones({...props, raster, spatialContext: GREAT_CIRCLE})
    ).toThrow(/raster needs planar positions/);
    expect(
      new GPUNetworkIsochrones({...props, cellOutline, spatialContext: GREAT_CIRCLE}).spatialContext
    ).toEqual(GREAT_CIRCLE);
    expect(() => new GPUNetworkIsochrones({...props, cellOutline, spatialContext: PLANAR})).toThrow(
      /cellOutline needs longitude-latitude/
    );
  });
});
