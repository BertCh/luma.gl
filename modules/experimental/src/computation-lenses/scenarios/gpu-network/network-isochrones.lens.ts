// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getGPUNetworkIsochronesParameterValues} from '@luma.gl/experimental/gpu-network';
import {createGPULocalFrame} from '../../../analysis-core/contracts/local-frame';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario,
  type NetworkRoads
} from './network-element-lens';
import {runExample} from './network-isochrones.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {gatherNodePoints, getSearchKeySpaces} from './search-lens-support';

const scenarioId = 'gpu-network.network-isochrones';
const roads = loadChicagoLoopRoads();
/** Cost breaks in seconds; band k holds costs in [breaks[k - 1], breaks[k]). */
const COST_BREAKS = [90, 180, 300];
/** Walk cost in seconds per meter off the network (about 1.4 m/s). */
const WALK_SECONDS_PER_METER = 0.7;
const TRIANGLE_CAPACITY = 16384;

// Local azimuthal-equidistant frame in meters at the centre of the file's bounding box: the raster
// path is planar, so longitude/latitude are projected on the CPU before upload (data preparation).
const [west, south, east, north] = roads.bbox;
const frame = createGPULocalFrame({origin: [(west + east) / 2, (south + north) / 2]});

function projectRoads(data: NetworkRoads): NetworkRoads {
  return {...data, nodes: Array.from(frame.projectArray(data.nodes))};
}

// A square extent around the projected nodes, so the 48 x 48 pixels are square.
function getExtent(): [number, number, number, number] {
  const planar = frame.projectArray(roads.nodes);
  let [minX, minY, maxX, maxY] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let node = 0; node < planar.length; node += 2) {
    minX = Math.min(minX, planar[node]);
    maxX = Math.max(maxX, planar[node]);
    minY = Math.min(minY, planar[node + 1]);
    maxY = Math.max(maxY, planar[node + 1]);
  }
  const half = Math.max(maxX - minX, maxY - minY) / 2 + 50;
  const [centerX, centerY] = [(minX + maxX) / 2, (minY + maxY) / 2];
  return [centerX - half, centerY - half, centerX + half, centerY + half];
}

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  prepareRoads: projectRoads,
  coordinateModel: 'projected',
  seed: roads.startNode,
  retain: ['costs', 'converged', 'triangles', 'triangleBands', 'count', 'overflow'],
  formulas: ['gpu-network.isochrones.cost'],
  keySpaces: getSearchKeySpaces(roads, 0),
  entityKeySpace: 'node',
  queuesResource: 'network-isochrones-search-queues',
  choices: [
    {
      stage: 'impedance',
      name: 'walkSecondsPerMeter',
      value: WALK_SECONDS_PER_METER,
      alternatives: [0],
      sweepable: true
    }
  ],
  async build(context) {
    const startNodes = context.upload('startNodes', Uint32Array.of(roads.startNode), 'uint32');
    const costBreaks = context.upload('costBreaks', Float32Array.from(COST_BREAKS), 'float32');
    const parameters = context.upload(
      'parameters',
      getGPUNetworkIsochronesParameterValues({
        breakCount: COST_BREAKS.length,
        extent: getExtent(),
        bufferRadius: 30,
        walkCostPerUnit: WALK_SECONDS_PER_METER
      }),
      'float32'
    );
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      startNodes,
      costBreaks,
      parameters,
      triangleCapacity: TRIANGLE_CAPACITY
    });
    return {
      views: {
        costs: example.costs,
        converged: example.converged,
        triangles: example.triangles,
        triangleBands: example.triangleBands,
        count: example.status.count!,
        overflow: example.status.overflow!
      }
    };
  },
  capture(final) {
    if (final('converged').values[0] === 0)
      throw new Error('Network isochrones: the cost search did not converge');
    if (final('overflow').values[0] !== 0)
      throw new Error('Network isochrones: the triangle output overflowed');
    const positions = final('nodePositions');
    const count = final('count').values[0];
    const triangles = final('triangles');
    const bands = final('triangleBands');
    return [
      gatherNodePoints(positions, [roads.startNode], {
        id: 'start-intersection',
        coordinateModel: 'projected',
        unit: 'm',
        meaning:
          'input: start intersection (position gathered from the nodePositions readback; local metre frame)'
      }),
      {...final('costs'), keySpace: 'node'},
      {
        ...bands,
        values: Uint32Array.from(Array.from(bands.values).slice(0, count)),
        meaning: 'band of each emitted triangle'
      },
      {
        ...triangles,
        id: 'triangle-offsets',
        name: 'triangle-offsets',
        elementType: 'u32',
        components: undefined,
        values: Uint32Array.from({length: count + 1}, (_, triangle) => triangle * 3),
        meaning:
          'derived: triangle k spans vertices 3k to 3k + 3 (triangle count from the GPU count readback)'
      },
      {
        ...triangles,
        id: 'isochrone-triangles',
        name: 'isochrone-triangles',
        values: Float32Array.from(Array.from(triangles.values).slice(0, count * 6)),
        role: 'output',
        meaning:
          'output: isoband triangles of the 48 x 48 sampled cost raster, coloured by travel-time band (local metre frame)',
        geometry: {
          kind: 'polygons',
          coordinateModel: 'projected',
          unit: 'm',
          offsetsCapture: 'triangle-offsets',
          valuesCapture: 'triangleBands'
        }
      }
    ];
  }
};

/** Executes the isochrone raster on the Chicago Loop roads and reads the triangles back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
