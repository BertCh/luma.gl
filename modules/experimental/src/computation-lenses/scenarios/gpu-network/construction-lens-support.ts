// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationCapture, GPUComputationKeySpace} from '@luma.gl/gpgpu';
import {createGPULocalFrame} from '../../../analysis-core/contracts/local-frame';
import crashes from '../../data/n-chicago-loop-crashes.json';
import odFlows from '../../data/n-chicago-od-flows.json';
import polylines from '../../data/n-chicago-loop-polylines.json';
import traces from '../../data/n-chicago-loop-traces.json';
import type {NetworkRoads} from './network-element-lens';
import {loadChicagoLoopRoads} from './network-lens-support';

/** Raw Chicago Loop centrelines for noding: lon/lat, 16 streets (8 east-west, 8 north-south). */
export const LOOP_POLYLINES = polylines;
/** SIMULATED GPS traces inside the Chicago Loop (synthetic, not real vehicles). */
export const LOOP_TRACES = traces;
/** Sampled 2023 traffic crashes inside the Chicago Loop (lon/lat). */
export const LOOP_CRASHES = crashes;
/** Taxi flows between Chicago community areas: annual rows (bundling) and hourly rows (aggregation). */
export const CHICAGO_OD_FLOWS = odFlows;

/** Dataset ids and licences of the construction-family scenarios. */
export const CONSTRUCTION_DATASETS = {
  polylines: {
    id: 'n-chicago-loop-polylines',
    licence: 'ODbL 1.0',
    source: 'OpenStreetMap via BBBike Chicago extract (raw centrelines)'
  },
  traces: {
    id: 'n-chicago-loop-traces',
    licence: 'Synthetic, derived from OpenStreetMap data (ODbL 1.0)',
    source: 'SIMULATED GPS traces on OpenStreetMap roads (not real vehicles)'
  },
  crashes: {
    id: 'n-chicago-loop-crashes',
    licence: 'City of Chicago Data Portal Terms of Use',
    source: 'City of Chicago traffic crashes 2023 (sampled)'
  },
  odFlows: {
    id: 'n-chicago-od-flows',
    licence: 'City of Chicago Data Portal Terms of Use',
    source: 'City of Chicago taxi trips 2023 between community areas'
  }
} as const;

/** Start intersection shared by the cost-table scenario. */
export function getLoopCenterNode(): number {
  return loadChicagoLoopRoads().startNode;
}

/**
 * Local azimuthal-equidistant metre frame at the centre of the Loop bounding box. Planar elements
 * (noding, map matching) take positions projected into it; the projection is data preparation.
 */
export function createLoopFrame() {
  const [west, south, east, north] = loadChicagoLoopRoads().bbox;
  return createGPULocalFrame({origin: [(west + east) / 2, (south + north) / 2]});
}

/** Projects the road node positions into the Loop metre frame; the CSR and weights are unchanged. */
export function projectLoopRoads(data: NetworkRoads): NetworkRoads {
  return {...data, nodes: Array.from(createLoopFrame().projectArray(data.nodes))};
}

/**
 * Makes the road extract a symmetric, simple network for the K function: every physical link
 * appears once in each direction with the same length, and parallel links keep the shorter one.
 * Weights are `lengthMeters`; travel time and class are carried over from the shorter link.
 * This is data preparation, stated in the scenario's choices.
 */
export function symmetrizeRoads(data: NetworkRoads): NetworkRoads {
  const links = new Map<
    number,
    {low: number; high: number; length: number; time: number; cls: number}
  >();
  for (let source = 0; source < data.nodeCount; source++)
    for (let edge = data.offsets[source]; edge < data.offsets[source + 1]; edge++) {
      const target = data.neighbors[edge];
      if (target === source) continue;
      const [low, high] = source < target ? [source, target] : [target, source];
      const key = low * data.nodeCount + high;
      const length = data.lengthMeters[edge];
      const known = links.get(key);
      if (!known || length < known.length)
        links.set(key, {
          low,
          high,
          length,
          time: data.travelTimeSeconds[edge],
          cls: data.roadClass[edge]
        });
    }
  const directed = [...links.values()].flatMap(link => [
    {source: link.low, target: link.high, link},
    {source: link.high, target: link.low, link}
  ]);
  directed.sort((a, b) => a.source - b.source || a.target - b.target);
  const offsets = new Array<number>(data.nodeCount + 1).fill(0);
  for (const {source} of directed) offsets[source + 1]++;
  for (let node = 0; node < data.nodeCount; node++) offsets[node + 1] += offsets[node];
  return {
    ...data,
    edgeCount: directed.length,
    offsets,
    neighbors: directed.map(edge => edge.target),
    travelTimeSeconds: directed.map(edge => edge.link.time),
    lengthMeters: directed.map(edge => edge.link.length),
    roadClass: directed.map(edge => edge.link.cls)
  };
}

/** Total network length with every physical link counted once: half of the symmetric weight sum. */
export function getUndirectedLength(data: NetworkRoads): number {
  return data.lengthMeters.reduce((sum, length) => sum + length, 0) / 2;
}

/** Key spaces for scenarios that do not run on the road graph. */
export function getKeySpaces(
  spaces: readonly {id: string; label: string; size: number}[]
): GPUComputationKeySpace[] {
  return spaces.map(space => ({...space, indexSpace: '1d' as const}));
}

/** Returns a capture whose values are rows `firstRow` up to `endRow` of another capture. */
export function sliceCapture(
  capture: GPUComputationCapture,
  firstRow: number,
  endRow: number,
  props: Partial<GPUComputationCapture> & {id: string}
): GPUComputationCapture {
  const components = capture.components ?? 1;
  const Values = capture.elementType === 'f32' ? Float32Array : Uint32Array;
  return {
    ...capture,
    ...props,
    name: props.id,
    values: Values.from(
      Array.from(capture.values).slice(firstRow * components, endRow * components)
    )
  };
}

/** Gathers the 2-component rows of a position capture at node ids (positions are a readback). */
export function gatherRows(
  positions: GPUComputationCapture,
  nodes: ArrayLike<number>
): Float32Array {
  return Float32Array.from(
    Array.from(nodes).flatMap(node => [positions.values[node * 2], positions.values[node * 2 + 1]])
  );
}
