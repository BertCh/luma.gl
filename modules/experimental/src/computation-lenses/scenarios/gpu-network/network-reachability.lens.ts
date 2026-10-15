// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, type Device, type Buffer} from '@luma.gl/core';
import {
  captureGPUCommandGraph,
  createGPUComputationCaptures,
  createGPUComputationMetrics,
  validateGPUComputationSidecars,
  type GPUComputationCapture,
  type GPUComputationRecordWithSidecars
} from '@luma.gl/gpgpu';
import type {CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {NETWORK_FORMULA_REFS} from '../../../gpu-network/computation-pack/network-formulas';
import {runExample} from './network-reachability.example';
import {
  assembleNetworkLensRecord,
  captureInputGeometry,
  createNetworkLensGraph,
  createNetworkRelations,
  findNetworkResource,
  getNetworkFinalCapture,
  loadChicagoLoopRoads,
  NETWORK_KEY_SPACES,
  uploadNetworkTopology,
  uploadNetworkView
} from './network-lens-support';

/** Fixed round budget: the single-hop proof must converge or fail visibly. */
export const NETWORK_REACHABILITY_MAXIMUM_ITERATIONS = 256;

const contributorId = 'reachability';
const retained = [
  'costs',
  'bands',
  'predecessors',
  'bandCounts',
  'converged',
  'iterationCount',
  `${contributorId}-queued-round`,
  `${contributorId}-control`
];

async function buildScenario(device: Device, chained: boolean, structureOnly: boolean) {
  const scenarioId = `gpu-network.network-reachability${chained ? '.chained' : ''}`;
  const graph = createNetworkLensGraph(device, scenarioId, retained);
  const data = loadChicagoLoopRoads();
  const ownedBuffers: Buffer[] = [];
  let compiled: CompiledGPUCommandGraph | undefined;
  try {
    const topology = uploadNetworkTopology(device, graph, data, {weights: 'travelTimeSeconds'});
    ownedBuffers.push(...topology.ownedBuffers);
    const startNodes = uploadNetworkView(
      device,
      graph,
      'startNodes',
      Uint32Array.of(data.startNode),
      'uint32'
    );
    ownedBuffers.push(startNodes.buffer);
    const activeIterations = uploadNetworkView(
      device,
      graph,
      'activeIterations',
      Uint32Array.of(NETWORK_REACHABILITY_MAXIMUM_ITERATIONS),
      'uint32'
    );
    ownedBuffers.push(activeIterations.buffer);
    const costBreaks = uploadNetworkView(
      device,
      graph,
      'costBreaks',
      Float32Array.of(60, 120, 180, 240, 300),
      'float32'
    );
    ownedBuffers.push(costBreaks.buffer);
    const example = await runExample(device, {
      graph,
      roads: topology,
      startNodes: startNodes.view,
      activeIterations: activeIterations.view,
      costBreaks: costBreaks.view,
      ...(chained ? {} : {localIterations: 1}),
      structureOnly
    });
    compiled = example.compiled;
    const description = compiled.describe({domain: 'gpu-network'});
    const localIterations = example.reachability.localIterations;
    const record = assembleNetworkLensRecord(device, {
      id: scenarioId,
      seed: data.startNode,
      variant: chained ? 'chained' : 'single-hop',
      graphs: [
        {
          ...description,
          graph: {
            ...description.graph,
            attrs: {
              ...description.graph.attrs,
              localIterations,
              maximumIterations: NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
              observation: chained
                ? 'end capture only'
                : 'step capture splits queue submissions; replay uses the same compiled graph'
            }
          }
        }
      ],
      formulas: Object.values(NETWORK_FORMULA_REFS),
      choices: [
        {
          stage: 'search',
          name: 'localIterations',
          value: localIterations,
          alternatives: [chained ? 1 : 32],
          sweepable: true
        }
      ],
      axes: {
        axes: [
          {
            id: 'entity',
            label: 'network node',
            kind: 'data',
            keySpace: 'node',
            size: data.nodeCount
          },
          {
            id: 'iteration',
            label: 'completed relax rounds',
            kind: 'execution',
            size: NETWORK_REACHABILITY_MAXIMUM_ITERATIONS + 1
          }
        ],
        keySpaces: NETWORK_KEY_SPACES,
        relations: createNetworkRelations(topology, `${contributorId}-queues`)
      }
    });
    return {example, compiled, record, topology, activeIterations, data, ownedBuffers};
  } catch (error) {
    compiled?.destroy();
    for (const buffer of ownedBuffers) buffer.destroy();
    throw error;
  }
}

/** Converts exact queuedRound readbacks into named round stamps; rejects tie-phase contamination. */
export function makeReachabilityStamp(
  queued: GPUComputationCapture,
  maximumIterations: number,
  chained: boolean
): GPUComputationCapture {
  const values = Float32Array.from(queued.values, stamp => {
    // A later tie phase shares this buffer. Positive road weights must leave it untouched.
    assert(stamp <= maximumIterations + 1);
    return stamp >= 1 ? stamp - 1 : NaN;
  });
  return {
    ...queued,
    id: chained ? 'lastQueuedRound' : 'lastImprovedRound',
    name: chained ? 'lastQueuedRound' : 'lastImprovedRound',
    keySpace: 'node',
    elementType: 'f32',
    values,
    meaning: chained
      ? 'last global-queue round (chained hops inside a round are not stamped)'
      : 'last-improved round (frontier queuedRound stamp − 1; exact because localIterations = 1)'
  };
}

/**
 * Selects seeded state and executed relax prefixes as log-spaced key frames: rounds 0, 1, 2 and
 * every power of two, plus the final state. Rounds that only repeat the final state are dropped.
 * Other executed rounds are not captured (the record size budget); each kept frame says so in its
 * meaning, and the per-node last-improved stamp still records exactly when every node settled.
 */
export function selectReachabilitySnapshots(
  captures: readonly GPUComputationCapture[],
  iterationCount: number
): GPUComputationCapture[] {
  const snapshots = captures
    .flatMap(capture => {
      if (capture.name !== 'costs') return [];
      const match = /^reachability-relax-(\d+)$/.exec(capture.afterNode ?? '');
      const iteration =
        capture.afterNode === 'reachability-seed' ? 0 : match ? Number(match[1]) + 1 : -1;
      return iteration >= 0 && iteration <= iterationCount
        ? [{...capture, id: `costs-round-${iteration}`, iteration, keySpace: 'node'}]
        : [];
    })
    .sort((left, right) => left.iteration - right.iteration);
  const final = snapshots.at(-1);
  if (!final) return snapshots;
  while (snapshots.length > 2) {
    const previous = snapshots[snapshots.length - 2];
    if (!Array.from(previous.values).every((value, index) => Object.is(value, final.values[index])))
      break;
    snapshots.splice(snapshots.length - 2, 1);
  }
  const isKeyFrame = (iteration: number) =>
    iteration <= 2 || (iteration & (iteration - 1)) === 0 || iteration === final.iteration;
  const keyFrames = snapshots.filter(snapshot => isKeyFrame(snapshot.iteration));
  if (keyFrames.length === snapshots.length) return snapshots;
  const meaning = `key frame: rounds 0, 1, 2, powers of two and the final round ${final.iteration} were captured; the ${snapshots.length - keyFrames.length} other executed rounds were not`;
  return keyFrames.map(snapshot => ({...snapshot, meaning}));
}

/** Runs either proof variant; all values are real GPU readbacks, with explicit GPU replay provenance. */
export async function runReachabilityScenario(
  device: Device,
  chained: boolean,
  options?: {signal?: AbortSignal}
): Promise<GPUComputationRecordWithSidecars> {
  options?.signal?.throwIfAborted();
  const built = await buildScenario(device, chained, false);
  const {compiled, example, topology, record, data, activeIterations} = built;
  try {
    const {output, commandNodes} = example;
    const views = {
      costs: output.costs,
      bands: output.bands,
      predecessors: output.predecessors,
      bandCounts: output.bandCounts,
      converged: output.status.converged,
      iterationCount: output.status.iterationCount,
      queuedRound: findNetworkResource(commandNodes, `${contributorId}-queued-round`),
      control: findNetworkResource(commandNodes, `${contributorId}-control`),
      offsets: topology.offsets,
      neighbors: topology.neighbors,
      nodePositions: topology.nodePositions
    };
    const captured = await captureGPUCommandGraph(compiled, {
      at: chained ? 'end' : 'step',
      encodeOptions: {parameters: undefined, coalesceComputePasses: false},
      views
    });
    options?.signal?.throwIfAborted();
    const final = (name: string) => getNetworkFinalCapture(captured.captures, name);
    if (final('converged').values[0] === 0)
      throw new Error('Network reachability did not converge within maximumIterations = 256');
    const iterationCount = final('iterationCount').values[0];
    // In step mode, read the stamps right after the last relax round, before the tie phase can push.
    const lastRelax = `${contributorId}-relax-${NETWORK_REACHABILITY_MAXIMUM_ITERATIONS - 1}`;
    const relaxedStamps = captured.captures.find(
      capture => capture.name === 'queuedRound' && capture.afterNode === lastRelax
    );
    const stamp = makeReachabilityStamp(
      relaxedStamps ? {...relaxedStamps, id: 'queuedRound'} : final('queuedRound'),
      NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
      chained
    );
    const snapshots = chained ? [] : selectReachabilitySnapshots(captured.captures, iterationCount);
    snapshots.push(
      ...[
        'costs',
        'bands',
        'predecessors',
        'bandCounts',
        'converged',
        'iterationCount',
        'queuedRound',
        'control'
      ].map(name => ({
        ...final(name),
        ...(['costs', 'bands', 'predecessors', 'queuedRound'].includes(name)
          ? {keySpace: 'node'}
          : {})
      }))
    );
    snapshots.push(...captureInputGeometry(captured.captures, [data.startNode]));
    const positions = final('nodePositions');
    snapshots.push({
      ...positions,
      id: 'reached-nodes',
      name: 'reached-nodes',
      keySpace: 'node',
      role: 'output',
      geometry: {kind: 'points', coordinateModel: 'geographic', valuesCapture: 'costs'},
      meaning:
        'output: all nodes coloured by final travel time; +Infinity is no data, including beyond the displayed bands'
    });
    const replays: GPUComputationCapture[] = [];
    if (!chained) {
      const rounds = Array.from(stamp.values)
        .filter(
          (value, node) => Number.isFinite(value) && Number.isFinite(final('costs').values[node])
        )
        .sort((left, right) => left - right);
      const replayRound = rounds[Math.max(0, Math.ceil(rounds.length / 2) - 1)];
      activeIterations.buffer.write(Uint32Array.of(replayRound));
      const replay = await captureGPUCommandGraph(compiled, {
        at: 'end',
        encodeOptions: {parameters: undefined, coalesceComputePasses: false},
        views: {costs: output.costs}
      });
      replays.push({
        ...getNetworkFinalCapture(replay.captures, 'costs'),
        id: `costs-replay-${replayRound}`,
        provenance: 'gpu-replay',
        keySpace: 'node',
        iteration: replayRound,
        meaning: 'replay: the same compiled graph re-run with activeIterations = k'
      });
    }
    const captures = createGPUComputationCaptures({
      runId: record.runId,
      snapshots,
      stamps: [stamp],
      replays
    });
    const metrics = [
      createGPUComputationMetrics({
        runId: record.runId,
        graph: record.graphs[0].id,
        timing: {nodes: []},
        encoding: {coalesced: false, timestamped: false, taps: 0, lensTwin: true, retained},
        series: [
          {
            name: 'frontier-size',
            axis: 'iteration',
            unit: 'nodes',
            values: Array.from(final('control').values).slice(4, 5 + iterationCount)
          }
        ]
      })
    ];
    const result = {...record, captures, metrics};
    const validation = validateGPUComputationSidecars(record, result);
    if (!validation.valid) throw new Error(JSON.stringify(validation.issues));
    return result;
  } finally {
    compiled.destroy();
    for (const buffer of built.ownedBuffers) buffer.destroy();
  }
}

/** Describes either variant on a null device, without submitting or reading buffers. */
export async function describeReachabilityStructure(device: Device, chained: boolean) {
  const built = await buildScenario(device, chained, true);
  try {
    return built.record;
  } finally {
    built.compiled.destroy();
    for (const buffer of built.ownedBuffers) buffer.destroy();
  }
}

/** Captures the single-hop wavefront proof and a separately labelled limited replay. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  return runReachabilityScenario(device, false, options);
}

/** Builds the single-hop structure without execution. */
export async function describeStructure(device: Device) {
  return describeReachabilityStructure(device, false);
}
