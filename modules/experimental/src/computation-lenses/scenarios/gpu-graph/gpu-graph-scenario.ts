// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, type Device} from '@luma.gl/core';
import {
  captureGPUCommandGraph,
  createGPUComputationCaptures,
  createGPUComputationMetrics,
  createGPUComputationRecord,
  getGPUComputationAdapter,
  validateGPUComputationRecord,
  validateGPUComputationSidecars,
  type GPUAnalysisChoice,
  type GPUComputationCapture,
  type GPUComputationFormulaRef,
  type GPUComputationKeySpace,
  type GPUComputationRecord,
  type GPUComputationRecordWithSidecars,
  type GPUComputationSeries
} from '@luma.gl/gpgpu';
import type {CompiledGPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVector, GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {GPU_CORE_FORMULAS} from '../../../computation-views/packs/gpu-core/formulas';
import {GPU_GRAPH_FORMULAS} from './computation-pack';
import {
  createRailGraphFixture,
  type RailGraphFixture,
  type RailGraphFixtureOptions
} from './rail-graph-fixture';

/** Builds the actual public example and tracks returned resources; does not submit commands. */
export async function buildGPUGraphScenario<Outputs extends Record<string, unknown>>(
  device: Device,
  options: RailGraphFixtureOptions & {
    example: (device: Device, input: RailGraphFixture) => Promise<Outputs> | Outputs;
  }
): Promise<{fixture: RailGraphFixture; outputs: Outputs}> {
  const fixture = createRailGraphFixture(device, options);
  try {
    const outputs = await options.example(device, fixture);
    for (const resource of Object.values(outputs)) {
      if (isDestroyable(resource)) fixture.track(resource);
    }
    return {fixture, outputs};
  } catch (error) {
    fixture.destroy();
    throw error;
  }
}

function isDestroyable(resource: unknown): resource is {destroy(): void} {
  return Boolean(
    resource &&
      typeof resource === 'object' &&
      'destroy' in resource &&
      typeof resource.destroy === 'function'
  );
}

/** Reuses the contributor's imported physical handle; call before compiling for new columns. */
export function getGPUGraphCaptureView<Format extends GPUVectorFormat>(
  fixture: RailGraphFixture,
  name: string,
  vector: GPUVector<Format>
): GraphDataView<Format> {
  // importGPUGraphColumn uses importGPUVector too. Its physical-buffer cache returns the SAME
  // handle, preserving write hazards; no second alias or CPU reconstruction is needed.
  return fixture.commandGraph.importGPUVector(name, vector).data[0];
}

/** A real output column; stationValues projects its values onto the read-back rail coordinates. */
export type GPUGraphOutputCapture = {
  view: GraphDataView;
  keySpace?: string;
  stationValues?: boolean;
  geometry?: GPUComputationCapture['geometry'];
};

/** One evolving column, optionally captured from an explicitly retained transient view. */
export type GPUGraphIterationFrames = {
  name: string;
  view?: GraphDataView;
  loopId?: string;
};

/** Inputs shared by recording and structural inspection of a compiled scenario. */
export type GPUGraphScenarioRecordOptions = {
  scenarioId: string;
  element: string;
  fixture: RailGraphFixture;
  compiled: CompiledGPUCommandGraph<void>;
  outputCaptures: Readonly<Record<string, GPUGraphOutputCapture>>;
  iterationFrames?: readonly GPUGraphIterationFrames[];
  /** Key spaces kernels bind beyond stations and edges, such as breadth-first seeds. */
  extraKeySpaces?: readonly GPUComputationKeySpace[];
  choices: readonly GPUAnalysisChoice[];
  highlight?: {
    capture: string;
    select: (
      values: ArrayLike<number>,
      fixture: RailGraphFixture
    ) => NonNullable<GPUComputationCapture['highlight']>;
  };
  series?: (
    finalCaptures: Readonly<Record<string, GPUComputationCapture>>
  ) => readonly GPUComputationSeries[];
};

/** Selects zero-based completed rounds, preserving the final round within an eight-frame budget. */
export function selectGPUGraphIterations(iterations: readonly number[]): number[] {
  const available = [...new Set(iterations)].sort((first, second) => first - second);
  if (!available.length) return [];
  const final = available[available.length - 1];
  const selected = available.filter(
    iteration =>
      iteration === 0 ||
      iteration === 1 ||
      (iteration > 0 && Number.isInteger(Math.log2(iteration)))
  );
  return [...new Set([...selected.filter(iteration => iteration !== final).slice(0, 7), final])];
}

function getInputViews(fixture: RailGraphFixture): Record<string, GraphDataView> {
  // The fixture registered these before compilation; each repeated import only creates a view.
  return {
    'csr-offsets': getGPUGraphCaptureView(
      fixture,
      'csr-offsets',
      fixture.railTopology.forward.offsets
    ),
    'csr-neighbors': getGPUGraphCaptureView(
      fixture,
      'csr-neighbors',
      fixture.railTopology.forward.neighbors
    ),
    'station-positions': getGPUGraphCaptureView(
      fixture,
      'station-positions',
      fixture.stations.positions
    )
  };
}

/** Describes the compiled graph and semantic axes without running or reading any GPU work. */
export function describeGPUGraphScenario(
  options: GPUGraphScenarioRecordOptions
): GPUComputationRecord {
  const {fixture, compiled, element, iterationFrames} = options;
  const inputs = getInputViews(fixture);
  const formula = GPU_GRAPH_FORMULAS[element];
  const description = compiled.describe({domain: 'gpu-graph', label: element});
  // Topology lowers to GPUScan passes whose annotations refer to the core pack's formula.
  // Preserve those references and reuse their canonical definitions alongside this element.
  const referencedFormulas = new Set(description.graph.nodes.map(node => node.formula));
  const formulas = Object.values({...GPU_CORE_FORMULAS, ...GPU_GRAPH_FORMULAS})
    .filter((candidate): candidate is GPUComputationFormulaRef => !('formula' in candidate))
    .filter(candidate => candidate === formula || referencedFormulas.has(candidate.id));
  return createGPUComputationRecord({
    id: options.scenarioId,
    adapter: getGPUComputationAdapter(fixture.device),
    domains: ['gpu-graph'],
    graphs: [description],
    axes: {
      keySpaces: [
        {id: 'vertex', label: 'station', size: fixture.stations.vertexCount, indexSpace: '1d'},
        {id: 'edge', label: 'edge', size: fixture.edgeCount},
        ...(options.extraKeySpaces ?? [])
      ],
      relations: [
        {
          id: 'adjacency',
          from: 'vertex',
          to: 'vertex',
          cardinality: 'N:M',
          kind: 'buffer',
          ref: {
            layout: 'csr',
            resource: inputs['csr-neighbors'].buffer.id,
            offsetsResource: inputs['csr-offsets'].buffer.id,
            capture: 'csr-neighbors'
          },
          timeDependent: false
        }
      ],
      axes: [
        {id: 'entity', label: 'station', kind: 'data', keySpace: 'vertex'},
        ...(iterationFrames?.length
          ? [
              {
                id: 'iteration',
                label: 'iteration',
                kind: 'execution' as const,
                size: Math.max(
                  0,
                  ...description.graph.nodes.map(node => (node.loop?.iteration ?? -1) + 1)
                )
              }
            ]
          : []),
        ...(options.series
          ? [{id: 'publication', label: 'final publication', kind: 'execution' as const, size: 1}]
          : [])
      ]
    },
    formulas,
    choices: options.choices,
    datasetVersion: 'gtfs-nl-rail-2026-10-09'
  });
}

/** Runs one observation graph, retains deterministic key frames, validates, then releases ownership. */
export async function recordGPUGraphScenario(
  options: GPUGraphScenarioRecordOptions
): Promise<GPUComputationRecordWithSidecars> {
  const {fixture, compiled, outputCaptures, iterationFrames = []} = options;
  try {
    const record = describeGPUGraphScenario(options);
    const graph = record.graphs[0];
    const views = getInputViews(fixture);
    for (const [name, output] of Object.entries(outputCaptures)) views[name] = output.view;
    for (const frame of iterationFrames) {
      const view = frame.view ?? outputCaptures[frame.name]?.view;
      // Each requested iteration series must identify a real, capturable GPU view.
      assert(view);
      views[`iteration:${frame.name}`] = view;
    }
    const stepped = iterationFrames.length > 0;
    const observed = await captureGPUCommandGraph(compiled, {
      encodeOptions: {parameters: undefined, coalesceComputePasses: !stepped},
      views,
      at: stepped ? 'step' : 'end',
      runId: record.runId,
      graphId: graph.id
    });
    const finalCaptures: Record<string, GPUComputationCapture> = {};
    for (const capture of observed.captures) finalCaptures[capture.name] = capture;
    const snapshots: GPUComputationCapture[] = [];
    for (const name of ['csr-offsets', 'csr-neighbors', 'station-positions']) {
      snapshots.push({
        ...finalCaptures[name],
        id: name,
        iteration: undefined,
        role: 'input',
        ...(name === 'station-positions' && {keySpace: 'vertex', components: 2}),
        ...(name === 'csr-offsets' && {
          geometry: {
            kind: 'graph',
            offsetsCapture: 'csr-offsets',
            targetsCapture: 'csr-neighbors',
            layoutCapture: 'station-positions',
            coordinateModel: 'geographic'
          }
        })
      });
    }
    for (const [name, output] of Object.entries(outputCaptures)) {
      const final = {
        ...finalCaptures[name],
        id: name,
        iteration: undefined,
        keySpace: output.keySpace,
        role: 'intermediate' as const
      };
      finalCaptures[name] = final;
      const frames = iterationFrames.find(frame => frame.name === name);
      if (frames) {
        const lastNodeByIteration = new Map<number, string>();
        for (const node of graph.nodes) {
          if (
            node.op === options.element &&
            node.loop &&
            (!frames.loopId || node.loop.id === frames.loopId)
          ) {
            lastNodeByIteration.set(node.loop.iteration, node.id);
          }
        }
        const selected = selectGPUGraphIterations([...lastNodeByIteration.keys()]);
        // Missing recording annotations must fail, never produce plausible invented frames.
        assert(selected.length > 0);
        for (const iteration of selected) {
          const afterNode = lastNodeByIteration.get(iteration);
          const capture = observed.captures.find(
            candidate => candidate.name === `iteration:${name}` && candidate.afterNode === afterNode
          );
          assert(capture);
          snapshots.push({
            ...capture,
            id: `${name}-${iteration}`,
            name,
            iteration,
            keySpace: output.keySpace,
            role: 'intermediate'
          });
        }
      } else {
        snapshots.push(final);
      }
      if (output.stationValues) {
        const positions = finalCaptures['station-positions'];
        snapshots.push({
          ...positions,
          id: `${name}-stations`,
          name: `${name}-stations`,
          iteration: undefined,
          keySpace: 'vertex',
          components: 2,
          role: 'output',
          geometry: {kind: 'points', coordinateModel: 'geographic', valuesCapture: name},
          ...(options.highlight?.capture === name && {
            highlight: options.highlight.select(final.values, fixture)
          })
        });
      } else if (output.geometry) {
        snapshots.push({
          ...final,
          id: `${name}-geometry`,
          name: `${name}-geometry`,
          role: 'output',
          geometry: output.geometry
        });
      }
    }
    const captures = createGPUComputationCaptures({runId: record.runId, snapshots});
    const metrics = [
      createGPUComputationMetrics({
        runId: record.runId,
        graph: graph.id,
        adapter: record.adapter,
        // captureGPUCommandGraph exposes no timings: an empty timing table is honest, not zero timings.
        timing: {nodes: []},
        encoding: {
          coalesced: !stepped,
          timestamped: false,
          // Count actual snapshot copies, including frames discarded from the compact sidecar.
          taps: observed.snapshotCount * Object.keys(views).length,
          lensTwin: true,
          retained: fixture.commandGraph.recording?.retain ?? []
        },
        series: options.series?.(finalCaptures)
      })
    ];
    const issues = [
      validateGPUComputationRecord(record),
      validateGPUComputationSidecars(record, {metrics, captures})
    ].flatMap(report => (report.valid ? [] : report.issues));
    if (issues.length) throw new Error(`Invalid gpu-graph lens record: ${JSON.stringify(issues)}`);
    return {...record, metrics, captures};
  } finally {
    try {
      compiled.destroy();
    } finally {
      fixture.destroy();
    }
  }
}
