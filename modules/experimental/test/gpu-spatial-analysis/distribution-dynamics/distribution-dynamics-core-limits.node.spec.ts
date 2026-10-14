// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  GraphDataView,
  type GPUCommandNode,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';
import type {VertexFormat} from '@luma.gl/core';
import {describe, expect, it} from 'vitest';
import {
  GPUClassAssignment,
  GPUClassificationFit,
  GPULISAMarkov,
  GPUSpatialMarkov,
  GPUTransitionDiagnostics,
  GPUTransitionMatrix,
  GPU_CLASSIFICATION_FIT_SUMMARY_LENGTH,
  GPU_TRANSITION_HOMOGENEITY_LENGTH,
  GPU_TRANSITION_STATE_SPARSITY_LENGTH
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/index';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

const MAXIMUM_BINDING_BYTES = 134217728;
let viewSerial = 0;

type InspectedGraph = {
  graph: GPUCommandGraph;
  transients: GraphBufferHandle[];
  view: <Format extends VertexFormat>(
    format: Format,
    length: number,
    name?: string
  ) => GraphDataView<Format>;
};

/** A graph on the core-limits device that records every transient buffer it creates. */
function createInspectedGraph(id: string): InspectedGraph {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id});
  const transients: GraphBufferHandle[] = [];
  const createTransientBuffer = graph.createTransientBuffer.bind(graph);
  graph.createTransientBuffer = descriptor => {
    const handle = createTransientBuffer(descriptor);
    transients.push(handle);
    return handle;
  };
  return {
    graph,
    transients,
    view: (format, length, name = 'v') =>
      createTransientView(graph, `${id}-${name}-${viewSerial++}`, format, length)
  };
}

function getBindingBytes(buffer: GraphDataView | GraphBufferHandle): number {
  return buffer instanceof GraphDataView
    ? buffer.byteOffset + Math.max(buffer.length - 1, 0) * buffer.byteStride + buffer.rowByteLength
    : buffer.byteLength;
}

/**
 * Asserts the three core kernel limits for every kernel of the nodes and that every transient and
 * every bound view fits one storage binding.
 */
function expectWithinCoreLimits(inspected: InspectedGraph, nodes: readonly GPUCommandNode[]) {
  expect(nodes.length).toBeGreaterThan(0);
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(sources.length).toBeGreaterThan(0);
  for (const {id, source} of sources) {
    expect(getStorageBindingCount(source), `${id} storage bindings`).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(source), `${id} private bytes`).toBeLessThanOrEqual(
      CORE_PRIVATE_MEMORY_LIMIT_BYTES
    );
    expect(getWorkgroupMemoryBytes(source), `${id} workgroup bytes`).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
  for (const handle of inspected.transients) {
    expect(handle.byteLength, `transient ${handle.id}`).toBeLessThanOrEqual(MAXIMUM_BINDING_BYTES);
  }
  for (const node of nodes) {
    for (const resource of node.resources ?? []) {
      if ('buffer' in resource) {
        expect(
          getBindingBytes(resource.buffer as GraphDataView),
          `${node.id} bound view`
        ).toBeLessThanOrEqual(MAXIMUM_BINDING_BYTES);
      }
    }
  }
}

const ROWS = 200;
const PERIODS = 5;

function createWeights({view}: InspectedGraph, rows: number) {
  return {
    offsets: view('uint32', rows + 1, 'offsets'),
    neighbors: view('uint32', rows * 8, 'neighbors'),
    weights: view('float32', rows * 8, 'weights')
  };
}

function createDiagnosticsOutput(
  {view}: InspectedGraph,
  classCount: number,
  conditionCount: number
) {
  const chains = conditionCount > 1 ? conditionCount + 1 : 1;
  return {
    homogeneity:
      conditionCount > 1 ? view('float32', GPU_TRANSITION_HOMOGENEITY_LENGTH, 'hom') : undefined,
    steadyStates: view('float32', chains * classCount, 'steady'),
    stateSparsity: view('uint32', GPU_TRANSITION_STATE_SPARSITY_LENGTH, 'sparsity'),
    status: {converged: view('uint32', 1, 'converged')}
  };
}

function createTransitionOutput(
  inspected: InspectedGraph,
  classCount: number,
  conditionCount: number
) {
  const {view} = inspected;
  return {
    ...createDiagnosticsOutput(inspected, classCount, conditionCount),
    counts: view('uint32', conditionCount * classCount * classCount, 'counts'),
    probabilities: view('float32', conditionCount * classCount * classCount, 'probabilities'),
    rowTotals: view('uint32', conditionCount * classCount, 'row-totals'),
    ignored: view('uint32', 1, 'ignored')
  };
}

describe('distribution-dynamics core limits', () => {
  it('GPUClassAssignment with mask', () => {
    const inspected = createInspectedGraph('cl-assign');
    const {view, graph} = inspected;
    const nodes = new GPUClassAssignment({
      values: view('float32', ROWS),
      breaks: view('float32', 65),
      classCount: view('uint32', 1),
      mask: view('uint32', ROWS),
      output: view('uint32', ROWS)
    }).getCommandNodes(graph);
    expectWithinCoreLimits(inspected, nodes);
  });

  it('GPUClassificationFit at 64 classes with every output', () => {
    const inspected = createInspectedGraph('cl-fit');
    const {view, graph} = inspected;
    const nodes = new GPUClassificationFit({
      values: view('float32', ROWS),
      classes: view('uint32', ROWS),
      classCount: 64,
      mask: view('uint32', ROWS),
      output: {
        counts: view('uint32', 65),
        medians: view('float32', 65),
        absoluteDeviations: view('float32', 65),
        squaredDeviations: view('float32', 65),
        summary: view('float32', GPU_CLASSIFICATION_FIT_SUMMARY_LENGTH)
      }
    }).getCommandNodes(graph);
    expectWithinCoreLimits(inspected, nodes);
  });

  for (const [classCount, conditionCount] of [
    [64, 1],
    [64, 8],
    [10, 4]
  ] as const) {
    it(`GPUTransitionMatrix at ${classCount} classes x ${conditionCount} conditions`, () => {
      const inspected = createInspectedGraph('transition');
      const {view, graph} = inspected;
      const nodes = new GPUTransitionMatrix({
        classes: view('uint32', ROWS * PERIODS),
        periodCount: PERIODS,
        classCount,
        periodLag: 2,
        ...(conditionCount > 1
          ? {conditionClasses: view('uint32', ROWS * PERIODS), conditionCount}
          : {}),
        mask: view('uint32', ROWS),
        minimumTransitions: 7,
        output: createTransitionOutput(inspected, classCount, conditionCount)
      }).getCommandNodes(graph);
      expectWithinCoreLimits(inspected, nodes);
    });

    it(`GPUTransitionDiagnostics at ${classCount} classes x ${conditionCount} conditions`, () => {
      const inspected = createInspectedGraph('diagnostics');
      const {view, graph} = inspected;
      const nodes = new GPUTransitionDiagnostics({
        counts: view('uint32', conditionCount * classCount * classCount),
        classCount,
        conditionCount,
        minimumTransitions: 7,
        output: createDiagnosticsOutput(inspected, classCount, conditionCount)
      }).getCommandNodes(graph);
      expectWithinCoreLimits(inspected, nodes);
    });
  }

  for (const gating of ['none', 'significance'] as const) {
    it(`GPULISAMarkov with gating ${gating} and every output`, () => {
      const inspected = createInspectedGraph('lisa');
      const {view, graph} = inspected;
      const nodes = new GPULISAMarkov({
        values: view('float32', ROWS * PERIODS),
        weights: {...createWeights(inspected, ROWS), distances: view('float32', ROWS * 8)},
        periodCount: PERIODS,
        periodLag: 2,
        parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
        mask: view('uint32', ROWS),
        gating,
        minimumTransitions: 7,
        output: {
          ...createTransitionOutput(inspected, 5, 1),
          quadrants: view('uint32', ROWS * PERIODS)
        }
      }).getCommandNodes(graph);
      expectWithinCoreLimits(inspected, nodes);
    });
  }

  for (const [classCount, lagMaximumClassCount, normalize] of [
    [16, 16, true],
    [64, 8, false],
    [5, 6, true]
  ] as const) {
    it(`GPUSpatialMarkov at ${classCount} classes x ${lagMaximumClassCount} lag classes`, () => {
      const inspected = createInspectedGraph('spatial-markov');
      const {view, graph} = inspected;
      const nodes = new GPUSpatialMarkov({
        values: view('float32', ROWS * PERIODS),
        classes: view('uint32', ROWS * PERIODS),
        classCount,
        weights: createWeights(inspected, ROWS),
        periodCount: PERIODS,
        periodLag: 2,
        normalize,
        maximumLagClassCount: lagMaximumClassCount,
        lagParameters: view('float32', 16 + lagMaximumClassCount),
        lagMethods: ['quantile', 'equal-interval', 'natural-breaks', 'box-plot'],
        mask: view('uint32', ROWS),
        minimumTransitions: 7,
        output: {
          ...createTransitionOutput(inspected, classCount, lagMaximumClassCount),
          lagBreaks: view('float32', lagMaximumClassCount + 1),
          lagClassCount: view('uint32', 1),
          lagValues: view('float32', ROWS * PERIODS),
          lagClasses: view('uint32', ROWS * PERIODS)
        }
      }).getCommandNodes(graph);
      expectWithinCoreLimits(inspected, nodes);
    });
  }
});
