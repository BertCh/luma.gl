// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUComputationCapture,
  GPUComputationFormulaRef,
  GPUComputationGraph,
  GPUComputationGraphForm,
  GPUComputationMetrics,
  GPUComputationNode,
  GPUComputationSourceRef
} from '@luma.gl/gpgpu';

import type {MarkProvenance} from '../types';
import type {LensRecord, ViewContext} from '../registry/view-registration';

/** Graph forms from most concrete to most abstract. */
const CONCRETE_FIRST: readonly GPUComputationGraphForm[] = [
  'encoded',
  'compiled',
  'lowered',
  'intent'
];

/** Most concrete graph of a form list (default: compiled, then lowered, then intent; encoded last). */
export function getGraphByForm(
  record: LensRecord,
  forms: readonly GPUComputationGraphForm[] = ['compiled', 'lowered', 'intent', 'encoded']
): GPUComputationGraph | undefined {
  for (const form of forms) {
    const graph = record.graphs.find(candidate => candidate.form === form);
    if (graph) {
      return graph;
    }
  }
  return undefined;
}

/** The most concrete graph of the record. */
export function getMostConcreteGraph(record: LensRecord): GPUComputationGraph | undefined {
  return getGraphByForm(record, CONCRETE_FIRST);
}

/** Graph by id. */
export function getGraph(record: LensRecord, graphId: string): GPUComputationGraph | undefined {
  return record.graphs.find(graph => graph.id === graphId);
}

/**
 * Resolves a node reference: `<graphId>:<nodeId>` or a bare node id searched in the compiled
 * graph first.
 */
export function findNode(
  record: LensRecord,
  reference: string
): {graph: GPUComputationGraph; node: GPUComputationNode} | undefined {
  const separator = reference.indexOf(':');
  if (separator > 0) {
    const graph = getGraph(record, reference.slice(0, separator));
    const node = graph?.nodes.find(candidate => candidate.id === reference.slice(separator + 1));
    if (graph && node) {
      return {graph, node};
    }
  }
  for (const form of ['compiled', 'lowered', 'intent', 'encoded'] as const) {
    for (const graph of record.graphs.filter(candidate => candidate.form === form)) {
      const node = graph.nodes.find(candidate => candidate.id === reference);
      if (node) {
        return {graph, node};
      }
    }
  }
  return undefined;
}

/** The node bound by `bindings.node`, if any. */
export function getBoundNode(
  context: ViewContext
): {graph: GPUComputationGraph; node: GPUComputationNode} | undefined {
  const reference = context.bindings.node;
  return typeof reference === 'string' ? findNode(context.record, reference) : undefined;
}

/** Formula by id. */
export function getFormula(
  record: LensRecord,
  formulaId: string | undefined
): GPUComputationFormulaRef | undefined {
  return formulaId ? record.refs.formulas.find(formula => formula.id === formulaId) : undefined;
}

/** Source by id. */
export function getSource(
  record: LensRecord,
  sourceId: string | undefined
): GPUComputationSourceRef | undefined {
  return sourceId ? record.refs.sources.find(source => source.id === sourceId) : undefined;
}

/** Every capture (snapshots, stamps, replays) of the record, in sidecar order. */
export function getAllCaptures(record: LensRecord): GPUComputationCapture[] {
  const captures = record.captures;
  return captures ? [...captures.snapshots, ...captures.stamps, ...captures.replays] : [];
}

/** Captures indexed by a key space, optionally at one iteration. */
export function getCapturesForKeySpace(
  record: LensRecord,
  keySpace: string,
  iteration?: number
): GPUComputationCapture[] {
  return getAllCaptures(record).filter(
    capture =>
      capture.keySpace === keySpace && (iteration === undefined || capture.iteration === iteration)
  );
}

/**
 * The data key space a datum pin (`bindings.entity`) refers to: `options.keySpace`, else the key
 * space of the first `data` axis, else the first key space with captures.
 */
export function getDatumKeySpace(context: ViewContext): string | undefined {
  const optionKeySpace = context.options?.['keySpace'];
  if (typeof optionKeySpace === 'string') {
    return optionKeySpace;
  }
  const dataAxis = context.record.axes.axes.find(axis => axis.kind === 'data' && axis.keySpace);
  if (dataAxis?.keySpace) {
    return dataAxis.keySpace;
  }
  return getAllCaptures(context.record).find(capture => capture.keySpace)?.keySpace;
}

/** The pinned entity index, if `bindings.entity` is a single number. */
export function getPinnedEntity(context: ViewContext): number | undefined {
  const entity = context.bindings.entity;
  return typeof entity === 'number' && Number.isInteger(entity) && entity >= 0 ? entity : undefined;
}

/** The bound iteration, if `bindings.iteration` is a single number. */
export function getBoundIteration(context: ViewContext): number | undefined {
  const iteration = context.bindings.iteration;
  return typeof iteration === 'number' && Number.isInteger(iteration) ? iteration : undefined;
}

/** Value of one key in a capture (first component), or undefined when out of range. */
export function getCaptureValue(capture: GPUComputationCapture, key: number): number | undefined {
  const components = capture.components ?? 1;
  const index = key * components;
  return index >= 0 && index < capture.values.length ? Number(capture.values[index]) : undefined;
}

/** Mark provenance of a capture: readback is solid, replay hatched. */
export function getCaptureProvenance(capture: GPUComputationCapture): MarkProvenance {
  return capture.provenance === 'gpu-replay' ? 'replay' : 'readback';
}

/** Metrics sidecar of a graph, if measured. */
export function getMetricsForGraph(
  record: LensRecord,
  graphId: string
): GPUComputationMetrics | undefined {
  return record.metrics?.find(metrics => metrics.graph === graphId);
}

/** Adapter label shown on every view (`Apple M3 · variant`). */
export function getAdapterLabel(record: LensRecord): string {
  const adapter = record.adapter;
  const name = adapter.description || adapter.vendor || adapter.architecture || adapter.backend;
  return adapter.variant ? `${name} · ${adapter.variant}` : name;
}

/**
 * Observer effects in force during the record's measured runs, worded for captions. Every
 * timing view states them (policy 3).
 */
export function getObserverEffects(record: LensRecord): string[] {
  const effects = new Set<string>();
  for (const metrics of record.metrics ?? []) {
    const encoding = metrics.encoding;
    if (encoding.timestamped && !encoding.coalesced) {
      effects.add('timestamps disabled compute-pass coalescing, so passes differ from production');
    } else if (!encoding.coalesced) {
      effects.add('compute passes were not coalesced as in production');
    }
    if (encoding.taps > 0) {
      effects.add(`${encoding.taps} observation tap${encoding.taps === 1 ? '' : 's'} inserted`);
    }
    if (encoding.lensTwin) {
      effects.add(
        encoding.retained && encoding.retained.length > 0
          ? `lens twin: transients ${encoding.retained.join(', ')} retained instead of aliased`
          : 'lens twin: docs-sized data'
      );
    }
  }
  const readable = record.graphs.flatMap(graph =>
    (graph.resources ?? []).filter(resource => resource.attrs?.['observerUsage'] === 'copy-src')
  );
  if (readable.length) {
    effects.add(
      `transients ${[...new Set(readable.map(resource => resource.id))].join(', ')} allocated with COPY_SRC for observation`
    );
  }
  return [...effects];
}

/** Splits a namespace path into segments (`''` is top level). */
export function getNamespaceSegments(namespace: string): string[] {
  return namespace.split('/').filter(segment => segment.length > 0);
}

/** Formats a number for labels: integers with grouping, others to 4 significant digits. */
export function formatViewNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return Number.isNaN(value) ? 'NaN' : value > 0 ? '∞' : '−∞';
  }
  if (Number.isInteger(value)) {
    return value.toLocaleString('en-US');
  }
  return Number(value.toPrecision(4)).toLocaleString('en-US', {maximumSignificantDigits: 4});
}
