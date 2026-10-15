// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_COMPUTATION_RECORD_SCHEMA} from './computation-record-types';
import {hashGPUComputationText} from './hash-computation-text';
import type {
  GPUComputationGraph,
  GPUComputationSourceRef,
  GPUComputationAdapter,
  GPUComputationAxes,
  GPUComputationFormulaRef,
  GPUDecisionRecord,
  GPUAnalysisChoice,
  GPUWorkflowRecord,
  GPUComputationRecord
} from './computation-record-types';

/** A graph, optionally accompanied by the sources returned by its description. */
export type GPUComputationGraphInput =
  | GPUComputationGraph
  | {
      graph: GPUComputationGraph;
      sources: readonly GPUComputationSourceRef[];
    };

/** Structure and identity used to assemble one computation record. */
export type CreateGPUComputationRecordProps = {
  id: string;
  adapter: GPUComputationAdapter;
  domains: readonly string[];
  graphs: readonly GPUComputationGraphInput[];
  axes?: Partial<GPUComputationAxes>;
  formulas?: readonly GPUComputationFormulaRef[];
  sources?: readonly GPUComputationSourceRef[];
  decisions?: readonly GPUDecisionRecord[];
  choices?: readonly GPUAnalysisChoice[];
  workflow?: GPUWorkflowRecord;
  runId?: string;
  createdAt?: string;
  datasetVersion?: string;
  seed?: string | number;
};

/** Assembles a record, deduplicating sources by id; conflicting retained text throws. */
export function createGPUComputationRecord(
  props: CreateGPUComputationRecordProps
): GPUComputationRecord {
  const graphs = props.graphs.map(input => ('graph' in input ? input.graph : input));
  const sources = new Map<string, GPUComputationSourceRef>();
  const addSource = (source: GPUComputationSourceRef): void => {
    const previous = sources.get(source.id);
    if (
      previous?.text !== undefined &&
      source.text !== undefined &&
      previous.text !== source.text
    ) {
      throw new Error(`Conflicting source text: ${source.id}`);
    }
    if (!previous || (previous.text === undefined && source.text !== undefined)) {
      sources.set(source.id, source);
    }
  };
  props.sources?.forEach(addSource);
  for (const input of props.graphs) {
    if ('graph' in input) input.sources.forEach(addSource);
  }
  return {
    schema: GPU_COMPUTATION_RECORD_SCHEMA,
    id: props.id,
    runId: props.runId ?? getGPUComputationRunId({...props, graphs}),
    createdAt: props.createdAt ?? new Date().toISOString(),
    adapter: props.adapter,
    domains: props.domains,
    graphs,
    axes: {
      axes: props.axes?.axes ?? [],
      keySpaces: props.axes?.keySpaces ?? [],
      relations: props.axes?.relations ?? []
    },
    refs: {formulas: props.formulas ?? [], sources: [...sources.values()]},
    decisions: props.decisions ?? [],
    choices: props.choices ?? [],
    ...(props.workflow !== undefined && {workflow: props.workflow})
  };
}

/** Hashes ordered graph/node ids, backend/variant, dataset version and seed into `run-<8 hex>`. */
export function getGPUComputationRunId(props: {
  graphs: readonly GPUComputationGraph[];
  adapter: GPUComputationAdapter;
  datasetVersion?: string;
  seed?: string | number;
}): string {
  return `run-${hashGPUComputationText(
    JSON.stringify([
      props.graphs.map(graph => [graph.id, graph.nodes.map(node => node.id)]),
      props.adapter.backend,
      props.adapter.variant ?? null,
      props.datasetVersion ?? null,
      props.seed ?? null
    ])
  )}`;
}

/** Reads the DeviceInfo vendor, architecture (falling back to GPU family), and renderer. */
export function getGPUComputationAdapter(
  device: {
    type: string;
    info: {vendor?: string; renderer?: string; gpu?: string; gpuArchitecture?: string};
  },
  variant?: string
): GPUComputationAdapter {
  const architecture = device.info.gpuArchitecture ?? device.info.gpu;
  return {
    backend: device.type,
    ...(device.info.vendor !== undefined && {vendor: device.info.vendor}),
    ...(architecture !== undefined && {architecture}),
    ...(device.info.renderer !== undefined && {description: device.info.renderer}),
    ...(variant !== undefined && {variant})
  };
}
