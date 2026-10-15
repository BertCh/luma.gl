// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {
  createGPUComputationRecord,
  getGPUComputationAdapter,
  createGPUComputationCaptures,
  type CreateGPUComputationRecordProps,
  type GPUComputationCapture,
  type GPUComputationCaptureElementType,
  type GPUComputationRecordWithSidecars,
  type GPUComputationKeySpace,
  type GPUComputationRelation
} from '@luma.gl/gpgpu';
import {
  captureGPUCommandGraph,
  type GPUAnalysisChoice,
  type GPUComputationCaptureGeometry,
  type GPUComputationFormulaRef
} from '@luma.gl/gpgpu';
import type {CompiledGPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getDataFrameOperatorEntry} from '../../../gpu-dataframe/computation-pack/index';
import {mergeGPUCoreFormulaRefsForGraphs} from '../../../computation-views/packs/gpu-core/formulas';
import {
  validateGPUComputationRecord,
  validateGPUComputationSidecars,
  type GPUComputationAxis
} from '@luma.gl/gpgpu';

import type {CommandEncoder} from '@luma.gl/core';
import type {GPUDataFrameQueryParameters} from '../../../gpu-dataframe/gpu-query-compiler';

/** Loads all Chicago tract columns, including nullable income. */
export function loadDataFrameLensData(
  name: 'f-chicago-tracts'
): Promise<typeof import('../../data/f-chicago-tracts.json')>;
/** Loads the deterministic sample of Chicago observations. */
export function loadDataFrameLensData(
  name: 'f-chicago-nature'
): Promise<typeof import('../../data/f-chicago-nature.json')>;
/** Loads the deterministic sample of NOAA weather stations. */
export function loadDataFrameLensData(
  name: 'f-ghcn-stations'
): Promise<typeof import('../../data/f-ghcn-stations.json')>;
/** Accepts a runtime choice of dataset as well as the precisely typed literal overloads. */
export function loadDataFrameLensData(
  name: 'f-chicago-tracts' | 'f-chicago-nature' | 'f-ghcn-stations'
): Promise<
  | typeof import('../../data/f-chicago-tracts.json')
  | typeof import('../../data/f-chicago-nature.json')
  | typeof import('../../data/f-ghcn-stations.json')
>;
/** Loads a deterministic local dataset only when a dataframe lens is activated. */
export async function loadDataFrameLensData(
  name: 'f-chicago-tracts' | 'f-chicago-nature' | 'f-ghcn-stations'
) {
  switch (name) {
    case 'f-chicago-tracts':
      return (await import('../../data/f-chicago-tracts.json')).default;
    case 'f-chicago-nature':
      return (await import('../../data/f-chicago-nature.json')).default;
    case 'f-ghcn-stations':
      return (await import('../../data/f-ghcn-stations.json')).default;
  }
}

/** Creates a source-recording graph, optionally retaining named transient resources. */
export function createDataFrameLensGraph<Parameters = void>(
  device: Device,
  id: string,
  retain?: readonly string[]
): GPUCommandGraph<Parameters> {
  return new GPUCommandGraph<Parameters>(device, {id, recording: {sourceText: true, retain}});
}

/** Reads fixed-width scalar/vector chunks in source order, respecting offsets and row stride. */
export async function readGPUVectorValues(
  vector: GPUVector,
  elementType: GPUComputationCaptureElementType
): Promise<number[]> {
  const chunks = await Promise.all(
    vector.data.map(async chunk => {
      if (!chunk.length) return [];
      const buffer = chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer;
      const bytes = await buffer.readAsync(
        chunk.byteOffset,
        (chunk.length - 1) * chunk.byteStride + chunk.rowByteLength
      );
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const values: number[] = [];
      for (let row = 0; row < chunk.length; row++) {
        for (let component = 0; component < chunk.rowByteLength / 4; component++) {
          const offset = row * chunk.byteStride + component * 4;
          values.push(
            elementType === 'f32'
              ? view.getFloat32(offset, true)
              : elementType === 'i32'
                ? view.getInt32(offset, true)
                : view.getUint32(offset, true)
          );
        }
      }
      return values;
    })
  );
  return chunks.flat();
}

/** Assembles a real dataframe run and its readbacks under the same deterministic run identity. */
export function createDataFrameLensRecord(props: {
  scenarioId: string;
  device: Device;
  graphs: CreateGPUComputationRecordProps['graphs'];
  formulas?: CreateGPUComputationRecordProps['formulas'];
  keySpaces: readonly GPUComputationKeySpace[];
  relations?: readonly GPUComputationRelation[];
  choices?: CreateGPUComputationRecordProps['choices'];
  captures: readonly GPUComputationCapture[];
}): GPUComputationRecordWithSidecars {
  const record = createGPUComputationRecord({
    id: props.scenarioId,
    adapter: getGPUComputationAdapter(props.device),
    domains: ['gpu-dataframe'],
    graphs: props.graphs,
    formulas: mergeGPUCoreFormulaRefsForGraphs(props.graphs, props.formulas),
    choices: props.choices,
    axes: {keySpaces: props.keySpaces, relations: props.relations ?? []},
    datasetVersion: `${props.scenarioId}:1`
  });
  return {
    ...record,
    captures: createGPUComputationCaptures({runId: record.runId, snapshots: props.captures})
  };
}

/** Tags actual GPU readback as a table column, without inventing geometry. */
export function createTableCapture(
  props: Pick<
    GPUComputationCapture,
    'id' | 'name' | 'keySpace' | 'role' | 'values' | 'elementType' | 'meaning'
  >
): GPUComputationCapture {
  return {...props, provenance: 'gpu-readback'};
}

/** One graph view a pair-statistics scenario snapshots, with the tags the result view reads. */
export type PairStatisticsCaptureSpec = {
  /** Packed input or output view; its capture name is the key it is registered under. */
  view: GraphDataView;
  /** Input, output or (for slices of an output that a formula symbol names) intermediate. */
  role: 'input' | 'output' | 'intermediate';
  /** Key space the rows index, declared in the record's `keySpaces`. */
  keySpace: string;
  /** Defaults to the view's 32-bit scalar type. */
  elementType?: GPUComputationCaptureElementType;
  /** Points geometry of projected positions. */
  geometry?: GPUComputationCaptureGeometry;
  /** Free-form note shown with the column. */
  meaning?: string;
};

/**
 * Executes a compiled one-pass graph once and copies the requested views at the end of the run.
 * The capture adds one submission after the run (observer effect); values are `gpu-readback`.
 */
export async function capturePairStatisticsViews(
  compiled: CompiledGPUCommandGraph<undefined>,
  graphId: string,
  specs: Readonly<Record<string, PairStatisticsCaptureSpec>>
): Promise<GPUComputationCapture[]> {
  const names = Object.keys(specs);
  const captured = await captureGPUCommandGraph(compiled, {
    encodeOptions: {parameters: undefined},
    views: Object.fromEntries(names.map(name => [name, specs[name].view])),
    elementTypes: Object.fromEntries(
      names.flatMap(name => {
        const elementType = specs[name].elementType;
        return elementType ? [[name, elementType]] : [];
      })
    ),
    at: 'end',
    graphId
  });
  return captured.captures.map(capture => {
    const spec = specs[capture.name];
    return {
      ...capture,
      role: spec.role,
      keySpace: spec.keySpace,
      ...(spec.geometry && {geometry: spec.geometry}),
      ...(spec.meaning && {meaning: spec.meaning})
    };
  });
}

/** Returns the curated pack formula of an element, or none when the pack declares it procedural. */
export function getDataFrameElementFormulas(element: string): GPUComputationFormulaRef[] {
  const formula = getDataFrameOperatorEntry(element)?.formula;
  return formula && 'tex' in formula ? [formula] : [];
}

/** Builds one recorded analyst choice; sweeping is off unless the caller says a sweep is meaningful. */
export function createDataFrameChoice(
  stage: string,
  name: string,
  value: GPUAnalysisChoice['value'],
  options: Pick<GPUAnalysisChoice, 'alternatives' | 'unit'> & {sweepable?: boolean} = {}
): GPUAnalysisChoice {
  const {sweepable = false, ...rest} = options;
  return {stage, name, value, sweepable, ...rest};
}

/** Inputs of {@link runDataFrameContributorLens}. */
export type DataFrameContributorLensProps = {
  /** Scenario id, also the record id and the prefix of the compiled graph id. */
  scenarioId: string;
  /** Exported element name; its pack entry supplies the formula. */
  element: string;
  /** Recording graph that already holds the contributor and its inputs. */
  graph: GPUCommandGraph<undefined>;
  /** Views to snapshot at the end of the run, tagged for the result view. */
  specs: Readonly<Record<string, PairStatisticsCaptureSpec>>;
  /** Every key space the captures use. */
  keySpaces: readonly GPUComputationKeySpace[];
  /** Key space a datum pin indexes; it becomes the record's data axis. */
  datumKeySpace: string;
  /** Relations between key spaces, for example the identity from invocation to row. */
  relations?: readonly GPUComputationRelation[];
  /** Analyst choices the example made, with alternatives from the exported constants. */
  choices: readonly GPUAnalysisChoice[];
  /** Dataset identity folded into the run id. */
  datasetVersion: string;
  /** Adds captures that are exact reshapes of the readback, for example one column of a table. */
  deriveCaptures?: (captures: readonly GPUComputationCapture[]) => readonly GPUComputationCapture[];
};

/**
 * Compiles the recording graph, runs it once with an end-of-run snapshot of the tagged views, and
 * returns the validated record with its captures. The snapshot adds one submission after the run
 * (observer effect). Imported buffers stay caller-owned; compiled transients are released here.
 */
export async function runDataFrameContributorLens(
  device: Device,
  props: DataFrameContributorLensProps
): Promise<GPUComputationRecordWithSidecars> {
  const graphId = `${props.scenarioId}-compiled`;
  const compiled = props.graph.compile();
  try {
    const description = compiled.describe({graphId, domain: 'gpu-dataframe'});
    const readback = await capturePairStatisticsViews(compiled, graphId, props.specs);
    const snapshots = props.deriveCaptures ? props.deriveCaptures(readback) : readback;
    const datumSpace = props.keySpaces.find(keySpace => keySpace.id === props.datumKeySpace);
    const axes: GPUComputationAxis[] = [
      {
        id: 'entity',
        label: datumSpace?.label ?? props.datumKeySpace,
        kind: 'data',
        keySpace: props.datumKeySpace,
        ...(datumSpace?.size !== undefined && {size: datumSpace.size})
      }
    ];
    const record = createGPUComputationRecord({
      id: props.scenarioId,
      adapter: getGPUComputationAdapter(device),
      domains: ['gpu-dataframe'],
      graphs: [description],
      axes: {axes, keySpaces: props.keySpaces, relations: props.relations ?? []},
      formulas: mergeGPUCoreFormulaRefsForGraphs(
        [description],
        getDataFrameElementFormulas(props.element)
      ),
      choices: props.choices,
      datasetVersion: props.datasetVersion
    });
    const validation = validateGPUComputationRecord(record);
    // A malformed record must fail here instead of rendering a plausible but wrong card.
    if (!validation.valid) throw new Error(JSON.stringify(validation.issues));
    const captures = createGPUComputationCaptures({runId: record.runId, snapshots: [...snapshots]});
    const sidecars = validateGPUComputationSidecars(record, {captures});
    if (!sidecars.valid) throw new Error(JSON.stringify(sidecars.issues));
    return {...record, captures};
  } finally {
    compiled.destroy();
  }
}

/** A compiled dataframe query that can be encoded once into a caller-owned encoder. */
export type EncodableDataFrameQuery = {
  encode(commandEncoder: CommandEncoder, parameters?: GPUDataFrameQueryParameters): unknown;
};

/** Encodes a compiled query into a fresh encoder, submits it once and releases the encoder. */
export function submitDataFrameQuery(
  device: Device,
  query: EncodableDataFrameQuery,
  parameters: GPUDataFrameQueryParameters = {}
): void {
  const commandEncoder = device.createCommandEncoder();
  try {
    query.encode(commandEncoder, parameters);
    device.submit(commandEncoder.finish());
  } finally {
    commandEncoder.destroy();
  }
}

/** Marks rows without a valid value as NaN, so a nullable column reads back as it is seen by the query. */
export function applyValidity(values: readonly number[], validity: readonly number[]): number[] {
  return values.map((value, index) => (validity[index] ? value : NaN));
}

/**
 * Exact gather of readbacks: `values[indices[i]]` for each published slot. Slots whose index is the
 * reserved no-match marker `0xffffffff`, or whose source value is invalid, read as NaN.
 */
export function gatherValues(
  values: readonly number[],
  indices: readonly number[],
  validity?: readonly number[]
): number[] {
  return indices.map(index =>
    index === 0xffffffff || (validity && !validity[index]) ? NaN : values[index]
  );
}

/** Inputs of {@link createDataFrameQueryLensRecord}. */
export type DataFrameQueryLensRecordProps = {
  /** Scenario id, also the record id. */
  scenarioId: string;
  /** Exported element name; its pack entry supplies the curated formula, if it has one. */
  element: string;
  device: Device;
  graphs: CreateGPUComputationRecordProps['graphs'];
  /** Derived formulas generated by the compiled query's `describe()`. */
  derivedFormulas?: CreateGPUComputationRecordProps['formulas'];
  keySpaces: readonly GPUComputationKeySpace[];
  /** Key space a datum pin indexes; it becomes the record's data axis. */
  datumKeySpace: string;
  relations?: readonly GPUComputationRelation[];
  choices?: CreateGPUComputationRecordProps['choices'];
  captures: readonly GPUComputationCapture[];
  datasetVersion?: string;
  /** Domains of the record's graphs; defaults to `gpu-dataframe`. */
  domains?: readonly string[];
};

/**
 * Assembles and validates the record of a one-shot query run. The pack's curated formula comes
 * first so the formula rung shows the element's own formula before the derived expression ones.
 */
export function createDataFrameQueryLensRecord(
  props: DataFrameQueryLensRecordProps
): GPUComputationRecordWithSidecars {
  const datumSpace = props.keySpaces.find(keySpace => keySpace.id === props.datumKeySpace);
  const record = createGPUComputationRecord({
    id: props.scenarioId,
    adapter: getGPUComputationAdapter(props.device),
    domains: props.domains ?? ['gpu-dataframe'],
    graphs: props.graphs,
    formulas: mergeGPUCoreFormulaRefsForGraphs(props.graphs, [
      ...getDataFrameElementFormulas(props.element),
      ...(props.derivedFormulas ?? [])
    ]),
    choices: props.choices,
    axes: {
      axes: [
        {
          id: 'entity',
          label: datumSpace?.label ?? props.datumKeySpace,
          kind: 'data',
          keySpace: props.datumKeySpace,
          ...(datumSpace?.size !== undefined && {size: datumSpace.size})
        }
      ],
      keySpaces: props.keySpaces,
      relations: props.relations ?? []
    },
    datasetVersion: props.datasetVersion ?? `${props.scenarioId}:1`
  });
  const validation = validateGPUComputationRecord(record);
  // A malformed record must fail here instead of rendering a plausible but wrong card.
  if (!validation.valid) throw new Error(JSON.stringify(validation.issues));
  const captures = createGPUComputationCaptures({
    runId: record.runId,
    snapshots: [...props.captures]
  });
  const sidecars = validateGPUComputationSidecars(record, {captures});
  if (!sidecars.valid) throw new Error(JSON.stringify(sidecars.issues));
  return {...record, captures};
}
