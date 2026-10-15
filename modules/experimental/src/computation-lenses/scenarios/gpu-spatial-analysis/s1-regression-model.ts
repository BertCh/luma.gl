// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import type {GPUComputationAxes, GPUComputationKeySpace} from '@luma.gl/gpgpu';
import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandGraphEncodeOptions,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createGPULocalFrame} from '../../../analysis-core/contracts/local-frame';
import type {SpatialOperatorEntry} from '../../../gpu-spatial-analysis/computation-pack/types';
import {createWGSLKernelNode} from '../../../utils/wgsl-kernel-nodes';
import {loadChicagoTracts, type ChicagoTracts} from './spatial-lens-data';
import {
  runSpatialScenario,
  uploadChicagoTractPolygons,
  getRetainedViewIds,
  uploadSpatialView,
  type SpatialCaptureTag,
  type SpatialScenarioRun,
  type TractCaptureTag
} from './spatial-scenario-utils';

/** Response column of the shared regression model: CDC PLACES diabetes prevalence (%). */
export const REGRESSION_RESPONSE_COLUMN = 'diabetes';
/** Predictor columns of the shared regression model, in coefficient order after the intercept. */
export const REGRESSION_PREDICTOR_COLUMNS = ['poverty150Pct', 'noVehiclePct', 'uninsuredPct'];
/** Row order of every coefficient table: intercept first, then the predictors. */
export const REGRESSION_TERM_NAMES = ['intercept', ...REGRESSION_PREDICTOR_COLUMNS];
/** Origin of the local frame in which GWR distances are measured (central Chicago). */
const FRAME_ORIGIN: [number, number] = [-87.65, 41.84];
const REGRESSION_DATASET_NOTE =
  'complete-case tracts of s1-chicago-tracts: diabetes ~ poverty150Pct + noVehiclePct + uninsuredPct';

/** The shared tract design: y = diabetes, X = poverty, car-free households and uninsured shares. */
export type RegressionModel = {
  /** Complete-case tract geometry and columns, in source order. */
  tracts: ChicagoTracts;
  /** Tracts of the source extract (791). */
  sourceCount: number;
  /** Tracts dropped because the response or a predictor is missing in the source. */
  droppedCount: number;
  /** Response, one value per kept tract. */
  response: Float32Array;
  /** Row-major predictors, `tract * predictorCount + column`, without the intercept. */
  predictors: Float32Array;
  /** Area-weighted tract centroids in local-frame metres (azimuthal equidistant), x then y. */
  positions: Float32Array;
};

/**
 * Loads the Chicago tracts and keeps those with a finite response and predictors. Several
 * estimators here (two-stage, error and likelihood models) cannot skip rows, and all ten cards
 * should fit the same sample, so the three tracts with a missing value are dropped before any
 * upload. Centroids are prepared on the CPU as inputs, in a local metric frame.
 */
export async function loadRegressionModel(): Promise<RegressionModel> {
  const source = await loadChicagoTracts();
  const columnNames = [REGRESSION_RESPONSE_COLUMN, ...REGRESSION_PREDICTOR_COLUMNS];
  const kept: number[] = [];
  for (let tract = 0; tract < source.count; tract++) {
    if (columnNames.every(name => Number.isFinite(source.columns[name][tract]))) kept.push(tract);
  }
  const count = kept.length;
  // The extract has exactly one polygon part and one ring per feature (checked by the node spec).
  assert(source.ringOffsets.length === source.count + 1);
  const ringOffsets = new Uint32Array(count + 1);
  for (const [index, tract] of kept.entries()) {
    ringOffsets[index + 1] =
      ringOffsets[index] + source.ringOffsets[tract + 1] - source.ringOffsets[tract];
  }
  const vertices = new Float32Array(ringOffsets[count] * 2);
  for (const [index, tract] of kept.entries()) {
    vertices.set(
      source.vertices.subarray(source.ringOffsets[tract] * 2, source.ringOffsets[tract + 1] * 2),
      ringOffsets[index] * 2
    );
  }
  const identity = Uint32Array.from({length: count + 1}, (_, index) => index);
  const tracts: ChicagoTracts = {
    count,
    vertices,
    ringOffsets,
    polygonRingOffsets: identity,
    partFeature: identity.slice(0, count),
    columns: Object.fromEntries(
      Object.entries(source.columns).map(([name, values]) => [
        name,
        Float32Array.from(kept, tract => values[tract])
      ])
    ),
    dataset: {
      ...source.dataset,
      id: `${source.dataset.id}; ${REGRESSION_DATASET_NOTE}`
    }
  };
  const predictors = new Float32Array(count * REGRESSION_PREDICTOR_COLUMNS.length);
  for (const [column, name] of REGRESSION_PREDICTOR_COLUMNS.entries()) {
    for (let tract = 0; tract < count; tract++) {
      predictors[tract * REGRESSION_PREDICTOR_COLUMNS.length + column] =
        tracts.columns[name][tract];
    }
  }
  return {
    tracts,
    sourceCount: source.count,
    droppedCount: source.count - count,
    response: tracts.columns[REGRESSION_RESPONSE_COLUMN],
    predictors,
    positions: getTractCentroids(tracts)
  };
}

/** Area-weighted ring centroids in the metre frame; a degenerate ring falls back to its vertex mean. */
function getTractCentroids(tracts: ChicagoTracts): Float32Array {
  const frame = createGPULocalFrame({origin: FRAME_ORIGIN, datum: 'EPSG:4326'});
  const projected = frame.projectArray(tracts.vertices);
  const centroids = new Float32Array(tracts.count * 2);
  for (let tract = 0; tract < tracts.count; tract++) {
    const first = tracts.ringOffsets[tract];
    const last = tracts.ringOffsets[tract + 1];
    let area = 0;
    let centroidX = 0;
    let centroidY = 0;
    let meanX = 0;
    let meanY = 0;
    for (let vertex = first; vertex < last; vertex++) {
      const next = vertex + 1 < last ? vertex + 1 : first;
      const x = projected[vertex * 2];
      const y = projected[vertex * 2 + 1];
      const nextX = projected[next * 2];
      const nextY = projected[next * 2 + 1];
      const cross = x * nextY - nextX * y;
      area += cross;
      centroidX += (x + nextX) * cross;
      centroidY += (y + nextY) * cross;
      meanX += x;
      meanY += y;
    }
    const nonDegenerate = Math.abs(area) > 1e-6;
    centroids[tract * 2] = nonDegenerate ? centroidX / (3 * area) : meanX / (last - first);
    centroids[tract * 2 + 1] = nonDegenerate ? centroidY / (3 * area) : meanY / (last - first);
  }
  return centroids;
}

/** Uploaded design and geometry handed to every regression example. */
export type RegressionExampleInput = {
  graph: GPUCommandGraph;
  /** Tract polygons as four-level views. */
  tracts: ReturnType<typeof uploadChicagoTractPolygons>['polygons'];
  /** Response, one value per tract. */
  response: GraphDataView<'float32'>;
  /** Row-major predictors, three per tract. */
  predictors: GraphDataView<'float32'>;
  /** Tract centroids in frame metres. */
  positions: GraphDataView<'float32x2'>;
  /** Float32 per-frame parameters, present when the scenario requested them. */
  parameters?: GraphDataView<'float32'>;
  /** Uint32 permutation parameters, present when the scenario requested them. */
  permutationParameters?: GraphDataView<'uint32'>;
};

/** What a regression scenario hands back after running its example and tagging its captures. */
export type RegressionExecution = {
  encodeOptions?: GPUCommandGraphEncodeOptions<void>;
  /** Every view to read back, by capture name; transient ones are retained automatically. */
  views: Readonly<Record<string, GraphDataView>>;
  /** Roles and supports; `polygons` draws the values on the tract outlines. */
  tags: Readonly<Record<string, TractCaptureTag>>;
  /** Key spaces of non-tract captures (coefficients, tests, matrix powers). */
  keySpaces?: readonly GPUComputationKeySpace[];
};

/**
 * Appends one kernel that copies strided columns of a GPU table into packed vectors, so a
 * `stride x rows` estimator table can be drawn column by column. It reads and writes only on the
 * GPU, adds one node to the recorded graph and changes no value.
 */
export function addColumnExtraction(
  graph: GPUCommandGraph,
  props: {
    id: string;
    source: GraphDataView<'float32'>;
    /** Float32 values per source row. */
    stride: number;
    rowCount: number;
    /** Source column copied into each output, in order. */
    columns: readonly number[];
    outputs: readonly GraphDataView<'float32'>[];
  }
): void {
  assert(props.columns.length === props.outputs.length && props.outputs.length <= 7);
  graph.add(
    createWGSLKernelNode(graph, {
      id: props.id,
      operation: 'GPURegressionLens',
      variant: 'extract-columns',
      // Lens-only presentation node: its own namespace keeps it apart from the estimator's nodes.
      record: {
        namespace: 'lens',
        attrs: {purpose: 'copies estimator table columns for the result figure; computes nothing'}
      },
      bindings: [
        {name: 'source', view: props.source, type: 'f32', access: 'read'},
        ...props.outputs.map((view, index) => ({
          name: `column${index}`,
          view,
          type: 'f32' as const,
          access: 'read_write' as const
        }))
      ],
      invocationCount: props.rowCount,
      body: props.columns
        .map(
          (column, index) =>
            `column${index}[column${index}Offset + index] = source[sourceOffset + index * ${props.stride}u + ${column}u];`
        )
        .join('\n  ')
    })
  );
}

/** Columns of the stride-4 table of the two-stage, error and likelihood estimators. */
export const ESTIMATOR_TABLE_COLUMNS = {
  estimates: 0,
  standardErrors: 1,
  zStatistics: 2,
  pValues: 3
} as const;

/** Views read back from a stride-4 estimator table, with their roles. */
export type ExtractedTable = {
  views: Record<string, GraphDataView<'float32'>>;
  tags: Record<string, TractCaptureTag>;
  keySpaces: GPUComputationKeySpace[];
};

/**
 * Creates one packed output per named column of a strided GPU table and appends the kernel that
 * fills them. Outputs are named after the columns, so captures keep the names the formulas use.
 */
export function extractColumns(
  graph: GPUCommandGraph,
  props: {
    id: string;
    source: GraphDataView<'float32'>;
    stride: number;
    rowCount: number;
    /** Source column of each output name. */
    columns: Readonly<Record<string, number>>;
  }
): Record<string, GraphDataView<'float32'>> {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const names = Object.keys(props.columns);
  const outputs = names.map(name =>
    createTransientView(graph, name, 'float32', props.rowCount, usage)
  );
  addColumnExtraction(graph, {
    id: props.id,
    source: props.source,
    stride: props.stride,
    rowCount: props.rowCount,
    columns: names.map(name => props.columns[name]),
    outputs
  });
  return Object.fromEntries(names.map((name, index) => [name, outputs[index]]));
}

/**
 * Splits a strided GPU table into one captured column per name, all output role and sharing one
 * key space, which is what the result figure draws as a table (CARD-DESIGN 2.5).
 */
export function extractTable(
  graph: GPUCommandGraph,
  props: {
    id: string;
    source: GraphDataView<'float32'>;
    stride: number;
    rowCount: number;
    /** Source column of each captured name. */
    columns: Readonly<Record<string, number>>;
    keySpace: GPUComputationKeySpace;
    /** Row order and units, stated on every column. */
    meaning: string;
  }
): ExtractedTable {
  const views = extractColumns(graph, props);
  return {
    views,
    tags: Object.fromEntries(
      Object.keys(views).map(name => [
        name,
        {
          role: 'output' as const,
          keySpace: props.keySpace.id,
          meaning: `${name}; rows: ${props.meaning}`
        }
      ])
    ),
    keySpaces: [props.keySpace]
  };
}

/** Float values uploaded beside the design, handed to the example as `parameters`. */
export type RegressionParameterUploads = {
  parameters?: Float32Array;
  permutationParameters?: Uint32Array;
};

/**
 * Runs one regression card: uploads the shared design, lets the scenario run its example and add its
 * column extraction, retains what it returns and records it with tract, vertex and estimator key
 * spaces. Tract outlines are read back only when `withMap` (the card draws a map).
 */
export async function runRegressionScenario(props: {
  device: Device;
  id: string;
  model: RegressionModel;
  entry: SpatialOperatorEntry;
  choices: SpatialScenarioRun['choices'];
  metadata?: SpatialScenarioRun['metadata'];
  withMap: boolean;
  uploads?: RegressionParameterUploads;
  execute(input: RegressionExampleInput): Promise<RegressionExecution>;
}): Promise<Awaited<ReturnType<typeof runSpatialScenario>>> {
  const {model} = props;
  const retained: string[] = [];
  const graph = new GPUCommandGraph(props.device, {id: props.id, recording: {retain: retained}});
  const geometry = uploadChicagoTractPolygons(graph, model.tracts);
  const response = uploadSpatialView(graph, 'response', model.response, 'float32');
  const predictors = uploadSpatialView(graph, 'predictors', model.predictors, 'float32');
  const positions = uploadSpatialView(graph, 'positions', model.positions, 'float32x2');
  const parameters = props.uploads?.parameters
    ? uploadSpatialView(graph, 'parameters', props.uploads.parameters, 'float32')
    : undefined;
  const permutationParameters = props.uploads?.permutationParameters
    ? uploadSpatialView(
        graph,
        'permutationParameters',
        props.uploads.permutationParameters,
        'uint32'
      )
    : undefined;
  const uploaded = [response, predictors, positions, parameters, permutationParameters];
  try {
    const execution = await props.execute({
      graph,
      tracts: geometry.polygons,
      response: response.view,
      predictors: predictors.view,
      positions: positions.view,
      parameters: parameters?.view,
      permutationParameters: permutationParameters?.view
    });
    retained.push(...getRetainedViewIds(execution.views));
    const count = model.tracts.count;
    const geometryReference = {verticesCapture: 'vertices', offsetsCapture: 'ringOffsets'};
    const tags: Record<string, SpatialCaptureTag> = {};
    if (props.withMap) tags['vertices'] = {role: 'intermediate', keySpace: 'vertex'};
    for (const [name, view] of Object.entries(execution.views)) {
      // Untagged tract-length outputs are intermediates of the same support.
      if (view.length === count) tags[name] = {role: 'intermediate', keySpace: 'tract'};
    }
    for (const [name, {polygons, keySpace, ...tag}] of Object.entries(execution.tags)) {
      tags[name] = {
        ...tag,
        keySpace: keySpace ?? 'tract',
        ...(polygons && {polygons: geometryReference})
      };
    }
    const keySpaces: GPUComputationAxes['keySpaces'] = [
      {id: 'tract', label: 'Chicago tract', size: count, indexSpace: '1d'},
      ...(props.withMap
        ? [
            {
              id: 'vertex',
              label: 'Ring vertex',
              size: geometry.views.vertices.length,
              indexSpace: '1d' as const
            }
          ]
        : []),
      ...(execution.keySpaces ?? [])
    ];
    return await runSpatialScenario({
      id: props.id,
      graph,
      encodeOptions: execution.encodeOptions ?? {parameters: undefined},
      views: {
        ...(props.withMap && geometry.views),
        ...execution.views
      },
      entry: props.entry,
      choices: props.choices,
      metadata: {
        sampleTractCount: count,
        droppedTractCount: model.droppedCount,
        ...props.metadata
      },
      datasetVersion: model.tracts.dataset.id,
      seed: 1,
      axes: {
        axes: [
          {id: 'entity', label: 'Chicago tract', kind: 'data', keySpace: 'tract', size: count}
        ],
        keySpaces
      },
      tags
    });
  } finally {
    geometry.destroy();
    for (const upload of uploaded) upload?.destroy();
  }
}
