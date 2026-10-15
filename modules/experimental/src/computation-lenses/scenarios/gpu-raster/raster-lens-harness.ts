// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
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
  type GPUComputationCaptureElementType,
  type GPUComputationCaptureGeometry,
  type GPUComputationFormulaRef,
  type GPUComputationKeySpace,
  type GPUComputationNode,
  type GPUComputationRelation
} from '@luma.gl/gpgpu';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterMetadata} from '../../../gpu-raster/types';
import {mergeGPUCoreFormulaRefsForGraphs} from '../../../computation-views/packs/gpu-core/formulas';
import type {LensScenarioResult} from '../../types';

/** Storage formats a lens can upload or capture. */
export type RasterLensFormat = 'float32' | 'uint32' | 'sint32';

/** How one captured view appears in the record. */
export type RasterLensCaptureSpec = {
  /** Role in the element's input → output figure; captures without a role are intermediates. */
  role?: 'input' | 'output' | 'context' | 'intermediate';
  /**
   * `'raster'` draws the view on the lens grid (one value per cell). Vector outputs pass explicit
   * point, line or polygon geometry with their offsets and values capture names.
   */
  geometry?: 'raster' | GPUComputationCaptureGeometry;
  /** Reader-facing meaning, such as `elevation (m)` or `settledAt round`. */
  meaning?: string;
  /** The single answer the card figure points at (output captures only). */
  highlight?: {key: number; label: string};
  /**
   * Picks the highlight from the captured final values (finite values only): the cell with the
   * largest or smallest value, labelled from that value. Never chosen by a view.
   */
  highlightExtreme?: {pick: 'max' | 'min'; label: (value: number, key: number) => string};
  /** Key space id; raster captures default to `cells`. */
  keySpace?: string;
  /** Values per key (2 for vec2 coordinates). */
  components?: number;
  /** Keep one snapshot per key frame of {@link RunRasterLensProps.keyFrames}. */
  keyFrames?: boolean;
  /** File the final value under per-datum stamps instead of snapshots (settledAt rounds). */
  stamp?: boolean;
};

/** Upload, output and capture helpers handed to a scenario's `build`. */
export type RasterLensSession = {
  device: Device;
  /** Recording graph the example adds its contributor to. */
  graph: GPUCommandGraph;
  /** Uploads values as an imported, capturable storage view and optionally captures it. */
  input<Format extends RasterLensFormat>(
    name: string,
    format: Format,
    values: ArrayLike<number>,
    spec?: RasterLensCaptureSpec | false
  ): GraphDataView<Format>;
  /** Creates a zero-filled imported output view (default one value per grid cell) and captures it. */
  output<Format extends RasterLensFormat>(
    name: string,
    format: Format,
    spec?: RasterLensCaptureSpec | false,
    length?: number
  ): GraphDataView<Format>;
  /** Captures an existing packed view under `name`. */
  capture(name: string, view: GraphDataView, spec?: RasterLensCaptureSpec): void;
  /** Destroys `resource` after the run (parameter buffers created by the example). */
  track(resource: {destroy(): void} | undefined): void;
};

/** Properties of {@link runRasterLens}. */
export type RunRasterLensProps = {
  device: Device;
  /** Scenario id; also the record id. */
  scenarioId: string;
  /** Entry id, also the record domain: `gpu-terrain` or `gpu-raster`. */
  domain: string;
  /** Human label of the computation graph. */
  label: string;
  /** Lens grid: shape and cell size of every `'raster'` capture. */
  grid: GPURasterMetadata;
  /** `projected` (metres) or `geographic` (degrees). */
  coordinateModel: 'projected' | 'geographic';
  /** Horizontal unit of the grid, such as `m` or `deg`. */
  unit: string;
  /** Dataset id and version stamped into the run id. */
  dataset: {id: string; version: string};
  /** Analyst choices the run actually used. */
  choices?: readonly GPUAnalysisChoice[];
  /** Formula refs (the pack's curated formula for this element). */
  formulas?: readonly GPUComputationFormulaRef[];
  /** Extra key spaces (observers, vertices, particles, zones); `cells` is always present. */
  keySpaces?: readonly GPUComputationKeySpace[];
  relations?: readonly GPUComputationRelation[];
  /**
   * Key frames of an iterative element: the graph is executed in steps of at most
   * `ceil(nodeCount / maximumSteps)` nodes (default: one node per step, so every executed round
   * is observed before rounds past the executed iteration count are dropped), and every capture with `keyFrames: true` keeps one
   * snapshot per distinct round of `loop` (a loop id or suffix of one), at most `maximumFrames`,
   * always including the final state. Stepping splits submissions (stated in the metrics encoding).
   */
  keyFrames?: {loop: string; maximumFrames?: number; maximumSteps?: number};
  /** Adds the contributor and declares captures. */
  build(session: RasterLensSession): Promise<void>;
};

const CAPTURE_USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
const ELEMENT_TYPES: Record<RasterLensFormat, GPUComputationCaptureElementType> = {
  float32: 'f32',
  uint32: 'u32',
  sint32: 'i32'
};

type RegisteredCapture = {name: string; view: GraphDataView; spec: RasterLensCaptureSpec};

/**
 * Runs one raster or terrain lens scenario on the real device: builds a recording graph, lets the
 * scenario add its element through the example code, compiles, describes, executes once (in steps
 * when key frames are requested) and returns the validated record with its captures sidecar.
 */
export async function runRasterLens(props: RunRasterLensProps): Promise<LensScenarioResult> {
  const {device, grid} = props;
  const cellCount = grid.width * grid.height;
  const graph = new GPUCommandGraph(device, {
    id: props.scenarioId.replace(/\./g, '-'),
    recording: {}
  });
  const owned: {destroy(): void}[] = [];
  const captures: RegisteredCapture[] = [];
  const register = (name: string, view: GraphDataView, spec: RasterLensCaptureSpec | false) => {
    if (spec === false) return;
    // Capture names are the record's join keys; a duplicate would silently shadow a figure layer.
    if (captures.some(capture => capture.name === name)) {
      throw new Error(`${props.scenarioId}: duplicate capture ${name}`);
    }
    captures.push({name, view, spec});
  };
  const importValues = <Format extends RasterLensFormat>(
    name: string,
    format: Format,
    data: ArrayBufferView
  ): GraphDataView<Format> => {
    const buffer = device.createBuffer({id: `${graph.id}-${name}`, data, usage: CAPTURE_USAGE});
    owned.push(buffer);
    const handle = graph.importBuffer(
      {id: name, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format, length: data.byteLength / 4});
  };
  const session: RasterLensSession = {
    device,
    graph,
    input: (name, format, values, spec = {role: 'input', geometry: 'raster'}) => {
      const view = importValues(name, format, toTypedArray(format, values));
      register(name, view, spec);
      return view;
    },
    output: (name, format, spec = {role: 'output', geometry: 'raster'}, length = cellCount) => {
      const view = importValues(name, format, toTypedArray(format, new Array(length).fill(0)));
      register(name, view, spec);
      return view;
    },
    capture: (name, view, spec = {}) => register(name, view, spec),
    track: resource => {
      if (resource) owned.push(resource);
    }
  };
  try {
    await props.build(session);
    const compiled = await graph.compileAsync();
    try {
      const description = compiled.describe({label: props.label, domain: props.domain});
      const record = createGPUComputationRecord({
        id: props.scenarioId,
        adapter: getGPUComputationAdapter(device),
        domains: [props.domain],
        graphs: [description],
        axes: {
          keySpaces: [
            {
              id: 'cells',
              label: 'grid cells',
              size: cellCount,
              indexSpace: '2d',
              shape: [grid.width, grid.height]
            },
            ...(props.keySpaces ?? [])
          ],
          relations: props.relations ?? []
        },
        formulas: mergeGPUCoreFormulaRefsForGraphs([description], props.formulas),
        choices: props.choices,
        datasetVersion: `${props.dataset.id}@${props.dataset.version}`
      });
      const graphId = description.graph.id;
      const nodeOrder = compiled.stats.nodeOrder;
      const maximumSteps = props.keyFrames?.maximumSteps ?? nodeOrder.length;
      const result = await captureGPUCommandGraph(compiled, {
        encodeOptions: {parameters: undefined},
        views: Object.fromEntries(captures.map(capture => [capture.name, capture.view])),
        elementTypes: Object.fromEntries(
          captures.map(capture => [
            capture.name,
            ELEMENT_TYPES[capture.view.format as RasterLensFormat] ?? 'u32'
          ])
        ),
        at: props.keyFrames ? 'step' : 'end',
        ...(props.keyFrames && {
          budget: {
            maximumInvocationCount: Number.MAX_SAFE_INTEGER,
            maximumNodeCount: Math.max(1, Math.ceil(nodeOrder.length / maximumSteps))
          }
        }),
        runId: record.runId,
        graphId
      });
      const sidecar = arrangeCaptures(props, captures, result.captures, description.graph.nodes, {
        nodeOrder
      });
      const metrics = createGPUComputationMetrics({
        runId: record.runId,
        graph: graphId,
        // No timestamp queries are written by lens runs: node timings stay absent, never estimated.
        timing: {nodes: []},
        encoding: {
          coalesced: !props.keyFrames,
          timestamped: false,
          taps: 0,
          lensTwin: true
        }
      });
      const captureSidecar = createGPUComputationCaptures({
        runId: record.runId,
        snapshots: sidecar.snapshots,
        stamps: sidecar.stamps
      });
      assertValid(props.scenarioId, validateGPUComputationRecord(record));
      assertValid(
        props.scenarioId,
        validateGPUComputationSidecars(record, {metrics: [metrics], captures: captureSidecar})
      );
      return {...record, metrics: [metrics], captures: captureSidecar};
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const resource of owned) resource.destroy();
  }
}

/** Applies roles and geometry, keeps final values, thins key frames and files stamps. */
function arrangeCaptures(
  props: RunRasterLensProps,
  registered: readonly RegisteredCapture[],
  raw: readonly GPUComputationCapture[],
  nodes: readonly GPUComputationNode[],
  plan: {nodeOrder: readonly string[]}
): {snapshots: GPUComputationCapture[]; stamps: GPUComputationCapture[]} {
  const loop = props.keyFrames?.loop;
  // Rounds of the key-frame loop completed after each node, in execution order.
  const roundsAfterNode = new Map<string, number>();
  let rounds = 0;
  const nodeById = new Map(nodes.map(node => [node.id, node]));
  for (const nodeId of plan.nodeOrder) {
    const loopCoordinate = nodeById.get(nodeId)?.loop;
    if (
      loop &&
      loopCoordinate &&
      nodeById.get(nodeId)?.attrs['loopRole'] !== 'gate' &&
      (loopCoordinate.id === loop || loopCoordinate.id.endsWith(loop))
    ) {
      rounds = loopCoordinate.iteration + 1;
    }
    roundsAfterNode.set(nodeId, rounds);
  }
  const snapshots: GPUComputationCapture[] = [];
  const stamps: GPUComputationCapture[] = [];
  for (const capture of registered) {
    const frames = raw.filter(candidate => candidate.name === capture.name);
    const finalFrame = frames.at(-1);
    // captureGPUCommandGraph returns at least one frame per requested view.
    if (!finalFrame) throw new Error(`${props.scenarioId}: no capture for ${capture.name}`);
    const decorate = (frame: GPUComputationCapture, iteration?: number): GPUComputationCapture => ({
      ...frame,
      ...(iteration !== undefined && {id: `${frame.id}-round-${iteration}`, iteration}),
      ...(capture.spec.role && {role: capture.spec.role}),
      ...(capture.spec.meaning && {meaning: capture.spec.meaning}),
      ...(capture.spec.highlight && {highlight: capture.spec.highlight}),
      ...(capture.spec.components && {components: capture.spec.components}),
      ...((capture.spec.keySpace || capture.spec.geometry === 'raster') && {
        keySpace: capture.spec.keySpace ?? 'cells'
      }),
      ...(capture.spec.geometry && {
        geometry:
          capture.spec.geometry === 'raster'
            ? {
                kind: 'raster',
                shape: [props.grid.width, props.grid.height],
                coordinateModel: props.coordinateModel,
                cellSize: Math.abs(props.grid.affine[0]),
                unit: props.unit
              }
            : capture.spec.geometry
      })
    });
    const extreme = capture.spec.highlightExtreme;
    if (extreme) {
      const key = findExtreme(finalFrame.values, extreme.pick);
      if (key !== undefined) {
        capture.spec = {
          ...capture.spec,
          highlight: {key, label: extreme.label(finalFrame.values[key], key)}
        };
      }
    }
    if (capture.spec.stamp) {
      stamps.push(decorate(finalFrame));
    } else if (capture.spec.keyFrames && loop && frames.length > 1) {
      snapshots.push(
        ...selectKeyFrames(
          frames,
          roundsAfterNode,
          props.keyFrames?.maximumFrames ?? 8,
          getExecutedRounds(registered, raw, loop)
        ).map(({frame, iteration}) => decorate(frame, iteration))
      );
    } else {
      snapshots.push(decorate(finalFrame));
    }
  }
  return {snapshots, stamps};
}

/**
 * Rounds the GPU really executed for `loop`, read from the capture whose meaning is
 * `executed iterations: <loop id>` (matched exactly or by suffix), or `undefined` without one.
 */
function getExecutedRounds(
  registered: readonly RegisteredCapture[],
  raw: readonly GPUComputationCapture[],
  loop: string
): number | undefined {
  const counter = registered.find(capture => {
    const prefix = 'executed iterations: ';
    const meaning = capture.spec.meaning;
    if (!meaning?.startsWith(prefix)) return false;
    const loopId = meaning.slice(prefix.length);
    return loopId === loop || loopId.endsWith(loop) || loop.endsWith(loopId);
  });
  const value =
    counter && raw.filter(candidate => candidate.name === counter.name).at(-1)?.values[0];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

/**
 * One frame per distinct completed round, thinned log-spaced to `maximum` (early rounds change
 * most), first and final rounds always kept. With `executedRounds`, rounds past it are dropped:
 * gated rounds after convergence never ran, so their observations are the final state, not rounds.
 */
export function selectKeyFrames(
  frames: readonly GPUComputationCapture[],
  roundsAfterNode: ReadonlyMap<string, number>,
  maximum: number,
  executedRounds?: number
): {frame: GPUComputationCapture; iteration: number}[] {
  const byRound = new Map<number, GPUComputationCapture>();
  for (const frame of frames) {
    const round = frame.afterNode === undefined ? 0 : (roundsAfterNode.get(frame.afterNode) ?? 0);
    if (executedRounds !== undefined && round > executedRounds) continue;
    // The last frame observed at a round count is the state after that round.
    byRound.set(round, frame);
  }
  const ordered = [...byRound.entries()].sort((first, second) => first[0] - second[0]);
  // The final frame is the last observation, even when trailing non-loop nodes ran after the loop.
  const final = frames[frames.length - 1];
  if (!ordered.length) return [{frame: final, iteration: executedRounds ?? 0}];
  ordered[ordered.length - 1] = [ordered[ordered.length - 1][0], final];
  if (ordered.length <= maximum) {
    return ordered.map(([iteration, frame]) => ({frame, iteration}));
  }
  const selected = new Set<number>([0, ordered.length - 1]);
  const last = Math.log(ordered.length);
  for (let step = 1; selected.size < maximum && step < 4 * maximum; step++) {
    // Candidate positions spaced evenly in log(index + 1).
    const position = Math.round(Math.exp((step * last) / (maximum - 1)) - 1);
    if (position > 0 && position < ordered.length - 1) selected.add(position);
  }
  for (let index = 1; selected.size < maximum && index < ordered.length - 1; index++) {
    selected.add(index);
  }
  return [...selected]
    .sort((first, second) => first - second)
    .map(index => ({frame: ordered[index][1], iteration: ordered[index][0]}));
}

function findExtreme(values: ArrayLike<number>, pick: 'max' | 'min'): number | undefined {
  let best: number | undefined;
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    if (best === undefined || (pick === 'max' ? value > values[best] : value < values[best])) {
      best = index;
    }
  }
  return best;
}

function toTypedArray(format: RasterLensFormat, values: ArrayLike<number>): ArrayBufferView {
  switch (format) {
    case 'float32':
      return Float32Array.from(values);
    case 'uint32':
      return Uint32Array.from(values);
    default:
      return Int32Array.from(values);
  }
}

function assertValid(
  scenarioId: string,
  report: {valid: boolean; issues: readonly {level: string; path: string; message: string}[]}
): void {
  if (report.valid) return;
  const errors = report.issues
    .filter(issue => issue.level === 'error')
    .map(issue => `${issue.path}: ${issue.message}`);
  throw new Error(`${scenarioId}: invalid record\n${errors.join('\n')}`);
}
