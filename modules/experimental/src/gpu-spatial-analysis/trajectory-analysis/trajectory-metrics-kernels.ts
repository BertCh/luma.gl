// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
// GPUSegmentedReduction is not re-exported from the gpu-core index yet.
import {GPUSegmentedReduction, type GPUSegmentedReductionOperation} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {TIME_WORDS_WGSL} from '../../gpu-dataframe/time-window-filter/time-words';

const OPERATION = 'GPUTrajectoryMetrics';

/** WebGPU default `maxStorageBuffersPerShaderStage`. Kernels needing more are split. */
const MAXIMUM_STORAGE_BINDINGS = 8;

/**
 * How sample times are stored: `'float32'` relative f32 (default), `'double-single'` f32 high
 * parts plus f32 low parts, or `'words'` Int64 `(low, high)` words.
 *
 * @internal
 */
export type TrajectoryTimeMode = 'float32' | 'double-single' | 'words';

/** Sample time inputs shared by every kernel that reads timestamps. @internal */
export type TrajectoryTimeViews = {
  /** f32 times (relative or double-single high parts) or Int64 words. */
  timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
  /** Double-single low parts, only with f32 `timestamps`. */
  timestampsLow?: GraphDataView<'float32'>;
};

/** Returns the time mode implied by the bound views. @internal */
export function getTrajectoryTimeMode(time: TrajectoryTimeViews): TrajectoryTimeMode {
  if (time.timestamps.format === 'uint32x2') {
    return 'words';
  }
  return time.timestampsLow ? 'double-single' : 'float32';
}

/** Returns the storage bindings (`timestamps`, then `timestampsLow` if present). @internal */
export function getTrajectoryTimeBindings(time: TrajectoryTimeViews): WGSLKernelBinding[] {
  const bindings: WGSLKernelBinding[] = [
    {
      name: 'timestamps',
      view: time.timestamps,
      type: time.timestamps.format === 'uint32x2' ? 'u32' : 'f32',
      access: 'read'
    }
  ];
  if (time.timestampsLow) {
    bindings.push({name: 'timestampsLow', view: time.timestampsLow, type: 'f32', access: 'read'});
  }
  return bindings;
}

/**
 * WGSL time helpers for one time mode, in the units of the timestamps.
 *
 * - `timeDifferenceRows(a, b) -> f32`: `t[a] - t[b]` for two rows. Words subtract exactly and then
 *   convert to f32; double-single is `(aHigh - bHigh) + (aLow - bLow)`; f32 is a plain subtraction.
 * - `isTimeNonDecreasing(a, b) -> bool`: exact `t[a] >= t[b]`. Double-single compares the high
 *   parts first and the low parts only on ties, so fast-math cannot drop the low parts.
 *
 * @internal
 */
export function getTrajectoryTimeSource(mode: TrajectoryTimeMode): string {
  switch (mode) {
    case 'words':
      return /* wgsl */ `${TIME_WORDS_WGSL}
fn getTimeWords(row: u32) -> vec2<u32> {
  return vec2<u32>(timestamps[timestampsOffset + 2u * row], timestamps[timestampsOffset + 2u * row + 1u]);
}

fn timeDifferenceRows(a: u32, b: u32) -> f32 {
  return timeWordsToF32(timeWordsSubtract(getTimeWords(a), getTimeWords(b)));
}

fn isTimeNonDecreasing(a: u32, b: u32) -> bool {
  return !timeWordsIsNegative(timeWordsSubtract(getTimeWords(a), getTimeWords(b)));
}
`;
    case 'double-single':
      return /* wgsl */ `
fn timeDifferenceRows(a: u32, b: u32) -> f32 {
  return (timestamps[timestampsOffset + a] - timestamps[timestampsOffset + b]) +
    (timestampsLow[timestampsLowOffset + a] - timestampsLow[timestampsLowOffset + b]);
}

fn isTimeNonDecreasing(a: u32, b: u32) -> bool {
  let aHigh = timestamps[timestampsOffset + a];
  let bHigh = timestamps[timestampsOffset + b];
  return aHigh > bHigh ||
    (aHigh == bHigh && timestampsLow[timestampsLowOffset + a] >= timestampsLow[timestampsLowOffset + b]);
}
`;
    default:
      return /* wgsl */ `
fn timeDifferenceRows(a: u32, b: u32) -> f32 {
  return timestamps[timestampsOffset + a] - timestamps[timestampsOffset + b];
}

fn isTimeNonDecreasing(a: u32, b: u32) -> bool {
  return timestamps[timestampsOffset + a] >= timestamps[timestampsOffset + b];
}
`;
  }
}

/**
 * WGSL helpers shared by the kernels that map rows to tracks.
 *
 * Requires a `trackOffsets` storage binding. `findTrack(row)` returns the track whose row range
 * `[offsets[track], offsets[track + 1])` contains `row`, or `NO_TRACK` for rows outside every
 * track. Empty tracks are skipped because the search takes the last track that starts at or before
 * the row.
 *
 * @internal
 */
export function getTrackSearchSource(trackCount: number): string {
  return /* wgsl */ `
const TRACK_COUNT: u32 = ${trackCount}u;
const NO_TRACK: u32 = 0xffffffffu;

fn findTrack(row: u32) -> u32 {
  var low = 0u;
  var high = TRACK_COUNT;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    if (trackOffsets[trackOffsetsOffset + middle] <= row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u || row >= trackOffsets[trackOffsetsOffset + TRACK_COUNT]) {
    return NO_TRACK;
  }
  return low - 1u;
}`;
}

/**
 * WGSL helpers for step distance and slowness. Requires `positions` (f32 pairs) and the time
 * bindings and {@link getTrajectoryTimeSource} helpers, and `parameters` when `isSlowStep` is used. `row` must be a step row (not the first
 * row of its track).
 *
 * @internal
 */
export function getStepSource(includeSlowStep: boolean = true): string {
  const stepSource = /* wgsl */ `
fn getStepDistance(row: u32) -> f32 {
  let deltaX = positions[positionsOffset + 2u * row] - positions[positionsOffset + 2u * (row - 1u)];
  let deltaY = positions[positionsOffset + 2u * row + 1u] - positions[positionsOffset + 2u * (row - 1u) + 1u];
  return sqrt(deltaX * deltaX + deltaY * deltaY);
}

fn getStepDuration(row: u32) -> f32 {
  return timeDifferenceRows(row, row - 1u);
}
`;
  // A step is slow when time does not run backwards and it is either a zero-distance step or its
  // speed is below the threshold. The comparison multiplies instead of dividing.
  const slowStepSource = /* wgsl */ `
fn isSlowStep(row: u32) -> bool {
  let stepDuration = getStepDuration(row);
  let stepDistance = getStepDistance(row);
  return isTimeNonDecreasing(row, row - 1u) &&
    (stepDistance == 0.0 || stepDistance < parameters[parametersOffset] * stepDuration);
}`;
  return includeSlowStep ? `${stepSource}${slowStepSource}` : stepSource;
}

/** Properties for {@link createTrajectoryStepsNode}. @internal */
export type TrajectoryStepsProps = TrajectoryTimeViews & {
  id: string;
  positions: GraphDataView<'float32x2'>;
  trackOffsets: GraphDataView<'uint32'>;
  /** Present when any stop output is requested. */
  parameters?: GraphDataView<'float32'>;
  stepDistances?: GraphDataView<'float32'>;
  stepSpeeds?: GraphDataView<'float32'>;
  runStartFlags?: GraphDataView<'uint32'>;
  runEndFlags?: GraphDataView<'uint32'>;
};

/**
 * Builds the per-row step kernel(s): step distance, step speed, and slow-run start and end flags.
 * Every output word is rewritten on every encoding.
 *
 * The all-outputs kernel binds up to eight storage buffers; double-single times add a ninth. Only
 * then is it split into a distance and speed node and a run-flag node, each well under the limit.
 *
 * @internal
 */
export function createTrajectoryStepsNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TrajectoryStepsProps
): GPUCommandNode<Parameters>[] {
  const hasRuns = Boolean(props.runStartFlags && props.runEndFlags && props.parameters);
  const bindingCount =
    3 +
    (props.timestampsLow ? 1 : 0) +
    (hasRuns ? 3 : 0) +
    (props.stepDistances ? 1 : 0) +
    (props.stepSpeeds ? 1 : 0);
  if (bindingCount <= MAXIMUM_STORAGE_BINDINGS || !hasRuns) {
    return [createTrajectoryStepsNode(graph, props)];
  }
  const {runStartFlags, runEndFlags, parameters, ...rowOutputs} = props;
  const nodes: GPUCommandNode<Parameters>[] = [];
  if (props.stepDistances || props.stepSpeeds) {
    nodes.push(createTrajectoryStepsNode(graph, rowOutputs));
  }
  nodes.push(
    createTrajectoryStepsNode(graph, {
      id: `${props.id}-runs`,
      positions: props.positions,
      timestamps: props.timestamps,
      timestampsLow: props.timestampsLow,
      trackOffsets: props.trackOffsets,
      parameters,
      runStartFlags,
      runEndFlags
    })
  );
  return nodes;
}

function createTrajectoryStepsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TrajectoryStepsProps
): GPUCommandNode<Parameters> {
  const trackCount = props.trackOffsets.length - 1;
  const hasRuns = Boolean(props.runStartFlags && props.runEndFlags && props.parameters);
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    ...getTrajectoryTimeBindings(props),
    {name: 'trackOffsets', view: props.trackOffsets, type: 'u32', access: 'read'}
  ];
  if (hasRuns) {
    bindings.push({name: 'parameters', view: props.parameters!, type: 'f32', access: 'read'});
  }
  if (props.stepDistances) {
    bindings.push({
      name: 'stepDistances',
      view: props.stepDistances,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (props.stepSpeeds) {
    bindings.push({name: 'stepSpeeds', view: props.stepSpeeds, type: 'f32', access: 'read_write'});
  }
  if (hasRuns) {
    bindings.push(
      {name: 'runStartFlags', view: props.runStartFlags!, type: 'u32', access: 'read_write'},
      {name: 'runEndFlags', view: props.runEndFlags!, type: 'u32', access: 'read_write'}
    );
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'steps',
    bindings,
    invocationCount: props.positions.length,
    declarations: `${getTrackSearchSource(trackCount)}
${getTrajectoryTimeSource(getTrajectoryTimeMode(props))}
${getStepSource(hasRuns)}`,
    body: /* wgsl */ `
  var stepDistance = 0.0;
  var stepSpeed = 0.0;
  var runStart = 0u;
  var runEnd = 0u;
  let track = findTrack(index);
  if (track != NO_TRACK) {
    let trackStart = trackOffsets[trackOffsetsOffset + track];
    let trackEnd = trackOffsets[trackOffsetsOffset + track + 1u];
    if (index > trackStart) {
      stepDistance = getStepDistance(index);
      let stepDuration = getStepDuration(index);
      if (stepDuration > 0.0) {
        stepSpeed = stepDistance / stepDuration;
      }
      ${
        hasRuns
          ? `if (isSlowStep(index)) {
        let previousSlow = index - 1u > trackStart && isSlowStep(index - 1u);
        let nextSlow = index + 1u < trackEnd && isSlowStep(index + 1u);
        runStart = select(1u, 0u, previousSlow);
        runEnd = select(1u, 0u, nextSlow);
      }`
          : ''
      }
    }
  }
  ${props.stepDistances ? 'stepDistances[stepDistancesOffset + index] = stepDistance;' : ''}
  ${props.stepSpeeds ? 'stepSpeeds[stepSpeedsOffset + index] = stepSpeed;' : ''}
  ${
    hasRuns
      ? `runStartFlags[runStartFlagsOffset + index] = runStart;
  runEndFlags[runEndFlagsOffset + index] = runEnd;`
      : ''
  }`
  });
}

/** Properties for {@link createTrajectoryStepColumnsNode}. @internal */
export type TrajectoryStepColumnsProps = TrajectoryTimeViews & {
  id: string;
  positions: GraphDataView<'float32x2'>;
  trackOffsets: GraphDataView<'uint32'>;
  stepSpeeds?: GraphDataView<'float32'>;
  stepHeadings?: GraphDataView<'float32'>;
  stepAccelerations?: GraphDataView<'float32'>;
};

/**
 * Builds the per-row kernel that writes the speed, heading and acceleration of the step ending at
 * each row (0 for the first row of a track and for rows outside every track). Every output word is
 * rewritten on every encoding. At most seven storage bindings (double-single times).
 *
 * @internal
 */
export function createTrajectoryStepColumnsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TrajectoryStepColumnsProps
): GPUCommandNode<Parameters> {
  const trackCount = props.trackOffsets.length - 1;
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    ...getTrajectoryTimeBindings(props),
    {name: 'trackOffsets', view: props.trackOffsets, type: 'u32', access: 'read'}
  ];
  for (const [name, view] of [
    ['stepSpeeds', props.stepSpeeds],
    ['stepHeadings', props.stepHeadings],
    ['stepAccelerations', props.stepAccelerations]
  ] as const) {
    if (view) {
      bindings.push({name, view, type: 'f32', access: 'read_write'});
    }
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'step-columns',
    bindings,
    invocationCount: props.positions.length,
    declarations: `${getTrackSearchSource(trackCount)}
${getTrajectoryTimeSource(getTrajectoryTimeMode(props))}
${getStepSource(false)}

fn getStepSpeed(row: u32) -> f32 {
  let duration = getStepDuration(row);
  if (duration > 0.0) {
    return getStepDistance(row) / duration;
  }
  return 0.0;
}`,
    body: /* wgsl */ `
  var speed = 0.0;
  var heading = 0.0;
  var acceleration = 0.0;
  let track = findTrack(index);
  if (track != NO_TRACK) {
    let trackStart = trackOffsets[trackOffsetsOffset + track];
    if (index > trackStart) {
      speed = getStepSpeed(index);
      let deltaX = positions[positionsOffset + 2u * index] - positions[positionsOffset + 2u * (index - 1u)];
      let deltaY = positions[positionsOffset + 2u * index + 1u] - positions[positionsOffset + 2u * (index - 1u) + 1u];
      if (deltaX != 0.0 || deltaY != 0.0) {
        heading = atan2(deltaY, deltaX);
      }
      let duration = getStepDuration(index);
      if (index - 1u > trackStart && duration > 0.0) {
        acceleration = (speed - getStepSpeed(index - 1u)) / duration;
      }
    }
  }
  ${props.stepSpeeds ? 'stepSpeeds[stepSpeedsOffset + index] = speed;' : ''}
  ${props.stepHeadings ? 'stepHeadings[stepHeadingsOffset + index] = heading;' : ''}
  ${props.stepAccelerations ? 'stepAccelerations[stepAccelerationsOffset + index] = acceleration;' : ''}`
  });
}

/** Properties for {@link createTrajectoryFinalizeNode}. @internal */
export type TrajectoryFinalizeProps = TrajectoryTimeViews & {
  id: string;
  trackOffsets: GraphDataView<'uint32'>;
  /** Required when `averageSpeeds` is given. */
  trackLengths?: GraphDataView<'float32'>;
  trackDurations?: GraphDataView<'float32'>;
  averageSpeeds?: GraphDataView<'float32'>;
};

/** Builds the per-track kernel that writes durations and average speeds. @internal */
export function createTrajectoryFinalizeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TrajectoryFinalizeProps
): GPUCommandNode<Parameters> {
  const trackCount = props.trackOffsets.length - 1;
  const bindings: WGSLKernelBinding[] = [
    ...getTrajectoryTimeBindings(props),
    {name: 'trackOffsets', view: props.trackOffsets, type: 'u32', access: 'read'}
  ];
  if (props.averageSpeeds && props.trackLengths) {
    bindings.push({name: 'trackLengths', view: props.trackLengths, type: 'f32', access: 'read'});
  }
  if (props.trackDurations) {
    bindings.push({
      name: 'trackDurations',
      view: props.trackDurations,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (props.averageSpeeds) {
    bindings.push({
      name: 'averageSpeeds',
      view: props.averageSpeeds,
      type: 'f32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'finalize',
    bindings,
    invocationCount: trackCount,
    declarations: getTrajectoryTimeSource(getTrajectoryTimeMode(props)),
    body: /* wgsl */ `
  let trackStart = trackOffsets[trackOffsetsOffset + index];
  let trackEnd = trackOffsets[trackOffsetsOffset + index + 1u];
  var trackDuration = 0.0;
  if (trackEnd >= trackStart + 2u) {
    trackDuration = timeDifferenceRows(trackEnd - 1u, trackStart);
  }
  ${props.trackDurations ? 'trackDurations[trackDurationsOffset + index] = trackDuration;' : ''}
  ${
    props.averageSpeeds
      ? `var averageSpeed = 0.0;
  if (trackDuration > 0.0) {
    averageSpeed = trackLengths[trackLengthsOffset + index] / trackDuration;
  }
  averageSpeeds[averageSpeedsOffset + index] = averageSpeed;`
      : ''
  }`
  });
}

/** Builds the kernel that writes identity row indices into `rowIds` and `runIndices`. @internal */
export function createTrajectoryRowIdsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rowIds: GraphDataView<'uint32'>;
    rowCount: number;
    runIndices: GraphDataView<'uint32'>;
    runCapacity: number;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'row-ids',
    bindings: [
      {name: 'rowIds', view: props.rowIds, type: 'u32', access: 'read_write'},
      {name: 'runIndices', view: props.runIndices, type: 'u32', access: 'read_write'}
    ],
    invocationCount: Math.max(props.rowCount, props.runCapacity),
    declarations: `const ROW_COUNT: u32 = ${props.rowCount}u;
const RUN_CAPACITY: u32 = ${props.runCapacity}u;`,
    body: /* wgsl */ `
  if (index < ROW_COUNT) {
    rowIds[rowIdsOffset + index] = index;
  }
  if (index < RUN_CAPACITY) {
    runIndices[runIndicesOffset + index] = index;
  }`
  });
}

/** Properties for {@link createTrajectoryQualifyNode}. @internal */
export type TrajectoryQualifyProps = TrajectoryTimeViews & {
  id: string;
  trackOffsets: GraphDataView<'uint32'>;
  parameters: GraphDataView<'float32'>;
  runStarts: GraphDataView<'uint32'>;
  runEnds: GraphDataView<'uint32'>;
  runCount: GraphDataView<'uint32'>;
  qualifyFlags: GraphDataView<'uint32'>;
  runTracks: GraphDataView<'uint32'>;
  runCapacity: number;
};

/**
 * Builds the per-run kernel that tests the minimum duration and writes the qualify flag and the
 * owning track (`0xffffffff` for runs that do not qualify or do not exist).
 *
 * The single kernel binds eight storage buffers; double-single times add a ninth. Only then it is
 * split into a flag node and a track node that reads the flags.
 *
 * @internal
 */
export function createTrajectoryQualifyNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TrajectoryQualifyProps
): GPUCommandNode<Parameters>[] {
  const timeBindings = getTrajectoryTimeBindings(props);
  const timeSource = getTrajectoryTimeSource(getTrajectoryTimeMode(props));
  const runBindings: WGSLKernelBinding[] = [
    {name: 'runStarts', view: props.runStarts, type: 'u32', access: 'read'},
    {name: 'runEnds', view: props.runEnds, type: 'u32', access: 'read'},
    {name: 'runCountIn', view: props.runCount, type: 'u32', access: 'read'}
  ];
  const trackBindings: WGSLKernelBinding[] = [
    {name: 'trackOffsets', view: props.trackOffsets, type: 'u32', access: 'read'}
  ];
  const outputBindings: WGSLKernelBinding[] = [
    {name: 'qualifyFlags', view: props.qualifyFlags, type: 'u32', access: 'read_write'},
    {name: 'runTracks', view: props.runTracks, type: 'u32', access: 'read_write'}
  ];
  const parameterBindings: WGSLKernelBinding[] = [
    {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'}
  ];
  const qualifySource = /* wgsl */ `
  var flag = 0u;
  if (index < runCountIn[runCountInOffset]) {
    let firstStepRow = runStarts[runStartsOffset + index];
    let lastStepRow = runEnds[runEndsOffset + index];
    if (timeDifferenceRows(lastStepRow, firstStepRow - 1u) >= parameters[parametersOffset + 1u]) {
      flag = 1u;
    }
  }`;
  const common = {operation: OPERATION, invocationCount: props.runCapacity} as const;
  const trackDeclarations = getTrackSearchSource(props.trackOffsets.length - 1);
  const allBindings = [
    ...runBindings,
    ...timeBindings,
    ...parameterBindings,
    ...trackBindings,
    ...outputBindings
  ];
  if (allBindings.length <= MAXIMUM_STORAGE_BINDINGS) {
    return [
      createWGSLKernelNode<Parameters>(graph, {
        ...common,
        id: props.id,
        variant: 'qualify',
        bindings: allBindings,
        declarations: `${trackDeclarations}\n${timeSource}`,
        body: `${qualifySource}
  var track = NO_TRACK;
  if (flag == 1u) {
    track = findTrack(runStarts[runStartsOffset + index]);
  }
  qualifyFlags[qualifyFlagsOffset + index] = flag;
  runTracks[runTracksOffset + index] = track;`
      })
    ];
  }
  return [
    createWGSLKernelNode<Parameters>(graph, {
      ...common,
      id: props.id,
      variant: 'qualify',
      bindings: [...runBindings, ...timeBindings, ...parameterBindings, outputBindings[0]],
      declarations: timeSource,
      body: `${qualifySource}
  qualifyFlags[qualifyFlagsOffset + index] = flag;`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      ...common,
      id: `${props.id}-tracks`,
      variant: 'qualify-tracks',
      bindings: [
        runBindings[0],
        runBindings[2],
        ...trackBindings,
        {name: 'qualifyFlags', view: props.qualifyFlags, type: 'u32', access: 'read'},
        outputBindings[1]
      ],
      declarations: trackDeclarations,
      body: /* wgsl */ `
  var track = NO_TRACK;
  if (index < runCountIn[runCountInOffset] && qualifyFlags[qualifyFlagsOffset + index] == 1u) {
    track = findTrack(runStarts[runStartsOffset + index]);
  }
  runTracks[runTracksOffset + index] = track;`
    })
  ];
}

/**
 * Builds the kernel that writes monotonic centroid segment offsets: segment `2r` is the row range
 * of dwell `r` and segment `2r + 1` is the gap before the next dwell. Nonexistent runs get empty
 * segments at `rowCount`.
 *
 * @internal
 */
export function createTrajectoryCentroidOffsetsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    runStarts: GraphDataView<'uint32'>;
    runEnds: GraphDataView<'uint32'>;
    runCount: GraphDataView<'uint32'>;
    centroidOffsets: GraphDataView<'uint32'>;
    runCapacity: number;
    rowCount: number;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'centroid-offsets',
    bindings: [
      {name: 'runStarts', view: props.runStarts, type: 'u32', access: 'read'},
      {name: 'runEnds', view: props.runEnds, type: 'u32', access: 'read'},
      {name: 'runCountIn', view: props.runCount, type: 'u32', access: 'read'},
      {name: 'centroidOffsets', view: props.centroidOffsets, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.runCapacity,
    declarations: `const ROW_COUNT: u32 = ${props.rowCount}u;
const RUN_CAPACITY: u32 = ${props.runCapacity}u;`,
    body: /* wgsl */ `
  var segmentStart = ROW_COUNT;
  var segmentEnd = ROW_COUNT;
  if (index < runCountIn[runCountInOffset]) {
    segmentStart = runStarts[runStartsOffset + index] - 1u;
    segmentEnd = runEnds[runEndsOffset + index] + 1u;
  }
  centroidOffsets[centroidOffsetsOffset + 2u * index] = segmentStart;
  centroidOffsets[centroidOffsetsOffset + 2u * index + 1u] = segmentEnd;
  if (index == 0u) {
    centroidOffsets[centroidOffsetsOffset + 2u * RUN_CAPACITY] = ROW_COUNT;
  }`
  });
}

/** Builds the kernel that splits interleaved positions into separate x and y columns. @internal */
export function createTrajectorySplitPositionsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    xs: GraphDataView<'float32'>;
    ys: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'split-positions',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'xs', view: props.xs, type: 'f32', access: 'read_write'},
      {name: 'ys', view: props.ys, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.positions.length,
    body: /* wgsl */ `
  xs[xsOffset + index] = positions[positionsOffset + 2u * index];
  ys[ysOffset + index] = positions[positionsOffset + 2u * index + 1u];`
  });
}

/** Properties for {@link createTrajectoryStopGatherNodes}. @internal */
export type TrajectoryStopGatherProps = TrajectoryTimeViews & {
  id: string;
  capacity: number;
  stopTotal: GraphDataView<'uint32'>;
  qualifiedRuns: GraphDataView<'uint32'>;
  runStarts: GraphDataView<'uint32'>;
  runEnds: GraphDataView<'uint32'>;
  runTracks: GraphDataView<'uint32'>;
  /** Required when `centroids` is given. */
  sumsX?: GraphDataView<'float32'>;
  /** Required when `centroids` is given. */
  sumsY?: GraphDataView<'float32'>;
  ids: GraphDataView<'uint32'>;
  startRows?: GraphDataView<'uint32'>;
  endRows?: GraphDataView<'uint32'>;
  centroids?: GraphDataView<'float32x2'>;
  durations?: GraphDataView<'float32'>;
};

/**
 * Builds the kernels that gather per-stop outputs for the bounded prefix and write sentinels
 * (`0xffffffff` rows and tracks, zero durations and centroids) after it. The work is split across
 * up to three kernels to stay within eight storage bindings each.
 *
 * @internal
 */
export function createTrajectoryStopGatherNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TrajectoryStopGatherProps
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const common: WGSLKernelBinding[] = [
    {name: 'stopTotalIn', view: props.stopTotal, type: 'u32', access: 'read'},
    {name: 'qualifiedRuns', view: props.qualifiedRuns, type: 'u32', access: 'read'},
    {name: 'runStarts', view: props.runStarts, type: 'u32', access: 'read'},
    {name: 'runEnds', view: props.runEnds, type: 'u32', access: 'read'}
  ];
  const declarations = `const CAPACITY: u32 = ${props.capacity}u;`;
  const prelude = /* wgsl */ `
  let count = min(stopTotalIn[stopTotalInOffset], CAPACITY);
  let isStop = index < count;
  var run = 0u;
  if (isStop) {
    run = qualifiedRuns[qualifiedRunsOffset + index];
  }`;

  // WebGPU drops unused bindings from the auto layout, so only bind what the body reads.
  const rowBindings: WGSLKernelBinding[] = [
    {name: 'stopTotalIn', view: props.stopTotal, type: 'u32', access: 'read'},
    {name: 'qualifiedRuns', view: props.qualifiedRuns, type: 'u32', access: 'read'},
    {name: 'runTracks', view: props.runTracks, type: 'u32', access: 'read'},
    {name: 'idsOut', view: props.ids, type: 'u32', access: 'read_write'}
  ];
  if (props.startRows) {
    rowBindings.push({name: 'runStarts', view: props.runStarts, type: 'u32', access: 'read'});
  }
  if (props.endRows) {
    rowBindings.push({name: 'runEnds', view: props.runEnds, type: 'u32', access: 'read'});
  }
  if (props.startRows) {
    rowBindings.push({
      name: 'startRowsOut',
      view: props.startRows,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.endRows) {
    rowBindings.push({name: 'endRowsOut', view: props.endRows, type: 'u32', access: 'read_write'});
  }
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: props.id,
      operation: OPERATION,
      variant: 'stop-gather',
      bindings: rowBindings,
      invocationCount: props.capacity,
      declarations,
      body: `${prelude}
  idsOut[idsOutOffset + index] = select(0xffffffffu, runTracks[runTracksOffset + run], isStop);
  ${props.startRows ? 'startRowsOut[startRowsOutOffset + index] = select(0xffffffffu, runStarts[runStartsOffset + run] - 1u, isStop);' : ''}
  ${props.endRows ? 'endRowsOut[endRowsOutOffset + index] = select(0xffffffffu, runEnds[runEndsOffset + run], isStop);' : ''}`
    })
  );

  if (props.durations) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-durations`,
        operation: OPERATION,
        variant: 'stop-durations',
        bindings: [
          ...common,
          ...getTrajectoryTimeBindings(props),
          {name: 'durationsOut', view: props.durations, type: 'f32', access: 'read_write'}
        ],
        invocationCount: props.capacity,
        declarations: `${declarations}
${getTrajectoryTimeSource(getTrajectoryTimeMode(props))}`,
        body: `${prelude}
  var stopDuration = 0.0;
  if (isStop) {
    stopDuration = timeDifferenceRows(runEnds[runEndsOffset + run], runStarts[runStartsOffset + run] - 1u);
  }
  durationsOut[durationsOutOffset + index] = stopDuration;`
      })
    );
  }

  if (props.centroids && props.sumsX && props.sumsY) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-centroids`,
        operation: OPERATION,
        variant: 'stop-centroids',
        bindings: [
          ...common,
          {name: 'sumsX', view: props.sumsX, type: 'f32', access: 'read'},
          {name: 'sumsY', view: props.sumsY, type: 'f32', access: 'read'},
          {name: 'centroidsOut', view: props.centroids, type: 'f32', access: 'read_write'}
        ],
        invocationCount: props.capacity,
        declarations,
        body: `${prelude}
  var centroid = vec2<f32>(0.0, 0.0);
  if (isStop) {
    let rowCount = f32(runEnds[runEndsOffset + run] - runStarts[runStartsOffset + run] + 2u);
    centroid = vec2<f32>(sumsX[sumsXOffset + 2u * run], sumsY[sumsYOffset + 2u * run]) / rowCount;
  }
  centroidsOut[centroidsOutOffset + 2u * index] = centroid.x;
  centroidsOut[centroidsOutOffset + 2u * index + 1u] = centroid.y;`
      })
    );
  }
  return nodes;
}

/** Properties for {@link getSegmentedReductionNodes}. @internal */
export type SegmentedReductionNodesProps = {
  id: string;
  input: GraphDataView<'float32'>;
  segmentOffsets: GraphDataView<'uint32'>;
  output: GraphDataView<'float32'>;
  operation: GPUSegmentedReductionOperation;
};

/**
 * Builds `GPUSegmentedReduction` nodes, splitting the segments into chunks of at most
 * `maximumSegmentCount` because the primitive dispatches one workgroup per segment without 3D
 * folding. Chunks are sub-views of the same offset and output buffers.
 *
 * @param maximumSegmentCount Normally `maxComputeWorkgroupsPerDimension`.
 * @internal
 */
export function getSegmentedReductionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SegmentedReductionNodesProps,
  maximumSegmentCount: number
): readonly GPUCommandNode<Parameters>[] {
  const segmentCount = props.output.length;
  if (segmentCount <= maximumSegmentCount) {
    return new GPUSegmentedReduction({...props}).getCommandNodes(graph);
  }
  const nodes: GPUCommandNode<Parameters>[] = [];
  for (
    let first = 0, chunkIndex = 0;
    first < segmentCount;
    first += maximumSegmentCount, chunkIndex++
  ) {
    const length = Math.min(maximumSegmentCount, segmentCount - first);
    const offsets = graph.createDataView(props.segmentOffsets.buffer, {
      format: 'uint32',
      length: length + 1,
      byteOffset: props.segmentOffsets.byteOffset + first * 4
    });
    const output = graph.createDataView(props.output.buffer, {
      format: 'float32',
      length,
      byteOffset: props.output.byteOffset + first * 4
    });
    nodes.push(
      ...new GPUSegmentedReduction({
        id: `${props.id}-chunk-${chunkIndex}`,
        input: props.input,
        segmentOffsets: offsets,
        output,
        operation: props.operation
      }).getCommandNodes(graph)
    );
  }
  return nodes;
}
