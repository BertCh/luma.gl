// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {TIME_WORDS_WGSL} from '../../gpu-dataframe/time-window-filter/time-words';
import {GPU_TRAJECTORY_PLAYHEAD_STATUS} from './trajectory-playhead-parameters';

const PLAYHEAD_OPERATION = 'GPUTrajectoryPlayhead';
const RESAMPLE_OPERATION = 'GPUTrajectoryResample';

/**
 * Words per track in the internal bracket column:
 * `[segmentRow, nextRow, status, fractionBits, durationBits]`.
 *
 * @internal
 */
export const TRAJECTORY_BRACKET_STRIDE = 5;

/** Sample time inputs: f32 relative times or Int64 `(low, high)` words. @internal */
export type TrajectoryInterpolationTimestamps =
  | GraphDataView<'float32'>
  | GraphDataView<'uint32x2'>;

const STATUS_CONSTANTS = /* wgsl */ `
const NO_ROW: u32 = 0xffffffffu;
const STATUS_EMPTY: u32 = ${GPU_TRAJECTORY_PLAYHEAD_STATUS.empty}u;
const STATUS_ACTIVE: u32 = ${GPU_TRAJECTORY_PLAYHEAD_STATUS.active}u;
const STATUS_BEFORE_START: u32 = ${GPU_TRAJECTORY_PLAYHEAD_STATUS.beforeStart}u;
const STATUS_AFTER_END: u32 = ${GPU_TRAJECTORY_PLAYHEAD_STATUS.afterEnd}u;
const STATUS_GAP: u32 = ${GPU_TRAJECTORY_PLAYHEAD_STATUS.gap}u;
`;

/** Returns the `timestamps` storage binding. @internal */
function getTimestampsBinding(timestamps: TrajectoryInterpolationTimestamps): WGSLKernelBinding {
  return {
    name: 'timestamps',
    view: timestamps,
    type: timestamps.format === 'uint32x2' ? 'u32' : 'f32',
    access: 'read'
  };
}

/**
 * WGSL row-time helpers. `getRowTimeDifference(a, b)` is `t[a] - t[b]` in f32; word times subtract
 * exactly with a borrow and only then round to f32.
 */
function getRowTimeSource(isWordMode: boolean): string {
  return isWordMode
    ? /* wgsl */ `${TIME_WORDS_WGSL}
fn getTimeWords(row: u32) -> vec2<u32> {
  return vec2<u32>(timestamps[timestampsOffset + 2u * row], timestamps[timestampsOffset + 2u * row + 1u]);
}

fn getRowTimeDifference(a: u32, b: u32) -> f32 {
  return timeWordsToF32(timeWordsSubtract(getTimeWords(a), getTimeWords(b)));
}
`
    : /* wgsl */ `
fn getRowTimeDifference(a: u32, b: u32) -> f32 {
  return timestamps[timestampsOffset + a] - timestamps[timestampsOffset + b];
}
`;
}

/**
 * WGSL playhead helpers over the `parameters` binding. Comparisons are exact in both modes:
 * f32 compares f32 values, words compare integer words first and the f32 fraction only on ties.
 */
function getPlayheadSource(isWordMode: boolean): string {
  return isWordMode
    ? /* wgsl */ `
fn getPlayheadWords() -> vec2<u32> {
  return vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
}

fn getPlayheadFraction() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 2u]);
}

fn getPlayheadMaxGap() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 3u]);
}

fn isRowAfterPlayhead(row: u32) -> bool {
  return !isTimeWordsAtLeast(getPlayheadWords(), getPlayheadFraction(), getTimeWords(row), 0.0);
}

fn isRowBeforePlayhead(row: u32) -> bool {
  return !isTimeWordsAtLeast(getTimeWords(row), 0.0, getPlayheadWords(), getPlayheadFraction());
}

fn getPlayheadElapsed(row: u32) -> f32 {
  return timeWordsDifference(getPlayheadWords(), getPlayheadFraction(), getTimeWords(row), 0.0);
}
`
    : /* wgsl */ `
fn getPlayheadMaxGap() -> f32 {
  return parameters[parametersOffset + 1u];
}

fn isRowAfterPlayhead(row: u32) -> bool {
  return timestamps[timestampsOffset + row] > parameters[parametersOffset];
}

fn isRowBeforePlayhead(row: u32) -> bool {
  return timestamps[timestampsOffset + row] < parameters[parametersOffset];
}

fn getPlayheadElapsed(row: u32) -> f32 {
  return parameters[parametersOffset] - timestamps[timestampsOffset + row];
}
`;
}

/**
 * WGSL common-clock helpers over the `clock` binding. `getClockOffset(firstRow)` is the clock start
 * minus the track's first sample time (exact for words before the f32 round); `getClockStep()` is
 * the f32 step between consecutive samples.
 */
function getClockSource(isWordMode: boolean): string {
  return isWordMode
    ? /* wgsl */ `
fn getClockOffset(firstRow: u32) -> f32 {
  let start = vec2<u32>(clockParameters[clockParametersOffset], clockParameters[clockParametersOffset + 1u]);
  return timeWordsDifference(start, bitcast<f32>(clockParameters[clockParametersOffset + 2u]), getTimeWords(firstRow), 0.0);
}

fn getClockStep() -> f32 {
  return bitcast<f32>(clockParameters[clockParametersOffset + 3u]);
}
`
    : /* wgsl */ `
fn getClockOffset(firstRow: u32) -> f32 {
  return clockParameters[clockParametersOffset] - timestamps[timestampsOffset + firstRow];
}

fn getClockStep() -> f32 {
  return clockParameters[clockParametersOffset + 1u];
}
`;
}

/** WGSL that clamps the row range of track `index` to `[0, ROW_COUNT]`. */
const TRACK_RANGE_SOURCE = /* wgsl */ `
  let trackStart = min(trackOffsets[trackOffsetsOffset + track], ROW_COUNT);
  let trackEnd = clamp(trackOffsets[trackOffsetsOffset + track + 1u], trackStart, ROW_COUNT);`;

/** Properties for {@link createTrajectoryBracketNode}. @internal */
export type TrajectoryBracketProps = {
  id: string;
  timestamps: TrajectoryInterpolationTimestamps;
  trackOffsets: GraphDataView<'uint32'>;
  parameters: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** `TRAJECTORY_BRACKET_STRIDE * trackCount` uint32 words. */
  bracket: GraphDataView<'uint32'>;
  rowCount: number;
  trackCount: number;
};

/**
 * Builds the per-track playhead search: classify the playhead against the track range and find
 * the bracketing segment with an upper-bound binary search (first row whose time is greater than
 * the playhead). Writes every bracket word on every encoding.
 *
 * @internal
 */
export function createTrajectoryBracketNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TrajectoryBracketProps
): GPUCommandNode<Parameters> {
  const isWordMode = props.timestamps.format === 'uint32x2';
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: PLAYHEAD_OPERATION,
    variant: isWordMode ? 'bracket-words' : 'bracket',
    bindings: [
      getTimestampsBinding(props.timestamps),
      {
        name: 'trackOffsets',
        view: props.trackOffsets,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'parameters',
        view: props.parameters,
        type: isWordMode ? 'u32' : 'f32',
        access: 'read'
      },
      {
        name: 'bracket',
        view: props.bracket,
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: props.trackCount,
    declarations: `const ROW_COUNT: u32 = ${props.rowCount}u;
const BRACKET_STRIDE: u32 = ${TRAJECTORY_BRACKET_STRIDE}u;
${STATUS_CONSTANTS}
${getRowTimeSource(isWordMode)}
${getPlayheadSource(isWordMode)}`,
    body: /* wgsl */ `let track = index;
  ${TRACK_RANGE_SOURCE}
  var status = STATUS_EMPTY;
  var segmentRow = NO_ROW;
  var nextRow = NO_ROW;
  var fraction = 0.0;
  var duration = 0.0;
  if (trackEnd > trackStart) {
    let lastRow = trackEnd - 1u;
    let lastSegmentRow = select(trackStart, lastRow - 1u, lastRow > trackStart);
    if (isRowAfterPlayhead(trackStart)) {
      status = STATUS_BEFORE_START;
      segmentRow = trackStart;
      nextRow = min(trackStart + 1u, lastRow);
    } else if (isRowBeforePlayhead(lastRow)) {
      status = STATUS_AFTER_END;
      segmentRow = lastSegmentRow;
      nextRow = lastRow;
      fraction = 1.0;
    } else {
      // t[trackStart] <= playhead <= t[lastRow]: first row in (trackStart, trackEnd) after it.
      var low = trackStart + 1u;
      var high = trackEnd;
      loop {
        if (low >= high) {
          break;
        }
        let middle = (low + high) / 2u;
        if (isRowAfterPlayhead(middle)) {
          high = middle;
        } else {
          low = middle + 1u;
        }
      }
      status = STATUS_ACTIVE;
      if (low == trackEnd) {
        // The playhead equals the last sample time.
        segmentRow = lastSegmentRow;
        nextRow = lastRow;
        fraction = 1.0;
      } else {
        // t[low - 1] <= playhead < t[low], so the interval is strictly positive.
        segmentRow = low - 1u;
        nextRow = low;
        let interval = getRowTimeDifference(nextRow, segmentRow);
        fraction = clamp(getPlayheadElapsed(segmentRow) / interval, 0.0, 1.0);
        let maxGap = getPlayheadMaxGap();
        if (maxGap > 0.0 && interval > maxGap && isRowBeforePlayhead(segmentRow)) {
          status = STATUS_GAP;
          fraction = 0.0;
        }
      }
    }
    duration = getRowTimeDifference(nextRow, segmentRow);
  }
  let base = bracketOffset + BRACKET_STRIDE * track;
  bracket[base] = segmentRow;
  bracket[base + 1u] = nextRow;
  bracket[base + 2u] = status;
  bracket[base + 3u] = bitcast<u32>(fraction);
  bracket[base + 4u] = bitcast<u32>(duration);`
  });
}

/** Properties for {@link createTrajectoryGeometryNode}. @internal */
export type TrajectoryGeometryProps = {
  id: string;
  bracket: GraphDataView<'uint32'>;
  positions: GraphDataView<'float32x2'>;
  elevations?: GraphDataView<'float32'>;
  currentPositions?: GraphDataView<'float32x2'>;
  currentElevations?: GraphDataView<'float32'>;
  headings?: GraphDataView<'float32'>;
  speeds?: GraphDataView<'float32'>;
  trackCount: number;
};

/**
 * Builds the per-track geometry kernel: interpolated position (and elevation), heading of the
 * bracketing segment, and speed for active tracks.
 *
 * @internal
 */
export function createTrajectoryGeometryNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TrajectoryGeometryProps
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'bracket', view: props.bracket, type: 'u32', access: 'read'},
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'}
  ];
  const optional = (
    name: string,
    view: GraphDataView | undefined,
    access: 'read' | 'read_write'
  ) => {
    if (view) {
      bindings.push({name, view, type: 'f32', access});
    }
  };
  const readsElevations = Boolean(props.elevations && props.currentElevations);
  optional('elevations', readsElevations ? props.elevations : undefined, 'read');
  optional('currentPositions', props.currentPositions, 'read_write');
  optional(
    'currentElevations',
    readsElevations ? props.currentElevations : undefined,
    'read_write'
  );
  optional('headings', props.headings, 'read_write');
  optional('speeds', props.speeds, 'read_write');
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: PLAYHEAD_OPERATION,
    variant: 'geometry',
    bindings,
    invocationCount: props.trackCount,
    declarations: `const BRACKET_STRIDE: u32 = ${TRAJECTORY_BRACKET_STRIDE}u;
${STATUS_CONSTANTS}
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,
    body: /* wgsl */ `let base = bracketOffset + BRACKET_STRIDE * index;
  let segmentRow = bracket[base];
  let nextRow = bracket[base + 1u];
  let status = bracket[base + 2u];
  let fraction = bitcast<f32>(bracket[base + 3u]);
  let duration = bitcast<f32>(bracket[base + 4u]);
  var position = vec2<f32>(0.0, 0.0);
  var elevation = 0.0;
  var heading = 0.0;
  var speed = 0.0;
  if (status != STATUS_EMPTY) {
    let startPosition = getPosition(segmentRow);
    let endPosition = getPosition(nextRow);
    let delta = endPosition - startPosition;
    position = select(startPosition + delta * fraction, endPosition, fraction >= 1.0);
    ${
      readsElevations
        ? `let startElevation = elevations[elevationsOffset + segmentRow];
    let endElevation = elevations[elevationsOffset + nextRow];
    elevation = select(
      startElevation + (endElevation - startElevation) * fraction,
      endElevation,
      fraction >= 1.0
    );`
        : ''
    }
    if (delta.x != 0.0 || delta.y != 0.0) {
      heading = atan2(delta.y, delta.x);
    }
    if (status == STATUS_ACTIVE && duration > 0.0) {
      speed = length(delta) / duration;
    }
  }
  ${props.currentPositions ? 'currentPositions[currentPositionsOffset + 2u * index] = position.x;\n  currentPositions[currentPositionsOffset + 2u * index + 1u] = position.y;' : ''}
  ${readsElevations ? 'currentElevations[currentElevationsOffset + index] = elevation;' : ''}
  ${props.headings ? 'headings[headingsOffset + index] = heading;' : ''}
  ${props.speeds ? 'speeds[speedsOffset + index] = speed;' : ''}
  _ = elevation;`
  });
}

/** Properties for {@link createTrajectoryColumnsNode}. @internal */
export type TrajectoryColumnsProps = {
  id: string;
  bracket: GraphDataView<'uint32'>;
  status?: GraphDataView<'uint32'>;
  segmentRows?: GraphDataView<'uint32'>;
  segmentFractions?: GraphDataView<'float32'>;
  /** Internal 0/1 flags of active tracks for compaction. */
  activeFlags?: GraphDataView<'uint32'>;
  /** Internal track indices `0..trackCount-1` for compaction. */
  trackIndices?: GraphDataView<'uint32'>;
  trackCount: number;
};

/** Builds the per-track bookkeeping kernel (status, segment row and fraction, active flags). @internal */
export function createTrajectoryColumnsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TrajectoryColumnsProps
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'bracket', view: props.bracket, type: 'u32', access: 'read'}
  ];
  const statements: string[] = [];
  for (const [name, view, type, value] of [
    ['status', props.status, 'u32', 'trackStatus'],
    ['segmentRows', props.segmentRows, 'u32', 'trackSegmentRow'],
    ['segmentFractions', props.segmentFractions, 'f32', 'trackFraction'],
    ['activeFlags', props.activeFlags, 'u32', 'select(0u, 1u, trackStatus == STATUS_ACTIVE)'],
    ['trackIndices', props.trackIndices, 'u32', 'index']
  ] as const) {
    if (view) {
      bindings.push({name, view, type, access: 'read_write'});
      statements.push(`${name}[${name}Offset + index] = ${value};`);
    }
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: PLAYHEAD_OPERATION,
    variant: 'columns',
    bindings,
    invocationCount: props.trackCount,
    declarations: `const BRACKET_STRIDE: u32 = ${TRAJECTORY_BRACKET_STRIDE}u;
${STATUS_CONSTANTS}`,
    body: /* wgsl */ `let base = bracketOffset + BRACKET_STRIDE * index;
  let trackSegmentRow = bracket[base];
  let trackStatus = bracket[base + 2u];
  let trackFraction = bitcast<f32>(bracket[base + 3u]);
  _ = trackSegmentRow;
  _ = trackFraction;
  _ = trackStatus;
  ${statements.join('\n  ')}`
  });
}

/** Properties for {@link createTrajectoryArcLengthNode}. @internal */
export type TrajectoryArcLengthProps = {
  id: string;
  positions: GraphDataView<'float32x2'>;
  trackOffsets: GraphDataView<'uint32'>;
  /** One f32 per row; rows outside every track are left untouched. */
  cumulativeLengths: GraphDataView<'float32'>;
  rowCount: number;
  trackCount: number;
};

/**
 * Builds the per-track cumulative planar path length, summed sequentially in row order so the
 * result is deterministic. One invocation per track.
 *
 * @internal
 */
export function createTrajectoryArcLengthNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TrajectoryArcLengthProps
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: RESAMPLE_OPERATION,
    variant: 'arc-length',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {
        name: 'trackOffsets',
        view: props.trackOffsets,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'cumulativeLengths',
        view: props.cumulativeLengths,
        type: 'f32',
        access: 'read_write'
      }
    ],
    invocationCount: props.trackCount,
    declarations: `const ROW_COUNT: u32 = ${props.rowCount}u;`,
    body: /* wgsl */ `let track = index;
  ${TRACK_RANGE_SOURCE}
  if (trackEnd > trackStart) {
    var total = 0.0;
    cumulativeLengths[cumulativeLengthsOffset + trackStart] = 0.0;
    for (var row = trackStart + 1u; row < trackEnd; row++) {
      let deltaX = positions[positionsOffset + 2u * row] - positions[positionsOffset + 2u * (row - 1u)];
      let deltaY = positions[positionsOffset + 2u * row + 1u] - positions[positionsOffset + 2u * (row - 1u) + 1u];
      total = total + sqrt(deltaX * deltaX + deltaY * deltaY);
      cumulativeLengths[cumulativeLengthsOffset + row] = total;
    }
  }`
  });
}

/** Properties for {@link createTrajectoryResampleNode}. @internal */
export type TrajectoryResampleNodeProps = {
  id: string;
  positions: GraphDataView<'float32x2'>;
  elevations?: GraphDataView<'float32'>;
  timestamps?: TrajectoryInterpolationTimestamps;
  trackOffsets: GraphDataView<'uint32'>;
  /** Present in arc-length mode. */
  cumulativeLengths?: GraphDataView<'float32'>;
  samples: GraphDataView<'float32x2'>;
  sampleElevations?: GraphDataView<'float32'>;
  sampleTimes?: GraphDataView<'float32'>;
  /**
   * Per-frame common clock (`'clock'` spacing): float32 `[start, step, 0, 0]`, or uint32 words
   * `[startLow, startHigh, startFraction, stepBits]` for word timestamps.
   */
  clock?: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  rowCount: number;
  trackCount: number;
  sampleCount: number;
};

/**
 * Builds the per-sample resampling kernel. Sample `k` of a track targets progress
 * `total * k / (sampleCount - 1)` (time since the first sample, or arc length) and interpolates
 * the segment found by the same upper-bound search as the playhead.
 *
 * @internal
 */
export function createTrajectoryResampleNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TrajectoryResampleNodeProps
): GPUCommandNode<Parameters> {
  const isArcLength = Boolean(props.cumulativeLengths);
  const isClock = Boolean(props.clock);
  const timestamps = props.timestamps;
  const isWordMode = timestamps?.format === 'uint32x2';
  const readsElevations = Boolean(props.elevations && props.sampleElevations);
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {
      name: 'trackOffsets',
      view: props.trackOffsets,
      type: 'u32',
      access: 'read'
    }
  ];
  if (timestamps) {
    bindings.push(getTimestampsBinding(timestamps));
  }
  if (props.clock) {
    bindings.push({
      name: 'clockParameters',
      view: props.clock,
      type: isWordMode ? 'u32' : 'f32',
      access: 'read'
    });
  }
  if (props.cumulativeLengths) {
    bindings.push({
      name: 'cumulativeLengths',
      view: props.cumulativeLengths,
      type: 'f32',
      access: 'read'
    });
  }
  if (readsElevations && props.elevations && props.sampleElevations) {
    bindings.push(
      {
        name: 'elevations',
        view: props.elevations,
        type: 'f32',
        access: 'read'
      },
      {
        name: 'sampleElevations',
        view: props.sampleElevations,
        type: 'f32',
        access: 'read_write'
      }
    );
  }
  bindings.push({
    name: 'samples',
    view: props.samples,
    type: 'f32',
    access: 'read_write'
  });
  if (props.sampleTimes) {
    bindings.push({
      name: 'sampleTimes',
      view: props.sampleTimes,
      type: 'f32',
      access: 'read_write'
    });
  }
  const progressSource = isArcLength
    ? `fn getProgress(row: u32, firstRow: u32) -> f32 {
  _ = firstRow;
  return cumulativeLengths[cumulativeLengthsOffset + row];
}`
    : `fn getProgress(row: u32, firstRow: u32) -> f32 {
  return getRowTimeDifference(row, firstRow);
}`;
  // Time at the sample: the targetProgress itself in time mode, interpolated row times in arc mode.
  const timeSource = !props.sampleTimes
    ? ''
    : isClock
      ? 'time = f32(sampleIndex) * clockStep;'
      : isArcLength
        ? `let startTime = getRowTimeDifference(row0, trackStart);
    let endTime = getRowTimeDifference(row1, trackStart);
    time = select(startTime + (endTime - startTime) * fraction, endTime, fraction >= 1.0);`
        : 'time = targetProgress;';
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: RESAMPLE_OPERATION,
    variant: isClock ? (isWordMode ? 'clock-words' : 'clock') : isArcLength ? 'arc-length' : 'time',
    bindings,
    invocationCount: props.trackCount * props.sampleCount,
    declarations: `const ROW_COUNT: u32 = ${props.rowCount}u;
const SAMPLE_COUNT: u32 = ${props.sampleCount}u;
${timestamps ? getRowTimeSource(isWordMode) : ''}
${isClock ? getClockSource(isWordMode) : ''}
${progressSource}
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,
    body: /* wgsl */ `let track = index / SAMPLE_COUNT;
  let sampleIndex = index - track * SAMPLE_COUNT;
  ${TRACK_RANGE_SOURCE}
  var position = vec2<f32>(0.0, 0.0);
  var elevation = 0.0;
  var time = 0.0;
  ${isClock ? 'let clockStep = getClockStep();\n  var isAbsent = trackEnd <= trackStart;' : ''}
  if (trackEnd > trackStart) {
    let lastRow = trackEnd - 1u;
    let total = getProgress(lastRow, trackStart);
    var targetProgress = 0.0;
    ${
      isClock
        ? `targetProgress = getClockOffset(trackStart) + f32(sampleIndex) * clockStep;
    isAbsent = !(targetProgress >= 0.0 && targetProgress <= total);`
        : ''
    }
    if (${isClock ? 'false' : 'SAMPLE_COUNT > 1u'}) {
      targetProgress = select(
        total * (f32(sampleIndex) / f32(SAMPLE_COUNT - 1u)),
        total,
        sampleIndex == SAMPLE_COUNT - 1u
      );
    }
    var low = trackStart + 1u;
    var high = trackEnd;
    loop {
      if (low >= high) {
        break;
      }
      let middle = (low + high) / 2u;
      if (getProgress(middle, trackStart) > targetProgress) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    var row0 = lastRow;
    var row1 = lastRow;
    var fraction = 1.0;
    if (low < trackEnd) {
      row0 = low - 1u;
      row1 = low;
      let startProgress = getProgress(row0, trackStart);
      fraction = clamp(
        (targetProgress - startProgress) / (getProgress(row1, trackStart) - startProgress),
        0.0,
        1.0
      );
    }
    let startPosition = getPosition(row0);
    let endPosition = getPosition(row1);
    position = select(startPosition + (endPosition - startPosition) * fraction, endPosition, fraction >= 1.0);
    ${
      readsElevations
        ? `let startElevation = elevations[elevationsOffset + row0];
    let endElevation = elevations[elevationsOffset + row1];
    elevation = select(
      startElevation + (endElevation - startElevation) * fraction,
      endElevation,
      fraction >= 1.0
    );`
        : ''
    }
    ${timeSource}
  }
  ${
    isClock
      ? `if (isAbsent) {
    // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
    position = vec2<f32>(bitcast<f32>(0x7fc00000u | (index & 0u)));
    elevation = position.x;
  }`
      : ''
  }
  samples[samplesOffset + 2u * index] = position.x;
  samples[samplesOffset + 2u * index + 1u] = position.y;
  ${readsElevations ? 'sampleElevations[sampleElevationsOffset + index] = elevation;' : ''}
  ${props.sampleTimes ? 'sampleTimes[sampleTimesOffset + index] = time;' : ''}
  _ = elevation;
  _ = time;`
  });
}
