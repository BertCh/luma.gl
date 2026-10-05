// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import {getGraphViewChunks} from '../map-graph-utils';
import {GPU_TIME_WINDOW_PARAMETER_LENGTH} from './time-window-parameters';
import {
  GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH,
  TIME_WORDS_WGSL,
  type GPUInt64TimeWordRows
} from './time-words';

type Float32Rows = GraphDataView<'float32'> | GraphVectorView<'float32'>;

/** Properties for {@link getTimeWindowClassifyNodes}. @internal */
export type TimeWindowClassifyProps = {
  id: string;
  /** Float32 rows, or exact Int64 word rows (`uint32x2`). */
  timestamps: Float32Rows | GPUInt64TimeWordRows;
  /** Float32 mode only. */
  timestampsLow?: Float32Rows;
  /** Same format as `timestamps`. */
  endTimestamps?: Float32Rows | GPUInt64TimeWordRows;
  /** Float32 mode only. */
  endTimestampsLow?: Float32Rows;
  /** Float32 window in float mode, packed `uint32` word window in word mode. */
  window: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  mask: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  fadeWeights?: Float32Rows;
  clipFractions?: GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>;
};

/**
 * Builds one classify node per nonempty timestamp chunk.
 *
 * Every per-row view shares the topology of `timestamps`. A kernel binds at most 8 storage
 * buffers, which is the WebGPU default `maxStorageBuffersPerShaderStage`.
 *
 * @internal
 */
export function getTimeWindowClassifyNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TimeWindowClassifyProps
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const isWordMode = props.timestamps.format === 'uint32x2';
  validateTimeWindowClassifyMode(props, isWordMode);
  const isVector = props.timestamps instanceof GraphVectorView;
  const chunks = getGraphViewChunks(props.timestamps);
  const getChunk = <View extends GraphDataView | GraphVectorView>(
    view: View | undefined,
    chunkIndex: number
  ): GraphDataView | undefined =>
    view ? getGraphViewChunks(view as GraphDataView | GraphVectorView)[chunkIndex] : undefined;

  for (const [chunkIndex, chunk] of chunks.entries()) {
    if (chunk.length === 0) {
      continue;
    }
    const startLow = getChunk(props.timestampsLow, chunkIndex);
    const endHigh = getChunk(props.endTimestamps, chunkIndex);
    const endLow = getChunk(props.endTimestampsLow, chunkIndex);
    const weights = getChunk(props.fadeWeights, chunkIndex);
    const clip = getChunk(props.clipFractions, chunkIndex);
    if (isWordMode) {
      nodes.push(createWordClassifyNode(graph, props, chunkIndex, isVector, chunk, getChunk));
      continue;
    }
    const bindings: MapGraphKernelBinding[] = [
      {name: 'params', view: props.window, type: 'f32', access: 'read'},
      {name: 'startHigh', view: chunk, type: 'f32', access: 'read'}
    ];
    if (startLow)
      bindings.push({
        name: 'startLow',
        view: startLow,
        type: 'f32',
        access: 'read'
      });
    if (endHigh)
      bindings.push({
        name: 'endHigh',
        view: endHigh,
        type: 'f32',
        access: 'read'
      });
    if (endLow)
      bindings.push({
        name: 'endLow',
        view: endLow,
        type: 'f32',
        access: 'read'
      });
    bindings.push({
      name: 'outputMask',
      view: getChunk(props.mask, chunkIndex) as GraphDataView,
      type: 'u32',
      access: 'read_write'
    });
    if (weights) {
      bindings.push({
        name: 'outputWeights',
        view: weights,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (clip)
      bindings.push({
        name: 'outputClip',
        view: clip,
        type: 'f32',
        access: 'read_write'
      });

    const rowEndLow = endLow ? 'endLow[endLowOffset + index]' : endHigh ? '0.0' : 'rowStartLow';
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: isVector ? `${props.id}-chunk-${chunkIndex}` : props.id,
        operation: 'GPUTimeWindowFilter',
        variant: 'classify',
        bindings,
        invocationCount: chunk.length,
        declarations: /* wgsl */ `
// Double-single difference (aHigh + aLow) - (bHigh + bLow). High parts are subtracted first so
// nearby large epoch values cancel exactly before the small low parts are added. The result is
// used for fade and clip ratios, where rounding is harmless.
fn timeDifference(aHigh: f32, aLow: f32, bHigh: f32, bLow: f32) -> f32 {
  return (aHigh - bHigh) + (aLow - bLow);
}

// Exact double-single a >= b. Backends with fast-math may reassociate the sum above and drop the
// low parts, so acceptance compares the high parts first and uses the low parts only on ties.
fn isTimeAtLeast(aHigh: f32, aLow: f32, bHigh: f32, bLow: f32) -> bool {
  return aHigh > bHigh || (aHigh == bHigh && aLow >= bLow);
}`,
        body: /* wgsl */ `
  let windowStartHigh = params[paramsOffset + 0u];
  let windowStartLow = params[paramsOffset + 1u];
  let windowEndHigh = params[paramsOffset + 2u];
  let windowEndLow = params[paramsOffset + 3u];
  let startFade = params[paramsOffset + 4u];
  let endFade = params[paramsOffset + 5u];

  let rowStartHigh = startHigh[startHighOffset + index];
  let rowStartLow = ${startLow ? 'startLow[startLowOffset + index]' : '0.0'};
  let rowEndHigh = ${endHigh ? 'endHigh[endHighOffset + index]' : 'rowStartHigh'};
  let rowEndLow = ${rowEndLow};

  // Ordered comparisons are false for NaN, so NaN rows are rejected.
  let endAfterWindowStart = timeDifference(rowEndHigh, rowEndLow, windowStartHigh, windowStartLow);
  let startBeforeWindowEnd = timeDifference(windowEndHigh, windowEndLow, rowStartHigh, rowStartLow);
  let accepted =
    isTimeAtLeast(rowEndHigh, rowEndLow, windowStartHigh, windowStartLow) &&
    isTimeAtLeast(windowEndHigh, windowEndLow, rowStartHigh, rowStartLow);
  outputMask[outputMaskOffset + index] = select(0u, 1u, accepted);
  ${
    weights
      ? `var startRamp = 1.0;
  if (startFade > 0.0) { startRamp = clamp(endAfterWindowStart / startFade, 0.0, 1.0); }
  var endRamp = 1.0;
  if (endFade > 0.0) { endRamp = clamp(startBeforeWindowEnd / endFade, 0.0, 1.0); }
  outputWeights[outputWeightsOffset + index] = select(0.0, min(startRamp, endRamp), accepted);`
      : ''
  }
  ${
    clip
      ? `let duration = timeDifference(rowEndHigh, rowEndLow, rowStartHigh, rowStartLow);
  var clipStart = 0.0;
  var clipEnd = 1.0;
  if (duration > 0.0) {
    clipStart = clamp(timeDifference(windowStartHigh, windowStartLow, rowStartHigh, rowStartLow) / duration, 0.0, 1.0);
    clipEnd = clamp(startBeforeWindowEnd / duration, 0.0, 1.0);
  }
  outputClip[outputClipOffset + 2u * index] = select(0.0, clipStart, accepted);
  outputClip[outputClipOffset + 2u * index + 1u] = select(0.0, clipEnd, accepted);`
      : ''
  }`
      })
    );
  }
  return nodes;
}

function validateTimeWindowClassifyMode(props: TimeWindowClassifyProps, isWordMode: boolean): void {
  const {id} = props;
  if (props.endTimestamps && (props.endTimestamps.format === 'uint32x2') !== isWordMode) {
    throw new Error(`${id} endTimestamps must have the same format as timestamps`);
  }
  if (isWordMode) {
    if (props.timestampsLow || props.endTimestampsLow) {
      throw new Error(`${id} timestampsLow and endTimestampsLow require float32 timestamps`);
    }
    if (props.window.format !== 'uint32') {
      throw new Error(`${id} uint32x2 word timestamps require a packed uint32 word window`);
    }
    if (props.window.length < GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH) {
      throw new Error(
        `${id} word window must hold ${GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH} uint32 values`
      );
    }
  } else {
    if (props.window.format !== 'float32') {
      throw new Error(`${id} float32 timestamps require a float32 window`);
    }
    if (props.window.length < GPU_TIME_WINDOW_PARAMETER_LENGTH) {
      throw new Error(`${id} window must hold ${GPU_TIME_WINDOW_PARAMETER_LENGTH} float32 values`);
    }
  }
}

/** Classify kernel for exact Int64 word timestamps. Storage bindings: at most 6. */
function createWordClassifyNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TimeWindowClassifyProps,
  chunkIndex: number,
  isVector: boolean,
  chunk: GraphDataView,
  getChunk: (
    view: GraphDataView | GraphVectorView | undefined,
    chunkIndex: number
  ) => GraphDataView | undefined
): GPUCommandNode<Parameters> {
  const endWords = getChunk(props.endTimestamps, chunkIndex);
  const weights = getChunk(props.fadeWeights, chunkIndex);
  const clip = getChunk(props.clipFractions, chunkIndex);
  const bindings: MapGraphKernelBinding[] = [
    {name: 'params', view: props.window, type: 'u32', access: 'read'},
    {name: 'startWords', view: chunk, type: 'u32', access: 'read'}
  ];
  if (endWords)
    bindings.push({
      name: 'endWords',
      view: endWords,
      type: 'u32',
      access: 'read'
    });
  bindings.push({
    name: 'outputMask',
    view: getChunk(props.mask, chunkIndex) as GraphDataView,
    type: 'u32',
    access: 'read_write'
  });
  if (weights) {
    bindings.push({
      name: 'outputWeights',
      view: weights,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (clip)
    bindings.push({
      name: 'outputClip',
      view: clip,
      type: 'f32',
      access: 'read_write'
    });

  return createMapGraphKernelNode<Parameters>(graph, {
    id: isVector ? `${props.id}-chunk-${chunkIndex}` : props.id,
    operation: 'GPUTimeWindowFilter',
    variant: 'classify-words',
    bindings,
    invocationCount: chunk.length,
    declarations: TIME_WORDS_WGSL,
    body: /* wgsl */ `
  let windowStart = vec2<u32>(params[paramsOffset + 0u], params[paramsOffset + 1u]);
  let windowEnd = vec2<u32>(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let windowStartFraction = bitcast<f32>(params[paramsOffset + 4u]);
  let windowEndFraction = bitcast<f32>(params[paramsOffset + 5u]);
  let startFade = bitcast<f32>(params[paramsOffset + 6u]);
  let endFade = bitcast<f32>(params[paramsOffset + 7u]);

  let rowStart = vec2<u32>(
    startWords[startWordsOffset + 2u * index],
    startWords[startWordsOffset + 2u * index + 1u]
  );
  ${
    endWords
      ? `let rowEnd = vec2<u32>(
    endWords[endWordsOffset + 2u * index],
    endWords[endWordsOffset + 2u * index + 1u]
  );`
      : 'let rowEnd = rowStart;'
  }

  let endAfterWindowStart = timeWordsDifference(rowEnd, 0.0, windowStart, windowStartFraction);
  let startBeforeWindowEnd = timeWordsDifference(windowEnd, windowEndFraction, rowStart, 0.0);
  let accepted =
    isTimeWordsAtLeast(rowEnd, 0.0, windowStart, windowStartFraction) &&
    isTimeWordsAtLeast(windowEnd, windowEndFraction, rowStart, 0.0);
  outputMask[outputMaskOffset + index] = select(0u, 1u, accepted);
  ${
    weights
      ? `var startRamp = 1.0;
  if (startFade > 0.0) { startRamp = clamp(endAfterWindowStart / startFade, 0.0, 1.0); }
  var endRamp = 1.0;
  if (endFade > 0.0) { endRamp = clamp(startBeforeWindowEnd / endFade, 0.0, 1.0); }
  outputWeights[outputWeightsOffset + index] = select(0.0, min(startRamp, endRamp), accepted);`
      : ''
  }
  ${
    clip
      ? `let duration = timeWordsDifference(rowEnd, 0.0, rowStart, 0.0);
  var clipStart = 0.0;
  var clipEnd = 1.0;
  if (duration > 0.0) {
    clipStart = clamp(timeWordsDifference(windowStart, windowStartFraction, rowStart, 0.0) / duration, 0.0, 1.0);
    clipEnd = clamp(startBeforeWindowEnd / duration, 0.0, 1.0);
  }
  outputClip[outputClipOffset + 2u * index] = select(0.0, clipStart, accepted);
  outputClip[outputClipOffset + 2u * index + 1u] = select(0.0, clipEnd, accepted);`
      : ''
  }`
  });
}
