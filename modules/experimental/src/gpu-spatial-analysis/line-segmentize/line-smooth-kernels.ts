// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';

/**
 * Per-line Chaikin preparation: `smoothCounts[p]` is the vertex count of a smoothable line (open
 * with at least two vertices, or closed with at least three after dropping a repeated closing
 * vertex), `plainCounts[p]` the vertex count of a line that is copied unchanged, and
 * `smoothFlags[p]` 1 for smoothable lines. Row `lineCount` of each column is 0 so an exclusive scan
 * ends with the totals.
 *
 * @internal
 */
export function createSmoothPrepareNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    positions: GraphDataView<'float32x2'>;
    lineOffsets: GraphDataView<'uint32'>;
    closed: boolean;
    smoothCounts: GraphDataView<'uint32'>;
    plainCounts: GraphDataView<'uint32'>;
    smoothFlags: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const lineCount = props.lineOffsets.length - 1;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'smooth-prepare',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
      {name: 'smoothCounts', view: props.smoothCounts, type: 'u32', access: 'read_write'},
      {name: 'plainCounts', view: props.plainCounts, type: 'u32', access: 'read_write'},
      {name: 'smoothFlags', view: props.smoothFlags, type: 'u32', access: 'read_write'}
    ],
    invocationCount: lineCount + 1,
    declarations: `const LINE_COUNT: u32 = ${lineCount}u;
const ROW_COUNT: u32 = ${props.positions.length}u;
const CLOSED: bool = ${props.closed};

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,
    body: /* wgsl */ `var smoothCount = 0u;
  var plainCount = 0u;
  if (index < LINE_COUNT) {
    let start = min(lineOffsets[lineOffsetsOffset + index], ROW_COUNT);
    let end = min(max(lineOffsets[lineOffsetsOffset + index + 1u], start), ROW_COUNT);
    let count = end - start;
    var effectiveCount = count;
    if (CLOSED && count >= 2u && all(getPosition(start) == getPosition(end - 1u))) {
      effectiveCount = count - 1u;
    }
    let smoothable = select(count >= 2u, effectiveCount >= 3u, CLOSED);
    smoothCount = select(0u, effectiveCount, smoothable);
    plainCount = select(count, 0u, smoothable);
  }
  smoothCounts[smoothCountsOffset + index] = smoothCount;
  plainCounts[plainCountsOffset + index] = plainCount;
  smoothFlags[smoothFlagsOffset + index] = select(0u, 1u, smoothCount > 0u);`
  });
}

/** WGSL helpers for line offsets at a Chaikin level, from the scanned prepare columns. */
const LEVEL_OFFSETS_WGSL = /* wgsl */ `
fn getLevelOffset(line: u32, scale: u32, withClosing: bool) -> u32 {
  var offset = smoothStarts[smoothStartsOffset + line] * scale + plainStarts[plainStartsOffset + line];
  if (withClosing) {
    offset += getClosingStart(line);
  }
  return offset;
}
`;

/**
 * Builds one Chaikin level: every output slot finds its line by binary search over the level's
 * line offsets and computes its vertex from two vertices of the previous level. Open lines keep
 * both endpoints; closed rings wrap and, at the final level, repeat their first vertex.
 *
 * @internal
 */
export function createSmoothLevelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    /** Level number, for the variant label only. */
    level: number;
    /** One-word state holding the current level; {@link createSmoothLevelAdvanceNode} steps it. */
    levelState: GraphDataView<'uint32'>;
    isFinal: boolean;
    closed: boolean;
    lineCount: number;
    /** Level 0 reads the input positions with `lineOffsets`; later levels read the previous level. */
    source: GraphDataView<'float32x2'>;
    lineOffsets: GraphDataView<'uint32'>;
    smoothStarts: GraphDataView<'uint32'>;
    plainStarts: GraphDataView<'uint32'>;
    smoothFlagStarts?: GraphDataView<'uint32'>;
    parameters: GraphDataView<'float32'>;
    destination: GraphDataView<'float32x2'>;
  }
): GPUCommandNode<Parameters> {
  const withClosing = props.isFinal && props.closed;
  const bindings: WGSLKernelBinding[] = [
    {name: 'source', view: props.source, type: 'f32', access: 'read'},
    {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
    {name: 'smoothStarts', view: props.smoothStarts, type: 'u32', access: 'read'},
    {name: 'plainStarts', view: props.plainStarts, type: 'u32', access: 'read'},
    {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
    {name: 'destination', view: props.destination, type: 'f32', access: 'read_write'},
    {name: 'levelState', view: props.levelState, type: 'u32', access: 'read'}
  ];
  if (withClosing && props.smoothFlagStarts) {
    bindings.push({
      name: 'smoothFlagStarts',
      view: props.smoothFlagStarts,
      type: 'u32',
      access: 'read'
    });
  }
  const capacity = props.destination.length;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: `smooth-level-${props.level}`,
    bindings,
    invocationCount: capacity,
    declarations: `const LINE_COUNT: u32 = ${props.lineCount}u;
const CAPACITY: u32 = ${capacity}u;
const SOURCE_LENGTH: u32 = ${props.source.length}u;
const IS_FIRST_LEVEL: bool = ${props.level === 0};
const CLOSED: bool = ${props.closed};
const WITH_CLOSING: bool = ${withClosing};

fn getClosingStart(line: u32) -> u32 {
  ${withClosing ? 'return smoothFlagStarts[smoothFlagStartsOffset + line];' : 'return 0u;'}
}
${LEVEL_OFFSETS_WGSL}

fn getSource(row: u32) -> vec2<f32> {
  return vec2<f32>(source[sourceOffset + 2u * row], source[sourceOffset + 2u * row + 1u]);
}`,
    body: /* wgsl */ `// The level lives in a GPU word, so no pipeline depends on the level (rule 12).
  let sourceScale = 1u << levelState[levelStateOffset];
  let destinationScale = sourceScale << 1u;
  let total = getLevelOffset(LINE_COUNT, destinationScale, WITH_CLOSING);
  if (index >= total) {
    return;
  }
  // Largest line whose destination offset is at most index.
  var low = 0u;
  var high = LINE_COUNT;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (getLevelOffset(middle, destinationScale, WITH_CLOSING) <= index) {
      low = middle;
    } else {
      high = middle;
    }
  }
  let line = low;
  let local = index - getLevelOffset(line, destinationScale, WITH_CLOSING);
  let smoothCount = smoothStarts[smoothStartsOffset + line + 1u] - smoothStarts[smoothStartsOffset + line];
  var sourceStart = getLevelOffset(line, sourceScale, false);
  if (IS_FIRST_LEVEL) {
    sourceStart = lineOffsets[lineOffsetsOffset + line];
  }
  var vertex: vec2<f32>;
  if (smoothCount == 0u) {
    // Copied line.
    if (sourceStart + local >= SOURCE_LENGTH) {
      return;
    }
    vertex = getSource(sourceStart + local);
  } else {
    let count = smoothCount * sourceScale;
    let ratio = parameters[parametersOffset];
    var edge = 0u;
    var isSecond = false;
    var copyRow = 0xffffffffu;
    if (CLOSED) {
      let wrapped = select(local, 0u, local == 2u * count);
      edge = wrapped / 2u;
      isSecond = (wrapped & 1u) == 1u;
    } else if (local == 0u) {
      copyRow = 0u;
    } else if (local == 2u * count - 1u) {
      copyRow = count - 1u;
    } else {
      edge = (local - 1u) / 2u;
      isSecond = ((local - 1u) & 1u) == 1u;
    }
    if (copyRow != 0xffffffffu) {
      if (sourceStart + copyRow >= SOURCE_LENGTH) {
        return;
      }
      vertex = getSource(sourceStart + copyRow);
    } else {
      let nextEdge = select(edge + 1u, 0u, edge + 1u == count);
      if (sourceStart + max(edge, nextEdge) >= SOURCE_LENGTH) {
        return;
      }
      let start = getSource(sourceStart + edge);
      let end = getSource(sourceStart + nextEdge);
      vertex = mix(start, end, select(ratio, 1.0 - ratio, isSecond));
    }
  }
  if (index < CAPACITY) {
    destination[destinationOffset + 2u * index] = vertex.x;
    destination[destinationOffset + 2u * index + 1u] = vertex.y;
  }`
  });
}

/**
 * Publishes Chaikin output line offsets (clamped), count, overflow, total and line count.
 *
 * @internal
 */
export function createSmoothPublishNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    iterationCount: number;
    closed: boolean;
    lineCount: number;
    smoothStarts: GraphDataView<'uint32'>;
    plainStarts: GraphDataView<'uint32'>;
    smoothFlagStarts?: GraphDataView<'uint32'>;
    capacity: number;
    outputLineOffsets: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    requiredCount?: GraphDataView<'uint32'>;
    lineCountOutput?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'smoothStarts', view: props.smoothStarts, type: 'u32', access: 'read'},
    {name: 'plainStarts', view: props.plainStarts, type: 'u32', access: 'read'}
  ];
  if (props.closed && props.smoothFlagStarts) {
    bindings.push({
      name: 'smoothFlagStarts',
      view: props.smoothFlagStarts,
      type: 'u32',
      access: 'read'
    });
  }
  bindings.push(
    {name: 'outputLineOffsets', view: props.outputLineOffsets, type: 'u32', access: 'read_write'},
    {name: 'countOut', view: props.count, type: 'u32', access: 'read_write'},
    {name: 'overflowOut', view: props.overflow, type: 'u32', access: 'read_write'}
  );
  if (props.requiredCount) {
    bindings.push({name: 'totalOut', view: props.requiredCount, type: 'u32', access: 'read_write'});
  }
  if (props.lineCountOutput) {
    bindings.push({
      name: 'lineCountOut',
      view: props.lineCountOutput,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'smooth-publish',
    bindings,
    invocationCount: props.lineCount + 1,
    declarations: `const LINE_COUNT: u32 = ${props.lineCount}u;
const CAPACITY: u32 = ${props.capacity}u;
const SCALE: u32 = ${2 ** props.iterationCount}u;

fn getClosingStart(line: u32) -> u32 {
  ${props.closed ? 'return smoothFlagStarts[smoothFlagStartsOffset + line];' : 'return 0u;'}
}
${LEVEL_OFFSETS_WGSL}`,
    body: /* wgsl */ `let total = getLevelOffset(LINE_COUNT, SCALE, true);
  outputLineOffsets[outputLineOffsetsOffset + index] = min(getLevelOffset(index, SCALE, true), CAPACITY);
  if (index == 0u) {
    countOut[countOutOffset] = min(total, CAPACITY);
    overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY);
    ${props.requiredCount ? 'totalOut[totalOutOffset] = total;' : ''}
    ${props.lineCountOutput ? 'lineCountOut[lineCountOutOffset] = LINE_COUNT;' : ''}
  }`
  });
}

/**
 * Resets (`reset: true`) or advances the one-word Chaikin level state read by
 * {@link createSmoothLevelNode}.
 *
 * @internal
 */
export function createSmoothLevelStateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; operation: string; levelState: GraphDataView<'uint32'>; reset: boolean}
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: props.reset ? 'smooth-level-reset' : 'smooth-level-advance',
    bindings: [{name: 'levelState', view: props.levelState, type: 'u32', access: 'read_write'}],
    invocationCount: 1,
    body: props.reset
      ? 'levelState[levelStateOffset] = 0u;'
      : 'levelState[levelStateOffset] = levelState[levelStateOffset] + 1u;'
  });
}
