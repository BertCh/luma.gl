// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandNode} from './gpu-command-node';
import {type GPUCommandGraph, type GraphDataView} from './gpu-command-graph';
import {
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  type GPUBoundedDispatchLayout
} from './gpu-dispatch-utils';
import type {GPUGridIndex, GPUGridIndexBounds} from './gpu-grid-index';
import {GPUScan, getGPUScanCommandNodesWithDispatchLimit} from './gpu-scan';
import {GPUScatter, getGPUScatterCommandNodesWithDispatchLimit} from './gpu-scatter';
import {
  createTransientView,
  createTransientVectorView,
  getViewElementOffset
} from './graph-data-view-utils';
import {getGraphVectorData} from './graph-vector-view-utils';
import {getGraphDataRange, createChunkNode, validateChunkViews} from './gpu-chunk-utils';

const GRID_INDEX_WORKGROUP_SIZE = 256;

/** Rebuilds counts and global destinations while retaining every source and destination chunk. @internal */
export function getGPUGridIndexCommandNodesWithDispatchLimit<Parameters>(
  index: GPUGridIndex,
  graph: GPUCommandGraph<Parameters>,
  maximum: number
): readonly GPUCommandNode<Parameters>[] {
  validateChunkViews(
    graph,
    [
      index.positions,
      ...(index.sourceIds ? [index.sourceIds] : []),
      ...(index.boundsBuffer ? [index.boundsBuffer] : [])
    ],
    [index.cellOffsets, index.objectIds, index.count, index.overflow]
  );
  const nodes: GPUCommandNode<Parameters>[] = [];
  const offsets = getGraphDataRange(graph, index.cellOffsets, 0, index.cellCount);
  const counts = createTransientVectorView(graph, `${index.id}-counts`, offsets);
  const cursors = createTransientVectorView(graph, `${index.id}-cursors`, offsets);
  const positions = getGraphVectorData(index.positions);
  const ranks = positions.map((chunk, chunkIndex) =>
    createTransientView(graph, `${index.id}-ranks-${chunkIndex}`, 'uint32', chunk.length)
  );
  const generatedIds = index.sourceIds
    ? undefined
    : positions.map((chunk, chunkIndex) =>
        createTransientView(graph, `${index.id}-ids-${chunkIndex}`, 'uint32', chunk.length)
      );
  let positionStart = index.firstSourceIndex;
  for (const [chunkIndex, chunk] of positions.entries()) {
    if (chunk.length) {
      const rank = ranks[chunkIndex];
      const ids = generatedIds?.[chunkIndex];
      const dispatch = getGPUGridIndexDispatchLayout(chunk.length, maximum);
      nodes.push(
        createChunkNode(graph, {
          id: `${index.id}-initialize-ranks-${chunkIndex}`,
          outputs: {ranks: rank, ...(ids ? {ids} : {})},
          dispatch,
          source: `
@group(0) @binding(0) var<storage, read_write> ranks: array<u32>;
${ids ? '@group(0) @binding(1) var<storage, read_write> ids: array<u32>;' : ''}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${getGPUGridIndexInvocationIndexSource(dispatch)}
  if (index >= ${chunk.length}u) { return; }
  ranks[index] = 0xffffffffu;
  ${ids ? `ids[index] = ${positionStart}u + index;` : ''}
}`
        })
      );
    }
    positionStart += chunk.length;
  }
  let cellStart = 0;
  for (const [cellChunkIndex, cellCounts] of counts.data.entries()) {
    const cellCursors = cursors.data[cellChunkIndex];
    const dispatch = getGPUGridIndexDispatchLayout(cellCounts.length, maximum);
    nodes.push(
      createChunkNode(graph, {
        id: `${index.id}-initialize-${cellChunkIndex}`,
        outputs: {cellCounts, cellCursors},
        dispatch,
        source: `
@group(0) @binding(0) var<storage, read_write> cellCounts: array<u32>;
@group(0) @binding(1) var<storage, read_write> cellCursors: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${getGPUGridIndexInvocationIndexSource(dispatch)}
  if (index >= ${cellCounts.length}u) { return; }
  cellCounts[index] = 0u;
  cellCursors[index] = 0u;
}`
      })
    );
    for (const [positionChunkIndex, chunk] of positions.entries()) {
      if (!chunk.length) continue;
      const dispatch = getGPUGridIndexDispatchLayout(chunk.length, maximum);
      nodes.push(
        createChunkNode(graph, {
          id: `${index.id}-count-${cellChunkIndex}-${positionChunkIndex}`,
          inputs: {positions: chunk, ...getBoundsInput(index)},
          outputs: {cellCounts},
          dispatch,
          source: makePositionPassSource(
            index,
            chunk,
            dispatch,
            binding =>
              `@group(0) @binding(${binding}) var<storage, read_write> cellCounts: array<atomic<u32>>;`,
            `
  if (accepted && cellIndex >= ${cellStart}u && cellIndex - ${cellStart}u < ${cellCounts.length}u) {
    atomicAdd(&cellCounts[cellIndex - ${cellStart}u], 1u);
  }`
          )
        })
      );
    }
    cellStart += cellCounts.length;
  }
  nodes.push(
    ...getGPUScanCommandNodesWithDispatchLimit(
      new GPUScan({id: `${index.id}-scan`, input: counts, output: offsets}),
      graph,
      maximum
    )
  );
  const lastCounts = counts.data[counts.data.length - 1];
  const lastOffsets = offsets.data[offsets.data.length - 1];
  const terminal = getGraphDataRange(graph, index.cellOffsets, index.cellCount, 1).data[0];
  nodes.push(
    createChunkNode(graph, {
      id: `${index.id}-finalize`,
      inputs: {counts: lastCounts, offsets: lastOffsets},
      outputs: {count: index.count, overflow: index.overflow},
      dispatch: {x: 1, y: 1, z: 1},
      workgroupSize: 1,
      source: `
@group(0) @binding(0) var<storage, read> counts: array<u32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> count: array<u32>;
@group(0) @binding(3) var<storage, read_write> overflow: array<u32>;
@compute @workgroup_size(1)
fn main() {
  let total = counts[${lastCounts.length - 1}u] + offsets[${getViewElementOffset(lastOffsets) + lastOffsets.length - 1}u];
  count[${getViewElementOffset(index.count)}u] = total;
  overflow[${getViewElementOffset(index.overflow)}u] = select(0u, 1u, total > ${index.objectIds.length}u);
}`
    })
  );
  nodes.push(
    createChunkNode(graph, {
      id: `${index.id}-terminal`,
      inputs: {count: index.count},
      outputs: {terminal},
      dispatch: {x: 1, y: 1, z: 1},
      workgroupSize: 1,
      source: `
@group(0) @binding(0) var<storage, read> count: array<u32>;
@group(0) @binding(1) var<storage, read_write> terminal: array<u32>;
@compute @workgroup_size(1)
fn main() {
  terminal[${getViewElementOffset(terminal)}u] = count[${getViewElementOffset(index.count)}u];
}`
    })
  );
  cellStart = 0;
  for (const [cellChunkIndex, cellOffsets] of offsets.data.entries()) {
    const cellCursors = cursors.data[cellChunkIndex];
    for (const [positionChunkIndex, chunk] of positions.entries()) {
      if (!chunk.length || !index.objectIds.length) continue;
      const dispatch = getGPUGridIndexDispatchLayout(chunk.length, maximum);
      nodes.push(
        createChunkNode(graph, {
          id: `${index.id}-rank-${cellChunkIndex}-${positionChunkIndex}`,
          inputs: {positions: chunk, ...getBoundsInput(index), cellOffsets},
          outputs: {cellCursors, ranks: ranks[positionChunkIndex]},
          dispatch,
          source: makePositionPassSource(
            index,
            chunk,
            dispatch,
            binding => `
@group(0) @binding(${binding}) var<storage, read> cellOffsets: array<u32>;
@group(0) @binding(${binding + 1}) var<storage, read_write> cellCursors: array<atomic<u32>>;
@group(0) @binding(${binding + 2}) var<storage, read_write> ranks: array<u32>;`,
            `
  if (accepted && cellIndex >= ${cellStart}u && cellIndex - ${cellStart}u < ${cellOffsets.length}u) {
    let cell = cellIndex - ${cellStart}u;
    ranks[index] = cellOffsets[${getViewElementOffset(cellOffsets)}u + cell] + atomicAdd(&cellCursors[cell], 1u);
  }`
          )
        })
      );
    }
    cellStart += cellOffsets.length;
  }
  let sourceStart = 0;
  for (const [chunkIndex, chunk] of positions.entries()) {
    if (chunk.length && index.objectIds.length)
      nodes.push(
        ...getGPUScatterCommandNodesWithDispatchLimit(
          new GPUScatter({
            id: `${index.id}-scatter-${chunkIndex}`,
            source: index.sourceIds
              ? getGraphDataRange(graph, index.sourceIds, sourceStart, chunk.length)
              : generatedIds![chunkIndex],
            indices: ranks[chunkIndex],
            output: index.objectIds
          }),
          graph,
          maximum
        )
      );
    sourceStart += chunk.length;
  }
  return nodes;
}

function makePositionPassSource(
  props: GPUGridIndex,
  positions: GraphDataView,
  dispatch: GPUBoundedDispatchLayout,
  getBindings: (firstBinding: number) => string,
  body: string
): string {
  const bounds = getGPUGridIndexBoundsSource(props.bounds, props.boundsBuffer, props.dimension, 1);
  const bindings = getBindings(props.boundsBuffer ? 2 : 1);
  const width = props.gridSize[0];
  const height = props.gridSize[1];
  const depth = props.dimension === 3 ? props.gridSize[2] : 1;
  const zDeclarations =
    props.dimension === 3
      ? `let z = positions[POSITIONS_OFFSET + index * 3u + 2u];
    let finiteZ = z == z && abs(z) <= 3.402823466e+38;
    let inZ = z >= ${bounds.minimum[2]} && z <= ${bounds.maximum[2]};`
      : 'let finiteZ = true;\n    let inZ = true;';
  const zCoordinate =
    props.dimension === 3
      ? `let layer = getCoordinate(z, ${bounds.minimum[2]}, ${bounds.maximum[2]}, DEPTH);`
      : 'let layer = 0u;';
  return /* wgsl */ `
const ELEMENT_COUNT: u32 = ${positions.length}u;
const POSITIONS_OFFSET: u32 = ${getViewElementOffset(positions)}u;
const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const DEPTH: u32 = ${depth}u;
${bounds.declarations}@group(0) @binding(0) var<storage, read> positions: array<f32>;
${bindings}

fn getCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
  if (maximum == minimum || value == minimum) { return 0u; }
  if (value == maximum) { return size - 1u; }
  if (minimum < 0.0 && maximum > 0.0) {
    let scale = max(abs(minimum), abs(maximum));
    let scaledValue = value / scale;
    let scaledMinimum = minimum / scale;
    let scaledMaximum = maximum / scale;
    return min(
      u32((scaledValue - scaledMinimum) / (scaledMaximum - scaledMinimum) * f32(size)),
      size - 1u
    );
  }
  return min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${getGPUGridIndexInvocationIndexSource(dispatch)}
  var accepted = false;
  var cellIndex = 0u;
  if (index < ELEMENT_COUNT) {
    let x = positions[POSITIONS_OFFSET + index * ${props.dimension}u];
    let y = positions[POSITIONS_OFFSET + index * ${props.dimension}u + 1u];
    ${zDeclarations}
    let finiteXY = x == x && y == y && abs(x) <= 3.402823466e+38 && abs(y) <= 3.402823466e+38;
    let inX = x >= ${bounds.minimum[0]} && x <= ${bounds.maximum[0]};
    let inY = y >= ${bounds.minimum[1]} && y <= ${bounds.maximum[1]};
    if (${bounds.validity}finiteXY && finiteZ && inX && inY && inZ) {
      accepted = true;
      let column = getCoordinate(x, ${bounds.minimum[0]}, ${bounds.maximum[0]}, WIDTH);
      let row = getCoordinate(y, ${bounds.minimum[1]}, ${bounds.maximum[1]}, HEIGHT);
      ${zCoordinate}
      cellIndex = (layer * HEIGHT + row) * WIDTH + column;
    }
  }
  ${body}
}`;
}

/** Plans bounded 3D dispatches for index kernels. @internal */
export function getGPUGridIndexDispatchLayout(
  count: number,
  maximum: number
): GPUBoundedDispatchLayout {
  return getBoundedDispatchLayout('GPUGridIndex', count, GRID_INDEX_WORKGROUP_SIZE, maximum);
}

/** Maps a bounded dispatch to logical rows. @internal */
export function getGPUGridIndexInvocationIndexSource(layout: GPUBoundedDispatchLayout): string {
  return getBoundedInvocationIndexSource(layout, GRID_INDEX_WORKGROUP_SIZE);
}

/** Returns the optional bounds buffer input for a kernel, bound directly after `positions`. @internal */
function getBoundsInput(index: GPUGridIndex): {boundsBuffer?: GraphDataView<'float32'>} {
  return index.boundsBuffer ? {boundsBuffer: index.boundsBuffer} : {};
}

/** Returns whether literal bounds are finite with minima not above maxima. @internal */
export function isOrderedFiniteBounds(bounds: GPUGridIndexBounds): boolean {
  const dimension = bounds.length / 2;
  return (
    bounds.every(Number.isFinite) &&
    Array.from({length: dimension}, (_, axis) => axis).every(
      axis => bounds[axis]! <= bounds[axis + dimension]!
    )
  );
}

/**
 * WGSL for an index domain: per-axis minimum and maximum expressions, module-scope declarations,
 * and a leading `&&` clause rejecting invalid run-time bounds.
 *
 * Without `boundsBuffer`, literal bounds yield float literals and no declarations or validity clause. A bounds buffer is read
 * through the `boundsValues` storage binding at `binding`, so the caller must bind it there. @internal
 */
export function getGPUGridIndexBoundsSource(
  bounds: GPUGridIndexBounds,
  boundsBuffer: GraphDataView<'float32'> | undefined,
  dimension: 2 | 3,
  binding: number
): {minimum: string[]; maximum: string[]; declarations: string; validity: string} {
  if (!boundsBuffer) {
    return {
      minimum: Array.from({length: dimension}, (_, axis) => getFloatLiteral(bounds[axis]!)),
      maximum: Array.from({length: dimension}, (_, axis) =>
        getFloatLiteral(bounds[axis + dimension]!)
      ),
      declarations: '',
      validity: ''
    };
  }
  const read = (component: number) => `boundsValues[BOUNDS_OFFSET + ${component}u]`;
  const minimum = Array.from({length: dimension}, (_, axis) => read(axis));
  const maximum = Array.from({length: dimension}, (_, axis) => read(axis + dimension));
  const validity = minimum.map(
    (value, axis) =>
      `${value} == ${value} && ${maximum[axis]} == ${maximum[axis]} && abs(${value}) <= 3.402823466e+38 && abs(${maximum[axis]}) <= 3.402823466e+38 && ${value} <= ${maximum[axis]}`
  );
  return {
    minimum,
    maximum,
    declarations: `const BOUNDS_OFFSET: u32 = ${getViewElementOffset(boundsBuffer)}u;
@group(0) @binding(${binding}) var<storage, read> boundsValues: array<f32>;
`,
    validity: `${validity.join(' && ')} && `
  };
}

export function getFloatLiteral(value: number): string {
  const literal = `${Math.fround(value)}`;
  return literal.includes('.') || literal.includes('e') ? literal : `${literal}.0`;
}
