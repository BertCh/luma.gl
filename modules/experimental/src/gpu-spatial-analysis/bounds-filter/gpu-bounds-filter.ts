// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUVisibilityWorkflow,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import {
  validateCompactOutput,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {createPublishNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';

const OPERATION = 'GPUBoundsFilter';

/** Number of float32 elements of the per-frame box: `[minX, minY, maxX, maxY]`. */
export const GPU_BOUNDS_FILTER_PARAMETER_LENGTH = 4;

/**
 * Relation between each feature's bounding box `F` and the query box `B`.
 *
 * - `'intersects'`: `F` and `B` overlap or touch (GeoPandas `.cx` on bounds).
 * - `'within'`: `F` lies inside `B`, borders included.
 * - `'contains'`: `F` covers `B`, borders included.
 */
export type GPUBoundsFilterMode = 'intersects' | 'within' | 'contains';

/** CPU description of the per-frame box of {@link GPUBoundsFilter}. */
export type GPUBoundsFilterBox = {
  /** Left edge. */
  minX: number;
  /** Bottom edge. */
  minY: number;
  /** Right edge, at least `minX`. */
  maxX: number;
  /** Top edge, at least `minY`. */
  maxY: number;
};

/**
 * Packs a {@link GPUBoundsFilterBox} into the 4-element float32 layout `[minX, minY, maxX, maxY]`.
 * Write the result into a `GPUParameterBuffer` between encodings to move the box without
 * recompiling.
 *
 * @param box Box to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is not finite, the box is inverted, or `target` is too short.
 */
export function getGPUBoundsFilterParameterValues(
  box: GPUBoundsFilterBox,
  target: Float32Array = new Float32Array(GPU_BOUNDS_FILTER_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_BOUNDS_FILTER_PARAMETER_LENGTH) {
    throw new Error(
      `Bounds filter target must hold ${GPU_BOUNDS_FILTER_PARAMETER_LENGTH} elements`
    );
  }
  const {minX, minY, maxX, maxY} = box;
  if (![minX, minY, maxX, maxY].every(Number.isFinite)) {
    throw new Error('Bounds filter box must be finite');
  }
  if (maxX < minX || maxY < minY) {
    throw new Error('Bounds filter requires maxX >= minX and maxY >= minY');
  }
  target.set([minX, minY, maxX, maxY]);
  return target;
}

/**
 * Properties for {@link GPUBoundsFilter}.
 *
 * Per-frame (no recompile): the box and every input buffer. Compile-time: view lengths, `mode`,
 * `output.ids.length` and which optional views are present.
 */
export type GPUBoundsFilterProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'bounds-filter'`. */
  id?: string;
  /**
   * Packed per-feature bounds `[minX, minY, maxX, maxY]`, for example `GPUGeometryMeasures`
   * bounds. Rows with a non-finite component (NaN or infinite, the empty-geometry marker) are
   * rejected in every mode.
   */
  bounds: GraphDataView<'float32x4'>;
  /** Per-frame box: a packed float32 view written with {@link getGPUBoundsFilterParameterValues}. */
  box: GraphDataView<'float32'>;
  /** Relation tested between each feature's bounds and the box. Defaults to `'intersects'`. */
  mode?: GPUBoundsFilterMode;
  /** Caller-owned bounded result of accepted feature rows, ascending. `ids.length` is the capacity. */
  output: GPUCompactOutput;
  /** Optional caller-owned canonical 0/1 mask, one row per feature. */
  mask?: GraphDataView<'uint32'>;
};

/**
 * Selects features whose bounding boxes relate to a per-frame box: a GPU `GeoDataFrame.cx[...]`
 * (bounds level) and `gdf.bounds` comparison.
 *
 * Note this is a bounds test, not an exact geometry test: `'intersects'` accepts a feature whose
 * bounding box overlaps the box even when the geometry misses it. Refine the survivors with
 * `GPUSpatialPredicateJoin` against a one-row box polygon for exact `cx` semantics.
 *
 * One mask kernel and a stable compaction publish the accepted row IDs, a clamped count and an
 * overflow flag. The box is read per frame, so a viewport or slider moves without recompiling.
 * Comparisons are closed (borders count) and exact in f32 (no arithmetic).
 */
export class GPUBoundsFilter implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUBoundsFilterProps;
  /** Resolved relation mode. */
  readonly mode: GPUBoundsFilterMode;

  constructor(props: GPUBoundsFilterProps) {
    this.id = props.id ?? 'bounds-filter';
    this.props = props;
    this.mode = props.mode ?? 'intersects';
    const {id} = this;
    if (!['intersects', 'within', 'contains'].includes(this.mode)) {
      throw new Error(`${id} mode must be 'intersects', 'within' or 'contains'`);
    }
    validatePackedView(props.bounds, ['float32x4'], `${id} bounds`);
    if (props.bounds.length < 1) {
      throw new Error(`${id} needs at least one bounds row`);
    }
    validatePackedView(props.box, ['float32'], `${id} box`);
    if (props.box.length < GPU_BOUNDS_FILTER_PARAMETER_LENGTH) {
      throw new Error(`${id} box must hold ${GPU_BOUNDS_FILTER_PARAMETER_LENGTH} float32 values`);
    }
    validateCompactOutput(id, props.output);
    if (props.output.ids.length < 1) {
      throw new Error(`${id} output.ids must contain at least one row`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== props.bounds.length) {
        throw new Error(`${id} mask length must equal bounds length`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.output.ids,
        props.output.count,
        props.output.overflow,
        props.output.totalCount,
        props.mask
      ],
      [props.bounds, props.box]
    );
  }

  /** Returns the mask kernel, the stable compaction nodes and the publish node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.bounds,
      props.box,
      props.mask,
      output.ids,
      output.count,
      output.overflow,
      output.totalCount
    ]);
    const rows = props.bounds.length;
    const mask = props.mask ?? createTransientView(graph, `${id}-mask`, 'uint32', rows);
    const comparison = {
      intersects: `fMin.x <= boxMax.x && fMax.x >= boxMin.x && fMin.y <= boxMax.y && fMax.y >= boxMin.y`,
      within: `fMin.x >= boxMin.x && fMax.x <= boxMax.x && fMin.y >= boxMin.y && fMax.y <= boxMax.y`,
      contains: `fMin.x <= boxMin.x && fMax.x >= boxMax.x && fMin.y <= boxMin.y && fMax.y >= boxMax.y`
    }[this.mode];
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mask`,
        operation: OPERATION,
        bindings: [
          {name: 'bounds', view: props.bounds, type: 'f32', access: 'read'},
          {name: 'box', view: props.box, type: 'f32', access: 'read'},
          {name: 'maskOut', view: mask, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `// Exponent bits all ones: NaN or infinity. A bit test, because Metal may fold NaN compares.
fn isFiniteBits(value: f32) -> bool {
  return ((bitcast<u32>(value) >> 23u) & 0xffu) != 0xffu;
}`,
        body: `let base = boundsOffset + 4u * index;
  let fMin = vec2<f32>(bounds[base], bounds[base + 1u]);
  let fMax = vec2<f32>(bounds[base + 2u], bounds[base + 3u]);
  let boxMin = vec2<f32>(box[boxOffset], box[boxOffset + 1u]);
  let boxMax = vec2<f32>(box[boxOffset + 2u], box[boxOffset + 3u]);
  let finite = isFiniteBits(fMin.x) && isFiniteBits(fMin.y) && isFiniteBits(fMax.x) && isFiniteBits(fMax.y);
  let accepted = finite && ${comparison};
  maskOut[maskOutOffset + index] = select(0u, 1u, accepted);`
      })
    ];
    // Compact straight into the caller's IDs when they hold every row; otherwise compact into
    // full-size scratch and let the publish kernel copy the bounded prefix.
    const direct = output.ids.length >= rows;
    const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
    const compactIds = direct
      ? output.ids
      : createTransientView(graph, `${id}-compact-ids`, 'uint32', rows);
    nodes.push(
      ...new GPUVisibilityWorkflow({
        id: `${id}-visibility`,
        predicates: [{kind: 'bounds', mask}],
        output: compactIds,
        count: total
      }).getCommandNodes(graph),
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        totalCount: total,
        compactIds: direct ? undefined : compactIds,
        output
      })
    );
    return nodes;
  }
}
