// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {createPathPrefixNode} from '../line-segmentize/line-segmentize-kernels';
import {
  createLineLocateNode,
  createPackedScatterNodes,
  LINE_LOCATE_RESULT_STRIDE
} from './linear-referencing-kernels';

const OPERATION = 'GPULineLocate';

/** Number of float32 elements in a `GPULineLocate` parameter buffer. */
export const GPU_LINE_LOCATE_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame parameters of `GPULineLocate`. */
export type GPULineLocateParameters = {
  /** Multiplies every event measure before locating. Default 1. */
  measureScale?: number;
  /** Added to every scaled event measure, for example elapsed distance in an animation. Default 0. */
  measureOffset?: number;
};

/**
 * Packs `GPULineLocate` parameters into the 4-element float32 layout
 * `[measureScale, measureOffset, 0, 0]`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is not finite or `target` is too short.
 */
export function getGPULineLocateParameterValues(
  parameters: GPULineLocateParameters,
  target: Float32Array = new Float32Array(GPU_LINE_LOCATE_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_LINE_LOCATE_PARAMETER_LENGTH) {
    throw new Error(`Line locate target must hold ${GPU_LINE_LOCATE_PARAMETER_LENGTH} elements`);
  }
  const measureScale = parameters.measureScale ?? 1;
  const measureOffset = parameters.measureOffset ?? 0;
  if (!Number.isFinite(measureScale) || !Number.isFinite(measureOffset)) {
    throw new Error('Line locate measureScale and measureOffset must be finite');
  }
  target.set([measureScale, measureOffset, 0, 0]);
  return target;
}

/** Per-event outputs of {@link GPULineLocate}; `eventPaths.length` rows each. */
export type GPULineLocateOutput = {
  /** Located (and laterally offset) positions; NaN for invalid events. */
  positions: GraphDataView<'float32x2'>;
  /** Segment index within the path, or `0xffffffff` for invalid events. */
  segmentIndices?: GraphDataView<'uint32'>;
  /** Unit direction of the segment, `(0, 0)` on single-vertex paths and zero-length segments. */
  tangents?: GraphDataView<'float32x2'>;
  /**
   * Segment direction in degrees counter-clockwise from the +x axis (deck.gl `getAngle`
   * convention for planar or Web Mercator-like coordinates); `0` where the tangent is `(0, 0)`.
   */
  angles?: GraphDataView<'float32'>;
  /** `GPU_LINE_LOCATE_STATUS` code per event. */
  statuses?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPULineLocate}.
 *
 * Per-frame (no recompile): the contents of `parameters` and of every input buffer. Compile-time:
 * view lengths, the path and event counts, `measureMode`, and which optional views are present.
 */
export type GPULineLocateProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'line-locate'`. */
  id?: string;
  /** Packed planar path vertices sorted by path. */
  positions: GraphDataView<'float32x2'>;
  /** `pathCount + 1` monotonic row offsets. */
  pathOffsets: GraphDataView<'uint32'>;
  /** Path index of each event. */
  eventPaths: GraphDataView<'uint32'>;
  /** Measure of each event: a distance from the path start, or a fraction of the path length. */
  eventMeasures: GraphDataView<'float32'>;
  /** Optional lateral offset of each event, positive to the left of the path direction. */
  eventOffsets?: GraphDataView<'float32'>;
  /** `'distance'` (default) or `'fraction'` of each path's length (turf `along` vs. `ST_LineInterpolatePoint`). */
  measureMode?: 'distance' | 'fraction';
  /**
   * Optional per-frame float32 view of {@link GPU_LINE_LOCATE_PARAMETER_LENGTH} elements written
   * with `getGPULineLocateParameterValues`; transforms every measure as `measure * scale + offset`.
   */
  parameters?: GraphDataView<'float32'>;
  /** Per-event outputs. */
  output: GPULineLocateOutput;
  /** Optional per-vertex cumulative path measure, `positions.length` rows. */
  vertexMeasures?: GraphDataView<'float32'>;
};

/**
 * Places events along paths by measure (turf `along`, PostGIS `ST_LineInterpolatePoint` and
 * `ST_LocateAlong` with offset): route events, mile markers, vehicles moving along routes.
 *
 * A per-path sequential prefix computes cumulative vertex measures (Neumaier-compensated,
 * deterministic). Each event clamps its measure into `[0, pathLength]` (status `clamped`), finds
 * the segment with an upper-bound binary search over the measures (so a measure that lands on a
 * vertex continues on the following segment, and zero-length segments are skipped), and
 * interpolates by measure. Lateral offsets move the point along the segment's left normal.
 * Events on empty or out-of-range paths are `invalid` with NaN positions.
 *
 * Planar only: measures are Euclidean in position units.
 */
export class GPULineLocate implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULineLocateProps;
  /** Resolved measure mode. */
  readonly measureMode: 'distance' | 'fraction';

  constructor(props: GPULineLocateProps) {
    this.id = props.id ?? 'line-locate';
    this.props = props;
    this.measureMode = props.measureMode ?? 'distance';
    const {id} = this;
    const {output} = props;
    const inputs = {
      positions: props.positions,
      pathOffsets: props.pathOffsets,
      eventPaths: props.eventPaths,
      eventMeasures: props.eventMeasures,
      eventOffsets: props.eventOffsets,
      parameters: props.parameters,
      vertexMeasures: props.vertexMeasures
    };
    for (const [name, view] of [
      ...Object.entries(inputs),
      ...Object.entries(output).map(([column, columnView]) => [`output.${column}`, columnView])
    ]) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    if (this.measureMode !== 'distance' && this.measureMode !== 'fraction') {
      throw new Error(`${id} measureMode must be 'distance' or 'fraction'`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    validatePackedUint32View(props.pathOffsets, `${id} pathOffsets`);
    if (props.pathOffsets.length < 2) {
      throw new Error(`${id} pathOffsets must contain at least two rows`);
    }
    validatePackedUint32View(props.eventPaths, `${id} eventPaths`);
    const eventCount = props.eventPaths.length;
    if (eventCount < 1) {
      throw new Error(`${id} needs at least one event`);
    }
    for (const [name, view] of [
      ['eventMeasures', props.eventMeasures],
      ['eventOffsets', props.eventOffsets]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== eventCount) {
          throw new Error(`${id} ${name} length must equal eventPaths length`);
        }
      }
    }
    if (props.parameters) {
      validatePackedView(props.parameters, ['float32'], `${id} parameters`);
      if (props.parameters.length < GPU_LINE_LOCATE_PARAMETER_LENGTH) {
        throw new Error(
          `${id} parameters must hold ${GPU_LINE_LOCATE_PARAMETER_LENGTH} float32 values`
        );
      }
    }
    if (props.vertexMeasures) {
      validatePackedView(props.vertexMeasures, ['float32'], `${id} vertexMeasures`);
      if (props.vertexMeasures.length !== props.positions.length) {
        throw new Error(`${id} vertexMeasures length must equal positions length`);
      }
    }
    const formats = {
      positions: 'float32x2',
      segmentIndices: 'uint32',
      tangents: 'float32x2',
      angles: 'float32',
      statuses: 'uint32'
    } as const;
    for (const [name, view] of Object.entries(output) as [
      keyof typeof formats,
      GraphDataView | undefined
    ][]) {
      if (!view) {
        continue;
      }
      if (!(name in formats)) {
        throw new Error(`${id} output.${name} is not a line-locate column`);
      }
      validatePackedView(view, [formats[name]], `${id} output.${name}`);
      if (view.length !== eventCount) {
        throw new Error(`${id} output.${name} must hold ${eventCount} rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [...Object.values(output), props.vertexMeasures],
      [
        props.positions,
        props.pathOffsets,
        props.eventPaths,
        props.eventMeasures,
        props.eventOffsets,
        props.parameters
      ]
    );
  }

  /** Returns the measure prefix node, the locate node and the scatter node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.pathOffsets,
      props.eventPaths,
      props.eventMeasures,
      props.eventOffsets,
      props.parameters,
      props.vertexMeasures,
      ...Object.values(output)
    ]);
    const eventCount = props.eventPaths.length;
    const vertexMeasures =
      props.vertexMeasures ??
      createTransientView(graph, `${id}-vertex-measures`, 'float32', props.positions.length);
    const results = createTransientView(
      graph,
      `${id}-results`,
      'uint32',
      eventCount * LINE_LOCATE_RESULT_STRIDE
    );
    return [
      createPathPrefixNode<Parameters>(graph, {
        id: `${id}-vertex-measures`,
        operation: OPERATION,
        positions: props.positions,
        pathOffsets: props.pathOffsets,
        coordinateSystem: 'planar',
        radius: 1,
        rowMeasures: vertexMeasures
      }),
      createLineLocateNode<Parameters>(graph, {
        id: `${id}-locate`,
        operation: OPERATION,
        positions: props.positions,
        pathOffsets: props.pathOffsets,
        vertexMeasures,
        eventPaths: props.eventPaths,
        eventMeasures: props.eventMeasures,
        eventOffsets: props.eventOffsets,
        parameters: props.parameters,
        measureMode: this.measureMode,
        results
      }),
      ...createPackedScatterNodes<Parameters>(graph, {
        id: `${id}-output`,
        operation: OPERATION,
        rowCount: eventCount,
        stride: LINE_LOCATE_RESULT_STRIDE,
        results,
        columns: [
          {
            name: 'positions',
            view: output.positions,
            type: 'f32',
            statement: `positions[positionsOffset + 2u * index] = bitcast<f32>(results[base]);
  positions[positionsOffset + 2u * index + 1u] = bitcast<f32>(results[base + 1u]);`
          },
          {
            name: 'segmentIndices',
            view: output.segmentIndices,
            type: 'u32',
            statement: 'segmentIndices[segmentIndicesOffset + index] = results[base + 2u];'
          },
          {
            name: 'tangents',
            view: output.tangents,
            type: 'f32',
            statement: `tangents[tangentsOffset + 2u * index] = bitcast<f32>(results[base + 3u]);
  tangents[tangentsOffset + 2u * index + 1u] = bitcast<f32>(results[base + 4u]);`
          },
          {
            name: 'angles',
            view: output.angles,
            type: 'f32',
            statement: `{
    let tangent = vec2<f32>(bitcast<f32>(results[base + 3u]), bitcast<f32>(results[base + 4u]));
    // atan2(0, 0) is undefined in WGSL: degenerate tangents report angle 0.
    angles[anglesOffset + index] = select(
      atan2(tangent.y, tangent.x) * 57.2957795130823,
      0.0,
      tangent.x == 0.0 && tangent.y == 0.0
    );
  }`
          },
          {
            name: 'statuses',
            view: output.statuses,
            type: 'u32',
            statement: 'statuses[statusesOffset + index] = results[base + 5u];'
          }
        ]
      })
    ];
  }
}
