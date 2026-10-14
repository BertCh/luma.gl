// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  getStorageBindingElementLimit,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getGPUSphereRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {GPULineGeometryFields} from '../../analysis-core/contracts/index';
import {createLinePrefixNode} from '../line-segmentize/line-segmentize-kernels';
import {
  createLineLocateNode,
  createPackedScatterNodes,
  LINE_LOCATE_RESULT_STRIDE
} from './linear-referencing-kernels';

const OPERATION = 'GPULineLocate';

/**
 * Largest `eventLineIndices` row count of {@link GPULineLocate} on `device`: the packed result
 * table holds 6 `uint32` words (24 bytes) per event in one storage binding, so 5592405 on a default
 * device (128 MiB bindings).
 */
export function getGPULineLocateMaximumEventCount(device: Device): number {
  return getStorageBindingElementLimit(device, LINE_LOCATE_RESULT_STRIDE * 4);
}

/** Number of float32 elements in a `GPULineLocate` parameter buffer. */
export const GPU_LINE_LOCATE_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame parameters of `GPULineLocate`. */
export type GPULineLocateSettings = {
  /** Multiplies every event measure before locating. Default 1. */
  measureScale?: number;
  /** Added to every scaled event measure, for example elapsed distance in an animation. Default 0. */
  measureOffset?: number;
};

/**
 * Packs `GPULineLocate` parameters into the 4-element float32 layout
 * `[measureScale, measureOffset, 0, 0]`.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is not finite or `target` is too short.
 */
export function getGPULineLocateParameterValues(
  settings: GPULineLocateSettings,
  target: Float32Array = new Float32Array(GPU_LINE_LOCATE_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_LINE_LOCATE_PARAMETER_LENGTH) {
    throw new Error(`Line locate target must hold ${GPU_LINE_LOCATE_PARAMETER_LENGTH} elements`);
  }
  const measureScale = settings.measureScale ?? 1;
  const measureOffset = settings.measureOffset ?? 0;
  if (!Number.isFinite(measureScale) || !Number.isFinite(measureOffset)) {
    throw new Error('Line locate measureScale and measureOffset must be finite');
  }
  target.set([measureScale, measureOffset, 0, 0]);
  return target;
}

/** Kernel model of {@link GPULineLocate}, derived from its `spatialContext` (not a prop). */
export type GPULineLocateCoordinateSystem = 'planar' | 'spherical';

/** Per-event outputs of {@link GPULineLocate}; `eventLineIndices.length` rows each. */
export type GPULineLocateOutput = {
  /** Located (and laterally offset) positions; NaN for invalid events. */
  positions: GraphDataView<'float32x2'>;
  /** Segment index within the line, or `0xffffffff` for invalid events. */
  segmentIndices?: GraphDataView<'uint32'>;
  /**
   * Unit direction of the segment (`(east, north)` compass components at the located point for
   * `'spherical'`), `(0, 0)` on single-vertex lines and zero-length segments. */
  tangents?: GraphDataView<'float32x2'>;
  /**
   * Segment direction in degrees counter-clockwise from the +x axis (deck.gl `getAngle`
   * convention for planar or Web Mercator-like coordinates); `0` where the tangent is `(0, 0)`.
   */
  angles?: GraphDataView<'float32'>;
  /** `GPU_LINE_LOCATE_STATUS` code per event. */
  statuses?: GraphDataView<'uint32'>;
  /** Optional per-vertex cumulative line measure, `positions.length` rows. */
  vertexMeasures?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPULineLocate}.
 *
 * Per-frame (no recompile): the contents of `parameters` and of every input buffer. Compile-time:
 * view lengths, the line and event counts, `measureMode`, and which optional views are present.
 */
export type GPULineLocateProps = GPULineGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'line-locate'`. */
  id?: string;
  /**
   * Planar (default): Euclidean segments in position units. `{coordinateSpace: 'longitude-latitude',
   * metric: 'great-circle'}`: positions are longitude/latitude degrees, segments are great-circle
   * arcs, and measures, distances and offsets are in the context unit on its `sphereRadius`
   * (meters on the mean Earth radius by default). Replaces the former `coordinateSystem` and
   * `sphereRadius` props.
   */
  spatialContext?: GPUSpatialContext;
  /** Line index of each event, at most {@link getGPULineLocateMaximumEventCount} rows (5592405 on a default device). */
  eventLineIndices: GraphDataView<'uint32'>;
  /** Measure of each event: a distance from the line start, or a fraction of the line length. */
  eventMeasures: GraphDataView<'float32'>;
  /** Optional lateral offset of each event, positive to the left of the line direction. */
  eventOffsets?: GraphDataView<'float32'>;
  /** `'distance'` (default) or `'fraction'` of each line's length (turf `along` vs. `ST_LineInterpolatePoint`). */
  measureMode?: 'distance' | 'fraction';
  /**
   * Optional per-frame float32 view of {@link GPU_LINE_LOCATE_PARAMETER_LENGTH} elements written
   * with `getGPULineLocateParameterValues`; transforms every measure as `measure * scale + offset`.
   */
  parameters?: GraphDataView<'float32'>;
  /** Per-event outputs. */
  output: GPULineLocateOutput;
};

/**
 * Places events along lines by measure (turf `along`, PostGIS `ST_LineInterpolatePoint` and
 * `ST_LocateAlong` with offset): route events, mile markers, vehicles moving along routes.
 *
 * A per-line sequential prefix computes cumulative vertex measures (Neumaier-compensated,
 * deterministic). Each event clamps its measure into `[0, lineLength]` (status `clamped`), finds
 * the segment with an upper-bound binary search over the measures (so a measure that lands on a
 * vertex continues on the following segment, and zero-length segments are skipped), and
 * interpolates by measure. Lateral offsets move the point along the segment's left normal.
 * Events on empty or out-of-range lines are `invalid` with NaN positions.
 *
 * A great-circle `spatialContext` measures in sphere-radius units (meters by default) and places
 * events on great-circle arcs (Geod `npts`, `ST_LineInterpolatePoint` on a sphere). Tangents are
 * the compass direction of the arc at the located point as `(east, north)`, offsets move along the
 * left compass bearing, and `angles` follow the same CCW-from-east convention. `measureMode:
 * 'fraction'` is shapely `line_interpolate_point(normalized=True)`. Precision is f32 (about 1 m).
 *
 * Planar by default: measures are Euclidean in position units.
 */
export class GPULineLocate implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULineLocateProps;
  /** Resolved measure mode. */
  readonly measureMode: 'distance' | 'fraction';
  /** Resolved spatial context (planar unless given). */
  readonly spatialContext: GPUSpatialContext;
  /** Kernel model derived from `spatialContext`; not a prop. */
  private readonly coordinateSystem: GPULineLocateCoordinateSystem;
  /** Resolved sphere radius in the context's length unit (1 for planar). */
  private readonly sphereRadius: number;

  constructor(props: GPULineLocateProps) {
    this.id = props.id ?? 'line-locate';
    this.props = props;
    this.measureMode = props.measureMode ?? 'distance';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
    );
    this.coordinateSystem =
      this.spatialContext.coordinateSpace === 'longitude-latitude' ? 'spherical' : 'planar';
    this.sphereRadius =
      this.coordinateSystem === 'spherical' ? getGPUSphereRadius(this.spatialContext) : 1;
    const {id} = this;
    const {output} = props;
    if (this.measureMode !== 'distance' && this.measureMode !== 'fraction') {
      throw new Error(`${id} measureMode must be 'distance' or 'fraction'`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    validatePackedView(props.lineOffsets, ['uint32'], `${id} lineOffsets`);
    if (props.lineOffsets.length < 2) {
      throw new Error(`${id} lineOffsets needs two rows`);
    }
    validatePackedView(props.eventLineIndices, ['uint32'], `${id} eventLineIndices`);
    const eventCount = props.eventLineIndices.length;
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
          throw new Error(`${id} ${name} length must equal eventLineIndices length`);
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
    const formats = {
      positions: 'float32x2',
      segmentIndices: 'uint32',
      tangents: 'float32x2',
      angles: 'float32',
      statuses: 'uint32',
      vertexMeasures: 'float32'
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
      const length = name === 'vertexMeasures' ? props.positions.length : eventCount;
      if (view.length !== length) {
        throw new Error(`${id} output.${name} must hold ${length} rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(id, Object.values(output), [
      props.positions,
      props.lineOffsets,
      props.eventLineIndices,
      props.eventMeasures,
      props.eventOffsets,
      props.parameters
    ]);
  }

  /** Returns the measure prefix node, the locate node and the scatter node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.lineOffsets,
      props.eventLineIndices,
      props.eventMeasures,
      props.eventOffsets,
      props.parameters,
      ...Object.values(output)
    ]);
    const eventCount = props.eventLineIndices.length;
    validateMaximumCount(
      id,
      'eventLineIndices rows',
      eventCount,
      getGPULineLocateMaximumEventCount(graph.device)
    );
    const vertexMeasures =
      output.vertexMeasures ??
      createTransientView(graph, `${id}-vertex-measures`, 'float32', props.positions.length);
    const results = createTransientView(
      graph,
      `${id}-results`,
      'uint32',
      eventCount * LINE_LOCATE_RESULT_STRIDE
    );
    return [
      createLinePrefixNode<Parameters>(graph, {
        id: `${id}-vertex-measures`,
        operation: OPERATION,
        positions: props.positions,
        lineOffsets: props.lineOffsets,
        coordinateSystem: this.coordinateSystem,
        radius: this.coordinateSystem === 'spherical' ? this.sphereRadius : 1,
        rowMeasures: vertexMeasures
      }),
      createLineLocateNode<Parameters>(graph, {
        id: `${id}-locate`,
        operation: OPERATION,
        positions: props.positions,
        lineOffsets: props.lineOffsets,
        vertexMeasures,
        eventLineIndices: props.eventLineIndices,
        eventMeasures: props.eventMeasures,
        eventOffsets: props.eventOffsets,
        parameters: props.parameters,
        measureMode: this.measureMode,
        coordinateSystem: this.coordinateSystem,
        sphereRadius: this.sphereRadius,
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
