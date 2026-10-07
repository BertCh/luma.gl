// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUSegmentRingAssembly,
  type GPUSegmentRingAssemblyOutput
} from '../../gpu-spatial-analysis/ring-assembly/index';
import {GPUIsobands} from './gpu-isobands';

/**
 * Caller-owned outputs of {@link GPUIsobandRings}: every output of `GPUSegmentRingAssembly`
 * (`ringOffsets`, `positions`, `count`, `overflow`, optional `ringAreas`, `ringIsHole`,
 * `ringShells`, `polygons`, ...) plus the band of each ring and the boundary edge counters.
 *
 * `ringGroups` receives the band index of each ring, and holes only attach to shells of their own
 * band. Without `ringGroups` rings are still band-pure.
 */
export type GPUIsobandRingsOutput = GPUSegmentRingAssemblyOutput & {
  /** Optional one-row scalar receiving the number of band boundary edges before ring assembly. */
  edgeCount?: GraphDataView<'uint32'>;
  /**
   * Optional one-row scalar receiving 1 when more boundary edges existed than `edgeCapacity`. The
   * rings are then incomplete (open chains are dropped), so raise the capacity.
   */
  edgeOverflow?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUIsobandRings}. The raster inputs match `GPUIsobandsProps`; every
 * per-frame value (breaks, parameters, band window) stays a buffer write.
 */
export type GPUIsobandRingsProps = {
  /** Prefix for generated node IDs. Defaults to `'isoband-rings'`. */
  id?: string;
  /** Raster width in samples, at least 2. */
  width: number;
  /** Raster height in samples, at least 2. */
  height: number;
  /** Packed row-major float32 raster. */
  values: GraphDataView<'float32'>;
  /** Optional row-major `uint32` validity; zero marks a nodata sample. */
  validity?: GraphDataView<'uint32'>;
  /** Optional finite nodata sentinel compared exactly against sample values. */
  noDataValue?: number;
  /** Break values, see `GPUIsobandsProps.breaks`. */
  breaks: GraphDataView<'float32'>;
  /** Per-frame parameters written with `getGPUIsobandsParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /**
   * Compile-time capacity of the intermediate band boundary edge list. A ring of `n` vertices
   * needs `n` edges; budget the total vertex count of every ring of every band.
   */
  edgeCapacity: number;
  /**
   * Vertex matching distance in world units, see `GPUSegmentRingAssemblyProps.vertexTolerance`.
   * Boundary vertices of neighbouring cells are bit-identical, so any value well below the
   * distance between distinct vertices works. Defaults to 1e-4 of the world unit.
   */
  vertexTolerance?: number;
  /** Split rings that touch themselves at a vertex, see `GPUSegmentRingAssemblyProps`. Defaults to true. */
  splitTouchingRings?: boolean;
  /** Caller-owned outputs. */
  output: GPUIsobandRingsOutput;
};

/**
 * Closed polygon rings of every isoband: shells and holes in GeoArrow layout.
 *
 * Composes `GPUIsobands` (band boundary edges: marching-squares level segments and cell border
 * edges that no neighbouring cell cancels) with `GPUSegmentRingAssembly` (edges chained into closed
 * rings, `ringGroups` = band, holes assigned to their innermost shell of the same band,
 * optional `polygons` regrouped as shell plus holes). Rings follow the segment direction, so shells
 * are counter-clockwise and holes clockwise; the band region lies on the left. Band boundaries are
 * bit-identical to `GPUIsolines` of the same breaks. A band region that touches itself at a
 * single vertex (a saddle with the joined centre rule) is split into separate rings when
 * `splitTouchingRings` is on. Cells with a nodata corner emit no edges, so their border with valid
 * cells becomes part of a ring.
 *
 * Changing breaks, `breakCount`, the extent or the band window never recompiles the graph. The
 * edge and ring capacities are compile-time; the `overflow` and `edgeOverflow` words say when they
 * were too small.
 */
export class GPUIsobandRings implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUIsobandRingsProps;

  constructor(props: GPUIsobandRingsProps) {
    this.id = props.id ?? 'isoband-rings';
    this.props = props;
    if (!Number.isInteger(props.edgeCapacity) || props.edgeCapacity < 1) {
      throw new Error(`${this.id} edgeCapacity must be a positive integer`);
    }
  }

  /** Returns the isoband edge nodes followed by the ring assembly nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const edges = createTransientView(graph, `${id}-edges`, 'float32x4', props.edgeCapacity);
    const edgeBands = createTransientView(graph, `${id}-edge-bands`, 'uint32', props.edgeCapacity);
    const edgeCount =
      output.edgeCount ?? createTransientView(graph, `${id}-edge-count`, 'uint32', 1);
    const edgeOverflow =
      output.edgeOverflow ?? createTransientView(graph, `${id}-edge-overflow`, 'uint32', 1);
    const isobands = new GPUIsobands({
      id: `${id}-isobands`,
      width: props.width,
      height: props.height,
      values: props.values,
      validity: props.validity,
      noDataValue: props.noDataValue,
      breaks: props.breaks,
      parameters: props.parameters,
      output: {edges, edgeBands, edgeCount, edgeOverflow}
    });
    const {edgeCount: _edgeCount, edgeOverflow: _edgeOverflow, ...assemblyOutput} = output;
    const ringAssembly = new GPUSegmentRingAssembly({
      id: `${id}-assembly`,
      endpoints: edges,
      count: edgeCount,
      groups: edgeBands,
      vertexTolerance: props.vertexTolerance ?? 1e-4,
      interiorSide: 'left',
      geographic: false,
      cancelOpposingSegments: true,
      splitTouchingRings: props.splitTouchingRings,
      output: assemblyOutput
    });
    return [...isobands.getCommandNodes(graph), ...ringAssembly.getCommandNodes(graph)];
  }
}
