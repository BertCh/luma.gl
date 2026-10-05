// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createRecipeTransientView} from '../cost-distance/raster-grid-utils';
import {validateRasterIterations} from '../cost-distance/raster-relaxation';
import {
  createMapGraphFillNode,
  createMapGraphKernelNode,
  type MapGraphKernelBinding
} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {
  validateTerrainBuffersDistinct,
  validateTerrainGrid
} from '../terrain-analysis/terrain-analysis-utils';
import {
  createTerrainFlowPointerNodes,
  getTerrainFlowTracingWGSL,
  validateTerrainTracingCellView,
  validateTerrainTracingConvergedView
} from './terrain-flow-tracing';

const OPERATION = 'GPUTerrainWatersheds';

/** Label of invalid cells, and of cells that drain to no pour point. */
export const GPU_TERRAIN_WATERSHED_NONE = 0xffffffff;

/**
 * Properties for {@link GPUTerrainWatersheds}.
 *
 * Compile-time: `width`, `height`, `maxIterations`, and the presence of `pourPoints`, `converged`.
 * Per-frame: the contents of `flowDirections` and `pourPoints`. The recipe works on D8 receiver
 * indices only, so it has no cell-size model (`uniform`, `web-mercator` and `geographic` spacing
 * give the same labels).
 *
 * Output aliasing: outputs never share a buffer with an input or with each other.
 */
export type GPUTerrainWatershedsProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-watersheds'`. Compile-time. */
  id?: string;
  /** Grid width in cells. Compile-time. */
  width: number;
  /** Grid height in cells. Compile-time. */
  height: number;
  /**
   * ESRI D8 codes, one per cell: 1 east, 2 south-east, 4 south, 8 south-west, 16 west, 32
   * north-west, 64 north, 128 north-east; 0 terminal; `0xffffffff` invalid. Any other code, and
   * any code pointing outside the grid, is treated as terminal. Typically
   * `GPUTerrainFlow.flowDirections`. Per-frame.
   */
  flowDirections: GraphDataView<'uint32'>;
  /**
   * Optional pour points as cell indices (`row * width + column`). Indices outside the grid and
   * indices of invalid cells are ignored. When several points share a cell the lowest point index
   * wins. Length is compile-time, contents per-frame. Without pour points the recipe labels
   * drainage basins.
   */
  pourPoints?: GraphDataView<'uint32'>;
  /**
   * With pour points: the index (into `pourPoints`) of the nearest pour point downstream of each
   * cell, inclusive, or `GPU_TERRAIN_WATERSHED_NONE` when the path reaches none. Without pour points:
   * the cell index of the terminal cell (outlet or pit) the cell drains to. Invalid cells are always
   * `GPU_TERRAIN_WATERSHED_NONE`.
   */
  labels: GraphDataView<'uint32'>;
  /** One row set to 1 when pointer jumping reached its fixpoint. Presence is compile-time. */
  converged?: GraphDataView<'uint32'>;
  /**
   * Maximum gated pointer-jumping rounds in `[1, 1024]`. Defaults to 32: paths of up to 2^32 cells
   * need at most 32 doubling rounds. Compile-time.
   */
  maxIterations?: number;
};

/**
 * Watershed delineation on one raster tile from D8 flow directions.
 *
 * Every cell follows its D8 receivers to a stop cell: a pour point when `pourPoints` is given, the
 * terminal cell otherwise. The path is resolved with GPU pointer jumping (in-place
 * `pointer[c] = pointer[pointer[c]]` rounds), the same technique as parallel list ranking
 * (Wyllie 1979). The fixpoint is unique, so labels never depend on thread scheduling. Pour points
 * are marked with `atomicMin` so that duplicates resolve to the lowest point index. Nested pour
 * points give nested watersheds: each cell takes its nearest downstream pour point, and cells that
 * drain past all pour points are `GPU_TERRAIN_WATERSHED_NONE`.
 *
 * Without pour points the labels are drainage basins (the outlet or pit cell index), as in the
 * basin step of standard hydrologic toolboxes. Receiver cycles, possible only in caller-supplied
 * directions, leave `converged` at 0.
 *
 * Compose it with {@link GPUTerrainFlow} through `flowDirections`; grids imported from TauDEM or
 * Whitebox work as well. Results are tile-local: watersheds do not continue across tile edges.
 */
export class GPUTerrainWatersheds implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'terrain-watersheds';
  /** Validated properties. */
  readonly props: GPUTerrainWatershedsProps;

  constructor(props: GPUTerrainWatershedsProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const cellCount = validateTerrainGrid(id, props.width, props.height);
    validateRasterIterations(id, 'maxIterations', props.maxIterations ?? 32);
    validateTerrainTracingCellView(id, 'flowDirections', props.flowDirections, 'uint32', cellCount);
    validateTerrainTracingCellView(id, 'labels', props.labels, 'uint32', cellCount);
    if (props.pourPoints) {
      validatePackedUint32View(props.pourPoints, `${id} pourPoints`);
    }
    validateTerrainTracingConvergedView(id, props.converged);
    const outputs = [props.labels, props.converged];
    validateGraphOutputsDisjointFromInputs(id, outputs, [props.flowDirections, props.pourPoints]);
    validateTerrainBuffersDistinct(id, outputs, []);
  }

  /** Returns marker, pointer-jumping and labeling nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.flowDirections,
      props.pourPoints,
      props.labels,
      props.converged
    ]);
    const cellCount = width * height;
    const pointer = createRecipeTransientView(
      graph,
      'GPUTerrainWatersheds',
      id,
      `${id}-pointer`,
      'uint32',
      cellCount
    );
    const nodes: GPUCommandNode<Parameters>[] = [];
    let markers: GraphDataView<'uint32'> | undefined;
    if (props.pourPoints) {
      markers = createRecipeTransientView(
        graph,
        'GPUTerrainWatersheds',
        id,
        `${id}-markers`,
        'uint32',
        cellCount
      );
      nodes.push(
        createMapGraphFillNode<Parameters>(graph, {
          id: `${id}-markers-init`,
          operation: OPERATION,
          view: markers,
          type: 'u32',
          value: '0xffffffffu'
        }),
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-markers-mark`,
          operation: OPERATION,
          variant: 'mark-pour-points',
          bindings: [
            {name: 'pourPoints', view: props.pourPoints, type: 'u32', access: 'read'},
            {name: 'flowDirections', view: props.flowDirections, type: 'u32', access: 'read'},
            {name: 'markers', view: markers, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: props.pourPoints.length,
          declarations: `const CELL_COUNT: u32 = ${cellCount}u;`,
          body: `let cell = pourPoints[pourPointsOffset + index];
  if (cell < CELL_COUNT && flowDirections[flowDirectionsOffset + cell] != 0xffffffffu) {
    atomicMin(&markers[markersOffset + cell], index);
  }`
        })
      );
    }
    nodes.push(
      ...createTerrainFlowPointerNodes(graph, {
        id,
        recipeId: id,
        operation: OPERATION,
        width,
        height,
        maxIterations: props.maxIterations ?? 32,
        flowDirections: props.flowDirections,
        pointer,
        stops: markers ? {kind: 'markers', markers} : {kind: 'none'},
        converged: props.converged
      })
    );
    const bindings: MapGraphKernelBinding[] = [
      {name: 'flowDirections', view: props.flowDirections, type: 'u32', access: 'read'},
      {name: 'pointer', view: pointer, type: 'u32', access: 'read'},
      ...(markers
        ? [{name: 'markers', view: markers, type: 'u32' as const, access: 'read' as const}]
        : []),
      {name: 'labels', view: props.labels, type: 'u32', access: 'read_write'}
    ];
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-labels`,
        operation: OPERATION,
        variant: markers ? 'label-pour-points' : 'label-basins',
        bindings,
        invocationCount: cellCount,
        declarations: getTerrainFlowTracingWGSL(width, height),
        body: `let reached = pointer[pointerOffset + index];
  var label = NO_CELL;
  if (flowDirections[flowDirectionsOffset + index] != NO_CELL) {
    ${
      markers
        ? 'label = markers[markersOffset + reached];'
        : 'if (flowDirections[flowDirectionsOffset + reached] != NO_CELL) { label = reached; }'
    }
  }
  labels[labelsOffset + index] = label;`
      })
    );
    return nodes;
  }
}
