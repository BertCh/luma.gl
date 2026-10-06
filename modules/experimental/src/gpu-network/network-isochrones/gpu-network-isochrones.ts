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
import {createFillNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {GPUIsobands, GPU_ISOBANDS_PARAMETER_LENGTH} from '../../gpu-raster/isolines/index';
import type {GPUIsobandsOutput} from '../../gpu-raster/isolines/index';
import {GPUCellAggregation} from '../../gpu-spatial-analysis/cell-aggregation/index';
import type {GPUCellTable} from '../../gpu-spatial-analysis/cell-aggregation/index';
import {GPUPointToCell} from '../../gpu-spatial-analysis/cell-indexing/index';
import {GPUCellSetOutline} from '../../gpu-spatial-analysis/cell-set-outline/index';
import type {
  GPUCellSetOutlineOutput,
  GPUCellSetOutlineRings
} from '../../gpu-spatial-analysis/cell-set-outline/index';
import {GPUNetworkReachability} from '../network-reachability/gpu-network-reachability';
import {GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH} from './network-isochrones-parameters';
import {
  createIsochronesBandParametersNode,
  createIsochronesDecodeNode,
  createIsochronesNodeMaskNode,
  createIsochronesSplatNode,
  getIsochronesEncodedInitialValue,
  type GPUNetworkIsochronesMode
} from './network-isochrones-passes';

const DEFAULT_ID = 'network-isochrones';
const DEFAULT_MAXIMUM_BUFFER_PIXELS = 6;
const MAXIMUM_BUFFER_PIXELS_LIMIT = 16;
const DEFAULT_MAXIMUM_SAMPLES_PER_EDGE = 64;
const MAXIMUM_SAMPLES_PER_EDGE_LIMIT = 1024;
const DEFAULT_UNREACHED_COST = 1e30;

/** Raster path of {@link GPUNetworkIsochrones}: splat edge costs to a raster, then isobands. */
export type GPUNetworkIsochroneRaster = {
  /** Raster width in pixels, at least 2. */
  width: number;
  /** Raster height in pixels, at least 2. */
  height: number;
  /**
   * `'min'` (default) keeps the smallest cost that lands on a pixel, the usual isochrone: a pixel
   * is as close as the closest street sample. `'max'` keeps the largest, a conservative surface
   * where a pixel is within a cost only when every street sample touching it is.
   */
  mode?: GPUNetworkIsochronesMode;
  /**
   * Compile-time cap in pixels of the half-window that one edge sample marks per axis, 1 to 16.
   * Defaults to 6. A buffer radius wider than this many pixels is truncated.
   */
  maximumBufferPixels?: number;
  /**
   * Compile-time cap on the samples taken along one edge, 2 to 1024. Defaults to 64. An edge longer
   * than `maximumSamplesPerEdge * bufferRadius / 2` leaves gaps, so raise it or use a larger buffer.
   */
  maximumSamplesPerEdge?: number;
  /**
   * Value of pixels that no edge sample reached. Defaults to 1e30, a finite number far above any
   * break, so unreached space lands in the last band. Pixels next to the network interpolate
   * toward it, which insets the outer contour by up to one pixel.
   */
  unreachedCost?: number;
  /**
   * Caller-owned outputs of {@link GPUIsobands} (band classes or triangles), plus the optional
   * `values` raster. Band `k` holds cost in `[breaks[k - 1], breaks[k])`.
   */
  output: GPUIsobandsOutput & {
    /** Optional `width * height` float32 cost raster, row 0 at `minY`. */
    values?: GraphDataView<'float32'>;
  };
};

/** Cell-outline path of {@link GPUNetworkIsochrones}: outline the cells that contain reached nodes. */
export type GPUNetworkIsochroneCellOutline = {
  /** Cell family of the outline. */
  family: 'h3' | 'quadbin';
  /** Cell resolution: Quadbin 0 to 26, H3 0 to 15. */
  resolution: number;
  /**
   * Optional caller-owned table that receives the reached cells (sorted, distinct). Provide it to
   * read its `count` and `overflow`. Defaults to a graph transient of `tableCapacity` rows.
   */
  table?: GPUCellTable;
  /** Row capacity of the transient table when `table` is omitted. Defaults to the node count. */
  tableCapacity?: number;
  /** Caller-owned boundary segments of the reached cells. */
  output: GPUCellSetOutlineOutput;
  /**
   * Optional ring assembly of the boundary segments (`GPUSegmentRingAssembly`): closed isochrone
   * rings as GeoArrow-style offsets plus positions, shells and holes classified, holes assigned to
   * shells. Rings are in longitude/latitude degrees. One run covers one cost limit
   * (`cellCostLimit`); several bands need one run each.
   */
  rings?: GPUCellSetOutlineRings;
};

/**
 * Properties for {@link GPUNetworkIsochrones}.
 *
 * Per-frame (no recompile): the contents of the CSR, node positions, costs or sources, `breaks` and
 * `parameters`. Compile-time: node, edge, pixel and break capacities, raster and cell options.
 */
export type GPUNetworkIsochronesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-isochrones'`. */
  id?: string;
  /** Forward CSR row offsets with `nodeCount + 1` rows. */
  offsets: GraphDataView<'uint32'>;
  /** CSR destination node per edge. Indices `>= nodeCount` are ignored. */
  neighbors: GraphDataView<'uint32'>;
  /** Non-negative travel cost per edge. Negative or NaN edges are skipped. */
  weights: GraphDataView<'float32'>;
  /**
   * Node positions. The raster path uses them in the units of the extent (projected meters give an
   * isotropic buffer; longitude/latitude degrees give an anisotropic one). The cell path needs
   * longitude/latitude degrees.
   */
  nodePositions: GraphDataView<'float32x2'>;
  /**
   * Per-node travel cost. An input (for example `GPUNetworkServiceAreas.costs`) unless `sources`
   * is given, in which case this contributor runs `GPUNetworkReachability` and writes it. Unreached
   * nodes are `+Infinity`. Its length defines the node count.
   */
  costs: GraphDataView<'float32'>;
  /** Optional facility nodes that start a multi-source search writing `costs`. */
  sources?: GraphDataView<'uint32'>;
  /** Optional per-source starting cost. */
  sourceCosts?: GraphDataView<'float32'>;
  /** Optional one-row active source count. */
  sourceCount?: GraphDataView<'uint32'>;
  /** Optional one-row search cost cutoff. */
  costLimit?: GraphDataView<'float32'>;
  /** Compile-time relaxation rounds of the search (with `sources`). Defaults to 64. */
  maxIterations?: number;
  /**
   * Ascending cost thresholds (isochrone times); the length is the maximum break count. The first
   * `breakCount` entries of `parameters` are active.
   */
  breaks: GraphDataView<'float32'>;
  /** Per-frame float32 parameters, written by `getGPUNetworkIsochroneParameterValues` (12 elements). */
  parameters: GraphDataView<'float32'>;
  /** Raster path. At least one of `raster` and `cellOutline` is required. */
  raster?: GPUNetworkIsochroneRaster;
  /** Cell-outline path. */
  cellOutline?: GPUNetworkIsochroneCellOutline;
};

/**
 * Network isochrone polygons: edge-interpolated travel costs splatted to a raster and contoured
 * into bands, or the outline of the cells that contain reached nodes.
 *
 * The contributor composes existing parts. Node costs come from the caller (any producer such as
 * `GPUNetworkServiceAreas`, whose `costs` are the multi-source minimum used here) or from a
 * `GPUNetworkReachability` search over `sources`. A splat kernel then samples every edge at a
 * bounded number of points, interpolates the cost linearly from the source node along the edge
 * weight, and writes the minimum (or maximum) cost to the pixels within the walking buffer using
 * integer atomics on order-preserving float bits. Results do not depend on thread order. The
 * raster feeds `GPUIsobands`, so band boundaries are the isochrone polygons as triangles per band.
 *
 * The raster path is planar in the units of `nodePositions`; the pixel grid has resolution
 * `(maxX - minX) / width`, so the polygons are accurate to about a pixel and thinner than a pixel
 * features need a buffer. Costs of pixels in the buffer are the sample cost plus
 * `walkCostPerUnit` times the distance, which gives the standard "street cost plus walk" isochrone.
 * Edges are straight segments between node positions, so curved street geometry is approximated.
 *
 * The cell path outlines cells holding at least one reached node (cost at or below
 * `cellCostLimit`) with `GPUCellSetOutline`; `cellOutline.rings` chains the outline into closed
 * isochrone rings with shells and holes. It samples nodes only, so on sparse networks long
 * edges leave their cells out; use the raster path for smooth polygons.
 *
 * Nothing here compiles, submits or reads back.
 */
export class GPUNetworkIsochrones implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkIsochronesProps;
  /** Number of network nodes, `costs.length`. */
  readonly nodeCount: number;
  /** Resolved raster accumulation mode, when the raster path is on. */
  readonly mode: GPUNetworkIsochronesMode;

  constructor(props: GPUNetworkIsochronesProps) {
    this.id = props.id ?? DEFAULT_ID;
    this.props = props;
    const {id} = this;
    const {raster, cellOutline} = props;
    if (!raster && !cellOutline) {
      throw new Error(`${id} needs raster or cellOutline`);
    }
    for (const [name, view] of [
      ['offsets', props.offsets],
      ['neighbors', props.neighbors],
      ['weights', props.weights],
      ['nodePositions', props.nodePositions],
      ['costs', props.costs],
      ['breaks', props.breaks],
      ['parameters', props.parameters],
      ['sources', props.sources]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedUint32View(props.offsets, `${id} offsets`);
    validatePackedUint32View(props.neighbors, `${id} neighbors`);
    validatePackedView(props.weights, ['float32'], `${id} weights`);
    validatePackedView(props.nodePositions, ['float32x2'], `${id} nodePositions`);
    validatePackedView(props.costs, ['float32'], `${id} costs`);
    validatePackedView(props.breaks, ['float32'], `${id} breaks`);
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    this.nodeCount = props.costs.length;
    if (this.nodeCount < 1) {
      throw new Error(`${id} costs must contain at least one node`);
    }
    if (props.offsets.length !== this.nodeCount + 1) {
      throw new Error(`${id} offsets must contain one more row than costs`);
    }
    if (props.weights.length !== props.neighbors.length) {
      throw new Error(`${id} weights length must equal neighbors length`);
    }
    if (props.neighbors.length < 1) {
      throw new Error(`${id} needs at least one edge`);
    }
    if (props.nodePositions.length !== this.nodeCount) {
      throw new Error(`${id} nodePositions length must equal the node count`);
    }
    if (props.breaks.length < 1) {
      throw new Error(`${id} breaks must contain at least one row`);
    }
    if (props.parameters.length < GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH} float32 rows`
      );
    }
    if (!props.sources) {
      for (const name of ['sourceCosts', 'sourceCount', 'costLimit'] as const) {
        if (props[name]) {
          throw new Error(`${id} ${name} needs sources`);
        }
      }
    }
    this.mode = raster?.mode ?? 'min';
    if (this.mode !== 'min' && this.mode !== 'max') {
      throw new Error(`${id} raster.mode must be 'min' or 'max'`);
    }
    if (raster) {
      for (const [name, value, minimum, maximum] of [
        ['raster.width', raster.width, 2, 0x7fffffff],
        ['raster.height', raster.height, 2, 0x7fffffff],
        [
          'raster.maximumBufferPixels',
          raster.maximumBufferPixels ?? DEFAULT_MAXIMUM_BUFFER_PIXELS,
          1,
          MAXIMUM_BUFFER_PIXELS_LIMIT
        ],
        [
          'raster.maximumSamplesPerEdge',
          raster.maximumSamplesPerEdge ?? DEFAULT_MAXIMUM_SAMPLES_PER_EDGE,
          2,
          MAXIMUM_SAMPLES_PER_EDGE_LIMIT
        ]
      ] as const) {
        if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
          throw new Error(`${id} ${name} must be an integer in [${minimum}, ${maximum}]`);
        }
      }
      const unreached = raster.unreachedCost ?? DEFAULT_UNREACHED_COST;
      if (!Number.isFinite(unreached) || !(unreached > 0) || unreached >= 3.0e38) {
        throw new Error(`${id} raster.unreachedCost must be a positive finite number below 3e38`);
      }
      const pixelCount = raster.width * raster.height;
      if (!Number.isSafeInteger(pixelCount) || pixelCount > 0x7fffffff) {
        throw new Error(`${id} raster width * height must fit in 31 bits`);
      }
      if (raster.output.values) {
        validatePackedView(raster.output.values, ['float32'], `${id} raster.output.values`);
        if (raster.output.values.length < pixelCount) {
          throw new Error(`${id} raster.output.values must hold width * height rows`);
        }
      }
    }
    if (cellOutline) {
      if (cellOutline.family !== 'h3' && cellOutline.family !== 'quadbin') {
        throw new Error(`${id} cellOutline.family must be 'h3' or 'quadbin'`);
      }
      if (
        cellOutline.tableCapacity !== undefined &&
        (!Number.isSafeInteger(cellOutline.tableCapacity) || cellOutline.tableCapacity < 1)
      ) {
        throw new Error(`${id} cellOutline.tableCapacity must be a positive integer`);
      }
    }
    const inputs = [
      props.offsets,
      props.neighbors,
      props.weights,
      props.nodePositions,
      props.breaks,
      props.parameters,
      props.sources,
      props.sourceCosts,
      props.sourceCount,
      props.costLimit,
      ...(props.sources ? [] : [props.costs])
    ];
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        ...(props.sources ? [props.costs] : []),
        ...Object.values(raster?.output ?? {}),
        ...Object.values(cellOutline?.output ?? {}),
        ...getRingOutputViews(cellOutline),
        ...(cellOutline?.table ? Object.values(cellOutline.table) : [])
      ],
      inputs
    );
  }

  /**
   * Returns the optional search nodes, then for the raster path: band parameters, encoded fill,
   * edge splat, decode and the isobands nodes; for the cell path: node mask, point keying (H3),
   * cell aggregation and the outline nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, mode} = this;
    const {raster, cellOutline} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.offsets,
      props.neighbors,
      props.weights,
      props.nodePositions,
      props.costs,
      props.sources,
      props.sourceCosts,
      props.sourceCount,
      props.costLimit,
      props.breaks,
      props.parameters,
      ...Object.values(raster?.output ?? {}),
      ...Object.values(cellOutline?.output ?? {}),
      ...getRingOutputViews(cellOutline),
      ...(cellOutline?.table ? Object.values(cellOutline.table) : [])
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (props.sources) {
      nodes.push(
        ...new GPUNetworkReachability({
          id: `${id}-search`,
          offsets: props.offsets,
          neighbors: props.neighbors,
          weights: props.weights,
          sources: props.sources,
          sourceCosts: props.sourceCosts,
          sourceCount: props.sourceCount,
          costLimit: props.costLimit,
          maxIterations: props.maxIterations,
          costs: props.costs
        }).getCommandNodes(graph)
      );
    }

    if (raster) {
      const {width, height} = raster;
      const pixelCount = width * height;
      const encoded = createTransientView(graph, `${id}-encoded`, 'uint32', pixelCount);
      const values =
        raster.output.values ?? createTransientView(graph, `${id}-values`, 'float32', pixelCount);
      const bandParameters = createTransientView(
        graph,
        `${id}-band-parameters`,
        'float32',
        GPU_ISOBANDS_PARAMETER_LENGTH
      );
      nodes.push(
        createIsochronesBandParametersNode<Parameters>(graph, {
          id: `${id}-band-parameters`,
          width,
          height,
          parameters: props.parameters,
          bandParameters
        }),
        createFillNode<Parameters>(graph, {
          id: `${id}-encoded-fill`,
          operation: 'GPUNetworkIsochrones',
          view: encoded,
          type: 'u32',
          value: `${getIsochronesEncodedInitialValue(mode)}u`
        }),
        createIsochronesSplatNode<Parameters>(graph, {
          id: `${id}-splat`,
          nodeCount,
          width,
          height,
          mode,
          maximumBufferPixels: raster.maximumBufferPixels ?? DEFAULT_MAXIMUM_BUFFER_PIXELS,
          maximumSamplesPerEdge: raster.maximumSamplesPerEdge ?? DEFAULT_MAXIMUM_SAMPLES_PER_EDGE,
          offsets: props.offsets,
          neighbors: props.neighbors,
          weights: props.weights,
          costs: props.costs,
          nodePositions: props.nodePositions,
          parameters: props.parameters,
          encoded
        }),
        createIsochronesDecodeNode<Parameters>(graph, {
          id: `${id}-decode`,
          mode,
          unreachedCost: raster.unreachedCost ?? DEFAULT_UNREACHED_COST,
          encoded,
          values,
          pixelCount
        }),
        ...new GPUIsobands({
          id: `${id}-isobands`,
          width,
          height,
          values,
          breaks: props.breaks,
          parameters: bandParameters,
          output: {
            bandClasses: raster.output.bandClasses,
            triangles: raster.output.triangles,
            triangleBands: raster.output.triangleBands,
            count: raster.output.count,
            overflow: raster.output.overflow,
            totalCount: raster.output.totalCount,
            vertexCount: raster.output.vertexCount
          }
        }).getCommandNodes(graph)
      );
    }

    if (cellOutline) {
      const mask = createTransientView(graph, `${id}-reached-mask`, 'uint32', nodeCount);
      nodes.push(
        createIsochronesNodeMaskNode<Parameters>(graph, {
          id: `${id}-reached-mask`,
          costs: props.costs,
          parameters: props.parameters,
          mask
        })
      );
      const capacity = cellOutline.tableCapacity ?? nodeCount;
      const table: GPUCellTable = cellOutline.table ?? {
        cells: createTransientView(graph, `${id}-reached-cells`, 'uint32x2', capacity),
        counts: createTransientView(graph, `${id}-reached-counts`, 'uint32', capacity),
        count: createTransientView(graph, `${id}-reached-count`, 'uint32', 1),
        overflow: createTransientView(graph, `${id}-reached-overflow`, 'uint32', 1)
      };
      if (cellOutline.family === 'quadbin') {
        nodes.push(
          ...new GPUCellAggregation({
            id: `${id}-reached-table`,
            family: 'quadbin',
            resolution: cellOutline.resolution,
            positions: props.nodePositions,
            mask,
            output: table
          }).getCommandNodes(graph)
        );
      } else {
        const nodeCells = createTransientView(graph, `${id}-node-cells`, 'uint32x2', nodeCount);
        nodes.push(
          ...new GPUPointToCell({
            id: `${id}-node-cells`,
            family: 'h3',
            resolution: cellOutline.resolution,
            positions: props.nodePositions,
            mask,
            output: {cells: nodeCells}
          }).getCommandNodes(graph),
          ...new GPUCellAggregation({
            id: `${id}-reached-table`,
            family: 'h3',
            resolution: cellOutline.resolution,
            cells: nodeCells,
            output: table
          }).getCommandNodes(graph)
        );
      }
      nodes.push(
        ...new GPUCellSetOutline({
          id: `${id}-outline`,
          family: cellOutline.family,
          cells: table.cells,
          count: table.count,
          output: cellOutline.output,
          rings: cellOutline.rings
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }
}

/** Views of the optional ring output, with the nested polygon layout flattened. */
function getRingOutputViews(cellOutline: GPUNetworkIsochroneCellOutline | undefined) {
  const output = cellOutline?.rings?.output;
  if (!output) {
    return [];
  }
  const {polygons, ...views} = output;
  return [...Object.values(views), ...(polygons ? Object.values(polygons) : [])];
}
