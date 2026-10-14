// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology,
  type GPUNetworkTopology
} from '../../analysis-core/network-topology';
import {
  GPUIsobandRings,
  GPUIsobands,
  GPU_ISOBANDS_PARAMETER_LENGTH
} from '../../gpu-raster/isolines/index';
import type {GPUIsobandRingsOutput, GPUIsobandsOutput} from '../../gpu-raster/isolines/index';
import type {GPURasterMetadata} from '../../gpu-raster/types';
import {GPUCellAggregation} from '../../gpu-spatial-analysis/cell-aggregation/index';
import type {GPUCellTable} from '../../gpu-spatial-analysis/cell-aggregation/index';
import {GPUPointToCell} from '../../gpu-spatial-analysis/cell-indexing/index';
import {GPUCellSetOutline} from '../../gpu-spatial-analysis/cell-set-outline/index';
import type {GPUSegmentRingAssemblyOutput} from '../../analysis-core/ring-assembly/index';
import type {
  GPUCellSetOutlineOutput,
  GPUCellSetOutlineRings
} from '../../gpu-spatial-analysis/cell-set-outline/index';
import {
  resolveGPUSpatialContext,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import {GPUNetworkServiceAreas} from '../network-analysis/gpu-network-service-areas';
import {GPUNetworkReachability} from '../network-reachability/gpu-network-reachability';
import {GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH} from './network-isochrones-parameters';
import {
  createIsochronesBandParametersNode,
  createIsochronesCellFacilityNodes,
  createIsochronesDecodeNode,
  createIsochronesNodeMaskNode,
  createIsochronesSplatNode,
  createIsochronesTriangleFacilityNode,
  getIsochronesEncodedInitialValue,
  GPU_NETWORK_ISOCHRONES_MAXIMUM_RASTER_FACILITIES,
  type GPUNetworkIsochronesMode
} from './network-isochrones-passes';

const DEFAULT_ID = 'network-isochrones';
const DEFAULT_MAXIMUM_BUFFER_PIXELS = 6;
const MAXIMUM_BUFFER_PIXELS_LIMIT = 16;
const DEFAULT_MAXIMUM_SAMPLES_PER_EDGE = 64;
const MAXIMUM_SAMPLES_PER_EDGE_LIMIT = 1024;
const DEFAULT_UNREACHED_COST = 1e30;
/** The raster path is planar; the cell path takes longitude/latitude. */
const ISOCHRONES_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  measure: 'distance'
});

/** Raster path of {@link GPUNetworkIsochrones}: splat edge costs to a raster, then isobands. */
export type GPUNetworkIsochronesRaster = {
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
   * `values` raster. Band `k` holds cost in `[costBreaks[k - 1], costBreaks[k])`.
   */
  output: GPUIsobandsOutput & {
    /** Optional `width * height` float32 cost raster, row 0 at `minY`. */
    values?: GraphDataView<'float32'>;
    /**
     * Optional `width * height` uint32 facility row (index into `startNodes`) of the edge sample that
     * set each pixel's cost, `0xffffffff` for unreached pixels. Needs `assignments`, mode `'min'`
     * and at most 255 start nodes. The facility shares the pixel word with the cost, which then keeps
     * its top 24 bits (relative precision 2^-15, rounded down); ties go to the lowest facility row.
     */
    pixelFacilities?: GraphDataView<'uint32'>;
    /**
     * Optional facility row per isoband triangle (length `triangleBands.length`), taken from the
     * raster sample nearest to the triangle centroid that has a facility; `0xffffffff` past
     * `count`. Needs `pixelFacilities` and the triangle output.
     */
    triangleFacilities?: GraphDataView<'uint32'>;
  };
  /**
   * Optional closed band rings (`GPUIsobandRings`): shells and holes per cost band as GeoArrow
   * offsets, in the raster's units. `rings.output.ringGroups` receives the band of each ring. Bands
   * of several facilities that touch merge into one ring; use the cell path with `byFacility` for
   * rings that never mix facilities.
   */
  rings?: {
    /** Compile-time capacity of the intermediate boundary edge list, see `GPUIsobandRings`. */
    edgeCapacity: number;
    /** Vertex matching distance in raster units, see `GPUIsobandRings`. */
    vertexTolerance?: number;
    /** Caller-owned ring outputs. */
    output: GPUIsobandRingsOutput;
  };
};

/** Cell-outline path of {@link GPUNetworkIsochrones}: outline the cells that contain reached nodes. */
export type GPUNetworkIsochronesCellOutline = {
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
  /**
   * Label every reached cell with the facility (row of `startNodes`) of its cheapest reached node and
   * outline each facility separately: the outline gets the labels as `groups`, so borders between
   * facilities are emitted from both sides, `output.groups` carries the facility of each segment
   * and rings never mix facilities (give `rings.output.ringGroups` to read each ring's facility).
   * Needs `assignments`. Cells are labelled by their lowest-cost node, ties to the lowest facility
   * row. Quadbin cells are keyed through `GPUPointToCell`, like H3.
   */
  byFacility?: boolean;
  /**
   * Optional caller-owned facility row per table row (`table.cells.length` rows, `0xffffffff` past
   * the cell count). Needs `byFacility`. Defaults to a graph transient.
   */
  cellFacilities?: GraphDataView<'uint32'>;
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

/** GPU-written views of {@link GPUNetworkIsochrones}. */
export type GPUNetworkIsochronesOutput = {
  /** Status words of the search. */
  status?: {
    /**
     * Optional one-row flag written by the search (with `startNodes`): 1 when it reached a fixpoint,
     * 0 when `maximumIterations` (or `maximumLabelIterations`) stopped it first. A 0 means `costs`
     * are truncated: nodes past the last round stay unreached or too costly, so every isochrone
     * polygon is smaller than the true one. As with `GPUNetworkReachability` and
     * `GPUNetworkServiceAreas`, omitting it accepts that truncation unchecked; pass it whenever the
     * network diameter is not known to fit `maximumIterations`. Not accepted without `startNodes`
     * (the caller's costs carry their own status).
     */
    converged?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUNetworkIsochrones}.
 *
 * Per-frame (no recompile): the contents of the topology, costs or start nodes, `costBreaks` and
 * `parameters`. Compile-time: node, edge, pixel and break capacities, a numeric `startCount`,
 * raster and cell options.
 *
 * `costs` and `assignments` stay top-level because each is an input when `startNodes` is absent and
 * an output the contributor writes when `startNodes` is given. The search convergence flag is
 * `output.status.converged`.
 */
export type GPUNetworkIsochronesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-isochrones'`. */
  id?: string;
  /**
   * Directed network with node positions. `neighbors` indices `>= nodeCount` are ignored; negative
   * or NaN `weights` are skipped. The raster path uses `nodePositions` in the units of the extent
   * (projected meters give an isotropic buffer; longitude/latitude degrees would give an
   * anisotropic one). The cell path needs longitude/latitude degrees.
   */
  topology: GPUNetworkTopology & {nodePositions: GraphDataView<'float32x2'>};
  /**
   * Optional coordinates of `topology.nodePositions`. The raster path is planar (pixel grid,
   * `bufferRadius` and `walkCostPerUnit` are Euclidean in the units of the extent), so a
   * longitude/latitude context with `raster` throws: project to a planar frame first. The cell
   * path indexes longitude/latitude degrees, so a planar context with `cellOutline` throws.
   * Omitted resolves to native planar for `raster`, or longitude/latitude with metric `'none'`
   * for `cellOutline` alone. With both paths and no context, the raster default applies and
   * cell-outline positions remain unchecked for compatibility. Distance buffers accept an
   * azimuthal-equidistant local frame and reject a Lambert equal-area frame.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Per-node travel cost. An input (for example `GPUNetworkServiceAreasOutput.costs`) unless
   * `startNodes` is given, in which case this contributor runs `GPUNetworkReachability` and writes
   * it. Unreached nodes are `+Infinity`. Its length defines the node count.
   */
  costs: GraphDataView<'float32'>;
  /**
   * Optional start nodes (facilities) that begin a multi-source search writing `costs`. Pass
   * `output.status.converged` to detect a search cut short by `maximumIterations`.
   */
  startNodes?: GraphDataView<'uint32'>;
  /**
   * Optional per-node facility row (index into `startNodes`) of the nearest facility,
   * `GPU_NETWORK_REACHABILITY_NONE` where none reaches it. Like `costs` it is an input (for
   * example `GPUNetworkServiceAreasOutput.assignments`) unless `startNodes` is given, in which case
   * this contributor writes it: the search is then a `GPUNetworkServiceAreas` run
   * (nearest-facility allocation, ties to the lowest row) rather than a plain multi-source
   * `GPUNetworkReachability`, and `costs` is the same minimum cost either way. It enables the
   * facility-labelled outputs (`raster.output.pixelFacilities`, `triangleFacilities`,
   * `cellOutline.byFacility`). Length `costs.length`.
   */
  assignments?: GraphDataView<'uint32'>;
  /** Compile-time label-phase rounds of the allocation (with `assignments`), see `GPUNetworkServiceAreas`. */
  maximumLabelIterations?: number;
  /** Optional per-start initial cost. */
  startCosts?: GraphDataView<'float32'>;
  /**
   * Optional active start count. A number is a compile-time constant in `[0, startNodes.length]`; a
   * one-row view is read per frame. Defaults to every row.
   */
  startCount?: number | GraphDataView<'uint32'>;
  /** Optional one-row search cost cutoff. */
  costLimit?: GraphDataView<'float32'>;
  /** Compile-time relaxation rounds of the search (with `startNodes`). Defaults to 64. */
  maximumIterations?: number;
  /**
   * Compile-time hops one workgroup chains per round of the search (with `startNodes`), forwarded
   * to `GPUNetworkReachability` (default 32) or `GPUNetworkServiceAreas` (default 16). A network
   * whose longest shortest path has `h` hops needs about `ceil(h / localIterations)` rounds.
   */
  localIterations?: number;
  /**
   * Optional GPU-written status of the search (with `startNodes`). Not accepted without
   * `startNodes` (the caller's costs carry their own status).
   */
  output?: GPUNetworkIsochronesOutput;
  /**
   * Ascending cost thresholds (isochrone times); the length is the maximum break count. The first
   * `breakCount` entries of `parameters` are active.
   */
  costBreaks: GraphDataView<'float32'>;
  /** Per-frame float32 parameters, written by `getGPUNetworkIsochronesParameterValues` (12 elements). */
  parameters: GraphDataView<'float32'>;
  /** Raster path. At least one of `raster` and `cellOutline` is required. */
  raster?: GPUNetworkIsochronesRaster;
  /** Cell-outline path. */
  cellOutline?: GPUNetworkIsochronesCellOutline;
};

/**
 * Network isochrone polygons: edge-interpolated travel costs splatted to a raster and contoured
 * into bands, or the outline of the cells that contain reached nodes.
 *
 * The contributor composes existing parts. Node costs come from the caller (any producer such as
 * `GPUNetworkServiceAreas`, whose `costs` are the multi-source minimum used here) or from a
 * `GPUNetworkReachability` search over `startNodes`, whose optional `output.status.converged` flag reports a search
 * cut short by `maximumIterations` (truncated costs, too-small polygons). A splat kernel then samples every edge at a
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
 * An explicit `spatialContext` checks both paths at construction; omitted defaults follow
 * `GPUNetworkIsochronesProps.spatialContext`.
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
  /** Resolved context: native planar for raster, longitude/latitude for cell outlines alone. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUNetworkIsochronesProps) {
    this.id = props.id ?? DEFAULT_ID;
    this.props = props;
    const {id} = this;
    const {raster, cellOutline} = props;
    if (!raster && !cellOutline) {
      throw new Error(`${id} needs raster or cellOutline`);
    }
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext ?? (raster ? undefined : GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT),
      ISOCHRONES_SPATIAL_CONTEXT_SUPPORT
    );
    const coordinateSpace = this.spatialContext.coordinateSpace;
    if (raster && coordinateSpace === 'longitude-latitude') {
      throw new Error(`${id} raster needs planar positions; project to a planar frame first`);
    }
    // Legacy combined outputs without a context leave cell-outline positions unchecked.
    if (cellOutline && coordinateSpace === 'planar' && props.spatialContext) {
      throw new Error(`${id} cellOutline needs longitude-latitude positions`);
    }
    for (const [name, view] of [
      ['costs', props.costs],
      ['costBreaks', props.costBreaks],
      ['parameters', props.parameters],
      ['startNodes', props.startNodes],
      ['assignments', props.assignments]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    const {nodeCount: networkNodeCount, edgeCount} = validateGPUNetworkTopology(
      id,
      props.topology,
      {weights: true, nodePositions: true}
    );
    validatePackedView(props.costs, ['float32'], `${id} costs`);
    validatePackedView(props.costBreaks, ['float32'], `${id} costBreaks`);
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    this.nodeCount = props.costs.length;
    if (this.nodeCount < 1) {
      throw new Error(`${id} costs must contain at least one node`);
    }
    if (networkNodeCount !== this.nodeCount) {
      throw new Error(`${id} topology must contain one node per costs row`);
    }
    if (edgeCount < 1) {
      throw new Error(`${id} needs at least one edge`);
    }
    if (props.costBreaks.length < 1) {
      throw new Error(`${id} costBreaks must contain at least one row`);
    }
    if (props.parameters.length < GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH} float32 rows`
      );
    }
    if (!props.startNodes) {
      for (const name of [
        'startCosts',
        'startCount',
        'costLimit',
        'output',
        'localIterations'
      ] as const) {
        if (props[name] !== undefined) {
          throw new Error(`${id} ${name} needs startNodes`);
        }
      }
    } else {
      validatePackedView(props.startNodes, ['uint32'], `${id} startNodes`);
      if (typeof props.startCount === 'number') {
        validateCapacity(id, 'startCount', props.startCount, 0, props.startNodes.length);
      }
    }
    if (props.output?.status?.converged) {
      validatePackedView(
        props.output.status.converged,
        ['uint32'],
        `${id} output.status.converged`
      );
      if (props.output.status.converged.length !== 1) {
        throw new Error(`${id} output.status.converged must contain exactly one row`);
      }
    }
    if (props.assignments) {
      validatePackedView(props.assignments, ['uint32'], `${id} assignments`);
      if (props.assignments.length !== this.nodeCount) {
        throw new Error(`${id} assignments length must equal the node count`);
      }
    }
    if (props.maximumLabelIterations !== undefined && !(props.assignments && props.startNodes)) {
      throw new Error(`${id} maximumLabelIterations needs startNodes and assignments`);
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
        validateCapacity(id, name, value, minimum, maximum);
      }
      const unreached = raster.unreachedCost ?? DEFAULT_UNREACHED_COST;
      if (!Number.isFinite(unreached) || !(unreached > 0) || unreached >= 3.0e38) {
        throw new Error(`${id} raster.unreachedCost must be a positive finite number below 3e38`);
      }
      const pixelCount = raster.width * raster.height;
      if (!Number.isSafeInteger(pixelCount) || pixelCount > 0x7fffffff) {
        throw new Error(`${id} raster width * height must fit in 31 bits`);
      }
      if (raster.output.pixelFacilities) {
        if (!props.assignments) {
          throw new Error(`${id} raster.output.pixelFacilities needs assignments`);
        }
        if (this.mode !== 'min') {
          throw new Error(`${id} raster.output.pixelFacilities needs raster.mode 'min'`);
        }
        if (
          props.startNodes &&
          props.startNodes.length > GPU_NETWORK_ISOCHRONES_MAXIMUM_RASTER_FACILITIES
        ) {
          throw new Error(
            `${id} raster.output.pixelFacilities supports at most ${GPU_NETWORK_ISOCHRONES_MAXIMUM_RASTER_FACILITIES} startNodes`
          );
        }
        validatePackedView(
          raster.output.pixelFacilities,
          ['uint32'],
          `${id} raster.output.pixelFacilities`
        );
        if (raster.output.pixelFacilities.length < pixelCount) {
          throw new Error(`${id} raster.output.pixelFacilities must hold width * height rows`);
        }
      }
      if (raster.output.triangleFacilities) {
        if (!raster.output.pixelFacilities || !raster.output.triangles) {
          throw new Error(
            `${id} raster.output.triangleFacilities needs pixelFacilities and the triangle output`
          );
        }
        validatePackedView(
          raster.output.triangleFacilities,
          ['uint32'],
          `${id} raster.output.triangleFacilities`
        );
        if (raster.output.triangleFacilities.length !== raster.output.triangleBands!.length) {
          throw new Error(
            `${id} raster.output.triangleFacilities length must equal triangleBands length`
          );
        }
      }
      if (raster.rings) {
        validateCapacity(id, 'raster.rings.edgeCapacity', raster.rings.edgeCapacity);
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
      if (cellOutline.byFacility && !props.assignments) {
        throw new Error(`${id} cellOutline.byFacility needs assignments`);
      }
      if (cellOutline.cellFacilities && !cellOutline.byFacility) {
        throw new Error(`${id} cellOutline.cellFacilities needs cellOutline.byFacility`);
      }
      if (cellOutline.byFacility && !cellOutline.output.groups) {
        throw new Error(`${id} cellOutline.byFacility needs cellOutline.output.groups`);
      }
      if (cellOutline.tableCapacity !== undefined) {
        validateCapacity(id, 'cellOutline.tableCapacity', cellOutline.tableCapacity);
      }
    }
    const inputs = [
      ...getGPUNetworkTopologyViews(props.topology),
      props.costBreaks,
      props.parameters,
      props.startNodes,
      props.startCosts,
      typeof props.startCount === 'object' ? props.startCount : undefined,
      props.costLimit,
      ...(props.startNodes ? [] : [props.costs, props.assignments])
    ];
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        ...(props.startNodes ? [props.costs, props.assignments] : []),
        cellOutline?.cellFacilities,
        ...collectGraphViews(raster?.output),
        ...collectGraphViews(cellOutline?.output),
        ...getRingOutputViews(cellOutline?.rings?.output),
        ...getRingOutputViews(raster?.rings?.output),
        ...collectGraphViews(cellOutline?.table)
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
    const {offsets, neighbors, weights, nodePositions} = props.topology;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGPUNetworkTopologyViews(props.topology),
      props.costs,
      props.startNodes,
      props.startCosts,
      typeof props.startCount === 'object' ? props.startCount : undefined,
      props.costLimit,
      props.output?.status?.converged,
      props.costBreaks,
      props.parameters,
      props.assignments,
      cellOutline?.cellFacilities,
      ...collectGraphViews(raster?.output),
      ...collectGraphViews(cellOutline?.output),
      ...getRingOutputViews(cellOutline?.rings?.output),
      ...getRingOutputViews(raster?.rings?.output),
      ...collectGraphViews(cellOutline?.table)
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (props.startNodes && props.assignments) {
      nodes.push(
        ...new GPUNetworkServiceAreas({
          id: `${id}-search`,
          topology: props.topology,
          facilities: props.startNodes,
          facilityCosts: props.startCosts,
          facilityCount: props.startCount,
          costLimit: props.costLimit,
          maximumIterations: props.maximumIterations,
          maximumLabelIterations: props.maximumLabelIterations,
          localIterations: props.localIterations,
          output: {
            assignments: props.assignments,
            costs: props.costs,
            status: {converged: props.output?.status?.converged}
          }
        }).getCommandNodes(graph)
      );
    } else if (props.startNodes) {
      nodes.push(
        ...new GPUNetworkReachability({
          id: `${id}-search`,
          topology: props.topology,
          startNodes: props.startNodes,
          startCosts: props.startCosts,
          startCount: props.startCount,
          costLimit: props.costLimit,
          maximumIterations: props.maximumIterations,
          localIterations: props.localIterations,
          output: {costs: props.costs, status: {converged: props.output?.status?.converged}}
        }).getCommandNodes(graph)
      );
    }

    if (raster) {
      const {width, height} = raster;
      const pixelCount = width * height;
      // The isobands read their extent from the per-frame parameters, so the grid only sizes them.
      const grid: GPURasterMetadata = {
        width,
        height,
        affine: [1, 0, 0, 0, 1, 0],
        pixelInterpretation: 'point'
      };
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
          offsets,
          neighbors,
          weights,
          costs: props.costs,
          nodePositions,
          parameters: props.parameters,
          encoded,
          assignments: raster.output.pixelFacilities ? props.assignments : undefined
        }),
        createIsochronesDecodeNode<Parameters>(graph, {
          id: `${id}-decode`,
          mode,
          unreachedCost: raster.unreachedCost ?? DEFAULT_UNREACHED_COST,
          encoded,
          values,
          pixelCount,
          pixelFacilities: raster.output.pixelFacilities
        }),
        ...new GPUIsobands({
          id: `${id}-isobands`,
          grid,
          values,
          breaks: props.costBreaks,
          parameters: bandParameters,
          output: {
            bandClasses: raster.output.bandClasses,
            triangles: raster.output.triangles,
            triangleBands: raster.output.triangleBands,
            edges: raster.output.edges,
            edgeBands: raster.output.edgeBands,
            status: raster.output.status
          }
        }).getCommandNodes(graph)
      );
      if (raster.output.triangleFacilities) {
        nodes.push(
          createIsochronesTriangleFacilityNode<Parameters>(graph, {
            id: `${id}-triangle-facilities`,
            width,
            height,
            triangles: raster.output.triangles!,
            triangleCount: raster.output.status!.count!,
            parameters: props.parameters,
            pixelFacilities: raster.output.pixelFacilities!,
            triangleFacilities: raster.output.triangleFacilities
          })
        );
      }
      if (raster.rings) {
        nodes.push(
          ...new GPUIsobandRings({
            id: `${id}-rings`,
            grid,
            values,
            breaks: props.costBreaks,
            parameters: bandParameters,
            edgeCapacity: raster.rings.edgeCapacity,
            vertexTolerance: raster.rings.vertexTolerance,
            output: raster.rings.output
          }).getCommandNodes(graph)
        );
      }
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
      let cellFacilities: GraphDataView<'uint32'> | undefined;
      const table: GPUCellTable = cellOutline.table ?? {
        cells: createTransientView(graph, `${id}-reached-cells`, 'uint32x2', capacity),
        counts: createTransientView(graph, `${id}-reached-counts`, 'uint32', capacity),
        status: {
          count: createTransientView(graph, `${id}-reached-count`, 'uint32', 1),
          overflow: createTransientView(graph, `${id}-reached-overflow`, 'uint32', 1)
        }
      };
      const byFacility = Boolean(cellOutline.byFacility);
      if (cellOutline.family === 'quadbin' && !byFacility) {
        nodes.push(
          ...new GPUCellAggregation({
            id: `${id}-reached-table`,
            family: 'quadbin',
            resolution: cellOutline.resolution,
            positions: nodePositions,
            mask,
            output: table
          }).getCommandNodes(graph)
        );
      } else {
        const nodeCells = createTransientView(graph, `${id}-node-cells`, 'uint32x2', nodeCount);
        nodes.push(
          ...new GPUPointToCell({
            id: `${id}-node-cells`,
            family: cellOutline.family,
            resolution: cellOutline.resolution,
            positions: nodePositions,
            mask,
            output: {cells: nodeCells}
          }).getCommandNodes(graph),
          ...new GPUCellAggregation({
            id: `${id}-reached-table`,
            family: cellOutline.family,
            resolution: cellOutline.resolution,
            cells: nodeCells,
            output: table
          }).getCommandNodes(graph)
        );
        if (byFacility) {
          cellFacilities =
            cellOutline.cellFacilities ??
            createTransientView(graph, `${id}-cell-facilities`, 'uint32', capacity);
          nodes.push(
            ...createIsochronesCellFacilityNodes<Parameters>(graph, {
              id: `${id}-cell-facilities`,
              nodeCount,
              nodeCells,
              tableCells: table.cells,
              tableCount: table.status.count,
              costs: props.costs,
              assignments: props.assignments!,
              cellMinimumCosts: createTransientView(
                graph,
                `${id}-cell-minimum-costs`,
                'uint32',
                capacity
              ),
              cellFacilities
            })
          );
        }
      }
      nodes.push(
        ...new GPUCellSetOutline({
          id: `${id}-outline`,
          family: cellOutline.family,
          cells: table.cells,
          count: table.status.count,
          groups: cellFacilities,
          output: cellOutline.output,
          rings: cellOutline.rings
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }
}

/** Returns every graph view in a bag of views, descending into nested status bags. */
function collectGraphViews(bag: object | undefined): GraphDataView[] {
  if (!bag) {
    return [];
  }
  return Object.values(bag).flatMap(value => {
    if (!value || typeof value !== 'object') {
      return [];
    }
    // A view carries its buffer; any other object is a nested bag such as `status`.
    return 'buffer' in value ? [value as GraphDataView] : collectGraphViews(value);
  });
}

/** Views of an optional ring output, with the nested polygon layout and status flattened. */
function getRingOutputViews(output: GPUSegmentRingAssemblyOutput | undefined): GraphDataView[] {
  return collectGraphViews(output);
}
