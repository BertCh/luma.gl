// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUGroupStatistics} from '../../gpu-dataframe/group-statistics/index';
import {GPUNetworkSnapping} from '../../gpu-network/network-accessibility/index';
import {GPUNetworkServiceAreas} from '../../gpu-network/network-analysis/index';
import {GPUNetworkIsochrones} from '../../gpu-network/network-isochrones/index';
import type {
  GPUNetworkIsochroneCellOutline,
  GPUNetworkIsochroneRaster
} from '../../gpu-network/network-isochrones/index';
import {GPUPointInPolygonJoin} from '../spatial-join/index';
import type {GPUSegmentRingAssemblyOutput} from '../ring-assembly/index';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {assertRecipe, getOrCreateView, RecipeBuilder, type GPURecipeResult} from './recipe-utils';

const ID = 'GPUDriveTimeCatchmentRecipe';

/** Band key written for demand points that are unsnapped or beyond every break. */
export const GPU_DRIVE_TIME_CATCHMENT_NO_BAND = 0xffffffff;

/** Road network as a CSR with planar node positions. */
export type GPUDriveTimeNetwork = {
  /** CSR row offsets, `nodeCount + 1` rows. */
  offsets: GraphDataView<'uint32'>;
  /** CSR edge targets. */
  neighbors: GraphDataView<'uint32'>;
  /** Travel cost per edge (also the cost used to split snapped edges). */
  weights: GraphDataView<'float32'>;
  /** Planar node positions in the coordinates of the facility and demand points. */
  nodePositions: GraphDataView<'float32x2'>;
};

/**
 * Join of the demand points against the isochrone rings (`cellOutline.rings.output.polygons`).
 * The polygon layout has GPU-written offsets over fixed-length views, so the join's compile-time
 * topology is the ring capacity; unused polygons and rings are empty.
 */
export type GPUDriveTimeIsochroneJoinOptions = {
  /** Maximum `(demand point, isochrone)` bounding-box candidates per encoding. */
  candidateCapacity: number;
  /** Optional caller-owned per demand point: 0 inside the isochrone polygons, else `GPU_SPATIAL_JOIN_NO_FEATURE`. */
  pointFeatureIds?: GraphDataView<'uint32'>;
  /** Optional one-row join overflow flag. */
  overflow?: GraphDataView<'uint32'>;
  /** Optional one-row count of candidates the join could not classify. */
  uncertainCount?: GraphDataView<'uint32'>;
};

/**
 * Optional isochrone polygons of the facility costs: raster isobands (triangles per band), or the
 * outline of the cells holding reached nodes with optional closed rings. At least one is required.
 */
export type GPUDriveTimeIsochroneOptions = {
  /** Band edges in cost units; its length is the maximum break count. */
  breaks: GraphDataView<'float32'>;
  /** `getGPUNetworkIsochroneParameterValues` view (12 elements). */
  parameters: GraphDataView<'float32'>;
  /** Raster size and caller-owned isoband outputs. */
  raster?: GPUNetworkIsochroneRaster;
  /**
   * Cell-outline path (positions must be longitude/latitude). Set `cellOutline.rings` for isochrone
   * rings (shells, holes, shell assignment) of the nodes with cost at most `cellCostLimit`.
   */
  cellOutline?: GPUNetworkIsochroneCellOutline;
  /** Join demand against the cell isochrone rings; needs `cellOutline.rings.output.polygons`. */
  joinDemand?: GPUDriveTimeIsochroneJoinOptions;
};

/** Properties for {@link addDriveTimeCatchmentRecipe}. */
export type GPUDriveTimeCatchmentRecipeProps = {
  /** Prefix for every node and transient ID. Defaults to `'drive-time-catchment'`. */
  id?: string;
  /** Road network. */
  network: GPUDriveTimeNetwork;
  /** Facility locations, snapped to the nearest edge. */
  facilities: GraphDataView<'float32x2'>;
  /** Demand points (households, customers), snapped to the nearest edge. */
  demand: GraphDataView<'float32x2'>;
  /** Optional per-demand-point value summarised per drive-time band. */
  demandValues?: GraphDataView<'float32'>;
  /** Optional one-row per-frame maximum snap distance (needed with `candidateCapacity`). */
  maxSnapDistance?: GraphDataView<'float32'>;
  /** Compile-time BVH candidate capacity of the snapping; omit for the exact all-edges scan. */
  candidateCapacity?: number;
  /** Optional one-row cost cutoff of the service-area search. */
  costLimit?: GraphDataView<'float32'>;
  /** Compile-time iteration bound of each service-area phase. */
  maxIterations?: number;
  /**
   * Ascending drive-time upper bounds, one per band. Band `k` is the first with
   * `time < bandBreaks[k]`; slower, unreached or unsnapped demand gets no band. Per-frame values.
   */
  bandBreaks: GraphDataView<'float32'>;
  /** Rows of the band table; defaults to `bandBreaks.length`. */
  bandCapacity?: number;
  /** Caller-owned per-node facility seed row (`seedNodes` row, so facility = row / 2). */
  assignments?: GraphDataView<'uint32'>;
  /** Caller-owned per-node minimum cost to a facility (`+Infinity` when unreached). */
  nodeCosts?: GraphDataView<'float32'>;
  /** Caller-owned drive time of each demand point (`+Infinity` when unreachable). */
  demandTimes?: GraphDataView<'float32'>;
  /** Caller-owned band of each demand point. */
  demandBands?: GraphDataView<'uint32'>;
  /** Caller-owned band table: ascending band keys, counts, and optional per-band sum/mean. */
  bands?: {
    keys?: GraphDataView<'uint32'>;
    counts?: GraphDataView<'uint32'>;
    count?: GraphDataView<'uint32'>;
    overflow?: GraphDataView<'uint32'>;
    sumValues?: GraphDataView<'float32'>;
    means?: GraphDataView<'float32'>;
  };
  /** Optional isochrone polygons of the same costs. */
  isochrones?: GPUDriveTimeIsochroneOptions;
};

/** Named outputs of {@link addDriveTimeCatchmentRecipe}. */
export type GPUDriveTimeCatchmentRecipeResult = GPURecipeResult & {
  /** Facility seed nodes (`2 * facilityCount` rows, two edge endpoints per facility). */
  seedNodes: GraphDataView<'uint32'>;
  seedCosts: GraphDataView<'float32'>;
  assignments: GraphDataView<'uint32'>;
  nodeCosts: GraphDataView<'float32'>;
  demandTimes: GraphDataView<'float32'>;
  demandBands: GraphDataView<'uint32'>;
  bands: {
    keys: GraphDataView<'uint32'>;
    counts: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    sumValues?: GraphDataView<'float32'>;
    means?: GraphDataView<'float32'>;
  };
  /** Isochrone ring outputs (the caller-owned `isochrones.cellOutline.rings.output`), if requested. */
  isochroneRings?: GPUSegmentRingAssemblyOutput;
  /** Join of demand against the isochrone rings, if `isochrones.joinDemand` was set. */
  isochroneDemand?: {
    pointFeatureIds: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    uncertainCount: GraphDataView<'uint32'>;
  };
};

/**
 * Drive-time catchment recipe: which demand is within each drive-time band of the facilities.
 *
 * Chain: `GPUNetworkSnapping` (facilities) -> `GPUNetworkServiceAreas` (multi-source costs) ->
 * optional `GPUNetworkIsochrones` (isoband triangles, or cell-outline rings) ->
 * `GPUNetworkSnapping` (demand) -> adapter kernels (demand time =
 * `min(cost[source] + sourceCost, cost[target] + targetCost)` of the snapped edge, then the band)
 * -> `GPUGroupStatistics` keyed by band.
 *
 * Bands are classified in network space (exact at the snapped edge, no polygon approximation):
 * isoband output has a GPU-written, data-dependent triangle count that `GPUPointInPolygonJoin`
 * cannot take as polygons. The cell path produces closed rings (`GPUSegmentRingAssembly`) whose
 * polygon layout does fit the join: its offsets are GPU-written over fixed-length views, so
 * `isochrones.joinDemand` joins the demand points against the isochrone at `cellCostLimit` in the
 * coordinates of the facilities and demand (longitude/latitude). That polygon is the union of
 * cells holding reached nodes, a coarser answer than the network-space bands.
 */
export function addDriveTimeCatchmentRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUDriveTimeCatchmentRecipeProps
): GPUDriveTimeCatchmentRecipeResult {
  const id = props.id ?? 'drive-time-catchment';
  const builder = new RecipeBuilder(graph);
  const {network} = props;
  const nodeCount = network.nodePositions.length;
  const facilityCount = props.facilities.length;
  const demandCount = props.demand.length;
  const bandCount = props.bandBreaks.length;
  assertRecipe(
    ID,
    network.offsets.length === nodeCount + 1,
    'offsets must have nodeCount + 1 rows'
  );
  assertRecipe(ID, bandCount >= 1, 'bandBreaks must not be empty');

  const seedNodes = getOrCreateView(graph, `${id}-seed-nodes`, 'uint32', 2 * facilityCount);
  const seedCosts = getOrCreateView(graph, `${id}-seed-costs`, 'float32', 2 * facilityCount);
  builder.add(
    new GPUNetworkSnapping({
      id: `${id}-snap-facilities`,
      points: props.facilities,
      nodePositions: network.nodePositions,
      offsets: network.offsets,
      edgeTargets: network.neighbors,
      edgeCosts: network.weights,
      maxSnapDistance: props.maxSnapDistance,
      candidateCapacity: props.candidateCapacity,
      snappedEdges: getOrCreateView(graph, `${id}-facility-edges`, 'uint32', facilityCount),
      seedNodes,
      seedCosts
    })
  );

  const assignments = getOrCreateView(
    graph,
    `${id}-assignments`,
    'uint32',
    nodeCount,
    props.assignments
  );
  const nodeCosts = getOrCreateView(
    graph,
    `${id}-node-costs`,
    'float32',
    nodeCount,
    props.nodeCosts
  );
  builder.add(
    new GPUNetworkServiceAreas({
      id: `${id}-service-areas`,
      offsets: network.offsets,
      neighbors: network.neighbors,
      weights: network.weights,
      facilities: seedNodes,
      facilityCosts: seedCosts,
      costLimit: props.costLimit,
      maxIterations: props.maxIterations,
      assignments,
      costs: nodeCosts
    })
  );

  const isochrones = props.isochrones;
  if (isochrones) {
    assertRecipe(
      ID,
      Boolean(isochrones.raster || isochrones.cellOutline),
      'isochrones needs raster or cellOutline'
    );
    assertRecipe(
      ID,
      !isochrones.joinDemand || Boolean(isochrones.cellOutline?.rings?.output.polygons),
      'isochrones.joinDemand needs cellOutline.rings.output.polygons'
    );
    builder.add(
      new GPUNetworkIsochrones({
        id: `${id}-isochrones`,
        offsets: network.offsets,
        neighbors: network.neighbors,
        weights: network.weights,
        nodePositions: network.nodePositions,
        costs: nodeCosts,
        breaks: isochrones.breaks,
        parameters: isochrones.parameters,
        raster: isochrones.raster,
        cellOutline: isochrones.cellOutline
      })
    );
  }
  let isochroneDemand: GPUDriveTimeCatchmentRecipeResult['isochroneDemand'];
  const polygons = isochrones?.cellOutline?.rings?.output.polygons;
  if (isochrones?.joinDemand && polygons) {
    const join = isochrones.joinDemand;
    isochroneDemand = {
      pointFeatureIds: getOrCreateView(
        graph,
        `${id}-isochrone-demand`,
        'uint32',
        demandCount,
        join.pointFeatureIds
      ),
      overflow: getOrCreateView(graph, `${id}-isochrone-join-overflow`, 'uint32', 1, join.overflow),
      uncertainCount: getOrCreateView(
        graph,
        `${id}-isochrone-join-uncertain`,
        'uint32',
        1,
        join.uncertainCount
      )
    };
    builder.add(
      new GPUPointInPolygonJoin({
        id: `${id}-isochrone-join`,
        points: props.demand,
        polygonPositions: polygons.positions,
        featureOffsets: polygons.featureOffsets,
        polygonOffsets: polygons.polygonOffsets,
        ringOffsets: polygons.ringOffsets,
        candidateCapacity: join.candidateCapacity,
        pointFeatureIds: isochroneDemand.pointFeatureIds,
        overflow: isochroneDemand.overflow,
        uncertainCount: isochroneDemand.uncertainCount
      })
    );
  }

  const snappedEdges = getOrCreateView(graph, `${id}-demand-edges`, 'uint32', demandCount);
  const sourceCosts = getOrCreateView(graph, `${id}-demand-source-costs`, 'float32', demandCount);
  const targetCosts = getOrCreateView(graph, `${id}-demand-target-costs`, 'float32', demandCount);
  builder.add(
    new GPUNetworkSnapping({
      id: `${id}-snap-demand`,
      points: props.demand,
      nodePositions: network.nodePositions,
      offsets: network.offsets,
      edgeTargets: network.neighbors,
      edgeCosts: network.weights,
      maxSnapDistance: props.maxSnapDistance,
      candidateCapacity: props.candidateCapacity,
      snappedEdges,
      sourceCosts,
      targetCosts
    })
  );

  // Adapter: demand drive time from the snapped edge and the node costs.
  const demandTimes = getOrCreateView(
    graph,
    `${id}-demand-times`,
    'float32',
    demandCount,
    props.demandTimes
  );
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-demand-times-kernel`,
      operation: 'GPURecipeDriveTimeCatchment',
      variant: 'demand-time',
      bindings: [
        {name: 'snappedEdges', view: snappedEdges, type: 'u32', access: 'read'},
        {name: 'sourceCosts', view: sourceCosts, type: 'f32', access: 'read'},
        {name: 'targetCosts', view: targetCosts, type: 'f32', access: 'read'},
        {name: 'offsets', view: network.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: network.neighbors, type: 'u32', access: 'read'},
        {name: 'nodeCosts', view: nodeCosts, type: 'f32', access: 'read'},
        {name: 'times', view: demandTimes, type: 'f32', access: 'read_write'}
      ],
      invocationCount: demandCount,
      declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const UNREACHED: f32 = 3.0e38;`,
      body: `let edge = snappedEdges[snappedEdgesOffset + index];
  var best = UNREACHED;
  if (edge != 0xffffffffu) {
    var lo = 0u;
    var hi = NODE_COUNT;
    while (lo + 1u < hi) {
      let mid = (lo + hi) / 2u;
      if (offsets[offsetsOffset + mid] <= edge) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    let targetNode = neighbors[neighborsOffset + edge];
    let viaSource = nodeCosts[nodeCostsOffset + lo] + sourceCosts[sourceCostsOffset + index];
    var viaTarget = UNREACHED;
    if (targetNode < NODE_COUNT) {
      viaTarget = nodeCosts[nodeCostsOffset + targetNode] + targetCosts[targetCostsOffset + index];
    }
    if (viaSource < best) {
      best = viaSource;
    }
    if (viaTarget < best) {
      best = viaTarget;
    }
  }
  if (best < UNREACHED) {
    times[timesOffset + index] = best;
  } else {
    var infinityBits = 0x7f800000u;
    times[timesOffset + index] = bitcast<f32>(infinityBits);
  }`
    })
  );

  const demandBands = getOrCreateView(
    graph,
    `${id}-demand-bands`,
    'uint32',
    demandCount,
    props.demandBands
  );
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-demand-bands-kernel`,
      operation: 'GPURecipeDriveTimeCatchment',
      variant: 'band',
      bindings: [
        {name: 'times', view: demandTimes, type: 'f32', access: 'read'},
        {name: 'bandBreaks', view: props.bandBreaks, type: 'f32', access: 'read'},
        {name: 'bands', view: demandBands, type: 'u32', access: 'read_write'}
      ],
      invocationCount: demandCount,
      declarations: `const BAND_COUNT: u32 = ${bandCount}u;`,
      body: `let time = times[timesOffset + index];
  var band = 0xffffffffu;
  if (time < 3.0e38) {
    for (var k = 0u; k < BAND_COUNT; k++) {
      if (time < bandBreaks[bandBreaksOffset + k]) {
        band = k;
        break;
      }
    }
  }
  bands[bandsOffset + index] = band;`
    })
  );

  const capacity = props.bandCapacity ?? bandCount;
  const provided = props.bands ?? {};
  const bands = {
    keys: getOrCreateView(graph, `${id}-band-keys`, 'uint32', capacity, provided.keys),
    counts: getOrCreateView(graph, `${id}-band-counts`, 'uint32', capacity, provided.counts),
    count: getOrCreateView(graph, `${id}-band-count`, 'uint32', 1, provided.count),
    overflow: getOrCreateView(graph, `${id}-band-overflow`, 'uint32', 1, provided.overflow),
    sumValues: props.demandValues
      ? getOrCreateView(graph, `${id}-band-sums`, 'float32', capacity, provided.sumValues)
      : undefined,
    means: props.demandValues
      ? getOrCreateView(graph, `${id}-band-means`, 'float32', capacity, provided.means)
      : undefined
  };
  builder.add(
    new GPUGroupStatistics({
      id: `${id}-band-statistics`,
      keys: demandBands,
      columns: props.demandValues
        ? [
            {
              values: props.demandValues,
              statistics: ['sum', 'mean'],
              output: {
                sums: getOrCreateView(graph, `${id}-band-sum-words`, 'uint32x2', capacity),
                sumValues: bands.sumValues,
                means: bands.means
              }
            }
          ]
        : [],
      output: {
        keys: bands.keys,
        counts: bands.counts,
        count: bands.count,
        overflow: bands.overflow
      }
    })
  );

  return {
    contributors: builder.contributors,
    seedNodes,
    seedCosts,
    assignments,
    nodeCosts,
    demandTimes,
    demandBands,
    bands,
    isochroneRings: isochrones?.cellOutline?.rings?.output,
    isochroneDemand
  };
}
