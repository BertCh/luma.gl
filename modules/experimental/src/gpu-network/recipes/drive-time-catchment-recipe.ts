// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUGroupStatistics} from '../../gpu-dataframe/group-statistics/index';
import {GPUNetworkSnapping} from '../network-accessibility/index';
import {createNetworkEdgeSourceNodes} from '../network-accessibility/network-edge-sources';
import {GPUNetworkServiceAreas} from '../network-analysis/index';
import {GPUNetworkIsochrones} from '../network-isochrones/index';
import type {
  GPUNetworkIsochronesCellOutline,
  GPUNetworkIsochronesRaster
} from '../network-isochrones/index';
import {
  validateGPUNetworkTopology,
  type GPUNetworkTopology
} from '../../analysis-core/network-topology';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport,
  getGPUSphereRadius
} from '../../analysis-core/contracts/spatial-context';
import type {
  GPUCardinalityPort,
  GPUOverflowStatusPort,
  GPURecipeStageStatusPort,
  GPURecipeStatusPort,
  GPUUncertaintyStatusPort
} from '../../analysis-core/contracts/index';
import {GPUPointInPolygonJoin} from '../../gpu-spatial-analysis/spatial-join/index';
import type {GPUSegmentRingAssemblyOutput} from '../../analysis-core/ring-assembly/index';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  assertRecipe,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeOverrides,
  type GPURecipeResult
} from '../../gpu-spatial-analysis/recipes/recipe-utils';

const ID = 'GPUDriveTimeCatchmentRecipe';

/**
 * Spatial contexts of {@link addDriveTimeCatchmentRecipe}: native planar coordinates (for example a
 * `createGPULocalFrame` frame in meters), or longitude/latitude degrees with the `'none'` or
 * `'great-circle'` metric. Both snappings measure distance in the context's length unit
 * (meters by default, or kilometers on longitude/latitude). Lambert equal-area frames are rejected.
 */
export const GPU_DRIVE_TIME_CATCHMENT_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport =
  Object.freeze({
    coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
    metrics: Object.freeze(['none', 'native', 'great-circle'] as const),
    measure: 'distance'
  });

/** Band key written for demand points that are unsnapped or beyond every break. */
export const GPU_DRIVE_TIME_CATCHMENT_NO_BAND = 0xffffffff;

/**
 * Join of the demand points against the isochrone rings (`cellOutline.rings.output.polygons`).
 * The polygon layout has GPU-written offsets over fixed-length views, so the join's compile-time
 * topology is the ring capacity; unused polygons and rings are empty.
 */
export type GPUDriveTimeIsochroneJoinOptions = {
  /** Maximum `(demand point, isochrone)` bounding-box candidates per encoding. */
  candidateCapacity: number;
};

/**
 * Optional isochrone polygons of the facility costs: raster isobands (triangles per band), or the
 * outline of the cells holding reached nodes with optional closed rings. At least one is required.
 */
export type GPUDriveTimeIsochroneOptions = {
  /** Band edges in cost units; its length is the maximum break count. */
  costBreaks: GraphDataView<'float32'>;
  /** `getGPUNetworkIsochronesParameterValues` view (12 elements). */
  parameters: GraphDataView<'float32'>;
  /** Raster size and caller-owned isoband outputs. */
  raster?: GPUNetworkIsochronesRaster;
  /**
   * Cell-outline path (positions must be longitude/latitude). Set `cellOutline.rings` for isochrone
   * rings (shells, holes, shell assignment) of the nodes with cost at most `cellCostLimit`.
   */
  cellOutline?: GPUNetworkIsochronesCellOutline;
  /** Join demand against the cell isochrone rings; needs `cellOutline.rings.output.polygons`. */
  joinDemand?: GPUDriveTimeIsochroneJoinOptions;
};

/** Properties for {@link addDriveTimeCatchmentRecipe}. */
export type GPUDriveTimeCatchmentRecipeProps = GPURecipeOverrides<
  Record<never, never>,
  {
    assignments?: GraphDataView<'uint32'>;
    nodeCosts?: GraphDataView<'float32'>;
    /** Mirrors `result.status`. */
    status?: {serviceAreaConverged?: GraphDataView<'uint32'>};
    demandTimes?: GraphDataView<'float32'>;
    demandBands?: GraphDataView<'uint32'>;
    bands?: {
      keys?: GraphDataView<'uint32'>;
      counts?: GraphDataView<'uint32'>;
      status?: Partial<GPUCardinalityPort & GPUOverflowStatusPort>;
      sumValues?: GraphDataView<'float32'>;
      means?: GraphDataView<'float32'>;
    };
    isochroneDemand?: {
      pointFeatureIds?: GraphDataView<'uint32'>;
      status?: Pick<GPUOverflowStatusPort, 'overflow'> & GPUUncertaintyStatusPort;
    };
    snapping?: {
      status?: {
        facilitiesOverflow?: GraphDataView<'uint32'>;
        demandOverflow?: GraphDataView<'uint32'>;
      };
    };
  },
  {
    startNodes?: GraphDataView<'uint32'>;
    startCosts?: GraphDataView<'float32'>;
    facilityEdges?: GraphDataView<'uint32'>;
    demandEdges?: GraphDataView<'uint32'>;
    demandSourceCosts?: GraphDataView<'float32'>;
    demandTargetCosts?: GraphDataView<'float32'>;
    bandSumWords?: GraphDataView<'uint32x2'>;
  }
> & {
  /** Prefix for every node and transient ID. Defaults to `'drive-time-catchment'`. */
  id?: string;
  /**
   * Coordinate space of the facility, demand and node positions. Omitted means native planar:
   * snapping compares Euclidean distances in the coordinate units (meters for a
   * `createGPULocalFrame` frame), and `maximumSnapDistance` is in those units. With
   * `coordinateSpace: 'longitude-latitude'` (metric `'none'` or `'great-circle'`; the cell-outline
   * isochrone path needs degrees) both snappings are great-circle: edges are ranked by great-circle
   * distance in the context's length unit (meters by default, or kilometers), as is
   * `maximumSnapDistance`, with no east-west bias at any latitude
   * (`sphereRadius` of a great-circle context is honored, else the mean Earth radius). Travel
   * costs (`topology.weights`) are independent of positions: the snapped fraction splits the edge
   * cost, so drive times stay in the units of the weights. Other geographic metrics (`'rhumb'`,
   * `'ellipsoidal'`) throw. Local planar frames must be azimuthal-equidistant, not Lambert
   * equal-area. The resolved context is returned as
   * the returned `spatialContext`.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Road network. `weights` are the travel cost per edge (also the cost used to split snapped
   * edges) and `nodePositions` are planar in the coordinates of the facility and demand points.
   */
  topology: GPUNetworkTopology & {nodePositions: GraphDataView<'float32x2'>};
  /** Facility locations, snapped to the nearest edge. */
  facilities: GraphDataView<'float32x2'>;
  /** Demand points (households, customers), snapped to the nearest edge. */
  demand: GraphDataView<'float32x2'>;
  /** Optional per-demand-point value summarised per drive-time band. */
  demandValues?: GraphDataView<'float32'>;
  /**
   * Optional one-row per-frame maximum snap distance in the context's length unit
   * (needed with `candidateCapacity`).
   * On longitude/latitude this distance arrives as a GPU view, so no short-distance warning is possible;
   * below about 17 m use createGPULocalFrame (see Precision of longitude/latitude positions).
   */
  maximumSnapDistance?: GraphDataView<'float32'>;
  /**
   * Compile-time BVH candidate capacity of the snapping; omit for the exact all-edges scan. When
   * set, each snapping reports candidate overflow (the returned `snapping`, stages `snap-facilities` and
   * `snap-demand`): an overflow means some facilities or demand may be snapped to a farther edge.
   * Great-circle searches ignore the capacity; kilometer contexts scan directly and report no overflow.
   */
  candidateCapacity?: number;
  /** Optional one-row cost cutoff of the service-area search. */
  costLimit?: GraphDataView<'float32'>;
  /** Compile-time iteration bound of each service-area phase. */
  maximumIterations?: number;
  /**
   * Ascending drive-time upper bounds, one per band. Band `k` is the first with
   * `time < costBreaks[k]`; slower, unreached or unsnapped demand gets no band. Per-frame values.
   */
  costBreaks: GraphDataView<'float32'>;
  /** Rows of the band table; defaults to `costBreaks.length`. */
  bandCapacity?: number;
  /** Optional isochrone polygons of the same costs. */
  isochrones?: GPUDriveTimeIsochroneOptions;
};

/**
 * Status stages of the caller-owned isochrone outputs. The views belong to the caller (the
 * isochrone contributor requires them together with their geometry), so only those present are
 * reported.
 */
function getIsochroneStatusStages(
  isochrones: GPUDriveTimeIsochroneOptions | undefined
): GPURecipeStageStatusPort[] {
  const stages: GPURecipeStageStatusPort[] = [];
  const raster = isochrones?.raster;
  const output = raster?.output;
  const rasterStatus = output?.status;
  if (rasterStatus?.count && rasterStatus.overflow) {
    stages.push({
      stage: 'isochrone-bands',
      status: {
        count: rasterStatus.count,
        overflow: rasterStatus.overflow,
        requiredCount: rasterStatus.requiredCount
      }
    });
  }
  if (rasterStatus?.edgeCount && rasterStatus.edgeOverflow) {
    stages.push({
      stage: 'isochrone-band-edges',
      status: {
        count: rasterStatus.edgeCount,
        overflow: rasterStatus.edgeOverflow,
        requiredCount: rasterStatus.edgeRequiredCount
      }
    });
  }
  const bandRings = raster?.rings?.output;
  if (bandRings) {
    stages.push({
      stage: 'isochrone-band-rings',
      status: {
        count: bandRings.status.count,
        overflow: bandRings.status.overflow,
        requiredCount: bandRings.status.requiredCount
      }
    });
  }
  const cellOutline = isochrones?.cellOutline;
  if (cellOutline?.table) {
    stages.push({
      stage: 'isochrone-cells',
      status: {
        count: cellOutline.table.status.count,
        overflow: cellOutline.table.status.overflow,
        requiredCount: cellOutline.table.status.requiredCount
      }
    });
  }
  if (cellOutline) {
    stages.push({
      stage: 'isochrone-outline',
      status: {
        count: cellOutline.output.status.count,
        overflow: cellOutline.output.status.overflow,
        requiredCount: cellOutline.output.status.requiredCount
      }
    });
  }
  const cellRings = cellOutline?.rings?.output;
  if (cellRings) {
    stages.push({
      stage: 'isochrone-rings',
      status: {
        count: cellRings.status.count,
        overflow: cellRings.status.overflow,
        requiredCount: cellRings.status.requiredCount
      }
    });
  }
  return stages;
}

/** Named outputs of {@link addDriveTimeCatchmentRecipe}. */
export type GPUDriveTimeCatchmentRecipeOutput = GPURecipeResult & {
  /** Resolved coordinate context of the positions. */
  spatialContext: GPUSpatialContext;
  /** Facility start nodes (`2 * facilityCount` rows, two edge endpoints per facility). */
  startNodes: GraphDataView<'uint32'>;
  /** Cost from each facility to its start nodes, aligned with `startNodes`. */
  startCosts: GraphDataView<'float32'>;
  assignments: GraphDataView<'uint32'>;
  nodeCosts: GraphDataView<'float32'>;
  demandTimes: GraphDataView<'float32'>;
  demandBands: GraphDataView<'uint32'>;
  /** Status words next to the per-stage `stages`. */
  status: {
    /** One-row flag: 1 when the service-area search converged (stage `service-areas`). */
    serviceAreaConverged: GraphDataView<'uint32'>;
  } & GPURecipeStatusPort;
  bands: {
    keys: GraphDataView<'uint32'>;
    counts: GraphDataView<'uint32'>;
    /** One-row status words of stage `band-statistics`; `requiredCount` is the unclamped number of bands present. */
    status: GPUCardinalityPort & GPUOverflowStatusPort;
    sumValues?: GraphDataView<'float32'>;
    means?: GraphDataView<'float32'>;
  };
  /** Isochrone ring outputs (the caller-owned `isochrones.cellOutline.rings.output`), if requested. */
  isochroneRings?: GPUSegmentRingAssemblyOutput;
  /** Join of demand against the isochrone rings, if `isochrones.joinDemand` was set. */
  isochroneDemand?: {
    pointFeatureIds: GraphDataView<'uint32'>;
    /** One-row status words of stage `isochrone-demand`. */
    status: {
      overflow: GraphDataView<'uint32'>;
      uncertainCount: GraphDataView<'uint32'>;
    };
  };
  /**
   * BVH candidate overflow flags of the facility and demand snapping, if `candidateCapacity` was
   * set: 1 when a candidate search dropped edges (the snap may not be the nearest edge).
   */
  snapping?: {
    status: {
      facilitiesOverflow: GraphDataView<'uint32'>;
      demandOverflow: GraphDataView<'uint32'>;
    };
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
 * cells holding reached nodes, a coarser answer than the network-space bands. The ring edges are
 * axis-aligned cell edges in degrees, so the join evaluates containment in the degree plane exactly
 * and is constructed with the native planar context whatever `spatialContext` says.
 *
 * Spatial context: see `GPUDriveTimeCatchmentRecipeProps.spatialContext`. Costs are in the units
 * of `topology.weights`; distances (`maximumSnapDistance`) are in the context's length unit.
 *
 * Status: `status.stages` reports `snap-facilities` and `snap-demand` (candidate overflow, with
 * `candidateCapacity`), `service-areas` (convergence), the isochrone stages for the caller-owned
 * isochrone outputs that carry status (`isochrone-bands`, `isochrone-band-edges`,
 * `isochrone-band-rings` on the raster path; `isochrone-cells`, `isochrone-outline`,
 * `isochrone-rings` on the cell path), `band-statistics` (count, overflow, required count) and,
 * with a demand join, `isochrone-demand`. The service-area `assignments` are handed to
 * `GPUNetworkIsochrones`, so `raster.output.pixelFacilities`, `triangleFacilities` and
 * `cellOutline.byFacility` work through the recipe; facility rows there are start rows
 * (`2 * facility + endpoint`).
 */
export function addDriveTimeCatchmentRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUDriveTimeCatchmentRecipeProps
): GPUDriveTimeCatchmentRecipeOutput {
  const id = props.id ?? 'drive-time-catchment';
  const spatialContext = resolveGPUSpatialContext(
    id,
    props.spatialContext,
    GPU_DRIVE_TIME_CATCHMENT_SPATIAL_CONTEXT_SUPPORT
  );
  const builder = new RecipeBuilder(graph);
  const {topology} = props;
  const outputs = props.outputs ?? {};
  const scratch = props.scratch ?? {};
  const {nodeCount} = validateGPUNetworkTopology(id, topology, {
    weights: true,
    nodePositions: true
  });
  const facilityCount = props.facilities.length;
  const demandCount = props.demand.length;
  const bandCount = props.costBreaks.length;
  assertRecipe(ID, bandCount >= 1, 'costBreaks must not be empty');

  // Edge source node per CSR edge, shared by the demand-time adapter (instead of a binary search).
  // The shared gpu-network nodes write every row, so rows past offsets[nodeCount] are never stale.
  const edgeCount = topology.neighbors.length;
  const edgeSources = createTransientView(
    graph,
    `${id}-edge-sources`,
    'uint32',
    Math.max(edgeCount, 1)
  );
  for (const node of createNetworkEdgeSourceNodes<Parameters>(graph, {
    id: `${id}-edge-sources-kernel`,
    operation: 'GPURecipeDriveTimeCatchment',
    offsets: topology.offsets,
    edgeSources,
    nodeCount
  })) {
    graph.add(node);
  }

  // Longitude/latitude positions snap great-circle in the context's length unit for either metric.
  const snappingContext: GPUSpatialContext =
    spatialContext.coordinateSpace === 'longitude-latitude'
      ? {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          sphereRadius: getGPUSphereRadius(spatialContext),
          ...(spatialContext.units ? {units: spatialContext.units} : {})
        }
      : spatialContext;
  const providedSnapping = outputs.snapping ?? {};
  const snapping: GPUDriveTimeCatchmentRecipeOutput['snapping'] =
    props.candidateCapacity !== undefined
      ? {
          status: {
            facilitiesOverflow: getOrCreateView(
              graph,
              `${id}-snap-facilities-overflow`,
              'uint32',
              1,
              providedSnapping.status?.facilitiesOverflow
            ),
            demandOverflow: getOrCreateView(
              graph,
              `${id}-snap-demand-overflow`,
              'uint32',
              1,
              providedSnapping.status?.demandOverflow
            )
          }
        }
      : undefined;

  const startNodes = getOrCreateView(
    graph,
    `${id}-start-nodes`,
    'uint32',
    2 * facilityCount,
    scratch.startNodes
  );
  const startCosts = getOrCreateView(
    graph,
    `${id}-start-costs`,
    'float32',
    2 * facilityCount,
    scratch.startCosts
  );
  builder.add(
    new GPUNetworkSnapping({
      id: `${id}-snap-facilities`,
      spatialContext: snappingContext,
      positions: props.facilities,
      topology,
      maximumSnapDistance: props.maximumSnapDistance,
      candidateCapacity: props.candidateCapacity,
      output: {
        status: {overflow: snapping?.status.facilitiesOverflow},
        snappedEdges: getOrCreateView(
          graph,
          `${id}-facility-edges`,
          'uint32',
          facilityCount,
          scratch.facilityEdges
        ),
        startNodes,
        startCosts
      }
    })
  );

  const assignments = getOrCreateView(
    graph,
    `${id}-assignments`,
    'uint32',
    nodeCount,
    outputs.assignments
  );
  const nodeCosts = getOrCreateView(
    graph,
    `${id}-node-costs`,
    'float32',
    nodeCount,
    outputs.nodeCosts
  );
  const serviceAreaConverged = getOrCreateView(
    graph,
    `${id}-service-area-converged`,
    'uint32',
    1,
    outputs.status?.serviceAreaConverged
  );
  builder.add(
    new GPUNetworkServiceAreas({
      id: `${id}-service-areas`,
      topology,
      facilities: startNodes,
      facilityCosts: startCosts,
      costLimit: props.costLimit,
      maximumIterations: props.maximumIterations,
      output: {assignments, costs: nodeCosts, status: {converged: serviceAreaConverged}}
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
        topology,
        costs: nodeCosts,
        // Input from the service areas: enables the facility-labelled outputs
        // (`raster.output.pixelFacilities`, `triangleFacilities`, `cellOutline.byFacility`). The
        // facility row is the start row, `2 * facility + endpoint`.
        assignments,
        costBreaks: isochrones.costBreaks,
        parameters: isochrones.parameters,
        raster: isochrones.raster,
        cellOutline: isochrones.cellOutline
      })
    );
  }
  let isochroneDemand: GPUDriveTimeCatchmentRecipeOutput['isochroneDemand'];
  const polygons = isochrones?.cellOutline?.rings?.output.polygons;
  if (isochrones?.joinDemand && polygons) {
    const join = isochrones.joinDemand;
    const provided = outputs.isochroneDemand ?? {};
    isochroneDemand = {
      pointFeatureIds: getOrCreateView(
        graph,
        `${id}-isochrone-demand`,
        'uint32',
        demandCount,
        provided.pointFeatureIds
      ),
      status: {
        overflow: getOrCreateView(
          graph,
          `${id}-isochrone-join-overflow`,
          'uint32',
          1,
          provided.status?.overflow
        ),
        uncertainCount: getOrCreateView(
          graph,
          `${id}-isochrone-join-uncertain`,
          'uint32',
          1,
          provided.status?.uncertainCount
        )
      }
    };
    builder.add(
      new GPUPointInPolygonJoin({
        id: `${id}-isochrone-join`,
        points: {kind: 'points', positions: props.demand},
        polygons: {
          kind: 'polygons',
          positions: polygons.positions,
          featureOffsets: polygons.featureOffsets,
          polygonOffsets: polygons.polygonOffsets,
          ringOffsets: polygons.ringOffsets
        },
        candidateCapacity: join.candidateCapacity,
        output: {
          pointFeatureIds: isochroneDemand.pointFeatureIds,
          status: isochroneDemand.status
        }
      })
    );
  }

  const snappedEdges = getOrCreateView(
    graph,
    `${id}-demand-edges`,
    'uint32',
    demandCount,
    scratch.demandEdges
  );
  const sourceCosts = getOrCreateView(
    graph,
    `${id}-demand-source-costs`,
    'float32',
    demandCount,
    scratch.demandSourceCosts
  );
  const targetCosts = getOrCreateView(
    graph,
    `${id}-demand-target-costs`,
    'float32',
    demandCount,
    scratch.demandTargetCosts
  );
  builder.add(
    new GPUNetworkSnapping({
      id: `${id}-snap-demand`,
      spatialContext: snappingContext,
      positions: props.demand,
      topology,
      maximumSnapDistance: props.maximumSnapDistance,
      candidateCapacity: props.candidateCapacity,
      output: {
        status: {overflow: snapping?.status.demandOverflow},
        snappedEdges,
        sourceCosts,
        targetCosts
      }
    })
  );

  // Adapter: demand drive time from the snapped edge and the node costs.
  const demandTimes = getOrCreateView(
    graph,
    `${id}-demand-times`,
    'float32',
    demandCount,
    outputs.demandTimes
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
        {name: 'edgeSources', view: edgeSources, type: 'u32', access: 'read'},
        {name: 'neighbors', view: topology.neighbors, type: 'u32', access: 'read'},
        {name: 'nodeCosts', view: nodeCosts, type: 'f32', access: 'read'},
        {name: 'times', view: demandTimes, type: 'f32', access: 'read_write'}
      ],
      invocationCount: demandCount,
      declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const UNREACHED: f32 = 3.0e38;`,
      body: `let edge = snappedEdges[snappedEdgesOffset + index];
  var best = UNREACHED;
  if (edge != 0xffffffffu) {
    let sourceNode = edgeSources[edgeSourcesOffset + edge];
    let targetNode = neighbors[neighborsOffset + edge];
    var viaSource = UNREACHED;
    if (sourceNode < NODE_COUNT) {
      viaSource = nodeCosts[nodeCostsOffset + sourceNode] + sourceCosts[sourceCostsOffset + index];
    }
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
    outputs.demandBands
  );
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-demand-bands-kernel`,
      operation: 'GPURecipeDriveTimeCatchment',
      variant: 'band',
      bindings: [
        {name: 'times', view: demandTimes, type: 'f32', access: 'read'},
        {name: 'bandBreaks', view: props.costBreaks, type: 'f32', access: 'read'},
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
  const provided = outputs.bands ?? {};
  const bands = {
    keys: getOrCreateView(graph, `${id}-band-keys`, 'uint32', capacity, provided.keys),
    counts: getOrCreateView(graph, `${id}-band-counts`, 'uint32', capacity, provided.counts),
    status: {
      count: getOrCreateView(graph, `${id}-band-count`, 'uint32', 1, provided.status?.count),
      overflow: getOrCreateView(
        graph,
        `${id}-band-overflow`,
        'uint32',
        1,
        provided.status?.overflow
      ),
      requiredCount: getOrCreateView(
        graph,
        `${id}-band-required-count`,
        'uint32',
        1,
        provided.status?.requiredCount
      )
    },
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
                sums: getOrCreateView(
                  graph,
                  `${id}-band-sum-words`,
                  'uint32x2',
                  capacity,
                  scratch.bandSumWords
                ),
                sumValues: bands.sumValues,
                means: bands.means
              }
            }
          ]
        : [],
      output: {
        keys: bands.keys,
        counts: bands.counts,
        status: bands.status
      }
    })
  );

  return {
    spatialContext,
    contributors: builder.contributors,
    startNodes,
    startCosts,
    assignments,
    nodeCosts,
    demandTimes,
    demandBands,
    bands,
    isochroneRings: isochrones?.cellOutline?.rings?.output,
    isochroneDemand,
    snapping,
    outputs: {
      assignments,
      nodeCosts,
      status: {serviceAreaConverged},
      demandTimes,
      demandBands,
      bands,
      ...(isochroneDemand ? {isochroneDemand} : {}),
      ...(snapping ? {snapping} : {})
    },
    intermediates: {startNodes, startCosts},
    status: {
      serviceAreaConverged,
      stages: [
        ...(snapping
          ? [
              {stage: 'snap-facilities', status: {overflow: snapping.status.facilitiesOverflow}},
              {stage: 'snap-demand', status: {overflow: snapping.status.demandOverflow}}
            ]
          : []),
        {stage: 'service-areas', status: {converged: serviceAreaConverged}},
        ...getIsochroneStatusStages(isochrones),
        {
          stage: 'band-statistics',
          status: bands.status
        },
        ...(isochroneDemand
          ? [
              {
                stage: 'isochrone-demand',
                status: isochroneDemand.status
              }
            ]
          : [])
      ]
    }
  };
}
