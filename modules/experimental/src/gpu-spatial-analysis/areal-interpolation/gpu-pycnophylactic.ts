// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGather,
  GPUReduction,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandGraphNodeCondition,
  type GPUCommandNode,
  type GraphBufferUse,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLDispatchGate,
  createWGSLKernelNode
} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  createSegmentSumNode,
  getSegmentSumDispatchShape,
  getSortKeyBits
} from '../../utils/sorted-segment-sums';
import {
  getGPUAuthalicSphereRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import {
  getPlacedCellAreaWGSL,
  validateGeographicRaster
} from '../../analysis-core/spherical-cell-area';
import type {GPUStatusScalar} from '../../analysis-core/contracts/status';
import {validateRasterGrid} from '../../gpu-raster/raster-band-utils';
import type {GPURasterMetadata} from '../../gpu-raster/types';
import {getZoneAnchorNodes} from './zone-anchor-nodes';

const OPERATION = 'GPUPycnophylactic';
/** Planar coordinates, or longitude/latitude rasters smoothed as density per spherical cell area. */
const SUPPORT: GPUSpatialContextSupport = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'area'
});

/** Maximum number of smoothing iterations of {@link GPUPycnophylactic}. */
export const GPU_PYCNOPHYLACTIC_MAXIMUM_ITERATIONS = 512;

/** Smoothing neighborhood of {@link GPUPycnophylactic}. */
export type GPUPycnophylacticKernel = 'rook' | 'box';

/** Caller-owned outputs of {@link GPUPycnophylactic}. */
export type GPUPycnophylacticOutput = {
  /**
   * `grid.width * grid.height` result: the mass per cell. The cells of zone `z` sum to `totals[z]` (to
   * float32 accuracy) and are non-negative. Cells outside the study area and cells of zones
   * without cells are 0. On a longitude/latitude raster this is density times cell area; see
   * `density`.
   */
  mass: GraphDataView<'float32'>;
  /**
   * Optional `grid.width * grid.height` density surface, the quantity that is smoothed. Mass per
   * cell on a planar raster; mass per square unit of the context (square meters by default) on a
   * longitude/latitude raster.
   */
  density?: GraphDataView<'float32'>;
  /** Optional status words; each is one uint32 row. */
  status?: {
    /**
     * 1 when the change fell to `tolerance` within `maximumIterations`, else 0. Requires
     * `tolerance`. Also 0 when `maximumIterations` is 0.
     */
    converged?: GPUStatusScalar;
    /**
     * The number of smoothing iterations that ran (at most `maximumIterations`). Requires
     * `tolerance`; without one every iteration runs.
     */
    iterationCount?: GPUStatusScalar;
    /** Zones with no cell (or no area) after anchoring, whose total is lost. */
    invalidCount?: GPUStatusScalar;
  };
};

/**
 * Properties for {@link GPUPycnophylactic}.
 *
 * Compile-time: the `grid` size, `zoneCount`, `maximumIterations`, `kernel`, `spatialContext` and which
 * optional views exist. Per-frame: the contents of `zones`, `totals`, `parameters`, `zoneAnchorCells`
 * and `tolerance`.
 */
export type GPUPycnophylacticProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'pycnophylactic'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels, and `zones` holds `grid.width * grid.height` rows. */
  grid: GPURasterMetadata;
  /**
   * Zone raster (`grid.width * grid.height` rows, row-major): the zone of each cell, or any value
   * `>= zoneCount` (such as `GPU_POLYGON_RASTERIZATION_NO_ZONE`) for a cell outside the study area.
   * Typically the `zones` output of a `GPUPolygonRasterization`.
   */
  zones: GraphDataView<'uint32'>;
  /** Number of zones `Z`. */
  zoneCount: number;
  /** Per-zone totals to preserve (`zoneCount` rows). Negative or non-finite totals count as 0. */
  totals: GraphDataView<'float32'>;
  /**
   * Maximum number of smoothing iterations, in `[0, GPU_PYCNOPHYLACTIC_MAXIMUM_ITERATIONS]`. All of
   * them run unless `tolerance` stops the surface earlier.
   */
  maximumIterations: number;
  /**
   * Optional one-row relative tolerance. After each iteration the largest change of any cell,
   * divided by the mean density of its zone (`totals[z] / area(z)`), is compared with it. Once the
   * change is at most `tolerance`, the surface is final: the per-cell kernels of the remaining
   * iterations are skipped through a GPU-written indirect dispatch (the compiled iteration count is
   * unchanged, and the surface is not copied forward). Required for `output.status.converged` and
   * `output.status.iterationCount`. Per-frame; a NaN or negative tolerance never converges.
   */
  tolerance?: GraphDataView<'float32'>;
  /**
   * Coordinate space of the raster. Omitted means native planar: every cell has the same area, so
   * mass and density coincide. A `longitude-latitude` context with metric `'great-circle'`
   * means a longitude/latitude grid placed by `parameters`: the surface smoothed is the density,
   * mass per square unit of the context (square meters by default). The zone totals are enforced
   * on mass = density times the spherical cell area `R^2 dLon (sin(latTop) - sin(latBottom))`, with `R` = `context.sphereRadius` or the
   * WGS84 authalic radius in the context's length unit. `output.mass` is then mass per cell and
   * `output.density` the surface itself.
   * Smoothing neighbors are still cell-index neighbors, so the focal mean is anisotropic on the
   * ground away from the equator. Other spaces throw.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * `longitude-latitude` only: the per-frame raster placement, the `getGPUPolygonRasterizationParameterValues(grid)`
   * view, the same view as `GPUPolygonRasterization.parameters` (float32, at least 4 values). A
   * negative cell height (north-up rows) is handled: cell areas follow the mirrored rows. Required
   * for, and only valid with, a longitude/latitude context.
   */
  parameters?: GraphDataView<'float32'>;
  /**
   * Optional per-zone fallback cell (`zoneCount` rows of cell indices; out of range means none).
   * A zone that rasterized to no cell claims its anchor cell, for example the cell containing the
   * zone's label point, so its total is not lost. The claim is made in a transient copy of `zones`
   * and only succeeds if the cell belongs to no zone or to a zone that keeps another cell; shared
   * anchors go to the lowest zone row. Zones that still have no cell are counted in
   * `output.status.invalidCount` and their total is lost.
   */
  zoneAnchorCells?: GraphDataView<'uint32'>;
  /**
   * Focal mean: `'rook'` (default, Tobler 1979) averages the 4 edge neighbors inside the study
   * area; `'box'` averages the 3x3 block including the cell itself.
   */
  kernel?: GPUPycnophylacticKernel;
  /** Caller-owned outputs. */
  output: GPUPycnophylacticOutput;
};

/**
 * Tobler's pycnophylactic (volume-preserving) interpolation on a raster: a smooth density surface
 * whose mass in every zone equals the zone total. Reference: Tobler, W. R. (1979), "Smooth
 * pycnophylactic interpolation for geographical regions", JASA 74(367), 519-530.
 *
 * Algorithm (at most `maximumIterations` iterations, deterministic). With `a(c)` the cell area (1 on a
 * planar raster, the spherical cell area on a longitude/latitude raster), `A(z)` the zone area and
 * `T(z)` the zone total:
 * 1. Initialize the density of every cell of zone `z` to `T(z) / A(z)`.
 * 2. Replace every cell by the focal mean of its in-area neighbors (`kernel`); add
 *    `(T(z) - M(z)) / A(z)` to every density of zone `z`, where `M(z) = sum a(c) density(c)`, so
 *    the mass is restored; clamp negative densities to 0; multiply the densities of the zone by
 *    `T(z) / M(z)` (a zone whose mass is 0 is filled uniformly). The last step guarantees the
 *    totals and non-negativity after every iteration.
 * 3. With `tolerance`: stop (further iterations become no-ops) when the largest change of any cell
 *    relative to its zone's mean density is at most `tolerance`; `output.status.converged` and
 *    `output.status.iterationCount` report what happened.
 *
 * Deviation from Tobler (1979). Tobler alternates the smoothing with a volume-restoring
 * adjustment; the restoring step here is additive then clamped then multiplicative. The
 * multiplicative step is kept because it is the only variant that keeps both the exact totals and
 * non-negativity without a second clamp; where the additive step clamped, it can leave flat
 * plateaus that a purely additive scheme would not. An additive-only variant would give negative
 * densities in zones next to much larger zones.
 *
 * Zones without cells. A zone smaller than a cell has no raster cell and its total would be lost.
 * `zoneAnchorCells` gives such a zone a fallback cell; zones that still have none are counted in
 * `output.status.invalidCount`. Totals of zones with no cell never appear in `output.mass`.
 *
 * Longitude/latitude rasters. With a `longitude-latitude` `spatialContext` the smoothed quantity is
 * the density per square unit of the context (square meters by default). The totals are enforced
 * on density times the spherical cell area, so the surface is a density per ground area and total
 * mass is exact per zone. Without it,
 * every cell counts the same, which is only correct on an equal-area raster.
 *
 * Zone sums use a fixed-order segmented reduction over cells sorted once by zone, so results are
 * bitwise reproducible. Smoothing converges geometrically with the zone size, so use `tolerance`
 * rather than guessing `maximumIterations`. Counts are float32 sums, exact up to 2^24 cells per zone;
 * larger zones lose count precision (the mass still sums to float32 accuracy).
 *
 * Together with `GPUArealInterpolation` this covers the tobler pycnophylactic and area
 * interpolation workloads. Unlike areal weighting, the surface is smooth across zone borders.
 * Limitation: no ancillary (dasymetric) initial surface and no barriers; the study area is the
 * union of the zone cells, and the focal mean is by cell index, not by ground distance.
 */
export class GPUPycnophylactic implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPycnophylacticProps;
  /** Resolved coordinate context: native planar unless a longitude/latitude raster was declared. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUPycnophylacticProps) {
    const id = props.id ?? 'pycnophylactic';
    this.id = id;
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext, SUPPORT);
    const cellCount = validateRasterGrid(id, props.grid);
    validateCapacity(id, 'zoneCount', props.zoneCount);
    validateCapacity(
      id,
      'maximumIterations',
      props.maximumIterations,
      0,
      GPU_PYCNOPHYLACTIC_MAXIMUM_ITERATIONS
    );
    if (props.kernel !== undefined && props.kernel !== 'rook' && props.kernel !== 'box') {
      throw new Error(`${id} kernel must be 'rook' or 'box'`);
    }
    const {output} = props;
    validatePackedView(props.zones, ['uint32'], `${id} zones`);
    validatePackedView(props.totals, ['float32'], `${id} totals`);
    validatePackedView(output.mass, ['float32'], `${id} output.mass`);
    if (props.zones.length !== cellCount) {
      throw new Error(`${id} zones must contain one row per grid cell`);
    }
    if (output.mass.length !== cellCount) {
      throw new Error(`${id} output.mass must contain one row per grid cell`);
    }
    if (props.totals.length !== props.zoneCount) {
      throw new Error(`${id} totals must contain zoneCount rows`);
    }
    if (this.spatialContext.coordinateSpace === 'longitude-latitude') {
      validateGeographicRaster(id, props.parameters);
    } else if (props.parameters) {
      throw new Error(`${id} parameters need a longitude-latitude spatialContext`);
    }
    if (output.density) {
      validatePackedView(output.density, ['float32'], `${id} output.density`);
      if (output.density.length !== cellCount) {
        throw new Error(`${id} output.density must contain one row per grid cell`);
      }
    }
    if (props.zoneAnchorCells) {
      validatePackedView(props.zoneAnchorCells, ['uint32'], `${id} zoneAnchorCells`);
      if (props.zoneAnchorCells.length !== props.zoneCount) {
        throw new Error(`${id} zoneAnchorCells must have zoneCount rows`);
      }
    }
    if (props.tolerance) {
      validatePackedView(props.tolerance, ['float32'], `${id} tolerance`);
      if (props.tolerance.length < 1) {
        throw new Error(`${id} tolerance must hold one float32`);
      }
    }
    for (const [name, view] of [
      ['output.status.converged', output.status?.converged],
      ['output.status.iterationCount', output.status?.iterationCount],
      ['output.status.invalidCount', output.status?.invalidCount]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must hold one uint32`);
        }
      }
    }
    if ((output.status?.converged || output.status?.iterationCount) && !props.tolerance) {
      throw new Error(
        `${id} output.status.converged and output.status.iterationCount need a tolerance`
      );
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
  }

  private getInputViews(): (GraphDataView | undefined)[] {
    const {props} = this;
    return [props.zones, props.totals, props.parameters, props.zoneAnchorCells, props.tolerance];
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return [
      output.mass,
      output.density,
      output.status?.converged,
      output.status?.iterationCount,
      output.status?.invalidCount
    ];
  }

  /** Returns the sort, initialization and per-iteration smoothing and rescaling nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props.grid;
    const {totals, zoneCount, maximumIterations: iterations, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    const cellCount = width * height;
    const kernel = props.kernel ?? 'rook';
    const geographic = this.spatialContext.coordinateSpace === 'longitude-latitude';
    const {tolerance} = props;
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Zones without cells claim their anchor cell in a transient copy of the raster.
    let zones = props.zones;
    if (props.zoneAnchorCells) {
      const anchored = getZoneAnchorNodes(graph, {
        id,
        operation: OPERATION,
        zones,
        zoneCount,
        anchorCells: props.zoneAnchorCells
      });
      zones = anchored.zones;
      nodes.push(...anchored.nodes);
    }

    // Cells sorted once by zone; zone sums are then fixed-order segmented sums.
    const counts = createTransientView(graph, `${id}-zone-counts`, 'uint32', zoneCount);
    const sortKeys = createTransientView(graph, `${id}-sort-keys`, 'uint32', cellCount);
    const sortIndices = createTransientView(graph, `${id}-sort-indices`, 'uint32', cellCount);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', cellCount);
    const sortedIndices = createTransientView(graph, `${id}-sorted-indices`, 'uint32', cellCount);
    const segmentOffsets = createTransientView(
      graph,
      `${id}-segment-offsets`,
      'uint32',
      zoneCount + 1
    );
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-counts-clear`,
        operation: OPERATION,
        view: counts,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-counts`,
        operation: OPERATION,
        variant: 'counts',
        bindings: [
          {name: 'zones', view: zones, type: 'u32', access: 'read'},
          {name: 'counts', view: counts, type: 'atomic<u32>', access: 'read_write'},
          {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
          {name: 'sortIndices', view: sortIndices, type: 'u32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: `const ZONE_COUNT: u32 = ${zoneCount}u;`,
        body: `let zone = zones[zonesOffset + index];
  if (zone < ZONE_COUNT) {
    atomicAdd(&counts[countsOffset + zone], 1u);
  }
  sortKeys[sortKeysOffset + index] = min(zone, ZONE_COUNT);
  sortIndices[sortIndicesOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${id}-sort`,
        keys: sortKeys,
        values: sortIndices,
        outputKeys: sortedKeys,
        outputValues: sortedIndices,
        keyBits: getSortKeyBits(zoneCount)
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-segment-scan`,
        input: counts,
        output: segmentOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-segment-total`,
        operation: OPERATION,
        variant: 'segment-total',
        bindings: [
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'segmentOffsets', view: segmentOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const LAST_ZONE: u32 = ${zoneCount - 1}u;`,
        body: `segmentOffsets[segmentOffsetsOffset + LAST_ZONE + 1u] =
    segmentOffsets[segmentOffsetsOffset + LAST_ZONE] + counts[countsOffset + LAST_ZONE];`
      })
    );

    const scratch = createTransientView(graph, `${id}-scratch`, 'float32', cellCount);
    const sortedValues = createTransientView(graph, `${id}-sorted-values`, 'float32', cellCount);
    const zoneSums = createTransientView(graph, `${id}-zone-sums`, 'float32', zoneCount);
    const zoneAreas = createTransientView(graph, `${id}-zone-areas`, 'float32', zoneCount);

    const sharedDeclarations = `const ZONE_COUNT: u32 = ${zoneCount}u;
const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
fn getTotal(zone: u32) -> f32 {
  let total = totals[totalsOffset + zone];
  // Negative, NaN and infinite totals count as 0.
  let isFinite = (bitcast<u32>(total) & 0x7f800000u) != 0x7f800000u;
  return select(0.0, total, isFinite && total > 0.0);
}`;
    const getSumNodes = (
      label: string,
      values: GraphDataView<'float32'>,
      sums: GraphDataView<'float32'>,
      isGated = false
    ) => [
      // `sortedIndices` is a permutation of the cells, so no index is out of range.
      ...new GPUGather({
        id: `${id}-${label}-gather`,
        source: values,
        indices: sortedIndices,
        output: sortedValues
      }).getCommandNodes(graph),
      createSegmentSumNode<Parameters>(graph, {
        id: `${id}-${label}-sum`,
        operation: OPERATION,
        segmentCount: zoneCount,
        input: sortedValues,
        segmentOffsets,
        output: sums,
        // Exclusive scan of per-zone counts of valid cells, so the last offset is at most `cellCount`.
        offsetsWithinInput: true,
        // Only the in-iteration sums are gated; `sumGate` is defined below, before they are built.
        ...(isGated ? sumGate : {})
      })
    ];

    // Cell areas: 1 per cell on a planar raster (the zone area is then the cell count).
    const cellAreas = geographic
      ? createTransientView(graph, `${id}-cell-areas`, 'float32', cellCount)
      : undefined;
    if (cellAreas) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-cell-areas`,
          operation: OPERATION,
          variant: 'cell-areas',
          bindings: [
            {name: 'zones', view: zones, type: 'u32', access: 'read'},
            {name: 'extent', view: props.parameters!, type: 'f32', access: 'read'},
            {name: 'cellAreas', view: cellAreas, type: 'f32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          declarations: `const ZONE_COUNT: u32 = ${zoneCount}u;
const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
${getPlacedCellAreaWGSL(getGPUAuthalicSphereRadius(this.spatialContext))}`,
          body: `var area = getPlacedCellArea(index / WIDTH);
  // Cells outside the zones and cells without a positive finite area carry no area.
  if (zones[zonesOffset + index] >= ZONE_COUNT || (bitcast<u32>(area) & 0x7f800000u) == 0x7f800000u || !(area > 0.0)) {
    area = 0.0;
  }
  cellAreas[cellAreasOffset + index] = area;`
        }),
        ...getSumNodes('zone-areas', cellAreas, zoneAreas)
      );
    } else {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-zone-areas`,
          operation: OPERATION,
          variant: 'zone-areas-from-counts',
          bindings: [
            {name: 'counts', view: counts, type: 'u32', access: 'read'},
            {name: 'zoneAreas', view: zoneAreas, type: 'f32', access: 'read_write'}
          ],
          invocationCount: zoneCount,
          body: 'zoneAreas[zoneAreasOffset + index] = f32(counts[countsOffset + index]);'
        })
      );
    }
    // Mass of the surface `values` (density times area), summed per zone into `zoneSums`.
    const getMassSumNodes = (label: string, values: GraphDataView<'float32'>) => {
      if (!cellAreas) {
        return getSumNodes(label, values, zoneSums, true);
      }
      const mass = createTransientView(graph, `${id}-${label}-mass`, 'float32', cellCount);
      return [
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${label}-mass`,
          operation: OPERATION,
          variant: 'mass',
          // Only called inside the iterations, so it shares their gate.
          ...gate,
          bindings: [
            {name: 'values', view: values, type: 'f32', access: 'read'},
            {name: 'cellAreas', view: cellAreas, type: 'f32', access: 'read'},
            {name: 'mass', view: mass, type: 'f32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          body: 'mass[massOffset + index] = values[valuesOffset + index] * cellAreas[cellAreasOffset + index];'
        }),
        ...getSumNodes(label, mass, zoneSums, true)
      ];
    };

    if (output.status?.invalidCount) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-invalid-count-clear`,
          operation: OPERATION,
          view: output.status?.invalidCount,
          type: 'u32',
          value: '0u',
          componentCount: 1
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-invalid-count`,
          operation: OPERATION,
          variant: 'invalid-count',
          bindings: [
            {name: 'zoneAreas', view: zoneAreas, type: 'f32', access: 'read'},
            {
              name: 'invalidCount',
              view: output.status?.invalidCount,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          invocationCount: zoneCount,
          body: `if (!(zoneAreas[zoneAreasOffset + index] > 0.0)) {
    atomicAdd(&invalidCount[invalidCountOffset], 1u);
  }`
        })
      );
    }

    // The final surface is a density. Planar rasters without a density output write it straight
    // into `output.mass` (mass equals density); otherwise a conversion pass writes mass and density.
    const needsConversion = geographic || Boolean(output.density);
    const finalSurface = needsConversion
      ? createTransientView(graph, `${id}-surface`, 'float32', cellCount)
      : output.mass;
    // The last iteration writes `finalSurface`; earlier ones alternate with the scratch raster.
    const bufferFor = (iteration: number) =>
      (iterations - 1 - iteration) % 2 === 0 ? finalSurface : scratch;
    const initial =
      iterations === 0 ? finalSurface : bufferFor(0) === finalSurface ? scratch : finalSurface;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize`,
        operation: OPERATION,
        variant: 'initialize',
        bindings: [
          {name: 'zones', view: zones, type: 'u32', access: 'read'},
          {name: 'totals', view: totals, type: 'f32', access: 'read'},
          {name: 'zoneAreas', view: zoneAreas, type: 'f32', access: 'read'},
          {name: 'destination', view: initial, type: 'f32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: sharedDeclarations,
        body: `let zone = zones[zonesOffset + index];
  var value = 0.0;
  if (zone < ZONE_COUNT) {
    let area = zoneAreas[zoneAreasOffset + zone];
    value = select(0.0, getTotal(zone) / area, area > 0.0);
  }
  destination[destinationOffset + index] = value;`
      })
    );

    // Convergence state: word 0 is set once the largest relative change falls to the tolerance,
    // word 1 counts the iterations that ran. Once converged, the converged-flag kernel zeroes the
    // indirect dispatch of every per-cell kernel and segment sum of the later iterations (rule 13);
    // the upstream gather and maximum nodes between them stay ungated and only re-reduce the live
    // surface. The surface is not copied forward: a final settle kernel copies the scratch raster
    // into the final surface only when the last iteration that ran wrote the scratch raster.
    const flag = tolerance
      ? createTransientView(graph, `${id}-converged-flag`, 'uint32', 2)
      : undefined;
    // Slot 0: every gated per-cell kernel is a guarded kernel, so they share one dispatch triple.
    // Slot 1: the zone sums, one workgroup per zone.
    const iterationGate = flag
      ? createWGSLDispatchGate(graph, {
          id: `${id}-iteration-dispatch`,
          operation: OPERATION,
          shapes: [[cellCount], getSegmentSumDispatchShape(zoneCount)]
        })
      : undefined;
    const gate: {
      condition?: GPUCommandGraphNodeCondition<Parameters>;
      extraResources?: GraphBufferUse[];
    } = iterationGate ? iterationGate.getGate(0) : {};
    const sumGate: typeof gate = iterationGate ? iterationGate.getGate(1) : {};
    if (flag && iterationGate) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-converged-flag-clear`,
          operation: OPERATION,
          variant: 'converged-flag-clear',
          bindings: [
            {name: 'flag', view: flag, type: 'u32', access: 'read_write'},
            iterationGate.binding
          ],
          invocationCount: 1,
          body: `flag[flagOffset] = 0u;
  flag[flagOffset + 1u] = 0u;
  ${iterationGate.armWGSL}`
        })
      );
    }

    let source = initial;
    for (let iteration = 0; iteration < iterations; iteration++) {
      const destination = bufferFor(iteration);
      const label = `iteration-${iteration}`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${label}-smooth`,
          operation: OPERATION,
          variant: `smooth-${kernel}`,
          bindings: [
            {name: 'zones', view: zones, type: 'u32', access: 'read'},
            {name: 'totals', view: totals, type: 'f32', access: 'read'},
            {name: 'source', view: source, type: 'f32', access: 'read'},
            {name: 'destination', view: destination, type: 'f32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          ...gate,
          declarations: sharedDeclarations,
          body: `let zone = zones[zonesOffset + index];
  var result = 0.0;
  if (zone < ZONE_COUNT) {
    let column = i32(index % WIDTH);
    let row = i32(index / WIDTH);
    var sum = 0.0;
    var count = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let isCenter = dx == 0 && dy == 0;
        let isDiagonal = dx != 0 && dy != 0;
        if ((isCenter && ${kernel === 'rook' ? 'true' : 'false'}) || (isDiagonal && ${kernel === 'rook' ? 'true' : 'false'})) {
          continue;
        }
        let x = column + dx;
        let y = row + dy;
        if (x < 0 || y < 0 || x >= i32(WIDTH) || y >= i32(HEIGHT)) {
          continue;
        }
        let neighbor = u32(y) * WIDTH + u32(x);
        if (zones[zonesOffset + neighbor] < ZONE_COUNT) {
          sum += source[sourceOffset + neighbor];
          count += 1.0;
        }
      }
    }
    result = select(source[sourceOffset + index], sum / count, count > 0.0);
  }
  destination[destinationOffset + index] = result;`
        }),
        ...getMassSumNodes(`${label}-restore`, destination),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${label}-restore-apply`,
          operation: OPERATION,
          variant: 'restore-and-clamp',
          record: {formula: 'gpu-spatial-analysis.pycnophylactic'},
          bindings: [
            {name: 'zones', view: zones, type: 'u32', access: 'read'},
            {name: 'totals', view: totals, type: 'f32', access: 'read'},
            {name: 'zoneAreas', view: zoneAreas, type: 'f32', access: 'read'},
            {name: 'zoneSums', view: zoneSums, type: 'f32', access: 'read'},
            {name: 'destination', view: destination, type: 'f32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          ...gate,
          declarations: sharedDeclarations,
          body: `let zone = zones[zonesOffset + index];
  if (zone < ZONE_COUNT) {
    let area = zoneAreas[zoneAreasOffset + zone];
    // The shift is a density, so it restores the mass of the zone: sum(a * shift) = T - M.
    let shift = select(0.0, (getTotal(zone) - zoneSums[zoneSumsOffset + zone]) / area, area > 0.0);
    destination[destinationOffset + index] = max(destination[destinationOffset + index] + shift, 0.0);
  }`
        }),
        ...getMassSumNodes(`${label}-scale`, destination),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${label}-scale-apply`,
          operation: OPERATION,
          variant: 'rescale',
          bindings: [
            {name: 'zones', view: zones, type: 'u32', access: 'read'},
            {name: 'totals', view: totals, type: 'f32', access: 'read'},
            {name: 'zoneAreas', view: zoneAreas, type: 'f32', access: 'read'},
            {name: 'zoneSums', view: zoneSums, type: 'f32', access: 'read'},
            {name: 'destination', view: destination, type: 'f32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          ...gate,
          declarations: sharedDeclarations,
          body: `let zone = zones[zonesOffset + index];
  if (zone < ZONE_COUNT) {
    let total = getTotal(zone);
    let sum = zoneSums[zoneSumsOffset + zone];
    let area = zoneAreas[zoneAreasOffset + zone];
    // A zone whose clamped mass vanished is filled uniformly.
    destination[destinationOffset + index] = select(
      select(0.0, total / area, area > 0.0),
      destination[destinationOffset + index] * (total / sum),
      sum > 0.0
    );
  }`
        })
      );
      if (flag && tolerance && iterationGate) {
        const change = createTransientView(graph, `${id}-${label}-change`, 'float32', cellCount);
        const maximumChange = createTransientView(graph, `${id}-${label}-maximum`, 'float32', 1);
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-${label}-change`,
            operation: OPERATION,
            variant: 'change',
            ...gate,
            bindings: [
              {name: 'zones', view: zones, type: 'u32', access: 'read'},
              {name: 'totals', view: totals, type: 'f32', access: 'read'},
              {name: 'zoneAreas', view: zoneAreas, type: 'f32', access: 'read'},
              {name: 'source', view: source, type: 'f32', access: 'read'},
              {name: 'destination', view: destination, type: 'f32', access: 'read'},
              {name: 'change', view: change, type: 'f32', access: 'read_write'}
            ],
            invocationCount: cellCount,
            declarations: sharedDeclarations,
            body: `let zone = zones[zonesOffset + index];
  var value = 0.0;
  if (zone < ZONE_COUNT) {
    let area = zoneAreas[zoneAreasOffset + zone];
    // Change relative to the mean density of the zone; empty zones cannot change.
    let reference = select(0.0, getTotal(zone) / area, area > 0.0);
    if (reference > 0.0) {
      value = abs(destination[destinationOffset + index] - source[sourceOffset + index]) / reference;
    }
  }
  change[changeOffset + index] = value;`
          }),
          ...new GPUReduction({
            id: `${id}-${label}-maximum`,
            input: change,
            output: maximumChange,
            operation: 'max'
          }).getCommandNodes(graph),
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-${label}-converged`,
            operation: OPERATION,
            variant: 'converged-flag',
            bindings: [
              {name: 'maximumChange', view: maximumChange, type: 'f32', access: 'read'},
              {name: 'tolerance', view: tolerance, type: 'f32', access: 'read'},
              {name: 'flag', view: flag, type: 'u32', access: 'read_write'},
              iterationGate.binding
            ],
            invocationCount: 1,
            // Ungated (it writes the gate). After convergence the maximum is stale and ignored.
            body: `if (flag[flagOffset] != 0u) {
    return;
  }
  flag[flagOffset + 1u] = flag[flagOffset + 1u] + 1u;
  if (maximumChange[maximumChangeOffset] <= tolerance[toleranceOffset]) {
    flag[flagOffset] = 1u;
    ${iterationGate.disarmWGSL}
  }`
          })
        );
      }
      source = destination;
    }
    if (flag && iterations > 0) {
      // Iteration i writes the final surface when (iterations - 1 - i) is even, so after
      // `executed` iterations the scratch raster is live exactly when (iterations - executed) is odd.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-settle`,
          operation: OPERATION,
          variant: 'settle',
          bindings: [
            {name: 'flag', view: flag, type: 'u32', access: 'read'},
            {name: 'scratch', view: scratch, type: 'f32', access: 'read'},
            {name: 'surface', view: finalSurface, type: 'f32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          declarations: `const ITERATIONS: u32 = ${iterations}u;`,
          body: `if ((ITERATIONS - flag[flagOffset + 1u]) % 2u == 1u) {
    surface[surfaceOffset + index] = scratch[scratchOffset + index];
  }`
        })
      );
    }

    if (needsConversion) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-convert`,
          operation: OPERATION,
          variant: `convert${cellAreas ? '-area' : ''}${output.density ? '-density' : ''}`,
          bindings: [
            {name: 'surface', view: finalSurface, type: 'f32', access: 'read'},
            ...(cellAreas
              ? [
                  {
                    name: 'cellAreas',
                    view: cellAreas,
                    type: 'f32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            {name: 'output', view: output.mass, type: 'f32', access: 'read_write'},
            ...(output.density
              ? [
                  {
                    name: 'density',
                    view: output.density,
                    type: 'f32' as const,
                    access: 'read_write' as const
                  }
                ]
              : [])
          ],
          invocationCount: cellCount,
          body: `let value = surface[surfaceOffset + index];
  output[outputOffset + index] = value${cellAreas ? ' * cellAreas[cellAreasOffset + index]' : ''};
  ${output.density ? 'density[densityOffset + index] = value;' : ''}`
        })
      );
    }
    if (output.status?.converged || output.status?.iterationCount) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-status`,
          operation: OPERATION,
          variant: 'status',
          bindings: [
            {name: 'flag', view: flag!, type: 'u32', access: 'read'},
            ...(output.status?.converged
              ? [
                  {
                    name: 'converged',
                    view: output.status?.converged,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  }
                ]
              : []),
            ...(output.status?.iterationCount
              ? [
                  {
                    name: 'iterationCount',
                    view: output.status?.iterationCount,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  }
                ]
              : [])
          ],
          invocationCount: 1,
          body: `${output.status?.converged ? 'converged[convergedOffset] = flag[flagOffset];' : ''}
  ${output.status?.iterationCount ? 'iterationCount[iterationCountOffset] = flag[flagOffset + 1u];' : ''}`
        })
      );
    }
    return nodes;
  }
}
