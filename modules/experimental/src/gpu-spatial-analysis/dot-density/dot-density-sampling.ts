// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  createPublishNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  DOT_DENSITY_CANDIDATE_PURPOSE,
  DOT_DENSITY_REMAINDER_PURPOSE,
  getDotSlotCountLimit
} from './dot-density-constants';
import {
  getPolygonTriangulationIndexCount,
  GPUPolygonTriangulation
} from '../polygon-triangulation/gpu-polygon-triangulation';
import {createSegmentSumNode} from '../../utils/sorted-segment-sums';
import {IS_FINITE_BITS_WGSL, IS_NAN_BITS_WGSL} from '../../utils/wgsl-common';
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import {GPU_DOT_DENSITY_PARAMETER_LENGTH} from './dot-density-parameters';
import {PHILOX_WGSL} from '../../utils/philox-random';

/** Feature ID written past the published count. */
const NO_FEATURE = '0xffffffffu';
/** Words per dot in the transient dot table: slot, rank, first triangle and end triangle. */
const DOT_WORDS = 4;

/**
 * Largest dot capacity (`output.positions.length`) of `GPUDotDensity` and the shared dot sampler on
 * `device`: the per-dot slot table holds 16 bytes per dot in one storage binding, so 8388608 on a
 * default device (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPUDotDensityMaximumCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, DOT_WORDS * 4);
}

/**
 * Fixed-point total of the per-feature triangle areas. Areas are scaled so one feature sums to at
 * most this value, which leaves about 7% of the u32 range as headroom for f32 rounding of the
 * area total. The selection bias of the quantization is below one part in 1e9 per triangle.
 */
const AREA_FIXED_POINT_TOTAL = '4.0e9';
/** Default mask-rejection attempts per dot. */
export const DEFAULT_DOT_MAXIMUM_ATTEMPTS = 32;

/** Optional dasymetric constraint shared by {@link GPUDotDensity} and {@link GPURandomPointsInPolygon}. */
export type GPUDotDensityMask = {
  /**
   * Packed row-major weights in `[0, 1]`, `width * height` rows, placed by the `maskExtent`
   * parameter. A candidate is kept with probability equal to the weight of its (nearest) cell;
   * outside the raster or on NaN the weight is 0.
   */
  weights: GraphDataView<'float32'>;
  /** Mask width in cells. Compile-time. */
  width: number;
  /** Mask height in cells. Compile-time. */
  height: number;
};

/** Caller-owned outputs shared by {@link GPUDotDensity} and {@link GPURandomPointsInPolygon}. */
export type GPUDotDensityOutput = {
  /**
   * Dot positions, capacity rows. Rows past `dots.count` are NaN. A dot of a feature that failed to
   * triangulate or has zero area, or whose mask attempts all failed, is NaN too (and counted in
   * `status.failedCount`), so it occupies its slot but is not drawn.
   */
  positions: GraphDataView<'float32x2'>;
  /**
   * Compact dot list: `ids` holds the feature row of each dot (capacity rows, `0xffffffff` past the
   * count), `count = min(total, capacity)` is safe as an instance count, and `overflow` is 1 when
   * the total exceeds the capacity or a slot count was clamped.
   */
  dots: GPUCompactOutput;
  /** Optional category per dot (dot density only), capacity rows. */
  categories?: GraphDataView<'uint32'>;
  /** Optional per-slot dot counts, one row per slot. */
  slotCounts?: GraphDataView<'uint32'>;
  /** Optional per-slot exclusive offsets into the dot list, one row per slot. */
  slotOffsets?: GraphDataView<'uint32'>;
  /** Optional completeness scalars. */
  status?: {
    /**
     * Optional one-row count of dots that could not be placed: dots of features whose polygons
     * failed to triangulate (or overflowed the triangulation work budget) or whose area is zero or
     * not finite, plus dots whose mask attempts all failed. Zero means every dot was placed.
     */
    failedCount?: GraphDataView<'uint32'>;
  };
};

/** Internal configuration of the shared dot sampling graph. @internal */
export type DotSamplingConfig = GPUPolygonGeometryFields & {
  id: string;
  operation: string;
  /** Float values per slot (dot density) or integer counts per feature (random points). */
  source:
    | {kind: 'values'; values: GraphDataView<'float32'>}
    | {kind: 'counts'; counts: GraphDataView<'uint32'>};
  categoryCount: number;
  parameters: GraphDataView<'uint32'>;
  maximumAttempts: number;
  mask?: GPUDotDensityMask;
  output: GPUDotDensityOutput;
};

/** Validates a shared dot sampling configuration and returns the feature count. @internal */
export function validateDotSamplingConfig(config: DotSamplingConfig): number {
  const {id, output, mask} = config;
  validatePackedView(config.positions, ['float32x2'], `${id} positions`);
  validatePackedView(config.featureOffsets, ['uint32'], `${id} featureOffsets`);
  validatePackedView(config.polygonOffsets, ['uint32'], `${id} polygonOffsets`);
  validatePackedView(config.ringOffsets, ['uint32'], `${id} ringOffsets`);
  const featureCount = config.featureOffsets.length - 1;
  if (featureCount < 1) {
    throw new Error(`${id} featureOffsets must describe at least one feature`);
  }
  validateCapacity(id, 'categoryCount', config.categoryCount);
  const slotCount = featureCount * config.categoryCount;
  if (config.source.kind === 'values') {
    validatePackedView(config.source.values, ['float32'], `${id} values`);
    if (config.source.values.length !== slotCount) {
      throw new Error(`${id} values must have featureCount * categoryCount rows`);
    }
  } else {
    validatePackedView(config.source.counts, ['uint32'], `${id} counts`);
    if (config.source.counts.length !== featureCount) {
      throw new Error(`${id} counts must have one row per feature`);
    }
  }
  validatePackedView(config.parameters, ['uint32'], `${id} parameters`);
  if (config.parameters.length < GPU_DOT_DENSITY_PARAMETER_LENGTH) {
    throw new Error(
      `${id} parameters must contain ${GPU_DOT_DENSITY_PARAMETER_LENGTH} uint32 rows`
    );
  }
  validateCapacity(id, 'maximumAttempts', config.maximumAttempts);
  if (mask) {
    validatePackedView(mask.weights, ['float32'], `${id} mask.weights`);
    if (
      !Number.isInteger(mask.width) ||
      !Number.isInteger(mask.height) ||
      mask.width < 1 ||
      mask.height < 1 ||
      mask.weights.length !== mask.width * mask.height
    ) {
      throw new Error(`${id} mask.weights must have mask.width * mask.height rows`);
    }
  }
  validatePackedView(output.positions, ['float32x2'], `${id} output.positions`);
  validateCompactOutput(id, output.dots);
  const capacity = output.positions.length;
  if (capacity < 1 || output.dots.ids.length !== capacity) {
    throw new Error(`${id} output.positions and output.dots.ids must share a positive capacity`);
  }
  if (output.categories) {
    validatePackedView(output.categories, ['uint32'], `${id} output.categories`);
    if (output.categories.length !== capacity) {
      throw new Error(`${id} output.categories must have capacity rows`);
    }
  }
  const failedCountView = output.status?.failedCount;
  if (failedCountView) {
    validatePackedView(failedCountView, ['uint32'], `${id} output.status.failedCount`);
    if (failedCountView.length < 1) {
      throw new Error(`${id} output.status.failedCount must contain one uint32 row`);
    }
  }
  for (const [name, view] of [
    ['slotCounts', output.slotCounts],
    ['slotOffsets', output.slotOffsets]
  ] as const) {
    if (view) {
      validatePackedView(view, ['uint32'], `${id} output.${name}`);
      if (view.length !== slotCount) {
        throw new Error(`${id} output.${name} must have one row per slot`);
      }
    }
  }
  validateGraphOutputsDisjointFromInputs(
    id,
    [
      output.positions,
      output.dots.ids,
      output.dots.status.count,
      output.dots.status.overflow,
      output.dots.status.requiredCount,
      output.categories,
      output.status?.failedCount,
      output.slotCounts,
      output.slotOffsets
    ],
    getDotSamplingInputs(config)
  );
  return featureCount;
}

function getDotSamplingInputs(config: DotSamplingConfig): GraphDataView[] {
  return [
    config.positions,
    config.featureOffsets,
    config.polygonOffsets,
    config.ringOffsets,
    config.source.kind === 'values' ? config.source.values : config.source.counts,
    config.parameters,
    ...(config.mask ? [config.mask.weights] : [])
  ];
}

/**
 * Builds the shared dot sampling nodes:
 * `triangulation` (GPUPolygonTriangulation) -> `triangle-areas` -> `feature-triangles` (triangle
 * range and validity per feature) -> feature area sums -> `cumulative-areas` (per-feature inclusive
 * prefix of fixed-point areas) -> `counts` (dots per slot) -> slot prefix scan -> `total` ->
 * `locate` (slot, rank and triangle range per dot, feature IDs, categories) -> `clear-failed` ->
 * `sample` (area-weighted triangle, then uniform point in the triangle) -> `publish`.
 *
 * @internal
 */
export function getDotSamplingNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  config: DotSamplingConfig,
  featureCount: number
): GPUCommandNode<Parameters>[] {
  const {id, operation, output, mask, categoryCount} = config;
  validateGraphViewsBelongToGraph(id, graph, [
    ...getDotSamplingInputs(config),
    output.positions,
    output.dots.ids,
    output.dots.status.count,
    output.dots.status.overflow,
    output.dots.status.requiredCount,
    output.categories,
    output.status?.failedCount,
    output.slotCounts,
    output.slotOffsets
  ]);
  const slotCount = featureCount * categoryCount;
  const capacity = output.positions.length;
  validateMaximumCount(
    id,
    'output.positions rows',
    capacity,
    getGPUDotDensityMaximumCapacity(graph.device)
  );
  const slotLimit = getDotSlotCountLimit(slotCount, capacity);
  const polygonCount = config.polygonOffsets.length - 1;
  const triangleCount = Math.floor(
    getPolygonTriangulationIndexCount(
      config.positions.length,
      config.ringOffsets.length - 1,
      polygonCount
    ) / 3
  );
  const nodes: GPUCommandNode<Parameters>[] = [];

  const triangleIndices = createTransientView(
    graph,
    `${id}-triangle-indices`,
    'uint32',
    Math.max(3 * triangleCount, 1)
  );
  const polygonValid = createTransientView(
    graph,
    `${id}-polygon-valid`,
    'uint32',
    Math.max(polygonCount, 1)
  );
  const triangleAreas = createTransientView(
    graph,
    `${id}-triangle-areas`,
    'float32',
    Math.max(triangleCount, 1)
  );
  const cumulativeAreas = createTransientView(
    graph,
    `${id}-cumulative-areas`,
    'uint32',
    Math.max(triangleCount, 1)
  );
  const featureTriangleOffsets = createTransientView(
    graph,
    `${id}-feature-triangle-offsets`,
    'uint32',
    featureCount + 1
  );
  const featureValid = createTransientView(graph, `${id}-feature-valid`, 'uint32', featureCount);
  const featureAreas = createTransientView(graph, `${id}-feature-areas`, 'float32', featureCount);
  const slotCounts =
    output.slotCounts ?? createTransientView(graph, `${id}-slot-counts`, 'uint32', slotCount);
  const slotOffsets =
    output.slotOffsets ?? createTransientView(graph, `${id}-slot-offsets`, 'uint32', slotCount);
  const clampFlag = createTransientView(graph, `${id}-clamped`, 'uint32', 1);
  const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
  const dotSlots = createTransientView(graph, `${id}-dot-slots`, 'uint32', capacity * DOT_WORDS);
  const failedCount =
    output.status?.failedCount ?? createTransientView(graph, `${id}-failed`, 'uint32', 1);

  // 1. Triangulate every polygon (shell first, later rings are holes, as GeoArrow positions say).
  nodes.push(
    ...new GPUPolygonTriangulation({
      id: `${id}-triangulation`,
      positions: config.positions,
      featureOffsets: config.featureOffsets,
      polygonOffsets: config.polygonOffsets,
      ringOffsets: config.ringOffsets,
      output: {indices: triangleIndices, valid: polygonValid}
    }).getCommandNodes(graph)
  );

  // 2. Area of every triangle slot; padding slots are degenerate and have area 0.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-triangle-areas`,
      operation,
      variant: 'triangle-areas',
      bindings: [
        {name: 'polygonPositions', view: config.positions, type: 'f32', access: 'read'},
        {name: 'triangleIndices', view: triangleIndices, type: 'u32', access: 'read'},
        {name: 'triangleAreas', view: triangleAreas, type: 'f32', access: 'read_write'}
      ],
      invocationCount: Math.max(triangleCount, 1),
      declarations: /* wgsl */ `
const TRIANGLE_COUNT: u32 = ${triangleCount}u;
fn readVertex(vertex: u32) -> vec2<f32> {
  return vec2<f32>(polygonPositions[polygonPositionsOffset + 2u * vertex], polygonPositions[polygonPositionsOffset + 2u * vertex + 1u]);
}`,
      body: /* wgsl */ `
  var area = 0.0;
  if (index < TRIANGLE_COUNT) {
    let base = triangleIndicesOffset + 3u * index;
    let first = readVertex(triangleIndices[base]);
    // Edge vectors from one vertex keep the cross product accurate far from the origin.
    let edgeOne = readVertex(triangleIndices[base + 1u]) - first;
    let edgeTwo = readVertex(triangleIndices[base + 2u]) - first;
    area = 0.5 * abs(edgeOne.x * edgeTwo.y - edgeOne.y * edgeTwo.x);
  }
  triangleAreas[triangleAreasOffset + index] = area;`
    })
  );

  // 3. Triangle range [offsets[f], offsets[f + 1]) of every feature and whether all of its polygons
  // triangulated. The slice of polygon p starts at triangle ringOffsets[polygonOffsets[p]] +
  // 2 * polygonOffsets[p] - 4 * p (the index prefix of GPUPolygonTriangulation divided by 3).
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-feature-triangles`,
      operation,
      variant: 'feature-triangles',
      bindings: [
        {name: 'featureOffsets', view: config.featureOffsets, type: 'u32', access: 'read'},
        {name: 'polygonOffsets', view: config.polygonOffsets, type: 'u32', access: 'read'},
        {name: 'ringOffsets', view: config.ringOffsets, type: 'u32', access: 'read'},
        {name: 'polygonValid', view: polygonValid, type: 'u32', access: 'read'},
        {
          name: 'featureTriangleOffsets',
          view: featureTriangleOffsets,
          type: 'u32',
          access: 'read_write'
        },
        {name: 'featureValid', view: featureValid, type: 'u32', access: 'read_write'}
      ],
      invocationCount: featureCount + 1,
      declarations: /* wgsl */ `
const FEATURE_COUNT: u32 = ${featureCount}u;
const POLYGON_COUNT: u32 = ${polygonCount}u;
const TRIANGLE_COUNT: u32 = ${triangleCount}u;
fn getTriangleStart(polygon: u32) -> u32 {
  let ringPrefix = polygonOffsets[polygonOffsetsOffset + polygon];
  return min(ringOffsets[ringOffsetsOffset + ringPrefix] + 2u * ringPrefix - 4u * polygon, TRIANGLE_COUNT);
}`,
      body: /* wgsl */ `
  let firstPolygon = min(featureOffsets[featureOffsetsOffset + index], POLYGON_COUNT);
  featureTriangleOffsets[featureTriangleOffsetsOffset + index] = getTriangleStart(firstPolygon);
  if (index < FEATURE_COUNT) {
    let endPolygon = min(featureOffsets[featureOffsetsOffset + index + 1u], POLYGON_COUNT);
    var valid = 1u;
    for (var polygon = firstPolygon; polygon < endPolygon; polygon = polygon + 1u) {
      valid = valid & polygonValid[polygonValidOffset + polygon];
    }
    featureValid[featureValidOffset + index] = valid;
  }`
    })
  );

  // 4. Total area per feature (deterministic segmented sum).
  nodes.push(
    createSegmentSumNode<Parameters>(graph, {
      id: `${id}-feature-areas`,
      operation,
      segmentCount: featureCount,
      input: triangleAreas,
      segmentOffsets: featureTriangleOffsets,
      output: featureAreas,
      offsetsWithinInput: true
    })
  );

  // 5. Cumulative fixed-point area inside every feature (inclusive prefix), one invocation per
  // feature in triangle order. A feature whose polygons failed to triangulate or whose area is
  // zero or not finite gets all zeros, which the sampler reports. Each triangle gets
  // floor(share * total) units, so the running sum never exceeds the fixed-point total.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-cumulative-areas`,
      operation,
      variant: 'cumulative-areas',
      bindings: [
        {name: 'triangleAreas', view: triangleAreas, type: 'f32', access: 'read'},
        {name: 'featureTriangleOffsets', view: featureTriangleOffsets, type: 'u32', access: 'read'},
        {name: 'featureValid', view: featureValid, type: 'u32', access: 'read'},
        {name: 'featureAreas', view: featureAreas, type: 'f32', access: 'read'},
        {name: 'cumulativeAreas', view: cumulativeAreas, type: 'u32', access: 'read_write'}
      ],
      invocationCount: featureCount,
      declarations: IS_FINITE_BITS_WGSL,
      body: /* wgsl */ `
  let first = featureTriangleOffsets[featureTriangleOffsetsOffset + index];
  let end = featureTriangleOffsets[featureTriangleOffsetsOffset + index + 1u];
  let featureArea = featureAreas[featureAreasOffset + index];
  let usable = featureValid[featureValidOffset + index] == 1u && isFiniteBits(featureArea) && featureArea > 0.0;
  var running = 0u;
  for (var triangle = first; triangle < end; triangle = triangle + 1u) {
    let area = triangleAreas[triangleAreasOffset + triangle];
    if (usable && isFiniteBits(area)) {
      running = running + u32(clamp(area / featureArea, 0.0, 1.0) * ${AREA_FIXED_POINT_TOTAL});
    }
    cumulativeAreas[cumulativeAreasOffset + triangle] = running;
  }`
    })
  );

  // 7. Dots per slot: ceil(value * dotsPerUnit - u) (dot density) or the integer count.
  nodes.push(
    createFillNode<Parameters>(graph, {
      id: `${id}-clear-clamped`,
      operation,
      view: clampFlag,
      type: 'u32',
      value: '0u'
    })
  );
  const sourceBinding: WGSLKernelBinding =
    config.source.kind === 'values'
      ? {
          name: 'sourceValues',
          view: config.source.values,
          type: 'f32',
          access: 'read'
        }
      : {
          name: 'sourceCounts',
          view: config.source.counts,
          type: 'u32',
          access: 'read'
        };
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-counts`,
      operation,
      variant: 'counts',
      bindings: [
        sourceBinding,
        {
          name: 'parameters',
          view: config.parameters,
          type: 'u32',
          access: 'read'
        },
        {
          name: 'slotCounts',
          view: slotCounts,
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'clampFlag',
          view: clampFlag,
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: slotCount,
      declarations: `${PHILOX_WGSL}
${IS_NAN_BITS_WGSL}
const SLOT_LIMIT: u32 = ${slotLimit}u;`,
      body:
        config.source.kind === 'values'
          ? /* wgsl */ `
  let seed = parameters[parametersOffset];
  let dotsPerUnit = bitcast<f32>(parameters[parametersOffset + 1u]);
  let scaled = sourceValues[sourceValuesOffset + index] * dotsPerUnit;
  var dots = 0u;
  var clamped = false;
  if (!isNanBits(scaled) && scaled > 0.0) {
    let remainder = philoxUnitFloat(
      philox4x32(vec4<u32>(index, 0u, 0u, 0u), vec2<u32>(seed, ${DOT_DENSITY_REMAINDER_PURPOSE}u)).x
    );
    let ceiling = ceil(scaled - remainder);
    if (ceiling > f32(SLOT_LIMIT)) {
      dots = SLOT_LIMIT;
      clamped = true;
    } else if (ceiling > 0.0) {
      dots = u32(ceiling);
      if (dots > SLOT_LIMIT) {
        dots = SLOT_LIMIT;
        clamped = true;
      }
    }
  }
  slotCounts[slotCountsOffset + index] = dots;
  if (clamped) {
    clampFlag[clampFlagOffset] = 1u;
  }`
          : /* wgsl */ `
  _ = parameters[parametersOffset];
  var dots = sourceCounts[sourceCountsOffset + index];
  if (dots > SLOT_LIMIT) {
    dots = SLOT_LIMIT;
    clampFlag[clampFlagOffset] = 1u;
  }
  slotCounts[slotCountsOffset + index] = dots;`
    })
  );

  // 8. Exclusive offsets and the total.
  nodes.push(
    ...new GPUScan({
      id: `${id}-slot-scan`,
      input: slotCounts,
      output: slotOffsets,
      mode: 'exclusive'
    }).getCommandNodes(graph)
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-total`,
      operation,
      variant: 'total',
      bindings: [
        {name: 'slotCounts', view: slotCounts, type: 'u32', access: 'read'},
        {name: 'slotOffsets', view: slotOffsets, type: 'u32', access: 'read'},
        {name: 'totalOut', view: total, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      body: `let last = ${slotCount - 1}u;
  totalOut[totalOutOffset] = slotOffsets[slotOffsetsOffset + last] + slotCounts[slotCountsOffset + last];`
    })
  );

  // 9. Slot, rank and triangle range of every published dot (binary search on the inclusive prefix).
  const locateBindings: WGSLKernelBinding[] = [
    {name: 'slotCounts', view: slotCounts, type: 'u32', access: 'read'},
    {name: 'slotOffsets', view: slotOffsets, type: 'u32', access: 'read'},
    {name: 'total', view: total, type: 'u32', access: 'read'},
    {name: 'featureTriangleOffsets', view: featureTriangleOffsets, type: 'u32', access: 'read'},
    {name: 'dotSlots', view: dotSlots, type: 'u32', access: 'read_write'},
    {
      name: 'featureIds',
      view: output.dots.ids,
      type: 'u32',
      access: 'read_write'
    }
  ];
  if (output.categories) {
    locateBindings.push({
      name: 'categories',
      view: output.categories,
      type: 'u32',
      access: 'read_write'
    });
  }
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-locate`,
      operation,
      variant: 'locate',
      bindings: locateBindings,
      invocationCount: capacity,
      declarations: `const SLOT_COUNT: u32 = ${slotCount}u;
const CATEGORY_COUNT: u32 = ${categoryCount}u;`,
      body: /* wgsl */ `
  let count = min(total[totalOffset], ${capacity}u);
  let table = dotSlotsOffset + ${DOT_WORDS}u * index;
  if (index >= count) {
    dotSlots[table] = ${NO_FEATURE};
    dotSlots[table + 1u] = 0u;
    dotSlots[table + 2u] = 0u;
    dotSlots[table + 3u] = 0u;
    featureIds[featureIdsOffset + index] = ${NO_FEATURE};
    ${output.categories ? `categories[categoriesOffset + index] = ${NO_FEATURE};` : ''}
    return;
  }
  // First slot whose inclusive prefix exceeds the dot index; empty slots never match.
  var low = 0u;
  var high = SLOT_COUNT - 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (slotOffsets[slotOffsetsOffset + middle] + slotCounts[slotCountsOffset + middle] > index) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  let feature = low / CATEGORY_COUNT;
  dotSlots[table] = low;
  dotSlots[table + 1u] = index - slotOffsets[slotOffsetsOffset + low];
  dotSlots[table + 2u] = featureTriangleOffsets[featureTriangleOffsetsOffset + feature];
  dotSlots[table + 3u] = featureTriangleOffsets[featureTriangleOffsetsOffset + feature + 1u];
  featureIds[featureIdsOffset + index] = feature;
  ${output.categories ? 'categories[categoriesOffset + index] = low % CATEGORY_COUNT;' : ''}`
    })
  );

  // 10. Area-weighted triangle, then a uniform point inside it.
  nodes.push(
    createFillNode<Parameters>(graph, {
      id: `${id}-clear-failed`,
      operation,
      view: failedCount,
      type: 'u32',
      value: '0u',
      componentCount: 1
    })
  );
  const sampleBindings: WGSLKernelBinding[] = [
    {
      name: 'polygonPositions',
      view: config.positions,
      type: 'f32',
      access: 'read'
    },
    {name: 'triangleIndices', view: triangleIndices, type: 'u32', access: 'read'},
    {name: 'cumulativeAreas', view: cumulativeAreas, type: 'u32', access: 'read'},
    {name: 'dotSlots', view: dotSlots, type: 'u32', access: 'read'},
    {
      name: 'parameters',
      view: config.parameters,
      type: 'u32',
      access: 'read'
    },
    {
      name: 'positions',
      view: output.positions,
      type: 'f32',
      access: 'read_write'
    },
    {
      name: 'failed',
      view: failedCount,
      type: 'atomic<u32>',
      access: 'read_write'
    }
  ];
  if (mask) {
    sampleBindings.push({
      name: 'maskWeights',
      view: mask.weights,
      type: 'f32',
      access: 'read'
    });
  }
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-sample`,
      operation,
      variant: mask ? 'sample-masked' : 'sample',
      bindings: sampleBindings,
      invocationCount: capacity,
      declarations: `${PHILOX_WGSL}
const CATEGORY_COUNT: u32 = ${categoryCount}u;
// Without a mask the first candidate is always kept; with one, candidates are rejected by weight.
const MAXIMUM_ATTEMPTS: u32 = ${mask ? config.maximumAttempts : 1}u;
fn readVertex(vertex: u32) -> vec2<f32> {
  return vec2<f32>(polygonPositions[polygonPositionsOffset + 2u * vertex], polygonPositions[polygonPositionsOffset + 2u * vertex + 1u]);
}
// Uniform point of a triangle: r1 = sqrt(u1), P = (1 - r1) A + r1 (1 - u2) B + r1 u2 C, written
// relative to A (the same point) for accuracy far from the origin.
fn getTrianglePoint(triangle: u32, u1: f32, u2: f32) -> vec2<f32> {
  let base = triangleIndicesOffset + 3u * triangle;
  let a = readVertex(triangleIndices[base]);
  let edgeB = readVertex(triangleIndices[base + 1u]) - a;
  let edgeC = readVertex(triangleIndices[base + 2u]) - a;
  return a + sqrt(u1) * ((1.0 - u2) * edgeB + u2 * edgeC);
}
${
  mask
    ? `${IS_NAN_BITS_WGSL}
const MASK_WIDTH: u32 = ${mask.width}u;
const MASK_HEIGHT: u32 = ${mask.height}u;
fn getMaskWeight(point: vec2<f32>) -> f32 {
  let origin = vec2<f32>(bitcast<f32>(parameters[parametersOffset + 2u]), bitcast<f32>(parameters[parametersOffset + 3u]));
  let cell = vec2<f32>(bitcast<f32>(parameters[parametersOffset + 4u]), bitcast<f32>(parameters[parametersOffset + 5u]));
  let local = (point - origin) / cell;
  if (isNanBits(local.x) || isNanBits(local.y) || local.x < 0.0 || local.y < 0.0 ||
      local.x >= f32(MASK_WIDTH) || local.y >= f32(MASK_HEIGHT)) {
    return 0.0;
  }
  let weight = maskWeights[maskWeightsOffset + u32(local.y) * MASK_WIDTH + u32(local.x)];
  return select(weight, 0.0, isNanBits(weight));
}`
    : ''
}`,
      body: /* wgsl */ `
  let table = dotSlotsOffset + ${DOT_WORDS}u * index;
  let slot = dotSlots[table];
  var nanBits = 0x7fc00000u;
  var position = vec2<f32>(bitcast<f32>(nanBits));
  if (slot != ${NO_FEATURE}) {
    let rank = dotSlots[table + 1u];
    let firstTriangle = dotSlots[table + 2u];
    let endTriangle = dotSlots[table + 3u];
    let seed = parameters[parametersOffset];
    // Fixed-point area of the whole feature; 0 when it failed to triangulate or has no area.
    var featureArea = 0u;
    if (endTriangle > firstTriangle) {
      featureArea = cumulativeAreas[cumulativeAreasOffset + endTriangle - 1u];
    }
    var accepted = false;
    if (featureArea > 0u) {
      for (var attempt = 0u; attempt < MAXIMUM_ATTEMPTS; attempt = attempt + 1u) {
        let random = philox4x32(vec4<u32>(slot, rank, attempt, 0u), vec2<u32>(seed, ${DOT_DENSITY_CANDIDATE_PURPOSE}u));
        let areaTarget = min(u32(philoxUnitFloat(random.x) * f32(featureArea)), featureArea - 1u);
        // First triangle whose cumulative area exceeds the target; zero-area triangles never match.
        var low = firstTriangle;
        var high = endTriangle - 1u;
        while (low < high) {
          let middle = (low + high) / 2u;
          if (cumulativeAreas[cumulativeAreasOffset + middle] > areaTarget) {
            high = middle;
          } else {
            low = middle + 1u;
          }
        }
        let candidate = getTrianglePoint(low, philoxUnitFloat(random.y), philoxUnitFloat(random.z));
        ${mask ? 'accepted = philoxUnitFloat(random.w) < getMaskWeight(candidate);' : 'accepted = true;'}
        if (accepted) {
          position = candidate;
          break;
        }
      }
    }
    if (!accepted) {
      atomicAdd(&failed[failedOffset], 1u);
    }
  }
  positions[positionsOffset + 2u * index] = position.x;
  positions[positionsOffset + 2u * index + 1u] = position.y;`
    })
  );

  // 11. Count, overflow (capacity or clamped slot) and the unclamped total.
  nodes.push(
    createPublishNode<Parameters>(graph, {
      id: `${id}-publish`,
      operation,
      requiredCount: total,
      output: output.dots,
      overflowSources: [clampFlag]
    })
  );
  return nodes;
}
