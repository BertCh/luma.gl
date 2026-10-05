// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  validatePackedView,
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
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {
  DOT_DENSITY_CANDIDATE_PURPOSE,
  DOT_DENSITY_REMAINDER_PURPOSE,
  getDotSlotCountLimit
} from './dot-density-cpu';
import {GPU_DOT_DENSITY_PARAMETER_LENGTH} from './dot-density-parameters';
import {PHILOX_WGSL} from './dot-density-random';

/** Feature ID written past the published count. */
const NO_FEATURE = '0xffffffffu';
/** Words per feature in the transient feature table: ring range plus f32 bounds bits. */
const FEATURE_WORDS = 6;
/** Default rejection-sampling attempts per dot. */
export const DEFAULT_DOT_MAXIMUM_ATTEMPTS = 32;

/** GeoArrow polygon columns shared by {@link GPUDotDensity} and {@link GPURandomPointsInPolygon}. */
export type GPUDotDensityPolygons = {
  /** Flattened polygon vertices (GeoArrow layout, as `GPUPointInPolygonJoin`). */
  polygonPositions: GraphDataView<'float32x2'>;
  /** Feature-to-polygon offsets with `featureCount + 1` entries, first 0. */
  featureOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets with a terminal entry. Ring 0 of a polygon is the shell; others are holes. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Ring-to-vertex offsets with a terminal entry. Rings close implicitly; a repeated first vertex is harmless. */
  ringOffsets: GraphDataView<'uint32'>;
};

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
   * Dot positions, capacity rows. Rows past `dots.count` are NaN. A dot whose rejection attempts
   * all failed is NaN too (and counted in `failedCount`), so it occupies its slot but is not drawn.
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
  /** Optional one-row count of dots whose attempts all failed. */
  failedCount?: GraphDataView<'uint32'>;
  /** Optional per-slot dot counts, one row per slot. */
  slotCounts?: GraphDataView<'uint32'>;
  /** Optional per-slot exclusive offsets into the dot list, one row per slot. */
  slotOffsets?: GraphDataView<'uint32'>;
};

/** Internal configuration of the shared dot sampling graph. @internal */
export type DotSamplingConfig = GPUDotDensityPolygons & {
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
  validatePackedView(config.polygonPositions, ['float32x2'], `${id} polygonPositions`);
  validatePackedUint32View(config.featureOffsets, `${id} featureOffsets`);
  validatePackedUint32View(config.polygonOffsets, `${id} polygonOffsets`);
  validatePackedUint32View(config.ringOffsets, `${id} ringOffsets`);
  const featureCount = config.featureOffsets.length - 1;
  if (featureCount < 1) {
    throw new Error(`${id} featureOffsets must describe at least one feature`);
  }
  if (!Number.isInteger(config.categoryCount) || config.categoryCount < 1) {
    throw new Error(`${id} categoryCount must be a positive integer`);
  }
  const slotCount = featureCount * config.categoryCount;
  if (config.source.kind === 'values') {
    validatePackedView(config.source.values, ['float32'], `${id} values`);
    if (config.source.values.length !== slotCount) {
      throw new Error(`${id} values must have featureCount * categoryCount rows`);
    }
  } else {
    validatePackedUint32View(config.source.counts, `${id} counts`);
    if (config.source.counts.length !== featureCount) {
      throw new Error(`${id} counts must have one row per feature`);
    }
  }
  validatePackedUint32View(config.parameters, `${id} parameters`);
  if (config.parameters.length < GPU_DOT_DENSITY_PARAMETER_LENGTH) {
    throw new Error(
      `${id} parameters must contain ${GPU_DOT_DENSITY_PARAMETER_LENGTH} uint32 rows`
    );
  }
  if (!Number.isInteger(config.maximumAttempts) || config.maximumAttempts < 1) {
    throw new Error(`${id} maximumAttempts must be a positive integer`);
  }
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
    validatePackedUint32View(output.categories, `${id} output.categories`);
    if (output.categories.length !== capacity) {
      throw new Error(`${id} output.categories must have capacity rows`);
    }
  }
  if (output.failedCount) {
    validatePackedUint32View(output.failedCount, `${id} output.failedCount`);
    if (output.failedCount.length < 1) {
      throw new Error(`${id} output.failedCount must contain one uint32 row`);
    }
  }
  for (const [name, view] of [
    ['slotCounts', output.slotCounts],
    ['slotOffsets', output.slotOffsets]
  ] as const) {
    if (view) {
      validatePackedUint32View(view, `${id} output.${name}`);
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
      output.dots.count,
      output.dots.overflow,
      output.dots.totalCount,
      output.categories,
      output.failedCount,
      output.slotCounts,
      output.slotOffsets
    ],
    getDotSamplingInputs(config)
  );
  return featureCount;
}

function getDotSamplingInputs(config: DotSamplingConfig): GraphDataView[] {
  return [
    config.polygonPositions,
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
 * `features` (ring range and bounds per feature) -> `counts` (dots per slot) -> prefix scan ->
 * `total` -> `locate` (slot and rank per dot, feature IDs, categories) -> `clear-failed` ->
 * `sample` (rejection sampling) -> `publish`.
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
    output.dots.count,
    output.dots.overflow,
    output.dots.totalCount,
    output.categories,
    output.failedCount,
    output.slotCounts,
    output.slotOffsets
  ]);
  const slotCount = featureCount * categoryCount;
  const capacity = output.positions.length;
  const slotLimit = getDotSlotCountLimit(slotCount, capacity);
  const nodes: GPUCommandNode<Parameters>[] = [];

  const featureTable = createTransientView(
    graph,
    `${id}-feature-table`,
    'uint32',
    featureCount * FEATURE_WORDS
  );
  const slotCounts =
    output.slotCounts ?? createTransientView(graph, `${id}-slot-counts`, 'uint32', slotCount);
  const slotOffsets =
    output.slotOffsets ?? createTransientView(graph, `${id}-slot-offsets`, 'uint32', slotCount);
  const clampFlag = createTransientView(graph, `${id}-clamped`, 'uint32', 1);
  const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
  const dotSlots = createTransientView(graph, `${id}-dot-slots`, 'uint32', capacity * 2);
  const failedCount = output.failedCount ?? createTransientView(graph, `${id}-failed`, 'uint32', 1);

  // 1. Ring range and vertex bounds per feature.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-features`,
      operation,
      variant: 'features',
      bindings: [
        {
          name: 'polygonPositions',
          view: config.polygonPositions,
          type: 'f32',
          access: 'read'
        },
        {
          name: 'featureOffsets',
          view: config.featureOffsets,
          type: 'u32',
          access: 'read'
        },
        {
          name: 'polygonOffsets',
          view: config.polygonOffsets,
          type: 'u32',
          access: 'read'
        },
        {
          name: 'ringOffsets',
          view: config.ringOffsets,
          type: 'u32',
          access: 'read'
        },
        {
          name: 'featureTable',
          view: featureTable,
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: featureCount,
      body: /* wgsl */ `
  let ringStart = polygonOffsets[polygonOffsetsOffset + featureOffsets[featureOffsetsOffset + index]];
  let ringEnd = polygonOffsets[polygonOffsetsOffset + featureOffsets[featureOffsetsOffset + index + 1u]];
  let vertexStart = ringOffsets[ringOffsetsOffset + ringStart];
  let vertexEnd = ringOffsets[ringOffsetsOffset + ringEnd];
  var minimum = vec2<f32>(bitcast<f32>(0x7f7fffffu));
  var maximum = vec2<f32>(bitcast<f32>(0xff7fffffu));
  for (var vertex = vertexStart; vertex < vertexEnd; vertex = vertex + 1u) {
    let point = vec2<f32>(
      polygonPositions[polygonPositionsOffset + 2u * vertex],
      polygonPositions[polygonPositionsOffset + 2u * vertex + 1u]
    );
    minimum = min(minimum, point);
    maximum = max(maximum, point);
  }
  let base = featureTableOffset + index * ${FEATURE_WORDS}u;
  featureTable[base] = ringStart;
  featureTable[base + 1u] = ringEnd;
  featureTable[base + 2u] = bitcast<u32>(minimum.x);
  featureTable[base + 3u] = bitcast<u32>(minimum.y);
  featureTable[base + 4u] = bitcast<u32>(maximum.x);
  featureTable[base + 5u] = bitcast<u32>(maximum.y);`
    })
  );

  // 2. Dots per slot: ceil(value * dotsPerUnit - u) (dot density) or the integer count.
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
const SLOT_LIMIT: u32 = ${slotLimit}u;
fn isNanFloat(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }`,
      body:
        config.source.kind === 'values'
          ? /* wgsl */ `
  let seed = parameters[parametersOffset];
  let dotsPerUnit = bitcast<f32>(parameters[parametersOffset + 1u]);
  let scaled = sourceValues[sourceValuesOffset + index] * dotsPerUnit;
  var dots = 0u;
  var clamped = false;
  if (!isNanFloat(scaled) && scaled > 0.0) {
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

  // 3. Exclusive offsets and the total.
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

  // 4. Slot and rank of every published dot (binary search on the inclusive prefix).
  const locateBindings: WGSLKernelBinding[] = [
    {name: 'slotCounts', view: slotCounts, type: 'u32', access: 'read'},
    {name: 'slotOffsets', view: slotOffsets, type: 'u32', access: 'read'},
    {name: 'total', view: total, type: 'u32', access: 'read'},
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
  if (index >= count) {
    dotSlots[dotSlotsOffset + 2u * index] = ${NO_FEATURE};
    dotSlots[dotSlotsOffset + 2u * index + 1u] = 0u;
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
  dotSlots[dotSlotsOffset + 2u * index] = low;
  dotSlots[dotSlotsOffset + 2u * index + 1u] = index - slotOffsets[slotOffsetsOffset + low];
  featureIds[featureIdsOffset + index] = low / CATEGORY_COUNT;
  ${output.categories ? 'categories[categoriesOffset + index] = low % CATEGORY_COUNT;' : ''}`
    })
  );

  // 5. Rejection sampling inside each dot's feature.
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
      view: config.polygonPositions,
      type: 'f32',
      access: 'read'
    },
    {
      name: 'ringOffsets',
      view: config.ringOffsets,
      type: 'u32',
      access: 'read'
    },
    {name: 'featureTable', view: featureTable, type: 'u32', access: 'read'},
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
const MAXIMUM_ATTEMPTS: u32 = ${config.maximumAttempts}u;
fn isNanFloat(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
fn readVertex(vertex: u32) -> vec2<f32> {
  return vec2<f32>(polygonPositions[polygonPositionsOffset + 2u * vertex], polygonPositions[polygonPositionsOffset + 2u * vertex + 1u]);
}
fn isInsideFeature(point: vec2<f32>, ringStart: u32, ringEnd: u32) -> bool {
  var inside = false;
  for (var ring = ringStart; ring < ringEnd; ring = ring + 1u) {
    let first = ringOffsets[ringOffsetsOffset + ring];
    let end = ringOffsets[ringOffsetsOffset + ring + 1u];
    if (end <= first) {
      continue;
    }
    var previous = readVertex(end - 1u);
    for (var vertex = first; vertex < end; vertex = vertex + 1u) {
      let current = readVertex(vertex);
      if ((current.y > point.y) != (previous.y > point.y)) {
        let t = (point.y - current.y) / (previous.y - current.y);
        let crossX = current.x + t * (previous.x - current.x);
        if (point.x < crossX) {
          inside = !inside;
        }
      }
      previous = current;
    }
  }
  return inside;
}
${
  mask
    ? `const MASK_WIDTH: u32 = ${mask.width}u;
const MASK_HEIGHT: u32 = ${mask.height}u;
fn getMaskWeight(point: vec2<f32>) -> f32 {
  let origin = vec2<f32>(bitcast<f32>(parameters[parametersOffset + 2u]), bitcast<f32>(parameters[parametersOffset + 3u]));
  let cell = vec2<f32>(bitcast<f32>(parameters[parametersOffset + 4u]), bitcast<f32>(parameters[parametersOffset + 5u]));
  let local = (point - origin) / cell;
  if (isNanFloat(local.x) || isNanFloat(local.y) || local.x < 0.0 || local.y < 0.0 ||
      local.x >= f32(MASK_WIDTH) || local.y >= f32(MASK_HEIGHT)) {
    return 0.0;
  }
  let weight = maskWeights[maskWeightsOffset + u32(local.y) * MASK_WIDTH + u32(local.x)];
  return select(weight, 0.0, isNanFloat(weight));
}`
    : ''
}`,
      body: /* wgsl */ `
  let slot = dotSlots[dotSlotsOffset + 2u * index];
  var nanBits = 0x7fc00000u;
  var position = vec2<f32>(bitcast<f32>(nanBits));
  if (slot != ${NO_FEATURE}) {
    let rank = dotSlots[dotSlotsOffset + 2u * index + 1u];
    let seed = parameters[parametersOffset];
    let base = featureTableOffset + (slot / CATEGORY_COUNT) * ${FEATURE_WORDS}u;
    let ringStart = featureTable[base];
    let ringEnd = featureTable[base + 1u];
    let minimum = vec2<f32>(bitcast<f32>(featureTable[base + 2u]), bitcast<f32>(featureTable[base + 3u]));
    let span = vec2<f32>(bitcast<f32>(featureTable[base + 4u]), bitcast<f32>(featureTable[base + 5u])) - minimum;
    var accepted = false;
    for (var attempt = 0u; attempt < MAXIMUM_ATTEMPTS; attempt = attempt + 1u) {
      let random = philox4x32(vec4<u32>(slot, rank, attempt, 0u), vec2<u32>(seed, ${DOT_DENSITY_CANDIDATE_PURPOSE}u));
      let candidate = minimum + vec2<f32>(philoxUnitFloat(random.x), philoxUnitFloat(random.y)) * span;
      var inside = isInsideFeature(candidate, ringStart, ringEnd);
      ${mask ? 'inside = inside && philoxUnitFloat(random.z) < getMaskWeight(candidate);' : ''}
      if (inside) {
        position = candidate;
        accepted = true;
        break;
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

  // 6. Count, overflow (capacity or clamped slot) and the unclamped total.
  nodes.push(
    createPublishNode<Parameters>(graph, {
      id: `${id}-publish`,
      operation,
      totalCount: total,
      output: output.dots,
      overflowSources: [clampFlag]
    })
  );
  return nodes;
}
