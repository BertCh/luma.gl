// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUCalendarBuckets} from '../../gpu-dataframe/calendar-buckets/index';
import {GPUColorScale} from '../../gpu-dataframe/column-classification/index';
import {GPUGroupStatistics} from '../../gpu-dataframe/group-statistics/index';
import {GPU_EMERGING_HOT_SPOT_CATEGORIES, GPUEmergingHotSpots} from '../emerging-hot-spots/index';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../utils/wgsl-kernel-nodes';
import type {GPUSpatialWeights} from '../spatial-weights/index';
import {
  assertRecipe,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeOverrides,
  type GPURecipeResult
} from './recipe-utils';

const ID = 'addSpaceTimeHotSpotsRecipe';

/** Calendar column that defines the time slices of the cube. */
export type GPUSpaceTimeSliceField =
  | 'year'
  | 'month'
  | 'dayOfMonth'
  | 'hour'
  | 'minute'
  | 'weekday'
  | 'dayOfYear'
  | 'isoWeek'
  | 'quarter';

/** Slice definition: slice `s` holds the rows whose calendar `field` equals `firstValue + s`. */
export type GPUSpaceTimeSlices = {
  /** Calendar column read from `GPUCalendarBuckets`. */
  field: GPUSpaceTimeSliceField;
  /** Calendar value of slice 0, for example 1 for January with `'month'`. */
  firstValue: number;
  /** Number of slices (compile-time, at most 256). */
  count: number;
};

/** Cells given as a precomputed index per event. Spatial neighbors then come from `weights`. */
export type GPUSpaceTimeCellIds = {
  kind: 'ids';
  /** Cell index per event (`0xffffffff` or `>= cellCount` skips the event). */
  cellIds: GraphDataView<'uint32'>;
  /** Number of cells (rows of `weights`). */
  cellCount: number;
  /** Square weights over the cells, used as given (Gi* neighborhoods). */
  weights: GPUSpatialWeights;
  /** Weight of the focal cell. Defaults to 1. */
  selfWeight?: number;
};

/** Cells given as a regular lattice over planar event positions (lattice neighborhood). */
export type GPUSpaceTimeLattice = {
  kind: 'lattice';
  /** Planar position per event. */
  positions: GraphDataView<'float32x2'>;
  /** Lattice columns. */
  width: number;
  /** Lattice rows. */
  height: number;
  /** `[minX, minY, maxX, maxY]` covered by the lattice; events outside are skipped. */
  bounds: readonly [number, number, number, number];
  /** Largest per-frame radius in cells (compile-time, default 4). */
  maximumRadius?: number;
};

/** Properties for {@link addSpaceTimeHotSpotsRecipe}. */
export type GPUSpaceTimeHotSpotsRecipeProps = GPURecipeOverrides<
  Record<never, never>,
  {
    cube?: GraphDataView<'uint32'>;
    giZScores?: GraphDataView<'float32'>;
    trendZ?: GraphDataView<'float32'>;
    trendP?: GraphDataView<'float32'>;
    trendS?: GraphDataView<'sint32'>;
    category?: GraphDataView<'uint32'>;
    hotSliceCount?: GraphDataView<'uint32'>;
    coldSliceCount?: GraphDataView<'uint32'>;
    colors?: GraphDataView<'uint32'>;
    /** Group-statistics overflow flag; setting it selects the sorted counting path. */
    groupOverflow?: GraphDataView<'uint32'>;
  },
  {
    calendarField?: GraphDataView<'sint32'> | GraphDataView<'uint32'>;
    /** Per-event cube key; setting it selects the sorted counting path. */
    keys?: GraphDataView<'uint32'>;
    /** Per-event key validity; setting it selects the sorted counting path. */
    keyMask?: GraphDataView<'uint32'>;
    /** Distinct keys in sorted order; setting it selects the sorted counting path. */
    groupKeys?: GraphDataView<'uint32'>;
    /** Event count per distinct key; setting it selects the sorted counting path. */
    groupCounts?: GraphDataView<'uint32'>;
    /** Number of distinct keys; setting it selects the sorted counting path. */
    groupCount?: GraphDataView<'uint32'>;
    colorDomain?: GraphDataView<'float32'>;
  }
> & {
  /** Prefix for every node and transient ID. Defaults to `'space-time-hot-spots'`. */
  id?: string;
  /** Epoch-millisecond event times as little-endian Int64 words. */
  timestamps: GraphDataView<'uint32x2'>;
  /** Optional per-event mask. */
  mask?: GraphDataView<'uint32'>;
  /** Optional per-event UTC offset in minutes. */
  utcOffsets?: GraphDataView<'sint32'>;
  /** `getGPUCalendarBucketsParameterValues` view (`sint32`). */
  calendarParameters: GraphDataView<'sint32'>;
  /** Time slices of the cube. */
  slices: GPUSpaceTimeSlices;
  /** Spatial cells and neighborhood. */
  cells: GPUSpaceTimeCellIds | GPUSpaceTimeLattice;
  /** Optional per-cell mask (one row per cell). */
  cellMask?: GraphDataView<'uint32'>;
  /** `getGPUEmergingHotSpotParameterValues` view. */
  parameters: GraphDataView<'float32'>;
  /** Category colors; skipped when absent. */
  color?: {
    /** Packed rgba8 palette indexed by category code, at least 17 entries. */
    palette: GraphDataView<'uint32'>;
    /** `getGPUColorScaleParameterValues({scale: 'ordinal', ...})` view. */
    parameters: GraphDataView<'float32'>;
    /** Compile-time palette bound. Defaults to the 17 categories. */
    maximumPaletteCount?: number;
  };
};

/** Named outputs of {@link addSpaceTimeHotSpotsRecipe}. */
export type GPUSpaceTimeHotSpotsRecipeResult = GPURecipeResult & {
  cellCount: number;
  sliceCount: number;
  cube: GraphDataView<'uint32'>;
  giZScores: GraphDataView<'float32'>;
  trendZ: GraphDataView<'float32'>;
  trendP: GraphDataView<'float32'>;
  trendS: GraphDataView<'sint32'>;
  category: GraphDataView<'uint32'>;
  hotSliceCount: GraphDataView<'uint32'>;
  coldSliceCount: GraphDataView<'uint32'>;
  /** Present when `color` was requested. */
  colors?: GraphDataView<'uint32'>;
};

/**
 * Space-time hot spot recipe: timestamped events to an ArcGIS-style emerging hot spot map.
 *
 * Chain: `GPUCalendarBuckets` decodes event times, an adapter kernel bins every event into
 * `cell * sliceCount + slice` of a zeroed dense `cell x slice` cube (`uint32`, the layout
 * `GPUEmergingHotSpots` reads) with an integer `atomicAdd`, `GPUEmergingHotSpots` classifies every
 * cell, and `GPUColorScale` (ordinal) colors the categories.
 *
 * Passing any of `scratch.keys`, `scratch.keyMask`, `scratch.groupKeys`, `scratch.groupCounts`,
 * `scratch.groupCount` or `outputs.groupOverflow` selects the sorted path instead: an adapter
 * writes the key per event, `GPUGroupStatistics` counts the events per key, and an adapter
 * scatters the sparse counts into the cube. Only that path fills those views, adds the
 * `GPUGroupStatistics` contributor and reports the `cube-counts` status stage. Both paths produce
 * the same cube, bit for bit.
 *
 * The roadmap chain names `GPUCellAggregation` for the per-slice counts. Its keys must be valid
 * Quadbin or H3 cells and cannot carry a slice index, so the recipe counts with
 * `GPUGroupStatistics` over a lattice or caller-supplied cell index instead; the
 * `GPUPointToCell` + `GPUCellAggregation` route to a cube would need a cell-key to row lookup.
 */
export function addSpaceTimeHotSpotsRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUSpaceTimeHotSpotsRecipeProps
): GPUSpaceTimeHotSpotsRecipeResult {
  const id = props.id ?? 'space-time-hot-spots';
  const builder = new RecipeBuilder(graph);
  const outputOverrides = props.outputs ?? {};
  const scratch = props.scratch ?? {};
  const {slices, cells} = props;
  const eventCount = props.timestamps.length;
  const sliceCount = slices.count;
  assertRecipe(
    ID,
    Number.isInteger(sliceCount) && sliceCount >= 1,
    'slices.count must be a positive integer'
  );
  const cellCount = cells.kind === 'ids' ? cells.cellCount : cells.width * cells.height;
  const cubeLength = cellCount * sliceCount;
  assertRecipe(ID, cubeLength < 0xfffffffe, 'cellCount * sliceCount is too large for a u32 key');

  // 1. Calendar columns of the event times.
  const calendarOutput: Record<string, GraphDataView<'sint32'> | GraphDataView<'uint32'>> = {};
  const fieldIsSigned = slices.field === 'year';
  const calendarView = fieldIsSigned
    ? getOrCreateView(
        graph,
        `${id}-calendar-field`,
        'sint32',
        eventCount,
        scratch.calendarField?.format === 'sint32' ? scratch.calendarField : undefined
      )
    : getOrCreateView(
        graph,
        `${id}-calendar-field`,
        'uint32',
        eventCount,
        scratch.calendarField?.format === 'uint32' ? scratch.calendarField : undefined
      );
  calendarOutput[slices.field] = calendarView;
  builder.add(
    new GPUCalendarBuckets({
      id: `${id}-calendar`,
      timestamps: props.timestamps,
      mask: props.mask,
      utcOffsets: props.utcOffsets,
      parameters: props.calendarParameters,
      output: calendarOutput
    })
  );

  // 2. Event -> (cell, slice) and validity, shared by both counting paths so they bin every event
  // identically.
  const cellBindings =
    cells.kind === 'ids'
      ? [{name: 'cellIds', view: cells.cellIds, type: 'u32', access: 'read'} as const]
      : [{name: 'eventPositions', view: cells.positions, type: 'f32', access: 'read'} as const];
  const cellSource =
    cells.kind === 'ids'
      ? `let cell = cellIds[cellIdsOffset + index];
  let cellValid = cell < ${cellCount}u;`
      : `let px = eventPositions[eventPositionsOffset + 2u * index];
  let py = eventPositions[eventPositionsOffset + 2u * index + 1u];
  let fx = floor((px - ${getWGSLFloatLiteral(cells.bounds[0])}) / ${getWGSLFloatLiteral((cells.bounds[2] - cells.bounds[0]) / cells.width)});
  let fy = floor((py - ${getWGSLFloatLiteral(cells.bounds[1])}) / ${getWGSLFloatLiteral((cells.bounds[3] - cells.bounds[1]) / cells.height)});
  let cellValid = fx >= 0.0 && fy >= 0.0 && fx < ${cells.width}.0 && fy < ${cells.height}.0;
  let cell = select(0u, u32(fy) * ${cells.width}u + u32(fx), cellValid);`;
  const eventBinSource = `${cellSource}
  let value = i32(calendarField[calendarFieldOffset + index]);
  let slice = value - ${Math.trunc(slices.firstValue)};
  let sliceValid = value != bitcast<i32>(0x80000000u) && slice >= 0 && slice < ${sliceCount};
  let valid = cellValid && sliceValid;`;
  const calendarBinding = {
    name: 'calendarField',
    view: calendarView,
    type: fieldIsSigned ? 'i32' : 'u32',
    access: 'read'
  } as const;

  // 3. Dense cube of event counts, zeroed first: both paths only write bins that hold events.
  const cube = getOrCreateView(graph, `${id}-cube`, 'uint32', cubeLength, outputOverrides.cube);
  graph.add(
    createFillNode<Parameters>(graph, {
      id: `${id}-cube-clear`,
      operation: 'GPUSpaceTimeHotSpotsCube',
      view: cube,
      type: 'u32',
      value: '0u'
    })
  );

  // Every cube value is an integer count, so a `u32` atomic increment per event builds the cube
  // exactly and deterministically (integer addition commutes), with no sort. The sorted
  // `GPUGroupStatistics` path runs only when the caller asks for its intermediates (keys, key
  // mask, group keys, counts or count) or for `outputs.groupOverflow`. Both paths write identical
  // cubes: counts never exceed `eventCount`, which is far below 2^32. `GPUEmergingHotSpots` reads
  // a count as `f32(count)`, exact below 2^24 events per bin, the same in both paths.
  const useGroupStatistics =
    scratch.keys !== undefined ||
    scratch.keyMask !== undefined ||
    scratch.groupKeys !== undefined ||
    scratch.groupCounts !== undefined ||
    scratch.groupCount !== undefined ||
    outputOverrides.groupOverflow !== undefined;
  let intermediates: GPUSpaceTimeHotSpotsRecipeResult['intermediates'] = {
    calendarField: calendarView
  };
  let status: GPUSpaceTimeHotSpotsRecipeResult['status'] = {stages: []};
  if (!useGroupStatistics) {
    graph.add(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-cube-count`,
        operation: 'GPUSpaceTimeHotSpotsCube',
        variant: 'atomic-count',
        bindings: [
          calendarBinding,
          ...cellBindings,
          {name: 'cubeOutput', view: cube, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: eventCount,
        body: `${eventBinSource}
  if (valid) {
    atomicAdd(&cubeOutput[cubeOutputOffset + cell * ${sliceCount}u + u32(slice)], 1u);
  }`
      })
    );
  } else {
    // Sorted path: key `cell * sliceCount + slice` per event, sparse counts per key, scatter.
    const keys = getOrCreateView(graph, `${id}-keys`, 'uint32', eventCount, scratch.keys);
    const keyMask = getOrCreateView(graph, `${id}-key-mask`, 'uint32', eventCount, scratch.keyMask);
    graph.add(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-keys`,
        operation: 'GPUSpaceTimeHotSpotsKeys',
        bindings: [
          calendarBinding,
          ...cellBindings,
          {name: 'keyOutput', view: keys, type: 'u32', access: 'read_write'},
          {name: 'keyMaskOutput', view: keyMask, type: 'u32', access: 'read_write'}
        ],
        invocationCount: eventCount,
        body: `${eventBinSource}
  keyOutput[keyOutputOffset + index] = select(0xffffffffu, cell * ${sliceCount}u + u32(slice), valid);
  keyMaskOutput[keyMaskOutputOffset + index] = select(0u, 1u, valid);`
      })
    );
    const groupKeys = getOrCreateView(
      graph,
      `${id}-group-keys`,
      'uint32',
      cubeLength,
      scratch.groupKeys
    );
    const groupCounts = getOrCreateView(
      graph,
      `${id}-group-counts`,
      'uint32',
      cubeLength,
      scratch.groupCounts
    );
    const groupCount = getOrCreateView(graph, `${id}-group-count`, 'uint32', 1, scratch.groupCount);
    const groupOverflow = getOrCreateView(
      graph,
      `${id}-group-overflow`,
      'uint32',
      1,
      outputOverrides.groupOverflow
    );
    builder.add(
      new GPUGroupStatistics({
        id: `${id}-counts`,
        keys,
        mask: keyMask,
        columns: [],
        output: {keys: groupKeys, counts: groupCounts, count: groupCount, overflow: groupOverflow}
      })
    );
    graph.add(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-cube-scatter`,
        operation: 'GPUSpaceTimeHotSpotsCube',
        variant: 'scatter',
        bindings: [
          {name: 'groupKeys', view: groupKeys, type: 'u32', access: 'read'},
          {name: 'groupCounts', view: groupCounts, type: 'u32', access: 'read'},
          {name: 'groupCountScalar', view: groupCount, type: 'u32', access: 'read'},
          {name: 'cubeOutput', view: cube, type: 'u32', access: 'read_write'}
        ],
        // Distinct groups never outnumber the events, so the scatter needs no more threads.
        invocationCount: Math.min(cubeLength, eventCount),
        body: `if (index < groupCountScalar[groupCountScalarOffset]) {
    let key = groupKeys[groupKeysOffset + index];
    if (key < ${cubeLength}u) {
      cubeOutput[cubeOutputOffset + key] = groupCounts[groupCountsOffset + index];
    }
  }`
      })
    );
    intermediates = {...intermediates, keys, keyMask, groupKeys, groupCounts, groupCount};
    status = {
      stages: [{stage: 'cube-counts', status: {count: groupCount, overflow: groupOverflow}}]
    };
  }

  // 4. Emerging hot spots.
  const outputs = {
    giZScores: getOrCreateView(
      graph,
      `${id}-gi-z`,
      'float32',
      cubeLength,
      outputOverrides.giZScores
    ),
    trendZ: getOrCreateView(graph, `${id}-trend-z`, 'float32', cellCount, outputOverrides.trendZ),
    trendP: getOrCreateView(graph, `${id}-trend-p`, 'float32', cellCount, outputOverrides.trendP),
    trendS: getOrCreateView(graph, `${id}-trend-s`, 'sint32', cellCount, outputOverrides.trendS),
    category: getOrCreateView(
      graph,
      `${id}-category`,
      'uint32',
      cellCount,
      outputOverrides.category
    ),
    hotSliceCount: getOrCreateView(
      graph,
      `${id}-hot-slices`,
      'uint32',
      cellCount,
      outputOverrides.hotSliceCount
    ),
    coldSliceCount: getOrCreateView(
      graph,
      `${id}-cold-slices`,
      'uint32',
      cellCount,
      outputOverrides.coldSliceCount
    )
  };
  builder.add(
    new GPUEmergingHotSpots({
      id: `${id}-emerging`,
      values: cube,
      ...(cells.kind === 'ids'
        ? {weights: cells.weights, selfWeight: cells.selfWeight}
        : {
            gridWidth: cells.width,
            gridHeight: cells.height,
            maximumRadius: cells.maximumRadius
          }),
      sliceCount,
      mask: props.cellMask,
      parameters: props.parameters,
      ...outputs
    })
  );

  const result: GPUSpaceTimeHotSpotsRecipeResult = {
    contributors: builder.contributors,
    cellCount,
    sliceCount,
    cube,
    ...outputs,
    outputs: {cube, ...outputs},
    intermediates,
    status
  };

  // 5. Category colors (ordinal scale).
  if (props.color) {
    const {color} = props;
    const maximumPaletteCount =
      color.maximumPaletteCount ?? Object.keys(GPU_EMERGING_HOT_SPOT_CATEGORIES).length;
    const colors = getOrCreateView(
      graph,
      `${id}-colors`,
      'uint32',
      cellCount,
      outputOverrides.colors
    );
    builder.add(
      new GPUColorScale({
        id: `${id}-color-scale`,
        values: outputs.category,
        domain: getOrCreateView(graph, `${id}-color-domain`, 'float32', 2, scratch.colorDomain),
        palette: color.palette,
        parameters: color.parameters,
        maximumDomainCount: 2,
        maximumPaletteCount,
        output: {colors}
      })
    );
    result.colors = colors;
    result.outputs['colors'] = colors;
  }
  return result;
}
