// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUCalendarBuckets} from '../../gpu-dataframe/calendar-buckets/index';
import {GPUColorScale} from '../../gpu-dataframe/column-classification/index';
import {GPUGroupStatistics} from '../../gpu-dataframe/group-statistics/index';
import type {
  GPUInferenceMetadata,
  GPUMultipleTestingMethod
} from '../../analysis-core/contracts/inference';
import {
  GPU_EMERGING_HOT_SPOTS_CATEGORIES,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
  GPUEmergingHotSpots,
  type GPUEmergingHotSpotsSettings
} from '../emerging-hot-spots/index';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {GPUStatusScalar} from '../../analysis-core/contracts/status';
import {GPUParameterBuffer} from '../../utils/gpu-contributor-utils';
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

/** Calendar unit of one time step of the cube. */
export type GPUSpaceTimeStep = 'year' | 'quarter' | 'month' | 'week' | 'day' | 'hour';

/**
 * First instant of slice 0 in the local calendar (the same UTC-offset semantics as
 * `GPUCalendarBuckets`). Unset fields default to the start of the step: month 1, day 1, hour 0.
 */
export type GPUSpaceTimeOrigin = {
  year: number;
  /** 1 to 12. Used by every step finer than `'year'`. */
  month?: number;
  /** Day of the month, 1 to 31. Used by `'week'`, `'day'` and `'hour'`; a week starts on this day. */
  day?: number;
  /** 0 to 23. Used by `'hour'`. */
  hour?: number;
};

/**
 * Absolute time-step slicing of the cube (ESRI "time step interval"). Slice
 * `floor((stepIndex(t) - stepIndex(origin)) / (interval * stepsPerUnit))` holds the events at local
 * time `t`, where `stepIndex` counts whole calendar steps (`year`, `year * 4 + quarter - 1`,
 * `year * 12 + month - 1`, days since the epoch for `'day'` and `'week'`, hours since the epoch for
 * `'hour'`) and a `'week'` is 7 days anchored to the origin day. Events before the origin or past
 * the last slice are skipped and counted in the `slice-range` `invalidCount` status.
 *
 * Cyclic calendar fields (month of year, weekday, hour of day, day of year) are not accepted:
 * `month = 1` would pool every January of a multi-year range, and a Mann-Kendall trend over hour
 * or weekday is meaningless. Cyclic profiles belong in a non-trend summary such as the
 * `hourWeekdayCounts` of `GPUCalendarBuckets`.
 */
export type GPUSpaceTimeSlices = {
  /** Calendar unit of one step. */
  step: GPUSpaceTimeStep;
  /** Steps per slice, a positive integer. Defaults to 1. */
  interval?: number;
  /** Start of slice 0. */
  origin: GPUSpaceTimeOrigin;
  /** Number of slices (compile-time, at most 256). */
  count: number;
};

/**
 * Days since 1970-01-01 of a proleptic Gregorian date (Hinnant's `days_from_civil`), in integer
 * arithmetic with truncating division, the same formula the adapter kernel evaluates.
 */
export function getSpaceTimeDaysFromCivil(year: number, month: number, day: number): number {
  const adjustedYear = month <= 2 ? year - 1 : year;
  const era = Math.trunc((adjustedYear >= 0 ? adjustedYear : adjustedYear - 399) / 400);
  const yearOfEra = adjustedYear - era * 400;
  const shiftedMonth = (month + 9) % 12;
  const dayOfYear = Math.trunc((153 * shiftedMonth + 2) / 5) + day - 1;
  const dayOfEra =
    yearOfEra * 365 + Math.trunc(yearOfEra / 4) - Math.trunc(yearOfEra / 100) + dayOfYear;
  return era * 146097 + dayOfEra - 719468;
}

/** Whole-step index of a local calendar time for `step` (see {@link GPUSpaceTimeSlices}). */
export function getSpaceTimeStepIndex(
  step: GPUSpaceTimeStep,
  fields: {year: number; month?: number; day?: number; hour?: number}
): number {
  const {year, month = 1, day = 1, hour = 0} = fields;
  switch (step) {
    case 'year':
      return year;
    case 'quarter':
      return year * 4 + Math.trunc((month - 1) / 3);
    case 'month':
      return year * 12 + month - 1;
    case 'week':
    case 'day':
      return getSpaceTimeDaysFromCivil(year, month, day);
    case 'hour':
      return getSpaceTimeDaysFromCivil(year, month, day) * 24 + hour;
  }
}

/**
 * Slice index of a local calendar time, or `-1` when it falls before the origin or past the last
 * slice. The CPU twin of the adapter kernel, used by tests and by callers that bin on the CPU.
 */
export function getSpaceTimeSliceIndex(
  slices: GPUSpaceTimeSlices,
  fields: {year: number; month?: number; day?: number; hour?: number}
): number {
  const difference =
    getSpaceTimeStepIndex(slices.step, fields) - getSpaceTimeStepIndex(slices.step, slices.origin);
  const slice = Math.trunc(difference / getSliceDivisor(slices));
  return difference >= 0 && slice < slices.count ? slice : -1;
}

function getSliceDivisor(slices: GPUSpaceTimeSlices): number {
  return (slices.interval ?? 1) * (slices.step === 'week' ? 7 : 1);
}

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
  /** Planar position per event in the coordinates of `spatialContext`. */
  positions: GraphDataView<'float32x2'>;
  /** Planar context of positions and bounds. Omitted means native planar. */
  spatialContext?: GPUSpatialContext;
  /** Lattice columns. */
  width: number;
  /** Lattice rows. */
  height: number;
  /** `[minX, minY, maxX, maxY]` covered by the lattice; events outside are skipped. */
  bounds: readonly [number, number, number, number];
  /** Largest per-frame radius in cells (compile-time, default 4). */
  maximumRadius?: number;
};

/**
 * Per-frame settings of {@link addSpaceTimeHotSpotsRecipe}: the `GPUEmergingHotSpots` settings
 * without `multipleTesting`, which the recipe takes as a compile-time prop.
 */
export type GPUSpaceTimeHotSpotsRecipeSettings = Omit<
  GPUEmergingHotSpotsSettings,
  'multipleTesting'
>;

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
    /** Per-bin two-sided Gi* p-values, cube layout. */
    pValues?: GraphDataView<'float32'>;
    /** Per-bin `multipleTesting`-adjusted p-values, cube layout. */
    adjustedPValues?: GraphDataView<'float32'>;
    /** Per-cell last finite slice index (`0xffffffff` for none). */
    lastValidSlice?: GraphDataView<'uint32'>;
    colors?: GraphDataView<'uint32'>;
    /** Status words of the `slice-range` and `cube-counts` stages. */
    status?: {
      /**
       * One-row count of events that have a valid cell and time but fall before `slices.origin` or
       * past the last slice. It is the `invalidCount` of the `slice-range` status stage.
       */
      skippedEventCount?: GPUStatusScalar;
      /** Group-statistics overflow flag; setting it selects the sorted counting path. */
      groupOverflow?: GPUStatusScalar;
      /**
       * Group-statistics unclamped distinct key count (stage `cube-counts`). Only written on the
       * sorted counting path; unlike `groupOverflow` it does not select that path.
       */
      groupRequiredCount?: GPUStatusScalar;
    };
  },
  {
    /** Calendar columns the step needs (year always; month, day of month and hour as needed). */
    calendarYear?: GraphDataView<'sint32'>;
    calendarMonth?: GraphDataView<'uint32'>;
    calendarDayOfMonth?: GraphDataView<'uint32'>;
    calendarHour?: GraphDataView<'uint32'>;
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
  /**
   * Per-frame emerging hot spot settings (`temporalWindow`, `radius`, `confidenceLevel` or
   * `criticalZ`, `trendSignificanceLevel`). The recipe packs them into `parameters` and derives
   * `result.inference` from the same object; change them per frame with `result.setSettings`.
   * `multipleTesting` is the recipe's compile-time prop, so it is not part of the settings.
   */
  settings: GPUSpaceTimeHotSpotsRecipeSettings;
  /**
   * A `GPUParameterBuffer` of at least `GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH` float32 values
   * that the recipe fills from `settings`.
   */
  parameters: GPUParameterBuffer<'float32'>;
  /**
   * Multiple-testing correction of the hot and cold decision, passed to `GPUEmergingHotSpots`.
   * Defaults to `'benjamini-hochberg'`; `'benjamini-yekutieli'` is valid under the dependence of
   * overlapping temporal windows and neighboring cells.
   */
  multipleTesting?: GPUMultipleTestingMethod;
  /** Final-slice rule passed to `GPUEmergingHotSpots`. Defaults to `'require'`. */
  finalSlice?: 'require' | 'last-valid';
  /** Category colors; skipped when absent. */
  color?: {
    /** Packed rgba8 palette indexed by category code, at least 18 entries (0 to 17). */
    palette: GraphDataView<'uint32'>;
    /** `getGPUColorScaleParameterValues({scale: 'ordinal', ...})` view. */
    parameters: GraphDataView<'float32'>;
    /** Compile-time palette bound. Defaults to the 18 category codes. */
    maximumPaletteCount?: number;
  };
};

/** Named outputs of {@link addSpaceTimeHotSpotsRecipe}. */
export type GPUSpaceTimeHotSpotsRecipeResult = GPURecipeResult & {
  /** Resolved lattice context, or the supplied weights context when it is declared. */
  spatialContext?: GPUSpatialContext;
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
  /** Present when the matching output override was given. */
  pValues?: GraphDataView<'float32'>;
  adjustedPValues?: GraphDataView<'float32'>;
  lastValidSlice?: GraphDataView<'uint32'>;
  /** Present when `color` was requested. */
  colors?: GraphDataView<'uint32'>;
  /**
   * How the hot and cold bins were decided, for building a legend (a plain object), derived from
   * the packed `settings`.
   */
  inference: GPUInferenceMetadata;
  /**
   * Writes `settings` into `parameters` for the next submitted encoding and updates `inference`
   * from the same object. Throws, writing nothing, on invalid settings.
   */
  setSettings: (settings: GPUSpaceTimeHotSpotsRecipeSettings) => void;
};

/**
 * Space-time hot spot recipe: timestamped events to an ArcGIS-style emerging hot spot map.
 *
 * Chain: `GPUCalendarBuckets` decodes event times, an adapter kernel bins every event into
 * `cell * sliceCount + slice` of a zeroed dense `cell x slice` cube (`uint32`, the layout
 * `GPUEmergingHotSpots` reads) with an integer `atomicAdd`, `GPUEmergingHotSpots` classifies every
 * cell, and `GPUColorScale` (ordinal) colors the categories.
 *
 * Slices are absolute time steps (`slices.step`, `slices.interval`, `slices.origin`), so the
 * Mann-Kendall trend runs over a monotone time axis. Events outside the slice range are skipped
 * and counted in the `slice-range` stage `invalidCount`. `multipleTesting`, `finalSlice` and the
 * other options of `GPUEmergingHotSpots` apply as documented there, and the result carries a plain
 * `inference` object (`GPUInferenceMetadata`).
 *
 * Passing any of `scratch.keys`, `scratch.keyMask`, `scratch.groupKeys`, `scratch.groupCounts`,
 * `scratch.groupCount` or `outputs.status.groupOverflow` selects the sorted path instead: an adapter
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
  const spatialContext =
    cells.kind === 'lattice'
      ? resolveGPUSpatialContext(id, cells.spatialContext)
      : cells.weights.spatialContext;
  const eventCount = props.timestamps.length;
  const sliceCount = slices.count;
  assertRecipe(
    ID,
    !('field' in slices),
    'slices.field is no longer accepted: cyclic calendar fields cannot carry a trend, use slices.step with an origin'
  );
  assertRecipe(
    ID,
    Number.isInteger(sliceCount) && sliceCount >= 1,
    'slices.count must be a positive integer'
  );
  const interval = slices.interval ?? 1;
  assertRecipe(
    ID,
    Number.isInteger(interval) && interval >= 1,
    'slices.interval must be a positive integer'
  );
  assertRecipe(
    ID,
    ['year', 'quarter', 'month', 'week', 'day', 'hour'].includes(slices.step),
    'slices.step must be year, quarter, month, week, day or hour'
  );
  const {origin} = slices;
  for (const [name, value, minimum, maximum] of [
    ['year', origin.year, -1_000_000, 1_000_000],
    ['month', origin.month ?? 1, 1, 12],
    ['day', origin.day ?? 1, 1, 31],
    ['hour', origin.hour ?? 0, 0, 23]
  ] as const) {
    assertRecipe(
      ID,
      Number.isInteger(value) && value >= minimum && value <= maximum,
      `slices.origin.${name} must be an integer in ${minimum}..${maximum}`
    );
  }
  const cellCount = cells.kind === 'ids' ? cells.cellCount : cells.width * cells.height;
  const cubeLength = cellCount * sliceCount;
  assertRecipe(ID, cubeLength < 0xfffffffe, 'cellCount * sliceCount is too large for a u32 key');

  // 1. Calendar columns of the event times: only the ones the step needs.
  const needsMonth = slices.step !== 'year';
  const needsDay = slices.step === 'week' || slices.step === 'day' || slices.step === 'hour';
  const needsHour = slices.step === 'hour';
  const calendarYear = getOrCreateView(
    graph,
    `${id}-calendar-year`,
    'sint32',
    eventCount,
    scratch.calendarYear
  );
  const calendarMonth = needsMonth
    ? getOrCreateView(graph, `${id}-calendar-month`, 'uint32', eventCount, scratch.calendarMonth)
    : undefined;
  const calendarDay = needsDay
    ? getOrCreateView(
        graph,
        `${id}-calendar-day-of-month`,
        'uint32',
        eventCount,
        scratch.calendarDayOfMonth
      )
    : undefined;
  const calendarHour = needsHour
    ? getOrCreateView(graph, `${id}-calendar-hour`, 'uint32', eventCount, scratch.calendarHour)
    : undefined;
  builder.add(
    new GPUCalendarBuckets({
      id: `${id}-calendar`,
      timestamps: props.timestamps,
      mask: props.mask,
      utcOffsets: props.utcOffsets,
      parameters: props.calendarParameters,
      output: {
        year: calendarYear,
        ...(calendarMonth ? {month: calendarMonth} : {}),
        ...(calendarDay ? {dayOfMonth: calendarDay} : {}),
        ...(calendarHour ? {hour: calendarHour} : {})
      }
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
  const stepsPerSlice = getSliceDivisor(slices);
  const originStep = getSpaceTimeStepIndex(slices.step, origin);
  const stepExpression = {
    year: 'year',
    quarter: 'year * 4 + (month - 1) / 3',
    month: 'year * 12 + month - 1',
    week: 'daysFromCivil(year, month, day)',
    day: 'daysFromCivil(year, month, day)',
    hour: 'daysFromCivil(year, month, day) * 24 + hour'
  }[slices.step];
  const eventBinSource = `${cellSource}
  let year = calendarYear[calendarYearOffset + index];
  ${calendarMonth ? 'let month = i32(calendarMonth[calendarMonthOffset + index]);' : ''}
  ${calendarDay ? 'let day = i32(calendarDay[calendarDayOffset + index]);' : ''}
  ${calendarHour ? 'let hour = i32(calendarHour[calendarHourOffset + index]);' : ''}
  let isCalendarValid = year != bitcast<i32>(0x80000000u);
  let stepOffset = select(-1, ${stepExpression} - (${originStep}), isCalendarValid);
  let slice = stepOffset / ${stepsPerSlice};
  let sliceValid = stepOffset >= 0 && slice < ${sliceCount};
  let valid = cellValid && sliceValid;`;
  const calendarBindings = [
    {name: 'calendarYear', view: calendarYear, type: 'i32', access: 'read'} as const,
    ...(calendarMonth
      ? [{name: 'calendarMonth', view: calendarMonth, type: 'u32', access: 'read'} as const]
      : []),
    ...(calendarDay
      ? [{name: 'calendarDay', view: calendarDay, type: 'u32', access: 'read'} as const]
      : []),
    ...(calendarHour
      ? [{name: 'calendarHour', view: calendarHour, type: 'u32', access: 'read'} as const]
      : [])
  ];
  // Hinnant's days_from_civil in i32 arithmetic with truncating division.
  const calendarDeclarations = `fn daysFromCivil(year: i32, month: i32, day: i32) -> i32 {
  let adjustedYear = select(year, year - 1, month <= 2);
  let era = select(adjustedYear, adjustedYear - 399, adjustedYear < 0) / 400;
  let yearOfEra = adjustedYear - era * 400;
  let shiftedMonth = (month + 9) % 12;
  let dayOfYear = (153 * shiftedMonth + 2) / 5 + day - 1;
  let dayOfEra = yearOfEra * 365 + yearOfEra / 4 - yearOfEra / 100 + dayOfYear;
  return era * 146097 + dayOfEra - 719468;
}`;

  // Events with a valid cell and time that fall before the origin or past the last slice.
  const skippedCount = getOrCreateView(
    graph,
    `${id}-skipped-count`,
    'uint32',
    1,
    outputOverrides.status?.skippedEventCount
  );
  graph.add(
    createFillNode<Parameters>(graph, {
      id: `${id}-skipped-count-clear`,
      operation: 'GPUSpaceTimeHotSpotsCube',
      view: skippedCount,
      type: 'u32',
      value: '0u'
    })
  );
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-skipped-count`,
      operation: 'GPUSpaceTimeHotSpotsCube',
      variant: 'slice-range',
      bindings: [
        ...calendarBindings,
        ...cellBindings,
        {name: 'skippedOutput', view: skippedCount, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: eventCount,
      declarations: calendarDeclarations,
      body: `${eventBinSource}
  if (cellValid && isCalendarValid && !sliceValid) {
    atomicAdd(&skippedOutput[skippedOutputOffset], 1u);
  }`
    })
  );

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
  // mask, group keys, counts or count) or for `outputs.status.groupOverflow`. Both paths write identical
  // cubes: counts never exceed `eventCount`, which is far below 2^32. `GPUEmergingHotSpots` reads
  // a count as `f32(count)`, exact below 2^24 events per bin, the same in both paths.
  const useGroupStatistics =
    scratch.keys !== undefined ||
    scratch.keyMask !== undefined ||
    scratch.groupKeys !== undefined ||
    scratch.groupCounts !== undefined ||
    scratch.groupCount !== undefined ||
    outputOverrides.status?.groupOverflow !== undefined;
  let intermediates: GPUSpaceTimeHotSpotsRecipeResult['intermediates'] = {
    calendarYear,
    ...(calendarMonth ? {calendarMonth} : {}),
    ...(calendarDay ? {calendarDayOfMonth: calendarDay} : {}),
    ...(calendarHour ? {calendarHour} : {})
  };
  let status: GPUSpaceTimeHotSpotsRecipeResult['status'] = {
    stages: [{stage: 'slice-range', status: {invalidCount: skippedCount}}]
  };
  if (!useGroupStatistics) {
    graph.add(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-cube-count`,
        operation: 'GPUSpaceTimeHotSpotsCube',
        variant: 'atomic-count',
        bindings: [
          ...calendarBindings,
          ...cellBindings,
          {name: 'cubeOutput', view: cube, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: eventCount,
        declarations: calendarDeclarations,
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
          ...calendarBindings,
          ...cellBindings,
          {name: 'keyOutput', view: keys, type: 'u32', access: 'read_write'},
          {name: 'keyMaskOutput', view: keyMask, type: 'u32', access: 'read_write'}
        ],
        invocationCount: eventCount,
        declarations: calendarDeclarations,
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
      outputOverrides.status?.groupOverflow
    );
    const groupRequiredCount = getOrCreateView(
      graph,
      `${id}-group-required-count`,
      'uint32',
      1,
      outputOverrides.status?.groupRequiredCount
    );
    builder.add(
      new GPUGroupStatistics({
        id: `${id}-counts`,
        keys,
        mask: keyMask,
        columns: [],
        output: {
          keys: groupKeys,
          counts: groupCounts,
          status: {count: groupCount, overflow: groupOverflow, requiredCount: groupRequiredCount}
        }
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
      stages: [
        ...status.stages,
        {
          stage: 'cube-counts',
          status: {count: groupCount, overflow: groupOverflow, requiredCount: groupRequiredCount}
        }
      ]
    };
  }

  // 4. Emerging hot spots.
  assertRecipe(
    ID,
    props.parameters.length >= GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
    `parameters needs ${GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH} values`
  );
  const parameterView = props.parameters.importToGraph(graph, `${id}-parameters`);
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
  const optionalOutputs = {
    ...(outputOverrides.pValues ? {pValues: outputOverrides.pValues} : {}),
    ...(outputOverrides.adjustedPValues ? {adjustedPValues: outputOverrides.adjustedPValues} : {}),
    ...(outputOverrides.lastValidSlice ? {lastValidSlice: outputOverrides.lastValidSlice} : {})
  };
  const emerging = builder.add(
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
      parameters: parameterView,
      multipleTesting: props.multipleTesting,
      finalSlice: props.finalSlice,
      output: {...outputs, ...optionalOutputs}
    })
  );
  // One source of truth (rule 19): the recipe packs `settings` and derives the metadata from the
  // same object; both validate before anything is written.
  const writeSettings = (settings: GPUSpaceTimeHotSpotsRecipeSettings): GPUInferenceMetadata => {
    const values = emerging.getParameterValues(settings);
    const metadata = emerging.getInferenceMetadata(settings);
    props.parameters.write(values);
    return metadata;
  };

  const result: GPUSpaceTimeHotSpotsRecipeResult = {
    contributors: builder.contributors,
    spatialContext,
    cellCount,
    sliceCount,
    cube,
    ...outputs,
    ...optionalOutputs,
    outputs: {cube, ...outputs, ...optionalOutputs},
    intermediates,
    status,
    inference: writeSettings(props.settings),
    setSettings: settings => {
      result.inference = writeSettings(settings);
    }
  };

  // 5. Category colors (ordinal scale).
  if (props.color) {
    const {color} = props;
    const maximumPaletteCount =
      color.maximumPaletteCount ?? Object.keys(GPU_EMERGING_HOT_SPOTS_CATEGORIES).length;
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
