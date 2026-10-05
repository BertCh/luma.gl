// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {
  createMapGraphFillNode,
  createMapGraphKernelNode,
  type MapGraphKernelBinding
} from '../map-graph-kernels';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {
  GPU_CALENDAR_BUCKETS_MATRIX_LENGTH,
  GPU_CALENDAR_BUCKETS_MAXIMUM_DAYS,
  GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH
} from './calendar-buckets-parameters';

const OPERATION = 'GPUCalendarBuckets';
/** Storage bindings a kernel may use on the WebGPU default limit. */
const MAXIMUM_BINDINGS = 8;

/**
 * Caller-owned output columns of {@link GPUCalendarBuckets}. Each is optional and which ones are
 * present is compile-time, but at least one must be given. Every column holds one row per input
 * row; a masked or invalid row writes `0xffffffff` to `uint32` columns and `-2^31` to `year` and
 * `isoWeekYear`.
 */
export type GPUCalendarBucketsOutput = {
  /** Proleptic Gregorian year of the local time, astronomical numbering (year 0 exists). */
  year?: GraphDataView<'sint32'>;
  /** Month, 1 to 12. */
  month?: GraphDataView<'uint32'>;
  /** Day of the month, 1 to 31. */
  dayOfMonth?: GraphDataView<'uint32'>;
  /** Hour, 0 to 23. */
  hour?: GraphDataView<'uint32'>;
  /** Minute, 0 to 59. */
  minute?: GraphDataView<'uint32'>;
  /** Weekday, 0 to 6, where 0 is the `firstDayOfWeek` parameter (Monday by default). */
  weekday?: GraphDataView<'uint32'>;
  /** Day of the year, 1 to 366. */
  dayOfYear?: GraphDataView<'uint32'>;
  /** ISO 8601 week number, 1 to 53 (weeks start on Monday, week 1 contains the first Thursday). */
  isoWeek?: GraphDataView<'uint32'>;
  /** ISO 8601 week-based year, which differs from `year` around January 1. */
  isoWeekYear?: GraphDataView<'sint32'>;
  /** Quarter of the year, 1 to 4. */
  quarter?: GraphDataView<'uint32'>;
  /**
   * Hour by weekday count matrix, `weekday * 24 + hour`, at least 168 elements. It is cleared on
   * every encoding, then every valid row adds one. The counts are exact integers.
   */
  hourWeekdayCounts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUCalendarBuckets}.
 *
 * Per-frame (no recompile): the contents of `parameters` and of every input buffer. Topology
 * (needs a new graph): the row count, which outputs are present, and whether `mask` and
 * `utcOffsets` are present.
 */
export type GPUCalendarBucketsProps = {
  /** Prefix for generated node IDs. Defaults to `'calendar-buckets'`. */
  id?: string;
  /**
   * Epoch milliseconds as signed 64-bit two's complement words, one `uint32x2` `(low, high)` row
   * per row (for example from `getInt64TimeWords`). Times before 1970 are negative.
   */
  timestamps: GraphDataView<'uint32x2'>;
  /** Optional packed `uint32` row mask; zero makes the row invalid. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Optional UTC offset in minutes per row (`sint32`, within `[-1440, 1440]`), which overrides the
   * fixed offset in `parameters`. This is the only way to express daylight saving time: the caller
   * precomputes each row's offset from its zone rules. An out-of-range offset makes the row invalid.
   */
  utcOffsets?: GraphDataView<'sint32'>;
  /**
   * Per-frame `sint32` view of at least 2 elements written with
   * `getGPUCalendarBucketsParameterValues`: the fixed UTC offset in minutes (a fixed offset, not a
   * zone) and the first day of the week.
   */
  parameters: GraphDataView<'sint32'>;
  /** Caller-owned outputs. At least one must be present. */
  output: GPUCalendarBucketsOutput;
};

type OutputSpec = {
  name: keyof GPUCalendarBucketsOutput;
  type: 'u32' | 'i32' | 'atomic<u32>';
  /** WGSL statement that writes the output for one row. */
  write: string;
};

const COLUMN_OUTPUTS: readonly {
  name: Exclude<keyof GPUCalendarBucketsOutput, 'hourWeekdayCounts'>;
  field: string;
  signed?: boolean;
}[] = [
  {name: 'year', field: 'year', signed: true},
  {name: 'month', field: 'month'},
  {name: 'dayOfMonth', field: 'day'},
  {name: 'hour', field: 'hour'},
  {name: 'minute', field: 'minute'},
  {name: 'weekday', field: 'weekday'},
  {name: 'dayOfYear', field: 'dayOfYear'},
  {name: 'isoWeek', field: 'isoWeek'},
  {name: 'isoWeekYear', field: 'isoWeekYear', signed: true},
  {name: 'quarter', field: 'quarter'}
];

/**
 * Decodes epoch-millisecond timestamps into calendar columns (year, month, day, hour, minute,
 * weekday, day of year, ISO week, ISO week-year, quarter) and an hour by weekday count matrix, so
 * time-of-day and weekday charts and calendar bucket keys need no per-row `Date` on the CPU.
 *
 * Time zone: local time is UTC plus an offset in minutes. The per-frame `parameters` carry one
 * fixed offset; an optional per-row `utcOffsets` column overrides it and is how daylight saving
 * time is supported (the caller precomputes each row's offset). There is no zone table on the GPU.
 *
 * Exactness: all arithmetic is integer. The signed 64-bit millisecond count plus offset is
 * floor-divided by 86,400,000 with u32 operations (nibble-wise long division, exact for negative
 * times too), then Howard Hinnant's `civil_from_days` gives year, month and day in i32. Weekday
 * comes from the day number, day of year from the month and leap rule, and the ISO week and
 * week-year from the Thursday of the row's ISO week. Results are bit-identical on every adapter.
 * Weekdays are ISO (Monday is 0) unless `firstDayOfWeek` says otherwise; ISO week numbers always
 * start on Monday.
 *
 * Range: rows whose local day number is beyond +-1e9 days (about +-2.7 million years), whose sum with
 * the offset overflows Int64, whose per-row offset is beyond +-1440 minutes, or that are masked,
 * are invalid and get the sentinels documented on {@link GPUCalendarBucketsOutput}. The hour by
 * weekday matrix counts only valid rows, with integer `atomicAdd`, so it is deterministic.
 *
 * Because WebGPU allows 8 storage bindings per kernel, the outputs are split over as many decode
 * kernels as needed; each recomputes the (cheap) decode. Inputs must be single packed views.
 */
export class GPUCalendarBuckets implements GPUMapGraphRecipe {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'calendar-buckets';
  /** Validated properties. */
  readonly props: GPUCalendarBucketsProps;
  /** Number of input rows. */
  readonly rowCount: number;

  constructor(props: GPUCalendarBucketsProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const id = this.id;
    const {output} = props;
    for (const [name, view] of [
      ['timestamps', props.timestamps],
      ['mask', props.mask],
      ['utcOffsets', props.utcOffsets],
      ['parameters', props.parameters]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.timestamps, ['uint32x2'], `${id} timestamps`);
    this.rowCount = props.timestamps.length;
    if (this.rowCount < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== this.rowCount) {
        throw new Error(`${id} mask length must equal timestamps length`);
      }
    }
    if (props.utcOffsets) {
      validatePackedView(props.utcOffsets, ['sint32'], `${id} utcOffsets`);
      if (props.utcOffsets.length !== this.rowCount) {
        throw new Error(`${id} utcOffsets length must equal timestamps length`);
      }
    }
    validatePackedView(props.parameters, ['sint32'], `${id} parameters`);
    if (props.parameters.length < GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH} sint32 values`
      );
    }
    let outputCount = 0;
    for (const column of COLUMN_OUTPUTS) {
      const view = output[column.name];
      if (!view) {
        continue;
      }
      outputCount++;
      validatePackedView(
        view,
        [column.signed ? 'sint32' : 'uint32'],
        `${id} output.${column.name}`
      );
      if (view.length < this.rowCount) {
        throw new Error(`${id} output.${column.name} must hold one row per timestamp`);
      }
    }
    if (output.hourWeekdayCounts) {
      outputCount++;
      validatePackedUint32View(output.hourWeekdayCounts, `${id} output.hourWeekdayCounts`);
      if (output.hourWeekdayCounts.length < GPU_CALENDAR_BUCKETS_MATRIX_LENGTH) {
        throw new Error(
          `${id} output.hourWeekdayCounts must hold ${GPU_CALENDAR_BUCKETS_MATRIX_LENGTH} elements`
        );
      }
    }
    if (outputCount === 0) {
      throw new Error(`${id} needs at least one output`);
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
  }

  /** Returns the optional fill node of the matrix, then the decode kernels in output order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, rowCount} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const specs: OutputSpec[] = [];
    for (const column of COLUMN_OUTPUTS) {
      if (output[column.name]) {
        specs.push({
          name: column.name,
          type: column.signed ? 'i32' : 'u32',
          write: `${column.name}Out[${column.name}OutOffset + index] = fields.${column.field};`
        });
      }
    }
    if (output.hourWeekdayCounts) {
      nodes.push(
        createMapGraphFillNode<Parameters>(graph, {
          id: `${id}-fill-matrix`,
          operation: OPERATION,
          view: output.hourWeekdayCounts,
          type: 'u32',
          value: '0u',
          componentCount: GPU_CALENDAR_BUCKETS_MATRIX_LENGTH
        })
      );
      specs.push({
        name: 'hourWeekdayCounts',
        type: 'atomic<u32>',
        write: `if (fields.valid) {
    atomicAdd(&hourWeekdayCountsOut[hourWeekdayCountsOutOffset + fields.weekday * 24u + fields.hour], 1u);
  }`
      });
    }

    const read = (
      name: string,
      view: GraphDataView,
      type: 'u32' | 'i32'
    ): MapGraphKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const inputBindings: MapGraphKernelBinding[] = [
      read('timestamps', props.timestamps, 'u32'),
      read('params', props.parameters, 'i32')
    ];
    if (props.mask) {
      inputBindings.push(read('rowMask', props.mask, 'u32'));
    }
    if (props.utcOffsets) {
      inputBindings.push(read('rowOffsets', props.utcOffsets, 'i32'));
    }
    const perKernel = MAXIMUM_BINDINGS - inputBindings.length;
    for (let first = 0, step = 0; first < specs.length; first += perKernel, step++) {
      const group = specs.slice(first, first + perKernel);
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-decode-${step}`,
          operation: OPERATION,
          variant: 'decode',
          bindings: [
            ...inputBindings,
            ...group.map(
              (spec): MapGraphKernelBinding => ({
                name: `${spec.name}Out`,
                view: output[spec.name] as GraphDataView,
                type: spec.type,
                access: 'read_write'
              })
            )
          ],
          invocationCount: rowCount,
          declarations: CALENDAR_WGSL,
          body: `let time = vec2<u32>(timestamps[timestampsOffset + 2u * index], timestamps[timestampsOffset + 2u * index + 1u]);
  let offsetMinutes = ${props.utcOffsets ? 'rowOffsets[rowOffsetsOffset + index]' : 'params[paramsOffset]'};
  let firstDay = u32(((params[paramsOffset + 1u] % 7) + 7) % 7);
  var fields = decodeCalendar(time, offsetMinutes, firstDay);
  ${props.mask ? 'if (rowMask[rowMaskOffset + index] == 0u) {\n    fields = invalidCalendar();\n  }' : ''}
  ${group.map(spec => spec.write).join('\n  ')}`
        })
      );
    }
    return nodes;
  }

  private getInputViews(): GraphDataView[] {
    const {props} = this;
    return [props.timestamps, props.mask, props.utcOffsets, props.parameters].filter(view =>
      Boolean(view)
    ) as GraphDataView[];
  }

  private getOutputViews(): GraphDataView[] {
    return Object.values(this.props.output).filter(view => Boolean(view)) as GraphDataView[];
  }
}

const CALENDAR_WGSL = /* wgsl */ `
const MILLISECONDS_PER_DAY: u32 = 86400000u;
const MAXIMUM_DAYS: u32 = ${GPU_CALENDAR_BUCKETS_MAXIMUM_DAYS}u;

struct CalendarFields {
  valid: bool,
  year: i32,
  month: u32,
  day: u32,
  hour: u32,
  minute: u32,
  weekday: u32,
  dayOfYear: u32,
  isoWeek: u32,
  isoWeekYear: i32,
  quarter: u32
}

fn invalidCalendar() -> CalendarFields {
  let invalidYear = bitcast<i32>(0x80000000u);
  return CalendarFields(false, invalidYear, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
    0xffffffffu, 0xffffffffu, 0xffffffffu, invalidYear, 0xffffffffu);
}

fn isLeapYear(year: i32) -> bool {
  return (year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0));
}

// Hinnant civil_from_days: days since 1970-01-01 to (year, month, day).
fn civilFromDays(days: i32) -> vec3<i32> {
  let z = days + 719468;
  let era = select((z - 146096) / 146097, z / 146097, z >= 0);
  let dayOfEra = z - era * 146097;
  let yearOfEra = (dayOfEra - dayOfEra / 1460 + dayOfEra / 36524 - dayOfEra / 146096) / 365;
  let dayOfMarchYear = dayOfEra - (365 * yearOfEra + yearOfEra / 4 - yearOfEra / 100);
  let monthPrime = (5 * dayOfMarchYear + 2) / 153;
  let day = dayOfMarchYear - (153 * monthPrime + 2) / 5 + 1;
  let month = select(monthPrime - 9, monthPrime + 3, monthPrime < 10);
  let year = yearOfEra + era * 400 + select(0, 1, month <= 2);
  return vec3<i32>(year, month, day);
}

fn getDayOfYear(date: vec3<i32>) -> i32 {
  let adjustment = select(select(-2, -1, isLeapYear(date.x)), 0, date.y <= 2);
  return (367 * date.y - 362) / 12 + date.z + adjustment;
}

struct DivideResult {
  quotientLow: u32,
  quotientHigh: u32,
  remainder: u32
}

// Exact unsigned 64-bit division by a divisor below 2^28, one nibble at a time so that
// remainder * 16 never overflows 32 bits.
fn divideUnsigned64(value: vec2<u32>, divisor: u32) -> DivideResult {
  var remainder = value.y % divisor;
  var quotientLow = 0u;
  for (var nibble = 7i; nibble >= 0i; nibble = nibble - 1i) {
    let shift = u32(nibble) * 4u;
    let current = (remainder << 4u) | ((value.x >> shift) & 0xfu);
    quotientLow = quotientLow | ((current / divisor) << shift);
    remainder = current % divisor;
  }
  return DivideResult(quotientLow, value.y / divisor, remainder);
}

fn decodeCalendar(time: vec2<u32>, offsetMinutes: i32, firstDay: u32) -> CalendarFields {
  let invalid = invalidCalendar();
  if (offsetMinutes < -1440 || offsetMinutes > 1440) {
    return invalid;
  }
  // Signed 64-bit local time = time + offset, with overflow detection.
  let offsetMilliseconds = offsetMinutes * 60000;
  let offsetLow = bitcast<u32>(offsetMilliseconds);
  let offsetHigh = select(0u, 0xffffffffu, offsetMilliseconds < 0);
  let low = time.x + offsetLow;
  let high = time.y + offsetHigh + select(0u, 1u, low < time.x);
  let timeSign = time.y >> 31u;
  let sumSign = high >> 31u;
  if (timeSign == (offsetHigh >> 31u) && sumSign != timeSign) {
    return invalid;
  }
  let isNegative = sumSign != 0u;
  var magnitude = vec2<u32>(low, high);
  if (isNegative) {
    let negatedLow = ~low + 1u;
    magnitude = vec2<u32>(negatedLow, ~high + select(0u, 1u, negatedLow == 0u));
  }
  let division = divideUnsigned64(magnitude, MILLISECONDS_PER_DAY);
  if (division.quotientHigh != 0u || division.quotientLow > MAXIMUM_DAYS) {
    return invalid;
  }
  var days = i32(division.quotientLow);
  var millisecondsOfDay = division.remainder;
  if (isNegative) {
    // floor(-(q * D + r) / D) = -(q + 1) with D - r left over when r is not zero.
    if (millisecondsOfDay != 0u) {
      days = days + 1;
      millisecondsOfDay = MILLISECONDS_PER_DAY - millisecondsOfDay;
    }
    days = -days;
  }
  let date = civilFromDays(days);
  let isoWeekday = u32(((days + 3) % 7 + 7) % 7);
  let thursday = days - i32(isoWeekday) + 3;
  let thursdayDate = civilFromDays(thursday);
  var fields = invalid;
  fields.valid = true;
  fields.year = date.x;
  fields.month = u32(date.y);
  fields.day = u32(date.z);
  fields.hour = millisecondsOfDay / 3600000u;
  fields.minute = (millisecondsOfDay % 3600000u) / 60000u;
  fields.weekday = (isoWeekday + 7u - firstDay) % 7u;
  fields.dayOfYear = u32(getDayOfYear(date));
  fields.isoWeek = u32((getDayOfYear(thursdayDate) - 1) / 7 + 1);
  fields.isoWeekYear = thursdayDate.x;
  fields.quarter = u32((date.y - 1) / 3 + 1);
  return fields;
}
`;
