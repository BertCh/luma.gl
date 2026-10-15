// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DataFrameFamilyOperators} from '../types';

/** Curated temporal formulas, preserving inclusive windows and exact calendar offset arithmetic. */
export const TEMPORAL_OPERATORS: DataFrameFamilyOperators = {
  /**
   * Primary: instant acceptance, inclusive at BOTH endpoints. Intervals use end>=startWindow and
   * start<=endWindow with a non-reversed window. NaN timestamps fail; optional predicates further
   * restrict acceptance. Double-single and Int64 words preserve the same comparisons; fade and
   * clipping are secondary outputs. Sorted mode finds the identical accepted contiguous range.
   */
  GPUTimeWindowFilter: {
    family: 'temporal',
    descriptor: {scope: 'local', reduction: 'select'},
    formula: {
      id: 'gpu-dataframe:GPUTimeWindowFilter',
      derivation: 'curated',
      tex: String.raw`\term{outputMask}{\mathrm{keep}}=\mathbf{1}[\term{start}{s}\leq\term{timestamps}{t}\leq\term{end}{e}]`,
      ascii:
        'keep=1[start <= t <= end]; reversed windows and NaN timestamps reject; additionalPredicates intersect the mask.',
      symbols: [
        {id: 'outputMask', role: 'output', binding: 'wgsl:outputMask'},
        {id: 'start', role: 'parameter', binding: 'wgsl:windowStartHigh'},
        {id: 'timestamps', role: 'input', binding: 'wgsl:startHigh'},
        {id: 'end', role: 'parameter', binding: 'wgsl:windowEndHigh'}
      ]
    },
    stages: ['filter', 'select'],
    datumKinds: [{kind: 'row', label: 'Timestamped rows', indexSpace: '1d'}],
    guards: [],
    choices: ['start', 'end', 'startFadeDuration', 'endFadeDuration', 'sorted', 'validateOrder'],
    variables: {
      timestamps: {level: 'interval', extent: 'intensive'},
      outputMask: {level: 'nominal'},
      fadeWeights: {level: 'ratio', extent: 'intensive'}
    },
    sourceFiles: [
      'modules/experimental/src/gpu-dataframe/time-window-filter/time-window-classify-node.ts'
    ]
  },
  /**
   * Primary: counts per (cell,time-bucket); this contributor has no value-sum output.
   * Float bucket b satisfies R(b*width)<=R(t-origin)<R((b+1)*width), with R=fround; the
   * quotient estimate is corrected against these products. Int64 words use exact division.
   * Invalid cells, masked rows, NaN values/times and out-of-range buckets are skipped.
   * Extrema and earliest/latest values are secondary outputs; time ties select the lowest row.
   */
  GPUTemporalReduction: {
    family: 'temporal',
    // A slot groups arbitrary rows by cell and time; the primary aggregation counts rows.
    descriptor: {scope: 'zonal', reduction: 'count'},
    formula: {
      id: 'gpu-dataframe:GPUTemporalReduction',
      derivation: 'curated',
      tex: String.raw`\operatorname{R}(\term{bucket}{b}\term{width}{w})\leq\operatorname{R}(\term{timestamps}{t}-\term{origin}{o})<\operatorname{R}((\term{bucket}{b}+1)\term{width}{w}),\quad \term{rowSlots}{s}=\term{cellIds}{c}\term{bucketCount}{B}+\term{bucket}{b},\quad \term{counts}{N_s}=\sum_{\text{included rows}}\mathbf{1}[\term{rowSlots}{s_{\rm row}}=\term{slot}{s}]`,
      ascii:
        'R(b*w)<=R(t-origin)<R((b+1)*w); slot=cell*bucketCount+b; counts[slot]=number of accepted rows (Int64: b=(t-origin) div width).',
      symbols: [
        {id: 'bucket', role: 'intermediate', binding: 'wgsl:bucket'},
        {id: 'width', role: 'parameter', binding: 'wgsl:width'},
        {id: 'timestamps', role: 'input', binding: 'wgsl:timestamps'},
        {id: 'origin', role: 'parameter', binding: 'wgsl:origin'},
        {id: 'rowSlots', role: 'intermediate', binding: 'wgsl:rowSlots'},
        {id: 'cellIds', role: 'input', binding: 'wgsl:cellIds'},
        {id: 'bucketCount', role: 'constant', binding: 'wgsl:BUCKET_COUNT'},
        {id: 'counts', role: 'output', binding: 'wgsl:counts'},
        {id: 'slot', role: 'index', binding: 'wgsl:slot'}
      ]
    },
    stages: ['filter', 'group', 'aggregate'],
    datumKinds: [
      {kind: 'row', label: 'Rows', indexSpace: '1d'},
      {kind: 'time-bucket', label: 'Cell and time slots', indexSpace: '1d'}
    ],
    guards: [],
    choices: ['origin', 'width', 'cellCount', 'bucketCount', 'sorted'],
    variables: {
      timestamps: {level: 'interval', extent: 'intensive'},
      values: {level: 'ratio', extent: 'intensive'},
      counts: {level: 'ratio', extent: 'extensive'},
      cellIds: {level: 'nominal'}
    },
    sourceFiles: [
      'modules/experimental/src/gpu-dataframe/temporal-reduction/gpu-temporal-reduction.ts'
    ]
  },
  /**
   * Primary: hour-by-weekday counts. Local days use floor division (also before 1970); mod is
   * nonnegative. Offset precedence is timeZone table or utcOffsets, then fixed utcOffsetMinutes.
   * The table's seconds offset at index #{transition<=instant} applies FROM the transition,
   * including daylight-saving changes; per-row/fixed offsets are minutes. No offset is inferred.
   * Invalid offsets, Int64 overflow, |local days|>1e9 and masked rows never enter the matrix.
   * Civil date/ISO-week fields use exact integer Gregorian arithmetic as secondary outputs.
   */
  GPUCalendarBuckets: {
    family: 'temporal',
    descriptor: {scope: 'zonal', scopeSecondary: 'local', reduction: 'count'},
    formula: {
      id: 'gpu-dataframe:GPUCalendarBuckets',
      derivation: 'curated',
      tex: String.raw`\term{days}{d}=\left\lfloor\frac{\term{timestamps}{t}+\term{offsetMilliseconds}{\Delta(t)}}{86400000}\right\rfloor,\quad \term{hour}{h}=\left\lfloor\frac{(\term{timestamps}{t}+\term{offsetMilliseconds}{\Delta(t)})\bmod86400000}{3600000}\right\rfloor,\quad \term{weekday}{v}=(\term{days}{d}+3-\term{firstDay}{f})\bmod7,\quad \term{hourWeekdayCounts}{C_{24v+h}}=\sum_{\text{valid rows}}\mathbf{1}[\term{weekday}{v_{\rm row}}=\term{weekday}{v}\land\term{hour}{h_{\rm row}}=\term{hour}{h}]`,
      ascii:
        'local=t+offsetMilliseconds(t); day=floor(local/86400000); hour=floor(mod(local,86400000)/3600000); weekday=mod(day+3-firstDay,7); count[weekday*24+hour]++; transitions use upper_bound(t).',
      symbols: [
        {id: 'days', role: 'intermediate', binding: 'wgsl:days'},
        {id: 'timestamps', role: 'input', binding: 'wgsl:timestamps'},
        {id: 'offsetMilliseconds', role: 'intermediate', binding: 'wgsl:offsetMilliseconds'},
        {id: 'hour', role: 'output', binding: 'wgsl:hour'},
        {id: 'weekday', role: 'output', binding: 'wgsl:weekday'},
        {id: 'firstDay', role: 'parameter', binding: 'wgsl:firstDay'},
        {id: 'hourWeekdayCounts', role: 'output', binding: 'wgsl:hourWeekdayCountsOut'}
      ]
    },
    stages: ['derive', 'group', 'aggregate'],
    datumKinds: [
      {kind: 'row', label: 'Instants', indexSpace: '1d'},
      {kind: 'time-bucket', label: 'Weekday and hour', indexSpace: '1d'}
    ],
    guards: [],
    choices: ['utcOffsetMinutes', 'firstDayOfWeek', 'utcOffsets', 'timeZone'],
    variables: {
      timestamps: {level: 'interval', extent: 'intensive'},
      hour: {level: 'cyclic'},
      weekday: {level: 'cyclic'},
      hourWeekdayCounts: {level: 'ratio', extent: 'extensive'}
    },
    sourceFiles: ['modules/experimental/src/gpu-dataframe/calendar-buckets/gpu-calendar-buckets.ts']
  }
};
