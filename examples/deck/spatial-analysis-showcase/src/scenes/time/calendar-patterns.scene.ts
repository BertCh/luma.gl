// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {CalendarPatternsOptions, CalendarStatistic} from './calendar-patterns.compute';

const GROUPS = [
  {value: 'all', label: 'All observations'},
  {value: '0', label: 'Plants'},
  {value: '1', label: 'Birds'},
  {value: '2', label: 'Insects'},
  {value: '3', label: 'Fungi'},
  {value: '4', label: 'Mammals'},
  {value: '5', label: 'Spiders and kin'},
  {value: '6', label: 'Amphibians and reptiles'},
  {value: '7', label: 'Snails and mussels'},
  {value: '8', label: 'Fish'},
  {value: '9', label: 'Other life'}
];

const DAYS_OF_YEAR_LABELS = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec'
];
const formatDate = (day: number) => {
  const date = new Date(Date.UTC(2023, 0, 1 + day));
  return `${DAYS_OF_YEAR_LABELS[date.getUTCMonth()]} ${date.getUTCDate()}`;
};
const formatHour = (hour: number) => `${String(hour).padStart(2, '0')}:00`;

const STATISTIC_TITLES: Record<
  CalendarStatistic,
  {title: string; unit: string; labels?: [string, string]}
> = {
  count: {title: 'Observations per community area', unit: 'observations'},
  density: {title: 'Observations per square kilometer', unit: 'per km2'},
  'mean-hour': {title: 'Mean hour of day', unit: 'h', labels: ['earlier', 'later']},
  'median-hour': {title: 'Median hour of day', unit: 'h', labels: ['earlier', 'later']},
  'hour-spread': {title: 'Spread of the hour (standard deviation)', unit: 'h'},
  'percentile-hour': {
    title: 'Percentile of the hour of day',
    unit: 'h',
    labels: ['earlier', 'later']
  },
  'mode-hour': {title: 'Most common hour of day', unit: 'h', labels: ['earlier', 'later']},
  'active-days': {title: 'Days with at least one observation', unit: 'days'},
  'research-grade-share': {title: 'Share with a community-confirmed ID', unit: 'share'},
  'introduced-share': {title: 'Share of introduced (non-native) taxa', unit: 'share'}
};

export default defineScene<CalendarPatternsOptions>({
  id: 'calendar-patterns',
  title: 'Chicago nature observations by hour and community area',
  chapter: 'time',
  order: 4,
  summary:
    'GPU calendar decoding and grouped statistics use 43,557 iNaturalist records to map hourly and seasonal observation metrics by Chicago community area; observer effort and linear hour summaries limit ecological interpretation.',
  contributors: ['GPUCalendarBuckets', 'GPUTimeWindowFilter', 'GPUGroupStatistics'],
  datasets: [
    {id: 'chicago-nature', role: 'timestamped nature observations'},
    {id: 'chicago-community-areas', role: 'zones for the group statistics'}
  ],
  initialView: {longitude: -87.5, latitude: 41.835, zoom: 9.7},
  basemap: ground('night'),
  furniture: {
    title: {
      title: 'Chicago observation calendar',
      subtitle: 'iNaturalist records by hour and community area'
    },
    scaleBar: {units: 'metric'},
    credit: 'iNaturalist contributors; Chicago community-area boundaries',
    caveat: 'Counts measure observer effort as well as wildlife activity.'
  },

  options: [
    {
      kind: 'range',
      id: 'dateRange',
      label: 'Date window',
      group: 'Time window',
      apply: 'param',
      min: 0,
      max: 365,
      step: 1,
      default: [0, 365],
      format: value => formatDate(value),
      help: 'Observations outside this window are dropped by GPUTimeWindowFilter before anything else runs. Times are exact Int64 milliseconds and the window is eight parameter words.'
    },
    {
      kind: 'toggle',
      id: 'playWindow',
      label: 'Slide a window through the year',
      group: 'Time window',
      apply: 'param',
      default: false,
      help: 'Moves a fixed-length window across 2023 and loops. Watch the matrix and the area statistics follow the seasons: spring migration, the summer insect peak, the autumn fall-off.'
    },
    {
      kind: 'slider',
      id: 'windowDays',
      label: 'Sliding window length',
      group: 'Time window',
      apply: 'param',
      min: 7,
      max: 120,
      step: 1,
      default: 28,
      unit: 'days',
      disabledWhen: state => !state.playWindow,
      help: 'Length of the sliding window.'
    },
    {
      kind: 'select',
      id: 'groupType',
      label: 'Group of life',
      group: 'Time window',
      apply: 'param',
      default: 'all',
      options: GROUPS,
      help: 'Passed to the time filter as an extra selection predicate (a mask buffer), so it combines with the date window.'
    },
    {
      kind: 'range',
      id: 'hours',
      label: 'Hours in the brush',
      group: 'Brush',
      apply: 'param',
      min: 0,
      max: 23,
      step: 1,
      default: [0, 23],
      format: value => formatHour(value),
      help: 'Only observations whose local hour falls in this range (inclusive) feed the area statistics. The matrix on the map keeps every hour and dims the rest.'
    },
    {
      kind: 'toggle',
      id: 'invertHours',
      label: 'Invert the hour brush',
      group: 'Brush',
      apply: 'param',
      default: false,
      help: 'Selects the hours outside the range, which makes windows that wrap past midnight easy: brush 07:00 to 14:00 and invert to see everything else.'
    },
    {
      kind: 'select',
      id: 'weekdays',
      label: 'Weekdays in the brush',
      group: 'Brush',
      apply: 'param',
      default: 'all',
      options: [
        {value: 'all', label: 'Every day'},
        {value: 'weekdays', label: 'Monday to Friday'},
        {value: 'weekend', label: 'Saturday and Sunday'},
        {value: 'friday-saturday', label: 'Friday and Saturday'},
        {value: 'mon', label: 'Monday'},
        {value: 'tue', label: 'Tuesday'},
        {value: 'wed', label: 'Wednesday'},
        {value: 'thu', label: 'Thursday'},
        {value: 'fri', label: 'Friday'},
        {value: 'sat', label: 'Saturday'},
        {value: 'sun', label: 'Sunday'}
      ],
      help: 'The weekday set of the brush, as a bit mask rewritten in a four-word parameter buffer.'
    },
    {
      kind: 'select',
      id: 'statistic',
      label: 'Statistic per area',
      group: 'Area statistic',
      apply: 'param',
      default: 'count',
      options: [
        {value: 'count', label: 'Observations (count)'},
        {value: 'density', label: 'Observations per square kilometer'},
        {value: 'mean-hour', label: 'Mean hour of day'},
        {value: 'median-hour', label: 'Median hour of day'},
        {value: 'percentile-hour', label: 'Percentile of the hour of day'},
        {value: 'mode-hour', label: 'Most common hour'},
        {value: 'hour-spread', label: 'Spread of the hour (standard deviation)'},
        {value: 'active-days', label: 'Days with at least one observation (unique count)'},
        {value: 'research-grade-share', label: 'Research-grade share (mean of a 0/1 flag)'},
        {value: 'introduced-share', label: 'Introduced-species share (mean of a 0/1 flag)'}
      ],
      help: 'GPUGroupStatistics computes them all every run; this only picks which output buffer the map reads. Hours are numeric, so a mean of 23 and 1 is 12: read mean hours for areas with a single daily peak.'
    },
    {
      kind: 'slider',
      id: 'percentile',
      label: 'Percentile',
      group: 'Area statistic',
      apply: 'param',
      min: 0.05,
      max: 0.95,
      step: 0.05,
      default: 0.9,
      format: value => `${Math.round(value * 100)}th`,
      disabledWhen: state => state.statistic !== 'percentile-hour',
      help: 'The quantile fraction of the hour distribution. It is a per-frame parameter, not a rebuild: 0.9 means nine in ten observations are made by that hour.'
    },
    {
      kind: 'select',
      id: 'variance',
      label: 'Standard deviation uses',
      group: 'Area statistic',
      apply: 'compile',
      default: 'sample',
      options: [
        {value: 'sample', label: 'Sample variance (n - 1)'},
        {value: 'population', label: 'Population variance (n)'}
      ],
      help: 'The variance kind is fixed when the statistics graph is compiled. It only changes the spread statistic, and only for small groups.'
    },
    {
      kind: 'slider',
      id: 'minimumEvents',
      label: 'Hide areas with fewer observations',
      group: 'Area statistic',
      apply: 'param',
      min: 0,
      max: 300,
      step: 5,
      default: 20,
      unit: 'observations',
      help: 'Areas whose brush holds fewer observations than this are left blank, because medians and rates of a handful of observations are noise.'
    },
    {
      kind: 'slider',
      id: 'utcOffset',
      label: 'Clock offset',
      group: 'Calendar',
      apply: 'param',
      min: -12,
      max: 12,
      step: 1,
      default: 0,
      format: value => (value === 0 ? 'as published' : `${value > 0 ? '+' : ''}${value} h`),
      help: 'The fixed UTC offset applied by GPUCalendarBuckets before it decodes hour and weekday. iNaturalist records the local clock time of each observation, so 0 is correct; other values show how the matrix shifts if a clock is misread.'
    },
    {
      kind: 'select',
      id: 'firstDay',
      label: 'Week starts on',
      group: 'Calendar',
      apply: 'param',
      default: '0',
      options: [
        {value: '0', label: 'Monday (ISO)'},
        {value: '6', label: 'Sunday (US)'},
        {value: '5', label: 'Saturday'}
      ],
      help: 'Decides which weekday is row 0 of the matrix and weekday 0 of the calendar columns. Changing it reorders the heat map rows but not the observations.'
    },
    {
      kind: 'toggle',
      id: 'daylightSaving',
      label: 'Pretend timestamps are UTC and apply Chicago daylight saving',
      group: 'Calendar',
      apply: 'compile',
      default: false,
      help: 'Adds the per-row UTC offset column (-6 h in winter, -5 h in summer) that is the only way GPUCalendarBuckets handles daylight saving. The data are already local time, so this is a demonstration of the mechanism: the whole matrix slides by five or six hours.'
    },
    {
      kind: 'toggle',
      id: 'showAreas',
      label: 'Community area statistic',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Colors each of the 77 community areas by the chosen statistic.'
    },
    {
      kind: 'toggle',
      id: 'showPoints',
      label: 'Observation points colored by hour',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws the observations inside the date window, compacted on the GPU and drawn with an indirect command, so no count is read back.'
    },
    {
      kind: 'toggle',
      id: 'showMatrix',
      label: 'Hour by weekday matrix in the lake',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'The calendar matrix drawn east of the Loop. Columns are hours 0 to 23 (grid lines at 06:00, 12:00, 18:00); rows are weekdays from the first day of the week at the top. Dimmed cells are outside the brush.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'cividis',
      options: [
        {value: 'ylgnbu', label: 'Yellow-green-blue'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
      ],
      help: 'Ramp of the area statistic.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Area opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.85,
      help: 'Lower it to read streets under the areas.'
    }
  ],

  readouts: [
    {id: 'windowDates', label: 'Date window'},
    {
      id: 'windowEvents',
      label: 'Observations in the window',
      help: 'Accepted by GPUTimeWindowFilter, counted on the GPU.'
    },
    {id: 'brushEvents', label: 'Observations in the brush'},
    {
      id: 'busiestCell',
      label: 'Busiest hour of the week',
      help: 'The largest cell of the hour by weekday matrix.'
    },
    {id: 'topArea', label: 'Highest area for the statistic'},
    {id: 'overflow', label: 'Capacity overflow'}
  ],

  legends: state => {
    const info = STATISTIC_TITLES[state.statistic];
    const legends: ReturnType<NonNullable<Parameters<typeof defineScene>[0]['legends']>>[number][] =
      [];
    if (state.showAreas) {
      legends.push({
        kind: 'ramp',
        id: 'metric',
        title: `${info.title}${state.statistic === 'percentile-hour' ? ` (${Math.round(state.percentile * 100)}th)` : ''}`,
        ramp: state.ramp,
        extent: 'gpu',
        sqrtScale: ['count', 'density', 'active-days'].includes(state.statistic),
        unit: info.unit,
        labels: info.labels,
        format: value =>
          state.statistic === 'research-grade-share' || state.statistic === 'introduced-share'
            ? `${(value * 100).toFixed(0)}%`
            : value >= 100
              ? value.toFixed(0)
              : value.toFixed(1)
      });
    }
    if (state.showMatrix) {
      legends.push({
        kind: 'ramp',
        id: 'matrix',
        title: 'Hour by weekday matrix (in the lake)',
        ramp: 'inferno',
        extent: 'gpu',
        sqrtScale: true,
        unit: 'observations',
        format: value => value.toFixed(0)
      });
    }
    if (state.showPoints) {
      legends.push({
        kind: 'ramp',
        title: 'Observation hour of day',
        ramp: 'cividis',
        extent: [0, 24],
        labels: ['00:00', '24:00'],
        unit: 'h'
      });
    }
    return legends;
  },

  snippet: state => `import {
  getGPUCalendarBucketsParameterValues, getGPUTimeWindowWordParameterValues,
  GPUCalendarBuckets, GPUGroupStatistics, GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';

// 1. exact Int64 time window, plus a group mask as an extra predicate
graph.add(new GPUTimeWindowFilter({
  timestamps: timeWords,                    // uint32x2 epoch milliseconds
  window: windowParameters,                 // 8 uint32 words
  additionalPredicates: [{kind: 'selection', mask: groupMask}],
  output: {ids, count, overflow}, outputMask: windowMask,
  drawInstanceCount: drawCommands.getInstanceCountData(0)   // indirect draw: no readback
}));
// 2. hour, weekday, day of year and the 7 x 24 count matrix
graph.add(new GPUCalendarBuckets({
  timestamps: timeWords, mask: windowMask,${state.daylightSaving ? '\n  utcOffsets,                               // per-row offset column: the daylight saving path' : ''}
  parameters: calendarParameters,
  output: {hour, weekday, dayOfYear, hourWeekdayCounts: matrix}
}));
// 3. per-area statistics of the brushed observations (dense keys: one row per community area)
graph.add(new GPUGroupStatistics({
  keys: areaKeys, mask: brush, keyCount: 77, variance: '${state.variance}', percentiles: percentileFractions,
  columns: [
    {values: hourFloat, statistics: ['count', 'mean', 'median', 'standardDeviation', 'percentiles', 'mode'], output: hourOutputs},
    {values: dayFloat, statistics: ['uniqueCount'], output: {uniqueCounts}},
    {values: researchGradeFlags, statistics: ['mean'], output: {means: researchGradeShare}}
  ],
  output: {keys, counts, count, overflow}
}));
// parameters, per frame: no rebuild
windowParameters.write(getGPUTimeWindowWordParameterValues({start: ${Date.UTC(2023, 0, 1 + state.dateRange[0])}, end: ${Date.UTC(2023, 0, 1 + state.dateRange[1]) - 1}}));
calendarParameters.write(getGPUCalendarBucketsParameterValues(${state.utcOffset * 60}, ${state.firstDay}));`,

  about: {
    what: '`GPUTimeWindowFilter` keeps the observations inside an exact Int64 time window (and any extra mask). `GPUCalendarBuckets` turns every kept timestamp into hour, weekday and day of the year with integer arithmetic on the GPU and counts an hour by weekday matrix. `GPUGroupStatistics` then reduces the brushed observations per community area: count, mean, median, a percentile, mode, spread, unique days and shares.',
    why: 'Where and when are usually asked together: "where do people see birds on a May morning" is a different map from "where do people see birds". Doing the calendar decode and the grouped statistics on the GPU means the brush, the window and the statistic respond immediately. Because the data are community-science records, the calendar also shows when the observers are out, not only when the wildlife is.',
    howToRead:
      'The matrix in the lake shows the whole week: bright cells are busy hours. Dimmed cells are outside your brush. The colored areas summarise only the brushed observations; hover one for its value and event count.'
  },

  create: async ctx => (await import('./calendar-patterns.compute')).createCalendarPatterns(ctx),

  story: [
    {
      id: 'question',
      title: 'When does the week of observing peak?',
      headline: 'Observation activity peaks Saturday late morning',
      textAlternative:
        'An hour-by-weekday matrix peaks on Saturday at 11:00 while community areas show annual observation counts.',
      body: 'Every iNaturalist observation has a timestamp. **`GPUCalendarBuckets`** decodes all 43,557 of them into hour and weekday on the GPU, with integer arithmetic and no per-row `Date`, and counts them in a 7 by 24 matrix. The matrix is drawn in Lake Michigan east of the Loop: columns are hours 0 to 23 (grid lines every six hours), rows are weekdays from Monday at the top.\n\nThe community areas are colored by the number of observations (**Statistic per area**, below). The busiest are Lincoln Park (5,526), Uptown (4,887) and Lincoln Square (4,142): the lakefront parks and Montrose Point draw the observers. The brightest matrix cell is Saturday at 11:00, with 1,348 observations. Almost nothing happens between midnight and 05:00: this is a daylight, human-paced rhythm.',
      options: {statistic: 'count', showMatrix: true},
      highlight: {readout: 'busiestCell'},
      controls: ['statistic', 'showMatrix'],
      readouts: ['busiestCell', 'topArea']
    },
    {
      id: 'window',
      title: 'Cut the year with an exact time window',
      headline: 'Four-week windows expose seasonal observation peaks',
      textAlternative:
        'A moving four-week window updates the calendar matrix and community-area counts as observation activity rises and falls through 2023.',
      body: '**`GPUTimeWindowFilter`** keeps only observations inside a date window. Times are exact Int64 milliseconds (an Arrow-style timestamp column) and the window is eight parameter words, so dragging **Date window** is a buffer write. The group mask (**Group of life**) rides along as an extra predicate.\n\nTurn on **Slide a window through the year**, set its length with **Sliding window length**, and watch the matrix and the area map follow the seasons. July is the peak month (6,554 observations) and December the quietest (551). Pick *Birds* and the window lights up in April and May, the spring migration; pick *Insects* and it swells in July; pick *Fungi* and it stays alive into October. The date window feeds the calendar decode, so the matrix always describes the window.',
      options: {playWindow: true, windowDays: 28},
      controls: ['dateRange', 'playWindow', 'windowDays', 'groupType'],
      readouts: ['windowDates', 'windowEvents']
    },
    {
      id: 'brush',
      title: 'Brush hours and weekdays',
      headline: 'Weekend morning records concentrate near lakefront parks',
      textAlternative:
        'The matrix highlights weekend hours from 06:00 to 10:00 and the map shows the density of matching records by community area.',
      body: 'Set **Hours in the brush** to 06:00 to 10:00 and **Weekdays in the brush** to *Saturday and Sunday*: the early weekend birding window. The matrix dims everything outside the brush and the area statistics now describe only those mornings. Tick **Invert the hour brush** to see the complement instead: midday, afternoon and evening.\n\nThe brush is a four-word parameter buffer applied by a small kernel to the calendar columns, so it responds instantly. Set **Statistic per area** to *Observations per square kilometer* and compare weekend mornings with weekday afternoons. Observers record more per day on weekends (about 139 per Saturday or Sunday against about 111 per weekday, from the 2023 totals). Compare the maps to see whether the same places lead.',
      options: {
        hours: [6, 10],
        invertHours: false,
        weekdays: 'weekend',
        statistic: 'density'
      },
      controls: ['hours', 'invertHours', 'weekdays', 'statistic'],
      readouts: ['brushEvents']
    },
    {
      id: 'group-statistics',
      title: 'Group statistics: when does each area peak?',
      headline: 'Median observation hour varies among community areas',
      textAlternative:
        'Community areas are shaded from earlier to later median observation hours after GPU grouped statistics.',
      body: '**`GPUGroupStatistics`** reduces the brushed observations per community area in dense mode (one output row per area, in feature order, so the table maps directly onto the polygons). It computes count, mean, median, standard deviation, a percentile, the mode, unique days, and the mean of the research-grade and introduced flags, all in one pass; the map just selects which output buffer to color.\n\nSwitch **Statistic per area** to *Median hour of day* with every day and hour brushed. Areas are shaded from earlier to later in the day: Uptown, home of Montrose Point, sits near 11:00, while Lincoln Square is closer to 16:00. Hours are numbers on a line, so these are best for areas with one daily peak.',
      options: {
        hours: [0, 23],
        invertHours: false,
        weekdays: 'all',
        statistic: 'median-hour',
        ramp: 'cividis',
        playWindow: false
      },
      controls: ['statistic'],
      readouts: ['topArea']
    },
    {
      id: 'percentile',
      title: 'Percentiles, shares and unique days',
      headline: 'Sparse areas produce unstable hourly summaries',
      textAlternative:
        'Community areas show the 90th-percentile observation hour, with areas below 30 records left blank.',
      body: "Set **Statistic per area** to *Percentile of the hour of day* and move **Percentile**: at 0.9, nine in ten of an area's observations are made by that hour. Try *Research-grade share* (the mean of a 0/1 flag): Uptown is near 78% community-confirmed, Lincoln Park near 51%, so the same species list is easier to confirm in some places than others. *Introduced-species share* maps where non-native taxa dominate (Lincoln Park is near 15%, Uptown near 11%). *Days with at least one observation* (a unique count of the day of the year) separates places watched all year from places visited in bursts.\n\nAreas with fewer observations than **Hide areas with fewer observations** are left blank, because shares and medians of a handful of records are noise. The **Standard deviation uses** option (sample or population variance) is compiled into the statistics graph, so it counts as a rebuild.",
      options: {statistic: 'percentile-hour', percentile: 0.9, minimumEvents: 30},
      controls: ['statistic', 'percentile', 'minimumEvents', 'variance'],
      readouts: ['topArea']
    },
    {
      id: 'clock',
      title: 'Clocks and daylight saving',
      headline: 'Fixed offsets shift the calendar matrix',
      textAlternative:
        'The hour-by-weekday matrix shifts when a synthetic UTC offset and Chicago daylight-saving adjustments are applied.',
      body: 'iNaturalist records the local clock time of each observation, so a **Clock offset** of 0 is right. Calendar decoding takes one fixed UTC offset as a parameter; set +6 and the matrix slides six columns, as if the timestamps had been in UTC. **Week starts on** reorders the rows (Monday, Sunday or Saturday first) without touching the observations.\n\nDaylight saving cannot be a single offset. **Pretend timestamps are UTC and apply Chicago daylight saving** adds a per-row offset column (-6 h in winter, -5 h in summer), which is how `GPUCalendarBuckets` handles zones with transitions. The data are already local, so the result is deliberately wrong: it shows the mechanism, and why a published clock time must never be converted twice.',
      options: {statistic: 'count', daylightSaving: true, utcOffset: 0},
      controls: ['utcOffset', 'firstDay', 'daylightSaving'],
      readouts: ['busiestCell']
    },
    {
      id: 'limits',
      title: 'Limits, and things to try',
      headline: 'Observation counts reflect effort and habitat',
      textAlternative:
        'Observation points and area counts show concentrated recording effort around parks and organized events.',
      body: 'Observations follow observers: counts measure effort as much as wildlife. The City Nature Challenge weekend (28 April to 1 May 2023) puts a spike into the calendar, and the busiest single day, 21 June, has 667 records. Timestamps are local clock times stored as UTC. Statistics of hours are linear, not circular. Community areas are large and uneven, so counts per area say more about area size and park cover than about abundance: use density.\n\nTry: a single **Group of life** (*Birds*, *Insects*, *Fungi*) and see its own weekly and seasonal rhythm; **Observation points colored by hour** to see the compacted point set drawn by an indirect command; and **Statistic per area** set to *Median hour of day* for *Birds* versus *Insects*.',
      options: {daylightSaving: false, showPoints: true, statistic: 'count', groupType: 'all'},
      controls: ['groupType', 'showPoints', 'statistic']
    }
  ]
});
