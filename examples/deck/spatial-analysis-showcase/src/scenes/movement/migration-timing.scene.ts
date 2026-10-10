// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CREDITS, joinCredits} from '../../cartography/credits';
import {labelsFor, WORLD} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import type {MapGround} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import type {PaletteColor} from '../../engine/ramps';
import {playbackOptions} from '../../engine/playback';
import {defineScene, type LegendSpec} from '../scene';
import type {MigrationTimingOptions} from './migration-timing.compute';
import {formatYearDay, MIGRATION_SEASONS} from './migration-shared';
import {getOverlayInks, getTimingTable, type TimingMode} from './migration-timing-panel';
import {SPECIES_MASKS, THIN_FIXES} from './migration-timing-stats';
import {FLYWAY} from './movement-places';
import {
  FOLDED_MONTH_TICKS,
  formatFoldedDay,
  getSpeciesLegend,
  MOVEMENT_CREDITS
} from './movement-style';

/** Camera of the first and last step: the tracks on the coast and the panel in the Atlantic. */
const MAP_AND_PANEL = {bounds: [-61, -5, 17, 63] as const};
/** Camera of the steps about the panel itself. */
const PANEL_ONLY = {bounds: [-61, -5, -18, 66] as const};

/** The cartouche of one step: the claim, the variable and method, and the honesty chips. */
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  chips: ['Years folded', 'A fix is not a bird'] as const
});

/** What the compute module publishes for the legends (`ctx.setLegendData('timing', ...)`). */
type TimingLegendData = {
  ground: MapGround;
  tables: Record<TimingMode, ClassTable>;
  classCounts: Record<TimingMode, number[]> | null;
  speciesCounts: number[];
  groupLabels: string[];
  groupColors: PaletteColor[];
};

const LEGEND_TITLES: Record<TimingMode, string> = {
  share: 'Share of the bucket’s fixes in the band',
  count: 'Fixes in the band and bucket',
  speed: 'Fastest step in the band and bucket'
};

function getLegends(
  state: MigrationTimingOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const timing = data['timing'] as TimingLegendData | undefined;
  const groundName: MapGround = timing?.ground ?? 'light';
  const table = timing?.tables[state.cellValue] ?? getTimingTable(state.cellValue, groundName, []);
  const legends: LegendSpec[] = [
    getClassTableLegend(table, {
      title: LEGEND_TITLES[state.cellValue],
      basis: state.cellValue === 'share' ? 'of that bucket’s fixes' : undefined,
      counts: timing?.classCounts?.[state.cellValue],
      layout: 'list'
    })
  ];
  const marks: {color: PaletteColor; label: string; shape?: 'swatch' | 'hatch'}[] = [
    {
      color: getOverlayInks(groundName).hatch,
      label: `Fewer than ${THIN_FIXES} fixes in the cell`,
      shape: 'hatch'
    }
  ];
  if (state.showGroups && timing) {
    for (let group = timing.groupLabels.length - 1; group >= 0; group--) {
      marks.push({
        color: timing.groupColors[group % timing.groupColors.length],
        label: `${timing.groupLabels[group]}° N`
      });
    }
  }
  legends.push({
    kind: 'categories',
    title: 'Marks on the panel',
    entries: marks,
    layout: 'list',
    note: state.showGroups
      ? 'The coloured ruler beside the panel marks the latitude groups of the band chart.'
      : 'Empty cells are transparent: the paper shows.'
  });
  if (state.showTracks) {
    legends.push(
      getSpeciesLegend(
        groundName,
        timing?.speciesCounts,
        'Faint lines are the tag tracks; an animal-year is one bird in one tagged year.'
      )
    );
  }
  return legends;
}

export default defineScene<MigrationTimingOptions>({
  id: 'migration-timing',
  title: 'When does the population pass each latitude?',
  chapter: 'movement',
  order: 6,
  summary:
    'A Hovmöller diagram drawn on the map: GPUTemporalReduction drops every GPS fix into a (species, latitude band, time bucket) slot, and the year becomes a slanted stripe whose slope is the speed of the migration.',
  contributors: ['GPUTemporalReduction'],
  datasets: [
    {id: 'poopdeck-animals', role: 'GPS tracks of 42 birds, years folded onto one calendar'}
  ],
  initialView: {longitude: -22, latitude: 33, zoom: 3.3},

  options: [
    {
      kind: 'select',
      id: 'species',
      label: 'Species',
      group: 'Space-time cube',
      apply: 'param',
      default: 'all',
      display: 'chips',
      help: 'Which species are folded into the matrix: a mask passed to the display kernel. The reduction keeps one cell per species and band, so switching never recompiles.',
      options: [
        {value: 'all', label: 'All three'},
        {value: 'marsh', label: 'Marsh harrier'},
        {value: 'montagu', label: "Montagu's harrier"},
        {value: 'spoonbill', label: 'Spoonbill'}
      ]
    },
    {
      kind: 'slider',
      id: 'bucketDays',
      label: 'Bucket width',
      group: 'Space-time cube',
      apply: 'param',
      min: 3,
      max: 14,
      step: 1,
      default: 7,
      unit: 'days',
      marks: [{value: 7, label: 'week'}],
      autoSweep: {from: 3, to: 14, durationMs: 9000},
      describe: value => `${value} days = ${Math.ceil(366 / value)} buckets`,
      help: 'Width of one time bucket: a parameter-buffer write that re-bins every fix without a recompile. The cube holds at most 128 buckets, so the narrowest width is 3 days (a year is 366 days).'
    },
    {
      kind: 'select',
      id: 'cellValue',
      label: 'Cell value',
      group: 'Space-time cube',
      apply: 'param',
      default: 'share',
      display: 'segmented',
      help: "Share is the part of the bucket's fixes that fall in the band (a column adds up to 100 %): where the population is. Count is the number of fixes: it follows how many fixes the tags delivered. Fastest step is the maximum the reduction kept for the slot.",
      options: [
        {value: 'share', label: 'Share'},
        {value: 'count', label: 'Count'},
        {value: 'speed', label: 'Fastest step'}
      ]
    },
    {
      kind: 'slider',
      id: 'probeLatitude',
      label: 'Probe latitude',
      group: 'Probe',
      apply: 'param',
      min: 6,
      max: 59.5,
      step: 0.5,
      default: 36,
      format: value => `${value}° N`,
      help: 'The latitude you read across to: a dashed line from the coast to the matrix row that holds it. The Probe band readout and the crossing dates follow it. Click the panel to set it with the day.'
    },
    {
      kind: 'toggle',
      id: 'showProbe',
      label: 'Show the probe',
      group: 'Probe',
      apply: 'param',
      default: true,
      help: 'Draws the probe latitude as a dashed line from the coast to the panel and outlines its row.'
    },
    {
      kind: 'toggle',
      id: 'showCrossings',
      label: 'Mark the crossing dates',
      group: 'Probe',
      apply: 'param',
      default: false,
      help: 'Pins the dates on which the median latitude of the selected species crosses the probe latitude, northbound and southbound, to the panel.'
    },
    {
      kind: 'toggle',
      id: 'showTrace',
      label: 'Show the median latitude',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws the median latitude of the selected species fixes in each bucket as a line over the matrix. Its slope is the speed of the migration.'
    },
    {
      kind: 'toggle',
      id: 'showBucketEdges',
      label: 'Show bucket edges',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws the edges between time buckets as faint vertical lines, so the width you choose is visible.'
    },
    {
      kind: 'toggle',
      id: 'showGroups',
      label: 'Show the latitude groups',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws a coloured ruler beside the panel for the five 12-degree latitude groups, the colours of the band chart.'
    },
    {
      kind: 'toggle',
      id: 'showTracks',
      label: 'Show the tracks',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws every animal-year as a thin line on the map, so the rows of the matrix can be compared with the real routes.'
    },
    {
      kind: 'toggle',
      id: 'showFixes',
      label: 'Show the fixes of the bucket',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws the GPS fixes of the time bucket under the cursor as dots on the map, rebuilt only when the bucket changes.'
    },
    {
      kind: 'slider',
      id: 'trackOpacity',
      label: 'Track opacity',
      group: 'Display',
      apply: 'param',
      min: 0.05,
      max: 1,
      step: 0.05,
      default: 0.3,
      expert: true,
      disabledWhen: state => !state.showTracks,
      help: 'Lower it to read the matrix; raise it to follow the routes.'
    },
    ...playbackOptions<MigrationTimingOptions>({
      ids: {play: 'play', time: 'day', speed: 'playSpeed', loop: 'loop'},
      group: 'Playback',
      playing: false,
      time: {
        min: 0,
        max: 365,
        step: 1,
        default: 120,
        label: 'Day of the year',
        format: formatYearDay,
        help: 'Moves the cursor across the matrix; the profile, the cursor readout and the dots follow it.'
      },
      speed: {
        min: 1,
        max: 30,
        step: 1,
        default: 6,
        unit: 'days/s',
        label: 'Play speed',
        help: 'Days of the folded year per real second. At 6 days per second a year takes a minute.'
      },
      loop: true
    })
  ],

  readouts: [
    {
      id: 'tracks',
      label: 'Sample',
      help: 'Tagged birds, animal-years (one bird in one tagged year) and GPS fixes in the data set.'
    },
    {
      id: 'bandChart',
      label: 'Where the fixes are, by latitude group',
      kind: 'chart',
      help: 'Share of the selected fixes in each 12-degree latitude group, bucket by bucket (the five areas add up to 100 %). The colours are the ruler beside the panel. The hand-over from the north to the south in late summer is the autumn migration. The rule is the cursor.'
    },
    {
      id: 'occupancyChart',
      label: 'Occupied slots by bucket width',
      kind: 'chart',
      help: 'The share of the (species, band, bucket) slots that hold at least one fix, for every width from 3 to 14 days. It was measured once at load by writing each width into the same compiled graph. Click the chart to set the bucket width.'
    },
    {
      id: 'buckets',
      label: 'Time buckets',
      help: 'How many buckets hold the folded year at this width. The last one is shorter when the width does not divide the year.'
    },
    {
      id: 'reduction',
      label: 'Occupied slots',
      hood: true,
      help: 'Slots (species, latitude band, bucket) that hold at least one fix, as reported by GPUTemporalReduction, out of all slots at this width.'
    },
    {
      id: 'weeklyFixes',
      label: 'Fixes per bucket',
      help: 'The fewest and the most fixes of the selection in a bucket that holds a full width: the denominator of every share.'
    },
    {
      id: 'fixesChart',
      label: 'The denominator: fixes per bucket',
      kind: 'chart',
      help: 'GPS fixes of the selection in each bucket across the year, with the seasons shaded. A share divides each column of the matrix by this number; a count is this number split into bands.'
    },
    {
      id: 'springWeeks',
      label: 'Northbound, 20 to 50° N',
      help: 'Time the median latitude of the selected species needs to rise from 20 to 50° N in spring, with the dates it crosses each (interpolated between bucket centres).'
    },
    {
      id: 'autumnWeeks',
      label: 'Southbound, 50 to 20° N',
      help: 'Time the median latitude needs to fall from 50 to 20° N in autumn, with the dates it crosses each.'
    },
    {
      id: 'crossings',
      label: 'The median crosses the probe latitude',
      help: 'The dates the median latitude of the selected species rises through (north) and falls through (south) the probe latitude.'
    },
    {
      id: 'fastest',
      label: 'Fastest step in the sample',
      help: 'The maximum the reduction kept in any cell of the selection: the fastest step between two consecutive fixes that ends in the cell, and where it is. One position error between two fixes can also make a step look fast.'
    },
    {
      id: 'stepLength',
      label: 'Time between fixes',
      help: 'The median and the shortest time between two consecutive fixes of one animal-year. The data were thinned to a fixed minimum, which sets the length of a step.'
    },
    {
      id: 'thinCells',
      label: 'Hatched cells',
      help: 'Occupied cells that hold fewer fixes than the suppression limit. Their shares and speeds rest on very few fixes.'
    },
    {
      id: 'profileChart',
      label: 'Latitude profile at the cursor',
      kind: 'chart',
      help: "The share of the fixes of the cursor's time bucket in each 3-degree latitude band, north at the top so the bars line up with the rows of the matrix. Press Play and watch the mass move."
    },
    {
      id: 'cursor',
      label: 'Cursor',
      help: 'The bucket under the cursor, its fixes and the median latitude.'
    },
    {
      id: 'probe',
      label: 'Probe band',
      help: "The probe band's fix count and its busiest buckets in the first and the second half of the year. For a band the birds merely cross, those are the northbound and southbound passages; for a band they live in, the busiest weeks."
    }
  ],

  pipeline: [
    {
      id: 'reduce',
      label: 'Reduce',
      detail:
        'Every fix goes into one (species, band, bucket) slot: count, minimum, maximum, first, last'
    },
    {
      id: 'fold',
      label: 'Fold',
      detail: 'A kernel folds the chosen species into a band-by-bucket matrix'
    },
    {
      id: 'classes',
      label: 'Classes',
      detail: 'Share, count or fastest step is cut into classes and drawn from the same buffer'
    },
    {
      id: 'read',
      label: 'Read back',
      detail: 'The CPU reads the cube once for the charts, the crossing dates and the tooltips'
    }
  ],

  legends: getLegends,

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUTemporalReduction, getGPUTemporalReductionParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// One cell per (species, 3-degree latitude band); 128 time buckets.
// cellIds: uint32 per fix, timestamps: float32 seconds of the folded year,
// values: the speed of the step that ends at each fix, in km/h.
graph.add(new GPUTemporalReduction({
  cellIds, timestamps, values: speeds,
  parameters: bucketParameters.importToGraph(graph),
  cellCount: 3 * rowCount, bucketCount: 128,
  output: {counts, min, max, first, last, occupiedSlots: {ids, count, overflow}}
}));
const compiled = graph.compile();            // once
// A bucket width is a buffer write: ${state.bucketDays} days = ${Math.ceil(366 / state.bucketDays)} buckets.
bucketParameters.write(getGPUTemporalReductionParameterValues(0, ${state.bucketDays} * 86400));

// Display kernel: fold the chosen species (mask ${SPECIES_MASKS[state.species]}) into a band-by-bucket matrix.
//   value = ${state.cellValue === 'share' ? 'count in the band / count in the column' : state.cellValue === 'count' ? 'count in the band' : 'max step speed of the slot'}
// A raster layer draws the matrix at its true latitudes with the class table of the legend.
new SpatialAnalysisRasterLayer({gridSize: [validBuckets, rowCount], bounds, values: matrix,
  colormap: 'uniform', ...getClassTableLayerProps(table), tessellation: rowCount});`,

  about: {
    what: 'Previously: *Play a year of migration* showed the whole flock as moving dots. Next: *Which way do the harriers and spoonbills fly?*\n\n`GPUTemporalReduction` drops every GPS fix into one slot per cell and time bucket, here a species and a 3-degree latitude band and a bucket of a few days. Each slot keeps its fix count and the minimum, maximum, first and last of a per-fix value, here the speed of the step that ends at the fix (so the maximum is the fastest step), with atomics, so the result does not depend on thread order. The bucket width is a parameter write, never a recompile; a display kernel then folds the species you pick into a latitude-by-bucket matrix.',
    why: 'Timing is conservation data: when a wetland should be quiet, and how fast a population passes a latitude, depend on where the birds are on the calendar. Time along one axis and latitude along the other (a Hovmöller diagram) turns a migration into a slanted stripe whose slope is its speed, for every latitude at once. Two choices change the picture: the bucket width (time has its own modifiable-unit problem) and whether a cell shows a count or a share.',
    howToRead:
      'The paper panel west of Africa is a chart drawn on the map. Columns are time buckets of the folded year (1 January at the left, 31 December at the right); rows are 3-degree latitude bands at their true latitudes, so you can read across to the coast (Web Mercator makes the northern rows taller, so the summer block looks bigger than it is). A stripe running down to the right is a southbound migration, one running up to the right is northbound, and horizontal blocks are stays. Empty cells are paper; hatched cells hold fewer than 20 fixes. Times are float32 seconds of the year, exact to 2 s after day 194: a fix within 2 s of a bucket edge late in the year can land in the neighbouring bucket (the contributor has an exact uint32x2 mode). Speeds are computed once at load on the CPU from consecutive fixes.'
  },

  // Paper ground: a figure on a quiet map, the matrix is the subject and the labels sit above it.
  basemap: ground('paperCity', {
    suppressNames: ['Sahara', 'Sahel', 'Iberia', 'Strait of Gibraltar', 'Low Countries']
  }),
  furniture: {
    title: cartouche(
      'When is the population at each latitude?',
      'Share of weekly GPS fixes by 3° band · years folded onto 2024'
    ),
    credit: joinCredits(MOVEMENT_CREDITS.birds, CREDITS.carto),
    caveat: 'Rows sit at true latitude; Web Mercator makes the northern rows taller.'
  },
  annotations: [
    ...labelsFor(FLYWAY, ['sahel', 'sahara', 'iberia', 'low-countries', 'banc-d-arguin'], {
      'banc-d-arguin': {minZoom: 3}
    }),
    ...labelsFor(WORLD, ['strait-of-gibraltar'], {'strait-of-gibraltar': {minZoom: 3}})
  ],

  timeline: {
    time: 'day',
    play: 'play',
    speed: 'playSpeed',
    format: formatFoldedDay,
    bands: [
      {from: 0, to: MIGRATION_SEASONS.spring.days[0], label: 'Winter'},
      {
        from: MIGRATION_SEASONS.spring.days[0],
        to: MIGRATION_SEASONS.spring.days[1],
        label: 'Spring'
      },
      {
        from: MIGRATION_SEASONS.breeding.days[0],
        to: MIGRATION_SEASONS.breeding.days[1],
        label: 'Breeding'
      },
      {
        from: MIGRATION_SEASONS.autumn.days[0],
        to: MIGRATION_SEASONS.autumn.days[1],
        label: 'Autumn'
      },
      {from: MIGRATION_SEASONS.winter.days[0], to: 366, label: 'Winter'}
    ],
    ticks: FOLDED_MONTH_TICKS
  },

  create: async ctx => (await import('./migration-timing.compute')).createMigrationTiming(ctx),

  story: [
    {
      id: 'the-chart-is-a-map',
      title: 'The chart is a map',
      headline: 'Rows of the chart sit at true latitudes',
      textAlternative:
        'Map of western Europe and North Africa with a white chart panel in the Atlantic. Its columns are weeks of the year and its rows are latitude bands, shaded blue-green by the share of GPS fixes; a dashed line joins one row to the coast.',
      body: "Tracks say where; this chart says when. The block west of Africa is a calendar drawn on the map: each column is a week, each row a latitude band at its true latitude, shaded by its share of that week's GPS fixes. Drag **Probe latitude** and read straight across to the coast. The sample is **{{tracks}}**.\n\n*Time is a coordinate here, not an animation.*",
      evidence:
        'The matrix and its latitude-group chart aggregate the same **{{tracks}}**, with every column normalised over the selected fixes.',
      caveat:
        '**{{tracks}}** reports animal-years and fixes, not a representative census of either species.',
      optionsMode: 'fresh',
      options: {showGroups: true},
      controls: ['probeLatitude'],
      readouts: ['bandChart', 'tracks'],
      camera: {...MAP_AND_PANEL, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'When is the population at each latitude?',
          'Share of weekly GPS fixes by 3° band · years folded onto 2024'
        )
      },
      stage: 'reduce'
    },
    {
      id: 'slice-the-year',
      title: 'A bucket width is a choice',
      headline: 'Narrow buckets are noisy, wide ones blur timing',
      textAlternative:
        'The chart panel with faint vertical lines between the time buckets; with narrow buckets many cells are empty, with wide buckets the stripe blurs.',
      body: '`GPUTemporalReduction` drops every fix into a slot: species, latitude band and a time bucket. Slide **Bucket width** (or press play on it): a parameter write, so the same compiled graph re-bins every fix. The year has **{{buckets}}** and **{{reduction}}**. Tick **Show bucket edges** to see them.\n\n*Time has its own MAUP: the bucket is a choice.*',
      evidence:
        'The current binning produces **{{buckets}}** and **{{reduction}}**; the occupancy chart compares that result with every available width.',
      caveat:
        '**{{reduction}}** is conditional on bucket width: narrower bins reveal timing but create more sparse slots.',
      optionsMode: 'fresh',
      options: {showBucketEdges: true, showTracks: false},
      controls: ['bucketDays', 'showBucketEdges'],
      readouts: ['occupancyChart', 'reduction'],
      camera: {...PANEL_ONLY, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'How wide should a time bucket be?',
          'Fixes per species, 3° band and bucket · 3 to 14 days'
        )
      },
      stage: 'reduce'
    },
    {
      id: 'counts-or-shares',
      title: 'Counts follow the tags, shares follow the birds',
      headline: 'Counts follow the tags, shares follow the birds',
      textAlternative:
        'The chart panel drawn with counts of fixes, with a strip of fixes per week below the legend; the share version of the same panel is one click away.',
      body: 'Flip **Cell value** between *Count* and *Share*. A count is the number of fixes the tags delivered, **{{weeklyFixes}}** across the year (strip below), so it follows the tags. A share divides each column by that total, so it follows the birds. Compare **Species**: a smaller sample means fewer fixes and noisier shares.\n\n*State the denominator.*',
      evidence:
        'The denominator chart exposes **{{weeklyFixes}}** for the selected species before those counts become column shares.',
      caveat:
        '**{{weeklyFixes}}** counts fixes, not distinct birds; unequal reporting cadence can therefore shape both counts and shares.',
      optionsMode: 'fresh',
      options: {cellValue: 'count', showTracks: false},
      controls: ['cellValue', 'species'],
      readouts: ['fixesChart', 'weeklyFixes'],
      camera: {...PANEL_ONLY, transitionMs: 1200},
      furniture: {
        title: cartouche(
          'Do counts or shares tell the story?',
          "Fixes, or share of the bucket's fixes, by 3° band"
        )
      },
      stage: 'fold'
    },
    {
      id: 'the-slope-is-speed',
      title: "The stripe's slope is the migration's speed",
      headline: 'A steeper stripe means a faster migration',
      textAlternative:
        'The chart panel for the marsh harrier with a line of median latitude climbing in spring and falling in autumn, and two pinned dates where it crosses the probe row.',
      body: 'Choose a **Species**. The line over the matrix is the median latitude of its fixes: the steeper it runs, the faster the population moves, as on a train timetable. Between the two reference latitudes the median needs **{{springWeeks}}** going north and **{{autumnWeeks}}** coming back. Move **Probe latitude** to read when it crosses any row.\n\n*Slope is speed.*',
      evidence:
        'The median trace takes **{{springWeeks}}** northbound and **{{autumnWeeks}}** southbound; at the probe its crossings are **{{crossings}}**.',
      caveat:
        'The crossing estimate **{{crossings}}** is interpolated between bucket centres, so bucket width limits its temporal precision.',
      optionsMode: 'fresh',
      options: {
        species: 'marsh',
        showTrace: true,
        showCrossings: true,
        showGroups: true,
        showTracks: false
      },
      controls: ['species', 'probeLatitude'],
      readouts: ['springWeeks', 'autumnWeeks', 'crossings', 'bandChart'],
      camera: {...PANEL_ONLY, transitionMs: 1200},
      furniture: {
        title: cartouche(
          'How fast does the population move?',
          'Median latitude of the fixes · crossing dates at the probe'
        )
      },
      stage: 'read'
    },
    {
      id: 'fastest-step',
      title: 'One fast step lights a whole cell',
      headline: 'A single fast step can light a whole cell',
      textAlternative:
        'The chart panel in orange classes: the fastest step in each cell, brightest along the migration stripe, with hatched cells where fewer than twenty fixes stand behind the maximum.',
      body: 'Set **Cell value** to *Fastest step*. Each cell shows the **maximum** the reduction kept for the slot, not a mean: the contributor keeps no sum. A step is the move between two consecutive fixes (**{{stepLength}}**), so one fast step is enough, and hatched cells hold too few fixes to trust. The fastest in the sample is **{{fastest}}**. Compare **Species**.\n\n*Read the stripe, not one cell.*',
      evidence:
        'The maximum retained step is **{{fastest}}**; **{{thinCells}}** identifies the cells whose maxima rest on sparse support.',
      caveat:
        'One displacement can set a maximum, and **{{stepLength}}** is the sampling interval—not a continuous speed measurement.',
      optionsMode: 'fresh',
      options: {cellValue: 'speed', species: 'all', showTracks: false},
      controls: ['cellValue', 'species'],
      readouts: ['fastest', 'thinCells', 'profileChart'],
      camera: {...PANEL_ONLY, transitionMs: 1200},
      furniture: {
        title: cartouche(
          'How fast is the fastest step?',
          'Maximum km/h of a step ending in the cell · thin cells hatched'
        )
      },
      stage: 'classes'
    },
    {
      id: 'play-the-year',
      title: 'Play the year: the column fills as birds move',
      headline: 'The column fills as the birds move',
      textAlternative:
        'The map and the panel together: a cursor sweeps the chart and the bucket under it is outlined, while that bucket’s GPS fixes appear as coloured dots over Europe and Africa.',
      body: 'Press **Play**: the cursor sweeps the year, the bucket under it is outlined, and its fixes appear on the map as dots. Drag **Day of the year** to jump and **Probe latitude** to move the row. The limits: **{{tracks}}**, tagged in the Low Countries; years are folded; a fix is not a bird, and the reduction counts fixes, not distinct birds.\n\nNext: [Which way do the harriers and spoonbills fly?](#/story/migration-flyways).',
      evidence:
        'The selected column is **{{cursor}}**; the horizontal probe reports **{{probe}}** from the same reduced cube.',
      caveat:
        'The animated dots inherit the limits of **{{tracks}}**: folded years, uneven fixes and no claim of population representativeness.',
      optionsMode: 'fresh',
      options: {play: true, day: 0, showFixes: true, showTrace: true, trackOpacity: 0.15},
      controls: ['play', 'day', 'probeLatitude'],
      readouts: ['cursor', 'profileChart', 'probe'],
      camera: {...MAP_AND_PANEL, transitionMs: 1400},
      furniture: {
        title: cartouche(
          'What does one column hold?',
          "Share of the bucket's fixes by 3° band · dots are the bucket's fixes"
        ),
        clock: {
          option: 'day',
          time: {origin: '2024-01-01T00:00:00Z', unit: 'days'},
          show: 'date',
          zones: ['UTC']
        }
      },
      stage: 'read'
    }
  ]
});
