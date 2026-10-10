// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CHICAGO, labelsFor} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {getRegistryColors} from '../../cartography/hue-registry';
import {defineScene, type LegendSpec} from '../scene';
import {
  getGhostColor,
  getParameterInk,
  getSubjectColor,
  nextStoryLine,
  POINTS_CREDITS,
  pointsCartouche
} from './b1-points-look';
import type {LassoOptions} from './lasso-explorer.compute';

/** The 77 official community areas, in id order (area index = id - 1). */
const AREA_NAMES = [
  'Rogers Park',
  'West Ridge',
  'Uptown',
  'Lincoln Square',
  'North Center',
  'Lake View',
  'Lincoln Park',
  'Near North Side',
  'Edison Park',
  'Norwood Park',
  'Jefferson Park',
  'Forest Glen',
  'North Park',
  'Albany Park',
  'Portage Park',
  'Irving Park',
  'Dunning',
  'Montclare',
  'Belmont Cragin',
  'Hermosa',
  'Avondale',
  'Logan Square',
  'Humboldt Park',
  'West Town',
  'Austin',
  'West Garfield Park',
  'East Garfield Park',
  'Near West Side',
  'North Lawndale',
  'South Lawndale',
  'Lower West Side',
  'Loop',
  'Near South Side',
  'Armour Square',
  'Douglas',
  'Oakland',
  'Fuller Park',
  'Grand Boulevard',
  'Kenwood',
  'Washington Park',
  'Hyde Park',
  'Woodlawn',
  'South Shore',
  'Chatham',
  'Avalon Park',
  'South Chicago',
  'Burnside',
  'Calumet Heights',
  'Roseland',
  'Pullman',
  'South Deering',
  'East Side',
  'West Pullman',
  'Riverdale',
  'Hegewisch',
  'Garfield Ridge',
  'Archer Heights',
  'Brighton Park',
  'Mckinley Park',
  'Bridgeport',
  'New City',
  'West Elsdon',
  'Gage Park',
  'Clearing',
  'West Lawn',
  'Chicago Lawn',
  'West Englewood',
  'Englewood',
  'Greater Grand Crossing',
  'Ashburn',
  'Auburn Gresham',
  'Beverly',
  'Washington Heights',
  'Mount Greenwood',
  'Morgan Park',
  'Ohare',
  'Edgewater'
];

const UPTOWN = CHICAGO.places.uptown.lngLat;
const LOOP = CHICAGO.places.loop.lngLat;
const MONTROSE_POINT = CHICAGO.places['montrose-point'].lngLat;

/** The scene options the story steps keep resetting: the story is "fresh" at every step. */
const SHARED_SHAPE = {shape: 'polygon', path: 'direct', space: 'world'} as const;

/** A `[west, south, east, north]` frame around a place, for a step camera. */
const frameAround = (
  center: readonly [number, number],
  halfWidth: number,
  halfHeight: number
): [number, number, number, number] => [
  center[0] - halfWidth,
  center[1] - halfHeight,
  center[0] + halfWidth,
  center[1] + halfHeight
];

/** The lake sits east of Montrose Point; the label is moved there so it stays on the water. */
const LAKE_LABEL = {
  'lake-michigan': {coordinate: [MONTROSE_POINT[0] + 0.03, MONTROSE_POINT[1] + 0.006] as const}
};

export default defineScene<LassoOptions>({
  id: 'lasso-explorer',
  title: 'What gets logged inside the area I draw?',
  chapter: 'points',
  order: 4,
  summary:
    'Draw a lasso, circle or rectangle on 43,557 Chicago nature observations and read a linked chart of when and what was logged inside it, as a share of the shape against the whole city. The GPU reduces the selection in one pass.',
  contributors: [
    'GPURegionStatistics',
    'GPURegionMask',
    'GPUPickRegionMask',
    'GPURegionStatisticsReadback'
  ],
  datasets: [
    {id: 'chicago-nature', role: 'iNaturalist observations (2023)'},
    {id: 'chicago-community-areas', role: 'preset lassos, outlines and tooltips'}
  ],
  initialView: {longitude: UPTOWN[0], latitude: UPTOWN[1], zoom: 12.4, pitch: 0, bearing: 0},

  // Night ground: the selection is the one bright figure on a dimmed cloud.
  basemap: ground('night'),
  furniture: {
    title: pointsCartouche('When does Uptown log nature?', 'Records inside the shape, by hour'),
    scaleBar: {units: 'metric'},
    credit: POINTS_CREDITS.nature
  },
  annotations: labelsFor(CHICAGO, ['lake-michigan', 'loop'], {
    ...LAKE_LABEL,
    loop: {minZoom: 10.2}
  }),

  options: [
    {
      kind: 'select',
      id: 'area',
      label: 'Lasso a community area',
      group: 'Selection',
      apply: 'param',
      default: '2',
      help: 'Replaces the lasso with the outline of one of the 77 community areas (simplified to at most 254 vertices). Click an area on the map to do the same, or draw your own with the button below.',
      options: AREA_NAMES.map((name, index) => ({value: String(index), label: name}))
    },
    {
      kind: 'select',
      id: 'shape',
      label: 'Shape',
      group: 'Selection',
      apply: 'compile',
      default: 'polygon',
      display: 'segmented',
      help: 'Lasso polygon, circle or rectangle. The shape kind is compile-time (one graph each); the shape itself is rewritten every frame.',
      options: [
        {value: 'polygon', label: 'Lasso'},
        {value: 'radius', label: 'Circle'},
        {value: 'rectangle', label: 'Rectangle'}
      ]
    },
    {
      kind: 'slider',
      id: 'radius',
      label: 'Circle radius',
      group: 'Selection',
      apply: 'param',
      min: 100,
      max: 5000,
      step: 10,
      default: 1500,
      unit: 'm',
      disabledWhen: state => state.shape !== 'radius',
      describe: value => `${((Math.PI * value * value) / 1e6).toFixed(1)} km² inside the circle`,
      help: 'Written into the circle parameter buffer every frame. Drag the circle on the map to move it.'
    },
    {
      kind: 'toggle',
      id: 'equalArea',
      label: 'Same area as the lasso',
      group: 'Selection',
      apply: 'param',
      default: false,
      disabledWhen: state => state.shape !== 'radius',
      help: 'Sets the radius so the circle has the same area as the community area (r = sqrt(area / pi)), which makes "the same place, drawn two ways" a fair comparison.'
    },
    {
      kind: 'select',
      id: 'circleAt',
      label: 'Circle centre',
      group: 'Selection',
      apply: 'param',
      default: 'area',
      disabledWhen: state => state.shape !== 'radius',
      help: 'Where the circle sits: on the area you lassoed, on Montrose Point, or wherever you last dragged it.',
      options: [
        {value: 'area', label: 'The lassoed area'},
        {value: 'montrose-point', label: 'Montrose Point'},
        {value: 'custom', label: 'Where I dragged it'}
      ]
    },
    {
      kind: 'select',
      id: 'space',
      label: 'Shape space',
      group: 'Selection',
      apply: 'compile',
      default: 'world',
      display: 'segmented',
      disabledWhen: state => state.shape === 'radius' || state.path === 'pick',
      help: 'World: the shape is in meters on the ground and follows the map. Screen: the shape is in screen pixels through a per-frame view-projection transform, so it stays put while you pan, rotate or tilt the map.',
      options: [
        {value: 'world', label: 'World'},
        {value: 'screen', label: 'Screen'}
      ]
    },
    {
      kind: 'button',
      id: 'drawLasso',
      label: 'Draw a lasso',
      group: 'Selection',
      help: 'Then drag on the map. Use the Lasso shape.'
    },
    {
      kind: 'button',
      id: 'clear',
      label: 'Clear the selection',
      group: 'Selection',
      help: 'Empties the shape (a community area from the list above restores it).'
    },
    {
      kind: 'select',
      id: 'value',
      label: 'Statistic of',
      group: 'Chart',
      apply: 'compile',
      default: 'hour',
      help: 'The per-point value that is summed, averaged and histogrammed. Switching rewrites the values buffer; the histogram bins and domain are compile-time, so each choice is its own compiled graph, built the first time you use it.',
      options: [
        {value: 'hour', label: 'Hour of day (24 bins)'},
        {value: 'weekday', label: 'Day of week (7 bins, Sunday first)'},
        {value: 'month', label: 'Month of year (12 bins)'},
        {value: 'category', label: 'Group (10 bins)'},
        {value: 'researchGrade', label: 'Identification (2 bins)'}
      ]
    },
    {
      kind: 'select',
      id: 'normalise',
      label: 'Chart shows',
      group: 'Chart',
      apply: 'param',
      default: 'share',
      display: 'segmented',
      disabledWhen: state => state.domain === 'selection',
      help: 'Counts: records per bin (not comparable between places). Share: the percent of the selection in each bin, against the citywide percent. Difference: selection minus citywide in percentage points, orange above the city and purple below.',
      options: [
        {value: 'counts', label: 'Counts'},
        {value: 'share', label: 'Share'},
        {value: 'difference', label: 'Difference'}
      ]
    },
    {
      kind: 'toggle',
      id: 'clockView',
      label: 'Clock view',
      group: 'Chart',
      apply: 'param',
      default: false,
      disabledWhen: state =>
        state.value === 'category' ||
        state.value === 'researchGrade' ||
        state.domain === 'selection',
      help: 'Adds a rose chart that bends hours, weekdays or months into a circle, so 23:00 joins 00:00 and December joins January. The dashed ring is the citywide shape.'
    },
    {
      kind: 'select',
      id: 'compareWith',
      label: 'Ghost series',
      group: 'Chart',
      apply: 'param',
      default: 'previous',
      disabledWhen: state => state.normalise === 'counts' || state.domain === 'selection',
      help: 'A second place kept as a dashed series (cyclic quantities, Share). Previous: the community area you lassoed before this one, so changing the area never loses the comparison.',
      options: [
        {value: 'previous', label: 'The previous area'},
        {value: 'none', label: 'None'},
        ...AREA_NAMES.map((name, index) => ({value: String(index), label: name}))
      ]
    },
    {
      kind: 'select',
      id: 'domain',
      label: 'Histogram range',
      group: 'Chart',
      apply: 'compile',
      default: 'fixed',
      help: "Fixed: the full natural range of the quantity. Selection: the histogram stretches over the selected values' own minimum and maximum (computed on the GPU), so it has no citywide baseline.",
      options: [
        {value: 'fixed', label: 'Fixed (for example 0 to 24 h)'},
        {value: 'selection', label: 'Stretch to the selection'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showAreas',
      label: 'Community area outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Faint boundaries of the 77 community areas for orientation; hover one for its numbers.'
    },
    {
      kind: 'slider',
      id: 'ghostOpacity',
      label: 'Other observations',
      group: 'Display',
      apply: 'param',
      min: 0,
      max: 0.3,
      step: 0.01,
      default: 0.15,
      format: value => `${Math.round(value * 100)}% opaque`,
      help: 'How strongly every observation outside the shape is drawn. Dimmed, not hidden: the selection reads against the rest.'
    },
    {
      kind: 'toggle',
      id: 'showBounds',
      label: 'Bounding box and grid cells',
      group: 'Display',
      apply: 'param',
      default: false,
      disabledWhen: state => state.path === 'pick',
      help: 'Draws the shape’s bounding box, the first thing the grid index looks at, and (with the grid index on) the cells it gathers candidates from.'
    },
    {
      kind: 'select',
      id: 'path',
      label: 'Selection path',
      group: 'Plumbing',
      expert: true,
      apply: 'compile',
      default: 'direct',
      display: 'segmented',
      help: 'Direct: the statistics contributor takes the shape. Mask: GPURegionMask writes a 0/1 mask first. Pick: GPUPickRegionMask selects the points visible under the cursor in an index-picking texture. The selection is drawn the same whichever path made it.',
      options: [
        {value: 'direct', label: 'Direct'},
        {value: 'mask', label: 'Mask'},
        {value: 'pick', label: 'Pick'}
      ]
    },
    {
      kind: 'toggle',
      id: 'gridIndex',
      label: 'Grid index',
      group: 'Plumbing',
      expert: true,
      apply: 'compile',
      default: false,
      disabledWhen: state => state.path !== 'direct' || state.selectedIds === 'ids',
      help: 'A uniform grid built once over the points; the statistics gather only the cells the shape touches instead of testing all 43,557 observations. Identical results unless the candidate capacity is exceeded.'
    },
    {
      kind: 'select',
      id: 'candidateCapacity',
      label: 'Grid candidate capacity',
      group: 'Plumbing',
      expert: true,
      apply: 'compile',
      default: '0.5',
      disabledWhen: state => state.path !== 'direct' || !state.gridIndex,
      help: 'Most candidate rows gathered per frame, as a share of all points. A selection that touches more cells sets the "grid candidates truncated" flag.',
      options: [
        {value: '1', label: 'Exact (every point)'},
        {value: '0.5', label: '50% of points'},
        {value: '0.25', label: '25% of points'},
        {value: '0.1', label: '10% of points'},
        {value: '0.02', label: '2% (will truncate big shapes)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'withMask',
      label: 'Selection mask output',
      group: 'Plumbing',
      expert: true,
      apply: 'compile',
      default: true,
      disabledWhen: state => state.path !== 'direct' || state.selectedIds === 'ids',
      help: 'Writes a 0/1 mask used to highlight the selected points. Off: statistics only, which lets the grid index skip the per-point mask work. Not combined with the compact id list (together they exceed the eight storage buffers a compute stage may bind).'
    },
    {
      kind: 'select',
      id: 'selectedIds',
      label: 'Selected points drawn from',
      group: 'Plumbing',
      expert: true,
      apply: 'compile',
      default: 'mask',
      display: 'segmented',
      disabledWhen: state => state.path !== 'direct' || state.gridIndex,
      help: 'The mask, or a compact list of selected ids that an indirect draw consumes directly (drawInstanceCount): the draw size comes from the GPU, no readback.',
      options: [
        {value: 'mask', label: 'Mask'},
        {value: 'ids', label: 'Id list'}
      ]
    },
    {
      kind: 'select',
      id: 'idCapacity',
      label: 'Id list capacity',
      group: 'Plumbing',
      expert: true,
      apply: 'compile',
      default: 'medium',
      disabledWhen: state => state.path !== 'direct' || state.selectedIds !== 'ids',
      help: 'Rows the compact id list can hold. A bigger selection sets the "id list truncated" flag and only the first rows are drawn.',
      options: [
        {value: 'small', label: '1,000'},
        {value: 'medium', label: '10,000'},
        {value: 'all', label: '65,536'}
      ]
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time grid index on vs off',
      group: 'Plumbing',
      expert: true,
      help: 'Compiles the other variant and times both outside the frame (direct path), best of two rounds each.'
    }
  ],

  readouts: [
    {id: 'points', label: 'Observations', help: 'Every record on the map, uploaded once.'},
    {
      id: 'selected',
      label: 'Records inside',
      emphasis: 'tile',
      help: 'Observations inside the shape, from the GPU summary. It lags the frame by a few frames by design (a bounded readback ring).'
    },
    {
      id: 'cpuCheck',
      label: 'CPU recount',
      help: 'The same shape counted on the CPU to confirm the GPU result (world-space shapes; it agrees with the simplified outline).'
    },
    {
      id: 'selectionChart',
      label: 'Linked chart',
      kind: 'chart',
      help: 'The selection by the chosen quantity, as counts, as a share against the citywide shape (counted once from all records), or as the difference in points.'
    },
    {
      id: 'clockChart',
      label: 'The same, round a clock',
      kind: 'chart',
      help: 'Shares of the selection round a clock; the dashed ring is the citywide shape.'
    },
    {
      id: 'peak',
      label: 'Busiest bin',
      help: 'The bin with most records and its share of the selection. A linear mean of hours would be wrong across midnight, so the peak and the busiest window are shown instead.'
    },
    {
      id: 'peakWindowShare',
      label: 'Share in the busiest three bins',
      help: 'The most records in three consecutive hours, days or months (the window may wrap across midnight or the new year).'
    },
    {
      id: 'weekendShare',
      label: 'Weekend share',
      help: 'Share of the selection logged on a Saturday or Sunday, counted on the CPU from the columns (world-space shapes).'
    },
    {
      id: 'cityWeekendShare',
      label: 'Citywide weekend share',
      help: 'The same share over all records.'
    },
    {
      id: 'circleArea',
      label: 'Circle area',
      help: 'Pi times the radius squared.'
    },
    {
      id: 'polygonArea',
      label: 'Community area size',
      help: 'Planar area of the lassoed community area.'
    },
    {
      id: 'polygonRecords',
      label: 'Records in the community area',
      help: 'Records whose community-area column is this area (the portal join, not the simplified outline).'
    },
    {
      id: 'mean',
      label: 'Confirmed share',
      help: 'Mean of the 0/1 research-grade flag: the share whose identification the community confirmed.'
    },
    {id: 'range', label: 'Min to max', hood: true},
    {id: 'valueCount', label: 'Valid values', hood: true},
    {id: 'histogramLabel', label: 'Histogram of', hood: true},
    {id: 'outside', label: 'Outside the histogram', hood: true},
    {id: 'flags', label: 'Flags', hood: true},
    {id: 'maskContributor', label: 'Selection path', hood: true},
    {id: 'ids', label: 'Compact id list', hood: true},
    {id: 'region', label: 'Region path', hood: true},
    {id: 'gridOn', label: 'Grid index on', hood: true},
    {id: 'gridOff', label: 'Grid index off', hood: true},
    {id: 'speedup', label: 'Speedup', hood: true},
    {id: 'timingStatus', label: 'Timing', hood: true}
  ],

  pipeline: [
    {
      id: 'shape',
      label: 'Shape',
      detail: 'The lasso, circle or rectangle is written to a parameter buffer'
    },
    {id: 'test', label: 'Test', detail: 'Every point thread tests inside or outside'},
    {
      id: 'reduce',
      label: 'Reduce',
      detail: 'Inside rows add to atomics: count, sum, min, max, bins'
    },
    {
      id: 'readback',
      label: 'Readback',
      detail: 'About 32 words return through a bounded ring, a few frames late'
    },
    {id: 'draw', label: 'Draw', detail: 'The mask or the id list draws the selected points'}
  ],

  legends: (state, data) => {
    const look = (data.look ?? {}) as {ground?: 'light' | 'dark'; theme?: 'light' | 'dark'};
    const mapGround = look.ground ?? 'dark';
    const legends: LegendSpec[] = [
      {
        kind: 'categories',
        title: 'Observations',
        layout: 'list',
        entries: [
          {color: getGhostColor(mapGround, 170), label: 'All the rest', shape: 'dot'},
          {
            color: getSubjectColor(mapGround, 255),
            label: 'Inside the shape',
            shape: 'dot',
            count: typeof data.selected === 'number' ? data.selected : undefined
          },
          {color: getParameterInk(mapGround, 255), label: 'The shape you drew', shape: 'line'}
        ],
        note: 'Dimmed, not hidden: the rest stays as context.'
      }
    ];
    if (state.normalise === 'difference') {
      const colors = getRegistryColors('deviation', look.theme ?? 'dark', 5);
      legends.push({
        kind: 'categories',
        title: 'Chart: selection minus citywide',
        entries: [
          {color: colors[colors.length - 1], label: 'More than citywide'},
          {color: colors[0], label: 'Fewer than citywide'}
        ],
        note: 'Percentage points; the zero line is "same as the city".'
      });
    }
    return legends;
  },

  snippet: state => {
    const bins = {hour: 24, weekday: 7, month: 12, category: 10, researchGrade: 2}[state.value];
    const domain = {
      hour: '[0, 24]',
      weekday: '[0, 7]',
      month: '[0, 12]',
      category: '[0, 10]',
      researchGrade: '[0, 1]'
    }[state.value];
    const shapeCode =
      state.shape === 'radius'
        ? "{kind: 'radius', circle: circleParameters.importToGraph(graph)}      // [x, y, radius]"
        : state.shape === 'rectangle'
          ? `{kind: 'rectangle', bounds: rectangleParameters.importToGraph(graph)${state.space === 'screen' ? ',\n   screenTransform: screenTransform.importToGraph(graph)' : ''}}`
          : `{kind: 'polygon', vertices, vertexCount: vertexCount.importToGraph(graph)${state.space === 'screen' ? ',\n   screenTransform: screenTransform.importToGraph(graph)' : ''}}`;
    const histogram = `{binCount: ${bins}, domain: ${state.domain === 'selection' ? "'selection'" : domain}}`;
    const selection =
      state.path === 'direct'
        ? `selection: ${shapeCode},`
        : state.path === 'mask'
          ? `selection: {kind: 'mask', mask: regionMask},        // written by GPURegionMask`
          : `selection: {kind: 'mask', mask: pickMask},          // written by GPUPickRegionMask`;
    const pre =
      state.path === 'mask'
        ? `graph.add(new GPURegionMask({
  positions, outputMask: regionMask, overflow,
  region: ${state.shape === 'rectangle' ? "{kind: 'rectangle', bounds}" : "{kind: 'polygon', vertices, vertexCount}"}${state.space === 'screen' ? ',  // plus screenTransform' : ''}
}));
`
        : state.path === 'pick'
          ? `// index-picking pass draws every point's row into a target texture,
// then target.addRegionPass({region, result}) reduces the window around the cursor
graph.add(new GPUPickRegionMask({result, outputMask: pickMask, overflow}));
`
          : '';
    return `${pre}graph.add(new GPURegionStatistics({
  ${selection}${state.path === 'direct' ? '\n  positions,' : ''}
  values: ${state.value},                             // float32 per observation
  histogram: ${histogram},${state.withMask && state.path === 'direct' ? '\n  outputMask: mask,' : ''}${state.selectedIds === 'ids' && state.path === 'direct' ? '\n  output: {ids, count, overflow, requiredCount},\n  drawInstanceCount: drawCommands.getInstanceCountData(0),' : ''}${state.gridIndex && state.path === 'direct' ? `\n  spatialIndex: {kind: 'grid', index: gridIndex, candidateCapacity: ${Math.ceil(43557 * Number(state.candidateCapacity))}},` : ''}
  summary
}));
const compiled = graph.compile();            // once per combination of options
// every frame: rewrite the shape, encode, then read the summary through a ring
compiled.encode(commandEncoder, {parameters: undefined});
const ticket = readback.encodeRead(commandEncoder, summary);
readback.read(ticket).then(result => {
  // result.histogram is the selection; divide by its total and by the citywide
  // shape (counted once on the CPU) to compare shares instead of counts
  console.log(result.selectedCount, result.histogram);
});`;
  },

  about: {
    what: '`GPURegionStatistics` selects the observations inside a rectangle, circle, lasso polygon or picked region and reduces them in one pass: selected count, value sum, mean, minimum, maximum and a histogram, plus an optional 0/1 mask and compact id list. `GPURegionMask` and `GPUPickRegionMask` produce the selection mask separately (from a shape, or from what is visible under the cursor), and `GPURegionStatisticsReadback` reads the small summary back through a bounded ring.',
    why: 'Interactive exploration needs numbers for an arbitrary shape the moment it is drawn, without sending 43,557 points to the CPU. The summary is a few dozen words, so a lasso over the whole lakefront updates at frame rate. The selection is only half the answer: the linked chart must be read against the citywide shape, as a share, because a bigger place always has more records.',
    howToRead:
      'Amber dots are inside the shape; the grey rest is dimmed, not hidden. The chart under the map shows the selection by the chosen quantity: counts, a share of the selection against the citywide grey line, or the difference in percentage points (orange above the city, purple below). Hours, weekdays and months go round a clock. The CPU recount confirms the GPU count.'
  },

  create: async ctx => (await import('./lasso-explorer.compute')).createLassoExplorer(ctx),

  story: [
    {
      id: 'uptown',
      title: 'Draw around Uptown, read its day',
      headline: 'The map selects, the chart answers',
      textAlternative:
        'Dark map of Uptown, Chicago: amber dots inside a thin white lasso on the lakefront, grey dots outside, and a bar chart of the hours at which they were logged.',
      body: 'The lasso is the Uptown boundary. **`GPURegionStatistics`** tests every record against it and the chart answers: when were its {{selected}} records logged? That is *brushing and linking*: the map selects, the chart reads. Pick another place with **Lasso a community area**, or press **Draw a lasso** and drag. These are observers’ logs, not wildlife, so effort shapes every bar.',
      optionsMode: 'fresh',
      options: {
        ...SHARED_SHAPE,
        area: '2',
        value: 'hour',
        normalise: 'counts',
        clockView: false,
        compareWith: 'none'
      },
      controls: ['area', 'drawLasso'],
      readouts: ['selected', 'cpuCheck', 'selectionChart'],
      camera: {bounds: frameAround(UPTOWN, 0.034, 0.024), pitch: 0, bearing: 0, transitionMs: 1400},
      stage: 'test',
      furniture: {
        title: pointsCartouche('When does Uptown log nature?', 'Records inside the shape, by hour'),
        northArrow: 'auto'
      },
      annotations: labelsFor(CHICAGO, ['uptown', 'montrose-point'], {
        'montrose-point': {minZoom: 11.5}
      }),
      highlight: {readout: 'selectionChart'}
    },
    {
      id: 'uptown-vs-loop',
      title: 'Counts mislead; shares against the city do not',
      headline: 'Shares against the city show what counts hide',
      textAlternative:
        'Map of Uptown and the Loop with the Loop lassoed, and a line chart of its share of records by hour against the grey citywide line and a dashed Uptown line.',
      body: 'In *Counts* the Loop’s {{selected}} records sit far below Uptown’s, whatever the hour; in *Share* each bar is a share of the place’s own records, drawn against the citywide shape (grey) and Uptown (dashed). Flip **Chart shows** and see. Weekends: {{weekendShare}} here, {{cityWeekendShare}} citywide; busiest {{peak}}. Try **Lasso a community area**.\n\n*Compare shares, not counts.*',
      optionsMode: 'fresh',
      options: {
        ...SHARED_SHAPE,
        area: '31',
        value: 'hour',
        normalise: 'share',
        clockView: false,
        compareWith: 'previous'
      },
      controls: ['normalise', 'area'],
      readouts: ['selected', 'weekendShare', 'peak', 'selectionChart'],
      camera: {
        bounds: [
          Math.min(UPTOWN[0], LOOP[0]) - 0.03,
          Math.min(UPTOWN[1], LOOP[1]) - 0.03,
          Math.max(UPTOWN[0], LOOP[0]) + 0.03,
          Math.max(UPTOWN[1], LOOP[1]) + 0.03
        ],
        pitch: 0,
        bearing: 0,
        transitionMs: 1600
      },
      stage: 'reduce',
      furniture: {
        title: pointsCartouche('Does the Loop log like Uptown?', 'Share of the selection by hour'),
        northArrow: 'auto'
      },
      annotations: labelsFor(CHICAGO, ['uptown', 'lincoln-park'], {
        uptown: {minZoom: 9},
        'lincoln-park': {tone: 'muted'}
      }),
      highlight: {readout: 'selectionChart'}
    },
    {
      id: 'clock',
      title: 'Hours and months go round a clock',
      headline: 'Hours and months go round a clock',
      textAlternative:
        'Uptown lassoed on the map, with a line chart of its share of records by month and a round clock chart whose dashed ring is the citywide shape.',
      body: 'Hours and months repeat, so a straight axis breaks the wrap. **Statistic of** switches the quantity; **Clock view** draws it round with the citywide shape as a dashed ring, so December joins January and 23:00 joins midnight. The busiest three bins hold {{peakWindowShare}}, peaking {{peak}}. *Group* asks what is logged instead of when.',
      optionsMode: 'fresh',
      options: {
        ...SHARED_SHAPE,
        area: '2',
        value: 'month',
        normalise: 'share',
        clockView: true,
        compareWith: 'none'
      },
      controls: ['value', 'clockView'],
      readouts: ['peak', 'peakWindowShare', 'clockChart', 'selectionChart'],
      camera: {bounds: frameAround(UPTOWN, 0.034, 0.024), pitch: 0, bearing: 0, transitionMs: 1400},
      stage: 'reduce',
      furniture: {
        title: pointsCartouche('Which season does Uptown log?', 'Share of the selection by month'),
        northArrow: 'auto'
      },
      annotations: labelsFor(CHICAGO, ['uptown', 'montrose-point'], {
        'montrose-point': {minZoom: 11.5}
      }),
      highlight: {readout: 'clockChart'}
    },
    {
      id: 'which-boundary',
      title: 'Draw a circle instead and the answer moves',
      headline: 'One place, drawn twice, gives two answers',
      textAlternative:
        'Map of Montrose Point with a circle the same size as Uptown, a chart of its hourly share and a dashed Uptown line for comparison.',
      body: 'A circle of the same area as Uptown (**Same area as the lasso**: {{circleArea}} against {{polygonArea}}), centred on Montrose Point, catches {{selected}} records where the boundary holds {{polygonRecords}}. Slide **Circle radius** and watch the dashed boundary line part from the circle’s. Part of the circle is lake. The same place drawn two ways: the *modifiable areal unit problem*. [The full lesson](#/story/nature-density).',
      optionsMode: 'fresh',
      options: {
        area: '2',
        shape: 'radius',
        path: 'direct',
        space: 'world',
        equalArea: true,
        circleAt: 'montrose-point',
        value: 'hour',
        normalise: 'share',
        clockView: false,
        compareWith: '2'
      },
      controls: ['radius', 'equalArea'],
      readouts: ['selected', 'circleArea', 'polygonArea', 'selectionChart'],
      camera: {
        longitude: MONTROSE_POINT[0],
        latitude: MONTROSE_POINT[1],
        zoom: 12.6,
        pitch: 0,
        bearing: 0,
        transitionMs: 1600
      },
      stage: 'shape',
      furniture: {
        title: pointsCartouche(
          'Does the boundary change the answer?',
          'Share of the selection by hour'
        ),
        northArrow: 'auto'
      },
      annotations: labelsFor(CHICAGO, ['montrose-point', 'uptown'], {
        'montrose-point': {minZoom: 10},
        uptown: {minZoom: 10}
      }),
      highlight: {readout: 'circleArea'}
    },
    {
      id: 'screen-space',
      title: 'Select what you see, not what is there',
      headline: 'A tilted screen lasso reaches farther up the map',
      textAlternative:
        'Tilted, rotated map of Uptown with a lasso fixed to the screen; a dashed outline shows the same pixels on a flat map, shorter at the far edge.',
      body: 'With **Shape space** on *Screen* the lasso is tested through the view matrix, so it stays on the glass while you pan or tilt. Tilted, the far edge covers more ground: the dashed line is the same pixels on a flat map. {{selected}} records are inside; the CPU recount reads {{cpuCheck}}. Press **Draw a lasso** and draw one across the map.',
      optionsMode: 'fresh',
      options: {
        ...SHARED_SHAPE,
        area: '2',
        space: 'screen',
        value: 'hour',
        normalise: 'share',
        clockView: false,
        compareWith: 'none'
      },
      controls: ['space', 'drawLasso'],
      readouts: ['selected', 'cpuCheck', 'selectionChart'],
      // cartography-allow: pitch (a screen-space selection under perspective is the lesson: the tilt is what foreshortens it)
      camera: {
        longitude: UPTOWN[0],
        latitude: UPTOWN[1],
        zoom: 12.8,
        pitch: 50,
        bearing: 20,
        transitionMs: 1800
      },
      stage: 'shape',
      furniture: {
        title: pointsCartouche('What does a tilted lasso reach?', 'Share of the selection by hour'),
        northArrow: 'always'
      },
      annotations: labelsFor(CHICAGO, ['uptown', 'montrose-point'], {
        'montrose-point': {minZoom: 11.5}
      }),
      highlight: {readout: 'selected'}
    },
    {
      id: 'under-the-hood',
      title: 'Three ways to hand a shape to the GPU',
      headline: 'Three ways to hand a shape to the GPU',
      textAlternative:
        'Paper-coloured map of Uptown with the lasso, a dashed bounding box around it and a faint grid of the cells the grid index would gather.',
      body: `Hand the shape to the GPU three ways: directly, as a \`GPURegionMask\`, or as points picked under the cursor (**Selection path**). The dashed box is the lasso’s bounding box; with **Grid index** only the faint cells are tested, which pays at millions of points, not {{points}}. **Selected points drawn from** a compact id list skips the readback; **Time grid index on vs off** measures it.\n\n${nextStoryLine('lasso-explorer')}`,
      optionsMode: 'fresh',
      options: {
        ...SHARED_SHAPE,
        area: '2',
        value: 'hour',
        normalise: 'share',
        clockView: false,
        compareWith: 'none',
        gridIndex: true,
        showBounds: true
      },
      controls: ['path', 'gridIndex', 'selectedIds', 'measure'],
      readouts: ['gridOn', 'gridOff', 'speedup', 'flags'],
      camera: {
        longitude: UPTOWN[0],
        latitude: UPTOWN[1],
        zoom: 12.4,
        pitch: 0,
        bearing: 0,
        transitionMs: 1800
      },
      // The subject here is the outline, the box and the cells: ink on paper.
      basemap: ground('paperCity', {suppressNames: ['Chicago', 'Lake Michigan']}),
      stage: 'draw',
      furniture: {
        title: pointsCartouche('How does the GPU get the shape?', 'Bounding box and grid cells'),
        northArrow: 'auto'
      },
      annotations: labelsFor(CHICAGO, ['uptown'], {uptown: {minZoom: 10}}),
      highlight: {readout: 'speedup'}
    }
  ]
});
