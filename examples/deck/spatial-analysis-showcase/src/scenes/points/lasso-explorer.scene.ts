// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
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

export default defineScene<LassoOptions>({
  id: 'lasso-explorer',
  title: 'What gets logged inside the area I draw?',
  chapter: 'points',
  order: 4,
  summary:
    'Draw a lasso, circle or rectangle (or pick the points under the cursor) on 43,557 Chicago nature observations and read the count, an hour-of-day histogram and statistics of the selection from the GPU.',
  contributors: [
    'GPURegionStatistics',
    'GPURegionMask',
    'GPUPickRegionMask',
    'GPURegionStatisticsReadback'
  ],
  datasets: [
    {id: 'chicago-nature', role: 'iNaturalist observations (2023)'},
    {id: 'chicago-community-areas', role: 'preset lassos and context outlines'}
  ],
  initialView: {longitude: -87.655, latitude: 41.965, zoom: 12.4},

  options: [
    {
      kind: 'select',
      id: 'area',
      label: 'Lasso a community area',
      group: 'Selection',
      apply: 'param',
      default: '2',
      help: 'Replaces the lasso with the outline of one of the 77 community areas (simplified to at most 254 vertices). Draw your own with the button below.',
      options: AREA_NAMES.map((name, index) => ({value: String(index), label: name}))
    },
    {
      kind: 'select',
      id: 'shape',
      label: 'Shape',
      group: 'Selection',
      apply: 'compile',
      default: 'polygon',
      help: 'Lasso polygon, circle or rectangle. The shape kind is compile-time (one graph each); the shape itself is rewritten every frame.',
      options: [
        {value: 'polygon', label: 'Lasso polygon'},
        {value: 'radius', label: 'Circle (click or drag to move)'},
        {value: 'rectangle', label: 'Rectangle (drag on the map)'}
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
      step: 100,
      default: 1500,
      unit: 'm',
      disabledWhen: state => state.shape !== 'radius',
      help: 'Written into the circle parameter buffer every frame.'
    },
    {
      kind: 'select',
      id: 'path',
      label: 'Selection path',
      group: 'Selection',
      apply: 'compile',
      default: 'direct',
      help: 'Direct: the statistics contributor takes the shape. Mask: GPURegionMask writes a 0/1 mask first. Pick: GPUPickRegionMask selects the points visible under the cursor in an index-picking texture.',
      options: [
        {value: 'direct', label: 'Direct (statistics take the shape)'},
        {value: 'mask', label: 'Mask (GPURegionMask, then statistics)'},
        {value: 'pick', label: 'Pick (points under the cursor)'}
      ]
    },
    {
      kind: 'select',
      id: 'space',
      label: 'Shape space',
      group: 'Selection',
      apply: 'compile',
      default: 'world',
      disabledWhen: state => state.shape === 'radius' || state.path === 'pick',
      help: 'World: the shape is in meters on the ground and follows the map. Screen: the shape is in screen pixels through a per-frame view-projection transform, so it stays put while you pan, rotate or tilt the map.',
      options: [
        {value: 'world', label: 'World (meters on the map)'},
        {value: 'screen', label: 'Screen (pixels, stays fixed on screen)'}
      ]
    },
    {
      kind: 'button',
      id: 'drawLasso',
      label: 'Draw a lasso',
      group: 'Draw',
      help: 'Then drag on the map. Use the Lasso polygon shape.'
    },
    {
      kind: 'button',
      id: 'clear',
      label: 'Clear the selection',
      group: 'Draw',
      help: 'Empties the shape (a community area from the list above restores it).'
    },
    {
      kind: 'select',
      id: 'value',
      label: 'Statistic of',
      group: 'Statistics',
      apply: 'param',
      default: 'hour',
      help: 'The per-point value that is summed, averaged and histogrammed. Switching rewrites the values buffer; the histogram bins and domain are compile-time and change with it.',
      options: [
        {value: 'hour', label: 'Hour of day (24 bins)'},
        {value: 'weekday', label: 'Day of week (7 bins, Sunday first)'},
        {value: 'month', label: 'Month of year (12 bins)'},
        {value: 'researchGrade', label: 'Research grade (mean = confirmed share)'}
      ]
    },
    {
      kind: 'select',
      id: 'domain',
      label: 'Histogram range',
      group: 'Statistics',
      apply: 'compile',
      default: 'fixed',
      help: "Fixed: the full natural range of the quantity. Selection: the histogram stretches over the selected values' own minimum and maximum (computed on the GPU).",
      options: [
        {value: 'fixed', label: 'Fixed (for example 0 to 24 h)'},
        {value: 'selection', label: 'Stretch to the selection'}
      ]
    },
    {
      kind: 'toggle',
      id: 'gridIndex',
      label: 'Grid index',
      group: 'Performance',
      apply: 'compile',
      default: false,
      disabledWhen: state => state.path !== 'direct' || state.selectedIds === 'ids',
      help: 'A uniform grid built once over the points; the statistics gather only the cells the shape touches instead of testing all 43,557 observations. Identical results unless the candidate capacity is exceeded.'
    },
    {
      kind: 'select',
      id: 'candidateCapacity',
      label: 'Grid candidate capacity',
      group: 'Performance',
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
      group: 'Performance',
      apply: 'compile',
      default: true,
      disabledWhen: state => state.path !== 'direct' || state.selectedIds === 'ids',
      help: 'Writes a 0/1 mask used to highlight the selected points. Off: statistics only, which lets the grid index skip the per-point mask work. Not combined with the compact id list (together they exceed the eight storage buffers a compute stage may bind).'
    },
    {
      kind: 'select',
      id: 'selectedIds',
      label: 'Selected points drawn from',
      group: 'Performance',
      apply: 'compile',
      default: 'mask',
      disabledWhen: state => state.path !== 'direct' || state.gridIndex,
      help: 'The mask, or a compact list of selected ids that an indirect draw consumes directly (drawInstanceCount): the draw size comes from the GPU, no readback.',
      options: [
        {value: 'mask', label: 'Selection mask'},
        {value: 'ids', label: 'Compact id list (indirect draw)'}
      ]
    },
    {
      kind: 'select',
      id: 'idCapacity',
      label: 'Id list capacity',
      group: 'Performance',
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
      kind: 'toggle',
      id: 'showAreas',
      label: 'Community area outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Faint boundaries of the 77 community areas for orientation.'
    },
    {
      kind: 'toggle',
      id: 'showPoints',
      label: 'All observations',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Every observation as a faint dot under the selection.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time grid index on vs off',
      group: 'Compare',
      help: 'Compiles the other variant and times both outside the frame (direct path), best of two rounds each.'
    }
  ],

  readouts: [
    {id: 'points', label: 'Observations', format: 'integer'},
    {id: 'selected', label: 'Selected'},
    {
      id: 'cpuCheck',
      label: 'CPU recount',
      help: 'The same shape counted on the CPU to confirm the GPU result (world-space shapes).'
    },
    {id: 'histogramLabel', label: 'Histogram of'},
    {id: 'histogram', label: 'Histogram', help: 'One bar per bin, scaled to the tallest bin.'},
    {id: 'peak', label: 'Peak bin'},
    {id: 'mean', label: 'Mean'},
    {id: 'range', label: 'Min to max'},
    {id: 'valueCount', label: 'Valid values'},
    {id: 'outside', label: 'Outside the histogram'},
    {id: 'flags', label: 'Flags'},
    {id: 'maskContributor', label: 'Selection path'},
    {id: 'ids', label: 'Compact id list'},
    {id: 'region', label: 'Region path'},
    {id: 'gridOn', label: 'Grid index on'},
    {id: 'gridOff', label: 'Grid index off'},
    {id: 'speedup', label: 'Speedup'},
    {id: 'timingStatus', label: 'Timing'}
  ],

  legends: state => [
    {
      kind: 'categories',
      title: 'Observations and selection',
      entries: [
        {color: [100, 130, 185, 150], label: 'All observations'},
        {
          color: state.path === 'direct' ? [255, 190, 60, 255] : [255, 110, 200, 255],
          label:
            state.path === 'direct'
              ? 'Selected (statistics mask)'
              : state.path === 'mask'
                ? 'Selected (GPURegionMask)'
                : 'Selected (GPUPickRegionMask)'
        },
        {color: [0, 170, 160, 255], label: 'Selection outline'}
      ]
    }
  ],

  snippet: state => {
    const shapeCode =
      state.shape === 'radius'
        ? "{kind: 'radius', circle: circleParameters.importToGraph(graph)}      // [x, y, radius]"
        : state.shape === 'rectangle'
          ? `{kind: 'rectangle', bounds: rectangleParameters.importToGraph(graph)${state.space === 'screen' ? ',\n   screenTransform: screenTransform.importToGraph(graph)' : ''}}`
          : `{kind: 'polygon', vertices, vertexCount: vertexCount.importToGraph(graph)${state.space === 'screen' ? ',\n   screenTransform: screenTransform.importToGraph(graph)' : ''}}`;
    const histogram = `{binCount: ${state.value === 'hour' ? 24 : state.value === 'weekday' ? 7 : state.value === 'month' ? 12 : 2}, domain: ${state.domain === 'selection' ? "'selection'" : state.value === 'hour' ? '[0, 24]' : state.value === 'weekday' ? '[0, 7]' : state.value === 'month' ? '[0, 12]' : '[0, 1]'}}`;
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
  histogram: ${histogram},${state.withMask && state.path === 'direct' ? '\n  outputMask: mask,' : ''}${state.selectedIds === 'ids' && state.path === 'direct' ? '\n  output: {ids, count, overflow, totalCount},\n  drawInstanceCount: drawCommands.getInstanceCountData(0),' : ''}${state.gridIndex && state.path === 'direct' ? `\n  spatialIndex: {kind: 'grid', index: gridIndex, candidateCapacity: ${Math.ceil(43557 * Number(state.candidateCapacity))}},` : ''}
  summary
}));
const compiled = graph.compile();            // once per combination of options
// every frame: rewrite the shape, encode, then read the summary through a ring
compiled.encode(commandEncoder, {parameters: undefined});
const ticket = readback.encodeRead(commandEncoder, summary);
readback.read(ticket).then(result => console.log(result.selectedCount, result.histogram));`;
  },

  about: {
    what: '`GPURegionStatistics` selects the observations inside a rectangle, circle, lasso polygon or picked region and reduces them in one pass: selected count, value sum, mean, minimum, maximum and a histogram, plus an optional 0/1 mask and compact id list. `GPURegionMask` and `GPUPickRegionMask` produce the selection mask separately (from a shape, or from what is visible under the cursor), and `GPURegionStatisticsReadback` reads the small summary back through a bounded ring.',
    why: 'Interactive exploration needs numbers for an arbitrary shape the moment the user draws it, without sending 43,557 points to the CPU. The summary is a few dozen words, so a lasso over the whole lakefront updates at frame rate.',
    howToRead:
      'Amber points are inside the shape; the teal line is the shape. The histogram under Readouts has one bar per bin of the chosen quantity, scaled to the tallest bin: hours of the day run left to right from midnight. The CPU recount confirms the GPU count.'
  },

  create: async ctx => (await import('./lasso-explorer.compute')).createLassoExplorer(ctx),

  story: [
    {
      id: 'the-question',
      controls: ['area', 'drawLasso'],
      readouts: ['selected', 'cpuCheck', 'histogram'],
      title: 'When do people log nature in Uptown?',
      body: 'The lasso is the outline of the Uptown community area, home of Montrose Point. **`GPURegionStatistics`** tests every one of the 43,557 observations against it on the GPU and reduces the inside ones to a count, statistics and a histogram of the hour of day. Nothing is sent to the CPU except a summary of about 32 numbers.\n\nThe **Histogram** readout runs from midnight (left) to 23:00 (right). Uptown holds 4,887 observations, more than half of them birds, and the counts climb through the morning to a peak between 11:00 and 12:00, with an early bump at 08:00 from the birders who arrive for the dawn migrants. The CPU recount confirms the GPU count to the observation.\n\nBelow, **Lasso a community area** swaps the shape, or press **Draw a lasso** and drag on the map.',
      options: {area: '2', shape: 'polygon', path: 'direct', value: 'hour'},
      camera: {longitude: -87.655, latitude: 41.965, zoom: 12.4},
      highlight: {readout: 'histogram'}
    },
    {
      id: 'compare-areas',
      controls: ['area'],
      readouts: ['peak', 'histogram'],
      title: 'Same question, another neighbourhood',
      body: 'Pick **Lasso a community area** to swap the shape: here the Loop, with 1,401 observations. The shape is just 254 vertices in a parameter buffer, so swapping it recompiles nothing.\n\nCompare the **Peak bin** and the histogram with Uptown: downtown observations peak at lunchtime, 12:00 to 13:00, and only 23% fall on a weekend against 38% in Uptown. The daily rhythm of an office district is different from a lakefront park. Lincoln Park, the busiest area with 5,526 observations, is another good one to try.',
      options: {area: '31'},
      camera: {longitude: -87.63, latitude: 41.882, zoom: 12.4},
      highlight: {readout: 'peak'}
    },
    {
      id: 'other-quantities',
      controls: ['value', 'domain'],
      readouts: ['histogram', 'mean'],
      title: 'Histogram anything you can put in a buffer',
      body: "The histogram is over a per-point float32 **values** buffer. Switch **Statistic of** to *Day of week*, *Month of year* or *Research grade*: for a 0/1 flag the mean is the share of the selection whose identification the community confirmed (63% citywide). Bins and domain are compile-time, so each choice is its own compiled graph (built the first time you use it, then cached).\n\nSet **Histogram range** to *Stretch to the selection* and the bins span the selected values' own minimum and maximum, computed on the GPU. Month of year is the most telling for wildlife: pick Uptown and the spring migration shows as a tall bar for April and May.",
      options: {value: 'month', area: '2'},
      camera: {longitude: -87.655, latitude: 41.965, zoom: 12.4}
    },
    {
      id: 'shapes-and-paths',
      controls: ['shape', 'radius', 'path'],
      readouts: ['selected', 'cpuCheck'],
      title: 'Circles, rectangles and mask paths',
      body: 'A **circle** is three numbers, a **rectangle** four, a **lasso** up to 256 vertices; each shape kind is a separate compiled graph. Here a 1.5 km circle is placed on the map: drag to move it, and change **Circle radius**.\n\nThe **Selection path** switches how the selection reaches the statistics. *Mask* runs **`GPURegionMask`** first and the statistics read its 0/1 mask (the circle becomes a 64-sided polygon); *Pick* uses **`GPUPickRegionMask`** to select the points visible in a window around your click. The count must match the direct path for the same shape.',
      options: {shape: 'radius', radius: 1500, path: 'mask'}
    },
    {
      id: 'screen-space',
      controls: ['space', 'drawLasso'],
      readouts: ['selected'],
      title: 'Select what you see: screen-space shapes',
      body: 'With **Shape space** set to *Screen*, the lasso is stored in screen pixels and tested through a per-frame view-projection matrix. It stays where you drew it while you pan, rotate or tilt the map, so a tilted view selects what is *under the lasso on screen*, not what is under it on the ground.\n\nThe map tilts here; pan the map and watch the count change while the outline stays put. Switch back to *World* and the shape sticks to the ground.',
      options: {shape: 'polygon', path: 'direct', space: 'screen', area: '2'},
      camera: {longitude: -87.65, latitude: 41.95, zoom: 12.8, pitch: 55, bearing: 20}
    },
    {
      id: 'index-and-ids',
      controls: ['gridIndex', 'withMask', 'measure'],
      readouts: ['gridOn', 'gridOff', 'speedup'],
      title: 'Grid index and compact id lists',
      body: 'A **grid index** over the 43,557 points, built once, lets a small shape gather only the cells it touches instead of testing every observation. The result is bit-identical unless the candidate capacity is exceeded; press **Time grid index on vs off** to measure it (with so few points the gain is small, and it grows with the point count). Turn **Selection mask output** off and the statistics skip the per-point mask work too.\n\nAlternatively (instead of, not together with, the grid index) the selection can be drawn from a **compact id list**: the contributor writes the selected row ids and the instance count straight into an indirect draw record. The grid index and the id list exclude each other, so **Selected points drawn from** is greyed out while **Grid index** is on. The next step switches **Grid index** off and the id list on, and with a small **Id list capacity** the flags show "id list truncated" for big shapes.',
      options: {
        space: 'world',
        gridIndex: true,
        selectedIds: 'mask',
        area: '2'
      },
      camera: {longitude: -87.655, latitude: 41.965, zoom: 12.4, pitch: 0, bearing: 0}
    },
    {
      id: 'limits',
      controls: ['selectedIds', 'idCapacity', 'drawLasso'],
      readouts: ['ids', 'flags'],
      title: 'Limits and things to try',
      body: 'Counts are of *observations*, not of animals or of biodiversity: one flock photographed twice counts twice, and about one observation in six shares its exact coordinate with another, so a thin lasso along a path can swing the count. Places nobody visits do not appear at all. The lasso is limited to 256 vertices and an even-odd rule; picks see only what is visible under the cursor, so overlapping points hide each other. **Grid index** is now off and **Selected points drawn from** is the compact id list with an **Id list capacity** of 1,000, so a big shape will truncate.\n\nTry: press **Draw a lasso** and draw a shape around Montrose Point, then a large one over the whole North Side, and read the **Flags** readout; raise **Id list capacity** to 10,000 and the truncation flag clears for most shapes.',
      options: {gridIndex: false, selectedIds: 'ids', idCapacity: 'small', path: 'direct'}
    }
  ]
});
