// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import type {TransitFrequencyOptions} from './transit-frequency.compute';
import {TRANSIT_MODE_LABELS, TRANSIT_MODES} from './transit-data';

const RANDSTAD_VIEW = {longitude: 4.75, latitude: 52.12, zoom: 8.8};

export default defineScene<TransitFrequencyOptions>({
  id: 'transit-frequency',
  title: 'Where does service concentrate?',
  chapter: 'networks',
  order: 21,
  summary:
    'Line density of every scheduled trip of the Randstad morning peak: length per grid cell turned into vehicles per hour, by mode, with per-line trips, distance and speed from group statistics.',
  contributors: ['GPULineDensity', 'GPUTrajectoryMetrics', 'GPUGroupStatistics'],
  datasets: [{id: 'poopdeck-gtfs-nl', role: 'scheduled trips (OVapi GTFS, 3 July 2026)'}],
  initialView: RANDSTAD_VIEW,

  options: [
    {
      kind: 'select',
      id: 'modeFilter',
      label: 'Mode',
      group: 'Density',
      apply: 'compile',
      default: 'all',
      help: 'Which trips feed the density. Each choice has its own GPULineDensity graph (its own positions and offsets), compiled the first time you pick it and kept; switching afterwards is instant.',
      options: [
        {value: 'all', label: 'All modes'},
        ...TRANSIT_MODES.map(mode => ({value: mode, label: TRANSIT_MODE_LABELS[mode]}))
      ]
    },
    {
      kind: 'slider',
      id: 'cellMeters',
      label: 'Cell size',
      group: 'Density',
      apply: 'param',
      min: 200,
      max: 800,
      step: 50,
      default: 250,
      unit: 'm',
      help: 'Edge of a square grid cell. The grid is 384 x 384 cells centred on the Randstad, so 200 m covers 77 km and 800 m 307 km. The cell size is a four-float parameter write: the graph is not recompiled, only re-run.'
    },
    {
      kind: 'slider',
      id: 'colorMax',
      label: 'Color scale maximum',
      group: 'Display',
      apply: 'param',
      min: 10,
      max: 300,
      step: 10,
      default: 100,
      unit: 'veh/h',
      help: 'Vehicles per hour at the top of the ramp. Cells above it are drawn in the last color. Lower it to see the quiet streets, raise it to separate the busiest corridors.'
    },
    {
      kind: 'select',
      id: 'scale',
      label: 'Color scale',
      group: 'Display',
      apply: 'param',
      default: 'sqrt',
      help: 'Square root spreads the many quiet cells over more of the ramp; linear keeps the ratio between cells honest.',
      options: [
        {value: 'sqrt', label: 'Square root'},
        {value: 'linear', label: 'Linear'}
      ]
    },
    {
      kind: 'slider',
      id: 'hideBelow',
      label: 'Hide cells below',
      group: 'Display',
      apply: 'param',
      min: 0,
      max: 20,
      step: 0.5,
      default: 0,
      unit: 'veh/h',
      help: 'Cells at or below this many vehicles per hour are transparent. 0 hides only empty cells.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'magma',
      help: 'Sequential ramp for vehicles per hour. All four are perceptually uniform.',
      options: [
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
      ]
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Opacity',
      group: 'Display',
      apply: 'param',
      min: 0.3,
      max: 1,
      step: 0.05,
      default: 0.85,
      help: 'Opacity of the grid over the basemap.'
    },
    {
      kind: 'toggle',
      id: 'showRoutes',
      label: 'Show routes faintly',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws the line of every trip as a thin gray line over the grid, to see which street a bright cell belongs to.'
    }
  ],

  readouts: [
    {
      id: 'concentration',
      label: 'How concentrated is service?',
      kind: 'chart',
      help: 'Occupied cells by the vehicles per hour that cross them. A long tail of busy cells hugs the corridors.'
    },
    {
      id: 'modeSpeed',
      label: 'Mean speed by mode',
      kind: 'chart',
      help: 'Total distance over total time of the trips of each mode in the window, from GPUTrajectoryMetrics and GPUGroupStatistics.'
    },
    {
      id: 'busiestLines',
      label: 'Busiest lines',
      kind: 'chart',
      help: 'Lines with most scheduled trips per hour in the window. Trips are grouped by (mode, line name), so a number shared by several cities counts together.'
    },
    {
      id: 'vehicleKilometers',
      label: 'Distance scheduled',
      help: 'Sum of every trip length inside the grid, in the window.'
    },
    {id: 'occupied', label: 'Cells with service'},
    {
      id: 'busiest',
      label: 'Busiest cell',
      help: 'Vehicles per hour through the busiest cell at the current cell size.'
    },
    {
      id: 'pieces',
      label: 'Line density pieces',
      help: 'Segment-cell pieces GPULineDensity produced; the capacity is four times the vertex count and an overflow would show here.'
    },
    {id: 'trips', label: 'Trips'},
    {
      id: 'modeTable',
      label: 'Trips, distance and speed by mode',
      layout: 'block',
      help: 'Per-mode totals added up from the per-line statistics (trips, kilometres inside the window, mean speed).'
    },
    {id: 'topLine', label: 'Busiest line'},
    {
      id: 'flex',
      label: 'On-demand Flex trips',
      help: 'The feed lists demand-responsive services as scheduled trips named Flex; they are left out of the busiest-lines chart.'
    },
    {
      id: 'cellValue',
      label: 'Clicked cell',
      help: 'Click the map to read the vehicles per hour of a cell.'
    }
  ],

  legends: state => [
    {
      kind: 'ramp' as const,
      id: 'frequency',
      title: 'Vehicles per hour through a cell',
      ramp: state.ramp,
      extent: [0, state.colorMax] as const,
      sqrtScale: state.scale === 'sqrt',
      unit: 'veh/h',
      format: (value: number) => value.toFixed(0)
    }
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPULineDensity, getGPULineDensityParameterValues, GPUTrajectoryMetrics} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUGroupStatistics} from '@luma.gl/experimental/gpu-dataframe';

// positions: planar meters of every vertex of every trip, pathOffsets: one path per trip
const graph = new GPUCommandGraph(device, {id: 'frequency'});
graph.add(new GPULineDensity({
  positions, pathOffsets,
  columns: 384, rows: 384,                 // compile time
  coordinateSystem: 'planar',
  parameters: gridParameters.importToGraph(graph),
  output: {lengths, overflow, totalRecords}
}));
const compiled = graph.compile();          // once per mode: ${state.modeFilter}

// per frame (or per change): the grid is a parameter write
gridParameters.write(getGPULineDensityParameterValues({
  minX: centerX - 384 * ${state.cellMeters} / 2, minY: centerY - 384 * ${state.cellMeters} / 2,
  cellWidth: ${state.cellMeters}, cellHeight: ${state.cellMeters}
}));
compiled.encode(commandEncoder, {parameters: undefined});
// vehicles per hour in a cell = lengths[cell] / ${state.cellMeters} / 2 hours

// per-trip metrics, then statistics per line
graph2.add(new GPUTrajectoryMetrics({positions, timestamps, trackOffsets, trackLengths, trackDurations, averageSpeeds}));
graph2.add(new GPUGroupStatistics({
  keys: lineIds, keyCount: lineCount,       // dense: row k is line k
  columns: [{values: trackLengths, statistics: ['sum'], output: {sumValues}},
            {values: averageSpeeds, statistics: ['mean', 'median'], output: {means, medians}}],
  output: {keys, counts: tripsPerLine, count, overflow}
}));`,

  about: {
    what: '`GPULineDensity` clips every segment of every trip to a grid (Liang-Barsky) and walks it through the cells it crosses, emitting one `(cell, length)` piece per cell; the pieces are summed per cell in a fixed order, so the result is reproducible. `GPUTrajectoryMetrics` measures each trip (length, duration, mean speed) and `GPUGroupStatistics` groups the trips by line and reports counts, sums, means and medians.',
    why: 'Service frequency is what a rider experiences as "I do not need a timetable". Summing scheduled length per cell turns thousands of trips into a map of where the service really is, which is the first screen of any question about transit coverage, equity of service or where a new line would add least.',
    howToRead:
      'A cell is coloured by the **vehicles per hour** that cross it: summed line length divided by the cell width and by the two hours of the window. A line that runs straight through a cell contributes its crossing length (one cell width), so the value is exact for straight-through traffic and a slight over-estimate for diagonals. Every vehicle in both directions counts, so a tram line with a 10-minute interval each way reads as 12 vehicles per hour.'
  },

  create: async ctx => (await import('./transit-frequency.compute')).createTransitFrequency(ctx),

  story: [
    {
      id: 'the-question',
      title: 'Where is the service?',
      body: 'Between 07:00 and 09:00 the Randstad schedules **8,448 trips** that cover about **112,000 vehicle-kilometres** of line. Where does all of that pile up?\n\n**`GPULineDensity`** answers it: each trip segment is clipped to a 384 x 384 grid and its length added to every cell it crosses. The map shows that length converted to **vehicles per hour through a cell**: dark cells are quiet, bright ones are corridors. The four big cities glow and the main rail and bus corridors draw bright threads between them.',
      camera: {...RANDSTAD_VIEW, transitionMs: 1200},
      options: {modeFilter: 'all', cellMeters: 250, colorMax: 100},
      controls: ['modeFilter', 'colorMax'],
      readouts: ['vehicleKilometers', 'busiest', 'concentration']
    },
    {
      id: 'cell-size',
      title: 'Resolution is a parameter',
      body: 'The grid is a four-float parameter: lower-left corner and cell width and height. Dragging **Cell size** below re-runs the same compiled graph on a coarser or finer grid, with no rebuild. At 200 m single streets separate; at 800 m the map becomes a regional heat map and the busiest cell tends to get *busier*, because one cell now holds several parallel lines.\n\nThat is the usual trade of any gridded measure: the number depends on the cell, so quote it with the cell size. Click the map to read the vehicles per hour of one cell.',
      options: {cellMeters: 250},
      camera: {longitude: 4.9, latitude: 52.37, zoom: 11, transitionMs: 1400},
      callout: {coordinate: [4.9003, 52.3791], text: 'Amsterdam Centraal'},
      controls: ['cellMeters', 'hideBelow', 'scale'],
      readouts: ['busiest', 'cellValue']
    },
    {
      id: 'rail',
      title: 'The railway skeleton',
      body: 'Set **Mode** below to *Train*: only the 454 train trips of the window remain, drawn from their own `GPULineDensity` graph (compiled the first time you choose it, which the Under the hood drawer counts). Intercity and sprinter trains share the same tracks, so the main corridors between the big cities are the brightest threads and the branch lines show up at a few trains an hour.\n\nLower **Color scale maximum** to 30 vehicles per hour to separate the main lines from the branches.',
      options: {modeFilter: 'rail', colorMax: 40, cellMeters: 250},
      camera: {...RANDSTAD_VIEW, transitionMs: 1400},
      callout: {coordinate: [5.1101, 52.0894], text: 'Utrecht Centraal'},
      controls: ['modeFilter', 'colorMax'],
      readouts: ['busiest', 'concentration']
    },
    {
      id: 'cities',
      title: 'Trams and metros own the city centres',
      body: 'Switch **Mode** to *Tram*: the 1,188 tram trips of the window form tight webs in the big cities. Switch to *Metro* for the Rotterdam and Amsterdam metro lines.\n\nIn a city the corridor is a street: a bright cell tells you where two or three lines share the rails, which is where the reliability problems of one line hit the most riders.',
      options: {modeFilter: 'tram', colorMax: 100, cellMeters: 200},
      camera: {longitude: 4.32, latitude: 52.08, zoom: 11.4, transitionMs: 1600},
      callout: {coordinate: [4.3247, 52.0808], text: 'Den Haag Centraal'},
      controls: ['modeFilter', 'colorMax', 'showRoutes'],
      readouts: ['busiest', 'cellValue']
    },
    {
      id: 'lines',
      title: 'Which lines carry the schedule?',
      body: '`GPUTrajectoryMetrics` measures every trip once: length, duration, mean speed. **`GPUGroupStatistics`** then groups the trips by line (mode and name) in dense mode, one row per line, and returns the trip count, the summed distance and the mean and median speed of each. The bar chart ranks the busiest lines; the table adds up trips, kilometres and speed per mode.\n\nTrams and buses average **19 and 27 km/h** against **66 km/h** for trains, dwell at stops included. The busiest lines are the trains (the Sprinter service and the Intercity), then a handful of tram lines. Where several cities use one line number the chart counts them together.',
      options: {modeFilter: 'all', colorMax: 100},
      controls: ['modeFilter'],
      readouts: ['busiestLines', 'modeSpeed', 'modeTable']
    },
    {
      id: 'limits',
      title: 'What to remember, and what to try',
      body: 'These are **scheduled** trips, not observed ones: a cancelled or late vehicle still counts. Trips are clipped to the Randstad box, so lines near the edges look quieter than they are, and each trip is simplified to 20 metres. About **9 percent** of the trips are on-demand *Flex* services that the feed lists as ordinary scheduled trips; they add vehicle-kilometres but not a fixed timetable.\n\nA cell value is the line length through the cell, so two lines on the same street add up, and a vehicle crossing a corner of a cell counts less than one crossing it. **Try it:** set **Cell size** to 200 and **Mode** to *Bus* to find the bus streets of the Hague; raise **Hide cells below** to 10 to keep only the real corridors; or set **Color scale** to *Linear* and see how few cells are really busy.',
      options: {modeFilter: 'bus', colorMax: 60, cellMeters: 200, hideBelow: 3},
      camera: {longitude: 4.4, latitude: 52.07, zoom: 10.2, transitionMs: 1400},
      controls: ['modeFilter', 'cellMeters', 'hideBelow', 'scale'],
      readouts: ['pieces', 'flex']
    }
  ]
});
