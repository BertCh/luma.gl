// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {TransitFrequencyOptions} from './transit-frequency.compute';
import {TRANSIT_MODE_LABELS, TRANSIT_MODES} from './transit-data';
import {
  getServiceClassColors,
  RANDSTAD_DATA_FRAME,
  RANDSTAD_ORIENTATION,
  RANDSTAD_SCHEDULE_CREDIT,
  SERVICE_LABELS
} from './randstad-network-cartography';

const RANDSTAD_VIEW = {longitude: 4.75, latitude: 52.12, zoom: 8.8};
const CONTINUOUS_NIGHT_RAMP = 'magma' as const;

export default defineScene<TransitFrequencyOptions>({
  id: 'transit-frequency',
  title: 'Where does service concentrate?',
  chapter: 'networks',
  order: 7,
  summary:
    'Line density of every scheduled trip of the Randstad morning peak: length per grid cell turned into vehicles per hour, by mode, with per-line trips, distance and speed from group statistics.',
  contributors: ['GPULineDensity', 'GPUTrajectoryMetrics', 'GPUGroupStatistics'],
  datasets: [{id: 'poopdeck-gtfs-nl', role: 'scheduled trips (OVapi GTFS, 3 July 2026)'}],
  initialView: RANDSTAD_VIEW,
  basemap: ground('night', {labels: 'above', labelPreset: 'places-only'}),
  furniture: {
    title: {subtitle: 'Scheduled service · 07:00–09:00 CEST · 250 m cells'},
    scaleBar: {units: 'metric'},
    credit: RANDSTAD_SCHEDULE_CREDIT,
    caveat: 'Both directions and parallel routes add; this is not observed frequency.'
  },
  annotations: RANDSTAD_ORIENTATION,

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
      kind: 'select',
      id: 'representation',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      default: 'cells',
      display: 'segmented',
      help: 'Cells teach the grid calculation; routes restore the actual route geometry. Both keeps the computation and its source visible together.',
      options: [
        {value: 'cells', label: 'Cells'},
        {value: 'routes', label: 'Routes'},
        {value: 'both', label: 'Both'}
      ]
    },
    {
      kind: 'toggle',
      id: 'frequentOnly',
      label: 'Emphasise 15-minute service',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Fades the first three classes below 8 both-direction vehicles per hour (about every 15 minutes per direction). It is an analytical lens, not a service guarantee.'
    },
    {
      kind: 'select',
      id: 'serviceStyle',
      label: 'Service display',
      group: 'Display',
      apply: 'param',
      default: 'classes',
      display: 'segmented',
      help: 'Night glow uses one fixed magma square-root scale. Paper steps use the frozen six rider-facing service classes.',
      options: [
        {value: 'continuous', label: 'Continuous glow'},
        {value: 'classes', label: 'Six classes'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showRoutes',
      label: 'Show route context',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Thin route geometry gives the blocky grid an honest reference.'
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
      id: 'cellComparison',
      label: 'Busiest value by cell size',
      kind: 'chart',
      help: 'The same scheduled segments summarised at 200, 400 and 800 metres with fixed service classes.'
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

  legends: (state, data) => [
    ...(state.representation === 'cells' || state.representation === 'both'
      ? [
          ...(state.serviceStyle === 'continuous'
            ? [
                {
                  kind: 'ramp' as const,
                  title: 'Scheduled service through a cell',
                  ramp: CONTINUOUS_NIGHT_RAMP,
                  extent: [0, 60] as const,
                  unit: 'veh/h',
                  format: (value: number) => value.toFixed(0),
                  note: 'Fixed square-root scale, both directions summed.'
                }
              ]
            : [
                {
                  kind: 'categories' as const,
                  title: 'Scheduled service through a cell',
                  entries: SERVICE_LABELS.map((label, index) => ({
                    color: getServiceClassColors(data.ground !== 'light')[index],
                    label
                  })),
                  note: 'Both directions summed; headway is approximate.'
                }
              ])
        ]
      : []),
    ...(state.representation === 'routes' || state.representation === 'both'
      ? [
          {
            kind: 'line' as const,
            title: 'Route geometry',
            entries: [
              {color: [66, 72, 84, 180] as const, widthPixels: 1, label: 'scheduled trip segment'}
            ]
          }
        ]
      : [])
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
      id: 'glow',
      headline: 'Service as light',
      textAlternative:
        'A dark map shows the morning timetable glowing around the four Randstad cities.',
      title: 'Service as light',
      body: 'Every scheduled segment contributes its length to the cells it crosses, so the two-hour timetable becomes a luminous network. This first view is continuous on purpose: it makes concentration visible before we turn it into rider-facing classes. The map is a schedule, not a live vehicle feed.',
      camera: {...RANDSTAD_VIEW, transitionMs: 1200},
      optionsMode: 'fresh',
      basemap: ground('night', {labels: 'above', labelPreset: 'places-only'}),
      furniture: {
        title: {subtitle: 'Scheduled service · continuous intensity'},
        credit: RANDSTAD_SCHEDULE_CREDIT,
        scaleBar: {units: 'metric'}
      },
      options: {
        modeFilter: 'all',
        cellMeters: 250,
        serviceStyle: 'continuous',
        representation: 'cells'
      },
      controls: ['modeFilter'],
      readouts: ['vehicleKilometers', 'busiest', 'concentration']
    },
    {
      id: 'service-ladder',
      headline: 'Translate frequency into waiting',
      textAlternative:
        'Classed cells distinguish infrequent from frequent scheduled service around Randstad cities.',
      title: 'Translate frequency into waiting',
      body: 'The same density is now classified as vehicles per hour and an approximate per-direction headway. A cell at 4–8 veh/h reads as roughly every 15–30 minutes each way. These are useful classes, but parallel routes and both directions add: they do not promise a single route frequency.',
      optionsMode: 'fresh',
      basemap: ground('paperCity', {labels: 'above'}),
      options: {
        modeFilter: 'all',
        cellMeters: 250,
        serviceStyle: 'classes',
        representation: 'cells'
      },
      controls: ['representation'],
      readouts: ['concentration', 'busiest'],
      annotations: RANDSTAD_DATA_FRAME
    },
    {
      id: 'cell-size',
      headline: 'The number belongs to the cell',
      textAlternative: 'The same timetable is rendered at a selectable grid size around Amsterdam.',
      title: 'The number belongs to the cell',
      body: 'Cell width is a parameter write, not a rebuild. At 200 m streets separate; at 800 m parallel routes fall together and the busiest cell can grow. This is the modifiable areal unit problem in miniature: quote the cell size with every density value.',
      camera: {longitude: 4.9, latitude: 52.37, zoom: 11, transitionMs: 1400},
      callout: {coordinate: [4.9003, 52.3791], text: 'Amsterdam Centraal'},
      optionsMode: 'fresh',
      options: {cellMeters: 400, serviceStyle: 'classes', representation: 'both'},
      controls: ['cellMeters', 'representation'],
      readouts: ['busiest', 'cellComparison', 'cellValue']
    },
    {
      id: 'modes',
      headline: 'Rail, tram and bus differ',
      textAlternative: 'Tram service is shown around The Hague with route geometry as context.',
      title: 'Rail, tram and bus differ',
      body: 'Mode chooses a cached compile variant because each choice has different trip buffers. The first selection builds its graph; returning to it is instant. Tram webs read at city scale, while rail joins the cities. Mode hue identifies routes only—service magnitude stays ordered by the class ladder.',
      optionsMode: 'fresh',
      options: {
        modeFilter: 'tram',
        cellMeters: 200,
        serviceStyle: 'classes',
        representation: 'both'
      },
      camera: {longitude: 4.32, latitude: 52.08, zoom: 11.4, transitionMs: 1600},
      callout: {coordinate: [4.3247, 52.0808], text: 'Den Haag Centraal'},
      controls: ['modeFilter'],
      readouts: ['modeSpeed', 'busiestLines']
    },
    {
      id: 'frequent',
      headline: 'Keep service you can rely on',
      textAlternative: 'Cells below the approximate 15-minute service threshold are faded.',
      title: 'Keep service you can rely on',
      body: 'The 15-minute threshold begins at 8 both-direction veh/h because the per-direction inversion is 120 ÷ veh/h. It fades the first three classes while preserving the fixed legend. This analytical lens distinguishes frequent from occasional service without claiming that every route in a busy cell follows that interval.',
      optionsMode: 'fresh',
      options: {
        modeFilter: 'all',
        cellMeters: 400,
        serviceStyle: 'classes',
        representation: 'cells',
        frequentOnly: true
      },
      controls: ['frequentOnly'],
      readouts: ['concentration', 'busiestLines']
    },
    {
      id: 'representation',
      headline: 'Cells simplify lines',
      textAlternative:
        'Grid cells and faint route geometry are compared in the full Randstad extent.',
      title: 'Cells simplify lines',
      body: 'A grid makes comparison easy, but its values depend on clipping: diagonal crossings can be up to √2 longer than a straight crossing, parallel routes add, and the dashed frame marks the data extent. Routes restore the geometry. Flex trips are included in scheduled kilometres but do not imply fixed service.',
      optionsMode: 'fresh',
      options: {
        modeFilter: 'all',
        cellMeters: 400,
        serviceStyle: 'classes',
        representation: 'both'
      },
      camera: {longitude: 4.4, latitude: 52.07, zoom: 10.2, transitionMs: 1400},
      controls: ['representation', 'cellMeters'],
      readouts: ['pieces', 'flex']
    }
  ]
});
