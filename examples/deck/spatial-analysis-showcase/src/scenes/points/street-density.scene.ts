// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CHICAGO, CITY_FRAMES, labelsFor} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import {POINTS_CREDITS} from './b1-points-look';
import type {PolygonLegendData, StreetDensityOptions} from './street-density.compute';
import {
  CELL_SWEEP_SIZES,
  FIELD_VARIABLE,
  getStreetLegendEntries,
  makeDensityTable,
  MILE_VIEW,
  streetCartouche,
  type StreetGround
} from './street-density-look';

/** Names that orient the reader in every step (drawn above the data). */
const ORIENTATION = labelsFor(CHICAGO, ['lake-michigan', 'loop'], {loop: {minZoom: 10.2}});

/** The Loop, the frame of the scale step. */
const LOOP = CHICAGO.places.loop.lngLat;

/** The paper ground of the classed steps: the basemap names we draw ourselves are left out. */
const PAPER = ground('paperCity', {suppressNames: ['Chicago', 'Lake Michigan']});

export default defineScene<StreetDensityOptions>({
  id: 'street-density',
  title: 'How much street does each part of Chicago have?',
  chapter: 'points',
  order: 5,
  summary:
    'Chicago’s drivable streets as a grid of lines, as street length per grid cell on a camera-following grid, and clipped into community areas or census tracts, divided by area or residents. A story of figure-ground, the mile grid and the scale you count in.',
  contributors: ['GPULineDensity', 'GPULineLengthPerPolygon'],
  datasets: [
    {id: 'chicago-roads', role: 'OpenStreetMap drivable streets'},
    {id: 'chicago-community-areas', role: 'polygons'},
    {id: 'chicago-tracts', role: 'polygons with population'},
    {id: 'chicago-boundary', role: 'city limit and lake shore (no data outside the city)'}
  ],
  initialView: {...CITY_FRAMES.chicago},

  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'View',
      group: 'View',
      apply: 'compile',
      default: 'density',
      display: 'segmented',
      help: 'Street length per grid cell (GPULineDensity), or street length clipped into polygons (GPULineLengthPerPolygon). The first use of each combination compiles its graph; later switches reuse it.',
      options: [
        {value: 'density', label: 'Grid cells'},
        {value: 'polygons', label: 'Polygons'}
      ]
    },
    {
      kind: 'select',
      id: 'layers',
      label: 'Layers',
      group: 'View',
      apply: 'param',
      default: 'field',
      display: 'segmented',
      disabledWhen: state => state.view !== 'density',
      help: 'The street lines alone (the figure), the classed density surface alone, or the surface over the lines. Streets are the input, the surface is a summary of them.',
      options: [
        {value: 'streets', label: 'Streets'},
        {value: 'field', label: 'Field'},
        {value: 'both', label: 'Both'}
      ]
    },
    {
      kind: 'select',
      id: 'streetEmphasis',
      label: 'Street hierarchy',
      group: 'View',
      apply: 'param',
      default: 'all',
      display: 'segmented',
      disabledWhen: state => state.view !== 'density' || state.layers === 'field',
      help: 'Draw every street alike, or bring motorways to secondary roads forward and push the tertiary, residential and service streets back. A display split of the same buffer: nothing is recompiled.',
      options: [
        {value: 'all', label: 'All alike'},
        {value: 'arterials', label: 'Arterials forward'}
      ]
    },
    {
      kind: 'select',
      id: 'roadClass',
      label: 'Street class',
      group: 'Streets',
      apply: 'compile',
      default: 'all',
      help: 'Which streets enter the analysis. A different subset is a different vertex buffer, so it is a compile-time choice (cached after first use). Two-way streets are counted once.',
      options: [
        {value: 'all', label: 'All drivable streets'},
        {value: 'arterial', label: 'Motorways to secondary roads'},
        {value: 'local', label: 'Tertiary, residential and service streets'}
      ]
    },
    {
      kind: 'slider',
      id: 'cellSize',
      label: 'Cell size',
      group: 'Grid',
      apply: 'param',
      min: 50,
      max: 600,
      step: 10,
      default: 350,
      unit: 'm',
      marks: [{value: 350, label: 'default'}],
      autoSweep: {from: CELL_SWEEP_SIZES[0], to: 600, durationMs: 9000, ease: 'in-out'},
      describe: value =>
        `${value} m cells, ${((192 * value) / 1000).toFixed(1)} km across the grid`,
      disabledWhen: state => state.view !== 'density',
      help: 'Edge of a grid cell. The grid always has 192 x 120 cells, so a bigger cell covers more of the map. The grid origin snaps to the cell size, so panning only rewrites a four-number parameter buffer.'
    },
    {
      kind: 'select',
      id: 'cellValue',
      label: 'Cell value',
      group: 'Grid',
      apply: 'param',
      default: 'density',
      display: 'segmented',
      disabledWhen: state => state.view !== 'density',
      help: 'Density divides length by cell area, so it does not change with cell size. Length is the raw meters of street in the cell and grows with the cell size: the wrong comparison across scales.',
      options: [
        {value: 'density', label: 'Density'},
        {value: 'length', label: 'Length'}
      ]
    },
    {
      kind: 'select',
      id: 'polygonSet',
      label: 'Polygons',
      group: 'Polygons',
      apply: 'compile',
      default: 'areas',
      display: 'segmented',
      disabledWhen: state => state.view !== 'polygons',
      help: 'The 77 community areas or the 791 census tracts of the city.',
      options: [
        {value: 'areas', label: 'Community areas'},
        {value: 'tracts', label: 'Census tracts'}
      ]
    },
    {
      kind: 'select',
      id: 'metric',
      label: 'Map value',
      group: 'Polygons',
      apply: 'param',
      default: 'length',
      display: 'chips',
      disabledWhen: state => state.view !== 'polygons',
      help: 'Every metric is written for every polygon by the same graph; this only picks which output buffer the layer reads. Natural breaks are computed once per set and value, then frozen.',
      options: [
        {value: 'length', label: 'Length'},
        {value: 'perArea', label: 'Per area'},
        {value: 'perResident', label: 'Per resident'},
        {value: 'meanAttribute', label: 'Mean of the attribute'},
        {value: 'segments', label: 'Segments'}
      ]
    },
    {
      kind: 'select',
      id: 'weight',
      label: 'Attribute for the weighted mean',
      group: 'Polygons',
      apply: 'param',
      default: 'speed',
      expert: true,
      disabledWhen: state => state.view !== 'polygons' || state.metric !== 'meanAttribute',
      help: "pathWeights multiplies each street's clipped length by a per-street value: speed limit in km/h (mean speed limit), 1 for a one-way street (share of one-way streets) or 1 for a major road (share of major roads). Dividing the weighted length by the length gives the mean. Rewriting the weights is a buffer write.",
      options: [
        {value: 'speed', label: 'Speed limit (km/h)'},
        {value: 'oneway', label: 'One-way street (share)'},
        {value: 'major', label: 'Major road (share)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showOutlines',
      label: 'Polygon outlines',
      group: 'Polygons',
      apply: 'param',
      default: true,
      disabledWhen: state => state.view !== 'polygons',
      help: 'Hairline boundaries of the polygons.'
    },
    {
      kind: 'select',
      id: 'system',
      label: 'Coordinates',
      group: 'Compare',
      apply: 'compile',
      default: 'planar',
      expert: true,
      display: 'segmented',
      help: 'Planar: positions are local meters and lengths are Euclidean. Spherical: positions are longitude and latitude, segments are clipped straight in degrees and each piece is measured as a great-circle distance. At city scale the two differ by a fraction of a percent.',
      options: [
        {value: 'planar', label: 'Planar'},
        {value: 'spherical', label: 'Spherical'}
      ]
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time planar vs spherical',
      group: 'Compare',
      expert: true,
      help: 'Runs the density graph in both coordinate systems outside the frame and reports the GPU time of each.'
    }
  ],

  readouts: [
    {
      id: 'streetKm',
      label: 'Street network',
      emphasis: 'tile',
      help: 'Length of the drivable streets, each two-way street counted once.'
    },
    {
      id: 'gridShare',
      label: 'On the grid',
      emphasis: 'tile',
      help: 'Share of street length within 5 degrees of north-south or east-west, weighted by length.'
    },
    {id: 'bearingRose', label: 'Street bearings', kind: 'chart'},
    {
      id: 'densityPeak',
      label: 'Busiest cell',
      emphasis: 'tile',
      help: 'The densest city cell in the grid, read back once the field is computed.'
    },
    {id: 'cellEdge', label: 'Cell size'},
    {
      id: 'cellLength',
      label: 'Most street in one cell',
      help: 'The longest street total of any city cell: it grows with the cell, which is why length is the wrong unit across scales.'
    },
    {
      id: 'coverage',
      label: 'Grid covers',
      help: 'The grid is always 192 x 120 cells, so its reach follows the cell size.'
    },
    {id: 'densityHistogram', label: 'City cells by density', kind: 'chart'},
    {id: 'cellSizeCurve', label: 'Peak against cell size', kind: 'chart'},
    {id: 'mileGrid', label: 'East-west arterials', kind: 'chart'},
    {
      id: 'mileShare',
      label: 'On a mile lattice',
      help: 'Share of east-west primary and secondary road length within 100 m of the best-fitting mile lattice, against the share a random lattice would catch.'
    },
    {
      id: 'halfMileShare',
      label: 'On a half-mile lattice',
      help: 'The same for a lattice at half-mile spacing.'
    },
    {id: 'rankFlip', label: 'Rank by length and by area', kind: 'chart'},
    {id: 'topLength', label: 'Most street'},
    {id: 'topDensity', label: 'Most street per area'},
    {
      id: 'residentMedian',
      label: 'Median street per resident',
      help: 'Median over the tracts that are not set aside.'
    },
    {
      id: 'suppressed',
      label: 'Set aside',
      help: 'Tracts with too few residents for a stable rate are hatched, not ranked.'
    },
    {id: 'segments', label: 'Streets analysed', hood: true},
    {id: 'density', label: 'Grid', hood: true},
    {id: 'grid', label: 'Cell size and extent', hood: true},
    {
      id: 'records',
      label: 'Clipped pieces',
      hood: true,
      help: 'Segment-cell pieces produced against the compile-time capacity.'
    },
    {id: 'overflow', label: 'Piece overflow', hood: true},
    {id: 'sweepCost', label: 'Cell-size sweep', hood: true},
    {id: 'polygons', label: 'Polygons', hood: true},
    {id: 'inside', label: 'Street length inside polygons', hood: true},
    {id: 'conservation', label: 'Conservation check', hood: true},
    {id: 'polygonOverflow', label: 'Polygon overflow', hood: true},
    {id: 'timing', label: 'Density graph timing', hood: true}
  ],

  pipeline: [
    {
      id: 'clip',
      label: 'Clip',
      detail: 'Each street is cut where it meets a cell edge or a polygon boundary'
    },
    {
      id: 'walk',
      label: 'Walk',
      detail: 'Each piece walks the cells it crosses and emits a (cell, length) record'
    },
    {id: 'sort', label: 'Sort', detail: 'Records are sorted by cell'},
    {
      id: 'sum',
      label: 'Sum',
      detail: 'A fixed-order segmented sum, so the result is bitwise reproducible'
    },
    {id: 'divide', label: 'Divide', detail: 'Length over cell area: km of street per km²'}
  ],

  legends: (state, data) => {
    const groundTone = (data.ground as StreetGround | undefined) ?? 'light';
    if (state.view === 'polygons') {
      const polygon = data.polygon as PolygonLegendData | undefined;
      if (!polygon) return [];
      return [
        getClassTableLegend(polygon.table, {
          title: polygon.title,
          id: 'polygon-classes',
          counts: polygon.counts,
          interactive: true,
          note: polygon.note
        })
      ];
    }
    if (state.layers === 'streets') {
      return [
        {
          kind: 'line',
          title: 'Drivable streets',
          entries: getStreetLegendEntries(groundTone, state.streetEmphasis),
          note: 'Two-way streets are drawn once. The data stops at the city limit.'
        }
      ] satisfies LegendSpec[];
    }
    const table = makeDensityTable(groundTone, state.cellValue);
    const field = data.field as {counts?: number[]; lengthCounts?: number[]} | undefined;
    const marker = data.marker as number | null | undefined;
    const legend = getClassTableLegend(table, {
      title: state.cellValue === 'length' ? 'Metres of street per cell' : 'Street density',
      id: 'density-classes',
      counts: state.cellValue === 'length' ? field?.lengthCounts : field?.counts,
      interactive: true,
      note:
        state.cellValue === 'length'
          ? 'Grows with the cell size: the wrong comparison across scales. Fixed breaks.'
          : 'Fixed breaks at every cell size. Cells with no street are left clear.'
    });
    return [marker !== null && marker !== undefined ? {...legend, marker} : legend];
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPULineDensity, GPULineLengthPerPolygon, getGPULineDensityParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';

// every street polyline is one path: positions sorted by path plus pathCount + 1 offsets
const density = new GPUCommandGraph(device, {id: 'line-density'});
density.add(new GPULineDensity({
  positions, pathOffsets,                    // ${state.system === 'spherical' ? 'longitude / latitude degrees' : 'float32x2 meters'}
  columns: 192, rows: 120,
  coordinateSystem: '${state.system}',
  maximumRecords: 4 * vertexCount,           // compile-time; \`overflow\` says if it was too small
  parameters: gridParameters.importToGraph(density),
  output: {lengths, densities, overflow, totalRecords}
}));
const compiledDensity = density.compile();   // once

// pan, zoom and cell size are one four-number write; the grid follows the camera
gridParameters.write(getGPULineDensityParameterValues({
  minX, minY, cellWidth: ${state.cellSize}${state.system === 'spherical' ? ' / metersPerDegreeLongitude' : ''}, cellHeight: ${state.cellSize}${state.system === 'spherical' ? ' / metersPerDegreeLatitude' : ''}
}));
compiledDensity.encode(commandEncoder, {parameters: undefined});

// classed by fixed breaks (km per km², densities are m per m² x 1000), zero left clear
new SpatialAnalysisRasterLayer({gridSize: [192, 120], bounds: gridBounds.buffer, values: densities,
  valueScale: 1000, classBreaks: [5, 10, 15, 20], classColors, discardAtOrBelow: 0});

// street length clipped into polygons
const perPolygon = new GPUCommandGraph(device, {id: 'line-length'});
perPolygon.add(new GPULineLengthPerPolygon({
  positions, pathOffsets,
  pathWeights: ${state.weight},                // optional per-street multiplier
  polygons: {kind: 'polygons', positions: polygonPositions, featureOffsets, polygonOffsets, ringOffsets},
  coordinateSystem: '${state.system}',
  output: {lengths, weightedLengths, segmentCounts, overflow}
}));`,

  about: {
    what: '`GPULineDensity` clips every street segment to the cells of a grid (Liang-Barsky, then a grid walk), sorts the pieces by cell and sums their lengths in a fixed order, then divides by cell area. `GPULineLengthPerPolygon` clips the same streets into polygons, cutting each segment at every polygon edge, and returns the street length, an attribute-weighted length and a segment count per polygon. Parity target: QGIS "Line density" and "Sum line lengths", and PostGIS `ST_Length(ST_Intersection(...))`.',
    why: 'Street density is a basic descriptor of urban form: walkability, connectivity, infrastructure cost per resident, and the grid that observation counts should be normalised by. The unit it is counted in is part of the answer, so the story shows the same city in three units: lines, cells and polygons.',
    howToRead:
      'Darker cells hold more street per square kilometre, on fixed breaks that never change across steps. A cell with no street is left clear, and outside the city limit is no data, not zero. Lines are not unioned, so a dual carriageway counts twice; planar and spherical lengths differ by a fraction of a percent at city scale.'
  },

  // The streets are the figure: the night ground until a step needs paper for classed fills.
  basemap: ground('night'),
  furniture: {
    title: streetCartouche('A city of grid lines', 'OpenStreetMap drivable streets'),
    scaleBar: {units: 'metric'},
    credit: POINTS_CREDITS.roads
  },
  annotations: ORIENTATION,

  create: async ctx => (await import('./street-density.compute')).createStreetDensity(ctx),

  story: [
    {
      id: 'grid-lines',
      title: 'Chicago is a grid',
      headline: 'Chicago’s streets run north-south and east-west',
      textAlternative:
        'Dark map of Chicago drawn only in thin warm-white street lines on a regular grid, beside a rose diagram of street bearings with four dominant directions.',
      body: 'Chicago was surveyed on a mile grid and its streets still run that way: **{{gridShare}}** of the **{{streetKm}}** of street, each counted once, points within a few degrees of north-south or east-west. The rose gives every bearing the length it carries. Switch **Layers** to *Field* to watch the lines become a surface.',
      optionsMode: 'fresh',
      options: {
        view: 'density',
        layers: 'streets',
        streetEmphasis: 'all',
        roadClass: 'all',
        cellSize: 350,
        cellValue: 'density'
      },
      controls: ['layers'],
      readouts: ['gridShare', 'streetKm', 'bearingRose'],
      camera: {...CITY_FRAMES.chicago, transitionMs: 1400},
      basemap: ground('night'),
      furniture: {title: streetCartouche('A city of grid lines', 'OpenStreetMap drivable streets')},
      annotations: labelsFor(CHICAGO, ['ohare', 'midway'], {
        ohare: {tone: 'muted'},
        midway: {tone: 'muted'}
      })
    },
    {
      id: 'length-becomes-surface',
      title: 'Street length becomes a surface',
      headline: 'Street length becomes a quiet surface',
      textAlternative:
        'Paper map of Chicago washed in five brown classes of street density, with clear gaps where airports, parks and rail yards are.',
      body: 'The classed surface reads on paper, so the lines go to the back. **`GPULineDensity`** cuts each street into the cells it crosses, sums the pieces and divides by cell area: **{{densityPeak}}** in the busiest **{{cellEdge}}** cell. Cells with no street stay clear. Streets are the figure, the surface only a wash; put the lines back with **Layers**.',
      optionsMode: 'fresh',
      options: {
        view: 'density',
        layers: 'field',
        streetEmphasis: 'all',
        roadClass: 'all',
        cellSize: 350,
        cellValue: 'density'
      },
      controls: ['layers'],
      readouts: ['densityPeak', 'cellEdge', 'densityHistogram'],
      stage: 'sum',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1600},
      basemap: PAPER,
      furniture: {title: streetCartouche('Street length becomes a surface', FIELD_VARIABLE)},
      annotations: []
    },
    {
      id: 'mile-grid',
      title: 'Arterials run every mile',
      headline: 'Arterials run every mile, or every half mile',
      textAlternative:
        'Dark map of the West and South Sides with bright arterial roads on a regular lattice, a one square mile frame, and a profile of east-west arterials by position.',
      body: 'The Public Land Survey set section lines a mile apart, and Chicago’s address grid counts eight hundred numbers to the mile. East-west arterials follow it: a mile lattice catches **{{mileShare}}** of their length, a half-mile lattice **{{halfMileShare}}**. **Street hierarchy** pushes the local streets back; the frame is one square mile.',
      optionsMode: 'fresh',
      options: {
        view: 'density',
        layers: 'streets',
        streetEmphasis: 'arterials',
        roadClass: 'all',
        cellSize: 350,
        cellValue: 'density'
      },
      controls: ['streetEmphasis'],
      readouts: ['mileGrid', 'mileShare', 'halfMileShare'],
      camera: {...MILE_VIEW, transitionMs: 1600},
      basemap: ground('night'),
      furniture: {
        title: streetCartouche(
          'Arterials sit on the survey lattice',
          'Motorways to secondary roads, OpenStreetMap'
        )
      },
      annotations: labelsFor(CHICAGO, ['garfield-park', 'bridgeport', 'midway'], {
        'garfield-park': {tone: 'muted'},
        bridgeport: {tone: 'muted'},
        midway: {tone: 'muted'}
      })
    },
    {
      id: 'scale-of-the-cell',
      title: 'The scale of the cell',
      headline: 'Smaller cells, higher peaks',
      textAlternative:
        'Paper map of the Loop in small street-density cells, beside a curve showing the busiest cell falling as the cell size grows.',
      body: 'Press Play on **Cell size**: the busiest cell reads **{{densityPeak}}** and the curve shows peaks falling as cells grow while the typical cell barely moves. Switch **Cell value** to *Length* for the wrong comparison: street per cell just grows with the cell ({{cellLength}} here).\n\n*The unit you count in is part of the answer.*',
      optionsMode: 'fresh',
      options: {
        view: 'density',
        layers: 'field',
        streetEmphasis: 'all',
        roadClass: 'all',
        cellSize: 100,
        cellValue: 'density'
      },
      controls: ['cellSize', 'cellValue'],
      readouts: ['densityPeak', 'cellLength', 'coverage', 'cellSizeCurve'],
      stage: 'divide',
      camera: {longitude: LOOP[0], latitude: LOOP[1], zoom: 13, transitionMs: 1600},
      basemap: PAPER,
      furniture: {
        title: streetCartouche(
          'The unit you count in',
          'km of street per km², cell size by slider, OpenStreetMap'
        )
      },
      annotations: []
    },
    {
      id: 'by-neighbourhood',
      title: 'Divide by area',
      headline: 'Divide by area and the ranking flips',
      textAlternative:
        'Paper map of the 77 community areas shaded by street length, beside a slope chart showing how their ranks change when street length is divided by area.',
      body: '**`GPULineLengthPerPolygon`** sums street length per community area, and length mostly measures size: **{{topLength}}** leads. Choose **Map value** *Per area* and **{{topDensity}}** leads; the slope chart shows the ranks crossing. Raw totals on unequal areas map area, not intensity ([counts against rates](#/story/observations-by-tract)).',
      optionsMode: 'fresh',
      options: {
        view: 'polygons',
        polygonSet: 'areas',
        metric: 'length',
        roadClass: 'all',
        showOutlines: true
      },
      controls: ['metric'],
      readouts: ['rankFlip', 'topLength', 'topDensity'],
      stage: 'clip',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1600},
      basemap: PAPER,
      furniture: {
        title: streetCartouche(
          'Divide by area and the ranking flips',
          'Street length in community areas, natural breaks'
        )
      },
      annotations: []
    },
    {
      id: 'per-resident',
      title: 'Street per resident',
      headline: 'Street per resident, with small tracts set aside',
      textAlternative:
        'Paper map of the 791 census tracts shaded by metres of street per resident, with the few tracts that have too few residents hatched.',
      body: 'Across the tracts the median is **{{residentMedian}}**; {{suppressed}}, because small populations make unstable rates. Lines are not unioned, so dual carriageways count twice. Explore **Polygons**, **Map value**, **Street class** and **View**.\n\nThe same streets become a graph next: [routing](#/story/routing).',
      optionsMode: 'fresh',
      options: {
        view: 'polygons',
        polygonSet: 'tracts',
        metric: 'perResident',
        roadClass: 'all',
        showOutlines: true
      },
      controls: ['polygonSet', 'metric', 'roadClass', 'view'],
      readouts: ['residentMedian', 'suppressed'],
      stage: 'divide',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1600},
      basemap: PAPER,
      furniture: {
        title: streetCartouche(
          'Where the network is thin on people',
          'm of street per resident, census tracts, natural breaks'
        )
      },
      annotations: []
    }
  ]
});
