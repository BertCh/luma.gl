// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {CHICAGO_VIEW} from './b1-nature-data';
import type {StreetDensityOptions} from './street-density.compute';

const RAMP_OPTIONS = [
  {value: 'inferno', label: 'Inferno'},
  {value: 'magma', label: 'Magma'},
  {value: 'viridis', label: 'Viridis'},
  {value: 'cividis', label: 'Cividis (color-blind optimised)'}
];

export default defineScene<StreetDensityOptions>({
  id: 'street-density',
  title: 'How much street does each part of Chicago have?',
  chapter: 'points',
  order: 5,
  summary:
    'Street length per grid cell on a camera-following grid, and street length clipped into community areas or census tracts, weighted by speed limit or one-way status and divided by area or residents.',
  contributors: ['GPULineDensity', 'GPULineLengthPerPolygon'],
  datasets: [
    {id: 'chicago-roads', role: 'OpenStreetMap drivable streets'},
    {id: 'chicago-community-areas', role: 'polygons'},
    {id: 'chicago-tracts', role: 'polygons with population'}
  ],
  initialView: {...CHICAGO_VIEW},

  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'View',
      group: 'View',
      apply: 'compile',
      default: 'density',
      help: 'Street length per grid cell (GPULineDensity), or street length clipped into polygons (GPULineLengthPerPolygon). The first use of each combination compiles its graph; later switches reuse it.',
      options: [
        {value: 'density', label: 'Density on a grid'},
        {value: 'polygons', label: 'Length per polygon'}
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
      kind: 'select',
      id: 'system',
      label: 'Coordinates',
      group: 'Streets',
      apply: 'compile',
      default: 'planar',
      help: 'Planar: positions are local meters and lengths are Euclidean. Spherical: positions are longitude and latitude, segments are clipped straight in degrees and each piece is measured as a great-circle distance.',
      options: [
        {value: 'planar', label: 'Planar meters'},
        {value: 'spherical', label: 'Spherical (longitude / latitude)'}
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
      disabledWhen: state => state.view !== 'density',
      help: 'Density divides length by cell area, so it does not change with cell size. Length is the raw meters of street in the cell and grows with the cell size.',
      options: [
        {value: 'density', label: 'Density (km of street per km²)'},
        {value: 'length', label: 'Length (meters of street per cell)'}
      ]
    },
    {
      kind: 'select',
      id: 'polygonSet',
      label: 'Polygons',
      group: 'Polygons',
      apply: 'compile',
      default: 'areas',
      disabledWhen: state => state.view !== 'polygons',
      help: 'The 77 community areas or the 791 census tracts of the city.',
      options: [
        {value: 'areas', label: 'Community areas (77)'},
        {value: 'tracts', label: 'Census tracts (791)'}
      ]
    },
    {
      kind: 'select',
      id: 'metric',
      label: 'Map value',
      group: 'Polygons',
      apply: 'param',
      default: 'length',
      disabledWhen: state => state.view !== 'polygons',
      help: 'Every metric is written for every polygon by the same graph; this only picks which output buffer the layer reads.',
      options: [
        {value: 'length', label: 'Street length (km)'},
        {value: 'perArea', label: 'Street length per area (km per km²)'},
        {value: 'perResident', label: 'Street length per resident (m)'},
        {value: 'meanAttribute', label: 'Length-weighted mean of the attribute'},
        {value: 'segments', label: 'Number of street segments'}
      ]
    },
    {
      kind: 'select',
      id: 'weight',
      label: 'Attribute for the weighted mean',
      group: 'Polygons',
      apply: 'param',
      default: 'speed',
      disabledWhen: state => state.view !== 'polygons',
      help: "pathWeights multiplies each street's clipped length by a per-street value: speed limit in km/h (mean speed limit), 1 for a one-way street (share of one-way streets) or 1 for a major road (share of major roads). Dividing the weighted length by the length gives the mean. Rewriting the weights is a buffer write.",
      options: [
        {value: 'speed', label: 'Speed limit (km/h)'},
        {value: 'oneway', label: 'One-way street (share)'},
        {value: 'major', label: 'Major road (share)'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'inferno',
      help: 'Perceptually uniform ramps; cividis is readable with common color-vision deficiencies.',
      options: RAMP_OPTIONS
    },
    {
      kind: 'toggle',
      id: 'showStreets',
      label: 'Street lines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws the streets that enter the analysis under the field.'
    },
    {
      kind: 'toggle',
      id: 'showOutlines',
      label: 'Polygon outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      disabledWhen: state => state.view !== 'polygons',
      help: 'Boundaries of the polygons.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time planar vs spherical',
      group: 'Compare',
      help: 'Runs the density graph in both coordinate systems outside the frame and reports the GPU time of each.'
    }
  ],

  readouts: [
    {id: 'segments', label: 'Streets analysed'},
    {id: 'density', label: 'Grid'},
    {id: 'grid', label: 'Cell size and extent'},
    {id: 'coverage', label: 'Grid covers'},
    {id: 'lengthInGrid', label: 'Street length inside the grid'},
    {id: 'densityPeak', label: 'Densest cells'},
    {
      id: 'records',
      label: 'Clipped pieces',
      help: 'Segment-cell pieces produced against the compile-time capacity.'
    },
    {id: 'overflow', label: 'Piece overflow'},
    {id: 'polygons', label: 'Polygons'},
    {id: 'inside', label: 'Street length inside polygons'},
    {id: 'conservation', label: 'Conservation check'},
    {id: 'polygonMaximum', label: 'Longest street total'},
    {id: 'polygonOverflow', label: 'Polygon overflow'},
    {id: 'hovered', label: 'Hovered polygon'},
    {id: 'timing', label: 'Density graph timing'}
  ],

  legends: state => {
    if (state.view === 'density') {
      return [
        {
          kind: 'ramp' as const,
          id: 'cells',
          title: state.cellValue === 'length' ? 'Street length per cell' : 'Street density',
          ramp: state.ramp,
          extent: 'gpu' as const,
          sqrtScale: true,
          unit: state.cellValue === 'length' ? 'm per cell' : 'km per km²',
          format: (value: number) =>
            state.cellValue === 'length' ? value.toFixed(0) : (value * 1000).toFixed(1)
        }
      ];
    }
    const titles = {
      length: 'Street length in the polygon',
      perArea: 'Street length per area',
      perResident: 'Street length per resident',
      segments: 'Street segments',
      meanAttribute:
        state.weight === 'speed'
          ? 'Mean speed limit'
          : state.weight === 'oneway'
            ? 'Share of one-way streets'
            : 'Share of major roads'
    };
    return [
      {
        kind: 'ramp' as const,
        id: 'metric',
        title: titles[state.metric],
        ramp: state.ramp,
        extent: 'gpu' as const,
        unit: state.metric === 'length' ? 'km' : undefined,
        format: (value: number) => {
          if (state.metric === 'length') return (value / 1000).toFixed(0);
          if (state.metric === 'meanAttribute')
            return state.weight === 'speed'
              ? `${value.toFixed(0)} km/h`
              : `${(value * 100).toFixed(0)}%`;
          return value < 10 ? value.toFixed(1) : value.toFixed(0);
        }
      }
    ];
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
  parameters: gridParameters.importToGraph(density),
  output: {lengths, densities, overflow, totalRecords}
}));
const compiledDensity = density.compile();   // once

// pan, zoom and cell size are one four-number write
gridParameters.write(getGPULineDensityParameterValues({
  minX, minY, cellWidth: ${state.cellSize}${state.system === 'spherical' ? ' / metersPerDegreeLongitude' : ''}, cellHeight: ${state.cellSize}${state.system === 'spherical' ? ' / metersPerDegreeLatitude' : ''}
}));
compiledDensity.encode(commandEncoder, {parameters: undefined});

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
    what: '`GPULineDensity` clips every street segment to the cells of a grid (Liang-Barsky, then a grid walk) and sums the clipped lengths per cell, optionally dividing by cell area. `GPULineLengthPerPolygon` clips the same streets into polygons, cutting each segment at every polygon edge, and returns the street length, an optional attribute-weighted length and a segment count per polygon. Parity target: QGIS "Line density" and "Sum line lengths", and PostGIS `ST_Length(ST_Intersection(...))`.',
    why: 'Street density is a basic descriptor of urban form: walkability, connectivity, infrastructure cost per resident, and the grid that observation counts or crash counts should be normalised by. Doing it on the GPU means the grid follows the camera at any zoom.',
    howToRead:
      'Bright cells hold more street per area. In the polygon view the legend gives the range up to the 98th percentile polygon. The "conservation check" compares the clipped length with the total length of the streets: a partition of the city should keep nearly all of it.'
  },

  create: async ctx => (await import('./street-density.compute')).createStreetDensity(ctx),

  story: [
    {
      id: 'the-question',
      controls: ['cellSize', 'cellValue'],
      readouts: ['densityPeak', 'grid'],
      title: "Where is Chicago's street grid densest?",
      body: "Chicago's grid is famously regular, but density still varies: parks, rail yards, industrial land and the lakefront have little street, while the near-north neighbourhoods have a lot. **`GPULineDensity`** clips every street to a grid laid over the screen and sums the length per cell; the colour shows kilometres of street per square kilometre.\n\nBright cells follow the dense, short-block neighbourhoods; dark gaps are O'Hare, rail corridors, the Stockyards and the parks. The grid is 192 × 120 cells of 350 m, which spans the whole city, and follows the camera, so panning never recompiles anything. Change **Cell size** below to see how the picture sharpens.",
      options: {
        view: 'density',
        roadClass: 'all',
        cellSize: 350,
        cellValue: 'density',
        system: 'planar'
      }
    },
    {
      id: 'cell-size',
      controls: ['cellSize', 'cellValue'],
      readouts: ['coverage'],
      title: 'Cell size: density is stable, length is not',
      body: 'Zoom into the Loop and shrink **Cell size** to 60 m: individual blocks appear. The grid always has 192 × 120 cells, so a smaller cell covers a smaller area; the **Grid covers** readout says how much of the view it spans.\n\nNow switch **Cell value** to *Length*: the same street now reads different values for different cell sizes, because a bigger cell simply contains more street. That is why densities (length divided by area) are what you compare across cell sizes.',
      options: {cellSize: 60},
      camera: {longitude: -87.64, latitude: 41.88, zoom: 12.9, transitionMs: 1800},
      highlight: {readout: 'coverage'}
    },
    {
      id: 'street-classes',
      controls: ['roadClass'],
      readouts: ['segments'],
      title: 'Arterials against neighbourhood streets',
      body: 'Choose a **Street class** below: motorways to secondary roads are the arterial skeleton, with long segments spaced about a mile apart; tertiary, residential and service streets are the fine grid. The subset is a different vertex buffer, so it is a compile-time choice; the first use compiles its graph, later switches reuse it.\n\nHere only the arterials are analysed. Densities are lower and more regular, and the lakefront and the expressways stand out.',
      options: {roadClass: 'arterial', cellSize: 350},
      camera: {longitude: -87.68, latitude: 41.835, zoom: 9.8, transitionMs: 1800}
    },
    {
      id: 'polygons',
      controls: ['view', 'polygonSet', 'metric'],
      readouts: ['conservation', 'inside', 'hovered'],
      title: 'Street length inside each community area',
      body: '**`GPULineLengthPerPolygon`** answers the polygon version of the question: how much street lies inside each of the 77 community areas? It cuts every street segment at every polygon edge and keeps the pieces whose midpoint is inside, then sums per polygon in a fixed order, so the result is reproducible.\n\nThe map shows kilometres of street per area. Big areas with many blocks lead; the **Conservation check** compares the total with the length of the network, which should be nearly 100% for a partition of the city. Hover a polygon for its figures.',
      options: {view: 'polygons', roadClass: 'all', polygonSet: 'areas', metric: 'length'},
      highlight: {readout: 'conservation'}
    },
    {
      id: 'normalise',
      controls: ['polygonSet', 'metric', 'weight'],
      readouts: ['hovered'],
      title: 'Normalise by area, people and attribute',
      body: "Length alone mostly maps size. Switch **Map value** to *Street length per area* (km per km²) or *Street length per resident*, which divides by the area and by the population the tracts add up to: sparsely populated industrial areas have a lot of street per resident. The census tracts give 791 polygons, ten times the detail.\n\nWith **pathWeights**, driven by **Attribute for the weighted mean**, each street's length is multiplied by its speed limit before summing; dividing by the length gives a *length-weighted mean speed limit*. The same graph yields the share of one-way streets or of major roads.",
      options: {polygonSet: 'tracts', metric: 'meanAttribute', weight: 'speed'},
      highlight: {readout: 'hovered'}
    },
    {
      id: 'spherical',
      controls: ['system', 'measure'],
      readouts: ['timing'],
      title: 'Planar or spherical lengths',
      body: 'By default positions are projected to local meters and lengths are Euclidean. With **Coordinates** set to *Spherical*, positions stay as longitude and latitude, segments are clipped straight in degrees and each piece is measured as a great-circle distance. At city scale the two agree to a fraction of a percent; over a continent the choice matters.\n\nPress **Time planar vs spherical** to compare the cost: the spherical walk does a little more arithmetic per piece.',
      options: {view: 'density', system: 'spherical', roadClass: 'all', cellSize: 350}
    },
    {
      id: 'limits',
      controls: ['view', 'roadClass', 'cellSize'],
      readouts: ['conservation'],
      title: 'Limits and things to try',
      body: 'Streets here are the OpenStreetMap drive network, so alleys, paths and private roads are missing, and two-way streets are counted once, not once per lane or direction. Lengths are summed per line, not unioned: overlapping lines count twice. The grid origin snaps to the cell size, so cell values jump slightly as you pan.\n\nTry: tracts per resident with the one-way share (**Polygons** and **Map value**); local streets only at 60 m in Lincoln Park (**Street class**, **Cell size**); planar against spherical with the **Conservation check** in the **Length per polygon** view.',
      options: {view: 'density', system: 'planar', roadClass: 'local', cellSize: 120}
    }
  ]
});
