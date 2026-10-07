// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import type {RasterJoinOptions} from './raster-join.compute';

const METRIC_TITLES: Record<RasterJoinOptions['metric'], string> = {
  count: 'Observations per area (raster join)',
  exact: 'Observations per area (exact join)',
  difference: 'Raster join minus exact join',
  boundaryShare: 'Share of an area’s points in boundary cells',
  researchGrade: 'Research-grade observations per area (raster join sum)'
};

export default defineScene<RasterJoinOptions>({
  id: 'raster-join',
  title: 'Raster join: count points without testing polygons',
  chapter: 'joins',
  order: 5,
  summary:
    'Rasterize the 77 Chicago community areas into a zone grid on the GPU and bin 43,557 nature observations into it in O(1) per point. Compare with the exact join, see where the error can be, and watch it shrink as the raster gets finer.',
  contributors: ['GPUPolygonRasterization', 'GPURasterJoin', 'GPUPointInPolygonJoin'],
  datasets: [
    {id: 'chicago-nature', role: 'points to join'},
    {id: 'chicago-community-areas', role: 'polygons'}
  ],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 10},

  options: [
    {
      kind: 'slider',
      id: 'resolutionLevel',
      label: 'Raster resolution',
      group: 'Raster',
      apply: 'compile',
      min: 0,
      max: 5,
      step: 1,
      default: 2,
      format: level => `${64 * 2 ** level} cells on the long side`,
      help: 'Doubles the grid per step, from 64 to 2,048 cells. The grid size is compile-time, so each step rebuilds the graph; cell size and error shrink together.'
    },
    {
      kind: 'select',
      id: 'extent',
      label: 'Raster placement',
      group: 'Raster',
      apply: 'param',
      default: 'city',
      help: 'The extent is four floats (origin and cell size) in a parameter buffer. Fixed covers the whole city; following the viewport re-fits the grid to the screen on every camera move with no rebuild, so cells shrink as you zoom in.',
      options: [
        {value: 'city', label: 'Whole city (fixed)'},
        {value: 'viewport', label: 'Follow the viewport'}
      ]
    },
    {
      kind: 'select',
      id: 'layer',
      label: 'Show',
      group: 'Display',
      apply: 'param',
      default: 'choropleth',
      help: 'The choropleth looks each cell up in the join result. The zone raster shows the raw cell-to-area table the join uses.',
      options: [
        {value: 'choropleth', label: 'Choropleth of the join'},
        {value: 'zones', label: 'Zone raster (cell to area)'}
      ]
    },
    {
      kind: 'select',
      id: 'metric',
      label: 'Metric',
      group: 'Display',
      apply: 'param',
      default: 'count',
      disabledWhen: state => state.layer !== 'choropleth',
      help: 'Which per-area number colours the map. Count and sum come straight from the join output buffers; the others are derived from them.',
      options: [
        {value: 'count', label: 'Observations per area (raster join)'},
        {value: 'exact', label: 'Observations per area (exact join)'},
        {
          value: 'difference',
          label: 'Raster minus exact',
          help: 'Diverging: blue is undercounted, red is overcounted.'
        },
        {
          value: 'boundaryShare',
          label: 'Share of points in boundary cells',
          help: 'The error bound of each area.'
        },
        {
          value: 'researchGrade',
          label: 'Research-grade observations per area (sum of a value)',
          help: 'GPURasterJoin also sums a per-point value.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Colour ramp',
      group: 'Display',
      apply: 'param',
      default: 'magma',
      disabledWhen: state => state.layer !== 'choropleth' || state.metric === 'difference',
      help: 'Perceptually uniform ramps. The difference metric always uses the diverging ramp.',
      options: [
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (colour-blind optimised)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showBoundary',
      label: 'Highlight boundary cells',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Cells any polygon edge touches. Only points in these cells can join a different area than the exact join gives.'
    },
    {
      kind: 'select',
      id: 'points',
      label: 'Observation points',
      group: 'Display',
      apply: 'param',
      default: 'off',
      help: 'Boundary-cell points use the join’s pointBoundaryMask output. Mismatched points are those whose raster area differs from the exact join (computed on the GPU).',
      options: [
        {value: 'off', label: 'Hidden'},
        {value: 'boundary', label: 'Points in boundary cells'},
        {value: 'mismatch', label: 'Points the raster join misassigns'}
      ]
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Fill opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.8,
      help: 'Lower it to read street names under the areas.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time raster join vs exact join',
      group: 'Compare',
      help: 'Times the whole graph, the exact join alone and rasterization plus raster join, outside the frame.'
    }
  ],

  readouts: [
    {id: 'cells', label: 'Zone raster'},
    {id: 'cellSize', label: 'Cell size', help: 'Updates when the extent follows the viewport.'},
    {id: 'joined', label: 'Points joined'},
    {
      id: 'mismatch',
      label: 'Misassigned points',
      help: 'Points whose raster area differs from the exact join.'
    },
    {
      id: 'explained',
      label: 'Where the error is',
      help: 'Every misassigned point must sit in a boundary cell. If that ever fails, the bound is violated.'
    },
    {
      id: 'bound',
      label: 'Documented error bound',
      help: 'For each area, the exact count must lie in [count - boundaryCount, count - boundaryCount + all boundary points].'
    },
    {id: 'difference', label: 'Largest area difference'},
    {id: 'boundaryJoined', label: 'Boundary-cell points'},
    {
      id: 'rasterOverflow',
      label: 'Rasterization capacity',
      help: 'Crossings of polygon edges with raster rows, against the buffer reserved. If it overflowed, the zone raster is left empty.'
    },
    {id: 'exactOverflow', label: 'Exact join overflow'},
    {id: 'timing', label: 'Timing'}
  ],

  legends: state => {
    const legends: LegendSpec[] = [];
    if (state.layer === 'zones') {
      legends.push({
        kind: 'categories',
        title: 'Zone raster: each cell holds one area',
        entries: [
          {color: [78, 201, 255, 200], label: 'Area row 0, 8, 16, ... (colours cycle)'},
          {color: [255, 148, 72, 200], label: 'Area row 1, 9, 17, ...'}
        ],
        note: 'Cells outside every area are empty.'
      });
    } else if (state.metric === 'difference') {
      legends.push({
        kind: 'ramp',
        id: 'metric',
        title: METRIC_TITLES[state.metric],
        ramp: 'diverging',
        extent: 'gpu',
        unit: 'observations',
        format: value => Math.round(value).toLocaleString('en-US')
      });
    } else {
      legends.push({
        kind: 'ramp',
        id: 'metric',
        title: METRIC_TITLES[state.metric],
        ramp: state.ramp,
        extent: 'gpu',
        sqrtScale: state.metric === 'count' || state.metric === 'exact',
        unit: state.metric === 'boundaryShare' ? 'share' : 'observations',
        format: value =>
          state.metric === 'boundaryShare'
            ? value.toFixed(3)
            : Math.round(value).toLocaleString('en-US')
      });
    }
    if (state.showBoundary) {
      legends.push({
        kind: 'categories',
        title: 'Overlay',
        entries: [{color: [255, 40, 60, 210], label: 'Boundary cell (a polygon edge touches it)'}]
      });
    }
    if (state.points !== 'off') {
      legends.push({
        kind: 'categories',
        title: 'Observation points',
        entries: [
          state.points === 'mismatch'
            ? {color: [255, 255, 60, 255], label: 'Raster area differs from exact area'}
            : {color: [255, 160, 40, 150], label: 'Point falls in a boundary cell'}
        ]
      });
    }
    return legends;
  },

  snippet: state => `import {
  getGPUPolygonRasterizationExtentValues,
  GPUPolygonRasterization,
  GPURasterJoin
} from '@luma.gl/experimental/gpu-raster';

const width = ${64 * 2 ** state.resolutionLevel}, height = ${'/* long side scaled to the city */ '}width;
extent.write(getGPUPolygonRasterizationExtentValues(originX, originY, cellWidth, cellHeight));

graph.add(
  new GPUPolygonRasterization({
    width, height, extent: extent.importToGraph(graph),   // per frame: ${state.extent === 'viewport' ? 'follows the camera' : 'fixed to the city'}
    polygonPositions, featureOffsets, polygonOffsets, ringOffsets,
    crossingCapacity,                                    // sized from the crossing count
    zones, boundary, overflow, crossingCount
  })
);
graph.add(
  new GPURasterJoin({
    width, height, extent: extent.importToGraph(graph),
    points, values: researchGrade, zones, boundary, zoneCount: 77,
    output: {counts, sums, boundaryCounts, unassignedBoundaryCount, outsideCount,
             pointZones, pointBoundaryMask}              // error accounting
  })
);
// exact reference: GPUPointInPolygonJoin({points, polygons..., pointFeatureIds, featureCounts, overflow})
const compiled = graph.compile();                        // once per resolution
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUPolygonRasterization` scan-converts polygons into a dense zone raster: each cell holds the feature whose polygon contains the cell centre (smallest row wins, even-odd over rings). `GPURasterJoin` then assigns each point the zone of the cell it falls in, so the cost per point is O(1) and independent of polygon complexity. Cells touched by a polygon edge are flagged so the error can be bounded.',
    why: 'Exact point-in-polygon cost grows with polygon complexity; a raster join does not. For dashboards that re-aggregate millions of points as the user pans, an O(1) join with a stated error bound (Zacharatou et al. 2017, "GPU Rasterization for Real-Time Spatial Aggregation over Arbitrary Polygons") is often the right trade.',
    howToRead:
      'Each cell holds one area and each point takes its cell’s area. Points are only ever wrong in boundary cells, so the error shrinks as cells shrink. The panel checks this live against the exact join: every misassigned point must be in a boundary cell and every area’s exact count must lie inside the documented bound.'
  },

  create: async ctx => (await import('./raster-join.compute')).createRasterJoin(ctx),

  story: [
    {
      id: 'the-question',
      controls: ['resolutionLevel', 'metric'],
      readouts: ['joined', 'mismatch'],
      title: 'Count observations per community area without testing polygons',
      body: 'An exact point-in-polygon join tests every observation against the polygons that might contain it. A **raster join** changes the problem: **`GPUPolygonRasterization`** first paints the 77 community areas into a grid in which every cell holds an area id, and **`GPURasterJoin`** gives each of the 43,557 points the id of the cell it lands in. The cost per point is one lookup, whatever the polygons look like.\n\nThe map is the choropleth of that join at 256 cells on the long side. Hover an area to compare its raster count with the exact count of **`GPUPointInPolygonJoin`**, which runs in the same graph as the reference. Below, **Metric** switches between the two counts and **Raster resolution** sets the grid.',
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10},
      options: {resolutionLevel: 2, layer: 'choropleth', metric: 'count', extent: 'city'}
    },
    {
      id: 'the-zone-raster',
      controls: ['layer'],
      readouts: ['cells', 'cellSize'],
      title: 'The zone raster: a table from cell to area',
      body: 'Set **Show** to *Zone raster (cell to area)*. Each cell is one area id; a cell belongs to the area whose polygon contains the cell’s centre, with a half-open edge rule so two areas that share an edge never both claim a cell on it. Where polygons overlap, the smallest row wins, exactly as in the exact join.\n\nThe raster is rewritten whenever its extent changes. The extent is just four floats: origin and cell size. The **Cell size** readout below reports it.',
      options: {layer: 'zones', opacity: 0.7}
    },
    {
      id: 'boundary-cells',
      controls: ['showBoundary', 'points'],
      readouts: ['mismatch', 'explained', 'boundaryJoined'],
      title: 'Only boundary cells can be wrong',
      body: 'A cell is fully inside or fully outside every polygon unless an edge passes through it. The rasterizer flags those **boundary cells** (red) conservatively, and the join reports how many points fall in them, per area and per point. A point in an unflagged cell joins exactly the area the exact join gives.\n\nTurn on **Highlight boundary cells** and set **Observation points** to *Points the raster join misassigns* to show the misassigned points (yellow): observations whose raster area differs from the exact area. **Where the error is** checks that every one of them sits in a boundary cell.',
      options: {layer: 'choropleth', metric: 'count', showBoundary: true, points: 'mismatch'},
      highlight: {readout: 'explained'}
    },
    {
      id: 'finer-raster',
      controls: ['resolutionLevel', 'metric'],
      readouts: ['difference', 'bound'],
      title: 'Finer cells, smaller error',
      body: 'Raise **Raster resolution** from 128 to 2,048 cells on the long side. Boundary cells get thinner, so fewer points fall in them and fewer are misassigned. Because the grid size is compile-time, each step rebuilds the graph once; **Under the hood** counts it.\n\nSet **Metric** to *Raster minus exact* to see the difference per area: a few observations at 256 cells, fewer at 2,048. The **documented error bound** readout checks, for each area, that the exact count lies in `[count - boundaryCount, count - boundaryCount + all boundary points]`.',
      options: {
        resolutionLevel: 4,
        layer: 'choropleth',
        metric: 'difference',
        showBoundary: false,
        points: 'off'
      },
      highlight: {readout: 'bound'}
    },
    {
      id: 'follow-the-viewport',
      controls: ['extent', 'resolutionLevel'],
      readouts: ['cellSize', 'rasterOverflow'],
      title: 'Let the raster follow the camera',
      body: 'Set **Raster placement** to follow the viewport and zoom in on a neighbourhood. The raster is re-fitted to the screen every frame by writing the four extent floats; the graph is not rebuilt. Cells shrink as you zoom, so the join is effectively exact in the area you are looking at.\n\nThis is the pattern for interactive aggregation: a coarse raster for the overview, a fine one under the cursor, and the cost per point stays one lookup. If the rasterization buffer is too small for a very detailed view, **Rasterization capacity** says so.',
      camera: {longitude: -87.63, latitude: 41.88, zoom: 12},
      options: {extent: 'viewport', metric: 'count', resolutionLevel: 3}
    },
    {
      id: 'cost-and-limits',
      controls: ['measure', 'metric'],
      readouts: ['timing'],
      title: 'What it costs, and when not to use it',
      body: 'Press **Time raster join vs exact join**. The exact join must test observations against polygon edges; the raster join pays for rasterization once per extent and then one lookup per point. With only 77 coarse polygons the gap is modest; it widens with polygon complexity and with the number of times the points are re-aggregated for the same extent.\n\n**Limits.** The result is approximate near boundaries by design; it needs a bound that fits your tolerance. Polygons smaller than a cell can be missed entirely, and the rasterization buffer is sized per resolution. Try **Metric** *Research-grade observations per area* (a per-point sum, 63.0% of observations citywide) or *Observations per area (exact join)* next to it.',
      options: {extent: 'city', resolutionLevel: 3, metric: 'researchGrade'}
    }
  ]
});
