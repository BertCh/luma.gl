// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import type {BuffersAndShapesOptions} from './buffers-and-shapes.compute';
import {B3_PALETTE} from './b3-palette';

const ROAD_CLASS_LABELS = [
  'Motorway',
  'Trunk',
  'Primary',
  'Secondary',
  'Tertiary',
  'Residential',
  'Service / other'
] as const;

const CITY_VIEW = {longitude: -87.7, latitude: 41.85, zoom: 9.55};

/** Buffers, generated shapes, grids, Hilbert keys and rectangle clipping. GPU work in `buffers-and-shapes.compute.ts`. */
export default defineScene<BuffersAndShapesOptions>({
  id: 'buffers-and-shapes',
  title: 'Buffers, shapes, grids and clips',
  chapter: 'geometry',
  order: 2,
  summary:
    'Draw a walking buffer around every Chicago L station, generate circles, sectors and ellipses, tile the city with square, hexagon or triangle grids ordered by a Hilbert curve, and clip two hundred thousand street vertices to a box you drag.',
  contributors: [
    'GPUOutlineGeometry',
    'GPUShapeGenerator',
    'GPUGridGenerator',
    'GPUHilbertKeys',
    'GPURectangleClip'
  ],
  datasets: [
    {id: 'cta-transit', role: 'L stations, all stops and rail lines'},
    {id: 'chicago-community-areas', role: 'polygon rings'},
    {id: 'chicago-roads', role: 'street centrelines to clip'}
  ],
  initialView: CITY_VIEW,
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Chicago construction geometry',
      subtitle: 'Straight-line reach, grids and clipping'
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(CREDITS.cta, CREDITS.cityOfChicago, CREDITS.openStreetMap)
  },

  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Tool',
      group: 'Tool',
      apply: 'compile',
      default: 'buffers',
      help: 'Each tool has its own compiled graphs, built the first time you open it. Sliders inside a tool never recompile.',
      options: [
        {
          value: 'buffers',
          label: 'Buffers',
          help: 'GPUOutlineGeometry: round-join buffer triangles around points, lines and rings.'
        },
        {
          value: 'shapes',
          label: 'Circles, sectors, ellipses',
          help: 'GPUShapeGenerator around every L station.'
        },
        {
          value: 'grid',
          label: 'Grids and Hilbert order',
          help: 'GPUGridGenerator tiles the city; GPUHilbertKeys orders the cells and the transit stops.'
        },
        {
          value: 'clip',
          label: 'Clip to a rectangle',
          help: 'GPURectangleClip cuts streets or community areas to a box you can drag.'
        }
      ]
    },

    {
      kind: 'select',
      id: 'bufferSource',
      label: 'Buffer around',
      group: 'Buffers',
      apply: 'compile',
      default: 'stations',
      disabledWhen: s => s.view !== 'buffers',
      help: 'The geometry type is compile-time: points, lines or rings each use a different kernel layout.',
      options: [
        {value: 'stations', label: 'Points: 135 L stations'},
        {value: 'lines', label: 'Lines: the eight L routes'},
        {value: 'rings', label: 'Rings: 77 community area boundaries'}
      ]
    },
    {
      kind: 'slider',
      id: 'bufferDistance',
      label: 'Buffer distance',
      group: 'Buffers',
      apply: 'param',
      min: 0,
      max: 2000,
      step: 25,
      default: 800,
      unit: 'm',
      disabledWhen: s => s.view !== 'buffers',
      help: 'A parameter-buffer write: the triangles are rewritten each frame, nothing is recompiled. 800 m is about a ten-minute walk.'
    },
    {
      kind: 'select',
      id: 'bufferSystem',
      label: 'Distance units',
      group: 'Buffers',
      apply: 'compile',
      default: 'planar',
      disabledWhen: s => s.view !== 'buffers',
      help: "Planar: positions are local metres. Spherical: positions stay in degrees and the distance is metres using each vertex's local east and north scale.",
      options: [
        {value: 'planar', label: 'Planar (local metres)'},
        {value: 'spherical', label: 'Spherical (degrees in, metres out)'}
      ]
    },
    {
      kind: 'slider',
      id: 'joinSegments',
      label: 'Round-join segments',
      group: 'Buffers',
      apply: 'compile',
      min: 3,
      max: 32,
      step: 1,
      default: 16,
      disabledWhen: s => s.view !== 'buffers',
      help: 'Triangles in each round join. The disc is inscribed, so it is slightly narrower than the true offset: 1.9% at 16, 13% at 6. Compile-time because it sets the output size.'
    },
    {
      kind: 'toggle',
      id: 'showSource',
      label: 'Show the source geometry',
      group: 'Buffers',
      apply: 'param',
      default: true,
      disabledWhen: s => s.view !== 'buffers',
      help: 'The points, lines or rings the buffer is built from.'
    },

    {
      kind: 'select',
      id: 'shapeKind',
      label: 'Shape',
      group: 'Shapes',
      apply: 'compile',
      default: 'circle',
      disabledWhen: s => s.view !== 'shapes',
      help: 'Circle: a radius per station. Sector: a pie slice facing the Loop. Ellipse: semi-axes rotated to point at the Loop. Compile-time per shape.',
      options: [
        {value: 'circle', label: 'Circle'},
        {value: 'sector', label: 'Sector (faces the Loop)'},
        {value: 'ellipse', label: 'Ellipse (long axis toward the Loop)'}
      ]
    },
    {
      kind: 'select',
      id: 'shapeSystem',
      label: 'Radii are',
      group: 'Shapes',
      apply: 'compile',
      default: 'planar',
      disabledWhen: s => s.view !== 'shapes',
      help: 'Planar: radii are local metres. Geodesic: centres are degrees and rings follow the sphere (the turf destination formula).',
      options: [
        {value: 'planar', label: 'Planar metres'},
        {value: 'geodesic', label: 'Geodesic metres'}
      ]
    },
    {
      kind: 'select',
      id: 'maxSegments',
      label: 'Maximum segments',
      group: 'Shapes',
      apply: 'compile',
      default: '64',
      disabledWhen: s => s.view !== 'shapes',
      help: 'The output capacity per ring (compile-time). The slider below can request up to this many without recompiling.',
      options: [
        {value: '16', label: '16'},
        {value: '32', label: '32'},
        {value: '64', label: '64'},
        {value: '128', label: '128'}
      ]
    },
    {
      kind: 'slider',
      id: 'segmentCount',
      label: 'Segments per ring',
      group: 'Shapes',
      apply: 'param',
      min: 3,
      max: 128,
      step: 1,
      default: 40,
      disabledWhen: s => s.view !== 'shapes',
      help: 'Smoothness of every ring this frame; clamped to the maximum above. A parameter write.'
    },
    {
      kind: 'slider',
      id: 'shapeRadius',
      label: 'Radius',
      group: 'Shapes',
      apply: 'param',
      min: 100,
      max: 3000,
      step: 50,
      default: 800,
      unit: 'm',
      disabledWhen: s => s.view !== 'shapes',
      help: 'The per-frame radius scale. Busier stations (more routes) draw larger around this mean.'
    },
    {
      kind: 'slider',
      id: 'sectorSweep',
      label: 'Sector sweep',
      group: 'Shapes',
      apply: 'param',
      min: 10,
      max: 360,
      step: 5,
      default: 120,
      unit: '°',
      disabledWhen: s => s.view !== 'shapes' || s.shapeKind !== 'sector',
      help: 'Angle of each pie slice, centred on the bearing to the Loop. Rewrites the bearings column; no recompile.'
    },
    {
      kind: 'slider',
      id: 'ellipseRatio',
      label: 'Minor / major axis',
      group: 'Shapes',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.5,
      disabledWhen: s => s.view !== 'shapes' || s.shapeKind !== 'ellipse',
      help: 'How flat the ellipses are. 1 is a circle.'
    },
    {
      kind: 'select',
      id: 'ellipseSpacing',
      label: 'Ellipse spacing',
      group: 'Shapes',
      apply: 'compile',
      default: 'arc-length',
      disabledWhen: s => s.view !== 'shapes' || s.shapeKind !== 'ellipse',
      help: 'Arc length evens out edge lengths around flat ellipses. Parameter angle preserves the original, cheaper sampling.',
      options: [
        {value: 'arc-length', label: 'Equal arc length'},
        {value: 'parameter', label: 'Equal parameter angle'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showShapeFill',
      label: 'Fill the shapes',
      group: 'Shapes',
      apply: 'param',
      default: true,
      disabledWhen: s => s.view !== 'shapes',
      help: 'Fan triangles from each centre, coloured by distance to the Loop.'
    },

    {
      kind: 'select',
      id: 'gridType',
      label: 'Grid type',
      group: 'Grid and Hilbert order',
      apply: 'compile',
      default: 'square',
      disabledWhen: s => s.view !== 'grid',
      help: 'Square, pointy-top hexagon, triangle strips or a point grid. Compile-time: the cell count and layout are fixed.',
      options: [
        {value: 'square', label: 'Square'},
        {value: 'hex', label: 'Hexagon'},
        {value: 'triangle', label: 'Triangle'},
        {value: 'point', label: 'Points (cell centres)'}
      ]
    },
    {
      kind: 'slider',
      id: 'cellSize',
      label: 'Cell width',
      group: 'Grid and Hilbert order',
      apply: 'param',
      min: 600,
      max: 1100,
      step: 25,
      default: 800,
      unit: 'm',
      disabledWhen: s => s.view !== 'grid' || s.hilbertTarget === 'stops',
      help: 'Resizes the whole tessellation every frame around the city centre. The cell count stays fixed (56 × 72), so the extent grows with the cell.'
    },
    {
      kind: 'slider',
      id: 'hilbertOrder',
      label: 'Hilbert order',
      group: 'Grid and Hilbert order',
      apply: 'compile',
      min: 1,
      max: 16,
      step: 1,
      default: 5,
      disabledWhen: s => s.view !== 'grid',
      help: 'Bits per axis: the bounds are split into 2^order × 2^order cells along the curve. Low orders give coarse blocks (cells share keys); 16 is the finest. Compile-time.'
    },
    {
      kind: 'select',
      id: 'hilbertTarget',
      label: 'Order',
      group: 'Grid and Hilbert order',
      apply: 'param',
      default: 'grid',
      disabledWhen: s => s.view !== 'grid',
      help: 'Both are keyed every time the graph runs; this picks which one is drawn.',
      options: [
        {value: 'grid', label: 'The grid cells'},
        {value: 'stops', label: 'All 10,601 transit stops'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showCurve',
      label: 'Draw the curve through the sorted items',
      group: 'Grid and Hilbert order',
      apply: 'param',
      default: true,
      disabledWhen: s => s.view !== 'grid',
      help: 'Joins consecutive items of the sorted order (the stable radix sort output of the keys).'
    },

    {
      kind: 'select',
      id: 'clipGeometry',
      label: 'Clip',
      group: 'Clip rectangle',
      apply: 'compile',
      default: 'roads',
      disabledWhen: s => s.view !== 'clip',
      help: 'Lines use Liang-Barsky per segment; polygons use Sutherland-Hodgman per ring. Compile-time geometry type.',
      options: [
        {value: 'roads', label: 'Street lines (Liang-Barsky)'},
        {value: 'areas', label: 'Community area rings (Sutherland-Hodgman)'}
      ]
    },
    {
      kind: 'slider',
      id: 'clipSize',
      label: 'Rectangle half-width',
      group: 'Clip rectangle',
      apply: 'param',
      min: 1,
      max: 15,
      step: 0.5,
      default: 4,
      unit: 'km',
      disabledWhen: s => s.view !== 'clip',
      help: 'Drag the rectangle on the map, or click to move it. Its bounds are a per-frame parameter write.'
    },
    {
      kind: 'toggle',
      id: 'showOriginal',
      label: 'Show the unclipped layer',
      group: 'Clip rectangle',
      apply: 'param',
      default: true,
      disabledWhen: s => s.view !== 'clip',
      help: 'The input geometry, faint, for comparison.'
    }
  ],

  story: [
    {
      id: 'walk-shed',
      title: 'Which parts of Chicago are a walk from an L station?',
      headline: 'Straight-line bands surround every station',
      textAlternative:
        'Translucent 800-metre discs surround 135 Chicago L station points, showing straight-line proximity to rail; overlapping discs remain separate and the result does not represent routes along the street network.',
      body: `A planner draws an 800 m circle around every rail station to see which neighbourhoods have rapid transit within about a ten-minute walk. \`GPUOutlineGeometry\` builds that picture on the GPU: for each of the 135 stations it writes a round-join disc as a triangle list, straight into a buffer the map draws.

The translucent blue is the buffered area; the dots are the stations. Slide **Buffer distance** below: the triangles are rewritten every frame with no recompile. The big gaps on the far south and west sides are the neighbourhoods more than a walk from the L. This is a picture only: overlaps are not merged, so use a distance query (\`dwithin\`) when you need the numbers.`,
      camera: {...CITY_VIEW, transitionMs: 1000},
      options: {view: 'buffers', bufferSource: 'stations', bufferDistance: 800},
      controls: ['bufferDistance']
    },
    {
      id: 'corridors',
      title: 'Buffer a line, not a point',
      headline: 'Translucent overlaps do not encode density',
      textAlternative:
        'A translucent 400-metre corridor follows eight Chicago L route lines using rectangular segments and round joins; darker overlaps are not density, buffers are not unioned, and low join-segment counts understate the requested distance.',
      body: `The same tool buffers **lines**: every vertex of the eight L routes gets a round join and a rectangle toward the next vertex, so the result is a continuous corridor. Set the distance to 400 m and you see the strip within a five-minute walk of the track, not just of the stations, a useful contrast with the station circles for noise or right-of-way questions.

Set **Buffer around** to *Rings* for buffers of the 77 community area outlines. **Distance units** switches between local metres and spherical mode (positions stay in degrees and metres are scaled per vertex). **Round-join segments** trades smoothness for work: the disc is inscribed, so with only 6 segments the corridor is visibly narrower than 400 m.`,
      options: {bufferSource: 'lines', bufferDistance: 400},
      controls: ['bufferDistance', 'bufferSource', 'bufferSystem', 'joinSegments'],
      readouts: ['bufferNarrowing']
    },
    {
      id: 'sectors',
      title: 'Sectors that face the Loop',
      headline: 'A shape needs direction and units',
      textAlternative:
        'Station-centred sectors point toward the Chicago Loop and are coloured by distance to it from 0 to 20 kilometres; radius, sweep and ring resolution are selectable, while planar and geodesic constructions use different distance models.',
      body: `\`GPUShapeGenerator\` generates **circles, sectors and ellipses** around every station. Here each sector is a pie slice centred on the bearing from the station to the Loop: an inbound catchment. The colour is distance to the Loop from 0 to 20 km, so the outer terminals are bright.

The compile-time choices are the shape and the coordinate system (planar or geodesic); **Segments per ring**, **Radius** and **Sector sweep** are parameter writes. Set **Shape** to *Ellipse* to point a long axis at the Loop, or **Radii are** to *Geodesic metres* to see the spherical ring: at city scale they agree to a metre or two. The readout reports vertices per ring.`,
      options: {view: 'shapes', shapeKind: 'sector', shapeRadius: 1500, sectorSweep: 110},
      controls: ['segmentCount', 'shapeRadius', 'sectorSweep', 'shapeKind', 'shapeSystem'],
      readouts: ['shapeVertices']
    },
    {
      id: 'grid-and-curve',
      title: 'Tile the city and order it along a curve',
      headline: 'Cell geometry changes the generated lattice',
      textAlternative:
        'A 56 by 72 square lattice is coloured by order-five Hilbert key and joined through sorted cell centres; cell size and geometry change the lattice, while Hilbert order limits key resolution rather than measuring intensity.',
      body: `\`GPUGridGenerator\` tessellates an extent: here a fixed 56 × 72 lattice of squares whose **Cell width** you resize live (below). Each cell is then keyed by \`GPUHilbertKeys\`, a space-filling curve whose consecutive keys are edge-adjacent cells, and the colour shows the position along that curve.

The line is the curve itself, drawn through the cells in sorted order. Raise **Hilbert order** from 3 to 7: each step splits every block in four, so more cells get distinct keys and the curve gets finer (order is compile-time because it sets the radix-sort width). Try **Grid type** *Hexagon*, *Triangle* and *Points*: the generator offers all four layouts.`,
      options: {view: 'grid', gridType: 'square', hilbertOrder: 5, cellSize: 800},
      controls: ['cellSize', 'hilbertOrder', 'gridType'],
      readouts: ['hilbertCells']
    },
    {
      id: 'sort-the-stops',
      title: 'Sort 10,601 transit stops along the curve',
      headline: 'Hilbert keys preserve local proximity imperfectly',
      textAlternative:
        'All 10,601 Chicago transit stops are ordered by order-eight Hilbert keys and connected in that sequence, with mean consecutive distance compared against file order; spatial proximity is encouraged but not guaranteed.',
      body: `Why order data by a space-filling curve? Locality: items that are near each other in the order are near each other on the map, so tiles, indexes and compression work better. Set **Order** to *All 10,601 transit stops*: each stop gets a Hilbert key within the Chicago bounds and a stable radix sort produces the curve order.

The readout compares the mean hop between consecutive stops along the curve with the hop in file order. Raise **Hilbert order** to 10: the curve now threads the bus network block by block.`,
      options: {hilbertTarget: 'stops', hilbertOrder: 8, showCurve: true},
      controls: ['hilbertTarget', 'hilbertOrder'],
      readouts: ['hop']
    },
    {
      id: 'clip',
      title: 'Cut everything to a box',
      headline: 'Clipping creates vertices at boundaries',
      textAlternative:
        'Chicago street lines are clipped to a movable axis-aligned rectangle and coloured by source road class, with surviving vertices and pieces reported; clipping is planar and output beyond the fixed capacity is truncated and flagged.',
      body: `\`GPURectangleClip\` clips line and polygon layers to an axis-aligned rectangle on the GPU. Here 100 000 street vertices are cut to a box you can **drag** (or click to move); **Rectangle half-width** below resizes it. Streets that leave and re-enter the box become several pieces; each piece keeps its source street through the \`sourcePaths\` output, so it can be coloured by road class.

The readouts show the vertices that survive, the number of pieces and whether the output overflowed. The rectangle is a per-frame parameter, so dragging costs only a parameter write and one encode.`,
      camera: {longitude: -87.64, latitude: 41.88, zoom: 11.3, transitionMs: 1500},
      options: {view: 'clip', clipGeometry: 'roads', clipSize: 4},
      controls: ['clipSize', 'clipGeometry'],
      readouts: ['clipVertices', 'clipPaths', 'clipOverflow']
    },
    {
      id: 'limits',
      title: 'Limits and things to try',
      headline: 'Buffers are not network walksheds',
      textAlternative:
        'Community-area rings are clipped to an eight-kilometre half-width rectangle and retain source categories; the planar clip has fixed output capacity, and polygon outlines may include zero-width traces along rectangle boundaries.',
      body: `**Limits.** Buffers are drawn, not unioned (no area, no queries) and have no negative or one-sided mode. Grids are not clipped to an extent and the lattice size is fixed at compile time. Hilbert keys inherit the f32 resolution of the cell choice. Rectangle clipping is planar and truncates output past its capacity (the overflow readout reports it); polygon clips keep zero-width bridges along the box edges, so the filled area is exact but the outline can trace the border.

**Try it.** Set **Clip** to *Community area rings* and watch each ring become one output ring. Switch **Tool** to *Grids and Hilbert order*, set **Hilbert order** to 1 and 2 and count the distinct colours, or to *Circles, sectors, ellipses* and make **Sector sweep** 360 degrees: a full turn keeps its centre spokes.`,
      options: {view: 'clip', clipGeometry: 'areas', clipSize: 8},
      controls: ['clipGeometry', 'view'],
      readouts: ['clipOverflow']
    }
  ],

  about: {
    what: 'Geometry generators and clippers that write triangles, rings, grid cells, curve keys and clipped paths into GPU buffers: round-join buffers (GPUOutlineGeometry), circles, sectors and ellipses (GPUShapeGenerator), square, hexagon and triangle grids (GPUGridGenerator), Hilbert curve keys and sorted order (GPUHilbertKeys) and rectangle clipping (GPURectangleClip).',
    why: 'These are the "make a shape" and "cut a shape" steps of vector workflows: catchment pictures, coverage fans, fishnets for aggregation, spatially sorted data and viewport clipping. On the GPU they are redone every frame while a slider moves.',
    howToRead:
      'The picture changes by tool. Buffers and shapes are drawn directly from the output buffers; colours are the legend. For the grid, colour is the position along the Hilbert curve; for the clip, colour is the road class of the street each piece came from.'
  },

  legends: state => {
    const legends: LegendSpec[] = [];
    if (state.view === 'buffers') {
      legends.push({
        kind: 'categories',
        title: 'Buffer',
        entries: [
          {color: [...B3_PALETTE[0], 200], label: `${state.bufferDistance} m round-join buffer`},
          {color: [20, 30, 50, 235], label: 'Source geometry'}
        ],
        note: 'Overlapping triangles are not merged: a picture, not a polygon.'
      });
    } else if (state.view === 'shapes') {
      legends.push({
        kind: 'ramp',
        title: 'Distance to the Loop',
        ramp: 'ylgnbu',
        extent: [0, 20],
        unit: 'km',
        format: value => `${value.toFixed(0)} km`
      });
    } else if (state.view === 'grid') {
      legends.push({
        kind: 'ramp',
        title: 'Position along the Hilbert curve',
        ramp: 'ylgnbu',
        extent: [0, 1],
        labels: ['curve start', 'curve end'],
        format: value => value.toFixed(1)
      });
      if (state.showCurve && state.hilbertTarget === 'grid') {
        legends.push({
          kind: 'ramp',
          title: 'The curve (consecutive cells)',
          ramp: 'ylorbr',
          extent: [0, 1],
          labels: ['first', 'last']
        });
      }
    } else {
      legends.push(
        state.clipGeometry === 'roads'
          ? {
              kind: 'categories',
              title: 'Street class of each clipped piece (sourcePaths)',
              entries: ROAD_CLASS_LABELS.map((label, index) => ({
                color: [...B3_PALETTE[index], 255] as const,
                label
              }))
            }
          : {
              kind: 'categories',
              title: 'Clipped community area rings',
              entries: [{color: [78, 168, 222, 255], label: 'Ring inside the rectangle'}]
            }
      );
      legends.push({
        kind: 'categories',
        title: 'Clip rectangle',
        entries: [{color: [200, 90, 10, 255], label: 'Drag me'}]
      });
    }
    return legends;
  },

  readouts: [
    {id: 'bufferInputs', label: 'Input vertices'},
    {id: 'bufferTriangles', label: 'Buffer triangles'},
    {
      id: 'bufferNarrowing',
      label: 'Join narrowing at this distance',
      help: 'The inscribed disc is up to distance × (1 - cos(π / segments)) narrower than the true offset.'
    },
    {id: 'shapeCount', label: 'Shapes', format: 'integer'},
    {id: 'shapeVertices', label: 'Vertices per shape'},
    {id: 'shapeArea', label: 'Area of an average shape'},
    {id: 'gridCells', label: 'Grid'},
    {id: 'gridExtent', label: 'Grid extent'},
    {id: 'hilbertCells', label: 'Curve resolution'},
    {
      id: 'hop',
      label: 'Mean hop between sorted stops',
      help: 'Distance between consecutive stops in Hilbert order against file order: the locality the curve buys.'
    },
    {id: 'clipVertices', label: 'Vertices kept'},
    {id: 'clipPaths', label: 'Output paths'},
    {
      id: 'clipTotal',
      label: 'Total vertices before capacity',
      help: 'The unclamped count (requiredCount output).'
    },
    {id: 'clipOverflow', label: 'Output overflowed'}
  ],

  snippet: state => {
    if (state.view === 'buffers') {
      return `import {GPUOutlineGeometry, getGPUOutlineGeometryParameterValues}
  from '@luma.gl/experimental/gpu-spatial-analysis';

graph.add(new GPUOutlineGeometry({
  positions,                                  // float32x2, ${state.bufferSystem === 'spherical' ? 'longitude/latitude degrees' : 'local metres'}
  geometryType: '${state.bufferSource === 'stations' ? 'points' : state.bufferSource}',${state.bufferSource === 'stations' ? '' : '\n  pathOffsets,                                // path or ring starts'}
  coordinateSystem: '${state.bufferSystem}',
  joinSegments: ${state.joinSegments},
  parameters: distance.importToGraph(graph),  // GPUParameterBuffer, length 4
  output: {positions: triangles}              // vertexCount * ${state.joinSegments * 3 + 6} float32x2 rows
}));

// per frame: one buffer write, no recompile
distance.write(getGPUOutlineGeometryParameterValues({distance: ${state.bufferDistance}}));`;
    }
    if (state.view === 'shapes') {
      return `import {GPUShapeGenerator, getGPUShapeGeneratorParameterValues}
  from '@luma.gl/experimental/gpu-spatial-analysis';

graph.add(new GPUShapeGenerator({
  shape: '${state.shapeKind}',
  coordinateSystem: '${state.shapeSystem}',
  ${state.shapeKind === 'ellipse' ? `ellipseSpacing: '${state.ellipseSpacing}',` : ''}
  maximumSegments: ${state.maxSegments},
  centers, radii,${state.shapeKind === 'sector' ? ' bearings,           // float32x2 start and end bearing per station' : ''}${state.shapeKind === 'ellipse' ? ' rotations,         // degrees clockwise per station' : ''}
  parameters: shapeParameters.importToGraph(graph),
  output: {positions: rings, offsets, vertexCount}
}));

shapeParameters.write(getGPUShapeGeneratorParameterValues({
  segmentCount: ${state.segmentCount}, radiusScale: ${state.shapeRadius}
}));`;
    }
    if (state.view === 'grid') {
      return `import {GPUGridGenerator, GPUHilbertKeys, getGPUGridGeneratorParameterValues}
  from '@luma.gl/experimental/gpu-spatial-analysis';

graph.add(new GPUGridGenerator({
  gridType: '${state.gridType}', columns: 56, rows: 72,
  parameters: gridParameters.importToGraph(graph),
  output: {positions: cells, centers}
}));
graph.add(new GPUHilbertKeys({
  order: ${state.hilbertOrder}, points: centers,
  bounds: bounds.importToGraph(graph),        // [minX, minY, maxX, maxY]
  output: {keys, sortedRows}                  // stable radix sort in curve order
}));

gridParameters.write(getGPUGridGeneratorParameterValues({
  minX, minY, cellWidth: ${state.cellSize}, cellHeight: ${state.cellSize}
}));`;
    }
    return `import {GPURectangleClip, getGPURectangleClipParameterValues}
  from '@luma.gl/experimental/gpu-spatial-analysis';

graph.add(new GPURectangleClip({
  positions, pathOffsets,
  geometryType: '${state.clipGeometry === 'roads' ? 'lines' : 'polygons'}',
  parameters: clipParameters.importToGraph(graph),
  output: {positions: out, pathOffsets: outOffsets, count, overflow${state.clipGeometry === 'roads' ? ', requiredCount, pathCount, sourcePaths' : ''}}
}));

// the rectangle is a per-frame parameter
clipParameters.write(getGPURectangleClipParameterValues({minX, minY, maxX, maxY}));`;
  },

  create: async ctx => (await import('./buffers-and-shapes.compute')).createBuffersAndShapes(ctx)
});
