// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import type {LineOperationsOptions} from './b3-line-options';
import {B3_PALETTE} from './b3-palette';

const formatKilometers = (value: number) => `${(value / 1000).toFixed(0)} km`;
const logMeters = (value: number) => {
  const meters = 10 ** value;
  return meters < 1000
    ? `${meters.toFixed(meters < 100 ? 0 : 0)} m`
    : `${(meters / 1000).toFixed(1)} km`;
};

const L_VIEW = {longitude: -87.7, latitude: 41.87, zoom: 9.9};

/** Line operations: densify, chunk, locate, snap, simplify, smooth. GPU work in `line-operations.compute.ts` and the `b3-line-views-*` files. */
export default defineScene<LineOperationsOptions>({
  id: 'line-operations',
  title: 'Cut, mark, snap and simplify lines',
  chapter: 'geometry',
  order: 3,
  summary:
    'Put the Chicago L routes through the line toolbox (densify, chunk, substring, locate), snap 110,000 traffic crashes to the street they happened on, and simplify or smooth 897 ship tracks in New York Harbor, all on the GPU.',
  contributors: [
    'GPULineSegmentize',
    'GPULineChunk',
    'GPULineLocate',
    'GPULinearReferencing',
    'GPULineSimplification',
    'GPULineSmooth'
  ],
  datasets: [
    {id: 'cta-transit', role: 'L route shapes'},
    {id: 'chicago-roads', role: 'street polylines'},
    {id: 'chicago-crashes', role: '110,000 crash points'},
    {id: 'ais-vessels', role: 'ship tracks with timestamps'}
  ],
  initialView: L_VIEW,

  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Layer and tools',
      group: 'Tool',
      apply: 'compile',
      default: 'reshape',
      help: 'Each layer has its own compiled graphs, built the first time you open it.',
      options: [
        {value: 'reshape', label: 'L routes: densify, chunk, locate'},
        {value: 'snap', label: 'Crashes snapped to streets'},
        {value: 'tracks', label: 'Ship tracks: simplify and smooth'}
      ]
    },

    {
      kind: 'select',
      id: 'lineTool',
      label: 'Line tool',
      group: 'L routes',
      apply: 'compile',
      default: 'densify',
      disabledWhen: s => s.view !== 'reshape',
      help: 'Each tool compiles its own graph the first time you pick it.',
      options: [
        {
          value: 'densify',
          label: 'Densify (GPULineSegmentize)',
          help: 'Split long segments into equal pieces of at most a given length.'
        },
        {
          value: 'chunk',
          label: 'Chunk (GPULineChunk)',
          help: 'Cut every route into pieces of a fixed length.'
        },
        {
          value: 'substring',
          label: 'Substring (GPULineChunk)',
          help: 'Extract the part of each route between two measures.'
        },
        {
          value: 'locate',
          label: 'Locate events (GPULineLocate)',
          help: 'Place markers at a distance or fraction along each route.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'lineSystem',
      label: 'Lengths measured',
      group: 'L routes',
      apply: 'compile',
      default: 'planar',
      disabledWhen: s => s.view !== 'reshape' || s.lineTool === 'locate',
      help: 'Planar: local metres. Spherical: positions stay in degrees and lengths are great-circle metres. Compile-time. Locate is planar only.',
      options: [
        {value: 'planar', label: 'Planar (local metres)'},
        {value: 'spherical', label: 'Spherical (degrees in, metres out)'}
      ]
    },
    {
      kind: 'slider',
      id: 'densifyLength',
      label: 'Maximum segment length',
      group: 'L routes',
      apply: 'param',
      min: 1,
      max: 3.3,
      step: 0.02,
      default: 2,
      format: logMeters,
      disabledWhen: s => s.view !== 'reshape' || s.lineTool !== 'densify',
      help: 'Log scale. Every input vertex is kept; longer segments gain equal pieces. A parameter write.'
    },
    {
      kind: 'slider',
      id: 'chunkLength',
      label: 'Chunk length',
      group: 'L routes',
      apply: 'param',
      min: 2,
      max: 4,
      step: 0.02,
      default: 3,
      format: logMeters,
      disabledWhen: s => s.view !== 'reshape' || s.lineTool !== 'chunk',
      help: 'Log scale. Consecutive chunks share their boundary point; the piece count is data-dependent and read back.'
    },
    {
      kind: 'slider',
      id: 'substringStart',
      label: 'Start measure',
      group: 'L routes',
      apply: 'param',
      min: 0,
      max: 40,
      step: 0.5,
      default: 5,
      unit: 'km',
      disabledWhen: s => s.view !== 'reshape' || s.lineTool !== 'substring',
      help: 'Distance from the start of each route. Measures are clamped to the route; a start past the end gives an empty path.'
    },
    {
      kind: 'slider',
      id: 'substringEnd',
      label: 'End measure',
      group: 'L routes',
      apply: 'param',
      min: 0,
      max: 40,
      step: 0.5,
      default: 12,
      unit: 'km',
      disabledWhen: s => s.view !== 'reshape' || s.lineTool !== 'substring',
      help: 'Distance from the start where the extracted part ends.'
    },
    {
      kind: 'select',
      id: 'locateMode',
      label: 'Measure is a',
      group: 'L routes',
      apply: 'compile',
      default: 'distance',
      disabledWhen: s => s.view !== 'reshape' || s.lineTool !== 'locate',
      help: 'Distance in metres from the route start (turf along) or a fraction of the route length (ST_LineInterpolatePoint). Compile-time.',
      options: [
        {value: 'distance', label: 'Distance (metres)'},
        {value: 'fraction', label: 'Fraction of the route'}
      ]
    },
    {
      kind: 'slider',
      id: 'locateSpacing',
      label: 'Spacing scale',
      group: 'L routes',
      apply: 'param',
      min: 0.25,
      max: 4,
      step: 0.25,
      default: 1,
      unit: '×',
      disabledWhen: s => s.view !== 'reshape' || s.lineTool !== 'locate',
      help: 'Multiplies every event measure (the measureScale parameter). 1 is a marker every kilometre in distance mode; above that, markers beyond the route end stack up clamped to it.'
    },
    {
      kind: 'slider',
      id: 'locateLateral',
      label: 'Lateral offset',
      group: 'L routes',
      apply: 'param',
      min: 0,
      max: 80,
      step: 5,
      default: 25,
      unit: 'm',
      disabledWhen: s => s.view !== 'reshape' || s.lineTool !== 'locate',
      help: 'Shifts markers left and right of the route in alternating lanes (the eventOffsets column).'
    },
    {
      kind: 'toggle',
      id: 'locateAnimate',
      label: 'Run the markers along the routes',
      group: 'L routes',
      apply: 'param',
      default: true,
      disabledWhen: s => s.view !== 'reshape' || s.lineTool !== 'locate',
      help: 'Animates the measureOffset parameter: one buffer write per frame.'
    },

    {
      kind: 'slider',
      id: 'snapRadius',
      label: 'Search radius',
      group: 'Crashes',
      apply: 'param',
      min: 5,
      max: 150,
      step: 5,
      default: 60,
      unit: 'm',
      disabledWhen: s => s.view !== 'snap',
      help: "A crash only snaps to a street within this distance. The dataset's own snap used 60 m. A per-frame parameter."
    },
    {
      kind: 'select',
      id: 'snapColor',
      label: 'Snap lines coloured by',
      group: 'Crashes',
      apply: 'param',
      default: 'side',
      disabledWhen: s => s.view !== 'snap',
      options: [
        {
          value: 'side',
          label: 'Side of the street',
          help: 'The signed offset: left or right of the street direction.'
        },
        {value: 'measure', label: 'Measure along the street'},
        {value: 'distance', label: 'Snap distance'}
      ],
      help: 'Which output column colours the line from each crash to its foot point.'
    },

    {
      kind: 'select',
      id: 'trackTool',
      label: 'Track tool',
      group: 'Ship tracks',
      apply: 'compile',
      default: 'simplify',
      disabledWhen: s => s.view !== 'tracks',
      help: 'Each tool compiles its own graphs the first time you pick it.',
      options: [
        {value: 'simplify', label: 'Simplify (GPULineSimplification)'},
        {value: 'smooth', label: 'Smooth (GPULineSmooth)'}
      ]
    },
    {
      kind: 'select',
      id: 'simplifyMetric',
      label: 'Distance measure',
      group: 'Ship tracks',
      apply: 'compile',
      default: 'segment',
      disabledWhen: s => s.view !== 'tracks' || s.trackTool !== 'simplify',
      help: 'Segment: classic Douglas-Peucker distance to the chord. Time-ratio: distance to where the vessel would be at that moment if it moved at constant speed (TD-TR). Compile-time.',
      options: [
        {value: 'segment', label: 'Segment distance (Douglas-Peucker)'},
        {value: 'time-ratio', label: 'Time-ratio (TD-TR)'}
      ]
    },
    {
      kind: 'select',
      id: 'simplifyRounds',
      label: 'Maximum rounds',
      group: 'Ship tracks',
      apply: 'compile',
      default: '64',
      disabledWhen: s => s.view !== 'tracks' || s.trackTool !== 'simplify',
      help: 'Cap on level-synchronous rounds of the importance pass. A balanced split needs about log2(n); a low cap leaves a superset of the exact result. Compile-time.',
      options: [
        {value: '4', label: '4'},
        {value: '8', label: '8'},
        {value: '16', label: '16'},
        {value: '64', label: '64'},
        {value: '256', label: '256'}
      ]
    },
    {
      kind: 'toggle',
      id: 'simplifyAuto',
      label: 'Tolerance follows the zoom',
      group: 'Ship tracks',
      apply: 'param',
      default: false,
      disabledWhen: s => s.view !== 'tracks' || s.trackTool !== 'simplify',
      help: 'Sets the tolerance to a few pixels at the current zoom: zoom out and the track gets coarser. Still only a parameter write.'
    },
    {
      kind: 'slider',
      id: 'simplifyTolerance',
      label: 'Tolerance',
      group: 'Ship tracks',
      apply: 'param',
      min: 0,
      max: 3.3,
      step: 0.05,
      default: 1.5,
      format: logMeters,
      disabledWhen: s => s.view !== 'tracks' || s.trackTool !== 'simplify' || s.simplifyAuto,
      help: 'Log scale. Vertices whose importance is below it are dropped; moving it re-runs only the selection, not the 64 rounds.'
    },
    {
      kind: 'slider',
      id: 'simplifyPixels',
      label: 'Automatic tolerance',
      group: 'Ship tracks',
      apply: 'param',
      min: 0.25,
      max: 12,
      step: 0.25,
      default: 2,
      unit: ' px',
      disabledWhen: s => s.view !== 'tracks' || s.trackTool !== 'simplify' || !s.simplifyAuto,
      help: 'Pixels of error allowed at the current zoom.'
    },
    {
      kind: 'slider',
      id: 'smoothIterations',
      label: 'Chaikin iterations',
      group: 'Ship tracks',
      apply: 'compile',
      min: 1,
      max: 5,
      step: 1,
      default: 2,
      disabledWhen: s => s.view !== 'tracks' || s.trackTool !== 'smooth',
      help: 'Each iteration doubles the vertex count and cuts the corners again. Compile-time because it sets the output size.'
    },
    {
      kind: 'slider',
      id: 'smoothRatio',
      label: 'Cut ratio',
      group: 'Ship tracks',
      apply: 'param',
      min: 0.05,
      max: 0.5,
      step: 0.05,
      default: 0.25,
      disabledWhen: s => s.view !== 'tracks' || s.trackTool !== 'smooth',
      help: "Where each corner is cut: 0.25 is Chaikin's classic quarter points; 0.5 would only join midpoints. A parameter write."
    },
    {
      kind: 'toggle',
      id: 'showOriginal',
      label: 'Show the original lines',
      group: 'Display',
      apply: 'param',
      default: true,
      disabledWhen: s => s.view === 'snap',
      help: 'The unmodified input, underneath the result.'
    }
  ],

  story: [
    {
      id: 'densify',
      title: 'A vertex every 100 metres',
      body: `A transit planner needs the Chicago L as points: a vertex every 100 m along each route, each carrying its **distance from the route start**. \`GPULineSegmentize\` does that on the GPU: every input vertex is kept and any long segment is split into equal pieces no longer than the **maximum segment length** you choose.

The colour is the measure output, from 0 at the start to the length of the longest route at the far end (see the legend). Slide **Maximum segment length** below (log scale) and watch the vertex count in the readout change; nothing recompiles. Switch **Lengths measured** to *Spherical* to measure along the sphere instead of the local plane: over a city the two agree to a few metres.`,
      camera: {...L_VIEW, transitionMs: 1000},
      options: {view: 'reshape', lineTool: 'densify', densifyLength: 2},
      highlight: {readout: 'reshapeVertices'},
      controls: ['densifyLength', 'lineSystem'],
      readouts: ['reshapeVertices']
    },
    {
      id: 'chunk',
      title: 'Cut the routes into kilometre pieces',
      body: `Now split each route into **one-kilometre chunks** with \`GPULineChunk\` in \`chunk\` mode (turf \`lineChunk\`). The colours cycle through the output pieces, so you can count kilometres along each line; consecutive chunks share their boundary point.

The number of pieces depends on the data, so the contributor reports it through the \`pathCount\` output (see the readout) and the offsets past it equal the vertex count. Slide **Chunk length** below from 100 m to 10 km. Set **Line tool** to *Substring* to extract the stretch between a start and an end measure instead, as ST_LineSubstring does: a quick way to reveal a route a bit at a time.`,
      options: {lineTool: 'chunk', chunkLength: 3},
      highlight: {readout: 'reshapePieces'},
      controls: ['chunkLength', 'lineTool'],
      readouts: ['reshapePieces']
    },
    {
      id: 'locate',
      title: 'Run a train along each line',
      body: `\`GPULineLocate\` turns "N km along route R" into a map position and a direction. Here 320 markers (40 per route) sit a kilometre apart and **run along the lines**: only the \`measureOffset\` parameter changes each frame, a single buffer write.

Each marker also reports a tangent (the white tick) so it can point along the track, and a status that says when it was clamped at the route end. Raise **Spacing scale** above 1 and markers pile up at the end of the shorter routes (see the clamped count). Add a **Lateral offset** to put alternate markers on either side. Set **Measure is a** to *Fraction* to place markers by fraction of route length instead of distance.`,
      options: {lineTool: 'locate', locateMode: 'distance'},
      highlight: {readout: 'reshapePieces'},
      controls: ['locateAnimate', 'locateSpacing', 'locateLateral', 'locateMode'],
      readouts: ['reshapePieces']
    },
    {
      id: 'snap',
      title: 'Which street was each crash on?',
      body: `A crash is recorded as a point; a safety analyst wants the **street**, the **distance along it** and the **side**. \`GPULinearReferencing\` snaps every point to the nearest of 49,000 street polylines within a search radius and returns the foot point, the measure, the signed offset and the street's row.

Lines run from each of 110,000 crashes to its foot point. The map colours them by side of the street (left or right of the street's direction). The readout compares the GPU result with the snap the dataset ships (built offline against the same streets): they agree where a crash has one clear nearest street, and differ at intersections where two streets are equally near. Shrink **Search radius** below to 15 m and unmatched crashes appear; grow it and mean snap distance rises.`,
      camera: {longitude: -87.66, latitude: 41.87, zoom: 11.2, transitionMs: 1600},
      options: {view: 'snap', snapRadius: 60, snapColor: 'side'},
      highlight: {readout: 'snapAgreement'},
      controls: ['snapRadius', 'snapColor'],
      readouts: ['snapAgreement', 'snapMatched', 'snapDistance']
    },
    {
      id: 'simplify',
      title: 'Thin the ship tracks',
      body: `897 AIS tracks of vessels in New York Harbor hold 124,780 position fixes. \`GPULineSimplification\` computes a Douglas-Peucker **importance** for every fix once (64 gated rounds on the GPU), then the **tolerance** you set, in metres or in pixels at the current zoom, only re-selects the vertices to keep.

Orange is the simplified track over the faint original. Slide **Tolerance** below: at 30 m the kept fraction drops to a few percent while the shape stays right. The readouts show whether the rounds converged (then the result equals the classic recursive algorithm) and how many rounds ran. Switch **Distance measure** to *Time-ratio* (or tick **Tolerance follows the zoom**) to keep the fixes that matter for speed, not just shape (TD-TR).`,
      camera: {longitude: -74.03, latitude: 40.66, zoom: 10.3, transitionMs: 1600},
      options: {view: 'tracks', trackTool: 'simplify', simplifyTolerance: 1.5},
      highlight: {readout: 'tracksRatio'},
      controls: ['simplifyTolerance', 'simplifyAuto', 'simplifyMetric'],
      readouts: ['tracksRatio', 'tracksConverged', 'tracksRounds']
    },
    {
      id: 'smooth',
      title: 'Round the corners',
      body: `The opposite operation: **smooth** a track by Chaikin corner cutting. \`GPULineSmooth\` replaces every corner by two points at a **cut ratio** along its two edges (0.25 is the classic quarter points) and repeats for the chosen number of iterations, each doubling the vertex count.

Set **Chaikin iterations** to two or three and the jagged fixes of the ferry crossings become flowing curves, which is a cosmetic fix only: smoothing invents positions and can cut corners across land. The iteration count is compile-time (it sets the output size); the cut ratio is a parameter write.`,
      options: {trackTool: 'smooth', smoothIterations: 3, smoothRatio: 0.25},
      highlight: {readout: 'tracksSmoothVertices'},
      controls: ['smoothIterations', 'smoothRatio'],
      readouts: ['tracksSmoothVertices']
    },
    {
      id: 'limits',
      title: 'Limits and things to try',
      body: `**Limits.** Densify, chunk, locate and snap are planar (Locate and Linear Referencing take projected coordinates) except where a spherical mode is offered. Output sizes are capacities fixed at compile time: a too-small capacity truncates and sets the overflow flag. Simplification does not preserve topology: two simplified tracks can cross. TD-TR needs timestamps; Chaikin smoothing can cut across land.

**Try it.** Set **Spacing scale** to 4 below and see which routes are shorter than 40 km. Then switch **Layer and tools** to *Crashes snapped to streets*, set **Search radius** to 5 m and read the **Snapped** readout to see how many crashes stay unmatched.`,
      options: {view: 'reshape', lineTool: 'locate'},
      controls: ['locateSpacing', 'view', 'snapRadius'],
      readouts: ['reshapePieces', 'snapMatched']
    }
  ],

  about: {
    what: 'The line toolbox on the GPU: add vertices (GPULineSegmentize), cut into pieces or extract a stretch (GPULineChunk), place events by distance (GPULineLocate), snap points to lines and measure along them (GPULinearReferencing), thin tracks at a tolerance (GPULineSimplification) and round corners (GPULineSmooth).',
    why: 'Linear referencing, route markers, map matching and track thinning are everyday transport and logistics tasks. Doing them on the GPU keeps the sliders interactive over hundreds of thousands of vertices.',
    howToRead:
      'The faint line is the input. The coloured line or markers are the output buffer, drawn straight from the GPU. Vertex colour is the distance along the route; chunk colours cycle per piece; snap lines are coloured by side, measure or distance (legend).'
  },

  legends: state => {
    const legends: LegendSpec[] = [];
    if (state.view === 'reshape') {
      if (state.lineTool === 'densify') {
        legends.push({
          kind: 'ramp',
          id: 'measure',
          title: 'Distance from the route start (measures)',
          ramp: 'viridis',
          extent: 'gpu',
          format: formatKilometers
        });
      } else if (state.lineTool === 'chunk') {
        legends.push({
          kind: 'categories',
          title: 'Chunks (colour cycles per output piece)',
          entries: B3_PALETTE.slice(0, 6).map((color, index) => ({
            color: [...color, 255] as const,
            label: `piece ${index}, ${index + 8}, ${index + 16}...`
          }))
        });
      } else if (state.lineTool === 'substring') {
        legends.push({
          kind: 'categories',
          title: 'Lines',
          entries: [
            {
              color: [240, 150, 60, 255],
              label: `Substring from ${state.substringStart} to ${state.substringEnd} km`
            },
            {color: [150, 160, 180, 120], label: 'Full route'}
          ]
        });
      } else {
        legends.push({
          kind: 'categories',
          title: 'Events',
          entries: [
            {color: [240, 150, 60, 255], label: 'Event position'},
            {color: [30, 40, 60, 230], label: 'Tangent tick (direction of travel)'},
            {color: [150, 160, 180, 120], label: 'Route'}
          ]
        });
      }
    } else if (state.view === 'snap') {
      const limit = Math.min(state.snapRadius, 40);
      legends.push(
        state.snapColor === 'side'
          ? {
              kind: 'ramp',
              title: 'Offset from the street',
              ramp: 'diverging',
              extent: [-limit, limit],
              labels: ['right', 'left'],
              format: value => `${value.toFixed(0)} m`
            }
          : state.snapColor === 'measure'
            ? {
                kind: 'ramp',
                title: 'Measure along the street',
                ramp: 'viridis',
                extent: [0, 600],
                format: value => `${value.toFixed(0)} m`
              }
            : {
                kind: 'ramp',
                title: 'Snap distance',
                ramp: 'inferno',
                extent: [0, limit],
                format: value => `${value.toFixed(0)} m`
              }
      );
    } else {
      legends.push({
        kind: 'categories',
        title: 'Tracks',
        entries: [
          {
            color: [240, 150, 60, 255],
            label: state.trackTool === 'simplify' ? 'Simplified (kept fixes)' : 'Smoothed (Chaikin)'
          },
          {color: [120, 170, 255, 120], label: 'Original AIS track'}
        ]
      });
    }
    return legends;
  },

  readouts: [
    {id: 'reshapeInputs', label: 'L routes'},
    {id: 'reshapeVertices', label: 'Output vertices'},
    {id: 'reshapeSpacing', label: 'Mean spacing'},
    {
      id: 'reshapePieces',
      label: 'Pieces / events',
      help: 'Chunks and substrings from the pathCount output; for Locate the status counts.'
    },
    {id: 'reshapeOverflow', label: 'Overflow'},
    {id: 'snapInputs', label: 'Crashes and streets'},
    {id: 'snapMatched', label: 'Snapped'},
    {id: 'snapSides', label: 'Side of the street'},
    {id: 'snapDistance', label: 'Mean snap distance'},
    {
      id: 'snapAgreement',
      label: 'Agrees with the dataset snap',
      help: "Share of crashes whose GPU-snapped street equals the street in the dataset's precomputed edgeIndex."
    },
    {
      id: 'snapCandidates',
      label: 'Candidates',
      help: 'Bounding-box (point, segment) candidates against the compile-time capacity.'
    },
    {id: 'tracksInputs', label: 'Ship tracks'},
    {id: 'tracksTolerance', label: 'Tolerance'},
    {id: 'tracksKept', label: 'Kept fixes'},
    {id: 'tracksRatio', label: 'Kept ratio'},
    {id: 'tracksConverged', label: 'Importance converged'},
    {id: 'tracksRounds', label: 'Rounds used'},
    {id: 'tracksSmoothVertices', label: 'Smoothed vertices'}
  ],

  snippet: state => {
    if (state.view === 'reshape') {
      if (state.lineTool === 'densify') {
        return `import {GPULineSegmentize, getGPULineSegmentizeParameterValues}
  from '@luma.gl/experimental/gpu-spatial-analysis';

graph.add(new GPULineSegmentize({
  positions, pathOffsets,
  coordinateSystem: '${state.lineSystem}',
  parameters: segmentize.importToGraph(graph),
  output: {positions: out, pathOffsets: outOffsets, count, overflow, measures}
}));

segmentize.write(getGPULineSegmentizeParameterValues({maximumSegmentLength: ${Math.round(10 ** state.densifyLength)}}));`;
      }
      if (state.lineTool === 'chunk' || state.lineTool === 'substring') {
        return `import {GPULineChunk, getGPULineChunkParameterValues}
  from '@luma.gl/experimental/gpu-spatial-analysis';

graph.add(new GPULineChunk({
  positions, pathOffsets,
  mode: '${state.lineTool}',
  coordinateSystem: '${state.lineSystem}',
  parameters: chunk.importToGraph(graph),
  output: {positions: out, pathOffsets: outOffsets, count, overflow, pathCount${state.lineTool === 'chunk' ? ', sourcePaths' : ''}}
}));

chunk.write(getGPULineChunkParameterValues(${state.lineTool === 'chunk' ? `{chunkLength: ${Math.round(10 ** state.chunkLength)}}` : `{startMeasure: ${state.substringStart * 1000}, endMeasure: ${state.substringEnd * 1000}}`}));`;
      }
      return `import {GPULineLocate, getGPULineLocateParameterValues}
  from '@luma.gl/experimental/gpu-spatial-analysis';

graph.add(new GPULineLocate({
  positions, pathOffsets,
  eventPaths, eventMeasures, eventOffsets,
  measureMode: '${state.locateMode}',
  parameters: locate.importToGraph(graph),
  output: {positions: eventPositions, tangents, statuses}
}));

// every frame: animate the offset
locate.write(getGPULineLocateParameterValues({measureScale: ${state.locateSpacing}, measureOffset: elapsedMetres}));`;
    }
    if (state.view === 'snap') {
      return `import {GPULinearReferencing} from '@luma.gl/experimental/gpu-spatial-analysis';

graph.add(new GPULinearReferencing({
  points, positions, pathOffsets,      // planar metres
  radius: radius.importToGraph(graph), // ${state.snapRadius} m, one float32 row
  candidateCapacity: 1 << 22,
  spatialSort: true,
  output: {footPoints, distances, measures, signedOffsets, pathIndices},
  overflow, candidateCount
}));`;
    }
    if (state.trackTool === 'smooth') {
      return `import {GPULineSmooth, getGPULineSmoothParameterValues}
  from '@luma.gl/experimental/gpu-spatial-analysis';

graph.add(new GPULineSmooth({
  positions, pathOffsets,
  iterations: ${state.smoothIterations},        // compile-time: output holds n * 2^iterations vertices
  closed: false,
  parameters: smooth.importToGraph(graph),
  output: {positions: out, pathOffsets: outOffsets, count, overflow}
}));

smooth.write(getGPULineSmoothParameterValues({ratio: ${state.smoothRatio}}));`;
    }
    return `import {GPULineSimplification, getGPULineSimplificationParameterValues}
  from '@luma.gl/experimental/gpu-spatial-analysis';

// Graph 1, encoded once: per-vertex importance
importanceGraph.add(new GPULineSimplification({
  positions, trackOffsets, importance,
  timestamps,                            // needed for 'time-ratio'
  metric: '${state.simplifyMetric}', maximumRounds: ${state.simplifyRounds},
  status: {converged, roundCount}
}));

// Graph 2, every frame the tolerance changes
selectionGraph.add(new GPULineSimplification({
  positions, trackOffsets, importance, computeImportance: false,
  parameters: tolerance.importToGraph(selectionGraph),
  selection: {output: {ids, count, overflow, totalCount}}
}));
tolerance.write(getGPULineSimplificationParameterValues({tolerance: 30}));`;
  },

  create: async ctx => (await import('./line-operations.compute')).createLineOperations(ctx)
});
