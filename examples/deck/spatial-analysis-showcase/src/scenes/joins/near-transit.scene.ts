// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import type {NearTransitOptions} from './near-transit.compute';

const LINE_LEGEND = [
  {color: [198, 12, 48, 255], label: 'Red'},
  {color: [150, 90, 230, 255], label: 'Purple'},
  {color: [249, 227, 0, 255], label: 'Yellow'},
  {color: [0, 161, 222, 255], label: 'Blue'},
  {color: [226, 126, 166, 255], label: 'Pink'},
  {color: [0, 155, 58, 255], label: 'Green'},
  {color: [249, 110, 40, 255], label: 'Orange'},
  {color: [160, 100, 60, 255], label: 'Brown'}
] as const;

const POINT_LABELS: Record<NearTransitOptions['points'], string> = {
  observations: 'nature observations',
  places: 'places',
  crashes: 'traffic crashes'
};

export default defineScene<NearTransitOptions>({
  id: 'near-transit',
  title: 'Select by distance from the L',
  chapter: 'joins',
  order: 3,
  summary:
    'Which nature observations, places or crashes lie within a distance of an L line, an L station or a bus stop? The distance is a per-frame float, the selection is drawn straight from the GPU with an indirect draw and no readback, and a category readout shows what the buffer over-represents.',
  contributors: ['GPUBufferSelection'],
  datasets: [
    {id: 'chicago-nature', role: 'points to select'},
    {id: 'chicago-places', role: 'points to select'},
    {id: 'chicago-crashes', role: 'points to select'},
    {id: 'cta-transit', role: 'L lines, L stations and bus stops'}
  ],
  initialView: {longitude: -87.66, latitude: 41.87, zoom: 10.2},

  options: [
    {
      kind: 'select',
      id: 'points',
      label: 'Select which points?',
      group: 'Selection',
      apply: 'compile',
      default: 'observations',
      help: 'The point layer that is tested. Its length is part of the compiled graph, so a new layer rebuilds.',
      options: [
        {value: 'observations', label: 'Nature observations, 2023 (43,557)'},
        {
          value: 'places',
          label: 'Places (105,808)',
          help: 'Overture places: a proxy for where people go.'
        },
        {value: 'crashes', label: 'Traffic crashes, 2023 (109,711)'}
      ]
    },
    {
      kind: 'select',
      id: 'features',
      label: 'Distance from what?',
      group: 'Selection',
      apply: 'compile',
      default: 'stations',
      help: 'Point features (stations, bus stops) or polyline features (the L track as segments). The feature kind is compile-time.',
      options: [
        {value: 'stations', label: 'L stations (135)'},
        {value: 'lines', label: 'L track'},
        {value: 'bus', label: 'Bus stops (10,466)'}
      ]
    },
    {
      kind: 'select',
      id: 'line',
      label: 'Which L line?',
      group: 'Selection',
      apply: 'param',
      default: 'all',
      disabledWhen: state => state.features === 'bus',
      help: 'Chooses a subset of the features without recompiling: the feature buffers are rewritten and the other rows are parked far outside the data, where no point can be in range.',
      options: [
        {value: 'all', label: 'All eight lines'},
        {value: 'Red', label: 'Red Line'},
        {value: 'Blue', label: 'Blue Line'},
        {value: 'Brn', label: 'Brown Line'},
        {value: 'G', label: 'Green Line'},
        {value: 'Org', label: 'Orange Line'},
        {value: 'P', label: 'Purple Line'},
        {value: 'Pink', label: 'Pink Line'},
        {value: 'Y', label: 'Yellow Line'}
      ]
    },
    {
      kind: 'slider',
      id: 'distance',
      label: 'Buffer distance',
      group: 'Selection',
      apply: 'param',
      min: 0,
      max: 2000,
      step: 25,
      default: 400,
      unit: 'm',
      help: 'One float32 rewritten per frame. A point is selected when its planar distance to the nearest feature is at most this. The candidate buffer is sized per feature layer, so the distance is capped at 500 m for bus stops, 1 km for track and 2 km for stations.'
    },
    {
      kind: 'toggle',
      id: 'spatialSort',
      label: 'Hilbert-sort the features',
      group: 'Selection',
      apply: 'compile',
      default: true,
      help: 'Reorders features along a space-filling curve before the BVH build. The selection is identical either way; only traversal cost changes.'
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Colour selected points by',
      group: 'Display',
      apply: 'param',
      default: 'distance',
      help: 'Distance uses the per-point distance output. Line uses the nearest feature id, mapped to the L line it belongs to.',
      options: [
        {value: 'distance', label: 'Distance to the nearest feature'},
        {value: 'line', label: 'Nearest L line (official colours)'},
        {value: 'single', label: 'One colour'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Distance ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state => state.colorBy !== 'distance',
      help: 'Perceptually uniform ramps. The legend and the map share one ramp table.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis (colour-blind optimised)'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showBuffer',
      label: 'Show the buffer footprint',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'A translucent band of the buffer width around the features, sized in pixels from the current zoom.'
    },
    {
      kind: 'toggle',
      id: 'showOutside',
      label: 'Show unselected points',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws the points outside the buffer small and gray, from the same mask buffer.'
    },
    {
      kind: 'slider',
      id: 'pointSize',
      label: 'Selected point size',
      group: 'Display',
      apply: 'param',
      min: 0.5,
      max: 5,
      step: 0.5,
      default: 1.5,
      unit: 'px',
      help: 'Disc radius of the selected points.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Selected opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.8,
      help: 'Lower it where selected points overlap.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time Hilbert sort on vs off',
      group: 'Compare',
      help: 'Builds the selection graph both ways and times it outside the frame.'
    }
  ],

  readouts: [
    {id: 'pointsTotal', label: 'Points tested'},
    {
      id: 'features',
      label: 'Features in play',
      help: 'After the line filter; the rest are parked.'
    },
    {id: 'distanceUsed', label: 'Distance in use'},
    {
      id: 'selected',
      label: 'Selected',
      help: 'The clamped count the GPU wrote for the indirect draw, read back once after the map settles.'
    },
    {
      id: 'composition',
      label: 'Compared with the whole city',
      help: 'Share of each category inside the buffer divided by its share citywide. 1x means no difference; categories with fewer than 50 selected points are left out.'
    },
    {id: 'overflow', label: 'Overflow'},
    {id: 'timeSort', label: 'Selection timing'}
  ],

  legends: state => {
    const legends: LegendSpec[] = [];
    if (state.colorBy === 'distance') {
      legends.push({
        kind: 'ramp',
        title: `Selected ${POINT_LABELS[state.points]}: distance to the nearest feature`,
        ramp: state.ramp,
        extent: [0, Math.max(state.distance, 1)],
        unit: 'm',
        format: value => Math.round(value).toLocaleString('en-US')
      });
    } else if (state.colorBy === 'line' && state.features !== 'bus') {
      legends.push({
        kind: 'categories',
        title: `Selected ${POINT_LABELS[state.points]}: nearest L line`,
        entries: LINE_LEGEND,
        note: 'Shared stations take the first line that calls there.'
      });
    }
    legends.push({
      kind: 'categories',
      title: 'Overlay',
      entries: [
        {color: [90, 100, 120, 160], label: 'Outside the buffer'},
        {color: [30, 120, 220, 110], label: `Buffer width ${state.distance} m`}
      ]
    });
    return legends;
  },

  snippet: state => `import {GPUCommandGraph, DrawCommandBuffer} from '@luma.gl/gpgpu/gpu-core';
import {GPUBufferSelection} from '@luma.gl/experimental/gpu-spatial-analysis';

const drawCommands = new DrawCommandBuffer(device, {
  type: 'draw',
  commands: [{vertexCount: 6, instanceCount: 0}]
});
const graph = new GPUCommandGraph(device, {id: 'near-transit'});
graph.add(
  new GPUBufferSelection({
    points,                                  // ${POINT_LABELS[state.points]}, float32x2 meters, uploaded once
    features: ${state.features === 'lines' ? "{kind: 'segments', starts, ends}" : "{kind: 'points', positions}"},
    distance: distance.importToGraph(graph), // one float32, rewritten per frame
    candidateCapacity: points.length * ${state.features === 'lines' ? 40 : 24},
    spatialSort: ${state.spatialSort},
    outputMask,                              // per point: 0 or 1
    output: {ids, count, overflow, totalCount}, // selected ids, ascending
    // the clamped count goes straight into the indirect draw record:
    drawInstanceCount: graph.importGPUData('count', drawCommands.getInstanceCountData(0)),
    distances, nearestFeatureIds, overflow
  })
);
const compiled = graph.compile();            // once
distance.write(Float32Array.of(${state.distance}));        // parameter write
compiled.encode(commandEncoder, {parameters: undefined});
// layer: positions + ids with drawCommands, so no readback is needed`,

  about: {
    what: '`GPUBufferSelection` answers "which points lie within d of these features?" (ArcGIS Select By Location, PostGIS ST_DWithin). It joins each point to its nearest feature with `GPUNearestFeatureJoin` inside a per-frame distance, publishes a 0/1 mask and the ascending ids of the selected points, and writes the selected count into an indirect draw record.',
    why: 'Planning questions are distance questions: who lives within a 10-minute walk of a station, which crashes happen along a corridor, where does a buffer capture activity. Because the distance is a buffer write, the answer follows a slider at display rate.',
    howToRead:
      'Coloured points are inside the buffer; gray points are outside. The translucent band is the buffer. **Compared with the whole city** divides each category’s share inside the buffer by its citywide share: values above 1 mean over-represented near the transit feature. Planar distance is not walking distance.'
  },

  create: async ctx => (await import('./near-transit.compute')).createNearTransit(ctx),

  story: [
    {
      id: 'the-question',
      controls: ['points', 'features', 'distance'],
      readouts: ['selected', 'composition'],
      title: 'How much wildlife is logged within a short walk of an L station?',
      body: 'The CTA runs 135 L stations inside Chicago. **`GPUBufferSelection`** selects the 43,557 nature observations of 2023 whose distance to the nearest station is at most the **Buffer distance**: here 400 m, about a five-minute walk. Selected observations are coloured by their distance; the rest stay gray.\n\nThe selected points are drawn through an **indirect draw**: the GPU writes the selected count into the draw record and the layer draws the compact ids, so nothing is read back to decide how many points to draw. Read **Selected** and **Compared with the whole city** for the share and what is over-represented.',
      camera: {longitude: -87.66, latitude: 41.87, zoom: 10.2},
      options: {points: 'observations', features: 'stations', distance: 400, colorBy: 'distance'},
      highlight: {readout: 'selected'}
    },
    {
      id: 'distance-is-a-float',
      controls: ['distance'],
      readouts: ['selected', 'distanceUsed'],
      title: 'The buffer distance is one float per frame',
      body: 'Widen **Buffer distance** to 800 m, then 1,600 m. The distance is a one-float parameter buffer, so the selection is re-run with no recompile and no CPU loop over points. The translucent band is the same distance drawn in pixels.\n\nWatch **Selected**: the share grows quickly with distance as the buffers merge along the lines. The candidate buffer limits stations to 2 km and bus stops to 500 m; the **Distance in use** readout shows the distance actually applied.',
      options: {distance: 1200}
    },
    {
      id: 'track-corridor',
      controls: ['features', 'distance', 'colorBy'],
      readouts: ['selected'],
      title: 'Distance to the track, not the station',
      body: 'Switch **Distance from what?** to *L track*: the segments of the route shapes, each a line feature. A point is selected when it is within the distance of any segment, so the elevated corridors show up along their whole length, not only around stations. Set **Buffer distance** to 150 m for the immediate corridor.\n\nLine features are compile-time, so the panel marks the change as a rebuild. The track shapes of lines that share the Loop are de-duplicated so each segment counts once.',
      camera: {longitude: -87.68, latitude: 41.89, zoom: 11},
      options: {features: 'lines', distance: 150, colorBy: 'line'}
    },
    {
      id: 'one-line',
      controls: ['line', 'colorBy'],
      readouts: ['composition'],
      title: 'One line, no recompile',
      body: 'Pick the **Red Line** in **Which L line?** The feature buffers are rewritten and the other segments are parked far outside the data, so the graph is not rebuilt and **Rebuilds** in **Under the hood** stays put. Colouring by the nearest line (**Colour selected points by**) uses the nearest-feature id output.\n\nCompare the over-represented categories of the Red Line with the Blue Line and the Green Line.',
      camera: {longitude: -87.64, latitude: 41.85, zoom: 10.5},
      options: {features: 'lines', line: 'Red', distance: 300, colorBy: 'line'}
    },
    {
      id: 'places-baseline',
      controls: ['points', 'features'],
      readouts: ['selected', 'composition'],
      title: 'Compare with where people go',
      body: 'Observations near stations may simply follow people, not wildlife: observers may log what they see on the way to and from the train. Set **Select which points?** to *Places* instead: Overture places are a proxy for where people are. The same 400 m buffer around the stations captures a different share, so the observation share near stations has to be read against it. Compare which groups are over-represented in the buffer.\n\nSwitching the point layer rebuilds the graph over a new buffer, and each graph keeps its own outputs. The *Traffic crashes* choice asks the same question for road safety.',
      camera: {longitude: -87.66, latitude: 41.87, zoom: 10.2},
      options: {
        points: 'places',
        features: 'stations',
        line: 'all',
        distance: 400,
        colorBy: 'distance'
      }
    },
    {
      id: 'bus-and-limits',
      controls: ['features', 'spatialSort', 'measure'],
      readouts: ['timeSort'],
      title: 'Bus stops, sorting, and what the map cannot say',
      body: 'With **10,466 bus stops** the feature BVH is large enough for the Hilbert order to matter. Press **Time Hilbert sort on vs off**: the selection is identical, only the traversal cost changes. Bus buffers are limited to 500 m by the candidate capacity.\n\n**Limits.** Distances are straight lines, not walking routes across rivers and rail yards. A buffer says nothing about causes: stations sit where people gather, and observations follow observers. Try *Traffic crashes* under **Select which points?** around the track, or the Green Line against the Pink Line (**Which L line?**).',
      options: {
        points: 'observations',
        features: 'bus',
        distance: 250,
        colorBy: 'distance',
        spatialSort: true
      }
    }
  ]
});
