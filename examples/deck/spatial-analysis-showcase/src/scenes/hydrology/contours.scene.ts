// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import type {ContoursOptions} from './contours.compute';

/**
 * Contours of the Grand Canyon: metadata, options and narrative (light); the GPU work lives in
 * `contours.compute.ts`.
 */
export default defineScene<ContoursOptions>({
  id: 'contours',
  title: 'Contours, bands and rings of the Canyon',
  chapter: 'hydrology',
  order: 4,
  summary:
    'Marching-squares contour lines, filled elevation bands and closed shell-and-hole polygon rings of the Grand Canyon, extracted on the GPU from elevation or slope and re-levelled by a slider.',
  contributors: ['GPUIsolines', 'GPUIsobands', 'GPUIsobandRings', 'GPUTerrainDerivatives'],
  datasets: [{id: 'grand-canyon-dem', role: 'elevation and slope on a 31 m grid'}],
  initialView: {longitude: -112.1, latitude: 36.1, zoom: 11.2},

  options: [
    {
      kind: 'select',
      id: 'source',
      label: 'Raster to contour',
      group: 'Levels',
      apply: 'param',
      default: 'elevation',
      help: 'Elevation, or slope in degrees from GPUTerrainDerivatives. Choosing is a buffer copy into the raster the contributors read: no recompile.',
      options: [
        {value: 'elevation', label: 'Elevation (m)'},
        {value: 'slope', label: 'Slope (degrees)'}
      ]
    },
    {
      kind: 'slider',
      id: 'elevationInterval',
      label: 'Elevation interval',
      group: 'Levels',
      apply: 'param',
      min: 50,
      max: 500,
      step: 10,
      default: 100,
      unit: 'm',
      disabledWhen: state => state.source !== 'elevation',
      help: 'Vertical distance between contour levels. The levels are written into the levels buffer, so the contours re-extract on the next frame without a recompile. The graph holds at most 64 levels.'
    },
    {
      kind: 'slider',
      id: 'slopeInterval',
      label: 'Slope interval',
      group: 'Levels',
      apply: 'param',
      min: 2,
      max: 20,
      step: 1,
      default: 5,
      unit: '°',
      disabledWhen: state => state.source !== 'slope',
      help: 'Degrees between slope levels.'
    },
    {
      kind: 'range',
      id: 'elevationWindow',
      label: 'Elevation window',
      group: 'Levels',
      apply: 'param',
      min: 680,
      max: 2640,
      step: 10,
      default: [680, 2640],
      unit: 'm',
      disabledWhen: state => state.source !== 'elevation',
      help: 'Only the bands inside this range are emitted as filled geometry (firstBand and lastBand of GPUIsobands). Lines and rings keep every level.'
    },
    {
      kind: 'range',
      id: 'slopeWindow',
      label: 'Slope window',
      group: 'Levels',
      apply: 'param',
      min: 0,
      max: 85,
      step: 1,
      default: [0, 85],
      unit: '°',
      disabledWhen: state => state.source !== 'slope',
      help: 'Only slope bands inside this range are filled. Set it to 30 to 85 to isolate cliffs.'
    },
    {
      kind: 'toggle',
      id: 'showRelief',
      label: 'Shaded relief',
      group: 'Layers',
      apply: 'param',
      default: true,
      help: 'Hillshade of the DEM under the vector layers.'
    },
    {
      kind: 'toggle',
      id: 'showBands',
      label: 'Filled bands (GPUIsobands)',
      group: 'Layers',
      apply: 'param',
      default: true,
      help: 'Triangles of each band, colored by band index through the palette, drawn from the GPU triangle buffer with an indirect vertex count.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Band palette',
      group: 'Layers',
      apply: 'param',
      default: 'cividis',
      help: 'Ramp sampled by band index. Written into a 256-entry palette buffer.',
      options: [
        {value: 'cividis', label: 'Cividis'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'grayscale', label: 'Grayscale'}
      ]
    },
    {
      kind: 'slider',
      id: 'bandOpacity',
      label: 'Band opacity',
      group: 'Layers',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.6,
      format: value => `${Math.round(value * 100)} %`
    },
    {
      kind: 'toggle',
      id: 'showLines',
      label: 'Contour lines (GPUIsolines)',
      group: 'Layers',
      apply: 'param',
      default: true,
      help: 'Marching-squares segments between samples, drawn as screen-space-width lines.'
    },
    {
      kind: 'toggle',
      id: 'stitchLines',
      label: 'Stitch into polylines',
      group: 'Layers',
      apply: 'compile',
      default: false,
      help: 'Chains the segments into polylines on the GPU (pointer jumping in about log2 of the capacity rounds). Compile-time: this switches to a second graph that has the polyline outputs.'
    },
    {
      kind: 'slider',
      id: 'indexEvery',
      label: 'Index contour every',
      group: 'Layers',
      apply: 'param',
      min: 0,
      max: 10,
      step: 1,
      default: 5,
      format: value => (value === 0 ? 'none' : `${value}th level`),
      help: 'Draws every nth level in a heavier color, as on a topographic map. A per-level style table.'
    },
    {
      kind: 'slider',
      id: 'lineWidth',
      label: 'Line width',
      group: 'Layers',
      apply: 'param',
      min: 0.6,
      max: 3,
      step: 0.1,
      default: 1.1,
      unit: 'px'
    },
    {
      kind: 'toggle',
      id: 'showRings',
      label: 'Band rings (GPUIsobandRings)',
      group: 'Rings',
      apply: 'compile',
      default: false,
      help: 'Chains the band boundary edges into closed polygon rings, white for shells (counter-clockwise) and magenta for holes (clockwise). Runs as a third compiled graph.'
    },
    {
      kind: 'select',
      id: 'ringTolerance',
      label: 'Vertex tolerance',
      group: 'Rings',
      apply: 'compile',
      default: 'fine',
      disabledWhen: state => !state.showRings,
      help: 'Distance below which ring vertices count as the same point (vertexTolerance). Boundary vertices are bit-identical, so a tiny tolerance is exact; a tolerance near the cell size starts to merge distinct vertices. Compile-time.',
      options: [
        {value: 'fine', label: '0.0001 m (exact)'},
        {value: 'half-cell', label: '15 m (half a cell)'},
        {value: 'cell', label: '31 m (one cell)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'splitTouchingRings',
      label: 'Split rings that touch themselves',
      group: 'Rings',
      apply: 'compile',
      default: true,
      disabledWhen: state => !state.showRings,
      help: 'A band that pinches to a single vertex at a saddle becomes two rings (on) or one self-touching ring (off). Compile-time.'
    },
    {
      kind: 'button',
      id: 'time',
      label: 'Time the three graphs',
      group: 'Under the hood',
      help: 'GPU time of lines + bands, the stitched variant and the rings graph.'
    }
  ],

  story: [
    {
      id: 'question',
      title: 'Where do the Canyon’s cliffs and benches sit?',
      body: 'A topographic map answers that with one trick: draw a line wherever the ground crosses a chosen height. Where lines crowd together, the ground is steep; where they space out, it is a bench. The Grand Canyon’s layered rock makes this unusually legible: wide Tonto Platform benches, Redwall cliffs, side canyons with their own tiny staircases.\n\nHere three GPU contributors do it on a 31 m raster of the 15 m DEM: `GPUIsolines` draws the **lines**, `GPUIsobands` the **filled bands** between them, and `GPUIsobandRings` the **polygons** of those bands. Bands run from blue (low, the river) to yellow (the rims); every fifth line is an *index contour*.',
      camera: {longitude: -112.1, latitude: 36.1, zoom: 11.2, transitionMs: 1400},
      controls: ['source', 'elevationInterval', 'indexEvery'],
      readouts: ['levels'],
      highlight: {readout: 'levels'}
    },
    {
      id: 'lines',
      title: 'Marching squares: GPUIsolines',
      body: '`GPUIsolines` walks every cell of the raster once. A corner is *high* when its value is at least the level; a cell with both high and low corners is crossed by the contour, and the crossing points on its edges are found by linear interpolation. When opposite corners agree and the other two do not (a *saddle*) the cell-centre average decides which way the line goes. The segments come out ordered by cell, level and slot, so the result is deterministic.\n\nBands are off now so only lines remain, at 50 m. Drag **Elevation interval** below: the levels are a buffer write and the lines re-extract on the next frame, no recompile. Zoom to the Bright Angel Trail: the lines bunch into the Redwall cliff and spread over the Tonto bench.',
      camera: {longitude: -112.125, latitude: 36.07, zoom: 12.7, transitionMs: 1800},
      options: {showBands: false, elevationInterval: 50, showRelief: true},
      controls: ['elevationInterval', 'lineWidth'],
      readouts: ['segments'],
      highlight: {readout: 'segments'}
    },
    {
      id: 'polylines',
      title: 'From segments to polylines',
      body: 'A million loose segments are fine to draw but useless to export or label. **Stitch into polylines** adds the second output of `GPUIsolines`: the segments are linked end to end on the GPU by pointer jumping (about log2 of the capacity rounds), then compacted into polylines with their level and a *closed* flag, plus offsets that index the shared vertex buffer.\n\nTurn on **Stitch into polylines** below. It is a compile-time choice, so the control is marked *rebuild*: it selects a second compiled graph that has the polyline buffers. Watch the readouts: *Polylines* counts lines and vertices; every closed contour around a butte becomes one line.',
      options: {showBands: false, elevationInterval: 50, stitchLines: true},
      controls: ['stitchLines'],
      readouts: ['polylines'],
      highlight: {readout: 'polylines'}
    },
    {
      id: 'bands',
      title: 'Fill between the lines: GPUIsobands',
      body: '`GPUIsobands` produces the **filled bands** between consecutive levels as triangles, by walking each cell’s boundary combinatorially (no floating-point polygon clipping), so a band’s edge matches the contour line of the same level **bit for bit**: no slivers, no gaps. A cell contributes at most two convex pieces per band. The triangles go to a GPU buffer and the layer draws them with a vertex count the GPU wrote.\n\nThe color of a band is its **index** sampled through the palette, so the legend is a ramp across the elevation range. Switch **Band palette** below, or change **Band opacity**; the 100 m default gives about 20 bands.',
      options: {showBands: true, elevationInterval: 100, stitchLines: false},
      camera: {longitude: -112.1, latitude: 36.1, zoom: 11.2, transitionMs: 1400},
      controls: ['ramp', 'bandOpacity', 'elevationInterval'],
      readouts: ['triangles'],
      highlight: {readout: 'triangles'}
    },
    {
      id: 'window',
      title: 'Isolate a layer of the Canyon',
      body: 'The **Elevation window** below keeps only the bands between two heights: here 1,000 to 1,600 m, roughly the Tonto Platform bench (Indian Garden is at about 1,150 m) and the Redwall cliff above it. It maps the heights to band numbers and writes `firstBand` and `lastBand` into the band parameters, so the triangles of every other band are not even emitted. Lines and rings are unaffected.\n\nThis is how you cut a *stratum* out of a terrain model: change the window and the geometry follows within a frame. Notice that the triangle readout drops with the window.',
      options: {elevationWindow: [1000, 1600], showLines: true, ramp: 'viridis'},
      controls: ['elevationWindow'],
      readouts: ['bandWindow', 'triangles'],
      highlight: {readout: 'bandWindow'}
    },
    {
      id: 'slope',
      title: 'Contour the slope: where are the cliffs?',
      body: 'The same three contributors can contour *any* raster. **Raster to contour** is now set to slope: the values are degrees from `GPUTerrainDerivatives` (steepest-descent from a 3 x 3 window), contoured every 5 degrees. The narrow, dark bands in the high end are cliffs; the broad low bands are benches and the rim plateaus. It is the picture a trail planner wants: the cliff limit of the least-cost scene is just the 38 degree line here.\n\nTry the **Slope window** at 35 to 85 degrees to isolate the walls, and a coarser **Slope interval** for a cleaner map.',
      options: {
        source: 'slope',
        slopeInterval: 5,
        ramp: 'inferno',
        showBands: true,
        elevationWindow: [680, 2640]
      },
      camera: {longitude: -112.1, latitude: 36.1, zoom: 11.4, transitionMs: 1400},
      controls: ['source', 'slopeInterval', 'slopeWindow'],
      readouts: ['domain'],
      highlight: {readout: 'domain'}
    },
    {
      id: 'rings',
      title: 'Polygons you can hand off: GPUIsobandRings, and limits',
      body: 'Triangles draw well but cannot be exported. `GPUIsobandRings` chains each band’s boundary edges into **closed rings**: counter-clockwise shells (white) and clockwise holes (magenta), with each hole attached to the shell it sits in and every ring tagged with its band. The readouts count them and report the boundary edges and open segments (should be 0). Change **Vertex tolerance** and **Split rings that touch themselves** below to see what they do to the counts; both are compile-time.\n\nLimits: the DEM is Terrarium (0.5 m vertical steps) resampled to 31 m, so cliffs under 31 m wide vanish; the buffers have fixed capacities (the readouts show use against capacity and flag overflow); the contours match marching squares and GDAL’s contour tools in method, not in output order. Try **Elevation interval** at 50 m with rings on.',
      options: {
        source: 'elevation',
        showRings: true,
        showLines: true,
        showBands: true,
        elevationInterval: 200,
        ramp: 'cividis'
      },
      camera: {longitude: -112.09, latitude: 36.105, zoom: 12.1, transitionMs: 1600},
      controls: ['showRings', 'ringTolerance', 'splitTouchingRings', 'elevationInterval'],
      readouts: ['rings', 'ringEdges'],
      highlight: {readout: 'rings'}
    }
  ],

  about: {
    what: 'Marching-squares lines (`GPUIsolines`), filled bands (`GPUIsobands`) and closed polygon rings (`GPUIsobandRings`) of one raster, with the levels, band window and palette all in parameter buffers.',
    why: 'Contours are the oldest terrain visualisation and still the clearest: they give slope, aspect and landform at a glance, and as polygons they are the basis of hypsometric tints, cut-and-fill volumes, viewshed masks and exported map layers.',
    howToRead:
      'Lines of equal height (or slope); heavier index lines every few levels. Colors give the band, low to high. White rings are band outlines (shells), magenta rings are holes inside a band.'
  },

  legends: state => {
    const legends: LegendSpec[] = [];
    const elevation = state.source === 'elevation';
    if (state.showBands) {
      legends.push({
        kind: 'ramp',
        title: elevation ? 'Elevation band' : 'Slope band',
        ramp: state.ramp,
        extent: elevation ? [688, 2626] : [0, 85],
        unit: elevation ? 'm' : '°',
        format: value => `${Math.round(value).toLocaleString('en-US')}`
      });
    }
    const entries: {color: readonly [number, number, number, number]; label: string}[] = [
      {
        color: [40, 40, 50, 255],
        label: `Contour every ${elevation ? state.elevationInterval : state.slopeInterval} ${elevation ? 'm' : '°'}`
      }
    ];
    if (state.indexEvery > 0) {
      entries.push({
        color: [10, 10, 20, 255],
        label: `Index contour every ${state.indexEvery}th level`
      });
    }
    if (state.showLines) legends.push({kind: 'categories', title: 'Lines', entries});
    if (state.showRings) {
      legends.push({
        kind: 'categories',
        title: 'Band rings',
        entries: [
          {color: [255, 255, 255, 235], label: 'Shell (counter-clockwise)'},
          {color: [255, 60, 200, 235], label: 'Hole (clockwise)'}
        ]
      });
    }
    return legends;
  },

  readouts: [
    {
      id: 'grid',
      label: 'Raster',
      help: 'Contoured raster: the 15 m DEM averaged over 2 x 2 blocks.'
    },
    {
      id: 'domain',
      label: 'Value range',
      help: 'Elevation range of the window and the slope domain used for slope levels.'
    },
    {
      id: 'levels',
      label: 'Levels',
      help: 'Number of active levels (compile-time limit 64) and their interval.'
    },
    {
      id: 'bandWindow',
      label: 'Band window',
      help: 'firstBand and lastBand written to the isoband parameters.'
    },
    {
      id: 'segments',
      label: 'Contour segments',
      help: 'Marching-squares segments against the compile-time capacity; OVERFLOW means the capacity was too small.'
    },
    {
      id: 'polylines',
      label: 'Polylines',
      help: 'Stitched polylines and their vertices (compile-time switch).'
    },
    {
      id: 'triangles',
      label: 'Band triangles',
      help: 'Triangles emitted for the drawn bands against capacity.'
    },
    {id: 'rings', label: 'Band rings', help: 'Shells and holes of GPUIsobandRings.'},
    {
      id: 'ringEdges',
      label: 'Ring boundary edges',
      help: 'Boundary edges before ring assembly, and edges that stayed open.'
    },
    {id: 'encode', label: 'Last encode', help: 'CPU time to encode the analysis graph.'},
    {
      id: 'timings',
      label: 'GPU timings',
      help: 'Press the "Time the three graphs" button to fill this in.'
    }
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUIsolines, GPUIsobands, GPUIsobandRings,
  getGPUIsolinesParameterValues, getGPUIsobandsParameterValues
} from '@luma.gl/experimental/gpu-raster';

const graph = new GPUCommandGraph(device, {id: 'contours'});
graph.add(new GPUIsobands({
  width, height, values, breaks: levels, parameters: isobandParameters.importToGraph(graph),
  output: {triangles, triangleBands, count, overflow, vertexCount}
}));
graph.add(new GPUIsolines({
  width, height, values, levels, parameters: isolineParameters.importToGraph(graph),
  output: {segments, segmentLevels, count, overflow}${
    state.stitchLines
      ? `,
  polylines: {vertices, polylineOffsets, polylineLevels, polylineClosed, polylineCount, vertexCount, overflow}`
      : ''
  }
}));${
    state.showRings
      ? `
ringGraph.add(new GPUIsobandRings({
  width, height, values, breaks: levels, parameters: isobandParameters.importToGraph(ringGraph),
  edgeCapacity: 1_200_000, vertexTolerance: ${state.ringTolerance === 'fine' ? '1e-4' : state.ringTolerance === 'cell' ? '31' : '15'}, splitTouchingRings: ${state.splitTouchingRings},
  output: {ringOffsets, positions, ringGroups, ringIsHole, count, overflow}
}));`
      : ''
  }
const compiled = graph.compile();            // once

// Per change (no recompile): levels, band window and extent are buffer writes.
levelsBuffer.write(Float32Array.from(levels));   // multiples of ${state.source === 'elevation' ? state.elevationInterval : state.slopeInterval}
isolineParameters.write(getGPUIsolinesParameterValues({width, height, levelCount, extent}));
isobandParameters.write(getGPUIsobandsParameterValues({
  width, height, breakCount: levelCount, extent, firstBand, lastBand
}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  create: async ctx => (await import('./contours.compute')).createContours(ctx)
});
