// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import type {SummitsOptions} from './summits.compute';

/**
 * Summits, peak snapping, critical points, relative height and contours of the Matterhorn DEM.
 * The GPU work lives in `summits.compute.ts`.
 */
export default defineScene<SummitsOptions>({
  id: 'summits',
  title: 'Where are the summits?',
  chapter: 'terrain',
  order: 7,
  summary:
    'Find every summit of the Matterhorn region on the GPU, snap a sloppy peak catalogue onto them, classify peaks, saddles and pits, map relative height with a summed-area table and draw contours with GPU-written draw records.',
  contributors: [
    'GPUTerrainSummits',
    'GPUTerrainPeakSnap',
    'GPUTerrainCriticalPoints',
    'GPUTerrainContours',
    'GPUTerrainDerivatives'
  ],
  datasets: [{id: 'alps-dem', role: 'terrain (Terrarium, Web Mercator)'}],
  initialView: {longitude: 7.742, latitude: 45.985, zoom: 11.7},

  options: [
    {
      kind: 'slider',
      id: 'summitRadius',
      label: 'Summit radius',
      group: 'Summits',
      apply: 'param',
      min: 100,
      max: 800,
      step: 25,
      default: 400,
      unit: 'm',
      help: 'A cell is a summit if it is the highest cell within this ground distance. Per-frame; always ground metres, whatever the pixel size.'
    },
    {
      kind: 'slider',
      id: 'summitMinimumDrop',
      label: 'Minimum drop',
      group: 'Summits',
      apply: 'param',
      min: 0,
      max: 500,
      step: 10,
      default: 100,
      unit: 'm',
      help: 'The summit must stand at least this far above the highest cell on the edge (ring) of its disc. A radius-limited lower bound of topographic prominence.'
    },
    {
      kind: 'select',
      id: 'summitMaximumRadius',
      label: 'Search bound',
      group: 'Summits',
      apply: 'compile',
      default: '32',
      help: 'Compile-time loop bound in pixels per axis (26.6 m each). A larger radius is clamped to this and the "Radius" readout says so. Bigger bounds cost quadratically more.',
      options: [
        {value: '16', label: '16 pixels (425 m)'},
        {value: '24', label: '24 pixels (640 m)'},
        {value: '32', label: '32 pixels (850 m)'}
      ]
    },
    {
      kind: 'select',
      id: 'incompleteNeighborhood',
      label: 'Window edge',
      group: 'Summits',
      apply: 'compile',
      default: 'ignore',
      help: 'Reject: a cell whose disc leaves the grid is never a summit. Ignore: outside cells are simply absent, so the edge of the window can produce summits that are really flanks.',
      options: [
        {value: 'ignore', label: 'Ignore missing cells'},
        {value: 'reject', label: 'Reject incomplete discs'}
      ]
    },
    {
      kind: 'slider',
      id: 'catalogueError',
      label: 'Catalogue position error',
      group: 'Peak snap',
      apply: 'param',
      min: 0,
      max: 400,
      step: 20,
      default: 120,
      unit: 'm',
      help: 'Moves the eight catalogue peaks by this distance in a fixed random direction, like a sloppy OpenStreetMap node. Per-frame: it rewrites the candidate buffer.'
    },
    {
      kind: 'slider',
      id: 'snapRadius',
      label: 'Snap radius',
      group: 'Peak snap',
      apply: 'param',
      min: 50,
      max: 400,
      step: 10,
      default: 200,
      unit: 'm',
      help: 'The disc searched around each candidate for the highest cell. Used for candidates without their own radius.'
    },
    {
      kind: 'slider',
      id: 'snapMaximumMove',
      label: 'Largest accepted move',
      group: 'Peak snap',
      apply: 'param',
      min: 0,
      max: 600,
      step: 10,
      default: 300,
      unit: 'm',
      help: 'A snap longer than this is rejected and the candidate keeps its position (red).'
    },
    {
      kind: 'slider',
      id: 'snapMaximumHeightChange',
      label: 'Largest height change',
      group: 'Peak snap',
      apply: 'param',
      min: 0,
      max: 300,
      step: 5,
      default: 80,
      unit: 'm',
      help: 'If the snapped DEM height differs from the reference height by more than this, the snap is rejected (magenta): the catalogue and the DEM disagree about which summit this is.'
    },
    {
      kind: 'toggle',
      id: 'snapInterior',
      label: 'Keep flank points (interior rule)',
      group: 'Peak snap',
      apply: 'param',
      default: true,
      help: 'If the best cell lies on the ring of the disc, the ground is still climbing beyond it: a flank, not a summit. The candidate keeps its position (orange).'
    },
    {
      kind: 'toggle',
      id: 'snapCatalogueHeights',
      label: 'Use catalogue elevations',
      group: 'Peak snap',
      apply: 'param',
      default: true,
      help: 'Passes the published elevation as the reference height of each named peak (candidateHeights). Off sends NaN, so the DEM height at the candidate is the reference.'
    },
    {
      kind: 'toggle',
      id: 'snapDistanceRule',
      label: 'Distance-dependent radius',
      group: 'Peak snap',
      apply: 'param',
      default: false,
      help: 'Gives each candidate its own radius min(250 m, 60 m + 0.4 % of its distance from Gornergrat), the rule of the mt-image skyline renderer for far peaks (candidateRadii).'
    },
    {
      kind: 'button',
      id: 'clear',
      label: 'Clear clicked candidates',
      group: 'Peak snap',
      help: 'Removes the points you added by clicking the map and keeps the eight catalogue peaks.'
    },
    {
      kind: 'select',
      id: 'connectivity',
      label: 'Critical point ring',
      group: 'Critical points',
      apply: 'compile',
      default: '8',
      help: 'The neighbour ring read for the sign changes. 8 is Peucker and Douglas; 6 is the Freudenthal triangulation, whose counts obey the Euler relation.',
      options: [
        {value: '8', label: '8 neighbours (Peucker-Douglas)'},
        {value: '6', label: '6 neighbours (Freudenthal)'}
      ]
    },
    {
      kind: 'slider',
      id: 'devRadius',
      label: 'Relative height window',
      group: 'Relative height (summed-area table)',
      apply: 'param',
      min: 3,
      max: 64,
      step: 1,
      default: 20,
      unit: 'cells',
      help: 'Half-width of the square window in cells (26.6 m each) whose mean and standard deviation the height is compared with. The table is built once per run; the radius is a parameter.'
    },
    {
      kind: 'slider',
      id: 'contourInterval',
      label: 'Contour interval',
      group: 'Contours',
      apply: 'param',
      min: 100,
      max: 500,
      step: 50,
      default: 100,
      unit: 'm',
      help: 'Spacing of the 40 contour levels, starting at the first multiple above the lowest point. The level values are a per-frame buffer: no recompile.'
    },
    {
      kind: 'slider',
      id: 'contourIndexEvery',
      label: 'Index contour every',
      group: 'Contours',
      apply: 'param',
      min: 2,
      max: 10,
      step: 1,
      default: 5,
      help: 'Every nth level is drawn thicker and darker.'
    },
    {
      kind: 'select',
      id: 'base',
      label: 'Base layer',
      group: 'Display',
      apply: 'param',
      default: 'hillshade',
      help: 'Hillshade, the elevation, or the relative height from the summed-area table.',
      options: [
        {value: 'hillshade', label: 'Hillshade'},
        {value: 'relative-height', label: 'Relative height (z-score in a window)'},
        {value: 'elevation', label: 'Elevation'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showSummits',
      label: 'Summits',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Summit discs sized by drop.'
    },
    {
      kind: 'toggle',
      id: 'showSnap',
      label: 'Peak snap',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'White rings are the catalogue positions, filled discs the snapped result. Click the map to add your own candidate.'
    },
    {
      kind: 'toggle',
      id: 'showCritical',
      label: 'Critical points',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Peaks (red), saddles (yellow) and pits (blue) from the sign changes around each cell.'
    },
    {
      kind: 'toggle',
      id: 'showContours',
      label: 'Contours',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Marching-squares contours of the full resolution DEM.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Measure GPU cost',
      group: 'Compare',
      help: 'Runs each graph outside the frame and reports GPU time.'
    }
  ],

  readouts: [
    {id: 'grid', label: 'Grids'},
    {
      id: 'summits',
      label: 'Summits',
      help: 'Counted on the GPU into a compact list with a total and an overflow flag.'
    },
    {
      id: 'summitRadius',
      label: 'Radius',
      help: 'Whether the requested radius fits the compile-time search bound.'
    },
    {
      id: 'snap',
      label: 'Peak snap',
      help: 'Outcome of each of the candidates, from the status codes of the contributor.'
    },
    {
      id: 'snapMove',
      label: 'Catalogue moves',
      help: 'Ground distance the named catalogue peaks moved.'
    },
    {id: 'critical', label: 'Critical points'},
    {
      id: 'euler',
      label: 'Euler count',
      help: 'Peaks minus saddles (with multiplicity c/2 - 1) plus pits. It equals the Euler characteristic only for a surface without boundary; here the window edge and the unclassified boundary cells break it.'
    },
    {id: 'contours', label: 'Contours'},
    {id: 'timing', label: 'GPU time'}
  ],

  legends: state => {
    const legends: LegendSpec[] = [];
    if (state.showSummits) {
      legends.push({
        kind: 'categories',
        title: 'Summit, by drop within its ring',
        entries: [
          {color: [255, 226, 110, 235], label: 'Up to 200 m'},
          {color: [255, 176, 70, 240], label: '200 to 400 m'},
          {color: [255, 118, 60, 245], label: '400 to 700 m'},
          {color: [235, 50, 70, 250], label: 'More than 700 m'}
        ],
        note: 'Bigger discs mark bigger drops.'
      });
    }
    if (state.showSnap) {
      legends.push({
        kind: 'categories',
        title: 'Peak snap result (white ring: catalogue position)',
        entries: [
          {color: [70, 235, 130, 255], label: 'Snapped'},
          {color: [190, 200, 215, 255], label: 'Unchanged'},
          {color: [255, 160, 60, 255], label: 'On the ring: a flank'},
          {color: [235, 60, 70, 255], label: 'Move too far'},
          {color: [220, 90, 230, 255], label: 'Height change too large'}
        ]
      });
    }
    if (state.showCritical) {
      legends.push({
        kind: 'categories',
        title: 'Critical points',
        entries: [
          {color: [235, 60, 70, 235], label: 'Peak'},
          {color: [255, 214, 70, 235], label: 'Saddle'},
          {color: [70, 140, 255, 235], label: 'Pit'}
        ]
      });
    }
    if (state.base === 'relative-height') {
      legends.push({
        kind: 'ramp',
        title: 'Height compared with the window (standard deviations)',
        ramp: 'diverging',
        extent: [-3, 3],
        labels: ['-3 below', '+3 above'],
        format: value => value.toFixed(0)
      });
    } else if (state.base === 'elevation') {
      legends.push({
        kind: 'ramp',
        title: 'Elevation',
        ramp: 'cividis',
        extent: [1503, 4476],
        unit: 'm',
        format: value => value.toFixed(0)
      });
    }
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTerrainSummits, GPUTerrainPeakSnap, GPUTerrainCriticalPoints, GPUTerrainContours,
  getGPUTerrainSummitsParameterValues, getGPUTerrainPeakSnapParameterValues
} from '@luma.gl/experimental/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'summits'});
graph.add(new GPUTerrainSummits({
  width, height, elevation,
  cellSizeMode: 'web-mercator',                  // cellSize + northEdge / southEdge, per row
  maximumRadiusPixels: ${state.summitMaximumRadius},                    // compile-time loop bound
  incompleteNeighborhood: '${state.incompleteNeighborhood}',
  settings: summitSettings.importToGraph(graph),
  output: {ids, count, overflow, totalCount},      // compact summit list
  outputDrop                                       // drop of each listed summit
}));
graph.add(new GPUTerrainPeakSnap({
  width, height, elevation, cellSizeMode: 'web-mercator', maximumRadiusPixels: 16,
  candidates, candidateHeights, candidateRadii,    // catalogue positions, elevations, radii
  settings: snapSettings.importToGraph(graph),
  positions, heights, status, snapDistance, overflow
}));
graph.add(new GPUTerrainCriticalPoints({
  width, height, elevation, connectivity: ${state.connectivity}, classes, signChanges, counts
}));
const compiled = graph.compile();                // once

summitSettings.write(getGPUTerrainSummitsParameterValues({
  radius: ${state.summitRadius}, minimumDrop: ${state.summitMinimumDrop},
  cellSize: [mercatorCell, mercatorCell], northEdge, southEdge   // Web Mercator: equatorial metres
}));
snapSettings.write(getGPUTerrainPeakSnapParameterValues({
  radius: ${state.snapRadius}, maximumMove: ${state.snapMaximumMove}, maximumHeightChange: ${state.snapMaximumHeightChange},
  interior: ${state.snapInterior}, cellSize: [mercatorCell, mercatorCell], northEdge, southEdge
}));
compiled.encode(commandEncoder, {parameters: undefined}); // when a parameter changed`,

  about: {
    what: '`GPUTerrainSummits` marks the cells that are the highest point of a ground-metre disc and rise at least a minimum drop above its rim. `GPUTerrainPeakSnap` moves catalogue points onto those summits, `GPUTerrainCriticalPoints` classifies every cell as a peak, saddle or pit from the sign changes around it, and `GPUTerrainContours` extracts marching-squares contour segments with GPU-written draw records. A summed-area table (`addTerrainSummedAreaTableNodes`) gives the mean and spread of the height in any window in constant time.',
    why: 'Summits are the nodes of every mountain map: they are what a peak list, a panorama label or a summit register refers to. Finding them on the GPU, with a scale you choose, separates real peaks from every boulder on a glacier, and snapping puts catalogue peaks on the ground they describe.',
    howToRead:
      'Bigger, redder discs are bigger drops. White rings are catalogue positions and filled discs where they snapped to. Red, yellow and blue cells are peaks, saddles and pits. Blue relative-height cells are below their surroundings (valleys), red above (ridges and summits).'
  },

  create: async ctx => (await import('./summits.compute')).createSummits(ctx),

  story: [
    {
      id: 'the-question',
      controls: ['summitRadius', 'summitMinimumDrop'],
      readouts: ['summits'],
      title: 'How many summits does the Matterhorn region have?',
      body: 'A summit list sounds simple until you ask what counts: every bump of a glacier, or only the horns? **`GPUTerrainSummits`** answers on the GPU for every cell: a cell is a summit if it is the **highest point of a disc** of a ground radius you choose, and it stands at least a **minimum drop** above the highest cell on the **ring** (the cells on the disc edge).\n\nThe discs here are 400 m (**Summit radius**) and the drop 100 m (**Minimum drop**); both are sliders below. Hover a disc for its elevation and drop. The summits are found on the DEM averaged to 26.6 m cells, because summit finding depends on scale: at 6.6 m every boulder is a summit.',
      camera: {longitude: 7.742, latitude: 45.985, zoom: 11.7, transitionMs: 900},
      highlight: {readout: 'summits'}
    },
    {
      id: 'radius',
      controls: ['summitRadius', 'summitMaximumRadius'],
      readouts: ['summits', 'summitRadius'],
      title: 'The radius decides what a summit is',
      body: 'Shrink the **Summit radius** below to 150 m and the count explodes: every shoulder and moraine crest is the highest point of a small disc. Grow it to 800 m and only the dominant peaks survive. The radius is always ground metres (the contributor scales each row by the cosine of latitude from the Web Mercator edges), so the answer does not depend on the pixel size of your raster.\n\nThe **Search bound** is a compile-time loop limit: ask for more than it allows and the **Radius** readout says the request was clamped.',
      options: {summitRadius: 150},
      highlight: {readout: 'summits'}
    },
    {
      id: 'drop',
      controls: ['summitMinimumDrop', 'incompleteNeighborhood'],
      readouts: ['summits'],
      title: 'The drop is a prominence lower bound',
      body: 'Any path from a summit to the outside of its disc must cross the ring, so it descends at least the **drop**: a radius-limited lower bound on topographic prominence. Set the **Minimum drop** below to 250 m with the 400 m disc and the map reduces to the real horns: the Matterhorn, the Breithorn, Pollux and the Obergabelhorn group.\n\nTrue prominence needs a whole-DEM Priority-Flood sweep, which is out of scope for one disc kernel; there is no GDAL equivalent of this tool. Near the window edge (the Matterhorn is 350 m from it) a disc leaves the grid: **Window edge** set to *Ignore* treats the missing cells as absent, *Reject* refuses such a cell as a summit.',
      options: {summitRadius: 400, summitMinimumDrop: 250},
      callout: {coordinate: [7.6586, 45.9766], text: 'Matterhorn'}
    },
    {
      id: 'peak-snap',
      controls: ['showSnap', 'catalogueError', 'snapMaximumHeightChange', 'clear'],
      readouts: ['snap', 'snapMove'],
      title: 'Snap a sloppy catalogue onto the summits',
      body: '**`GPUTerrainPeakSnap`** moves catalogue points onto the DEM. Each candidate searches a **metric disc** for the highest cell, rejects a move that is too far or changes the height too much, and keeps flank points whose best cell lies on the disc ring. The eight named peaks here are shifted about 120 m in random directions, like a hand-placed OpenStreetMap node.\n\nSwitch **Peak snap** on below: white rings are the catalogue positions, discs where they snapped. Slide the **Catalogue position error** to 400 m to see rejected moves turn red, tighten the **Largest height change** to see magenta, and click the map to add your own candidate (**Clear clicked candidates** removes them). A square search window reaches 1.41 times its radius at the corners; the disc does not.',
      options: {showSnap: true, showSummits: false, summitMinimumDrop: 100},
      camera: {longitude: 7.74, latitude: 45.975, zoom: 12.1, transitionMs: 1200},
      highlight: {readout: 'snapMove'}
    },
    {
      id: 'critical-points',
      controls: ['showCritical', 'connectivity'],
      readouts: ['critical', 'euler'],
      title: 'Peaks, saddles and pits from sign changes',
      body: '**`GPUTerrainCriticalPoints`** walks the ring of neighbours of every cell and counts the **sign changes** of the height difference (higher or lower than the centre, ties broken by index so plateaus resolve deterministically). Zero changes with all neighbours lower is a **peak**, all higher a **pit**, two changes a regular slope, four or more a **saddle** (a pass) of multiplicity `c/2 - 1`.\n\nThe **Critical points** layer is on. Compare the 8-neighbour ring (Peucker and Douglas 1975) with the 6-neighbour **Freudenthal** ring using **Critical point ring** below: the 8-ring is not a consistent triangulation and can report extra saddles at diagonal ambiguities, the 6-ring is the piecewise-linear one whose counts obey the Euler relation.',
      options: {showSnap: false, showSummits: false, showCritical: true},
      camera: {longitude: 7.742, latitude: 45.985, zoom: 11.7, transitionMs: 1200},
      highlight: {readout: 'euler'}
    },
    {
      id: 'relative-height',
      controls: ['base', 'devRadius'],
      readouts: ['timing'],
      title: 'Relative height from a summed-area table',
      body: 'A summit stands out from its surroundings. `addTerrainSummedAreaTableNodes` builds an exact **summed-area table** of elevation, its square and a valid-cell count (a row scan, a transpose and a column scan, in modular 64-bit integers), so the mean and variance of any square window are four table reads. The kernel here turns that into the **z-score** `(z - mean) / sd` of every cell in a window whose radius is a per-frame parameter.\n\nRed cells are higher than their window, blue lower: summits and ridges against valleys and glaciers. Slide the **Relative height window** below from 6 to 60 cells and the same table gives the picture at every scale. This is the machinery behind `GPUTerrainTopographicPosition`.',
      options: {showCritical: false, showSummits: true, base: 'relative-height', devRadius: 20},
      highlight: {readout: 'timing'}
    },
    {
      id: 'contours-and-limits',
      controls: ['contourInterval', 'incompleteNeighborhood', 'contourIndexEvery', 'measure'],
      readouts: ['contours', 'timing'],
      title: 'Contours, limits, and things to try',
      body: '**`GPUTerrainContours`** extracts marching-squares segments for 40 level slots on the full 2048 × 2048 DEM. The levels are a per-frame buffer, and each level also rewrites a **16-byte draw record** on the GPU (`[6, segments, 0, 0]`), so the segment layer draws exactly the segments found with no readback. **Contour interval** changes the levels without a recompile.\n\nLimits: the DEM is 13.6 km wide, so peaks at the window edge (the Liskamm flank in the south-east corner) are cut; the summit scale is a choice, not a fact; and drops are lower bounds. **Try:** set **Window edge** to *Reject* and watch the Matterhorn vanish; use a 100 m **Contour interval** with **Index contour every** 5; press **Measure GPU cost**.',
      options: {showContours: true, base: 'hillshade', contourInterval: 100}
    }
  ]
});
