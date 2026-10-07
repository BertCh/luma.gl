// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import {
  ANALYSIS_CELL_METERS,
  getSummitsLegends,
  SCALE_PRESETS,
  SUMMIT_FRAMES,
  type SummitsOptions
} from './summits-style';
import {getTerrainFurniture, terrainCartouche} from './terrain-furniture';
import {terrainLabel, terrainLabels} from './terrain-places';

/** The title cartouche of a step: its question, with a subtitle the scene replaces live. */
const furnitureFor = (question: string, subtitle: string) =>
  getTerrainFurniture({cartouche: terrainCartouche(question, subtitle)});

/**
 * Summits of the Matterhorn region: which bump is a mountain? A summit is a definition (the
 * highest ground in a disc, a minimum drop above its ring), and the list changes with it. The GPU
 * work lives in `summits.compute.ts`.
 */
export default defineScene<SummitsOptions>({
  id: 'summits',
  title: 'Which bump is a mountain?',
  chapter: 'terrain',
  order: 4,
  summary:
    'Find every summit of the Matterhorn region on the GPU as the highest ground in a disc with a minimum drop above its ring, watch the list change with the radius, the drop and the window edge, snap a real OpenStreetMap peak catalogue onto the summits, and draw critical points and contours.',
  contributors: [
    'GPUTerrainSummits',
    'GPUTerrainPeakSnap',
    'GPUTerrainCriticalPoints',
    'GPUTerrainContours'
  ],
  datasets: [
    {id: 'alps-dem', role: 'terrain (Terrarium, Web Mercator)'},
    {id: 'alps-context', role: 'glaciers and the OpenStreetMap peak catalogue'}
  ],
  initialView: {...SUMMIT_FRAMES.home},
  basemap: ground('relief'),
  furniture: furnitureFor('Which bump is a mountain?', 'Highest within the radius, above its ring'),

  options: [
    {
      kind: 'preset',
      id: 'scale',
      label: 'Scale',
      group: 'Summit definition',
      help: 'Three ways to ask the same question: a boulder (small disc, tiny drop), a horn (the default) and a massif (large disc, large drop). Each chip writes both the radius and the drop.',
      presets: SCALE_PRESETS.map(preset => ({
        label: `${preset.label} ${preset.radius} m`,
        values: {summitRadius: preset.radius, summitMinimumDrop: preset.drop}
      }))
    },
    {
      kind: 'slider',
      id: 'summitRadius',
      label: 'Summit radius',
      group: 'Summit definition',
      apply: 'param',
      min: 100,
      max: 800,
      step: 25,
      default: 400,
      unit: 'm',
      marks: [
        {value: 150, label: 'boulder'},
        {value: 400, label: 'horn'},
        {value: 800, label: 'massif'}
      ],
      describe: value =>
        `${Math.round(value / ANALYSIS_CELL_METERS)} cells from the centre to the rim`,
      help: 'A cell is a summit if it is the highest cell within this ground distance. A per-frame parameter, always in ground metres whatever the pixel size.'
    },
    {
      kind: 'slider',
      id: 'summitMinimumDrop',
      label: 'Minimum drop',
      group: 'Summit definition',
      apply: 'param',
      min: 0,
      max: 500,
      step: 10,
      default: 100,
      unit: 'm',
      help: 'The summit must stand at least this far above the highest cell on the rim (ring) of its disc: a radius-limited lower bound of topographic prominence. The GPU lists every disc maximum with its drop; this threshold is applied to that list, so moving it costs no GPU work.'
    },
    {
      kind: 'select',
      id: 'incompleteNeighborhood',
      label: 'Window edge',
      group: 'Summit definition',
      apply: 'compile',
      display: 'segmented',
      default: 'ignore',
      help: 'Ignore: cells beyond the DEM are simply absent, so the edge of the window can produce summits that are really flanks. Reject: a cell whose disc leaves the grid is never a summit. A compile option: each choice is its own compiled graph.',
      options: [
        {value: 'ignore', label: 'Ignore'},
        {value: 'reject', label: 'Reject'}
      ]
    },
    {
      kind: 'select',
      id: 'summitMaximumRadius',
      label: 'Search bound',
      group: 'Engine',
      apply: 'compile',
      expert: true,
      default: '32',
      help: 'Compile-time loop bound in cells per axis. A larger radius is clamped to this, and the radius readout under the hood says so. A bigger bound costs quadratically more.',
      options: [
        {value: '16', label: '16 cells (425 m)'},
        {value: '24', label: '24 cells (640 m)'},
        {value: '32', label: '32 cells (850 m)'}
      ]
    },
    {
      kind: 'slider',
      id: 'catalogueError',
      label: 'Catalogue position error',
      group: 'Peak snap',
      apply: 'param',
      min: 0,
      max: 200,
      step: 10,
      default: 0,
      unit: 'm',
      help: 'Moves every catalogue peak by this distance in a fixed random direction, like a hand-placed OpenStreetMap node that is a little off. Per-frame: it rewrites the candidate buffer.'
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
      help: 'If the snapped DEM height differs from the published height by more than this, the snap is rejected (pink): the catalogue and the DEM disagree about which summit this is.'
    },
    {
      kind: 'slider',
      id: 'snapRadius',
      label: 'Snap radius',
      group: 'Peak snap',
      apply: 'param',
      expert: true,
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
      expert: true,
      min: 0,
      max: 600,
      step: 10,
      default: 300,
      unit: 'm',
      help: 'A snap longer than this is rejected and the candidate keeps its position (vermilion).'
    },
    {
      kind: 'toggle',
      id: 'snapInterior',
      label: 'Keep flank points (interior rule)',
      group: 'Peak snap',
      apply: 'param',
      expert: true,
      default: true,
      help: 'If the best cell lies on the ring of the disc, the ground is still climbing beyond it: a flank, not a summit. The candidate keeps its position (orange).'
    },
    {
      kind: 'toggle',
      id: 'snapCatalogueHeights',
      label: 'Use catalogue elevations',
      group: 'Peak snap',
      apply: 'param',
      expert: true,
      default: true,
      help: 'Passes the published elevation as the reference height of each peak (candidateHeights). Off sends NaN, so the DEM height at the candidate is the reference.'
    },
    {
      kind: 'toggle',
      id: 'snapDistanceRule',
      label: 'Distance-dependent radius',
      group: 'Peak snap',
      apply: 'param',
      expert: true,
      default: false,
      help: 'Gives each candidate its own radius, min(250 m, 60 m + 0.4 % of its distance from Gornergrat): the rule of the mt-image skyline renderer for far peaks (candidateRadii).'
    },
    {
      kind: 'select',
      id: 'connectivity',
      label: 'Critical point ring',
      group: 'Critical points',
      apply: 'compile',
      display: 'segmented',
      default: '8',
      help: 'The neighbour ring read for the sign changes. Eight neighbours is Peucker and Douglas; six is the Freudenthal triangulation, whose counts obey the Euler relation.',
      options: [
        {value: '8', label: '8 neighbours'},
        {value: '6', label: '6 neighbours'}
      ]
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
      help: 'Spacing of the 40 contour levels, starting at the DEM minimum rounded down to this interval. The level values are a per-frame buffer: no recompile.'
    },
    {
      kind: 'slider',
      id: 'contourIndexEvery',
      label: 'Index contour every',
      group: 'Contours',
      apply: 'param',
      expert: true,
      min: 2,
      max: 10,
      step: 1,
      default: 5,
      help: 'Every nth level is drawn heavier, as the index contours of a topographic map are.'
    },
    {
      kind: 'toggle',
      id: 'showSummits',
      label: 'Summit triangles',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Summits as ink triangles, sized by drop. Size is the only thing the triangles encode.'
    },
    {
      kind: 'toggle',
      id: 'showRejected',
      label: 'Failed candidates',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Hollow triangles mark disc maxima that fail the drop test, so the filter is visible.'
    },
    {
      kind: 'toggle',
      id: 'showProbe',
      label: 'Disc probe on hover',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Hover any cell to see its disc outlined in true ground metres and the same test the GPU runs: is it the highest cell in the disc, and what is the highest cell on the ring?'
    },
    {
      kind: 'toggle',
      id: 'showEdge',
      label: 'Incomplete-disc band',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Hatches the band along the edge of the DEM where a disc of this radius leaves the data, and frames the edge of the data.'
    },
    {
      kind: 'toggle',
      id: 'showSnap',
      label: 'Peak snap',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Paper rings are the catalogue positions, filled circles the cells they snapped to, coloured by outcome.'
    },
    {
      kind: 'toggle',
      id: 'showCritical',
      label: 'Critical points',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Peaks (ink triangles), saddles (orange diamonds) and pits (blue rings) from the sign changes around each cell. Drawn from zoom 12.5 on.'
    },
    {
      kind: 'toggle',
      id: 'showContours',
      label: 'Contours',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Marching-squares contours of the full-resolution DEM, brown on rock, with every fifth heavier.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Measure GPU cost',
      group: 'Under the hood',
      expert: true,
      help: 'Runs each graph outside the frame and reports GPU time.'
    }
  ],

  readouts: [
    {
      id: 'summitCount',
      label: 'Summits',
      unit: 'summits',
      format: 'integer',
      emphasis: 'tile',
      help: 'Disc maxima whose drop to the ring passes the minimum drop.'
    },
    {
      id: 'rejectedCount',
      label: 'Failed the drop test',
      unit: 'candidates',
      format: 'integer',
      emphasis: 'tile',
      help: 'Cells that are the highest in their disc but stand less than the minimum drop above its ring.'
    },
    {
      id: 'edgeLost',
      label: 'Lost under Reject',
      unit: 'summits',
      format: 'integer',
      emphasis: 'tile',
      help: 'Summits whose disc leaves the DEM: Reject refuses them, Ignore keeps them.'
    },
    {
      id: 'countByRadius',
      label: 'Summits by radius',
      kind: 'chart',
      help: 'The summit kernel run at six radii as parameter writes, at the current minimum drop and window-edge rule.'
    },
    {
      id: 'dropHistogram',
      label: 'Drop of every disc maximum',
      kind: 'chart',
      help: 'Candidates left of the marked minimum drop fail the test. Click or drag the chart to set the drop.'
    },
    {
      id: 'snapOutcomes',
      label: 'Snap outcomes',
      kind: 'chart',
      help: 'Catalogue peaks by what the snap did, in the map colours.'
    },
    {
      id: 'medianMove',
      label: 'Median snap distance',
      format: 'meters',
      emphasis: 'tile',
      help: 'Median ground distance the snapped catalogue peaks moved.'
    },
    {
      id: 'catalogueSize',
      label: 'Catalogue peaks',
      unit: 'peaks',
      format: 'integer',
      emphasis: 'tile',
      help: 'Named OpenStreetMap peaks inside the tile.'
    },
    {
      id: 'criticalCounts',
      label: 'Peaks, saddles and pits',
      help: 'Cells classified by the sign changes around them, on the analysis grid.'
    },
    {
      id: 'euler',
      label: 'Euler count',
      hood: true,
      help: 'Peaks minus saddles (with multiplicity c/2 - 1) plus pits. It equals the Euler characteristic only for a surface without boundary; here the window edge breaks it.'
    },
    {id: 'contourSummary', label: 'Contours', hood: true},
    {id: 'timing', label: 'GPU time', hood: true},
    {id: 'grid', label: 'Grids', hood: true},
    {
      id: 'listUse',
      label: 'Candidate list',
      hood: true,
      help: 'The GPU writes every disc maximum (minimum drop 0) into a compact list with a total and an overflow flag; the drop threshold is applied on the read-back list.'
    },
    {
      id: 'radiusClamp',
      label: 'Search radius',
      hood: true,
      help: 'Whether the requested radius fits the compile-time search bound.'
    },
    {id: 'radiusNow', label: 'Radius', format: 'meters', hood: true},
    {id: 'dropNow', label: 'Minimum drop', format: 'meters', hood: true}
  ],

  pipeline: [
    {id: 'average', label: 'Average 4x4', detail: 'Box-average the DEM to 26.6 m analysis cells'},
    {id: 'disc', label: 'Disc max', detail: 'Is this cell the highest in its metric disc?'},
    {id: 'drop', label: 'Ring drop', detail: 'Height above the highest cell on the disc rim'},
    {id: 'snap', label: 'Snap', detail: 'Move catalogue points to the highest cell in a disc'},
    {id: 'contours', label: 'Contours', detail: 'Marching squares on the full DEM, 40 level slots'}
  ],

  legends: getSummitsLegends,

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTerrainSummits, GPUTerrainPeakSnap, GPUTerrainCriticalPoints, GPUTerrainContours,
  getGPUTerrainSummitsParameterValues, getGPUTerrainPeakSnapParameterValues
} from '@luma.gl/experimental/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'summits'});
graph.add(new GPUTerrainSummits({
  width, height, elevation,                       // the DEM averaged 4 x 4
  cellSizeMode: 'web-mercator',                   // cellSize + northEdge / southEdge, per row
  maximumRadiusPixels: ${state.summitMaximumRadius},                     // compile-time loop bound
  incompleteNeighborhood: '${state.incompleteNeighborhood}',
  settings: summitSettings.importToGraph(graph),
  output: {ids, count, overflow, totalCount},     // compact list of disc maxima
  outputDrop                                      // drop of each listed maximum
}));
graph.add(new GPUTerrainPeakSnap({
  width, height, elevation, cellSizeMode: 'web-mercator', maximumRadiusPixels: 16,
  candidates, candidateHeights, candidateRadii,   // catalogue positions, elevations, radii
  settings: snapSettings.importToGraph(graph),
  positions, heights, status, snapDistance, overflow
}));
graph.add(new GPUTerrainCriticalPoints({
  width, height, elevation, connectivity: ${state.connectivity}, classes, signChanges, counts
}));
const compiled = graph.compile();                 // once

// minimumDrop 0: list every disc maximum, then threshold the read-back list on the CPU
summitSettings.write(getGPUTerrainSummitsParameterValues({
  radius: ${state.summitRadius}, minimumDrop: 0,
  cellSize: [mercatorCell, mercatorCell], northEdge, southEdge   // Web Mercator: equatorial metres
}));
const summits = candidates.filter(candidate => candidate.drop >= ${state.summitMinimumDrop});
snapSettings.write(getGPUTerrainPeakSnapParameterValues({
  radius: ${state.snapRadius}, maximumMove: ${state.snapMaximumMove}, maximumHeightChange: ${state.snapMaximumHeightChange},
  interior: ${state.snapInterior}, cellSize: [mercatorCell, mercatorCell], northEdge, southEdge
}));
compiled.encode(commandEncoder, {parameters: undefined}); // when a parameter changed`,

  about: {
    what: '`GPUTerrainSummits` marks the cells that are the highest point of a ground-metre disc and rise at least a minimum drop above its rim; the scene runs it with no drop test, keeps the drop of every disc maximum, and applies the threshold on the list it reads back. `GPUTerrainPeakSnap` moves catalogue points onto the highest cell of a disc, `GPUTerrainCriticalPoints` classifies every cell as a peak, saddle or pit from the sign changes around it, and `GPUTerrainContours` extracts marching-squares contour segments with GPU-written draw records.',
    why: 'Summits are what a peak list, a panorama label or a summit register refers to. There is no single list: it depends on the radius and the drop you choose, on what the data window cuts off, and on how well a catalogue sits on the ground it describes. The drop is only a lower bound of prominence; true prominence needs the key col, a sweep over the whole DEM.',
    howToRead:
      'Ink triangles are summits, bigger for a bigger drop. Hollow grey triangles are disc maxima that fail the drop test. Hatching marks where a disc would leave the data. Paper rings are catalogue positions, filled circles the cells they snapped to (green snapped, grey unchanged, orange flank, vermilion too far, pink height mismatch). Orange diamonds are saddles, blue rings pits. Brown lines are contours, every fifth heavier. Names are real OpenStreetMap peaks within a short distance of a summit; their heights are the published ones, the DEM cells read lower. The terrain data are bundled, so `?data=synthetic` makes no difference here.'
  },

  create: async ctx => (await import('./summits.compute')).createSummits(ctx),

  story: [
    {
      id: 'the-question',
      title: 'Which bump is a mountain?',
      headline: 'Every triangle is the highest ground in its disc',
      textAlternative:
        'Shaded relief of the Matterhorn, Zermatt and the Gornergrat with dark triangles on the summits, larger where the drop is larger, and the names of the largest.',
      body: 'Every triangle marks a cell that is the highest within **{{radiusNow}}** and stands at least **{{dropNow}}** above the highest cell on the rim of its disc. `GPUTerrainSummits` asks that of every cell at once: **{{summitCount}}** pass, and **{{rejectedCount}}** are the highest in their disc but fail the drop test. Size shows the drop.\n\n*A summit is a definition, not a fact.*',
      optionsMode: 'fresh',
      options: {summitRadius: 400, summitMinimumDrop: 100},
      controls: [],
      readouts: ['summitCount', 'rejectedCount'],
      camera: {...SUMMIT_FRAMES.home, transitionMs: 1400},
      stage: 'disc',
      furniture: furnitureFor(
        'Which bump is a mountain?',
        'Highest within the radius, above its ring'
      ),
      annotations: [terrainLabel('zermatt')],
      highlight: {readout: 'summitCount'}
    },
    {
      id: 'radius',
      title: 'The radius decides what a summit is',
      headline: 'Wider discs keep fewer, bigger summits',
      textAlternative:
        'The same map with a chart of summit count against radius; hovering a cell outlines the disc that is tested around it.',
      body: 'Pick a **Scale** or slide the **Summit radius**: with a small disc every shoulder of a glacier is the highest ground nearby, with a large one only the tops of the massif survive. The curve runs the kernel at six radii as parameter writes with no recompile, and now holds **{{summitCount}}** at **{{radiusNow}}**. Hover a cell to see its disc.\n\n*Scale: the answer depends on how big a question you ask.*',
      optionsMode: 'fresh',
      options: {showProbe: true},
      controls: ['scale', 'summitRadius'],
      readouts: ['summitCount', 'countByRadius'],
      camera: {...SUMMIT_FRAMES.home, transitionMs: 1200},
      stage: 'disc',
      furniture: furnitureFor(
        'Does a wider disc keep fewer summits?',
        'Highest within the radius, above its ring'
      ),
      highlight: {readout: 'countByRadius'}
    },
    {
      id: 'drop',
      title: 'Drop is a lower bound of prominence',
      headline: 'Drop is the lower bound of prominence',
      textAlternative:
        'The Matterhorn and its neighbours with solid triangles on the summits that pass a larger drop and hollow grey triangles on disc maxima that fail it; a histogram of drops sits in the card.',
      body: 'A disc maximum is a summit only if it also stands at least the **Minimum drop** above its ring. Every way out of the disc crosses the ring, so the drop is a lower bound on true prominence. Hollow triangles fail: **{{rejectedCount}}** of them, against **{{summitCount}}** that pass. Drag the mark on the histogram to move the line.\n\n*A summit is a definition: radius and drop decide the list.*',
      optionsMode: 'fresh',
      options: {summitMinimumDrop: 250, showRejected: true, showProbe: true},
      controls: ['summitMinimumDrop'],
      readouts: ['summitCount', 'rejectedCount', 'dropHistogram'],
      camera: {...SUMMIT_FRAMES.drop, transitionMs: 1400},
      stage: 'drop',
      furniture: furnitureFor(
        'How far above its ring is a summit?',
        'Highest within the radius, above its ring'
      ),
      highlight: {readout: 'rejectedCount'}
    },
    {
      id: 'edge',
      title: 'The window edge eats mountains',
      headline: 'The window edge eats mountains',
      textAlternative:
        'The western edge of the data with a hatched band along it where a disc would leave the DEM, a frame labelled Data ends here, and the summit triangles inside it.',
      body: 'A disc that leaves the data has no honest ring. Set **Window edge** to Reject and every cell whose disc crosses the hatched band is refused: **{{edgeLost}}** disappear. At this radius the Matterhorn lies closer to the west edge than the disc is wide, so watch its label. Ignore keeps those cells but compares each with a clipped disc.\n\n*Edge effects: a clipped window changes the answer.*',
      optionsMode: 'fresh',
      options: {showEdge: true, incompleteNeighborhood: 'ignore'},
      controls: ['incompleteNeighborhood'],
      readouts: ['summitCount', 'edgeLost'],
      camera: {...SUMMIT_FRAMES.edge, transitionMs: 1400},
      stage: 'disc',
      furniture: furnitureFor(
        'What does the window edge remove?',
        'Highest within the radius, above its ring'
      ),
      highlight: {readout: 'edgeLost'}
    },
    {
      id: 'peak-snap',
      title: 'Snap a catalogue onto the DEM summits',
      headline: 'Catalogue peaks sit beside the summits they name',
      textAlternative:
        'Paper rings at the OpenStreetMap peak positions, joined by short lines to coloured circles on the highest DEM cells, with the names of six peaks and a stacked bar of the outcomes.',
      body: 'A hand-placed OpenStreetMap node need not sit on the highest cell. `GPUTerrainPeakSnap` searches a disc around each of the **{{catalogueSize}}** named peaks for the highest cell and keeps the point where the answer is a flank, a long move or a height that disagrees. The median snap is **{{medianMove}}**. Hover a peak: the published height and the DEM cell differ. Add **Catalogue position error** to stress it.\n\n*A catalogue is only as good as how it was placed.*',
      optionsMode: 'fresh',
      options: {showSnap: true, showSummits: false},
      controls: ['catalogueError', 'snapMaximumHeightChange'],
      readouts: ['snapOutcomes', 'medianMove', 'catalogueSize'],
      camera: {...SUMMIT_FRAMES.snap, transitionMs: 1400},
      stage: 'snap',
      furniture: furnitureFor(
        'Where does the catalogue put each peak?',
        'Catalogue peaks snapped to the highest cell'
      ),
      highlight: {readout: 'medianMove'}
    },
    {
      id: 'critical-and-contours',
      title: 'Contours and critical points draw the frame',
      headline: 'Passes, pits and contours frame every summit',
      textAlternative:
        'Brown contours around the Breithorn and the Klein Matterhorn with ink triangles for peaks, orange diamonds for saddles and blue rings for pits.',
      body: '`GPUTerrainCriticalPoints` counts how often the neighbours of a cell switch between lower and higher: no switch at all is a peak or a pit, four or more is a saddle, a pass. Contours (`GPUTerrainContours`) draw the same surface as lines, every fifth heavier. Switch the **Critical point ring** and compare: **{{criticalCounts}}**. Zoom in; the glyphs appear from a closer view.\n\n*The same surface, drawn as points and as lines.*',
      optionsMode: 'fresh',
      options: {showSummits: false, showCritical: true, showContours: true},
      controls: ['connectivity', 'contourInterval', 'showSummits'],
      readouts: ['criticalCounts', 'euler', 'contourSummary', 'timing'],
      camera: {...SUMMIT_FRAMES.critical, transitionMs: 1600},
      stage: 'contours',
      furniture: furnitureFor('Where do passes and contours meet?', 'Contours and critical points'),
      annotations: terrainLabels(['breithorn', 'klein-matterhorn', 'theodulpass']),
      highlight: {readout: 'criticalCounts'}
    }
  ]
});
