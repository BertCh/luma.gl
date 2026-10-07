// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import {earthDrop} from './cpu-dem';
import {
  BARE_EARTH_CHIP,
  TERRAIN_CREDIT,
  TERRAIN_FRAMES,
  terrainCartouche
} from './terrain-furniture';
import {makeMarkVisibleTable, makeVisibilityTable} from './terrain-palettes';
import {
  getCumulativeLegend,
  getFlipLegend,
  getLookoutLegend,
  getSightLineLegend,
  makePyramidTable
} from './viewshed-style';
import type {ViewshedOptions} from './viewshed.compute';

const REFRACTION = {'mt-image': 0.13, gdal: 1 / 7} as const;

/** The legends of the current state; the ground tone and the lookout names come from the compute module. */
function getLegends(state: ViewshedOptions, data: Readonly<Record<string, unknown>>): LegendSpec[] {
  const tone = (data.ground as 'light' | 'dark' | undefined) ?? 'light';
  const names = (data.lookouts as string[] | undefined) ?? [];
  const legends: LegendSpec[] = [];
  if (state.display === 'viewshed') {
    const table = state.markVisible ? makeMarkVisibleTable(tone) : makeVisibilityTable(tone);
    legends.push(
      getClassTableLegend(table, {
        title: 'Ground seen from the gold observer',
        id: 'visibility',
        layout: 'list',
        note: state.markVisible
          ? 'The wrong way round: painting what is seen buries the ground you are judging.'
          : state.toleranceMeters > 0 || state.tolerancePerKilometer > 0
            ? "Marginal (hatched): within the height uncertainty, after Fisher's probable viewshed."
            : 'Cells beyond the maximum distance are left clear.'
      })
    );
    if (state.showSightLine) legends.push(getSightLineLegend(tone));
    if (state.earthModel === 'changes') legends.push(getFlipLegend(tone));
  } else if (state.display === 'cumulative') {
    legends.push(getCumulativeLegend(tone, state.cumulativeObservers));
    legends.push(getLookoutLegend(tone, names.slice(0, Math.max(1, state.cumulativeObservers))));
  } else {
    legends.push(
      getClassTableLegend(makePyramidTable(tone), {
        title: 'Highest ground in the pyramid block',
        id: 'pyramid',
        layout: 'bar'
      })
    );
  }
  return legends;
}

/**
 * What can you see from Gornergrat? A viewshed, one line of sight, the cumulative viewshed of six
 * lookouts and the extrema pyramid, over a wide DEM of the whole cirque. The metadata, options and
 * narrative live here (light, loaded by the gallery); the GPU work lives in `viewshed.compute.ts`.
 */
export default defineScene<ViewshedOptions>({
  id: 'viewshed',
  title: 'What can you see from Gornergrat?',
  chapter: 'terrain',
  order: 5,
  summary:
    'Drag an observer across the Gornergrat cirque and see every cell it can and cannot see, how far a summit could sink and still be in sight, and which slopes six lookouts overlook together.',
  contributors: [
    'GPUTerrainViewshed',
    'GPUTerrainLineOfSight',
    'GPUTerrainCumulativeViewshed',
    'GPURasterExtremaPyramid'
  ],
  datasets: [
    {id: 'alps-dem-wide', role: 'terrain (Terrarium, Web Mercator), the whole cirque'},
    {id: 'alps-context', role: 'glaciers, peaks and stations (OpenStreetMap)'}
  ],
  initialView: {...TERRAIN_FRAMES.gornergratWide},
  basemap: ground('relief'),
  furniture: {
    title: terrainCartouche(
      'What can you see from Gornergrat?',
      'Seen from the gold observer',
      undefined,
      [BARE_EARTH_CHIP]
    ),
    scaleBar: {units: 'metric'},
    credit: TERRAIN_CREDIT
  },

  options: [
    {
      kind: 'slider',
      id: 'observerHeight',
      label: 'Eye height',
      group: 'Observer',
      apply: 'param',
      min: 0,
      max: 60,
      step: 1,
      default: 2,
      unit: 'm',
      help: 'Height of the eye above the ground at the observer. A person is about 2 m; a viewing tower or a mast is 30 to 60 m. Per-frame parameter.'
    },
    {
      kind: 'slider',
      id: 'targetHeight',
      label: 'Target height',
      group: 'Observer',
      apply: 'param',
      min: 0,
      max: 100,
      step: 1,
      default: 0,
      unit: 'm',
      help: 'Added to every target cell: 0 asks whether the ground is visible, 30 asks whether a 30 m tower standing there is visible.'
    },
    {
      kind: 'slider',
      id: 'maxDistance',
      label: 'Maximum distance',
      group: 'Observer',
      apply: 'param',
      min: 2,
      max: 13,
      step: 0.5,
      default: 10,
      unit: 'km',
      marks: [{value: 10, label: 'default'}],
      describe: (value, state) =>
        `The earth drops ${earthDrop(value * 1000, REFRACTION[state.refraction]).toFixed(1)} m by then. The data ends 13 km from the station.`,
      help: 'Cells farther than this are "out of range" and excluded from the shares. The wide DEM keeps 13 km of ground on every side of Gornergrat station.'
    },
    {
      kind: 'select',
      id: 'earthModel',
      label: 'Earth model',
      group: 'Earth model',
      apply: 'param',
      display: 'segmented',
      default: 'curved',
      help: 'Curved lowers terrain at distance d by c d squared, flat sets c to zero, and Cells that change draws the curved map and marks the cells that would flip against the flat one.',
      options: [
        {value: 'curved', label: 'Curved', help: 'The default: curvature with refraction.'},
        {value: 'flat', label: 'Flat', help: 'No curvature drop: c = 0.'},
        {
          value: 'changes',
          label: 'Cells that change',
          help: 'The curved map, with the cells that flip between curved and flat marked.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'refraction',
      label: 'Refraction coefficient',
      group: 'Earth model',
      apply: 'param',
      default: 'mt-image',
      expert: true,
      help: 'Terrain at distance d is lowered by c d squared, with c = (1 - k) / 2R, R = 6371008.8 m and refraction coefficient k.',
      options: [
        {
          value: 'mt-image',
          label: 'k = 0.13 (geodetic default)',
          help: 'The convention of mt-image and geodetic practice.'
        },
        {
          value: 'gdal',
          label: 'k = 1/7 (GDAL -cc 0.85714)',
          help: 'GDAL gdal_viewshed uses cc = 1 - k with k = 1/7.'
        }
      ]
    },
    {
      kind: 'preset',
      id: 'uncertainty',
      label: 'Height uncertainty',
      group: 'Uncertainty',
      help: 'How wrong the DEM heights may be. A band around the sight line turns the hard cut into a third class, marginal.',
      presets: [
        {
          label: 'None',
          values: {
            toleranceMeters: 0,
            tolerancePerKilometer: 0,
            targetIgnoreDistance: 0,
            targetIgnoreFraction: 0
          },
          help: 'The classic two-class viewshed.'
        },
        {
          label: 'mt-image',
          values: {
            toleranceMeters: 2,
            tolerancePerKilometer: 1,
            targetIgnoreDistance: 150,
            targetIgnoreFraction: 0.02
          },
          help: '2 m plus 1 m per km, ignoring the last 150 m or 2 percent.'
        },
        {
          label: 'Wide',
          values: {
            toleranceMeters: 5,
            tolerancePerKilometer: 3,
            targetIgnoreDistance: 150,
            targetIgnoreFraction: 0.02
          },
          help: '5 m plus 3 m per km: a pessimistic height error.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'toleranceMeters',
      label: 'Tolerance band (constant)',
      group: 'Uncertainty',
      apply: 'param',
      min: 0,
      max: 10,
      step: 0.5,
      default: 0,
      unit: 'm',
      expert: true,
      help: 'Half-width of a band around the sight line at the target. Cells inside it are "marginal" instead of visible or hidden. 0 gives the classic two-class viewshed.'
    },
    {
      kind: 'slider',
      id: 'tolerancePerKilometer',
      label: 'Tolerance band (per km)',
      group: 'Uncertainty',
      apply: 'param',
      min: 0,
      max: 5,
      step: 0.5,
      default: 0,
      unit: 'm/km',
      expert: true,
      help: 'Extra half-width per kilometre of distance, because DEM height errors matter more at long range.'
    },
    {
      kind: 'slider',
      id: 'targetIgnoreDistance',
      label: 'Ignore the last stretch before the target',
      group: 'Uncertainty',
      apply: 'param',
      min: 0,
      max: 500,
      step: 10,
      default: 0,
      unit: 'm',
      expert: true,
      help: 'Samples closer than this to the target are not tested, so a summit is not hidden by its own flank.'
    },
    {
      kind: 'slider',
      id: 'targetIgnoreFraction',
      label: 'Ignore the last fraction of the distance',
      group: 'Uncertainty',
      apply: 'param',
      min: 0,
      max: 0.1,
      step: 0.005,
      default: 0,
      expert: true,
      format: value => `${(value * 100).toFixed(1)} %`,
      help: 'The same idea as a fraction of the target distance, added to the fixed stretch above.'
    },
    {
      kind: 'toggle',
      id: 'markVisible',
      label: 'Mark visible instead',
      group: 'Display',
      apply: 'param',
      default: false,
      disabledWhen: state => state.display !== 'viewshed',
      help: 'Paints the cells you can see in gold and leaves the hidden ones bare: the wrong way round, shown only to compare.'
    },
    {
      kind: 'toggle',
      id: 'showSightLine',
      label: 'Sight line to a target',
      group: 'Sight line',
      apply: 'param',
      default: false,
      disabledWhen: state => state.display !== 'viewshed',
      help: 'Draws the straight line from the observer to the teal target, its profile and the steepest ground it passes. Drag the teal dot, or click the map to move it.'
    },
    {
      kind: 'select',
      id: 'rayTarget',
      label: 'Target',
      group: 'Sight line',
      apply: 'param',
      display: 'segmented',
      default: 'matterhorn',
      help: 'A summit to look at: the Matterhorn, or Zumsteinspitze, which lies behind the Monte Rosa ridge. Drag the teal dot to choose your own.',
      options: [
        {value: 'matterhorn', label: 'Matterhorn'},
        {value: 'zumsteinspitze', label: 'Zumsteinspitze'},
        {value: 'custom', label: 'Your pick'}
      ]
    },
    {
      kind: 'slider',
      id: 'rayPosition',
      label: 'Ray position',
      group: 'Sight line',
      apply: 'param',
      min: 0,
      max: 100,
      step: 1,
      default: 100,
      format: value =>
        value >= 100 ? 'at the target' : value <= 0 ? 'at the eye' : `${value} % of the way`,
      help: 'Moves the scan dot from the eye to the target: the running horizon is the steepest ground passed so far.'
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Show',
      group: 'Display',
      apply: 'param',
      default: 'viewshed',
      help: 'The single-observer viewshed, the cumulative count of several lookouts, or one level of the extrema pyramid.',
      options: [
        {value: 'viewshed', label: 'Viewshed from the gold observer'},
        {value: 'cumulative', label: 'Cumulative viewshed (six lookouts)'},
        {value: 'pyramid', label: 'Extrema pyramid, highest ground per block'}
      ]
    },
    {
      kind: 'slider',
      id: 'cumulativeObservers',
      label: 'Stations',
      group: 'Display',
      apply: 'param',
      min: 1,
      max: 6,
      step: 1,
      default: 6,
      disabledWhen: state => state.display !== 'cumulative',
      help: 'Gornergrat station plus up to five more stations of the railway and lifts. Unused slots are parked off the grid, so changing the count never recompiles.'
    },
    {
      kind: 'slider',
      id: 'pyramidLevel',
      label: 'Pyramid level',
      group: 'Display',
      apply: 'param',
      min: 0,
      max: 9,
      step: 1,
      default: 4,
      expert: true,
      disabledWhen: state => state.display !== 'pyramid',
      help: 'Level 0 is the finest. Each level doubles the block size; the last level is one cell holding the extremes of the whole DEM.'
    },
    {
      kind: 'select',
      id: 'cellSize',
      label: 'Cell size',
      group: 'Compile-time choices',
      apply: 'compile',
      display: 'segmented',
      default: '2',
      help: 'The analysis grid is the DEM averaged 1 x 1, 2 x 2 or 4 x 4: a finer grid costs more per frame and keeps narrower ridges. Changing it rebuilds the grid and its graphs.',
      options: [
        {value: '1', label: '13.3 m', help: 'The native cell of the wide DEM.'},
        {
          value: '2',
          label: '26.6 m',
          help: 'Averaged 2 x 2: the default, light enough to drag the observer.'
        },
        {value: '4', label: '53.1 m', help: 'Averaged 4 x 4: a coarse preview.'}
      ]
    },
    {
      kind: 'select',
      id: 'traversal',
      label: 'Traversal',
      group: 'Compile-time choices',
      apply: 'compile',
      display: 'segmented',
      default: 'march',
      help: 'March tests every sample; pyramid skips samples a min-max pyramid proves cannot matter. The answers are bit-identical. The graph for each is compiled the first time you choose it.',
      options: [
        {value: 'march', label: 'March'},
        {value: 'pyramid', label: 'Pyramid'}
      ]
    },
    {
      kind: 'select',
      id: 'pyramidBlockSize',
      label: 'Pyramid first block size',
      group: 'Compile-time choices',
      apply: 'compile',
      default: '4',
      expert: true,
      help: 'Pixel size of a level-0 pyramid cell. Smaller cells skip more tightly but cost more levels; changing it rebuilds the pyramid and the pyramid graphs.',
      options: [
        {value: '4', label: '4 pixels'},
        {value: '8', label: '8 pixels'},
        {value: '16', label: '16 pixels'}
      ]
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time and compare',
      group: 'Compare',
      help: 'Runs both traversals outside the frame, reports the GPU time of each, then compares every output word.'
    },
    {
      kind: 'button',
      id: 'verify',
      label: 'Verify pyramid = march',
      group: 'Compare',
      expert: true,
      help: 'Runs both traversals and compares every output word on the GPU buffers, without timing them.'
    }
  ],

  readouts: [
    {
      id: 'visibleShare',
      label: 'Visible',
      emphasis: 'tile',
      help: 'Share of the in-range ground cells that are clearly visible.'
    },
    {
      id: 'visibleArea',
      label: 'Visible area',
      emphasis: 'tile',
      help: 'Ground area of the visible cells.'
    },
    {
      id: 'hiddenPeaks',
      label: 'Out of sight',
      help: 'Named summits within reach whose summit cell is hidden, read on the GPU at their cells.'
    },
    {
      id: 'marginalShare',
      label: 'Marginal',
      emphasis: 'tile',
      help: 'Cells inside the tolerance band: neither clearly visible nor clearly hidden.'
    },
    {
      id: 'clearance',
      label: 'Clearance at the target',
      emphasis: 'tile',
      help: 'GPUTerrainLineOfSight: how far the target could sink and stay visible. Negative: how much taller it would have to be.'
    },
    {
      id: 'rayLength',
      label: 'The ray',
      help: 'Length of the sight line, and where the steepest ground along it lies.'
    },
    {
      id: 'profile',
      label: 'Profile along the sight line',
      kind: 'chart',
      help: 'Ground, the ground lowered by the earth curve, the sight line and the horizon so far, from the same model the GPU runs per cell.'
    },
    {
      id: 'flippedCells',
      label: 'Cells that flip',
      emphasis: 'tile',
      help: 'Cells whose class differs between the curved and the flat earth model.'
    },
    {
      id: 'dropAtReach',
      label: 'Earth drop at the reach',
      emphasis: 'tile',
      help: 'How far the earth falls away from the horizontal at the maximum distance, c d squared.'
    },
    {
      id: 'seenByNone',
      label: 'Seen from none',
      emphasis: 'tile',
      help: 'Share of the ground within reach of at least one station that no station sees.'
    },
    {
      id: 'seenByAll',
      label: 'Seen from all',
      emphasis: 'tile',
      help: 'Share of that ground that every active station sees.'
    },
    {
      id: 'unseenGap',
      label: 'Largest unseen gap',
      help: 'The largest circle holding only unseen cells, found on a coarse mask read back from the GPU.'
    },
    {
      id: 'visibleByCell',
      label: 'Visible share by cell size',
      kind: 'chart',
      help: 'Filled in as you try each cell size with the same observer and settings.'
    },
    {
      id: 'marchTime',
      label: 'Graph, march',
      hood: true,
      help: 'Measured GPU time of the march graph, not assumed.'
    },
    {
      id: 'pyramidTime',
      label: 'Graph, pyramid',
      hood: true,
      help: 'Measured GPU time of the pyramid graph, not assumed.'
    },
    {id: 'identical', label: 'Pyramid = march', hood: true},
    {
      id: 'grid',
      label: 'Analysis grid',
      hood: true,
      help: 'The analysis grid: the wide DEM averaged by the chosen stride.'
    },
    {
      id: 'range',
      label: 'In range',
      hood: true,
      help: 'Counted on the GPU with GPUHistogram; no per-cell readback.'
    },
    {id: 'pyramid', label: 'Extrema pyramid', hood: true}
  ],

  pipeline: [
    {
      id: 'observer',
      label: 'Observer',
      detail: 'Eye height, target height and reach are per-frame parameters: no rebuild'
    },
    {
      id: 'march',
      label: 'March rays',
      detail: 'One ray per cell toward the observer, keeping the steepest slope seen'
    },
    {
      id: 'tolerance',
      label: 'Tolerance',
      detail: 'A band around the sight line makes the marginal middle class'
    },
    {
      id: 'classes',
      label: 'Classes',
      detail: 'Visible, marginal, hidden and out of range, drawn as a veil'
    },
    {
      id: 'cumulative',
      label: 'Cumulative',
      detail: 'Counts how many lookouts see each cell',
      show: {option: 'display', value: 'cumulative'}
    }
  ],

  legends: getLegends,

  snippet: state => `import {GPUCommandGraph, GPUHistogram} from '@luma.gl/gpgpu/gpu-core';
import {GPURasterExtremaPyramid} from '@luma.gl/experimental/gpu-raster';
import {
  GPUTerrainViewshed,
  GPUTerrainLineOfSight,
  GPUTerrainCumulativeViewshed,
  getGPUTerrainViewshedParameterValues,
  getGPUTerrainVisibilityToleranceParameterValues,
  getGPUTerrainCurvatureCoefficient
} from '@luma.gl/experimental/gpu-terrain';

// The wide DEM averaged ${state.cellSize} x ${state.cellSize}; elevation is a float32 band with a validity mask.
const graph = new GPUCommandGraph(device, {id: 'viewshed'});
const elevation = {id: 'dem', format: 'float32', storage: {kind: 'buffer', values}, validity};
${
  state.traversal === 'pyramid'
    ? `
// One min-max pyramid, built once, read by every consumer through \`pyramid\`.
const extrema = new GPURasterExtremaPyramid({
  width, height, input: elevation, firstBlockSize: ${state.pyramidBlockSize}, footprint: 'bilinear', combined
});
graph.add(extrema);
`
    : ''
}
graph.add(new GPUTerrainViewshed({
  width, height, elevation,
  traversal: '${state.traversal}',${state.traversal === 'pyramid' ? '\n  pyramid: extrema.output,' : ''}
  settings: viewshedSettings.importToGraph(graph),   // observer, heights, reach, curvature
  tolerance: toleranceSettings.importToGraph(graph), // band: marginal cells
  visibility                                          // 0 hidden 1 visible 2 out of range 3 no data 4 marginal
}));
graph.add(new GPUTerrainLineOfSight({
  width, height, elevation, traversal: '${state.traversal}', pairs, settings: sightSettings.importToGraph(graph),
  visibility: losCode, clearance                      // metres the target could sink and stay visible
}));
${
  state.display === 'cumulative'
    ? `graph.add(new GPUTerrainCumulativeViewshed({
  width, height, elevation, traversal: '${state.traversal}',
  observers,                                          // [column, row] x ${state.cumulativeObservers}, the rest parked off the grid
  settings: sightSettings.importToGraph(graph), visibleCount
}));
`
    : ''
}graph.add(new GPUHistogram({input: visibility, output: counts, edges: [0, 1, 2, 3, 4, 5]}));
const compiled = graph.compile();                    // once

// Gornergrat station, eye ${state.observerHeight} m, ${state.maxDistance} km, ${state.earthModel === 'flat' ? 'flat earth' : state.refraction === 'gdal' ? 'k = 1/7' : 'k = 0.13'}:
viewshedSettings.write(getGPUTerrainViewshedParameterValues({
  observer: [column, row],                            // pixel-centre index space
  observerHeight: ${state.observerHeight}, targetHeight: ${state.targetHeight},
  maxDistance: ${state.maxDistance * 1000},
  cellSize: [groundCellSize, groundCellSize],         // Web Mercator: scale by cos(latitude)
  curvatureCoefficient: ${state.earthModel === 'flat' ? '0' : `getGPUTerrainCurvatureCoefficient(${state.refraction === 'gdal' ? '1 / 7' : '0.13'})`}
}));
toleranceSettings.write(getGPUTerrainVisibilityToleranceParameterValues({
  toleranceMeters: ${state.toleranceMeters}, tolerancePerKilometer: ${state.tolerancePerKilometer},
  targetIgnoreDistance: ${state.targetIgnoreDistance}, targetIgnoreFraction: ${state.targetIgnoreFraction}
}));
compiled.encode(commandEncoder, {parameters: undefined}); // every change, no recompile`,

  about: {
    what: '`GPUTerrainViewshed` tests, for every cell, whether the straight line to the observer is blocked by terrain. `GPUTerrainLineOfSight` does the same for chosen pairs and reports the clearance in metres, `GPUTerrainCumulativeViewshed` counts how many observers see each cell, and `GPURasterExtremaPyramid` is the min-max mip chain that lets a traversal skip samples that provably cannot matter.',
    why: 'Viewsheds answer siting questions: where a tower, a hut or a wind turbine is visible from, which slopes a lookout overlooks, and how much of a landscape a road exposes. The clearance number turns a yes or no into a margin you can reason about.',
    howToRead:
      'The relief is the ground; a dark veil lies over what the gold observer cannot see and the visible ground stays clear. Hatched orange is marginal, within the height uncertainty. The data are bare earth: trees and buildings are not in them, so a viewshed from a village is optimistic.'
  },

  create: async ctx => (await import('./viewshed.compute')).createViewshed(ctx),

  story: [
    {
      id: 'the-question',
      title: 'What can you see from Gornergrat?',
      headline: 'From Gornergrat, much of the cirque is hidden',
      textAlternative:
        'Relief map of the Gornergrat cirque around a gold observer: a dark indigo veil covers the ground the station cannot see, and the visible ground is left clear, with dashed rings at fixed distances.',
      body: 'Gornergrat station is the end of the cog railway above Zermatt. **{{visibleShare}}** of the ground in reach is visible (**{{visibleArea}}**); the dark veil is ground the station cannot see. `GPUTerrainViewshed` marches one ray per cell. Raise **Eye height**, or switch on **Mark visible instead** to see why this map veils the hidden. Out of sight: {{hiddenPeaks}}.\n\n*Veil what is hidden; leave what you can see clear.*',
      optionsMode: 'fresh',
      controls: ['observerHeight', 'markVisible'],
      readouts: ['visibleShare', 'visibleArea', 'hiddenPeaks'],
      camera: {...TERRAIN_FRAMES.gornergratWide, transitionMs: 1400},
      furniture: {title: {title: 'What can you see from Gornergrat?'}},
      stage: 'observer'
    },
    {
      id: 'line-of-sight',
      title: 'One ray decides each cell',
      headline: 'The Matterhorn is in sight; Zumsteinspitze is not',
      textAlternative:
        'Closer map with a sight line from the gold observer to a teal target and a profile chart beneath: the straight line clears the ground for the Matterhorn and is cut by a ridge for Zumsteinspitze.',
      body: "One cell's answer is one ray. `GPUTerrainLineOfSight` walks from the eye to the target, keeps the steepest ground it passes and compares it with the sight line: the target could sink **{{clearance}}** and stay in sight (negative: how much taller it must be). Slide **Ray position** to scan, or change **Target**.\n\n*Visibility is a comparison of angles.*",
      options: {
        showSightLine: true,
        rayTarget: 'matterhorn',
        targetIgnoreDistance: 150,
        targetIgnoreFraction: 0.02
      },
      optionsMode: 'fresh',
      controls: ['rayTarget', 'rayPosition'],
      readouts: ['clearance', 'rayLength', 'profile'],
      camera: {longitude: 7.72, latitude: 45.981, zoom: 12.3, transitionMs: 1400},
      furniture: {title: {title: 'How does one ray decide?'}},
      stage: 'march'
    },
    {
      id: 'curvature',
      title: 'The earth falls away, and light bends',
      headline: 'The earth falls away under long sight lines',
      textAlternative:
        'The same map with the cells whose class changes between a curved and a flat earth marked in purple; they are few and lie at the far edge of the reach.',
      body: 'Terrain at distance `d` sits lower by `c d²`, where `c` is the curve of the earth less what refraction bends back. It falls **{{dropAtReach}}** and flips **{{flippedCells}}** (purple). Compare **Earth model**, or stretch **Maximum distance**. On this terrain the effect is small; it grows with the square of distance.\n\n*A small term, made visible.*',
      options: {earthModel: 'changes'},
      optionsMode: 'fresh',
      controls: ['earthModel', 'maxDistance'],
      readouts: ['dropAtReach', 'flippedCells', 'visibleShare'],
      camera: {...TERRAIN_FRAMES.gornergratWide, transitionMs: 1400},
      furniture: {title: {title: "Does the earth's curve matter here?"}},
      stage: 'march'
    },
    {
      id: 'marginal',
      title: 'Marginal cells: the honest middle',
      headline: 'The shadow edge is a band, not a line',
      textAlternative:
        'Close map of the ridges beside the station: a hatched orange band of marginal cells runs between the clear visible ground and the dark hidden ground.',
      body: 'A DEM has height errors, so a ray that misses a ridge by less than the error is not really hidden. The hatched orange band is **marginal**: **{{marginalShare}}** of the ground in reach. Pick a **Height uncertainty**: none gives the classic two-class map, Wide thickens the band around every shadow.\n\n*Fisher: a probable viewshed, not a hard line.*',
      options: {
        toleranceMeters: 2,
        tolerancePerKilometer: 1,
        targetIgnoreDistance: 150,
        targetIgnoreFraction: 0.02
      },
      optionsMode: 'fresh',
      controls: ['uncertainty'],
      readouts: ['marginalShare', 'visibleShare'],
      camera: {longitude: 7.77, latitude: 45.98, zoom: 13, transitionMs: 1400},
      furniture: {title: {title: 'How sure is the shadow edge?'}},
      stage: 'tolerance'
    },
    {
      id: 'cumulative',
      title: 'Which slopes does no station overlook?',
      headline: 'Most of the ground is seen from no station',
      textAlternative:
        'Map of six numbered stations coloured by how many see each cell, from pale yellow for one to dark red for all six; a dark veil covers the large area that none sees, with a note on the largest gap.',
      body: 'Six stations of the Gornergrat railway and the lifts stand at different heights. `GPUTerrainCumulativeViewshed` counts, for each cell, how many see it: **{{seenByNone}}** of the ground within reach is seen from none (dark), **{{seenByAll}}** from all. Remove some with **Stations**; the legend lists them.\n\n*A count needs its zero.*',
      options: {display: 'cumulative', cumulativeObservers: 6},
      optionsMode: 'fresh',
      controls: ['cumulativeObservers'],
      readouts: ['seenByNone', 'seenByAll', 'unseenGap'],
      camera: {longitude: 7.745, latitude: 45.975, zoom: 11.5, transitionMs: 1400},
      furniture: {title: {title: 'Which slopes do the stations overlook?'}},
      stage: 'cumulative'
    },
    {
      id: 'resolution-and-pyramid',
      title: 'Coarser cells change the answer; the pyramid does not',
      headline: 'Coarser cells change the answer; the pyramid does not',
      textAlternative:
        'The wide map again with three bars comparing the visible share at three cell sizes, and two measured timings for the march and the pyramid traversal.',
      body: 'The same observer, three cell sizes: **Cell size** changes which narrow ridges survive the averaging, so the visible share moves with it (bars). **Traversal** swaps the ray walk for the extrema pyramid, which skips samples that cannot matter: press **Time and compare** and read the measured times and the check. Try a mast with **Eye height**, a tower with **Target height**.\n\n*Resolution is part of the answer.*',
      optionsMode: 'fresh',
      controls: ['cellSize', 'traversal', 'measure', 'observerHeight', 'targetHeight'],
      readouts: ['visibleByCell', 'marchTime', 'pyramidTime', 'identical'],
      camera: {...TERRAIN_FRAMES.gornergratWide, transitionMs: 1400},
      furniture: {title: {title: 'Does the cell size change the answer?'}},
      stage: 'march'
    }
  ]
});
