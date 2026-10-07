// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import type {ViewshedOptions} from './viewshed.compute';

/**
 * What can you see from Gornergrat? Viewshed, line of sight, cumulative viewshed and an elevation
 * profile of the Matterhorn DEM. The metadata, options and narrative live here (light, loaded by
 * the gallery); the GPU work lives in `viewshed.compute.ts`.
 */
export default defineScene<ViewshedOptions>({
  id: 'viewshed',
  title: 'What can you see from Gornergrat?',
  chapter: 'terrain',
  order: 5,
  summary:
    'Drag an observer across the Matterhorn region and see every cell it can and cannot see, how many metres of ridge hide the Matterhorn, and which slopes are visible from six lookouts at once.',
  contributors: [
    'GPUTerrainViewshed',
    'GPUTerrainCumulativeViewshed',
    'GPUTerrainLineOfSight',
    'GPURasterExtremaPyramid',
    'GPURasterProfile',
    'GPUTerrainDerivatives'
  ],
  datasets: [{id: 'alps-dem', role: 'terrain (Terrarium, Web Mercator)'}],
  initialView: {longitude: 7.738, latitude: 45.984, zoom: 11.7},

  options: [
    {
      kind: 'slider',
      id: 'observerHeight',
      label: 'Eye height above ground',
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
      label: 'Height of what you want to see',
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
      min: 1,
      max: 14,
      step: 0.5,
      default: 14,
      unit: 'km',
      help: 'Cells farther than this are "out of range" and excluded from the percentages. The window is 13.6 km wide.'
    },
    {
      kind: 'select',
      id: 'refraction',
      label: 'Earth curvature and refraction',
      group: 'Earth model',
      apply: 'param',
      default: 'mt-image',
      help: 'Terrain at distance d is lowered by c d², with c = (1 - k) / 2R, R = 6371008.8 m and refraction coefficient k.',
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
        },
        {value: 'none', label: 'Flat earth', help: 'No curvature drop: c = 0.'}
      ]
    },
    {
      kind: 'slider',
      id: 'toleranceMeters',
      label: 'Tolerance band (constant)',
      group: 'Tolerance',
      apply: 'param',
      min: 0,
      max: 10,
      step: 0.5,
      default: 0,
      unit: 'm',
      help: 'Half-width of a band around the sight line at the target. Cells inside it are "marginal" instead of visible or hidden. 0 gives the classic two-class viewshed.'
    },
    {
      kind: 'slider',
      id: 'tolerancePerKilometer',
      label: 'Tolerance band (per km)',
      group: 'Tolerance',
      apply: 'param',
      min: 0,
      max: 5,
      step: 0.5,
      default: 0,
      unit: 'm/km',
      help: 'Extra half-width per kilometre of distance, because DEM height errors matter more at long range. mt-image uses 2 m + 1 m/km.'
    },
    {
      kind: 'slider',
      id: 'targetIgnoreDistance',
      label: 'Ignore the last stretch before the target',
      group: 'Tolerance',
      apply: 'param',
      min: 0,
      max: 500,
      step: 10,
      default: 0,
      unit: 'm',
      help: 'Samples closer than this to the target are not tested, so a summit is not hidden by its own flank. mt-image ignores the last max(150 m, 2 %).'
    },
    {
      kind: 'slider',
      id: 'targetIgnoreFraction',
      label: 'Ignore the last fraction of the distance',
      group: 'Tolerance',
      apply: 'param',
      min: 0,
      max: 0.1,
      step: 0.005,
      default: 0,
      format: value => `${(value * 100).toFixed(1)} %`,
      help: 'The same idea as a fraction of the target distance, added to the fixed stretch above.'
    },
    {
      kind: 'toggle',
      id: 'showProfile',
      label: 'Profile along the sight line',
      group: 'Profile',
      apply: 'param',
      default: false,
      help: 'Dots along the sight line from the red observer to the cyan target, sampled by GPURasterProfile and coloured by how far the ground rises above (red) or sits below (blue) the straight line.'
    },
    {
      kind: 'slider',
      id: 'profileSpacing',
      label: 'Profile sample spacing',
      group: 'Profile',
      apply: 'param',
      min: 5,
      max: 100,
      step: 5,
      default: 20,
      unit: 'm',
      disabledWhen: state => !state.showProfile,
      help: 'Distance between profile samples. The profile is a per-frame parameter: the graph is not rebuilt.'
    },
    {
      kind: 'select',
      id: 'profileMethod',
      label: 'Profile interpolation',
      group: 'Profile',
      apply: 'param',
      default: 'bilinear',
      disabledWhen: state => !state.showProfile,
      help: 'How the profile reads the DEM between pixel centres.',
      options: [
        {value: 'nearest', label: 'Nearest'},
        {value: 'bilinear', label: 'Bilinear'},
        {value: 'bicubic', label: 'Bicubic'}
      ]
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Show',
      group: 'Display',
      apply: 'param',
      default: 'viewshed',
      help: 'The single-observer viewshed, the cumulative count of several observers, or one level of the extrema pyramid.',
      options: [
        {value: 'viewshed', label: 'Viewshed from the red observer'},
        {value: 'cumulative', label: 'Cumulative viewshed (several observers)'},
        {value: 'pyramid', label: 'Extrema pyramid, maximum per cell'}
      ]
    },
    {
      kind: 'slider',
      id: 'cumulativeObservers',
      label: 'Observers in the cumulative viewshed',
      group: 'Display',
      apply: 'param',
      min: 1,
      max: 6,
      step: 1,
      default: 6,
      disabledWhen: state => state.display !== 'cumulative',
      help: 'The red observer plus up to five fixed lookouts (orange). Unused slots are parked off the grid, so changing the count never recompiles.'
    },
    {
      kind: 'select',
      id: 'cumulativeMetric',
      label: 'Count',
      group: 'Display',
      apply: 'param',
      default: 'visible',
      disabledWhen: state => state.display !== 'cumulative',
      help: 'How many observers see each cell clearly, or how many see it only marginally (inside the tolerance band).',
      options: [
        {value: 'visible', label: 'Visible from'},
        {value: 'marginal', label: 'Marginal from'}
      ]
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
      disabledWhen: state => state.display !== 'pyramid',
      help: 'Level 0 is the finest. Each level doubles the block size; the last level is one cell holding the extremes of the whole DEM.'
    },
    {
      kind: 'select',
      id: 'base',
      label: 'Terrain base',
      group: 'Display',
      apply: 'param',
      default: 'hillshade',
      help: 'Hillshade from GPUTerrainDerivatives (Web Mercator cell model), or the elevation itself.',
      options: [
        {value: 'hillshade', label: 'Hillshade'},
        {value: 'elevation', label: 'Elevation'}
      ]
    },
    {
      kind: 'select',
      id: 'traversal',
      label: 'Traversal',
      group: 'Compile-time choices',
      apply: 'compile',
      default: 'march',
      help: 'march tests every sample; pyramid skips samples a min-max pyramid proves cannot matter. The answers are bit-identical. The graph for each is compiled the first time you choose it.',
      options: [
        {value: 'march', label: 'March (every sample)'},
        {value: 'pyramid', label: 'Pyramid (min-max skip)'}
      ]
    },
    {
      kind: 'select',
      id: 'pyramidBlockSize',
      label: 'Pyramid first block size',
      group: 'Compile-time choices',
      apply: 'compile',
      default: '4',
      help: 'Pixel size of a level-0 pyramid cell. Smaller cells skip more tightly but cost more levels; changing it rebuilds the pyramid and the pyramid graphs.',
      options: [
        {value: '4', label: '4 pixels'},
        {value: '8', label: '8 pixels'},
        {value: '16', label: '16 pixels'}
      ]
    },
    {
      kind: 'button',
      id: 'verify',
      label: 'Verify pyramid = march',
      group: 'Compare',
      help: 'Runs both traversals and compares every output word on the GPU buffers.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time march vs pyramid',
      group: 'Compare',
      help: 'Runs each graph outside the frame and reports GPU time.'
    }
  ],

  readouts: [
    {
      id: 'grid',
      label: 'Analysis grid',
      help: 'The DEM is averaged 2 × 2 so an observer drag stays interactive.'
    },
    {
      id: 'visible',
      label: 'Visible',
      help: 'Share and area of in-range cells that are clearly visible.'
    },
    {
      id: 'marginal',
      label: 'Marginal',
      help: 'Cells inside the tolerance band: neither clearly visible nor clearly hidden.'
    },
    {id: 'hidden', label: 'Hidden'},
    {
      id: 'range',
      label: 'In range',
      help: 'Counted on the GPU with GPUHistogram; no per-cell readback.'
    },
    {
      id: 'lineOfSight',
      label: 'Sight line to target',
      help: 'GPUTerrainLineOfSight: the code, the distance and the clearance in metres. Positive clearance is how far the target could sink and stay visible; negative is how much taller it would have to be.'
    },
    {
      id: 'profile',
      label: 'Profile',
      help: 'Gain, loss and lowest point along the sight line, from GPURasterProfile.'
    },
    {
      id: 'drop',
      label: 'Curvature drop',
      help: 'How far the earth falls away from the sight line at the maximum distance.'
    },
    {id: 'pyramid', label: 'Extrema pyramid'},
    {id: 'identical', label: 'Pyramid = march'},
    {id: 'marchTime', label: 'Graph, march'},
    {id: 'pyramidTime', label: 'Graph, pyramid'}
  ],

  legends: state => {
    const legends: LegendSpec[] = [];
    if (state.display === 'viewshed') {
      const hasTolerance = state.toleranceMeters > 0 || state.tolerancePerKilometer > 0;
      legends.push({
        kind: 'categories',
        title: 'Cell seen from the red observer',
        entries: [
          {color: [60, 230, 110, 200], label: 'Visible'},
          ...(hasTolerance
            ? [{color: [255, 170, 40, 220] as const, label: 'Marginal (inside the tolerance band)'}]
            : []),
          {color: [12, 14, 40, 200], label: 'Hidden behind terrain'}
        ],
        note: 'Cells farther than the maximum distance are left clear.'
      });
    } else if (state.display === 'cumulative') {
      legends.push({
        kind: 'ramp',
        title:
          state.cumulativeMetric === 'visible'
            ? 'Observers that see the cell'
            : 'Observers for which the cell is marginal',
        ramp: 'viridis',
        extent: [1, Math.max(state.cumulativeObservers, 2)],
        unit: 'observers',
        format: value => value.toFixed(0)
      });
    } else {
      legends.push({
        kind: 'ramp',
        title: 'Highest ground in the pyramid cell',
        ramp: 'viridis',
        extent: [1503, 4476],
        unit: 'm',
        format: value => value.toFixed(0)
      });
    }
    legends.push({
      kind: 'categories',
      title: 'Sight line to the cyan target',
      entries: [
        {color: [40, 200, 100, 255], label: 'Visible'},
        {color: [255, 170, 40, 255], label: 'Marginal'},
        {color: [235, 60, 70, 255], label: 'Hidden'}
      ]
    });
    if (state.showProfile) {
      legends.push({
        kind: 'ramp',
        title: 'Ground above (+) or below (-) the sight line',
        ramp: 'diverging',
        extent: [-80, 80],
        unit: 'm',
        labels: ['-80 m', '+80 m']
      });
    }
    if (state.base === 'elevation') {
      legends.push({
        kind: 'ramp',
        title: 'Terrain elevation',
        ramp: 'cividis',
        extent: [1503, 4476],
        unit: 'm',
        format: value => value.toFixed(0)
      });
    }
    return legends;
  },

  snippet: state => `import {GPUCommandGraph, GPUHistogram} from '@luma.gl/gpgpu/gpu-core';
import {GPURasterExtremaPyramid, GPURasterProfile} from '@luma.gl/experimental/gpu-raster';
import {
  GPUTerrainViewshed,
  GPUTerrainLineOfSight,
  GPUTerrainCumulativeViewshed,
  getGPUTerrainViewshedParameterValues,
  getGPUTerrainVisibilityToleranceParameterValues,
  getGPUTerrainCurvatureCoefficient
} from '@luma.gl/experimental/gpu-terrain';

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
  settings: viewshedSettings.importToGraph(graph),   // observer, heights, radius, curvature
  tolerance: toleranceSettings.importToGraph(graph), // band: marginal cells
  visibility                                          // 0 hidden 1 visible 2 out of range 3 no data 4 marginal
}));
graph.add(new GPUTerrainLineOfSight({
  width, height, elevation, traversal: '${state.traversal}', pairs, settings: sightSettings.importToGraph(graph),
  visibility: losCode, clearance                      // metres the target could sink and stay visible
}));
graph.add(new GPUHistogram({input: visibility, output: counts, edges: [0, 1, 2, 3, 4, 5]}));
const compiled = graph.compile();                    // once

// Gornergrat, 2 m eye height, ${state.maxDistance} km, k = ${state.refraction === 'gdal' ? '1/7' : state.refraction === 'none' ? 'none' : '0.13'}:
viewshedSettings.write(getGPUTerrainViewshedParameterValues({
  observer: [column, row],                            // pixel-centre index space
  observerHeight: ${state.observerHeight}, targetHeight: ${state.targetHeight},
  maxDistance: ${state.maxDistance * 1000},
  cellSize: [groundCellSize, groundCellSize],         // Web Mercator: scale by cos(latitude)
  curvatureCoefficient: ${state.refraction === 'none' ? '0' : `getGPUTerrainCurvatureCoefficient(${state.refraction === 'gdal' ? '1 / 7' : '0.13'})`}
}));
toleranceSettings.write(getGPUTerrainVisibilityToleranceParameterValues({
  toleranceMeters: ${state.toleranceMeters}, tolerancePerKilometer: ${state.tolerancePerKilometer}
}));
compiled.encode(commandEncoder, {parameters: undefined}); // every change, no recompile`,

  about: {
    what: '`GPUTerrainViewshed` tests, for every cell, whether the straight line to the observer is blocked by terrain. `GPUTerrainLineOfSight` does the same for chosen pairs and reports the clearance in metres, `GPUTerrainCumulativeViewshed` counts how many observers see each cell, and `GPURasterExtremaPyramid` is the min-max mip chain that lets a traversal skip samples that provably cannot matter.',
    why: 'Viewsheds answer siting questions: where a tower, a hut or a wind turbine is visible from, which slopes a lookout overlooks, and how much of a landscape a road exposes. The clearance number turns a yes/no into a margin you can reason about.',
    howToRead:
      'Green cells are visible from the red observer, dark cells are hidden, and orange cells are inside the tolerance band. The cyan target and the line to it show the single sight line; its colour matches the readout. The width of the window is 13.6 km, so everything beyond is simply not in the data.'
  },

  create: async ctx => (await import('./viewshed.compute')).createViewshed(ctx),

  story: [
    {
      id: 'the-question',
      controls: ['observerHeight'],
      readouts: ['visible', 'hidden'],
      title: 'What can you see from Gornergrat?',
      body: 'Gornergrat (about 3,100 m) is the end of the cog railway above Zermatt, and its terrace is famous for one view. **Where, exactly, is that view?** The red dot is the observer; every green cell is a place whose ground you could see from a standing person there.\n\n**`GPUTerrainViewshed`** tests all one million cells on the GPU. For every cell it marches the straight line towards the observer, one sample per pixel with bilinear heights, and keeps the largest slope seen. The cell is **visible** when no sample rises above the line from the eye to the cell, **hidden** when one does. Drag the red dot (or click to move the cyan target) and watch the map recompute. **Eye height above ground** below lifts the observer off the terrace.',
      camera: {longitude: 7.738, latitude: 45.984, zoom: 11.7, transitionMs: 900}
    },
    {
      id: 'curvature',
      controls: ['refraction', 'maxDistance'],
      readouts: ['visible', 'drop', 'range'],
      title: 'The earth falls away, and light bends',
      body: 'Terrain at distance `d` is lowered by `c d²` with `c = (1 - k) / 2R`: the earth curves away, but the atmosphere bends light back, so only a fraction `1 - k` of the geometric drop counts. With `k = 0.13` (the geodetic default) the drop at 10 km is 6.8 m; GDAL `gdal_viewshed -cc 0.85714` is the same model with `k = 1/7`.\n\nSet **Earth curvature and refraction** below to *Flat earth* and compare the **Visible** readout with the earlier value. Over 14 km the effect is a few metres: enough to flip cells that graze a ridge, invisible on the map. Over 100 km it decides whether a peak shows at all. The **Maximum distance** slider below limits how far a cell counts; cells beyond it are "out of range".',
      options: {refraction: 'none'},
      highlight: {readout: 'visible'}
    },
    {
      id: 'marginal',
      controls: ['toleranceMeters', 'tolerancePerKilometer', 'targetIgnoreDistance'],
      readouts: ['marginal'],
      title: 'Marginal cells: the honest middle',
      body: 'Back on the geodetic curvature (`k = 0.13`): a DEM has height errors, so a cell that is hidden by 20 cm is not really hidden. The tolerance band (**Tolerance band (constant)** and **Tolerance band (per km)**) adds a third class: with the constant 2 m and 1 m per kilometre of mt-image, a cell is **hidden** if the highest sample rises more than the band above the line, **visible** if it stays more than the band below, and **marginal** (orange) in between.\n\nThe last stretch before the target is skipped too (**Ignore the last stretch before the target**), so a summit is not hidden by its own flank. Widen the band below to see the orange fringe grow around every ridge shadow.',
      options: {
        refraction: 'mt-image',
        toleranceMeters: 2,
        tolerancePerKilometer: 1,
        targetIgnoreDistance: 150,
        targetIgnoreFraction: 0.02
      },
      highlight: {readout: 'marginal'}
    },
    {
      id: 'line-of-sight',
      controls: ['showProfile', 'profileSpacing'],
      readouts: ['lineOfSight', 'profile'],
      title: 'Is the Matterhorn really visible? Clearance in metres',
      body: '**`GPUTerrainLineOfSight`** runs the same sight-line model for a chosen pair. The cyan target sits on the Matterhorn summit about 9.7 km west of the terrace, and the **Sight line to target** readout reports its code and the **clearance**: how many metres the summit could sink and stay visible. A negative number would be the height you would need to add.\n\nThe **Profile along the sight line** is on: **`GPURasterProfile`** samples the DEM along the line (at the **Profile sample spacing** you choose) and the dots show how far the ground rises above the straight line. Drag the cyan target onto a ridge and watch the clearance turn negative.',
      camera: {longitude: 7.72, latitude: 45.98, zoom: 12.4, transitionMs: 1200},
      options: {
        showProfile: true,
        tolerancePerKilometer: 0,
        toleranceMeters: 0,
        targetIgnoreDistance: 0,
        targetIgnoreFraction: 0
      },
      callout: {coordinate: [7.6586, 45.9766], text: 'Matterhorn, 4,478 m'},
      highlight: {readout: 'lineOfSight'}
    },
    {
      id: 'cumulative',
      controls: ['cumulativeObservers', 'cumulativeMetric'],
      readouts: [],
      title: 'Scenic exposure: how many lookouts see each slope?',
      body: '**`GPUTerrainCumulativeViewshed`** runs many observers and counts, per cell, how many see it: the same idea as GDAL cumulative mode. Here the observers are Gornergrat plus Klein Matterhorn, Riffelhorn, Theodulhorn, Zermatt and the Matterhorn summit.\n\nBright cells are overlooked from many places (a scenic or an exposed slope); dark cells are seen from only one. Drag **Observers in the cumulative viewshed** down to remove lookouts: unused slots are parked off the grid, a buffer write rather than a recompile. Switch **Count** to *Marginal from* to count the cells seen only inside the tolerance band.',
      camera: {longitude: 7.738, latitude: 45.984, zoom: 11.7, transitionMs: 1200},
      options: {display: 'cumulative', showProfile: false, cumulativeObservers: 6}
    },
    {
      id: 'pyramid',
      controls: ['traversal', 'verify', 'measure', 'pyramidLevel'],
      readouts: ['identical', 'marchTime', 'pyramidTime'],
      title: 'Same answer, fewer samples: the extrema pyramid',
      body: "**`GPURasterExtremaPyramid`** builds an exact min-max mip chain of the DEM (Tevs, Ihrke and Seidel 2008). A ray can skip a whole block when the block's highest point is below the line, and the answer stays **bit-identical** to the marching traversal. Set **Traversal** to *Pyramid*, press **Verify pyramid = march**, then **Time march vs pyramid**; **Pyramid level** below steps through the mip chain.\n\nOn this DEM at 1024 × 1024 the pyramid graph took about half the GPU time of the march in our test (8 ms against 16 ms on an Apple-silicon laptop); your GPU and the observer position will differ, so read your own numbers. The pyramid is built once and shared by the viewshed, the sight line and the cumulative graph.",
      options: {display: 'pyramid', traversal: 'pyramid', pyramidLevel: 4}
    },
    {
      id: 'limits-and-ideas',
      controls: [
        'observerHeight',
        'targetHeight',
        'display',
        'cumulativeMetric',
        'toleranceMeters'
      ],
      readouts: [],
      title: 'Limits, and things to try',
      body: 'This DEM is 13.6 km wide, so the viewshed ends at the window edge: Monte Rosa and the Liskamm lie just outside. The Web Mercator pixels are 9.6 m but the ground size is 6.6 m at this latitude; the planar contributors here use one uniform ground size (the cosine of latitude changes 0.1 % across the window) after a 2 × 2 average. No trees, buildings or glacier change are in the surface, so a viewshed from a village is optimistic.\n\n**Try:** raise **Eye height above ground** to 40 m for a mast; **Height of what you want to see** is already 30 m here, to ask about a tower; switch **Show** to the cumulative viewshed and **Count** to *Marginal from* with a nonzero **Tolerance band (constant)**.',
      options: {display: 'viewshed', traversal: 'march', targetHeight: 30, observerHeight: 2}
    }
  ]
});
