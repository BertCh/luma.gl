// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import type {HorizonOptions} from './horizon.compute';

/**
 * "Which peaks can I see from Gornergrat?" A 360 degree skyline from a draggable observer, the
 * peaks of that skyline, and a visible / hidden / on-the-skyline label for every catalogue peak.
 * The GPU work lives in `horizon.compute.ts`.
 */
export default defineScene<HorizonOptions>({
  id: 'horizon',
  title: 'Which peaks can I see from Gornergrat?',
  chapter: 'terrain',
  order: 6,
  summary:
    'The 360 degree skyline from a draggable viewpoint above Zermatt, its peaks, and a visible, hidden or on-the-skyline label for the Matterhorn, Breithorn, Pollux and a dozen more summits.',
  contributors: [
    'GPUPointHorizonProfile',
    'GPUPointHorizonVisibility',
    'GPUProfilePeaks',
    'GPUTerrainHorizon',
    'GPURasterExtremaPyramid',
    'GPUTerrainDerivatives'
  ],
  datasets: [{id: 'alps-dem', role: 'terrain (Terrarium, Web Mercator)'}],
  initialView: {longitude: 7.738, latitude: 45.984, zoom: 11.7},

  options: [
    {
      kind: 'slider',
      id: 'observerHeight',
      label: 'Eye height',
      group: 'Observer',
      apply: 'param',
      min: 0,
      max: 100,
      step: 1,
      default: 2,
      unit: 'm',
      help: 'Eye height above the ground at the observer (or above sea level with the absolute height reference). Per-frame: it is a value in the observer row.'
    },
    {
      kind: 'slider',
      id: 'maxDistance',
      label: 'Maximum ray length',
      group: 'Observer',
      apply: 'param',
      min: 1,
      max: 20,
      step: 0.5,
      default: 20,
      unit: 'km',
      help: 'Per-frame cap of the ray length; the compile-time lattice reaches 20 km, longer than the 13.6 km window.'
    },
    {
      kind: 'select',
      id: 'refraction',
      label: 'Earth curvature and refraction',
      group: 'Observer',
      apply: 'param',
      default: 'mt-image',
      help: 'Terrain at distance d is lowered by c d², c = (1 - k) / 2R. The skyline angle includes it.',
      options: [
        {value: 'mt-image', label: 'k = 0.13 (geodetic default)'},
        {value: 'gdal', label: 'k = 1/7 (GDAL -cc 0.85714)'},
        {value: 'none', label: 'Flat earth'}
      ]
    },
    {
      kind: 'select',
      id: 'projection',
      label: 'Projection model',
      group: 'Skyline model',
      apply: 'compile',
      default: 'web-mercator',
      help: 'Web Mercator follows great-circle rays on a sphere over the world-pixel grid, so the 9.6 m pixel and the 6.6 m ground size are handled by the projection. Planar treats the grid as a local metric raster with one ground cell size.',
      options: [
        {value: 'web-mercator', label: 'Web Mercator (great-circle rays)'},
        {value: 'planar', label: 'Planar (one ground cell size)'}
      ]
    },
    {
      kind: 'select',
      id: 'traversal',
      label: 'Traversal',
      group: 'Skyline model',
      apply: 'compile',
      default: 'pyramid',
      help: 'Pyramid skips samples that a shared min-max pyramid proves cannot raise the skyline; march evaluates every sample. Bit-identical results; the "Samples evaluated" readout shows the saving.',
      options: [
        {value: 'pyramid', label: 'Pyramid (min-max skip)'},
        {value: 'march', label: 'March (every sample)'}
      ]
    },
    {
      kind: 'select',
      id: 'azimuthCount',
      label: 'Azimuth divisions',
      group: 'Skyline model',
      apply: 'compile',
      default: '720',
      help: 'Rays per full circle: 720 is a ray every half degree. More rays resolve narrow summits and cost proportionally more.',
      options: [
        {value: '180', label: '180 (2°)'},
        {value: '360', label: '360 (1°)'},
        {value: '720', label: '720 (0.5°)'},
        {value: '1440', label: '1440 (0.25°)'}
      ]
    },
    {
      kind: 'select',
      id: 'sector',
      label: 'Sector',
      group: 'Skyline model',
      apply: 'compile',
      default: 'full',
      help: 'Cast all rays, or only the half circle from south through west to north (azimuths 180 to 360), which holds the Matterhorn. The sector covers firstAzimuth and azimuthSpan of the contributor.',
      options: [
        {value: 'full', label: 'Full circle'},
        {value: 'west', label: 'West half (180° to 360°)'}
      ]
    },
    {
      kind: 'select',
      id: 'heightReference',
      label: 'Height reference',
      group: 'Skyline model',
      apply: 'compile',
      default: 'ground',
      help: 'Ground: the eye is the ground plus the eye height. Absolute: the eye height value is the eye elevation itself (the scene passes ground plus eye height).',
      options: [
        {value: 'ground', label: 'Above the ground'},
        {value: 'absolute', label: 'Absolute elevation'}
      ]
    },
    {
      kind: 'select',
      id: 'peakWindow',
      label: 'Peak prominence window',
      group: 'Skyline peaks',
      apply: 'compile',
      default: '16',
      help: 'Half-window of the prominence walk, in rays: 16 rays at 0.5° is 8° each side. Compile-time because it bounds the loops.',
      options: [
        {value: '8', label: '8 rays'},
        {value: '16', label: '16 rays'},
        {value: '32', label: '32 rays'},
        {value: '64', label: '64 rays'}
      ]
    },
    {
      kind: 'slider',
      id: 'minProminence',
      label: 'Minimum skyline prominence',
      group: 'Skyline peaks',
      apply: 'param',
      min: 0,
      max: 3,
      step: 0.1,
      default: 0.3,
      unit: '°',
      help: 'A skyline peak must stand out from the lower of its two surrounding valleys by this many degrees. 0 accepts every local maximum.'
    },
    {
      kind: 'slider',
      id: 'toleranceDegrees',
      label: 'Occlusion tolerance',
      group: 'Peak visibility',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.01,
      default: 0.02,
      unit: '°',
      help: 'A peak is hidden if it sits more than this angle below the occluding ridge and marginal if within it. Widened by sigmaZ / distance.'
    },
    {
      kind: 'slider',
      id: 'sigmaZ',
      label: 'Vertical standard error',
      group: 'Peak visibility',
      apply: 'param',
      min: 0,
      max: 50,
      step: 1,
      default: 5,
      unit: 'm',
      help: 'DEM height error in metres; it widens the tolerance by sigmaZ / distance radians, so far peaks get a wider benefit of the doubt.'
    },
    {
      kind: 'slider',
      id: 'skylineToleranceDegrees',
      label: 'On-the-skyline tolerance',
      group: 'Peak visibility',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.01,
      default: 0.05,
      unit: '°',
      help: 'A visible peak counts as on the skyline when it is within this angle of the skyline at its bearing.'
    },
    {
      kind: 'slider',
      id: 'targetIgnoreDistance',
      label: 'Ignore the last stretch before the peak',
      group: 'Peak visibility',
      apply: 'param',
      min: 0,
      max: 1000,
      step: 25,
      default: 150,
      unit: 'm',
      help: 'The occlusion test stops this far before the peak, so its own flank cannot hide it (mt-image uses 150 m).'
    },
    {
      kind: 'slider',
      id: 'targetIgnoreFraction',
      label: 'Ignore the last fraction of the distance',
      group: 'Peak visibility',
      apply: 'param',
      min: 0,
      max: 0.1,
      step: 0.005,
      default: 0,
      format: value => `${(value * 100).toFixed(1)} %`,
      help: 'Added to the fixed stretch above, as a fraction of the peak distance.'
    },
    {
      kind: 'select',
      id: 'gridDirections',
      label: 'Grid horizon sectors',
      group: 'Grid horizon (every cell)',
      apply: 'compile',
      default: '16',
      help: 'Azimuth sectors of GPUTerrainHorizon, one compute node each: 4 to 64. More sectors give a smoother sky-view factor.',
      options: [
        {value: '8', label: '8'},
        {value: '16', label: '16'},
        {value: '32', label: '32'},
        {value: '64', label: '64'}
      ]
    },
    {
      kind: 'select',
      id: 'gridRadius',
      label: 'Grid horizon search radius',
      group: 'Grid horizon (every cell)',
      apply: 'compile',
      default: '256',
      help: 'How many pixels each ray searches (26.6 m per pixel on this grid). Cells near the edge see a truncated horizon.',
      options: [
        {value: '64', label: '64 px (1.7 km)'},
        {value: '128', label: '128 px (3.4 km)'},
        {value: '256', label: '256 px (6.8 km)'}
      ]
    },
    {
      kind: 'select',
      id: 'gridAlgorithm',
      label: 'Grid horizon algorithm',
      group: 'Grid horizon (every cell)',
      apply: 'compile',
      default: 'march',
      help: 'March: bounded ray march per sector. Sweep: exact upper-hull sweep on digital lines, amortised O(1) per cell, which wins at large radii.',
      options: [
        {value: 'march', label: 'Ray march'},
        {value: 'sweep', label: 'Sweep (digital lines)'}
      ]
    },
    {
      kind: 'select',
      id: 'base',
      label: 'Terrain base',
      group: 'Display',
      apply: 'param',
      default: 'hillshade',
      help: 'Hillshade, or the grid-wide sky-view factor or positive openness from GPUTerrainHorizon.',
      options: [
        {value: 'hillshade', label: 'Hillshade'},
        {value: 'sky-view', label: 'Sky-view factor (GPUTerrainHorizon)'},
        {value: 'openness', label: 'Positive openness (GPUTerrainHorizon)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showSkyline',
      label: 'Skyline ring',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'One dot per ray where the skyline is hit, coloured by elevation angle; white dots mark the skyline peaks.'
    },
    {
      kind: 'toggle',
      id: 'showPeaks',
      label: 'Catalogue peaks',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Peaks classified by GPUPointHorizonVisibility. Hover one for its name, distance and angle.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time the skyline graph',
      group: 'Compare',
      help: 'Runs the skyline and peaks graph outside the frame and reports GPU time. Switch the traversal and press again to compare.'
    }
  ],

  readouts: [
    {id: 'grid', label: 'Raster'},
    {
      id: 'catalogue',
      label: 'Peaks tested',
      help: 'Named catalogue peaks (snapped to the DEM summit) plus the highest unnamed summits.'
    },
    {
      id: 'highest',
      label: 'Highest skyline',
      help: 'The largest elevation angle over all azimuths.'
    },
    {
      id: 'panorama',
      label: 'Skyline panorama',
      help: 'Highest angle per azimuth band, left to right, as bars. Compare it with the ring on the map.'
    },
    {
      id: 'skylinePeaks',
      label: 'Skyline peaks',
      help: 'Peaks of the circular skyline from GPUProfilePeaks: azimuth and elevation angle, most prominent first.'
    },
    {id: 'peaksVisible', label: 'Visible peaks'},
    {id: 'peaksMarginal', label: 'Marginal peaks'},
    {id: 'peaksHidden', label: 'Peaks behind a ridge'},
    {id: 'peaksOnSkyline', label: 'On the skyline'},
    {
      id: 'samples',
      label: 'Samples evaluated',
      help: 'The debug counter of the skyline contributor: how many lattice samples were evaluated. Pyramid skips most.'
    },
    {
      id: 'gridCompare',
      label: 'Grid horizon vs skyline',
      help: 'The horizon of the 512 × 512 grid at the observer cell against the point skyline in the same directions.'
    },
    {id: 'drop', label: 'Curvature drop'},
    {id: 'timing', label: 'Skyline graph'}
  ],

  legends: state => {
    const legends: LegendSpec[] = [];
    if (state.showSkyline) {
      legends.push({
        kind: 'ramp',
        id: 'skyline',
        title: 'Skyline elevation angle',
        ramp: 'inferno',
        extent: 'gpu',
        unit: '° above the horizontal',
        format: value => value.toFixed(0)
      });
    }
    if (state.showPeaks) {
      legends.push({
        kind: 'categories',
        title: 'Catalogue peak',
        entries: [
          {color: [40, 205, 105, 255], label: 'Visible'},
          {color: [255, 175, 45, 255], label: 'Marginal'},
          {color: [150, 70, 90, 255], label: 'Behind a ridge'},
          {color: [255, 255, 255, 255], label: 'White ring: on the skyline'}
        ]
      });
    }
    if (state.base === 'sky-view') {
      legends.push({
        kind: 'ramp',
        title: 'Sky-view factor',
        ramp: 'cividis',
        extent: [0.5, 1],
        labels: ['0.5 enclosed', '1 open sky']
      });
    } else if (state.base === 'openness') {
      legends.push({
        kind: 'ramp',
        title: 'Positive openness',
        ramp: 'magma',
        extent: [60, 90],
        unit: '°',
        format: value => value.toFixed(0)
      });
    }
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUPointHorizonProfile,
  GPUPointHorizonVisibility,
  GPUProfilePeaks,
  getGPUPointHorizonParameterValues,
  getGPUPointHorizonVisibilityParameterValues,
  getGPUProfilePeaksParameterValues
} from '@luma.gl/experimental/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'skyline'});
graph.add(new GPUPointHorizonProfile({
  width, height, elevation,
  projection: '${state.projection}',       // Web Mercator window of a z13 tile
  traversal: '${state.traversal}',${state.traversal === 'pyramid' ? '\n  pyramid: extrema.output,               // one shared min-max pyramid' : ''}
  azimuthCount: ${state.azimuthCount}, maximumDistance: 20000, cellSize: groundCellSize,
  observers,                                 // float32x4 [column, row, eyeHeight, 0]
  settings: horizonSettings.importToGraph(graph),
  skylineAngle, distance, samples
}));
graph.add(new GPUProfilePeaks({               // peaks of the circular skyline
  values: skylineAngle, offsets, settings: peakSettings.importToGraph(graph),
  window: ${state.peakWindow}, wrap: ${state.sector === 'full'}, prominence, refinedIndex, refinedValue, peakMask, converged
}));
graph.add(new GPUPointHorizonVisibility({
  width, height, elevation, projection: '${state.projection}', traversal: '${state.traversal}', maximumDistance: 20000, cellSize: groundCellSize,
  observers, targets,                        // float32x4 [column, row, height, observerIndex]
  settings: visibilitySettings.importToGraph(graph),
  visibility, details                        // hidden / visible / marginal; details: angles + onSkyline
}));
const compiled = graph.compile();            // once

horizonSettings.write(getGPUPointHorizonParameterValues({
  worldPixelSize: 512 * 2 ** 13, originY,    // Web Mercator window
  curvatureCoefficient, maximumDistance: ${state.maxDistance * 1000}
}));
peakSettings.write(getGPUProfilePeaksParameterValues({minProminence: ${state.minProminence}}));
visibilitySettings.write(getGPUPointHorizonVisibilityParameterValues({
  worldPixelSize: 512 * 2 ** 13, originY, curvatureCoefficient,
  toleranceDegrees: ${state.toleranceDegrees}, sigmaZ: ${state.sigmaZ}, targetIgnoreDistance: ${state.targetIgnoreDistance}
}));
compiled.encode(commandEncoder, {parameters: undefined}); // when the observer moves`,

  about: {
    what: '`GPUPointHorizonProfile` casts a ray in every direction from one eye and keeps the highest elevation angle it meets: the skyline. `GPUProfilePeaks` finds the peaks of that circular profile, `GPUPointHorizonVisibility` tests catalogue peaks against it, and `GPUTerrainHorizon` computes horizon angles, sky-view factor and openness for every cell of a grid.',
    why: 'It is the computation behind a mountain panorama label: which peaks are visible, which are hidden behind a nearer ridge, and which touch the skyline. The same machinery tells you how enclosed a place is (sky-view factor) and how much sky a solar panel sees.',
    howToRead:
      'The ring of dots is the skyline seen from above: each dot sits where its ray first meets the highest ground, coloured by the elevation angle. White-ringed dots are skyline peaks. Catalogue peaks are green when visible, orange when marginal, wine-red when hidden, and have a white ring when they touch the skyline.'
  },

  create: async ctx => (await import('./horizon.compute')).createHorizon(ctx),

  story: [
    {
      id: 'the-question',
      controls: ['showSkyline', 'showPeaks'],
      readouts: ['catalogue'],
      title: 'Which peaks can I see from Gornergrat?',
      body: 'From the terrace at Gornergrat (about 3,100 m) you look west at the Matterhorn, south at the Breithorn and Pollux, and north across the Mattertal. **Which of the summits in this 13.6 km window can you actually see, and which are hiding behind a nearer ridge?**\n\nThe red dot is the viewpoint. Each coloured dot on the ring is where one ray (every half degree) first meets the highest ground in that direction, coloured by its elevation angle: the **skyline**. Green peaks are visible, wine-red peaks are behind a ridge. Hover a peak, then drag the viewpoint. Toggle **Skyline ring** and **Catalogue peaks** below to look at each layer on its own.',
      camera: {longitude: 7.738, latitude: 45.984, zoom: 11.7, transitionMs: 900}
    },
    {
      id: 'skyline',
      controls: ['azimuthCount', 'refraction'],
      readouts: ['panorama', 'highest', 'drop'],
      title: 'The skyline is a maximum over distance',
      body: '**`GPUPointHorizonProfile`** marches each ray over an exact power-of-two distance lattice with bilinear heights and keeps the largest apparent tangent `(h - h_eye) / d - c d`, where `c d` is the curvature drop. The angle is its arctangent. Ray positions follow great circles on a sphere over the Web Mercator world-pixel grid, so the 9.6 m pixel and the 6.6 m ground size are handled by the projection, not by you.\n\nThe **Skyline panorama** readout is the same data as bars: highest angle per azimuth band, left to right. The tall bars are the Matterhorn (west) and the Breithorn and Pollux (south).\n\nChange **Azimuth divisions** below to see the panorama sharpen or blur, and **Earth curvature and refraction** to see how the **Curvature drop** `c d` lowers distant ridges.',
      camera: {longitude: 7.738, latitude: 45.984, zoom: 12, transitionMs: 1200},
      highlight: {readout: 'panorama'}
    },
    {
      id: 'skyline-peaks',
      controls: ['minProminence', 'peakWindow'],
      readouts: ['skylinePeaks'],
      title: 'Peaks of the skyline',
      body: '**`GPUProfilePeaks`** treats the 720-ray skyline as a circular 1-D profile and finds its local maxima. Each candidate must stand out from the lower of its two sides over a window by the **Minimum skyline prominence**, its position is refined with a parabola, and a greedy non-maximum suppression (run exactly, in parallel rounds) removes lesser peaks nearby. These are the white-ringed dots.\n\nThe step starts at 1.5°, where only the great horns survive; lower **Minimum skyline prominence** below to 0 and every notch counts. The **Peak prominence window** is compile-time; the prominence is a parameter.',
      options: {minProminence: 1.5},
      highlight: {readout: 'skylinePeaks'}
    },
    {
      id: 'classify-peaks',
      controls: ['toleranceDegrees'],
      readouts: ['peaksVisible', 'peaksMarginal', 'peaksHidden'],
      title: 'Visible, marginal, hidden, on the skyline',
      body: '**`GPUPointHorizonVisibility`** (the classifier of mt-image) marches the ray to each catalogue peak, remembers the highest angle `αOcc` met *before* the last stretch, and compares it with the peak angle `αP`: **hidden** if `αP < αOcc - tol`, **marginal** if within `tol`, else **visible**, where `tol = tolerance + (σz / d)`. A visible peak that is within the skyline tolerance of the skyline at its bearing is **on the skyline**.\n\nWiden the **Occlusion tolerance** below to 0.3° and watch peaks move between hidden and marginal. Hover one for the exact angles.',
      options: {minProminence: 0.3, toleranceDegrees: 0.3},
      highlight: {readout: 'peaksHidden'}
    },
    {
      id: 'move-the-eye',
      controls: ['observerHeight', 'projection'],
      readouts: ['peaksVisible'],
      title: 'Move the eye: from the terrace to a mast',
      body: 'Raise the **Eye height** below to 100 m, or drag the viewpoint down into the Gorner glacier valley. Peaks that were hidden by the nearest ridges emerge, and the skyline drops. Click anywhere on the map to jump there.\n\nTry the **Projection model** below: with Web Mercator great-circle rays the contributor handles the stretch of the pixel grid; with planar it uses one ground cell size. Over 14 km the angles agree to a few hundredths of a degree, which is why the planar model is a safe shortcut for a window this small and not for a whole country.',
      options: {observerHeight: 100, toleranceDegrees: 0.02},
      highlight: {readout: 'peaksVisible'}
    },
    {
      id: 'grid-horizon',
      controls: ['base', 'gridRadius'],
      readouts: ['gridCompare'],
      title: 'The horizon for every cell: sky-view factor',
      body: '**`GPUTerrainHorizon`** computes the same kind of horizon for *every* cell of a grid, here the DEM averaged to 512 × 512, in 16 azimuth sectors with a search radius you choose (6.8 km here). From it come the **sky-view factor** `1 - mean(sin max(h, 0))` (how much sky a cell sees) and **openness**. Switch **Terrain base** below to see enclosed valleys in dark and open ridges in bright.\n\nThe **Grid horizon vs skyline** readout compares the 16 sector horizons at the observer cell with the point skyline in the same directions: the gap is the radius limit, the coarser grid and the eye height. From a high perch it is large: the point skyline sees the Matterhorn at 9.7 km, a 6.8 km search cannot. Drop the **Grid horizon search radius** to 64 px and the gap grows.',
      options: {observerHeight: 2, base: 'sky-view'},
      highlight: {readout: 'gridCompare'}
    },
    {
      id: 'limits-and-ideas',
      controls: ['traversal', 'measure', 'sector', 'gridAlgorithm'],
      readouts: ['samples', 'timing'],
      title: 'Limits, and things to try',
      body: 'The catalogue is small and hand-made: eight named peaks, snapped to the DEM summit within 160 m as OpenStreetMap peak nodes would be, plus the highest unnamed summits. Peaks beyond the window (Monte Rosa, Dent Blanche) cannot appear, and a skyline from a point near the window edge is cut where the data ends. Glaciers and clouds are not occluders; trees and buildings are not in the surface.\n\n**Try:** set **Traversal** to *March* and press **Time the skyline graph**, then switch back to *Pyramid* and compare both the time and the **Samples evaluated** readout; choose the *West half* **Sector**; set **Grid horizon algorithm** to *Sweep* with a 256 px radius.',
      options: {base: 'hillshade'}
    }
  ]
});
