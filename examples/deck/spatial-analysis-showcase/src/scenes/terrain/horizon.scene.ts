// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import {
  DEFAULT_BEARING_DEGREES,
  formatBearing,
  getHorizonLegends,
  type HorizonOptions
} from './horizon-style';
import {getTerrainFurniture, terrainCartouche, TERRAIN_FRAMES} from './terrain-furniture';

/**
 * The frame of every step: the whole cirque around the eye, the wide DEM window inset a little, so
 * the far peaks and the dashed "data ends here" frame are inside it. Fitted to the free map area.
 */
const CIRQUE_BOUNDS = [7.59, 45.87, 7.97, 46.11] as const;

/** The title cartouche of a step: its question, with a fallback subtitle the scene replaces live. */
const furnitureFor = (question: string, subtitle: string) =>
  getTerrainFurniture({cartouche: terrainCartouche(question, subtitle)});

/**
 * "Which peaks can I see from Gornergrat?" A 360 degree skyline from a draggable eye, the peaks of
 * that skyline, a visible / marginal / hidden verdict for every named summit, and the same horizon
 * for every cell. The panorama chart is the hero; the GPU work lives in `horizon.compute.ts`.
 */
export default defineScene<HorizonOptions>({
  id: 'horizon',
  title: 'Which peaks can I see from Gornergrat?',
  chapter: 'terrain',
  order: 6,
  summary:
    'The 360 degree skyline from a draggable eye above Zermatt as a panorama, the named summits of the Monte Rosa, Matterhorn and Weisshorn cirque classified as visible, marginal or hidden, and the sky-view factor of every cell.',
  contributors: [
    'GPUPointHorizonProfile',
    'GPUPointHorizonVisibility',
    'GPUProfilePeaks',
    'GPUTerrainHorizon',
    'GPURasterExtremaPyramid'
  ],
  datasets: [
    {id: 'alps-dem-wide', role: 'terrain (Terrarium, Web Mercator)'},
    {id: 'alps-context', role: 'named peaks and glaciers (OpenStreetMap)'}
  ],
  initialView: {...TERRAIN_FRAMES.gornergratWide, zoom: 10.9},
  basemap: ground('relief'),
  furniture: furnitureFor(
    'Which peaks can I see from Gornergrat?',
    'Elevation angle of the skyline'
  ),

  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Map layers',
      group: 'Display',
      apply: 'param',
      default: 'plain',
      help: 'Which stage of the computation the map shows: the peaks alone, the rays and the skyline ring, the skyline peaks, the classified peaks with one cast ray, or the sky-view of every cell.',
      options: [
        {value: 'plain', label: 'Peaks only'},
        {value: 'rays', label: 'Rays and skyline'},
        {value: 'skyline', label: 'Skyline'},
        {value: 'skyline-peaks', label: 'Skyline peaks'},
        {value: 'visibility', label: 'Peak visibility'},
        {value: 'sky-view', label: 'Sky-view of every cell'}
      ]
    },
    {
      kind: 'slider',
      id: 'observerHeight',
      label: 'Eye height',
      group: 'Observer',
      apply: 'param',
      min: 2,
      max: 100,
      step: 1,
      default: 2,
      unit: 'm',
      marks: [
        {value: 2, label: 'Standing'},
        {value: 40, label: 'Mast'}
      ],
      help: 'Height of the eye above the ground under it. Drag the gold eye on the map to move the viewpoint itself.'
    },
    {
      kind: 'slider',
      id: 'bearing',
      label: 'Ray bearing',
      group: 'Observer',
      apply: 'param',
      min: 180,
      max: 540,
      step: 0.5,
      default: DEFAULT_BEARING_DEGREES,
      format: value => formatBearing(value),
      marks: [
        {value: 180, label: 'S'},
        {value: 270, label: 'W'},
        {value: 360, label: 'N'},
        {value: 450, label: 'E'},
        {value: 540, label: 'S'}
      ],
      help: 'Bearing of the ray the visibility view casts, degrees clockwise from north. It runs from south round to south, like the panorama. Click a peak or the panorama to set it.'
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
      help: 'Rays stop at this distance (the compile-time lattice reaches 20 km). Peaks beyond it are out of range.'
    },
    {
      kind: 'select',
      id: 'azimuthCount',
      label: 'Rays per circle',
      group: 'Skyline',
      apply: 'compile',
      display: 'segmented',
      default: '720',
      help: 'Rays cast over the full circle: 720 is one every half degree. More rays resolve narrow summits and cost proportionally more; the graph is rebuilt.',
      options: [
        {value: '180', label: '180'},
        {value: '720', label: '720'},
        {value: '1440', label: '1440'}
      ]
    },
    {
      kind: 'slider',
      id: 'minProminence',
      label: 'Minimum prominence',
      group: 'Skyline',
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
      help: 'A peak is hidden if its angle is more than this below the highest ground in front of it, marginal if within it. The tolerance is widened by the vertical error divided by the distance.'
    },
    {
      kind: 'select',
      id: 'gridRadius',
      label: 'Search radius',
      group: 'Sky-view (every cell)',
      apply: 'compile',
      display: 'segmented',
      default: '256',
      help: 'How many grid steps each horizon ray searches. A short search misses the high ground beyond it, so cells see more sky than they should; cells near the edge see a truncated horizon.',
      options: [
        {value: '64', label: '64 steps'},
        {value: '128', label: '128 steps'},
        {value: '256', label: '256 steps'}
      ]
    },
    {
      kind: 'select',
      id: 'gridDirections',
      label: 'Sectors',
      group: 'Sky-view (every cell)',
      apply: 'compile',
      display: 'segmented',
      default: '16',
      help: 'Azimuth sectors of GPUTerrainHorizon, one compute node each. More sectors give a smoother sky-view factor.',
      options: [
        {value: '8', label: '8'},
        {value: '16', label: '16'},
        {value: '32', label: '32'}
      ]
    },
    {
      kind: 'select',
      id: 'refraction',
      label: 'Earth curvature and refraction',
      group: 'Engine',
      apply: 'param',
      default: 'mt-image',
      expert: true,
      help: 'Terrain at distance d is lowered by c d squared, c = (1 - k) / 2R. Owned by the viewshed story; here it moves a skyline by hundredths of a degree.',
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
      group: 'Engine',
      apply: 'compile',
      default: 'web-mercator',
      expert: true,
      help: 'Web Mercator follows great-circle rays over the world-pixel grid, so the pixel size and the ground size are handled by the projection. Planar treats the grid as a local metric raster with one ground cell size; over this window the two agree to hundredths of a degree.',
      options: [
        {value: 'web-mercator', label: 'Web Mercator (great-circle rays)'},
        {value: 'planar', label: 'Planar (one ground cell size)'}
      ]
    },
    {
      kind: 'select',
      id: 'traversal',
      label: 'Traversal',
      group: 'Engine',
      apply: 'compile',
      default: 'pyramid',
      expert: true,
      help: 'Pyramid skips samples that a shared min-max pyramid proves cannot raise the skyline; march evaluates every sample. Bit-identical results; the Samples evaluated readout shows the saving.',
      options: [
        {value: 'pyramid', label: 'Pyramid (min-max skip)'},
        {value: 'march', label: 'March (every sample)'}
      ]
    },
    {
      kind: 'select',
      id: 'peakWindow',
      label: 'Peak prominence window',
      group: 'Engine',
      apply: 'compile',
      default: '16',
      expert: true,
      help: 'Half-window of the prominence walk, in rays: 16 rays at half a degree is 8 degrees each side. Compile-time because it bounds the loops.',
      options: [
        {value: '8', label: '8 rays'},
        {value: '16', label: '16 rays'},
        {value: '32', label: '32 rays'},
        {value: '64', label: '64 rays'}
      ]
    },
    {
      kind: 'slider',
      id: 'sigmaZ',
      label: 'Vertical standard error',
      group: 'Engine',
      apply: 'param',
      min: 0,
      max: 50,
      step: 1,
      default: 5,
      unit: 'm',
      expert: true,
      help: 'DEM height error in metres; it widens the tolerance by the error divided by the distance, so far peaks get a wider benefit of the doubt.'
    },
    {
      kind: 'slider',
      id: 'skylineToleranceDegrees',
      label: 'On-the-skyline tolerance',
      group: 'Engine',
      apply: 'param',
      min: 0,
      max: 1,
      step: 0.01,
      default: 0.05,
      unit: '°',
      expert: true,
      help: 'A visible peak counts as on the skyline when it is within this angle of the skyline at its bearing.'
    },
    {
      kind: 'slider',
      id: 'targetIgnoreDistance',
      label: 'Ignore the last stretch before the peak',
      group: 'Engine',
      apply: 'param',
      min: 0,
      max: 1000,
      step: 25,
      default: 150,
      unit: 'm',
      expert: true,
      help: 'The occlusion test stops this far before the peak, so its own flank cannot hide it (mt-image uses 150 m).'
    },
    {
      kind: 'slider',
      id: 'targetIgnoreFraction',
      label: 'Ignore the last fraction of the distance',
      group: 'Engine',
      apply: 'param',
      min: 0,
      max: 0.1,
      step: 0.005,
      default: 0,
      expert: true,
      format: value => `${(value * 100).toFixed(1)} %`,
      help: 'Added to the fixed stretch above, as a fraction of the peak distance.'
    },
    {
      kind: 'select',
      id: 'gridAlgorithm',
      label: 'Grid horizon algorithm',
      group: 'Engine',
      apply: 'compile',
      default: 'march',
      expert: true,
      help: 'March: bounded ray march per sector. Sweep: exact upper-hull sweep on digital lines, amortised constant cost per cell, which wins at large radii.',
      options: [
        {value: 'march', label: 'Ray march'},
        {value: 'sweep', label: 'Sweep (digital lines)'}
      ]
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time the skyline graph',
      group: 'Engine',
      expert: true,
      help: 'Runs the skyline and peaks graph outside the frame and reports GPU time. Switch the traversal and press again to compare.'
    }
  ],

  readouts: [
    {
      id: 'peaksInRange',
      format: 'integer',
      label: 'Named summits in range',
      emphasis: 'tile',
      help: 'Named OpenStreetMap summits above the height floor, one per massif, within the ray length.'
    },
    {
      id: 'highestPeak',
      label: 'Highest of them',
      help: 'Name and published elevation of the highest summit in range.'
    },
    {
      id: 'panorama',
      label: 'Panorama from the eye',
      kind: 'chart',
      placement: 'map',
      mapCorner: 'bottom-right',
      help: 'Elevation angle against bearing, south round to south. The filled curve is the skyline; in the visibility view visible peaks touch it and hidden peaks sit below it. Click it to cast a ray.'
    },
    {
      id: 'highestAngle',
      label: 'Steepest skyline angle',
      emphasis: 'tile',
      help: 'The largest elevation angle over all bearings, and the compass point it lies in.'
    },
    {
      id: 'rayCount',
      label: 'Rays cast',
      emphasis: 'tile',
      help: 'Rays over the full circle, one per bearing step.'
    },
    {
      id: 'visibleCount',
      format: 'integer',
      label: 'Visible peaks',
      emphasis: 'tile',
      help: 'Summits whose angle clears the highest ground in front of them by more than the tolerance.'
    },
    {
      id: 'hiddenCount',
      format: 'integer',
      label: 'Hidden peaks',
      emphasis: 'tile',
      help: 'Summits below the highest ground in front of them by more than the tolerance.'
    },
    {id: 'marginalCount', label: 'Marginal peaks', format: 'integer', hood: true},
    {id: 'onSkylineCount', label: 'Peaks on the skyline', format: 'integer', hood: true},
    {
      id: 'rayProfile',
      label: 'Angle along the ray',
      kind: 'chart',
      help: 'Elevation angle of the ground at every distance along the cast ray, the highest angle so far (it only rises) and the ridge that sets it. A peak is hidden when its angle guide sits below the curve before the peak.'
    },
    {
      id: 'rayVerdict',
      label: 'The cast ray',
      layout: 'block',
      help: 'What the GPU found for the peak on the cast ray, or what sets the skyline at its bearing.'
    },
    {
      id: 'skylinePeaks',
      label: 'Skyline peaks by prominence',
      kind: 'chart',
      help: 'The maxima of the skyline from GPUProfilePeaks, ranked by prominence in degrees. Click a bar to mark that peak on the map.'
    },
    {
      id: 'skylinePeakCount',
      format: 'integer',
      label: 'Skyline peaks',
      emphasis: 'tile',
      help: 'Maxima of the skyline that stand out by at least the minimum prominence.'
    },
    {
      id: 'skyViewHere',
      label: 'Sky-view at the eye',
      emphasis: 'tile',
      help: 'Sky-view factor of the grid cell under the eye: 1 is a flat horizon, lower is enclosed.'
    },
    {
      id: 'gridCompare',
      label: 'Grid horizon vs point skyline',
      help: 'The grid horizon at the eye cell against the point skyline in the same directions: the gap is the search radius, the coarser grid and the eye height.'
    },
    {id: 'eyeHeight', label: 'Eye height', hood: true},
    {id: 'rayStep', label: 'Ray spacing', hood: true},
    {id: 'peakFloor', label: 'Height floor of the catalogue', hood: true},
    {id: 'gridDirectionsReadout', label: 'Grid horizon sectors', hood: true},
    {id: 'gridRadiusReadout', label: 'Grid horizon search radius', hood: true},
    {id: 'catalogue', label: 'Peak catalogue', hood: true},
    {id: 'grid', label: 'Raster', hood: true},
    {
      id: 'peakSuppression',
      label: 'Skyline peak suppression',
      hood: true,
      help: 'Whether the parallel non-maximum suppression of GPUProfilePeaks converged.'
    },
    {
      id: 'samples',
      label: 'Samples evaluated',
      hood: true,
      help: 'The debug counter of the skyline contributor: how many lattice samples were evaluated. Pyramid skips most.'
    },
    {id: 'drop', label: 'Curvature drop', hood: true},
    {id: 'timing', label: 'Skyline graph', hood: true}
  ],

  pipeline: [
    {
      id: 'rays',
      label: 'Rays',
      detail: 'One ray per bearing step from the eye, marched over a distance lattice',
      show: {option: 'view', value: 'rays'}
    },
    {
      id: 'skyline',
      label: 'Skyline',
      detail: 'The steepest elevation angle each ray meets, with curvature c d squared',
      show: {option: 'view', value: 'skyline'}
    },
    {
      id: 'peaks',
      label: 'Peaks',
      detail: 'Maxima of the circular skyline ranked by prominence',
      show: {option: 'view', value: 'skyline-peaks'}
    },
    {
      id: 'visibility',
      label: 'Visibility',
      detail: 'Each peak angle against the highest ground in front of it, with a tolerance',
      show: {option: 'view', value: 'visibility'}
    },
    {
      id: 'sky-view',
      label: 'Sky-view',
      detail: 'The same horizon for every cell, folded into the share of sky it sees',
      show: {option: 'view', value: 'sky-view'}
    }
  ],

  legends: getHorizonLegends,

  story: [
    {
      id: 'the-view',
      title: 'A ring of four-thousanders around the terrace',
      headline: 'Peaks ring the terrace in every direction',
      textAlternative:
        'Relief map of the Gornergrat cirque centred on a gold eye, with the Matterhorn, Dufourspitze, Weisshorn, Dom and other named summits marked by ink triangles and a dashed frame where the data ends.',
      body: 'From the terrace at Gornergrat the horizon is a ring of summits. **{{peaksInRange}}** named peaks above **{{peakFloor}}** lie in range, the highest **{{highestPeak}}**. Which can you actually see, and which hide behind a nearer ridge? The gold eye is you; drag it. The relief is the whole cirque, cut off where the data ends.\n\n*Ask the question before running the method.*',
      optionsMode: 'fresh',
      options: {view: 'plain'},
      controls: [],
      readouts: ['peaksInRange', 'highestPeak'],
      camera: {bounds: CIRQUE_BOUNDS, pitch: 0, bearing: 0, transitionMs: 1400},
      stage: 'rays',
      furniture: furnitureFor(
        'Which peaks can I see from Gornergrat?',
        'Named summits, one per massif'
      ),
      highlight: {readout: 'peaksInRange'}
    },
    {
      id: 'rays',
      title: 'Every half degree, keep the steepest angle',
      headline: 'Each ray keeps only its steepest angle',
      textAlternative:
        'The same map with a ring of skyline segments coloured by elevation angle in blue and green classes, a fan of thin rays from the gold eye, and a panorama inset drawing the skyline as a filled curve against bearing.',
      body: 'From the eye **{{rayCount}}** rays fan out, one every **{{rayStep}}**. Along each, `GPUPointHorizonProfile` keeps only the steepest elevation angle it meets: the skyline. The ring is where those angles occur; the panorama unrolls it. The steepest is **{{highestAngle}}**. Change **Rays per circle**: coarse rays miss narrow summits.\n\n*A skyline is a maximum of angles, not of heights.*',
      optionsMode: 'fresh',
      options: {view: 'rays'},
      controls: ['azimuthCount'],
      readouts: ['panorama', 'highestAngle', 'rayCount'],
      camera: {bounds: CIRQUE_BOUNDS, pitch: 0, bearing: 0, transitionMs: 1200},
      stage: 'skyline',
      furniture: furnitureFor('How high does the skyline stand?', 'Elevation angle of the skyline'),
      highlight: {readout: 'panorama'}
    },
    {
      id: 'hidden-behind',
      title: 'A near ridge can hide a higher peak',
      headline: 'A nearer ridge can hide a taller peak',
      textAlternative:
        'Summits drawn as filled blue circles where visible and hollow grey rings where hidden, a solid ray from the eye to the ridge that hides the selected peak and a dashed ray beyond it, a panorama with hidden peaks below the skyline and a profile of angle against distance.',
      body: 'A peak is visible only if its own elevation angle beats the highest angle in front of it. Click a peak or the panorama to cast its ray: **{{rayVerdict}}**. Now **{{visibleCount}}** summits are visible and **{{hiddenCount}}** hidden. Widen **Occlusion tolerance** and borderline peaks turn marginal.\n\n*Visibility is an angle, not a height.*',
      optionsMode: 'fresh',
      options: {view: 'visibility'},
      controls: ['toleranceDegrees'],
      readouts: ['panorama', 'rayProfile', 'visibleCount', 'hiddenCount'],
      camera: {bounds: CIRQUE_BOUNDS, pitch: 0, bearing: 0, transitionMs: 1200},
      stage: 'visibility',
      furniture: furnitureFor('Which peaks does a ridge hide?', 'Peak visibility'),
      highlight: {readout: 'rayProfile'}
    },
    {
      id: 'skyline-peaks',
      title: "The skyline's own summits, ranked by prominence",
      headline: 'The skyline has summits of its own',
      textAlternative:
        'The skyline ring with ink diamonds at its maxima, a panorama with the highest skyline peaks marked by name, and a ranked bar list of their prominence in degrees.',
      body: '`GPUProfilePeaks` treats the skyline as a closed curve and finds its own maxima. A maximum counts when it stands out from the lower side of its window by **Minimum prominence**; **{{skylinePeakCount}}** survive. Raise the slider and only the great horns remain; lower it to zero and every notch counts. Ink ticks on the ring and markers on the panorama show them.\n\n*Prominence is relative height, here in degrees.*',
      optionsMode: 'fresh',
      options: {view: 'skyline-peaks', minProminence: 1.5},
      controls: ['minProminence'],
      readouts: ['panorama', 'skylinePeaks', 'skylinePeakCount'],
      camera: {bounds: CIRQUE_BOUNDS, pitch: 0, bearing: 0, transitionMs: 1200},
      stage: 'peaks',
      furniture: furnitureFor('Where does the skyline peak?', 'Skyline maxima'),
      highlight: {readout: 'skylinePeaks'}
    },
    {
      id: 'move-the-eye',
      title: 'Climb a mast and the panorama opens',
      headline: 'A higher eye opens the panorama',
      textAlternative:
        'The classified peaks and the panorama for an eye raised on a mast, with the same camera so only the angles change.',
      body: 'Raise **Eye height** and ridges that hid peaks drop below them: at **{{eyeHeight}}**, **{{visibleCount}}** summits are visible. Or drag the gold eye down the Gorner glacier and watch the skyline close in. The camera stays put, so only the angles change.\n\n*Height buys angle, and angle buys sight.*',
      optionsMode: 'fresh',
      options: {view: 'visibility', observerHeight: 40},
      controls: ['observerHeight'],
      readouts: ['panorama', 'visibleCount', 'hiddenCount'],
      camera: {bounds: CIRQUE_BOUNDS, pitch: 0, bearing: 0, transitionMs: 1200},
      stage: 'visibility',
      furniture: furnitureFor('What does a higher eye gain?', 'Peak visibility'),
      highlight: {readout: 'visibleCount'}
    },
    {
      id: 'every-cell',
      title: 'The same trick for every cell: sky-view',
      headline: 'Every cell sees its own share of sky',
      textAlternative:
        'The relief darkened in grey where terrain encloses the sky, valleys darkest and ridges open, with the gold eye and a dashed frame where the data ends.',
      body: "`GPUTerrainHorizon` repeats the horizon search for every cell: **{{gridDirectionsReadout}}** out to **{{gridRadiusReadout}}**. Sky-view factor is the share of sky a cell sees: valleys dark, ridges open. The eye's own cell scores **{{skyViewHere}}**. Shorten **Search radius** and cells see more sky than they should. Curvature and tolerance are taught in the viewshed story.\n\n*A horizon is local: every cell has its own.*",
      optionsMode: 'fresh',
      options: {view: 'sky-view'},
      controls: ['gridRadius', 'gridDirections'],
      readouts: ['skyViewHere', 'gridCompare'],
      camera: {bounds: CIRQUE_BOUNDS, pitch: 0, bearing: 0, transitionMs: 1200},
      stage: 'sky-view',
      furniture: furnitureFor('How much sky does each cell see?', 'Sky-view factor'),
      highlight: {readout: 'skyViewHere'}
    }
  ],

  about: {
    what: '`GPUPointHorizonProfile` casts a ray in every direction from one eye and keeps the highest elevation angle it meets: the skyline. `GPUProfilePeaks` finds the maxima of that circular profile, `GPUPointHorizonVisibility` compares each catalogue peak with the highest ground in front of it, and `GPUTerrainHorizon` computes horizon angles and the sky-view factor for every cell of a grid.',
    why: 'It is the computation behind a mountain panorama label: which peaks are visible, which are hidden behind a nearer ridge, and which touch the skyline. The same machinery says how enclosed a place is (sky-view factor), which matters for solar panels, radio links and cold-air pooling.',
    howToRead:
      'The ring is the skyline seen from above: each segment sits where its ray first meets the highest ground, coloured by the elevation angle. The panorama is the same curve unrolled by bearing. Peaks are filled blue circles when visible, orange rings when marginal and hollow grey rings when hidden; an ink ring marks the peaks on the skyline. The terrain is bare earth from a digital elevation model of swissALTI3D and Copernicus data: trees, buildings, glaciers as surfaces and clouds are not occluders. Named summits are OpenStreetMap peaks above a height floor, one per massif, snapped to the highest DEM cell within 160 m; the DEM cell reads a few metres lower than the published elevation. Drag the gold eye, or click a peak or the panorama to cast a ray.'
  },

  create: async ctx => (await import('./horizon.compute')).createHorizon(ctx),

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUPointHorizonProfile,
  GPUPointHorizonVisibility,
  GPUProfilePeaks,
  GPUTerrainHorizon,
  getGPUPointHorizonParameterValues,
  getGPUPointHorizonVisibilityParameterValues,
  getGPUProfilePeaksParameterValues
} from '@luma.gl/experimental/gpu-terrain';

const graph = new GPUCommandGraph(device, {id: 'skyline'});
graph.add(new GPUPointHorizonProfile({
  width, height, elevation,
  projection: '${state.projection}',       // Web Mercator window of a z12 tile
  traversal: '${state.traversal}',${state.traversal === 'pyramid' ? '\n  pyramid: extrema.output,               // one shared min-max pyramid' : ''}
  azimuthCount: ${state.azimuthCount}, maximumDistance: 20000, cellSize: groundCellSize,
  observers,                                 // float32x4 [column, row, eyeHeight, 0]
  settings: horizonSettings.importToGraph(graph),
  skylineAngle, distance, samples
}));
graph.add(new GPUProfilePeaks({               // maxima of the circular skyline
  values: skylineAngle, offsets, settings: peakSettings.importToGraph(graph),
  window: ${state.peakWindow}, wrap: true, prominence, refinedIndex, refinedValue, peakMask, converged
}));
graph.add(new GPUPointHorizonVisibility({
  width, height, elevation, projection: '${state.projection}', traversal: '${state.traversal}', maximumDistance: 20000, cellSize: groundCellSize,
  observers, targets,                        // float32x4 [column, row, height, observerIndex]
  settings: visibilitySettings.importToGraph(graph),
  visibility, details                        // hidden / visible / marginal; details: angles + onSkyline
}));
graph.add(new GPUTerrainHorizon({             // the same horizon for every cell of a coarser grid
  width: gridWidth, height: gridHeight, elevation: gridElevation, settings: gridSettings.importToGraph(graph),
  directionCount: ${state.gridDirections}, maximumRadius: ${state.gridRadius}, algorithm: '${state.gridAlgorithm}',
  cellSizeMode: 'web-mercator', horizon, skyViewFactor
}));
const compiled = graph.compile();            // once

horizonSettings.write(getGPUPointHorizonParameterValues({
  worldPixelSize: 512 * 2 ** 12, originY,    // Web Mercator window
  curvatureCoefficient, maximumDistance: ${state.maxDistance * 1000}
}));
peakSettings.write(getGPUProfilePeaksParameterValues({minProminence: ${state.minProminence}}));
visibilitySettings.write(getGPUPointHorizonVisibilityParameterValues({
  worldPixelSize: 512 * 2 ** 12, originY, curvatureCoefficient,
  toleranceDegrees: ${state.toleranceDegrees}, sigmaZ: ${state.sigmaZ}, targetIgnoreDistance: ${state.targetIgnoreDistance}
}));
compiled.encode(commandEncoder, {parameters: undefined}); // when the eye moves`
});
