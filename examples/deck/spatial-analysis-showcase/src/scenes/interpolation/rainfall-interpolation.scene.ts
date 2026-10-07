// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {getRainfallSurfaceStyle, RAINFALL_VARIABLES} from './b7-rainfall-style';
import type {RainfallOptions} from './rainfall-interpolation.compute';

const RAMP_OPTIONS = [
  {value: 'viridis', label: 'Viridis'},
  {value: 'cividis', label: 'Cividis (color-blind optimised)'},
  {value: 'magma', label: 'Magma'},
  {value: 'inferno', label: 'Inferno'}
] as const;

/** Chapter `interpolation`, scene 1: Hurricane Helene rain gauges turned into a surface. */
export default defineScene<RainfallOptions>({
  id: 'rainfall-interpolation',
  title: 'Where did Helene’s rain fall between the gauges?',
  chapter: 'interpolation',
  order: 1,
  summary:
    'Six thousand rain gauges from Hurricane Helene become a continuous surface on the GPU, by inverse distance weighting and by ordinary kriging with a variogram fitted on the fly. Map the uncertainty, validate against held-out gauges and summarise the surface with focal statistics.',
  contributors: ['GPUInverseDistanceWeighting', 'GPUKriging', 'GPUVariogram', 'GPUFocalStatistics'],
  datasets: [{id: 'ghcn-stations', role: 'daily gauge readings'}],
  initialView: {longitude: -83, latitude: 32.3, zoom: 5.2},

  options: [
    {
      kind: 'select',
      id: 'variable',
      label: 'Variable',
      group: 'Data',
      apply: 'param',
      default: 'prcp',
      help: 'Which GHCN-Daily reading is interpolated. Switching rewrites the value buffer; nothing is recompiled. Gauges without a reading (NaN) are skipped by every contributor.',
      options: [
        {
          value: 'prcp',
          label: RAINFALL_VARIABLES.prcp.label,
          help: 'Landfall day: 6,135 gauges.'
        },
        {
          value: 'prcpPrevDay',
          label: RAINFALL_VARIABLES.prcpPrevDay.label,
          help: 'The day before landfall: 5,759 gauges. Rain-band arrival ahead of the eye.'
        },
        {
          value: 'tmax',
          label: RAINFALL_VARIABLES.tmax.label,
          help: 'Only 1,100 stations report temperature: a sparse network, so the uncertainty map matters more.'
        }
      ]
    },
    {
      kind: 'toggle',
      id: 'holdOut',
      label: 'Hold out 10% of gauges',
      group: 'Data',
      apply: 'param',
      default: false,
      help: 'Removes a fixed random tenth of the gauges from the interpolation and from the variogram (a mask buffer), and draws them with a ring. The readouts then score both methods at those gauges.'
    },
    {
      kind: 'button',
      id: 'reseed',
      label: 'Draw a new hold-out split',
      group: 'Data',
      help: 'Picks a different random tenth of the gauges. Scores change a little; a stable ranking is more believable.'
    },
    {
      kind: 'select',
      id: 'surface',
      label: 'Surface',
      group: 'Surface',
      apply: 'param',
      default: 'idw',
      help: 'Both interpolators run in one compiled graph; a parameter word picks which result is drawn (and fed to the focal statistics).',
      options: [
        {
          value: 'idw',
          label: 'Inverse distance weighting',
          help: 'Weighted average of nearby gauges, weight 1 / d^p.'
        },
        {
          value: 'kriging',
          label: 'Ordinary kriging',
          help: 'Best linear unbiased prediction under the fitted variogram.'
        },
        {
          value: 'difference',
          label: 'Kriging minus IDW',
          help: 'Where the two methods disagree (diverging ramp, zero is white).'
        },
        {
          value: 'error',
          label: 'Kriging standard error',
          help: 'Square root of the kriging variance: how uncertain the prediction is. It depends on gauge geometry and the variogram, not on the observed values.'
        },
        {
          value: 'support',
          label: 'Gauges used per cell (IDW)',
          help: 'How many gauges contribute to each cell after the radius and nearest-k limits. Cells with fewer than the minimum are no data.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'searchRadiusKm',
      label: 'Search radius',
      group: 'Surface',
      apply: 'param',
      min: 20,
      max: 500,
      step: 10,
      default: 120,
      unit: 'km',
      help: 'Gauges farther than this are ignored by both methods. Cells with too few gauges inside the radius are no data and stay transparent, which is how the map avoids inventing rain over the ocean.'
    },
    {
      kind: 'slider',
      id: 'minimumStations',
      label: 'Minimum gauges per cell',
      group: 'Surface',
      apply: 'param',
      min: 1,
      max: 8,
      step: 1,
      default: 3,
      help: 'A cell with fewer contributing gauges is no data. Kriging always needs at least 3 (a smaller value is raised to 3 for kriging only).'
    },
    {
      kind: 'slider',
      id: 'power',
      label: 'Distance power p',
      group: 'Inverse distance weighting',
      apply: 'param',
      min: 0,
      max: 6,
      step: 0.5,
      default: 2,
      help: 'Weight is 1 / d^p. p = 0 is a plain average of the neighbours (flat plateaus), p = 2 is the common default, large p snaps each cell to its nearest gauge and leaves bullseyes around every one.',
      disabledWhen: state => state.surface === 'kriging' || state.surface === 'error'
    },
    {
      kind: 'slider',
      id: 'neighborCount',
      label: 'Nearest gauges k',
      group: 'Inverse distance weighting',
      apply: 'param',
      min: 0,
      max: 32,
      step: 1,
      default: 12,
      format: value => (value === 0 ? 'all in radius' : String(value)),
      help: 'Keeps only the k nearest gauges inside the radius (compile-time capacity 32). 0 uses every gauge in the radius, which smooths away small features in dense networks.'
    },
    {
      kind: 'select',
      id: 'variogramModel',
      label: 'Variogram model',
      group: 'Variogram',
      apply: 'param',
      default: 'spherical',
      help: 'Curve fitted to the empirical semivariances on the CPU. The fit is written into the kriging parameters, so changing it never recompiles.',
      options: [
        {value: 'spherical', label: 'Spherical', help: 'Reaches the sill at a finite range.'},
        {
          value: 'exponential',
          label: 'Exponential',
          help: 'Rises faster, approaches the sill asymptotically (95% at the range).'
        },
        {
          value: 'gaussian',
          label: 'Gaussian',
          help: 'Very smooth near zero; assumes an extremely continuous field. Can be unstable with a small nugget.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'weighting',
      label: 'Fit weighting',
      group: 'Variogram',
      apply: 'param',
      default: 'cressie',
      help: 'Least-squares weights of the model fit. Cressie (1985) weights N / γ² and is the gstat default; it trusts short lags with many pairs.',
      options: [
        {value: 'cressie', label: 'Cressie (N / γ²)'},
        {value: 'pairs', label: 'Pair counts (N)'},
        {value: 'none', label: 'Ordinary least squares'}
      ]
    },
    {
      kind: 'toggle',
      id: 'robust',
      label: 'Robust (Cressie–Hawkins) estimator',
      group: 'Variogram',
      apply: 'param',
      default: false,
      help: 'Fits the Cressie–Hawkins robust semivariances instead of the Matheron mean of squared differences. Robust bins are less pulled by a few extreme storm totals.'
    },
    {
      kind: 'slider',
      id: 'maximumLagKm',
      label: 'Maximum lag',
      group: 'Variogram',
      apply: 'param',
      min: 100,
      max: 1000,
      step: 50,
      default: 300,
      unit: 'km',
      help: 'Pairs farther apart than this are ignored; the 16 lag bins split [0, max]. A common rule is about half the extent of the study area, but here the rain has a regional trend, so very long lags overshoot the sample variance.'
    },
    {
      kind: 'select',
      id: 'direction',
      label: 'Direction sector (diagnostic)',
      group: 'Variogram',
      apply: 'param',
      default: 'all',
      help: 'GPUVariogram bins pairs into 4 direction sectors of 45°. Pick one to fit its range separately; a different range means the field is anisotropic. GPUKriging itself is isotropic and always uses the omnidirectional fit.',
      options: [
        {value: 'all', label: 'All directions'},
        {value: '0', label: 'Sector 1'},
        {value: '1', label: 'Sector 2'},
        {value: '2', label: 'Sector 3'},
        {value: '3', label: 'Sector 4'}
      ]
    },
    {
      kind: 'slider',
      id: 'azimuth',
      label: 'Sector start angle',
      group: 'Variogram',
      apply: 'param',
      min: 0,
      max: 135,
      step: 15,
      default: 0,
      unit: '°',
      help: 'Rotates the 4 sectors (counter-clockwise from east). Sector 1 covers this angle to +45°.',
      disabledWhen: state => state.direction === 'all'
    },
    {
      kind: 'slider',
      id: 'krigingNeighborCount',
      label: 'Neighbourhood size k',
      group: 'Kriging',
      apply: 'param',
      min: 3,
      max: 16,
      step: 1,
      default: 12,
      help: 'Each cell solves a (k + 1) × (k + 1) kriging system over its k nearest gauges (compile-time capacity 16). Larger k adds distant information and cost; local kriging assumes the variogram holds in the neighbourhood.'
    },
    {
      kind: 'select',
      id: 'focalStatistic',
      label: 'Window statistic',
      group: 'Focal statistics',
      apply: 'param',
      default: 'mean',
      help: 'GPUFocalStatistics computes every statistic over a moving window of cells; this chooses which output buffer is drawn. It only applies when the window radius is above 0.',
      options: [
        {value: 'mean', label: 'Mean', help: 'Smooths the surface.'},
        {
          value: 'max',
          label: 'Maximum',
          help: 'Highest value in the window: the peak-rain footprint.'
        },
        {value: 'min', label: 'Minimum'},
        {
          value: 'range',
          label: 'Range (max − min)',
          help: 'Local contrast: where the surface changes quickly.'
        },
        {
          value: 'standardDeviation',
          label: 'Standard deviation',
          help: 'Local roughness of the surface.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'focalRadius',
      label: 'Window radius',
      group: 'Focal statistics',
      apply: 'param',
      min: 0,
      max: 8,
      step: 1,
      default: 0,
      format: value => (value === 0 ? 'off' : `${value} cells`),
      help: 'Radius of the moving window in raster cells (compile-time cap 8). The raster follows the camera, so a cell is about 5 km at full view and 1 km zoomed in.'
    },
    {
      kind: 'select',
      id: 'focalShape',
      label: 'Window shape',
      group: 'Focal statistics',
      apply: 'param',
      default: 'circle',
      disabledWhen: state => state.focalRadius === 0,
      help: 'A circle treats all directions alike; a square reaches farther along the diagonals.',
      options: [
        {value: 'circle', label: 'Circle'},
        {value: 'square', label: 'Square'}
      ]
    },
    {
      kind: 'slider',
      id: 'focalMinimumCount',
      label: 'Minimum valid cells',
      group: 'Focal statistics',
      apply: 'param',
      min: 1,
      max: 20,
      step: 1,
      default: 1,
      disabledWhen: state => state.focalRadius === 0,
      help: 'A cell whose window holds fewer valid (non no-data) cells becomes no data. Raise it to trim ragged edges.'
    },
    {
      kind: 'toggle',
      id: 'keepHoles',
      label: 'Keep no-data holes',
      group: 'Focal statistics',
      apply: 'param',
      default: true,
      disabledWhen: state => state.focalRadius === 0,
      help: 'On: a no-data centre cell stays no data. Off: windows fill holes and grow the surface outward by the window radius (ArcGIS default).'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time the interpolation and variogram graphs',
      group: 'Compare',
      help: 'Runs both compiled graphs outside the frame and reports GPU time per run.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      help: 'Used for the interpolated value and for the gauge dots, so they can be compared. Spread and error maps use fixed ramps.',
      options: RAMP_OPTIONS
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Surface opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.9,
      help: 'Lower it to read the basemap under the surface.'
    },
    {
      kind: 'toggle',
      id: 'showStations',
      label: 'Show gauges',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Dots use the same ramp as the surface. A ringed dot is a held-out gauge.'
    }
  ],

  readouts: [
    {id: 'stations', label: 'Gauges', help: 'Gauges in the dataset (NaN readings are skipped).'},
    {id: 'training', label: 'Gauges used', help: 'Gauges with a reading that are not held out.'},
    {id: 'heldOut', label: 'Held out', help: 'Gauges removed for validation.'},
    {
      id: 'cellSize',
      label: 'Cell size',
      help: 'Width × height of one raster cell. Follows the camera: the extent buffer is rewritten each frame.'
    },
    {
      id: 'variogramModel',
      label: 'Fitted variogram',
      help: 'Nugget (noise at zero distance), partial sill (variance that is spatially structured) and effective range.'
    },
    {
      id: 'empirical',
      label: 'Empirical γ(h)',
      help: 'GPU semivariance per lag, omnidirectional (bars rise with distance). Read it left to right, from short to long lag.'
    },
    {
      id: 'fitted',
      label: 'Fitted model γ(h)',
      help: 'The fitted curve at the same lags and the same vertical scale.'
    },
    {id: 'lagAxis', label: 'Lags'},
    {
      id: 'sampleVariance',
      label: 'Sample variance',
      help: 'Variance of the gauge values: the sill a stationary field should approach.'
    },
    {id: 'sectorRange', label: 'Direction sector'},
    {
      id: 'validationIdw',
      label: 'IDW at held-out gauges',
      help: 'Root-mean-square error, mean absolute error and bias of the displayed raster at held-out gauges on screen.'
    },
    {id: 'validationKriging', label: 'Kriging at held-out gauges'},
    {id: 'validationVerdict', label: 'Verdict'},
    {id: 'graphTime', label: 'Interpolation graph'},
    {id: 'variogramTime', label: 'Variogram graph'}
  ],

  legends: state => {
    const style = getRainfallSurfaceStyle(state);
    const meta = RAINFALL_VARIABLES[state.variable];
    const entries = [
      {
        kind: 'ramp' as const,
        id: 'surface',
        title: style.title,
        ramp: style.ramp,
        unit: style.unit,
        extent: style.range ?? ('gpu' as const),
        format: (value: number) => (Math.abs(value) < 10 ? value.toFixed(1) : value.toFixed(0))
      }
    ];
    if (state.showStations && (style.ramp !== state.ramp || state.surface === 'error')) {
      entries.push({
        kind: 'ramp' as const,
        id: 'stations',
        title: `Gauges: ${meta.short.toLowerCase()}`,
        ramp: state.ramp,
        unit: meta.unit,
        extent: meta.range as unknown as 'gpu',
        format: (value: number) => value.toFixed(0)
      });
    }
    return state.holdOut && state.showStations
      ? [
          ...entries,
          {
            kind: 'categories' as const,
            title: 'Gauges',
            entries: [
              {color: [160, 160, 160, 255] as const, label: 'Used by the surface'},
              {color: [255, 255, 255, 255] as const, label: 'Ringed: held out for validation'}
            ],
            note: 'Dots are colored by the gauge reading with the ramp above.'
          }
        ]
      : entries;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUInverseDistanceWeighting, GPUKriging, GPUFocalStatistics,
  getGPUInverseDistanceWeightingParameterValues, getGPUKrigingParameterValues,
  getGPUFocalStatisticsParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUVariogram, fitVariogramModel, getGPUVariogramParameterValues}
  from '@luma.gl/experimental/gpu-dataframe';

// 1. Empirical variogram on the GPU, model fit on the CPU (once per data change)
variogramGraph.add(new GPUVariogram({
  positions, values, mask,                  // mask: 0 = held-out gauge
  parameters: variogramParameters.importToGraph(variogramGraph),
  gridSize: [32, 32], lagCount: 16, directionCount: 4,
  semivariances, pairCounts, meanDistances, robustSemivariances
}));
variogramParameters.write(getGPUVariogramParameterValues({
  bounds: indexBounds, maximumDistance: ${state.maximumLagKm * 1000}
}));
const model = fitVariogramModel(
  {distances, semivariances, pairCounts},
  {model: '${state.variogramModel}', weighting: '${state.weighting}'}
);

// 2. One graph: IDW + kriging + focal statistics over a raster that follows the camera
const common = {positions, values, mask, width: 384, height: 240,
  indexGridSize: [64, 64], indexBounds};
graph.add(new GPUInverseDistanceWeighting({...common, maximumNeighborCount: 32,
  parameters: idwParameters.importToGraph(graph), output: {values: idw, counts}}));
graph.add(new GPUKriging({...common, maximumNeighborCount: 16,
  parameters: krigingParameters.importToGraph(graph), output: {values: kriging, variance}}));
graph.add(new GPUFocalStatistics({values: selected, width: 384, height: 240,
  maximumRadius: 8, parameters: focalParameters.importToGraph(graph),
  output: {mean, max, range}}));
const compiled = graph.compile();            // once

// 3. Every frame: parameter writes, then encode
idwParameters.write(getGPUInverseDistanceWeightingParameterValues({
  extent: viewBounds, searchRadius: ${state.searchRadiusKm * 1000}, power: ${state.power},
  neighborCount: ${state.neighborCount}, minimumNeighborCount: ${state.minimumStations}
}));
krigingParameters.write(getGPUKrigingParameterValues({
  extent: viewBounds, searchRadius: ${state.searchRadiusKm * 1000},
  neighborCount: ${state.krigingNeighborCount}, variogram: model   // {model, nugget, sill, range}
}));
focalParameters.write(getGPUFocalStatisticsParameterValues({
  radius: ${state.focalRadius}, shape: '${state.focalShape}', minimumCount: ${state.focalMinimumCount}
}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUInverseDistanceWeighting` and `GPUKriging` turn scattered samples into a raster, one invocation per cell over a rebuilt grid index of the samples. IDW averages the nearest gauges with weights 1/d^p. Ordinary kriging solves a small linear system over the k nearest gauges using a variogram `γ(h)` fitted by `GPUVariogram` bins, and also returns the kriging variance. `GPUFocalStatistics` then summarises the raster over a moving window.',
    why: 'Rainfall is measured at points but decisions (flood warnings, crop losses, insurance) are about places. How you interpolate decides the answer between gauges, and kriging adds the part IDW cannot: an honest uncertainty map and a variogram that says how far rain correlates.',
    howToRead:
      'Colors show the interpolated value in the legend units; transparent cells have too few gauges inside the radius. Dots are the gauges on the same scale: where a dot disagrees with the surface under it, the surface is smoothing. In the error map brighter means less certain, and it is brightest far from any gauge.'
  },

  create: async ctx =>
    (await import('./rainfall-interpolation.compute')).createRainfallInterpolation(ctx),

  story: [
    {
      id: 'the-question',
      title: 'Where did Helene’s rain fall between the gauges?',
      body: 'On 26–27 September 2024 Hurricane Helene crossed Florida’s Big Bend and drove record rain into the southern Appalachians. NOAA’s GHCN-Daily network recorded **6,135 gauges** across the Southeast that day, but a gauge only knows its own backyard. A flood manager needs the rain *between* the gauges.\n\nThe colored surface is built on the GPU from those points by **`GPUInverseDistanceWeighting`**, over a raster that follows the camera. Dots are the gauges on the same scale: hover one for its name and reading, hover anywhere else for the interpolated value. Transparent cells are where no gauge is close enough to say anything. **Variable** and **Surface**, below, choose the reading and the method.',
      camera: {longitude: -83, latitude: 32.3, zoom: 5.2},
      options: {surface: 'idw'},
      callout: {coordinate: [-83.58, 30.12], text: 'Landfall near Perry, FL'},
      controls: ['variable', 'surface'],
      readouts: ['stations']
    },
    {
      id: 'idw',
      title: 'Nearby gauges vote, nearer ones louder',
      body: 'Inverse distance weighting estimates a cell as a weighted average of the gauges around it, `z = Σ wᵢ zᵢ / Σ wᵢ` with `wᵢ = 1 / dᵢᵖ`. Doubling the distance quarters a gauge’s say when **p = 2**.\n\nThe Blue Ridge near Asheville is the story: Connestee Falls, NC recorded 343 mm. Slide **Distance power p** to 0 (a plain average, flat plateaus) and to 6 (each cell snaps to its nearest gauge, bullseyes around every dot), and watch **Search radius** and **Nearest gauges k** carve no-data holes and limit the vote. IDW never predicts above the highest or below the lowest gauge, so it can only spread a peak, never place one between gauges.',
      camera: {longitude: -82.8, latitude: 35.4, zoom: 7.6, transitionMs: 2200},
      options: {surface: 'idw', power: 2, searchRadiusKm: 120, neighborCount: 12},
      callout: {coordinate: [-82.7364, 35.141], text: 'Connestee Falls: 343 mm'},
      controls: ['power', 'searchRadiusKm', 'neighborCount']
    },
    {
      id: 'variogram',
      title: 'How fast does rain stop resembling its neighbour?',
      body: 'IDW assumes one fixed distance decay everywhere. The **semivariogram** measures the real one: for every pair of gauges, half the squared difference of their readings, averaged by separation, `γ(h) = ½ · mean (zᵢ − zⱼ)²`. **`GPUVariogram`** computes that over all pairs inside the maximum lag in one deterministic GPU pass (6,000 gauges is 18 million pairs).\n\nThe bar charts below (**Empirical γ(h)** and **Fitted model γ(h)**) are the result, read from short to long lag. Pairs of close gauges differ little, far pairs differ as much as random gauges (the sample variance). A model is fitted on the CPU to the bins: the **nugget** is noise at zero distance, the **sill** the structured variance, and the **range** how far gauges stay similar. Switch **Variogram model** and **Fit weighting** and watch the fitted bars follow the empirical ones. Stretch **Maximum lag** past about 400 km and the bars overshoot the sample variance: Helene’s rain has a regional trend that a stationary variogram cannot explain.',
      camera: {longitude: -83, latitude: 32.3, zoom: 5.2, transitionMs: 2000},
      options: {surface: 'idw', variogramModel: 'spherical', maximumLagKm: 300},
      highlight: {readout: 'variogramModel'},
      controls: ['variogramModel', 'weighting', 'maximumLagKm'],
      readouts: ['variogramModel', 'empirical', 'fitted']
    },
    {
      id: 'kriging',
      title: 'Kriging lets the variogram choose the weights',
      body: '**`GPUKriging`** is ordinary kriging, as in `gstat` and PyKrige: for each cell it takes the k nearest gauges (**Neighbourhood size k**) and solves for weights λ that sum to 1 and minimise the prediction variance under the fitted `γ(h)`. Unlike IDW the weights account for how the *gauges relate to each other*, so a cluster of gauges counts less than the sum of its parts, and the surface can pass through gauges exactly (when the nugget allows).\n\nNothing was recompiled to switch: the surface is a parameter word and the variogram is a 12-float buffer. Flip **Surface** between *Inverse distance weighting* and *Ordinary kriging*, then pick *Kriging minus IDW* to see where they disagree (mostly the data-poor edges and the peaks).',
      camera: {longitude: -82.8, latitude: 35.4, zoom: 7.3, transitionMs: 2000},
      options: {surface: 'kriging', krigingNeighborCount: 12, searchRadiusKm: 120},
      controls: ['surface', 'krigingNeighborCount']
    },
    {
      id: 'uncertainty',
      title: 'Where the map is guesswork',
      body: 'Kriging also returns the **kriging variance**, shown here as a standard error in the units of the variable. It is zero at a gauge and rises with distance from the nearest ones, following the variogram: it depends on where the gauges *are*, not on what they read. Brighter means less certain.\n\nWiden the **Search radius** to see predictions pushed out over the Gulf and the Atlantic where the error is largest, and the gaps in the Appalachian valleys. Set **Surface** to *Gauges used per cell (IDW)* for the raw data density behind it. This map is how you decide where a new gauge would add the most information.',
      camera: {longitude: -83, latitude: 32.3, zoom: 5.2, transitionMs: 2000},
      options: {surface: 'error', searchRadiusKm: 220},
      controls: ['searchRadiusKm', 'surface']
    },
    {
      id: 'validation',
      title: 'Does kriging actually beat IDW? Hold gauges out and score',
      body: 'A prettier map is not a better one. **Hold out 10% of gauges** removes a random tenth from the variogram and from both interpolators (a mask buffer, no recompile), draws them ringed, and scores both rasters at those gauges: root-mean-square error, mean absolute error and bias.\n\nRead the **Verdict** row below. With a network this dense (median gauge spacing about 5 km) the two methods are usually within a few percent of each other: kriging’s edge here is a smaller bias and the error map, not a dramatically better surface. Press **Draw a new hold-out split** a few times: a ranking that survives different splits is believable. Caveat: neighbouring gauges are so close that a random split flatters both methods; a spatially blocked split would be a harsher test, and elevation (the orographic effect that put 343 mm on the Blue Ridge) is not used by either.',
      camera: {longitude: -83, latitude: 32.3, zoom: 5.2, transitionMs: 1800},
      options: {surface: 'kriging', holdOut: true, searchRadiusKm: 120},
      highlight: {readout: 'validationVerdict'},
      controls: ['holdOut', 'reseed'],
      readouts: ['validationIdw', 'validationKriging', 'validationVerdict']
    },
    {
      id: 'focal',
      title: 'Summarise the surface with focal statistics, then try your own',
      body: '**`GPUFocalStatistics`** slides a window over the raster and reports its mean, minimum, maximum, range or standard deviation (like ArcGIS `FocalStatistics` and GRASS `r.neighbors`). Here a 3-cell circle (**Window radius**) with the *Maximum* **Window statistic** turns the surface into a peak-rain footprint; *Range* or *Standard deviation* highlight where rain changes abruptly. Switching **Keep no-data holes** off lets the window grow the surface outward.\n\nLimits: the raster is 384 × 240 cells whatever the zoom, so zooming in refines it; interpolation is isotropic here although Helene’s rain was banded along its track (a **Direction sector** in the variogram would show it, but it does not change the map). Try **Variable** *Daily high temperature* (only 1,100 stations), or set **Surface** to the difference map.',
      camera: {longitude: -82.8, latitude: 35.4, zoom: 7.0, transitionMs: 1800},
      options: {
        surface: 'kriging',
        holdOut: false,
        focalStatistic: 'max',
        focalRadius: 3,
        focalShape: 'circle'
      },
      controls: ['focalStatistic', 'focalRadius', 'keepHoles', 'variable', 'surface']
    }
  ]
});
