// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import {playbackOptions} from '../../engine/playback';
import type {OceanDriftersVsModelOptions} from './ocean-drifters-vs-model.compute';

const GLOBAL_VIEW = {longitude: -20, latitude: 15, zoom: 1.5};
const REAL_LEGEND_COLOR = [70, 150, 255, 255] as const;
const MODEL_LEGEND_COLOR = [255, 150, 40, 255] as const;

export default defineScene<OceanDriftersVsModelOptions>({
  id: 'ocean-drifters-vs-model',
  title: 'Ocean drifters against the model',
  chapter: 'earth',
  order: 1,
  summary:
    'Release a virtual particle at every real 2017 drifter, advect it through the ECCO ocean model on the GPU, and measure on every day how far it ends up from the buoy that was really there. A skill curve, a failure map and the western boundary currents.',
  contributors: ['GPUParticleAdvection', 'GPUGeodesicPairs', 'GPULineDensity'],
  datasets: [
    {id: 'poopdeck-drifters', role: 'real drifters (NOAA Global Drifter Program, 2017)'},
    {id: 'poopdeck-ecco-currents', role: 'modelled annual-mean currents (ECCO V4r4)'}
  ],
  initialView: GLOBAL_VIEW,

  options: [
    ...playbackOptions<OceanDriftersVsModelOptions>({
      group: 'Lead time',
      time: {
        min: 0,
        max: 30,
        step: 0.25,
        default: 0,
        label: 'Lead time',
        unit: 'days',
        help: 'Days since release. The markers, trails, links and charts show this moment. While playing the slider follows the clock; drag it to jump.'
      },
      speed: {
        min: 0.5,
        max: 6,
        step: 0.5,
        default: 2,
        unit: 'days/s',
        label: 'Playback speed',
        help: 'Lead days per real second. 2 days/s plays the 30 days in 15 seconds.'
      },
      loop: true
    }),
    {
      kind: 'slider',
      id: 'speedScale',
      label: 'Current speed scale',
      group: 'Model',
      apply: 'param',
      min: 0.25,
      max: 3,
      step: 0.05,
      default: 1,
      unit: 'x',
      help: 'Multiplies the ECCO current speed before advecting (the speed scale of GPUParticleAdvection). 1x is the model as published; a value above 1 asks "how much faster would the ocean have to be?". Rerunning the 30 days takes a fraction of a second.'
    },
    {
      kind: 'select',
      id: 'coast',
      label: 'Particles at the coast',
      group: 'Model',
      apply: 'param',
      default: 'remove',
      help: 'The field has no data on land and shelves. Remove: a particle that would step onto a cell without data is lost for the rest of the run (and counted). Stick: those cells become zero velocity and hold the particle where it is.',
      options: [
        {value: 'remove', label: 'Remove (lost)'},
        {value: 'stick', label: 'Stick (held in place)'}
      ]
    },
    {
      kind: 'select',
      id: 'distanceModel',
      label: 'Distance model',
      group: 'Model',
      apply: 'compile',
      default: 'sphere',
      help: 'How GPUGeodesicPairs measures the separation: great circle on a sphere, Vincenty on the WGS84 ellipsoid, or a rhumb line of constant bearing. All three graphs are compiled up front and switched, so this shows a rebuild badge once. The differences are well under 1% here.',
      options: [
        {value: 'sphere', label: 'Great circle (sphere)'},
        {value: 'wgs84', label: 'WGS84 ellipsoid (Vincenty)'},
        {value: 'rhumb', label: 'Rhumb line'}
      ]
    },
    {
      kind: 'button',
      id: 'sweep',
      label: 'Find the best speed scale',
      group: 'Model',
      help: 'Runs the 30 days at nine scales from 0.5x to 3x, scores each by the mean median separation over lead days 1 to 30 for the drifters currently selected, and sets the slider to the winner.'
    },
    {
      kind: 'select',
      id: 'releaseSet',
      label: 'Drifters',
      group: 'Selection',
      apply: 'param',
      default: 'all',
      help: 'Which releases enter the statistics and the map. New deployments start after 5 January; the others were already at sea on 1 January and are released at their first fix of 2017.',
      options: [
        {value: 'all', label: 'All drifters'},
        {value: 'deployed', label: 'New deployments and records'},
        {value: 'atSea', label: 'Already at sea on 1 January'}
      ]
    },
    {
      kind: 'select',
      id: 'region',
      label: 'Region',
      group: 'Selection',
      apply: 'param',
      default: 'all',
      help: 'Keeps drifters released inside the region, for the statistics, the charts and the map, and flies the camera there.',
      options: [
        {value: 'all', label: 'All oceans'},
        {value: 'gulfStream', label: 'Gulf Stream'},
        {value: 'kuroshio', label: 'Kuroshio'},
        {value: 'agulhas', label: 'Agulhas'},
        {value: 'southern', label: 'Southern Ocean (south of 45°S)'},
        {value: 'tropicalPacific', label: 'Tropical Pacific'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showReal',
      label: 'Real drifters',
      group: 'Layers',
      apply: 'param',
      default: true,
      help: 'The buoys as observed, interpolated to the lead time (blue).'
    },
    {
      kind: 'toggle',
      id: 'showVirtual',
      label: 'Virtual drifters',
      group: 'Layers',
      apply: 'param',
      default: true,
      help: 'The particles advected through the model current field (orange).'
    },
    {
      kind: 'toggle',
      id: 'showTrails',
      label: 'Trails',
      group: 'Layers',
      apply: 'param',
      default: true,
      help: 'The daily path of each real and virtual drifter from release to the lead time. Segments are drawn by an indirect draw whose count the GPU is given each frame.'
    },
    {
      kind: 'toggle',
      id: 'showLinks',
      label: 'Separation links',
      group: 'Layers',
      apply: 'param',
      default: false,
      help: 'A line from each real drifter to its virtual twin at the lead time, colored by how far apart they are.'
    },
    {
      kind: 'toggle',
      id: 'showReleases',
      label: 'Release points (failure map)',
      group: 'Layers',
      apply: 'param',
      default: false,
      help: 'A dot at every release point, colored by the separation at the lead time. This is the map of where the model fails: scrub to day 30 and look for the bright dots.'
    },
    {
      kind: 'slider',
      id: 'pointSize',
      label: 'Marker size',
      group: 'Layers',
      apply: 'param',
      min: 2,
      max: 9,
      step: 0.5,
      default: 4,
      unit: 'px',
      help: 'Radius of the release dots; the drifter markers are 80% of it.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Separation color ramp',
      group: 'Layers',
      apply: 'param',
      default: 'inferno',
      help: 'Color of links and release dots. All four are perceptually uniform; cividis is color-blind optimised.',
      options: [
        {value: 'inferno', label: 'Inferno'},
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis'}
      ]
    },
    {
      kind: 'slider',
      id: 'sepMax',
      label: 'Separation at the top of the ramp',
      group: 'Layers',
      apply: 'param',
      min: 200,
      max: 3000,
      step: 100,
      default: 1500,
      unit: 'km',
      help: 'Separation (km) that gets the last color of the ramp. Also the right edge of the histogram.'
    },
    {
      kind: 'select',
      id: 'background',
      label: 'Map background',
      group: 'Background',
      apply: 'param',
      default: 'none',
      help: 'Current speed: the ECCO annual mean. Track density: kilometers of drifter track per 1,000 square kilometers, from GPULineDensity, for the real drifters or for the virtual ones. Difference: where one has more track than the other.',
      options: [
        {value: 'none', label: 'None'},
        {value: 'speed', label: 'Model current speed'},
        {value: 'realDensity', label: 'Real track density'},
        {value: 'modelDensity', label: 'Model track density'},
        {value: 'densityDiff', label: 'Model minus real density'}
      ]
    },
    {
      kind: 'slider',
      id: 'speedMax',
      label: 'Speed at the top of the ramp',
      group: 'Background',
      apply: 'param',
      min: 0.1,
      max: 0.8,
      step: 0.05,
      default: 0.4,
      unit: 'm/s',
      disabledWhen: state => state.background !== 'speed',
      help: 'Mean current speed (m/s) that gets the last color of the speed ramp.'
    },
    {
      kind: 'slider',
      id: 'densityMax',
      label: 'Density at the top of the ramp',
      group: 'Background',
      apply: 'param',
      min: 5,
      max: 200,
      step: 5,
      default: 60,
      unit: 'km / 1000 km²',
      disabledWhen: state =>
        state.background !== 'realDensity' && state.background !== 'modelDensity',
      help: 'Track length per 1,000 square kilometers that gets the last color. The ramp is a square root, so sparse ocean stays visible.'
    },
    {
      kind: 'slider',
      id: 'densityMinKm',
      label: 'Minimum track length for the difference',
      group: 'Background',
      apply: 'param',
      min: 0,
      max: 500,
      step: 10,
      default: 50,
      unit: 'km',
      disabledWhen: state => state.background !== 'densityDiff',
      help: 'A cell needs at least this much real plus model track before its normalised difference (model - real) / (model + real) is drawn; smaller cells are too noisy.'
    },
    {
      kind: 'slider',
      id: 'backgroundOpacity',
      label: 'Background opacity',
      group: 'Background',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.8,
      disabledWhen: state => state.background === 'none',
      help: 'Opacity of the background raster over the basemap.'
    }
  ],

  readouts: [
    {
      id: 'separationChart',
      label: 'Separation against lead time',
      kind: 'chart',
      help: 'Median distance between each virtual drifter and its real twin (solid), the 90th percentile (dashed), and the median if the particle had simply stayed where it was released (dashed, the benchmark to beat). The band is the middle half of the drifters. The vertical rule is the lead time.'
    },
    {
      id: 'separationHistogram',
      label: 'Separation at the lead day',
      kind: 'chart',
      help: 'How many drifters are each distance from their virtual twin at the lead day (rounded to a whole day).'
    },
    {
      id: 'regionChart',
      label: 'Median separation by region',
      kind: 'chart',
      help: 'Median separation at the lead day for drifters released in each region (the drifter set filter applies, the region filter does not).'
    },
    {id: 'lead', label: 'Lead time'},
    {
      id: 'comparedCount',
      label: 'Drifters compared',
      format: 'integer',
      help: 'Selected drifters with a real fix at the lead day and a virtual particle still in the model.'
    },
    {
      id: 'median',
      label: 'Median separation',
      help: 'At the lead day, over the compared drifters.'
    },
    {id: 'middleHalf', label: 'Middle half of separations', help: '25th to 75th percentile.'},
    {
      id: 'stayPut',
      label: 'If the particle had stayed put',
      help: 'Median distance from the release point to the real drifter at the lead day, over the same drifters.'
    },
    {
      id: 'improvement',
      label: 'Improvement over staying put',
      help: '1 minus (median model separation / median staying-put separation). Zero means the model is no better than not moving; negative would be worse.'
    },
    {
      id: 'lost',
      label: 'Lost to the coast or the edge',
      help: 'Share of drifters with a real fix whose virtual particle has left the field (or stepped onto a cell without data) by the lead day. They are removed from the separation statistics.'
    },
    {
      id: 'selected',
      label: 'Selected',
      help: 'Drifters in the selection (inside the model and the filters).'
    },
    {id: 'releases', label: 'Releases'},
    {id: 'field', label: 'Current field'},
    {id: 'samples', label: 'Model samples binned'},
    {
      id: 'gulfStreamLength',
      label: 'Track length, Gulf Stream box',
      help: 'Total length of real and virtual tracks (first 30 days) inside 82°W to 40°W, 28°N to 48°N, from GPULineDensity.'
    },
    {
      id: 'kuroshioLength',
      label: 'Track length, Kuroshio box',
      help: 'Inside 125°E to 180°, 20°N to 45°N.'
    },
    {
      id: 'agulhasLength',
      label: 'Track length, Agulhas box',
      help: 'Inside 10°E to 60°E, 48°S to 20°S.'
    },
    {id: 'sweep', label: 'Speed-scale sweep'},
    {id: 'runTime', label: 'Last model run'}
  ],

  legends: state => {
    const legends: LegendSpec[] = [];
    if (state.showLinks || state.showReleases) {
      legends.push({
        kind: 'ramp',
        title: 'Separation from the real drifter',
        ramp: state.ramp,
        extent: [0, state.sepMax],
        unit: 'km',
        format: value => Math.round(value).toLocaleString('en-US')
      });
    }
    if (state.showReal || state.showVirtual || state.showTrails) {
      legends.push({
        kind: 'categories',
        title: 'Drifters',
        entries: [
          {color: REAL_LEGEND_COLOR, label: 'Real drifter (NOAA GDP)'},
          {color: MODEL_LEGEND_COLOR, label: 'Virtual drifter (ECCO annual mean)'}
        ]
      });
    }
    if (state.background === 'speed') {
      legends.push({
        kind: 'ramp',
        title: 'Mean current speed (ECCO)',
        ramp: 'viridis',
        extent: [0, state.speedMax],
        unit: 'm/s',
        format: value => value.toFixed(2)
      });
    } else if (state.background === 'realDensity' || state.background === 'modelDensity') {
      legends.push({
        kind: 'ramp',
        title:
          state.background === 'realDensity'
            ? 'Real track length per 1,000 km²'
            : 'Model track length per 1,000 km²',
        ramp: 'inferno',
        extent: [0, state.densityMax],
        sqrtScale: true,
        unit: 'km / 1000 km²',
        format: value => Math.round(value).toString()
      });
    } else if (state.background === 'densityDiff') {
      legends.push({
        kind: 'ramp',
        title: 'Model minus real track length',
        ramp: 'diverging',
        extent: [-1, 1],
        labels: ['real tracks only', 'model tracks only'],
        format: value => value.toFixed(1)
      });
    }
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUParticleAdvection, getGPUParticleAdvectionParameterValues, getGPUParticleAdvectionWordParameterValues
} from '@luma.gl/experimental/gpu-raster';
import {GPUGeodesicPairs, GPULineDensity} from '@luma.gl/experimental/gpu-spatial-analysis';

// velocities: float32x2 (u, v) in degrees per day on a 720 x 320 lon/lat grid, row 0 = south
// positions: float32x2, one particle per drifter, written once with the real release points
const advect = new GPUCommandGraph(device, {id: 'advection'});
advect.add(new GPUParticleAdvection({
  velocities, fieldWidth: 720, fieldHeight: 320,
  parameters: params.importToGraph(advect), wordParameters: words.importToGraph(advect),
  state: {positions, ages, generations}
}));
// (a small kernel copies positions into snapshots[track * 31 + day] after every step)
const stepAdvection = advect.compile();

const pairs = new GPUCommandGraph(device, {id: 'pairs'});
pairs.add(new GPUGeodesicPairs({
  origins: snapshots,           // virtual drifter at day d, tracks x 31 rows
  targets: realDaily,           // real drifter at day d, same layout (NaN = no fix)
  model: '${state.distanceModel}',
  output: {distances}
}));

const density = new GPUCommandGraph(device, {id: 'density'});
for (const [positions, pathOffsets] of [[realTrack, realOffsets], [snapshots, snapshotOffsets]]) {
  density.add(new GPULineDensity({
    positions, pathOffsets, columns: 720, rows: 320, coordinateSystem: 'spherical',
    parameters: grid.importToGraph(density),     // [west, south, cellWidth, cellHeight]
    output: {lengths, densities, overflow}
  }));
}

// Thirty daily steps at ${state.speedScale}x, one submit each (the frame word is the day):
params.write(getGPUParticleAdvectionParameterValues({
  fieldExtent: [-180, -80, 0.5, 0.5], timeStep: 1, speedScale: ${state.speedScale},
  spawnBounds: [0, 89.95, 0, 89.95]          // lost particles park outside the field
}, [720, 320]));
for (let day = 1; day <= 30; day++) {
  words.write(getGPUParticleAdvectionWordParameterValues({seed: 1, frame: day, maximumAge: 0}));
  const encoder = device.createCommandEncoder();
  stepAdvection.encode(encoder, {parameters: undefined});
  device.submit(encoder.finish());
}`,

  about: {
    what: '`GPUParticleAdvection` moves one virtual particle per real drifter through the annual-mean surface current of the ECCO ocean state estimate: one second-order Runge-Kutta step per day, thirty days, all particles at once. `GPUGeodesicPairs` then measures the great-circle (or ellipsoidal, or rhumb) distance between every virtual particle and the real drifter it was released with, for every lead day in one dispatch. `GPULineDensity` turns the real and the virtual tracks into track length per half-degree cell so the two can be compared on a map.',
    why: 'This is how ocean models are checked against buoys: release virtual particles where real ones were, and ask how fast the pair separates. The curve of separation against lead time is the skill of the model for tracking surface drift (search and rescue, oil spills, plastic), and the map of where it is worst shows which parts of the ocean the model does not capture.',
    howToRead:
      'Blue is the real buoy, orange its virtual twin. A short link means the model got the drift right; a bright one means they are far apart (see the legend). In the chart the solid line is the median separation; the dashed line below it is what you would get by assuming the buoy never moved. A model that beats that benchmark has skill. Caveats: the field is an **annual mean** (no seasons, no eddies, no diffusion), ECCO resolves about 1 degree so the narrow western boundary currents are weak, real drifters also slip with the wind (windage) and some lose their drogue, which this data does not flag, and particles are advected with one-day steps from the first fix of 2017.'
  },

  create: async ctx =>
    (await import('./ocean-drifters-vs-model.compute')).createOceanDriftersVsModel(ctx),

  story: [
    {
      id: 'the-question',
      title: 'If the model had a buoy at the same spot, where would it drift?',
      body: "Since 1979 the NOAA **Global Drifter Program** has put satellite-tracked buoys in the ocean. This chapter uses **2,811 of them from 2017**, each followed for the first 60 days of its 2017 record. For every one we release a **virtual** twin at the same place and run it through the surface currents of NASA/JPL's **ECCO** ocean model.\n\nBlue dots are real buoys, orange dots are their virtual twins, and the trails are the paths so far. Press **Play** below, or drag **Lead time**: at day 0 each pair sits together, and every day after that they can drift apart. The **Playback speed** control sets how many days pass per second.",
      camera: {...GLOBAL_VIEW, transitionMs: 1200},
      options: {play: true, time: 0, showTrails: true},
      controls: ['play', 'time', 'speed'],
      readouts: ['lead', 'releases']
    },
    {
      id: 'the-model-field',
      title: 'The model ocean: one mean current for the whole year',
      body: 'The virtual twins are pushed by a field we **derived from all 1.44 million ECCO particle displacements** of the archive: each 3.5-day move gives a velocity, and every velocity is averaged into half-degree cells. Switch **Map background** to *Model current speed*: the equatorial currents, the gyres and the Antarctic Circumpolar Current appear.\n\nLook at the Gulf Stream and Kuroshio: they are **weak and smooth** here, because ECCO resolves about one degree and the field is a mean. **Particles at the coast** decides what happens when a particle runs out of data: it is removed (and counted) or held in place. Tune **Speed at the top of the ramp** if the field looks too dark.',
      options: {
        background: 'speed',
        showTrails: false,
        play: false,
        time: 0,
        backgroundOpacity: 0.75
      },
      camera: {...GLOBAL_VIEW, transitionMs: 1400},
      controls: ['background', 'speedMax', 'coast'],
      readouts: ['field', 'samples']
    },
    {
      id: 'separation-grows',
      title: 'How fast do the twins come apart?',
      body: '**`GPUGeodesicPairs`** measures the distance between every virtual particle and its real twin on every lead day. Each link below joins a pair, and the dot at the release point takes its color: dark is close, bright is far (the legend gives the kilometers, set by **Separation at the top of the ramp**).\n\nDrag **Lead time** from 0 to 30 days and watch the links stretch. The chart is the whole answer: the solid line is the median separation at each lead day and the band is the middle half of the drifters. Switch **Distance model** to *Rhumb line* or *WGS84 ellipsoid* and compare: the three measures differ by well under one percent at these distances.',
      options: {
        showTrails: false,
        showLinks: true,
        showReleases: true,
        play: false,
        time: 15,
        background: 'none'
      },
      camera: {...GLOBAL_VIEW, transitionMs: 1400},
      controls: ['time', 'sepMax', 'ramp', 'distanceModel'],
      readouts: ['median', 'middleHalf', 'comparedCount', 'separationChart']
    },
    {
      id: 'beating-nothing',
      title: 'Is that better than assuming the buoy never moved?',
      body: 'A skill number needs a benchmark. The dashed line in the chart is the median distance from the release point to the real buoy: what you get by **ignoring the ocean**. The model is only useful where its solid line is clearly below that one, and the **Improvement over staying put** readout turns that into a percentage.\n\nChange the **Drifters** filter to compare drifters that were *already at sea* with *new deployments*, and pick a **Region** to see its own curve: the histogram shows how spread the separations are, and the bars compare regions at the same lead day. A large share of **lost** particles means the comparison is made on the survivors.',
      options: {
        showTrails: false,
        showLinks: true,
        showReleases: true,
        play: false,
        time: 30,
        background: 'none'
      },
      controls: ['releaseSet', 'region', 'time'],
      readouts: ['stayPut', 'improvement', 'lost', 'separationHistogram', 'regionChart']
    },
    {
      id: 'where-it-fails',
      title: 'Where it fails: the western boundary currents',
      body: 'Look for the model to be worst where the real ocean is fastest and narrowest. **`GPULineDensity`** adds up the length of real tracks and of virtual tracks in every half-degree cell. The map shows **(model - real) / (model + real)**: **blue** cells have more real track than model track (the buoys went there and the model particles did not), **red** the other way round.\n\nSet **Region** to *Gulf Stream*, *Kuroshio* or *Agulhas*: real buoys are carried along these currents, while the weak mean field moves its twins less, and the **track length** readouts below compare the real and the model total in each box. Raise **Minimum track length for the difference** to hide noisy cells.',
      options: {
        showTrails: false,
        showLinks: false,
        showReleases: false,
        showReal: false,
        showVirtual: false,
        background: 'densityDiff',
        backgroundOpacity: 0.85,
        play: false,
        time: 30,
        region: 'gulfStream'
      },
      camera: {longitude: -62, latitude: 38, zoom: 4.1, transitionMs: 1500},
      callout: {coordinate: [-72, 37], text: 'Gulf Stream'},
      controls: ['region', 'background', 'densityMinKm'],
      readouts: ['gulfStreamLength', 'kuroshioLength', 'agulhasLength']
    },
    {
      id: 'tune-and-limits',
      title: 'Turn the ocean up, and what to remember',
      body: 'How much faster would the model ocean have to be? Press **Find the best speed scale**: the 30 days are run at nine scales and the slider is set to the one with the smallest mean median separation for the current selection. A winner near 1x says the mean model speed is about right on average; a winner above 1x says it underestimates the real drift. Either way a single number hides the regions: real drifters also feel the **wind** (windage) and the **Ekman layer**, which an annual-mean ocean current does not contain, and the boundary currents are far faster than the mean field.\n\nRemember: the field is a **time mean** (no seasons, no eddies), there is **no diffusion**, ECCO is coarse, some drifters **lost their drogue** (this archive does not say which), steps are one day long, and lost particles are excluded from the statistics. Try it for one region at a time.',
      options: {
        showTrails: true,
        showLinks: true,
        showReleases: false,
        showReal: true,
        showVirtual: true,
        background: 'none',
        play: false,
        time: 30,
        region: 'all'
      },
      camera: {...GLOBAL_VIEW, transitionMs: 1400},
      controls: ['sweep', 'speedScale', 'coast', 'region'],
      readouts: ['sweep', 'median', 'improvement', 'runTime']
    }
  ]
});
