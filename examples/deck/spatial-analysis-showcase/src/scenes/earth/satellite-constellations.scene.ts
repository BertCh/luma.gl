// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {SatelliteConstellationsOptions} from './satellite-constellations.compute';

const VIEW = {longitude: 10, latitude: 15, zoom: 1.5, pitch: 55, bearing: 0};
const elapsed = (seconds: number) => formatPlaybackTime.duration(seconds) || '0 min';

export default defineScene<SatelliteConstellationsOptions>({
  id: 'satellite-constellations',
  title: 'Starlink shells versus GPS orbits',
  chapter: 'earth',
  order: 32,
  summary:
    'Let GPU k-means group Starlink and GPS satellites by orbital inclination and altitude, and watch each shell fly: tilted bands a few hundred kilometres up against a navigation constellation at 20,200 km.',
  contributors: ['GPUKMeans', 'GPUTrajectoryPlayhead', 'GPUTimeWindowFilter'],
  datasets: [{id: 'celestrak-ground-tracks', role: 'SGP4 ground tracks (7 October 2026, 3 h)'}],
  initialView: VIEW,

  options: [
    {
      kind: 'select',
      id: 'family',
      label: 'Satellites to cluster',
      group: 'Shells (k-means)',
      apply: 'compile',
      default: 'both',
      help: 'Which satellites are grouped and shown. The set of points is fixed when the k-means graph compiles, so each family compiles its own graph on first use; the panel counts that as a rebuild.',
      options: [
        {value: 'both', label: 'Starlink sample + GPS'},
        {value: 'starlink', label: 'Starlink sample (800)'},
        {value: 'gps', label: 'GPS (32)'},
        {value: 'all', label: 'All 911 satellites'}
      ]
    },
    {
      kind: 'slider',
      id: 'k',
      label: 'Number of shells (k)',
      group: 'Shells (k-means)',
      apply: 'compile',
      min: 1,
      max: 8,
      step: 1,
      default: 5,
      help: 'How many clusters GPUKMeans looks for. k is a compile-time property of the contributor, so each value compiles its own graph on first use. Too small merges real shells; too large splits one shell by tiny altitude differences.'
    },
    {
      kind: 'select',
      id: 'seed',
      label: 'Initialization seed',
      group: 'Shells (k-means)',
      apply: 'compile',
      default: '7',
      help: 'k-means++ chooses its first centers at random; the seed makes that choice reproducible. A different seed can land in a different local optimum, which is itself a finding about how stable the shells are.',
      options: [
        {value: '7', label: 'Seed 7'},
        {value: '11', label: 'Seed 11'},
        {value: '23', label: 'Seed 23'},
        {value: '101', label: 'Seed 101'}
      ]
    },
    {
      kind: 'slider',
      id: 'altitudeWeight',
      label: 'Altitude weight',
      group: 'Shells (k-means)',
      apply: 'param',
      min: 5,
      max: 200,
      step: 5,
      default: 30,
      unit: '° per decade',
      help: 'k-means measures distance in one space, so inclination (degrees) and altitude must be put in common units. Features are inclination and weight x log10(altitude in km): this is how many degrees of inclination one factor of ten in altitude is worth. Only the feature buffer changes; the graph re-runs without recompiling.'
    },
    ...playbackOptions<SatelliteConstellationsOptions>({
      time: {
        min: 0,
        max: 10800,
        step: 30,
        default: 3600,
        label: 'Time since 00:00 UTC',
        format: elapsed,
        help: 'Time zero is 7 October 2026, 00:00 UTC.'
      },
      speed: {
        min: 30,
        max: 900,
        step: 30,
        default: 120,
        unit: 'x',
        help: 'Simulated seconds per real second.'
      },
      loop: {default: true}
    }),
    {
      kind: 'select',
      id: 'altitudeDisplay',
      label: 'Altitude scale',
      group: 'Altitude',
      apply: 'param',
      default: 'compressed',
      help: 'Linear is the true height times the exaggeration; compressed is logarithmic so GPS at 20,200 km and Starlink at 450 km fit one view (heights are ordered, not proportional).',
      options: [
        {value: 'compressed', label: 'Compressed (logarithmic)'},
        {value: 'linear', label: 'Linear (true proportions)'}
      ]
    },
    {
      kind: 'slider',
      id: 'altitudeScale',
      label: 'Altitude exaggeration',
      group: 'Altitude',
      apply: 'param',
      min: 1,
      max: 20,
      step: 0.5,
      default: 3,
      unit: 'x',
      help: 'Multiplies the displayed height.'
    },
    {
      kind: 'toggle',
      id: 'showStems',
      label: 'Drop lines to the ground',
      group: 'Altitude',
      apply: 'param',
      default: false,
      help: 'Vertical line from each satellite to the point on the ground below it.'
    },
    {
      kind: 'slider',
      id: 'markerSize',
      label: 'Marker size',
      group: 'Display',
      apply: 'param',
      min: 2,
      max: 12,
      step: 0.5,
      default: 4,
      unit: 'px',
      help: 'Radius of each satellite disc in screen pixels.'
    },
    {
      kind: 'toggle',
      id: 'showTrails',
      label: 'Show trails',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws the recent ground track at the satellite altitude, colored by shell.'
    },
    {
      kind: 'slider',
      id: 'trailMinutes',
      label: 'Trail length',
      group: 'Display',
      apply: 'param',
      min: 5,
      max: 120,
      step: 5,
      default: 30,
      unit: 'min',
      disabledWhen: state => !state.showTrails,
      help: 'Length of the time window behind each satellite.'
    }
  ],

  story: [
    {
      id: 'shells',
      title: 'Which satellites share an orbit?',
      body: 'A mega-constellation is built from shells: groups of satellites flying the same altitude at the same tilt. `GPUKMeans` finds them without being told how many there are in each place: every satellite becomes a point (its inclination, and the logarithm of its altitude), and the GPU repeats assign-and-update steps until the centers stop moving.\n\nEach color is one shell found. Start the clock with **Play** below, tilt the map, and read the shells in the readout.',
      camera: {...VIEW, transitionMs: 1400},
      options: {family: 'both', k: 5, altitudeWeight: 30, play: true},
      controls: ['play', 'k', 'family'],
      readouts: ['shells', 'converged']
    },
    {
      id: 'starlink',
      title: 'Four shells, unequal in size',
      body: 'The Starlink sample holds satellites at 43, 53, 70 and 97 degrees. In this sample the 53 degree shell has 372 satellites and the 43 degree one 280; the 70 degree shell has 53 and the polar 97 degree shell 95. At each tilt a few satellites sit lower than the rest, still climbing from their launch orbit or about to be retired.\n\nWith 4 shells selected, the chart counts each. Try **Number of shells (k)** at 3 and at 6: with too few clusters real shells merge, with too many one shell is split by small altitude differences.',
      camera: {...VIEW, transitionMs: 1400},
      options: {family: 'starlink', k: 4},
      controls: ['k', 'seed', 'family'],
      readouts: ['shellChart', 'inclinationChart', 'shells']
    },
    {
      id: 'gps',
      title: 'GPS is a different world',
      body: 'The 32 operational GPS satellites fly at about 20,200 km, roughly 40 times higher than Starlink, in orbits tilted by 53 to 57 degrees, and take 12 hours to circle the Earth. Over three hours each moves a quarter of a lap. With **Altitude scale** compressed they stay on the same screen as Starlink; switch to linear and low orbit collapses onto the map.\n\nGPS is six orbital planes. k-means on inclination and altitude cannot see them, because the planes differ by where they cross the equator, which is not one of the features. Only its altitude and tilt are clustered.',
      camera: {...VIEW, zoom: 1.2, pitch: 60, transitionMs: 1400},
      options: {family: 'both', k: 5, showStems: true, altitudeDisplay: 'compressed'},
      controls: ['altitudeDisplay', 'altitudeScale', 'showStems'],
      readouts: ['shells']
    },
    {
      id: 'weights',
      title: 'How much does altitude matter?',
      body: 'k-means uses plain distance, so the choice of units decides the answer. With altitude counted lightly (a weight of 5) inclination dominates and the four Starlink tilts separate cleanly at k = 4. Around 30 the lowest satellites of the 53 and 43 degree shells start to split off as extra clusters at k = 5. Counted heavily (200), inclination stops mattering: the 43 and 53 degree shells merge into one cluster of about 600 satellites and the small groups are told apart by altitude. There is no single correct weight; it encodes what you mean by "the same shell".\n\nDrag **Altitude weight** below and watch the colors and the chart change; the graph reruns without recompiling.',
      camera: {...VIEW, transitionMs: 1400},
      options: {family: 'starlink', k: 5, altitudeWeight: 30, showStems: false},
      controls: ['altitudeWeight', 'k', 'seed'],
      readouts: ['shellChart', 'converged']
    },
    {
      id: 'limits',
      title: 'A sample, a snapshot, and no conjunctions',
      body: 'Starlink is a random sample of 800 of about 10,600 satellites, and the altitude of each is its mean over three hours, so satellites still raising their orbit appear between shells. SGP4 positions are predictions that degrade with the age of the element sets. This scene does not look for close approaches: `GPUTrajectoryEncounters` is two-dimensional and would report satellites hundreds of kilometres apart in height as near misses.\n\nSet **Satellites to cluster** to all 911 and see how the weather, Earth observation and station satellites fall into the same tilted bands.',
      camera: {...VIEW, zoom: 1.4, transitionMs: 1400},
      options: {family: 'all', k: 8, showTrails: true},
      controls: ['family', 'k'],
      readouts: ['shells', 'satellites']
    }
  ],

  legends: (_state, data) => [
    {
      kind: 'categories' as const,
      title: 'Shell (inclination, mean altitude, satellites)',
      entries:
        (data.shells as
          | {color: readonly [number, number, number, number]; label: string}[]
          | undefined) ?? [],
      note: 'Shells are ordered by altitude; colors are stable across settings.'
    }
  ],

  readouts: [
    {
      id: 'shells',
      label: 'Shells found',
      layout: 'block',
      help: 'Center of each k-means cluster: inclination and altitude (the geometric mean, since altitude is clustered on a log scale), and its satellite count, lowest first.'
    },
    {
      id: 'shellChart',
      label: 'Satellites per shell',
      kind: 'chart',
      help: 'Cluster sizes from GPUKMeans, labeled by inclination / altitude (km).'
    },
    {
      id: 'inclinationChart',
      label: 'Inclination of the family',
      kind: 'chart',
      help: 'Histogram of orbital inclination in 2-degree bins; the vertical rules are the k-means centers.'
    },
    {
      id: 'converged',
      label: 'k-means',
      help: 'Iterations used before the largest center shift reached zero, out of the 24 allowed.'
    },
    {id: 'clock', label: 'Playhead'},
    {id: 'trailSegments', label: 'Trail segments live', format: 'integer'},
    {id: 'satellites', label: 'Dataset'}
  ],

  snippet: state => `import {GPUKMeans} from '@luma.gl/experimental/gpu-spatial-analysis';

// One 2D point per satellite: [inclination (deg), ${state.altitudeWeight} * log10(mean altitude in km)]
graph.add(new GPUKMeans({
  positions: features,
  k: ${state.k},                    // compile-time: a new k compiles a new graph
  iterations: 24, initialization: 'kmeans++', seed: ${state.seed},
  labels, centers, sizes, convergence   // [iterationsUsed, converged]
}));
compiled.encode(commandEncoder, {parameters: undefined});

// Rewriting 'features' (a different altitude weight) re-runs the same compiled graph.
// The playhead scene colors each satellite by labels[satellite] and lifts it by its altitude.`,

  about: {
    what: 'A look at constellation geometry: satellites grouped by orbital inclination and altitude with `GPUKMeans`, then flown on the globe with the same GPU playhead as the playback scene. The dataset holds a random sample of 800 Starlink satellites, the operational GPS constellation and other families, all propagated with SGP4 from current CelesTrak elements.',
    why: 'Mega-constellations are designed as shells, and navigation constellations as planes; the shell a satellite belongs to determines its coverage latitude, its drag and its neighbours. Clustering turns a list of elements into that structure.',
    howToRead:
      'Colors are k-means clusters in the plane of inclination and log altitude, lowest shell first. The chart counts satellites per shell. Clusters depend on k, the seed and the altitude weight, so treat them as a way to look, not as truth. Altitude is the mean over three hours. Close approaches are deliberately not computed: `GPUTrajectoryEncounters` is two-dimensional.'
  },

  create: async ctx =>
    (await import('./satellite-constellations.compute')).createSatelliteConstellations(ctx)
});
