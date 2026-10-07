// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {formatPlaybackTime, playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {SatelliteConstellationsOptions} from './satellite-constellations.compute';

const VIEW = {longitude: 10, latitude: 15, zoom: 1.5, pitch: 0, bearing: 0};
const elapsed = (seconds: number) => formatPlaybackTime.duration(seconds) || '0 min';

export default defineScene<SatelliteConstellationsOptions>({
  id: 'satellite-constellations',
  title: 'What makes an orbital shell?',
  chapter: 'earth',
  order: 12,
  summary:
    'Let GPU k-means group loaded satellites by orbital inclination and altitude, then inspect the iterative assignments and sensitivity of the resulting shells.',
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
        {value: 'starlink', label: 'Bounded Starlink sample'},
        {value: 'gps', label: 'GPS records'},
        {value: 'all', label: 'All loaded records'}
      ]
    },
    {
      kind: 'select',
      id: 'iterationFrame',
      label: 'Iteration frame',
      group: 'Shells (k-means)',
      apply: 'param',
      default: '24',
      help: 'Runs GPUKMeans for the selected authored iteration limit, so the linked map and scatter show its actual intermediate assignments and centres before the final 24-iteration result.',
      options: [
        {value: '1', label: '1. Assign'},
        {value: '2', label: '2. Move centres'},
        {value: '4', label: '3. Repeat'},
        {value: '24', label: 'Final GPU result'}
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
      id: 'features',
      title: 'A shell begins with chosen features',
      headline: 'K-means sees tilt and mean altitude',
      textAlternative:
        'Satellite clusters are explained as a two-feature partition rather than physical orbital planes.',
      optionsMode: 'fresh',
      body: 'Every satellite first becomes a point in a deliberately small feature space: inclination in degrees and mean altitude on a logarithmic scale. The linked scatter shows those measured points with units; the map is only the playback view. `GPUKMeans` then assigns every point to its nearest centre and moves that centre until it stops changing.\n\nThis is a classification, not recovered physical orbital planes: RAAN is absent. Start the clock with **Play** and read the provisional shell labels with the feature chart.',
      camera: {...VIEW, transitionMs: 1400},
      options: {family: 'both', k: 5, altitudeWeight: 30, play: true},
      controls: ['play', 'k', 'family'],
      readouts: ['featureChart', 'shells', 'converged']
    },
    {
      id: 'iterate',
      title: 'Assign, move centres, repeat',
      headline: 'Clusters are an iterative choice',
      textAlternative:
        'Cluster colours and convergence readout explain the k-means assignment and centre-update cycle.',
      optionsMode: 'fresh',
      body: 'The linked feature chart colours assignments and the stage chart keeps the actual k-means sequence visible: assign points to nearest centres, move each centre to its mean, then repeat until the GPU convergence check stops.\n\nChange k to see a live partition, rather than treating a final scatter as a fixed taxonomy.',
      camera: {...VIEW, transitionMs: 1400},
      options: {family: 'starlink', k: 4},
      controls: ['iterationFrame', 'k', 'seed', 'family'],
      readouts: ['featureChart', 'iterationChart', 'shellChart', 'shells']
    },
    {
      id: 'starlink',
      title: 'One sample separates into orbital shells',
      headline: 'A sample can still reveal bands',
      textAlternative:
        'Starlink sample clusters show stable tilt-and-altitude groups over a dark world map.',
      optionsMode: 'fresh',
      body: 'The Starlink input is a deliberately bounded sample, not a census. Vary **k** to see how a classification can merge or split the feature-space bands. The live shell chart supplies each cluster’s N.\n\nCluster labels are descriptive shell IDs, not physical names or orbital planes.',
      camera: {...VIEW, zoom: 1.2, transitionMs: 1400},
      options: {family: 'starlink', k: 4, showStems: false, altitudeDisplay: 'compressed'},
      controls: ['k', 'altitudeDisplay', 'altitudeScale'],
      readouts: ['featureChart', 'shells', 'shellChart']
    },
    {
      id: 'gps',
      title: 'Altitude and tilt cannot reveal orbital planes',
      headline: 'GPS has 32 records, not 37',
      textAlternative:
        'GPS points remain distinct in altitude while the story explains that RAAN is absent from the clustering features.',
      optionsMode: 'fresh',
      body: 'The live readout reports the loaded GPS record count. GPS sits high above the bounded Starlink sample in the scatter, but inclination and mean altitude cannot recover orbital planes: RAAN is absent from these features.\n\nDrag **Altitude weight** to see the axis-scale decision change the provisional partition. Compressed height is display-only and never changes clustering features.',
      camera: {...VIEW, transitionMs: 1400},
      options: {family: 'gps', k: 3, altitudeWeight: 30, showStems: false},
      controls: ['altitudeWeight', 'k', 'seed'],
      readouts: ['featureChart', 'shellChart', 'converged']
    },
    {
      id: 'stability',
      title: 'Weights and seeds change the partition',
      headline: 'Feature units decide cluster boundaries',
      textAlternative:
        'Changing altitude weight and seed demonstrates a different k-means partition without presenting it as orbital truth.',
      optionsMode: 'fresh',
      body: 'The synchronized before/after assignment comparison and reassignment matrix show what changes when seed or altitude weight changes. Shell colours remain stable for a fixed family, k, seed and weight, but are descriptive feature-space labels—not planes.\n\nThe bounded Starlink sample, dated TLE snapshot and absent RAAN remain visible limits.',
      camera: {...VIEW, zoom: 1.4, transitionMs: 1400},
      options: {family: 'all', k: 8, showTrails: true},
      controls: ['family', 'k', 'seed', 'altitudeWeight'],
      readouts: ['reassignmentMatrix', 'reassignedShare', 'shells', 'satellites']
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
    {id: 'iterationChart', label: 'Assign → move → repeat', kind: 'chart'},
    {id: 'reassignmentMatrix', label: 'Before / after assignments', kind: 'chart'},
    {id: 'reassignedShare', label: 'Reassigned share'},
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
      id: 'featureChart',
      label: 'Inclination–altitude feature space',
      kind: 'chart',
      help: 'One dot per selected satellite. Mean altitude uses a logarithmic kilometre axis; colours are provisional nearest-centre classes, not orbital-plane labels.'
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
    what: 'A look at constellation geometry: satellites grouped by orbital inclination and altitude with `GPUKMeans`, then flown on the globe with the same GPU playhead as the playback scene. The dataset holds a bounded 800-object Starlink sample, 32 GPS records and other families, all propagated with SGP4 from a dated CelesTrak TLE snapshot.',
    why: 'Clustering is a way to inspect a chosen feature space. It does not recover orbital planes, coverage, or conjunctions.',
    howToRead:
      'Colors are k-means clusters in the plane of inclination and log altitude, lowest shell first. The chart counts satellites per shell. Clusters depend on k, the seed and the altitude weight, so treat them as a way to look, not as truth. Altitude is the mean over three hours. Close approaches are deliberately not computed: `GPUTrajectoryEncounters` is two-dimensional.'
  },

  pipeline: [
    {
      id: 'features',
      label: 'Features',
      detail: 'Build inclination and weighted log-altitude values'
    },
    {id: 'assign', label: 'Assign', detail: 'Give each satellite its nearest centre'},
    {id: 'update', label: 'Update', detail: 'Move centres to fixed-order cluster means'},
    {id: 'draw', label: 'Draw', detail: 'Link modelled tracks to provisional shell labels'}
  ],
  basemap: ground('space', {
    labels: 'none',
    graticule: true,
    referenceLines: ['equator', 'tropics']
  }),
  furniture: {
    title: {
      title: 'What makes an orbital shell?',
      subtitle: 'Inclination + mean altitude · GPU k-means · SGP4 model'
    },
    credit: joinCredits('CelesTrak GP elements; propagation with SGP4', CREDITS.naturalEarth),
    caveat: 'RAAN is absent: these feature clusters cannot recover orbital planes or conjunctions.'
  },

  create: async ctx =>
    (await import('./satellite-constellations.compute')).createSatelliteConstellations(ctx)
});
