// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {WORLD, labelsFor} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {HurricaneFamiliesOptions} from './hurricane-families.compute';
import {
  HURRICANE_CATEGORY_COLORS,
  HURRICANE_CATEGORY_LABELS,
  HURRICANE_FAMILY_COLORS
} from './hurricane-data';

const BASIN_VIEW = {longitude: -55, latitude: 29, zoom: 3.05};
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  chips: ['route similarity'] as const
});
const FAMILY_LABELS = labelsFor(WORLD, [
  'atlantic-ocean',
  'caribbean-sea',
  'gulf-of-mexico',
  'sargasso-sea'
]);

export default defineScene<HurricaneFamiliesOptions>({
  id: 'hurricane-families',
  title: 'Do hurricane routes form stable families?',
  chapter: 'earth',
  order: 5,
  summary:
    'Resample Atlantic storm tracks, score route similarity, then test whether k-means summaries remain useful as their settings change. Select a storm to compare it with the closest route and its rendered discrete-Frechet leash.',
  contributors: ['GPUTrajectoryResample', 'GPUTrackSimilarity', 'GPUKMeans'],
  datasets: [{id: 'ibtracs-north-atlantic', role: 'storm tracks (IBTrACS, 1980-2025)'}],
  initialView: BASIN_VIEW,

  options: [
    {
      kind: 'select',
      id: 'routeSpacing',
      label: 'Route samples spaced by',
      group: 'Routes (resample)',
      apply: 'compile',
      default: 'arc-length',
      help: 'Each storm is rebuilt as the same number of points before comparison. Equal distance compares the shape of the route; equal time also weighs how long the storm lingered in each place.',
      options: [
        {value: 'arc-length', label: 'Equal distance along the track'},
        {value: 'time', label: 'Equal time steps'}
      ]
    },
    {
      kind: 'select',
      id: 'routeSamples',
      label: 'Samples per route',
      group: 'Routes (resample)',
      apply: 'compile',
      default: '48',
      help: 'Points per storm. More points follow bends more closely and cost more in the Frechet pass (quadratic in the samples). A compile-time size: changing it rebuilds the resample and similarity graphs.',
      options: [
        {value: '32', label: '32 points'},
        {value: '48', label: '48 points'},
        {value: '64', label: '64 points'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showRoutes',
      label: 'Show resampled routes',
      group: 'Routes (resample)',
      apply: 'param',
      default: false,
      help: 'Draws the resampled points and the polyline through them, colored by family, over the faded original tracks.'
    },
    {
      kind: 'select',
      id: 'distanceMeasure',
      label: 'Route distance',
      group: 'Route distance (similarity)',
      apply: 'param',
      default: 'frechet',
      help: 'GPUTrackSimilarity scores both measures for every pair in one pass; this picks the matrix that is embedded and clustered.',
      options: [
        {
          value: 'frechet',
          label: 'Discrete Frechet (the dog-walk distance)',
          help: 'The shortest leash that lets two walkers follow their routes in order. It respects direction and sequence, so a storm going out and one coming back are far apart.'
        },
        {
          value: 'hausdorff',
          label: 'Hausdorff',
          help: 'The largest gap from any point of one route to the other route. It ignores order, so two routes that are the same curve walked in opposite directions are close.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'similarityRangeKm',
      label: 'Distance color range',
      group: 'Route distance (similarity)',
      apply: 'param',
      min: 500,
      max: 8000,
      step: 250,
      default: 3000,
      unit: 'km',
      help: 'The distance from the selected storm that reaches the end of the color ramp (and the marker in the histogram). Storms farther than this show the darkest color.'
    },
    {
      kind: 'slider',
      id: 'familyCount',
      label: 'Number of families',
      group: 'Families (k-means)',
      apply: 'compile',
      min: 2,
      max: 8,
      step: 1,
      default: 5,
      help: 'k, the number of clusters. It is fixed when GPUKMeans is compiled, so changing it rebuilds the clustering graph (the distances are not recomputed).'
    },
    {
      kind: 'select',
      id: 'initialization',
      label: 'Starting centers',
      group: 'Families (k-means)',
      apply: 'compile',
      default: 'kmeans++',
      help: 'How the first centers are picked. k-means++ draws them spread apart with a seeded hash; first-valid takes the first k storms in time order, which can start two centers in the same cluster.',
      options: [
        {value: 'kmeans++', label: 'k-means++ (seeded)'},
        {value: 'first-valid', label: 'The first k storms'}
      ]
    },
    {
      kind: 'slider',
      id: 'seed',
      label: 'k-means++ seed',
      group: 'Families (k-means)',
      apply: 'compile',
      min: 0,
      max: 99,
      step: 1,
      default: 7,
      disabledWhen: state => state.initialization !== 'kmeans++',
      help: 'Seed of the k-means++ draw. The same seed gives the same families on one device; change it to see whether the families are stable.'
    },
    {
      kind: 'slider',
      id: 'iterations',
      label: 'Iterations',
      group: 'Families (k-means)',
      apply: 'compile',
      min: 2,
      max: 64,
      step: 1,
      default: 24,
      help: 'Maximum Lloyd iterations (assign to the nearest center, then move the centers). The graph holds this many; once nothing moves, the rest return immediately. The readout says how many were needed.'
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Color storms by',
      group: 'Display',
      apply: 'param',
      default: 'family',
      help: 'Family colors need the clustering. Distance colors need a selected storm.',
      options: [
        {value: 'family', label: 'Track family'},
        {value: 'similarity', label: 'Distance to the selected storm'},
        {value: 'peak', label: 'Peak category (Saffir-Simpson)'},
        {value: 'season', label: 'Season (year)'},
        {value: 'plain', label: 'One color'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'cividis',
      help: 'Used by the distance and season colors. Distance is read in kilometers; season runs from 1980 to 2025.',
      options: [
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis'}
      ]
    },
    {
      kind: 'select',
      id: 'representative',
      label: 'Family representative',
      group: 'Display',
      apply: 'param',
      default: 'medoid',
      help: 'A medoid is a real route with the smallest total distance to its family. A mean averages matching resample points and may not be a storm that occurred.',
      options: [
        {value: 'medoid', label: 'Medoid (a real central route)'},
        {value: 'mean', label: 'Point-by-point mean route'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showFamilyMeans',
      label: 'Show family representatives',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draw the chosen central route for each family above the individual tracks.'
    },
    {
      kind: 'slider',
      id: 'trackOpacity',
      label: 'Track opacity',
      group: 'Display',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.55,
      help: 'Opacity of the original tracks. With 739 overlapping tracks, lower values show where routes pile up.'
    }
  ],

  legends: (state, data) => {
    const familyLabels = (data.familyLabels as string[] | undefined) ?? [];
    switch (state.colorBy) {
      case 'family':
        return [
          {
            kind: 'categories' as const,
            title: 'Track family',
            entries: (familyLabels.length
              ? familyLabels
              : Array.from({length: state.familyCount}, (_, family) => `Family ${family + 1}`)
            ).map((label, family) => ({color: HURRICANE_FAMILY_COLORS[family], label})),
            note: 'Numbered from the most easterly mean start to the most westerly.'
          }
        ];
      case 'similarity':
        return [
          {
            kind: 'ramp' as const,
            title: `${state.distanceMeasure === 'frechet' ? 'Frechet' : 'Hausdorff'} distance to the selected storm`,
            ramp: state.ramp,
            extent: [0, state.similarityRangeKm] as const,
            unit: 'km',
            format: (value: number) => value.toFixed(0)
          }
        ];
      case 'peak':
        return [
          {
            kind: 'categories' as const,
            title: 'Peak intensity of the storm',
            entries: HURRICANE_CATEGORY_LABELS.map((label, index) => ({
              color: HURRICANE_CATEGORY_COLORS[index],
              label
            })),
            note: 'Saffir-Simpson classes come from the 1-minute sustained wind, not from the storm damage.'
          }
        ];
      case 'season':
        return [
          {
            kind: 'ramp' as const,
            title: 'Season',
            ramp: state.ramp,
            extent: [1980, 2025] as const,
            format: (value: number) => value.toFixed(0)
          }
        ];
      default:
        return [];
    }
  },

  readouts: [
    {
      id: 'embeddingChart',
      label: 'Route embedding',
      kind: 'chart',
      help: 'Each dot is one storm in a two-dimensional distance embedding, not a geographic location. Colours are the live k-means families; click a dot to select its route on the map.'
    },
    {
      id: 'familyChart',
      label: 'Family sizes',
      kind: 'chart',
      help: 'Storms in each k-means family. The bar of the selected storm is highlighted.'
    },
    {
      id: 'distanceChart',
      label: 'Distances from the selected storm',
      kind: 'chart',
      help: 'How far every other storm is from the selected one, by Frechet or Hausdorff distance. Without a selection, a sample of all pairs.'
    },
    {
      id: 'storms',
      label: 'Storms',
      help: 'Tracks in the dataset (best tracks since 1980, 6-hourly).'
    },
    {
      id: 'pairs',
      label: 'Pairs scored',
      help: 'Every unordered pair of storms, scored by GPUTrackSimilarity.'
    },
    {
      id: 'embedding',
      label: 'Two-dimensional embedding',
      help: 'GPUKMeans clusters points in the plane, so the distance matrix is first embedded in two dimensions (classical multidimensional scaling, on the CPU). This is how much of the distance variance the plane keeps.'
    },
    {
      id: 'kmeans',
      label: 'k-means',
      help: 'Iterations used, convergence and the variance the families explain.'
    },
    {
      id: 'families',
      label: 'Families',
      layout: 'block',
      help: 'For each family: storms, mean start and end positions, mean track length and the share of storms that reached hurricane strength.'
    },
    {
      id: 'leash',
      label: 'Rendered route comparison',
      help: 'The orange route is the selected storm’s closest route under the active measure. The orange connector is the widest matched pair in their discrete-Frechet coupling.'
    },
    {
      id: 'selected',
      label: 'Selected storm',
      help: 'Click a storm to read its peak, dates, family and its closest track.'
    }
  ],

  pipeline: [
    {
      id: 'resample',
      label: 'Resample',
      detail: 'Make unequal tracks comparable at equal fractions'
    },
    {
      id: 'distance',
      label: 'Route distance',
      detail: 'Score every unordered pair with a shape metric'
    },
    {id: 'embed', label: 'Embed', detail: 'Place the distance matrix in two dimensions on CPU'},
    {id: 'kmeans', label: 'k-means', detail: 'Assign, update centres, repeat'},
    {id: 'draw', label: 'Draw', detail: 'Routes and the linked family summary'}
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTrajectoryResample, GPUTrackSimilarity, GPUKMeans
} from '@luma.gl/experimental/gpu-spatial-analysis';

// positions: float32x2 azimuthal-equidistant meters, timestamps: float32 seconds since each storm's start
const routes = new GPUCommandGraph(device, {id: 'routes'});
routes.add(new GPUTrajectoryResample({
  positions, timestamps, trackOffsets,
  sampleCount: ${state.routeSamples}, spacing: '${state.routeSpacing}',      // compile-time
  samples                                  // storms x ${state.routeSamples} float32x2 rows
}));

const similarity = new GPUCommandGraph(device, {id: 'similarity'});
similarity.add(new GPUTrackSimilarity({
  positionsA: samples, offsetsA: sampleOffsets,
  pairA, pairB,                            // every unordered pair of storms
  frechet, hausdorff, status,              // one float32 per pair
  maxFrechetVertices: ${state.routeSamples}
}));

// distances -> two-dimensional embedding on the CPU (classical scaling), then:
const families = new GPUCommandGraph(device, {id: 'families'});
families.add(new GPUKMeans({
  positions: embedding,                    // float32x2 per storm
  k: ${state.familyCount}, iterations: ${state.iterations},
  initialization: '${state.initialization}'${state.initialization === 'kmeans++' ? `, seed: ${state.seed}` : ''},
  labels, centers, sizes, convergence, squaredDistances
}));`,

  about: {
    what: '`GPUTrajectoryResample` rebuilds every storm as the same number of points, evenly spaced along the path (or in time). `GPUTrackSimilarity` scores each pair of routes with the discrete Frechet distance (and the Hausdorff distance) on the GPU. The 739 x 739 distance matrix is embedded in two dimensions on the CPU, and `GPUKMeans` groups the embedded storms into families.',
    why: 'Forecasters and insurers talk about hurricane "types": Cape Verde storms that cross the whole Atlantic, storms born in the western Caribbean or Gulf, storms that recurve north and out to sea. A shape-aware distance lets the data propose those groups instead of drawing them by hand, and pick-a-storm distance shows which past storms had the most similar route.',
    howToRead:
      'Each line is one storm. In family colors, lines of one color follow a similar route; the thick lines are the mean route of each family. In distance colors, bright is close to the selected storm and dark is far. The histogram shows how many storms lie at each distance. Families are a summary of a continuum, not natural classes: change the number of families and the groups split or merge.'
  },

  basemap: ground('paperSheet'),
  furniture: {
    title: cartouche(
      'Do hurricane routes form stable families?',
      'Route similarity · Fréchet / Hausdorff · k-means'
    ),
    scaleBar: {units: 'metric'},
    credit: joinCredits(CREDITS.noaaNhc, 'NOAA IBTrACS', CREDITS.naturalEarth, CREDITS.okabeIto),
    caveat:
      'Distances are computed in an azimuthal metric frame and drawn in Web Mercator; families are a lens, not storm species.'
  },
  annotations: FAMILY_LABELS,

  create: async ctx => (await import('./hurricane-families.compute')).createHurricaneFamilies(ctx),

  story: [
    {
      id: 'the-question',
      title: 'Do Atlantic hurricanes follow a few routes?',
      headline: 'Routes form a continuum, not boxes',
      textAlternative: 'Fine neutral hurricane routes cover a quiet Atlantic paper map.',
      optionsMode: 'fresh',
      body: 'Every dot in the Atlantic archive is a six-hourly NOAA IBTrACS best-track fix. The **Storms** card reports the loaded record; this story stays in its satellite-era regional sample because older offshore observations are not directly comparable.\n\nLines are colored by peak wind category, which is a wind classification rather than a damage measure. The routes suggest several broad patterns, but this scene tests whether those patterns remain useful when distance and clustering choices change.',
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {colorBy: 'peak', showRoutes: false, showFamilyMeans: false, trackOpacity: 0.55},
      controls: ['colorBy', 'trackOpacity'],
      readouts: ['storms']
    },
    {
      id: 'resample',
      title: 'First make every route the same length',
      headline: 'Equal samples make routes comparable',
      textAlternative:
        'One storm route carries evenly spaced resample dots over faint context routes.',
      optionsMode: 'fresh',
      body: 'To compare two storms point by point they need the same number of points. **`GPUTrajectoryResample`** rebuilds every track at equal fractions along its path, so a slow loop and a fast run of the same shape have comparable samples. The dots are drawn over the faded tracks.\n\nSwitch **Route samples spaced by** to *Equal time steps*: a storm that stalled in the Gulf now puts more samples there, so the comparison also weighs duration. **Samples per route** and spacing are compile-time options (a rebuild badge shows), and the similarity pass is redone for the new routes.',
      camera: {longitude: -62, latitude: 27, zoom: 3.3, transitionMs: 1400},
      options: {showRoutes: true, colorBy: 'plain', trackOpacity: 0.55},
      controls: ['showRoutes', 'routeSpacing', 'routeSamples'],
      readouts: ['storms']
    },
    {
      id: 'frechet',
      title: 'How alike are two routes? The Frechet distance',
      headline: 'A leash measures whole-route likeness',
      textAlternative:
        'Two selected hurricane routes are compared against dim Atlantic context paths.',
      optionsMode: 'fresh',
      body: '**`GPUTrackSimilarity`** scores every unordered pair of routes in one GPU pass. The **discrete Frechet distance** is the shortest leash that lets two walkers follow their routes in order, which makes it sensitive to shape *and* direction, unlike the Hausdorff distance (the largest gap between curves, ignoring order).\n\nThe white route is the selected storm; the orange route is its closest match. The orange connector is the widest matched pair along their discrete-Frechet coupling. **Click any storm** to choose another. The histogram shows every distance from the selection; its marker is the active **Distance color range**. Change **Route distance** to *Hausdorff* and see which routes move closer.',
      camera: {longitude: -75, latitude: 27, zoom: 3.6, transitionMs: 1400},
      options: {
        showRoutes: false,
        colorBy: 'similarity',
        trackOpacity: 0.7,
        similarityRangeKm: 3000
      },
      controls: ['colorBy', 'distanceMeasure', 'similarityRangeKm'],
      readouts: ['selected', 'leash', 'distanceChart', 'pairs']
    },
    {
      id: 'families',
      title: 'Let k-means find the families',
      headline: 'Clusters change when choices change',
      textAlternative: 'Qualitative family routes pair with their linked clustering summary.',
      optionsMode: 'fresh',
      body: 'Clustering needs points in a plane, so the distance matrix is **embedded in two dimensions** (classical scaling) and **`GPUKMeans`** groups the storms: assign every storm to the nearest center, move the centers, repeat. The colors are families and the bars show how many storms each holds.\n\nTurn on family representatives. The default **medoid** is a real, central route: the member with the smallest total route distance to its family. The **mean** averages resample points and can be useful but need not be a storm that occurred. The live **Families** readout gives each group’s start, end, length and hurricane share. Slide **Number of families**: groups split and merge because routes form a continuum.',
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {
        colorBy: 'family',
        showFamilyMeans: true,
        representative: 'medoid',
        showRoutes: false,
        familyCount: 5
      },
      controls: ['familyCount', 'representative', 'showFamilyMeans'],
      readouts: ['embeddingChart', 'familyChart', 'families']
    },
    {
      id: 'stability',
      title: 'Distances become a two-dimensional map',
      headline: 'Nearby dots are similar routes, not places',
      textAlternative:
        'A linked embedding scatter uses the same family colours as the faint Atlantic routes.',
      optionsMode: 'fresh',
      body: 'The scatter is the space `GPUKMeans` actually sees: each dot is a route after the distance matrix is embedded in two dimensions. Near dots mean similar **paths**, not nearby locations. Click a dot to select its matching Atlantic route.\n\nk-means can settle into different local splits. Change **Starting centers** or the **k-means++ seed** and watch the family colours move across the same continuum. The live **k-means** readout reports the iterations and the variance explained; the default is an example, never a natural taxonomy.',
      options: {colorBy: 'family', showFamilyMeans: true, familyCount: 5},
      controls: ['initialization', 'seed', 'iterations'],
      readouts: ['embeddingChart', 'kmeans', 'embedding', 'familyChart']
    },
    {
      id: 'limits',
      title: 'What to remember, and what to try',
      headline: 'Families are a lens, not a taxonomy',
      textAlternative:
        'Neutral routes and a few family summaries stress continuous route variation.',
      optionsMode: 'fresh',
      body: 'Families describe **routes**, not causes: a hurricane’s path depends on the steering flow of its year. Six-hourly fixes are straight-lined, tight loops can be smoothed, and this regional time span is a sample rather than climate. The embedding keeps only part of the distance variation; use its live readout before treating small cluster boundaries as meaningful.\n\n**Try it:** select any storm, then compare Hausdorff and Frechet closest routes; set **Number of families** to 3 and 8; switch **Color storms by** to *Season*; click a recurving storm and compare its nearest route and rendered leash.',
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {colorBy: 'family', showFamilyMeans: true},
      controls: ['colorBy', 'distanceMeasure', 'familyCount'],
      readouts: ['selected', 'families']
    }
  ]
});
