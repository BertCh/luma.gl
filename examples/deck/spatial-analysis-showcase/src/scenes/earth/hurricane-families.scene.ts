// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import type {HurricaneFamiliesOptions} from './hurricane-families.compute';
import {
  HURRICANE_CATEGORY_COLORS,
  HURRICANE_CATEGORY_LABELS,
  HURRICANE_FAMILY_COLORS
} from './hurricane-data';

const BASIN_VIEW = {longitude: -55, latitude: 29, zoom: 3.05};

export default defineScene<HurricaneFamiliesOptions>({
  id: 'hurricane-families',
  title: 'Do Atlantic hurricanes follow a few routes?',
  chapter: 'earth',
  order: 10,
  summary:
    'Every Atlantic storm since 1980 (739 tracks from NOAA IBTrACS) resampled to equal-length routes, all 273,000 pairs scored with Frechet distance on the GPU and clustered into track families with k-means. Click a storm to color every other by how far its route is from it.',
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
      default: 'viridis',
      help: 'Used by the distance and season colors. Distance is read in kilometers; season runs from 1980 to 2025.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showFamilyMeans',
      label: 'Show family mean routes',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'The average of the resampled routes in each family, point by point: a typical storm of the family.'
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
      id: 'selected',
      label: 'Selected storm',
      help: 'Click a storm to read its peak, dates, family and its closest track.'
    }
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

  create: async ctx => (await import('./hurricane-families.compute')).createHurricaneFamilies(ctx),

  story: [
    {
      id: 'the-question',
      title: 'Do Atlantic hurricanes follow a few routes?',
      body: 'Every dot of the Atlantic season is a six-hourly fix in the **NOAA IBTrACS** best tracks. This is the satellite era only, **739 storms from 1980 to 2025** (20,818 fixes). Older records exist back to 1851, but before satellites storms far from land were missed or undersampled, so their routes and counts are not comparable.\n\nThe lines are colored by **peak category**. The Saffir-Simpson class comes from the 1-minute sustained wind (35 kt tropical storm, 64 kt Category 1, 137 kt Category 5), not from damage. Some groups are easy to see: a stream from the Cape Verde islands crossing the ocean, a tangle in the Gulf, and curves that turn north and out to sea. Can we get the computer to find them?',
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {colorBy: 'peak', showRoutes: false, showFamilyMeans: false, trackOpacity: 0.55},
      controls: ['colorBy', 'trackOpacity'],
      readouts: ['storms']
    },
    {
      id: 'resample',
      title: 'First make every route the same length',
      body: 'To compare two storms point by point they need the same number of points. **`GPUTrajectoryResample`** rebuilds every track as **48 samples** placed at equal distances along the path, so a slow loop and a fast run of the same shape have the same samples. The dots are drawn over the faded tracks.\n\nSwitch **Route samples spaced by** to *Equal time steps*: now a storm that stalled in the Gulf piles its samples there, and the shape comparison also weighs how long a storm took. **Samples per route** and the spacing are compile-time options (a rebuild badge shows), and the similarity pass is redone for the new routes.',
      camera: {longitude: -62, latitude: 27, zoom: 3.3, transitionMs: 1400},
      options: {showRoutes: true, colorBy: 'plain', trackOpacity: 0.55},
      controls: ['showRoutes', 'routeSpacing', 'routeSamples'],
      readouts: ['storms']
    },
    {
      id: 'frechet',
      title: 'How alike are two routes? The Frechet distance',
      body: '**`GPUTrackSimilarity`** scores every one of the **272,691 pairs** of routes in one GPU pass. The **discrete Frechet distance** is the shortest leash that lets two walkers follow their routes in order, which makes it sensitive to shape *and* direction, unlike the Hausdorff distance (the largest gap between the curves, ignoring order).\n\nThe storm outlined in white is **Katrina (2005)**. Every other storm is colored by its distance to it: bright means a similar route. **Click any storm** to choose another. The histogram shows how many storms lie at each distance; the marker is the end of the color range, set with **Distance color range**. Change **Route distance** to *Hausdorff* and see which storms move closer.',
      camera: {longitude: -75, latitude: 27, zoom: 3.6, transitionMs: 1400},
      options: {
        showRoutes: false,
        colorBy: 'similarity',
        trackOpacity: 0.7,
        similarityRangeKm: 3000
      },
      callout: {coordinate: [-88.6, 26.3], text: 'Katrina near its peak, 28 Aug 2005'},
      controls: ['colorBy', 'distanceMeasure', 'similarityRangeKm'],
      readouts: ['selected', 'distanceChart', 'pairs']
    },
    {
      id: 'families',
      title: 'Let k-means find the families',
      body: "Clustering needs points in a plane, so the distance matrix is **embedded in two dimensions** (classical scaling) and **`GPUKMeans`** groups the storms: assign every storm to the nearest center, move the centers, repeat. The colors are the families, the thick lines their mean routes, and the bars show how many storms each holds.\n\nWith **5 families** you typically get a long Cape Verde family crossing the tropical Atlantic, a mid-Atlantic family that turns north, recurving storms that end near Europe (the longest tracks, with the highest share of hurricanes), storms that form near Florida and run up the Southeast coast, and a Gulf and western Caribbean family. The **Families** readout gives each one's start, end, length and hurricane share. Slide **Number of families**: groups split and merge, which is the honest message, because the routes form a continuum.",
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {colorBy: 'family', showFamilyMeans: true, showRoutes: false, familyCount: 5},
      controls: ['familyCount', 'colorBy', 'showFamilyMeans'],
      readouts: ['familyChart', 'families']
    },
    {
      id: 'stability',
      title: 'Are the families stable?',
      body: 'k-means finds a local optimum, so the answer depends on the **starting centers**. Switch **Starting centers** to *The first k storms* (the earliest storms of 1980, wherever they happen to lie): the split comes out different. With *k-means++* the centers start spread apart; change the **k-means++ seed** and the family sizes shift by tens of storms while the same few routes stay recognisable, which is what real structure looks like.\n\nThe **k-means** readout shows how many iterations were needed and what share of the embedding variance the families explain, and **Iterations** caps the loop. The seed, the iterations and the starting centers are compile-time options of `GPUKMeans`, so each change rebuilds only the clustering graph; the 273,000 distances are kept.',
      options: {colorBy: 'family', showFamilyMeans: true, familyCount: 5},
      controls: ['initialization', 'seed', 'iterations'],
      readouts: ['kmeans', 'embedding', 'familyChart']
    },
    {
      id: 'limits',
      title: 'What to remember, and what to try',
      body: "Families describe **routes**, not causes: a hurricane's path depends on the steering flow of its year, and the same Cape Verde wave can recurve or hit the Caribbean. Six-hourly fixes are straight-lined, so tight loops are smoothed; tracks are cut at 105 W; and 1980-2025 is a sample of 46 seasons, not the climate. The two-dimensional embedding keeps most but not all of the distance variance (about 84% with 48 samples; see the readout), which limits how finely families can be separated.\n\n**Try it:** select **Katrina**, then Hausdorff and Frechet and compare the closest storms; set **Number of families** to 3 and 8; switch **Color storms by** to *Season* to see whether the families drift in time; click a recurving storm and read how far the nearest Gulf storm is from it.",
      camera: {...BASIN_VIEW, transitionMs: 1200},
      options: {colorBy: 'family', showFamilyMeans: true},
      controls: ['colorBy', 'distanceMeasure', 'familyCount'],
      readouts: ['selected', 'families']
    }
  ]
});
