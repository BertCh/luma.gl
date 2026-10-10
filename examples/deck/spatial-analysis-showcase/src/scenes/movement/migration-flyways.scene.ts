// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import type {MigrationFlywaysOptions} from './migration-flyways.compute';
import {MOVEMENT_CREDITS} from './movement-style';
import {
  MIGRATION_SEASONS,
  MIGRATION_SPECIES_COLORS,
  MIGRATION_SPECIES_LABELS
} from './migration-shared';

const EUROPE_AFRICA_VIEW = {longitude: 3, latitude: 33, zoom: 3.1};

export default defineScene<MigrationFlywaysOptions>({
  id: 'migration-flyways',
  title: 'Which way do the harriers and spoonbills fly?',
  chapter: 'movement',
  order: 7,
  summary:
    "101 GPS-tracked animal-years of marsh harriers, Montagu's harriers and spoonbills from the Low Countries: line density turns two hundred thousand fixes into flyways, by species and season, and a Frechet comparison asks whether a bird repeats its own route.",
  contributors: ['GPULineDensity', 'GPUTrajectoryResample', 'GPUTrackSimilarity'],
  datasets: [
    {id: 'poopdeck-animals', role: 'GPS tracks of 42 birds, years folded onto one calendar'}
  ],
  initialView: EUROPE_AFRICA_VIEW,
  basemap: ground('night', {labels: 'none'}),
  furniture: {
    title: {
      title: 'Where birds concentrate',
      subtitle: 'Track length per cell · years folded',
      chips: ['Track kilometres are not birds']
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(MOVEMENT_CREDITS.birds)
  },

  options: [
    {
      kind: 'select',
      id: 'species',
      label: 'Species',
      group: 'Flyway density',
      apply: 'compile',
      default: 'all',
      help: 'Which tracks are summed. Each species and season is a different set of path inputs, so a combination is compiled when you first pick it (a one-off graph build) and cached after that.',
      options: [
        {value: 'all', label: 'All three species'},
        {value: 'marsh', label: 'Western marsh harrier'},
        {value: 'montagu', label: "Montagu's harrier"},
        {value: 'spoonbill', label: 'Eurasian spoonbill'}
      ]
    },
    {
      kind: 'select',
      id: 'season',
      label: 'Season',
      group: 'Flyway density',
      apply: 'compile',
      default: 'year',
      help: 'Only fixes inside the season are summed (tracks are cut to it). Spring and autumn show the two passages separately; years are folded onto one calendar, so a season mixes every tagged year.',
      options: Object.entries(MIGRATION_SEASONS).map(([value, season]) => ({
        value,
        label: season.label
      }))
    },
    {
      kind: 'slider',
      id: 'cellSize',
      label: 'Cell size',
      group: 'Flyway density',
      apply: 'param',
      min: 0.25,
      max: 1,
      step: 0.05,
      default: 0.35,
      unit: 'degrees',
      help: 'Side of a grid cell in degrees of longitude and latitude (0.35 degrees is about 39 km by 25 km at 50 N). The grid is 320 by 280 cells anchored at 20 W, 3 N: the cell size is a parameter-buffer write, so dragging it never recompiles.'
    },
    {
      kind: 'select',
      id: 'cellValue',
      label: 'Cell value',
      group: 'Flyway density',
      apply: 'param',
      default: 'density',
      help: 'Line density divides the track length by the exact spherical area of each cell, so cells at different latitudes compare fairly. Length is the plain sum in kilometers (cells shrink poleward, so the same traffic reads lower).',
      options: [
        {value: 'density', label: 'Line density (km per 1,000 km2)'},
        {value: 'length', label: 'Track length (km per cell)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'liftLow',
      label: 'Square-root color scale',
      group: 'Flyway density',
      apply: 'param',
      default: true,
      help: 'Applies a square root after normalizing, which lifts thin corridors out from under the busy ones. The color range always ends at the 99.5th percentile of the non-empty cells.'
    },
    {
      kind: 'slider',
      id: 'probeLatitude',
      label: 'Probe latitude',
      group: 'Flyway density',
      apply: 'param',
      min: 10,
      max: 58,
      step: 0.5,
      default: 37,
      unit: 'N',
      help: 'The latitude row whose track length is summarized in the Corridor readout and marked on the corridor chart: the central 80% of the track length lies between the 10th and 90th percentile longitudes.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'inferno',
      help: 'Ramp of the density cells.',
      options: [
        {value: 'inferno', label: 'Inferno'},
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showTracks',
      label: 'Show the tracks',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws every animal-year as a thin line over the density cells.'
    },
    {
      kind: 'select',
      id: 'colorBy',
      label: 'Color tracks by',
      group: 'Display',
      apply: 'param',
      default: 'species',
      disabledWhen: state => !state.showTracks,
      help: 'Species uses one color per species; date uses the day of the (folded) year; route distance colors every track by its distance to the selected one (click a track to select it).',
      options: [
        {value: 'species', label: 'Species'},
        {value: 'date', label: 'Day of the year'},
        {value: 'similarity', label: 'Route distance to the selected track'},
        {value: 'plain', label: 'One color'}
      ]
    },
    {
      kind: 'slider',
      id: 'trackOpacity',
      label: 'Track opacity',
      group: 'Display',
      apply: 'param',
      min: 0.05,
      max: 1,
      step: 0.05,
      default: 0.35,
      disabledWhen: state => !state.showTracks,
      help: 'Lower it to read the density underneath; raise it to follow single birds.'
    },
    {
      kind: 'select',
      id: 'routeSpacing',
      label: 'Route samples spaced by',
      group: 'Route similarity',
      apply: 'compile',
      default: 'arc-length',
      help: 'Each track is resampled to 96 points before the comparison (a different compiled resample graph each; both are built up front). Equal distance compares the shape of the route; equal time also weighs where the bird spent its time.',
      options: [
        {value: 'arc-length', label: 'Equal distance along the path'},
        {value: 'time', label: 'Equal time steps'}
      ]
    },
    {
      kind: 'select',
      id: 'similarityMetric',
      label: 'Route distance',
      group: 'Route similarity',
      apply: 'param',
      default: 'frechet',
      help: 'Both distances are computed for every pair in one pass; this picks the column that is charted and colored.',
      options: [
        {
          value: 'frechet',
          label: 'Frechet',
          help: 'Shortest leash for two walkers following the routes in order: sensitive to timing and direction.'
        },
        {
          value: 'hausdorff',
          label: 'Hausdorff',
          help: 'Farthest any point of one route is from the other, ignoring order.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'minCoverageDays',
      label: 'Minimum days of data',
      group: 'Route similarity',
      apply: 'param',
      min: 30,
      max: 330,
      step: 10,
      default: 240,
      unit: 'days',
      help: 'Only tracks that span at least this many days enter the comparison and the chart: a tag that ran for two months cannot repeat a whole annual route. It is a CPU filter on pairs already scored.'
    },
    {
      kind: 'slider',
      id: 'similarityRangeKm',
      label: 'Distance color range',
      group: 'Route similarity',
      apply: 'param',
      min: 200,
      max: 3000,
      step: 100,
      default: 1500,
      unit: 'km',
      disabledWhen: state => state.colorBy !== 'similarity' || !state.showTracks,
      help: 'Route distance at the top of the ramp when tracks are colored by distance to the selected track. Tracks that do not meet the minimum days are drawn grey.'
    }
  ],

  readouts: [
    {
      id: 'corridorChart',
      label: 'How wide is the flyway at each latitude?',
      kind: 'chart',
      help: 'For each latitude row of the density grid, the longitude of the median track length (line) and of the 10th to 90th percentile (band). A narrow band is a bottleneck; the vertical rule is the probe latitude.'
    },
    {
      id: 'evidenceChart',
      label: 'What sample makes this map?',
      kind: 'chart',
      help: 'Tagged animal-years with at least two recorded fixes in the selected season, by species. Tagged years are repeated measurements of a small convenience sample, not a population count.'
    },
    {
      id: 'fidelityChart',
      label: 'Does a bird repeat its own route?',
      kind: 'chart',
      help: 'Share of pairs at each route distance for the same bird in two different years, for two different birds of one species and for two species. If the same-bird curve sits left of the others, birds are faithful to their route.'
    },
    {id: 'birds', label: 'Tracks'},
    {
      id: 'evidence',
      label: 'Mapped tracking evidence',
      help: 'The animal-years, individual birds and recorded fixes actually feeding the selected seasonal density. Origin rings are averages of each species’ first recorded positions.'
    },
    {id: 'flyway', label: 'Flyway density'},
    {id: 'busiest', label: 'Busiest cell'},
    {
      id: 'probe',
      label: 'Corridor at the probe latitude',
      help: 'Width in kilometers of the band holding the central 80% of the track length in the probe row.'
    },
    {id: 'routes', label: 'Route comparison'},
    {
      id: 'fidelity',
      label: 'Route fidelity',
      help: 'Median distance between routes of the same bird in different years, of different birds of the species and of different species, for tracks that meet the minimum days.'
    },
    {id: 'selected', label: 'Selected track'}
  ],

  legends: state => [
    state.cellValue === 'density'
      ? {
          kind: 'ramp' as const,
          id: 'density',
          title: 'Track length per 1,000 km2',
          ramp: state.ramp,
          extent: 'gpu' as const,
          sqrtScale: state.liftLow,
          unit: 'km / 1,000 km2'
        }
      : {
          kind: 'ramp' as const,
          id: 'length',
          title: 'Track length in the cell',
          ramp: state.ramp,
          extent: 'gpu' as const,
          sqrtScale: state.liftLow,
          unit: 'km'
        },
    ...(state.showTracks && state.colorBy === 'species'
      ? [
          {
            kind: 'categories' as const,
            title: 'Species',
            entries: MIGRATION_SPECIES_COLORS.map((color, index) => ({
              color,
              label: MIGRATION_SPECIES_LABELS[index]
            }))
          }
        ]
      : []),
    ...(state.showTracks && state.colorBy === 'date'
      ? [
          {
            kind: 'ramp' as const,
            title: 'Day of the folded year',
            ramp: 'cividis' as const,
            extent: [0, 366] as const,
            labels: ['1 Jan', '31 Dec'] as const
          }
        ]
      : []),
    ...(state.showTracks && state.colorBy === 'similarity'
      ? [
          {
            kind: 'ramp' as const,
            title: `${state.similarityMetric === 'frechet' ? 'Frechet' : 'Hausdorff'} distance to the selected track`,
            ramp: 'cividis' as const,
            extent: [0, state.similarityRangeKm] as const,
            unit: 'km',
            format: (value: number) => value.toFixed(0)
          }
        ]
      : []),
    {
      kind: 'categories' as const,
      title: 'Selection',
      entries: [
        {color: [255, 255, 255, 255] as const, label: 'Selected track (click one)'},
        {color: [255, 214, 64, 255] as const, label: 'Same bird, other tagged years'}
      ]
    }
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPULineDensity, GPUTrajectoryResample, GPUTrackSimilarity, getGPULineDensityParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';

// positions: longitude/latitude degrees of the ${state.species === 'all' ? 'tracks' : 'selected species'} cut to the ${state.season} window
// 1. Line length and density per grid cell, measured as great-circle pieces
graph.add(new GPULineDensity({
  positions, pathOffsets, columns: 320, rows: 280,
  spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'},                  // lengths in meters, densities in 1/m
  parameters: densityParameters.importToGraph(graph),
  output: {lengths, densities, overflow, totalRecords}
}));
densityParameters.write(getGPULineDensityParameterValues({
  minX: -20, minY: 3, cellWidth: ${state.cellSize}, cellHeight: ${state.cellSize}      // degrees: a buffer write
}));

// 2. Routes: ${96} points per track (planar azimuthal-equidistant meters), then every pair
graph.add(new GPUTrajectoryResample({
  positions: meters, timestamps, trackOffsets, sampleCount: 96, spacing: '${state.routeSpacing}', samples: routes
}));
graph.add(new GPUTrackSimilarity({
  positionsA: routes, offsetsA: routeOffsets, pairA, pairB,
  hausdorff, frechet, maxFrechetVertices: 96      // distance column: ${state.similarityMetric}
}));`,

  about: {
    what: '`GPULineDensity` clips every track segment to a grid of longitude/latitude cells, walks it through the cells it crosses and sums the great-circle length per cell (and the length per unit of exact spherical cell area). `GPUTrajectoryResample` rebuilds each track as 96 evenly spaced points and `GPUTrackSimilarity` scores every pair of routes with the Hausdorff and the discrete Frechet distance.',
    why: 'A flyway is where animals concentrate, not where one bird went. Summing track length per cell shows corridors and bottlenecks that conservation planning cares about, and comparing a bird with itself in another year says whether a route is learned and repeated or improvised.',
    howToRead:
      'Bright cells carry the most track length per area. A thin bright line is a bottleneck, a broad glow a front. The thin colored lines are the tracks; the white line is the selected animal-year and the yellow ones are the same bird in other years. The map is Web Mercator; densities and distances are measured on the sphere. Tracks are folded onto a single calendar year, so a season mixes every tagged year of a bird.'
  },

  create: async ctx => (await import('./migration-flyways.compute')).createMigrationFlyways(ctx),

  story: [
    {
      id: 'the-question',
      headline: 'Dense tracks reveal a flyway',
      textAlternative: 'A luminous density map of Europe and Africa shows bird flyways.',
      optionsMode: 'fresh',
      title: "Where do the Low Countries' raptors and spoonbills spend the winter?",
      body: "Forty-two birds from Flanders, the Netherlands and the border between them carried GPS tags that logged a fix about every hour, thinned here to one every two hours: **55 marsh harrier**, **23 Montagu's harrier** and **23 spoonbill** animal-years, 226,000 fixes in all. The source is the poopdeck.gl `animals` archive, which folds every tagged year onto one calendar year, so a bird tracked for six years appears as six overlapping tracks.\n\nThe lines are colored by species. The glow behind them is what the next step builds. Where do the harriers go, and do the spoonbills follow them?",
      camera: {...EUROPE_AFRICA_VIEW, transitionMs: 1200},
      controls: [],
      readouts: ['birds', 'selected']
    },
    {
      id: 'sample',
      headline: 'A bright corridor is a sample, not a census',
      textAlternative:
        'Map labels mark the mean first recorded locations for each tracked species, alongside a bar chart of the animal-years in the sample.',
      optionsMode: 'fresh',
      title: 'First inspect the tracking evidence',
      body: 'The density is built from a **convenience sample**: 42 tagged birds, with several years from some of the same individuals. The ring labels mark each species’ mean first recorded position and the bar chart counts the tagged animal-years that actually contribute at least two fixes to the selected season. Those are inputs to a route-density estimate, not counts of birds using a corridor.\n\nChange **Season** and **Species**. The **Mapped tracking evidence** readout changes with the tracks and fixes that enter the calculation. This is why a narrow bright band is evidence about these tagged routes, not a measured share of the population.',
      options: {season: 'year', species: 'all', showTracks: true, trackOpacity: 0.25},
      camera: {...EUROPE_AFRICA_VIEW, transitionMs: 1200},
      highlight: {readout: 'evidence'},
      controls: ['season', 'species'],
      readouts: ['evidence', 'evidenceChart']
    },
    {
      id: 'line-density',
      headline: 'Every cell counts track length',
      textAlternative: 'A gridded density surface replaces overlapping routes.',
      optionsMode: 'fresh',
      title: 'Summing the track length of every cell',
      body: '**`GPULineDensity`** clips every segment of every track to a grid and adds up the great-circle length that falls in each cell, then divides by the exact spherical area of the cell. A cell that many birds cross in many years is bright; a cell crossed once is dim. This is the QGIS "line density" tool, run on 226,000 fixes in one GPU pass.\n\nSlide **Cell size** below: coarser cells smooth the corridors, finer ones follow single tracks, and nothing recompiles because the size is a parameter write. Switch **Cell value** to *Track length* and the northern cells darken, because a degree of longitude gets narrower toward the pole. Turn **Square-root color scale** off to see only the busiest cells.',
      options: {showTracks: true, trackOpacity: 0.25},
      highlight: {readout: 'busiest'},
      controls: ['cellSize', 'cellValue', 'liftLow'],
      readouts: ['flyway', 'busiest']
    },
    {
      id: 'species',
      headline: 'Species use different corridors',
      textAlternative: 'Species routes appear over a shared flyway surface.',
      optionsMode: 'fresh',
      title: 'Three species, three flyways',
      body: "Pick a species in **Species** below. **Marsh and Montagu's harriers** leave the Low Countries, funnel through Iberia and across the Strait of Gibraltar, and fan out over the Sahel from Senegal to Mali, where the median bird on 15 January is at about 14 N. **Spoonbills** do not: most winter on the Atlantic coast of France and Iberia, a few hundred kilometers from home, and none of these tags went south of 33 N.\n\nEach species is a different set of paths, so the first time you pick one the shell builds a graph for it (the **rebuild** badge); after that the pick is instant.",
      options: {species: 'marsh', showTracks: false},
      highlight: {readout: 'flyway'},
      controls: ['species', 'showTracks', 'ramp'],
      readouts: ['flyway']
    },
    {
      id: 'bottleneck',
      headline: 'Cell size changes the bottleneck',
      textAlternative: 'A density grid changes resolution over a narrow passage.',
      optionsMode: 'fresh',
      title: 'Where does the flyway pinch?',
      body: 'Choose the **Autumn** passage for all species and drag **Probe latitude** below. The chart shows, for every latitude row, the longitude range that holds the central 80% of the track length. At **50 N** the harriers leave from a belt only about 2 degrees wide (Belgium and the Netherlands); at **37 N** the band is still only a few degrees wide, the funnel of southern Iberia and the Strait of Gibraltar, and farther south it fans out over the Sahara and the Sahel.\n\nA narrow band is a bottleneck: whatever happens in that strip (a wind farm, a drained wetland, a hunting season) happens to most of the population. The readout gives the width in kilometers.',
      options: {species: 'all', season: 'autumn', probeLatitude: 37, showTracks: false},
      camera: {longitude: -4, latitude: 36, zoom: 4.3, transitionMs: 1400},
      highlight: {readout: 'probe'},
      controls: ['season', 'probeLatitude'],
      readouts: ['probe', 'corridorChart']
    },
    {
      id: 'spring',
      headline: 'Season changes the route picture',
      textAlternative: 'Spring and autumn densities are compared on one map.',
      optionsMode: 'fresh',
      title: 'Do they come back the way they left?',
      body: "Set **Season** to *Spring* and compare it with *Autumn*. Marsh harriers run north on almost the same corridor they used in autumn, a few weeks earlier than Montagu's harriers, which cross 20 N about a month later (early April against early March). Pick **Montagu's harrier** in **Species** and look at the central Mediterranean: one spring track comes north through Italy, a route no autumn track uses.\n\nOne track is an anecdote, not a loop migration: with 23 animal-years of Montagu's harrier all you can say is that the return trip is a little less tidy than the departure.",
      options: {season: 'spring', species: 'montagu', showTracks: true, trackOpacity: 0.5},
      camera: {longitude: 2, latitude: 35, zoom: 3.4, transitionMs: 1400},
      controls: ['season', 'species'],
      readouts: ['flyway']
    },
    {
      id: 'fidelity',
      headline: 'A long track is not many birds',
      textAlternative: 'One selected route is compared with the population density.',
      optionsMode: 'fresh',
      title: 'Does a bird repeat its own route?',
      body: '**`GPUTrajectoryResample`** rebuilds every track as 96 points and **`GPUTrackSimilarity`** scores all 5,050 pairs with the discrete **Frechet distance**: the shortest leash that lets two walkers follow their routes in order. The chart compares pairs that are the same bird in two different years against pairs of different birds, for tracks with enough days of data.\n\nSet **Color tracks by** to *Route distance to the selected track* and click a track: its other tagged years (yellow) should be the coldest lines if birds are faithful to a route. Use **Minimum days of data** to include or drop short tags, and **Route distance** to switch to Hausdorff.\n\nThe honest limits: tags are on 42 birds, only a few of which have two or more complete years; the years are folded onto one calendar; and Frechet on 96 points describes the shape of a route, not every detour. **Try:** set the minimum to 300 days and see how the same-bird curve changes.',
      options: {
        colorBy: 'similarity',
        showTracks: true,
        trackOpacity: 0.8,
        season: 'year',
        species: 'all',
        cellSize: 0.5
      },
      camera: {...EUROPE_AFRICA_VIEW, transitionMs: 1200},
      highlight: {readout: 'fidelity'},
      controls: ['colorBy', 'minCoverageDays', 'similarityMetric'],
      readouts: ['fidelity', 'fidelityChart', 'selected']
    }
  ]
});
