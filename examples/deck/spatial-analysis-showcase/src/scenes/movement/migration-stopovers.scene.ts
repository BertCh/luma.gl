// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import type {MigrationStopoversOptions} from './migration-stopovers.compute';
import {MOVEMENT_CREDITS} from './movement-style';
import {
  formatYearDay,
  MIGRATION_SPECIES_COLORS,
  MIGRATION_SPECIES_LABELS
} from './migration-shared';

const EUROPE_AFRICA_VIEW = {longitude: 2, latitude: 33, zoom: 3.1};

const CLUSTER_PALETTE = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
] as const;

export default defineScene<MigrationStopoversOptions>({
  id: 'migration-stopovers',
  title: 'Where do migrating raptors stop, and for how long?',
  chapter: 'movement',
  order: 8,
  summary:
    "Dwells found in 101 GPS-tracked animal-years of marsh harriers, Montagu's harriers and spoonbills, filtered to stopovers and clustered on the GPU into ranked sites with their dwell in bird-days.",
  contributors: ['GPUTrajectoryMetrics', 'GPUSpatialClustering'],
  datasets: [{id: 'poopdeck-animals', role: 'GPS tracks of 42 birds, a fix every two hours'}],
  initialView: EUROPE_AFRICA_VIEW,
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Where migrating raptors stop',
      subtitle: 'Slow runs clustered into sites · years folded',
      chips: ['A stop is a rule']
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(MOVEMENT_CREDITS.birds)
  },

  options: [
    {
      kind: 'slider',
      id: 'stopSpeed',
      label: 'Slow-step speed',
      group: 'Dwells (metrics)',
      apply: 'param',
      min: 0.25,
      max: 4,
      step: 0.25,
      default: 1.5,
      unit: 'm/s',
      help: 'A two-hour step is slow when the bird moves less than this speed times the step length (1.5 m/s is 10.8 km in two hours). A dwell is a run of slow steps. A parameter-buffer write.'
    },
    {
      kind: 'slider',
      id: 'minStayHours',
      label: 'Minimum dwell',
      group: 'Dwells (metrics)',
      apply: 'param',
      min: 12,
      max: 240,
      step: 6,
      default: 72,
      unit: 'h',
      format: value =>
        value >= 48 ? `${(value / 24).toFixed(value % 24 ? 1 : 0)} days` : `${value} h`,
      help: 'A run of slow steps is reported as a dwell when it lasts at least this long. Overnight roosts are far shorter; three days separates a real stay from a rest. A parameter-buffer write.'
    },
    {
      kind: 'select',
      id: 'species',
      label: 'Species',
      group: 'Stopover rules',
      apply: 'param',
      default: 'all',
      help: 'Which species count. A mask passed to the selection kernel, so switching never recompiles.',
      options: [
        {value: 'all', label: 'All three species'},
        {value: 'marsh', label: 'Western marsh harrier'},
        {value: 'montagu', label: "Montagu's harrier"},
        {value: 'spoonbill', label: 'Eurasian spoonbill'}
      ]
    },
    {
      kind: 'slider',
      id: 'maxStayDays',
      label: 'Longest stopover',
      group: 'Stopover rules',
      apply: 'param',
      min: 3,
      max: 120,
      step: 1,
      default: 14,
      unit: 'days',
      help: 'Dwells longer than this are home ranges (a breeding territory, a wintering area), not stopovers, and are dropped before clustering. See the dwell histogram: the long tail beyond about two weeks is residence.'
    },
    {
      kind: 'range',
      id: 'dayRange',
      label: 'Stopovers that begin between',
      group: 'Stopover rules',
      apply: 'param',
      min: 0,
      max: 366,
      step: 1,
      default: [0, 366],
      format: value => formatYearDay(Math.min(365, value)),
      help: 'Keeps the dwells that start inside this window of the folded year: autumn is about 1 Aug to 30 Nov, spring 1 Mar to 15 Jun.'
    },
    {
      kind: 'slider',
      id: 'epsilonKm',
      label: 'Site radius (epsilon)',
      group: 'Sites (clustering)',
      apply: 'param',
      min: 10,
      max: 300,
      step: 5,
      default: 80,
      unit: 'km',
      help: 'Two stopovers are neighbors when their centers are at most this far apart (planar azimuthal-equidistant meters). A larger radius merges nearby sites; a smaller one splits them. A parameter write.'
    },
    {
      kind: 'slider',
      id: 'minimumStops',
      label: 'Minimum stops per site',
      group: 'Sites (clustering)',
      apply: 'param',
      min: 1,
      max: 20,
      step: 1,
      default: 3,
      help: 'A stopover is a core point when this many stopovers (itself included) lie within the radius. Sparse stops that do not reach it are noise: the isolated stops of an unusual detour.'
    },
    {
      kind: 'select',
      id: 'rankBy',
      label: 'Rank sites by',
      group: 'Sites (clustering)',
      apply: 'param',
      default: 'dwell',
      help: 'Dwell adds up the bird-days spent at the site, birds counts the distinct tagged birds that used it, stops counts the stopovers. A site used briefly by many birds ranks differently from one occupied for long by a few.',
      options: [
        {value: 'dwell', label: 'Total dwell (bird-days)'},
        {value: 'birds', label: 'Distinct birds'},
        {value: 'stops', label: 'Number of stops'}
      ]
    },
    {
      kind: 'select',
      id: 'colorStops',
      label: 'Color stops by',
      group: 'Display',
      apply: 'param',
      default: 'stay',
      help: 'Length of stay uses the ramp below; site colors the stops by DBSCAN cluster (grey is noise); species uses the species colors.',
      options: [
        {value: 'stay', label: 'Length of stay'},
        {value: 'cluster', label: 'Site (cluster)'},
        {value: 'species', label: 'Species'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showSites',
      label: 'Show ranked sites',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws the 24 highest-ranked sites as discs colored by their share of the top site, with a halo on the top three. Click a disc to select it.'
    },
    {
      kind: 'toggle',
      id: 'showTracks',
      label: 'Show the tracks',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws every animal-year as a thin line under the stops.'
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
      default: 0.25,
      disabledWhen: state => !state.showTracks,
      help: 'Lower it to read the stops; raise it to follow the routes between them.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'magma',
      help: 'Used for the length of stay and the site discs.',
      options: [
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis (color-blind optimised)'}
      ]
    }
  ],

  readouts: [
    {
      id: 'stayChart',
      label: 'How long do dwells last?',
      kind: 'chart',
      help: 'Every dwell the metrics graph found at the slow-step speed and minimum dwell, in two-day bins. The tall bars on the left are stopovers, the thin tail is home ranges. The rule is the longest stay counted as a stopover.'
    },
    {
      id: 'seasonChart',
      label: 'When do the stopovers begin?',
      kind: 'chart',
      help: 'Stopovers kept by the rules, begun in each week of the folded year, one line per species. Autumn and spring humps are the two migrations.'
    },
    {
      id: 'siteChart',
      label: 'The ten top sites',
      kind: 'chart',
      help: 'The ten highest-ranked sites by the chosen measure. Bar numbers are the ranks in the list below.'
    },
    {id: 'tracks', label: 'Tracks'},
    {
      id: 'stops',
      label: 'Dwells',
      help: 'Dwells found by GPUTrajectoryMetrics and how many of them pass the stopover rules.'
    },
    {id: 'tracksWithStops', label: 'Tracks with dwells'},
    {
      id: 'sites',
      label: 'Sites',
      help: 'Clusters found by GPUSpatialClustering and the stopovers left out as noise.'
    },
    {
      id: 'siteList',
      label: 'Ranked sites',
      layout: 'block',
      help: 'Sites by the chosen measure. Names are the nearest well-known area within 150 km, or coordinates; bird-days is the summed stay.'
    },
    {id: 'selectedSite', label: 'Selected site'}
  ],

  legends: state => [
    ...(state.colorStops === 'stay'
      ? [
          {
            kind: 'ramp' as const,
            title: 'Length of stay',
            ramp: state.ramp,
            extent: [0, 30] as const,
            unit: 'days',
            format: (value: number) => value.toFixed(0)
          }
        ]
      : []),
    ...(state.colorStops === 'cluster'
      ? [
          {
            kind: 'categories' as const,
            title: 'Stops by site',
            entries: [
              {color: CLUSTER_PALETTE[0], label: 'A site (eight colors, repeated)'},
              {color: [150, 156, 168, 255] as const, label: 'Noise: an isolated stop'}
            ]
          }
        ]
      : []),
    ...(state.colorStops === 'species'
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
    ...(state.showSites
      ? [
          {
            kind: 'ramp' as const,
            title: 'Site size relative to the top-ranked site',
            ramp: state.ramp,
            extent: [0, 1] as const,
            labels: ['small', 'the top site'] as const
          }
        ]
      : [])
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTrajectoryMetrics, GPUSpatialClustering,
  getGPUTrajectoryMetricsParameterValues, getGPUSpatialClusteringParameterValues
} from '@luma.gl/experimental/gpu-spatial-analysis';

// positions: azimuthal-equidistant meters, timestamps: float32 seconds of the folded year
// 1. Dwells: runs of slow steps lasting at least the minimum stay
metricsGraph.add(new GPUTrajectoryMetrics({spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
  positions, timestamps, trackOffsets, parameters: metricsParameters.importToGraph(metricsGraph),
  stops: {output: {ids, count, overflow, requiredCount}, startRows, endRows, centroids, durations}  // capacity 16,384
}));
metricsParameters.write(getGPUTrajectoryMetricsParameterValues({
  stopSpeedThreshold: ${state.stopSpeed}, stopMinimumDuration: ${state.minStayHours * 3600}     // m/s, s
}));

// 2. Kernel: keep stays <= ${state.maxStayDays} d that begin in the window; rejected stops get NaN centroids
// 3. DBSCAN over the kept centroids: sites
clusterGraph.add(new GPUSpatialClustering({
  positions: keptCentroids, parameters: clusterParameters.importToGraph(clusterGraph),
  gridSize: [192, 192], labels,
  clusters: {ids, count, overflow}, clusterSizes, clusterCentroids   // capacity 512
}));
clusterParameters.write(getGPUSpatialClusteringParameterValues({
  bounds: dataBounds, epsilon: ${state.epsilonKm * 1000}, minimumPoints: ${state.minimumStops}   // meters
}));
// Ranking by ${state.rankBy} is a few lines on the CPU over the read-back labels and durations.`,

  about: {
    what: '`GPUTrajectoryMetrics` scans every track for runs of slow steps and reports each dwell with its start and end rows, mean position and duration. A selection kernel keeps the dwells that are short enough to be stopovers, begin in the chosen season and belong to the chosen species. `GPUSpatialClustering` runs DBSCAN on the kept centers to group them into sites.',
    why: 'Conservation money goes to places, not to GPS fixes. Ranking the places where birds actually stop, by bird-days of dwell or by the number of birds that use them, says which wetlands and plains the population depends on, and when.',
    howToRead:
      'Each dot is one dwell, colored by how long the bird stayed. The large discs are the ranked sites (the top three have a halo): a disc is the centroid of a cluster of dwells. Tracks run under the dots. All durations are in days of the folded year; the same bird in different years counts as one bird, and its stays add up as dwell.'
  },

  create: async ctx =>
    (await import('./migration-stopovers.compute')).createMigrationStopovers(ctx),

  story: [
    {
      id: 'the-question',
      headline: 'Tracks thicken where birds stay',
      textAlternative: 'A paper atlas map draws migration routes and proportional stop discs.',
      optionsMode: 'fresh',
      title: 'Where do the birds stop on the way?',
      body: "Marsh harriers and Montagu's harriers fly from the Low Countries to the Sahel and back; spoonbills move to the Atlantic coast of France and Iberia. A GPS fix every two hours says where each bird was, so a bird that stays within a few kilometers for days shows up as a run of fixes that barely move: a **dwell**.\n\nEvery dot is a dwell, colored by the length of the stay. Where do they cluster, and are the places the birds stop the same places they live?",
      camera: {...EUROPE_AFRICA_VIEW, transitionMs: 1200},
      controls: [],
      readouts: ['tracks', 'stops']
    },
    {
      id: 'dwells',
      headline: 'A stay is a run of slow steps',
      textAlternative: 'Slow and fast segments reveal a dwell rule on one route.',
      optionsMode: 'fresh',
      title: 'Finding the dwells',
      body: '**`GPUTrajectoryMetrics`** marks each two-hour step as slow when the bird covers less than the **Slow-step speed** times the step, then reports every run of slow steps that lasts at least the **Minimum dwell**: its first and last fix, the mean position and the duration. It runs once for all 101 tracks and re-runs on every slider move, because both values are parameters.\n\nTry **Slow-step speed** at 0.5 m/s and the dwells thin out, then at 4 m/s and short flights get glued into one long stay. The readout counts the dwells; the histogram shows how long they last.',
      options: {showSites: false, showTracks: true, trackOpacity: 0.2},
      highlight: {readout: 'stops'},
      controls: ['stopSpeed', 'minStayHours'],
      readouts: ['stops', 'stayChart']
    },
    {
      id: 'stopover-or-home',
      headline: 'A stopover is not a home range',
      textAlternative: 'Short stays and long home ranges are separated by hollow rings.',
      optionsMode: 'fresh',
      title: 'A stopover is not a home range',
      body: "The histogram has two populations. Most dwells last a few days: **stopovers**. A thin tail runs to months: a breeding territory in Flanders, a wintering range in the Sahel. Drag **Longest stopover** below and the long dots disappear from the map: only dwells shorter than the limit go on to the next stage.\n\nPick one species in **Species** to see how it moves: the spoonbills' stops sit on the Atlantic seaboard of France and Iberia, the harriers' in Flanders and the Sahel. **Stopovers that begin between** limits the map to a season.",
      options: {showSites: false, maxStayDays: 14, colorStops: 'stay'},
      highlight: {readout: 'stops'},
      controls: ['maxStayDays', 'species', 'dayRange'],
      readouts: ['stops', 'stayChart']
    },
    {
      id: 'clustering',
      headline: 'A radius makes a site',
      textAlternative: 'A metric radius ring groups individual stop points into a site.',
      optionsMode: 'fresh',
      title: 'From stops to sites',
      body: '**`GPUSpatialClustering`** runs DBSCAN on the stopover centers: two stopovers are neighbors when they lie within the **Site radius**, and a stop is a core point when **Minimum stops per site** of them are near. Clusters become **sites**; stops that never reach the minimum are noise (grey). The clustering runs on the GPU in planar kilometers, and moving either slider just rewrites a parameter buffer.\n\nColor the stops by **Site** to see the clusters. A radius of 80 km gives about two dozen sites; 300 km merges most of them into a handful, and 10 km shatters them.',
      options: {showSites: true, colorStops: 'cluster', maxStayDays: 14},
      highlight: {readout: 'sites'},
      controls: ['colorStops', 'epsilonKm', 'minimumStops'],
      readouts: ['sites']
    },
    {
      id: 'ranking',
      headline: 'Rank changes with what counts',
      textAlternative: 'Proportional discs resize as the site statistic changes.',
      optionsMode: 'fresh',
      title: 'Which sites matter most?',
      body: 'The sites are ranked by **total dwell**, the bird-days spent there, and the list says how many birds, how many stops and when the stays usually begin. Click a disc on the map to read it. By dwell, the top of the list is dominated by the two ends of the journey: the **Flanders** breeding area, where the stays begin in early August after breeding, then the **Sahel of Mali and Mauritania** where harriers settle in early October, then the **Vendee coast** of France for the spoonbills.\n\nSwitch **Rank sites by** to *Distinct birds* and an en-route site climbs the list: the Atlantic plains of Morocco, used by about half a dozen birds. A real stopover is brief, so it does not win on bird-days.',
      options: {showSites: true, colorStops: 'stay', rankBy: 'dwell', maxStayDays: 14},
      highlight: {readout: 'siteList'},
      controls: ['rankBy', 'showSites'],
      readouts: ['siteList', 'siteChart', 'selectedSite']
    },
    {
      id: 'spring',
      headline: 'Spring changes the stopover ranking',
      textAlternative: 'A spring stopover atlas ranks sites using the chosen rule.',
      optionsMode: 'fresh',
      title: 'When do they stop, and what does the sample hide?',
      body: 'Set **Stopovers that begin between** to spring, 1 March to 15 June: the northbound stops are fewer than the autumn ones, and the Atlantic plains of **Morocco** appear: about half a dozen birds stopping there between late March and mid-April. The chart shows the weeks in which stopovers begin: the autumn hump (late summer to early autumn) is larger and broader than the spring one, and the spike in the first week of January is an artifact of the fold, where stays are cut at New Year.\n\nThe honest limits: tags are on 42 birds, and some individuals appear in several years, which makes a site look busier than the population is; the archive folds years onto one calendar, so "April" blends every year; a two-hour fix cannot see a rest of a few hours; and whether a stay is a stopover or a home range is a threshold (**Longest stopover**), not a fact. **Try:** set **Minimum stops per site** to 1 and see how many single-bird sites appear.',
      options: {
        showSites: true,
        rankBy: 'birds',
        dayRange: [60, 167],
        maxStayDays: 14,
        colorStops: 'species'
      },
      camera: {longitude: -2, latitude: 36, zoom: 3.8, transitionMs: 1400},
      highlight: {readout: 'siteList'},
      controls: ['dayRange', 'rankBy', 'minimumStops'],
      readouts: ['seasonChart', 'siteList']
    }
  ]
});
