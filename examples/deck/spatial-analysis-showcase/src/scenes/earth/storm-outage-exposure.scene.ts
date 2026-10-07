// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {US, labelsFor} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {StormOutageExposureOptions} from './storm-outage-exposure.compute';
import {formatStormClock, METRIC_LABELS, STORM_VERIFICATION_RANGE, STORM_VIEW} from './storm-data';

const PLAINS_VIEW = {longitude: -94.5, latitude: 40.5, zoom: 5.6};
const EAST_VIEW = {longitude: -80.5, latitude: 41.3, zoom: 5.4};
const OUTAGE_LABELS = labelsFor(US, ['msp', 'ord', 'dfw']);
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  chips: ['population-normalised rate', 'county aggregation'] as const
});

type LegendRange = {low: number; high: number; metric: StormOutageExposureOptions['metric']};

export default defineScene<StormOutageExposureOptions>({
  id: 'storm-outage-exposure',
  title: 'Did the counties the storms crossed lose power?',
  chapter: 'earth',
  order: 7,
  summary:
    'Compare population-normalised outage burden with county-scale storm exposure, without treating association as a causal estimate.',
  contributors: [
    'GPULineLengthPerPolygon',
    'GPUZonalStatistics',
    'GPUGroupStatistics',
    'GPUTimeWindowFilter'
  ],
  datasets: [
    {id: 'us-counties', role: 'county polygons and population'},
    {id: 'poopdeck-mrms-storm3d-outages', role: 'county outage snapshots (EAGLE-I)'},
    {id: 'poopdeck-mrms-precip-tracks', role: 'storm-cell tracks (MRMS)'},
    {id: 'poopdeck-goes-glm-lightning', role: 'lightning flashes (GOES-16 GLM)'}
  ],
  initialView: STORM_VIEW,

  options: [
    ...playbackOptions<StormOutageExposureOptions>({
      time: {
        min: STORM_VERIFICATION_RANGE[0],
        max: STORM_VERIFICATION_RANGE[1],
        step: 300,
        default: 43200,
        label: 'Time (UTC)',
        format: formatStormClock,
        help: 'Outage snapshots exist from 17:30 UTC on 21 May to 03:00 UTC on 22 May 2024, one every 15 minutes.'
      },
      playing: false,
      speed: {
        min: 1,
        max: 40,
        step: 1,
        default: 10,
        unit: ' min/s',
        label: 'Playback speed',
        help: 'Simulated minutes per real second. Each five simulated minutes the exposure graph runs again.'
      },
      loop: true
    }),
    {
      kind: 'select',
      id: 'metric',
      label: 'Color counties by',
      group: 'Map',
      apply: 'param',
      default: 'outageNow',
      help: 'What the county fill shows. A kernel composes the chosen value for all 3,109 counties from the contributor outputs, so switching is a one-number parameter write.',
      options: [
        {value: 'outageNow', label: 'Customers out now, per 1,000 residents'},
        {value: 'outagePeak', label: 'Peak customers out, per 1,000 residents'},
        {value: 'outageCount', label: 'Customers out now (count)'},
        {value: 'trackKm', label: 'Storm-cell track inside the county (km)'},
        {value: 'flashDensity', label: 'Lightning flashes per 1,000 km2'},
        {value: 'meanDbz', label: 'Mean peak reflectivity of the cells that crossed'},
        {value: 'bivariate', label: 'Exposure × outage burden (3 × 3 classes)'}
      ]
    },
    {
      kind: 'select',
      id: 'exposureWindow',
      label: 'Storm exposure counted',
      group: 'Exposure',
      apply: 'param',
      default: 'so-far',
      help: 'Cell tracks and flashes up to the playhead (exposure builds as the evening goes), or the whole event.',
      options: [
        {value: 'so-far', label: 'Up to the playhead'},
        {value: 'event', label: 'Whole event'}
      ]
    },
    {
      kind: 'select',
      id: 'sumOrder',
      label: 'Flash sum order',
      group: 'Exposure',
      apply: 'compile',
      default: 'sorted',
      help: 'How the zonal sums are accumulated (compile-time). Sorted is reproducible: points are sorted by county and reduced in a fixed order. Atomic skips the sort but can differ in the last bits between runs.',
      options: [
        {value: 'sorted', label: 'Sorted (reproducible)'},
        {value: 'atomic', label: 'Atomic float adds'}
      ]
    },
    {
      kind: 'select',
      id: 'exposureMeasure',
      label: 'Exposure measure for the comparison',
      group: 'Comparison',
      apply: 'param',
      default: 'trackKm',
      help: 'Which exposure the comparison chart groups counties by.',
      options: [
        {value: 'trackKm', label: 'Kilometers of cell track'},
        {value: 'flashDensity', label: 'Flashes per 1,000 km2'}
      ]
    },
    {
      kind: 'slider',
      id: 'exposureThreshold',
      label: 'Exposure threshold',
      group: 'Comparison',
      apply: 'param',
      min: 2,
      max: 60,
      step: 1,
      default: 15,
      help: 'Counties are grouped as none, under the threshold, one to three times it, and over three times it. The unit follows the measure (kilometers, or flashes per 1,000 km2).'
    },
    {
      kind: 'slider',
      id: 'outageThreshold',
      label: 'Outage threshold',
      group: 'Comparison',
      apply: 'param',
      min: 1,
      max: 100,
      step: 1,
      default: 10,
      unit: 'per 1,000',
      help: 'A county counts as hit when its peak number of customers without power reached this many per 1,000 residents.'
    },
    {
      kind: 'slider',
      id: 'fillOpacity',
      label: 'Fill opacity',
      group: 'Map',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.8,
      help: 'Opacity of the county fill over the basemap.'
    },
    {
      kind: 'toggle',
      id: 'showCounties',
      label: 'Show county outlines',
      group: 'Map',
      apply: 'param',
      default: true,
      help: 'Thin outlines of every county.'
    },
    {
      kind: 'toggle',
      id: 'showTracks',
      label: 'Show storm-cell tracks',
      group: 'Map',
      apply: 'param',
      default: true,
      help: 'Draws the 289 cell tracks over the counties.'
    }
  ],

  readouts: [
    {
      id: 'totalChart',
      label: 'Customers without power through the evening',
      kind: 'chart',
      help: 'Total customers without power in the storm-region counties at each 15-minute snapshot. The marker is the playhead.'
    },
    {
      id: 'exposureChart',
      label: 'Counties hit, by storm exposure',
      kind: 'chart',
      help: 'Share of storm-region counties whose peak outage reached the outage threshold, grouped by exposure so far (county counts in the labels).'
    },
    {id: 'clock', label: 'Playhead', help: 'Simulated UTC time.'},
    {
      id: 'customersNow',
      label: 'Out now',
      help: 'Customers without power at the latest snapshot, summed over the storm-region counties.'
    },
    {
      id: 'exposedShare',
      label: 'Hit, among exposed counties',
      help: 'Share of counties with at least the exposure threshold whose peak outage reached the outage threshold.'
    },
    {
      id: 'unexposedShare',
      label: 'Hit, among the others',
      help: 'Same share for counties with no exposure or less than the threshold.'
    },
    {
      id: 'ratio',
      label: 'Ratio',
      help: 'How many times more likely an exposed county was to be hit. An association, not a cause: wind, trees and the grid matter.'
    },
    {
      id: 'peakTotal',
      label: 'Highest total out',
      help: 'Peak of the regional total over the event.'
    },
    {
      id: 'flashesCounted',
      label: 'Flashes counted',
      help: 'Estimated flashes inside the exposure window, scaled up from the sample.'
    },
    {
      id: 'overflow',
      label: 'Capacities',
      help: 'Whether the zonal join or the outage reduction overflowed a capacity.'
    },
    {id: 'inputs', label: 'Inputs'},
    {
      id: 'hovered',
      label: 'Selected county',
      help: 'Hover a county for its numbers; click one to keep it in this row.'
    }
  ],

  legends: (state, data) => {
    if (state.metric === 'bivariate') {
      const bivariateColors = [
        [238, 232, 229, 255],
        [204, 207, 220, 255],
        [151, 177, 205, 255],
        [230, 195, 205, 255],
        [181, 166, 197, 255],
        [116, 139, 184, 255],
        [206, 139, 169, 255],
        [143, 113, 161, 255],
        [75, 87, 145, 255]
      ] as const;
      return [
        {
          kind: 'matrix' as const,
          title: 'County comparison: exposure × outage burden',
          rows: ['outage burden low', 'outage burden middle', 'outage burden high'],
          columns: ['exposure low', 'exposure middle', 'exposure high'],
          colors: bivariateColors,
          rowTitle: 'customers out per 1,000 residents',
          columnTitle: 'storm exposure',
          note: 'Nine combined classes: exposure low to high on x; outage burden low to high on y.'
        }
      ];
    }
    const range = data.range as LegendRange | undefined;
    const labels = METRIC_LABELS[state.metric];
    return [
      {
        kind: 'ramp' as const,
        title: labels.title,
        ramp: state.metric === 'trackKm' || state.metric === 'flashDensity' ? 'mako' : 'magma',
        extent: range ? ([range.low, range.high] as const) : ([0, 1] as const),
        sqrtScale: true,
        unit: labels.unit,
        format: (value: number) =>
          value >= 100 ? value.toFixed(0) : value >= 10 ? value.toFixed(1) : value.toFixed(2)
      }
    ];
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPULineLengthPerPolygon, GPUZonalStatistics} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUGroupStatistics, GPUTimeWindowFilter, getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// Once: join every flash to its county and keep the point-to-county rows.
staticGraph.add(new GPUZonalStatistics({
  features: {kind: 'polygons', polygonPositions, featureOffsets, polygonOffsets, ringOffsets, candidateCapacity},
  points: flashMeters,
  output: {counts: flashTotals, pointFeatureRows}
}));
// Once: the peak customers out per county, dense keys = county rows.
staticGraph.add(new GPUGroupStatistics({
  keys: snapshotCounty, keyCount: countyCount,
  columns: [{values: customersOut, statistics: ['maximum'], output: {maximums: peakOutage}}],
  output: {keys, counts, count, overflow}
}));

// When the playhead moves five minutes:
graph.add(new GPUTimeWindowFilter({timestamps: flashTimes, window: flashWindow.importToGraph(graph),
  output: {ids, count, overflow}, fadeWeights: flashWeights}));      // 1 inside the window, 0 outside
graph.add(new GPUZonalStatistics({
  features: {kind: 'feature-rows', pointFeatureRows, featureCount: countyCount},
  values: ones, weights: flashWeights, sumOrder: '${state.sumOrder}',  // compile-time
  output: {sums: flashSums}                                           // flashes in the window per county
}));
graph.add(new GPULineLengthPerPolygon({
  positions: tracksClippedAtPlayhead, pathOffsets, pathWeights: peakDbz,
  polygons: {kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets},
  output: {lengths: trackLengths, weightedLengths, overflow}          // meters of track inside each county
}));
graph.add(new GPUGroupStatistics({
  keys: snapshotCounty, mask: snapshotInWindow, keyCount: countyCount,
  columns: [{values: customersOut, statistics: ['sum'], output: {sumValues: outageNow}}],
  output: {keys, counts, count, overflow}
}));
flashWindow.write(getGPUTimeWindowParameterValues({start: 0, end: ${state.exposureWindow === 'so-far' ? 'playhead' : 'eventEnd'}}));`,

  about: {
    what: '`GPULineLengthPerPolygon` cuts every storm-cell track where it crosses a county boundary and sums the length inside each county. `GPUZonalStatistics` joins the lightning flashes to counties once, then sums the flashes of a time window per county from the stored rows. `GPUGroupStatistics` reduces the 40,000-row outage table to a peak and to the latest snapshot per county, and `GPUTimeWindowFilter` picks the flashes and the outage snapshots that belong to the playhead.',
    why: 'Utilities, emergency managers and researchers want to know how well a storm footprint predicts where power fails, and which counties failed with little storm overhead. County polygons are the unit outages are reported in, so every exposure has to be turned into a county number. The point of the GPU is that the join is done once and the window sums are cheap, so exposure can be replayed.',
    howToRead:
      'Color is the county value for the metric you pick; transparent counties have none. Outage values are customers (meter accounts) per 1,000 residents, which is not a share of households. The bars compare how often counties with more storm overhead had an outage above the threshold; they show association only. The study region is the counties whose centroid is inside the storm box, and counties that never appear in the outage table are treated as having none.'
  },

  pipeline: [
    {id: 'snapshots', label: 'Snapshots', detail: 'Window outage rows at the playhead'},
    {
      id: 'exposure',
      label: 'County exposure',
      detail: 'Cut tracks and sum sampled flashes by polygon'
    },
    {
      id: 'rate',
      label: 'Normalise',
      detail: 'Divide customers out by residents, not customers served'
    },
    {
      id: 'compare',
      label: 'Compare',
      detail: 'Show group size and rate distribution, not a causal effect'
    }
  ],
  basemap: ground('paperCity'),
  furniture: {
    title: cartouche(
      'Did exposed counties report more outages?',
      'Outage burden · per 1,000 residents · county comparison · 21 May 2024'
    ),
    scaleBar: {units: 'metric'},
    northArrow: 'always',
    credit: joinCredits(CREDITS.usCensus, 'DOE EAGLE-I; NOAA MRMS and GOES-16 GLM (public domain)'),
    caveat:
      'Customers out is not a customers-served denominator; county association does not establish a storm cause.'
  },
  annotations: OUTAGE_LABELS,

  create: async ctx =>
    (await import('./storm-outage-exposure.compute')).createStormOutageExposure(ctx),

  story: [
    {
      id: 'snapshots',
      title: 'Outage counts follow population too',
      headline: 'Counts are an orientation, not a rate',
      textAlternative:
        'A paper county map shows raw customers-out counts beside the outage timeline.',
      optionsMode: 'fresh',
      body: 'The outage table records nonzero county snapshots at a fixed cadence. Start with raw customers out to see the reporting table and its clock; the timeline supplies the total at each snapshot.\n\nRaw counts are an orientation view only. A large county can have more customers out without a larger burden on residents.',
      camera: {...PLAINS_VIEW, transitionMs: 1400},
      options: {metric: 'outageNow', time: 43200, play: false, showTracks: false},
      controls: ['play', 'time', 'metric'],
      readouts: ['clock', 'customersNow', 'totalChart']
    },
    {
      id: 'rates',
      title: 'A denominator changes the map',
      headline: 'Map customers out per 1,000 residents',
      textAlternative: 'A rate map uses warm classes to show customers out per 1,000 residents.',
      optionsMode: 'fresh',
      body: '**`GPUGroupStatistics`** reduces the long outage table by county. Its live result can show a current snapshot or each county’s event peak.\n\nThe authored map divides customers out by residents and labels the denominator. The data contain no served-account or household denominator.',
      camera: {...STORM_VIEW, transitionMs: 1400},
      options: {metric: 'outagePeak', time: 54000, play: false, showTracks: false},
      controls: ['metric', 'time'],
      readouts: ['peakTotal', 'customersNow', 'hovered']
    },
    {
      id: 'track-exposure',
      title: 'Cut every track at county edges',
      headline: 'Exposure begins as line length in polygons',
      textAlternative:
        'Violet county classes and cased radar-track segments show track length per county.',
      optionsMode: 'fresh',
      body: '**`GPULineLengthPerPolygon`** splits every radar-derived path where it meets a county boundary, then sums the in-county length in metres. The violet classes are a footprint proxy, not measured damage.\n\nSwitch between the clock-to-date and whole-event windows to see exposure accumulate. The computation uses an azimuthal metric frame; map display remains Web Mercator.',
      camera: {...PLAINS_VIEW, transitionMs: 1400},
      options: {
        metric: 'trackKm',
        exposureWindow: 'event',
        time: 54000,
        play: false,
        showTracks: true
      },
      controls: ['metric', 'exposureWindow'],
      readouts: ['clock', 'inputs']
    },
    {
      id: 'flash-exposure',
      title: 'A second exposure changes the groups',
      headline: 'Sampled flashes offer another exposure proxy',
      textAlternative:
        'Violet county classes show estimated flashes per area without radar tracks.',
      optionsMode: 'fresh',
      body: '**`GPUZonalStatistics`** joins sampled flashes to counties once, then re-sums only the active time window. The map reports estimated flashes per area and keeps the sampled count visible.\n\nChanging the exposure definition changes the county groups. Sorted accumulation is reproducible; atomic accumulation is an expert numerical trade-off.',
      camera: {...EAST_VIEW, transitionMs: 1400},
      options: {
        metric: 'flashDensity',
        exposureWindow: 'event',
        time: 54000,
        play: false,
        showTracks: false
      },
      controls: ['metric', 'exposureWindow', 'sumOrder'],
      readouts: ['flashesCounted', 'overflow']
    },
    {
      id: 'comparison',
      title: 'Compare rates, not just totals',
      headline: 'Keep N beside every exposure group',
      textAlternative:
        'A county comparison shows exposure groups, outage-rate threshold shares, and group sizes.',
      optionsMode: 'fresh',
      body: 'The comparison groups counties by the selected proxy, then reports the share crossing the rate threshold with each group’s **N** in the chart label. Move either threshold: the grouping rule is part of the result.\n\nThis is association at county scale, not an estimate that tracked cells or lightning caused the outages. The bivariate reading is exposure on one axis and population-normalised outage burden on the other.',
      camera: {...STORM_VIEW, transitionMs: 1400},
      options: {
        metric: 'bivariate',
        exposureWindow: 'event',
        time: 54000,
        play: false,
        showTracks: true,
        exposureMeasure: 'trackKm',
        exposureThreshold: 15,
        outageThreshold: 10
      },
      controls: ['exposureMeasure', 'exposureThreshold', 'outageThreshold'],
      readouts: ['exposedShare', 'unexposedShare', 'ratio', 'exposureChart']
    },
    {
      id: 'limits',
      title: 'Exposure is not proof of cause',
      headline: 'County associations have ecological limits',
      textAlternative:
        'A rate map and comparison chart retain the group counts while explaining the limits of inference.',
      optionsMode: 'fresh',
      body: 'Radar cells and lightning are storm-intensity proxies, not measurements of damaging wind at the ground. Infrastructure, vegetation, reporting practice and timing are absent; counties are unequal aggregation units; and nonzero outage rows require careful zero/no-data interpretation.\n\nTry the time-to-date window and change exposure definition. The next story asks a different three-stage question: whether a report was inside a compatible, valid warning.',
      camera: {...PLAINS_VIEW, transitionMs: 1400},
      options: {
        metric: 'outageNow',
        exposureWindow: 'so-far',
        time: 46800,
        play: false,
        showTracks: true,
        exposureMeasure: 'flashDensity',
        exposureThreshold: 15,
        outageThreshold: 10
      },
      controls: ['exposureWindow', 'exposureMeasure', 'outageThreshold', 'speed'],
      readouts: ['ratio', 'customersNow']
    }
  ]
});
