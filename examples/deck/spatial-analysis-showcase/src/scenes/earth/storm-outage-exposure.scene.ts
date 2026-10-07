// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {StormOutageExposureOptions} from './storm-outage-exposure.compute';
import {formatStormClock, METRIC_LABELS, STORM_VERIFICATION_RANGE, STORM_VIEW} from './storm-data';

const PLAINS_VIEW = {longitude: -94.5, latitude: 40.5, zoom: 5.6};
const EAST_VIEW = {longitude: -80.5, latitude: 41.3, zoom: 5.4};

type LegendRange = {low: number; high: number; metric: StormOutageExposureOptions['metric']};

export default defineScene<StormOutageExposureOptions>({
  id: 'storm-outage-exposure',
  title: 'Did the counties the storms crossed lose power?',
  chapter: 'earth',
  order: 22,
  summary:
    'Compare county power outages on 21-22 May 2024 with storm exposure: kilometers of storm-cell track inside each county (GPULineLengthPerPolygon) and lightning flashes per county (GPUZonalStatistics), with outage tables reduced by GPUGroupStatistics.',
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
        {value: 'meanDbz', label: 'Mean peak reflectivity of the cells that crossed'}
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
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Map',
      apply: 'param',
      default: 'magma',
      help: 'Color ramp of the county fill.',
      options: [
        {value: 'magma', label: 'Magma'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis'}
      ]
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
    const range = data.range as LegendRange | undefined;
    const labels = METRIC_LABELS[state.metric];
    return [
      {
        kind: 'ramp' as const,
        title: labels.title,
        ramp: state.ramp,
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

  create: async ctx =>
    (await import('./storm-outage-exposure.compute')).createStormOutageExposure(ctx),

  story: [
    {
      id: 'the-question',
      title: 'Which counties lost power on the evening of 21 May?',
      body: 'The DOE EAGLE-I archive records how many customers each utility reports without power, county by county, every 15 minutes. Here that is **40,291 county snapshots** from 2,111 counties between 17:30 UTC on 21 May and 03:00 UTC on 22 May 2024.\n\nPress **Play** below or drag **Time (UTC)**: counties are colored by the customers out per 1,000 residents, and counties with no outage stay clear. The chart under the map is the regional total; its marker is the playhead.',
      camera: {...PLAINS_VIEW, transitionMs: 1400},
      options: {metric: 'outageNow', time: 43200, play: false, showTracks: false},
      controls: ['play', 'time', 'metric'],
      readouts: ['clock', 'customersNow', 'totalChart']
    },
    {
      id: 'group-statistics',
      title: 'Reducing the outage table with GPUGroupStatistics',
      body: 'The outage table is long and thin (county, time, customers). **`GPUGroupStatistics`** groups its rows by county in a dense table and computes, in one pass, the **peak** of each county (run once) and the **sum of the latest snapshot** (run again whenever the playhead moves). The latest snapshot is chosen by a **`GPUTimeWindowFilter`** mask over the snapshot times.\n\nSwitch **Color counties by** to *Peak customers out, per 1,000 residents*: every county is colored by the worst moment of the whole evening, not the current one. Then try *Customers out now (count)*: the biggest counties dominate because they have the most customers, which is why the map defaults to a rate.',
      camera: {...STORM_VIEW, transitionMs: 1400},
      options: {metric: 'outagePeak', time: 54000, play: false, showTracks: false},
      controls: ['metric', 'ramp'],
      readouts: ['peakTotal', 'customersNow']
    },
    {
      id: 'track-length',
      title: 'Exposure 1: kilometers of storm track in each county',
      body: '**`GPULineLengthPerPolygon`** cuts every cell track where it crosses a county border and adds up the length inside each county, in meters, on an azimuthal frame whose distances are accurate. The tracks are clipped at the playhead, so press **Play** and watch exposure accumulate.\n\nSwitch **Storm exposure counted** to *Whole event* to see the total footprint, then back to *Up to the playhead*. Darker counties had more kilometers of intense cells (51 dBZ and above) cross them; try **Color counties by** *Mean peak reflectivity* to see which counties had the strongest cores.',
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
      id: 'lightning',
      title: 'Exposure 2: flashes per county',
      body: '**`GPUZonalStatistics`** has two modes and this scene uses both. In polygon mode it joined every flash to its county **once**, keeping the county of every point. In feature-rows mode it sums, county by county, the flashes whose time falls in the window (their fade weight is 1 inside and 0 outside), without repeating the join.\n\nThe map shows an estimate of flashes per 1,000 square kilometers (the sample is scaled up by its sampling fraction). The **Flash sum order** option is compile-time: *Sorted* gives the same answer every run; *Atomic* is faster but can differ in the last digits.',
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
      title: 'Were exposed counties more likely to lose power?',
      body: 'Group the storm-region counties by exposure (none, under the **Exposure threshold**, one to three times it, three times or more) and ask what share had a peak outage above the **Outage threshold**. The bars show it; **Ratio** is how many times more likely an exposed county was to be hit than the others.\n\nSwitch **Exposure measure for the comparison** between cell track and lightning, and move the thresholds: the ratio changes, but the pattern is the test of whether storm footprint and outages go together at all. This is an association: the path of a storm cell does not say where wind brought lines down.',
      camera: {...STORM_VIEW, transitionMs: 1400},
      options: {
        metric: 'outagePeak',
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
      title: 'What this can and cannot say',
      body: 'Outages depend on wind, falling trees, how old the lines are and how each utility reports, none of which is in this data. Radar cells at 51 dBZ and above and lightning are proxies for a storm’s intensity, not measures of wind at the ground, and the tracker’s cells are about 10 minutes apart. Counties are large and unequal, customers are meters not people, and counties missing from the outage table are treated as having none.\n\n**Try it:** set **Storm exposure counted** to *Up to the playhead* and play the evening at 20 min/s to watch the outage map respond after the storms pass; switch the comparison to lightning and see whether the ratio holds; raise **Outage threshold** to 50 per 1,000 and see how few counties are hit hard.',
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
