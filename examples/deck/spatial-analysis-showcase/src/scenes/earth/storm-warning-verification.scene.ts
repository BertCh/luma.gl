// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {StormWarningVerificationOptions} from './storm-warning-verification.compute';
import {formatStormClock, STORM_VERIFICATION_RANGE, STORM_VIEW} from './storm-data';

const UPPER_MIDWEST_VIEW = {longitude: -93.2, latitude: 43.4, zoom: 6.6};
const SOUTHERN_PLAINS_VIEW = {longitude: -97.4, latitude: 33.4, zoom: 6.2};

/** Verdict legend entries (kept here so the scene file does not import GPU code). */
const VERDICT_ENTRIES = [
  {color: [30, 150, 235, 255] as const, label: 'Inside a matching active warning'},
  {color: [176, 176, 190, 255] as const, label: 'Inside an active warning of another type'},
  {color: [235, 60, 160, 255] as const, label: 'No active warning at the report time'}
];
const KIND_ENTRIES = [
  {color: [213, 94, 0, 255] as const, label: 'Tornado'},
  {color: [0, 114, 178, 255] as const, label: 'Wind (measured)'},
  {color: [86, 180, 233, 255] as const, label: 'Hail'},
  {color: [0, 158, 115, 255] as const, label: 'Flood'},
  {color: [204, 121, 167, 255] as const, label: 'Wind damage'},
  {color: [150, 156, 168, 255] as const, label: 'Other'}
];
const WARNING_ENTRIES = [
  {color: [220, 50, 47, 255] as const, label: 'Tornado warning'},
  {color: [240, 190, 40, 255] as const, label: 'Severe thunderstorm warning'},
  {color: [60, 170, 90, 255] as const, label: 'Flash flood warning'}
];

export default defineScene<StormWarningVerificationOptions>({
  id: 'storm-warning-verification',
  title: 'Was there already a warning when the report came in?',
  chapter: 'earth',
  order: 21,
  summary:
    'Check 860 storm reports from 21-22 May 2024 against 750 National Weather Service warning polygons: a spatial join on the GPU plus a time test, with lead time as a histogram.',
  contributors: ['GPUSpatialPredicateJoin', 'GPUTimeWindowFilter'],
  datasets: [
    {id: 'poopdeck-mrms-storm3d-reports', role: 'storm reports (SPC)'},
    {id: 'poopdeck-mrms-storm3d-warnings', role: 'warning polygons (NWS)'}
  ],
  initialView: STORM_VIEW,

  options: [
    ...playbackOptions<StormWarningVerificationOptions>({
      time: {
        min: STORM_VERIFICATION_RANGE[0],
        max: STORM_VERIFICATION_RANGE[1],
        step: 300,
        default: 41400,
        label: 'Time (UTC)',
        format: formatStormClock,
        help: 'From 17:30 UTC on 21 May, when the first warning of the data starts, to 03:00 UTC on 22 May 2024.'
      },
      playing: false,
      speed: {
        min: 1,
        max: 40,
        step: 1,
        default: 10,
        unit: ' min/s',
        label: 'Playback speed',
        help: 'Simulated minutes per real second. 10 plays the whole evening in about a minute.'
      },
      loop: true
    }),
    {
      kind: 'select',
      id: 'spatialTest',
      label: 'Spatial test',
      group: 'Verification',
      apply: 'compile',
      default: 'dwithin',
      help: 'The predicate of the GPUSpatialPredicateJoin, evaluated as predicate(report, polygon). Within counts only the interior; intersects also counts the boundary; dwithin adds a tolerance distance. A different predicate is a different kernel, so each choice compiles once and is cached.',
      options: [
        {value: 'dwithin', label: 'Within a distance of the polygon'},
        {value: 'intersects', label: 'Intersects (boundary counts)'},
        {value: 'within', label: 'Within (interior only)'}
      ]
    },
    {
      kind: 'slider',
      id: 'toleranceKm',
      label: 'Location tolerance',
      group: 'Verification',
      apply: 'param',
      min: 0,
      max: 40,
      step: 1,
      default: 0,
      unit: 'km',
      disabledWhen: state => state.spatialTest !== 'dwithin',
      help: 'A report this close to a warning polygon still counts as inside. The distance is a one-row parameter buffer of the join, so moving the slider never recompiles. Reports are often located to the nearest town, so a few kilometers is fair.'
    },
    {
      kind: 'slider',
      id: 'graceMinutes',
      label: 'Grace after expiry',
      group: 'Verification',
      apply: 'param',
      min: 0,
      max: 30,
      step: 1,
      default: 0,
      unit: 'min',
      help: 'A report this long after a warning version is replaced or expires still counts. Reports are filed after the fact, so a few minutes are often fair.'
    },
    {
      kind: 'select',
      id: 'matching',
      label: 'Warning type must fit the hazard',
      group: 'Verification',
      apply: 'param',
      default: 'hazard',
      help: 'Which warning types can verify which report kinds. Parameter buffer: the kernels read an allowed-type table.',
      options: [
        {value: 'any', label: 'Any active warning counts'},
        {
          value: 'hazard',
          label: 'Warned at least as severe',
          help: 'Tornado reports need a tornado warning; wind and hail reports accept a severe thunderstorm or tornado warning; flood reports need a flash flood warning.'
        },
        {
          value: 'exact',
          label: 'Same hazard only',
          help: 'Tornado reports need a tornado warning; wind and hail reports need a severe thunderstorm warning; flood reports need a flash flood warning.'
        }
      ]
    },
    {
      kind: 'toggle',
      id: 'showWarnings',
      label: 'Show active warnings (fill)',
      group: 'Map',
      apply: 'param',
      default: true,
      help: 'Fills the warning polygons that are valid at the playhead, selected by GPUTimeWindowFilter in interval mode.'
    },
    {
      kind: 'slider',
      id: 'warningOpacity',
      label: 'Warning fill opacity',
      group: 'Map',
      apply: 'param',
      min: 0.05,
      max: 0.8,
      step: 0.05,
      default: 0.3,
      disabledWhen: state => !state.showWarnings,
      help: 'Opacity of the warning fill.'
    },
    {
      kind: 'toggle',
      id: 'showOutlines',
      label: 'Show warning outlines',
      group: 'Map',
      apply: 'param',
      default: true,
      help: 'Draws the edges of the active warning polygons.'
    },
    {
      kind: 'toggle',
      id: 'showReports',
      label: 'Show storm reports',
      group: 'Map',
      apply: 'param',
      default: true,
      help: 'Draws each report in the window ending at the playhead.'
    },
    {
      kind: 'select',
      id: 'colorReportsBy',
      label: 'Color reports by',
      group: 'Map',
      apply: 'param',
      default: 'verdict',
      disabledWhen: state => !state.showReports,
      help: 'Verification verdict (blue: matching active warning, gray: another type, magenta: none) or the kind of report.',
      options: [
        {value: 'verdict', label: 'Verification verdict'},
        {value: 'kind', label: 'Kind of report'}
      ]
    },
    {
      kind: 'slider',
      id: 'reportMinutes',
      label: 'Reports shown for',
      group: 'Map',
      apply: 'param',
      min: 10,
      max: 600,
      step: 10,
      default: 600,
      unit: 'min',
      disabledWhen: state => !state.showReports,
      help: 'How long a report stays on the map after it arrives. 600 minutes keeps them all, so the map accumulates through the evening.'
    },
    {
      kind: 'slider',
      id: 'reportSize',
      label: 'Report size',
      group: 'Map',
      apply: 'param',
      min: 2,
      max: 9,
      step: 0.5,
      default: 4,
      unit: 'px',
      disabledWhen: state => !state.showReports,
      help: 'Radius of a report in screen pixels.'
    }
  ],

  readouts: [
    {
      id: 'leadChart',
      label: 'Lead time of verified reports',
      kind: 'chart',
      help: 'Minutes from the first issue of the matching warning to the report, in 5-minute bars up to 60 minutes. The line is the median.'
    },
    {
      id: 'kindChart',
      label: 'Verified share by kind of report',
      kind: 'chart',
      help: 'Percent of the reports of each kind (count in the label) that a matching active warning covered.'
    },
    {
      id: 'timelineChart',
      label: 'Reports through the evening',
      kind: 'chart',
      help: 'Reports per half hour (area) and how many were verified (line). The marker is the playhead.'
    },
    {id: 'clock', label: 'Playhead', help: 'Simulated UTC time.'},
    {
      id: 'verifiedShare',
      label: 'Reports verified',
      help: 'Reports that fell inside a matching warning polygon that was valid at the report time (plus grace).'
    },
    {
      id: 'anyWarningShare',
      label: 'Reports inside any active warning',
      help: 'Same test without the hazard match: was there any active warning over the spot?'
    },
    {
      id: 'unwarned',
      label: 'No active warning',
      help: 'Reports with no active warning polygon over them at the report time.'
    },
    {
      id: 'warningsVerified',
      label: 'Warnings with a report',
      help: 'Share of warnings (chains of re-issued polygon versions) that contained at least one matching report. Reports can be missing where few people are watching.'
    },
    {
      id: 'medianLead',
      label: 'Median lead time',
      help: 'Median minutes from the first issue of the matching warning to the report, over verified reports.'
    },
    {
      id: 'activeWarnings',
      label: 'Warnings valid now',
      format: 'integer',
      help: 'Polygon versions valid at the playhead, counted by GPUTimeWindowFilter.'
    },
    {
      id: 'reportsSeen',
      label: 'Reports on the map',
      format: 'integer',
      help: 'Reports inside the report window, counted by GPUTimeWindowFilter.'
    },
    {
      id: 'pairs',
      label: 'Join output',
      help: 'Matched (report, polygon) pairs of the join, its candidate count and whether a capacity overflowed.'
    },
    {id: 'inputs', label: 'Inputs'},
    {
      id: 'selected',
      label: 'Selected report',
      help: 'Click a report to read its kind, time and verdict.'
    }
  ],

  legends: state => [
    ...(state.showReports
      ? [
          state.colorReportsBy === 'verdict'
            ? {
                kind: 'categories' as const,
                title: 'Report verdict',
                entries: VERDICT_ENTRIES
              }
            : {
                kind: 'categories' as const,
                title: 'Kind of report',
                entries: KIND_ENTRIES
              }
        ]
      : []),
    ...(state.showWarnings || state.showOutlines
      ? [
          {
            kind: 'categories' as const,
            title: 'Active warnings',
            entries: WARNING_ENTRIES,
            note: 'Storm-based polygons, drawn while valid.'
          }
        ]
      : [])
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUSpatialJoinPrepared, GPUSpatialPredicateJoin} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUTimeWindowFilter, getGPUTimeWindowParameterValues} from '@luma.gl/experimental/gpu-dataframe';

// reports: points; warnings: polygon versions in an azimuthal-equidistant frame (meters)
const graph = new GPUCommandGraph(device, {id: 'verify'});
const prepared = new GPUSpatialJoinPrepared({geometry: warningPolygons});
graph.add(prepared);
graph.add(new GPUSpatialPredicateJoin({
  left: {kind: 'points', positions: reportMeters},
  right: warningPolygons,
  predicate: '${state.spatialTest}',${state.spatialTest === 'dwithin' ? `\n  distance: tolerance.importToGraph(graph),     // ${state.toleranceKm} km, a parameter buffer` : ''}
  candidateCapacity: 600000, prepared,
  pairs: {leftIds, rightIds, count, overflow}    // every (report, polygon) that touches in space
}));

// The time and hazard tests run on the pairs: one thread per report.
//   verified  = a pair whose polygon is valid at the report time (+ ${state.graceMinutes} min grace)
//               and whose warning type is allowed for the report kind ('${state.matching}')
//   lead time = report time - issue time of the earliest-issued matching warning
addKernelPass(graph, {id: 'verify-reports', invocationCount: reportCount, bindings, body: verifyWgsl});

// Which warnings to draw now: interval mode over each polygon version's valid time.
display.add(new GPUTimeWindowFilter({
  timestamps: validFrom, endTimestamps: validUntil,
  window: window.importToGraph(display),
  output: {ids, count, overflow}, outputMask
}));
window.write(getGPUTimeWindowParameterValues({start: playhead, end: playhead}));`,

  about: {
    what: '`GPUSpatialPredicateJoin` pairs every storm report with every warning polygon version that touches it in space, in one pass over a prepared index of the polygons. Two small kernels then add the time test (is the polygon valid at the report time?) and the hazard test (does the warning type fit the report kind?), and keep the earliest matching issue time for the lead time. `GPUTimeWindowFilter` picks the warnings valid at the playhead and the reports seen so far.',
    why: 'Warning skill is judged on two questions: was the hazard warned before it happened (probability of detection and lead time), and did warnings turn out to be right? Verification needs a join in space and in time, over polygons that are re-issued every few minutes. Pushing the pairing onto the GPU keeps tolerance, grace period and matching rule interactive.',
    howToRead:
      'Colored translucent polygons are warnings valid right now (red tornado, yellow severe thunderstorm, green flash flood). Dots are reports: blue is inside a matching active warning, gray inside an active warning of another type, magenta had no active warning when it arrived. The histogram is how many minutes before the report the matching warning was first issued. Reports are not ground truth: they depend on people seeing and reporting events, they are preliminary, and they are thin where few people live.'
  },

  create: async ctx =>
    (await import('./storm-warning-verification.compute')).createStormWarningVerification(ctx),

  story: [
    {
      id: 'the-question',
      title: 'Was there a warning when the storm reached them?',
      body: 'On the evening of 21 May 2024 the National Weather Service issued **750 warning polygon versions** (284 separate tornado, severe thunderstorm and flash flood warnings) while the Storm Prediction Center logged **860 reports** of tornadoes, hail, damaging wind and flooding. Warning skill asks: how many of those reports fell inside a warning that was already active?\n\nPress **Play** below, or drag **Time (UTC)**. Translucent polygons are the warnings valid right now (red tornado, yellow severe thunderstorm, green flash flood); dots are reports as they arrive. The slider follows the clock.',
      camera: {...UPPER_MIDWEST_VIEW, transitionMs: 1400},
      options: {time: 38700, play: false, colorReportsBy: 'kind', reportMinutes: 90},
      controls: ['play', 'time', 'speed'],
      readouts: ['clock', 'activeWarnings', 'reportsSeen']
    },
    {
      id: 'spatial-join',
      title: 'Step 1: which polygons touch each report?',
      body: '**`GPUSpatialPredicateJoin`** builds an index of the warning polygons once, then for every report finds the polygons that contain it, using exact point-in-polygon tests. The result is a list of (report, polygon) pairs; the **Join output** readout shows how many and whether any capacity overflowed.\n\nThis step is purely spatial: a polygon from 19:00 UTC pairs with a report at 01:00 UTC if they overlap. Choose a **Spatial test** of *Within a distance of the polygon* and slide **Location tolerance** up: the join widens the polygons by a distance, which is a parameter buffer write and never recompiles.',
      camera: {...STORM_VIEW, transitionMs: 1400},
      options: {
        time: 54000,
        play: false,
        reportMinutes: 600,
        colorReportsBy: 'kind',
        spatialTest: 'dwithin',
        toleranceKm: 0
      },
      controls: ['spatialTest', 'toleranceKm'],
      readouts: ['pairs', 'inputs']
    },
    {
      id: 'time-test',
      title: 'Step 2: was the polygon valid at that moment?',
      body: 'A warning polygon is valid only from the moment it is issued until it is replaced or expires, so a kernel keeps only the pairs where the **report time falls inside the valid interval**. Blue reports are inside a matching active warning, gray ones inside a warning of another type, and magenta ones had no active warning at all.\n\nReports are often filed a few minutes late, so slide **Grace after expiry** to 10 minutes and watch magenta turn blue. The **Reports verified** readout is the number the rest of the story is about.',
      camera: {...STORM_VIEW, transitionMs: 1400},
      options: {
        time: 54000,
        play: false,
        colorReportsBy: 'verdict',
        toleranceKm: 0,
        graceMinutes: 0
      },
      controls: ['graceMinutes', 'colorReportsBy'],
      readouts: ['verifiedShare', 'anyWarningShare', 'unwarned']
    },
    {
      id: 'hazard-match',
      title: 'Step 3: does the warning fit the hazard?',
      body: 'A tornado warning does not verify a flood report. Choose how strict to be with **Warning type must fit the hazard**: *Any active warning counts*, *Warned at least as severe* (a tornado warning also warns of wind and hail) or *Same hazard only*. It is a small table of allowed (report kind, warning type) pairs in a parameter buffer.\n\nThe bars below the map show the verified share for each kind of report. Hail and wind damage are far more often covered than flooding, partly because there are only a handful of flash flood warnings in this evening’s data.',
      camera: {...SOUTHERN_PLAINS_VIEW, transitionMs: 1400},
      options: {time: 54000, play: false, colorReportsBy: 'verdict', matching: 'hazard'},
      controls: ['matching', 'colorReportsBy'],
      readouts: ['verifiedShare', 'kindChart']
    },
    {
      id: 'lead-time',
      title: 'Lead time: how early was the warning?',
      body: 'For every verified report the **lead time** is the minutes between the **first issue** of the matching warning and the report. NWS re-issues a storm-based polygon every few minutes as the storm moves; chaining those versions into one warning gives its true issue time.\n\nThe histogram groups lead times in 5-minute bars, and the line is the median. A report at zero minutes arrived as the warning went out; the long tail is warnings issued half an hour or more ahead of the hazard. Change the **Location tolerance** or **Grace after expiry** and the histogram redraws from the GPU result.',
      camera: {...STORM_VIEW, transitionMs: 1400},
      options: {
        time: 54000,
        play: false,
        matching: 'hazard',
        showWarnings: false,
        showOutlines: false
      },
      controls: ['toleranceKm', 'graceMinutes', 'matching'],
      readouts: ['medianLead', 'leadChart', 'warningsVerified']
    },
    {
      id: 'limits',
      title: 'What the numbers can and cannot say',
      body: 'Storm reports are **not ground truth**: they exist where people were looking, are logged to the nearest place and minute, and are preliminary. A report with no warning may sit just outside a polygon, or arrive after a warning expired; a warning with no report may have been right in an empty field. Warning polygons here are chained by event number, phenomenon and position, which is a heuristic, and the verified share is not the official verification statistic.\n\n**Try it:** set **Location tolerance** to 10 km and **Grace after expiry** to 10 minutes and see how much of the gap closes; then switch to *Same hazard only* and see how much it opens; replay the evening at 20 min/s and watch warnings lead the reports across Iowa and Minnesota.',
      camera: {...UPPER_MIDWEST_VIEW, transitionMs: 1400},
      options: {
        time: 45000,
        play: false,
        showWarnings: true,
        showOutlines: true,
        spatialTest: 'dwithin',
        toleranceKm: 5,
        graceMinutes: 5,
        matching: 'hazard',
        colorReportsBy: 'verdict'
      },
      controls: ['toleranceKm', 'graceMinutes', 'matching', 'speed'],
      readouts: ['verifiedShare', 'warningsVerified']
    }
  ]
});
