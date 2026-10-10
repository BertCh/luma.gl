// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {US, labelsFor} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {playbackOptions} from '../../engine/playback';
import {defineScene} from '../scene';
import type {StormWarningVerificationOptions} from './storm-warning-verification.compute';
import {formatStormClock, STORM_VERIFICATION_RANGE, STORM_VIEW} from './storm-data';

const UPPER_MIDWEST_VIEW = {longitude: -93.2, latitude: 43.4, zoom: 6.6};
const SOUTHERN_PLAINS_VIEW = {longitude: -97.4, latitude: 33.4, zoom: 6.2};
const WARNING_LABELS = labelsFor(US, ['msp', 'ord', 'dfw']);
const cartouche = (title: string, subtitle: string) => ({
  title,
  subtitle,
  chips: ['space → time → hazard join'] as const
});

/** Verdict legend entries (kept here so the scene file does not import GPU code). */
const VERDICT_ENTRIES = [
  {color: [0, 128, 128, 255] as const, label: 'Verified: teal disc, matching active warning'},
  {color: [215, 140, 35, 255] as const, label: 'Hazard mismatch: amber diamond'},
  {color: [190, 45, 135, 255] as const, label: 'No warning: magenta ring'}
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
  {color: [230, 135, 30, 255] as const, label: 'Severe thunderstorm warning'},
  {color: [38, 150, 145, 255] as const, label: 'Flash flood warning'}
];

export default defineScene<StormWarningVerificationOptions>({
  id: 'storm-warning-verification',
  title: 'Was there already a warning when the report came in?',
  chapter: 'earth',
  order: 8,
  summary:
    'Verify reports by a three-stage spatial, temporal and hazard join; retain misses and the visible denominator at every stage.',
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
    {
      id: 'funnelChart',
      label: 'Verification funnel',
      kind: 'chart',
      help: 'Monotone unique-report counts: all reports, reports touching at least one polygon, time-valid reports, and hazard-compatible reports.'
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

  pipeline: [
    {id: 'space', label: 'Space', detail: 'Pair reports with candidate warning polygons'},
    {id: 'time', label: 'Time', detail: 'Keep polygons valid at the report time'},
    {id: 'hazard', label: 'Hazard', detail: 'Apply the selected report-to-warning matching table'},
    {id: 'lead', label: 'Lead', detail: 'Retain the earliest valid matching issue time'}
  ],
  basemap: ground('paperCity'),
  furniture: {
    title: cartouche(
      'Was each report inside a matching warning?',
      'Verification · reports and warning polygons · 21 May 2024'
    ),
    scaleBar: {units: 'metric'},
    credit: joinCredits(
      'NWS storm-based warnings; SPC storm reports (public domain)',
      CREDITS.naturalEarth
    ),
    caveat:
      'Reports are not a census of hazards; this dataset cannot measure quiet-place false alarms.'
  },
  annotations: WARNING_LABELS,

  create: async ctx =>
    (await import('./storm-warning-verification.compute')).createStormWarningVerification(ctx),

  story: [
    {
      id: 'event',
      title: 'Warnings and reports overlap in space and time',
      headline: 'Start with the evening, not a verdict',
      textAlternative:
        'Warning polygons and reports appear through time over a quiet Plains paper map.',
      optionsMode: 'fresh',
      body: 'Press **Play** or scrub the UTC clock. Warning polygons are shown only while valid, and reports appear in a short trailing window. The live input readout keeps the report and polygon-version counts visible.\n\nThis opening view is not verification yet: it establishes the event layers that the next three tests will join.',
      evidence:
        'At **{{clock}}**, **{{activeWarnings}}** warning versions are valid and **{{reportsSeen}}** reports are visible within the selected trailing window; the complete inputs are **{{inputs}}**.',
      caveat:
        'Visible overlap is descriptive only. A report and polygon must still pass explicit spatial, temporal and hazard-compatibility rules.',
      camera: {...UPPER_MIDWEST_VIEW, transitionMs: 1400},
      options: {time: 38700, play: false, colorReportsBy: 'kind', reportMinutes: 90},
      controls: ['play', 'time', 'speed'],
      readouts: ['clock', 'activeWarnings', 'reportsSeen', 'inputs']
    },
    {
      id: 'space',
      title: 'First ask which polygons contain the report',
      headline: 'A spatial pair is only a candidate',
      textAlternative:
        'Reports and warning outlines show the spatial-candidate stage of verification.',
      optionsMode: 'fresh',
      body: '**`GPUSpatialPredicateJoin`** indexes warning polygons, then emits report–polygon pairs that pass the selected spatial test. The join output exposes its candidate count and capacity state.\n\nTolerance is a rule choice: it changes which reports are close enough to an edge, and writes only a parameter buffer. Space alone cannot tell whether a polygon was valid then.',
      evidence:
        'The selected boundary rule yields **{{pairs}}** from **{{inputs}}** before warning validity or hazard type is considered.',
      caveat:
        'Boundary coordinates and the tolerance choice can change edge cases; a spatial candidate says nothing about whether the warning was active at report time.',
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
      id: 'time',
      title: 'Then test the warning validity window',
      headline: 'A spatial pair must also be timely',
      textAlternative:
        'Report verdict symbols retain no-warning rings while valid warning outlines remain visible.',
      optionsMode: 'fresh',
      body: 'The second kernel keeps candidates whose warning was valid at the report time; the grace control makes its policy visible. A no-warning report stays on the map rather than disappearing.\n\nThe three verdict symbols have redundant form as well as colour: verified, another warning type, and no active warning.',
      evidence:
        '**{{anyWarningShare}}** reports intersect a time-valid warning, while **{{unwarned}}** remain visible rather than leaving the denominator.',
      caveat:
        'Grace minutes are an analyst-selected policy, not part of the original warning validity interval; the hazard match is applied only in the next stage.',
      camera: {...STORM_VIEW, transitionMs: 1400},
      options: {
        time: 54000,
        play: false,
        colorReportsBy: 'verdict',
        toleranceKm: 0,
        graceMinutes: 0
      },
      controls: ['graceMinutes', 'toleranceKm'],
      readouts: ['verifiedShare', 'anyWarningShare', 'unwarned']
    },
    {
      id: 'hazard',
      title: 'Finally require the hazard to match',
      headline: 'Matching rules change the verdict',
      textAlternative:
        'Verdict symbols and a kind chart show the hazard-matching stage of verification.',
      optionsMode: 'fresh',
      body: 'The final report-side test uses an allowed report-kind / warning-kind table. Change the matching rule to see the same spatial and temporal pairs receive different verdicts.\n\nThe kind chart prints **N** beside each category: a small kind should not be read like a stable rate. Every result retains the original report denominator.',
      evidence:
        'Under the selected matching rule, **{{verifiedShare}}** pass all three stages; the kind and funnel charts retain category counts and the full report denominator.',
      caveat:
        'Hazard taxonomies are not interchangeable, and percentages for small report kinds are unstable even when the overall denominator is visible.',
      camera: {...SOUTHERN_PLAINS_VIEW, transitionMs: 1400},
      options: {time: 54000, play: false, colorReportsBy: 'verdict', matching: 'hazard'},
      controls: ['matching', 'colorReportsBy'],
      readouts: ['verifiedShare', 'kindChart', 'funnelChart']
    },
    {
      id: 'lead',
      title: 'Lead time excludes no one silently',
      headline: 'Keep misses beside lead-time classes',
      textAlternative:
        'A lead-time histogram and verdict map distinguish valid matches from retained misses.',
      optionsMode: 'fresh',
      body: 'For verified reports, lead time is minutes from the earliest matching issue to the report. The histogram reads from the live join result and marks its median.\n\nNo-warning reports remain a separate verdict rather than being averaged away. Move tolerance or grace and the entire funnel and lead distribution update from the same result.',
      evidence:
        'Among matching reports, **{{medianLead}}**; separately, **{{warningsVerified}}** warning chains contain at least one compatible report.',
      caveat:
        'Lead time is conditional on a reported event that matched. Misses stay in the chart but have no lead value, and warnings without reports are not confirmed false alarms.',
      camera: {...STORM_VIEW, transitionMs: 1400},
      options: {
        time: 54000,
        play: false,
        matching: 'hazard',
        showWarnings: true,
        showOutlines: true
      },
      controls: ['toleranceKm', 'graceMinutes', 'matching'],
      readouts: ['medianLead', 'leadChart', 'funnelChart', 'warningsVerified']
    },
    {
      id: 'limits',
      title: 'Reports are not all hazardous events',
      headline: 'The funnel has no quiet-place denominator',
      textAlternative:
        'A verdict map and verification funnel explain reporting, boundary, and false-alarm limitations.',
      optionsMode: 'fresh',
      body: 'Reports depend on observers and reporting practice; polygons have boundary uncertainty; event taxonomy and a single time window shape the join. The report table cannot enumerate quiet places, so it cannot supply a false-alarm denominator.\n\nTry a different tolerance, grace period, or matching rule. Keep the counts and rule alongside the verified share rather than treating it as an official performance statistic.',
      evidence:
        'The sensitivity setting still exposes **{{verifiedShare}}**, **{{unwarned}}** and **{{warningsVerified}}** alongside the unchanged inputs: **{{inputs}}**.',
      caveat:
        'Reports are observer-dependent positive events, not a census of all hazardous and quiet places; this dataset cannot estimate an official false-alarm ratio.',
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
      readouts: ['verifiedShare', 'unwarned', 'warningsVerified', 'inputs']
    }
  ]
});
