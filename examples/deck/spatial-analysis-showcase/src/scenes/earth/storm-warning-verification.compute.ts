// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';
import {
  GPUSpatialJoinPrepared,
  GPUSpatialPredicateJoin,
  type GPUSpatialJoinGeometry
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {ChoroplethFillLayer} from '../statistics/b5-choropleth-layer';
import {buildPolygonMesh, getPolygonLayout} from '../statistics/b5-geometry';
import type {SceneContext, SceneInstance} from '../scene';
import {
  binValues,
  formatInteger,
  formatStormClock,
  formatStormHourMinute,
  histogramChart,
  projectToStormFrame,
  quantile,
  seriesChart,
  STORM_VERIFICATION_RANGE
} from './storm-data';

/** Option state of the storm-warning-verification scene. */
export type StormWarningVerificationOptions = {
  play: boolean;
  time: number;
  speed: number;
  loop: boolean;
  spatialTest: 'dwithin' | 'intersects' | 'within';
  toleranceKm: number;
  graceMinutes: number;
  matching: 'any' | 'hazard' | 'exact';
  showWarnings: boolean;
  warningOpacity: number;
  showOutlines: boolean;
  showReports: boolean;
  colorReportsBy: 'verdict' | 'kind';
  reportMinutes: number;
  reportSize: number;
};

/** Verdict codes of a report, in palette order. */
export const VERDICT_LABELS = [
  'No active warning',
  'Active warning of another type',
  'Verified by a matching warning'
] as const;
/** Verdict colors: unwarned (magenta), another type (gray), verified (blue). */
export const VERDICT_COLORS: readonly (readonly [number, number, number, number])[] = [
  [235, 60, 160, 255],
  [176, 176, 190, 255],
  [30, 150, 235, 255]
];
/** Warning fill colors by phenomenon: tornado, severe thunderstorm, flash flood. */
export const WARNING_COLORS: readonly (readonly [number, number, number, number])[] = [
  [220, 50, 47, 255],
  [240, 190, 40, 255],
  [60, 170, 90, 255]
];
export const WARNING_LABELS = [
  'Tornado warning',
  'Severe thunderstorm warning',
  'Flash flood warning'
];
/** Colors of the report kinds, by name (Okabe-Ito based). */
export const KIND_COLORS: Record<string, readonly [number, number, number, number]> = {
  tornado: [213, 94, 0, 255],
  wind: [0, 114, 178, 255],
  hail: [86, 180, 233, 255],
  flood: [0, 158, 115, 255],
  damage: [204, 121, 167, 255],
  other: [150, 156, 168, 255]
};
export const KIND_LABELS: Record<string, string> = {
  tornado: 'Tornado',
  wind: 'Wind (measured)',
  hail: 'Hail',
  flood: 'Flood',
  damage: 'Wind damage',
  other: 'Other'
};

const PAIR_CAPACITY = 200000;
const CANDIDATE_CAPACITY = 600000;
const PARAMETER_LENGTH = 40;
const STATUS_INTERVAL_FRAMES = 10;
const TIMELINE_BIN_SECONDS = 1800;

type JoinVariant = {
  key: string;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  resources: SpatialAnalysisResources;
};

type PhenomenonMatrix = readonly (readonly string[])[];

/**
 * Warning verification for 21-22 May 2024. One `GPUSpatialPredicateJoin` pairs every storm report
 * with every warning polygon version that touches its location (purely spatial, all times). Two
 * kernels then apply the time test (the report falls inside the version's valid interval, plus a
 * grace period) and the hazard test (the warning type fits the report kind), and keep the earliest
 * matching issue time for the lead time. `GPUTimeWindowFilter` selects the warnings that are
 * active at the playhead and the reports seen so far for display. The spatial test is
 * compile-time (cached variants); tolerance, grace period and matching rule are parameter writes.
 */
export async function createStormWarningVerification(
  ctx: SceneContext<StormWarningVerificationOptions>
): Promise<SceneInstance<StormWarningVerificationOptions>> {
  const {device} = ctx;
  const reportData = ctx.datasets.get('poopdeck-mrms-storm3d-reports');
  const warningData = ctx.datasets.get('poopdeck-mrms-storm3d-warnings');
  const resources = new SpatialAnalysisResources(device, 'storm-verify');
  const lngLatDraw = {coordinateSystem: COORDINATE_SYSTEM.LNGLAT} as const;

  // ---- Reports ------------------------------------------------------------------------------
  const reportLngLat = reportData.column<Float32Array>('position');
  const reportCount = reportData.count;
  const reportTimes = Float32Array.from(reportData.column<Uint32Array>('timestamp'));
  const reportKinds = reportData.column<Uint8Array>('kind');
  const kindNames = reportData.categories('kind');
  const reportMagnitude = reportData.column<Float32Array>('magnitude');
  const reportAeqd = projectToStormFrame(reportLngLat);
  const reportInfo = new Float32Array(reportCount * 4);
  const kindCodes = new Uint32Array(reportCount);
  for (let report = 0; report < reportCount; report++) {
    reportInfo[report * 4] = reportTimes[report];
    reportInfo[report * 4 + 1] = reportKinds[report];
    kindCodes[report] = reportKinds[report];
  }

  // ---- Warnings -----------------------------------------------------------------------------
  const layout = getPolygonLayout(warningData);
  const warningCount = warningData.count;
  const warningLngLat = layout.vertices;
  const warningAeqd = projectToStormFrame(warningLngLat);
  const phenomenonNames = warningData.categories('phenomenon');
  const warningStart = Float32Array.from(warningData.column<Uint32Array>('startTime'));
  const warningEnd = Float32Array.from(warningData.column<Uint32Array>('endTime'));
  const warningIssue = Float32Array.from(warningData.column<Uint32Array>('issueTime'));
  const warningId = warningData.column<Uint32Array>('warning');
  const warningPhenomenon = warningData.column<Uint8Array>('phenomenon');
  let chainCount = 0;
  for (const value of warningId) chainCount = Math.max(chainCount, value + 1);
  const polygonInfo = new Float32Array(warningCount * 8);
  for (let warning = 0; warning < warningCount; warning++) {
    polygonInfo.set(
      [
        warningStart[warning],
        warningEnd[warning],
        warningIssue[warning],
        warningPhenomenon[warning],
        warningId[warning]
      ],
      warning * 8
    );
  }
  const featureOffsets = Uint32Array.from({length: warningCount + 1}, (_, index) => index);
  const mesh = buildPolygonMesh(layout);
  // One outline segment per ring edge and the polygon it belongs to.
  const edgeSegments: number[] = [];
  const edgePolygons: number[] = [];
  for (let polygon = 0; polygon < warningCount; polygon++) {
    for (
      let ring = layout.polygonRingOffsets[polygon];
      ring < layout.polygonRingOffsets[polygon + 1];
      ring++
    ) {
      const first = layout.ringOffsets[ring];
      const last = layout.ringOffsets[ring + 1];
      for (let vertex = first; vertex < last; vertex++) {
        const next = vertex + 1 < last ? vertex + 1 : first;
        edgeSegments.push(
          warningLngLat[vertex * 2],
          warningLngLat[vertex * 2 + 1],
          warningLngLat[next * 2],
          warningLngLat[next * 2 + 1]
        );
        edgePolygons.push(polygon);
      }
    }
  }
  const edgeCount = edgePolygons.length;

  // ---- Buffers ------------------------------------------------------------------------------
  const reportAeqdBuffer = resources.createBuffer('report-aeqd', reportAeqd);
  const reportLngLatBuffer = resources.createBuffer('report-lng-lat', reportLngLat);
  const reportInfoBuffer = resources.createBuffer('report-info', reportInfo);
  const reportTimesBuffer = resources.createBuffer('report-times', reportTimes);
  const kindCodeBuffer = resources.createBuffer('kind-codes', kindCodes);
  const warningVerticesBuffer = resources.createBuffer('warning-aeqd', warningAeqd);
  const featureOffsetsBuffer = resources.createBuffer('warning-feature-offsets', featureOffsets);
  const polygonOffsetsBuffer = resources.createBuffer(
    'warning-polygon-offsets',
    layout.polygonRingOffsets
  );
  const ringOffsetsBuffer = resources.createBuffer('warning-ring-offsets', layout.ringOffsets);
  const polygonInfoBuffer = resources.createBuffer('polygon-info', polygonInfo);
  const polygonStartBuffer = resources.createBuffer('polygon-start', warningStart);
  const polygonEndBuffer = resources.createBuffer('polygon-end', warningEnd);
  const meshPositionsBuffer = resources.createBuffer('mesh-positions', mesh.positions);
  const meshFeaturesBuffer = resources.createBuffer('mesh-features', mesh.featureRows);
  const edgeBuffer = resources.createBuffer('edges', Float32Array.from(edgeSegments));
  const edgePolygonBuffer = resources.createBuffer('edge-polygons', Uint32Array.from(edgePolygons));

  // Join outputs (shared by every spatial-test variant) and the verification results.
  const leftIds = resources.createBuffer('pair-left', PAIR_CAPACITY * 4);
  const rightIds = resources.createBuffer('pair-right', PAIR_CAPACITY * 4);
  const pairCount = resources.createBuffer('pair-count', 4);
  const pairOverflow = resources.createBuffer('pair-overflow', 4);
  const pairTotal = resources.createBuffer('pair-total', 4);
  const candidateCount = resources.createBuffer('candidate-count', 4);
  const reportResults = resources.createBuffer('report-results', reportCount * 16);
  const verdictCodes = resources.createBuffer('verdict-codes', reportCount * 4);
  const polygonHits = resources.createBuffer('polygon-hits', warningCount * 4);
  const verifyParameters = resources.createParameterBuffer('verify', 'float32', PARAMETER_LENGTH);
  const toleranceParameters = resources.createParameterBuffer('tolerance', 'float32', 1);

  // ---- Matching rule ------------------------------------------------------------------------
  function getMatrix(mode: StormWarningVerificationOptions['matching']): PhenomenonMatrix {
    return kindNames.map(kind => {
      if (mode === 'any') return phenomenonNames;
      if (kind === 'tornado') return phenomenonNames.filter(name => name === 'TO');
      if (kind === 'flood') return phenomenonNames.filter(name => name === 'FF');
      if (kind === 'wind' || kind === 'hail' || kind === 'damage') {
        return phenomenonNames.filter(name => (mode === 'hazard' ? name !== 'FF' : name === 'SV'));
      }
      return phenomenonNames;
    });
  }
  function writeVerifyParameters(): void {
    const values = new Float32Array(PARAMETER_LENGTH);
    values[0] = ctx.options.graceMinutes * 60;
    const matrix = getMatrix(ctx.options.matching);
    matrix.forEach((allowed, kind) => {
      phenomenonNames.forEach((name, phenomenon) => {
        values[4 + kind * 4 + phenomenon] = allowed.includes(name) ? 1 : 0;
      });
    });
    verifyParameters.write(values);
    toleranceParameters.write(Float32Array.of(ctx.options.toleranceKm * 1000));
  }
  writeVerifyParameters();

  // ---- Join variants (spatial test is compile-time) -----------------------------------------
  const variants = new Map<string, JoinVariant>();
  let current: JoinVariant | null = null;
  let verifyDirty = true;
  let destroyed = false;
  let results: {
    verdict: Uint8Array;
    leadMinutes: Float32Array;
    warning: Float32Array;
    polygonHits: Uint32Array;
  } | null = null;

  function buildVariant(spatialTest: StormWarningVerificationOptions['spatialTest']): JoinVariant {
    const existing = variants.get(spatialTest);
    if (existing) return existing;
    const own = new SpatialAnalysisResources(device, `storm-verify-${spatialTest}`);
    const graph = new GPUCommandGraph<void>(device, {id: `storm-verify-${spatialTest}`});
    const left: GPUSpatialJoinGeometry = {
      kind: 'points',
      positions: importGraphBuffer(graph, 'reports', reportAeqdBuffer, 'float32x2', reportCount)
    };
    const right: GPUSpatialJoinGeometry = {
      kind: 'polygons',
      positions: importGraphBuffer(
        graph,
        'warning-vertices',
        warningVerticesBuffer,
        'float32x2',
        warningAeqd.length / 2
      ),
      featureOffsets: importGraphBuffer(
        graph,
        'warning-feature-offsets',
        featureOffsetsBuffer,
        'uint32',
        featureOffsets.length
      ),
      polygonOffsets: importGraphBuffer(
        graph,
        'warning-polygon-offsets',
        polygonOffsetsBuffer,
        'uint32',
        layout.polygonRingOffsets.length
      ),
      ringOffsets: importGraphBuffer(
        graph,
        'warning-ring-offsets',
        ringOffsetsBuffer,
        'uint32',
        layout.ringOffsets.length
      )
    };
    const prepared = new GPUSpatialJoinPrepared({id: 'warning-index', geometry: right});
    graph.add(prepared);
    const leftView = importGraphBuffer(graph, 'pair-left', leftIds, 'uint32', PAIR_CAPACITY);
    const rightView = importGraphBuffer(graph, 'pair-right', rightIds, 'uint32', PAIR_CAPACITY);
    const countView = importGraphBuffer(graph, 'pair-count', pairCount, 'uint32', 1);
    graph.add(
      new GPUSpatialPredicateJoin({
        id: 'report-in-warning',
        left,
        right,
        predicate: spatialTest,
        ...(spatialTest === 'dwithin' ? {distance: toleranceParameters.importToGraph(graph)} : {}),
        candidateCapacity: CANDIDATE_CAPACITY,
        prepared,
        pairs: {
          leftIds: leftView,
          rightIds: rightView,
          count: countView,
          overflow: importGraphBuffer(graph, 'pair-overflow', pairOverflow, 'uint32', 1),
          totalCount: importGraphBuffer(graph, 'pair-total', pairTotal, 'uint32', 1)
        },
        candidateCount: importGraphBuffer(graph, 'candidates', candidateCount, 'uint32', 1)
      })
    );
    const reportInfoView = importGraphBuffer(
      graph,
      'report-info',
      reportInfoBuffer,
      'float32',
      reportCount * 4
    );
    const polygonInfoView = importGraphBuffer(
      graph,
      'polygon-info',
      polygonInfoBuffer,
      'float32',
      warningCount * 8
    );
    const parametersView = verifyParameters.importToGraph(graph);
    // Time test and hazard test per report: the earliest-issued matching warning gives the lead.
    addKernelPass(graph, {
      id: 'verify-reports',
      invocationCount: reportCount,
      bindings: [
        {name: 'leftIds', view: leftView, type: 'u32', access: 'read'},
        {name: 'rightIds', view: rightView, type: 'u32', access: 'read'},
        {name: 'pairCount', view: countView, type: 'u32', access: 'read'},
        {name: 'reportInfo', view: reportInfoView, type: 'f32', access: 'read'},
        {name: 'polygonInfo', view: polygonInfoView, type: 'f32', access: 'read'},
        {name: 'parameters', view: parametersView, type: 'f32', access: 'read'},
        {
          name: 'results',
          view: importGraphBuffer(
            graph,
            'report-results',
            reportResults,
            'float32',
            reportCount * 4
          ),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'verdicts',
          view: importGraphBuffer(graph, 'verdict-codes', verdictCodes, 'uint32', reportCount),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: `let reportTime = reportInfo[reportInfoOffset + index * 4u];
  let kind = u32(reportInfo[reportInfoOffset + index * 4u + 1u]);
  let grace = parameters[parametersOffset];
  let total = min(pairCount[pairCountOffset], ${PAIR_CAPACITY}u);
  var verdict = 0.0;
  var bestLead = -1.0e9;
  var chain = -1.0;
  for (var pair = 0u; pair < total; pair = pair + 1u) {
    let owner = leftIds[leftIdsOffset + pair];
    if (owner < index) { continue; }
    if (owner > index) { break; }
    let polygon = rightIds[rightIdsOffset + pair];
    let base = polygon * 8u;
    let validFrom = polygonInfo[polygonInfoOffset + base];
    let validUntil = polygonInfo[polygonInfoOffset + base + 1u] + grace;
    if (reportTime < validFrom || reportTime > validUntil) { continue; }
    verdict = max(verdict, 1.0);
    let phenomenon = u32(polygonInfo[polygonInfoOffset + base + 3u]);
    if (parameters[parametersOffset + 4u + kind * 4u + phenomenon] > 0.5) {
      let lead = reportTime - polygonInfo[polygonInfoOffset + base + 2u];
      if (verdict < 2.0 || lead > bestLead) {
        bestLead = lead;
        chain = polygonInfo[polygonInfoOffset + base + 4u];
      }
      verdict = 2.0;
    }
  }
  results[resultsOffset + index * 4u] = verdict;
  results[resultsOffset + index * 4u + 1u] = select(-1.0, bestLead, verdict > 1.5);
  results[resultsOffset + index * 4u + 2u] = chain;
  results[resultsOffset + index * 4u + 3u] = 0.0;
  verdicts[verdictsOffset + index] = u32(verdict);`
    });
    // Reports each warning version verified (same two tests, from the polygon's side).
    addKernelPass(graph, {
      id: 'verify-warnings',
      invocationCount: warningCount,
      bindings: [
        {name: 'leftIds', view: leftView, type: 'u32', access: 'read'},
        {name: 'rightIds', view: rightView, type: 'u32', access: 'read'},
        {name: 'pairCount', view: countView, type: 'u32', access: 'read'},
        {name: 'reportInfo', view: reportInfoView, type: 'f32', access: 'read'},
        {name: 'polygonInfo', view: polygonInfoView, type: 'f32', access: 'read'},
        {name: 'parameters', view: parametersView, type: 'f32', access: 'read'},
        {
          name: 'hits',
          view: importGraphBuffer(graph, 'polygon-hits', polygonHits, 'uint32', warningCount),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: `let base = index * 8u;
  let validFrom = polygonInfo[polygonInfoOffset + base];
  let validUntil = polygonInfo[polygonInfoOffset + base + 1u] + parameters[parametersOffset];
  let phenomenon = u32(polygonInfo[polygonInfoOffset + base + 3u]);
  let total = min(pairCount[pairCountOffset], ${PAIR_CAPACITY}u);
  var count = 0u;
  for (var pair = 0u; pair < total; pair = pair + 1u) {
    if (rightIds[rightIdsOffset + pair] != index) { continue; }
    let owner = leftIds[leftIdsOffset + pair];
    let reportTime = reportInfo[reportInfoOffset + owner * 4u];
    if (reportTime < validFrom || reportTime > validUntil) { continue; }
    let kind = u32(reportInfo[reportInfoOffset + owner * 4u + 1u]);
    if (parameters[parametersOffset + 4u + kind * 4u + phenomenon] > 0.5) { count = count + 1u; }
  }
  hits[hitsOffset + index] = count;`
    });
    const sources = [
      {buffer: pairCount, size: 4},
      {buffer: pairOverflow, size: 4},
      {buffer: pairTotal, size: 4},
      {buffer: candidateCount, size: 4},
      {buffer: reportResults, size: reportCount * 16},
      {buffer: polygonHits, size: warningCount * 4}
    ];
    const built: JoinVariant = {
      key: spatialTest,
      compiled: undefined as never,
      reader: undefined as never,
      resources: own
    };
    built.reader = new SummaryReader(own, `summary-${spatialTest}`, sources, bytes => {
      if (destroyed || current !== built) return;
      handleSummary(bytes);
    });
    built.compiled = own.track(graph.compile());
    own.track({destroy: () => prepared.destroy()});
    variants.set(spatialTest, built);
    return built;
  }

  // ---- Display graph: warnings active now and reports seen so far -----------------------------
  const warningMask = resources.createBuffer('warning-mask', warningCount * 4);
  const warningIds = resources.createBuffer('warning-ids', warningCount * 4);
  const warningActive = resources.createBuffer('warning-active', 4);
  const warningOverflow = resources.createBuffer('warning-overflow', 4);
  const activeCategory = resources.createBuffer('active-category', warningCount * 4);
  const warningWindow = resources.createParameterBuffer(
    'warning-window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  const reportIds = resources.createBuffer('report-ids', reportCount * 4);
  const reportSeen = resources.createBuffer('report-seen', 4);
  const reportOverflow = resources.createBuffer('report-overflow', 4);
  const reportWindow = resources.createParameterBuffer(
    'report-window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  const reportDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'storm-report-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const displayGraph = new GPUCommandGraph<void>(device, {id: 'storm-verify-display'});
  const maskView = importGraphBuffer(
    displayGraph,
    'warning-mask',
    warningMask,
    'uint32',
    warningCount
  );
  displayGraph.add(
    new GPUTimeWindowFilter({
      id: 'warning-window',
      timestamps: importGraphBuffer(
        displayGraph,
        'polygon-start',
        polygonStartBuffer,
        'float32',
        warningCount
      ),
      endTimestamps: importGraphBuffer(
        displayGraph,
        'polygon-end',
        polygonEndBuffer,
        'float32',
        warningCount
      ),
      window: warningWindow.importToGraph(displayGraph),
      output: {
        ids: importGraphBuffer(displayGraph, 'warning-ids', warningIds, 'uint32', warningCount),
        count: importGraphBuffer(displayGraph, 'warning-active', warningActive, 'uint32', 1),
        overflow: importGraphBuffer(displayGraph, 'warning-overflow', warningOverflow, 'uint32', 1)
      },
      outputMask: maskView
    })
  );
  addKernelPass(displayGraph, {
    id: 'active-category',
    invocationCount: warningCount,
    bindings: [
      {name: 'mask', view: maskView, type: 'u32', access: 'read'},
      {
        name: 'polygonInfo',
        view: importGraphBuffer(
          displayGraph,
          'polygon-info',
          polygonInfoBuffer,
          'float32',
          warningCount * 8
        ),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'category',
        view: importGraphBuffer(
          displayGraph,
          'active-category',
          activeCategory,
          'uint32',
          warningCount
        ),
        type: 'u32',
        access: 'read_write'
      }
    ],
    body: `let isActive = mask[maskOffset + index] != 0u;
  category[categoryOffset + index] = select(0xffffffffu, u32(polygonInfo[polygonInfoOffset + index * 8u + 3u]), isActive);`
  });
  displayGraph.add(
    new GPUTimeWindowFilter({
      id: 'report-window',
      timestamps: importGraphBuffer(
        displayGraph,
        'report-times',
        reportTimesBuffer,
        'float32',
        reportCount
      ),
      window: reportWindow.importToGraph(displayGraph),
      output: {
        ids: importGraphBuffer(displayGraph, 'report-ids', reportIds, 'uint32', reportCount),
        count: importGraphBuffer(displayGraph, 'report-seen', reportSeen, 'uint32', 1),
        overflow: importGraphBuffer(displayGraph, 'report-overflow', reportOverflow, 'uint32', 1)
      },
      drawInstanceCount: displayGraph.importGPUData(
        'report-draw-count',
        reportDraw.getInstanceCountData(0)
      )
    })
  );
  const displayCompiled = resources.track(displayGraph.compile());

  // ---- State ------------------------------------------------------------------------------------
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: STORM_VERIFICATION_RANGE, rate: 60, step: 300}
  );
  let playhead = ctx.options.time;
  let statusStale = true;
  let selectedReport = -1;
  let chartMarker = -1;

  ctx.setReadout(
    'inputs',
    `${formatInteger(reportCount)} reports, ${formatInteger(warningCount)} polygon versions of ${formatInteger(chainCount)} warnings`
  );

  function selectVariant(): void {
    current = buildVariant(ctx.options.spatialTest);
    verifyDirty = true;
  }
  selectVariant();

  // Reports per half hour and how many were verified, for the timeline chart.
  const timelineBins = Math.ceil(
    (STORM_VERIFICATION_RANGE[1] - STORM_VERIFICATION_RANGE[0]) / TIMELINE_BIN_SECONDS
  );
  const timelineX = Array.from(
    {length: timelineBins},
    (_, bin) => (STORM_VERIFICATION_RANGE[0] + (bin + 0.5) * TIMELINE_BIN_SECONDS) / 3600
  );
  const reportsPerBin = new Float64Array(timelineBins);
  for (const time of reportTimes) {
    reportsPerBin[
      Math.min(
        timelineBins - 1,
        Math.max(0, Math.floor((time - STORM_VERIFICATION_RANGE[0]) / TIMELINE_BIN_SECONDS))
      )
    ]++;
  }

  function updateTimelineChart(force = false): void {
    const marker = Math.round(playhead / 300) * 300;
    if (!force && marker === chartMarker) return;
    chartMarker = marker;
    const verifiedPerBin = new Float64Array(timelineBins);
    if (results) {
      for (let report = 0; report < reportCount; report++) {
        if (results.verdict[report] !== 2) continue;
        verifiedPerBin[
          Math.min(
            timelineBins - 1,
            Math.max(
              0,
              Math.floor((reportTimes[report] - STORM_VERIFICATION_RANGE[0]) / TIMELINE_BIN_SECONDS)
            )
          )
        ]++;
      }
    }
    ctx.setChart(
      'timelineChart',
      seriesChart(
        [
          {label: 'reports', x: timelineX, y: reportsPerBin, area: true},
          ...(results ? [{label: 'verified', x: timelineX, y: verifiedPerBin, color: 2}] : [])
        ],
        {
          xLabel: 'hours after 12:00 UTC on 21 May',
          yLabel: 'reports / 30 min',
          xDomain: [STORM_VERIFICATION_RANGE[0] / 3600, STORM_VERIFICATION_RANGE[1] / 3600],
          markers: [{x: marker / 3600, label: 'now'}],
          formatX: value => value.toFixed(0),
          formatY: value => value.toFixed(0),
          description:
            'Storm reports per half hour (area) and how many of them were verified by a matching active warning (line); a marker shows the playhead.'
        }
      )
    );
  }

  function handleSummary(bytes: ArrayBuffer): void {
    const words = new Uint32Array(bytes);
    const floats = new Float32Array(bytes);
    const resultStart = 4;
    const hitStart = resultStart + reportCount * 4;
    const verdict = new Uint8Array(reportCount);
    const leadMinutes = new Float32Array(reportCount).fill(Number.NaN);
    const warning = new Float32Array(reportCount);
    const byKind = kindNames.map(() => [0, 0]);
    const leads: number[] = [];
    let verified = 0;
    let anyActive = 0;
    for (let report = 0; report < reportCount; report++) {
      const code = Math.round(floats[resultStart + report * 4]);
      verdict[report] = code;
      warning[report] = floats[resultStart + report * 4 + 2];
      byKind[reportKinds[report]][0]++;
      if (code >= 1) anyActive++;
      if (code === 2) {
        verified++;
        byKind[reportKinds[report]][1]++;
        const minutes = floats[resultStart + report * 4 + 1] / 60;
        leadMinutes[report] = minutes;
        leads.push(minutes);
      }
    }
    const hits = words.slice(hitStart, hitStart + warningCount);
    const chainHit = new Uint8Array(chainCount);
    for (let polygon = 0; polygon < warningCount; polygon++) {
      if (hits[polygon] > 0) chainHit[warningId[polygon]] = 1;
    }
    let warningsVerified = 0;
    for (const hit of chainHit) warningsVerified += hit;
    results = {verdict, leadMinutes, warning, polygonHits: hits};
    const share = verified / Math.max(1, reportCount);
    ctx.setReadout(
      'verifiedShare',
      `${formatInteger(verified)} of ${formatInteger(reportCount)} reports (${(share * 100).toFixed(0)}%)`
    );
    ctx.setReadout(
      'anyWarningShare',
      `${formatInteger(anyActive)} of ${formatInteger(reportCount)} (${((anyActive / Math.max(1, reportCount)) * 100).toFixed(0)}%)`
    );
    ctx.setReadout(
      'unwarned',
      `${formatInteger(reportCount - anyActive)} reports with no active warning`
    );
    ctx.setReadout(
      'warningsVerified',
      `${formatInteger(warningsVerified)} of ${formatInteger(chainCount)} warnings (${((warningsVerified / Math.max(1, chainCount)) * 100).toFixed(0)}%)`
    );
    ctx.setReadout(
      'medianLead',
      leads.length
        ? `${quantile(leads, 0.5).toFixed(0)} min (middle half ${quantile(leads, 0.25).toFixed(0)} to ${quantile(leads, 0.75).toFixed(0)})`
        : 'no verified reports'
    );
    ctx.setReadout(
      'pairs',
      `${formatCount(words[0])} pairs of ${formatCount(PAIR_CAPACITY)} (${words[1] ? 'OVERFLOW' : 'no overflow'}); ${formatCount(words[3])} candidates of ${formatCount(CANDIDATE_CAPACITY)}`
    );
    const medianLead = leads.length ? quantile(leads, 0.5) : Number.NaN;
    ctx.setChart(
      'leadChart',
      histogramChart(binValues(leads, 0, 60, 12), 0, 60, {
        xLabel: 'minutes from first issue to the report (60+ in the last bar)',
        yLabel: 'reports',
        formatX: value => value.toFixed(0),
        markers: Number.isFinite(medianLead)
          ? [{x: Math.min(60, medianLead), label: 'median'}]
          : [],
        description:
          'Histogram of lead time: minutes between the first issue of the matching warning and the storm report.'
      })
    );
    ctx.setChart('kindChart', {
      kind: 'bars',
      values: byKind.map(([total, hit]) => (total ? (hit / total) * 100 : 0)),
      labels: kindNames.map(
        name => `${KIND_LABELS[name] ?? name} (${byKind[kindNames.indexOf(name)][0]})`
      ),
      height: 130,
      yLabel: '% verified',
      yDomain: [0, 100],
      highlight: [],
      description: 'Share of reports of each kind that fell inside a matching active warning.'
    });
    updateTimelineChart(true);
    describeSelection();
    ctx.requestLayers();
  }

  const statusReader = new SummaryReader(
    resources,
    'storm-verify-status',
    [
      {buffer: warningActive, size: 4},
      {buffer: reportSeen, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      ctx.setReadout('activeWarnings', words[0]);
      ctx.setReadout('reportsSeen', words[1]);
      updateTimelineChart();
    }
  );

  function describeReport(report: number): string {
    const kind = KIND_LABELS[kindNames[reportKinds[report]]] ?? kindNames[reportKinds[report]];
    const magnitude = reportMagnitude[report];
    let text = `${kind}${Number.isFinite(magnitude) && magnitude > 0 ? ` ${magnitude}` : ''} at ${formatStormHourMinute(reportTimes[report])} UTC`;
    if (results) {
      const verdict = results.verdict[report];
      text += `: ${VERDICT_LABELS[verdict].toLowerCase()}`;
      if (verdict === 2) {
        const chain = results.warning[report];
        text += ` (lead ${results.leadMinutes[report].toFixed(0)} min, warning ${chain + 1})`;
      }
    }
    return text;
  }
  function describeSelection(): void {
    ctx.setReadout(
      'selected',
      selectedReport < 0 ? 'click a report' : describeReport(selectedReport)
    );
  }
  describeSelection();

  function pickReport(pixel: readonly [number, number]): number {
    const viewport = ctx.getViewport();
    if (!viewport) return -1;
    const memory = ctx.options.reportMinutes * 60;
    let best = -1;
    let bestDistance = 12 * 12;
    for (let report = 0; report < reportCount; report++) {
      if (reportTimes[report] > playhead || reportTimes[report] < playhead - memory) continue;
      const [x, y] = viewport.project([reportLngLat[report * 2], reportLngLat[report * 2 + 1]]);
      const squared = (x - pixel[0]) ** 2 + (y - pixel[1]) ** 2;
      if (squared < bestDistance) {
        bestDistance = squared;
        best = report;
      }
    }
    return best;
  }

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () =>
      [
        ...(current ? [current.compiled] : []),
        displayCompiled
      ] as unknown as CompiledGPUCommandGraph<never>[],

    setOption(id) {
      switch (id) {
        case 'spatialTest':
          selectVariant();
          writeVerifyParameters();
          break;
        case 'toleranceKm':
        case 'graceMinutes':
        case 'matching':
          writeVerifyParameters();
          verifyDirty = true;
          break;
        case 'time':
        case 'play':
        case 'speed':
        case 'loop':
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      playhead = clock.advance(frame);
      ctx.setReadout('clock', `${formatStormClock(playhead)}`);
      if (current && verifyDirty) {
        current.compiled.encode(commandEncoder, {parameters: undefined});
        current.reader.request(commandEncoder);
        verifyDirty = false;
      } else {
        current?.reader.flush(commandEncoder);
      }
      warningWindow.write(getGPUTimeWindowParameterValues({start: playhead, end: playhead}));
      reportWindow.write(
        getGPUTimeWindowParameterValues({
          start: playhead - options.reportMinutes * 60,
          end: playhead
        })
      );
      displayCompiled.encode(commandEncoder, {parameters: undefined});
      if (statusStale || frame.frameIndex % STATUS_INTERVAL_FRAMES === 0) {
        statusReader.markStale();
        statusStale = false;
      }
      statusReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.showWarnings) {
        layers.push(
          new ChoroplethFillLayer({
            id: 'storm-warning-fill',
            positions: meshPositionsBuffer,
            featureRows: meshFeaturesBuffer,
            vertexCount: mesh.triangleCount * 3,
            values: activeCategory,
            mode: 'category',
            palette: WARNING_COLORS,
            fillOpacity: options.warningOpacity
          })
        );
      }
      if (options.showOutlines) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'storm-warning-outline',
            ...lngLatDraw,
            segments: edgeBuffer,
            instanceCount: edgeCount,
            values: activeCategory,
            valueFormat: 'uint32',
            valueIndices: edgePolygonBuffer,
            colormap: 'category',
            palette: WARNING_COLORS,
            noDataColor: [0, 0, 0, 0],
            widthPixels: 1.8
          })
        );
      }
      if (options.showReports) {
        const byVerdict = options.colorReportsBy === 'verdict';
        const palette = byVerdict
          ? VERDICT_COLORS
          : kindNames.map(name => KIND_COLORS[name] ?? KIND_COLORS.other);
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'storm-report-halo',
            ...lngLatDraw,
            positions: reportLngLatBuffer,
            ids: reportIds,
            drawCommands: reportDraw,
            radiusPixels: options.reportSize + 1.6,
            color: dark ? [10, 12, 18, 235] : [255, 255, 255, 235]
          }),
          new SpatialAnalysisPointLayer({
            id: `storm-reports-${options.colorReportsBy}`,
            ...lngLatDraw,
            positions: reportLngLatBuffer,
            ids: reportIds,
            drawCommands: reportDraw,
            radiusPixels: options.reportSize,
            values: byVerdict ? verdictCodes : kindCodeBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      const report = pickReport(event.pixel);
      return report < 0 ? null : describeReport(report);
    },

    onClick(event) {
      const report = pickReport(event.pixel);
      selectedReport = report < 0 || report === selectedReport ? -1 : report;
      describeSelection();
      return report >= 0;
    },

    destroy() {
      destroyed = true;
      statusReader.stop();
      for (const variant of variants.values()) {
        variant.reader.stop();
        variant.resources.destroy();
      }
      resources.destroy();
    }
  };
}
