// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPUGroupStatistics,
  GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';
import {
  GPULineLengthPerPolygon,
  GPUZonalStatistics,
  type GPUZonalStatisticsSumOrder
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {fetchJson, getDataFileUrl} from '../../data/loaders';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {ChoroplethFillLayer} from '../statistics/b5-choropleth-layer';
import {buildPolygonMesh, createFeatureLocator, getPolygonLayout} from '../statistics/b5-geometry';
import type {SceneContext, SceneInstance} from '../scene';
import {
  formatInteger,
  formatStormClock,
  loadStormFlashes,
  loadStormTracks,
  METRIC_MODES,
  projectToStormFrame,
  seriesChart,
  STORM_VERIFICATION_RANGE
} from './storm-data';

/** Option state of the storm-outage-exposure scene. */
export type StormOutageExposureOptions = {
  play: boolean;
  time: number;
  speed: number;
  loop: boolean;
  metric:
    | 'outageNow'
    | 'outagePeak'
    | 'outageCount'
    | 'trackKm'
    | 'flashDensity'
    | 'meanDbz'
    | 'bivariate';
  exposureWindow: 'so-far' | 'event';
  exposureMeasure: 'trackKm' | 'flashDensity';
  exposureThreshold: number;
  outageThreshold: number;
  sumOrder: 'sorted' | 'atomic';
  fillOpacity: number;
  showCounties: boolean;
  showTracks: boolean;
};

const STATUS_STEP_SECONDS = 300;
const OUTAGE_WINDOW_SECONDS = 899;
const REGION = {west: -104.2, south: 24, east: -70.8, north: 48.8};
const BIVARIATE_COLORS = [
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

type ExposureVariant = {
  compiled: CompiledGPUCommandGraph<void>;
};

/**
 * Outage exposure. A static graph joins every flash to its county once (`GPUZonalStatistics`
 * polygon mode, which also writes the point-to-county rows) and reduces the whole outage table to
 * a peak per county (`GPUGroupStatistics`, dense keys). A second graph, re-encoded when the
 * playhead moves by five minutes, windows the flashes (`GPUTimeWindowFilter` fade weights), sums
 * them per county from the stored rows (`GPUZonalStatistics` feature-rows mode), clips the tracks
 * at the playhead and measures their length inside each county (`GPULineLengthPerPolygon`), windows
 * the outage snapshots and sums them per county. One kernel composes the county value drawn.
 */
export async function createStormOutageExposure(
  ctx: SceneContext<StormOutageExposureOptions>
): Promise<SceneInstance<StormOutageExposureOptions>> {
  const {device} = ctx;
  const countyData = ctx.datasets.get('us-counties');
  const outageData = ctx.datasets.get('poopdeck-mrms-storm3d-outages');
  const tracks = loadStormTracks(ctx.datasets.get('poopdeck-mrms-precip-tracks'));
  const flashes = loadStormFlashes(ctx.datasets.get('poopdeck-goes-glm-lightning'));
  const resources = new SpatialAnalysisResources(device, 'storm-outage');
  const lngLatDraw = {coordinateSystem: COORDINATE_SYSTEM.LNGLAT} as const;

  // ---- Counties -----------------------------------------------------------------------------
  ctx.setStatus('Reading county names');
  const names = await fetchJson<{fips: string[]; name: string[]; state: string[]}>(
    getDataFileUrl('us-counties', 'names.json'),
    ctx.signal
  );
  const layout = getPolygonLayout(countyData);
  const countyCount = layout.featureCount;
  const countyFips = countyData.column<Uint32Array>('fips');
  const population = countyData.column<Float32Array>('population');
  const countyAeqd = projectToStormFrame(layout.vertices);
  const countyPolygonOffsets = countyData.column<Uint32Array>('countyPolygonOffsets');
  const centroid = countyData.column<Float32Array>('centroid');
  const mesh = buildPolygonMesh(layout);
  const locator = createFeatureLocator(layout);
  const fipsToCounty = new Map<number, number>();
  countyFips.forEach((fips, county) => fipsToCounty.set(fips, county));

  // Planar area in square kilometers (shells add, holes subtract) and the study region mask.
  const countyStatic = new Float32Array(countyCount * 4);
  const inRegion = new Uint8Array(countyCount);
  let regionCount = 0;
  for (let county = 0; county < countyCount; county++) {
    let areaSquareMeters = 0;
    for (let part = countyPolygonOffsets[county]; part < countyPolygonOffsets[county + 1]; part++) {
      for (
        let ring = layout.polygonRingOffsets[part];
        ring < layout.polygonRingOffsets[part + 1];
        ring++
      ) {
        const first = layout.ringOffsets[ring];
        const last = layout.ringOffsets[ring + 1];
        let twice = 0;
        for (let vertex = first; vertex < last; vertex++) {
          const next = vertex + 1 < last ? vertex + 1 : first;
          twice +=
            countyAeqd[vertex * 2] * countyAeqd[next * 2 + 1] -
            countyAeqd[next * 2] * countyAeqd[vertex * 2 + 1];
        }
        areaSquareMeters +=
          (ring === layout.polygonRingOffsets[part] ? 1 : -1) * Math.abs(twice / 2);
      }
    }
    const longitude = centroid[county * 2];
    const latitude = centroid[county * 2 + 1];
    inRegion[county] =
      longitude >= REGION.west &&
      longitude <= REGION.east &&
      latitude >= REGION.south &&
      latitude <= REGION.north
        ? 1
        : 0;
    regionCount += inRegion[county];
    countyStatic.set([population[county], areaSquareMeters / 1e6, inRegion[county], 0], county * 4);
  }

  // ---- Outage table -------------------------------------------------------------------------
  const snapshotFips = outageData.column<Uint32Array>('fips');
  const snapshotCount = snapshotFips.length;
  const snapshotCounty = new Uint32Array(snapshotCount);
  for (let row = 0; row < snapshotCount; row++) {
    snapshotCounty[row] = fipsToCounty.get(snapshotFips[row]) ?? 0xffffffff;
  }
  const snapshotTimes = Float32Array.from(outageData.column<Uint32Array>('timestamp'));
  const snapshotValues = Float32Array.from(outageData.column<Uint32Array>('customersOut'));

  // Total customers out in the study region at each snapshot time (static, for the chart).
  const totalByTime = new Map<number, number>();
  for (let row = 0; row < snapshotCount; row++) {
    const county = snapshotCounty[row];
    if (county === 0xffffffff || !inRegion[county]) continue;
    totalByTime.set(
      snapshotTimes[row],
      (totalByTime.get(snapshotTimes[row]) ?? 0) + snapshotValues[row]
    );
  }
  const totalTimes = [...totalByTime.keys()].sort((a, b) => a - b);
  const totalX = totalTimes.map(time => time / 3600);
  const totalY = totalTimes.map(time => (totalByTime.get(time) ?? 0) / 1000);
  let peakTotal = 0;
  for (const value of totalByTime.values()) peakTotal = Math.max(peakTotal, value);

  // ---- Buffers ------------------------------------------------------------------------------
  const flashCount = flashes.count;
  const trackCount = tracks.trackCount;
  const vertexCount = tracks.vertexCount;
  const trackAeqd = projectToStormFrame(tracks.lngLat);
  const countyVerticesBuffer = resources.createBuffer('county-aeqd', countyAeqd);
  const countyFeatureOffsets = resources.createBuffer(
    'county-feature-offsets',
    countyPolygonOffsets
  );
  const countyPolygonRingOffsets = resources.createBuffer(
    'county-polygon-offsets',
    layout.polygonRingOffsets
  );
  const countyRingOffsets = resources.createBuffer('county-ring-offsets', layout.ringOffsets);
  const countyStaticBuffer = resources.createBuffer('county-static', countyStatic);
  const flashAeqdBuffer = resources.createBuffer('flash-aeqd', flashes.aeqd);
  const flashTimesBuffer = resources.createBuffer('flash-times', flashes.times);
  const flashOnesBuffer = resources.createBuffer(
    'flash-ones',
    new Float32Array(flashCount).fill(1)
  );
  const trackAeqdBuffer = resources.createBuffer('track-aeqd', trackAeqd);
  const trackTimesBuffer = resources.createBuffer('track-times', tracks.timestamps);
  const trackOffsetsBuffer = resources.createBuffer('track-offsets', tracks.offsets);
  const trackPeakBuffer = resources.createBuffer('track-peak-dbz', tracks.peakDbz);
  const snapshotCountyBuffer = resources.createBuffer('snapshot-county', snapshotCounty);
  const snapshotTimesBuffer = resources.createBuffer('snapshot-times', snapshotTimes);
  const snapshotValuesBuffer = resources.createBuffer('snapshot-values', snapshotValues);
  const meshPositionsBuffer = resources.createBuffer('mesh-positions', mesh.positions);
  const meshFeaturesBuffer = resources.createBuffer('mesh-features', mesh.featureRows);
  const outlineBuffer = resources.createBuffer('county-outline', mesh.outline);
  const segmentsBuffer = resources.createBuffer('track-segments', tracks.segments);

  // Static results.
  const pointCounty = resources.createBuffer('point-county', flashCount * 4);
  const flashTotals = resources.createBuffer('flash-totals', countyCount * 4);
  const zonalOverflow = resources.createBuffer('zonal-overflow', 4);
  const peakOutage = resources.createBuffer('peak-outage', countyCount * 4);
  const outageCustomerSnapshots = resources.createBuffer('outage-sums', countyCount * 4);
  const peakKeys = resources.createBuffer('peak-keys', countyCount * 4);
  const peakCounts = resources.createBuffer('peak-counts', countyCount * 4);
  const peakCount = resources.createBuffer('peak-count', 4);
  const peakOverflow = resources.createBuffer('peak-overflow', 4);

  // Per-change results.
  const flashIds = resources.createBuffer('flash-ids', flashCount * 4);
  const flashSelected = resources.createBuffer('flash-selected', 4);
  const flashOverflow = resources.createBuffer('flash-overflow', 4);
  const flashWeights = resources.createBuffer('flash-weights', flashCount * 4);
  const flashSums = resources.createBuffer('flash-sums', countyCount * 4);
  const clipped = resources.createBuffer('clipped-aeqd', vertexCount * 8);
  const trackLengths = resources.createBuffer('county-track-lengths', countyCount * 4);
  const trackWeighted = resources.createBuffer('county-track-weighted', countyCount * 4);
  const lineOverflow = resources.createBuffer('line-overflow', 4);
  const snapshotIds = resources.createBuffer('snapshot-ids', snapshotCount * 4);
  const snapshotSelected = resources.createBuffer('snapshot-selected', 4);
  const snapshotOverflow = resources.createBuffer('snapshot-overflow', 4);
  const snapshotMask = resources.createBuffer('snapshot-mask', snapshotCount * 4);
  const nowKeys = resources.createBuffer('now-keys', countyCount * 4);
  const nowCounts = resources.createBuffer('now-counts', countyCount * 4);
  const nowCount = resources.createBuffer('now-count', 4);
  const nowOverflow = resources.createBuffer('now-overflow', 4);
  const outageNow = resources.createBuffer('outage-now', countyCount * 4);
  const countyValues = resources.createBuffer('county-values', countyCount * 4);
  const bivariateClasses = resources.createBuffer('bivariate-classes', countyCount * 4);

  const flashWindow = resources.createParameterBuffer(
    'flash-window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  const snapshotWindow = resources.createParameterBuffer(
    'snapshot-window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  const clipLimit = resources.createParameterBuffer('clip-limit', 'float32', 4);
  const composeParameters = resources.createParameterBuffer('compose', 'float32', 4);

  // ---- Static graph -------------------------------------------------------------------------
  const staticGraph = new GPUCommandGraph<void>(device, {id: 'storm-outage-static'});
  staticGraph.add(
    new GPUZonalStatistics({
      id: 'flash-county',
      features: {
        kind: 'polygons',
        polygonPositions: importGraphBuffer(
          staticGraph,
          'county-aeqd',
          countyVerticesBuffer,
          'float32x2',
          countyAeqd.length / 2
        ),
        featureOffsets: importGraphBuffer(
          staticGraph,
          'county-feature-offsets',
          countyFeatureOffsets,
          'uint32',
          countyPolygonOffsets.length
        ),
        polygonOffsets: importGraphBuffer(
          staticGraph,
          'county-polygon-offsets',
          countyPolygonRingOffsets,
          'uint32',
          layout.polygonRingOffsets.length
        ),
        ringOffsets: importGraphBuffer(
          staticGraph,
          'county-ring-offsets',
          countyRingOffsets,
          'uint32',
          layout.ringOffsets.length
        ),
        candidateCapacity: flashCount * 4
      },
      points: importGraphBuffer(
        staticGraph,
        'flash-aeqd',
        flashAeqdBuffer,
        'float32x2',
        flashCount
      ),
      output: {
        counts: importGraphBuffer(staticGraph, 'flash-totals', flashTotals, 'uint32', countyCount),
        pointFeatureRows: importGraphBuffer(
          staticGraph,
          'point-county',
          pointCounty,
          'uint32',
          flashCount
        ),
        overflow: importGraphBuffer(staticGraph, 'zonal-overflow', zonalOverflow, 'uint32', 1)
      }
    })
  );
  staticGraph.add(
    new GPUGroupStatistics({
      id: 'county-peak',
      keys: importGraphBuffer(
        staticGraph,
        'snapshot-county',
        snapshotCountyBuffer,
        'uint32',
        snapshotCount
      ),
      keyCount: countyCount,
      columns: [
        {
          values: importGraphBuffer(
            staticGraph,
            'snapshot-values',
            snapshotValuesBuffer,
            'float32',
            snapshotCount
          ),
          statistics: ['maximum', 'sum'],
          output: {
            maximums: importGraphBuffer(
              staticGraph,
              'peak-outage',
              peakOutage,
              'float32',
              countyCount
            ),
            sumValues: importGraphBuffer(
              staticGraph,
              'outage-sums',
              outageCustomerSnapshots,
              'float32',
              countyCount
            )
          }
        }
      ],
      output: {
        keys: importGraphBuffer(staticGraph, 'peak-keys', peakKeys, 'uint32', countyCount),
        counts: importGraphBuffer(staticGraph, 'peak-counts', peakCounts, 'uint32', countyCount),
        count: importGraphBuffer(staticGraph, 'peak-count', peakCount, 'uint32', 1),
        overflow: importGraphBuffer(staticGraph, 'peak-overflow', peakOverflow, 'uint32', 1)
      }
    })
  );
  const staticCompiled = resources.track(staticGraph.compile());

  // ---- Exposure graphs (the zonal sum order is compile-time) ----------------------------------
  const variants = new Map<string, ExposureVariant>();

  function buildVariant(sumOrder: GPUZonalStatisticsSumOrder): ExposureVariant {
    const existing = variants.get(sumOrder);
    if (existing) return existing;
    const graph = new GPUCommandGraph<void>(device, {id: `storm-outage-exposure-${sumOrder}`});
    // Each buffer is imported once per graph so the graph orders writers before readers.
    const flashWeightsView = importGraphBuffer(
      graph,
      'flash-weights',
      flashWeights,
      'float32',
      flashCount
    );
    const flashSumsView = importGraphBuffer(graph, 'flash-sums', flashSums, 'float32', countyCount);
    const trackLengthsView = importGraphBuffer(
      graph,
      'track-lengths',
      trackLengths,
      'float32',
      countyCount
    );
    const trackWeightedView = importGraphBuffer(
      graph,
      'track-weighted',
      trackWeighted,
      'float32',
      countyCount
    );
    const outageNowView = importGraphBuffer(graph, 'outage-now', outageNow, 'float32', countyCount);
    const trackOffsetsView = importGraphBuffer(
      graph,
      'track-offsets',
      trackOffsetsBuffer,
      'uint32',
      trackCount + 1
    );
    // Flashes in the window: fade weights are 1 inside and 0 outside.
    graph.add(
      new GPUTimeWindowFilter({
        id: 'flash-window',
        timestamps: importGraphBuffer(
          graph,
          'flash-times',
          flashTimesBuffer,
          'float32',
          flashCount
        ),
        window: flashWindow.importToGraph(graph),
        output: {
          ids: importGraphBuffer(graph, 'flash-ids', flashIds, 'uint32', flashCount),
          count: importGraphBuffer(graph, 'flash-selected', flashSelected, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'flash-overflow', flashOverflow, 'uint32', 1)
        },
        fadeWeights: flashWeightsView
      })
    );
    graph.add(
      new GPUZonalStatistics({
        id: 'flash-sums',
        features: {
          kind: 'feature-rows',
          pointFeatureRows: importGraphBuffer(
            graph,
            'point-county',
            pointCounty,
            'uint32',
            flashCount
          ),
          featureCount: countyCount
        },
        values: importGraphBuffer(graph, 'flash-ones', flashOnesBuffer, 'float32', flashCount),
        weights: flashWeightsView,
        sumOrder,
        output: {sums: flashSumsView}
      })
    );
    // Cell tracks up to the clip limit: later vertices repeat the last visible one (zero length).
    const clippedView = importGraphBuffer(graph, 'clipped-aeqd', clipped, 'float32x2', vertexCount);
    addKernelPass(graph, {
      id: 'clip-tracks',
      invocationCount: trackCount,
      bindings: [
        {
          name: 'positions',
          view: importGraphBuffer(graph, 'track-aeqd', trackAeqdBuffer, 'float32x2', vertexCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'times',
          view: importGraphBuffer(graph, 'track-times', trackTimesBuffer, 'float32', vertexCount),
          type: 'f32',
          access: 'read'
        },
        {name: 'offsets', view: trackOffsetsView, type: 'u32', access: 'read'},
        {name: 'limit', view: clipLimit.importToGraph(graph), type: 'f32', access: 'read'},
        {name: 'clippedPositions', view: clippedView, type: 'f32', access: 'read_write'}
      ],
      body: `let firstRow = offsets[offsetsOffset + index];
  let lastRow = offsets[offsetsOffset + index + 1u];
  var visibleRow = firstRow;
  for (var row = firstRow; row < lastRow; row = row + 1u) {
    let seen = times[timesOffset + row] <= limit[limitOffset];
    if (seen) { visibleRow = row; }
    let source = select(visibleRow, row, seen);
    clippedPositions[clippedPositionsOffset + row * 2u] = positions[positionsOffset + source * 2u];
    clippedPositions[clippedPositionsOffset + row * 2u + 1u] = positions[positionsOffset + source * 2u + 1u];
  }`
    });
    graph.add(
      new GPULineLengthPerPolygon({
        id: 'track-length',
        positions: clippedView,
        pathOffsets: trackOffsetsView,
        maximumCandidatePairs: Math.max(1024, 64 * vertexCount),
        pathWeights: importGraphBuffer(
          graph,
          'track-peak-dbz',
          trackPeakBuffer,
          'float32',
          trackCount
        ),
        polygons: {
          kind: 'polygons',
          positions: importGraphBuffer(
            graph,
            'county-aeqd',
            countyVerticesBuffer,
            'float32x2',
            countyAeqd.length / 2
          ),
          featureOffsets: importGraphBuffer(
            graph,
            'county-feature-offsets',
            countyFeatureOffsets,
            'uint32',
            countyPolygonOffsets.length
          ),
          polygonOffsets: importGraphBuffer(
            graph,
            'county-polygon-offsets',
            countyPolygonRingOffsets,
            'uint32',
            layout.polygonRingOffsets.length
          ),
          ringOffsets: importGraphBuffer(
            graph,
            'county-ring-offsets',
            countyRingOffsets,
            'uint32',
            layout.ringOffsets.length
          )
        },
        output: {
          lengths: trackLengthsView,
          weightedLengths: trackWeightedView,
          overflow: importGraphBuffer(graph, 'line-overflow', lineOverflow, 'uint32', 1)
        }
      })
    );
    // The latest outage snapshot of every county, summed per county.
    const maskView = importGraphBuffer(
      graph,
      'snapshot-mask',
      snapshotMask,
      'uint32',
      snapshotCount
    );
    graph.add(
      new GPUTimeWindowFilter({
        id: 'snapshot-window',
        timestamps: importGraphBuffer(
          graph,
          'snapshot-times',
          snapshotTimesBuffer,
          'float32',
          snapshotCount
        ),
        window: snapshotWindow.importToGraph(graph),
        output: {
          ids: importGraphBuffer(graph, 'snapshot-ids', snapshotIds, 'uint32', snapshotCount),
          count: importGraphBuffer(graph, 'snapshot-selected', snapshotSelected, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'snapshot-overflow', snapshotOverflow, 'uint32', 1)
        },
        outputMask: maskView
      })
    );
    graph.add(
      new GPUGroupStatistics({
        id: 'outage-now',
        keys: importGraphBuffer(
          graph,
          'snapshot-county',
          snapshotCountyBuffer,
          'uint32',
          snapshotCount
        ),
        mask: maskView,
        keyCount: countyCount,
        columns: [
          {
            values: importGraphBuffer(
              graph,
              'snapshot-values',
              snapshotValuesBuffer,
              'float32',
              snapshotCount
            ),
            statistics: ['sum'],
            output: {sumValues: outageNowView}
          }
        ],
        output: {
          keys: importGraphBuffer(graph, 'now-keys', nowKeys, 'uint32', countyCount),
          counts: importGraphBuffer(graph, 'now-counts', nowCounts, 'uint32', countyCount),
          count: importGraphBuffer(graph, 'now-count', nowCount, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'now-overflow', nowOverflow, 'uint32', 1)
        }
      })
    );
    // One kernel composes the value of the chosen metric for every county.
    addKernelPass(graph, {
      id: 'compose-county-values',
      invocationCount: countyCount,
      bindings: [
        {
          name: 'county',
          view: importGraphBuffer(
            graph,
            'county-static',
            countyStaticBuffer,
            'float32',
            countyCount * 4
          ),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'outageNow',
          view: outageNowView,
          type: 'f32',
          access: 'read'
        },
        {
          name: 'outagePeak',
          view: importGraphBuffer(graph, 'peak-outage-read', peakOutage, 'float32', countyCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'trackLengths',
          view: trackLengthsView,
          type: 'f32',
          access: 'read'
        },
        {
          name: 'trackWeighted',
          view: trackWeightedView,
          type: 'f32',
          access: 'read'
        },
        {
          name: 'flashSums',
          view: flashSumsView,
          type: 'f32',
          access: 'read'
        },
        {name: 'mode', view: composeParameters.importToGraph(graph), type: 'f32', access: 'read'},
        {
          name: 'values',
          view: importGraphBuffer(graph, 'county-values', countyValues, 'float32', countyCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  let people = max(county[countyOffset + index * 4u], 1.0);
  let area = max(county[countyOffset + index * 4u + 1u], 0.001);
  let inside = county[countyOffset + index * 4u + 2u] > 0.5;
  let selected = u32(mode[modeOffset]);
  var value = nan;
  if (selected == 0u) {
    value = select(nan, outageNow[outageNowOffset + index] / people * 1000.0, outageNow[outageNowOffset + index] > 0.0);
  } else if (selected == 1u) {
    value = select(nan, outagePeak[outagePeakOffset + index] / people * 1000.0, outagePeak[outagePeakOffset + index] > 0.0);
  } else if (selected == 2u) {
    value = select(nan, outageNow[outageNowOffset + index], outageNow[outageNowOffset + index] > 0.0);
  } else if (selected == 3u) {
    value = select(nan, trackLengths[trackLengthsOffset + index] / 1000.0, trackLengths[trackLengthsOffset + index] > 100.0);
  } else if (selected == 4u) {
    value = select(nan, flashSums[flashSumsOffset + index] * mode[modeOffset + 1u] / area * 1000.0, flashSums[flashSumsOffset + index] > 0.0);
  } else {
    value = select(nan, trackWeighted[trackWeightedOffset + index] / trackLengths[trackLengthsOffset + index], trackLengths[trackLengthsOffset + index] > 100.0);
  }
  values[valuesOffset + index] = select(nan, value, inside);`
    });
    const built = {compiled: resources.track(graph.compile())};
    variants.set(sumOrder, built);
    return built;
  }

  // ---- State ------------------------------------------------------------------------------------
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: STORM_VERIFICATION_RANGE, rate: 60, step: 300}
  );
  let playhead = ctx.options.time;
  let destroyed = false;
  let staticEncoded = false;
  let exposureDirty = true;
  let lastStep = -1;
  let current = buildVariant(ctx.options.sumOrder);
  let peakPerThousand: Float32Array | null = null;
  let latest: {
    values: Float32Array;
    outageNow: Float32Array;
    trackLengths: Float32Array;
    flashSums: Float32Array;
  } | null = null;
  let valueRange: [number, number] = [0, 1];
  let hoveredCounty = -1;
  let legendMetric = '';
  const flashScale = 1 / flashes.sampleFraction;

  function writeComposeParameters(): void {
    composeParameters.write(
      Float32Array.of(
        METRIC_MODES.indexOf(
          ctx.options.metric === 'bivariate' ? 'outagePeak' : ctx.options.metric
        ),
        flashScale,
        0,
        0
      )
    );
    exposureDirty = true;
  }
  function writeWindows(): void {
    const options = ctx.options;
    const quantized = Math.floor(playhead / STATUS_STEP_SECONDS) * STATUS_STEP_SECONDS;
    const limit = options.exposureWindow === 'so-far' ? quantized : 1e9;
    flashWindow.write(getGPUTimeWindowParameterValues({start: 0, end: Math.min(limit, 1e7)}));
    clipLimit.write(Float32Array.of(Math.min(limit, 1e7), 0, 0, 0));
    snapshotWindow.write(
      getGPUTimeWindowParameterValues({start: quantized - OUTAGE_WINDOW_SECONDS, end: quantized})
    );
  }
  writeComposeParameters();
  writeWindows();

  ctx.setReadout(
    'inputs',
    `${formatInteger(regionCount)} counties in the storm region, ${formatInteger(trackCount)} cell tracks, ${formatInteger(flashCount)} flashes, ${formatInteger(snapshotCount)} outage snapshots`
  );
  ctx.setReadout('peakTotal', `${formatInteger(peakTotal)} customers (all storm-region counties)`);

  let chartMarker = -1;
  function updateTotalChart(force = false): void {
    const marker = Math.floor(playhead / STATUS_STEP_SECONDS) * STATUS_STEP_SECONDS;
    if (!force && marker === chartMarker) return;
    chartMarker = marker;
    ctx.setChart(
      'totalChart',
      seriesChart(
        [{label: 'customers without power (thousands)', x: totalX, y: totalY, area: true}],
        {
          xLabel: 'hours after 12:00 UTC on 21 May',
          yLabel: 'thousand customers',
          xDomain: [STORM_VERIFICATION_RANGE[0] / 3600, STORM_VERIFICATION_RANGE[1] / 3600],
          markers: [{x: marker / 3600, label: 'now'}],
          formatX: value => value.toFixed(0),
          formatY: value => value.toFixed(0),
          description:
            'Customers without power in the storm-region counties at each 15-minute snapshot, with a marker at the playhead.'
        }
      )
    );
  }
  updateTotalChart(true);

  // ---- Readbacks --------------------------------------------------------------------------------
  const staticReader = new SummaryReader(
    resources,
    'storm-outage-static',
    [
      {buffer: peakOutage, size: countyCount * 4},
      {buffer: zonalOverflow, size: 4},
      {buffer: peakOverflow, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const peaks = new Float32Array(bytes, 0, countyCount);
      peakPerThousand = Float32Array.from(peaks, (value, county) =>
        Number.isFinite(value) ? (value / Math.max(1, population[county])) * 1000 : 0
      );
      const flags = new Uint32Array(bytes, countyCount * 4, 2);
      ctx.setReadout(
        'overflow',
        flags[0] || flags[1] ? 'a capacity overflowed' : 'no capacity overflowed'
      );
      refreshStatistics();
    }
  );
  const exposureReader = new SummaryReader(
    resources,
    'storm-outage-exposure',
    [
      {buffer: countyValues, size: countyCount * 4},
      {buffer: outageNow, size: countyCount * 4},
      {buffer: trackLengths, size: countyCount * 4},
      {buffer: flashSums, size: countyCount * 4},
      {buffer: flashSelected, size: 4},
      {buffer: lineOverflow, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const stride = countyCount;
      latest = {
        values: new Float32Array(bytes, 0, stride).slice(),
        outageNow: new Float32Array(bytes, stride * 4, stride).slice(),
        trackLengths: new Float32Array(bytes, stride * 8, stride).slice(),
        flashSums: new Float32Array(bytes, stride * 12, stride).slice()
      };
      const tail = new Uint32Array(bytes, stride * 16, 2);
      ctx.setReadout('flashesCounted', formatInteger(tail[0] * flashScale));
      refreshStatistics();
    }
  );

  function updateRange(): void {
    if (!latest) return;
    const finite: number[] = [];
    for (const value of latest.values) if (Number.isFinite(value)) finite.push(value);
    finite.sort((a, b) => a - b);
    const high = finite.length
      ? finite[Math.min(finite.length - 1, Math.floor(0.98 * finite.length))]
      : 1;
    const next: [number, number] = [0, Math.max(high, 1e-6)];
    const changed =
      Math.abs(next[1] - valueRange[1]) > 0.02 * valueRange[1] ||
      legendMetric !== ctx.options.metric;
    valueRange = next;
    if (changed) {
      legendMetric = ctx.options.metric;
      ctx.setLegendData('range', {low: 0, high: next[1], metric: ctx.options.metric});
    }
  }

  function exposureOf(county: number): number {
    if (!latest) return 0;
    if (ctx.options.exposureMeasure === 'trackKm') return latest.trackLengths[county] / 1000;
    return (
      ((latest.flashSums[county] * flashScale) / Math.max(0.001, countyStatic[county * 4 + 1])) *
      1000
    );
  }

  function writeBivariateClasses(): void {
    if (!latest || !peakPerThousand) return;
    const values = new Uint32Array(countyCount).fill(0xffffffff);
    const exposureBreaks = [ctx.options.exposureThreshold / 3, ctx.options.exposureThreshold];
    const outageBreaks = [ctx.options.outageThreshold / 3, ctx.options.outageThreshold];
    for (let county = 0; county < countyCount; county++) {
      if (!inRegion[county]) continue;
      const exposure = exposureOf(county);
      const exposureClass = exposure < exposureBreaks[0] ? 0 : exposure < exposureBreaks[1] ? 1 : 2;
      const outage = peakPerThousand[county];
      const outageClass = outage < outageBreaks[0] ? 0 : outage < outageBreaks[1] ? 1 : 2;
      values[county] = outageClass * 3 + exposureClass;
    }
    bivariateClasses.write(values);
  }

  function refreshStatistics(): void {
    if (!latest) return;
    updateRange();
    ctx.requestLayers();
    ctx.setReadout(
      'customersNow',
      `${formatInteger(sumNow())} customers without power (storm-region counties)`
    );
    if (!peakPerThousand) return;
    const threshold = ctx.options.exposureThreshold;
    const outageLimit = ctx.options.outageThreshold;
    const classes = [0, 0, 0, 0];
    const withOutage = [0, 0, 0, 0];
    for (let county = 0; county < countyCount; county++) {
      if (!inRegion[county]) continue;
      const exposure = exposureOf(county);
      const index = exposure <= 0 ? 0 : exposure < threshold ? 1 : exposure < 3 * threshold ? 2 : 3;
      classes[index]++;
      if (peakPerThousand[county] >= outageLimit) withOutage[index]++;
    }
    const share = (index: number) =>
      classes[index] ? (withOutage[index] / classes[index]) * 100 : 0;
    const unit = ctx.options.exposureMeasure === 'trackKm' ? 'km' : '/1,000 km2';
    ctx.setChart('exposureChart', {
      kind: 'bars',
      values: [share(0), share(1), share(2), share(3)],
      labels: [
        `none (${classes[0]})`,
        `under ${threshold} ${unit} (${classes[1]})`,
        `${threshold} to ${3 * threshold} (${classes[2]})`,
        `${3 * threshold}+ (${classes[3]})`
      ],
      height: 130,
      yLabel: '% of counties',
      yDomain: [0, 100],
      highlight: [3],
      description:
        'Share of storm-region counties whose peak outage reached the threshold, grouped by how much storm exposure they had so far.'
    });
    writeBivariateClasses();
    const exposedCount = classes[2] + classes[3];
    const unexposedCount = classes[0] + classes[1];
    const exposedShare = exposedCount ? (withOutage[2] + withOutage[3]) / exposedCount : 0;
    const unexposedShare = unexposedCount ? (withOutage[0] + withOutage[1]) / unexposedCount : 0;
    ctx.setReadout(
      'exposedShare',
      `${(exposedShare * 100).toFixed(0)}% of ${formatInteger(exposedCount)} counties`
    );
    ctx.setReadout(
      'unexposedShare',
      `${(unexposedShare * 100).toFixed(0)}% of ${formatInteger(unexposedCount)} counties`
    );
    ctx.setReadout(
      'ratio',
      unexposedShare > 0
        ? `${(exposedShare / unexposedShare).toFixed(1)}x`
        : 'n/a (no outages in the lighter group)'
    );
  }

  function sumNow(): number {
    if (!latest) return 0;
    let total = 0;
    for (let county = 0; county < countyCount; county++) {
      if (inRegion[county] && Number.isFinite(latest.outageNow[county]))
        total += latest.outageNow[county];
    }
    return total;
  }

  function describeCounty(county: number): string {
    const label = `${names.name[county]} County, ${names.state[county]}`;
    if (!latest) return label;
    const parts = [
      `${formatInteger(latest.outageNow[county])} customers without power now`,
      `peak ${peakPerThousand ? peakPerThousand[county].toFixed(1) : 'n/a'} per 1,000 residents`,
      `${(latest.trackLengths[county] / 1000).toFixed(0)} km of cell track`,
      `${(((latest.flashSums[county] * flashScale) / Math.max(0.001, countyStatic[county * 4 + 1])) * 1000).toFixed(0)} flashes per 1,000 km2`
    ];
    return `${label}: ${parts.join('; ')}`;
  }

  ctx.setReadout('hovered', 'hover a county');

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () =>
      [staticCompiled, current.compiled] as unknown as CompiledGPUCommandGraph<never>[],

    setOption(id) {
      switch (id) {
        case 'metric':
          writeComposeParameters();
          break;
        case 'exposureWindow':
          writeWindows();
          exposureDirty = true;
          break;
        case 'sumOrder':
          current = buildVariant(ctx.options.sumOrder);
          exposureDirty = true;
          break;
        case 'exposureMeasure':
        case 'exposureThreshold':
        case 'outageThreshold':
          refreshStatistics();
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

    onGroundChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      playhead = clock.advance(frame);
      ctx.setReadout('clock', formatStormClock(playhead));
      if (!staticEncoded) {
        staticCompiled.encode(commandEncoder, {parameters: undefined});
        staticReader.request(commandEncoder);
        staticEncoded = true;
      } else {
        staticReader.flush(commandEncoder);
      }
      const step = Math.floor(playhead / STATUS_STEP_SECONDS);
      if (step !== lastStep) {
        lastStep = step;
        writeWindows();
        exposureDirty = true;
        updateTotalChart();
      }
      if (exposureDirty && staticEncoded && !exposureReader.isPending) {
        current.compiled.encode(commandEncoder, {parameters: undefined});
        exposureReader.request(commandEncoder);
        exposureDirty = false;
      } else {
        exposureReader.flush(commandEncoder);
      }
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.ground() === 'dark';
      const layers: Layer[] = [
        new ChoroplethFillLayer({
          id: 'storm-county-fill',
          positions: meshPositionsBuffer,
          featureRows: meshFeaturesBuffer,
          vertexCount: mesh.triangleCount * 3,
          values: options.metric === 'bivariate' ? bivariateClasses : countyValues,
          mode: options.metric === 'bivariate' ? 'category' : 'ramp',
          ...(options.metric === 'bivariate' ? {palette: BIVARIATE_COLORS} : {}),
          ramp:
            options.metric === 'trackKm' || options.metric === 'flashDensity' ? 'mako' : 'magma',
          valueRange,
          sqrtScale: true,
          noDataColor: [0, 0, 0, 0],
          selectedRow: hoveredCounty,
          fillOpacity: options.fillOpacity
        })
      ];
      if (options.showCounties) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'storm-county-outline',
            ...lngLatDraw,
            segments: outlineBuffer,
            instanceCount: mesh.outline.length / 4,
            widthPixels: 0.7,
            color: dark ? [210, 215, 230, 60] : [40, 50, 70, 70]
          })
        );
      }
      if (options.showTracks) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'storm-outage-tracks',
            ...lngLatDraw,
            segments: segmentsBuffer,
            instanceCount: tracks.segmentCount,
            widthPixels: 1.4,
            color: dark ? [255, 255, 255, 120] : [20, 24, 32, 140]
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const county = locator.locate(event.coordinate[0], event.coordinate[1]);
      if (county < 0 || !inRegion[county]) return null;
      return describeCounty(county);
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const county = locator.locate(event.coordinate[0], event.coordinate[1]);
      hoveredCounty = county < 0 || county === hoveredCounty ? -1 : county;
      ctx.setReadout(
        'hovered',
        hoveredCounty < 0 ? 'click a county' : describeCounty(hoveredCounty)
      );
      ctx.requestLayers();
      return county >= 0;
    },

    destroy() {
      destroyed = true;
      staticReader.stop();
      exposureReader.stop();
      resources.destroy();
    }
  };
}
