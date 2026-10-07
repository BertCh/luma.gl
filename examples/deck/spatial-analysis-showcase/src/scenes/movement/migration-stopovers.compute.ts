// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUSpatialClusteringParameterValues,
  getGPUTrajectoryMetricsParameterValues,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
  GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH,
  GPUSpatialClustering,
  GPUTrajectoryMetrics
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {binValues, histogramChart} from './f-chart-helpers';
import {formatDuration} from './b12-tracks';
import {
  formatYearDay,
  loadMigrationTracks,
  MIGRATION_DATASET_ID,
  MIGRATION_SPECIES_COLORS,
  MIGRATION_SPECIES_LABELS,
  nameSite,
  SECONDS_PER_DAY
} from './migration-shared';

/** Option state of the migration stopovers scene. */
export type MigrationStopoversOptions = {
  species: 'all' | 'marsh' | 'montagu' | 'spoonbill';
  stopSpeed: number;
  minStayHours: number;
  maxStayDays: number;
  dayRange: readonly [number, number];
  epsilonKm: number;
  minimumStops: number;
  rankBy: 'dwell' | 'birds' | 'stops';
  colorStops: 'stay' | 'cluster' | 'species';
  showSites: boolean;
  showTracks: boolean;
  trackOpacity: number;
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
};

const STOP_CAPACITY = 16384;
const CLUSTER_CAPACITY = 512;
const SITES_DRAWN = 24;
const SITES_LISTED = 10;
const SETTLE_MILLISECONDS = 200;
const NOISE = 0xffffffff;
const SPECIES_MASKS = {all: 7, marsh: 1, montagu: 2, spoonbill: 4} as const;
const SPECIES_INDEX = {marsh: 0, montagu: 1, spoonbill: 2} as const;
/** Stay length at the top of the stay ramp. */
const STAY_RAMP_DAYS = 30;

type Site = {
  rank: number;
  cluster: number;
  lng: number;
  lat: number;
  name: string;
  stops: number;
  birds: number;
  dwellDays: number;
  medianStayDays: number;
  medianStartDay: number;
  species: number[];
};

/**
 * Migration stopovers: `GPUTrajectoryMetrics` finds every dwell (a run of slow two-hour steps that
 * lasts at least the minimum stay); a kernel keeps the stays that are short enough to be a stopover
 * rather than a home range, started in the chosen season, by the chosen species; `GPUSpatialClustering`
 * (DBSCAN) groups the surviving stops into sites; the CPU ranks the sites by bird-days of dwell.
 * Analysis runs in azimuthal-equidistant meters, drawing in longitude/latitude.
 */
export async function createMigrationStopovers(
  ctx: SceneContext<MigrationStopoversOptions>
): Promise<SceneInstance<MigrationStopoversOptions>> {
  const tracks = loadMigrationTracks(ctx.datasets.get(MIGRATION_DATASET_ID));
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = tracks;
  const resources = new SpatialAnalysisResources(device, 'stopovers');
  const drawProps = {coordinateSystem: COORDINATE_SYSTEM.LNGLAT} as const;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    graph: GPUCommandGraph<void>,
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, name, buffer, format, length);

  // ---- Static inputs ----------------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', tracks.positions);
  const lngLatBuffer = resources.createBuffer('lng-lat', tracks.lngLat);
  const timestampsBuffer = resources.createBuffer('timestamps', tracks.timestamps);
  const offsetsBuffer = resources.createBuffer('offsets', tracks.offsets);
  const segmentsBuffer = resources.createBuffer('segments', tracks.segments);
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', tracks.segmentTracks);
  const speciesBuffer = resources.createBuffer('species', Uint32Array.from(tracks.species));

  // ---- Metrics graph: stops and their drawable centroids ----------------------------------------
  const trackLengths = resources.createBuffer('track-lengths', trackCount * 4);
  const maximumSpeeds = resources.createBuffer('maximum-speeds', trackCount * 4);
  const trackStopCounts = resources.createBuffer('track-stop-counts', trackCount * 4);
  const stopIds = resources.createBuffer('stop-ids', STOP_CAPACITY * 4);
  const stopCount = resources.createBuffer('stop-count', 4);
  const stopOverflow = resources.createBuffer('stop-overflow', 4);
  const stopTotal = resources.createBuffer('stop-total', 4);
  const stopCentroids = resources.createBuffer('stop-centroids', STOP_CAPACITY * 8);
  const stopDurations = resources.createBuffer('stop-durations', STOP_CAPACITY * 4);
  const stopStartRows = resources.createBuffer('stop-start-rows', STOP_CAPACITY * 4);
  const stopEndRows = resources.createBuffer('stop-end-rows', STOP_CAPACITY * 4);
  const stopLngLat = resources.createBuffer('stop-lng-lat', STOP_CAPACITY * 8);
  const metricsParameters = resources.createParameterBuffer(
    'metrics-parameters',
    'float32',
    GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH
  );
  const metricsGraph = new GPUCommandGraph<void>(device, {id: 'stopovers-metrics'});
  const stopCountView = view(metricsGraph, 'stop-count', stopCount, 'uint32', 1);
  const startRowsView = view(
    metricsGraph,
    'stop-start-rows',
    stopStartRows,
    'uint32',
    STOP_CAPACITY
  );
  const endRowsView = view(metricsGraph, 'stop-end-rows', stopEndRows, 'uint32', STOP_CAPACITY);
  metricsGraph.add(
    new GPUTrajectoryMetrics({
      id: 'metrics',
      positions: view(metricsGraph, 'positions', positionsBuffer, 'float32x2', vertexCount),
      timestamps: view(metricsGraph, 'timestamps', timestampsBuffer, 'float32', vertexCount),
      trackOffsets: view(metricsGraph, 'offsets', offsetsBuffer, 'uint32', trackCount + 1),
      parameters: metricsParameters.importToGraph(metricsGraph),
      trackLengths: view(metricsGraph, 'track-lengths', trackLengths, 'float32', trackCount),
      maximumSpeeds: view(metricsGraph, 'maximum-speeds', maximumSpeeds, 'float32', trackCount),
      trackStopCounts: view(
        metricsGraph,
        'track-stop-counts',
        trackStopCounts,
        'uint32',
        trackCount
      ),
      stops: {
        output: {
          ids: view(metricsGraph, 'stop-ids', stopIds, 'uint32', STOP_CAPACITY),
          count: stopCountView,
          overflow: view(metricsGraph, 'stop-overflow', stopOverflow, 'uint32', 1),
          totalCount: view(metricsGraph, 'stop-total', stopTotal, 'uint32', 1)
        },
        startRows: startRowsView,
        endRows: endRowsView,
        centroids: view(metricsGraph, 'stop-centroids', stopCentroids, 'float32x2', STOP_CAPACITY),
        durations: view(metricsGraph, 'stop-durations', stopDurations, 'float32', STOP_CAPACITY)
      }
    })
  );
  // Planar centroids drive the clustering; drawing needs degrees, so average the lng/lat of each stop's rows.
  addKernelPass(metricsGraph, {
    id: 'stop-lng-lat',
    invocationCount: STOP_CAPACITY,
    bindings: [
      {name: 'stopCount', view: stopCountView, type: 'u32', access: 'read'},
      {name: 'startRows', view: startRowsView, type: 'u32', access: 'read'},
      {name: 'endRows', view: endRowsView, type: 'u32', access: 'read'},
      {
        name: 'lngLat',
        view: view(metricsGraph, 'lng-lat', lngLatBuffer, 'float32x2', vertexCount),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'output',
        view: view(metricsGraph, 'stop-lng-lat', stopLngLat, 'float32x2', STOP_CAPACITY),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  var lng = nan;
  var lat = nan;
  if (index < stopCount[stopCountOffset]) {
    let first = startRows[startRowsOffset + index];
    let last = endRows[endRowsOffset + index];
    var sumLng = 0.0;
    var sumLat = 0.0;
    for (var row = first; row <= last; row++) {
      sumLng += lngLat[lngLatOffset + row * 2u];
      sumLat += lngLat[lngLatOffset + row * 2u + 1u];
    }
    let n = f32(last - first + 1u);
    lng = sumLng / n;
    lat = sumLat / n;
  }
  output[outputOffset + index * 2u] = lng;
  output[outputOffset + index * 2u + 1u] = lat;`
  });
  const metricsCompiled = resources.track(metricsGraph.compile());

  // ---- Selection graph: which stops are stopovers of the chosen species and season --------------
  const selectedMeters = resources.createBuffer('selected-meters', STOP_CAPACITY * 8);
  const selectedLngLat = resources.createBuffer('selected-lng-lat', STOP_CAPACITY * 8);
  // [species mask, longest stay (s), season start (s), season end (s)]
  const selectionParameters = resources.createParameterBuffer('selection-parameters', 'float32', 4);
  const selectionGraph = new GPUCommandGraph<void>(device, {id: 'stopovers-selection'});
  addKernelPass(selectionGraph, {
    id: 'stop-selection',
    invocationCount: STOP_CAPACITY,
    bindings: [
      {
        name: 'stopCount',
        view: view(selectionGraph, 'stop-count', stopCount, 'uint32', 1),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'stopIds',
        view: view(selectionGraph, 'stop-ids', stopIds, 'uint32', STOP_CAPACITY),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'startRows',
        view: view(selectionGraph, 'stop-start-rows', stopStartRows, 'uint32', STOP_CAPACITY),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'durations',
        view: view(selectionGraph, 'stop-durations', stopDurations, 'float32', STOP_CAPACITY),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'centroids',
        view: view(selectionGraph, 'stop-centroids', stopCentroids, 'float32x2', STOP_CAPACITY),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'lngLatIn',
        view: view(selectionGraph, 'stop-lng-lat', stopLngLat, 'float32x2', STOP_CAPACITY),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'times',
        view: view(selectionGraph, 'timestamps', timestampsBuffer, 'float32', vertexCount),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'trackSpecies',
        view: view(selectionGraph, 'species', speciesBuffer, 'uint32', trackCount),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'rules',
        view: selectionParameters.importToGraph(selectionGraph),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'meters',
        view: view(selectionGraph, 'selected-meters', selectedMeters, 'float32x2', STOP_CAPACITY),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'lngLatOut',
        view: view(selectionGraph, 'selected-lng-lat', selectedLngLat, 'float32x2', STOP_CAPACITY),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  var accepted = false;
  if (index < stopCount[stopCountOffset]) {
    let track = stopIds[stopIdsOffset + index];
    let startTime = times[timesOffset + startRows[startRowsOffset + index]];
    let speciesBit = 1u << trackSpecies[trackSpeciesOffset + track];
    accepted = (u32(rules[rulesOffset]) & speciesBit) != 0u
      && durations[durationsOffset + index] <= rules[rulesOffset + 1u]
      && startTime >= rules[rulesOffset + 2u]
      && startTime <= rules[rulesOffset + 3u];
  }
  let meter = vec2<f32>(centroids[centroidsOffset + index * 2u], centroids[centroidsOffset + index * 2u + 1u]);
  let degree = vec2<f32>(lngLatIn[lngLatInOffset + index * 2u], lngLatIn[lngLatInOffset + index * 2u + 1u]);
  let hiddenMeter = select(vec2<f32>(nan), meter, accepted);
  let hiddenDegree = select(vec2<f32>(nan), degree, accepted);
  meters[metersOffset + index * 2u] = hiddenMeter.x;
  meters[metersOffset + index * 2u + 1u] = hiddenMeter.y;
  lngLatOut[lngLatOutOffset + index * 2u] = hiddenDegree.x;
  lngLatOut[lngLatOutOffset + index * 2u + 1u] = hiddenDegree.y;`
  });
  const selectionCompiled = resources.track(selectionGraph.compile());

  // ---- Clustering graph --------------------------------------------------------------------------
  const labels = resources.createBuffer('labels', STOP_CAPACITY * 4);
  const clusterCountBuffer = resources.createBuffer('cluster-count', 4);
  const clusterIds = resources.createBuffer('cluster-ids', CLUSTER_CAPACITY * 4);
  const clusterListCount = resources.createBuffer('cluster-list-count', 4);
  const clusterOverflow = resources.createBuffer('cluster-overflow', 4);
  const clusterSizes = resources.createBuffer('cluster-sizes', CLUSTER_CAPACITY * 4);
  const clusterCentroids = resources.createBuffer('cluster-centroids', CLUSTER_CAPACITY * 8);
  const clusterParameters = resources.createParameterBuffer(
    'cluster-parameters',
    'float32',
    GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH
  );
  const clusterGraph = new GPUCommandGraph<void>(device, {id: 'stopovers-clusters'});
  clusterGraph.add(
    new GPUSpatialClustering({
      id: 'sites',
      positions: view(clusterGraph, 'selected-meters', selectedMeters, 'float32x2', STOP_CAPACITY),
      parameters: clusterParameters.importToGraph(clusterGraph),
      gridSize: [192, 192],
      labels: view(clusterGraph, 'labels', labels, 'uint32', STOP_CAPACITY),
      clusterCount: view(clusterGraph, 'cluster-count', clusterCountBuffer, 'uint32', 1),
      clusters: {
        ids: view(clusterGraph, 'cluster-ids', clusterIds, 'uint32', CLUSTER_CAPACITY),
        count: view(clusterGraph, 'cluster-list-count', clusterListCount, 'uint32', 1),
        overflow: view(clusterGraph, 'cluster-overflow', clusterOverflow, 'uint32', 1)
      },
      clusterSizes: view(clusterGraph, 'cluster-sizes', clusterSizes, 'uint32', CLUSTER_CAPACITY),
      clusterCentroids: view(
        clusterGraph,
        'cluster-centroids',
        clusterCentroids,
        'float32x2',
        CLUSTER_CAPACITY
      )
    })
  );
  const clusterCompiled = resources.track(clusterGraph.compile());

  // Planar extent of the data: stops outside the bounds would be noise.
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < tracks.positions.length; index += 2) {
    minX = Math.min(minX, tracks.positions[index]);
    maxX = Math.max(maxX, tracks.positions[index]);
    minY = Math.min(minY, tracks.positions[index + 1]);
    maxY = Math.max(maxY, tracks.positions[index + 1]);
  }
  const bounds = [minX - 1e5, minY - 1e5, maxX + 1e5, maxY + 1e5] as const;

  // ---- Site drawing buffers (CPU-ranked) --------------------------------------------------------
  const siteLngLat = resources.createBuffer('site-lng-lat', SITES_DRAWN * 8);
  const siteValues = resources.createBuffer('site-values', SITES_DRAWN * 4);
  const topSiteLngLat = resources.createBuffer('top-site-lng-lat', 3 * 8);
  const selectedSiteLngLat = resources.createBuffer('selected-site-lng-lat', 8);

  // ---- State ------------------------------------------------------------------------------------
  let destroyed = false;
  let metricsDirty = true;
  let selectionDirty = true;
  let clusterDirty = true;
  let settleStale = true;
  let lastChange = performance.now();
  let sites: Site[] = [];
  let selectedSite = -1;
  let snapshot: {
    stopCount: number;
    stopTotal: number;
    stopOverflow: boolean;
    clusterCount: number;
    clusterOverflow: boolean;
    ids: Uint32Array;
    durations: Float32Array;
    startRows: Uint32Array;
    labels: Uint32Array;
    lngLat: Float32Array;
    centroids: Float32Array;
    trackStopCounts: Uint32Array;
    trackLengths: Float32Array;
    maxima: Float32Array;
  } | null = null;

  const markChanged = () => {
    lastChange = performance.now();
    settleStale = true;
  };

  function writeMetricsParameters(): void {
    metricsParameters.write(
      getGPUTrajectoryMetricsParameterValues({
        stopSpeedThreshold: ctx.options.stopSpeed,
        stopMinimumDuration: ctx.options.minStayHours * 3600
      })
    );
    metricsDirty = true;
    selectionDirty = true;
    clusterDirty = true;
    markChanged();
  }

  function writeSelectionParameters(): void {
    const options = ctx.options;
    selectionParameters.write(
      Float32Array.of(
        SPECIES_MASKS[options.species],
        options.maxStayDays * SECONDS_PER_DAY,
        options.dayRange[0] * SECONDS_PER_DAY,
        options.dayRange[1] * SECONDS_PER_DAY
      )
    );
    selectionDirty = true;
    clusterDirty = true;
    markChanged();
  }

  function writeClusterParameters(): void {
    clusterParameters.write(
      getGPUSpatialClusteringParameterValues({
        bounds,
        epsilon: ctx.options.epsilonKm * 1000,
        minimumPoints: ctx.options.minimumStops
      })
    );
    clusterDirty = true;
    markChanged();
  }

  const median = (values: number[]) =>
    values.length ? [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] : NaN;

  // ---- Summary ----------------------------------------------------------------------------------
  function summarize(): void {
    if (!snapshot) return;
    const options = ctx.options;
    const listed = Math.min(snapshot.stopCount, STOP_CAPACITY);
    // Every stop the metrics found, for the stay histogram.
    const stayDays = Float32Array.from(
      snapshot.durations.subarray(0, listed),
      seconds => seconds / SECONDS_PER_DAY
    );
    ctx.setChart(
      'stayChart',
      listed
        ? histogramChart(binValues(stayDays, 0, 90, 45), 0, 90, {
            xLabel: 'length of a dwell (days, 90 and over in the last bin)',
            yLabel: 'dwells',
            color: 3,
            markers: [{x: Math.min(89, options.maxStayDays), label: 'longest stopover'}],
            formatX: value => `${Math.round(value)}`,
            description:
              'Histogram of every dwell found at the current speed threshold and minimum stay. Short dwells are stopovers; the long tail is home ranges (breeding and wintering). The rule marks the longest stay counted as a stopover.'
          })
        : null
    );
    let kept = 0;
    const weekStarts: number[][] = [
      new Array(53).fill(0),
      new Array(53).fill(0),
      new Array(53).fill(0)
    ];
    const perCluster = new Map<number, number[]>();
    for (let stop = 0; stop < listed; stop++) {
      if (!Number.isFinite(snapshot.lngLat[stop * 2])) continue;
      kept++;
      const track = snapshot.ids[stop];
      const startDay = tracks.timestamps[snapshot.startRows[stop]] / SECONDS_PER_DAY;
      weekStarts[tracks.species[track]][Math.min(52, Math.floor(startDay / 7))]++;
      const label = snapshot.labels[stop];
      if (label !== NOISE && label < CLUSTER_CAPACITY) {
        const members = perCluster.get(label) ?? [];
        members.push(stop);
        perCluster.set(label, members);
      }
    }
    // Stopovers per week of the year, one line per species.
    const weekCenters = Array.from({length: 53}, (_, week) => week * 7 + 3.5);
    const speciesShown = options.species === 'all' ? [0, 1, 2] : [SPECIES_INDEX[options.species]];
    ctx.setChart(
      'seasonChart',
      kept
        ? {
            kind: 'line',
            series: speciesShown.map(species => ({
              label: MIGRATION_SPECIES_LABELS[species],
              x: weekCenters,
              y: weekStarts[species],
              color: species
            })),
            xDomain: [0, 371],
            xLabel: 'week of the year',
            yLabel: 'stopovers begun',
            height: 130,
            formatX: value => formatYearDay(Math.min(365, value)),
            formatY: value => `${Math.round(value)}`,
            description:
              'Stopovers that began in each week of the folded year, per species, after the stay, season and species filters.'
          }
        : null
    );

    // Sites: clusters ranked by the chosen measure.
    const all: Site[] = [];
    for (const [cluster, members] of perCluster) {
      const birds = new Set<number>();
      const speciesCounts = [0, 0, 0];
      const stays: number[] = [];
      const starts: number[] = [];
      let dwell = 0;
      for (const stop of members) {
        const track = snapshot.ids[stop];
        birds.add(tracks.individual[track]);
        speciesCounts[tracks.species[track]]++;
        stays.push(snapshot.durations[stop] / SECONDS_PER_DAY);
        starts.push(tracks.timestamps[snapshot.startRows[stop]] / SECONDS_PER_DAY);
        dwell += snapshot.durations[stop] / SECONDS_PER_DAY;
      }
      const [lng, lat] = tracks.unproject(
        snapshot.centroids[cluster * 2],
        snapshot.centroids[cluster * 2 + 1]
      );
      all.push({
        rank: 0,
        cluster,
        lng,
        lat,
        name: nameSite(lng, lat),
        stops: members.length,
        birds: birds.size,
        dwellDays: dwell,
        medianStayDays: median(stays),
        medianStartDay: median(starts),
        species: speciesCounts
      });
    }
    const measure = (site: Site) =>
      options.rankBy === 'dwell'
        ? site.dwellDays
        : options.rankBy === 'birds'
          ? site.birds
          : site.stops;
    all.sort((a, b) => measure(b) - measure(a) || b.dwellDays - a.dwellDays);
    all.forEach((site, index) => {
      site.rank = index + 1;
    });
    sites = all;
    if (selectedSite >= sites.length) selectedSite = -1;

    const noise = kept - sites.reduce((sum, site) => sum + site.stops, 0);
    ctx.setReadout(
      'stops',
      `${formatCount(snapshot.stopTotal)} dwells found${snapshot.stopOverflow ? ' (list truncated)' : ''}, ${formatCount(kept)} kept as stopovers`
    );
    ctx.setReadout(
      'sites',
      `${formatCount(sites.length)} sites${snapshot.clusterOverflow ? ' (capacity reached)' : ''}, ${formatCount(noise)} isolated stops (noise)`
    );
    ctx.setReadout(
      'siteList',
      sites.length
        ? sites
            .slice(0, SITES_LISTED)
            .map(
              site =>
                `${String(site.rank).padStart(2)}. ${site.name}\n    ${site.birds} birds, ${site.stops} stops, ${site.dwellDays.toFixed(0)} bird-days, median ${formatDuration(site.medianStayDays * SECONDS_PER_DAY)}, typically ${formatYearDay(site.medianStartDay)}`
            )
            .join('\n')
        : 'no site at these settings'
    );
    let birdsWithStops = 0;
    for (let track = 0; track < trackCount; track++)
      if (snapshot.trackStopCounts[track] > 0) birdsWithStops++;
    ctx.setReadout(
      'tracksWithStops',
      `${birdsWithStops} of ${trackCount} tracks have at least one dwell`
    );
    const top = sites.slice(0, SITES_LISTED);
    ctx.setChart(
      'siteChart',
      top.length
        ? {
            kind: 'bars',
            values: top.map(measure),
            labels: top.map(site => `${site.rank}`),
            highlight: selectedSite >= 0 && selectedSite < top.length ? [selectedSite] : [0],
            height: 110,
            yLabel:
              options.rankBy === 'dwell'
                ? 'bird-days'
                : options.rankBy === 'birds'
                  ? 'birds'
                  : 'stops',
            formatY: value => `${Math.round(value)}`,
            description:
              'The ten highest-ranked stopover sites by the chosen measure; bar numbers are the ranks in the list.'
          }
        : null
    );
    writeSiteBuffers();
    describeSelectedSite();
    ctx.requestLayers();
  }

  function writeSiteBuffers(): void {
    const options = ctx.options;
    const positions = new Float32Array(SITES_DRAWN * 2).fill(Number.NaN);
    const values = new Float32Array(SITES_DRAWN).fill(Number.NaN);
    const top = new Float32Array(6).fill(Number.NaN);
    const maxMeasure = sites.length
      ? options.rankBy === 'dwell'
        ? sites[0].dwellDays
        : options.rankBy === 'birds'
          ? Math.max(...sites.map(site => site.birds))
          : Math.max(...sites.map(site => site.stops))
      : 1;
    sites.slice(0, SITES_DRAWN).forEach((site, index) => {
      positions[index * 2] = site.lng;
      positions[index * 2 + 1] = site.lat;
      const measure =
        options.rankBy === 'dwell'
          ? site.dwellDays
          : options.rankBy === 'birds'
            ? site.birds
            : site.stops;
      values[index] = measure / Math.max(1, maxMeasure);
      if (index < 3) {
        top[index * 2] = site.lng;
        top[index * 2 + 1] = site.lat;
      }
    });
    siteLngLat.write(positions);
    siteValues.write(values);
    topSiteLngLat.write(top);
    const selected = new Float32Array(2).fill(Number.NaN);
    if (selectedSite >= 0 && sites[selectedSite]) {
      selected[0] = sites[selectedSite].lng;
      selected[1] = sites[selectedSite].lat;
    }
    selectedSiteLngLat.write(selected);
  }

  function describeSelectedSite(): void {
    const site = selectedSite >= 0 ? sites[selectedSite] : null;
    if (!site) {
      ctx.setReadout('selectedSite', 'click a site');
      return;
    }
    const species = site.species
      .map((count, index) =>
        count ? `${count} ${MIGRATION_SPECIES_LABELS[index].toLowerCase()}` : ''
      )
      .filter(Boolean)
      .join(', ');
    ctx.setReadout(
      'selectedSite',
      `#${site.rank} ${site.name}: ${site.birds} birds, ${site.stops} stops (${species}), ${site.dwellDays.toFixed(0)} bird-days, median stay ${formatDuration(site.medianStayDays * SECONDS_PER_DAY)}, typically from ${formatYearDay(site.medianStartDay)}`
    );
  }

  // ---- Reader -----------------------------------------------------------------------------------
  const reader = new SummaryReader(
    resources,
    'stopovers-summary',
    [
      {buffer: stopCount, size: 4},
      {buffer: stopOverflow, size: 4},
      {buffer: stopTotal, size: 4},
      {buffer: clusterCountBuffer, size: 4},
      {buffer: clusterOverflow, size: 4},
      {buffer: stopIds, size: STOP_CAPACITY * 4},
      {buffer: stopDurations, size: STOP_CAPACITY * 4},
      {buffer: stopStartRows, size: STOP_CAPACITY * 4},
      {buffer: labels, size: STOP_CAPACITY * 4},
      {buffer: selectedLngLat, size: STOP_CAPACITY * 8},
      {buffer: clusterCentroids, size: CLUSTER_CAPACITY * 8},
      {buffer: trackStopCounts, size: trackCount * 4},
      {buffer: trackLengths, size: trackCount * 4},
      {buffer: maximumSpeeds, size: trackCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      let word = 5;
      const take = (length: number) => {
        const start = word;
        word += length;
        return start;
      };
      const idsStart = take(STOP_CAPACITY);
      const durationsStart = take(STOP_CAPACITY);
      const startRowsStart = take(STOP_CAPACITY);
      const labelsStart = take(STOP_CAPACITY);
      const lngLatStart = take(STOP_CAPACITY * 2);
      const centroidsStart = take(CLUSTER_CAPACITY * 2);
      const stopCountsStart = take(trackCount);
      const lengthsStart = take(trackCount);
      const maximaStart = take(trackCount);
      snapshot = {
        stopCount: words[0],
        stopOverflow: words[1] !== 0,
        stopTotal: words[2],
        clusterCount: words[3],
        clusterOverflow: words[4] !== 0,
        ids: words.slice(idsStart, idsStart + STOP_CAPACITY),
        durations: floats.slice(durationsStart, durationsStart + STOP_CAPACITY),
        startRows: words.slice(startRowsStart, startRowsStart + STOP_CAPACITY),
        labels: words.slice(labelsStart, labelsStart + STOP_CAPACITY),
        lngLat: floats.slice(lngLatStart, lngLatStart + STOP_CAPACITY * 2),
        centroids: floats.slice(centroidsStart, centroidsStart + CLUSTER_CAPACITY * 2),
        trackStopCounts: words.slice(stopCountsStart, stopCountsStart + trackCount),
        trackLengths: floats.slice(lengthsStart, lengthsStart + trackCount),
        maxima: floats.slice(maximaStart, maximaStart + trackCount)
      };
      summarize();
    }
  );

  writeMetricsParameters();
  writeSelectionParameters();
  writeClusterParameters();
  ctx.setReadout(
    'tracks',
    `${trackCount} animal-years, ${formatCount(vertexCount)} fixes, one every two hours`
  );
  ctx.setReadout('selectedSite', 'click a site');

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [metricsCompiled, selectionCompiled, clusterCompiled],

    setOption(id) {
      switch (id) {
        case 'stopSpeed':
        case 'minStayHours':
          writeMetricsParameters();
          break;
        case 'species':
        case 'maxStayDays':
        case 'dayRange':
          writeSelectionParameters();
          break;
        case 'epsilonKm':
        case 'minimumStops':
          writeClusterParameters();
          break;
        case 'rankBy':
          if (snapshot) summarize();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      if (metricsDirty) {
        metricsCompiled.encode(commandEncoder, {parameters: undefined});
        metricsDirty = false;
      }
      if (selectionDirty) {
        selectionCompiled.encode(commandEncoder, {parameters: undefined});
        selectionDirty = false;
      }
      if (clusterDirty) {
        clusterCompiled.encode(commandEncoder, {parameters: undefined});
        clusterDirty = false;
      }
      if (settleStale && performance.now() - lastChange > SETTLE_MILLISECONDS) {
        reader.markStale();
        settleStale = false;
      }
      reader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.showTracks) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'stopover-tracks',
            ...drawProps,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            values: speciesBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentTracksBuffer,
            colormap: 'category',
            palette: MIGRATION_SPECIES_COLORS,
            widthPixels: 1,
            opacity: options.trackOpacity
          })
        );
      }
      let stopColors: Record<string, unknown>;
      switch (options.colorStops) {
        case 'cluster':
          stopColors = {
            values: labels,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [
              [78, 201, 255, 255],
              [255, 148, 72, 255],
              [189, 122, 255, 255],
              [87, 235, 168, 255],
              [255, 105, 168, 255],
              [245, 220, 87, 255],
              [107, 158, 255, 255],
              [255, 92, 92, 255]
            ],
            noDataColor: dark ? [170, 176, 190, 110] : [90, 96, 110, 120]
          };
          break;
        case 'species':
          stopColors = {
            values: speciesBuffer,
            valueFormat: 'uint32',
            valueIndices: stopIds,
            colormap: 'category',
            palette: MIGRATION_SPECIES_COLORS
          };
          break;
        default:
          stopColors = {
            values: stopDurations,
            valueFormat: 'float32',
            colormap: options.ramp,
            valueScale: 1 / SECONDS_PER_DAY,
            valueRange: [0, STAY_RAMP_DAYS]
          };
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'stopover-stops',
          ...drawProps,
          positions: selectedLngLat,
          instanceCount: STOP_CAPACITY,
          radiusPixels: 3.2,
          opacity: 0.9,
          ...stopColors
        })
      );
      if (options.showSites) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'stopover-sites',
            ...drawProps,
            positions: siteLngLat,
            instanceCount: SITES_DRAWN,
            values: siteValues,
            valueFormat: 'float32',
            colormap: options.ramp,
            valueRange: [0, 1],
            radiusPixels: 9,
            opacity: 0.95
          }),
          new SpatialAnalysisPointLayer({
            id: 'stopover-top-sites',
            ...drawProps,
            positions: topSiteLngLat,
            instanceCount: 3,
            radiusPixels: 15,
            color: dark ? [255, 255, 255, 120] : [20, 24, 32, 110]
          }),
          new SpatialAnalysisPointLayer({
            id: 'stopover-selected-site',
            ...drawProps,
            positions: selectedSiteLngLat,
            instanceCount: 1,
            radiusPixels: 20,
            color: dark ? [255, 255, 255, 220] : [20, 24, 32, 220]
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      const site = pickSite(event.pixel);
      if (site < 0) return null;
      const entry = sites[site];
      return `#${entry.rank} ${entry.name}: ${entry.birds} birds, ${entry.dwellDays.toFixed(0)} bird-days`;
    },

    onClick(event) {
      const site = pickSite(event.pixel);
      selectedSite = site < 0 || site === selectedSite ? -1 : site;
      writeSiteBuffers();
      describeSelectedSite();
      summarizeChartHighlight();
      ctx.requestLayers();
      return site >= 0;
    },

    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
    }
  };

  /** Rank number of the site bar to emphasise after a selection change. */
  function summarizeChartHighlight(): void {
    if (snapshot) summarize();
  }

  /** Nearest drawn site within 18 CSS pixels of the pointer, as an index into `sites`, or -1. */
  function pickSite(pixel: readonly [number, number]): number {
    const viewport = ctx.getViewport();
    if (!viewport || !ctx.options.showSites) return -1;
    let best = -1;
    let bestDistance = 18 * 18;
    sites.slice(0, SITES_DRAWN).forEach((site, index) => {
      const [x, y] = viewport.project([site.lng, site.lat]);
      const squared = (x - pixel[0]) ** 2 + (y - pixel[1]) ** 2;
      if (squared < bestDistance) {
        bestDistance = squared;
        best = index;
      }
    });
    return best;
  }
}
