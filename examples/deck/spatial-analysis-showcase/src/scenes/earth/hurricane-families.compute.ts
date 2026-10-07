// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUKMeans,
  GPUTrackSimilarity,
  GPUTrajectoryResample
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {findNearestTrack} from '../movement/b12-tracks';
import {binValues, histogramChart} from '../movement/f-chart-helpers';
import {
  ATLANTIC_CENTER,
  describeStorm,
  findLongestStorm,
  findStorm,
  getStormLabel,
  HURRICANE_CATEGORY_COLORS,
  HURRICANE_FAMILY_COLORS,
  loadHurricaneTracks
} from './hurricane-data';

/** Option state of the hurricane families scene. */
export type HurricaneFamiliesOptions = {
  routeSamples: '32' | '48' | '64';
  routeSpacing: 'arc-length' | 'time';
  distanceMeasure: 'frechet' | 'hausdorff';
  familyCount: number;
  initialization: 'kmeans++' | 'first-valid';
  seed: number;
  iterations: number;
  colorBy: 'family' | 'similarity' | 'peak' | 'season' | 'plain';
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  similarityRangeKm: number;
  trackOpacity: number;
  showRoutes: boolean;
  showFamilyMeans: boolean;
};

const MAXIMUM_FAMILIES = 8;
const NO_STORM = 0xffffffff;
const EARTH_RADIUS_METERS = 6371008.8;
const PICK_RADIUS_METERS = 450000;
const MDS_ITERATIONS = 90;

type RouteVariant = {
  key: string;
  sampleCount: number;
  routes: Buffer;
  routeOffsets: Buffer;
  routeDegrees: Buffer;
  routeSegments: Buffer;
  resample: CompiledGPUCommandGraph<void>;
  similarity: CompiledGPUCommandGraph<void>;
  encoded: boolean;
};

type KMeansVariant = {
  key: string;
  compiled: CompiledGPUCommandGraph<void>;
  encoded: boolean;
};

/**
 * Hurricane families: every Atlantic storm since 1980 is resampled to a fixed number of points
 * (`GPUTrajectoryResample`), all 273,000 pairs are scored with the discrete Frechet and Hausdorff
 * distances (`GPUTrackSimilarity`), the distance matrix is embedded in two dimensions on the CPU
 * (classical multidimensional scaling) and `GPUKMeans` clusters the embedding into families.
 *
 * Analysis runs in azimuthal-equidistant meters about 28 N, 60 W so distances between tracks are
 * true over the whole basin; drawing uses the longitude and latitude of the same vertices.
 */
export async function createHurricaneFamilies(
  ctx: SceneContext<HurricaneFamiliesOptions>
): Promise<SceneInstance<HurricaneFamiliesOptions>> {
  const storms = loadHurricaneTracks(ctx.datasets.get('ibtracs-north-atlantic'), {
    projection: 'azimuthal',
    timeBase: 'storm'
  });
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = storms;
  const resources = new SpatialAnalysisResources(device, 'hurricane-families');
  const drawProps = {coordinateSystem: COORDINATE_SYSTEM.LNGLAT} as const;

  // ---- Static inputs --------------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', storms.positions);
  const timestampsBuffer = resources.createBuffer('timestamps', storms.timestamps);
  const offsetsBuffer = resources.createBuffer('offsets', storms.offsets);
  const segmentsBuffer = resources.createBuffer('segments', storms.segments);
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', storms.segmentTracks);
  const maxCategoryBuffer = resources.createBuffer('max-category', storms.maxCategoryWords);
  const seasonBuffer = resources.createBuffer('season', Float32Array.from(storms.season));
  const familyBuffer = resources.createBuffer('family', new Uint32Array(trackCount));
  const distanceBuffer = resources.createBuffer(
    'selected-distance',
    new Float32Array(trackCount).fill(Number.NaN)
  );
  const selectedSegments = resources.createBuffer(
    'selected-segments',
    new Float32Array(storms.longestTrack * 4).fill(Number.NaN)
  );
  const familyIndexBuffer = resources.createBuffer(
    'family-index',
    Uint32Array.from({length: MAXIMUM_FAMILIES}, (_, family) => family)
  );
  const meanSegments = resources.createBuffer(
    'family-means',
    new Float32Array(MAXIMUM_FAMILIES * 63 * 4).fill(Number.NaN)
  );

  // ---- Pairs (all unordered storm pairs) -------------------------------------------------------
  const pairCount = (trackCount * (trackCount - 1)) / 2;
  const pairA = new Uint32Array(pairCount);
  const pairB = new Uint32Array(pairCount);
  let pairIndex = 0;
  for (let a = 0; a < trackCount; a++) {
    for (let b = a + 1; b < trackCount; b++) {
      pairA[pairIndex] = a;
      pairB[pairIndex] = b;
      pairIndex++;
    }
  }
  const pairABuffer = resources.createBuffer('pair-a', pairA);
  const pairBBuffer = resources.createBuffer('pair-b', pairB);
  const pairHausdorff = resources.createBuffer('pair-hausdorff', pairCount * 4);
  const pairFrechet = resources.createBuffer('pair-frechet', pairCount * 4);
  const pairStatus = resources.createBuffer('pair-status', pairCount * 4);

  // ---- Embedding and k-means buffers -----------------------------------------------------------
  const embeddingBuffer = resources.createBuffer('embedding', trackCount * 8);
  const labelsBuffer = resources.createBuffer('labels', trackCount * 4);
  const centersBuffer = resources.createBuffer('centers', MAXIMUM_FAMILIES * 8);
  const sizesBuffer = resources.createBuffer('sizes', MAXIMUM_FAMILIES * 4);
  const convergenceBuffer = resources.createBuffer('convergence', 8);
  const squaredDistancesBuffer = resources.createBuffer('squared-distances', trackCount * 4);

  // ---- State -----------------------------------------------------------------------------------
  let destroyed = false;
  let routeVariant: RouteVariant | null = null;
  let kmeansVariant: KMeansVariant | null = null;
  let similarityDirty = true;
  let kmeansDirty = false;
  let pairSnapshot: {hausdorff: Float32Array; frechet: Float32Array} | null = null;
  let matrix: Float32Array | null = null;
  let embedding: Float32Array | null = null;
  let embeddingVariance = 0;
  let embeddingTotal = 0;
  let routesSnapshot: Float32Array | null = null;
  let families = new Uint32Array(trackCount);
  let familySizes: number[] = [];
  let selectedStorm = NO_STORM;
  const katrina = findStorm(storms, 'katrina', 2005);
  selectedStorm = katrina >= 0 ? katrina : findLongestStorm(storms);

  // ---- Route variants (compile-time sample count and spacing) ----------------------------------
  function buildRoutes(sampleCount: number, spacing: 'arc-length' | 'time'): RouteVariant {
    const key = `${sampleCount}-${spacing}`;
    const routes = resources.createBuffer(`routes-${key}`, trackCount * sampleCount * 8);
    const routeDegrees = resources.createBuffer(
      `route-degrees-${key}`,
      trackCount * sampleCount * 8
    );
    const routeSegments = resources.createBuffer(
      `route-segments-${key}`,
      trackCount * (sampleCount - 1) * 16
    );
    const routeOffsetValues = new Uint32Array(trackCount + 1);
    for (let track = 0; track <= trackCount; track++)
      routeOffsetValues[track] = track * sampleCount;
    const routeOffsets = resources.createBuffer(`route-offsets-${key}`, routeOffsetValues);

    const resampleGraph = new GPUCommandGraph<void>(device, {id: `hurricane-routes-${key}`});
    const routesView = importGraphBuffer(
      resampleGraph,
      'routes',
      routes,
      'float32x2',
      trackCount * sampleCount
    );
    resampleGraph.add(
      new GPUTrajectoryResample({
        id: 'routes',
        positions: importGraphBuffer(
          resampleGraph,
          'positions',
          positionsBuffer,
          'float32x2',
          vertexCount
        ),
        timestamps: importGraphBuffer(
          resampleGraph,
          'timestamps',
          timestampsBuffer,
          'float32',
          vertexCount
        ),
        trackOffsets: importGraphBuffer(
          resampleGraph,
          'offsets',
          offsetsBuffer,
          'uint32',
          trackCount + 1
        ),
        sampleCount,
        spacing,
        samples: routesView
      })
    );
    // Back to longitude and latitude so the samples can be drawn: the inverse of the
    // azimuthal-equidistant projection the analysis runs in.
    const degreesView = importGraphBuffer(
      resampleGraph,
      'route-degrees',
      routeDegrees,
      'float32x2',
      trackCount * sampleCount
    );
    const phi0 = (ATLANTIC_CENTER[1] * Math.PI) / 180;
    const literal = (value: number) =>
      Math.fround(value)
        .toString()
        .replace(/^(-?\d+)$/, '$1.0');
    addKernelPass(resampleGraph, {
      id: 'route-degrees',
      invocationCount: trackCount * sampleCount,
      declarations: `const EARTH_RADIUS: f32 = ${literal(EARTH_RADIUS_METERS)};
const CENTER_LONGITUDE: f32 = ${literal((ATLANTIC_CENTER[0] * Math.PI) / 180)};
const CENTER_LATITUDE: f32 = ${literal(phi0)};
const SIN_CENTER: f32 = ${literal(Math.sin(phi0))};
const COS_CENTER: f32 = ${literal(Math.cos(phi0))};`,
      bindings: [
        {name: 'samples', view: routesView, type: 'f32', access: 'read'},
        {name: 'lngLatDegrees', view: degreesView, type: 'f32', access: 'read_write'}
      ],
      body: `let x = samples[samplesOffset + index * 2u];
  let y = samples[samplesOffset + index * 2u + 1u];
  let rho = sqrt(x * x + y * y);
  var longitude = CENTER_LONGITUDE;
  var latitude = CENTER_LATITUDE;
  if (rho > 1.0) {
    let c = rho / EARTH_RADIUS;
    let sinC = sin(c);
    let cosC = cos(c);
    latitude = asin(clamp(cosC * SIN_CENTER + (y * sinC / rho) * COS_CENTER, -1.0, 1.0));
    longitude = CENTER_LONGITUDE + atan2(x * sinC, rho * COS_CENTER * cosC - y * SIN_CENTER * sinC);
  }
  lngLatDegrees[lngLatDegreesOffset + index * 2u] = longitude * 57.29577951;
  lngLatDegrees[lngLatDegreesOffset + index * 2u + 1u] = latitude * 57.29577951;`
    });
    addKernelPass(resampleGraph, {
      id: 'route-segments',
      invocationCount: trackCount * (sampleCount - 1),
      declarations: `const SAMPLES: u32 = ${sampleCount}u;`,
      bindings: [
        {name: 'lngLatDegrees', view: degreesView, type: 'f32', access: 'read'},
        {
          name: 'segments',
          view: importGraphBuffer(
            resampleGraph,
            'route-segments',
            routeSegments,
            'float32',
            trackCount * (sampleCount - 1) * 4
          ),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `let track = index / (SAMPLES - 1u);
  let k = index % (SAMPLES - 1u);
  let row = (track * SAMPLES + k) * 2u;
  segments[segmentsOffset + index * 4u] = lngLatDegrees[lngLatDegreesOffset + row];
  segments[segmentsOffset + index * 4u + 1u] = lngLatDegrees[lngLatDegreesOffset + row + 1u];
  segments[segmentsOffset + index * 4u + 2u] = lngLatDegrees[lngLatDegreesOffset + row + 2u];
  segments[segmentsOffset + index * 4u + 3u] = lngLatDegrees[lngLatDegreesOffset + row + 3u];`
    });

    const similarityGraph = new GPUCommandGraph<void>(device, {id: `hurricane-similarity-${key}`});
    similarityGraph.add(
      new GPUTrackSimilarity({
        id: 'route-similarity',
        positionsA: importGraphBuffer(
          similarityGraph,
          'routes',
          routes,
          'float32x2',
          trackCount * sampleCount
        ),
        offsetsA: importGraphBuffer(
          similarityGraph,
          'route-offsets',
          routeOffsets,
          'uint32',
          trackCount + 1
        ),
        pairA: importGraphBuffer(similarityGraph, 'pair-a', pairABuffer, 'uint32', pairCount),
        pairB: importGraphBuffer(similarityGraph, 'pair-b', pairBBuffer, 'uint32', pairCount),
        hausdorff: importGraphBuffer(
          similarityGraph,
          'pair-hausdorff',
          pairHausdorff,
          'float32',
          pairCount
        ),
        frechet: importGraphBuffer(
          similarityGraph,
          'pair-frechet',
          pairFrechet,
          'float32',
          pairCount
        ),
        status: importGraphBuffer(similarityGraph, 'pair-status', pairStatus, 'uint32', pairCount),
        maxFrechetVertices: sampleCount
      })
    );
    return {
      key,
      sampleCount,
      routes,
      routeOffsets,
      routeDegrees,
      routeSegments,
      resample: resources.track(resampleGraph.compile()),
      similarity: resources.track(similarityGraph.compile()),
      encoded: false
    };
  }

  function replaceRoutes(sampleCount: number, spacing: 'arc-length' | 'time'): void {
    const previous = routeVariant;
    routeVariant = buildRoutes(sampleCount, spacing);
    if (previous) {
      resources.release(previous.similarity);
      resources.release(previous.resample);
      resources.release(previous.routes);
      resources.release(previous.routeOffsets);
      resources.release(previous.routeDegrees);
      resources.release(previous.routeSegments);
    }
    similarityDirty = true;
  }

  // ---- k-means variant (compile-time k, iterations, initialization, seed) ----------------------
  function buildKMeans(options: HurricaneFamiliesOptions): KMeansVariant {
    const k = options.familyCount;
    const key = `${k}-${options.iterations}-${options.initialization}-${options.seed}`;
    const graph = new GPUCommandGraph<void>(device, {id: `hurricane-kmeans-${key}`});
    graph.add(
      new GPUKMeans({
        id: 'families',
        positions: importGraphBuffer(graph, 'embedding', embeddingBuffer, 'float32x2', trackCount),
        k,
        iterations: options.iterations,
        initialization: options.initialization,
        seed: options.seed,
        labels: importGraphBuffer(graph, 'labels', labelsBuffer, 'uint32', trackCount),
        centers: importGraphBuffer(graph, 'centers', centersBuffer, 'float32x2', k),
        sizes: importGraphBuffer(graph, 'sizes', sizesBuffer, 'uint32', k),
        convergence: importGraphBuffer(graph, 'convergence', convergenceBuffer, 'uint32', 2),
        squaredDistances: importGraphBuffer(
          graph,
          'squared-distances',
          squaredDistancesBuffer,
          'float32',
          trackCount
        )
      })
    );
    return {key, compiled: resources.track(graph.compile()), encoded: false};
  }

  function replaceKMeans(): void {
    const previous = kmeansVariant;
    kmeansVariant = buildKMeans(ctx.options as HurricaneFamiliesOptions);
    if (previous) resources.release(previous.compiled);
    kmeansDirty = embedding !== null;
  }

  // ---- Readbacks ---------------------------------------------------------------------------------
  const similarityReader = new SummaryReader(
    resources,
    'hurricane-similarity',
    [
      {buffer: pairHausdorff, size: pairCount * 4},
      {buffer: pairFrechet, size: pairCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      pairSnapshot = {
        hausdorff: new Float32Array(bytes, 0, pairCount).slice(),
        frechet: new Float32Array(bytes, pairCount * 4, pairCount).slice()
      };
      buildMatrixAndEmbedding();
      ctx.setReadout(
        'pairs',
        `${formatCount(trackCount)} routes, ${formatCount(pairCount)} pairs scored`
      );
    }
  );

  let routeReader: SummaryReader | null = null;
  function createRouteReader(variant: RouteVariant): SummaryReader {
    return new SummaryReader(
      resources,
      `hurricane-route-samples-${variant.key}`,
      [{buffer: variant.routes, size: trackCount * variant.sampleCount * 8}],
      bytes => {
        if (destroyed || routeVariant !== variant) return;
        routesSnapshot = new Float32Array(bytes).slice();
        updateFamilyMeans();
      }
    );
  }

  const kmeansReader = new SummaryReader(
    resources,
    'hurricane-kmeans',
    [
      {buffer: labelsBuffer, size: trackCount * 4},
      {buffer: sizesBuffer, size: MAXIMUM_FAMILIES * 4},
      {buffer: convergenceBuffer, size: 8},
      {buffer: squaredDistancesBuffer, size: trackCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const labels = words.slice(0, trackCount);
      const convergence = words.slice(
        trackCount + MAXIMUM_FAMILIES,
        trackCount + MAXIMUM_FAMILIES + 2
      );
      let inertia = 0;
      for (let track = 0; track < trackCount; track++) {
        const value = floats[trackCount + MAXIMUM_FAMILIES + 2 + track];
        if (Number.isFinite(value)) inertia += value;
      }
      assignFamilies(labels);
      const explained = embeddingTotal > 0 ? Math.max(0, 1 - inertia / embeddingTotal) : 0;
      ctx.setReadout(
        'kmeans',
        `${convergence[0]} of ${ctx.options.iterations} iterations${convergence[1] ? ', converged' : ', not converged'}; families explain ${(explained * 100).toFixed(0)}% of the embedding variance`
      );
    }
  );

  // ---- CPU post-processing -----------------------------------------------------------------------
  function buildMatrixAndEmbedding(): void {
    if (!pairSnapshot) return;
    const source =
      ctx.options.distanceMeasure === 'frechet' ? pairSnapshot.frechet : pairSnapshot.hausdorff;
    matrix = new Float32Array(trackCount * trackCount);
    for (let pair = 0; pair < pairCount; pair++) {
      matrix[pairA[pair] * trackCount + pairB[pair]] = source[pair];
      matrix[pairB[pair] * trackCount + pairA[pair]] = source[pair];
    }
    const result = classicalScaling(matrix, trackCount);
    embedding = result.coordinates;
    embeddingVariance = result.explained;
    embeddingTotal = result.totalVariance;
    embeddingBuffer.write(embedding);
    kmeansDirty = true;
    ctx.setReadout(
      'embedding',
      `${(embeddingVariance * 100).toFixed(0)}% of the distance variance in two dimensions`
    );
    writeSelectedDistances();
    updateSelection();
    updateDistanceChart();
  }

  /** Renumbers k-means labels from the most easterly mean start to the most westerly. */
  function assignFamilies(labels: Uint32Array): void {
    const k = ctx.options.familyCount;
    const startLongitude = new Float64Array(k);
    const counts = new Uint32Array(k);
    for (let track = 0; track < trackCount; track++) {
      const label = labels[track];
      if (label >= k) continue;
      startLongitude[label] += storms.lngLat[storms.offsets[track] * 2];
      counts[label]++;
    }
    const order = Array.from({length: k}, (_, label) => label)
      .filter(label => counts[label] > 0)
      .sort((a, b) => startLongitude[b] / counts[b] - startLongitude[a] / counts[a]);
    const rank = new Uint32Array(k).fill(0);
    order.forEach((label, index) => {
      rank[label] = index;
    });
    families = new Uint32Array(trackCount);
    familySizes = order.map(label => counts[label]);
    for (let track = 0; track < trackCount; track++) {
      families[track] = labels[track] < k ? rank[labels[track]] : 0;
    }
    familyBuffer.write(families);
    updateFamilyMeans();
    updateFamilyText();
    updateFamilyChart();
    updateSelection();
    ctx.requestLayers();
  }

  function updateFamilyText(): void {
    const lines: string[] = [];
    const labels: string[] = [];
    for (let family = 0; family < familySizes.length; family++) {
      let startLongitude = 0;
      let startLatitude = 0;
      let endLongitude = 0;
      let endLatitude = 0;
      let hurricanes = 0;
      let length = 0;
      let count = 0;
      for (let track = 0; track < trackCount; track++) {
        if (families[track] !== family) continue;
        const first = storms.offsets[track];
        const last = storms.offsets[track + 1] - 1;
        startLongitude += storms.lngLat[first * 2];
        startLatitude += storms.lngLat[first * 2 + 1];
        endLongitude += storms.lngLat[last * 2];
        endLatitude += storms.lngLat[last * 2 + 1];
        if (storms.maxCategory[track] >= 2) hurricanes++;
        for (let vertex = first + 1; vertex <= last; vertex++) {
          length += Math.hypot(
            storms.positions[vertex * 2] - storms.positions[vertex * 2 - 2],
            storms.positions[vertex * 2 + 1] - storms.positions[vertex * 2 - 1]
          );
        }
        count++;
      }
      if (!count) continue;
      const format = (longitude: number, latitude: number) =>
        `${Math.abs(latitude / count).toFixed(0)}N ${Math.abs(longitude / count).toFixed(0)}${longitude < 0 ? 'W' : 'E'}`;
      lines.push(
        `${family + 1}: ${count} storms, ${format(startLongitude, startLatitude)} to ${format(endLongitude, endLatitude)}, ${formatCount(length / count / 1000)} km, ${((100 * hurricanes) / count).toFixed(0)}% hurricanes`
      );
      labels.push(`Family ${family + 1} (${count} storms)`);
    }
    ctx.setReadout('families', lines.join('\n'));
    ctx.setLegendData('familyLabels', labels);
  }

  function updateFamilyChart(): void {
    const selectedFamily = selectedStorm === NO_STORM ? -1 : families[selectedStorm];
    ctx.setChart('familyChart', {
      kind: 'bars',
      values: familySizes,
      labels: familySizes.map((_, family) => `F${family + 1}`),
      highlight: selectedFamily >= 0 ? [selectedFamily] : [],
      height: 130,
      yLabel: 'storms',
      formatY: value => value.toFixed(0),
      description:
        'Number of storms in each track family. The family of the selected storm is highlighted.'
    });
  }

  function updateDistanceChart(): void {
    if (!matrix) return;
    const maximumKm = 8000;
    const values: number[] = [];
    if (selectedStorm !== NO_STORM) {
      for (let track = 0; track < trackCount; track++) {
        if (track !== selectedStorm) values.push(matrix[selectedStorm * trackCount + track] / 1000);
      }
    } else {
      for (let pair = 0; pair < pairCount; pair += 7) {
        values.push(matrix[pairA[pair] * trackCount + pairB[pair]] / 1000);
      }
    }
    ctx.setChart(
      'distanceChart',
      histogramChart(binValues(values, 0, maximumKm, 32), 0, maximumKm, {
        xLabel:
          selectedStorm === NO_STORM
            ? 'distance between two storms (km)'
            : `${ctx.options.distanceMeasure === 'frechet' ? 'Frechet' : 'Hausdorff'} distance to ${getStormLabel(storms, selectedStorm)} (km)`,
        yLabel: 'storms',
        markers: [{x: ctx.options.similarityRangeKm, label: 'color range'}],
        formatX: value => `${Math.round(value / 100) * 100}`,
        description:
          'Histogram of the route distance from the selected storm to every other storm. The marker is the end of the color range.'
      })
    );
  }

  function writeSelectedDistances(): void {
    const distances = new Float32Array(trackCount).fill(Number.NaN);
    if (matrix && selectedStorm !== NO_STORM) {
      for (let track = 0; track < trackCount; track++) {
        distances[track] = track === selectedStorm ? 0 : matrix[selectedStorm * trackCount + track];
      }
    }
    distanceBuffer.write(distances);
  }

  function writeSelectedOutline(): void {
    const outline = new Float32Array(storms.longestTrack * 4).fill(Number.NaN);
    if (selectedStorm !== NO_STORM) {
      let row = 0;
      for (
        let vertex = storms.offsets[selectedStorm];
        vertex < storms.offsets[selectedStorm + 1] - 1;
        vertex++, row++
      ) {
        outline.set(storms.lngLat.subarray(vertex * 2, vertex * 2 + 4), row * 4);
      }
    }
    selectedSegments.write(outline);
  }

  /** Mean resampled route of each family, in longitude and latitude. */
  function updateFamilyMeans(): void {
    const variant = routeVariant;
    const sampleCount = variant?.sampleCount ?? 0;
    const segments = new Float32Array(MAXIMUM_FAMILIES * 63 * 4).fill(Number.NaN);
    if (variant && routesSnapshot && familySizes.length) {
      for (let family = 0; family < familySizes.length; family++) {
        const sumX = new Float64Array(sampleCount);
        const sumY = new Float64Array(sampleCount);
        let count = 0;
        for (let track = 0; track < trackCount; track++) {
          if (families[track] !== family) continue;
          for (let sample = 0; sample < sampleCount; sample++) {
            sumX[sample] += routesSnapshot[(track * sampleCount + sample) * 2];
            sumY[sample] += routesSnapshot[(track * sampleCount + sample) * 2 + 1];
          }
          count++;
        }
        if (!count) continue;
        const points = Array.from({length: sampleCount}, (_, sample) =>
          storms.unproject(sumX[sample] / count, sumY[sample] / count)
        );
        for (let sample = 0; sample + 1 < sampleCount; sample++) {
          segments.set(
            [points[sample][0], points[sample][1], points[sample + 1][0], points[sample + 1][1]],
            (family * 63 + sample) * 4
          );
        }
      }
    }
    meanSegments.write(segments);
    ctx.requestLayers();
  }

  function updateSelection(): void {
    if (selectedStorm === NO_STORM) {
      ctx.setReadout('selected', 'click a storm');
      return;
    }
    let text = describeStorm(storms, selectedStorm);
    if (familySizes.length) {
      text += `; family ${families[selectedStorm] + 1} of ${familySizes.length}`;
    }
    if (matrix) {
      let nearest = -1;
      let nearestDistance = Infinity;
      for (let track = 0; track < trackCount; track++) {
        if (track === selectedStorm) continue;
        const distance = matrix[selectedStorm * trackCount + track];
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = track;
        }
      }
      if (nearest >= 0) {
        text += `; closest track: ${getStormLabel(storms, nearest)} at ${formatCount(nearestDistance / 1000)} km`;
      }
    }
    ctx.setReadout('selected', text);
  }

  function pickStorm(coordinate: readonly [number, number] | null): number {
    if (!coordinate) return -1;
    const [x, y] = storms.project(coordinate[0], coordinate[1]);
    const {track, distance} = findNearestTrack(storms, x, y);
    return track >= 0 && distance < PICK_RADIUS_METERS ? track : -1;
  }

  // ---- Initial work ------------------------------------------------------------------------------
  ctx.setReadout(
    'storms',
    `${formatCount(trackCount)} storms, ${formatCount(vertexCount)} six-hourly fixes, seasons ${Math.min(...storms.season)} to ${Math.max(...storms.season)}`
  );
  replaceRoutes(Number(ctx.options.routeSamples), ctx.options.routeSpacing);
  routeReader = createRouteReader(routeVariant!);
  replaceKMeans();
  writeSelectedOutline();
  updateSelection();

  // ---- Instance ----------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [
      ...(routeVariant ? [routeVariant.resample, routeVariant.similarity] : []),
      ...(kmeansVariant ? [kmeansVariant.compiled] : [])
    ],

    setOption(id, _value, state) {
      switch (id) {
        case 'routeSamples':
        case 'routeSpacing':
          routeReader?.stop();
          replaceRoutes(Number(state.routeSamples), state.routeSpacing);
          routeReader = createRouteReader(routeVariant!);
          routesSnapshot = null;
          break;
        case 'distanceMeasure':
          buildMatrixAndEmbedding();
          break;
        case 'familyCount':
        case 'initialization':
        case 'seed':
        case 'iterations':
          replaceKMeans();
          break;
        case 'similarityRangeKm':
          updateDistanceChart();
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      const variant = routeVariant;
      if (variant && similarityDirty) {
        variant.resample.encode(commandEncoder, {parameters: undefined});
        variant.similarity.encode(commandEncoder, {parameters: undefined});
        variant.encoded = true;
        similarityDirty = false;
        similarityReader.request(commandEncoder);
        routeReader?.request(commandEncoder);
      }
      similarityReader.flush(commandEncoder);
      routeReader?.flush(commandEncoder);

      if (kmeansVariant && kmeansDirty) {
        kmeansVariant.compiled.encode(commandEncoder, {parameters: undefined});
        kmeansVariant.encoded = true;
        kmeansDirty = false;
        kmeansReader.request(commandEncoder);
      }
      kmeansReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      let colorProps: Record<string, unknown>;
      switch (options.colorBy) {
        case 'family':
          colorProps = {
            values: familyBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentTracksBuffer,
            colormap: 'category',
            palette: HURRICANE_FAMILY_COLORS
          };
          break;
        case 'similarity':
          colorProps = {
            values: distanceBuffer,
            valueFormat: 'float32',
            valueIndices: segmentTracksBuffer,
            valueScale: 0.001,
            valueRange: [0, options.similarityRangeKm],
            colormap: options.ramp,
            noDataColor: [140, 146, 160, 60]
          };
          break;
        case 'peak':
          colorProps = {
            values: maxCategoryBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentTracksBuffer,
            colormap: 'category',
            palette: HURRICANE_CATEGORY_COLORS
          };
          break;
        case 'season':
          colorProps = {
            values: seasonBuffer,
            valueFormat: 'float32',
            valueIndices: segmentTracksBuffer,
            valueRange: [1980, 2025],
            colormap: options.ramp
          };
          break;
        default:
          colorProps = {color: dark ? [200, 210, 230, 255] : [60, 70, 95, 255]};
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'hurricane-tracks',
          ...drawProps,
          segments: segmentsBuffer,
          instanceCount: segmentCount,
          widthPixels: 1.4,
          opacity: options.showRoutes ? options.trackOpacity * 0.3 : options.trackOpacity,
          ...colorProps
        })
      );
      if (options.showRoutes && routeVariant) {
        const variant = routeVariant;
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: `hurricane-routes-${variant.key}`,
            ...drawProps,
            segments: variant.routeSegments,
            instanceCount: trackCount * (variant.sampleCount - 1),
            values: familyBuffer,
            valueFormat: 'uint32',
            valueDivisor: variant.sampleCount - 1,
            colormap: 'category',
            palette: HURRICANE_FAMILY_COLORS,
            widthPixels: 1.4,
            opacity: 0.55
          }),
          new SpatialAnalysisPointLayer({
            id: `hurricane-route-samples-${variant.key}`,
            ...drawProps,
            positions: variant.routeDegrees,
            instanceCount: trackCount * variant.sampleCount,
            values: familyBuffer,
            valueFormat: 'uint32',
            valueDivisor: variant.sampleCount,
            colormap: 'category',
            palette: HURRICANE_FAMILY_COLORS,
            radiusPixels: 1.9,
            opacity: 0.9
          })
        );
      }
      if (options.showFamilyMeans) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'hurricane-family-means',
            ...drawProps,
            segments: meanSegments,
            instanceCount: MAXIMUM_FAMILIES * 63,
            color: dark ? [255, 255, 255, 255] : [15, 20, 30, 255],
            widthPixels: 7,
            opacity: 0.95
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'hurricane-family-means-color',
            ...drawProps,
            segments: meanSegments,
            instanceCount: MAXIMUM_FAMILIES * 63,
            values: familyIndexBuffer,
            valueFormat: 'uint32',
            valueDivisor: 63,
            colormap: 'category',
            palette: HURRICANE_FAMILY_COLORS,
            widthPixels: 4,
            opacity: 1
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'hurricane-selected',
          ...drawProps,
          segments: selectedSegments,
          instanceCount: storms.longestTrack,
          widthPixels: 3.4,
          color: dark ? [255, 255, 255, 240] : [15, 20, 30, 240]
        })
      );
      return layers;
    },

    getTooltip(event) {
      const track = pickStorm(event.coordinate);
      if (track < 0) return null;
      const family = familySizes.length ? `, family ${families[track] + 1}` : '';
      return `${describeStorm(storms, track)}${family}`;
    },

    onClick(event) {
      const track = pickStorm(event.coordinate);
      if (track < 0) return false;
      selectedStorm = track;
      writeSelectedOutline();
      writeSelectedDistances();
      updateSelection();
      updateFamilyChart();
      updateDistanceChart();
      ctx.requestLayers();
      return true;
    },

    destroy() {
      destroyed = true;
      similarityReader.stop();
      routeReader?.stop();
      kmeansReader.stop();
      resources.destroy();
    }
  };
}

/**
 * Classical multidimensional scaling of a symmetric distance matrix into two dimensions by power
 * iteration on the double-centered squared distances. Returns coordinates scaled to unit spread of
 * the first axis, the share of the total variance the two axes keep and the total variance of the
 * scaled coordinates (for the explained-variance readout of the k-means inertia).
 */
export function classicalScaling(
  distances: Float32Array,
  count: number
): {coordinates: Float32Array; explained: number; totalVariance: number} {
  const centered = new Float64Array(count * count);
  const rowMeans = new Float64Array(count);
  let grandMean = 0;
  for (let row = 0; row < count; row++) {
    let sum = 0;
    for (let column = 0; column < count; column++) {
      const value = distances[row * count + column];
      sum += value * value;
    }
    rowMeans[row] = sum / count;
    grandMean += rowMeans[row];
  }
  grandMean /= count;
  let trace = 0;
  for (let row = 0; row < count; row++) {
    for (let column = 0; column < count; column++) {
      const value = distances[row * count + column];
      centered[row * count + column] =
        -0.5 * (value * value - rowMeans[row] - rowMeans[column] + grandMean);
    }
    trace += centered[row * count + row];
  }
  const axes: {vector: Float64Array; value: number}[] = [];
  for (let axis = 0; axis < 2; axis++) {
    let vector = Float64Array.from({length: count}, (_, index) => Math.sin(index * 1.7 + axis + 1));
    let value = 0;
    for (let iteration = 0; iteration < MDS_ITERATIONS; iteration++) {
      for (const previous of axes) {
        let dot = 0;
        for (let index = 0; index < count; index++) dot += vector[index] * previous.vector[index];
        for (let index = 0; index < count; index++) vector[index] -= dot * previous.vector[index];
      }
      const next = new Float64Array(count);
      for (let row = 0; row < count; row++) {
        let sum = 0;
        for (let column = 0; column < count; column++) {
          sum += centered[row * count + column] * vector[column];
        }
        next[row] = sum;
      }
      let norm = 0;
      for (let index = 0; index < count; index++) norm += next[index] * next[index];
      norm = Math.sqrt(norm) || 1;
      value = norm;
      for (let index = 0; index < count; index++) next[index] /= norm;
      vector = next;
    }
    axes.push({vector, value});
  }
  const scale = 1 / Math.sqrt(Math.max(axes[0].value, 1e-9) / count);
  const coordinates = new Float32Array(count * 2);
  let total = 0;
  for (let index = 0; index < count; index++) {
    const x = axes[0].vector[index] * Math.sqrt(axes[0].value) * scale;
    const y = axes[1].vector[index] * Math.sqrt(axes[1].value) * scale;
    coordinates[index * 2] = x;
    coordinates[index * 2 + 1] = y;
    total += x * x + y * y;
  }
  return {
    coordinates,
    explained: trace > 0 ? (axes[0].value + axes[1].value) / trace : 0,
    totalVariance: total
  };
}
