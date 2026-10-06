// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Nearest road: `GPUNearestFeatureJoin` snaps every New York point of interest to its nearest road
 * segment within a per-frame radius. A mode-local layer draws a connector from each POI to the
 * closest point on its matched segment straight from the join output, POIs are colored by snap
 * distance, unmatched POIs are red, and segments that received at least one POI are highlighted
 * from `featureCounts`. Only a small summary (matched count, overflow) is read back.
 *
 * A second join, "k nearest", runs `GPUNearestFeatureJoin` in neighbors mode: a sample of drifting query
 * points against road segments, road polylines or synthetic polygons. The join is compiled once with
 * capacity `k = 8`; "k shown" and the per-frame `maxDistance` are parameters (the shown slots are the
 * first n of the ordered neighbors, which is exactly what a smaller k would return). Foot-point
 * links are drawn straight from the `neighborFootPoints` and `neighborIds` outputs. The ties mode and
 * the feature geometry are compile-time choices; polylines and polygons use a
 * `GPUSpatialJoinPrepared` index that is built once while the queries move.
 *
 * `spatialSort` is a compile-time option (the toggle rebuilds the graph). "Shuffle road rows" applies
 * one seeded permutation to every per-segment buffer so the road table becomes spatially incoherent
 * without changing the picture; it is a plain buffer rewrite. "Measure spatialSort on vs off" times
 * the active graph and a temporary graph of the other setting outside the frame.
 *
 * `GPUNearestFeatureWeights` turns the k-nearest output into a CSR weights matrix (query rows to
 * feature IDs) with a compile-time `k` and rule (binary or inverse distance). Weighted links are
 * drawn from that CSR; the readout shows rows and nonzeros.
 */

import type {Layer} from '@deck.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUNearestFeatureJoin,
  GPUNearestFeatureWeights,
  GPUSpatialJoinPrepared,
  type GPUNearestFeatureGeometry
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {createSeededRandom, LocalMetricProjection} from '../spatial-analysis-data';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {formatCompiledGraphTiming, formatSpeedup, measureCompiledGraph} from './vector-timing';
import {NearestSnapLayer, NeighborLinkLayer, WeightedLinkLayer} from './nearest-join-layers';
import {SummaryReader} from './summary-reader';

const MAXIMUM_RADIUS = 300;
const DEFAULT_RADIUS = 60;
/** Frames between summary readbacks. */
const READBACK_INTERVAL = 15;
const CANDIDATES_PER_POINT = 256;
/** Frames after creation before the automatic spatialSort measurement runs. */
const AUTO_MEASURE_FRAME = 40;
const SHUFFLE_SEED = 20240611;
/** Query points sampled from the points of interest for the k-nearest join. */
const NEIGHBOR_QUERY_COUNT = 256;
/** Compiled `k` of the k-nearest join. */
const NEIGHBOR_K = 8;
/** Slots per query when `ties: 'all'` may keep more than `k` features. */
const NEIGHBOR_TIE_CAPACITY = 16;
const POLYGON_FEATURE_COUNT = 48;
/** Peak drift radius of the sampled queries, meters. */
const NEIGHBOR_DRIFT_METERS = 30;

type JoinKind = 'snap' | 'neighbors';
type NeighborFeatureKind = 'segments' | 'lines' | 'polygons';
type NeighborTies = 'lowest-id' | 'all';
type WeightRule = 'binary' | 'inverse-distance';
/** Selectable compiled `k` of the nearest-feature weights. */
const WEIGHT_K_OPTIONS = [1, 3, 5];

/** Chains consecutive road segments that share an endpoint into polylines. */
function chainRoadLines(segments: Float32Array): {
  positions: Float32Array;
  lineOffsets: Uint32Array;
} {
  const vertices: number[] = [];
  const offsets: number[] = [0];
  const count = segments.length / 4;
  for (let row = 0; row < count; row++) {
    const continues =
      row > 0 &&
      vertices[vertices.length - 2] === segments[row * 4] &&
      vertices[vertices.length - 1] === segments[row * 4 + 1];
    if (!continues) {
      if (row > 0) offsets.push(vertices.length / 2);
      vertices.push(segments[row * 4], segments[row * 4 + 1]);
    }
    vertices.push(segments[row * 4 + 2], segments[row * 4 + 3]);
  }
  offsets.push(vertices.length / 2);
  return {positions: Float32Array.from(vertices), lineOffsets: Uint32Array.from(offsets)};
}

/** Seeded small triangles, squares and hexagons scattered around the points of interest. */
function createPolygonFeatures(pointPositions: Float32Array) {
  const random = createSeededRandom(4242);
  const pointCount = pointPositions.length / 2;
  const positions: number[] = [];
  const ringOffsets: number[] = [0];
  const outline: number[] = [];
  for (let feature = 0; feature < POLYGON_FEATURE_COUNT; feature++) {
    const anchor = Math.floor(random() * pointCount);
    const centerX = pointPositions[anchor * 2] + (random() - 0.5) * 200;
    const centerY = pointPositions[anchor * 2 + 1] + (random() - 0.5) * 200;
    const sides = [3, 4, 6][Math.floor(random() * 3)];
    const radius = 25 + random() * 65;
    const rotation = random() * Math.PI;
    const ring: [number, number][] = [];
    for (let side = 0; side < sides; side++) {
      const angle = rotation + (side / sides) * Math.PI * 2;
      ring.push([centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius]);
    }
    for (const [x, y] of ring) positions.push(x, y);
    ringOffsets.push(positions.length / 2);
    ring.forEach(([x0, y0], side) => {
      const [x1, y1] = ring[(side + 1) % sides];
      outline.push(x0, y0, x1, y1);
    });
  }
  const unit = Uint32Array.from({length: POLYGON_FEATURE_COUNT + 1}, (_, row) => row);
  return {
    positions: Float32Array.from(positions),
    featureOffsets: unit,
    polygonOffsets: unit,
    ringOffsets: Uint32Array.from(ringOffsets),
    outline: Float32Array.from(outline)
  };
}

export const nearestJoinMode: SpatialAnalysisModeDefinition = {
  id: 'nearest-join',
  title: 'Nearest road',
  contributors: ['GPUNearestFeatureJoin', 'GPUSpatialJoinPrepared', 'GPUNearestFeatureWeights'],
  description:
    'Each point of interest snaps to its nearest road segment within a radius, computed on the GPU. ' +
    'Drag the radius slider: the join re-runs every frame without recompiling. Switch the join to "k nearest" for ' +
    'foot-point links to the k closest roads, polylines or polygons, with ties and a per-frame maxDistance; weighted links show the GPUNearestFeatureWeights CSR.',
  initialViewState: {longitude: -73.985, latitude: 40.755, zoom: 15},

  async create(context) {
    const [pois, roads] = await Promise.all([
      context.data.getNewYorkPointsOfInterest(),
      context.data.getNewYorkRoads()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'nearest-join');
    const pointCount = pois.positions.length / 2;
    const segmentCount = roads.segments.length / 4;

    // The contributor takes packed starts and ends; the float32x4 rows are kept for the layers.
    const starts = new Float32Array(segmentCount * 2);
    const ends = new Float32Array(segmentCount * 2);
    for (let row = 0; row < segmentCount; row++) {
      starts.set(roads.segments.subarray(row * 4, row * 4 + 2), row * 2);
      ends.set(roads.segments.subarray(row * 4 + 2, row * 4 + 4), row * 2);
    }
    const positionsBuffer = resources.createBuffer('positions', pois.positions);
    const segmentsBuffer = resources.createBuffer('segments', roads.segments);
    const startsBuffer = resources.createBuffer('starts', starts);
    const endsBuffer = resources.createBuffer('ends', ends);
    const radius = resources.createParameterBuffer(
      'radius',
      'float32',
      1,
      Float32Array.of(DEFAULT_RADIUS)
    );
    const nearestFeatureIds = resources.createBuffer('nearest-feature-ids', pointCount * 4);
    const nearestDistances = resources.createBuffer('nearest-distances', pointCount * 4);
    const featureCounts = resources.createBuffer('feature-counts', segmentCount * 4);
    const overflow = resources.createBuffer('overflow', 4);
    const candidateCount = resources.createBuffer('candidate-count', 4);
    const matchIds = resources.createBuffer('match-ids', pointCount * 4);
    const matchCount = resources.createBuffer('match-count', 4);
    const matchOverflow = resources.createBuffer('match-overflow', 4);

    const buildGraph = (spatialSort: boolean): CompiledGPUCommandGraph<void> => {
      const graph = new GPUCommandGraph<void>(device, {
        id: `nearest-join-${spatialSort ? 'sorted' : 'unsorted'}`
      });
      graph.add(
        new GPUNearestFeatureJoin({
          id: 'nearest-join',
          points: importGraphBuffer(graph, 'points', positionsBuffer, 'float32x2', pointCount),
          features: {
            kind: 'segments',
            starts: importGraphBuffer(graph, 'starts', startsBuffer, 'float32x2', segmentCount),
            ends: importGraphBuffer(graph, 'ends', endsBuffer, 'float32x2', segmentCount)
          },
          radius: radius.importToGraph(graph),
          // Sized for the maximum radius; the candidate readout shows the real demand.
          candidateCapacity: pointCount * CANDIDATES_PER_POINT,
          spatialSort,
          nearestFeatureIds: importGraphBuffer(
            graph,
            'nearest-feature-ids',
            nearestFeatureIds,
            'uint32',
            pointCount
          ),
          nearestDistances: importGraphBuffer(
            graph,
            'nearest-distances',
            nearestDistances,
            'float32',
            pointCount
          ),
          featureCounts: importGraphBuffer(
            graph,
            'feature-counts',
            featureCounts,
            'uint32',
            segmentCount
          ),
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
          candidateCount: importGraphBuffer(graph, 'candidate-count', candidateCount, 'uint32', 1),
          matches: {
            ids: importGraphBuffer(graph, 'match-ids', matchIds, 'uint32', pointCount),
            count: importGraphBuffer(graph, 'match-count', matchCount, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'match-overflow', matchOverflow, 'uint32', 1)
          }
        })
      );
      return graph.compile();
    };
    let spatialSort = false;
    let compiled = resources.track(buildGraph(spatialSort));

    const readbackRing = new GPUReadbackRing(device, {id: 'nearest-join-readback', byteLength: 16});
    resources.track({destroy: () => readbackRing.destroy()});

    let currentRadius = DEFAULT_RADIUS;
    let showConnectors = true;
    let destroyed = false;
    let readbackPending = false;
    let measuring = false;
    let measureAgain = false;
    let encodedFrames = 0;
    let autoMeasured = false;

    // One deterministic Fisher-Yates permutation; shuffled row r holds original row permutation[r].
    // starts, ends and the float32x4 segments are permuted together, so the nearestFeatureIds and
    // featureCounts the join writes keep indexing matching rows in every layer.
    let shuffledRows: {segments: Float32Array; starts: Float32Array; ends: Float32Array} | null =
      null;
    const getShuffledRows = () => {
      if (shuffledRows) return shuffledRows;
      const random = createSeededRandom(SHUFFLE_SEED);
      const permutation = Uint32Array.from({length: segmentCount}, (_, row) => row);
      for (let row = segmentCount - 1; row > 0; row--) {
        const other = Math.floor(random() * (row + 1));
        [permutation[row], permutation[other]] = [permutation[other], permutation[row]];
      }
      const segments = new Float32Array(segmentCount * 4);
      const shuffledStarts = new Float32Array(segmentCount * 2);
      const shuffledEnds = new Float32Array(segmentCount * 2);
      for (let row = 0; row < segmentCount; row++) {
        const source = permutation[row];
        segments.set(roads.segments.subarray(source * 4, source * 4 + 4), row * 4);
        shuffledStarts.set(starts.subarray(source * 2, source * 2 + 2), row * 2);
        shuffledEnds.set(ends.subarray(source * 2, source * 2 + 2), row * 2);
      }
      shuffledRows = {segments, starts: shuffledStarts, ends: shuffledEnds};
      return shuffledRows;
    };

    // ------------------------------------------------------------------ k-nearest join
    const roadLines = chainRoadLines(roads.segments);
    const queryCount = Math.min(NEIGHBOR_QUERY_COUNT, pointCount);
    // The points of interest closest to the first view's center, so the links are dense there.
    const [centerX, centerY] = new LocalMetricProjection(roads.origin).project(-73.985, 40.755);
    const nearOrigin = Array.from({length: pointCount}, (_, row) => row)
      .sort(
        (a, b) =>
          Math.hypot(pois.positions[a * 2] - centerX, pois.positions[a * 2 + 1] - centerY) -
          Math.hypot(pois.positions[b * 2] - centerX, pois.positions[b * 2 + 1] - centerY)
      )
      .slice(0, queryCount);
    const queryBase = new Float32Array(queryCount * 2);
    nearOrigin.forEach((row, query) => {
      queryBase[query * 2] = pois.positions[row * 2];
      queryBase[query * 2 + 1] = pois.positions[row * 2 + 1];
    });
    const polygonFeatures = createPolygonFeatures(queryBase);
    const queryAnimated = new Float32Array(queryBase);
    const queryBuffer = resources.createBuffer('query-positions', queryBase);
    const lineVertexBuffer = resources.createBuffer('line-positions', roadLines.positions);
    const lineOffsetBuffer = resources.createBuffer('line-offsets', roadLines.lineOffsets);
    const polygonVertexBuffer = resources.createBuffer(
      'polygon-positions',
      polygonFeatures.positions
    );
    const polygonFeatureOffsets = resources.createBuffer(
      'polygon-feature-offsets',
      polygonFeatures.featureOffsets
    );
    const polygonPolygonOffsets = resources.createBuffer(
      'polygon-polygon-offsets',
      polygonFeatures.polygonOffsets
    );
    const polygonRingOffsets = resources.createBuffer(
      'polygon-ring-offsets',
      polygonFeatures.ringOffsets
    );
    const polygonOutline = resources.createBuffer('polygon-outline', polygonFeatures.outline);
    const neighborIds = resources.createBuffer(
      'neighbor-ids',
      queryCount * NEIGHBOR_TIE_CAPACITY * 4
    );
    const neighborCounts = resources.createBuffer('neighbor-counts', queryCount * 4);
    const neighborDistances = resources.createBuffer(
      'neighbor-distances',
      queryCount * NEIGHBOR_TIE_CAPACITY * 4
    );
    const neighborFeet = resources.createBuffer(
      'neighbor-feet',
      queryCount * NEIGHBOR_TIE_CAPACITY * 8
    );
    const neighborSegments = resources.createBuffer(
      'neighbor-segments',
      queryCount * NEIGHBOR_TIE_CAPACITY * 4
    );
    const neighborOverflow = resources.createBuffer('neighbor-overflow', 4);
    const weightOffsets = resources.createBuffer('weight-offsets', (queryCount + 1) * 4);
    const weightNeighbors = resources.createBuffer('weight-neighbors', queryCount * NEIGHBOR_K * 4);
    const weightValues = resources.createBuffer('weight-values', queryCount * NEIGHBOR_K * 4);
    const weightDistances = resources.createBuffer('weight-distances', queryCount * NEIGHBOR_K * 4);
    const weightOverflow = resources.createBuffer('weight-overflow', 4);
    const neighborMaxDistance = resources.createParameterBuffer(
      'neighbor-max-distance',
      'float32',
      1,
      Float32Array.of(400)
    );

    let join: JoinKind = 'snap';
    let neighborFeatureKind: NeighborFeatureKind = 'segments';
    let neighborTies: NeighborTies = 'lowest-id';
    let shownNeighbors = 3;
    let weightK = 3;
    let weightRule: WeightRule = 'binary';
    let showWeights = true;
    let currentMaxDistance = 400;
    let driftQueries = true;
    let colorByRank = true;
    let neighborBuild: {
      compiled: CompiledGPUCommandGraph<void>;
      prepared: GPUSpatialJoinPrepared | null;
      stride: number;
      weightK: number;
      reader: SummaryReader;
      frames: number;
    } | null = null;

    const releaseNeighborBuild = (build: NonNullable<typeof neighborBuild>) => {
      build.reader.stop();
      resources.release(build.compiled);
      if (build.prepared) build.prepared.destroy();
    };

    const buildNeighborGraph = (): NonNullable<typeof neighborBuild> => {
      const stride = neighborTies === 'all' ? NEIGHBOR_TIE_CAPACITY : NEIGHBOR_K;
      const graph = new GPUCommandGraph<void>(device, {
        id: `nearest-neighbors-${neighborFeatureKind}-${neighborTies}`
      });
      const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
        name: string,
        buffer: Parameters<typeof importGraphBuffer>[2],
        format: Format,
        length: number
      ) => importGraphBuffer(graph, name, buffer, format, length);
      let features: GPUNearestFeatureGeometry;
      let prepared: GPUSpatialJoinPrepared | null = null;
      if (neighborFeatureKind === 'segments') {
        features = {
          kind: 'segments',
          starts: view('starts', startsBuffer, 'float32x2', segmentCount),
          ends: view('ends', endsBuffer, 'float32x2', segmentCount)
        };
      } else {
        features =
          neighborFeatureKind === 'lines'
            ? {
                kind: 'lines',
                positions: view(
                  'line-positions',
                  lineVertexBuffer,
                  'float32x2',
                  roadLines.positions.length / 2
                ),
                lineOffsets: view(
                  'line-offsets',
                  lineOffsetBuffer,
                  'uint32',
                  roadLines.lineOffsets.length
                )
              }
            : {
                kind: 'polygons',
                positions: view(
                  'polygon-positions',
                  polygonVertexBuffer,
                  'float32x2',
                  polygonFeatures.positions.length / 2
                ),
                featureOffsets: view(
                  'polygon-feature-offsets',
                  polygonFeatureOffsets,
                  'uint32',
                  POLYGON_FEATURE_COUNT + 1
                ),
                polygonOffsets: view(
                  'polygon-polygon-offsets',
                  polygonPolygonOffsets,
                  'uint32',
                  POLYGON_FEATURE_COUNT + 1
                ),
                ringOffsets: view(
                  'polygon-ring-offsets',
                  polygonRingOffsets,
                  'uint32',
                  polygonFeatures.ringOffsets.length
                )
              };
        // A static feature set: the BVH is built on the first encoding and reused while queries move.
        prepared = new GPUSpatialJoinPrepared({id: 'neighbor-features', geometry: features});
        graph.add(prepared);
      }
      const joinNeighborIds = view('neighbor-ids', neighborIds, 'uint32', queryCount * stride);
      const joinNeighborCounts = view('neighbor-counts', neighborCounts, 'uint32', queryCount);
      const joinNeighborDistances = view(
        'neighbor-distances',
        neighborDistances,
        'float32',
        queryCount * stride
      );
      graph.add(
        new GPUNearestFeatureJoin({
          id: 'neighbors',
          points: view('queries', queryBuffer, 'float32x2', queryCount),
          features,
          k: NEIGHBOR_K,
          neighborCapacity: stride,
          ties: neighborTies,
          maxDistance: neighborMaxDistance.importToGraph(graph),
          ...(prepared ? {prepared} : {}),
          neighborIds: joinNeighborIds,
          neighborCounts: joinNeighborCounts,
          neighborDistances: joinNeighborDistances,
          neighborFootPoints: view('neighbor-feet', neighborFeet, 'float32x2', queryCount * stride),
          neighborSegmentIndices: view(
            'neighbor-segments',
            neighborSegments,
            'uint32',
            queryCount * stride
          ),
          overflow: view('neighbor-overflow', neighborOverflow, 'uint32', 1)
        })
      );
      // Cross weights from every query to its `weightK` nearest features. `k` and the rule are
      // compile-time; the rows are the same drifting queries, so the CSR follows them each frame.
      graph.add(
        new GPUNearestFeatureWeights({
          id: 'neighbor-weights',
          neighborIds: joinNeighborIds,
          neighborCounts: joinNeighborCounts,
          neighborDistances: joinNeighborDistances,
          slotCapacity: stride,
          k: weightK,
          weightType: weightRule,
          weights: {
            offsets: view('weight-offsets', weightOffsets, 'uint32', queryCount + 1),
            neighbors: view('weight-neighbors', weightNeighbors, 'uint32', queryCount * NEIGHBOR_K),
            weights: view('weight-values', weightValues, 'float32', queryCount * NEIGHBOR_K),
            distances: view('weight-distances', weightDistances, 'float32', queryCount * NEIGHBOR_K)
          },
          overflow: view('weight-overflow', weightOverflow, 'uint32', 1)
        })
      );
      const compiled = resources.track(graph.compile());
      const build: NonNullable<typeof neighborBuild> = {
        compiled,
        prepared,
        stride,
        weightK,
        frames: 0,
        reader: new SummaryReader(
          resources,
          'neighbors',
          [
            {buffer: neighborCounts, size: queryCount * 4},
            {buffer: neighborDistances, size: queryCount * stride * 4},
            {buffer: neighborOverflow, size: 4},
            {buffer: weightOffsets, size: (queryCount + 1) * 4},
            {buffer: weightOverflow, size: 4}
          ],
          bytes => {
            if (destroyed || build !== neighborBuild) return;
            const words = new Uint32Array(bytes);
            const distances = new Float32Array(bytes, queryCount * 4, queryCount * stride);
            let matched = 0;
            let extended = 0;
            let tied = 0;
            let nearestSum = 0;
            for (let query = 0; query < queryCount; query++) {
              if (words[query] > 0) {
                matched++;
                nearestSum += distances[query * stride];
              }
              if (words[query] > NEIGHBOR_K) extended++;
              for (let slot = 1; slot < Math.min(words[query], stride); slot++) {
                if (distances[query * stride + slot] === distances[query * stride + slot - 1]) {
                  tied++;
                  break;
                }
              }
            }
            neighborMatchedReadout.setValue(`${matched} of ${queryCount} queries have a neighbor`);
            neighborNearestReadout.setValue(
              matched ? `${(nearestSum / matched).toFixed(1)} m mean nearest distance` : 'none'
            );
            neighborTiesReadout.setValue(
              `${tied} of ${queryCount} queries have tied distances; ` +
                (neighborTies === 'all'
                  ? `${extended} keep ties beyond k = ${NEIGHBOR_K}`
                  : 'lowest-id cuts ties at k')
            );
            const weightBase = queryCount + queryCount * stride + 1;
            const nonzeros = words[weightBase + queryCount];
            weightRowsReadout.setValue(
              `${queryCount} rows, ${nonzeros} nonzeros (k = ${build.weightK}, ${weightRule})` +
                (words[weightBase + queryCount + 1] ? ', OVERFLOW' : '')
            );
            neighborOverflowReadout.setValue(
              words[queryCount + queryCount * stride] ? 'YES' : 'no'
            );
            neighborBuildsReadout.setValue(
              build.prepared
                ? `${build.prepared.encodedBuildCount} in ${formatCount(build.frames)} frames (reused)`
                : 'rebuilt every encoding (segments cannot be prepared)'
            );
          }
        )
      };
      return build;
    };

    const rebuildNeighbors = () => {
      const previous = neighborBuild;
      neighborBuild = buildNeighborGraph();
      if (previous) {
        // The old graph is freed once no frame can encode it any more.
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            if (!destroyed) releaseNeighborBuild(previous);
          })
        );
      }
      context.updateLayers();
    };

    context.controls.addSelect<JoinKind>({
      label: 'Join (switching to k nearest compiles its graph once)',
      options: [
        {value: 'snap', label: 'Snap to nearest road within a radius'},
        {value: 'neighbors', label: 'k nearest features with foot points'}
      ],
      value: join,
      onChange: value => {
        join = value;
        if (join === 'neighbors' && !neighborBuild) neighborBuild = buildNeighborGraph();
        context.updateLayers();
      }
    });
    context.controls.addSelect<NeighborFeatureKind>({
      label: 'k nearest: feature geometry (compile-time: rebuilds graph)',
      options: [
        {value: 'segments', label: 'Road segments'},
        {value: 'lines', label: 'Road polylines (prepared index)'},
        {value: 'polygons', label: 'Synthetic polygons (prepared index)'}
      ],
      value: neighborFeatureKind,
      onChange: value => {
        neighborFeatureKind = value;
        if (join === 'neighbors') rebuildNeighbors();
        else if (neighborBuild) {
          releaseNeighborBuild(neighborBuild);
          neighborBuild = null;
        }
      }
    });
    context.controls.addSelect<NeighborTies>({
      label: 'k nearest: ties (compile-time: rebuilds graph)',
      options: [
        {value: 'lowest-id', label: 'lowest-id (cut at k)'},
        {value: 'all', label: 'all (keep every tie, capacity 16)'}
      ],
      value: neighborTies,
      onChange: value => {
        neighborTies = value;
        if (join === 'neighbors') rebuildNeighbors();
        else if (neighborBuild) {
          releaseNeighborBuild(neighborBuild);
          neighborBuild = null;
        }
      }
    });
    context.controls.addSelect<string>({
      label: 'k nearest: weights k (GPUNearestFeatureWeights, compile-time: rebuilds graph)',
      options: WEIGHT_K_OPTIONS.map(value => ({value: String(value), label: `k = ${value}`})),
      value: String(weightK),
      onChange: value => {
        weightK = Number(value);
        if (join === 'neighbors') rebuildNeighbors();
      }
    });
    context.controls.addSelect<WeightRule>({
      label: 'k nearest: weight rule (compile-time: rebuilds graph)',
      options: [
        {value: 'binary', label: 'binary (1 per neighbor)'},
        {value: 'inverse-distance', label: 'inverse distance (1 / d)'}
      ],
      value: weightRule,
      onChange: value => {
        weightRule = value;
        if (join === 'neighbors') rebuildNeighbors();
      }
    });
    context.controls.addToggle({
      label: 'k nearest: draw weighted links (width = row-standardized weight)',
      value: showWeights,
      onChange: value => {
        showWeights = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: `k nearest: k shown (compiled capacity ${NEIGHBOR_K}; per-frame)`,
      min: 1,
      max: NEIGHBOR_K,
      step: 1,
      value: shownNeighbors,
      format: value => `${value}`,
      onChange: value => {
        shownNeighbors = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'k nearest: maxDistance (per-frame parameter)',
      min: 20,
      max: 1500,
      step: 10,
      value: currentMaxDistance,
      format: value => `${value} m`,
      onChange: value => {
        currentMaxDistance = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'k nearest: drift the queries (per-frame join)',
      value: driftQueries,
      onChange: value => {
        driftQueries = value;
        if (!value) queryBuffer.write(queryBase);
      }
    });
    context.controls.addToggle({
      label: 'k nearest: color links by rank (off: by distance)',
      value: colorByRank,
      onChange: value => {
        colorByRank = value;
        context.updateLayers();
      }
    });

    context.controls.addSlider({
      label: 'Search radius (per-frame parameter)',
      min: 10,
      max: MAXIMUM_RADIUS,
      step: 5,
      value: currentRadius,
      format: value => `${value} m`,
      onChange: value => {
        currentRadius = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'spatialSort (compile-time: rebuilds graph)',
      value: spatialSort,
      onChange: value => {
        spatialSort = value;
        const previous = compiled;
        compiled = resources.track(buildGraph(spatialSort));
        // The old graph is freed once no frame can encode it any more.
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            if (!destroyed) resources.release(previous);
          })
        );
        if (measuring) measureAgain = true;
      }
    });
    context.controls.addToggle({
      label: 'Shuffle road rows (per-frame buffer rewrite, no recompile)',
      value: false,
      onChange: value => {
        const rows = value ? getShuffledRows() : {segments: roads.segments, starts, ends};
        segmentsBuffer.write(rows.segments);
        startsBuffer.write(rows.starts);
        endsBuffer.write(rows.ends);
        if (measuring) measureAgain = true;
        else void measureSpatialSort();
      }
    });
    context.controls.addButton({
      label: 'Measure spatialSort on vs off',
      onClick: () => void measureSpatialSort()
    });
    context.controls.addToggle({
      label: 'Show snap connectors',
      value: showConnectors,
      onChange: value => {
        showConnectors = value;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'POI color: distance to nearest road',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: '0 m',
        maximumLabel: 'radius'
      }
    });
    context.controls.addLegend({
      title: 'Overlays',
      entries: [
        {color: [255, 255, 255], label: 'Snap connector to nearest road point'},
        {color: [255, 170, 60], label: 'Road that received a POI'},
        {color: [255, 70, 70], label: 'POI with no road in radius'}
      ]
    });
    context.controls.addReadout('Points of interest', formatCount(pointCount));
    context.controls.addReadout('Road segments', formatCount(segmentCount));
    const matchedReadout = context.controls.addReadout('Matched', '...');
    const overflowReadout = context.controls.addReadout('Overflow', '...');
    const candidateReadout = context.controls.addReadout('BVH candidates', '...');
    const sortedReadout = context.controls.addReadout('spatialSort on', '...');
    const unsortedReadout = context.controls.addReadout('spatialSort off', '...');
    const speedupReadout = context.controls.addReadout('Speedup', '...');
    const neighborMatchedReadout = context.controls.addReadout('k nearest: matched', '...');
    const neighborNearestReadout = context.controls.addReadout('k nearest: nearest', '...');
    const neighborTiesReadout = context.controls.addReadout('k nearest: ties', '...');
    const neighborOverflowReadout = context.controls.addReadout('k nearest: overflow', '...');
    const weightRowsReadout = context.controls.addReadout('k nearest: weights', '...');
    const neighborBuildsReadout = context.controls.addReadout(
      'k nearest: feature index builds',
      '...'
    );
    context.controls.addNote(
      'Timed outside the frame: GPU timestamps when the device has timestamp-query, otherwise wall clock / 8 repetitions (upper bound). ' +
        'Sorting helps most when road rows are spatially incoherent; try shuffling the rows.'
    );
    context.controls.addReadout('Data', `${pois.attribution}; ${roads.attribution}`);

    const readSummary = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      [matchCount, overflow, candidateCount].forEach((sourceBuffer, index) => {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          destinationBuffer: ticket.buffer,
          destinationOffset: index * 4,
          size: 4
        });
      });
      ticket.markEncoded({byteLength: 12});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, 3);
        matchedReadout.setValue(`${formatCount(words[0])} of ${formatCount(pointCount)}`);
        overflowReadout.setValue(words[1] ? 'YES' : 'no');
        candidateReadout.setValue(formatCount(words[2]));
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    /** Times the active graph and a temporary graph of the other spatialSort setting. */
    const measureSpatialSort = async () => {
      if (measuring || destroyed) return;
      measuring = true;
      measureAgain = false;
      sortedReadout.setValue('measuring...');
      unsortedReadout.setValue('measuring...');
      speedupReadout.setValue('...');
      let temporary: CompiledGPUCommandGraph<void> | null = null;
      try {
        const activeSort = spatialSort;
        const active = compiled;
        temporary = buildGraph(!activeSort);
        const options = {parameters: undefined, completionBuffer: overflow, signal: context.signal};
        const activeTiming = await measureCompiledGraph(device, active, options);
        const otherTiming = await measureCompiledGraph(device, temporary, options);
        if (destroyed) return;
        const sorted = activeSort ? activeTiming : otherTiming;
        const unsorted = activeSort ? otherTiming : activeTiming;
        sortedReadout.setValue(formatCompiledGraphTiming(sorted));
        unsortedReadout.setValue(formatCompiledGraphTiming(unsorted));
        speedupReadout.setValue(formatSpeedup(unsorted.milliseconds, sorted.milliseconds));
      } catch {
        if (!destroyed) {
          sortedReadout.setValue('interrupted');
          unsortedReadout.setValue('interrupted');
          measureAgain = true;
        }
      } finally {
        temporary?.destroy();
        measuring = false;
        if (measureAgain && !destroyed) void measureSpatialSort();
      }
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () =>
        join === 'neighbors' && neighborBuild ? [neighborBuild.compiled] : [compiled],
      encode(commandEncoder, frame) {
        if (join === 'neighbors' && neighborBuild) {
          const build = neighborBuild;
          if (driftQueries) {
            for (let query = 0; query < queryCount; query++) {
              const angle = query * 2.399963 + frame.timeSeconds * 0.8;
              queryAnimated[query * 2] =
                queryBase[query * 2] + Math.cos(angle) * NEIGHBOR_DRIFT_METERS;
              queryAnimated[query * 2 + 1] =
                queryBase[query * 2 + 1] + Math.sin(angle) * NEIGHBOR_DRIFT_METERS;
            }
            queryBuffer.write(queryAnimated);
          }
          neighborMaxDistance.write(Float32Array.of(currentMaxDistance));
          build.compiled.encode(commandEncoder, {parameters: undefined});
          build.frames++;
          if (frame.frameIndex % READBACK_INTERVAL === 0) build.reader.markStale();
          build.reader.flush(commandEncoder);
          return;
        }
        encodedFrames++;
        if (!autoMeasured && encodedFrames >= AUTO_MEASURE_FRAME) {
          autoMeasured = true;
          void measureSpatialSort();
        }
        radius.write(Float32Array.of(currentRadius));
        compiled.encode(commandEncoder, {parameters: undefined});
        if (!readbackPending && frame.frameIndex % READBACK_INTERVAL === 0) {
          void readSummary(commandEncoder);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [roads.origin[0], roads.origin[1], 0];
        if (join === 'neighbors' && neighborBuild) {
          return [
            neighborFeatureKind === 'polygons'
              ? new SpatialAnalysisSegmentLayer({
                  id: 'neighbors-polygons',
                  coordinateOrigin,
                  segments: polygonOutline,
                  instanceCount: polygonFeatures.outline.length / 4,
                  color: [255, 170, 60, 230],
                  widthPixels: 2.5
                })
              : new SpatialAnalysisSegmentLayer({
                  id: 'neighbors-roads',
                  coordinateOrigin,
                  segments: segmentsBuffer,
                  instanceCount: segmentCount,
                  color: [150, 170, 200, 150],
                  widthPixels: 1.5
                }),
            new NeighborLinkLayer({
              id: 'neighbors-links',
              coordinateOrigin,
              queries: queryBuffer,
              neighborIds,
              footPoints: neighborFeet,
              queryCount,
              stride: neighborBuild.stride,
              shown: shownNeighbors,
              colorByRank,
              maximumDistance: currentMaxDistance,
              widthPixels: 1.6,
              alpha: 0.95
            }),
            ...(showWeights
              ? [
                  new WeightedLinkLayer({
                    id: 'neighbors-weighted-links',
                    coordinateOrigin,
                    queries: queryBuffer,
                    offsets: weightOffsets,
                    neighbors: weightNeighbors,
                    weights: weightValues,
                    joinIds: neighborIds,
                    joinFootPoints: neighborFeet,
                    queryCount,
                    stride: neighborBuild.stride,
                    weightK: neighborBuild.weightK,
                    widthPixels: 2.2,
                    alpha: 0.9
                  })
                ]
              : []),
            new SpatialAnalysisPointLayer({
              id: 'neighbors-queries',
              coordinateOrigin,
              positions: queryBuffer,
              instanceCount: queryCount,
              color: [255, 255, 255, 255],
              radiusPixels: 3.5
            })
          ];
        }
        const layers: Layer[] = [
          new SpatialAnalysisSegmentLayer({
            id: 'nearest-join-roads',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            color: [150, 170, 200, 110],
            widthPixels: 1.5
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'nearest-join-used-roads',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            values: featureCounts,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [255, 170, 60, 130],
            noDataColor: [0, 0, 0, 0],
            widthPixels: 2.5
          })
        ];
        if (showConnectors) {
          layers.push(
            new NearestSnapLayer({
              id: 'nearest-join-connectors',
              coordinateOrigin,
              positions: positionsBuffer,
              nearestFeatureIds,
              segments: segmentsBuffer,
              instanceCount: pointCount,
              color: [255, 255, 255, 235],
              widthPixels: 1.5
            })
          );
        }
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'nearest-join-unmatched',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: pointCount,
            values: nearestFeatureIds,
            valueFormat: 'uint32',
            colormap: 'category',
            // A single transparent palette entry hides matched points; the no-data sentinel is red.
            palette: [[0, 0, 0, 0]],
            noDataColor: [255, 70, 70, 255],
            radiusPixels: 4
          }),
          new SpatialAnalysisPointLayer({
            id: 'nearest-join-points',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: pointCount,
            values: nearestDistances,
            valueFormat: 'float32',
            colormap: 'viridis',
            valueRange: [0, currentRadius],
            // Unmatched points carry the distance -1 and are drawn by the red layer instead.
            discardAtOrBelow: -0.5,
            radiusPixels: 4
          })
        );
        return layers;
      },
      destroy() {
        destroyed = true;
        neighborBuild?.prepared?.destroy();
        neighborBuild?.reader.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};
