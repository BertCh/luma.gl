// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Relate: `GPUSpatialPredicateJoin` evaluates a topological predicate (or any DE-9IM pattern) between
 * a small set of movable query shapes (the left side) and the San Francisco ZIP-code polygons (the
 * right side). The ZIP side is a `GPUSpatialJoinPrepared` handle: its bounds and BVH are built once and
 * reused while the queries move every frame, and a readout counts the builds. An anti join over the
 * same predicate lists the queries that match nothing, and `GPUSpatialJoinCandidates` shows the bounding
 * box candidates that the exact predicate filters down.
 *
 * Predicate and query-set kind are compile-time choices (the graph is rebuilt and labelled so): the
 * predicate picks the kernel family. The DE-9IM pattern of `relate` and the distance of `dwithin` are
 * per-frame parameter buffers, so changing them is a buffer write and the rebuild counter stays at
 * zero. Query positions, dragging and the invalidate-every-frame comparison are buffer writes too.
 * The pair table (a few hundred rows) is read back; the picture colors ZIP outlines by match count
 * and queries by matched or unmatched from it. The graph time is measured outside the frame.
 */

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  formatGPUSpatialRelate,
  GPUSpatialJoinCandidates,
  GPUSpatialJoinPrepared,
  GPUSpatialPredicateJoin,
  type GPUSpatialJoinGeometry,
  type GPUSpatialPredicate
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  GPU_SPATIAL_RELATE_PATTERN_WORDS,
  packGPUSpatialRelatePattern
} from '@luma.gl/experimental/gpu-spatial-analysis';
import type {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {LocalMetricProjection, type SpatialAnalysisPolygons} from '../spatial-analysis-data';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {SummaryReader} from './summary-reader';
import {findContainingFeature} from './raster-join-layers';
import {formatCompiledGraphTiming, measureCompiledGraph} from './vector-timing';

type QuerySet = 'polygons' | 'lines';
type RelatePredicate = Extract<
  GPUSpatialPredicate,
  | 'intersects'
  | 'contains'
  | 'within'
  | 'covers'
  | 'coveredBy'
  | 'touches'
  | 'crosses'
  | 'overlaps'
  | 'equals'
  | 'containsProperly'
  | 'relate'
  | 'dwithin'
>;

const PREDICATES: readonly RelatePredicate[] = [
  'intersects',
  'contains',
  'within',
  'covers',
  'coveredBy',
  'touches',
  'crosses',
  'overlaps',
  'equals',
  'containsProperly',
  'relate',
  'dwithin'
];
/**
 * Patterns that cannot match disjoint geometries (the engine rejects those). A value holds one
 * pattern, or several joined by `|` (an any-of list); see {@link parsePattern}.
 */
const PATTERNS: readonly {value: string; label: string}[] = [
  {value: 'T*T***T**', label: 'T*T***T** interiors overlap, both have exterior'},
  {value: '212101212', label: '212101212 areas partially overlap'},
  {value: 'T*****FF*', label: 'T*****FF* contains'},
  {value: 'T*F**F***', label: 'T*F**F*** within'},
  {value: '*T*******', label: '*T******* left interior meets right boundary'},
  {value: 'FT*******', label: 'FT******* only boundary-interior contact'},
  {
    value: 'T********|*T*******|***T*****|****T****',
    label: 'any of four: intersects (a list of patterns)'
  }
];
/** Most patterns an any-of list may hold: slots of the per-frame pattern buffer. */
const PATTERN_SLOTS = 4;
const MAXIMUM_DISTANCE = 3000;
const DEFAULT_DISTANCE = 400;
/** Frame at which the graph time is measured once, outside the frame. */
const AUTO_MEASURE_FRAME = 40;
/** Milliseconds after the last pattern or distance change before the graph is timed again. */
const REMEASURE_DELAY = 500;

/** Splits a {@link PATTERNS} value into its DE-9IM patterns. */
function parsePattern(value: string): string[] {
  return value.split('|');
}
/** Per-frame drift speed of unpinned queries, radians per second. */
const DRIFT_SPEED = 0.35;
const ZIP_COLOR = [160, 178, 205, 150] as const;

type QueryShape = {
  /** Vertices relative to `center`. */
  offsets: Float32Array;
  center: [number, number];
  /** Drift amplitude in meters (0 keeps the shape fixed until dragged). */
  drift: number;
  phase: number;
  pickRadius: number;
  /** Added to `offsets` (the shape's own center, or the origin for absolute vertices). */
  anchor: readonly [number, number];
};

type Scene = {
  shapes: QueryShape[];
  /** Total vertices per shape. */
  vertexCounts: number[];
  vertexOffsets: Uint32Array;
  positions: Float32Array;
  segmentCount: number;
  segments: Float32Array;
  segmentRows: Uint32Array;
};

/** Returns the bounding box of the ZIP polygons. */
function getBounds(positions: Float32Array): [number, number, number, number] {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < positions.length; index += 2) {
    minX = Math.min(minX, positions[index]);
    maxX = Math.max(maxX, positions[index]);
    minY = Math.min(minY, positions[index + 1]);
    maxY = Math.max(maxY, positions[index + 1]);
  }
  return [minX, minY, maxX, maxY];
}

function makeShape(
  center: [number, number],
  vertices: [number, number][],
  drift: number,
  phase: number,
  anchor?: readonly [number, number]
): QueryShape {
  const offsets = new Float32Array(vertices.length * 2);
  let pickRadius = 0;
  vertices.forEach(([x, y], index) => {
    offsets[index * 2] = x;
    offsets[index * 2 + 1] = y;
    pickRadius = Math.max(pickRadius, Math.hypot(x, y));
  });
  return {offsets, center, drift, phase, pickRadius, anchor: anchor ?? center};
}

function makeDisc(radius: number, count: number): [number, number][] {
  return Array.from({length: count}, (_, index) => {
    const angle = (index / count) * Math.PI * 2;
    return [Math.cos(angle) * radius, Math.sin(angle) * radius] as [number, number];
  });
}

function makeRectangle(halfWidth: number, halfHeight: number): [number, number][] {
  return [
    [-halfWidth, -halfHeight],
    [halfWidth, -halfHeight],
    [halfWidth, halfHeight],
    [-halfWidth, halfHeight]
  ];
}

/** Builds the movable query shapes around the ZIP extent. */
function createScene(kind: QuerySet, zips: SpatialAnalysisPolygons): Scene {
  const [minX, minY, maxX, maxY] = getBounds(zips.polygonPositions);
  const span = Math.min(maxX - minX, maxY - minY);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const at = (fractionX: number, fractionY: number): [number, number] => [
    centerX + fractionX * span,
    centerY + fractionY * span
  ];
  // The smallest ZIP's own shell (copy): `equals` and `coveredBy` hold until it is moved.
  // The ZIP with the fewest vertices: a copy of a detailed ZIP would cost a 700 x 700 edge product
  // against every neighbor that shares its boundary.
  let copyFeature = 0;
  let copyVertexCount = Infinity;
  for (let feature = 0; feature < zips.featureOffsets.length - 1; feature++) {
    const shell = zips.polygonOffsets[zips.featureOffsets[feature]];
    const count = zips.ringOffsets[shell + 1] - zips.ringOffsets[shell];
    if (count < copyVertexCount) {
      copyVertexCount = count;
      copyFeature = feature;
    }
  }
  const copyShell = zips.polygonOffsets[zips.featureOffsets[copyFeature]];
  const firstShellStart = zips.ringOffsets[copyShell];
  const firstShellEnd = zips.ringOffsets[copyShell + 1];
  const zipVertices: [number, number][] = [];
  for (let vertex = firstShellStart; vertex < firstShellEnd; vertex++) {
    zipVertices.push([zips.polygonPositions[vertex * 2], zips.polygonPositions[vertex * 2 + 1]]);
  }
  const shapes: QueryShape[] = [];
  const drift = 0.08 * span;
  if (kind === 'polygons') {
    shapes.push(
      makeShape(at(-0.02, 0.05), makeDisc(0.2 * span, 48), drift, 0),
      makeShape(at(0.2, -0.15), makeRectangle(0.11 * span, 0.11 * span), drift, 1.7),
      makeShape(at(-0.22, 0.2), makeRectangle(0.02 * span, 0.02 * span), drift, 3.1),
      makeShape(at(-0.1, -0.05), makeRectangle(0.03 * span, 0.3 * span), drift, 4.4),
      makeShape(
        at(0.25, 0.25),
        [
          [-0.1, -0.09],
          [0.1, -0.09],
          [0, 0.11]
        ].map(([x, y]) => [x * span, y * span] as [number, number]),
        drift,
        5.2
      )
    );
  } else {
    for (let index = 0; index < 4; index++) {
      const points: [number, number][] = [];
      for (let step = 0; step <= 9; step++) {
        const fraction = step / 9 - 0.5;
        points.push([
          fraction * 0.9 * span,
          Math.sin(fraction * 6 + index * 1.3) * 0.07 * span + (index - 1.5) * 0.12 * span * 0.3
        ]);
      }
      shapes.push(makeShape(at(0, (index - 1.5) * 0.2), points, drift, index * 1.9));
    }
  }
  // The stationary ZIP copy: a polygon for the polygon set, its closed boundary line otherwise.
  const zipCenter: [number, number] = [0, 0];
  for (const [x, y] of zipVertices) {
    zipCenter[0] += x / zipVertices.length;
    zipCenter[1] += y / zipVertices.length;
  }
  // Absolute vertices (anchor at the origin): `anchor + offset + 0` stays bit-exact, so `equals`
  // holds until the copy is moved.
  const zipShapeVertices = zipVertices.slice();
  if (kind === 'lines') zipShapeVertices.push(zipShapeVertices[0]);
  const zipShape = makeShape(zipCenter, zipShapeVertices, 0, 0, [0, 0]);
  zipShape.pickRadius = Math.max(
    ...zipVertices.map(([x, y]) => Math.hypot(x - zipCenter[0], y - zipCenter[1]))
  );
  shapes.push(zipShape);

  const vertexCounts = shapes.map(shape => shape.offsets.length / 2);
  const vertexOffsets = new Uint32Array(shapes.length + 1);
  shapes.forEach((_, row) => {
    vertexOffsets[row + 1] = vertexOffsets[row] + vertexCounts[row];
  });
  const vertexTotal = vertexOffsets[shapes.length];
  const segmentRows: number[] = [];
  shapes.forEach((_, row) => {
    const count = kind === 'polygons' ? vertexCounts[row] : vertexCounts[row] - 1;
    for (let segment = 0; segment < count; segment++) segmentRows.push(row);
  });
  return {
    shapes,
    vertexCounts,
    vertexOffsets,
    positions: new Float32Array(vertexTotal * 2),
    segmentCount: segmentRows.length,
    segments: new Float32Array(segmentRows.length * 4),
    segmentRows: Uint32Array.from(segmentRows)
  };
}

function writeScene(
  scene: Scene,
  kind: QuerySet,
  centers: Float32Array,
  timeSeconds: number,
  pinned: (readonly [number, number] | null)[],
  animate: boolean
): void {
  const {shapes, vertexOffsets, positions, segments} = scene;
  let segment = 0;
  shapes.forEach((shape, row) => {
    const pin = pinned[row];
    let x = shape.center[0];
    let y = shape.center[1];
    if (pin) {
      x = pin[0];
      y = pin[1];
    } else if (animate && shape.drift > 0) {
      const angle = timeSeconds * DRIFT_SPEED + shape.phase;
      x += Math.cos(angle) * shape.drift;
      y += Math.sin(angle * 1.31) * shape.drift;
    }
    centers[row * 2] = x;
    centers[row * 2 + 1] = y;
    const base = vertexOffsets[row];
    const count = shape.offsets.length / 2;
    for (let vertex = 0; vertex < count; vertex++) {
      positions[(base + vertex) * 2] =
        shape.anchor[0] + shape.offsets[vertex * 2] + (x - shape.center[0]);
      positions[(base + vertex) * 2 + 1] =
        shape.anchor[1] + shape.offsets[vertex * 2 + 1] + (y - shape.center[1]);
    }
    const edges = kind === 'polygons' ? count : count - 1;
    for (let edge = 0; edge < edges; edge++) {
      const start = base + edge;
      const end = base + ((edge + 1) % count);
      segments[segment * 4] = positions[start * 2];
      segments[segment * 4 + 1] = positions[start * 2 + 1];
      segments[segment * 4 + 2] = positions[end * 2];
      segments[segment * 4 + 3] = positions[end * 2 + 1];
      segment++;
    }
  });
}

/** Names the kernel family a join resolved to. */
function describeEngine(join: GPUSpatialPredicateJoin): string {
  if (join.usesRelateEngine) return 'relate';
  return join.usesWorkgroupDistance ? 'workgroup distance' : 'fast';
}

type Build = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  prepared: GPUSpatialJoinPrepared;
  engineText: string;
  kind: QuerySet;
  scene: Scene;
  reader: SummaryReader;
  positionsBuffer: Buffer;
  segmentsBuffer: Buffer;
  segmentRowsBuffer: Buffer;
  pairCountBuffer: Buffer;
  matchedFlags: Buffer;
  unmatchedFlags: Buffer;
  rightCounts: Buffer;
  centers: Float32Array;
  pairCapacity: number;
  frames: number;
  /** Latest decoded pair table. */
  pairs: {left: number; right: number; matrix: number}[];
};

export const relateMode: SpatialAnalysisModeDefinition = {
  id: 'relate',
  title: 'Relate',
  contributors: ['GPUSpatialPredicateJoin', 'GPUSpatialJoinPrepared', 'GPUSpatialJoinCandidates'],
  description:
    'Movable query shapes are related to ZIP-code polygons by any DE-9IM predicate on the GPU. ' +
    'Drag a shape, switch predicate or anti join, change the pattern or the dwithin distance live (no rebuild, the graph time updates), and hover a ZIP for its relate matrices.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 11.7},

  async create(context) {
    const zips = await context.data.getSanFranciscoZipCodes();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'relate');
    const projection = new LocalMetricProjection(zips.origin);
    const zipCount = zips.featureOffsets.length - 1;
    const outlineSegmentCount = zips.outlineSegments.length / 4;

    const zipPositions = resources.createBuffer('zip-positions', zips.polygonPositions);
    const zipFeatureOffsets = resources.createBuffer('zip-feature-offsets', zips.featureOffsets);
    const zipPolygonOffsets = resources.createBuffer('zip-polygon-offsets', zips.polygonOffsets);
    const zipRingOffsets = resources.createBuffer('zip-ring-offsets', zips.ringOffsets);
    const outlineSegments = resources.createBuffer('outline-segments', zips.outlineSegments);
    const outlineRows = resources.createBuffer('outline-rows', zips.outlineFeatureRows);

    let predicate: RelatePredicate = 'intersects';
    let pattern = PATTERNS[0].value;
    let distanceMeters = DEFAULT_DISTANCE;
    // Per-frame parameter buffers: they outlive graph rebuilds and are rewritten every encoding.
    const patternParameters = resources.createParameterBuffer(
      'pattern',
      'uint32',
      PATTERN_SLOTS * GPU_SPATIAL_RELATE_PATTERN_WORDS
    );
    const distanceParameters = resources.createParameterBuffer(
      'distance',
      'float32',
      1,
      Float32Array.of(DEFAULT_DISTANCE)
    );
    const writePattern = () =>
      patternParameters.write(packGPUSpatialRelatePattern(parsePattern(pattern), PATTERN_SLOTS));
    writePattern();
    let measuring = false;
    let measureTimer: ReturnType<typeof setTimeout> | undefined;
    let autoMeasureScheduled = false;
    let querySet: QuerySet = 'polygons';
    let engine: 'auto' | 'fast' | 'relate' = 'auto';
    let antiView = false;
    let animate = true;
    let invalidateEveryFrame = false;
    let destroyed = false;
    let dragged = -1;
    let grabOffset: [number, number] = [0, 0];
    const pinned: (readonly [number, number] | null)[] = [];

    const buildScene = (): Build => {
      const scene = createScene(querySet, zips);
      const leftCount = scene.shapes.length;
      const build = new SpatialAnalysisResources(device, `relate-${querySet}-${predicate}`);
      const positionsBuffer = build.createBuffer('positions', scene.positions);
      const vertexOffsets = build.createBuffer('vertex-offsets', scene.vertexOffsets);
      const unitOffsets = build.createBuffer(
        'unit-offsets',
        Uint32Array.from({length: leftCount + 1}, (_, row) => row)
      );
      const segmentsBuffer = build.createBuffer('segments', scene.segments);
      const segmentRowsBuffer = build.createBuffer('segment-rows', scene.segmentRows);
      const matchedFlags = build.createBuffer('matched-flags', leftCount * 4);
      const unmatchedFlags = build.createBuffer('unmatched-flags', leftCount * 4);
      const rightCounts = build.createBuffer('right-counts', zipCount * 4);
      const pairCapacity = Math.max(256, leftCount * zipCount);
      const candidateCapacity = Math.max(1024, leftCount * zipCount);
      const leftIds = build.createBuffer('left-ids', pairCapacity * 4);
      const rightIds = build.createBuffer('right-ids', pairCapacity * 4);
      const pairCount = build.createBuffer('pair-count', 4);
      const pairOverflow = build.createBuffer('pair-overflow', 4);
      const pairTotal = build.createBuffer('pair-total', 4);
      const relateBuffer = build.createBuffer('relate', pairCapacity * 4);
      const uncertain = build.createBuffer('uncertain', 4);
      const candidateTotal = build.createBuffer('candidate-total', 4);
      const antiIds = build.createBuffer('anti-ids', leftCount * 4);
      const antiCount = build.createBuffer('anti-count', 4);
      const antiOverflow = build.createBuffer('anti-overflow', 4);
      const candidateLeft = build.createBuffer('candidate-left', candidateCapacity * 4);
      const candidateRight = build.createBuffer('candidate-right', candidateCapacity * 4);
      const candidatePairCount = build.createBuffer('candidate-pair-count', 4);
      const candidateOverflow = build.createBuffer('candidate-overflow', 4);

      const graph = new GPUCommandGraph<void>(device, {id: `relate-${querySet}-${predicate}`});
      const view = <Format extends 'uint32' | 'float32x2'>(
        name: string,
        buffer: Buffer,
        format: Format,
        length: number
      ) => importGraphBuffer(graph, name, buffer, format, length);
      const left: GPUSpatialJoinGeometry =
        querySet === 'polygons'
          ? {
              kind: 'polygons',
              positions: view(
                'left-positions',
                positionsBuffer,
                'float32x2',
                scene.positions.length / 2
              ),
              featureOffsets: view('left-feature-offsets', unitOffsets, 'uint32', leftCount + 1),
              polygonOffsets: view('left-polygon-offsets', unitOffsets, 'uint32', leftCount + 1),
              ringOffsets: view('left-ring-offsets', vertexOffsets, 'uint32', leftCount + 1)
            }
          : {
              kind: 'lines',
              positions: view(
                'left-positions',
                positionsBuffer,
                'float32x2',
                scene.positions.length / 2
              ),
              lineOffsets: view('left-line-offsets', vertexOffsets, 'uint32', leftCount + 1)
            };
      const right: GPUSpatialJoinGeometry = {
        kind: 'polygons',
        positions: view(
          'zip-positions',
          zipPositions,
          'float32x2',
          zips.polygonPositions.length / 2
        ),
        featureOffsets: view('zip-feature-offsets', zipFeatureOffsets, 'uint32', zipCount + 1),
        polygonOffsets: view(
          'zip-polygon-offsets',
          zipPolygonOffsets,
          'uint32',
          zips.polygonOffsets.length
        ),
        ringOffsets: view('zip-ring-offsets', zipRingOffsets, 'uint32', zips.ringOffsets.length)
      };
      // One handle for every join in the graph: the right-side BVH is built once and reused.
      const prepared = new GPUSpatialJoinPrepared({id: 'relate-prepared', geometry: right});
      graph.add(prepared);
      const common = {
        left,
        right,
        predicate,
        // Per-frame views: the pattern and the distance are never compiled in.
        ...(predicate === 'relate' ? {pattern: patternParameters.importToGraph(graph)} : {}),
        ...(predicate === 'dwithin' ? {distance: distanceParameters.importToGraph(graph)} : {}),
        candidateCapacity,
        prepared
      };
      const innerJoin = new GPUSpatialPredicateJoin({
        ...common,
        id: 'relate-inner',
        pairs: {
          leftIds: view('left-ids', leftIds, 'uint32', pairCapacity),
          rightIds: view('right-ids', rightIds, 'uint32', pairCapacity),
          count: view('pair-count', pairCount, 'uint32', 1),
          overflow: view('pair-overflow', pairOverflow, 'uint32', 1),
          requiredCount: view('pair-total', pairTotal, 'uint32', 1)
        },
        // dwithin has no DE-9IM matrix.
        ...(predicate === 'dwithin'
          ? {}
          : {relate: view('relate', relateBuffer, 'uint32', pairCapacity)}),
        uncertainCount: view('uncertain', uncertain, 'uint32', 1),
        candidateCount: view('candidate-total', candidateTotal, 'uint32', 1)
      });
      graph.add(innerJoin);
      const antiJoin = new GPUSpatialPredicateJoin({
        ...common,
        // 'fast' exists only for the four legacy predicates.
        ...(['intersects', 'contains', 'within', 'dwithin'].includes(predicate) ? {engine} : {}),
        id: 'relate-anti',
        how: 'anti',
        unmatched: {
          ids: view('anti-ids', antiIds, 'uint32', leftCount),
          count: view('anti-count', antiCount, 'uint32', 1),
          overflow: view('anti-overflow', antiOverflow, 'uint32', 1)
        }
      });
      graph.add(antiJoin);
      graph.add(
        new GPUSpatialJoinCandidates({
          id: 'relate-candidates',
          left,
          right,
          prepared,
          pairs: {
            leftIds: view('candidate-left', candidateLeft, 'uint32', candidateCapacity),
            rightIds: view('candidate-right', candidateRight, 'uint32', candidateCapacity),
            count: view('candidate-pair-count', candidatePairCount, 'uint32', 1),
            overflow: view('candidate-overflow', candidateOverflow, 'uint32', 1)
          }
        })
      );
      const compiled = build.track(graph.compile());
      build.track({destroy: () => prepared.destroy()});

      const result: Build = {
        resources: build,
        compiled,
        prepared,
        engineText: `inner ${describeEngine(innerJoin)}${predicate === 'dwithin' ? '' : ' (matrix output)'}, anti ${describeEngine(antiJoin)}`,
        kind: querySet,
        scene,
        reader: undefined as unknown as SummaryReader,
        positionsBuffer,
        segmentsBuffer,
        segmentRowsBuffer,
        pairCountBuffer: pairCount,
        matchedFlags,
        unmatchedFlags,
        rightCounts,
        centers: new Float32Array(leftCount * 2),
        pairCapacity,
        frames: 0,
        pairs: []
      };
      // Header words: count, overflow, total, uncertain, candidates, anti count, anti overflow,
      // candidate pair count, candidate overflow; then pairs, relate and anti ids.
      result.reader = new SummaryReader(
        build,
        'relate',
        [
          {buffer: pairCount, size: 4},
          {buffer: pairOverflow, size: 4},
          {buffer: pairTotal, size: 4},
          {buffer: uncertain, size: 4},
          {buffer: candidateTotal, size: 4},
          {buffer: antiCount, size: 4},
          {buffer: antiOverflow, size: 4},
          {buffer: candidatePairCount, size: 4},
          {buffer: candidateOverflow, size: 4},
          {buffer: leftIds, size: pairCapacity * 4},
          {buffer: rightIds, size: pairCapacity * 4},
          {buffer: relateBuffer, size: pairCapacity * 4},
          {buffer: antiIds, size: leftCount * 4}
        ],
        bytes => handleSummary(result, bytes)
      );
      build.track({destroy: () => result.reader.stop()});
      return result;
    };

    let active: Build;

    const pairsReadout = (() => {
      const handles = {
        pairs: context.controls.addReadout('Matched pairs', '...'),
        anti: context.controls.addReadout('Anti join (unmatched queries)', '...'),
        candidates: context.controls.addReadout('Bounding-box candidates', '...'),
        matrices: context.controls.addReadout('Relate matrices', '...'),
        flags: context.controls.addReadout('Overflow / uncertain', '...'),
        engine: context.controls.addReadout('Engine path', '...'),
        timing: context.controls.addReadout('Graph time (joins + candidates)', '...'),
        reuse: context.controls.addReadout('Right-side BVH builds', '...')
      };
      return handles;
    })();

    /** The predicate with its per-frame parameter, for readouts and the tooltip. */
    const describePredicate = () =>
      predicate === 'relate'
        ? `pattern ${parsePattern(pattern).join(' | ')}`
        : predicate === 'dwithin'
          ? `dwithin ${distanceMeters} m`
          : predicate;

    const handleSummary = (build: Build, bytes: ArrayBuffer) => {
      if (destroyed || build !== active) return;
      const words = new Uint32Array(bytes);
      const capacity = build.pairCapacity;
      const leftCount = build.scene.shapes.length;
      const count = Math.min(words[0], capacity);
      const leftIds = words.subarray(9, 9 + capacity);
      const rightIds = words.subarray(9 + capacity, 9 + capacity * 2);
      const matrices = words.subarray(9 + capacity * 2, 9 + capacity * 3);
      const antiIds = words.subarray(9 + capacity * 3, 9 + capacity * 3 + leftCount);
      const matchedFlags = new Float32Array(leftCount);
      const rightCounts = new Float32Array(zipCount);
      const histogram = new Map<string, number>();
      build.pairs = [];
      for (let slot = 0; slot < count; slot++) {
        const leftRow = leftIds[slot];
        const rightRow = rightIds[slot];
        if (leftRow >= leftCount || rightRow >= zipCount) continue;
        matchedFlags[leftRow] = 1;
        rightCounts[rightRow] += 1;
        const matrix = formatGPUSpatialRelate(matrices[slot] & 0x3ffff);
        histogram.set(matrix, (histogram.get(matrix) ?? 0) + 1);
        build.pairs.push({left: leftRow, right: rightRow, matrix: matrices[slot]});
      }
      const unmatchedFlags = new Float32Array(leftCount);
      const antiCount = Math.min(words[5], leftCount);
      for (let slot = 0; slot < antiCount; slot++) {
        if (antiIds[slot] < leftCount) unmatchedFlags[antiIds[slot]] = 1;
      }
      build.matchedFlags.write(matchedFlags);
      build.unmatchedFlags.write(unmatchedFlags);
      build.rightCounts.write(rightCounts);
      let complement = true;
      for (let row = 0; row < leftCount; row++) {
        if (matchedFlags[row] === unmatchedFlags[row]) complement = false;
      }
      const topMatrices = [...histogram.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([matrix, matrixCount]) => `${matrix} x${matrixCount}`)
        .join(', ');
      pairsReadout.pairs.setValue(
        `${formatCount(words[2])} pairs over ${formatCount(matchedFlags.reduce((a, b) => a + b, 0))}` +
          ` of ${leftCount} queries, ${formatCount(rightCounts.filter(value => value > 0).length)}` +
          ` of ${zipCount} ZIPs`
      );
      pairsReadout.anti.setValue(
        `${formatCount(words[5])} of ${leftCount}; ${complement ? 'equals' : 'DIFFERS FROM'} ` +
          'the complement of the pairs'
      );
      pairsReadout.candidates.setValue(
        `${formatCount(words[4])} candidate pairs, ${formatCount(words[2])} pass ${describePredicate()}`
      );
      pairsReadout.matrices.setValue(
        predicate === 'dwithin' ? 'none: dwithin has no matrix' : topMatrices || 'none'
      );
      pairsReadout.flags.setValue(
        `${words[1] || words[6] || words[8] ? 'OVERFLOW' : 'no'} / ${formatCount(words[3])} pairs`
      );
      pairsReadout.engine.setValue(build.engineText);
      const builds = build.prepared.encodedBuildCount;
      pairsReadout.reuse.setValue(
        `${builds} in ${formatCount(build.frames)} frames` +
          (invalidateEveryFrame ? ' (rebuilding every frame)' : ' (reused)')
      );
    };

    /** Times the active graph outside the frame (GPU timestamps when available). */
    const measureGraph = async () => {
      if (measuring || destroyed) return;
      measuring = true;
      const build = active;
      try {
        const timing = await measureCompiledGraph(device, build.compiled, {
          parameters: undefined,
          completionBuffer: build.pairCountBuffer
        });
        if (!destroyed && build === active) {
          pairsReadout.timing.setValue(
            `${formatCompiledGraphTiming(timing)} at ${describePredicate()}`
          );
        }
      } catch {
        // The build was released, or the device was destroyed, while measuring.
      } finally {
        measuring = false;
      }
    };
    /** Re-times shortly after the last per-frame parameter change. */
    const scheduleMeasure = () => {
      clearTimeout(measureTimer);
      measureTimer = setTimeout(() => void measureGraph(), REMEASURE_DELAY);
    };

    const rebuild = () => {
      const previous = active;
      active = buildScene();
      pinned.length = 0;
      autoMeasureScheduled = false;
      if (previous) {
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            if (!destroyed) previous.resources.destroy();
          })
        );
      }
      context.updateLayers();
    };

    context.controls.addSelect({
      label: 'Predicate (compile-time: rebuilds graph)',
      options: PREDICATES.map(value => ({value, label: value})),
      value: predicate,
      onChange: value => {
        predicate = value;
        rebuild();
      }
    });
    context.controls.addSelect({
      label: 'DE-9IM pattern for predicate "relate" (per-frame: no rebuild)',
      options: PATTERNS,
      value: pattern,
      onChange: value => {
        pattern = value;
        writePattern();
        if (predicate === 'relate') scheduleMeasure();
      }
    });
    context.controls.addSlider({
      label: 'dwithin distance (per-frame parameter: no rebuild)',
      min: 0,
      max: MAXIMUM_DISTANCE,
      step: 50,
      value: distanceMeters,
      format: value => `${value} m`,
      onChange: value => {
        distanceMeters = value;
        if (predicate === 'dwithin') scheduleMeasure();
      }
    });
    context.controls.addSelect<'auto' | 'fast' | 'relate'>({
      label: 'Anti join engine for intersects/contains/within (compile-time)',
      options: [
        {value: 'auto', label: 'auto'},
        {value: 'fast', label: 'fast (short-circuit kernel)'},
        {value: 'relate', label: 'relate (DE-9IM workgroup engine)'}
      ],
      value: engine,
      onChange: value => {
        engine = value;
        rebuild();
      }
    });
    context.controls.addSelect<QuerySet>({
      label: 'Query geometry (compile-time: rebuilds graph)',
      options: [
        {value: 'polygons', label: 'Polygons (discs, squares, triangle, a ZIP copy)'},
        {value: 'lines', label: 'Polylines (routes, a ZIP boundary)'}
      ],
      value: querySet,
      onChange: value => {
        querySet = value;
        rebuild();
      }
    });
    context.controls.addToggle({
      label: 'Anti join view: show queries with no match (no recompile)',
      value: antiView,
      onChange: value => {
        antiView = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Drift the queries (BVH is reused)',
      value: animate,
      onChange: value => {
        animate = value;
      }
    });
    context.controls.addToggle({
      label: 'Invalidate the prepared BVH every frame (contrast)',
      value: invalidateEveryFrame,
      onChange: value => {
        invalidateEveryFrame = value;
      }
    });
    context.controls.addButton({
      label: 'Measure graph time',
      onClick: () => void measureGraph()
    });
    context.controls.addButton({
      label: 'Release dragged shapes',
      onClick: () => {
        pinned.length = 0;
      }
    });
    context.controls.addLegend({
      title: 'Queries and ZIPs',
      entries: [
        {color: [80, 230, 140], label: 'Query with at least one match'},
        {color: [255, 70, 70], label: 'Anti view: query with no match'},
        {color: [253, 231, 37], label: 'ZIP matched by queries (thick; hue = count)'},
        {color: [160, 178, 205], label: 'ZIP polygon'}
      ]
    });
    context.controls.addNote(
      'The ZIP index is a GPUSpatialJoinPrepared handle shared by the predicate join, the anti join and the candidate stage. ' +
        'It builds on the first frame only; tick the contrast toggle to rebuild it every frame. ' +
        'Pattern and distance are parameter buffers written every frame; only the predicate, the engine and the query kind rebuild. ' +
        'The graph time is measured outside the frame after the first frames and shortly after a pattern or distance change.'
    );
    context.controls.addReadout(
      'ZIP rings',
      `full resolution, ${formatCount(zips.polygonPositions.length / 2)} vertices`
    );
    context.controls.addReadout('Data', zips.attribution);

    active = buildScene();

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [active.compiled],
      encode(commandEncoder, frame) {
        const build = active;
        writeScene(build.scene, build.kind, build.centers, frame.timeSeconds, pinned, animate);
        build.positionsBuffer.write(build.scene.positions);
        build.segmentsBuffer.write(build.scene.segments);
        if (invalidateEveryFrame) build.prepared.invalidate();
        writePattern();
        distanceParameters.write(Float32Array.of(distanceMeters));
        build.compiled.encode(commandEncoder, {parameters: undefined});
        build.frames++;
        build.reader.markStale();
        build.reader.flush(commandEncoder);
        if (!autoMeasureScheduled && build.frames >= AUTO_MEASURE_FRAME) {
          autoMeasureScheduled = true;
          void measureGraph();
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [zips.origin[0], zips.origin[1], 0];
        const layers: Layer[] = [
          new SpatialAnalysisSegmentLayer({
            id: 'relate-zip-outline',
            coordinateOrigin,
            segments: outlineSegments,
            instanceCount: outlineSegmentCount,
            color: ZIP_COLOR,
            widthPixels: 1.5
          })
        ];
        if (!antiView) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'relate-zip-matched',
              coordinateOrigin,
              segments: outlineSegments,
              instanceCount: outlineSegmentCount,
              values: active.rightCounts,
              valueFormat: 'float32',
              valueIndices: outlineRows,
              colormap: 'viridis',
              valueRange: [0, Math.max(1, active.scene.shapes.length)],
              discardAtOrBelow: 0.5,
              widthPixels: 7,
              opacity: 0.9
            })
          );
        }
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: `relate-queries-${antiView ? 'anti' : 'matched'}`,
            coordinateOrigin,
            segments: active.segmentsBuffer,
            instanceCount: active.scene.segmentCount,
            values: antiView ? active.unmatchedFlags : active.matchedFlags,
            valueFormat: 'float32',
            valueIndices: active.segmentRowsBuffer,
            colormap: 'mask',
            color: antiView ? [255, 70, 70, 255] : [80, 230, 140, 255],
            noDataColor: antiView ? [140, 150, 175, 110] : [255, 255, 255, 190],
            widthPixels: 3.5
          })
        );
        return layers;
      },
      onDragStart(event) {
        if (!event.coordinate) return false;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        const centers = active.centers;
        let best = -1;
        let bestDistance = Infinity;
        active.scene.shapes.forEach((shape, row) => {
          const distance = Math.hypot(centers[row * 2] - x, centers[row * 2 + 1] - y);
          if (distance <= Math.max(shape.pickRadius, 300) && distance < bestDistance) {
            best = row;
            bestDistance = distance;
          }
        });
        if (best < 0) return false;
        dragged = best;
        grabOffset = [centers[best * 2] - x, centers[best * 2 + 1] - y];
        pinned[best] = [centers[best * 2], centers[best * 2 + 1]];
        context.setMapDragEnabled(false);
        return true;
      },
      onDrag(event) {
        if (dragged < 0 || !event.coordinate) return;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        pinned[dragged] = [x + grabOffset[0], y + grabOffset[1]];
      },
      onDragEnd() {
        dragged = -1;
        context.setMapDragEnabled(true);
      },
      getTooltip(event) {
        if (!event.coordinate) return null;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        const row = findContainingFeature(zips, x, y);
        if (row < 0) return null;
        const label = describePredicate();
        const lines = active.pairs
          .filter(pair => pair.right === row)
          .slice(0, 6)
          .map(pair =>
            predicate === 'dwithin'
              ? `query ${pair.left}`
              : `query ${pair.left}: ${formatGPUSpatialRelate(pair.matrix & 0x3ffff)}`
          );
        return (
          `ZIP ${zips.featureNames[row] ?? zips.featureIds[row]}\n` +
          (lines.length ? `${label} matches (DE-9IM)\n${lines.join('\n')}` : `no query ${label}`)
        );
      },
      destroy() {
        destroyed = true;
        clearTimeout(measureTimer);
        active.resources.destroy();
        resources.destroy();
      }
    };
    return instance;
  }
};
