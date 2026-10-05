// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Raster join: `GPUPolygonRasterization` scan-converts the San Francisco ZIP-code polygons into a
 * zone raster whose placement is a four-float `extent` parameter, and `GPURasterJoin` bins every
 * bike-parking point into that raster (O(1) per point) to count points and sum parking spaces per
 * ZIP. The extent follows the viewport, so panning and zooming rewrite one buffer and re-encode
 * the same compiled graph; the metric and the cell scale are buffer writes too.
 *
 * The join is approximate only in boundary cells. The recipe flags them, reports the points they
 * hold per zone (`boundaryCounts`) and per point (`pointBoundaryMask`), and this mode checks the
 * bound live against an exact `GPUPointInPolygonJoin` of the same points: every point whose zone
 * differs must sit in a boundary cell, and every zone's exact count must lie inside the interval
 * the recipe documents. A small readback (per-zone columns plus three per-point columns) feeds
 * the readouts; it is requested only after the extent changed.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {getGPUPolygonRasterizationExtentValues, GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH, GPU_POLYGON_RASTERIZATION_NO_ZONE, GPUPolygonRasterization, GPURasterJoin} from '@luma.gl/experimental/gpu-raster';
import {GPUPointInPolygonJoin} from '@luma.gl/experimental/geospatial';
import {importGraphBuffer} from '@luma.gl/experimental/UNRESOLVED';
import {LocalMetricProjection} from '../map-graphs-data';
import {
  MapGraphsPointLayer,
  MapGraphsRasterLayer,
  MapGraphsSegmentLayer
} from '../map-graphs-layers';
import type {
  MapGraphsFrame,
  MapGraphsModeDefinition,
  MapGraphsModeInstance,
  MapGraphsPointerEvent
} from '../map-graphs-mode';
import {formatCount, getViewportMetricBounds, MapGraphsResources} from '../map-graphs-resources';
import {
  addRasterJoinDisplayPass,
  findContainingFeature,
  RASTER_JOIN_HIDDEN,
  RASTER_JOIN_METRIC
} from './raster-join-layers';
import {formatCompiledGraphTiming, formatSpeedup, measureCompiledGraph} from './vector-timing';

/** Raster size in cells. Compile-time. */
const RASTER_WIDTH = 640;
const RASTER_HEIGHT = 448;
/** Maximum (edge, row) crossings per encoding. Compile-time; `overflow` and `crossingCount` report. */
const CROSSING_CAPACITY = 1 << 16;
/** Fraction of the city bounds added on each side by the fixed extent. */
const FIXED_EXTENT_MARGIN = 0.03;
/** Frames between readback attempts while the extent keeps changing. */
const READBACK_INTERVAL_FRAMES = 12;
/** Frames after creation before the automatic timing runs. */
const AUTO_MEASURE_FRAME = 60;
const BOUNDARY_COLOR = [255, 140, 30, 235] as const;
const ZONE_PALETTE = [
  [78, 201, 255, 175],
  [255, 148, 72, 175],
  [189, 122, 255, 175],
  [87, 235, 168, 175],
  [255, 105, 168, 175],
  [245, 220, 87, 175],
  [107, 158, 255, 175],
  [255, 92, 92, 175]
] as const;

type Metric = keyof typeof RASTER_JOIN_METRIC;
type Display = 'choropleth' | 'zones';
type ExtentMode = 'viewport' | 'city';

/** Everything one readback carries, split into typed columns. */
type JoinSnapshot = {
  /** Raster placement `[originX, originY, cellWidth, cellHeight]` the columns belong to. */
  extent: Float32Array;
  counts: Uint32Array;
  boundaryCounts: Uint32Array;
  sums: Float32Array;
  pointZones: Uint32Array;
  pointBoundaryMask: Uint32Array;
  exactZones: Uint32Array;
  unassignedBoundaryCount: number;
  outsideCount: number;
  overflow: number;
  crossingCount: number;
};

/** Raster join demo: ZIP polygons rasterized per frame, bike parking joined by cell lookup. */
export const rasterJoinMode: MapGraphsModeDefinition = {
  id: 'raster-join',
  title: 'Raster join',
  recipes: ['GPUPolygonRasterization', 'GPURasterJoin', 'GPUPointInPolygonJoin'],
  description:
    'ZIP polygons are scan-converted on the GPU into a zone raster that follows the viewport, and ' +
    'bike parking is joined to it by cell lookup. Orange cells and points mark the only places ' +
    'the approximate join can differ from an exact point-in-polygon join, which runs alongside.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 11.6},

  async create(context) {
    const [parking, zips] = await Promise.all([
      context.data.getSanFranciscoBikeParking(),
      context.data.getSanFranciscoZipCodes()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(zips.origin);
    const resources = new MapGraphsResources(device, 'raster-join');
    const pointCount = parking.positions.length / 2;
    const zoneCount = zips.featureOffsets.length - 1;
    const cellCount = RASTER_WIDTH * RASTER_HEIGHT;
    const vertexCount = zips.polygonPositions.length / 2;

    // --- Buffers -------------------------------------------------------------------------------
    const positions = resources.createBuffer('positions', parking.positions);
    const spaces = resources.createBuffer('spaces', parking.spaces);
    const polygonPositions = resources.createBuffer('polygon-positions', zips.polygonPositions);
    const featureOffsets = resources.createBuffer('feature-offsets', zips.featureOffsets);
    const polygonOffsets = resources.createBuffer('polygon-offsets', zips.polygonOffsets);
    const ringOffsets = resources.createBuffer('ring-offsets', zips.ringOffsets);
    const outlineSegments = resources.createBuffer('outline-segments', zips.outlineSegments);

    const zones = resources.createBuffer('zones', cellCount * 4);
    const boundary = resources.createBuffer('boundary', cellCount * 4);
    const display = resources.createBuffer('display', cellCount * 4);
    const overflow = resources.createBuffer('overflow', 4);
    const crossingCount = resources.createBuffer('crossing-count', 4);
    const counts = resources.createBuffer('counts', zoneCount * 4);
    const sums = resources.createBuffer('sums', zoneCount * 4);
    const boundaryCounts = resources.createBuffer('boundary-counts', zoneCount * 4);
    const unassignedBoundaryCount = resources.createBuffer('unassigned-boundary-count', 4);
    const outsideCount = resources.createBuffer('outside-count', 4);
    const pointZones = resources.createBuffer('point-zones', pointCount * 4);
    const pointBoundaryMask = resources.createBuffer('point-boundary-mask', pointCount * 4);
    const exactZones = resources.createBuffer('exact-zones', pointCount * 4);
    const exactCounts = resources.createBuffer('exact-counts', zoneCount * 4);
    const exactOverflow = resources.createBuffer('exact-overflow', 4);
    const exactCandidates = resources.createBuffer('exact-candidates', 4);
    // Layer bounds `[minX, minY, maxX, maxY]` of the raster, rewritten with the extent.
    const rasterBounds = resources.createBuffer('raster-bounds', 16);

    const extentParameter = resources.createParameterBuffer(
      'extent',
      'float32',
      GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH
    );
    const metricParameter = resources.createParameterBuffer('metric', 'uint32', 1);

    // --- Graphs --------------------------------------------------------------------------------
    const rasterGraph = new GPUCommandGraph<void>(device, {id: 'raster-join'});
    const extentView = extentParameter.importToGraph(rasterGraph);
    const zonesView = importGraphBuffer(rasterGraph, 'zones', zones, 'uint32', cellCount);
    const boundaryView = importGraphBuffer(rasterGraph, 'boundary', boundary, 'uint32', cellCount);
    const countsView = importGraphBuffer(rasterGraph, 'counts', counts, 'uint32', zoneCount);
    const sumsView = importGraphBuffer(rasterGraph, 'sums', sums, 'float32', zoneCount);
    const boundaryCountsView = importGraphBuffer(
      rasterGraph,
      'boundary-counts',
      boundaryCounts,
      'uint32',
      zoneCount
    );
    rasterGraph.add(
      new GPUPolygonRasterization({
        id: 'rasterize',
        width: RASTER_WIDTH,
        height: RASTER_HEIGHT,
        extent: extentView,
        polygonPositions: importGraphBuffer(
          rasterGraph,
          'polygon-positions',
          polygonPositions,
          'float32x2',
          vertexCount
        ),
        featureOffsets: importGraphBuffer(
          rasterGraph,
          'feature-offsets',
          featureOffsets,
          'uint32',
          zips.featureOffsets.length
        ),
        polygonOffsets: importGraphBuffer(
          rasterGraph,
          'polygon-offsets',
          polygonOffsets,
          'uint32',
          zips.polygonOffsets.length
        ),
        ringOffsets: importGraphBuffer(
          rasterGraph,
          'ring-offsets',
          ringOffsets,
          'uint32',
          zips.ringOffsets.length
        ),
        crossingCapacity: CROSSING_CAPACITY,
        zones: zonesView,
        boundary: boundaryView,
        overflow: importGraphBuffer(rasterGraph, 'overflow', overflow, 'uint32', 1),
        crossingCount: importGraphBuffer(rasterGraph, 'crossing-count', crossingCount, 'uint32', 1)
      })
    );
    rasterGraph.add(
      new GPURasterJoin({
        id: 'join',
        width: RASTER_WIDTH,
        height: RASTER_HEIGHT,
        extent: extentView,
        points: importGraphBuffer(rasterGraph, 'points', positions, 'float32x2', pointCount),
        values: importGraphBuffer(rasterGraph, 'spaces', spaces, 'float32', pointCount),
        zones: zonesView,
        boundary: boundaryView,
        zoneCount,
        output: {
          counts: countsView,
          sums: sumsView,
          boundaryCounts: boundaryCountsView,
          unassignedBoundaryCount: importGraphBuffer(
            rasterGraph,
            'unassigned-boundary-count',
            unassignedBoundaryCount,
            'uint32',
            1
          ),
          outsideCount: importGraphBuffer(rasterGraph, 'outside-count', outsideCount, 'uint32', 1),
          pointZones: importGraphBuffer(
            rasterGraph,
            'point-zones',
            pointZones,
            'uint32',
            pointCount
          ),
          pointBoundaryMask: importGraphBuffer(
            rasterGraph,
            'point-boundary-mask',
            pointBoundaryMask,
            'uint32',
            pointCount
          )
        }
      })
    );
    addRasterJoinDisplayPass(rasterGraph, {
      id: 'display',
      cellCount,
      zoneCount,
      zones: zonesView,
      counts: countsView,
      sums: sumsView,
      boundaryCounts: boundaryCountsView,
      metric: metricParameter.importToGraph(rasterGraph),
      display: importGraphBuffer(rasterGraph, 'display', display, 'float32', cellCount)
    });
    const rasterCompiled: CompiledGPUCommandGraph<void> = resources.track(rasterGraph.compile());

    // The existing exact join over the same points and polygons. The points never move, so it is
    // encoded once and serves as the reference for the live error check.
    const exactGraph = new GPUCommandGraph<void>(device, {id: 'raster-join-exact'});
    exactGraph.add(
      new GPUPointInPolygonJoin({
        id: 'exact-join',
        points: importGraphBuffer(exactGraph, 'points', positions, 'float32x2', pointCount),
        polygonPositions: importGraphBuffer(
          exactGraph,
          'polygon-positions',
          polygonPositions,
          'float32x2',
          vertexCount
        ),
        featureOffsets: importGraphBuffer(
          exactGraph,
          'feature-offsets',
          featureOffsets,
          'uint32',
          zips.featureOffsets.length
        ),
        polygonOffsets: importGraphBuffer(
          exactGraph,
          'polygon-offsets',
          polygonOffsets,
          'uint32',
          zips.polygonOffsets.length
        ),
        ringOffsets: importGraphBuffer(
          exactGraph,
          'ring-offsets',
          ringOffsets,
          'uint32',
          zips.ringOffsets.length
        ),
        candidateCapacity: Math.max(1024, pointCount * 4),
        pointFeatureIds: importGraphBuffer(
          exactGraph,
          'exact-zones',
          exactZones,
          'uint32',
          pointCount
        ),
        featureCounts: importGraphBuffer(
          exactGraph,
          'exact-counts',
          exactCounts,
          'uint32',
          zoneCount
        ),
        overflow: importGraphBuffer(exactGraph, 'exact-overflow', exactOverflow, 'uint32', 1),
        candidateCount: importGraphBuffer(
          exactGraph,
          'exact-candidates',
          exactCandidates,
          'uint32',
          1
        )
      })
    );
    const exactCompiled: CompiledGPUCommandGraph<void> = resources.track(exactGraph.compile());

    const summaryWords = zoneCount * 3 + pointCount * 3 + 4;
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'raster-join-summary', byteLength: summaryWords * 4})
    );

    // --- Extent --------------------------------------------------------------------------------
    let polygonBounds: [number, number, number, number] = [
      Infinity,
      Infinity,
      -Infinity,
      -Infinity
    ];
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const x = zips.polygonPositions[vertex * 2];
      const y = zips.polygonPositions[vertex * 2 + 1];
      polygonBounds = [
        Math.min(polygonBounds[0], x),
        Math.min(polygonBounds[1], y),
        Math.max(polygonBounds[2], x),
        Math.max(polygonBounds[3], y)
      ];
    }
    const cityBounds: [number, number, number, number] = [
      polygonBounds[0] - (polygonBounds[2] - polygonBounds[0]) * FIXED_EXTENT_MARGIN,
      polygonBounds[1] - (polygonBounds[3] - polygonBounds[1]) * FIXED_EXTENT_MARGIN,
      polygonBounds[2] + (polygonBounds[2] - polygonBounds[0]) * FIXED_EXTENT_MARGIN,
      polygonBounds[3] + (polygonBounds[3] - polygonBounds[1]) * FIXED_EXTENT_MARGIN
    ];

    let metric: Metric = 'count';
    let displayKind: Display = 'choropleth';
    let extentMode: ExtentMode = 'viewport';
    let cellScale = 1;
    let showBoundary = true;
    let dirty = true;
    let destroyed = false;
    let readbackWanted = false;
    let readbackPending = false;
    let lastReadbackFrame = -READBACK_INTERVAL_FRAMES;
    let encodedFrames = 0;
    let autoMeasured = false;
    let measuring = false;
    let snapshot: JoinSnapshot | null = null;
    let colorMaximum = 1;
    const extentValues = new Float32Array(GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH);

    /** Square cells that cover `bounds` (scaled by `cellScale`), centered on it. */
    const writeExtent = (bounds: readonly [number, number, number, number]): boolean => {
      const cellSize =
        Math.max((bounds[2] - bounds[0]) / RASTER_WIDTH, (bounds[3] - bounds[1]) / RASTER_HEIGHT) *
          cellScale || 1;
      const originX = (bounds[0] + bounds[2]) / 2 - (RASTER_WIDTH * cellSize) / 2;
      const originY = (bounds[1] + bounds[3]) / 2 - (RASTER_HEIGHT * cellSize) / 2;
      const next = getGPUPolygonRasterizationExtentValues(originX, originY, cellSize, cellSize);
      const changed = next.some((value, index) => value !== extentValues[index]);
      if (!changed) return false;
      extentValues.set(next);
      extentParameter.write(extentValues);
      rasterBounds.write(
        Float32Array.of(
          originX,
          originY,
          originX + RASTER_WIDTH * cellSize,
          originY + RASTER_HEIGHT * cellSize
        )
      );
      cellReadout.setValue(`${cellSize.toFixed(1)} m (${RASTER_WIDTH} x ${RASTER_HEIGHT} cells)`);
      return true;
    };
    const writeMetric = () => {
      metricParameter.write(Uint32Array.of(RASTER_JOIN_METRIC[metric]));
      dirty = true;
    };

    // --- Controls ------------------------------------------------------------------------------
    context.controls.addSelect<Display>({
      label: 'Show',
      options: [
        {value: 'choropleth', label: 'Choropleth of the raster join'},
        {value: 'zones', label: 'Zone raster (cell to ZIP row)'}
      ],
      value: displayKind,
      onChange: value => {
        displayKind = value;
        context.updateLayers();
      }
    });
    context.controls.addSelect<Metric>({
      label: 'Metric (per-frame buffer write)',
      options: [
        {value: 'count', label: 'Bike-parking points per ZIP'},
        {value: 'sum', label: 'Parking spaces per ZIP (sum)'},
        {value: 'boundaryShare', label: 'Boundary-point share (error bound)'}
      ],
      value: metric,
      onChange: value => {
        metric = value;
        writeMetric();
        updateColorMaximum();
        context.updateLayers();
      }
    });
    context.controls.addSelect<ExtentMode>({
      label: 'Raster extent (per-frame buffer write)',
      options: [
        {value: 'viewport', label: 'Follow the viewport'},
        {value: 'city', label: 'Whole city (fixed)'}
      ],
      value: extentMode,
      onChange: value => {
        extentMode = value;
        dirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Cell size (x the fit-to-extent size)',
      min: 1,
      max: 8,
      step: 0.5,
      value: cellScale,
      format: value => `${value}x`,
      onChange: value => {
        cellScale = value;
        dirty = true;
      }
    });
    context.controls.addToggle({
      label: 'Highlight boundary cells',
      value: showBoundary,
      onChange: value => {
        showBoundary = value;
        context.updateLayers();
      }
    });
    context.controls.addButton({
      label: 'Time raster join vs exact join',
      onClick: () => void measureGraphs()
    });
    context.controls.addLegend({
      title: 'Choropleth: metric per ZIP, from the cell lookup',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: '0',
        maximumLabel: 'max'
      }
    });
    context.controls.addLegend({
      title: 'Error bound',
      entries: [
        {color: BOUNDARY_COLOR, label: 'Boundary cell / point in a boundary cell'},
        {color: [235, 240, 255, 255], label: 'Point in an interior cell (joined exactly)'}
      ]
    });
    context.controls.addNote(
      'A point in an unflagged cell joins exactly the ZIP an exact join gives. Orange points are ' +
        'the only ones that can differ; coarser cells flag more of them. Counts cover only the points inside the raster, so zoomed-in views show partial totals.'
    );
    const cellReadout = context.controls.addReadout('Cell size', '...');
    context.controls.addReadout('Points / ZIPs', `${formatCount(pointCount)} / ${zoneCount}`);
    const crossingReadout = context.controls.addReadout('Crossings (capacity)', '...');
    const insideReadout = context.controls.addReadout('Points in raster', '...');
    const boundaryReadout = context.controls.addReadout('Points in boundary cells', '...');
    const exactReadout = context.controls.addReadout('Vs exact join', '...');
    const boundHoldsReadout = context.controls.addReadout('Count bound holds', '...');
    const worstShareReadout = context.controls.addReadout('Widest ZIP bound', '...');
    const worstDifferenceReadout = context.controls.addReadout('Largest ZIP difference', '...');
    const rasterTimingReadout = context.controls.addReadout('Raster join graph', '...');
    const exactTimingReadout = context.controls.addReadout('Exact join graph', '...');
    const speedupReadout = context.controls.addReadout('Raster vs exact', '...');
    context.controls.addReadout('Data', `${parking.attribution}; ${zips.attribution}`);

    const getZoneName = (row: number) => {
      const id = String(zips.featureIds[row]);
      const name = zips.featureNames[row];
      return name && name !== id ? `${id} ${name}` : id;
    };

    // --- Summary -------------------------------------------------------------------------------
    function getMetricValues(source: JoinSnapshot): Uint32Array | Float32Array {
      if (metric === 'count') return source.counts;
      if (metric === 'sum') return source.sums;
      return Float32Array.from(source.counts, (count, zone) =>
        count > 0 ? source.boundaryCounts[zone] / count : 0
      );
    }
    function updateColorMaximum(): void {
      if (!snapshot) return;
      let maximum = 0;
      for (const value of getMetricValues(snapshot)) {
        if (Number.isFinite(value)) maximum = Math.max(maximum, value);
      }
      colorMaximum = Math.max(maximum, metric === 'boundaryShare' ? 0.01 : 1);
    }

    /** Compares the raster join with the exact join using the snapshot's extent. */
    function refreshReadouts(source: JoinSnapshot): void {
      const [originX, originY, cellWidth, cellHeight] = source.extent;
      const exactInside = new Uint32Array(zoneCount);
      let inside = 0;
      let insideBoundary = 0;
      let differing = 0;
      let interiorDifferences = 0;
      for (let point = 0; point < pointCount; point++) {
        const column = Math.floor((parking.positions[point * 2] - originX) / cellWidth);
        const row = Math.floor((parking.positions[point * 2 + 1] - originY) / cellHeight);
        if (column < 0 || row < 0 || column >= RASTER_WIDTH || row >= RASTER_HEIGHT) continue;
        inside++;
        const exactZone = source.exactZones[point];
        if (exactZone !== GPU_POLYGON_RASTERIZATION_NO_ZONE) exactInside[exactZone]++;
        if (source.pointBoundaryMask[point]) insideBoundary++;
        if (source.pointZones[point] !== exactZone) {
          differing++;
          if (!source.pointBoundaryMask[point]) interiorDifferences++;
        }
      }
      let boundarySum = source.unassignedBoundaryCount;
      for (const count of source.boundaryCounts) boundarySum += count;
      let boundHolds = 0;
      let worstShare = -1;
      let worstShareZone = -1;
      let worstDifference = -1;
      let worstDifferenceZone = -1;
      for (let zone = 0; zone < zoneCount; zone++) {
        const count = source.counts[zone];
        const low = count - source.boundaryCounts[zone];
        if (exactInside[zone] >= low && exactInside[zone] <= low + boundarySum) boundHolds++;
        const share = count > 0 ? source.boundaryCounts[zone] / count : 0;
        if (share > worstShare) {
          worstShare = share;
          worstShareZone = zone;
        }
        const difference = Math.abs(count - exactInside[zone]);
        if (difference > worstDifference) {
          worstDifference = difference;
          worstDifferenceZone = zone;
        }
      }
      crossingReadout.setValue(
        `${formatCount(source.crossingCount)} of ${formatCount(CROSSING_CAPACITY)}` +
          (source.overflow ? ' OVERFLOW (raster empty)' : '')
      );
      insideReadout.setValue(
        `${formatCount(inside)} of ${formatCount(pointCount)} ` +
          `(${formatCount(source.outsideCount)} outside, joined to no ZIP)`
      );
      boundaryReadout.setValue(
        inside > 0
          ? `${formatCount(insideBoundary)} (${((100 * insideBoundary) / inside).toFixed(1)}% of in-raster points)`
          : '-'
      );
      exactReadout.setValue(
        `${formatCount(differing)} points differ, ${formatCount(interiorDifferences)} in interior cells ` +
          (interiorDifferences === 0 ? '(as guaranteed)' : '(VIOLATES THE GUARANTEE)')
      );
      boundHoldsReadout.setValue(
        `${boundHolds} of ${zoneCount} ZIPs (exact count inside [c-b, c-b+${boundarySum}])`
      );
      worstShareReadout.setValue(
        worstShareZone < 0
          ? '-'
          : `${getZoneName(worstShareZone)}: ${source.boundaryCounts[worstShareZone]} of ` +
              `${source.counts[worstShareZone]} points (${(100 * worstShare).toFixed(1)}%)`
      );
      worstDifferenceReadout.setValue(
        worstDifferenceZone < 0
          ? '-'
          : `${getZoneName(worstDifferenceZone)}: raster ${source.counts[worstDifferenceZone]} vs ` +
              `exact ${exactInside[worstDifferenceZone]}`
      );
    }

    const readSummary = (commandEncoder: CommandEncoder): boolean => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return false;
      const sources: [Buffer, number][] = [
        [counts, zoneCount],
        [boundaryCounts, zoneCount],
        [sums, zoneCount],
        [pointZones, pointCount],
        [pointBoundaryMask, pointCount],
        [exactZones, pointCount],
        [unassignedBoundaryCount, 1],
        [outsideCount, 1],
        [overflow, 1],
        [crossingCount, 1]
      ];
      let wordOffset = 0;
      for (const [buffer, words] of sources) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: buffer,
          destinationBuffer: ticket.buffer,
          destinationOffset: wordOffset * 4,
          size: words * 4
        });
        wordOffset += words;
      }
      ticket.markEncoded({byteOffset: 0, byteLength: summaryWords * 4});
      const requestedExtent = Float32Array.from(extentValues);
      readbackPending = true;
      void (async () => {
        try {
          const bytes = await ticket.read();
          if (destroyed) return;
          const words = new Uint32Array(
            bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + summaryWords * 4)
          );
          let cursor = 0;
          const take = (length: number) => {
            const column = words.subarray(cursor, cursor + length);
            cursor += length;
            return column;
          };
          const countColumn = take(zoneCount);
          const boundaryColumn = take(zoneCount);
          const sumColumn = take(zoneCount);
          snapshot = {
            extent: requestedExtent,
            counts: countColumn,
            boundaryCounts: boundaryColumn,
            sums: new Float32Array(sumColumn.buffer, sumColumn.byteOffset, zoneCount),
            pointZones: take(pointCount),
            pointBoundaryMask: take(pointCount),
            exactZones: take(pointCount),
            unassignedBoundaryCount: take(1)[0],
            outsideCount: take(1)[0],
            overflow: take(1)[0],
            crossingCount: take(1)[0]
          };
          const previousMaximum = colorMaximum;
          updateColorMaximum();
          refreshReadouts(snapshot);
          if (colorMaximum !== previousMaximum) context.updateLayers();
        } catch {
          // The ring or device was destroyed while the read was in flight.
        } finally {
          readbackPending = false;
        }
      })();
      return true;
    };

    /** Times both graphs outside the frame; the exact join is the cost the raster join avoids. */
    const measureGraphs = async () => {
      if (measuring || destroyed) return;
      measuring = true;
      rasterTimingReadout.setValue('measuring...');
      exactTimingReadout.setValue('measuring...');
      speedupReadout.setValue('...');
      try {
        const options = {parameters: undefined, completionBuffer: overflow, signal: context.signal};
        const raster = await measureCompiledGraph(device, rasterCompiled, options);
        const exact = await measureCompiledGraph(device, exactCompiled, {
          ...options,
          completionBuffer: exactOverflow
        });
        if (destroyed) return;
        rasterTimingReadout.setValue(formatCompiledGraphTiming(raster));
        exactTimingReadout.setValue(formatCompiledGraphTiming(exact));
        speedupReadout.setValue(formatSpeedup(exact.milliseconds, raster.milliseconds));
      } catch {
        if (!destroyed) {
          rasterTimingReadout.setValue('interrupted');
          exactTimingReadout.setValue('interrupted');
        }
      } finally {
        measuring = false;
      }
    };

    // --- Tooltip -------------------------------------------------------------------------------
    const describeZone = (event: MapGraphsPointerEvent): string | null => {
      if (!event.coordinate) return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const zone = findContainingFeature(zips, x, y);
      if (zone < 0 || !snapshot) return zone < 0 ? null : getZoneName(zone);
      const count = snapshot.counts[zone];
      const boundaryPoints = snapshot.boundaryCounts[zone];
      let boundarySum = snapshot.unassignedBoundaryCount;
      for (const value of snapshot.boundaryCounts) boundarySum += value;
      return (
        `${getZoneName(zone)}\n${formatCount(count)} points, ` +
        `${formatCount(snapshot.sums[zone])} spaces\n` +
        `${formatCount(boundaryPoints)} points in boundary cells\n` +
        `exact count in [${count - boundaryPoints}, ${count - boundaryPoints + boundarySum}]`
      );
    };

    // Seed the buffers so the first frame has a valid extent and metric.
    writeMetric();
    writeExtent(cityBounds);

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => [rasterCompiled, exactCompiled],
      encode(commandEncoder: CommandEncoder, frame: MapGraphsFrame) {
        encodedFrames++;
        // The rasterization and join depend only on the extent, the points and the metric, so
        // re-encode when one changed; every output stays valid in its buffer in between.
        const bounds =
          extentMode === 'viewport'
            ? getViewportMetricBounds(frame.viewport, projection)
            : cityBounds;
        if (writeExtent(bounds)) dirty = true;
        if (frame.frameIndex === 0) {
          exactCompiled.encode(commandEncoder, {parameters: undefined});
        }
        if (dirty) {
          rasterCompiled.encode(commandEncoder, {parameters: undefined});
          dirty = false;
          readbackWanted = true;
        }
        if (
          readbackWanted &&
          !readbackPending &&
          frame.frameIndex - lastReadbackFrame >= READBACK_INTERVAL_FRAMES &&
          readSummary(commandEncoder)
        ) {
          readbackWanted = false;
          lastReadbackFrame = frame.frameIndex;
        }
        if (!autoMeasured && encodedFrames >= AUTO_MEASURE_FRAME) {
          autoMeasured = true;
          void measureGraphs();
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [zips.origin[0], zips.origin[1], 0];
        const grid = {
          coordinateOrigin,
          gridSize: [RASTER_WIDTH, RASTER_HEIGHT] as const,
          bounds: rasterBounds,
          rowOrigin: 'south' as const
        };
        const layers: Layer[] = [
          displayKind === 'choropleth'
            ? new MapGraphsRasterLayer({
                ...grid,
                id: 'raster-join-choropleth',
                values: display,
                valueFormat: 'float32',
                colormap: 'viridis',
                valueRange: [0, colorMaximum],
                discardAtOrBelow: RASTER_JOIN_HIDDEN / 2,
                color: [255, 255, 255, 205],
                noDataColor: [0, 0, 0, 0]
              })
            : new MapGraphsRasterLayer({
                ...grid,
                id: 'raster-join-zones',
                values: zones,
                valueFormat: 'uint32',
                colormap: 'category',
                palette: ZONE_PALETTE,
                noDataValue: GPU_POLYGON_RASTERIZATION_NO_ZONE,
                noDataColor: [0, 0, 0, 0]
              })
        ];
        if (showBoundary) {
          layers.push(
            new MapGraphsRasterLayer({
              ...grid,
              id: 'raster-join-boundary',
              values: boundary,
              valueFormat: 'uint32',
              colormap: 'mask',
              color: BOUNDARY_COLOR,
              noDataColor: [0, 0, 0, 0]
            })
          );
        }
        layers.push(
          new MapGraphsSegmentLayer({
            id: 'raster-join-outline',
            coordinateOrigin,
            segments: outlineSegments,
            instanceCount: zips.outlineSegments.length / 4,
            widthPixels: 1.4,
            color: [255, 255, 255, 235]
          }),
          new MapGraphsPointLayer({
            id: 'raster-join-points',
            coordinateOrigin,
            positions,
            instanceCount: pointCount,
            values: pointBoundaryMask,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: BOUNDARY_COLOR,
            noDataColor: [235, 240, 255, 230],
            radiusPixels: 3
          })
        );
        return layers;
      },
      getTooltip: describeZone,
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};
