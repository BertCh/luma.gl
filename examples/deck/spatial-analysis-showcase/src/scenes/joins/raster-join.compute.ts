// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUPolygonRasterizationExtentValues,
  GPUPolygonRasterization,
  GPURasterJoin
} from '@luma.gl/experimental/gpu-raster';
import {GPUPointInPolygonJoin} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisColormap
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {RampName} from '../../engine/ramps';
import {getViewportMetricBounds, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneFrame, SceneInstance} from '../scene';
import {createPolygonSet, findPolygonAt, formatCompact, formatInteger} from './b2-geometry';
import {estimateCrossings, importPolygons, uploadPolygons} from './b2-graph';
import {ZoneFillLayer} from './b2-layers';

/** Option state of the raster-join scene. */
export type RasterJoinOptions = {
  /** Raster size level: the long side has `64 * 2 ** level` cells. */
  resolutionLevel: number;
  extent: 'city' | 'viewport';
  metric: 'count' | 'exact' | 'difference' | 'boundaryShare' | 'researchGrade';
  layer: 'choropleth' | 'zones';
  showBoundary: boolean;
  points: 'off' | 'boundary' | 'mismatch';
  ramp: RampName;
  opacity: number;
};

const NO_ZONE = 0xffffffff;

/** Cells along the long side of the raster for a resolution level. */
export const getResolution = (level: number): number => 64 * 2 ** level;
const SETTLE_FRAMES = 4;
const ZONE_PALETTE = [
  [78, 201, 255, 200],
  [255, 148, 72, 200],
  [189, 122, 255, 200],
  [87, 235, 168, 200],
  [255, 105, 168, 200],
  [245, 220, 87, 200],
  [107, 158, 255, 200],
  [255, 92, 92, 200]
] as const;

type Variant = {
  resolution: number;
  width: number;
  height: number;
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  zones: Buffer;
  boundary: Buffer;
  counts: Buffer;
  sums: Buffer;
  exactCounts: Buffer;
  pointBoundary: Buffer;
  mismatch: Buffer;
  rasterOverflow: Buffer;
  crossingCapacity: number;
};

/**
 * Raster join versus exact join. `GPUPolygonRasterization` scan-converts the 77 community areas into
 * a zone raster whose placement is a four-float extent, `GPURasterJoin` bins every observation into that
 * raster in O(1) per point, and `GPUPointInPolygonJoin` joins the same points exactly, so the error
 * of the approximation can be measured live. The raster resolution is compile-time (one graph per
 * resolution); the extent follows the viewport as a buffer write.
 */
export async function createRasterJoin(
  ctx: SceneContext<RasterJoinOptions>
): Promise<SceneInstance<RasterJoinOptions>> {
  const {device} = ctx;
  const observations = ctx.datasets.get('chicago-nature');
  const areasData = ctx.datasets.get('chicago-community-areas');
  const origin = areasData.defaultOrigin;
  const projection = areasData.getProjection(origin);

  const positions = observations.projectColumn('position', origin);
  const pointCount = positions.length / 2;
  const researchGrade = observations.column<Uint8Array>('researchGrade');
  const values = new Float32Array(pointCount);
  for (let index = 0; index < pointCount; index++) values[index] = researchGrade[index] ? 1 : 0;

  const areaSet = createPolygonSet(areasData, origin);
  const zoneCount = areaSet.featureCount;
  const areaNames = (areasData.geojson?.features ?? []).map(feature =>
    String((feature.properties as Record<string, unknown>)?.name ?? '')
  );

  const resources = new SpatialAnalysisResources(device, 'raster-join');
  const polygons = uploadPolygons(resources, 'areas', areaSet);
  const positionsBuffer = resources.createBuffer('positions', positions);
  const valuesBuffer = resources.createBuffer('values', values);
  const extentParameters = resources.createParameterBuffer('extent', 'float32', 4);
  const boundsBuffer = resources.createBuffer('bounds', 16);
  const exactIds = resources.createBuffer('exact-ids', pointCount * 4);
  const exactOverflow = resources.createBuffer('exact-overflow', 4);
  const zoneValues = resources.createBuffer('zone-values', zoneCount * 4);
  const boundaryBuffer = resources.createBuffer('unused-boundary', 4);
  void boundaryBuffer;

  // --- state -----------------------------------------------------------------------------------
  let destroyed = false;
  let variant: Variant;
  let dirty = true;
  let settle = 0;
  let measuring = false;
  let frames = 0;
  let lastBounds: [number, number, number, number] | null = null;
  let rasterCounts = new Uint32Array(zoneCount);
  let exactCounts = new Uint32Array(zoneCount);
  let valueRange: [number, number] = [0, 1];

  const [cityMinX, cityMinY, cityMaxX, cityMaxY] = areaSet.bounds;

  /** Builds the graph of one raster resolution (cells along the longer side of the city). */
  function buildVariant(resolution: number, forTiming = false): Variant {
    const cityWidth = cityMaxX - cityMinX;
    const cityHeight = cityMaxY - cityMinY;
    const cell = Math.max(cityWidth, cityHeight) / resolution;
    const width = Math.ceil(cityWidth / cell) + 1;
    const height = Math.ceil(cityHeight / cell) + 1;
    const cellCount = width * height;
    const own = new SpatialAnalysisResources(device, `rj-${resolution}`);
    const graph = new GPUCommandGraph<void>(device, {id: `rj-${resolution}`});
    const views = importPolygons(graph, 'areas', polygons);
    const crossingEstimate = estimateCrossings(areaSet, cityMinY - cell / 2, cell, height);
    const crossingCapacity = Math.max(8192, Math.ceil(crossingEstimate * 2.5));

    const zones = own.createBuffer('zones', cellCount * 4);
    const boundary = own.createBuffer('boundary', cellCount * 4);
    const rasterOverflow = own.createBuffer('raster-overflow', 4);
    const crossingCount = own.createBuffer('crossing-count', 4);
    const counts = own.createBuffer('counts', zoneCount * 4);
    const sums = own.createBuffer('sums', zoneCount * 4);
    const boundaryCounts = own.createBuffer('boundary-counts', zoneCount * 4);
    const unassigned = own.createBuffer('unassigned', 4);
    const outside = own.createBuffer('outside', 4);
    const pointZones = own.createBuffer('point-zones', pointCount * 4);
    const pointBoundary = own.createBuffer('point-boundary', pointCount * 4);
    const exactCountsBuffer = own.createBuffer('exact-counts', zoneCount * 4);
    const exactCandidates = own.createBuffer('exact-candidates', 4);
    const mismatch = own.createBuffer('mismatch', pointCount * 4);

    const extentView = extentParameters.importToGraph(graph);
    const zonesView = importGraphBuffer(graph, 'zones', zones, 'uint32', cellCount);
    const boundaryView = importGraphBuffer(graph, 'boundary', boundary, 'uint32', cellCount);
    const pointsView = importGraphBuffer(graph, 'points', positionsBuffer, 'float32x2', pointCount);
    const pointZonesView = importGraphBuffer(
      graph,
      'point-zones',
      pointZones,
      'uint32',
      pointCount
    );
    const exactIdsView = importGraphBuffer(graph, 'exact-ids', exactIds, 'uint32', pointCount);

    graph.add(
      new GPUPolygonRasterization({
        id: 'rasterize',
        width,
        height,
        extent: extentView,
        polygonPositions: views.positions,
        featureOffsets: views.featureOffsets,
        polygonOffsets: views.polygonOffsets,
        ringOffsets: views.ringOffsets,
        crossingCapacity,
        zones: zonesView,
        boundary: boundaryView,
        overflow: importGraphBuffer(graph, 'raster-overflow', rasterOverflow, 'uint32', 1),
        crossingCount: importGraphBuffer(graph, 'crossing-count', crossingCount, 'uint32', 1)
      })
    );
    graph.add(
      new GPURasterJoin({
        id: 'raster-join',
        width,
        height,
        extent: extentView,
        points: pointsView,
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', pointCount),
        zones: zonesView,
        boundary: boundaryView,
        zoneCount,
        output: {
          counts: importGraphBuffer(graph, 'counts', counts, 'uint32', zoneCount),
          sums: importGraphBuffer(graph, 'sums', sums, 'float32', zoneCount),
          boundaryCounts: importGraphBuffer(
            graph,
            'boundary-counts',
            boundaryCounts,
            'uint32',
            zoneCount
          ),
          unassignedBoundaryCount: importGraphBuffer(graph, 'unassigned', unassigned, 'uint32', 1),
          outsideCount: importGraphBuffer(graph, 'outside', outside, 'uint32', 1),
          pointZones: pointZonesView,
          pointBoundaryMask: importGraphBuffer(
            graph,
            'point-boundary',
            pointBoundary,
            'uint32',
            pointCount
          )
        }
      })
    );
    graph.add(
      new GPUPointInPolygonJoin({
        id: 'exact-join',
        points: pointsView,
        polygonPositions: views.positions,
        featureOffsets: views.featureOffsets,
        polygonOffsets: views.polygonOffsets,
        ringOffsets: views.ringOffsets,
        candidateCapacity: pointCount * 6,
        pointFeatureIds: exactIdsView,
        featureCounts: importGraphBuffer(
          graph,
          'exact-counts',
          exactCountsBuffer,
          'uint32',
          zoneCount
        ),
        overflow: importGraphBuffer(graph, 'exact-overflow', exactOverflow, 'uint32', 1),
        candidateCount: importGraphBuffer(graph, 'exact-candidates', exactCandidates, 'uint32', 1)
      })
    );
    // Points whose raster zone differs from the exact zone: the error the boundary flags must explain.
    addKernelPass(graph, {
      id: 'mismatch',
      invocationCount: pointCount,
      bindings: [
        {name: 'rasterZone', view: pointZonesView, type: 'u32', access: 'read'},
        {name: 'exactZone', view: exactIdsView, type: 'u32', access: 'read'},
        {
          name: 'mismatch',
          view: importGraphBuffer(graph, 'mismatch', mismatch, 'uint32', pointCount),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: `mismatch[mismatchOffset + index] = select(0u, 1u, rasterZone[rasterZoneOffset + index] != exactZone[exactZoneOffset + index]);`
    });

    const reader = new SummaryReader(
      own,
      `summary-${resolution}`,
      [
        {buffer: counts, size: zoneCount * 4},
        {buffer: boundaryCounts, size: zoneCount * 4},
        {buffer: exactCountsBuffer, size: zoneCount * 4},
        {buffer: sums, size: zoneCount * 4},
        {buffer: unassigned, size: 4},
        {buffer: outside, size: 4},
        {buffer: rasterOverflow, size: 4},
        {buffer: crossingCount, size: 4},
        {buffer: exactOverflow, size: 4},
        {buffer: mismatch, size: pointCount * 4},
        {buffer: pointBoundary, size: pointCount * 4}
      ],
      bytes => {
        if (destroyed || variant?.resolution !== resolution) return;
        handleSummary(bytes, width, height, crossingCapacity);
      }
    );
    const compiled = own.track(graph.compile());
    void forTiming;
    return {
      resolution,
      width,
      height,
      resources: own,
      compiled,
      reader,
      zones,
      boundary,
      counts,
      sums,
      exactCounts: exactCountsBuffer,
      pointBoundary,
      mismatch,
      rasterOverflow,
      crossingCapacity
    };
  }

  function handleSummary(
    bytes: ArrayBuffer,
    width: number,
    height: number,
    capacity: number
  ): void {
    const state = ctx.options;
    let offset = 0;
    const take = (length: number) => {
      const slice = bytes.slice(offset, offset + length * 4);
      offset += length * 4;
      return slice;
    };
    const counts = new Uint32Array(take(zoneCount));
    const boundaryCounts = new Uint32Array(take(zoneCount));
    const exact = new Uint32Array(take(zoneCount));
    const sums = new Float32Array(take(zoneCount));
    const [unassigned, outside, rasterOverflow, crossings, exactOverflow] = new Uint32Array(
      take(5)
    );
    const mismatch = new Uint32Array(take(pointCount));
    const pointBoundary = new Uint32Array(take(pointCount));
    rasterCounts = counts;
    exactCounts = exact;

    let joined = 0;
    let exactJoined = 0;
    let boundaryJoined = 0;
    let maximumDifference = 0;
    let withinBound = 0;
    let allBoundary = unassigned;
    for (let zone = 0; zone < zoneCount; zone++) allBoundary += boundaryCounts[zone];
    for (let zone = 0; zone < zoneCount; zone++) {
      joined += counts[zone];
      exactJoined += exact[zone];
      boundaryJoined += boundaryCounts[zone];
      maximumDifference = Math.max(maximumDifference, Math.abs(counts[zone] - exact[zone]));
      const low = counts[zone] - boundaryCounts[zone];
      if (exact[zone] >= low && exact[zone] <= low + allBoundary) withinBound++;
    }
    let mismatches = 0;
    let mismatchesInBoundary = 0;
    let pointsInBoundary = 0;
    for (let point = 0; point < pointCount; point++) {
      if (pointBoundary[point]) pointsInBoundary++;
      if (mismatch[point]) {
        mismatches++;
        if (pointBoundary[point]) mismatchesInBoundary++;
      }
    }
    const cells = width * height;
    ctx.setReadout(
      'cells',
      `${formatInteger(width)} x ${formatInteger(height)} = ${formatInteger(cells)} cells`
    );
    ctx.setReadout(
      'joined',
      `${formatInteger(joined)} by raster, ${formatInteger(exactJoined)} exact, ${formatInteger(outside)} outside the raster`
    );
    ctx.setReadout(
      'mismatch',
      `${formatInteger(mismatches)} points (${((100 * mismatches) / pointCount).toFixed(3)}%) join a different area than the exact join`
    );
    ctx.setReadout(
      'explained',
      mismatches === 0
        ? 'no mismatches'
        : `${mismatchesInBoundary === mismatches ? 'all' : formatInteger(mismatchesInBoundary)} ${mismatchesInBoundary === mismatches ? '' : `of ${formatInteger(mismatches)} `}mismatches sit in boundary cells (${formatInteger(pointsInBoundary)} points do)`
    );
    ctx.setReadout(
      'bound',
      `${withinBound} of ${zoneCount} areas have an exact count inside the documented bound`
    );
    ctx.setReadout(
      'difference',
      `${formatInteger(maximumDifference)} observations (largest area difference)`
    );
    ctx.setReadout(
      'boundaryJoined',
      `${formatInteger(boundaryJoined)} joined points are in boundary cells`
    );
    ctx.setReadout(
      'rasterOverflow',
      rasterOverflow
        ? `YES: ${formatInteger(crossings)} crossings need more than the ${formatInteger(capacity)} reserved, so the raster is empty`
        : `no (${formatInteger(crossings)} of ${formatInteger(capacity)} crossings)`
    );
    ctx.setReadout('exactOverflow', exactOverflow ? 'YES' : 'no');

    // Per-area display value for the metrics the join does not write directly.
    const metric = state.metric;
    const display = new Float32Array(zoneCount);
    let minimum = Infinity;
    let maximum = -Infinity;
    for (let zone = 0; zone < zoneCount; zone++) {
      let value = 0;
      if (metric === 'difference') value = counts[zone] - exact[zone];
      else if (metric === 'boundaryShare')
        value = counts[zone] ? boundaryCounts[zone] / counts[zone] : 0;
      else if (metric === 'count') value = counts[zone];
      else if (metric === 'exact') value = exact[zone];
      else value = sums[zone];
      display[zone] = value;
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
    if (metric === 'difference') {
      const bound = Math.max(1, Math.abs(minimum), Math.abs(maximum));
      valueRange = [-bound, bound];
    } else {
      valueRange = [0, Math.max(maximum, 1e-6)];
    }
    zoneValues.write(display);
    ctx.setLegendExtent('metric', valueRange);
    ctx.requestLayers();
  }

  function replaceVariant(resolution: number): void {
    const previous = variant;
    variant = buildVariant(resolution);
    dirty = true;
    if (previous) {
      previous.reader.stop();
      requestAnimationFrame(() => requestAnimationFrame(() => previous.resources.destroy()));
    }
  }

  variant = buildVariant(getResolution(ctx.options.resolutionLevel));

  /** Times rasterization, the raster join alone, and the exact join, outside the frame. */
  async function measureAll(): Promise<void> {
    if (measuring || destroyed) return;
    measuring = true;
    ctx.setReadout('timing', 'measuring...');
    const resolution = getResolution(ctx.options.resolutionLevel);
    const results: string[] = [];
    try {
      const built = buildVariant(resolution, true);
      try {
        const options = {
          parameters: undefined,
          completionBuffer: exactOverflow,
          signal: ctx.signal
        };
        const all = await measureCompiledGraph(device, built.compiled, options);
        results.push(
          `rasterize + raster join + exact join ${formatCompiledGraphTiming(all).split(' · ')[0]}`
        );
      } finally {
        built.compiled.destroy();
        built.reader.stop();
        built.resources.destroy();
      }
      const exactOnly = buildExactGraph();
      try {
        const timing = await measureCompiledGraph(device, exactOnly.compiled, {
          parameters: undefined,
          completionBuffer: exactOnly.overflow,
          signal: ctx.signal
        });
        results.push(`exact join alone ${formatCompiledGraphTiming(timing).split(' · ')[0]}`);
      } finally {
        exactOnly.compiled.destroy();
        exactOnly.resources.destroy();
      }
      const rasterOnly = buildRasterOnlyGraph(resolution);
      try {
        const timing = await measureCompiledGraph(device, rasterOnly.compiled, {
          parameters: undefined,
          completionBuffer: rasterOnly.overflow,
          signal: ctx.signal
        });
        results.push(
          `rasterize + raster join ${formatCompiledGraphTiming(timing).split(' · ')[0]}`
        );
      } finally {
        rasterOnly.compiled.destroy();
        rasterOnly.resources.destroy();
      }
      if (!destroyed) ctx.setReadout('timing', results.join(' | '));
    } catch {
      // Device destroyed or measurement aborted.
    } finally {
      measuring = false;
      dirty = true;
    }
  }

  function buildExactGraph() {
    const own = new SpatialAnalysisResources(device, 'rj-exact-time');
    const graph = new GPUCommandGraph<void>(device, {id: 'rj-exact-time'});
    const views = importPolygons(graph, 'areas', polygons);
    const ids = own.createBuffer('ids', pointCount * 4);
    const counts = own.createBuffer('counts', zoneCount * 4);
    const overflow = own.createBuffer('overflow', 4);
    graph.add(
      new GPUPointInPolygonJoin({
        id: 'exact-join',
        points: importGraphBuffer(graph, 'points', positionsBuffer, 'float32x2', pointCount),
        polygonPositions: views.positions,
        featureOffsets: views.featureOffsets,
        polygonOffsets: views.polygonOffsets,
        ringOffsets: views.ringOffsets,
        candidateCapacity: pointCount * 6,
        pointFeatureIds: importGraphBuffer(graph, 'ids', ids, 'uint32', pointCount),
        featureCounts: importGraphBuffer(graph, 'counts', counts, 'uint32', zoneCount),
        overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
      })
    );
    return {resources: own, compiled: own.track(graph.compile()), overflow};
  }

  function buildRasterOnlyGraph(resolution: number) {
    const cityWidth = cityMaxX - cityMinX;
    const cityHeight = cityMaxY - cityMinY;
    const cell = Math.max(cityWidth, cityHeight) / resolution;
    const width = Math.ceil(cityWidth / cell) + 1;
    const height = Math.ceil(cityHeight / cell) + 1;
    const cellCount = width * height;
    const own = new SpatialAnalysisResources(device, 'rj-raster-time');
    const graph = new GPUCommandGraph<void>(device, {id: 'rj-raster-time'});
    const views = importPolygons(graph, 'areas', polygons);
    const zones = own.createBuffer('zones', cellCount * 4);
    const rasterOverflow = own.createBuffer('raster-overflow', 4);
    const counts = own.createBuffer('counts', zoneCount * 4);
    const extentView = extentParameters.importToGraph(graph);
    const zonesView = importGraphBuffer(graph, 'zones', zones, 'uint32', cellCount);
    graph.add(
      new GPUPolygonRasterization({
        id: 'rasterize',
        width,
        height,
        extent: extentView,
        polygonPositions: views.positions,
        featureOffsets: views.featureOffsets,
        polygonOffsets: views.polygonOffsets,
        ringOffsets: views.ringOffsets,
        crossingCapacity: Math.max(
          8192,
          Math.ceil(estimateCrossings(areaSet, cityMinY - cell / 2, cell, height) * 2.5)
        ),
        zones: zonesView,
        overflow: importGraphBuffer(graph, 'raster-overflow', rasterOverflow, 'uint32', 1)
      })
    );
    graph.add(
      new GPURasterJoin({
        id: 'raster-join',
        width,
        height,
        extent: extentView,
        points: importGraphBuffer(graph, 'points', positionsBuffer, 'float32x2', pointCount),
        zones: zonesView,
        zoneCount,
        output: {counts: importGraphBuffer(graph, 'counts', counts, 'uint32', zoneCount)}
      })
    );
    return {resources: own, compiled: own.track(graph.compile()), overflow: rasterOverflow};
  }

  /** Writes the raster placement and the matching layer bounds for the chosen extent. */
  function writeExtent(frame: SceneFrame | null): void {
    const state = ctx.options;
    const {width, height} = variant;
    let bounds: [number, number, number, number];
    if (state.extent === 'viewport' && frame) {
      bounds = getViewportMetricBounds(frame.viewport, projection);
    } else {
      const cell =
        Math.max(cityMaxX - cityMinX, cityMaxY - cityMinY) / getResolution(state.resolutionLevel);
      bounds = [cityMinX - cell / 2, cityMinY - cell / 2, 0, 0];
      bounds[2] = bounds[0] + width * cell;
      bounds[3] = bounds[1] + height * cell;
    }
    if (lastBounds && bounds.every((value, index) => value === lastBounds![index])) return;
    lastBounds = bounds;
    extentParameters.write(
      getGPUPolygonRasterizationExtentValues(
        bounds[0],
        bounds[1],
        (bounds[2] - bounds[0]) / width,
        (bounds[3] - bounds[1]) / height
      )
    );
    boundsBuffer.write(Float32Array.from(bounds));
    const cellWidth = (bounds[2] - bounds[0]) / width;
    const cellHeight = (bounds[3] - bounds[1]) / height;
    ctx.setReadout(
      'cellSize',
      `${formatCompact(cellWidth)} x ${formatCompact(cellHeight)} m per cell`
    );
    dirty = true;
  }

  return {
    getCompiledGraphs: () => [variant.compiled as CompiledGPUCommandGraph<never>],

    setOption(id, _value, state) {
      if (id === 'resolutionLevel' && getResolution(state.resolutionLevel) !== variant.resolution) {
        replaceVariant(getResolution(state.resolutionLevel));
        lastBounds = null;
      } else if (id === 'extent') {
        lastBounds = null;
      } else if (id === 'metric') {
        dirty = true;
      }
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'measure') void measureAll();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const zone = findPolygonAt(areaSet, x, y);
      if (zone < 0) return null;
      return `${areaNames[zone]}\nraster join ${formatInteger(rasterCounts[zone])}, exact ${formatInteger(exactCounts[zone])}`;
    },

    encode(commandEncoder, frame) {
      frames++;
      writeExtent(frame);
      if (dirty) {
        variant.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        settle = SETTLE_FRAMES;
      } else if (settle > 0) {
        settle--;
        if (settle === 0) variant.reader.request(commandEncoder);
      }
      variant.reader.flush(commandEncoder);
      void frames;
    },

    getLayers() {
      const state = ctx.options;
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const grid = {
        coordinateOrigin,
        gridSize: [variant.width, variant.height] as [number, number],
        bounds: boundsBuffer,
        rowOrigin: 'south' as const
      };
      if (state.layer === 'zones') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'zone-raster',
            ...grid,
            values: variant.zones,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: ZONE_PALETTE,
            noDataValue: NO_ZONE,
            noDataColor: [0, 0, 0, 0],
            opacity: state.opacity
          })
        );
      } else {
        const direct =
          state.metric === 'count' || state.metric === 'exact' || state.metric === 'researchGrade';
        layers.push(
          new ZoneFillLayer({
            id: 'zone-choropleth',
            ...grid,
            valueIndices: variant.zones,
            values:
              state.metric === 'count'
                ? variant.counts
                : state.metric === 'exact'
                  ? variant.exactCounts
                  : state.metric === 'researchGrade'
                    ? variant.sums
                    : zoneValues,
            valueFormat:
              state.metric === 'count' || state.metric === 'exact' ? 'uint32' : 'float32',
            colormap: (state.metric === 'difference'
              ? 'diverging'
              : state.ramp) as SpatialAnalysisColormap,
            valueRange,
            sqrtScale: direct && state.metric !== 'researchGrade',
            noDataColor: [0, 0, 0, 0],
            opacity: state.opacity
          })
        );
      }
      if (state.showBoundary) {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'boundary-cells',
            ...grid,
            values: variant.boundary,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [255, 40, 60, 210],
            noDataColor: [0, 0, 0, 0]
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'area-outline',
          coordinateOrigin,
          segments: polygons.outline,
          instanceCount: areaSet.outline.length / 4,
          color: dark ? [235, 240, 250, 190] : [20, 30, 50, 200],
          widthPixels: 1.2
        })
      );
      if (state.points !== 'off') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'flagged-points',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: pointCount,
            values: state.points === 'mismatch' ? variant.mismatch : variant.pointBoundary,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: state.points === 'mismatch' ? [255, 255, 60, 255] : [255, 160, 40, 150],
            noDataColor: [0, 0, 0, 0],
            radiusPixels: state.points === 'mismatch' ? 3.5 : 1.2
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      variant.reader.stop();
      variant.resources.destroy();
      resources.destroy();
    }
  };
}
