// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';
import {GPUGeometryMeasures} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  buildRingEdges,
  createGraphImporter,
  findFeatureAt,
  formatNumber,
  triangulatePolygons
} from './b3-common';
import {FeatureTriangleLayer} from './b3-layers';
import {
  ACRES_PER_SQUARE_METER,
  describeFire,
  formatWildfireDay,
  loadWildfires,
  YEAR_COLORS,
  type WildfireData
} from './wildfire-data';

/** First and last playhead day (days since 2020-01-01): a little before the first and after the last perimeter. */
export const SEASON_RANGE = [140, 1450] as const;
/** Days of playhead per real second at speed 1. */
export const SEASON_RATE = 55;
const YEAR_STARTS = [0, 366, 731, 1096];

/** Option state of the wildfire-seasons scene. */
export type WildfireSeasonsOptions = {
  time: number;
  play: boolean;
  speed: string;
  loop: boolean;
  trailDays: number;
  tailFade: number;
  colorBy: 'age' | 'year' | 'sizeClass';
  areaSystem: 'planar' | 'spherical' | 'wgs84' | 'geodesic';
  groupBy: 'year' | 'sizeClass';
  showOutlines: boolean;
  showMarkers: boolean;
  highlightNewest: boolean;
  follow: boolean;
};

const YEAR_PALETTE = YEAR_COLORS.map(([r, g, b]) => [r, g, b, 255] as const);
const SIZE_PALETTE = [
  [110, 140, 235, 255],
  [70, 190, 150, 255],
  [240, 170, 50, 255],
  [232, 90, 60, 255]
] as const;

/**
 * Wildfire seasons: a playback of the perimeters in the order of their NIFC dates, filtered on the
 * GPU by `GPUTimeWindowFilter` (trail window, fade weights, accepted mask) while
 * `GPUGeometryMeasures` supplies the area of every fire and the area per year or size class. The
 * cumulative acres curve comes from the measured areas.
 */
export async function createWildfireSeasons(
  ctx: SceneContext<WildfireSeasonsOptions>
): Promise<SceneInstance<WildfireSeasonsOptions>> {
  const {device} = ctx;
  const data: WildfireData = loadWildfires(ctx.datasets.get('poopdeck-wildfires'));
  const {layout, count} = data;
  const resources = new SpatialAnalysisResources(device, 'wildfire-seasons');
  let destroyed = false;

  // ---- Buffers ------------------------------------------------------------------------------------
  const lngLatBuffer = resources.createBuffer('lnglat', layout.lngLat);
  const planarBuffer = resources.createBuffer('planar', data.mercator);
  const acreageValues = resources.createBuffer('agency-acres', data.acres);
  const ringOffsetsBuffer = resources.createBuffer('ring-offsets', layout.ringOffsets);
  const featureRingsBuffer = resources.createBuffer('feature-rings', layout.featureRingOffsets);
  const groupIdsBuffer = resources.createBuffer('group-ids', Uint32Array.from(data.yearIndex));
  const daysBuffer = resources.createBuffer('days', data.days);
  const yearFloat = resources.createBuffer('year-f', Float32Array.from(data.yearIndex));
  const sizeFloat = resources.createBuffer('size-f', Float32Array.from(data.sizeClass));
  const triangulated = triangulatePolygons(layout);
  const fill = {
    corners: resources.createBuffer('fill-corners', triangulated.corners),
    featureRows: resources.createBuffer('fill-feature-rows', triangulated.featureRows),
    triangleCount: triangulated.triangleCount
  };
  const edgeColumns = buildRingEdges(layout);
  const edgeSegments = new Float32Array(edgeColumns.edgeCount * 4);
  const edgesPerFire = new Uint32Array(count);
  for (let edge = 0; edge < edgeColumns.edgeCount; edge++) {
    edgeSegments.set(edgeColumns.starts.subarray(edge * 2, edge * 2 + 2), edge * 4);
    edgeSegments.set(edgeColumns.ends.subarray(edge * 2, edge * 2 + 2), edge * 4 + 2);
    edgesPerFire[edgeColumns.featureRows[edge]]++;
  }
  const edges = {
    segments: resources.createBuffer('edge-segments', edgeSegments),
    featureRows: resources.createBuffer('edge-feature-rows', edgeColumns.featureRows),
    edgeCount: edgeColumns.edgeCount
  };
  const selectionSegments = resources.createBuffer(
    'newest-segments',
    Math.max(1, ...edgesPerFire) * 16
  );
  const column = (name: string, rows = count) =>
    resources.createBuffer(name, Math.max(rows, 1) * 4);
  const areas = column('areas');
  const centroids = column('centroids', count * 2);
  const groupAreas = column('group-areas', 4);
  const groupCounts = column('group-counts', 4);
  const filterIds = column('filter-ids');
  const filterCount = column('filter-count', 1);
  const filterOverflow = column('filter-overflow', 1);
  const acceptedMask = column('accepted-mask');
  const fadeWeights = column('fade-weights');
  const display = column('display');
  const displayCategory = column('display-category');
  const windowParameters = resources.createParameterBuffer(
    'window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  const modeParameters = resources.createParameterBuffer('mode', 'float32', 4);

  // ---- Graphs -------------------------------------------------------------------------------------
  /** Per-fire area and the area per group; the centroids always come from the WGS84 node. */
  const buildMeasures = (options: WildfireSeasonsOptions): CompiledGPUCommandGraph<void> => {
    const graph = new GPUCommandGraph<void>(device, {id: 'seasons-measures'});
    const imp = createGraphImporter(graph);
    const ringOffsets = imp('ring-offsets', ringOffsetsBuffer, 'uint32', layout.ringCount + 1);
    const featureRings = imp('feature-rings', featureRingsBuffer, 'uint32', count + 1);
    const lngLat = imp('lnglat', lngLatBuffer, 'float32x2', layout.vertexCount);
    graph.add(
      new GPUGeometryMeasures({
        id: 'seasons-area',
        positions:
          options.areaSystem === 'planar'
            ? imp('planar', planarBuffer, 'float32x2', layout.vertexCount)
            : lngLat,
        ringOffsets,
        featureRingOffsets: featureRings,
        geometryType: 'polygons',
        coordinateSystem: options.areaSystem,
        groupIds: imp('group-ids', groupIdsBuffer, 'uint32', count),
        groupCount: 4,
        output: {areas: imp('areas', areas, 'float32', count)},
        groupOutput: {
          areas: imp('group-areas', groupAreas, 'float32', 4),
          featureCounts: imp('group-counts', groupCounts, 'uint32', 4)
        }
      })
    );
    graph.add(
      new GPUGeometryMeasures({
        id: 'seasons-centroids',
        positions: lngLat,
        ringOffsets,
        featureRingOffsets: featureRings,
        geometryType: 'polygons',
        coordinateSystem: 'wgs84',
        output: {centroids: imp('centroids', centroids, 'float32x2', count)}
      })
    );
    return resources.track(graph.compile());
  };

  /** Time window filter and the display columns it feeds. Encoded whenever the playhead moves. */
  const buildWindow = (): CompiledGPUCommandGraph<void> => {
    const graph = new GPUCommandGraph<void>(device, {id: 'seasons-window'});
    const imp = createGraphImporter(graph);
    const maskView = imp('accepted-mask', acceptedMask, 'uint32', count);
    const weightView = imp('fade-weights', fadeWeights, 'float32', count);
    graph.add(
      new GPUTimeWindowFilter({
        id: 'season-window',
        timestamps: imp('days', daysBuffer, 'float32', count),
        window: windowParameters.importToGraph(graph),
        output: {
          ids: imp('filter-ids', filterIds, 'uint32', count),
          count: imp('filter-count', filterCount, 'uint32', 1),
          overflow: imp('filter-overflow', filterOverflow, 'uint32', 1)
        },
        outputMask: maskView,
        fadeWeights: weightView
      })
    );
    addKernelPass(graph, {
      id: 'season-display',
      invocationCount: count,
      bindings: [
        {name: 'mask', view: maskView, type: 'u32', access: 'read'},
        {name: 'weights', view: weightView, type: 'f32', access: 'read'},
        {
          name: 'years',
          view: imp('year-f', yearFloat, 'float32', count),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'sizes',
          view: imp('size-f', sizeFloat, 'float32', count),
          type: 'f32',
          access: 'read'
        },
        {name: 'modes', view: modeParameters.importToGraph(graph), type: 'f32', access: 'read'},
        {
          name: 'shown',
          view: imp('display', display, 'float32', count),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'shownCategory',
          view: imp('display-category', displayCategory, 'uint32', count),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  var nanBits = 0x7fc00000u;
  var value = bitcast<f32>(nanBits);
  var category = 0xffffffffu;
  if (mask[maskOffset + index] != 0u) {
    let mode = u32(modes[modesOffset]);
    if (mode == 1u) {
      value = years[yearsOffset + index];
    } else if (mode == 2u) {
      value = sizes[sizesOffset + index];
    } else {
      value = max(weights[weightsOffset + index], 0.02);
    }
    category = u32(max(value, 0.0));
  }
  shown[shownOffset + index] = value;
  shownCategory[shownCategoryOffset + index] = category;`
    });
    return resources.track(graph.compile());
  };

  let measures = buildMeasures(ctx.options);
  const windowGraph = buildWindow();
  const clock = createPlaybackClock<WildfireSeasonsOptions>(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: SEASON_RANGE, rate: SEASON_RATE, step: 1}
  );

  // ---- Readback -----------------------------------------------------------------------------------
  let areaAcres = new Float32Array(count);
  let groupAcres = new Float32Array(4);
  let groupFires = new Uint32Array(4);
  let measuresReady = false;
  let windowMask = new Uint32Array(count);
  let windowCount = 0;
  let newest = -1;
  let newestEdgeCount = 0;
  let lastFlight = -Infinity;
  let lastChartDay = Number.NaN;
  const dirty = {measures: true, window: true, chart: true};

  const modeCode = (options: WildfireSeasonsOptions) =>
    options.colorBy === 'year' ? 1 : options.colorBy === 'sizeClass' ? 2 : 0;

  const writeWindow = (time: number, options: WildfireSeasonsOptions) => {
    windowParameters.write(
      getGPUTimeWindowParameterValues({
        start: time - options.trailDays,
        end: time,
        startFadeDuration: options.trailDays * options.tailFade
      })
    );
    modeParameters.write(Float32Array.of(modeCode(options), 0, 0, 0));
  };

  /** Newest fire (last perimeter date at or before the playhead), or -1. */
  const findNewest = (time: number) => {
    let low = 0;
    let high = count - 1;
    let found = -1;
    while (low <= high) {
      const middle = (low + high) >> 1;
      if (data.days[data.byDate[middle]] <= time) {
        found = middle;
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    return found < 0 ? -1 : data.byDate[found];
  };

  const updateNewest = (time: number, options: WildfireSeasonsOptions) => {
    const fire = findNewest(time);
    if (fire === newest) return;
    newest = fire;
    newestEdgeCount = 0;
    if (fire >= 0) {
      const segments: number[] = [];
      for (let edge = 0; edge < edges.edgeCount; edge++) {
        if (edgeColumns.featureRows[edge] === fire) {
          segments.push(
            edgeColumns.starts[edge * 2],
            edgeColumns.starts[edge * 2 + 1],
            edgeColumns.ends[edge * 2],
            edgeColumns.ends[edge * 2 + 1]
          );
        }
      }
      selectionSegments.write(Float32Array.from(segments));
      newestEdgeCount = segments.length / 4;
    }
    ctx.setReadout('newest', fire < 0 ? 'none yet' : describeFire(data, fire));
    if (options.follow && fire >= 0 && performance.now() - lastFlight > 1100) {
      const b = fire * 4;
      lastFlight = performance.now();
      ctx.flyTo(
        {
          longitude: (layout.featureBounds[b] + layout.featureBounds[b + 2]) / 2,
          latitude: (layout.featureBounds[b + 1] + layout.featureBounds[b + 3]) / 2,
          zoom: Math.min(
            9,
            Math.max(
              6,
              9.2 -
                Math.log2(
                  Math.max(0.2, layout.featureBounds[b + 2] - layout.featureBounds[b] + 0.1)
                )
            )
          )
        },
        {transitionMs: 900}
      );
    }
    ctx.requestLayers();
  };

  const updateCharts = (time: number) => {
    if (!measuresReady) return;
    // Cumulative acres in date order, as a step curve with the year starts and the playhead.
    const x: number[] = [SEASON_RANGE[0]];
    const y: number[] = [0];
    let total = 0;
    for (const fire of data.byDate) {
      x.push(data.days[fire], data.days[fire]);
      y.push(total, total + areaAcres[fire]);
      total += areaAcres[fire];
    }
    x.push(SEASON_RANGE[1]);
    y.push(total);
    ctx.setChart('cumulativeChart', {
      kind: 'line',
      series: [{label: 'cumulative acres (GPU area)', x, y, area: true}],
      xDomain: SEASON_RANGE,
      xLabel: 'perimeter date',
      yLabel: 'acres',
      formatX: value => formatWildfireDay(value).slice(3),
      formatY: value =>
        value >= 1e6 ? `${(value / 1e6).toFixed(1)}M` : `${Math.round(value / 1000)}k`,
      markers: [
        ...YEAR_STARTS.slice(1).map((start, index) => ({x: start, label: String(2021 + index)})),
        {x: time, label: 'now'}
      ],
      description: 'Cumulative acres of the fire perimeters by perimeter date, with the playhead'
    });
    ctx.setChart('yearChart', {
      kind: 'bars',
      values: Array.from(groupAcres, value => value),
      labels:
        ctx.options.groupBy === 'year'
          ? ['2020', '2021', '2022', '2023']
          : ['<10k', '11-33k', '50-97k', '>300k'],
      highlight: [],
      yLabel: 'acres',
      formatY: value =>
        value >= 1e6 ? `${(value / 1e6).toFixed(1)}M` : `${Math.round(value / 1000)}k`,
      description: 'Area of the fires per group, measured on the GPU'
    });
    // Cumulative readouts at the playhead come from the accepted mask of the filter.
    lastChartDay = time;
  };

  const updateReadouts = (time: number) => {
    ctx.setReadout('date', formatWildfireDay(time));
    let shown = 0;
    let acres = 0;
    for (let fire = 0; fire < count; fire++) {
      if (windowMask[fire]) {
        shown++;
        acres += areaAcres[fire];
      }
    }
    ctx.setReadout('firesShown', shown);
    ctx.setReadout('acresShown', measuresReady ? acres : null);
    ctx.setReadout('gpuCount', windowCount);
    let cumulative = 0;
    let cumulativeCount = 0;
    for (const fire of data.byDate) {
      if (data.days[fire] > time) break;
      cumulative += areaAcres[fire];
      cumulativeCount++;
    }
    ctx.setReadout(
      'cumulative',
      measuresReady ? `${formatNumber(cumulative)} acres in ${cumulativeCount} fires` : null
    );
  };

  const measuresReader = new SummaryReader(
    resources,
    'seasons-measures',
    [
      {buffer: areas, size: count * 4},
      {buffer: groupAreas, size: 16},
      {buffer: groupCounts, size: 16}
    ],
    bytes => {
      if (destroyed) return;
      areaAcres = Float32Array.from(
        new Float32Array(bytes, 0, count),
        value => value * ACRES_PER_SQUARE_METER
      );
      groupAcres = Float32Array.from(
        new Float32Array(bytes, count * 4, 4),
        value => value * ACRES_PER_SQUARE_METER
      );
      groupFires = new Uint32Array(bytes.slice(count * 4 + 16, count * 4 + 32));
      measuresReady = true;
      let nifc = 0;
      for (let fire = 0; fire < count; fire++) nifc += data.acres[fire];
      ctx.setReadout(
        'totalGpu',
        areaAcres.reduce((a, b) => a + b, 0)
      );
      ctx.setReadout('totalNifc', nifc);
      const relativeDifferences = Array.from(
        areaAcres,
        (acres, fire) => Math.abs(acres - data.acres[fire]) / Math.max(data.acres[fire], 1)
      ).sort((left, right) => left - right);
      const medianDifference = relativeDifferences[Math.floor(relativeDifferences.length / 2)] ?? 0;
      const maximumDifference = relativeDifferences.at(-1) ?? 0;
      ctx.setReadout(
        'medianRelativeDifference',
        `${(medianDifference * 100).toFixed(1)}% median; ${(maximumDifference * 100).toFixed(1)}% max`
      );
      ctx.setChart('ledgerChart', {
        kind: 'scatter',
        x: Array.from(data.acres),
        y: Array.from(areaAcres, (acres, fire) => acres - data.acres[fire]),
        xLabel: 'agency acres',
        yLabel: 'GPU − agency acres',
        description:
          'One point per loaded perimeter; the vertical difference is GPU WGS84 area minus the agency acreage.'
      });
      ctx.setReadout('groupFires', Array.from(groupFires).join(' / '));
      dirty.chart = true;
      updateReadouts(clock.time);
    }
  );
  const windowReader = new SummaryReader(
    resources,
    'seasons-window',
    [
      {buffer: filterCount, size: 4},
      {buffer: filterOverflow, size: 4},
      {buffer: acceptedMask, size: count * 4}
    ],
    bytes => {
      if (destroyed) return;
      windowCount = new Uint32Array(bytes, 0, 1)[0];
      windowMask = new Uint32Array(bytes.slice(8, 8 + count * 4));
      updateReadouts(clock.time);
      ctx.requestLayers();
    }
  );

  ctx.setReadout('newest', 'none yet');
  writeWindow(clock.time, ctx.options);

  return {
    getCompiledGraphs: () => [measures, windowGraph],

    setOption(id, _value, options) {
      if (id === 'areaSystem') {
        resources.release(measures);
        measures = buildMeasures(options);
        dirty.measures = true;
      } else if (id === 'groupBy') {
        groupIdsBuffer.write(
          Uint32Array.from(options.groupBy === 'year' ? data.yearIndex : data.sizeClass)
        );
        dirty.measures = true;
      } else if (id === 'follow' && options.follow) {
        lastFlight = -Infinity;
        newest = -2;
      }
      writeWindow(clock.time, options);
      dirty.window = true;
      dirty.chart = true;
      ctx.requestLayers();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const fire = findFeatureAt(layout, event.coordinate[0], event.coordinate[1]);
      if (fire < 0 || !windowMask[fire]) return null;
      return `${describeFire(data, fire)}${measuresReady ? ` · GPU area ${formatNumber(areaAcres[fire])} acres` : ''}`;
    },

    encode(commandEncoder, frame) {
      const time = clock.advance(frame);
      const options = ctx.options;
      if (dirty.measures) {
        measures.encode(commandEncoder, {parameters: undefined});
        measuresReader.request(commandEncoder);
        dirty.measures = false;
      }
      if (clock.moved || dirty.window) {
        writeWindow(time, options);
        windowGraph.encode(commandEncoder, {parameters: undefined});
        updateNewest(time, options);
        windowReader.request(commandEncoder);
        dirty.window = false;
      }
      if (dirty.chart || (clock.moved && Math.abs(time - lastChartDay) >= 5)) {
        updateCharts(time);
        dirty.chart = false;
      }
      measuresReader.flush(commandEncoder);
      windowReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const age = options.colorBy === 'age';
      const palette = options.colorBy === 'year' ? YEAR_PALETTE : SIZE_PALETTE;
      layers.push(
        new FeatureTriangleLayer({
          id: 'season-fill',
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          corners: fill.corners,
          featureRows: fill.featureRows,
          instanceCount: fill.triangleCount,
          values: display,
          valueMapping: age ? 'ramp' : 'category',
          colormap: 'fire',
          valueRange: [0, 1],
          color: [255, 255, 255, (ctx.getViewport()?.zoom ?? 0) >= 5.7 ? 215 : 75]
        })
      );
      if (options.showOutlines) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'season-outline',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            segments: edges.segments,
            instanceCount: edges.edgeCount,
            valueIndices: edges.featureRows,
            values: acceptedMask,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: dark ? [235, 240, 248, 140] : [40, 52, 70, 140],
            noDataColor: [0, 0, 0, 0],
            widthPixels: 1
          })
        );
      }
      if (options.showMarkers) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'season-markers',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            positions: centroids,
            instanceCount: count,
            radiusPixels: 16,
            radiusMinPixels: 2,
            radiusMaxPixels: 16,
            sizeValues: acreageValues,
            sizeMaximumValue: 1_000_000,
            sizeScale: 'sqrt',
            shape: 'ring',
            outlineColor: dark ? [229, 233, 240, 220] : [31, 41, 51, 210],
            values: age ? display : displayCategory,
            valueFormat: age ? 'float32' : 'uint32',
            colormap: age ? 'fire' : 'category',
            palette,
            valueRange: [0, 1],
            noDataColor: [0, 0, 0, 0]
          })
        );
      }
      if (options.highlightNewest && newest >= 0 && newestEdgeCount > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'season-newest',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            segments: selectionSegments,
            instanceCount: newestEdgeCount,
            color: [255, 214, 90, 255],
            widthPixels: 2.5
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      measuresReader.stop();
      windowReader.stop();
      resources.destroy();
    }
  };
}
