// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {
  and,
  column,
  getGPUTimeWindowParameterValues,
  GPUDataFrame,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPUTimeWindowFilter,
  parameter,
  type GPUDataFrameQueryParameters
} from '@luma.gl/experimental/gpu-dataframe';
import {GPUTable} from '@luma.gl/experimental/gpu-tables';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {GPUData, GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {createPlaybackClock} from '../../engine/playback';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  createViewImporter,
  dayToYear,
  formatMonthIndex,
  formatShare,
  loadOsmHistory,
  OSM_DATASET_ID,
  OSM_KIND_COLORS,
  OSM_KIND_LABELS,
  OSM_KINDS,
  scaleToFull
} from './osm-history-data';

/** Option state of the import-waves scene. */
export type OsmImportWavesOptions = {
  play: boolean;
  time: number;
  speed: number;
  loop: boolean;
  width: number;
  colorBy: 'kind' | 'prolific';
  prolificCount: number;
  spikeRatio: number;
  spikeMinimum: number;
  pointSize: number;
  showContext: boolean;
  contextOpacity: number;
};

type FrameSchema = {day: 'float32'; kind: 'uint32'; contributor: 'uint32'};

/** Spike runs of consecutive months collapse to their peak; one run ends after a calm month. */
export function findSpikes(monthly: ArrayLike<number>, ratio: number, minimum: number): number[] {
  const flagged: number[] = [];
  const neighbourhood = 12;
  for (let month = 0; month < monthly.length; month++) {
    const others: number[] = [];
    for (
      let other = Math.max(0, month - neighbourhood);
      other <= Math.min(monthly.length - 1, month + neighbourhood);
      other++
    ) {
      if (other !== month) others.push(monthly[other]);
    }
    others.sort((a, b) => a - b);
    const baseline = Math.max(others[Math.floor(others.length / 2)], minimum / 10);
    if (monthly[month] > ratio * baseline && monthly[month] >= minimum) flagged.push(month);
  }
  const peaks: number[] = [];
  let run: number[] = [];
  const closeRun = () => {
    if (run.length)
      peaks.push(run.reduce((best, m) => (monthly[m] > monthly[best] ? m : best), run[0]));
    run = [];
  };
  for (const month of flagged) {
    if (run.length && month - run[run.length - 1] > 1) closeRun();
    run.push(month);
  }
  closeRun();
  return peaks;
}

/**
 * Import waves. A `GPUDataFrame` over the sample's day, kind and contributor columns answers two
 * queries on the GPU: a histogram with irregular edges at the true calendar-month boundaries (the
 * monthly line), and a filtered, grouped count by kind inside the playing window (the bars; the
 * window is two query parameters). A `GPUTimeWindowFilter` draws the same window on the map. The
 * CPU only reads the 228 monthly counts back to mark spikes.
 */
export async function createOsmImportWaves(
  ctx: SceneContext<OsmImportWavesOptions>
): Promise<SceneInstance<OsmImportWavesOptions>> {
  const {device} = ctx;
  const history = loadOsmHistory(ctx.datasets.get(OSM_DATASET_ID));
  const count = history.count;
  const monthCount = history.full.monthCount;
  const fraction = history.full.sampleFraction;
  const resources = new SpatialAnalysisResources(device, 'osm-waves');
  const coordinateOrigin: [number, number, number] = [history.origin[0], history.origin[1], 0];
  let destroyed = false;

  const positionsBuffer = resources.createBuffer('positions', history.positions);
  const daysBuffer = resources.createBuffer('days', history.days);
  const kindBuffer = resources.createBuffer('kind', history.kind);
  const contributorBuffer = resources.createBuffer('contributor', history.contributor);
  const prolificBuffer = resources.createBuffer('prolific', count * 4);

  // ---- Dataframe over the same buffers, two compiled queries --------------------------------------
  const makeVector = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    buffer: Buffer
  ) =>
    new GPUVector({
      type: 'data',
      name,
      format,
      data: [new GPUData<Format>({buffer, format, length: count, ownsBuffer: false})],
      ownsData: false
    });
  const frame = new GPUDataFrame<FrameSchema>({
    table: new GPUTable<FrameSchema>({
      vectors: {
        day: makeVector('day', 'float32', daysBuffer),
        kind: makeVector('kind', 'uint32', kindBuffer),
        contributor: makeVector('contributor', 'uint32', contributorBuffer)
      }
    }),
    ownership: 'borrowed'
  });
  const monthlyQuery = frame
    .histogram('day', {edges: Array.from(history.monthEdges)})
    .compile(new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'osm-waves-monthly'}));
  const windowQuery = frame
    .filter(
      and(
        column('day').greaterThanOrEqual(parameter('from', 0)),
        column('day').lessThan(parameter('to', history.maxDay + 1))
      )
    )
    .groupBy('kind', {groupCount: OSM_KINDS.length})
    .aggregate({nodes: 'count'})
    .compile(new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {id: 'osm-waves-window'}));
  resources.track({
    destroy: () => {
      windowQuery.destroy();
      monthlyQuery.destroy();
      frame.destroy();
    }
  });
  const physicalBuffer = (data: GPUData): Buffer =>
    data.buffer instanceof Buffer ? data.buffer : data.buffer.buffer;
  const monthlyData = monthlyQuery.table.gpuVectors.count.data[0];
  const kindData = windowQuery.table.gpuVectors.nodes.data[0];
  if (monthlyData.byteOffset !== 0 || kindData.byteOffset !== 0) {
    throw new Error('Dataframe results are expected at offset 0 of their own buffers');
  }

  // ---- Window filter for the map --------------------------------------------------------------------
  const windowIds = resources.createBuffer('window-ids', count * 4);
  const windowCount = resources.createBuffer('window-count', 4);
  const windowOverflow = resources.createBuffer('window-overflow', 4);
  const windowParameters = resources.createParameterBuffer(
    'window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  const windowDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'osm-waves-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const mapGraph = new GPUCommandGraph<void>(device, {id: 'osm-waves-map'});
  const v = createViewImporter(mapGraph, 'm');
  mapGraph.add(
    new GPUTimeWindowFilter({
      id: 'waves-window',
      timestamps: v('days', daysBuffer, 'float32', count),
      window: windowParameters.importToGraph(mapGraph),
      output: {
        ids: v('ids', windowIds, 'uint32', count),
        count: v('count', windowCount, 'uint32', 1),
        overflow: v('overflow', windowOverflow, 'uint32', 1)
      },
      drawInstanceCount: mapGraph.importGPUData(
        'waves-draw-count',
        windowDraw.getInstanceCountData(0)
      )
    })
  );
  const mapCompiled: CompiledGPUCommandGraph<void> = resources.track(mapGraph.compile());

  // ---- State ----------------------------------------------------------------------------------------
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [0, monthCount - 1], secondsPerLoop: 70, step: 1}
  );
  let monthlySample: Float32Array | null = null;
  let monthlyScaled: Float32Array | null = null;
  let spikes: number[] = [];
  let kindCounts = new Uint32Array(OSM_KINDS.length);
  let windowKey = '';
  let monthlyEncoded = false;
  let lastChartMonth = -1;
  let prolificLimit = -1;
  const monthMiddle = Float32Array.from({length: monthCount}, (_, month) =>
    dayToYear(history, (history.monthEdges[month] + history.monthEdges[month + 1]) / 2)
  );
  const exactMonthly = Float32Array.from(history.full.monthTotal);

  function writeProlific(limit: number): void {
    if (limit === prolificLimit) return;
    prolificLimit = limit;
    const flags = new Uint32Array(count);
    for (let i = 0; i < count; i++) flags[i] = history.contributor[i] < limit ? 1 : 0;
    prolificBuffer.write(flags);
  }

  function updateSpikes(): void {
    if (!monthlyScaled) return;
    spikes = findSpikes(monthlyScaled, ctx.options.spikeRatio, ctx.options.spikeMinimum);
    const lines = spikes.map(
      month =>
        `${formatMonthIndex(history, month)}: ${formatCount(history.full.monthTotal[month])} nodes, ${formatShare(history.full.monthTopShare[month])} by one account`
    );
    ctx.setReadout('spikes', lines.length ? lines.join('\n') : 'no spikes at this sensitivity');
    lastChartMonth = -1;
    publishCharts(Math.round(clock.time));
  }

  function publishCharts(month: number): void {
    if (!monthlyScaled || month === lastChartMonth) return;
    lastChartMonth = month;
    ctx.setReadout(
      'check',
      `${formatCount(monthlyScaled[month])} from the sample, ${formatCount(history.full.monthTotal[month])} exact`
    );
    const now = {x: monthMiddle[month], label: formatMonthIndex(history, month)};
    ctx.setChart('monthly', {
      kind: 'line',
      xLabel: 'month',
      yLabel: 'nodes created per month',
      height: 150,
      formatX: value => value.toFixed(0),
      series: [
        {
          label: 'GPU histogram of the sample, scaled up',
          x: monthMiddle,
          y: monthlyScaled,
          color: 0
        },
        {label: 'exact full-history count', x: monthMiddle, y: exactMonthly, dashed: true, color: 5}
      ],
      markers: [
        ...spikes.filter(spike => spike !== month).map(spike => ({x: monthMiddle[spike]})),
        now
      ],
      description:
        'Nodes created per calendar month. Vertical rules mark detected spikes; the labelled rule is the playing window.'
    });
    ctx.setChart('share', {
      kind: 'line',
      xLabel: 'month',
      yLabel: "share by the month's top account",
      height: 100,
      formatX: value => value.toFixed(0),
      yDomain: [0, 1],
      formatY: value => `${Math.round(value * 100)}%`,
      guides: [{y: 0.5, label: 'half'}],
      series: [
        {label: 'top account share', x: monthMiddle, y: history.full.monthTopShare, color: 2}
      ],
      markers: [now],
      description:
        "Share of each month's nodes made by that month's single most active contributor (anonymous), exact over the full history."
    });
  }

  const reader = new SummaryReader(
    resources,
    'osm-waves-summary',
    [
      {buffer: physicalBuffer(monthlyData), size: monthCount * 4},
      {buffer: physicalBuffer(kindData), size: OSM_KINDS.length * 4},
      {buffer: windowCount, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      monthlySample = Float32Array.from(words.subarray(0, monthCount));
      monthlyScaled = Float32Array.from(monthlySample, value => value / fraction);
      kindCounts = words.slice(monthCount, monthCount + OSM_KINDS.length);
      const shown = words[monthCount + OSM_KINDS.length];
      ctx.setReadout('window', `${formatCount(scaleToFull(history, shown))} nodes`);
      let top = 0;
      for (let k = 1; k < OSM_KINDS.length; k++) if (kindCounts[k] > kindCounts[top]) top = k;
      ctx.setReadout('topKind', shown > 0 ? OSM_KIND_LABELS[OSM_KINDS[top]] : 'none');
      ctx.setChart('kinds', {
        kind: 'bars',
        labels: OSM_KINDS.map(name => OSM_KIND_LABELS[name]),
        values: Float32Array.from(kindCounts, value => value / fraction),
        highlight: shown > 0 ? [top] : [],
        height: 110,
        description: 'Nodes created inside the playing window, by kind (full-history estimate).'
      });
      updateSpikes();
    }
  );

  ctx.setReadout('rows', `${formatCount(count)} of ${formatCount(history.full.fullCount)} nodes`);
  writeProlific(ctx.options.prolificCount);

  return {
    // The dataframe queries own their compiled graphs privately; the map graph is the one exposed.
    getCompiledGraphs: () => [mapCompiled],

    setOption(id, _value, state) {
      switch (id) {
        case 'width':
          windowKey = '';
          break;
        case 'prolificCount':
          writeProlific(state.prolificCount);
          ctx.requestLayers();
          break;
        case 'spikeRatio':
        case 'spikeMinimum':
          updateSpikes();
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

    encode(commandEncoder, frameInfo) {
      const options = ctx.options;
      const playhead = clock.advance(frameInfo);
      const month = Math.min(Math.max(Math.round(playhead), 0), monthCount - 1);
      const last = Math.min(month + Math.round(options.width), monthCount);
      const from = history.monthEdges[month];
      const to = history.monthEdges[last];
      ctx.setReadout(
        'clock',
        options.width > 1
          ? `${formatMonthIndex(history, month)} to ${formatMonthIndex(history, last - 1)}`
          : formatMonthIndex(history, month)
      );
      const key = `${month}:${options.width}`;
      let encoded = false;
      if (!monthlyEncoded) {
        monthlyQuery.encode(commandEncoder, {});
        monthlyEncoded = true;
        encoded = true;
      }
      if (key !== windowKey) {
        windowKey = key;
        windowQuery.encode(commandEncoder, {from, to});
        windowParameters.write(getGPUTimeWindowParameterValues({start: from, end: to - 0.01}));
        mapCompiled.encode(commandEncoder, {parameters: undefined});
        encoded = true;
        ctx.setReadout(
          'topAccount',
          `${formatShare(history.full.monthTopShare[month])} of ${formatMonthIndex(history, month)}`
        );
        publishCharts(month);
      }
      if (encoded) reader.request(commandEncoder);
      else reader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.showContext) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'osm-waves-context',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: count,
            colormap: 'uniform',
            color: dark ? [200, 205, 220, 255] : [70, 75, 90, 255],
            radiusPixels: 0.8,
            opacity: options.contextOpacity
          })
        );
      }
      const byKind = options.colorBy === 'kind';
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'osm-waves-window',
          coordinateOrigin,
          positions: positionsBuffer,
          ids: windowIds,
          drawCommands: windowDraw,
          values: byKind ? kindBuffer : prolificBuffer,
          valueFormat: 'uint32',
          colormap: byKind ? 'category' : 'mask',
          palette: OSM_KIND_COLORS,
          color: [255, 150, 40, 255],
          noDataColor: dark ? [150, 190, 235, 235] : [40, 100, 170, 235],
          radiusPixels: options.pointSize,
          opacity: 0.95
        })
      );
      return layers;
    },

    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
    }
  };
}
