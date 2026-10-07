// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {GPUCrossfilter} from '@luma.gl/experimental/gpu-crossfilter';
import {DrawCommandBuffer, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import type {RampName} from '../../engine/ramps';
import {
  formatCount,
  getViewportMetricBounds,
  SpatialAnalysisResources
} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  createViewImporter,
  dayToYear,
  formatShare,
  loadOsmHistory,
  OSM_DATASET_ID,
  OSM_KIND_COLORS,
  OSM_KIND_LABELS,
  OSM_KINDS,
  scaleToFull,
  yearToDay
} from './osm-history-data';

/** Option state of the crossfilter scene. */
export type OsmCrossfilterOptions = {
  years: readonly [number, number];
  kind: string;
  topContributors: string;
  viewBrush: boolean;
  colorBy: 'kind' | 'year';
  ramp: Exclude<RampName, 'grayscale' | 'diverging'>;
  pointSize: number;
  showContext: boolean;
  contextOpacity: number;
};

/** Histogram bins over the whole history: 76 quarter years of 91.3 days. */
const TIME_BINS = 76;
const BIN_DAYS = 365.2425 / 4;
const TIME_DOMAIN: [number, number] = [0, TIME_BINS * BIN_DAYS];
const FIRST_YEAR = 2007.5;
const LAST_YEAR = 2026.4;
const NO_CONTRIBUTOR_LIMIT = 'all';

/**
 * Linked brushing over 400,000 OSM node creations. `GPUCrossfilter` owns four selection
 * dimensions (map view rectangle, creation time, kind, contributor rank) and publishes, in one
 * compiled graph, two time histograms, a per-kind count, the visible row ids with a draw count and
 * the selected count. A brush change is a small control-buffer write followed by one re-encode of
 * the same graph; the map redraws from the GPU-compacted ids with an indirect draw.
 */
export async function createOsmCrossfilter(
  ctx: SceneContext<OsmCrossfilterOptions>
): Promise<SceneInstance<OsmCrossfilterOptions>> {
  const {device} = ctx;
  const history = loadOsmHistory(ctx.datasets.get(OSM_DATASET_ID));
  const count = history.count;
  const resources = new SpatialAnalysisResources(device, 'osm-xf');
  const coordinateOrigin: [number, number, number] = [history.origin[0], history.origin[1], 0];
  let destroyed = false;

  // ---- Source columns (one buffer per column, shared by selections, views and layers) ---------------
  const xs = new Float32Array(count);
  const ys = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    xs[i] = history.positions[i * 2];
    ys[i] = history.positions[i * 2 + 1];
  }
  const positionsBuffer = resources.createBuffer('positions', history.positions);
  const xBuffer = resources.createBuffer('x', xs);
  const yBuffer = resources.createBuffer('y', ys);
  const daysBuffer = resources.createBuffer('days', history.days);
  const kindBuffer = resources.createBuffer('kind', history.kind);
  const contributorBuffer = resources.createBuffer('contributor', history.contributor);

  // ---- Outputs --------------------------------------------------------------------------------------
  const timeOthers = resources.createBuffer('time-others', TIME_BINS * 4);
  const timeSelected = resources.createBuffer('time-selected', TIME_BINS * 4);
  const kindCounts = resources.createBuffer('kind-counts', OSM_KINDS.length * 4);
  const selectedCount = resources.createBuffer('selected-count', 4);
  const visibleIds = resources.createBuffer('visible-ids', count * 4);
  const draw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'osm-xf-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );

  // ---- Graph ----------------------------------------------------------------------------------------
  const graph = new GPUCommandGraph<void>(device, {id: 'osm-crossfilter'});
  const v = createViewImporter(graph, 'xf');
  const x = v('x', xBuffer, 'float32', count);
  const y = v('y', yBuffer, 'float32', count);
  const days = v('days', daysBuffer, 'float32', count);
  const kind = v('kind', kindBuffer, 'uint32', count);
  const contributor = v('contributor', contributorBuffer, 'uint32', count);
  const filter = new GPUCrossfilter<void>(graph, {
    id: 'osm-crossfilter',
    dimensions: [
      {id: 'map', kind: 'bounds', x, y},
      {id: 'time', kind: 'range', input: days},
      {id: 'kind', kind: 'range', input: kind},
      {id: 'contributor', kind: 'range', input: contributor}
    ],
    views: [
      {
        id: 'time-others',
        kind: 'histogram',
        dimension: 'time',
        input: days,
        domain: TIME_DOMAIN,
        output: v('time-others', timeOthers, 'uint32', TIME_BINS)
      },
      {
        id: 'time-selected',
        kind: 'histogram',
        dimension: 'time',
        includeOwnSelection: true,
        input: days,
        domain: TIME_DOMAIN,
        output: v('time-selected', timeSelected, 'uint32', TIME_BINS)
      },
      {
        id: 'kind-others',
        kind: 'group',
        dimension: 'kind',
        keys: kind,
        output: v('kind-counts', kindCounts, 'uint32', OSM_KINDS.length)
      },
      {
        id: 'selected-count',
        kind: 'count',
        output: v('selected-count', selectedCount, 'uint32', 1)
      },
      {
        id: 'visible',
        kind: 'visibility',
        output: v('visible-ids', visibleIds, 'uint32', count),
        count: graph.importGPUData('visible-draw-count', draw.getInstanceCountData(0))
      }
    ]
  });
  // The controller is destroyed after the graph that was compiled from it.
  resources.track(filter);
  filter.addToGraph(graph);
  const compiled = resources.track(graph.compile());

  // ---- Brush state ----------------------------------------------------------------------------------
  let dirty = true;
  let mapBounds: [number, number, number, number] | null = null;
  let timeBrush: [number, number] | null = null;
  let selectedKind = -1;
  let latest: {
    others: Float32Array;
    selected: Float32Array;
    kinds: Uint32Array;
    selectedCount: number;
  } | null = null;

  function applyBrushes(state: OsmCrossfilterOptions): void {
    const [from, to] = state.years;
    if (from <= FIRST_YEAR + 0.01 && to >= LAST_YEAR - 0.01) {
      filter.clear('time');
      timeBrush = null;
    } else {
      timeBrush = [yearToDay(history, from), yearToDay(history, to)];
      filter.setRange('time', timeBrush);
    }
    if (state.kind === 'all') {
      filter.clear('kind');
      selectedKind = -1;
    } else {
      selectedKind = Number(state.kind);
      filter.setRange('kind', [selectedKind, selectedKind]);
    }
    if (state.topContributors === NO_CONTRIBUTOR_LIMIT) {
      filter.clear('contributor');
    } else {
      filter.setRange('contributor', [0, Number(state.topContributors) - 1]);
    }
    if (!state.viewBrush) {
      filter.clear('map');
      mapBounds = null;
    }
    dirty = true;
  }

  function publishCharts(): void {
    if (!latest) return;
    const fraction = history.full.sampleFraction;
    const centers = Float32Array.from({length: TIME_BINS}, (_, bin) =>
      dayToYear(history, (bin + 0.5) * BIN_DAYS)
    );
    const markers = timeBrush
      ? [
          {x: dayToYear(history, timeBrush[0]), label: 'from'},
          {x: dayToYear(history, timeBrush[1]), label: 'to'}
        ]
      : [];
    ctx.setChart('timeChart', {
      kind: 'line',
      xLabel: 'creation date',
      yLabel: 'nodes per quarter (estimate)',
      height: 130,
      formatX: value => value.toFixed(0),
      markers,
      series: [
        {
          label: 'passing the map, kind and contributor filters',
          x: centers,
          y: Float32Array.from(latest.others, value => value / fraction),
          color: 5
        },
        {
          label: 'also inside the time brush',
          x: centers,
          y: Float32Array.from(latest.selected, value => value / fraction),
          area: true,
          color: 0
        }
      ],
      description:
        'Nodes created per quarter among the rows that pass every filter except time, and the part inside the time brush.'
    });
    ctx.setChart('kindChart', {
      kind: 'bars',
      labels: OSM_KINDS.map(name => OSM_KIND_LABELS[name]),
      values: Float32Array.from(latest.kinds, value => value / fraction),
      highlight: selectedKind >= 0 ? [selectedKind] : [],
      height: 120,
      description:
        'Nodes per kind among the rows that pass every filter except kind (full-history estimate).'
    });
  }

  const reader = new SummaryReader(
    resources,
    'osm-xf-summary',
    [
      {buffer: timeOthers, size: TIME_BINS * 4},
      {buffer: timeSelected, size: TIME_BINS * 4},
      {buffer: kindCounts, size: OSM_KINDS.length * 4},
      {buffer: selectedCount, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      latest = {
        others: Float32Array.from(words.subarray(0, TIME_BINS)),
        selected: Float32Array.from(words.subarray(TIME_BINS, TIME_BINS * 2)),
        kinds: words.slice(TIME_BINS * 2, TIME_BINS * 2 + OSM_KINDS.length),
        selectedCount: words[TIME_BINS * 2 + OSM_KINDS.length]
      };
      ctx.setReadout(
        'selected',
        `${formatCount(scaleToFull(history, latest.selectedCount))} nodes`
      );
      ctx.setReadout('share', formatShare(latest.selectedCount / count));
      ctx.setReadout('sampleRows', `${formatCount(latest.selectedCount)} of ${formatCount(count)}`);
      let top = 0;
      for (let k = 1; k < OSM_KINDS.length; k++) if (latest.kinds[k] > latest.kinds[top]) top = k;
      ctx.setReadout('topKind', latest.kinds[top] > 0 ? OSM_KIND_LABELS[OSM_KINDS[top]] : 'none');
      publishCharts();
    }
  );

  applyBrushes(ctx.options);

  return {
    getCompiledGraphs: () => [compiled],

    setOption(id, _value, state) {
      switch (id) {
        case 'years':
        case 'kind':
        case 'topContributors':
        case 'viewBrush':
          applyBrushes(state);
          break;
        default:
          ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id === 'reset') {
        ctx.setOptions(
          {
            years: [FIRST_YEAR, LAST_YEAR],
            kind: 'all',
            topContributors: NO_CONTRIBUTOR_LIMIT,
            viewBrush: false
          },
          {notify: true}
        );
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (ctx.options.viewBrush) {
        // The map brush follows the camera; snap to 50 m so a still view is not re-brushed.
        const [minX, minY, maxX, maxY] = getViewportMetricBounds(
          frame.viewport,
          history.projection
        );
        const next: [number, number, number, number] = [
          Math.floor(minX / 50) * 50,
          Math.floor(minY / 50) * 50,
          Math.ceil(maxX / 50) * 50,
          Math.ceil(maxY / 50) * 50
        ];
        if (!mapBounds || next.some((value, index) => value !== mapBounds![index])) {
          mapBounds = next;
          filter.setBounds('map', next);
          dirty = true;
        }
      }
      if (dirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.request(commandEncoder);
      } else {
        reader.flush(commandEncoder);
      }
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.showContext) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'osm-xf-context',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: count,
            colormap: 'uniform',
            color: dark ? [200, 205, 220, 255] : [70, 75, 90, 255],
            radiusPixels: 0.9,
            opacity: options.contextOpacity
          })
        );
      }
      const byKind = options.colorBy === 'kind';
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'osm-xf-selected',
          coordinateOrigin,
          positions: positionsBuffer,
          ids: visibleIds,
          drawCommands: draw,
          values: byKind ? kindBuffer : daysBuffer,
          valueFormat: byKind ? 'uint32' : 'float32',
          colormap: byKind ? 'category' : options.ramp,
          valueRange: [0, history.maxDay],
          palette: OSM_KIND_COLORS,
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
