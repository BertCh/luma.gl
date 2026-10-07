// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {GPUGroupStatistics} from '@luma.gl/experimental/gpu-dataframe';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {SpatialAnalysisPointLayer, type SpatialAnalysisColor} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {RampName} from '../../engine/ramps';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  createViewImporter,
  formatMonthYear,
  formatShare,
  loadOsmHistory,
  OSM_DATASET_ID,
  scaleToFull,
  yearToDay
} from './osm-history-data';

/** Option state of the contributor concentration scene. */
export type OsmContributorsOptions = {
  years: readonly [number, number];
  kind: string;
  topN: number;
  inspect: number;
  colorBy: 'highlight' | 'rank';
  ramp: Exclude<RampName, 'grayscale' | 'diverging' | 'cividis'>;
  pointSize: number;
  contextOpacity: number;
};

const FIRST_YEAR = 2007.5;
const LAST_YEAR = 2026.4;
const LORENZ_POINTS = 101;
const TOP_BARS = 10;

/**
 * Who drew New York? A kernel marks the nodes inside a year and kind window; `GPUGroupStatistics`
 * groups those nodes by anonymous contributor rank (dense keys, one row per contributor) and
 * reports each contributor's count and the minimum, maximum, mean and median creation day. A
 * second kernel turns the top-N rank threshold into a per-node state the map colors by. The CPU
 * only sorts the 11,361 per-contributor counts to draw the Lorenz curve.
 */
export async function createOsmContributors(
  ctx: SceneContext<OsmContributorsOptions>
): Promise<SceneInstance<OsmContributorsOptions>> {
  const {device} = ctx;
  const history = loadOsmHistory(ctx.datasets.get(OSM_DATASET_ID));
  const count = history.count;
  const keyCount = history.full.contributorCount;
  const resources = new SpatialAnalysisResources(device, 'osm-who');
  const coordinateOrigin: [number, number, number] = [history.origin[0], history.origin[1], 0];
  let destroyed = false;

  const positionsBuffer = resources.createBuffer('positions', history.positions);
  const daysBuffer = resources.createBuffer('days', history.days);
  const kindBuffer = resources.createBuffer('kind', history.kind);
  const contributorBuffer = resources.createBuffer('contributor', history.contributor);
  const maskBuffer = resources.createBuffer('mask', count * 4);
  const stateBuffer = resources.createBuffer('state', count * 4);
  const rankBuffer = resources.createBuffer('rank', count * 4);
  const parameters = resources.createParameterBuffer('window', 'float32', 4);

  const groupKeys = resources.createBuffer('group-keys', keyCount * 4);
  const groupCounts = resources.createBuffer('group-counts', keyCount * 4);
  const groupCount = resources.createBuffer('group-count', 4);
  const groupOverflow = resources.createBuffer('group-overflow', 4);
  const minimums = resources.createBuffer('minimums', keyCount * 4);
  const maximums = resources.createBuffer('maximums', keyCount * 4);
  const means = resources.createBuffer('means', keyCount * 4);
  const medians = resources.createBuffer('medians', keyCount * 4);

  // ---- Graph 1: window mask and per-contributor statistics ------------------------------------------
  const statsGraph = new GPUCommandGraph<void>(device, {id: 'osm-who-stats'});
  const s = createViewImporter(statsGraph, 's');
  const maskView = s('mask', maskBuffer, 'uint32', count);
  addKernelPass(statsGraph, {
    id: 'osm-who-mask',
    invocationCount: count,
    bindings: [
      {name: 'days', view: s('days', daysBuffer, 'float32', count), type: 'f32', access: 'read'},
      {name: 'kinds', view: s('kind', kindBuffer, 'uint32', count), type: 'u32', access: 'read'},
      {
        name: 'window',
        view: parameters.importToGraph(statsGraph),
        type: 'f32',
        access: 'read'
      },
      {name: 'mask', view: maskView, type: 'u32', access: 'read_write'}
    ],
    body: `let day = days[daysOffset + index];
  let kindWanted = window[windowOffset + 2u];
  let kindOk = kindWanted < 0.0 || kinds[kindsOffset + index] == u32(kindWanted);
  let inside = day >= window[windowOffset] && day <= window[windowOffset + 1u] && kindOk;
  mask[maskOffset + index] = select(0u, 1u, inside);`
  });
  statsGraph.add(
    new GPUGroupStatistics({
      id: 'osm-who-groups',
      keys: s('contributor', contributorBuffer, 'uint32', count),
      mask: maskView,
      keyCount,
      columns: [
        {
          values: s('days', daysBuffer, 'float32', count),
          statistics: ['minimum', 'maximum', 'mean', 'median'],
          output: {
            minimums: s('minimums', minimums, 'float32', keyCount),
            maximums: s('maximums', maximums, 'float32', keyCount),
            means: s('means', means, 'float32', keyCount),
            medians: s('medians', medians, 'float32', keyCount)
          }
        }
      ],
      output: {
        keys: s('group-keys', groupKeys, 'uint32', keyCount),
        counts: s('group-counts', groupCounts, 'uint32', keyCount),
        count: s('group-count', groupCount, 'uint32', 1),
        overflow: s('group-overflow', groupOverflow, 'uint32', 1)
      }
    })
  );
  const statsCompiled = resources.track(statsGraph.compile());

  // ---- Graph 2: per-node state for the map (cheap; runs when the threshold moves) -------------------
  const stateGraph = new GPUCommandGraph<void>(device, {id: 'osm-who-state'});
  const t = createViewImporter(stateGraph, 't');
  addKernelPass(stateGraph, {
    id: 'osm-who-state',
    invocationCount: count,
    bindings: [
      {name: 'mask', view: t('mask', maskBuffer, 'uint32', count), type: 'u32', access: 'read'},
      {
        name: 'ranks',
        view: t('contributor', contributorBuffer, 'uint32', count),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'window',
        view: parameters.importToGraph(stateGraph),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'states',
        view: t('state', stateBuffer, 'uint32', count),
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'rankValues',
        view: t('rank', rankBuffer, 'float32', count),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let inside = mask[maskOffset + index] != 0u;
  let rank = ranks[ranksOffset + index];
  let limit = u32(window[windowOffset + 3u]);
  var level = 0u;
  if (inside) {
    level = 1u;
    if (rank < limit) { level = 2u; }
    if (rank == 0u) { level = 3u; }
  }
  states[statesOffset + index] = level;
  let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  rankValues[rankValuesOffset + index] = select(nan, f32(rank), inside);`
  });
  const stateCompiled = resources.track(stateGraph.compile());

  // ---- Readback and CPU summary ---------------------------------------------------------------------
  let latest: {
    counts: Uint32Array;
    minimums: Float32Array;
    maximums: Float32Array;
    medians: Float32Array;
  } | null = null;

  const reader = new SummaryReader(
    resources,
    'osm-who-summary',
    [
      {buffer: groupCounts, size: keyCount * 4},
      {buffer: minimums, size: keyCount * 4},
      {buffer: maximums, size: keyCount * 4},
      {buffer: medians, size: keyCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      latest = {
        counts: new Uint32Array(bytes, 0, keyCount),
        minimums: new Float32Array(bytes, keyCount * 4, keyCount),
        maximums: new Float32Array(bytes, keyCount * 8, keyCount),
        medians: new Float32Array(bytes, keyCount * 12, keyCount)
      };
      summarize();
    }
  );

  function summarize(): void {
    if (!latest) return;
    const options = ctx.options;
    const {counts} = latest;
    let total = 0;
    let active = 0;
    for (let rank = 0; rank < keyCount; rank++) {
      total += counts[rank];
      if (counts[rank] > 0) active++;
    }
    ctx.setReadout('window', `${formatCount(scaleToFull(history, total))} nodes`);
    ctx.setReadout(
      'active',
      `${formatCount(active)} of ${formatCount(keyCount)} (seen in the sample)`
    );
    if (total === 0) {
      ctx.setReadout('topShare', 'no nodes in this window');
      ctx.setReadout('gini', 'n/a');
      ctx.setChart('lorenz', null);
      ctx.setChart('topBars', null);
      ctx.setReadout('inspected', 'no nodes in this window');
      return;
    }
    const limit = Math.min(options.topN, keyCount);
    let topCount = 0;
    for (let rank = 0; rank < limit; rank++) topCount += counts[rank];
    const wholeHistory = isWholeHistory(options);
    ctx.setReadout(
      'topShare',
      `${formatShare(topCount / total)} (top ${formatCount(limit)} by full-history rank)`
    );
    ctx.setReadout(
      'fullShare',
      wholeHistory && limit <= history.full.topShares.length
        ? `${formatShare(history.full.topShares[limit - 1])} (exact, all 901,827 nodes)`
        : wholeHistory
          ? 'beyond the 200 exact shares stored'
          : 'only for the whole history'
    );

    // Lorenz curve over the contributors active in the window (ascending by node count).
    const activeCounts = Float64Array.from(counts.subarray(0, keyCount).filter(value => value > 0));
    activeCounts.sort();
    const cumulative = new Float64Array(activeCounts.length + 1);
    for (let i = 0; i < activeCounts.length; i++)
      cumulative[i + 1] = cumulative[i] + activeCounts[i];
    let weighted = 0;
    for (let i = 0; i < activeCounts.length; i++) weighted += (i + 1) * activeCounts[i];
    const gini =
      (2 * weighted) / (activeCounts.length * total) -
      (activeCounts.length + 1) / activeCounts.length;
    ctx.setReadout(
      'gini',
      `${gini.toFixed(3)} in the sample (full history: ${history.full.gini.toFixed(3)})`
    );
    const px = new Float32Array(LORENZ_POINTS);
    const py = new Float32Array(LORENZ_POINTS);
    for (let p = 0; p < LORENZ_POINTS; p++) {
      px[p] = p;
      py[p] = (cumulative[Math.round((p / 100) * activeCounts.length)] / total) * 100;
    }
    const topStart = Math.max(0, (1 - Math.min(limit, active) / active) * 100);
    ctx.setChart('lorenz', {
      kind: 'line',
      xLabel: 'contributors, least to most active (%)',
      yLabel: 'cumulative share of nodes (%)',
      height: 150,
      xDomain: [0, 100],
      yDomain: [0, 100],
      series: [
        {label: 'Lorenz curve', x: px, y: py, color: 0, area: true},
        {label: 'equal shares', x: [0, 100], y: [0, 100], dashed: true, color: 5}
      ],
      markers: [{x: topStart, label: `top ${formatCount(limit)}`}],
      description:
        'Lorenz curve of nodes per contributor. The further the curve bends below the diagonal, the more a few contributors dominate; the rule marks where the top N begin.'
    });
    const barCount = Math.min(TOP_BARS, keyCount);
    ctx.setChart('topBars', {
      kind: 'bars',
      labels: Array.from({length: barCount}, (_, i) => `#${i + 1}`),
      values: Array.from({length: barCount}, (_, i) => (counts[i] / total) * 100),
      highlight: Array.from({length: Math.min(limit, barCount)}, (_, i) => i),
      height: 110,
      yLabel: 'share of nodes (%)',
      description:
        "Share of the window's nodes made by each of the ten most active contributors (full-history rank)."
    });

    const rank = Math.min(options.inspect, keyCount) - 1;
    const nodes = counts[rank];
    ctx.setReadout(
      'inspected',
      nodes === 0
        ? `Contributor #${rank + 1}: no nodes in this window`
        : `Contributor #${rank + 1}: ${formatCount(scaleToFull(history, nodes))} nodes (${formatShare(nodes / total)}), first ${formatMonthYear(history, latest.minimums[rank])}, median ${formatMonthYear(history, latest.medians[rank])}, last ${formatMonthYear(history, latest.maximums[rank])}`
    );
  }

  function isWholeHistory(options: OsmContributorsOptions): boolean {
    return (
      options.kind === 'all' &&
      options.years[0] <= FIRST_YEAR + 0.01 &&
      options.years[1] >= LAST_YEAR - 0.01
    );
  }

  function writeParameters(state: OsmContributorsOptions): void {
    const wholeTime = state.years[0] <= FIRST_YEAR + 0.01 && state.years[1] >= LAST_YEAR - 0.01;
    parameters.write(
      Float32Array.of(
        wholeTime ? -1 : yearToDay(history, state.years[0]),
        wholeTime ? 1e9 : yearToDay(history, state.years[1]),
        state.kind === 'all' ? -1 : Number(state.kind),
        Math.max(1, Math.round(state.topN))
      )
    );
  }

  writeParameters(ctx.options);
  let statsDirty = true;
  let stateDirty = true;
  ctx.setReadout('rows', `${formatCount(count)} of ${formatCount(history.full.fullCount)} nodes`);

  return {
    getCompiledGraphs: () => [statsCompiled, stateCompiled],

    setOption(id, _value, state) {
      switch (id) {
        case 'years':
        case 'kind':
          writeParameters(state);
          statsDirty = true;
          stateDirty = true;
          break;
        case 'topN':
          writeParameters(state);
          stateDirty = true;
          summarize();
          break;
        case 'inspect':
          summarize();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      if (statsDirty) {
        statsCompiled.encode(commandEncoder, {parameters: undefined});
        statsDirty = false;
        reader.request(commandEncoder);
      } else {
        reader.flush(commandEncoder);
      }
      if (stateDirty) {
        stateCompiled.encode(commandEncoder, {parameters: undefined});
        stateDirty = false;
      }
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const palette: SpatialAnalysisColor[] = [
        [0, 0, 0, 0],
        dark ? [190, 200, 220, 255] : [60, 70, 90, 255],
        [255, 170, 50, 255],
        [255, 70, 130, 255]
      ];
      const layers: Layer[] = [];
      if (options.colorBy === 'highlight') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'osm-who-rest',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: count,
            values: stateBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [palette[0], palette[1], palette[0], palette[0]],
            radiusPixels: options.pointSize,
            opacity: options.contextOpacity
          }),
          new SpatialAnalysisPointLayer({
            id: 'osm-who-top',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: count,
            values: stateBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [palette[0], palette[0], palette[2], palette[3]],
            radiusPixels: options.pointSize + 0.4,
            opacity: 0.95
          })
        );
      } else {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'osm-who-rank',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: count,
            values: rankBuffer,
            valueFormat: 'float32',
            colormap: options.ramp,
            valueRange: [0, 1000],
            sqrtScale: true,
            radiusPixels: options.pointSize,
            opacity: 0.9
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
    }
  };
}
