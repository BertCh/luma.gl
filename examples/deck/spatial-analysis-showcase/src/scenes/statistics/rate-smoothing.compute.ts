// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  addRateClusterMapRecipe,
  getGPUPermutationParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  GPU_EMPIRICAL_BAYES_SUMMARY,
  GPU_GLOBAL_SPATIAL_STATISTIC_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPU_RATE_CLUSTER_MAP_PALETTE_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPUGlobalSpatialStatistics,
  GPUSpatialEmpiricalBayesRates,
  type GPUContiguityCriterion,
  type GPUPermutationAlternative
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import type {RampName} from '../../engine/ramps';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  createByteReader,
  createChoroplethGeometry,
  formatNumber,
  getOutlineColor,
  getQuantile,
  getSortedFinite
} from './b5-common';
import {packColor} from './b5-palettes';
import {clearLegendData, setLegendData} from './b5-legend-bus';

/** Option state of the rate-smoothing scene. */
export type RateSmoothingOptions = {
  map: 'raw' | 'smoothed' | 'spatial' | 'standardized' | 'cluster';
  analyze: 'standardized' | 'smoothed';
  criterion: GPUContiguityCriterion;
  gating: 'permutation' | 'analytic';
  alternative: Exclude<GPUPermutationAlternative, 'less'>;
  falseDiscoveryRate: boolean;
  permutations: number;
  seed: number;
  significance: number;
  ramp: RampName;
  outlines: boolean;
};

/** Quadrant colors of the cluster map: not significant, HH, LH, LL, HL. */
export const RATE_CLUSTER_COLORS = [
  [176, 182, 194, 120],
  [215, 48, 39, 235],
  [145, 191, 219, 235],
  [49, 54, 149, 235],
  [253, 174, 97, 235]
] as const;

const MAXIMUM_PERMUTATIONS = 999;
const MAXIMUM_LOCAL_NEIGHBORS = 32;
const RATE_SCALE = 1e5;
const PERSON_YEARS_SMALL = 7 * 5000;

type Variant = {compiled: CompiledGPUCommandGraph<void>; key: string};

/** Data shared with `legends(state)`. */
export type RateSmoothingLegend = {rateRange: [number, number]};

/**
 * Empirical-Bayes smoothing of county traffic-death rates. One graph chains the contributors:
 * `addRateClusterMapRecipe` (`GPUEmpiricalBayesRates`, `GPUContiguityWeights`, row-standardised
 * weights, `GPULocalMoran` and `GPULocalPermutationTest`), then `GPUSpatialEmpiricalBayesRates`
 * on the recipe's own weights and four `GPUGlobalSpatialStatistics` nodes (Moran's I of each rate
 * variable). The compile-time options (weights criterion, analysed rate, gating, tail, FDR) select
 * among graphs compiled on demand; every slider is a parameter-buffer write.
 */
export async function createRateSmoothing(
  ctx: SceneContext<RateSmoothingOptions>
): Promise<SceneInstance<RateSmoothingOptions>> {
  const counties = ctx.datasets.get('us-counties');
  const mortality = ctx.datasets.get('us-county-mortality');
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'rate-smoothing');
  const geometry = createChoroplethGeometry(resources, counties);
  const {layout} = geometry;
  const n = layout.featureCount;

  const deaths = mortality.column<Float32Array>('deaths');
  const personYears = mortality.column<Float32Array>('personYears');
  const features = counties.geojson?.features ?? [];
  const nameOf = (row: number): string => {
    const properties = features[row]?.properties;
    return properties ? `${properties.name}, ${properties.state}` : `County ${row}`;
  };

  const verticesBuffer = resources.createBuffer('vertices', layout.vertices);
  const ringOffsetsBuffer = resources.createBuffer('ring-offsets', layout.ringOffsets);
  const featureRingOffsetsBuffer = resources.createBuffer(
    'feature-ring-offsets',
    layout.featureRingOffsets
  );
  const eventsBuffer = resources.createBuffer('events', deaths);
  const populationsBuffer = resources.createBuffer('person-years', personYears);
  const rawBuffer = resources.createBuffer('raw-rates', n * 4);
  const smoothedBuffer = resources.createBuffer('smoothed-rates', n * 4);
  const spatialBuffer = resources.createBuffer('spatial-rates', n * 4);
  const standardizedBuffer = resources.createBuffer('standardized-rates', n * 4);
  const summaryBuffer = resources.createBuffer(
    'eb-summary',
    GPU_EMPIRICAL_BAYES_SUMMARY.length * 4
  );
  const neighborCapacity = n * 24;
  const offsetsBuffer = resources.createBuffer('weights-offsets', (n + 1) * 4);
  const neighborsBuffer = resources.createBuffer('weights-neighbors', neighborCapacity * 4);
  const weightsBuffer = resources.createBuffer('weights-values', neighborCapacity * 4);
  const weightsOverflowBuffer = resources.createBuffer('weights-overflow', 4);
  const zScoresBuffer = resources.createBuffer('z-scores', n * 4);
  const quadrantsBuffer = resources.createBuffer('quadrants', n * 4);
  const colorsBuffer = resources.createBuffer('cluster-colors', n * 4);
  const pseudoPBuffer = resources.createBuffer('pseudo-p', n * 4);
  const significantBuffer = resources.createBuffer('significant', n * 4);
  const paletteBuffer = resources.createBuffer(
    'quadrant-palette',
    Uint32Array.from(RATE_CLUSTER_COLORS, color =>
      packColor(color[0], color[1], color[2], color[3])
    )
  );
  const globalBuffers = {
    raw: resources.createBuffer('moran-raw', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4),
    smoothed: resources.createBuffer(
      'moran-smoothed',
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4
    ),
    spatial: resources.createBuffer(
      'moran-spatial',
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4
    ),
    standardized: resources.createBuffer(
      'moran-standardized',
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4
    )
  };
  const autocorrelationParameters = resources.createParameterBuffer(
    'autocorrelation-parameters',
    'float32',
    GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  );
  const permutationParameters = resources.createParameterBuffer(
    'permutation-parameters',
    'uint32',
    GPU_PERMUTATION_PARAMETER_LENGTH
  );

  const compileVariant = (options: RateSmoothingOptions): Variant => {
    const key = compileKey(options);
    const graph = new GPUCommandGraph<void>(device, {id: `rate-smoothing-${key}`});
    const float = (id: string, buffer: typeof rawBuffer, length = n) =>
      importGraphBuffer(graph, id, buffer, 'float32', length);
    const word = (id: string, buffer: typeof rawBuffer, length = n) =>
      importGraphBuffer(graph, id, buffer, 'uint32', length);
    const events = float('events', eventsBuffer);
    const populations = float('person-years', populationsBuffer);
    const raw = float('raw', rawBuffer);
    const smoothed = float('smoothed', smoothedBuffer);
    const standardized = float('standardized', standardizedBuffer);
    const recipe = addRateClusterMapRecipe(graph, {
      id: 'rates',
      events,
      populations,
      positions: importGraphBuffer(
        graph,
        'vertices',
        verticesBuffer,
        'float32x2',
        layout.vertices.length / 2
      ),
      ringOffsets: word('ring-offsets', ringOffsetsBuffer, layout.ringOffsets.length),
      polygonOffsets: word(
        'feature-ring-offsets',
        featureRingOffsetsBuffer,
        layout.featureRingOffsets.length
      ),
      criterion: options.criterion,
      neighborCapacity,
      pairCapacity: n * 96,
      analyze: options.analyze,
      parameters: autocorrelationParameters.importToGraph(graph),
      permutation:
        options.gating === 'permutation'
          ? {
              parameters: permutationParameters.importToGraph(graph),
              maximumPermutations: MAXIMUM_PERMUTATIONS,
              alternative: options.alternative,
              maximumNeighbors: MAXIMUM_LOCAL_NEIGHBORS,
              falseDiscoveryRate: options.falseDiscoveryRate,
              pseudoPValues: float('pseudo-p', pseudoPBuffer),
              significant: word('significant', significantBuffer)
            }
          : undefined,
      palette: word('palette', paletteBuffer, GPU_RATE_CLUSTER_MAP_PALETTE_LENGTH),
      standardizedRates: standardized,
      smoothedRates: smoothed,
      rawRates: raw,
      summary: float('summary', summaryBuffer, GPU_EMPIRICAL_BAYES_SUMMARY.length),
      weights: {
        offsets: word('weights-offsets', offsetsBuffer, n + 1),
        neighbors: word('weights-neighbors', neighborsBuffer, neighborCapacity),
        weights: float('weights-values', weightsBuffer, neighborCapacity)
      },
      weightsOverflow: word('weights-overflow', weightsOverflowBuffer, 1),
      zScores: float('z-scores', zScoresBuffer),
      quadrants: word('quadrants', quadrantsBuffer),
      colors: word('cluster-colors', colorsBuffer)
    });
    const spatial = float('spatial', spatialBuffer);
    graph.add(
      new GPUSpatialEmpiricalBayesRates({
        id: 'spatial-eb',
        events,
        populations,
        weights: recipe.weights,
        smoothedRates: spatial
      })
    );
    for (const [name, values] of [
      ['raw', raw],
      ['smoothed', smoothed],
      ['spatial', spatial],
      ['standardized', standardized]
    ] as const) {
      graph.add(
        new GPUGlobalSpatialStatistics({
          id: `moran-${name}`,
          weights: recipe.weights,
          values,
          statistics: ['moran'],
          results: float(
            `moran-${name}`,
            globalBuffers[name],
            GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length
          )
        })
      );
    }
    return {key, compiled: resources.track(graph.compile())};
  };

  const variants = new Map<string, Variant>();
  const getVariant = (options: RateSmoothingOptions): Variant => {
    const key = compileKey(options);
    let variant = variants.get(key);
    if (!variant) {
      variant = compileVariant(options);
      variants.set(key, variant);
    }
    return variant;
  };
  function compileKey(options: RateSmoothingOptions): string {
    return [
      options.criterion,
      options.analyze,
      options.gating,
      options.gating === 'permutation' ? options.alternative : '-',
      options.gating === 'permutation' && options.falseDiscoveryRate ? 'fdr' : '-'
    ].join('|');
  }

  let active = getVariant(ctx.options);
  let dirty = true;
  let selectedRow = -1;

  type Latest = {
    raw: Float32Array;
    smoothed: Float32Array;
    spatial: Float32Array;
    standardized: Float32Array;
    quadrants: Uint32Array;
    pseudoP: Float32Array;
    zScores: Float32Array;
  };
  let latest: Latest | null = null;

  const writeParameters = () => {
    const {significance, seed, permutations} = ctx.options;
    autocorrelationParameters.write(
      getGPUSpatialAutocorrelationParameterValues({significanceLevel: significance})
    );
    permutationParameters.write(
      getGPUPermutationParameterValues({seed, permutations, significanceLevel: significance})
    );
    dirty = true;
    reader.markStale();
  };

  const describeRow = (row: number): string => {
    const lines = [
      nameOf(row),
      `${formatNumber(deaths[row])} deaths in 7 years among ${formatNumber(personYears[row] / 7)} residents`
    ];
    if (latest) {
      const rate = (value: number) => `${formatNumber(value * RATE_SCALE, 1)} per 100k per year`;
      lines.push(
        `Raw rate ${rate(latest.raw[row])}`,
        `Empirical Bayes ${rate(latest.smoothed[row])}`,
        `Spatial empirical Bayes ${rate(latest.spatial[row])}`,
        `Standardized z ${formatNumber(latest.standardized[row], 2)}`
      );
      const quadrant = [
        'not significant',
        'high-high cluster',
        'low-high outlier',
        'low-low cluster',
        'high-low outlier'
      ][latest.quadrants[row]];
      lines.push(`Local Moran: ${quadrant}`);
    }
    return lines.join('\n');
  };

  const reader = new SummaryReader(
    resources,
    'rate-smoothing',
    [
      {buffer: rawBuffer, size: n * 4},
      {buffer: smoothedBuffer, size: n * 4},
      {buffer: spatialBuffer, size: n * 4},
      {buffer: standardizedBuffer, size: n * 4},
      {buffer: quadrantsBuffer, size: n * 4},
      {buffer: pseudoPBuffer, size: n * 4},
      {buffer: zScoresBuffer, size: n * 4},
      {buffer: summaryBuffer, size: GPU_EMPIRICAL_BAYES_SUMMARY.length * 4},
      {buffer: globalBuffers.raw, size: GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4},
      {buffer: globalBuffers.smoothed, size: GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4},
      {buffer: globalBuffers.spatial, size: GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4},
      {buffer: globalBuffers.standardized, size: GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4},
      {buffer: offsetsBuffer, size: (n + 1) * 4},
      {buffer: weightsOverflowBuffer, size: 4}
    ],
    bytes => {
      const read = createByteReader(bytes);
      const raw = read.floats(n);
      const smoothed = read.floats(n);
      const spatial = read.floats(n);
      const standardized = read.floats(n);
      const quadrants = read.words(n);
      const pseudoP = read.floats(n);
      const zScores = read.floats(n);
      const summary = read.floats(GPU_EMPIRICAL_BAYES_SUMMARY.length);
      const moran = (buffer: Float32Array) =>
        buffer[
          GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.statistic
        ];
      const globals = ['raw', 'smoothed', 'spatial', 'standardized'].map(() =>
        read.floats(GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length)
      );
      const offsets = read.words(n + 1);
      const overflow = read.words(1)[0];
      latest = {
        raw: raw.slice(),
        smoothed: smoothed.slice(),
        spatial: spatial.slice(),
        standardized: standardized.slice(),
        quadrants: quadrants.slice(),
        pseudoP: pseudoP.slice(),
        zScores: zScores.slice()
      };
      // Shared color range of the three rate maps: the central 96% of the smoothed rates, so raw
      // outliers visibly saturate.
      const sortedSmoothed = getSortedFinite(smoothed);
      const lo = getQuantile(sortedSmoothed, 0.02);
      const hi = getQuantile(sortedSmoothed, 0.98);
      rateRange = [lo, hi];
      setLegendData<RateSmoothingLegend>('rate-smoothing', {
        rateRange: [lo * RATE_SCALE, hi * RATE_SCALE]
      });
      ctx.setLegendExtent('rate', [lo * RATE_SCALE, hi * RATE_SCALE]);

      ctx.setReadout('counties', n);
      ctx.setReadout(
        'pooled',
        `${formatNumber(summary[GPU_EMPIRICAL_BAYES_SUMMARY.pooledRate] * RATE_SCALE, 2)} per 100k per year`
      );
      ctx.setReadout(
        'shrinkageWeight',
        `prior variance a = ${(summary[GPU_EMPIRICAL_BAYES_SUMMARY.priorVariance] * RATE_SCALE * RATE_SCALE).toFixed(1)}`
      );
      ctx.setReadout(
        'moran',
        `${moran(globals[0]).toFixed(3)} / ${moran(globals[1]).toFixed(3)} / ${moran(globals[2]).toFixed(3)} / ${moran(globals[3]).toFixed(3)}`
      );
      // Spread of the small counties (under 5,000 residents) before and after smoothing.
      const smallRaw: number[] = [];
      const smallSmoothed: number[] = [];
      for (let row = 0; row < n; row++) {
        if (personYears[row] < PERSON_YEARS_SMALL && Number.isFinite(raw[row])) {
          smallRaw.push(raw[row]);
          smallSmoothed.push(smoothed[row]);
        }
      }
      const sortedRaw = getSortedFinite(smallRaw);
      const sortedSmall = getSortedFinite(smallSmoothed);
      ctx.setReadout(
        'smallCounties',
        `${smallRaw.length} counties: raw ${formatNumber(getQuantile(sortedRaw, 0.05) * RATE_SCALE)}-${formatNumber(getQuantile(sortedRaw, 0.95) * RATE_SCALE)}, smoothed ${formatNumber(getQuantile(sortedSmall, 0.05) * RATE_SCALE)}-${formatNumber(getQuantile(sortedSmall, 0.95) * RATE_SCALE)}`
      );
      const counts = [0, 0, 0, 0, 0];
      for (let row = 0; row < n; row++) counts[Math.min(quadrants[row], 4)]++;
      ctx.setReadout(
        'clusters',
        `${formatCount(counts[1])} HH / ${formatCount(counts[2])} LH / ${formatCount(counts[3])} LL / ${formatCount(counts[4])} HL`
      );
      ctx.setReadout('notSignificant', `${formatCount(counts[0])} of ${formatCount(n)}`);
      let islands = 0;
      for (let row = 0; row < n; row++) if (offsets[row + 1] === offsets[row]) islands++;
      ctx.setReadout(
        'neighbors',
        `${(offsets[n] / n).toFixed(2)} on average, ${islands} without neighbors${overflow ? ' (CAPACITY OVERFLOW)' : ''}`
      );
      ctx.requestLayers();
    }
  );
  let rateRange: [number, number] = [0, 3e-4];

  writeParameters();

  return {
    getCompiledGraphs: () => [...variants.values()].map(variant => variant.compiled),

    setOption(id) {
      if (['criterion', 'analyze', 'gating', 'alternative', 'falseDiscoveryRate'].includes(id)) {
        active = getVariant(ctx.options);
        dirty = true;
        reader.markStale();
      } else if (['permutations', 'seed', 'significance'].includes(id)) {
        writeParameters();
      }
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (dirty || frame.frameIndex < 2) {
        active.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.markStale();
      }
      if (frame.frameIndex >= 2) reader.flush(commandEncoder);
    },

    getLayers() {
      const {map, ramp, outlines} = ctx.options;
      const layers: Layer[] = [];
      if (map === 'cluster') {
        layers.push(
          geometry.createFillLayer('rate-clusters', {
            values: colorsBuffer,
            mode: 'packed',
            selectedRow,
            fillOpacity: 0.9
          })
        );
      } else {
        const buffer =
          map === 'raw'
            ? rawBuffer
            : map === 'smoothed'
              ? smoothedBuffer
              : map === 'spatial'
                ? spatialBuffer
                : standardizedBuffer;
        layers.push(
          geometry.createFillLayer(`rate-${map}`, {
            values: buffer,
            mode: 'ramp',
            ramp: map === 'standardized' ? 'diverging' : ramp,
            valueRange: map === 'standardized' ? [-3, 3] : rateRange,
            noDataColor: [128, 128, 128, 90],
            selectedRow,
            fillOpacity: 0.9
          })
        );
      }
      if (outlines) {
        layers.push(
          geometry.createOutlineLayer('rate-outline', getOutlineColor(ctx.theme(), 70), 0.7)
        );
      }
      return layers;
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const row = geometry.locator.locate(event.coordinate[0], event.coordinate[1]);
      return row >= 0 ? describeRow(row) : null;
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const row = geometry.locator.locate(event.coordinate[0], event.coordinate[1]);
      selectedRow = row === selectedRow ? -1 : row;
      ctx.setReadout(
        'selected',
        selectedRow >= 0 ? describeRow(selectedRow).replace(/\n/g, ' | ') : null
      );
      ctx.requestLayers();
      return true;
    },

    destroy() {
      reader.stop();
      clearLegendData('rate-smoothing');
      resources.destroy();
    }
  };
}
