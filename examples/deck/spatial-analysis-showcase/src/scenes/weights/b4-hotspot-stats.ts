// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {
  GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH,
  GPUHotSpotAnalysis,
  GPULocalMoran,
  GPULocalPermutationTest,
  type GPUParameterBuffer,
  type GPUPermutationAlternative
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  GPUCommandGraph,
  GPUHistogram,
  GPUReduction,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {SpatialAnalysisResources} from '../../engine/resources';

/** Class value of rows that are masked out; outside every histogram edge. */
export const MASKED_CLASS = 100;
/** Gi* bins -3..3 become histogram rows 0..6. */
export const GI_BIN_COUNT = 7;
/** Local Moran class codes 0 (not significant), 1 HH, 2 LH, 3 LL, 4 HL. */
export const MORAN_CLASS_COUNT = 5;
/** Largest permutation count; the permutation graphs reserve this many. */
export const MAXIMUM_PERMUTATIONS = 999;

/** Which local statistic and which inference a stat graph computes. */
export type StatVariant = {
  statistic: 'gi-star' | 'local-moran';
  /** `analytic` normal p-values, `none` (Moran only: every quadrant), or a conditional permutation test. */
  inference: 'analytic' | 'permutation' | 'none';
  /** Weight of the focal place in Gi*: 1 is Gi*, 0 is Gi. */
  selfWeight: 0 | 1;
  /** Benjamini-Hochberg false discovery rate control. */
  falseDiscoveryRate: boolean;
  alternative: GPUPermutationAlternative;
  maximumNeighbors: number;
  /** The Gi* z-scores, bins and permutation were already produced (by `addHotSpotAnalysisRecipe`). */
  precomputed?: boolean;
};

/** Per-row and summary buffers one world owns for the local statistics. */
export type StatBuffers = {
  zScores: Buffer;
  pValues: Buffer;
  /** Gi* bins (sint32) or ungated Moran quadrants (uint32), before the final gate. */
  base: Buffer;
  /** Final class per row: Gi* bin or quadrant, 0 when not significant, `MASKED_CLASS` when masked. */
  classes: Buffer;
  counts: Buffer;
  statistics: Buffer;
  neighborCounts: Buffer;
  exceedances: Buffer;
  pseudoPValues: Buffer;
  significant: Buffer;
  permutationOverflow: Buffer;
  significantCounts: Buffer;
  valuesExtent: Buffer;
};

/** Allocates the buffers of {@link StatBuffers}. */
export function createStatBuffers(
  resources: SpatialAnalysisResources,
  rows: number,
  id: string
): StatBuffers {
  const create = (name: string, size: number) => resources.createBuffer(`${id}-${name}`, size);
  return {
    zScores: create('z-scores', rows * 4),
    pValues: create('p-values', rows * 4),
    base: create('base', rows * 4),
    classes: create('classes', rows * 4),
    counts: create('counts', GI_BIN_COUNT * 4),
    statistics: create('statistics', GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH * 4),
    neighborCounts: create('neighbor-counts', rows * 4),
    exceedances: create('exceedances', rows * 4),
    pseudoPValues: create('pseudo-p-values', rows * 4),
    significant: create('significant', rows * 4),
    permutationOverflow: create('permutation-overflow', 4),
    significantCounts: create('significant-counts', 8),
    valuesExtent: create('values-extent', 8)
  };
}

/** Inputs of {@link compileStatGraph}. */
export type StatGraphProps = {
  device: Device;
  resources: SpatialAnalysisResources;
  id: string;
  rows: number;
  slots: number;
  weights: {offsets: Buffer; neighbors: Buffer; weights: Buffer};
  values: Buffer;
  /** Uint32 per row: zero rows are excluded. */
  mask: Buffer;
  buffers: StatBuffers;
  parameters: GPUParameterBuffer<'float32'>;
  permutationParameters: GPUParameterBuffer<'uint32'>;
  variant: StatVariant;
};

/**
 * Compiles one local-statistics graph: `GPUHotSpotAnalysis` (Gi*) or `GPULocalMoran`, optionally
 * `GPULocalPermutationTest`, a gate that keeps only significant classes, a `GPUHistogram` of the
 * classes and the extent of the values. With `precomputed`, the Gi* and the permutation test came
 * from `addHotSpotAnalysisRecipe` and only the gate, the histogram and the extent are added.
 */
export function compileStatGraph(props: StatGraphProps): CompiledGPUCommandGraph<void> {
  const {device, resources, rows, slots, buffers, variant} = props;
  const graph = new GPUCommandGraph<void>(device, {id: props.id});
  const weights = {
    offsets: importGraphBuffer(graph, 'offsets', props.weights.offsets, 'uint32', rows + 1),
    neighbors: importGraphBuffer(graph, 'neighbors', props.weights.neighbors, 'uint32', slots),
    weights: importGraphBuffer(graph, 'weights', props.weights.weights, 'float32', slots)
  };
  const values = importGraphBuffer(graph, 'values', props.values, 'float32', rows);
  const mask = importGraphBuffer(graph, 'mask', props.mask, 'uint32', rows);
  const parameters = props.parameters.importToGraph(graph);
  const permutationParameters = props.permutationParameters.importToGraph(graph);
  const zScores = importGraphBuffer(graph, 'z-scores', buffers.zScores, 'float32', rows);
  const pValues = importGraphBuffer(graph, 'p-values', buffers.pValues, 'float32', rows);
  const statistics = importGraphBuffer(
    graph,
    'statistics',
    buffers.statistics,
    'float32',
    GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH
  );
  const neighborCounts = importGraphBuffer(
    graph,
    'neighbor-counts',
    buffers.neighborCounts,
    'uint32',
    rows
  );
  const isGi = variant.statistic === 'gi-star';
  const usePermutation = variant.inference === 'permutation';
  const baseFormat = isGi ? ('sint32' as const) : ('uint32' as const);
  const base = importGraphBuffer(graph, 'base', buffers.base, baseFormat, rows);
  const significant = importGraphBuffer(graph, 'significant', buffers.significant, 'uint32', rows);
  // One view per buffer: the gate and the histogram reuse the views the statistics write.
  const classes = importGraphBuffer(graph, 'classes', buffers.classes, baseFormat, rows);

  if (!variant.precomputed) {
    if (isGi) {
      graph.add(
        new GPUHotSpotAnalysis({
          id: 'gi-star',
          weights,
          values,
          mask,
          parameters,
          zScores,
          bins: base as ReturnType<typeof importGraphBuffer<'sint32', void>>,
          pValues,
          neighborCounts,
          globalStatistics: statistics,
          selfWeight: variant.selfWeight,
          falseDiscoveryRate: variant.falseDiscoveryRate
        })
      );
    } else {
      graph.add(
        new GPULocalMoran({
          id: 'local-moran',
          weights,
          values,
          mask,
          parameters,
          zScores,
          pValues,
          quadrants: base as ReturnType<typeof importGraphBuffer<'uint32', void>>,
          neighborCounts,
          globalStatistics: statistics,
          quadrantGating: variant.inference === 'analytic' ? 'analytic' : 'none',
          falseDiscoveryRate: variant.inference === 'analytic' ? variant.falseDiscoveryRate : false
        })
      );
    }
    if (usePermutation) {
      graph.add(
        new GPULocalPermutationTest({
          id: 'local-permutation',
          weights,
          values,
          mask,
          statistic: isGi ? (variant.selfWeight === 1 ? 'localGStar' : 'localG') : 'localMoran',
          alternative: variant.alternative,
          parameters: permutationParameters,
          maximumPermutations: MAXIMUM_PERMUTATIONS,
          maximumNeighbors: variant.maximumNeighbors,
          falseDiscoveryRate: variant.falseDiscoveryRate,
          exceedances: importGraphBuffer(graph, 'exceedances', buffers.exceedances, 'uint32', rows),
          pseudoPValues: importGraphBuffer(
            graph,
            'pseudo-p-values',
            buffers.pseudoPValues,
            'float32',
            rows
          ),
          significant,
          overflow: importGraphBuffer(
            graph,
            'permutation-overflow',
            buffers.permutationOverflow,
            'uint32',
            1
          )
        })
      );
    }
  }
  if (usePermutation) {
    graph.add(
      new GPUHistogram({
        id: 'significant-counts',
        input: significant,
        output: importGraphBuffer(
          graph,
          'significant-counts',
          buffers.significantCounts,
          'uint32',
          2
        ),
        edges: [0, 1, 2]
      })
    );
  }

  // Gate: keep the class only where the permutation test confirmed it; mark masked rows.
  addKernelPass(graph, {
    id: `${props.id}-gate`,
    invocationCount: rows,
    bindings: [
      {name: 'base', view: base, type: 'u32', access: 'read'},
      {name: 'significant', view: significant, type: 'u32', access: 'read'},
      {name: 'mask', view: mask, type: 'u32', access: 'read'},
      {name: 'classes', view: classes, type: 'u32', access: 'read_write'}
    ],
    body: `
  if (mask[maskOffset + index] == 0u) {
    classes[classesOffset + index] = ${MASKED_CLASS}u;
    return;
  }
  var value = base[baseOffset + index];
  let confirmed = significant[significantOffset + index];
  if (${usePermutation ? 'true' : 'false'} && confirmed == 0u) { value = 0u; }
  classes[classesOffset + index] = value;`
  });
  graph.add(
    new GPUHistogram({
      id: 'class-counts',
      input: classes,
      output: importGraphBuffer(
        graph,
        'counts',
        buffers.counts,
        'uint32',
        isGi ? GI_BIN_COUNT : MORAN_CLASS_COUNT
      ),
      edges: isGi ? [-3, -2, -1, 0, 1, 2, 3, 4] : [0, 1, 2, 3, 4, 5]
    })
  );
  graph.add(
    new GPUReduction({
      id: 'values-extent',
      input: values,
      mask,
      output: importGraphBuffer(graph, 'values-extent', buffers.valuesExtent, 'float32', 2),
      operation: 'extent'
    })
  );
  return resources.track(graph.compile());
}
