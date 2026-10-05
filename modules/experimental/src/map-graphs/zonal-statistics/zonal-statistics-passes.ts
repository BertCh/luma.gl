// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientVectorView,
  createTransientView,
  GPUReduction,
  GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import {getGraphViewChunks} from '../map-graph-utils';

const OPERATION = 'GPUZonalStatistics';

/** Bit pattern of the quiet NaN written for undefined statistics. @internal */
export const ZONAL_STATISTICS_NAN_BITS = '0x7fc00000u';

/** Statistic whose `[min, max]` extent can be published. */
export type ZonalStatisticsExtentStatistic =
  | 'count'
  | 'sum'
  | 'mean'
  | 'minimum'
  | 'maximum'
  | 'density';

/** Which statistics must be computed for one set of requested outputs. @internal */
export type ZonalStatisticsPlan = {
  /** Per-feature assigned point counts are needed. */
  needCounts: boolean;
  /** Per-feature counts of value-valid rows are needed. */
  needValueCounts: boolean;
  /** Per-feature sums are needed. */
  needSums: boolean;
  /** Per-feature weight sums are needed. */
  needWeightSums: boolean;
  /** Per-feature means are needed. */
  needMeans: boolean;
  /** Per-feature minima are needed. */
  needMinima: boolean;
  /** Per-feature maxima are needed. */
  needMaxima: boolean;
  /** Per-feature densities are needed. */
  needDensities: boolean;
  /** The GPU must compute polygon areas (otherwise caller `areas` are used or areas are not needed). */
  needGPUAreas: boolean;
  /** Any statistic that depends on a per-point value is needed. */
  needValueStatistics: boolean;
};

/** Output presence flags consumed by {@link getZonalStatisticsPlan}. @internal */
export type ZonalStatisticsPlanInput = {
  /** Requested outputs, reduced to presence flags. */
  outputs: {
    counts: boolean;
    valueCounts: boolean;
    sums: boolean;
    weightSums: boolean;
    means: boolean;
    minima: boolean;
    maxima: boolean;
    densities: boolean;
    featureAreas: boolean;
    extent: boolean;
  };
  /** Statistic that `extent` summarizes. */
  extentStatistic?: ZonalStatisticsExtentStatistic;
  /** Whether per-point weights were given. */
  hasWeights: boolean;
  /** Whether caller-provided per-feature areas were given. */
  hasAreas: boolean;
  /** Whether sums use the sorted segmented reduction. */
  sorted: boolean;
};

/** Resolves which statistics are computed, including transients for derived statistics. @internal */
export function getZonalStatisticsPlan(input: ZonalStatisticsPlanInput): ZonalStatisticsPlan {
  const {outputs, extentStatistic, hasWeights, hasAreas, sorted} = input;
  const extent = outputs.extent ? extentStatistic : undefined;
  const needMeans = outputs.means || extent === 'mean';
  const needDensities = outputs.densities || extent === 'density';
  const needMinima = outputs.minima || extent === 'minimum';
  const needMaxima = outputs.maxima || extent === 'maximum';
  const needSums = outputs.sums || needMeans || extent === 'sum';
  const needValueCounts = outputs.valueCounts || (needMeans && !hasWeights);
  const needWeightSums = outputs.weightSums || (needMeans && hasWeights);
  const needCounts =
    outputs.counts ||
    needDensities ||
    outputs.extent ||
    (sorted && (needSums || needWeightSums)) ||
    extent === 'count';
  return {
    needCounts,
    needValueCounts,
    needSums,
    needWeightSums,
    needMeans,
    needMinima,
    needMaxima,
    needDensities,
    needGPUAreas: outputs.featureAreas || (needDensities && !hasAreas),
    needValueStatistics: needValueCounts || needSums || needWeightSums || needMinima || needMaxima
  };
}

/** Creates float32 scratch rows with the chunk topology of `template`. @internal */
export function createTransientFloat32Rows<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  template: GraphDataView<'float32'> | GraphVectorView<'float32'>
): GraphDataView<'float32'> | GraphVectorView<'float32'> {
  return template instanceof GraphVectorView
    ? createTransientVectorView(graph, id, template)
    : createTransientView(graph, id, 'float32', template.length);
}

const FINITE_HELPERS = /* wgsl */ `
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
fn isFiniteValue(value: f32) -> bool { return value == value && abs(value) <= FLOAT32_MAXIMUM; }`;

/** Rows produced by the per-point prepare kernels. @internal */
export type ZonalStatisticsPreparedRows = {
  /** 1 when the row has a finite value (and finite weight when weights are given). */
  valueMask: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** `w * v` or `v` for valid rows and 0 otherwise, chunked like the values. */
  sumContributions?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /** `w` for valid rows and 0 otherwise, chunked like the values. */
  weightContributions?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
};

/** Writes validity and contribution rows, one kernel per non-empty value chunk. @internal */
export function createZonalStatisticsPrepareNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    values: GraphDataView<'float32'> | GraphVectorView<'float32'>;
    weights?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
    valueMask: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
    sumContributions?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
    weightContributions?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const valueChunks = getGraphViewChunks(props.values);
  const weightChunks = props.weights ? getGraphViewChunks(props.weights) : [];
  const maskChunks = getGraphViewChunks(props.valueMask);
  const sumChunks = props.sumContributions ? getGraphViewChunks(props.sumContributions) : [];
  const weightContributionChunks = props.weightContributions
    ? getGraphViewChunks(props.weightContributions)
    : [];
  const isVector = props.values instanceof GraphVectorView;
  for (const [chunkIndex, valueChunk] of valueChunks.entries()) {
    if (valueChunk.length === 0) {
      continue;
    }
    const weightChunk = weightChunks[chunkIndex];
    const sumChunk = sumChunks[chunkIndex];
    const weightContributionChunk = weightContributionChunks[chunkIndex];
    const bindings: MapGraphKernelBinding[] = [
      {name: 'values', view: valueChunk, type: 'f32', access: 'read'}
    ];
    if (weightChunk) {
      bindings.push({name: 'weights', view: weightChunk, type: 'f32', access: 'read'});
    }
    bindings.push({
      name: 'valueMask',
      view: maskChunks[chunkIndex],
      type: 'u32',
      access: 'read_write'
    });
    if (sumChunk) {
      bindings.push({name: 'sumContributions', view: sumChunk, type: 'f32', access: 'read_write'});
    }
    if (weightContributionChunk) {
      bindings.push({
        name: 'weightContributions',
        view: weightContributionChunk,
        type: 'f32',
        access: 'read_write'
      });
    }
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: isVector ? `${props.id}-prepare-chunk-${chunkIndex}` : `${props.id}-prepare`,
        operation: OPERATION,
        variant: 'prepare',
        bindings,
        invocationCount: valueChunk.length,
        declarations: FINITE_HELPERS,
        body: `let value = values[valuesOffset + index];
  ${weightChunk ? 'let weight = weights[weightsOffset + index];' : ''}
  let valid = isFiniteValue(value)${weightChunk ? ' && isFiniteValue(weight)' : ''};
  valueMask[valueMaskOffset + index] = select(0u, 1u, valid);
  ${
    sumChunk
      ? `let product = ${weightChunk ? 'weight * value' : 'value'};
  sumContributions[sumContributionsOffset + index] = select(0.0, product, valid && isFiniteValue(product));`
      : ''
  }
  ${
    weightContributionChunk
      ? 'weightContributions[weightContributionsOffset + index] = select(0.0, weight, valid);'
      : ''
  }`
      })
    );
  }
  return nodes;
}

/**
 * Computes one polygon-feature area per invocation, `|shell| - sum |holes|` summed over polygons.
 *
 * Each ring's shoelace sum is taken relative to its first finite vertex to reduce float32
 * cancellation. Non-finite vertices are skipped, joining their neighbors. Rings that do not fit
 * the offset arrays contribute 0, and so do features whose polygon range is invalid.
 *
 * @internal
 */
export function createZonalStatisticsAreaNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    featureCount: number;
    polygonPositions: GraphDataView<'float32x2'>;
    featureOffsets: GraphDataView<'uint32'>;
    polygonOffsets: GraphDataView<'uint32'>;
    ringOffsets: GraphDataView<'uint32'>;
    areas: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'area',
    bindings: [
      {name: 'polygonPositions', view: props.polygonPositions, type: 'f32', access: 'read'},
      {name: 'featureOffsets', view: props.featureOffsets, type: 'u32', access: 'read'},
      {name: 'polygonOffsets', view: props.polygonOffsets, type: 'u32', access: 'read'},
      {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
      {name: 'areas', view: props.areas, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.featureCount,
    declarations: `${FINITE_HELPERS}
const POLYGON_COUNT: u32 = ${props.polygonOffsets.length - 1}u;
const RING_COUNT: u32 = ${props.ringOffsets.length - 1}u;
const VERTEX_COUNT: u32 = ${props.polygonPositions.length}u;

fn getRingArea(vertexStart: u32, vertexEnd: u32) -> f32 {
  var origin = vec2f(0.0);
  var hasOrigin = false;
  var previous = vec2f(0.0);
  var twiceArea = 0.0;
  for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
    let position = vec2f(
      polygonPositions[polygonPositionsOffset + vertex * 2u],
      polygonPositions[polygonPositionsOffset + vertex * 2u + 1u]
    );
    if (!isFiniteValue(position.x) || !isFiniteValue(position.y)) {
      continue;
    }
    if (!hasOrigin) {
      origin = position;
      hasOrigin = true;
      previous = vec2f(0.0);
      continue;
    }
    let current = position - origin;
    // The closing edge ends at the origin, so its cross product is zero.
    twiceArea += previous.x * current.y - previous.y * current.x;
    previous = current;
  }
  return abs(twiceArea) * 0.5;
}`,
    body: `var area = 0.0;
  let polygonStart = featureOffsets[featureOffsetsOffset + index];
  let polygonEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  if (polygonStart <= polygonEnd && polygonEnd <= POLYGON_COUNT) {
    for (var polygon = polygonStart; polygon < polygonEnd; polygon++) {
      let ringStart = polygonOffsets[polygonOffsetsOffset + polygon];
      let ringEnd = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
      if (ringStart > ringEnd || ringEnd > RING_COUNT) {
        continue;
      }
      for (var ring = ringStart; ring < ringEnd; ring++) {
        let vertexStart = ringOffsets[ringOffsetsOffset + ring];
        let vertexEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
        if (vertexStart > vertexEnd || vertexEnd > VERTEX_COUNT) {
          continue;
        }
        let ringArea = getRingArea(vertexStart, vertexEnd);
        area += select(-ringArea, ringArea, ring == ringStart);
      }
    }
  }
  areas[areasOffset + index] = area;`
  });
}

/** Writes means: `sum / valueCount`, or `sum / weightSum` with weights; NaN when undefined. @internal */
export function createZonalStatisticsMeanNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    featureCount: number;
    sums: GraphDataView<'float32'>;
    valueCounts?: GraphDataView<'uint32'>;
    weightSums?: GraphDataView<'float32'>;
    means: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const weighted = Boolean(props.weightSums);
  const bindings: MapGraphKernelBinding[] = [
    {name: 'sums', view: props.sums, type: 'f32', access: 'read'}
  ];
  if (props.weightSums) {
    bindings.push({name: 'weightSums', view: props.weightSums, type: 'f32', access: 'read'});
  } else {
    bindings.push({name: 'valueCounts', view: props.valueCounts!, type: 'u32', access: 'read'});
  }
  bindings.push({name: 'means', view: props.means, type: 'u32', access: 'read_write'});
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'means',
    bindings,
    invocationCount: props.featureCount,
    body: weighted
      ? `let weightSum = weightSums[weightSumsOffset + index];
  var mean = ${ZONAL_STATISTICS_NAN_BITS};
  if (weightSum != 0.0) {
    mean = bitcast<u32>(sums[sumsOffset + index] / weightSum);
  }
  means[meansOffset + index] = mean;`
      : `let valueCount = valueCounts[valueCountsOffset + index];
  var mean = ${ZONAL_STATISTICS_NAN_BITS};
  if (valueCount != 0u) {
    mean = bitcast<u32>(sums[sumsOffset + index] / f32(valueCount));
  }
  means[meansOffset + index] = mean;`
  });
}

/** Writes `count / area` where the area is finite and positive, NaN otherwise. @internal */
export function createZonalStatisticsDensityNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    featureCount: number;
    counts: GraphDataView<'uint32'>;
    areas: GraphDataView<'float32'>;
    densities: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'densities',
    bindings: [
      {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
      {name: 'areas', view: props.areas, type: 'f32', access: 'read'},
      {name: 'densities', view: props.densities, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.featureCount,
    declarations: FINITE_HELPERS,
    body: `let area = areas[areasOffset + index];
  var density = ${ZONAL_STATISTICS_NAN_BITS};
  if (isFiniteValue(area) && area > 0.0) {
    density = bitcast<u32>(f32(counts[countsOffset + index]) / area);
  }
  densities[densitiesOffset + index] = density;`
  });
}

/**
 * Publishes the `[min, max]` extent of one per-feature statistic over qualifying features.
 *
 * A kernel writes the statistic as float32 plus a 0/1 qualification mask (count > 0 and finite
 * statistic); `GPUReduction` `'extent'` reduces the masked values and writes `[0, 0]` when no
 * feature qualifies.
 *
 * @internal
 */
export function getZonalStatisticsExtentNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    featureCount: number;
    statistic: ZonalStatisticsExtentStatistic;
    counts: GraphDataView<'uint32'>;
    /** The float32 per-feature statistic view; unused for `'count'`. */
    statisticView?: GraphDataView<'float32'>;
    extent: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const values = createTransientView(
    graph,
    `${props.id}-extent-values`,
    'float32',
    props.featureCount
  );
  const mask = createTransientView(graph, `${props.id}-extent-mask`, 'uint32', props.featureCount);
  const bindings: MapGraphKernelBinding[] = [
    {name: 'counts', view: props.counts, type: 'u32', access: 'read'}
  ];
  if (props.statisticView) {
    bindings.push({name: 'statistic', view: props.statisticView, type: 'f32', access: 'read'});
  }
  bindings.push(
    {name: 'extentValues', view: values, type: 'f32', access: 'read_write'},
    {name: 'extentMask', view: mask, type: 'u32', access: 'read_write'}
  );
  const nodes: GPUCommandNode<Parameters>[] = [
    createMapGraphKernelNode<Parameters>(graph, {
      id: `${props.id}-extent-prepare`,
      operation: OPERATION,
      variant: 'extent-prepare',
      bindings,
      invocationCount: props.featureCount,
      declarations: FINITE_HELPERS,
      body: `let count = counts[countsOffset + index];
  let value = ${props.statisticView ? 'statistic[statisticOffset + index]' : 'f32(count)'};
  let qualifies = count > 0u && isFiniteValue(value);
  extentValues[extentValuesOffset + index] = select(0.0, value, qualifies);
  extentMask[extentMaskOffset + index] = select(0u, 1u, qualifies);`
    })
  ];
  nodes.push(
    ...new GPUReduction({
      id: `${props.id}-extent`,
      input: values,
      mask,
      output: props.extent,
      operation: 'extent'
    }).getCommandNodes(graph)
  );
  return nodes;
}
