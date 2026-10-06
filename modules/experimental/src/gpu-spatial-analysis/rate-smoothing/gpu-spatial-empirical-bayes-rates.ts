// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {SPATIAL_AUTOCORRELATION_FLOAT_WGSL} from '../spatial-autocorrelation/spatial-autocorrelation-kernels';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';

const OPERATION = 'GPUSpatialEmpiricalBayesRates';

/**
 * Properties for {@link GPUSpatialEmpiricalBayesRates}.
 *
 * Per-frame (no rebuild or recompile): the contents of `events`, `populations`, `mask` and
 * `weights`. Compile-time: the row count and which outputs exist.
 */
export type GPUSpatialEmpiricalBayesRatesProps = {
  /** Prefix for generated node IDs. Defaults to `'spatial-empirical-bayes-rates'`. */
  id?: string;
  /** Event counts `e`, one per row. */
  events: GraphDataView<'float32'>;
  /** Populations at risk `b`, one per row. Rows with `b <= 0` or a non-finite value are excluded. */
  populations: GraphDataView<'float32'>;
  /**
   * Square self-join spatial weights. Only the neighbor structure is used (esda's binary
   * transform): every listed neighbor counts once whatever its weight value.
   */
  weights: GPUSpatialWeights;
  /** Optional row selection: nonzero includes the row, both as a focus row and as a neighbor. */
  mask?: GraphDataView<'uint32'>;
  /** Optional caller-owned spatial rate per row (esda `Spatial_Rate.r`), NaN for excluded rows. */
  spatialRates?: GraphDataView<'float32'>;
  /** Optional caller-owned spatial empirical-Bayes rate per row (esda `Spatial_Empirical_Bayes.r`), NaN if excluded. */
  smoothedRates?: GraphDataView<'float32'>;
  /** Optional caller-owned local prior mean per row (the neighborhood pooled rate), NaN if excluded. */
  priorMeans?: GraphDataView<'float32'>;
  /** Optional caller-owned local prior variance per row, after clamping at zero, NaN if excluded. */
  priorVariances?: GraphDataView<'float32'>;
};

/**
 * Spatial (neighborhood-pooled) rate smoothing, matching PySAL esda 2.10 `smoothing.Spatial_Rate`
 * and `smoothing.Spatial_Empirical_Bayes`.
 *
 * For each included focus row `i`, the neighborhood is `i` itself plus its listed neighbors that
 * are included (mask nonzero, finite `e`, finite `b > 0`). Weights are used as binary structure
 * (esda sets `w.transform = 'b'`). With `E = sum e`, `B = sum b` and `n` members:
 * - Spatial rate: `r_i = E / B`, exactly `Spatial_Rate.r` (`(e_i + lag e) / (b_i + lag b)`).
 * - Local prior mean `m = E / B`, local variance `s2 = sum b_j (e_j / b_j - m)^2 / B`,
 *   `a = s2 - m / (B / n)`, clamped at zero.
 * - Smoothed rate: `w = a / (a + m / b_i)`, `r_i = w e_i / b_i + (1 - w) m`.
 *
 * Verified against esda 2.10.0 source and values (lat2W 6x6 queen fixture, max relative error
 * under 2e-4 in float32). Deviations from esda: rows excluded by the mask or by a non-positive
 * population are skipped as neighbors instead of raising; a neighborhood with all-zero events gives
 * `0 / 0 = NaN` in esda and gives `r_i = 0` here (`w = 0`); a neighbor ID equal to the focus row is
 * skipped, whereas esda would count it twice. Isolates smooth to their own raw rate.
 *
 * Each row is reduced by one invocation over its ascending neighbor slots, so results are bitwise
 * reproducible. Neighbor IDs equal to the focus row are skipped (self is always included once).
 */
export class GPUSpatialEmpiricalBayesRates implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialEmpiricalBayesRatesProps;

  constructor(props: GPUSpatialEmpiricalBayesRatesProps) {
    const id = props.id ?? 'spatial-empirical-bayes-rates';
    this.id = id;
    this.props = props;
    validatePackedView(props.events, ['float32'], `${id} events`);
    validatePackedView(props.populations, ['float32'], `${id} populations`);
    const rows = props.events.length;
    if (rows < 1) {
      throw new Error(`${id} events must hold at least one row`);
    }
    if (props.populations.length !== rows) {
      throw new Error(`${id} populations length must equal the events length`);
    }
    if (validateGPUSpatialWeights(id, props.weights) !== rows) {
      throw new Error(`${id} weights row count must equal the events length`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal the events length`);
      }
    }
    const outputs = this._getOutputs();
    for (const [name, view] of outputs) {
      validatePackedView(view, ['float32'], `${id} ${name}`);
      if (view.length !== rows) {
        throw new Error(`${id} ${name} length must equal the events length`);
      }
    }
    if (outputs.length === 0) {
      throw new Error(`${id} requests no output`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      outputs.map(([, view]) => view),
      [
        props.events,
        props.populations,
        props.mask,
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights
      ]
    );
  }

  /** Returns the smoothing node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {events, populations, weights, mask} = props;
    const outputs = this._getOutputs();
    validateGraphViewsBelongToGraph(id, graph, [
      events,
      populations,
      mask,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      ...outputs.map(([, view]) => view)
    ]);
    const rows = events.length;
    const maskBinding: WGSLKernelBinding[] = mask
      ? [{name: 'mask', view: mask, type: 'u32', access: 'read'}]
      : [];
    const includedWGSL = (row: string) =>
      `(${mask ? `mask[maskOffset + ${row}] != 0u && ` : ''}isFiniteFloat(events[eventsOffset + ${row}]) && isFiniteFloat(populations[populationsOffset + ${row}]) && populations[populationsOffset + ${row}] > 0.0)`;
    const outputBindings: WGSLKernelBinding[] = outputs.map(([name, view]) => ({
      name,
      view,
      type: 'f32',
      access: 'read_write'
    }));
    const expressions: Record<string, string> = {
      spatialRates: 'spatialRate',
      smoothedRates: 'smoothed',
      priorMeans: 'priorMean',
      priorVariances: 'priorVariance'
    };
    const writeOutputs = (valueFor: (name: string) => string) =>
      outputs.map(([name]) => `${name}[${name}Offset + index] = ${valueFor(name)};`).join('\n  ');
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-smooth`,
        operation: OPERATION,
        variant: 'smooth',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'events', view: events, type: 'f32', access: 'read'},
          {name: 'populations', view: populations, type: 'f32', access: 'read'},
          ...maskBinding,
          ...outputBindings
        ],
        invocationCount: rows,
        declarations: `const ROWS: u32 = ${rows}u;
${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}`,
        body: `let nan = getQuietNaN(index);
  if (!${includedWGSL('index')}) {
    ${writeOutputs(() => 'nan')}
    return;
  }
  let start = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  var eventSum = events[eventsOffset + index];
  var populationSum = populations[populationsOffset + index];
  var count = 1.0;
  for (var slot = start; slot < end; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < ROWS && neighbor != index && ${includedWGSL('neighbor')}) {
      eventSum += events[eventsOffset + neighbor];
      populationSum += populations[populationsOffset + neighbor];
      count += 1.0;
    }
  }
  let priorMean = eventSum / populationSum;
  let ownDeviation = events[eventsOffset + index] / populations[populationsOffset + index] - priorMean;
  var deviationSum = populations[populationsOffset + index] * ownDeviation * ownDeviation;
  for (var slot = start; slot < end; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < ROWS && neighbor != index && ${includedWGSL('neighbor')}) {
      let b = populations[populationsOffset + neighbor];
      let deviation = events[eventsOffset + neighbor] / b - priorMean;
      deviationSum += b * deviation * deviation;
    }
  }
  let spatialRate = priorMean;
  let priorVariance = max(deviationSum / populationSum - priorMean / (populationSum / count), 0.0);
  let ownPopulation = populations[populationsOffset + index];
  let ownRate = events[eventsOffset + index] / ownPopulation;
  let denominator = priorVariance + priorMean / ownPopulation;
  let weight = select(0.0, priorVariance / denominator, denominator > 0.0);
  let smoothed = weight * ownRate + (1.0 - weight) * priorMean;
  ${writeOutputs(name => expressions[name])}`
      })
    ];
  }

  private _getOutputs(): [string, GraphDataView<'float32'>][] {
    const {props} = this;
    const outputs: [string, GraphDataView<'float32'> | undefined][] = [
      ['spatialRates', props.spatialRates],
      ['smoothedRates', props.smoothedRates],
      ['priorMeans', props.priorMeans],
      ['priorVariances', props.priorVariances]
    ];
    return outputs.filter(
      (entry): entry is [string, GraphDataView<'float32'>] => entry[1] !== undefined
    );
  }
}
