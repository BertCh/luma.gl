// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {assertCompatibleGPUSpatialContexts} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPUHuffTradeAreas';

/** Trade-area value of a demand location that reaches no attractive facility. */
export const GPU_HUFF_NO_TRADE_AREA = 0xffffffff;

/** Number of `float32` words of the optional per-frame parameter view of {@link GPUHuffTradeAreas}. */
export const GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH = 1;

/** Per-frame settings of {@link GPUHuffTradeAreas}. */
export type GPUHuffTradeAreasSettings = {
  /** Attractiveness exponent `alpha`, positive and finite. Defaults to 1. */
  exponent?: number;
};

/**
 * Packs {@link GPUHuffTradeAreasSettings} into the `GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH` `float32`
 * words `[exponent]` read by {@link GPUHuffTradeAreas}.
 *
 * @throws If `exponent` is not a positive finite number.
 */
export function getGPUHuffTradeAreasParameterValues(
  settings: GPUHuffTradeAreasSettings = {}
): Float32Array {
  const exponent = settings.exponent ?? 1;
  if (!Number.isFinite(exponent) || !(exponent > 0)) {
    throw new Error('GPUHuffTradeAreas exponent must be positive and finite');
  }
  return Float32Array.of(exponent);
}

/** Caller-owned outputs of {@link GPUHuffTradeAreas}. */
export type GPUHuffTradeAreasOutput = {
  /**
   * Trade area per demand location: the facility with the highest probability (the lowest ID wins
   * ties), or {@link GPU_HUFF_NO_TRADE_AREA} when none is reachable.
   */
  tradeArea: GraphDataView<'uint32'>;
  /**
   * Optional Huff probability per demand-weights slot, aligned with `demandWeights.neighbors`:
   * `p_ij = A_j^alpha w_ij / sum_k A_k^alpha w_ik` (0 for slots past the last row, and for rows
   * whose denominator is 0). Rows sum to 1.
   */
  probabilities?: GraphDataView<'float32'>;
  /** Optional probability `p_ij` of the trade-area facility per demand location. */
  tradeAreaProbability?: GraphDataView<'float32'>;
  /**
   * Optional expected demand `E_j = sum_i P_i p_ij` captured per facility. Needs `facilityWeights`
   * and `demand`.
   */
  expectedDemand?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUHuffTradeAreas}.
 *
 * Compile-time: view lengths, and which optional views and outputs exist. Per-frame: every view's
 * contents, including the attractiveness exponent in `parameters`.
 */
export type GPUHuffTradeAreasProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'huff-trade-areas'`. */
  id?: string;
  /** Attractiveness `A_j` (floor area, capacity) per facility, non-negative. */
  attractiveness: GraphDataView<'float32'>;
  /**
   * Demand-to-facility weights: row `i` lists facilities `j` demand location `i` can reach, with
   * the distance-decay value `w_ij = f(c_ij)` as weight. Neighbor IDs index `attractiveness`.
   */
  demandWeights: GPUSpatialWeights;
  /**
   * Facility-to-demand weights for `expectedDemand`: row `j` lists the demand locations of facility
   * `j` with `w_ji`. Required only when `expectedDemand` is requested; it must be the transpose of
   * `demandWeights`, and when `demand` and `attractiveness` have equal length with symmetric
   * weights `demandWeights` itself may be passed.
   */
  facilityWeights?: GPUSpatialWeights;
  /** Demand `P_i` per demand location. Required with `expectedDemand`. */
  demand?: GraphDataView<'float32'>;
  /**
   * Optional per-frame parameters, `GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH` `float32` words written
   * with {@link getGPUHuffTradeAreasParameterValues}. Without it `alpha = 1`.
   */
  parameters?: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPUHuffTradeAreasOutput;
};

/**
 * Huff gravity-model trade areas on any {@link GPUSpatialWeights}: the probability that demand
 * location `i` patronises facility `j`, `p_ij = A_j^alpha w_ij / sum_k A_k^alpha w_ik`, the modal
 * facility per demand location (its trade area), and the expected demand captured by each
 * facility.
 *
 * The weights carry the decay and the catchment cut-off, so a Euclidean kernel from
 * `GPUNeighborSearch` gives the classic distance-decay Huff model. Rows accumulate in slot order,
 * so results are deterministic and match a sequential CPU evaluation within f32 rounding.
 * Facilities with zero attractiveness have probability 0.
 *
 * When both weights declare spatial contexts, they must be compatible.
 *
 * Spatial context: this operator reads no positions, only the weights, so it has no
 * `spatialContext` prop; distance units and the coordinate space are those of the weights' producer.
 */
export class GPUHuffTradeAreas implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUHuffTradeAreasProps;
  /** Number of facilities. */
  readonly facilityCount: number;
  /** Number of demand locations. */
  readonly demandCount: number;

  constructor(props: GPUHuffTradeAreasProps) {
    const id = props.id ?? 'huff-trade-areas';
    this.id = id;
    this.props = props;
    this.demandCount = validateGPUSpatialWeights(id, props.demandWeights, 'demandWeights');
    this.facilityCount = props.attractiveness.length;
    if (props.demandWeights?.spatialContext && props.facilityWeights?.spatialContext) {
      assertCompatibleGPUSpatialContexts(
        id,
        props.demandWeights.spatialContext,
        props.facilityWeights.spatialContext
      );
    }
    const {output} = props;
    validatePackedView(props.attractiveness, ['float32'], `${id} attractiveness`);
    validatePackedView(output.tradeArea, ['uint32'], `${id} output.tradeArea`);
    for (const [name, view] of [
      ['demand', props.demand],
      ['parameters', props.parameters],
      ['output.probabilities', output.probabilities],
      ['output.tradeAreaProbability', output.tradeAreaProbability],
      ['output.expectedDemand', output.expectedDemand]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    if (output.tradeArea.length !== this.demandCount) {
      throw new Error(`${id} output.tradeArea length must equal the demand count`);
    }
    if (output.tradeAreaProbability && output.tradeAreaProbability.length !== this.demandCount) {
      throw new Error(`${id} output.tradeAreaProbability length must equal the demand count`);
    }
    if (
      output.probabilities &&
      output.probabilities.length !== props.demandWeights.neighbors.length
    ) {
      throw new Error(
        `${id} output.probabilities length must equal demandWeights.neighbors length`
      );
    }
    if (props.parameters && props.parameters.length < GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters must hold ${GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH} word`);
    }
    if (output.expectedDemand) {
      if (!props.facilityWeights || !props.demand) {
        throw new Error(`${id} output.expectedDemand needs facilityWeights and demand`);
      }
      if (output.expectedDemand.length !== this.facilityCount) {
        throw new Error(`${id} output.expectedDemand length must equal the facility count`);
      }
      if (props.demand.length !== this.demandCount) {
        throw new Error(`${id} demand length must equal the demandWeights row count`);
      }
      const facilityRows = validateGPUSpatialWeights(id, props.facilityWeights, 'facilityWeights');
      if (facilityRows !== this.facilityCount) {
        throw new Error(`${id} facilityWeights row count must equal the facility count`);
      }
    }
    const outputs = getOutputs(output);
    validateGraphOutputsDisjointFromInputs(id, outputs, getInputs(props));
    const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /** Returns `denominators`, `trade-area`, `probabilities` and `expected-demand` nodes as requested. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, facilityCount, demandCount} = this;
    const {attractiveness, demandWeights, parameters, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(output)]);
    const denominators = createTransientView(graph, `${id}-denominators`, 'float32', demandCount);
    const parameterBinding = parameters
      ? [{name: 'parameters', view: parameters, type: 'f32' as const, access: 'read' as const}]
      : [];
    const declarations = `const FACILITIES: u32 = ${facilityCount}u;
const DEMANDS: u32 = ${demandCount}u;
fn getAttraction(value: f32) -> f32 {
  let exponent = ${parameters ? 'parameters[parametersOffset]' : '1.0'};
  return select(0.0, pow(value, exponent), value > 0.0);
}`;
    const rowLoop = (
      inner: string
    ) => `for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < FACILITIES) {
      let share = getAttraction(attractiveness[attractivenessOffset + partner]) * weights[weightsOffset + slot];
      ${inner}
    }
  }`;
    const demandBindings = [
      {name: 'offsets', view: demandWeights.offsets, type: 'u32' as const, access: 'read' as const},
      {
        name: 'neighbors',
        view: demandWeights.neighbors,
        type: 'u32' as const,
        access: 'read' as const
      },
      {name: 'weights', view: demandWeights.weights, type: 'f32' as const, access: 'read' as const},
      {name: 'attractiveness', view: attractiveness, type: 'f32' as const, access: 'read' as const},
      ...parameterBinding
    ];
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-trade-area`,
        operation: OPERATION,
        variant: 'trade-area',
        record: {formula: 'gpu-spatial-analysis.huff-trade-areas'},
        bindings: [
          ...demandBindings,
          {name: 'denominators', view: denominators, type: 'f32', access: 'read_write'},
          {name: 'tradeArea', view: output.tradeArea, type: 'u32', access: 'read_write'},
          ...(output.tradeAreaProbability
            ? [
                {
                  name: 'tradeProbability',
                  view: output.tradeAreaProbability,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: demandCount,
        declarations,
        body: `var total = 0.0;
  var best = 0.0;
  var bestFacility = ${GPU_HUFF_NO_TRADE_AREA}u;
  ${rowLoop(`total += share;
      if (share > best) {
        best = share;
        bestFacility = partner;
      }`)}
  denominators[denominatorsOffset + index] = total;
  tradeArea[tradeAreaOffset + index] = select(${GPU_HUFF_NO_TRADE_AREA}u, bestFacility, total > 0.0);
  ${output.tradeAreaProbability ? 'tradeProbability[tradeProbabilityOffset + index] = select(0.0, best / total, total > 0.0);' : ''}`
      })
    ];
    if (output.probabilities) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-probabilities`,
          operation: OPERATION,
          variant: 'probabilities',
          bindings: [
            ...demandBindings,
            {name: 'denominators', view: denominators, type: 'f32', access: 'read'},
            {name: 'probabilities', view: output.probabilities, type: 'f32', access: 'read_write'}
          ],
          invocationCount: demandCount,
          declarations,
          body: `let total = denominators[denominatorsOffset + index];
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    var probability = 0.0;
    if (partner < FACILITIES && total > 0.0) {
      probability = getAttraction(attractiveness[attractivenessOffset + partner]) * weights[weightsOffset + slot] / total;
    }
    probabilities[probabilitiesOffset + slot] = probability;
  }`
        })
      );
    }
    if (output.expectedDemand && props.facilityWeights && props.demand) {
      const facilityWeights = props.facilityWeights;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-expected-demand`,
          operation: OPERATION,
          variant: 'expected-demand',
          bindings: [
            {name: 'offsets', view: facilityWeights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: facilityWeights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: facilityWeights.weights, type: 'f32', access: 'read'},
            {name: 'attractiveness', view: attractiveness, type: 'f32', access: 'read'},
            {name: 'demand', view: props.demand, type: 'f32', access: 'read'},
            {name: 'denominators', view: denominators, type: 'f32', access: 'read'},
            ...parameterBinding,
            {name: 'expectedDemand', view: output.expectedDemand, type: 'f32', access: 'read_write'}
          ],
          invocationCount: facilityCount,
          declarations,
          body: `var captured = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < DEMANDS) {
      let total = denominators[denominatorsOffset + partner];
      if (total > 0.0) {
        captured += demand[demandOffset + partner] * weights[weightsOffset + slot] / total;
      }
    }
  }
  expectedDemand[expectedDemandOffset + index] = getAttraction(attractiveness[attractivenessOffset + index]) * captured;`
        })
      );
    }
    return nodes;
  }
}

/** Returns every caller-owned output view of a Huff contributor. */
function getOutputs(output: GPUHuffTradeAreasOutput): (GraphDataView | undefined)[] {
  return [
    output.probabilities,
    output.tradeArea,
    output.tradeAreaProbability,
    output.expectedDemand
  ];
}

/** Returns every read-only view of a Huff contributor. */
function getInputs(props: GPUHuffTradeAreasProps): (GraphDataView | undefined)[] {
  return [
    props.attractiveness,
    props.demand,
    props.parameters,
    props.demandWeights.offsets,
    props.demandWeights.neighbors,
    props.demandWeights.weights,
    props.facilityWeights?.offsets,
    props.facilityWeights?.neighbors,
    props.facilityWeights?.weights
  ];
}
