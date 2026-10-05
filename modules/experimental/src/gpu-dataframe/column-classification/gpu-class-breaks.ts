// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  CLASS_BREAKS_BIN_WGSL,
  CLASS_BREAKS_EXTREMES,
  CLASS_BREAKS_HEAD_TAIL,
  CLASS_BREAKS_STATE,
  getClassBreaksMomentNodes,
  getClassBreaksMomentTileCount,
  MOMENT_STRIDE
} from './class-breaks-kernels';
import {
  getGPUClassBreaksParameterLength,
  GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT,
  GPU_CLASS_BREAKS_METHOD_CODES,
  GPU_CLASS_BREAKS_METHODS,
  GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH,
  type GPUClassBreaksMethod
} from './class-breaks-parameters';
import {COLUMN_ORDERED_KEY_WGSL} from './column-classification-shared';
import {getColumnQuantileNodes} from './column-quantiles-nodes';

const OPERATION = 'GPUClassBreaks';
/** Largest supported class count; the quantile pass refines at most 64 targets. */
const MAXIMUM_CLASS_COUNT = 64;
/** Largest row count: moment counts are f32 and quantile ranks are exact up to 2^24. */
const MAXIMUM_ROW_COUNT = 2 ** 24;
const DEFAULT_NATURAL_BREAKS_BIN_COUNT = 1024;
const MAXIMUM_NATURAL_BREAKS_BIN_COUNT = 2048;
/** Quantile parameter code for linear (R-7) interpolation, as numpy and mapclassify use. */
const LINEAR_INTERPOLATION_CODE = 3;
const INVALID_METHOD = 0xffffffff;
const LARGE_COST = '3.0e38';
const CODES = GPU_CLASS_BREAKS_METHOD_CODES;

/** Caller-owned outputs of {@link GPUClassBreaks}. */
export type GPUClassBreaksOutput = {
  /**
   * Class edges `e[0..k]` (`maximumClassCount + 1` rows): `e[0]` is the smallest and `e[k]` the
   * largest finite value, inner edges ascend. Rows past `k` are NaN. Rows `e[1..k-1]` feed
   * `GPUColorScale` or any threshold scale directly as a domain.
   */
  breaks: GraphDataView<'float32'>;
  /** One row receiving the produced class count `k` (0 when no finite value is selected). */
  classCount: GraphDataView<'uint32'>;
  /**
   * Optional rows per class (`maximumClassCount` rows), counted with the stream's class rule: a
   * value is in class `i` when `i` inner edges have an ordered key at or below its key; values
   * outside `[e[0], e[k]]` (such as infinities) clamp to the first or last class.
   */
  classCounts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUClassBreaks}.
 *
 * Per-frame (no recompile): the contents of `parameters` (method, class count, method settings,
 * custom edges), `values`, and `mask`. Topology (needs a new graph): `maximumClassCount`, `methods`,
 * `naturalBreaksBinCount`, view lengths, and whether `mask` and `output.classCounts` are present.
 */
export type GPUClassBreaksProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'class-breaks'`. */
  id?: string;
  /** Packed float32 column. NaN rows are skipped; infinities count only toward the end classes. */
  values: GraphDataView<'float32'>;
  /** Optional packed `uint32` row mask; zero skips the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Float32 parameter view of `getGPUClassBreaksParameterLength(maximumClassCount)` elements
   * written with `getGPUClassBreaksParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /** Compile-time class capacity, 1 to 64 (at least 6 when `'box-plot'` is compiled). */
  maximumClassCount: number;
  /**
   * Methods compiled into the graph; the per-frame method code chooses among them and every other
   * method's kernels return immediately. Defaults to every method. `'maximum-breaks'` adds two
   * full radix sorts that run every frame whatever the selected method, so leave it out unless it
   * is needed.
   */
  methods?: readonly GPUClassBreaksMethod[];
  /** Equal-width bins of the natural-breaks histogram, 2 to 2048. Default 1024. */
  naturalBreaksBinCount?: number;
  /** Caller-owned outputs. */
  output: GPUClassBreaksOutput;
};

/** First word of the head/tail break values in the head/tail buffer, after the state words. */
const HEAD_TAIL_BREAKS_OFFSET = 4;

/**
 * Computes choropleth class breaks of one column on the GPU every frame, with no readback: equal
 * interval, quantile, standard deviation, head/tail, box plot, maximum breaks, natural breaks
 * (Fisher-Jenks on a binned histogram), and custom edges.
 *
 * Methods (`n` unmasked non-NaN rows, `min`/`max` over the finite ones, `k` classes):
 * - `'equal-interval'`: `e[i] = min + ((max - min) / k) * i` in f32.
 * - `'quantile'`: `e[i]` is the linear (R-7, numpy and d3 default) quantile at `fround(i / k)`,
 *   computed exactly by `GPUColumnQuantiles` radix select. Ties may repeat edges (empty classes).
 * - `'standard-deviation'`: inner edges at `mean + (j - (k - 2) / 2) * interval * sd` for
 *   `j = 0..k-2`, centred on the mean (population sd, fixed-order Chan merge).
 * - `'head-tail'`: Jiang's head/tail breaks. Splits at the mean of the current head, then keeps
 *   the values above it, while the next head holds at most `headTailRatio` of the current rows, has
 *   more than one row, and fewer than `k` classes exist. Produces a data-dependent class count.
 * - `'box-plot'`: six classes with edges `[min, q1 - h * iqr, q1, median, q3, q3 + h * iqr, max]`.
 * - `'maximum-breaks'`: sorts the column, takes the `k - 1` largest gaps between consecutive
 *   distinct values (ties to the lower position), and breaks at their midpoints. Fewer classes when
 *   there are fewer distinct values.
 * - `'natural-breaks'`: exact Fisher-Jenks dynamic program over `naturalBreaksBinCount`
 *   equal-width bins of `[min, max]`, each bin weighted by its count at its centre. Breaks snap to
 *   bin edges, so the error is bounded by one bin width. Cost O(k * B^2) in parallel over bins.
 * - `'custom'`: the edges from the parameter view, used as given.
 *
 * Every computed inner edge is clamped to `[min, max]`. With no finite values the class count is 0
 * and every edge is NaN; when every finite value is equal the class count is 1 (except `'custom'`).
 * Infinite values are excluded from the moments, extremes, gaps, and histogram, but take part in
 * quantiles and classify into the end classes.
 *
 * Determinism: extremes, quantiles, counts, histograms, and sorts are integer-exact. Moments are
 * reduced in a fixed tree order and the dynamic program runs a fixed loop per bin, so every output
 * is bitwise identical across runs. Edges that come out of f32 arithmetic (equal interval,
 * standard deviation, box-plot whiskers, natural-breaks bin edges) may differ from a CPU f32
 * evaluation by an ulp where a driver fuses a multiply-add.
 */
export class GPUClassBreaks implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUClassBreaksProps;
  /** Compiled methods. */
  readonly methods: readonly GPUClassBreaksMethod[];
  /** Number of equal-width bins of the natural-breaks histogram. */
  readonly naturalBreaksBinCount: number;
  /** Number of quantile targets refined per frame. */
  readonly quantileCount: number;

  constructor(props: GPUClassBreaksProps) {
    this.id = props.id ?? 'class-breaks';
    this.props = props;
    const id = this.id;
    const {maximumClassCount, output} = props;
    if (
      !Number.isInteger(maximumClassCount) ||
      maximumClassCount < 1 ||
      maximumClassCount > MAXIMUM_CLASS_COUNT
    ) {
      throw new Error(`${id} maximumClassCount must be an integer in [1, ${MAXIMUM_CLASS_COUNT}]`);
    }
    this.methods = [...new Set(props.methods ?? GPU_CLASS_BREAKS_METHODS)];
    if (this.methods.length === 0) {
      throw new Error(`${id} needs at least one method`);
    }
    for (const method of this.methods) {
      if (CODES[method] === undefined) {
        throw new Error(`${id} unknown method ${String(method)}`);
      }
    }
    if (
      this.methods.includes('box-plot') &&
      maximumClassCount < GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT
    ) {
      throw new Error(`${id} box-plot needs maximumClassCount of at least 6`);
    }
    this.naturalBreaksBinCount = props.naturalBreaksBinCount ?? DEFAULT_NATURAL_BREAKS_BIN_COUNT;
    if (
      !Number.isInteger(this.naturalBreaksBinCount) ||
      this.naturalBreaksBinCount < 2 ||
      this.naturalBreaksBinCount > MAXIMUM_NATURAL_BREAKS_BIN_COUNT
    ) {
      throw new Error(
        `${id} naturalBreaksBinCount must be an integer in [2, ${MAXIMUM_NATURAL_BREAKS_BIN_COUNT}]`
      );
    }
    this.quantileCount = Math.max(maximumClassCount - 1, 3);
    for (const [name, view] of [
      ['values', props.values],
      ['mask', props.mask],
      ['parameters', props.parameters],
      ['output.breaks', output.breaks],
      ['output.classCount', output.classCount],
      ['output.classCounts', output.classCounts]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.values, ['float32'], `${id} values`);
    const rowCount = props.values.length;
    if (rowCount < 1 || rowCount > MAXIMUM_ROW_COUNT) {
      throw new Error(`${id} values must hold between 1 and 2^24 rows`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== rowCount) {
        throw new Error(`${id} mask length must equal values length`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    const parameterLength = getGPUClassBreaksParameterLength(maximumClassCount);
    if (props.parameters.length < parameterLength) {
      throw new Error(`${id} parameters must hold ${parameterLength} float32 values`);
    }
    validatePackedView(output.breaks, ['float32'], `${id} output.breaks`);
    if (output.breaks.length < maximumClassCount + 1) {
      throw new Error(`${id} output.breaks must hold maximumClassCount + 1 rows`);
    }
    validatePackedUint32View(output.classCount, `${id} output.classCount`);
    if (output.classCount.length < 1) {
      throw new Error(`${id} output.classCount must hold one row`);
    }
    if (output.classCounts) {
      validatePackedUint32View(output.classCounts, `${id} output.classCounts`);
      if (output.classCounts.length < maximumClassCount) {
        throw new Error(`${id} output.classCounts must hold maximumClassCount rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.breaks, output.classCount, output.classCounts],
      [props.values, props.mask, props.parameters]
    );
  }

  /** Returns prepare, extremes, per-method, finish, publish, and class-count nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, methods, quantileCount} = this;
    const {values, mask, parameters, output, maximumClassCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      values,
      mask,
      parameters,
      output.breaks,
      output.classCount,
      output.classCounts
    ]);
    const rowCount = values.length;
    const has = (method: GPUClassBreaksMethod) => methods.includes(method);
    const usesQuantiles = has('quantile') || has('box-plot');
    const usesStandardDeviation = has('standard-deviation');
    const usesHeadTail = has('head-tail');
    const headTailRounds = maximumClassCount - 1;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const read = (
      name: string,
      view: GraphDataView,
      type: 'u32' | 'f32'
    ): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const write = (
      name: string,
      view: GraphDataView,
      type: 'u32' | 'f32' | 'atomic<u32>' = 'u32'
    ): WGSLKernelBinding => ({name, view, type, access: 'read_write'});
    const maskBinding = mask ? [read('rowMask', mask, 'u32')] : [];
    const maskTest = mask ? 'rowMask[rowMaskOffset + index] != 0u' : 'true';

    const state = u32('state', 2);
    const extremes = u32('extremes', 4);
    const edges = f32('edges', maximumClassCount + 1);
    const edgeState = u32('edge-state', 1);
    const quantileGate = usesQuantiles ? u32('quantile-gate', 1) : undefined;
    const quantileParameters = usesQuantiles
      ? f32('quantile-parameters', 4 + quantileCount)
      : undefined;
    const quantileValues = usesQuantiles ? f32('quantile-values', quantileCount) : undefined;
    const usesMoments = usesStandardDeviation || usesHeadTail;
    const momentSlotCount = (usesHeadTail ? headTailRounds : 0) + 1;
    const moments = usesMoments ? f32('moments', momentSlotCount * MOMENT_STRIDE) : undefined;
    const partials = usesMoments
      ? f32('moment-partials', getClassBreaksMomentTileCount(rowCount) * MOMENT_STRIDE)
      : undefined;
    // Head/tail state words, then the break values as f32 bit patterns. One buffer keeps the
    // finish kernel within the default limit of 8 storage buffers per stage.
    const headTail = usesHeadTail
      ? u32('head-tail', HEAD_TAIL_BREAKS_OFFSET + Math.max(headTailRounds, 1))
      : undefined;
    const compiledMask = methods.reduce((bits, method) => bits | (1 << CODES[method]), 0);
    const methodRead = `state[stateOffset + ${CLASS_BREAKS_STATE.method}u]`;
    const classCountRead = `state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u]`;
    const nodes: GPUCommandNode<Parameters>[] = [];

    // 1. Decode the per-frame method and class count, and reset every accumulator.
    const prepareBindings: WGSLKernelBinding[] = [
      read('params', parameters, 'f32'),
      write('state', state),
      write('extremes', extremes),
      write('edgeState', edgeState),
      ...(quantileGate && quantileParameters
        ? [
            write('quantileGate', quantileGate),
            write('quantileParameters', quantileParameters, 'f32')
          ]
        : []),
      ...(headTail ? [write('headTail', headTail)] : [])
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-prepare`,
        operation: OPERATION,
        variant: 'prepare',
        bindings: prepareBindings,
        invocationCount: 1,
        declarations: `const COMPILED_METHODS: u32 = ${compiledMask}u;
const MAXIMUM_CLASS_COUNT: u32 = ${maximumClassCount}u;
const QUANTILE_COUNT: u32 = ${quantileCount}u;
const HEADER: u32 = ${GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH}u;`,
        body: `let methodValue = params[paramsOffset];
  var method = ${INVALID_METHOD}u;
  if (methodValue >= 0.0 && methodValue < 32.0 && floor(methodValue) == methodValue) {
    let code = u32(methodValue);
    if ((COMPILED_METHODS & (1u << code)) != 0u) {
      method = code;
    }
  }
  let classValue = params[paramsOffset + 1u];
  var classCount = 1u;
  if (classValue >= 1.0) {
    classCount = u32(min(classValue, f32(MAXIMUM_CLASS_COUNT)));
  }
  if (method == ${CODES['box-plot']}u) {
    classCount = ${GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT}u;
  }
  state[stateOffset + ${CLASS_BREAKS_STATE.method}u] = method;
  state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u] = classCount;
  extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u] = 0xffffffffu;
  extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u] = 0u;
  extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.validCount}u] = 0u;
  extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.finiteCount}u] = 0u;
  edgeState[edgeStateOffset] = 0u;
  ${
    quantileGate
      ? `quantileGate[quantileGateOffset] = select(0u, 1u, method == ${CODES.quantile}u || method == ${CODES['box-plot']}u);
  quantileParameters[quantileParametersOffset] = ${LINEAR_INTERPOLATION_CODE}.0;
  quantileParameters[quantileParametersOffset + 1u] = 0.0;
  quantileParameters[quantileParametersOffset + 2u] = 1.0;
  quantileParameters[quantileParametersOffset + 3u] = 0.0;
  for (var slot = 0u; slot < QUANTILE_COUNT; slot = slot + 1u) {
    quantileParameters[quantileParametersOffset + 4u + slot] = params[paramsOffset + HEADER + slot];
  }`
      : ''
  }
  ${
    headTail
      ? `headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.done}u] = 0u;
  headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.breakCount}u] = 0u;
  headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.thresholdKey}u] = 0u;
  headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.previousCount}u] = 0u;`
      : ''
  }`
      })
    );

    // 2. Exact extremes of the finite values and counts, on ordered keys.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-extremes`,
        operation: OPERATION,
        variant: 'extremes',
        bindings: [
          read('values', values, 'f32'),
          ...maskBinding,
          write('extremes', extremes, 'atomic<u32>')
        ],
        invocationCount: rowCount,
        declarations: COLUMN_ORDERED_KEY_WGSL,
        body: `let value = values[valuesOffset + index];
  if (isNanBits(value) || !(${maskTest})) {
    return;
  }
  atomicAdd(&extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.validCount}u], 1u);
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
    return;
  }
  let key = getOrderedKey(value);
  atomicAdd(&extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.finiteCount}u], 1u);
  atomicMin(&extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u], key);
  atomicMax(&extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u], key);`
      })
    );

    // 3. Exact linear quantiles for 'quantile' and 'box-plot', gated per frame.
    if (quantileGate && quantileParameters && quantileValues) {
      nodes.push(
        ...getColumnQuantileNodes<Parameters>(graph, {
          id: `${id}-quantiles`,
          operation: OPERATION,
          values,
          mask,
          parameters: quantileParameters,
          quantileCount,
          output: {quantiles: quantileValues, validCount: u32('quantile-valid-count', 1)},
          gate: {view: quantileGate, value: 1}
        })
      );
    }

    // 4. Fixed-order moments for 'standard-deviation' (the last slot).
    if (moments && partials && usesStandardDeviation) {
      nodes.push(
        ...getClassBreaksMomentNodes<Parameters>(graph, {
          id: `${id}-moments`,
          operation: OPERATION,
          values,
          mask,
          state,
          partials,
          moments,
          slot: momentSlotCount - 1,
          gate: {kind: 'method', methodCode: CODES['standard-deviation']}
        })
      );
    }

    // 5. Head/tail rounds: the mean of the current head, then a one-thread split decision.
    if (moments && partials && headTail) {
      for (let round = 0; round < headTailRounds; round++) {
        nodes.push(
          ...getClassBreaksMomentNodes<Parameters>(graph, {
            id: `${id}-head-tail-${round}`,
            operation: OPERATION,
            values,
            mask,
            state,
            partials,
            moments,
            slot: round,
            gate: {kind: 'head-tail', methodCode: CODES['head-tail'], headTail, round}
          })
        );
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-head-tail-${round}-split`,
            operation: OPERATION,
            variant: 'head-tail-split',
            bindings: [
              read('params', parameters, 'f32'),
              read('state', state, 'u32'),
              read('moments', moments, 'f32'),
              write('headTail', headTail)
            ],
            invocationCount: 1,
            declarations: COLUMN_ORDERED_KEY_WGSL,
            body: headTailSplitBody(round, methodRead, classCountRead)
          })
        );
      }
    }

    // 6. Maximum breaks: sort the finite keys, sort the gaps between distinct neighbours.
    if (has('maximum-breaks')) {
      nodes.push(...this.getMaximumBreaksNodes(graph, {state, extremes, edges, edgeState}));
    }

    // 7. Natural breaks: histogram, class cost matrix, dynamic program, backtrack.
    if (has('natural-breaks')) {
      nodes.push(...this.getNaturalBreaksNodes(graph, {state, extremes, edges, edgeState}));
    }

    // 8. Closed-form edges for equal interval, quantile, standard deviation, box plot, custom.
    const finishBindings: WGSLKernelBinding[] = [
      read('params', parameters, 'f32'),
      read('state', state, 'u32'),
      read('extremes', extremes, 'u32'),
      ...(quantileValues ? [read('quantileValues', quantileValues, 'f32')] : []),
      ...(moments ? [read('moments', moments, 'f32')] : []),
      ...(headTail ? [read('headTail', headTail, 'u32')] : []),
      write('edges', edges, 'f32')
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation: OPERATION,
        variant: 'finish',
        bindings: [...finishBindings, write('edgeState', edgeState)],
        invocationCount: 1,
        declarations: `${COLUMN_ORDERED_KEY_WGSL}
const HEADER: u32 = ${GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH}u;`,
        body: this.getFinishBody({
          methodRead,
          classCountRead,
          hasQuantiles: Boolean(quantileValues),
          momentsSlot: usesStandardDeviation ? momentSlotCount - 1 : -1,
          hasHeadTail: Boolean(headTail)
        })
      })
    );

    // 9. Publish: clamp, fill the end edges, apply the degenerate-data rules, NaN the tail.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        variant: 'publish',
        bindings: [
          read('state', state, 'u32'),
          read('extremes', extremes, 'u32'),
          read('edges', edges, 'f32'),
          read('edgeState', edgeState, 'u32'),
          write('breaksOut', output.breaks, 'f32'),
          write('classCountOut', output.classCount)
        ],
        invocationCount: 1,
        declarations: `${COLUMN_ORDERED_KEY_WGSL}
const MAXIMUM_CLASS_COUNT: u32 = ${maximumClassCount}u;`,
        body: `let method = ${methodRead};
  let finiteCount = extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.finiteCount}u];
  let minimum = decodeOrderedKey(extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u]);
  let maximum = decodeOrderedKey(extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u]);
  var classes = edgeState[edgeStateOffset];
  let isCustom = method == ${CODES.custom}u;
  if (method == ${INVALID_METHOD}u) {
    classes = 0u;
  } else if (!isCustom && finiteCount == 0u) {
    classes = 0u;
  } else if (!isCustom && minimum == maximum) {
    classes = 1u;
  }
  classes = min(classes, MAXIMUM_CLASS_COUNT);
  let nan = getNaN();
  for (var edge = 0u; edge <= MAXIMUM_CLASS_COUNT; edge = edge + 1u) {
    var value = nan;
    if (classes > 0u && edge <= classes) {
      if (isCustom) {
        value = edges[edgesOffset + edge];
      } else if (edge == 0u) {
        value = minimum;
      } else if (edge == classes) {
        value = maximum;
      } else {
        value = clamp(edges[edgesOffset + edge], minimum, maximum);
      }
    }
    breaksOut[breaksOutOffset + edge] = value;
  }
  classCountOut[classCountOutOffset] = classes;`
      })
    );

    // 10. Optional per-class row counts with the stream's class rule.
    if (output.classCounts) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-class-counts-clear`,
          operation: OPERATION,
          variant: 'class-counts-clear',
          bindings: [write('classCounts', output.classCounts)],
          invocationCount: maximumClassCount,
          body: 'classCounts[classCountsOffset + index] = 0u;'
        })
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-class-counts`,
          operation: OPERATION,
          variant: 'class-counts',
          bindings: [
            read('values', values, 'f32'),
            ...maskBinding,
            read('breaks', output.breaks, 'f32'),
            read('classCountIn', output.classCount, 'u32'),
            write('classCounts', output.classCounts, 'atomic<u32>')
          ],
          invocationCount: rowCount,
          declarations: COLUMN_ORDERED_KEY_WGSL,
          body: `let value = values[valuesOffset + index];
  let classes = classCountIn[classCountInOffset];
  if (classes == 0u || isNanBits(value) || !(${maskTest})) {
    return;
  }
  let key = getOrderedKey(value);
  // Binary search for the number of inner edges e[1..classes-1] whose key is at or below key.
  var low = 1u;
  var high = classes;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (getOrderedKey(breaks[breaksOffset + middle]) <= key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  atomicAdd(&classCounts[classCountsOffset + low - 1u], 1u);`
        })
      );
    }
    return nodes;
  }

  /** Builds the WGSL body of the closed-form finish kernel. */
  private getFinishBody(props: {
    methodRead: string;
    classCountRead: string;
    hasQuantiles: boolean;
    momentsSlot: number;
    hasHeadTail: boolean;
  }): string {
    const {methodRead, classCountRead, hasQuantiles, momentsSlot, hasHeadTail} = props;
    const minimumKey = `extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u]`;
    const maximumKey = `extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u]`;
    const momentBase = momentsSlot * MOMENT_STRIDE;
    return `let method = ${methodRead};
  let classCount = ${classCountRead};
  let minimum = decodeOrderedKey(${minimumKey});
  let maximum = decodeOrderedKey(${maximumKey});
  if (method == ${CODES['equal-interval']}u) {
    let width = (maximum - minimum) / f32(classCount);
    for (var edge = 1u; edge < classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = minimum + width * f32(edge);
    }
    edgeState[edgeStateOffset] = classCount;
  }
  ${
    hasQuantiles
      ? `if (method == ${CODES.quantile}u) {
    for (var edge = 1u; edge < classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = quantileValues[quantileValuesOffset + edge - 1u];
    }
    edgeState[edgeStateOffset] = classCount;
  }
  if (method == ${CODES['box-plot']}u) {
    let hinge = params[paramsOffset + 4u];
    let lowerQuartile = quantileValues[quantileValuesOffset];
    let median = quantileValues[quantileValuesOffset + 1u];
    let upperQuartile = quantileValues[quantileValuesOffset + 2u];
    let spread = upperQuartile - lowerQuartile;
    edges[edgesOffset + 1u] = lowerQuartile - hinge * spread;
    edges[edgesOffset + 2u] = lowerQuartile;
    edges[edgesOffset + 3u] = median;
    edges[edgesOffset + 4u] = upperQuartile;
    edges[edgesOffset + 5u] = upperQuartile + hinge * spread;
    edgeState[edgeStateOffset] = ${GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT}u;
  }`
      : ''
  }
  ${
    momentsSlot >= 0
      ? `if (method == ${CODES['standard-deviation']}u) {
    let count = moments[momentsOffset + ${momentBase}u];
    let mean = moments[momentsOffset + ${momentBase + 1}u];
    let deviation = sqrt(moments[momentsOffset + ${momentBase + 2}u] / max(count, 1.0));
    let interval = params[paramsOffset + 2u] * deviation;
    let center = f32(classCount - 2u) * 0.5;
    for (var edge = 1u; edge < classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = mean + (f32(edge - 1u) - center) * interval;
    }
    edgeState[edgeStateOffset] = classCount;
  }`
      : ''
  }
  ${
    hasHeadTail
      ? `if (method == ${CODES['head-tail']}u) {
    let breakCount = headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.breakCount}u];
    for (var edge = 1u; edge <= breakCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = bitcast<f32>(
        headTail[headTailOffset + ${HEAD_TAIL_BREAKS_OFFSET - 1}u + edge]
      );
    }
    edgeState[edgeStateOffset] = breakCount + 1u;
  }`
      : ''
  }
  if (method == ${CODES.custom}u) {
    for (var edge = 0u; edge <= classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = params[paramsOffset + HEADER + edge];
    }
    edgeState[edgeStateOffset] = classCount;
  }`;
  }

  /** Sort-based exact maximum breaks. */
  private getMaximumBreaksNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    shared: {
      state: GraphDataView<'uint32'>;
      extremes: GraphDataView<'uint32'>;
      edges: GraphDataView<'float32'>;
      edgeState: GraphDataView<'uint32'>;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {values, mask, maximumClassCount} = props;
    const rowCount = values.length;
    const prefix = `${id}-maximum-breaks`;
    const u32 = (name: string) =>
      createTransientView(graph, `${prefix}-${name}`, 'uint32', rowCount);
    const keys = u32('keys');
    const rows = u32('rows');
    const sortedKeys = u32('sorted-keys');
    const sortedRows = u32('sorted-rows');
    const gapKeys = u32('gap-keys');
    const gapPositions = u32('gap-positions');
    const sortedGapKeys = u32('sorted-gap-keys');
    const sortedGapPositions = u32('sorted-gap-positions');
    const gate = `state[stateOffset + ${CLASS_BREAKS_STATE.method}u] != ${CODES['maximum-breaks']}u`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${prefix}-keys`,
        operation: OPERATION,
        variant: 'maximum-breaks-keys',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          ...(mask ? [{name: 'rowMask', view: mask, type: 'u32', access: 'read'} as const] : []),
          {name: 'state', view: shared.state, type: 'u32', access: 'read'},
          {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
          {name: 'rows', view: rows, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: COLUMN_ORDERED_KEY_WGSL,
        body: `if (${gate}) {
    return;
  }
  let value = values[valuesOffset + index];
  var include = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
  ${mask ? 'include = include && rowMask[rowMaskOffset + index] != 0u;' : ''}
  // Excluded rows take the NaN key 0xffffffff, which no finite value has, and sort last.
  keys[keysOffset + index] = select(0xffffffffu, getOrderedKey(value), include);
  rows[rowsOffset + index] = index;`
      })
    );
    nodes.push(
      ...new GPUSort({
        id: `${prefix}-sort`,
        keys,
        values: rows,
        outputKeys: sortedKeys,
        outputValues: sortedRows,
        algorithm: 'radix'
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${prefix}-gaps`,
        operation: OPERATION,
        variant: 'maximum-breaks-gaps',
        bindings: [
          {name: 'state', view: shared.state, type: 'u32', access: 'read'},
          {name: 'extremes', view: shared.extremes, type: 'u32', access: 'read'},
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'gapKeys', view: gapKeys, type: 'u32', access: 'read_write'},
          {name: 'gapPositions', view: gapPositions, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: COLUMN_ORDERED_KEY_WGSL,
        body: `if (${gate}) {
    return;
  }
  let finiteCount = extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.finiteCount}u];
  var gapKey = 0xffffffffu;
  if (index > 0u && index < finiteCount) {
    let lowerKey = sortedKeys[sortedKeysOffset + index - 1u];
    let upperKey = sortedKeys[sortedKeysOffset + index];
    if (upperKey != lowerKey) {
      let gap = decodeOrderedKey(upperKey) - decodeOrderedKey(lowerKey);
      // Inverted keys sort the largest gap first; the stable sort keeps lower positions first.
      gapKey = ~getOrderedKey(gap);
    }
  }
  gapKeys[gapKeysOffset + index] = gapKey;
  gapPositions[gapPositionsOffset + index] = index;`
      })
    );
    nodes.push(
      ...new GPUSort({
        id: `${prefix}-gap-sort`,
        keys: gapKeys,
        values: gapPositions,
        outputKeys: sortedGapKeys,
        outputValues: sortedGapPositions,
        algorithm: 'radix'
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${prefix}-finish`,
        operation: OPERATION,
        variant: 'maximum-breaks-finish',
        bindings: [
          {name: 'state', view: shared.state, type: 'u32', access: 'read'},
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'sortedGapKeys', view: sortedGapKeys, type: 'u32', access: 'read'},
          {name: 'sortedGapPositions', view: sortedGapPositions, type: 'u32', access: 'read'},
          {name: 'edges', view: shared.edges, type: 'f32', access: 'read_write'},
          {name: 'edgeState', view: shared.edgeState, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `${COLUMN_ORDERED_KEY_WGSL}
const ROW_COUNT: u32 = ${rowCount}u;
const MAXIMUM_CLASS_COUNT: u32 = ${maximumClassCount}u;`,
        body: `if (${gate}) {
    return;
  }
  let classCount = state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u];
  var midpoints: array<f32, ${maximumClassCount}>;
  var found = 0u;
  for (var rank = 0u; rank + 1u < classCount && rank < ROW_COUNT; rank = rank + 1u) {
    if (sortedGapKeys[sortedGapKeysOffset + rank] == 0xffffffffu) {
      break;
    }
    let position = sortedGapPositions[sortedGapPositionsOffset + rank];
    let lower = decodeOrderedKey(sortedKeys[sortedKeysOffset + position - 1u]);
    let upper = decodeOrderedKey(sortedKeys[sortedKeysOffset + position]);
    // (upper - lower) * 0.5 is exact, so a fused multiply-add rounds the same way.
    let midpoint = lower + (upper - lower) * 0.5;
    // Insertion sort keeps the midpoints ascending.
    var slot = found;
    while (slot > 0u && midpoints[slot - 1u] > midpoint) {
      midpoints[slot] = midpoints[slot - 1u];
      slot = slot - 1u;
    }
    midpoints[slot] = midpoint;
    found = found + 1u;
  }
  for (var edge = 0u; edge < found; edge = edge + 1u) {
    edges[edgesOffset + edge + 1u] = midpoints[edge];
  }
  edgeState[edgeStateOffset] = found + 1u;`
      })
    );
    return nodes;
  }

  /** Fisher-Jenks over an equal-width histogram. */
  private getNaturalBreaksNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    shared: {
      state: GraphDataView<'uint32'>;
      extremes: GraphDataView<'uint32'>;
      edges: GraphDataView<'float32'>;
      edgeState: GraphDataView<'uint32'>;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props, naturalBreaksBinCount: binCount} = this;
    const {values, mask, maximumClassCount} = props;
    const rowCount = values.length;
    const prefix = `${id}-natural-breaks`;
    const histogram = createTransientView(graph, `${prefix}-histogram`, 'uint32', binCount);
    const costs = createTransientView(
      graph,
      `${prefix}-class-costs`,
      'float32',
      binCount * binCount
    );
    const totals = createTransientView(
      graph,
      `${prefix}-totals`,
      'float32',
      maximumClassCount * binCount
    );
    const starts = createTransientView(
      graph,
      `${prefix}-starts`,
      'uint32',
      maximumClassCount * binCount
    );
    const stateRead = {name: 'state', view: shared.state, type: 'u32', access: 'read'} as const;
    const gate = `state[stateOffset + ${CLASS_BREAKS_STATE.method}u] != ${CODES['natural-breaks']}u`;
    const binDeclarations = `${COLUMN_ORDERED_KEY_WGSL}
const BIN_COUNT: u32 = ${binCount}u;
const INVERSE_BIN_COUNT: f32 = ${getWGSLFloatLiteral(1 / binCount)};
${CLASS_BREAKS_BIN_WGSL}`;
    const extremesRead = `let minimum = decodeOrderedKey(extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u]);
  let maximum = decodeOrderedKey(extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u]);
  let width = (maximum - minimum) * INVERSE_BIN_COUNT;`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${prefix}-clear`,
        operation: OPERATION,
        variant: 'natural-breaks-clear',
        bindings: [
          stateRead,
          {name: 'histogram', view: histogram, type: 'u32', access: 'read_write'}
        ],
        invocationCount: binCount,
        body: `if (${gate}) {
    return;
  }
  histogram[histogramOffset + index] = 0u;`
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${prefix}-histogram`,
        operation: OPERATION,
        variant: 'natural-breaks-histogram',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          ...(mask ? [{name: 'rowMask', view: mask, type: 'u32', access: 'read'} as const] : []),
          stateRead,
          {name: 'extremes', view: shared.extremes, type: 'u32', access: 'read'},
          {name: 'histogram', view: histogram, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: binDeclarations,
        body: `if (${gate}) {
    return;
  }
  let value = values[valuesOffset + index];
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
    return;
  }
  ${mask ? 'if (rowMask[rowMaskOffset + index] == 0u) {\n    return;\n  }' : ''}
  ${extremesRead}
  var bin = 0u;
  if (width > 0.0) {
    bin = getBinIndex(value, minimum, width);
  }
  atomicAdd(&histogram[histogramOffset + bin], 1u);`
      })
    );
    // Class cost of bins [start, end]: weighted sum of squared deviations of bin centres, by a
    // weighted Welford update in a fixed loop per start bin. Bin indices stand in for centres,
    // since the dynamic program is invariant to the affine map from bin index to value.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${prefix}-class-costs`,
        operation: OPERATION,
        variant: 'natural-breaks-class-costs',
        bindings: [
          stateRead,
          {name: 'histogram', view: histogram, type: 'u32', access: 'read'},
          {name: 'costs', view: costs, type: 'f32', access: 'read_write'}
        ],
        invocationCount: binCount,
        declarations: `const BIN_COUNT: u32 = ${binCount}u;`,
        body: `if (${gate}) {
    return;
  }
  let start = index;
  var weight = 0.0;
  var mean = 0.0;
  var squares = 0.0;
  for (var end = start; end < BIN_COUNT; end = end + 1u) {
    let count = f32(histogram[histogramOffset + end]);
    if (count > 0.0) {
      let center = f32(end);
      let total = weight + count;
      let delta = center - mean;
      mean = mean + delta * (count / total);
      squares = squares + count * delta * (center - mean);
      weight = total;
    }
    costs[costsOffset + start * BIN_COUNT + end] = squares;
  }`
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${prefix}-layer-0`,
        operation: OPERATION,
        variant: 'natural-breaks-layer',
        bindings: [
          stateRead,
          {name: 'costs', view: costs, type: 'f32', access: 'read'},
          {name: 'totals', view: totals, type: 'f32', access: 'read_write'},
          {name: 'starts', view: starts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: binCount,
        declarations: `const BIN_COUNT: u32 = ${binCount}u;`,
        body: `if (${gate}) {
    return;
  }
  totals[totalsOffset + index] = costs[costsOffset + index];
  starts[startsOffset + index] = 0u;`
      })
    );
    // Layer c: the best split of bins [0, end] into c + 1 classes. Ties keep the lowest start.
    for (let layer = 1; layer < maximumClassCount; layer++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${prefix}-layer-${layer}`,
          operation: OPERATION,
          variant: 'natural-breaks-layer',
          bindings: [
            stateRead,
            {name: 'costs', view: costs, type: 'f32', access: 'read'},
            {name: 'totals', view: totals, type: 'f32', access: 'read_write'},
            {name: 'starts', view: starts, type: 'u32', access: 'read_write'}
          ],
          invocationCount: binCount,
          declarations: `const BIN_COUNT: u32 = ${binCount}u;
const LAYER: u32 = ${layer}u;`,
          body: `if (${gate} || LAYER >= state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u]) {
    return;
  }
  let end = index;
  var best = ${LARGE_COST};
  var bestStart = LAYER;
  for (var start = LAYER; start <= end; start = start + 1u) {
    let candidate = totals[totalsOffset + (LAYER - 1u) * BIN_COUNT + start - 1u] + costs[costsOffset + start * BIN_COUNT + end];
    if (candidate < best) {
      best = candidate;
      bestStart = start;
    }
  }
  totals[totalsOffset + LAYER * BIN_COUNT + end] = best;
  starts[startsOffset + LAYER * BIN_COUNT + end] = bestStart;`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${prefix}-backtrack`,
        operation: OPERATION,
        variant: 'natural-breaks-backtrack',
        bindings: [
          stateRead,
          {name: 'extremes', view: shared.extremes, type: 'u32', access: 'read'},
          {name: 'starts', view: starts, type: 'u32', access: 'read'},
          {name: 'edges', view: shared.edges, type: 'f32', access: 'read_write'},
          {name: 'edgeState', view: shared.edgeState, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: binDeclarations,
        body: `if (${gate}) {
    return;
  }
  ${extremesRead}
  let classCount = min(state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u], BIN_COUNT);
  var end = BIN_COUNT - 1u;
  for (var layer = classCount - 1u; layer > 0u; layer = layer - 1u) {
    let start = starts[startsOffset + layer * BIN_COUNT + end];
    // The class starting at bin 'start' begins at that bin's lower edge.
    edges[edgesOffset + layer] = minimum + f32(start) * width;
    end = start - 1u;
  }
  edgeState[edgeStateOffset] = classCount;`
      })
    );
    return nodes;
  }
}

/** One head/tail split decision for `round`: take the mean of the current head as a break. */
function headTailSplitBody(round: number, methodRead: string, classCountRead: string): string {
  const base = round * MOMENT_STRIDE;
  const done = `headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.done}u]`;
  return `if (${methodRead} != ${CODES['head-tail']}u || ${done} != 0u) {
    return;
  }
  let count = moments[momentsOffset + ${base}u];
  let mean = moments[momentsOffset + ${base + 1}u];
  let minimum = moments[momentsOffset + ${base + 3}u];
  let maximum = moments[momentsOffset + ${base + 4}u];
  var ratio = params[paramsOffset + 3u];
  if (!(ratio >= 0.0)) {
    ratio = 0.4;
  }
  // At most k - 1 breaks; stop on an empty or single-valued head.
  if (${round}u + 1u >= ${classCountRead} || count < 1.0 || !(minimum < maximum)) {
    ${done} = 1u;
    return;
  }
  ${
    round > 0
      ? `let previousCount = bitcast<f32>(headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.previousCount}u]);
  // The head kept from the previous split must be a minority (the 40% rule).
  if (count > ratio * previousCount || count <= 1.0) {
    ${done} = 1u;
    return;
  }`
      : ''
  }
  headTail[headTailOffset + ${HEAD_TAIL_BREAKS_OFFSET + round}u] = bitcast<u32>(mean);
  headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.breakCount}u] = ${round + 1}u;
  headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.thresholdKey}u] = getOrderedKey(mean);
  headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.previousCount}u] = bitcast<u32>(count);`;
}
