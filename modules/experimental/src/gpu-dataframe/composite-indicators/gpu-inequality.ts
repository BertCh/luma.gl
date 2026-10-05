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
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH,
  GPU_INEQUALITY_PARAMETER_LENGTH
} from './inequality-parameters';
import {
  getInequalityGlobalGiniBody,
  getInequalityKeysBody,
  getInequalityZoneStatsBody,
  INEQUALITY_GLOBAL_SUMS_BODY,
  INEQUALITY_HELPERS_WGSL,
  INEQUALITY_STAT,
  INEQUALITY_STATS_STRIDE,
  INEQUALITY_ZONE_KEYS_BODY
} from './inequality-wgsl';

const OPERATION = 'GPUInequality';

/** Smallest number of Lorenz knots. */
const MINIMUM_LORENZ_KNOT_COUNT = 2;
/** Largest number of Lorenz knots. */
const MAXIMUM_LORENZ_KNOT_COUNT = 1025;

/** Caller-owned outputs of {@link GPUInequality}. Every output is optional; at least one is needed. */
export type GPUInequalityOutput = {
  /** Gini coefficient per zone; NaN for an empty zone or zero total income. */
  gini?: GraphDataView<'float32'>;
  /** Theil T index per zone (0 at perfect equality, `ln n` at maximal inequality). */
  theilT?: GraphDataView<'float32'>;
  /** Theil L (mean log deviation) per zone; NaN when the zone has a zero value. */
  theilL?: GraphDataView<'float32'>;
  /**
   * Atkinson index per zone for the per-frame epsilon; NaN when epsilon is at least 1 and the zone
   * has a zero value.
   */
  atkinson?: GraphDataView<'float32'>;
  /** Hoover (Robin Hood) index per zone, the share of income to redistribute for equality. */
  hoover?: GraphDataView<'float32'>;
  /** Palma ratio per zone, income share of the top cut over the bottom cut; NaN if the bottom share is 0. */
  palma?: GraphDataView<'float32'>;
  /** Weighted mean value per zone; NaN for an empty zone. */
  mean?: GraphDataView<'float32'>;
  /** Number of included rows per zone. */
  count?: GraphDataView<'uint32'>;
  /** Lorenz curve knots, row-major `zone * lorenzKnotCount + knot`; NaN for an empty zone. */
  lorenzKnots?: GraphDataView<'float32'>;
  /**
   * Pooled statistics and the Theil T decomposition, indexed by `GPU_INEQUALITY_GLOBAL_SUMMARY`.
   * At least `GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH` values.
   */
  globalSummary?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUInequality}.
 *
 * Per-frame (no rebuild or recompile): the contents of `values`, `zoneIds`, `weights`, `mask` and
 * `parameters` (Atkinson epsilon and the Palma cuts). Compile-time: the row count, `zoneCount`,
 * `lorenzKnotCount`, and which optional views are present.
 */
export type GPUInequalityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'inequality'`. */
  id?: string;
  /** Packed float32 incomes. Negative and non-finite rows are excluded. */
  values: GraphDataView<'float32'>;
  /** Packed uint32 zone per row. `0xffffffff` or any value `>= zoneCount` excludes the row. */
  zoneIds: GraphDataView<'uint32'>;
  /** Number of zones. Compile-time. */
  zoneCount: number;
  /** Optional packed float32 population weights; rows without a finite weight above 0 are excluded. */
  weights?: GraphDataView<'float32'>;
  /** Optional packed `uint32` row mask; zero excludes the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Per-frame parameters: float32 view of at least `GPU_INEQUALITY_PARAMETER_LENGTH` elements
   * written with `getGPUInequalityParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /** Lorenz knots per zone at evenly spaced population fractions, 2 to 1025. Defaults to 11. Compile-time. */
  lorenzKnotCount?: number;
  /** Caller-owned outputs. */
  output: GPUInequalityOutput;
};

/**
 * Per-zone income inequality indices from a column of non-negative values: Gini, Theil T and L,
 * Atkinson, Hoover, Palma, the Lorenz curve, plus the between/within-zone decomposition of Theil T.
 *
 * A row is included when its mask is non-zero, its zone is below `zoneCount`, its value is finite
 * and non-negative, and (with `weights`) its weight is finite and positive. Everything below is
 * weighted by `w` (1 without weights), with `W = sum w`, `S = sum w x` and `mu = S / W`. Zones with
 * `S = 0` report NaN for every index but the mean and count.
 *
 * - Gini: `1 - sum_i (P_i - P_(i-1)) (L_i + L_(i-1))`, the trapezoid area under the Lorenz curve
 *   of the rows sorted by value, `P` and `L` being the cumulative population and income shares.
 *   With unit weights this equals `2 sum_i i x_(i) / (n S) - (n + 1) / n`; one holder of all
 *   income among `n` rows gives `(n - 1) / n`.
 * - Theil T: `(1 / W) sum w r ln r` with `r = x / mu` and `0 ln 0 = 0`.
 * - Theil L: `-(1 / W) sum w ln r`. A zone containing a zero value reports NaN (zeros are not
 *   dropped, so a result never silently describes a different population).
 * - Atkinson with epsilon `e`: `1 - ((1 / W) sum w r^(1 - e))^(1 / (1 - e))`, and
 *   `1 - exp((1 / W) sum w ln r)` at `e = 1`. For `e < 1` zeros are fine; for `e >= 1` a zero
 *   gives NaN. A result that overflows f32 is NaN.
 * - Hoover: `sum w |x - mu| / (2 S)`.
 * - Palma: `(1 - L(1 - top)) / L(bottom)` for the per-frame cuts (0.1 and 0.4 by default), where
 *   `L(p)` is the Lorenz curve of the rows sorted by value. The curve is linear inside each row
 *   (a row's income is spread over its population share), so cuts that fall inside a row take a
 *   fractional share of it. NaN when `L(bottom)` is 0.
 * - Lorenz knots: `L(k / (K - 1))` for `k = 0 .. K - 1` with the same linear interpolation, so
 *   `L(0) = 0` and `L(1) = 1`.
 * - Global summary: pooled Theil T computed directly from all included rows, and the decomposition
 *   `between = sum_g s_g ln(s_g / p_g)`, `within = sum_g s_g T_g` with income share `s_g` and
 *   population share `p_g`; total equals between plus within up to f32 rounding. Pooled Gini
 *   comes from the globally value-sorted rows.
 *
 * Algorithm: rows are sorted by an order-preserving u32 value key with `GPUSort`, re-keyed by zone
 * and sorted again with a stable radix `GPUSort`, so each zone owns one value-ordered segment.
 * One thread per zone binary-searches its segment in the sorted zone keys and walks it twice in
 * fixed order. No atomics are used at all, so results are bitwise reproducible on one adapter.
 *
 * Cost: one thread walks a whole zone, O(segment length) serial work, so one zone holding most
 * rows serializes the pass; the pooled Gini and the global sums are single-thread passes over all
 * rows or zones. Sums are plain f32 in sorted order, so precision degrades for segments of
 * millions of rows.
 */
export class GPUInequality implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUInequalityProps;
  /** Row count, `values.length`. */
  readonly rowCount: number;
  /** Lorenz knots per zone. */
  readonly lorenzKnotCount: number;

  constructor(props: GPUInequalityProps) {
    this.id = props.id ?? 'inequality';
    this.props = props;
    const id = this.id;
    const {output, zoneCount} = props;
    if (!Number.isInteger(zoneCount) || zoneCount < 1 || zoneCount >= 0x7fffffff) {
      throw new Error(`${id} zoneCount must be a positive integer`);
    }
    this.lorenzKnotCount = props.lorenzKnotCount ?? 11;
    if (
      !Number.isInteger(this.lorenzKnotCount) ||
      this.lorenzKnotCount < MINIMUM_LORENZ_KNOT_COUNT ||
      this.lorenzKnotCount > MAXIMUM_LORENZ_KNOT_COUNT
    ) {
      throw new Error(
        `${id} lorenzKnotCount must be an integer in [${MINIMUM_LORENZ_KNOT_COUNT}, ${MAXIMUM_LORENZ_KNOT_COUNT}]`
      );
    }
    for (const [name, view] of [
      ['values', props.values],
      ['zoneIds', props.zoneIds],
      ['weights', props.weights],
      ['mask', props.mask],
      ['parameters', props.parameters]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.values, ['float32'], `${id} values`);
    validatePackedUint32View(props.zoneIds, `${id} zoneIds`);
    this.rowCount = props.values.length;
    if (this.rowCount < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (props.zoneIds.length !== this.rowCount) {
      throw new Error(`${id} zoneIds length must equal the row count`);
    }
    if (props.weights) {
      validatePackedView(props.weights, ['float32'], `${id} weights`);
      if (props.weights.length !== this.rowCount) {
        throw new Error(`${id} weights length must equal the row count`);
      }
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== this.rowCount) {
        throw new Error(`${id} mask length must equal the row count`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_INEQUALITY_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_INEQUALITY_PARAMETER_LENGTH} float32 values`
      );
    }
    const lengths: [keyof GPUInequalityOutput, number, 'float32' | 'uint32'][] = [
      ['gini', zoneCount, 'float32'],
      ['theilT', zoneCount, 'float32'],
      ['theilL', zoneCount, 'float32'],
      ['atkinson', zoneCount, 'float32'],
      ['hoover', zoneCount, 'float32'],
      ['palma', zoneCount, 'float32'],
      ['mean', zoneCount, 'float32'],
      ['count', zoneCount, 'uint32'],
      ['lorenzKnots', zoneCount * this.lorenzKnotCount, 'float32'],
      ['globalSummary', GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH, 'float32']
    ];
    let outputCount = 0;
    for (const [name, length, format] of lengths) {
      const view = output[name];
      if (!view) {
        continue;
      }
      outputCount++;
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} output.${name} must be a single packed view, not a chunked vector`);
      }
      validatePackedView(view, [format], `${id} output.${name}`);
      if (view.length < length) {
        throw new Error(`${id} output.${name} must hold at least ${length} values`);
      }
    }
    if (outputCount === 0) {
      throw new Error(`${id} needs at least one output`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      lengths.map(([name]) => output[name]),
      [props.values, props.zoneIds, props.weights, props.mask, props.parameters]
    );
  }

  /**
   * Returns the key, value-sort, zone-key, zone-sort, zone-statistics, output-copy and optional
   * global-summary nodes in order. Node IDs are `${id}-keys`, `${id}-value-sort-*`,
   * `${id}-zone-keys`, `${id}-zone-sort-*`, `${id}-zone-stats`, `${id}-publish-indices`,
   * `${id}-publish-moments`, `${id}-global-sums` and `${id}-global-gini`.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, rowCount, lorenzKnotCount} = this;
    const {output, zoneCount, values, zoneIds, weights, mask, parameters} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      values,
      zoneIds,
      weights,
      mask,
      parameters,
      ...Object.values(output)
    ]);
    const transient = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, Math.max(length, 1));
    const valueKeys = transient('value-keys', 'uint32', rowCount);
    const rowIndices = transient('row-indices', 'uint32', rowCount);
    const sortedValueKeys = transient('sorted-value-keys', 'uint32', rowCount);
    const sortedRows = transient('sorted-rows', 'uint32', rowCount);
    const zoneKeys = transient('zone-keys', 'uint32', rowCount);
    const sortedZoneKeys = transient('sorted-zone-keys', 'uint32', rowCount);
    const zoneSortedRows = transient('zone-sorted-rows', 'uint32', rowCount);
    const stats = transient('stats', 'float32', zoneCount * INEQUALITY_STATS_STRIDE);
    const counts = output.count ?? transient('counts', 'uint32', zoneCount);
    const lorenz =
      output.lorenzKnots ?? transient('lorenz', 'float32', zoneCount * lorenzKnotCount);

    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const write = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read_write'
    });
    const declarations = `const ROW_COUNT: u32 = ${rowCount}u;
const ZONE_COUNT: u32 = ${zoneCount}u;
const KNOT_COUNT: u32 = ${lorenzKnotCount}u;
const STATS_STRIDE: u32 = ${INEQUALITY_STATS_STRIDE}u;
${INEQUALITY_HELPERS_WGSL}`;
    const weightExpression = weights ? 'weights[weightsOffset + row]' : '1.0';
    const nodes: GPUCommandNode<Parameters>[] = [];

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-keys`,
        operation: OPERATION,
        variant: 'keys',
        bindings: [
          read('values', values, 'f32'),
          read('zoneIds', zoneIds, 'u32'),
          ...(mask ? [read('rowMask', mask, 'u32')] : []),
          ...(weights ? [read('weights', weights, 'f32')] : []),
          write('valueKeys', valueKeys, 'u32'),
          write('rowIndices', rowIndices, 'u32')
        ],
        invocationCount: rowCount,
        declarations,
        body: getInequalityKeysBody({hasMask: Boolean(mask), hasWeights: Boolean(weights)})
      })
    );
    nodes.push(
      ...new GPUSort({
        id: `${id}-value-sort`,
        keys: valueKeys,
        values: rowIndices,
        outputKeys: sortedValueKeys,
        outputValues: sortedRows,
        keyBits: 32
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-zone-keys`,
        operation: OPERATION,
        variant: 'zone-keys',
        bindings: [
          read('sortedValueKeys', sortedValueKeys, 'u32'),
          read('sortedRows', sortedRows, 'u32'),
          read('zoneIds', zoneIds, 'u32'),
          write('zoneKeys', zoneKeys, 'u32')
        ],
        invocationCount: rowCount,
        declarations,
        body: INEQUALITY_ZONE_KEYS_BODY
      })
    );
    // Stable radix keeps the value order inside each zone. Keys span [0, zoneCount].
    const zoneKeyBits = Math.max(1, Math.ceil(Math.log2(zoneCount + 1)));
    nodes.push(
      ...new GPUSort({
        id: `${id}-zone-sort`,
        keys: zoneKeys,
        values: sortedRows,
        outputKeys: sortedZoneKeys,
        outputValues: zoneSortedRows,
        algorithm: 'radix',
        keyBits: Math.min(32, zoneKeyBits)
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-zone-stats`,
        operation: OPERATION,
        variant: 'zone-stats',
        bindings: [
          read('sortedZones', sortedZoneKeys, 'u32'),
          read('sortedRows', zoneSortedRows, 'u32'),
          read('values', values, 'f32'),
          ...(weights ? [read('weights', weights, 'f32')] : []),
          read('params', parameters, 'f32'),
          write('stats', stats, 'f32'),
          write('lorenz', lorenz, 'f32'),
          write('counts', counts, 'u32')
        ],
        invocationCount: zoneCount,
        declarations,
        body: getInequalityZoneStatsBody(weightExpression)
      })
    );

    const publish = (
      step: string,
      entries: [string, number, GraphDataView<'float32'> | undefined][]
    ) => {
      const present = entries.filter(([, , view]) => view);
      if (present.length === 0) {
        return;
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${step}`,
          operation: OPERATION,
          variant: step,
          bindings: [
            read('stats', stats, 'f32'),
            ...present.map(([name, , view]) => write(name, view!, 'f32'))
          ],
          invocationCount: zoneCount,
          declarations,
          body: present
            .map(
              ([name, slot]) =>
                `${name}[${name}Offset + index] = stats[statsOffset + index * STATS_STRIDE + ${slot}u];`
            )
            .join('\n  ')
        })
      );
    };
    publish('publish-indices', [
      ['gini', INEQUALITY_STAT.GINI, output.gini],
      ['theilT', INEQUALITY_STAT.THEIL_T, output.theilT],
      ['theilL', INEQUALITY_STAT.THEIL_L, output.theilL],
      ['atkinson', INEQUALITY_STAT.ATKINSON, output.atkinson]
    ]);
    publish('publish-moments', [
      ['hoover', INEQUALITY_STAT.HOOVER, output.hoover],
      ['palma', INEQUALITY_STAT.PALMA, output.palma],
      ['mean', INEQUALITY_STAT.MEAN, output.mean]
    ]);

    if (output.globalSummary) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-global-sums`,
          operation: OPERATION,
          variant: 'global-sums',
          bindings: [
            read('stats', stats, 'f32'),
            read('counts', counts, 'u32'),
            write('summary', output.globalSummary, 'f32')
          ],
          invocationCount: 1,
          declarations,
          body: INEQUALITY_GLOBAL_SUMS_BODY
        })
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-global-gini`,
          operation: OPERATION,
          variant: 'global-gini',
          bindings: [
            read('sortedValueKeys', sortedValueKeys, 'u32'),
            read('sortedRows', sortedRows, 'u32'),
            read('values', values, 'f32'),
            ...(weights ? [read('weights', weights, 'f32')] : []),
            write('summary', output.globalSummary, 'f32')
          ],
          invocationCount: 1,
          declarations,
          body: getInequalityGlobalGiniBody(weightExpression)
        })
      );
    }
    return nodes;
  }
}
