// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUVisibilityWorkflow,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  createPublishNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {
  getGPUAttributeCrossfilterHistogramLayout,
  getGPUAttributeCrossfilterParameterLength,
  GPU_ATTRIBUTE_CROSSFILTER_BRUSH_ENABLED_OFFSET,
  GPU_ATTRIBUTE_CROSSFILTER_BRUSH_MAX_OFFSET,
  GPU_ATTRIBUTE_CROSSFILTER_BRUSH_MIN_OFFSET,
  GPU_ATTRIBUTE_CROSSFILTER_DOMAIN_MAX_OFFSET,
  GPU_ATTRIBUTE_CROSSFILTER_DOMAIN_MIN_OFFSET,
  GPU_ATTRIBUTE_CROSSFILTER_PARAMETER_STRIDE
} from './attribute-crossfilter-parameters';

/** Maximum number of crossfilter dimensions: one fail bit each in a `u32` per row. */
export const GPU_ATTRIBUTE_CROSSFILTER_MAXIMUM_DIMENSION_COUNT = 32;
/** Maximum bins per dimension: the per-workgroup histogram must fit 16 KiB of workgroup memory. */
export const GPU_ATTRIBUTE_CROSSFILTER_MAXIMUM_BIN_COUNT = 4096;

const WORKGROUP_SIZE = 256;
/** Columns fused into one fail-bit pass: 8 bindings minus the parameters and failBits bindings. */
const FAIL_PASS_COLUMN_COUNT = 6;
const FINITE_EXPONENT_MASK = '0x7f800000u';

/** One crossfilter dimension: a numeric column and its histogram shape. */
export type GPUAttributeCrossfilterDimension = {
  /**
   * Packed float32 or uint32 column with one row per table row. Each dimension owns its buffer;
   * columns are never packed together. `uint32` values are binned as numbers after conversion to
   * f32, so values above 2^24 lose precision. Non-finite floats (NaN, +-Infinity) count as
   * missing: such a row fails this dimension.
   */
  column: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Histogram bins, fixed at compile time: an integer in `[1, 4096]`. */
  binCount: number;
  /**
   * Where the histogram domain `[min, max]` comes from. `'auto'` (default) is the min/max of live,
   * finite rows, recomputed on the GPU on every encoding and independent of all brushes.
   * `'parameters'` reads the domain from the parameter view.
   */
  domain?: 'auto' | 'parameters';
};

/**
 * Properties for {@link GPUAttributeCrossfilter}.
 *
 * Per-frame (no recompile): the contents of `parameters` (brushes, enabled flags, parameter
 * domains), every column, and `liveMask`. Topology (needs a new graph): row count, dimension count,
 * bin counts, domain modes, and which optional outputs, `sourceIds` and `liveMask` are present.
 */
export type GPUAttributeCrossfilterProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'attribute-crossfilter'`. */
  id?: string;
  /** One to 32 dimensions. All columns must have the same row count. */
  dimensions: readonly GPUAttributeCrossfilterDimension[];
  /**
   * Optional per-row liveness, nonzero = live, for example a time-window or residency mask. Dead
   * rows never enter a domain, histogram, count, selection, or compact output.
   */
  liveMask?: GraphDataView<'uint32'>;
  /**
   * Caller-owned per-frame float32 storage view. Dimension `d` reads
   * {@link GPU_ATTRIBUTE_CROSSFILTER_PARAMETER_STRIDE} floats from `d * stride`:
   * `[brushMin, brushMax, brushEnabled, domainMin, domainMax, 0, 0, 0]`. Pack it with
   * `getGPUAttributeCrossfilterParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Dimension-major `uint32` histograms with `sum(binCount)` rows; see
   * `getGPUAttributeCrossfilterHistogramLayout`. Dimension `d` counts rows that are live, finite in
   * `d`, and pass every other dimension's brush (not its own). Rewritten on every encoding.
   * Values outside a parameter domain are not counted; the domain maximum lands in the last bin.
   */
  histograms: GraphDataView<'uint32'>;
  /** Optional `float32` rows, `[min, max]` per dimension: the domain actually used. `[0, 0]` when no live row. */
  domains?: GraphDataView<'float32'>;
  /** Optional one-row `uint32` count of rows that are live and pass every brush. */
  selectedCount?: GraphDataView<'uint32'>;
  /** Optional one-row `uint32` count of live rows. */
  liveCount?: GraphDataView<'uint32'>;
  /** Optional per-row `uint32` canonical mask, 1 when the row is live and passes every brush. */
  selection?: GraphDataView<'uint32'>;
  /** Optional stable IDs aligned with rows. Zero-based row indices are emitted when omitted. */
  sourceIds?: GraphDataView<'uint32'>;
  /** Optional caller-owned bounded compact list of selected row IDs. `output.ids.length` is the capacity. */
  output?: GPUCompactOutput;
};

function getBit(dimensionIndex: number): string {
  return `${(2 ** dimensionIndex) >>> 0}u`;
}

/** WGSL statements loading `value` (f32) and `finite` (bool) from column binding `name` at `index`. */
function getLoadSource(name: string, isFloat: boolean, suffix: string): string {
  return isFloat
    ? `let raw${suffix} = ${name}[${name}Offset + index];
  let finite${suffix} = (raw${suffix} & ${FINITE_EXPONENT_MASK}) != ${FINITE_EXPONENT_MASK};
  let value${suffix} = select(0.0, bitcast<f32>(raw${suffix}), finite${suffix});`
    : `let finite${suffix} = true;
  let value${suffix} = f32(${name}[${name}Offset + index]);`;
}

const ORDERED_KEY_DECLARATIONS = /* wgsl */ `
fn encodeOrderedKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, (bits >> 31u) == 1u);
}
fn decodeOrderedKey(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key & 0x7fffffffu, (key >> 31u) == 1u));
}`;

/**
 * Linked histograms over numeric attribute columns with per-frame brushes (classic crossfilter).
 *
 * One recipe instance covers one table of vertices or edges. Each dimension's histogram counts
 * rows that are live and pass every OTHER dimension's brush, so brushing one chart updates the
 * others and never its own. Everything stays on the GPU and nothing recompiles when brushes,
 * columns, or the live mask change.
 *
 * Passes (all linear in rows, within the 8-storage-binding limit, no packed multi-column buffer):
 * 1. init: zero histograms and counts, seed auto-domain keys;
 * 2. per `'auto'` dimension: live-row min/max with order-preserving `u32` atomic keys, reduced per
 *    workgroup first;
 * 3. domain resolve: write the domain used by each dimension;
 * 4. fail bits: up to six columns per pass OR one bit per dimension into a per-row `u32`;
 * 5. per dimension: workgroup-local histogram over rows with `(failBits & ~ownBit) == 0`;
 * 6. optional selection mask and counts, then optional compaction of selected row IDs.
 *
 * Semantics: brushes are inclusive-minimum and exclusive-maximum so adjacent brushes partition a
 * range; pass `Infinity` as the maximum to include the upper end. Non-finite floats fail their
 * own dimension even when its brush is disabled, so rows with missing values are excluded from
 * every other dimension's histogram and from the selection. Bins use f32 math:
 * `bin = min(floor((v - min) * (bins / (max - min))), bins - 1)`.
 */
export class GPUAttributeCrossfilter implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUAttributeCrossfilterProps;

  constructor(props: GPUAttributeCrossfilterProps) {
    this.id = props.id ?? 'attribute-crossfilter';
    this.props = props;
    const id = this.id;
    const {dimensions} = props;
    if (
      dimensions.length < 1 ||
      dimensions.length > GPU_ATTRIBUTE_CROSSFILTER_MAXIMUM_DIMENSION_COUNT
    ) {
      throw new Error(
        `${id} needs 1 to ${GPU_ATTRIBUTE_CROSSFILTER_MAXIMUM_DIMENSION_COUNT} dimensions`
      );
    }
    const rows = dimensions[0].column.length;
    for (const [dimensionIndex, dimension] of dimensions.entries()) {
      validatePackedView(
        dimension.column,
        ['float32', 'uint32'],
        `${id} dimensions[${dimensionIndex}].column`
      );
      if (dimension.column.length !== rows) {
        throw new Error(`${id} dimensions must have equal row counts`);
      }
      if (
        !Number.isInteger(dimension.binCount) ||
        dimension.binCount < 1 ||
        dimension.binCount > GPU_ATTRIBUTE_CROSSFILTER_MAXIMUM_BIN_COUNT
      ) {
        throw new Error(
          `${id} dimensions[${dimensionIndex}].binCount must be an integer in [1, ${GPU_ATTRIBUTE_CROSSFILTER_MAXIMUM_BIN_COUNT}]`
        );
      }
      if (dimension.domain && dimension.domain !== 'auto' && dimension.domain !== 'parameters') {
        throw new Error(
          `${id} dimensions[${dimensionIndex}].domain must be 'auto' or 'parameters'`
        );
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    const parameterLength = getGPUAttributeCrossfilterParameterLength(dimensions.length);
    if (props.parameters.length < parameterLength) {
      throw new Error(`${id} parameters must hold ${parameterLength} float32 values`);
    }
    for (const [name, view] of [
      ['liveMask', props.liveMask],
      ['selection', props.selection],
      ['sourceIds', props.sourceIds]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length !== rows) {
          throw new Error(`${id} ${name} length must equal the column row count`);
        }
      }
    }
    const {totalBinCount} = getGPUAttributeCrossfilterHistogramLayout(
      dimensions.map(dimension => dimension.binCount)
    );
    validatePackedUint32View(props.histograms, `${id} histograms`);
    if (props.histograms.length < totalBinCount) {
      throw new Error(`${id} histograms must hold ${totalBinCount} uint32 rows`);
    }
    if (props.domains) {
      validatePackedView(props.domains, ['float32'], `${id} domains`);
      if (props.domains.length < 2 * dimensions.length) {
        throw new Error(`${id} domains must hold ${2 * dimensions.length} float32 rows`);
      }
    }
    for (const [name, view] of [
      ['selectedCount', props.selectedCount],
      ['liveCount', props.liveCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must contain one uint32 row`);
        }
      }
    }
    if (props.output) {
      validateCompactOutput(id, props.output);
    }
    const outputs = [
      props.histograms,
      props.domains,
      props.selectedCount,
      props.liveCount,
      props.selection,
      props.output?.ids,
      props.output?.count,
      props.output?.overflow,
      props.output?.totalCount
    ];
    validateGraphOutputsDisjointFromInputs(id, outputs, [
      ...dimensions.map(dimension => dimension.column),
      props.liveMask,
      props.parameters,
      props.sourceIds
    ]);
    const outputBuffers = outputs.filter(Boolean).map(view => view!.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must not share buffers with each other`);
    }
  }

  /** Returns init, domain, fail-bit, histogram, selection, and compaction nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {dimensions, liveMask, parameters, histograms, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...dimensions.map(dimension => dimension.column),
      liveMask,
      parameters,
      histograms,
      props.domains,
      props.selectedCount,
      props.liveCount,
      props.selection,
      props.sourceIds,
      output?.ids,
      output?.count,
      output?.overflow,
      output?.totalCount
    ]);
    const rows = dimensions[0].column.length;
    const dimensionCount = dimensions.length;
    const layout = getGPUAttributeCrossfilterHistogramLayout(
      dimensions.map(dimension => dimension.binCount)
    );
    const isAuto = dimensions.map(dimension => (dimension.domain ?? 'auto') === 'auto');
    const hasAuto = isAuto.some(Boolean);
    const autoMask = isAuto.reduce((mask, auto, index) => (auto ? mask | (1 << index) : mask), 0);
    const keys = hasAuto
      ? createTransientView(graph, `${id}-domain-keys`, 'uint32', 2 * dimensionCount)
      : undefined;
    const domains =
      props.domains ?? createTransientView(graph, `${id}-domains`, 'float32', 2 * dimensionCount);
    const failBits = createTransientView(graph, `${id}-fail-bits`, 'uint32', rows);
    const needsRowPass = Boolean(
      props.selection || props.selectedCount || props.liveCount || output
    );
    const selection =
      props.selection ??
      (output ? createTransientView(graph, `${id}-selection`, 'uint32', rows) : undefined);
    const liveSource = liveMask ? 'liveMask[liveMaskOffset + index] != 0u' : 'true';
    const nodes: GPUCommandNode<Parameters>[] = [];

    // 1. init
    {
      const bindings: WGSLKernelBinding[] = [
        {
          name: 'histogramOut',
          view: histograms,
          type: 'u32',
          access: 'read_write'
        }
      ];
      if (keys)
        bindings.push({
          name: 'keysOut',
          view: keys,
          type: 'u32',
          access: 'read_write'
        });
      if (props.selectedCount) {
        bindings.push({
          name: 'selectedOut',
          view: props.selectedCount,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (props.liveCount) {
        bindings.push({
          name: 'liveOut',
          view: props.liveCount,
          type: 'u32',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-init`,
          operation: 'GPUAttributeCrossfilter',
          variant: 'init',
          bindings,
          invocationCount: Math.max(layout.totalBinCount, 2 * dimensionCount),
          body: `if (index < ${layout.totalBinCount}u) { histogramOut[histogramOutOffset + index] = 0u; }
  ${keys ? `if (index < ${2 * dimensionCount}u) { keysOut[keysOutOffset + index] = select(0xffffffffu, 0u, (index & 1u) == 1u); }` : ''}
  if (index == 0u) {
    ${props.selectedCount ? 'selectedOut[selectedOutOffset] = 0u;' : ''}
    ${props.liveCount ? 'liveOut[liveOutOffset] = 0u;' : ''}
  }`
        })
      );
    }

    // 2. auto domains
    for (const [dimensionIndex, dimension] of dimensions.entries()) {
      if (!isAuto[dimensionIndex]) continue;
      const isFloat = dimension.column.format === 'float32';
      const bindings: WGSLKernelBinding[] = [
        {name: 'column', view: dimension.column, type: 'u32', access: 'read'}
      ];
      if (liveMask)
        bindings.push({
          name: 'liveMask',
          view: liveMask,
          type: 'u32',
          access: 'read'
        });
      bindings.push({
        name: 'keys',
        view: keys!,
        type: 'atomic<u32>',
        access: 'read_write'
      });
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-domain-${dimensionIndex}`,
          operation: 'GPUAttributeCrossfilter',
          variant: 'domain',
          bindings,
          invocationCount: rows,
          guardIndex: false,
          declarations: `${ORDERED_KEY_DECLARATIONS}
var<workgroup> workgroupMinimum: atomic<u32>;
var<workgroup> workgroupMaximum: atomic<u32>;`,
          body: `if (localInvocationIndex == 0u) {
    atomicStore(&workgroupMinimum, 0xffffffffu);
    atomicStore(&workgroupMaximum, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    ${getLoadSource('column', isFloat, '')}
    if (finite && ${liveSource}) {
      let key = encodeOrderedKey(value);
      atomicMin(&workgroupMinimum, key);
      atomicMax(&workgroupMaximum, key);
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let minimumKey = atomicLoad(&workgroupMinimum);
    let maximumKey = atomicLoad(&workgroupMaximum);
    if (minimumKey <= maximumKey) {
      atomicMin(&keys[keysOffset + ${2 * dimensionIndex}u], minimumKey);
      atomicMax(&keys[keysOffset + ${2 * dimensionIndex + 1}u], maximumKey);
    }
  }`
        })
      );
    }

    // 3. resolve domains
    {
      const bindings: WGSLKernelBinding[] = [
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {
          name: 'domainsOut',
          view: domains,
          type: 'f32',
          access: 'read_write'
        }
      ];
      if (keys)
        bindings.push({
          name: 'keys',
          view: keys,
          type: 'u32',
          access: 'read'
        });
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-resolve-domains`,
          operation: 'GPUAttributeCrossfilter',
          variant: 'resolve-domains',
          bindings,
          invocationCount: dimensionCount,
          declarations: ORDERED_KEY_DECLARATIONS,
          body: `let parameterBase = parametersOffset + index * ${GPU_ATTRIBUTE_CROSSFILTER_PARAMETER_STRIDE}u;
  var minimum = parameters[parameterBase + ${GPU_ATTRIBUTE_CROSSFILTER_DOMAIN_MIN_OFFSET}u];
  var maximum = parameters[parameterBase + ${GPU_ATTRIBUTE_CROSSFILTER_DOMAIN_MAX_OFFSET}u];
  ${
    keys
      ? `if (((${autoMask >>> 0}u >> index) & 1u) == 1u) {
    let minimumKey = keys[keysOffset + 2u * index];
    let maximumKey = keys[keysOffset + 2u * index + 1u];
    if (minimumKey > maximumKey) {
      minimum = 0.0;
      maximum = 0.0;
    } else {
      minimum = decodeOrderedKey(minimumKey);
      maximum = decodeOrderedKey(maximumKey);
    }
  }`
      : ''
  }
  domainsOut[domainsOutOffset + 2u * index] = minimum;
  domainsOut[domainsOutOffset + 2u * index + 1u] = maximum;`
        })
      );
    }

    // 4. fail bits, FAIL_PASS_COLUMN_COUNT columns per pass
    for (let first = 0; first < dimensionCount; first += FAIL_PASS_COLUMN_COUNT) {
      const group = dimensions.slice(first, first + FAIL_PASS_COLUMN_COUNT);
      const bindings: WGSLKernelBinding[] = [
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'failBits', view: failBits, type: 'u32', access: 'read_write'},
        ...group.map(
          (dimension, offset): WGSLKernelBinding => ({
            name: `column${first + offset}`,
            view: dimension.column,
            type: 'u32',
            access: 'read'
          })
        )
      ];
      const statements = group.map((dimension, offset) => {
        const dimensionIndex = first + offset;
        const base = `parametersOffset + ${dimensionIndex * GPU_ATTRIBUTE_CROSSFILTER_PARAMETER_STRIDE}u`;
        return `{
    ${getLoadSource(`column${dimensionIndex}`, dimension.column.format === 'float32', '')}
    let brushMinimum = parameters[${base} + ${GPU_ATTRIBUTE_CROSSFILTER_BRUSH_MIN_OFFSET}u];
    let brushMaximum = parameters[${base} + ${GPU_ATTRIBUTE_CROSSFILTER_BRUSH_MAX_OFFSET}u];
    let brushEnabled = parameters[${base} + ${GPU_ATTRIBUTE_CROSSFILTER_BRUSH_ENABLED_OFFSET}u] != 0.0;
    let fails = !finite || (brushEnabled && !(value >= brushMinimum && value < brushMaximum));
    bits |= select(0u, ${getBit(dimensionIndex)}, fails);
  }`;
      });
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-fail-${first}`,
          operation: 'GPUAttributeCrossfilter',
          variant: 'fail-bits',
          bindings,
          invocationCount: rows,
          body: `var bits = ${first === 0 ? '0u' : 'failBits[failBitsOffset + index]'};
  ${statements.join('\n  ')}
  failBits[failBitsOffset + index] = bits;`
        })
      );
    }

    // 5. histograms
    for (const [dimensionIndex, dimension] of dimensions.entries()) {
      const isFloat = dimension.column.format === 'float32';
      const binCount = dimension.binCount;
      const bindings: WGSLKernelBinding[] = [
        {name: 'column', view: dimension.column, type: 'u32', access: 'read'},
        {name: 'domains', view: domains, type: 'f32', access: 'read'},
        {name: 'failBits', view: failBits, type: 'u32', access: 'read'}
      ];
      if (liveMask)
        bindings.push({
          name: 'liveMask',
          view: liveMask,
          type: 'u32',
          access: 'read'
        });
      bindings.push({
        name: 'histogramOut',
        view: histograms,
        type: 'atomic<u32>',
        access: 'read_write'
      });
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-histogram-${dimensionIndex}`,
          operation: 'GPUAttributeCrossfilter',
          variant: 'histogram',
          bindings,
          invocationCount: rows,
          guardIndex: false,
          declarations: `const BIN_COUNT: u32 = ${binCount}u;
const HISTOGRAM_OFFSET: u32 = ${layout.offsets[dimensionIndex]}u;
var<workgroup> localBins: array<atomic<u32>, ${binCount}>;`,
          body: `for (var bin = localInvocationIndex; bin < BIN_COUNT; bin += ${WORKGROUP_SIZE}u) {
    atomicStore(&localBins[bin], 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    ${getLoadSource('column', isFloat, '')}
    if (finite && ${liveSource} && (failBits[failBitsOffset + index] & ${(~(2 ** dimensionIndex) >>> 0).toString()}u) == 0u) {
      let minimum = domains[domainsOffset + ${2 * dimensionIndex}u];
      let maximum = domains[domainsOffset + ${2 * dimensionIndex + 1}u];
      if (value >= minimum && value <= maximum) {
        var bin = 0u;
        if (maximum > minimum) {
          let scaled = (value - minimum) * (f32(BIN_COUNT) / (maximum - minimum));
          bin = u32(min(floor(scaled), f32(BIN_COUNT - 1u)));
        }
        atomicAdd(&localBins[bin], 1u);
      }
    }
  }
  workgroupBarrier();
  for (var bin = localInvocationIndex; bin < BIN_COUNT; bin += ${WORKGROUP_SIZE}u) {
    let count = atomicLoad(&localBins[bin]);
    if (count > 0u) {
      atomicAdd(&histogramOut[histogramOutOffset + HISTOGRAM_OFFSET + bin], count);
    }
  }`
        })
      );
    }

    // 6. selection mask and counts
    if (needsRowPass) {
      const bindings: WGSLKernelBinding[] = [
        {name: 'failBits', view: failBits, type: 'u32', access: 'read'}
      ];
      if (liveMask)
        bindings.push({
          name: 'liveMask',
          view: liveMask,
          type: 'u32',
          access: 'read'
        });
      if (selection) {
        bindings.push({
          name: 'selectionOut',
          view: selection,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (props.selectedCount) {
        bindings.push({
          name: 'selectedOut',
          view: props.selectedCount,
          type: 'atomic<u32>',
          access: 'read_write'
        });
      }
      if (props.liveCount) {
        bindings.push({
          name: 'liveOut',
          view: props.liveCount,
          type: 'atomic<u32>',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-selection`,
          operation: 'GPUAttributeCrossfilter',
          variant: 'selection',
          bindings,
          invocationCount: rows,
          guardIndex: false,
          declarations: `var<workgroup> workgroupSelected: atomic<u32>;
var<workgroup> workgroupLive: atomic<u32>;`,
          body: `if (localInvocationIndex == 0u) {
    atomicStore(&workgroupSelected, 0u);
    atomicStore(&workgroupLive, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let live = ${liveSource};
    let selected = live && failBits[failBitsOffset + index] == 0u;
    ${selection ? 'selectionOut[selectionOutOffset + index] = select(0u, 1u, selected);' : ''}
    if (live) { atomicAdd(&workgroupLive, 1u); }
    if (selected) { atomicAdd(&workgroupSelected, 1u); }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    ${props.selectedCount ? 'let selectedTotal = atomicLoad(&workgroupSelected); if (selectedTotal > 0u) { atomicAdd(&selectedOut[selectedOutOffset], selectedTotal); }' : ''}
    ${props.liveCount ? 'let liveTotal = atomicLoad(&workgroupLive); if (liveTotal > 0u) { atomicAdd(&liveOut[liveOutOffset], liveTotal); }' : ''}
  }`
        })
      );
    }

    // 7. compaction of selected rows
    if (output && selection) {
      const direct = rows > 0 && output.ids.length >= rows;
      const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
      const compactIds = direct
        ? output.ids
        : createTransientView(graph, `${id}-compact-ids`, 'uint32', rows);
      nodes.push(
        ...new GPUVisibilityWorkflow({
          id: `${id}-visibility`,
          predicates: [{kind: 'selection', mask: selection}],
          output: compactIds,
          count: total,
          sourceIds: props.sourceIds
        }).getCommandNodes(graph)
      );
      nodes.push(
        createPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: 'GPUAttributeCrossfilter',
          totalCount: total,
          compactIds: direct ? undefined : compactIds,
          output
        })
      );
    }
    return nodes;
  }
}
