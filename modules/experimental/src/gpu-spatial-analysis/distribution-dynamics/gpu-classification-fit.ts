// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';

const OPERATION = 'GPUClassificationFit';
const MAXIMUM_CLASS_COUNT = 256;

/** Rows of the optional `summary` output of {@link GPUClassificationFit}. */
export const GPU_CLASSIFICATION_FIT_SUMMARY_LENGTH = 5;
/** Row of `summary` holding GADF, `1 - ADCM / ADAM`. */
export const GPU_CLASSIFICATION_FIT_GADF = 0;
/** Row of `summary` holding ADCM, the summed absolute deviation around class medians. */
export const GPU_CLASSIFICATION_FIT_ADCM = 1;
/** Row of `summary` holding ADAM, the summed absolute deviation around the overall median. */
export const GPU_CLASSIFICATION_FIT_ADAM = 2;
/** Row of `summary` holding GVF, `1 - SDCM / total sum of squares` (not in mapclassify). */
export const GPU_CLASSIFICATION_FIT_GVF = 3;
/**
 * Row of `summary` holding mapclassify's `Classifier.tss`: despite the name, the summed squared
 * deviation of each value from its own class mean (SDCM).
 */
export const GPU_CLASSIFICATION_FIT_TSS = 4;

/**
 * Caller-owned outputs of {@link GPUClassificationFit}. Every per-class view has `classCount + 1`
 * rows: rows `0..k-1` describe each class, and the last row (`k`) describes all counted rows
 * together (the "overall" statistics: ADAM, TSS).
 */
export type GPUClassificationFitOutput = {
  /** Counted rows per class (last row: all counted rows). */
  counts: GraphDataView<'uint32'>;
  /** Exact median per class, numpy convention (mean of the two middle values). 0 for an empty class. */
  medians: GraphDataView<'float32'>;
  /** Sum of absolute deviations from the class median (last row: ADAM). */
  absoluteDeviations: GraphDataView<'float32'>;
  /** Sum of squared deviations from the class mean (last row: total sum of squares about the overall mean). */
  squaredDeviations: GraphDataView<'float32'>;
  /** Optional five rows: GADF, ADCM, ADAM, GVF, TSS (see the `GPU_CLASSIFICATION_FIT_*` row indices). */
  summary?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUClassificationFit}.
 *
 * Topology: `classCount`, view lengths and which optional views exist. Per-frame: view contents.
 */
export type GPUClassificationFitProps = {
  /** Prefix for generated node IDs. Defaults to `'classification-fit'`. */
  id?: string;
  /** Packed float32 values that were classified. */
  values: GraphDataView<'float32'>;
  /**
   * Class per value (same length as `values`), `0..classCount-1`, for example the output of
   * `GPUClassAssignment`. Rows with a class at or above `classCount` (such as the no-class value
   * of masked or NaN rows) are not counted.
   */
  classes: GraphDataView<'uint32'>;
  /** Number of classes `k`, 1 to 256. */
  classCount: number;
  /** Optional row selection (same length as `values`): zero rows are not counted. */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUClassificationFitOutput;
};

/**
 * Goodness of fit of a classification, matching mapclassify 2.11 `Classifier.adcm` /
 * `get_adcm()`, `get_gadf()` and `get_tss()`:
 *
 * - `ADCM = sum over classes of sum |x - median(class)|`
 * - `ADAM = sum |x - median(all)|`
 * - `GADF = 1 - ADCM / ADAM`
 * - `TSS` (mapclassify's name) `= sum over classes of sum (x - mean(class))^2`, which is the
 *   within-class sum of squares (SDCM), not a total sum of squares
 * - `GVF = 1 - TSS / sum (x - mean(all))^2` (extension; mapclassify 2.11 has no `gvf`)
 *
 * Confirmed from the mapclassify 2.11.0 source: medians use `np.median` (mean of the two middle
 * values for an even count); empty classes are skipped and add 0 (here: count 0, median 0, all
 * deviations 0); `get_gadf()` returns 1 when ADAM is 0 (a constant column), as this class does.
 * The same convention is used for GVF when its denominator is 0.
 *
 * Method: a stable sort by value key, then a stable sort by class, leaves every class as a
 * contiguous ascending-value segment (plus the value-ordered list of all counted rows for the
 * overall statistics). Each class invocation binary-searches its segment bounds, reads the exact
 * medians at their positions and sums its segment in sorted order, so the work is two radix sorts
 * plus one pass over the values in total, with no atomics and bit-reproducible results. Rows are
 * counted only when they have a class below `classCount`, pass the mask, and hold a finite value
 * (NaN and infinities are skipped).
 *
 * Deviations from mapclassify: sums and the median average are float32 (and accumulate in sorted
 * order rather than index order), so results agree to float32 precision, not bitwise. Requires at
 * least one row.
 */
export class GPUClassificationFit implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUClassificationFitProps;

  constructor(props: GPUClassificationFitProps) {
    const id = props.id ?? 'classification-fit';
    this.id = id;
    this.props = props;
    const {values, classes, classCount, mask, output} = props;
    validatePackedView(values, ['float32'], `${id} values`);
    validatePackedUint32View(classes, `${id} classes`);
    if (!Number.isInteger(classCount) || classCount < 1 || classCount > MAXIMUM_CLASS_COUNT) {
      throw new Error(`${id} classCount must be an integer from 1 to ${MAXIMUM_CLASS_COUNT}`);
    }
    if (values.length < 1) {
      throw new Error(`${id} values must hold at least one row`);
    }
    if (classes.length !== values.length) {
      throw new Error(`${id} classes length must equal values length`);
    }
    if (mask) {
      validatePackedUint32View(mask, `${id} mask`);
      if (mask.length !== values.length) {
        throw new Error(`${id} mask length must equal values length`);
      }
    }
    const slots = classCount + 1;
    validatePackedUint32View(output.counts, `${id} counts`);
    validatePackedView(output.medians, ['float32'], `${id} medians`);
    validatePackedView(output.absoluteDeviations, ['float32'], `${id} absoluteDeviations`);
    validatePackedView(output.squaredDeviations, ['float32'], `${id} squaredDeviations`);
    for (const [name, view] of [
      ['counts', output.counts],
      ['medians', output.medians],
      ['absoluteDeviations', output.absoluteDeviations],
      ['squaredDeviations', output.squaredDeviations]
    ] as const) {
      if (view.length !== slots) {
        throw new Error(`${id} ${name} must hold classCount + 1 (${slots}) rows`);
      }
    }
    if (output.summary) {
      validatePackedView(output.summary, ['float32'], `${id} summary`);
      if (output.summary.length !== GPU_CLASSIFICATION_FIT_SUMMARY_LENGTH) {
        throw new Error(`${id} summary must hold ${GPU_CLASSIFICATION_FIT_SUMMARY_LENGTH} rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.counts,
        output.medians,
        output.absoluteDeviations,
        output.squaredDeviations,
        output.summary
      ],
      [values, classes, mask]
    );
  }

  /**
   * Returns the key-build, value sort, class-key, class sort, `stats` and, when `summary` is
   * requested, `summary` nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {values, classes, classCount, mask, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      values,
      classes,
      mask,
      output.counts,
      output.medians,
      output.absoluteDeviations,
      output.squaredDeviations,
      output.summary
    ]);
    const rows = values.length;
    const valueKeys = createTransientView(graph, `${id}-value-keys`, 'uint32', rows);
    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rows);
    const sortedValueKeys = createTransientView(graph, `${id}-sorted-value-keys`, 'uint32', rows);
    const rowsByValue = createTransientView(graph, `${id}-rows-by-value`, 'uint32', rows);
    const classKeys = createTransientView(graph, `${id}-class-keys`, 'uint32', rows);
    const sortedClassKeys = createTransientView(graph, `${id}-sorted-class-keys`, 'uint32', rows);
    const rowsByClass = createTransientView(graph, `${id}-rows-by-class`, 'uint32', rows);
    const constants = `const ROWS: u32 = ${rows}u;
const CLASSES: u32 = ${classCount}u;
const UNCOUNTED_KEY: u32 = 0xffffffffu;`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-keys`,
        operation: OPERATION,
        variant: 'keys',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'classes', view: classes, type: 'u32', access: 'read'},
          ...(mask
            ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'valueKeys', view: valueKeys, type: 'u32', access: 'read_write'},
          {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: constants,
        body: `let value = values[valuesOffset + index];
  let bits = bitcast<u32>(value);
  // WGSL gives no NaN guarantees to comparisons: counted values have a non-all-ones exponent.
  let counted = (bits & 0x7f800000u) != 0x7f800000u && classes[classesOffset + index] < CLASSES
    && ${mask ? 'mask[maskOffset + index] != 0u' : 'true'};
  // Order-preserving key; the all-ones key (above every finite value) marks uncounted rows.
  let key = select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
  valueKeys[valueKeysOffset + index] = select(UNCOUNTED_KEY, key, counted);
  rowIds[rowIdsOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${id}-value-sort`,
        keys: valueKeys,
        values: rowIds,
        outputKeys: sortedValueKeys,
        outputValues: rowsByValue
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-class-keys`,
        operation: OPERATION,
        variant: 'class-keys',
        bindings: [
          {name: 'sortedValueKeys', view: sortedValueKeys, type: 'u32', access: 'read'},
          {name: 'rowsByValue', view: rowsByValue, type: 'u32', access: 'read'},
          {name: 'classes', view: classes, type: 'u32', access: 'read'},
          {name: 'classKeys', view: classKeys, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: constants,
        // Uncounted rows take the key CLASSES, which sorts after every class.
        body: `let counted = sortedValueKeys[sortedValueKeysOffset + index] != UNCOUNTED_KEY;
  classKeys[classKeysOffset + index] =
    select(CLASSES, classes[classesOffset + rowsByValue[rowsByValueOffset + index]], counted);`
      }),
      // Stable, so rows stay in ascending value order inside each class.
      ...new GPUSort({
        id: `${id}-class-sort`,
        keys: classKeys,
        values: rowsByValue,
        outputKeys: sortedClassKeys,
        outputValues: rowsByClass,
        keyBits: Math.max(1, Math.ceil(Math.log2(classCount + 1)))
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-stats`,
        operation: OPERATION,
        variant: 'stats',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'sortedClassKeys', view: sortedClassKeys, type: 'u32', access: 'read'},
          {name: 'rowsByValue', view: rowsByValue, type: 'u32', access: 'read'},
          {name: 'rowsByClass', view: rowsByClass, type: 'u32', access: 'read'},
          {name: 'counts', view: output.counts, type: 'u32', access: 'read_write'},
          {name: 'medians', view: output.medians, type: 'f32', access: 'read_write'},
          {
            name: 'absoluteDeviations',
            view: output.absoluteDeviations,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'squaredDeviations',
            view: output.squaredDeviations,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: classCount + 1,
        declarations: `${constants}

// First position in sortedClassKeys whose key is at least bound.
fn lowerBound(bound: u32) -> u32 {
  var low = 0u;
  var high = ROWS;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedClassKeys[sortedClassKeysOffset + middle] < bound) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}

fn valueAt(slot: u32, position: u32) -> f32 {
  // Slot CLASSES (all counted rows) reads the value-ordered rows, a class slot its class segment.
  let row = select(
    rowsByClass[rowsByClassOffset + position],
    rowsByValue[rowsByValueOffset + position],
    slot == CLASSES
  );
  return values[valuesOffset + row];
}`,
        body: `let slot = index;
  let start = select(lowerBound(slot), 0u, slot == CLASSES);
  let end = lowerBound(select(slot + 1u, CLASSES, slot == CLASSES));
  let count = end - start;
  counts[countsOffset + slot] = count;
  var median = 0.0;
  var absolute = 0.0;
  var squared = 0.0;
  if (count > 0u) {
    let lower = valueAt(slot, start + (count - 1u) / 2u);
    let upper = valueAt(slot, start + count / 2u);
    median = lower + (upper - lower) * 0.5;
    var sum = 0.0;
    for (var position = start; position < end; position++) {
      sum += valueAt(slot, position);
    }
    let mean = sum / f32(count);
    for (var position = start; position < end; position++) {
      let value = valueAt(slot, position);
      absolute += abs(value - median);
      let centered = value - mean;
      squared += centered * centered;
    }
  }
  medians[mediansOffset + slot] = median;
  absoluteDeviations[absoluteDeviationsOffset + slot] = absolute;
  squaredDeviations[squaredDeviationsOffset + slot] = squared;`
      })
    ];
    if (output.summary) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-summary`,
          operation: OPERATION,
          variant: 'summary',
          bindings: [
            {
              name: 'absoluteDeviations',
              view: output.absoluteDeviations,
              type: 'f32',
              access: 'read'
            },
            {
              name: 'squaredDeviations',
              view: output.squaredDeviations,
              type: 'f32',
              access: 'read'
            },
            {name: 'summary', view: output.summary, type: 'f32', access: 'read_write'}
          ],
          invocationCount: 1,
          declarations: `const CLASSES: u32 = ${classCount}u;`,
          body: `var adcm = 0.0;
  var sdcm = 0.0;
  for (var slot = 0u; slot < CLASSES; slot++) {
    adcm += absoluteDeviations[absoluteDeviationsOffset + slot];
    sdcm += squaredDeviations[squaredDeviationsOffset + slot];
  }
  let adam = absoluteDeviations[absoluteDeviationsOffset + CLASSES];
  let tss = squaredDeviations[squaredDeviationsOffset + CLASSES];
  summary[summaryOffset + ${GPU_CLASSIFICATION_FIT_GADF}u] = select(1.0, 1.0 - adcm / adam, adam > 0.0);
  summary[summaryOffset + ${GPU_CLASSIFICATION_FIT_ADCM}u] = adcm;
  summary[summaryOffset + ${GPU_CLASSIFICATION_FIT_ADAM}u] = adam;
  summary[summaryOffset + ${GPU_CLASSIFICATION_FIT_GVF}u] = select(1.0, 1.0 - sdcm / tss, tss > 0.0);
  summary[summaryOffset + ${GPU_CLASSIFICATION_FIT_TSS}u] = sdcm;`
        })
      );
    }
    return nodes;
  }
}
