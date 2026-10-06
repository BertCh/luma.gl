// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
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
import {COLUMN_ORDERED_KEY_WGSL} from './column-classification-shared';
import {COLOR_SCALE_WGSL} from './color-scale-wgsl';
import {GPU_COLOR_SCALE_PARAMETER_LENGTH} from './color-scale-parameters';

const OPERATION = 'GPUColorScale';

/** Caller-owned outputs of {@link GPUColorScale}. At least one is required. */
export type GPUColorScaleOutput = {
  /** Packed `rgba8` colour per row (`r | g << 8 | b << 16 | a << 24`). No-data rows get the no-data colour. */
  colors?: GraphDataView<'uint32'>;
  /**
   * Palette class per row, `0xffffffff` for no-data rows. Classes are always clamped to
   * `[0, paletteCount - 1]`, so a class always indexes the palette.
   */
  classIndices?: GraphDataView<'uint32'>;
  /** Rows per class, `maximumPaletteCount` rows. Rewritten every encoding; no-data rows are not counted. */
  classCounts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUColorScale}.
 *
 * Per-frame (no recompile): the contents of `parameters` (scale type, counts, interpolation,
 * clamp, no-data colour, log floor, exponent), `domain`, `palette`, `domainCount`, `mask` and
 * `values`. Topology (needs a new graph): view lengths, `maximumDomainCount`,
 * `maximumPaletteCount`, the `values` format, and which optional views and outputs are present.
 */
export type GPUColorScaleProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'color-scale'`. */
  id?: string;
  /**
   * Packed values. `float32` serves every scale except `ordinal`; `uint32` category codes select
   * the `ordinal` scale at compile time (the `scale` parameter is then ignored).
   */
  values: GraphDataView<'float32'> | GraphDataView<'uint32'> | GraphDataView<'sint32'>;
  /**
   * How an integer `values` view is read. `'ordinal'` (default) treats `uint32` codes as
   * categories for the `ordinal` scale, as above. `'numeric'` converts the integers to f32 on
   * read (exact up to 2^24 in magnitude) and applies the per-frame `scale` parameter as for a
   * float column, so a count column can be colored without a cast pass. A `sint32` view is
   * always numeric. Ignored for `float32`. Topology.
   */
  integerValues?: 'ordinal' | 'numeric';
  /** Optional packed `uint32` row mask; zero makes the row no-data. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Packed float32 domain, non-decreasing, at least `maximumDomainCount` rows. Edges or stops, for
   * example the `breaks` of `GPUClassBreaks` or quantiles of `GPUColumnQuantiles`, used directly
   * with no readback. Entries may be `+-Infinity`; edges must not be NaN.
   */
  domain: GraphDataView<'float32'>;
  /**
   * Optional one-row `uint32` view holding a CLASS count `k` (for example the `classCount` output
   * of `GPUClassBreaks`). The active domain then has exactly `k + 1` edges, clamped to
   * `maximumDomainCount`; `k = 0` makes every row no-data. When absent, the `domainCount`
   * parameter is the number of active domain entries.
   */
  domainCount?: GraphDataView<'uint32'>;
  /** Packed `uint32` `rgba8` palette, at least `maximumPaletteCount` rows. */
  palette: GraphDataView<'uint32'>;
  /** Per-frame parameters written with `getGPUColorScaleParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Largest active domain length, compile-time. */
  maximumDomainCount: number;
  /** Largest active palette length, compile-time. */
  maximumPaletteCount: number;
  /** Caller-owned outputs. */
  output: GPUColorScaleOutput;
};

/**
 * Applies a colour scale to a column on the GPU, with the scale type, domain, and palette all
 * per-frame data, so switching them never rebuilds or recompiles the graph.
 *
 * Semantics follow d3 scales. Continuous scales (`linear`, `sqrt`, `pow` with `exponent`, `log`,
 * `symlog`) transform the value and the domain ends, then normalise over `[domain[0],
 * domain[domainCount - 1]]` to `t`. When the domain has more than 2 entries and
 * `paletteCount == domainCount`, stop `i` maps to palette entry `i` and the mapping is piecewise
 * linear in the transformed space (a d3 multi-stop scale); otherwise the palette is a uniform
 * ramp. A degenerate domain (zero span) maps every valid row to `t = 0.5`. With `clamp` the
 * position is clamped to the ends; without it an out-of-domain row is no-data. The class is
 * `min(floor(t * paletteCount), paletteCount - 1)` (piecewise: `min(floor(position),
 * paletteCount - 1)`, the stop segment). `step` interpolation outputs `palette[class]` bit exactly;
 * `linear` interpolation blends adjacent entries per channel in f32 and rounds half up to the
 * nearest byte (expect up to 1 per channel difference from a double precision reference).
 * `log` replaces values and domain entries `<= 0` with `logFloor`; a NaN floor makes them no-data.
 *
 * `quantize` splits `[domain[0], domain[last]]` into `paletteCount` equal intervals; `clamp`
 * applies as above. `threshold` and `quantile` classify by the shared edge convention: the class
 * is the number of inner edges `domain[1 .. n - 2]` that are `<= v` (d3 `bisectRight`), found by
 * binary search on order-preserving integer keys, so class ids are exact; values outside the
 * ends fall into the first or last class regardless of `clamp`. Classes beyond the palette clamp
 * to the last entry. These class-based scales and `ordinal` ignore `interpolation`.
 * `ordinal` uses `palette[code]` for `code < paletteCount` and no-data (d3 `unknown`) otherwise.
 *
 * NaN and masked rows get the no-data colour and class `0xffffffff`, and are not counted.
 * Class counts use integer atomics and are exact; the colours are a pure function of the inputs
 * and so are run-to-run identical.
 *
 * Inputs must be single packed views (chunked vectors are not supported).
 */
export class GPUColorScale implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUColorScaleProps;

  constructor(props: GPUColorScaleProps) {
    this.id = props.id ?? 'color-scale';
    this.props = props;
    const id = this.id;
    const {output, maximumDomainCount, maximumPaletteCount} = props;
    for (const [name, count] of [
      ['maximumDomainCount', maximumDomainCount],
      ['maximumPaletteCount', maximumPaletteCount]
    ] as const) {
      if (!Number.isInteger(count) || count < 1) {
        throw new Error(`${id} ${name} must be a positive integer`);
      }
    }
    for (const [name, view] of [
      ['values', props.values],
      ['mask', props.mask],
      ['domain', props.domain],
      ['domainCount', props.domainCount],
      ['palette', props.palette],
      ['parameters', props.parameters],
      ['output.colors', output.colors],
      ['output.classIndices', output.classIndices],
      ['output.classCounts', output.classCounts]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    if (!output.colors && !output.classIndices && !output.classCounts) {
      throw new Error(`${id} needs at least one output (colors, classIndices, or classCounts)`);
    }
    const rows = props.values.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    validatePackedView(props.values, ['float32', 'uint32', 'sint32'], `${id} values`);
    if (
      props.integerValues !== undefined &&
      !['ordinal', 'numeric'].includes(props.integerValues)
    ) {
      throw new Error(`${id} integerValues must be 'ordinal' or 'numeric'`);
    }
    validatePackedView(props.domain, ['float32'], `${id} domain`);
    validatePackedUint32View(props.palette, `${id} palette`);
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal values length`);
      }
    }
    if (props.domain.length < maximumDomainCount) {
      throw new Error(`${id} domain must hold maximumDomainCount rows`);
    }
    if (props.palette.length < maximumPaletteCount) {
      throw new Error(`${id} palette must hold maximumPaletteCount rows`);
    }
    if (props.parameters.length < GPU_COLOR_SCALE_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_COLOR_SCALE_PARAMETER_LENGTH} float32 values`
      );
    }
    if (props.domainCount) {
      validatePackedUint32View(props.domainCount, `${id} domainCount`);
      if (props.domainCount.length < 1) {
        throw new Error(`${id} domainCount must contain one uint32 row`);
      }
    }
    for (const name of ['colors', 'classIndices'] as const) {
      const view = output[name];
      if (view) {
        validatePackedUint32View(view, `${id} output.${name}`);
        if (view.length < rows) {
          throw new Error(`${id} output.${name} must hold one row per value`);
        }
      }
    }
    if (output.classCounts) {
      validatePackedUint32View(output.classCounts, `${id} output.classCounts`);
      if (output.classCounts.length < maximumPaletteCount) {
        throw new Error(`${id} output.classCounts must hold maximumPaletteCount rows`);
      }
    }
    const outputs = [output.colors, output.classIndices, output.classCounts].filter(
      (view): view is GraphDataView<'uint32'> => Boolean(view)
    );
    if (new Set(outputs.map(view => view.buffer)).size !== outputs.length) {
      throw new Error(`${id} outputs must not share buffers with each other`);
    }
    validateGraphOutputsDisjointFromInputs(id, outputs, [
      props.values,
      props.mask,
      props.domain,
      props.domainCount,
      props.palette,
      props.parameters
    ]);
  }

  /** Returns the optional init, classify, and colorize nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output, maximumDomainCount, maximumPaletteCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.mask,
      props.domain,
      props.domainCount,
      props.palette,
      props.parameters,
      output.colors,
      output.classIndices,
      output.classCounts
    ]);
    const rows = props.values.length;
    const isOrdinal = props.values.format === 'uint32' && props.integerValues !== 'numeric';
    const integerType =
      props.values.format === 'float32'
        ? undefined
        : props.values.format === 'uint32'
          ? 'u32'
          : 'i32';
    const classes =
      output.classIndices ?? createTransientView(graph, `${id}-classes`, 'uint32', rows);
    const positions = createTransientView(graph, `${id}-positions`, 'float32', rows);
    const read = (
      name: string,
      view: GraphDataView,
      type: 'u32' | 'i32' | 'f32'
    ): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const declarations = `const MAXIMUM_DOMAIN_COUNT: u32 = ${maximumDomainCount}u;
const MAXIMUM_PALETTE_COUNT: u32 = ${maximumPaletteCount}u;
${COLUMN_ORDERED_KEY_WGSL}
${COLOR_SCALE_WGSL}`;
    const nodes: GPUCommandNode<Parameters>[] = [];

    if (output.classCounts) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-init`,
          operation: OPERATION,
          variant: 'init',
          bindings: [
            {
              name: 'counts',
              view: output.classCounts,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: maximumPaletteCount,
          body: 'counts[countsOffset + index] = 0u;'
        })
      );
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify`,
        operation: OPERATION,
        variant: isOrdinal ? 'classify-ordinal' : integerType ? 'classify-integer' : 'classify',
        bindings: [
          read('values', props.values, integerType ?? 'f32'),
          ...(props.mask ? [read('rowMask', props.mask, 'u32')] : []),
          read('domain', props.domain, 'f32'),
          ...(props.domainCount ? [read('domainCountIn', props.domainCount, 'u32')] : []),
          read('params', props.parameters, 'f32'),
          {name: 'classes', view: classes, type: 'u32', access: 'read_write'},
          {
            name: 'positions',
            view: positions,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: rows,
        declarations: `const DOMAIN_LENGTH: u32 = ${props.domain.length}u;
${declarations}`,
        body: `classes[classesOffset + index] = NO_CLASS;
  positions[positionsOffset + index] = 0.0;
  ${props.mask ? 'if (rowMask[rowMaskOffset + index] == 0u) {\n    return;\n  }' : ''}
  let paletteCount = min(u32(max(params[paramsOffset + 2u], 0.0)), MAXIMUM_PALETTE_COUNT);
  if (paletteCount == 0u) {
    return;
  }
    ${isOrdinal ? ORDINAL_BODY : getFloatBody(props, integerType !== undefined)}`
      })
    );

    if (output.colors || output.classCounts) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-colorize`,
          operation: OPERATION,
          variant: 'colorize',
          bindings: [
            read('classes', classes, 'u32'),
            read('positions', positions, 'f32'),
            read('palette', props.palette, 'u32'),
            read('params', props.parameters, 'f32'),
            ...(output.colors
              ? [
                  {
                    name: 'colors',
                    view: output.colors,
                    type: 'u32',
                    access: 'read_write'
                  } as WGSLKernelBinding
                ]
              : []),
            ...(output.classCounts
              ? [
                  {
                    name: 'counts',
                    view: output.classCounts,
                    type: 'atomic<u32>',
                    access: 'read_write'
                  } as WGSLKernelBinding
                ]
              : [])
          ],
          invocationCount: rows,
          declarations: `const IS_ORDINAL: bool = ${isOrdinal};
${declarations}`,
          body: `let classIndex = classes[classesOffset + index];
  let paletteCount = min(u32(max(params[paramsOffset + 2u], 0.0)), MAXIMUM_PALETTE_COUNT);
  let isValid = classIndex != NO_CLASS && classIndex < paletteCount;
  ${output.classCounts ? 'if (isValid) {\n    atomicAdd(&counts[countsOffset + classIndex], 1u);\n  }' : ''}
  ${
    output.colors
      ? `let noDataColor = u32(params[paramsOffset + 5u]) | (u32(params[paramsOffset + 6u]) << 16u);
  var color = noDataColor;
  if (isValid) {
    color = palette[paletteOffset + classIndex];
    let scale = u32(max(params[paramsOffset], 0.0));
    if (!IS_ORDINAL && scale <= 4u && params[paramsOffset + 3u] == 1.0) {
      let position = positions[positionsOffset + index];
      let lower = min(u32(floor(position)), paletteCount - 1u);
      let upper = min(lower + 1u, paletteCount - 1u);
      color = blendColors(
        palette[paletteOffset + lower],
        palette[paletteOffset + upper],
        position - f32(lower)
      );
    }
  }
  colors[colorsOffset + index] = color;`
      : ''
  }`
        })
      );
    }
    return nodes;
  }
}

const ORDINAL_BODY = /* wgsl */ `let code = values[valuesOffset + index];
  if (code < paletteCount) {
    classes[classesOffset + index] = code;
  }`;

function getFloatBody(props: GPUColorScaleProps, isInteger: boolean): string {
  const domainCountSource = props.domainCount
    ? `let classTotal = domainCountIn[domainCountInOffset];
  domainCount = select(0u, min(classTotal, MAXIMUM_DOMAIN_COUNT) + 1u, classTotal > 0u);`
    : '';
  return `var domainCount = u32(max(params[paramsOffset + 1u], 0.0));
  ${domainCountSource}
  domainCount = min(domainCount, min(MAXIMUM_DOMAIN_COUNT, DOMAIN_LENGTH));
  if (domainCount == 0u) {
    return;
  }
  let value = ${isInteger ? 'f32(values[valuesOffset + index])' : 'values[valuesOffset + index]'};
  if (isNanBits(value)) {
    return;
  }
  let scale = u32(max(params[paramsOffset], 0.0));
  let isClamped = params[paramsOffset + 4u] != 0.0;
  let logFloor = params[paramsOffset + 7u];
  let exponent = params[paramsOffset + 8u];
  var classIndex = NO_CLASS;
  var rampPosition = 0.0;
  if (scale == 6u || scale == 7u) {
    // Threshold and quantile: count the inner edges domain[1 .. domainCount - 2] that are <= value.
    if (domainCount < 2u) {
      return;
    }
    let valueKey = getComparisonKey(value);
    var low = 1u;
    var high = domainCount - 1u;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (getComparisonKey(domain[domainOffset + middle]) <= valueKey) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    classIndex = min(low - 1u, paletteCount - 1u);
  } else if (scale == 5u) {
    if (domainCount < 2u) {
      return;
    }
    let lowEdge = domain[domainOffset];
    let span = domain[domainOffset + domainCount - 1u] - lowEdge;
    var t = select(0.5, (value - lowEdge) / span, span != 0.0);
    if (isNanBits(t) || (!isClamped && (t < 0.0 || t > 1.0))) {
      return;
    }
    t = clamp(t, 0.0, 1.0);
    classIndex = min(u32(floor(t * f32(paletteCount))), paletteCount - 1u);
  } else if (scale <= 4u) {
    if (domainCount < 2u) {
      return;
    }
    let firstStop = transformValue(domain[domainOffset], scale, exponent, logFloor);
    let lastStop = transformValue(domain[domainOffset + domainCount - 1u], scale, exponent, logFloor);
    let transformed = transformValue(value, scale, exponent, logFloor);
    let span = lastStop - firstStop;
    let t = select(0.5, (transformed - firstStop) / span, span != 0.0);
    if (isNanBits(t) || (!isClamped && (t < 0.0 || t > 1.0))) {
      return;
    }
    if (domainCount > 2u && paletteCount == domainCount) {
      // Multi-stop scale: stop i maps to palette entry i, piecewise linear in transformed space.
      if (t <= 0.0) {
        rampPosition = 0.0;
      } else if (t >= 1.0) {
        rampPosition = f32(paletteCount - 1u);
      } else {
        var segment = 0u;
        for (var stopIndex = 1u; stopIndex + 1u < domainCount; stopIndex++) {
          if (transformValue(domain[domainOffset + stopIndex], scale, exponent, logFloor) <= transformed) {
            segment = stopIndex;
          }
        }
        let segmentStart = transformValue(domain[domainOffset + segment], scale, exponent, logFloor);
        let segmentEnd = transformValue(domain[domainOffset + segment + 1u], scale, exponent, logFloor);
        let width = segmentEnd - segmentStart;
        let fraction = select(0.0, clamp((transformed - segmentStart) / width, 0.0, 1.0), width > 0.0);
        rampPosition = f32(segment) + fraction;
      }
      classIndex = min(u32(floor(rampPosition)), paletteCount - 1u);
    } else {
      let clampedT = clamp(t, 0.0, 1.0);
      rampPosition = clampedT * f32(paletteCount - 1u);
      classIndex = min(u32(floor(clampedT * f32(paletteCount))), paletteCount - 1u);
    }
  } else {
    return;
  }
  classes[classesOffset + index] = classIndex;
  positions[positionsOffset + index] = rampPosition;`;
}
