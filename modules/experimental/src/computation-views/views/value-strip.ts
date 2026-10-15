// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {renderRasterImage} from '../encoding/raster-image';
import type {GPUComputationCapture} from '@luma.gl/gpgpu';

import {CARD_PALETTE_SOURCES} from '../encoding/card-tokens';
import {getSelectionInk} from '../encoding/selection-ink';
import {crispnessAttributes} from '../encoding/crispness';
import {provenanceMarkAttributes} from '../encoding/provenance';
import {rgbToHex} from '../scales/class-table';
import {NO_DATA_COLOR} from '../scales/hue-registry';
import {createQuantileScaleSpec, type ResolvedScale, ScaleRegistry} from '../scales/scale-registry';
import type {MarkProvenance, ProvisionalState, VariableKind, ViewGround} from '../types';
import {h, type ViewElementNode} from '../view-node';
import type {LensRecord, ViewContext} from '../registry/view-registration';
import {getAllCaptures} from './record-queries';

const RUN_SCALE_REGISTRIES = new WeakMap<LensRecord, ScaleRegistry>();

/**
 * The run's scale registry: `context.scales` when the host supplies one, otherwise one registry
 * shared by every view of the same record object, so all views of a lens page agree (§E rule 1).
 */
export function getRunScaleRegistry(context: ViewContext): ScaleRegistry {
  if (context.scales) {
    return context.scales;
  }
  let registry = RUN_SCALE_REGISTRIES.get(context.record);
  if (!registry) {
    registry = new ScaleRegistry({runId: context.record.runId});
    RUN_SCALE_REGISTRIES.set(context.record, registry);
  }
  return registry;
}

/**
 * Final-state values of a captured variable: the capture with that name at the highest
 * iteration (or without iteration). Iteration domains come from the final state (§E rule 2).
 */
export function getFinalStateCapture(
  record: LensRecord,
  name: string
): GPUComputationCapture | undefined {
  let finalCapture: GPUComputationCapture | undefined;
  for (const capture of getAllCaptures(record)) {
    if (capture.name !== name) {
      continue;
    }
    if (!finalCapture || (capture.iteration ?? Infinity) >= (finalCapture.iteration ?? Infinity)) {
      finalCapture = capture;
    }
  }
  return finalCapture;
}

/** Declared variable kind of a capture from the domain packs, or undefined (never guessed). */
export function getDeclaredVariableKind(
  context: ViewContext,
  name: string
): VariableKind | undefined {
  for (const pack of context.packs ?? []) {
    const kind = pack.variableKinds?.[name];
    if (kind) {
      return kind;
    }
  }
  return undefined;
}

/**
 * Resolves the one value scale of a captured variable for this run: an existing registration,
 * else a final-state quantile scale with lightness ramp and a stated beyond-range class. Returns
 * undefined when the final state has no finite values.
 *
 * When the measurement level is not declared the scale is ordered (lightness) and the legend says
 * so; hue is never chosen without a declared nominal level.
 */
export function resolveValueScale(
  context: ViewContext,
  name: string
): {scale: ResolvedScale; levelDeclared: boolean} | undefined {
  const registry = getRunScaleRegistry(context);
  const existing = registry.getByVariable(name);
  const declaredKind = getDeclaredVariableKind(context, name);
  if (existing) {
    return {scale: registry.get(existing), levelDeclared: Boolean(declaredKind)};
  }
  const finalCapture = getFinalStateCapture(context.record, name);
  if (!finalCapture || !hasFiniteValue(finalCapture.values)) {
    return undefined;
  }
  const spec = createQuantileScaleSpec({
    id: `${context.record.runId}:${name}`,
    variable: name,
    variableKind: declaredKind ?? {level: 'interval'},
    ramp: CARD_PALETTE_SOURCES.seq[context.ground].ramp,
    noDataColor: rgbToHex(NO_DATA_COLOR[context.ground]),
    ground: context.ground,
    finalStateValues: finalCapture.values,
    classCount: 5
  });
  return {scale: registry.get(registry.register(spec)), levelDeclared: Boolean(declaredKind)};
}

/** Props of {@link renderValueStrip}. */
export type ValueStripProps = {
  values: ArrayLike<number>;
  scale: ResolvedScale | undefined;
  provenance: MarkProvenance;
  idPrefix: string;
  ground: ViewGround;
  width?: number;
  height?: number;
  /** Largest number of drawn cells; longer buffers are drawn as means per bin. */
  maxCells?: number;
  /** Key of `values[0]` (for windows into a larger buffer). */
  keyOffset?: number;
  /** Pinned key, drawn with the selection core and halo. */
  pinnedKey?: number;
  /** Highlighted keys (selection image), drawn with an achromatic outline. */
  highlightedKeys?: ReadonlySet<number>;
  /** Convergence state, encoded by crispness only. */
  provisional?: ProvisionalState;
  /** Accessible label of the strip. */
  label: string;
};

/**
 * A 1-D index-space strip: one cell per key (or a mean per bin when the buffer is
 * longer than `maxCells`). Unaggregated small strips carry `data-action="pick" data-axis="entity" data-key`.
 * Value is lightness from the run's scale; lineage is texture; provisional state is blur.
 */
export function renderValueStrip(props: ValueStripProps): ViewElementNode {
  const width = props.width ?? 640;
  const height = props.height ?? 28;
  const keyOffset = props.keyOffset ?? 0;
  const count = props.values.length;
  const maxCells = Math.min(160, Math.max(1, props.maxCells ?? 160));
  const binSize = Math.max(1, Math.ceil(count / maxCells));
  const binCount = Math.min(count, maxCells);
  const cellWidth = binCount > 0 ? width / binCount : width;
  const ink = getSelectionInk(props.ground);
  const noData = props.scale?.noDataColor ?? rgbToHex(NO_DATA_COLOR[props.ground]);
  const colorFor = (value: number) =>
    props.scale && Number.isFinite(value) ? props.scale.colorFor(value) : noData;
  const cells: ViewElementNode[] = [];
  const outlines: ViewElementNode[] = [];
  for (let bin = 0; bin < binCount; bin++) {
    const first = Math.floor((bin * count) / binCount);
    const last = Math.floor(((bin + 1) * count) / binCount) - 1;
    const x = bin * cellWidth;
    const firstKey = keyOffset + first;
    const lastKey = keyOffset + last;
    const markAttributes = {
      'data-action': 'pick',
      'data-axis': 'entity',
      'data-key': firstKey,
      'data-key-end': binSize > 1 ? lastKey : undefined
    };
    if (binSize === 1 && count <= 32) {
      const value = Number(props.values[first]);
      cells.push(
        h(
          'rect',
          {
            x: round(x),
            y: 0,
            width: round(Math.max(cellWidth, 0.5)),
            height,
            ...markAttributes,
            ...provenanceMarkAttributes(props.provenance, colorFor(value), props.idPrefix)
          },
          h('title', null, `${firstKey}: ${formatCellValue(value)}`)
        )
      );
    }
    const highlighted = rangeIntersects(props.highlightedKeys, firstKey, lastKey);
    const pinned =
      props.pinnedKey !== undefined && props.pinnedKey >= firstKey && props.pinnedKey <= lastKey;
    if (pinned) {
      outlines.push(
        h('rect', {
          x: round(x - 1.5),
          y: -1.5,
          width: round(Math.max(cellWidth, 0.5) + 3),
          height: height + 3,
          fill: 'none',
          stroke: ink.halo,
          'stroke-width': 3,
          'data-state': 'selection-halo'
        }),
        h('rect', {
          x: round(x),
          y: 0,
          width: round(Math.max(cellWidth, 0.5)),
          height,
          fill: 'none',
          stroke: ink.core,
          'stroke-width': 1.5,
          'data-state': 'selection'
        })
      );
    } else if (highlighted) {
      outlines.push(
        h('rect', {
          x: round(x),
          y: 0,
          width: round(Math.max(cellWidth, 0.5)),
          height,
          fill: 'none',
          stroke: ink.core,
          'stroke-width': 1,
          'data-state': 'highlight'
        })
      );
    }
  }
  if (count > 32 || binSize > 1)
    cells.push(
      renderRasterImage({
        columns: count,
        rows: 1,
        limit: Math.min(160, maxCells),
        valueAt: index => Number(props.values[index]),
        categorical: props.scale?.variableKind.level === 'nominal',
        colorFor,
        attributes: {width, height, 'data-provenance': props.provenance}
      })
    );
  return h(
    'svg',
    {
      class: 'lens-value-strip',
      width,
      height: height + 4,
      viewBox: `0 -2 ${width} ${height + 4}`,
      role: 'img',
      'aria-label': props.label,
      'data-aggregate': binSize > 1 ? `bins of ${binSize}` : undefined
    },
    h('g', crispnessAttributes(props.provisional ?? 'final', props.idPrefix), cells),
    outlines
  );
}

function hasFiniteValue(values: ArrayLike<number>): boolean {
  for (let index = 0; index < values.length; index++) {
    if (Number.isFinite(Number(values[index]))) {
      return true;
    }
  }
  return false;
}

function rangeIntersects(
  keys: ReadonlySet<number> | undefined,
  firstKey: number,
  lastKey: number
): boolean {
  if (!keys || keys.size === 0) {
    return false;
  }
  if (lastKey - firstKey < keys.size) {
    for (let key = firstKey; key <= lastKey; key++) {
      if (keys.has(key)) {
        return true;
      }
    }
    return false;
  }
  for (const key of keys) {
    if (key >= firstKey && key <= lastKey) {
      return true;
    }
  }
  return false;
}

function formatCellValue(value: number): string {
  if (Number.isNaN(value)) {
    return 'NaN';
  }
  if (!Number.isFinite(value)) {
    return value > 0 ? '∞' : '−∞';
  }
  return Number.isInteger(value) ? String(value) : value.toPrecision(4);
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}
