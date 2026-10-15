// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getRasterSamplingNote} from '../../encoding/raster-image';
import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import type {ViewContext} from '../../registry/view-registration';
import type {DescriptionKeyNumber, MarkProvenance} from '../../types';
import {createDivergingScaleSpec, type ResolvedScale} from '../../scales/scale-registry';
import {CARD_FIGURE, getCardPaletteForVariable, type CardPalette} from '../../encoding/card-tokens';
import {getAllCaptures, getCaptureProvenance} from '../../views/record-queries';
import {
  getDeclaredVariableKind,
  getFinalStateCapture,
  getRunScaleRegistry,
  resolveValueScale
} from '../../views/value-strip';

/** One visible cell or an explicit omitted range in a captured array. */
export type ArrayCell = {index: number; key: number} | {gap: string};
/** Pure panel model shared by strip, table, and tile renderers. */
export type ArrayResultPanel = {
  capture: GPUComputationCapture;
  role: 'input' | 'output' | 'context';
  tokens: string[];
  first: number;
  total?: number;
  values: number[];
  cells: ArrayCell[];
  shape?: readonly [number, number];
  blockSize: number;
  palette: CardPalette;
  scale?: ResolvedScale;
  provenance: MarkProvenance;
  numbered: number[];
  x: number;
  y: number;
  width: number;
  height: number;
};
/** Complete array figure, projected only from captured evidence and authored metadata. */
export type GPUCoreArrayResultModel = {
  panels: ArrayResultPanel[];
  captures: GPUComputationCapture[];
  segments: number[];
  segmentCapture?: GPUComputationCapture;
  thumbnail: boolean;
  tiled: boolean;
  width: number;
  height: number;
  keyNumber: DescriptionKeyNumber;
  notes: string[];
  grid?: {originX: number; originY: number; cellSize: number; unit: string};
};

/** Final role-tagged captures; untagged intermediate snapshots never become result evidence. */
export function getArrayResultCaptures(context: ViewContext): GPUComputationCapture[] {
  return [...new Set(getAllCaptures(context.record).map(capture => capture.name))]
    .map(name => getFinalStateCapture(context.record, name)!)
    .filter(
      capture => capture.role === 'input' || capture.role === 'output' || capture.role === 'context'
    );
}

/** Space-separated, scenario-authored drawing vocabulary. */
export function getArrayTokens(capture: GPUComputationCapture): string[] {
  return capture.meaning?.split(/\s+/).filter(Boolean) ?? [];
}

/** Reads one parameterized token without guessing a missing declaration. */
export function getArrayTokenValue(tokens: readonly string[], name: string): string | undefined {
  return tokens.find(token => token.startsWith(`${name}:`))?.slice(name.length + 1);
}

function buildPanel(
  capture: GPUComputationCapture,
  context: ViewContext,
  thumbnail: boolean
): ArrayResultPanel {
  const tokens = getArrayTokens(capture);
  const components = capture.components ?? 1;
  const first = Number(getArrayTokenValue(tokens, 'first') ?? 0);
  const keySpace = context.record.axes.keySpaces.find(space => space.id === capture.keySpace);
  const shape: [number, number] | undefined =
    keySpace?.shape?.length === 2 ? [keySpace.shape[0], keySpace.shape[1]] : undefined;
  const magnitude = tokens.includes('complex-magnitude') && components === 2;
  const logarithmic = magnitude && tokens.includes('log');
  // CSR cell starts (cells + 1 values) are drawn as per-cell counts offsets[c + 1] - offsets[c].
  const csrOffsets = tokens.includes('csr-offsets');
  const derived = magnitude || csrOffsets;
  let values = Array.from({length: Math.floor(capture.values.length / components)}, (_, index) => {
    const real = Number(capture.values[index * components]);
    const value = magnitude
      ? Math.hypot(real, Number(capture.values[index * components + 1]))
      : real;
    return logarithmic ? Math.log10(1 + value) : value;
  });
  if (csrOffsets) {
    const offsets = values;
    values = offsets.slice(1).map((value, index) => value - offsets[index]!);
  }
  const countCapture = tokens.includes('compact')
    ? getArrayResultCaptures(context).find(
        candidate => candidate.role === 'output' && getArrayTokens(candidate).includes('count')
      )
    : undefined;
  const compactCount = countCapture ? Number(countCapture.values[0]) : undefined;
  if (compactCount !== undefined && Number.isSafeInteger(compactCount) && compactCount >= 0)
    values = values.slice(0, Math.max(0, compactCount - first));
  const total = compactCount ?? keySpace?.size;
  // Derived display quantities get a separate run-scoped variable: never classify magnitudes by real/imaginary samples.
  const scaleName = magnitude
    ? `${capture.name}:magnitude${logarithmic ? ':log' : ''}`
    : csrOffsets
      ? `${capture.name}:counts`
      : capture.name;
  const scaleContext: ViewContext = derived
    ? {
        ...context,
        scales: getRunScaleRegistry(context),
        record: {
          ...context.record,
          captures: {
            schema: 'luma.computation-captures/1',
            runId: context.record.runId,
            snapshots: [{...capture, name: scaleName, values, components: 1}],
            stamps: [],
            replays: []
          }
        }
      }
    : context;
  const registry = getRunScaleRegistry(context);
  const kind = getDeclaredVariableKind(context, capture.name);
  if (
    !derived &&
    !registry.getByVariable(scaleName) &&
    kind &&
    'diverging' in kind &&
    kind.diverging === true &&
    'midpoint' in kind &&
    typeof kind.midpoint === 'number' &&
    values.some(Number.isFinite)
  ) {
    const minimum = values.reduce(
      (previous, value) => (Number.isFinite(value) ? Math.min(previous, value) : previous),
      Infinity
    );
    const maximum = values.reduce(
      (previous, value) => (Number.isFinite(value) ? Math.max(previous, value) : previous),
      -Infinity
    );
    if (kind.midpoint >= minimum && kind.midpoint <= maximum)
      registry.register(
        createDivergingScaleSpec({
          id: `${context.record.runId}:${scaleName}`,
          variable: scaleName,
          variableKind: kind,
          midpoint: kind.midpoint,
          finalStateValues: values,
          noDataColor: '#d9d9d9',
          ground: context.ground,
          classCount: 5
        })
      );
  }
  const scale = resolveValueScale(scaleContext, scaleName)?.scale;
  const maximum = thumbnail ? 8 : 16;
  const indices =
    values.length <= maximum
      ? values.map((_, index) => index)
      : [
          ...Array.from({length: thumbnail ? 4 : 8}, (_, index) => index),
          -1,
          ...Array.from(
            {length: thumbnail ? 4 : 8},
            (_, index) => values.length - (thumbnail ? 4 : 8) + index
          )
        ];
  const cells: ArrayCell[] = indices.map(index =>
    index < 0 ? {gap: `… ${values.length - maximum} more …`} : {index, key: first + index}
  );
  const remaining = Math.max(0, (total ?? first + values.length) - first - values.length);
  if (remaining) {
    cells.push({gap: `… ${remaining} more`});
  }
  return {
    capture,
    tokens,
    first,
    total,
    values,
    cells,
    shape,
    role: capture.role === 'input' ? 'input' : capture.role === 'context' ? 'context' : 'output',
    blockSize: shape ? Math.max(1, Math.ceil(Math.max(...shape) / (thumbnail ? 64 : 160))) : 1,
    palette: getCardPaletteForVariable(context, capture.name, capture),
    scale,
    provenance: derived ? 'derived' : getCaptureProvenance(capture),
    numbered: [],
    x: 0,
    y: 0,
    width: 0,
    height: 0
  };
}

/** Builds the array view without a GPU, DOM, invented roles, or inferred highlights. */
export function buildGPUCoreArrayResultModel(context: ViewContext): GPUCoreArrayResultModel {
  const captures = getArrayResultCaptures(context);
  const thumbnail = context.options?.['size'] === 'thumbnail';
  const segmentCapture = captures.find(capture =>
    getArrayTokens(capture).some(
      token => token === 'segments-offsets' || token === 'segments-flags'
    )
  );
  const segmentFirst = Number(
    (segmentCapture && getArrayTokenValue(getArrayTokens(segmentCapture), 'first')) || 0
  );
  const segments = segmentCapture
    ? Array.from(segmentCapture.values, Number).flatMap((value, index) =>
        getArrayTokens(segmentCapture).includes('segments-offsets')
          ? index < segmentCapture.values.length - 1
            ? [value]
            : []
          : value !== 0
            ? [segmentFirst + index]
            : []
      )
    : [];
  const panels = captures
    .filter(
      capture =>
        capture !== segmentCapture &&
        !getArrayTokens(capture).includes('count') &&
        !capture.geometry
    )
    .map(capture => buildPanel(capture, context, thumbnail));
  const tables = panels.some(panel => getArrayTokenValue(panel.tokens, 'column') !== undefined);
  const tiled = panels.some(panel => panel.shape || panel.tokens.includes('stencil'));
  const width = thumbnail ? CARD_FIGURE.thumbnailWidth : CARD_FIGURE.width;
  const height = thumbnail
    ? CARD_FIGURE.thumbnailHeight
    : tiled || tables
      ? CARD_FIGURE.height
      : Math.max(1, panels.length) * 28 + Math.max(0, panels.length - 1) * 12 + 32;
  panels.forEach((panel, index) => {
    panel.x = tiled
      ? (thumbnail ? 6 : 24) + (index * (width - (thumbnail ? 12 : 48))) / panels.length
      : thumbnail
        ? 6
        : 112;
    panel.y = tiled
      ? thumbnail
        ? 8
        : 40
      : thumbnail
        ? 8 + (index * 80) / Math.max(1, panels.length)
        : 16 + index * 40;
    panel.width = tiled
      ? (width - (thumbnail ? 12 : 48)) / panels.length - (thumbnail ? 4 : 24)
      : width - panel.x - (thumbnail ? 6 : 16);
    panel.height = tiled
      ? height - (thumbnail ? 16 : 80)
      : thumbnail
        ? Math.min(20, 64 / Math.max(1, panels.length))
        : 28;
  });
  // One shared numeric-label budget, fairly distributed across rows; the declared answer has priority.
  if (!thumbnail) {
    let budget = 16;
    for (const panel of panels) {
      if (panel.capture.highlight && panel.palette !== 'div') {
        const index = panel.capture.highlight.key - panel.first;
        if (index >= 0 && index < panel.values.length) {
          panel.numbered.push(index);
          budget--;
          break;
        }
      }
    }
    for (const panel of panels.filter(candidate => candidate.tokens.includes('stencil'))) {
      for (let index = 0; index < panel.values.length && budget > 0; index++) {
        if (!panel.numbered.includes(index)) {
          panel.numbered.push(index);
          budget--;
        }
      }
    }
    for (let position = 0; position < 16 && budget > 0; position++) {
      for (const panel of panels) {
        const cell = panel.cells[position];
        if (
          !cell ||
          !('index' in cell) ||
          panel.numbered.includes(cell.index) ||
          (panel.shape && Math.max(...panel.shape) > 6) ||
          getArrayTokenValue(panel.tokens, 'column') !== undefined
        )
          continue;
        panel.numbered.push(cell.index);
        budget--;
        if (!budget) break;
      }
    }
  }
  const notes: string[] = [];
  if (captures.some(capture => capture.geometry?.kind === 'points'))
    notes.push(
      `Point overlays use stride decimation, capped at ${thumbnail ? 64 : 400} points per input.`
    );
  for (const panel of panels.filter(candidate => candidate.tokens.includes('compact'))) {
    const flags = captures.find(capture => getArrayTokens(capture).includes('flags'));
    if (!flags || Number(getArrayTokenValue(getArrayTokens(flags), 'first') ?? 0) !== 0)
      notes.push(
        `${panel.capture.name}: compaction connectors unknown — complete flags from key 0 are required`
      );
    else if (Array.from(flags.values, Number).some(value => value !== 0 && value !== 1))
      notes.push(`${panel.capture.name}: nonbinary flags; raw scan offsets can leave gaps`);
  }
  for (const panel of panels) {
    if (panel.tokens.includes('complex-magnitude'))
      notes.push(
        `${panel.capture.name}: ${panel.tokens.includes('log') ? 'log10(1 + magnitude), ' : ''}magnitude = sqrt(re² + im²), derived from captured components`
      );
    if (panel.tokens.includes('csr-offsets'))
      notes.push(
        `${panel.capture.name}: cell value = offsets[c + 1] − offsets[c], the count per cell, derived from captured CSR offsets`
      );
    if (panel.shape)
      notes.push(
        `${panel.capture.name}: ${getRasterSamplingNote(...panel.shape, thumbnail ? 64 : 160, panel.palette === 'nom') || 'captured cells at native resolution'}`
      );
    if (!panel.shape)
      notes.push(
        `${panel.capture.name}: ${panel.values.length} captured values (${panel.cells.filter(cell => 'key' in cell).length} shown)`
      );
    if (panel.first || (panel.total !== undefined && panel.total > panel.values.length))
      notes.push(
        `${panel.capture.name}: captured window starts at ${panel.first}, ${panel.values.length} of ${panel.total ?? 'unknown'} values`
      );
  }
  const choice = (name: string) =>
    context.record.choices.find(candidate => candidate.name === name);
  const originX = choice('gridOriginX');
  const originY = choice('gridOriginY');
  const cellSize = choice('gridCellSize');
  const grid =
    typeof originX?.value === 'number' &&
    typeof originY?.value === 'number' &&
    typeof cellSize?.value === 'number' &&
    cellSize.value > 0 &&
    cellSize.unit
      ? {
          originX: originX.value,
          originY: originY.value,
          cellSize: cellSize.value,
          unit: cellSize.unit
        }
      : undefined;
  if (panels.some(panel => panel.tokens.includes('grid')))
    notes.push(
      grid
        ? `Grid origin (${grid.originX}, ${grid.originY}); cell size ${grid.cellSize} ${grid.unit}`
        : 'Grid origin/cell size unknown — required recorded choices with unit are missing; points cannot be placed'
    );
  const count = captures.find(
    capture => capture.role === 'output' && getArrayTokens(capture).includes('count')
  );
  const highlighted = panels.find(panel => panel.role === 'output' && panel.capture.highlight);
  const outputPanel = panels.find(panel => panel.role === 'output');
  const finiteValues = outputPanel?.values.filter(Number.isFinite) ?? [];
  const maximum = finiteValues.reduce((current, value) => Math.max(current, value), -Infinity);
  const keyNumber: DescriptionKeyNumber = count
    ? {label: 'count', value: Number(count.values[0] ?? 0)}
    : highlighted
      ? {
          label: `${highlighted.capture.name}[${highlighted.capture.highlight!.key}]`,
          value:
            highlighted.values[highlighted.capture.highlight!.key - highlighted.first] ?? 'unknown'
        }
      : {
          label: `max ${outputPanel?.capture.name ?? 'output'}`,
          value: Number.isFinite(maximum) ? maximum : 'unknown'
        };
  return {
    panels,
    captures,
    segments,
    segmentCapture,
    thumbnail,
    tiled,
    width,
    height,
    keyNumber,
    notes,
    grid
  };
}
