// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {renderRasterImage, getRasterSamplingNote} from '../encoding/raster-image';
import {assert} from '@luma.gl/core';
import type {GPUComputationCapture, GPUComputationCaptureGeometry} from '@luma.gl/gpgpu';
import type {LensRecord, ViewContext, ViewRegistration} from '../registry/view-registration';
import {h, findViewNodes, viewNodeToDom, type ViewElementNode} from '../view-node';
import {
  provenanceBadge,
  provenanceMarkAttributes,
  provenancePatternDefs
} from '../encoding/provenance';
import {crispnessDefs} from '../encoding/crispness';
import {getSelectionInk} from '../encoding/selection-ink';
import {
  CARD_FIGURE,
  CARD_COLORS,
  getCardMarkAttributes,
  getCardPaletteColors,
  getCardPaletteForVariable,
  type CardRole,
  type CardMarkShape,
  type CardPalette
} from '../encoding/card-tokens';
import type {ResolvedScale} from '../scales/scale-registry';
import {
  getAdapterLabel,
  getAllCaptures,
  getCaptureProvenance,
  getCaptureValue,
  getPinnedEntity
} from './record-queries';
import {getFinalStateCapture, resolveValueScale} from './value-strip';
import {createGeometryProjection, type GeometryProjection} from './geometry-projection';

type ResultPanel = {
  capture: GPUComputationCapture;
  geometry?: GPUComputationCaptureGeometry;
  role: 'input' | 'output' | 'context';
  points: [number, number][];
  offsets?: GPUComputationCapture;
  targets?: GPUComputationCapture;
  values?: GPUComputationCapture;
  count: number | 'unknown';
  noun: 'points' | 'lines' | 'polygons' | 'cells' | 'nodes' | 'values' | 'rows';
  blockSize: number;
  notes: string[];
  unknown?: string;
  palette: CardPalette;
  scale?: ResolvedScale;
  categories?: number[];
};

/** Final input/output captures and their single shared native-space projection. */
export type ResultViewModel = {
  panels: ResultPanel[];
  output: ResultPanel;
  projection: GeometryProjection;
  layout: 'side-by-side' | 'overlay';
  thumbnail: boolean;
  width: number;
  height: number;
  grammar: 'geometry' | 'strips' | 'table';
  frames: ResultPanel[];
};

function getResultCaptures(record: LensRecord): GPUComputationCapture[] {
  const names = [...new Set(getAllCaptures(record).map(capture => capture.name))];
  return names
    .map(name => getFinalStateCapture(record, name)!)
    .filter(capture => ['input', 'output', 'context'].includes(capture.role ?? ''));
}

function resolveCapture(record: LensRecord, reference?: string): GPUComputationCapture | undefined {
  return reference === undefined
    ? undefined
    : (getAllCaptures(record).find(capture => capture.id === reference) ??
        getFinalStateCapture(record, reference));
}

function getPoints(capture: GPUComputationCapture | undefined): [number, number][] {
  if (!capture) return [];
  const components = capture.components ?? 2;
  return Array.from({length: Math.floor(capture.values.length / components)}, (_, index) => [
    Number(capture.values[index * components]),
    Number(capture.values[index * components + 1])
  ]);
}

function getRole(capture: GPUComputationCapture): ResultPanel['role'] {
  return capture.role === 'context' ? 'context' : capture.role === 'input' ? 'input' : 'output';
}

function buildPanel(capture: GPUComputationCapture, context: ViewContext): ResultPanel {
  const geometry = capture.geometry;
  const resolve = (reference?: string) => {
    const finalCapture = resolveCapture(context.record, reference);
    return capture.iteration === undefined || !finalCapture || finalCapture.id === reference
      ? finalCapture
      : (getAllCaptures(context.record).find(
          candidate =>
            candidate.name === finalCapture.name && candidate.iteration === capture.iteration
        ) ?? finalCapture);
  };
  const offsets =
    geometry && 'offsetsCapture' in geometry ? resolve(geometry.offsetsCapture) : undefined;
  const targets = geometry?.kind === 'graph' ? resolve(geometry.targetsCapture) : undefined;
  const values =
    !geometry || geometry.kind === 'raster'
      ? capture
      : geometry.valuesCapture !== undefined
        ? resolve(geometry.valuesCapture)
        : geometry.kind === 'graph' && capture.values.length
          ? capture
          : undefined;
  const points =
    geometry?.kind === 'graph'
      ? getPoints(resolve(geometry.layoutCapture))
      : geometry && geometry.kind !== 'raster'
        ? getPoints(capture)
        : [];
  const palette = getCardPaletteForVariable(
    context,
    values?.name ?? capture.name,
    values?.meaning ? values : capture
  );
  const panel: ResultPanel = {
    capture,
    geometry,
    role: getRole(capture),
    points,
    offsets,
    targets,
    values,
    palette,
    scale: values ? resolveValueScale(context, values.name)?.scale : undefined,
    count: !geometry
      ? capture.values.length
      : geometry.kind === 'raster'
        ? geometry.shape[0] * geometry.shape[1]
        : geometry.kind === 'points'
          ? points.length
          : offsets
            ? Math.max(0, offsets.values.length - 1)
            : 'unknown',
    noun: !geometry
      ? 'values'
      : geometry.kind === 'raster'
        ? 'cells'
        : geometry.kind === 'graph'
          ? 'nodes'
          : geometry.kind,
    blockSize: 1,
    notes: []
  };
  if (palette === 'nom' && values) {
    const frequencies = new Map<number, number>();
    const finalValues = getFinalStateCapture(context.record, values.name)?.values ?? values.values;
    for (const value of Array.from(finalValues)) {
      if (Number.isFinite(value)) frequencies.set(value, (frequencies.get(value) ?? 0) + 1);
    }
    panel.categories = [...frequencies.keys()];
    if (panel.categories.length > 5) {
      panel.notes.push(`${panel.categories.length} categories; 5 largest coloured`);
      panel.categories.sort((left, right) => frequencies.get(right)! - frequencies.get(left)!);
      panel.categories = panel.categories.slice(0, 5);
    }
  }
  if (points.length)
    panel.notes.push(
      `${points.length} source vertices/points; stride decimation to the figure budget (400 vertices, 64 in thumbnails, shared across panels and frames)`
    );
  if (!geometry) {
    if (capture.values.length > CARD_FIGURE.maxStripCells)
      panel.notes.push(`${capture.values.length} values, 16 shown`);
    return panel;
  }
  if (geometry.kind === 'raster') {
    const limit = context.options?.['size'] === 'thumbnail' ? 64 : 160;
    panel.blockSize = Math.max(1, Math.ceil(Math.max(...geometry.shape) / limit));
    if (panel.blockSize > 1)
      panel.notes.push(getRasterSamplingNote(...geometry.shape, limit, panel.palette === 'nom'));
    panel.notes.push(
      geometry.cellSize !== undefined
        ? `cell size ${geometry.cellSize} ${geometry.unit ?? '(unit unknown)'}`
        : 'cell size unknown'
    );
    if (!geometry.origin) panel.notes.push('Raster pixel grid; origin not recorded');
  }
  if ((geometry.kind === 'lines' || geometry.kind === 'polygons') && !offsets)
    panel.unknown = 'offsets capture missing';
  if ('valuesCapture' in geometry && geometry.valuesCapture && !values)
    panel.notes.push('unknown — values capture missing');
  if (geometry.kind === 'graph') {
    if (!points.length)
      panel.notes.push('no layout recorded: adjacency table instead of a node-link drawing');
    if (!offsets || !targets) panel.unknown = 'CSR offsets or targets capture missing';
    if (points.length && targets && targets.values.length > 2000)
      panel.notes.push(`${targets.values.length} edges; stride decimation for the figure budget`);
  }
  return panel;
}

function sampleEvenly<Value>(values: Value[], limit: number): Value[] {
  if (values.length <= limit) return values;
  return Array.from(
    {length: limit},
    (_, index) => values[Math.round((index * (values.length - 1)) / (limit - 1))]
  );
}

/** Builds the final input → output comparison, retaining recorded geometry and feature order. */
export function buildResultViewModel(context: ViewContext): ResultViewModel {
  const captures = getResultCaptures(context.record);
  const outputCapture = captures.filter(capture => capture.role === 'output').at(-1);
  // satisfies requires a tagged output, with or without spatial geometry.
  assert(outputCapture);
  const panels = ['context', 'input', 'output'].flatMap(role =>
    captures
      .filter(capture => getRole(capture) === role)
      .map(capture => buildPanel(capture, context))
  );
  const output = panels[panels.length - 1];
  const rolePanels = panels.filter(panel => panel.role !== 'context');
  const arrays = rolePanels.filter(panel => !panel.geometry);
  const table =
    arrays.length >= 2 &&
    (arrays.filter(panel => panel.capture.meaning?.startsWith('column')).length >= 2 ||
      arrays.some(
        (panel, index) =>
          panel.capture.keySpace &&
          arrays.some(
            (other, otherIndex) =>
              index !== otherIndex && other.capture.keySpace === panel.capture.keySpace
          )
      ));
  const grammar = output.geometry ? 'geometry' : table ? 'table' : 'strips';
  if (grammar === 'table')
    for (const panel of arrays) {
      panel.noun = 'rows';
      panel.notes = panel.notes.filter(note => !note.endsWith('values, 16 shown'));
    }
  const thumbnail = context.options?.['size'] === 'thumbnail';
  const sameSupport = rolePanels.some(
    panel =>
      panel.role === 'input' &&
      panel.geometry?.kind === output.geometry?.kind &&
      panel.count !== 'unknown' &&
      panel.count === output.count
  );
  const contextRaster = panels.some(
    panel => panel.role === 'context' && panel.geometry?.kind === 'raster'
  );
  const defaultLayout =
    output.geometry &&
    output.values &&
    sameSupport &&
    !(output.geometry.kind === 'raster' && contextRaster)
      ? 'side-by-side'
      : 'overlay';
  const requestedLayout = context.options?.['layout'];
  const layout =
    requestedLayout === 'overlay' || requestedLayout === 'side-by-side'
      ? requestedLayout
      : defaultLayout;
  const width = thumbnail
    ? CARD_FIGURE.thumbnailWidth
    : layout === 'side-by-side' && grammar === 'geometry'
      ? (CARD_FIGURE.width - CARD_FIGURE.panelGap * (rolePanels.length - 1)) / rolePanels.length
      : CARD_FIGURE.width;
  const height = thumbnail
    ? CARD_FIGURE.thumbnailHeight
    : grammar === 'strips'
      ? arrays.length * CARD_FIGURE.stripRowHeight +
        (arrays.length - 1) * CARD_FIGURE.stripRowGap +
        2 * CARD_FIGURE.stripPadding
      : CARD_FIGURE.height;
  const spatialPanels = panels.filter(panel => panel.geometry);
  const models = spatialPanels.map(panel =>
    panel.geometry!.kind === 'raster'
      ? // A raster without a recorded origin is drawn in its own cell space.
        panel.geometry!.origin
        ? (panel.geometry!.coordinateModel ?? 'unknown')
        : ('projected' as const)
      : (panel.geometry!.coordinateModel ?? 'unknown')
  );
  const points = spatialPanels.flatMap(panel => {
    const geometry = panel.geometry;
    if (geometry?.kind !== 'raster') return panel.points;
    const {columns, rows} = getRasterPlacement(geometry);
    return [
      [columns(0), rows(0)],
      [columns(geometry.shape[0]), rows(geometry.shape[1])]
    ] as [number, number][];
  });
  const units = spatialPanels.map(panel =>
    panel.geometry && 'unit' in panel.geometry ? panel.geometry.unit : undefined
  );
  const projection = createGeometryProjection({
    coordinateModel: models.every(model => model === models[0])
      ? (models[0] ?? 'unknown')
      : 'unknown',
    points,
    width,
    height,
    padding: thumbnail ? CARD_FIGURE.thumbnailPadding : CARD_FIGURE.padding,
    unit: units.every(unit => unit === units[0]) ? units[0] : undefined
  });
  if (!models.every(model => model === models[0]))
    projection.statement += ' Captures have differing coordinate models; alignment is unknown.';
  const iterations = new Map<number, GPUComputationCapture>();
  if (
    !thumbnail &&
    grammar !== 'table' &&
    (layout === 'overlay' || output.geometry?.kind === 'raster')
  ) {
    for (const capture of context.record.captures?.snapshots ?? []) {
      if (capture.name === outputCapture.name && capture.iteration !== undefined)
        iterations.set(capture.iteration, capture);
    }
  }
  const snapshots = [...iterations.values()].sort(
    (left, right) => left.iteration! - right.iteration!
  );
  const candidates = snapshots.map(capture => buildPanel(capture, context));
  if (snapshots.length >= 2 && snapshots.at(-1)?.id !== outputCapture.id) candidates.push(output);
  const signature = (panel: ResultPanel) =>
    Array.from(panel.values?.values ?? panel.capture.values);
  const firstValues = candidates[0] && signature(candidates[0]);
  const changed =
    firstValues &&
    candidates.some(panel => {
      const values = signature(panel);
      return (
        values.length !== firstValues.length ||
        values.some((value, index) => !Object.is(value, firstValues[index]))
      );
    });
  const frames =
    changed && snapshots.length >= 2 ? sampleEvenly(candidates, CARD_FIGURE.maxFrames) : [];
  if (frames.length && output.geometry?.kind === 'raster') {
    const note = getRasterSamplingNote(...output.geometry.shape, 96, output.palette === 'nom');
    if (note) output.notes.push(`Animation: ${note}`);
  }
  return {panels, output, projection, layout, thumbnail, width, height, grammar, frames};
}

function getMarkStyle(
  panel: ResultPanel,
  shape: CardMarkShape,
  value: number | undefined,
  model: ResultViewModel,
  role: CardRole = panel.role
) {
  const valued =
    Boolean(panel.values) &&
    role !== 'context' &&
    role !== 'highlight' &&
    (Boolean(panel.geometry) || role === 'output');
  const noData = valued && !Number.isFinite(value);
  const classIndex =
    !valued || noData
      ? undefined
      : panel.categories
        ? panel.categories.indexOf(value!)
        : panel.scale?.classFor(value!);
  const other = valued && panel.categories && classIndex === -1;
  return getCardMarkAttributes(other ? 'context' : role, shape, {
    palette: other ? undefined : panel.palette,
    classIndex: other ? undefined : classIndex,
    noData,
    scale: model.thumbnail ? CARD_FIGURE.thumbnailMarkScale : 1
  });
}

function drawMark(
  tag: string,
  attributes: ViewElementNode['attributes'],
  panel: ResultPanel,
  key: number,
  shape: CardMarkShape,
  model: ResultViewModel,
  context: ViewContext,
  value = panel.values ? getCaptureValue(panel.values, key) : undefined,
  role: CardRole = panel.role,
  selected = getPinnedEntity(context) === key
): ViewElementNode {
  const style = getMarkStyle(panel, shape, value, model, role);
  const provenance = provenanceMarkAttributes(
    getCaptureProvenance(panel.values ?? panel.capture),
    style['fill'],
    context.idPrefix
  );
  const interactive = style['data-role'] !== 'context' && role !== 'highlight';
  const mark = h(tag, {
    ...style,
    ...attributes,
    'data-provenance': provenance['data-provenance'],
    ...(interactive ? {'data-action': 'pick', 'data-axis': 'entity', 'data-key': key} : {})
  });
  const overlays: ViewElementNode[] = [];
  if (getCaptureProvenance(panel.values ?? panel.capture) === 'replay')
    overlays.push(
      h(tag, {
        ...attributes,
        ...(shape === 'point' ? {r: style['r']} : {}),
        fill: shape === 'line' ? 'none' : provenance['fill'],
        stroke: shape === 'line' ? provenance['fill'] : 'none',
        'stroke-width': style['stroke-width'],
        opacity: 0.55,
        'pointer-events': 'none',
        'data-provenance': 'replay'
      })
    );
  if (selected && interactive) {
    const ink = getSelectionInk(context.ground);
    for (const [state, color, width] of [
      ['selection-halo', ink.halo, 4],
      ['selection', ink.core, 2]
    ] as const)
      overlays.push(
        h(tag, {
          ...attributes,
          ...(shape === 'point' ? {r: style['r']} : {}),
          fill: 'none',
          stroke: color,
          'stroke-width': width,
          'data-state': state,
          'pointer-events': 'none'
        })
      );
  }
  return overlays.length ? h('g', null, mark, overlays) : mark;
}

function drawPanel(
  panel: ResultPanel,
  model: ResultViewModel,
  context: ViewContext,
  highlightOnly = false
): ViewElementNode[] {
  if (panel.unknown) return [];
  const project = (horizontal: number, vertical: number) =>
    model.projection.project(horizontal, vertical).map(value => Math.round(value * 100) / 100);
  const panelBudget = Math.max(
    3,
    Math.floor(
      (model.thumbnail ? 64 : 400) /
        (Math.max(1, model.panels.length) * Math.max(1, model.frames.length))
    )
  );
  const marks: ViewElementNode[] = [];
  const geometry = panel.geometry;
  if (!geometry) return marks;
  const highlight = getHighlight(panel);
  if (highlightOnly && !highlight) return marks;
  const role = highlightOnly ? 'highlight' : panel.role;
  const addMark = (
    tag: string,
    attributes: ViewElementNode['attributes'],
    key: number,
    shape: CardMarkShape,
    value?: number,
    selected?: boolean
  ) => {
    if (highlightOnly && key !== highlight!.key) return;
    marks.push(
      drawMark(
        tag,
        attributes,
        panel,
        key,
        shape,
        model,
        context,
        value ?? (panel.values ? getCaptureValue(panel.values, key) : undefined),
        role,
        selected
      )
    );
    if (highlightOnly && !model.thumbnail) {
      const point =
        panel.points[
          geometry.kind === 'lines' || geometry.kind === 'polygons'
            ? Number(panel.offsets?.values[key])
            : key
        ];
      const position = point
        ? project(...point)
        : [Number(attributes['x']) + Number(attributes['width'] ?? 0) / 2, Number(attributes['y'])];
      marks.push(
        h(
          'text',
          {
            class: 'cv-label cv-label--subject',
            x: position[0],
            y: position[1] - 8,
            'text-anchor': 'middle'
          },
          highlight!.label
        )
      );
    }
  };
  if (geometry.kind === 'raster') {
    const [columns, rows] = geometry.shape;
    const placement = getRasterPlacement(geometry);
    const [left, firstEdge] = project(placement.columns(0), placement.rows(0));
    const [right, lastEdge] = project(placement.columns(columns), placement.rows(rows));
    if (!highlightOnly) {
      marks.push(
        renderRasterImage({
          columns,
          rows,
          limit: model.thumbnail ? 64 : model.frames.length ? 96 : 160,
          valueAt: key => Number(panel.capture.values[key]),
          categorical: panel.palette === 'nom',
          flipRows: firstEdge > lastEdge,
          colorFor: value => String(getMarkStyle(panel, 'cell', value, model)['fill']),
          attributes: {
            ...getMarkStyle(panel, 'cell', 0, model),
            x: left,
            y: Math.min(firstEdge, lastEdge),
            width: right - left,
            height: Math.abs(lastEdge - firstEdge),
            'data-provenance': getCaptureProvenance(panel.capture)
          }
        })
      );
    }
    const selectedKey = highlightOnly ? highlight?.key : getPinnedEntity(context);
    if (selectedKey !== undefined && selectedKey >= 0 && selectedKey < columns * rows) {
      const column = selectedKey % columns;
      const row = Math.floor(selectedKey / columns);
      const [startX, startY] = project(placement.columns(column), placement.rows(row));
      const [endX, endY] = project(placement.columns(column + 1), placement.rows(row + 1));
      addMark(
        'rect',
        {
          x: startX,
          y: Math.min(startY, endY),
          width: endX - startX,
          height: Math.abs(endY - startY),
          fill: 'none'
        },
        selectedKey,
        'cell'
      );
    }
  } else if (geometry.kind === 'points' || geometry.kind === 'graph') {
    if (
      !highlightOnly &&
      geometry.kind === 'graph' &&
      panel.offsets &&
      panel.targets &&
      panel.points.length
    ) {
      let edgeCount = 0;
      const edgeStride = Math.max(
        1,
        Math.ceil(panel.targets.values.length / Math.floor(panelBudget / 2))
      );
      for (let node = 0; node < panel.offsets.values.length - 1; node++) {
        for (
          let edge = Number(panel.offsets.values[node]);
          edge < Number(panel.offsets.values[node + 1]);
          edge++
        ) {
          const from = panel.points[node];
          const to = panel.points[Number(panel.targets.values[edge])];
          if (
            edgeCount % edgeStride === 0 &&
            from?.every(Number.isFinite) &&
            to?.every(Number.isFinite)
          ) {
            const [startX, startY] = project(...from);
            const [endX, endY] = project(...to);
            marks.push(
              drawMark(
                'line',
                {x1: startX, y1: startY, x2: endX, y2: endY},
                panel,
                node,
                'line',
                model,
                context,
                undefined,
                panel.role === 'output' ? 'output' : 'context',
                false
              )
            );
          }
          edgeCount++;
        }
      }
    }
    const pointStride = Math.max(1, Math.ceil(panel.points.length / Math.min(2000, panelBudget)));
    panel.points.forEach((point, key) => {
      if ((!highlightOnly && key % pointStride !== 0) || !point.every(Number.isFinite)) return;
      const [horizontal, vertical] = project(...point);
      addMark('circle', {cx: horizontal, cy: vertical}, key, 'point');
    });
  } else if (panel.offsets) {
    const stride = Math.max(1, Math.ceil(panel.points.length / panelBudget));
    for (let feature = 0; feature < panel.offsets.values.length - 1; feature++) {
      const start = Number(panel.offsets.values[feature]);
      const vertices = panel.points
        .slice(Number(panel.offsets.values[feature]), Number(panel.offsets.values[feature + 1]))
        .filter((_, index) => (start + index) % stride === 0);
      if (
        vertices.length < (geometry.kind === 'polygons' ? 3 : 2) ||
        vertices.some(point => !point.every(Number.isFinite))
      )
        continue;
      const path =
        vertices
          .map((point, index) => `${index ? 'L' : 'M'}${project(...point).join(',')}`)
          .join(' ') + (geometry.kind === 'polygons' ? ' Z' : '');
      addMark(
        'path',
        {d: path, 'fill-rule': 'evenodd'},
        feature,
        geometry.kind === 'lines' ? 'line' : 'area'
      );
    }
  }
  return marks;
}

function renderAdjacency(panel: ResultPanel, model: ResultViewModel): ViewElementNode | null {
  if (
    panel.geometry?.kind !== 'graph' ||
    panel.points.length ||
    panel.unknown ||
    !panel.offsets ||
    !panel.targets
  )
    return null;
  const count = Number(panel.count);
  const limit = Math.min(count, model.thumbnail ? 8 : 32);
  return h(
    'div',
    null,
    h(
      'table',
      {class: 'lens-adjacency'},
      h('thead', null, h('tr', null, h('th', null, 'node'), h('th', null, 'neighbours'))),
      h(
        'tbody',
        null,
        Array.from({length: limit}, (_, node) => {
          const first = Number(panel.offsets!.values[node]);
          const last = Number(panel.offsets!.values[node + 1]);
          const neighbours = Array.from({length: last - first}, (_, index) =>
            Number(panel.targets!.values[first + index])
          );
          return h(
            'tr',
            {
              'data-action': 'pick',
              'data-axis': 'entity',
              'data-key': node,
              'data-provenance': getCaptureProvenance(panel.capture)
            },
            h('td', null, String(node)),
            h('td', null, neighbours.join(', ') || '—')
          );
        })
      )
    ),
    count > limit ? h('p', null, `${count - limit} more rows`) : null
  );
}

function getHighlight(panel: ResultPanel): {key: number; label: string} | undefined {
  if (panel.role !== 'output' || panel.palette === 'div') return undefined;
  return panel.capture.highlight;
}

/**
 * World coordinate of a raster's column and row edges. Without a recorded `origin` the raster
 * sits in its own cell space with row 0 at the top (`rows × cellSize`), as before placement existed.
 */
function getRasterPlacement(geometry: Extract<GPUComputationCaptureGeometry, {kind: 'raster'}>): {
  columns: (edge: number) => number;
  rows: (edge: number) => number;
} {
  const cellSize = geometry.cellSize ?? 1;
  const north = geometry.rowDirection === 'north';
  const [originX, originY] = geometry.origin ?? [0, north ? 0 : geometry.shape[1] * cellSize];
  return {
    columns: edge => originX + edge * cellSize,
    rows: edge => (north ? originY + edge * cellSize : originY - edge * cellSize)
  };
}

function renderLegend(model: ResultViewModel, figure: ViewElementNode): ViewElementNode {
  const valuedMarks = findViewNodes(figure, node =>
    String(node.attributes['class'] ?? '')
      .split(' ')
      .includes('cv-v')
  );
  const panel = model.output.values
    ? model.output
    : model.panels.find(candidate => candidate.values && candidate.role !== 'context');
  if (!panel || !valuedMarks.length) {
    const inputs = model.panels
      .filter(candidate => candidate.role === 'input')
      .map(candidate => candidate.capture.name);
    const outputs = model.panels
      .filter(candidate => candidate.role === 'output')
      .map(candidate => candidate.capture.name);
    return h(
      'p',
      {class: 'cv-legend'},
      [
        inputs.length ? `grey: input ${inputs.join(', ')}` : '',
        outputs.length ? `blue: output ${outputs.join(', ')}` : ''
      ]
        .filter(Boolean)
        .join(' · ')
    );
  }
  const labels =
    panel.categories?.map(String) ??
    panel.scale
      ?.legend()
      .classes.filter(entry => !entry.beyondRange)
      .slice(0, 5)
      .map(entry => entry.label) ??
    [];
  const colors = getCardPaletteColors(panel.palette, 'light');
  const noData = valuedMarks.some(mark => String(mark.attributes['class']).includes('cv-no-data'));
  const swatch = (className: string, fill: string, label: string) =>
    h(
      'span',
      null,
      h(
        'svg',
        {class: 'cv-legend__swatch', width: 12, height: 12, 'aria-hidden': true},
        h('rect', {class: `cv-v ${className}`, width: 12, height: 12, fill})
      ),
      ` ${label}`
    );
  return h(
    'p',
    {class: 'cv-legend'},
    `${panel.values!.name}: `,
    labels
      .slice(0, 5)
      .map((label, index) => swatch(`cv-${panel.palette}-${index + 1}`, colors[index], label)),
    noData ? swatch('cv-no-data', CARD_COLORS.light['no-data'], 'No data') : null
  );
}

function renderSvg(
  model: ResultViewModel,
  context: ViewContext,
  children: ViewElementNode[],
  panel = false
): ViewElementNode {
  const patterns = provenancePatternDefs(context.idPrefix, context.ground);
  for (const pattern of findViewNodes(patterns, node => node.tag === 'pattern')) {
    pattern.attributes['width'] = 3;
    pattern.attributes['height'] = 3;
  }
  return h(
    'svg',
    {
      class: `cv-figure${panel ? ' cv-panel' : ''}`,
      width: model.width,
      height: model.height,
      viewBox: `0 0 ${model.width} ${model.height}`,
      preserveAspectRatio: 'xMidYMid meet',
      'data-extent': JSON.stringify(model.projection.extent),
      role: 'img',
      'aria-label': `Result: ${model.output.capture.name}`
    },
    patterns,
    crispnessDefs(context.idPrefix),
    children
  );
}

function renderScaleTick(panel: ResultPanel, model: ResultViewModel): ViewElementNode[] {
  const geometry = panel.geometry;
  if (model.thumbnail || geometry?.kind !== 'raster' || geometry.cellSize === undefined) return [];
  const start = model.projection.project(0, 0);
  const end = model.projection.project(geometry.cellSize, 0);
  const cellWidth = Math.abs(end[0] - start[0]);
  if (!cellWidth) return [];
  const multiple = Math.max(1, Math.ceil(12 / cellWidth));
  const left = CARD_FIGURE.padding;
  const bottom = model.height - 12;
  return [
    h('line', {
      ...getCardMarkAttributes('context', 'line'),
      x1: left,
      y1: bottom,
      x2: left + cellWidth * multiple,
      y2: bottom
    }),
    h(
      'text',
      {class: 'cv-label', x: left, y: bottom - 4},
      `${Number((geometry.cellSize * multiple).toPrecision(3))} ${geometry.unit ?? '(unit unknown)'}`
    )
  ];
}

function renderSpatialGroups(
  panels: ResultPanel[],
  model: ResultViewModel,
  context: ViewContext,
  output = model.output
): ViewElementNode[] {
  const groups = panels.map(panel =>
    h('g', {'data-role': panel.role}, drawPanel(panel, model, context))
  );
  groups.push(h('g', {'data-role': 'highlight'}, drawPanel(output, model, context, true)));
  return groups;
}

function renderSpatialFigure(model: ResultViewModel, context: ViewContext): ViewElementNode {
  const contexts = model.panels.filter(panel => panel.role === 'context');
  const roles = model.panels.filter(panel => panel.role !== 'context');
  if (model.thumbnail) {
    const output = model.output;
    if (
      output.geometry?.kind === 'graph' &&
      !output.points.length &&
      output.offsets &&
      output.targets &&
      !output.unknown
    ) {
      const count = Math.min(8, Number(output.count));
      const size = (model.height - 12) / Math.max(1, count);
      const left = (model.width - count * size) / 2;
      const marks: ViewElementNode[] = [];
      for (let row = 0; row < count; row++) {
        const neighbours = new Set<number>();
        for (
          let edge = Number(output.offsets.values[row]);
          edge < Number(output.offsets.values[row + 1]);
          edge++
        )
          neighbours.add(Number(output.targets.values[edge]));
        for (let column = 0; column < count; column++)
          marks.push(
            drawMark(
              'rect',
              {x: left + column * size, y: 6 + row * size, width: size, height: size},
              output,
              row,
              'cell',
              model,
              context,
              undefined,
              neighbours.has(column) ? 'output' : 'context',
              false
            )
          );
      }
      return renderSvg(model, context, marks);
    }
    return renderSvg(model, context, renderSpatialGroups([...contexts, output], model, context));
  }
  const renderOutput = (panel: ResultPanel, panelContext: ViewContext) =>
    model.frames.length && panel === model.output
      ? model.frames.map((frame, index) =>
          h(
            'g',
            {class: 'cv-frame', 'data-cv-frame': index, hidden: index !== model.frames.length - 1},
            renderSpatialGroups([frame], model, panelContext, frame)
          )
        )
      : panel === model.output
        ? renderSpatialGroups([panel], model, panelContext, panel)
        : [h('g', {'data-role': panel.role}, drawPanel(panel, model, panelContext))];
  if (model.layout === 'side-by-side')
    return h(
      'div',
      {class: 'cv-panels'},
      roles.map((panel, index) => {
        const panelContext = {...context, idPrefix: `${context.idPrefix}-panel-${index}`};
        return h(
          'figure',
          {class: 'lens-result-panel', 'data-role': panel.role},
          renderSvg(
            model,
            panelContext,
            [
              ...contexts.map(background =>
                h('g', {'data-role': 'context'}, drawPanel(background, model, panelContext))
              ),
              ...renderOutput(panel, panelContext),
              ...renderScaleTick(panel, model)
            ],
            true
          ),
          renderAdjacency(panel, model),
          h(
            'figcaption',
            {class: 'cv-panel-caption'},
            `${panel.role}: ${panel.capture.name} `,
            provenanceBadge(getCaptureProvenance(panel.values ?? panel.capture))
          )
        );
      })
    );
  return h(
    'div',
    null,
    renderSvg(model, context, [
      ...model.panels
        .filter(panel => panel !== model.output)
        .map(panel => h('g', {'data-role': panel.role}, drawPanel(panel, model, context))),
      ...renderOutput(model.output, context),
      ...renderScaleTick(model.output, model)
    ]),
    roles.map(panel => renderAdjacency(panel, model))
  );
}

function formatValue(value: number | undefined): string {
  if (value === undefined) return '—';
  if (!Number.isFinite(value)) return Number.isNaN(value) ? 'NaN' : value > 0 ? '∞' : '−∞';
  return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(3)));
}

function renderStrips(model: ResultViewModel, context: ViewContext): ViewElementNode {
  const rows = model.panels.filter(panel => !panel.geometry && panel.role !== 'context');
  const padding = model.thumbnail ? CARD_FIGURE.thumbnailPadding : CARD_FIGURE.stripPadding;
  const gutter = model.thumbnail ? 0 : 72;
  const rowHeight = model.thumbnail
    ? Math.min(28, (model.height - padding * 2) / rows.length - 4)
    : CARD_FIGURE.stripRowHeight;
  const renderRow = (panel: ResultPanel, row: number): ViewElementNode => {
    const count = panel.capture.values.length;
    const keys: (number | null)[] = model.thumbnail
      ? Array.from({length: Math.min(8, count)}, (_, index) => index)
      : count <= 16
        ? Array.from({length: count}, (_, index) => index)
        : [
            ...Array.from({length: 8}, (_, index) => index),
            null,
            ...Array.from({length: 8}, (_, index) => count - 8 + index)
          ];
    const cellWidth =
      (model.width - padding * 2 - gutter) /
      Math.max(
        1,
        ...rows.map(candidate =>
          Math.min(
            model.thumbnail ? 8 : candidate.capture.values.length > 16 ? 17 : 16,
            candidate.capture.values.length
          )
        )
      );
    const top = padding + row * (rowHeight + (model.thumbnail ? 4 : CARD_FIGURE.stripRowGap));
    const marks: ViewElementNode[] = [];
    if (!model.thumbnail)
      marks.push(
        h(
          'text',
          {
            class: 'cv-label',
            x: padding,
            y: top + rowHeight / 2,
            'font-size': 11,
            'dominant-baseline': 'central'
          },
          panel.capture.name
        )
      );
    keys.forEach((key, index) => {
      const left = padding + gutter + index * cellWidth;
      if (key === null) {
        marks.push(
          h('rect', {
            ...getCardMarkAttributes('context', 'cell'),
            x: left,
            y: top,
            width: cellWidth,
            height: rowHeight
          })
        );
        if (!model.thumbnail)
          marks.push(
            h(
              'text',
              {
                class: 'cv-cell-text',
                x: left + cellWidth / 2,
                y: top + rowHeight / 2,
                'font-size': 7
              },
              `… ${count - 16} more …`
            )
          );
        return;
      }
      marks.push(
        drawMark(
          'rect',
          {x: left, y: top, width: cellWidth, height: rowHeight},
          panel,
          key,
          'cell',
          model,
          context
        )
      );
      const highlight =
        panel.capture.name === model.output.capture.name ? getHighlight(panel) : undefined;
      if (highlight?.key === key) {
        marks.push(
          drawMark(
            'rect',
            {x: left, y: top, width: cellWidth, height: rowHeight},
            panel,
            key,
            'cell',
            model,
            context,
            undefined,
            'highlight',
            false
          )
        );
        if (!model.thumbnail)
          marks.push(
            h(
              'text',
              {
                class: 'cv-label cv-label--subject',
                x: left + cellWidth / 2,
                y: top - 4,
                'text-anchor': 'middle'
              },
              highlight.label
            )
          );
      }
      if (!model.thumbnail)
        marks.push(
          h(
            'text',
            {class: 'cv-cell-text', x: left + cellWidth / 2, y: top + rowHeight / 2},
            formatValue(getCaptureValue(panel.capture, key))
          )
        );
    });
    return h('g', {'data-role': panel.role}, marks);
  };
  const groups = rows.map((panel, row) =>
    panel === model.output && model.frames.length
      ? h(
          'g',
          {'data-role': panel.role},
          model.frames.map((frame, index) =>
            h(
              'g',
              {
                class: 'cv-frame',
                'data-cv-frame': index,
                hidden: index !== model.frames.length - 1
              },
              renderRow(frame, row)
            )
          )
        )
      : renderRow(panel, row)
  );
  return renderSvg(model, context, groups);
}

function renderTable(model: ResultViewModel, context: ViewContext): ViewElementNode {
  const columns = model.panels.filter(panel => !panel.geometry && panel.role !== 'context');
  const rowCount = Math.max(...columns.map(panel => panel.capture.values.length));
  const count = Math.min(6, rowCount);
  if (model.thumbnail)
    return renderSvg(
      model,
      context,
      Array.from({length: count}, (_, row) =>
        h(
          'g',
          null,
          columns.map((panel, column) =>
            h('rect', {
              ...getCardMarkAttributes(panel.role, 'cell', {scale: CARD_FIGURE.thumbnailMarkScale}),
              x: 6 + (column * 148) / columns.length,
              y: 6 + row * 14,
              width: 148 / columns.length,
              height: 14,
              'data-provenance': getCaptureProvenance(panel.capture)
            })
          )
        )
      )
    );
  const columnAttributes = (panel: ResultPanel) => ({
    class: panel.role === 'output' ? 'cv-col--output' : undefined
  });
  return h(
    'table',
    {class: 'cv-table'},
    h(
      'thead',
      null,
      h(
        'tr',
        null,
        columns.map(panel =>
          h('th', {...columnAttributes(panel), scope: 'col'}, panel.capture.name)
        )
      )
    ),
    h(
      'tbody',
      null,
      Array.from({length: count}, (_, row) =>
        h(
          'tr',
          null,
          columns.map(panel =>
            h(
              'td',
              {...columnAttributes(panel), 'data-provenance': getCaptureProvenance(panel.capture)},
              formatValue(getCaptureValue(panel.capture, row))
            )
          )
        )
      ),
      rowCount > count
        ? h('tr', null, h('td', {colspan: columns.length}, `… ${rowCount - count} more rows`))
        : null
    )
  );
}

function renderMotion(model: ResultViewModel, context: ViewContext): ViewElementNode[] {
  if (!model.frames.length || model.thumbnail) return [];
  const smallModel = {
    ...model,
    thumbnail: true,
    width: 96,
    height: 58,
    projection: createGeometryProjection({
      coordinateModel: model.projection.model,
      points: model.projection.extent,
      width: 96,
      height: 58,
      padding: 6
    })
  };
  return [
    h(
      'button',
      {type: 'button', class: 'cv-play', 'data-cv-play': true},
      context.reducedMotion ? 'Step ▸' : '▶ Play'
    ),
    h('span', {class: 'cv-play-status', 'aria-live': 'polite'}, 'final'),
    h(
      'div',
      {class: 'cv-key-frames'},
      sampleEvenly(model.frames, 6).map((frame, index, frames) => {
        const frameContext = {
          ...context,
          bindings: {},
          idPrefix: `${context.idPrefix}-key-frame-${index}`
        };
        return h(
          'figure',
          null,
          model.grammar === 'strips'
            ? renderStrips(
                {...smallModel, panels: [frame], output: frame, frames: []},
                frameContext
              )
            : renderSvg(smallModel, frameContext, [
                h('g', null, drawPanel(frame, smallModel, frameContext))
              ]),
          h(
            'figcaption',
            null,
            index === frames.length - 1 ? 'final' : `round ${frame.capture.iteration}`
          )
        );
      })
    )
  ];
}

/** Renders role-styled native geometry, strips or tables from recorded evidence only. */
export function renderResultView(model: ResultViewModel, context: ViewContext): ViewElementNode {
  const figure =
    model.grammar === 'geometry'
      ? renderSpatialFigure(model, context)
      : model.grammar === 'table'
        ? renderTable(model, context)
        : renderStrips(model, context);
  const notes = model.panels.flatMap(panel => [
    ...(panel.unknown ? [`unknown — ${panel.unknown}`] : []),
    ...panel.notes
  ]);
  if (model.grammar === 'geometry') notes.unshift(model.projection.statement);
  return h(
    'section',
    {class: 'lens-view lens-result-view'},
    figure,
    model.thumbnail
      ? null
      : [
          renderMotion(model, context),
          renderLegend(model, figure),
          // The adapter line belongs to the host caption (twin provenance, lens footer).
          notes.length ? h('p', {class: 'cv-note'}, notes.join(' · ')) : null
        ]
  );
}

/** Public native-geometry result rung for tagged input and output captures. */
export const RESULT_VIEW: ViewRegistration<ResultViewModel> = {
  id: 'result-view',
  label: 'Result',
  rung: 'result',
  recordType: 'record',
  order: 50,
  signature: {binds: [], free: ['entity'], layout: {entity: 'position'}},
  descriptor: {audience: 'public', task: 'present', interaction: 'low'},
  level: 'realized',
  grain: 'contributor',
  satisfies: context =>
    getResultCaptures(context.record).some(capture => capture.role === 'output')
      ? {ok: true}
      : {
          ok: false,
          reason:
            'no captures tagged role: output (set capture.role, and capture.geometry for spatial results, when recording)'
        },
  build: buildResultViewModel,
  render: renderResultView,
  describe: (model, context) => ({
    subject: `Result of ${context.record.id}: ${
      model.panels
        .filter(panel => panel.role === 'input')
        .map(panel => panel.capture.name)
        .join(', ') || 'no recorded inputs'
    } → ${model.output.capture.name}`,
    keyNumber: {label: model.output.noun, value: model.output.count},
    facts: [
      ...model.panels
        .filter(panel => panel.role === 'input')
        .map(panel => `input ${panel.capture.name}: ${panel.count} ${panel.noun}`),
      model.projection.statement,
      ...model.panels.flatMap(panel => panel.notes)
    ],
    provenance: getCaptureProvenance(model.output.capture),
    adapterLabel: getAdapterLabel(context.record)
  })
};

let mountCounter = 0;

/** Mounts a host-owned result view, with replacement updates and explicit teardown. */
export function mountResultView(
  element: HTMLElement,
  record: LensRecord,
  options: Partial<Omit<ViewContext, 'record'>> = {}
): {update(record: LensRecord): void; destroy(): void} {
  const defaults = {
    bindings: {},
    ground: 'light' as const,
    idPrefix: `lens-result-view-${++mountCounter}`,
    reducedMotion: false,
    ...options
  };
  const update = (nextRecord: LensRecord) => {
    const context: ViewContext = {...defaults, record: nextRecord};
    const satisfaction = RESULT_VIEW.satisfies(context);
    const tree = !('reason' in satisfaction)
      ? renderResultView(buildResultViewModel(context), context)
      : h('p', {class: 'lens-disabled'}, satisfaction.reason);
    element.replaceChildren(viewNodeToDom(tree, element.ownerDocument));
  };
  update(record);
  return {update, destroy: () => element.replaceChildren()};
}
