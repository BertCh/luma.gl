// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationCapture} from '@luma.gl/gpgpu';

import {
  provenanceBadge,
  provenancePatternDefs
} from '../../../computation-views/encoding/provenance';
import type {
  ViewContext,
  ViewRegistration
} from '../../../computation-views/registry/view-registration';
import type {MarkProvenance} from '../../../computation-views/types';
import {h, type ViewElementNode} from '../../../computation-views/view-node';
import {
  formatViewNumber,
  getAdapterLabel,
  getAllCaptures,
  getCaptureProvenance,
  getCaptureValue,
  getPinnedEntity
} from '../../../computation-views/views/record-queries';

/** D8 directions in kernel order (`getD8ColumnOffset`/`getD8RowOffset`, rows advance south). */
const D8_DIRECTIONS = [
  {label: 'E', column: 1, row: 0},
  {label: 'SE', column: 1, row: 1},
  {label: 'S', column: 0, row: 1},
  {label: 'SW', column: -1, row: 1},
  {label: 'W', column: -1, row: 0},
  {label: 'NW', column: -1, row: -1},
  {label: 'N', column: 0, row: -1},
  {label: 'NE', column: 1, row: -1}
] as const;

/** One neighbour of the pinned cell. */
export type D8StencilNeighbour = {
  direction: number;
  label: string;
  cell: number;
  elevation?: number;
  /** 1 for cardinal moves, √2 for diagonal ones (in cell sizes). */
  distanceFactor: number;
  /** Ground distance in the grid's unit. */
  distance: number;
  /** Elevation drop divided by distance (positive = downhill), derived from captured elevations. */
  gradient?: number;
};

/** Model of the D8 stencil receipt. */
export type D8StencilModel = {
  entity: number;
  column: number;
  row: number;
  width: number;
  cellSize: number;
  unit: string;
  surfaceName: string;
  surfaceProvenance: MarkProvenance;
  centerElevation?: number;
  neighbours: D8StencilNeighbour[];
  /** Direction chosen by the GPU (from the captured ESRI code), when one was captured. */
  chosenDirection?: number;
  /** Raw captured code when it is not a single D8 power of two (no receiver, unresolved). */
  chosenCode?: number;
  directionName?: string;
};

function isRasterCapture(capture: GPUComputationCapture): boolean {
  return capture.geometry?.kind === 'raster' && capture.iteration === undefined;
}

/** The elevation surface the D8 drops are taken on: filled elevation when captured, else the input. */
function getSurfaceCapture(context: ViewContext): GPUComputationCapture | undefined {
  const captures = getAllCaptures(context.record).filter(isRasterCapture);
  const option = context.options?.['surface'];
  return (
    (typeof option === 'string' ? captures.find(capture => capture.name === option) : undefined) ??
    captures.find(capture => capture.meaning?.startsWith('filled elevation')) ??
    captures.find(capture => capture.role === 'input' && capture.meaning?.startsWith('elevation'))
  );
}

/** The captured D8 code raster (`meaning` starts with `D8`). */
function getDirectionCapture(context: ViewContext): GPUComputationCapture | undefined {
  return getAllCaptures(context.record).find(
    capture => isRasterCapture(capture) && capture.meaning?.startsWith('D8')
  );
}

/** Builds the stencil from captured values only; gradients are arithmetic on readbacks. */
export function buildD8StencilModel(context: ViewContext): D8StencilModel {
  const entity = getPinnedEntity(context)!;
  const surface = getSurfaceCapture(context)!;
  const geometry = surface.geometry;
  // satisfies() admits only raster surfaces, which always carry a shape.
  const [width, height] = geometry?.kind === 'raster' ? geometry.shape : [1, 1];
  const cellSize = geometry?.kind === 'raster' ? (geometry.cellSize ?? 1) : 1;
  const unit = geometry?.kind === 'raster' ? (geometry.unit ?? 'cell') : 'cell';
  const column = entity % width;
  const row = Math.floor(entity / width);
  const centerElevation = getCaptureValue(surface, entity);
  const neighbours = D8_DIRECTIONS.map((direction, index) => {
    const neighbourColumn = column + direction.column;
    const neighbourRow = row + direction.row;
    const inside =
      neighbourColumn >= 0 && neighbourRow >= 0 && neighbourColumn < width && neighbourRow < height;
    const cell = neighbourRow * width + neighbourColumn;
    const distanceFactor = direction.column !== 0 && direction.row !== 0 ? Math.SQRT2 : 1;
    const elevation = inside ? getCaptureValue(surface, cell) : undefined;
    const distance = distanceFactor * cellSize;
    return {
      direction: index,
      label: direction.label,
      cell,
      elevation,
      distanceFactor,
      distance,
      gradient:
        elevation !== undefined && centerElevation !== undefined && Number.isFinite(elevation)
          ? (centerElevation - elevation) / distance
          : undefined
    };
  });
  const directions = getDirectionCapture(context);
  const code = directions ? getCaptureValue(directions, entity) : undefined;
  const chosen =
    code !== undefined && code > 0 && Number.isInteger(Math.log2(code)) && code <= 128
      ? Math.log2(code)
      : undefined;
  return {
    entity,
    column,
    row,
    width,
    cellSize,
    unit,
    surfaceName: surface.meaning ?? surface.name,
    surfaceProvenance: getCaptureProvenance(surface),
    centerElevation,
    neighbours,
    ...(chosen !== undefined && {chosenDirection: chosen}),
    ...(code !== undefined && chosen === undefined && {chosenCode: code}),
    ...(directions && {directionName: directions.meaning ?? directions.name})
  };
}

const CELL = 64;

/** Draws the 3×3 stencil: elevations, the √2 diagonal distance and the GPU-chosen receiver. */
export function renderD8Stencil(model: D8StencilModel, context: ViewContext): ViewElementNode {
  const size = CELL * 3;
  const cells: ViewElementNode[] = [];
  const centre = {x: CELL * 1.5, y: CELL * 1.5};
  const ordered = [
    {column: 0, row: 0, label: 'C', elevation: model.centerElevation} as const,
    ...model.neighbours.map(neighbour => {
      const offset = D8_DIRECTIONS[neighbour.direction];
      return {...neighbour, column: offset.column, row: offset.row};
    })
  ];
  for (const cell of ordered) {
    const x = (cell.column + 1) * CELL;
    const y = (cell.row + 1) * CELL;
    const isCentre = cell.column === 0 && cell.row === 0;
    const neighbour = 'direction' in cell ? cell : undefined;
    const chosen = neighbour !== undefined && neighbour.direction === model.chosenDirection;
    cells.push(
      h(
        'g',
        {
          'data-action': neighbour ? 'pick' : undefined,
          'data-axis': neighbour ? 'entity' : undefined,
          'data-key': neighbour ? neighbour.cell : undefined
        },
        h('rect', {
          x: x + 1,
          y: y + 1,
          width: CELL - 2,
          height: CELL - 2,
          fill: 'none',
          stroke: 'currentColor',
          'stroke-opacity': isCentre || chosen ? 1 : 0.2,
          'stroke-width': chosen ? 2.5 : 1,
          class: chosen ? 'cv-mark cv-mark--highlight' : 'cv-mark cv-mark--context'
        }),
        h(
          'text',
          {x: x + CELL / 2, y: y + 24, 'text-anchor': 'middle', class: 'cv-label'},
          cell.elevation === undefined ? '—' : formatViewNumber(cell.elevation)
        ),
        neighbour
          ? h(
              'text',
              {x: x + CELL / 2, y: y + 44, 'text-anchor': 'middle', class: 'cv-label'},
              `${neighbour.label} ×${neighbour.distanceFactor === 1 ? '1' : '1.414'}`
            )
          : h(
              'text',
              {x: x + CELL / 2, y: y + 44, 'text-anchor': 'middle', class: 'cv-label'},
              'centre'
            )
      )
    );
  }
  if (model.chosenDirection !== undefined) {
    const offset = D8_DIRECTIONS[model.chosenDirection];
    cells.push(
      h('line', {
        x1: centre.x,
        y1: centre.y,
        x2: centre.x + offset.column * CELL * 0.6,
        y2: centre.y + offset.row * CELL * 0.6,
        stroke: 'currentColor',
        'stroke-width': 2,
        class: 'cv-mark cv-mark--highlight'
      })
    );
  }
  const rows = model.neighbours.map(neighbour =>
    h(
      'tr',
      {'data-key': neighbour.cell},
      h('td', null, neighbour.label),
      h(
        'td',
        null,
        neighbour.elevation === undefined ? 'outside grid' : formatViewNumber(neighbour.elevation)
      ),
      h(
        'td',
        null,
        `${neighbour.distanceFactor === 1 ? '1' : '√2 ≈ 1.414'} × ${formatViewNumber(model.cellSize)} ${model.unit}`
      ),
      h(
        'td',
        null,
        neighbour.gradient === undefined ? '—' : formatViewNumber(neighbour.gradient),
        ' ',
        provenanceBadge('derived')
      ),
      h('td', null, neighbour.direction === model.chosenDirection ? 'receiver (GPU)' : '')
    )
  );
  return h(
    'section',
    {class: 'lens-view lens-d8-stencil'},
    h(
      'svg',
      {
        class: 'cv-figure',
        viewBox: `0 0 ${size} ${size}`,
        role: 'img',
        'aria-label': `D8 stencil of cell ${model.entity}`
      },
      provenancePatternDefs(context.idPrefix, context.ground),
      ...cells
    ),
    h(
      'table',
      {class: 'lens-receipt'},
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          h('th', null, 'neighbour'),
          h('th', null, 'elevation'),
          h('th', null, 'distance'),
          h('th', null, 'drop / distance'),
          h('th', null, 'chosen')
        )
      ),
      h('tbody', null, ...rows)
    ),
    h(
      'p',
      {class: 'lens-caption'},
      'D8 sends all of a cell’s input to one neighbour; diagonals are √2 ≈ 1.414 cells away, not 1. ',
      model.chosenDirection === undefined
        ? h(
            'span',
            {class: 'lens-unknown'},
            model.directionName === undefined
              ? 'unknown — no D8 direction captured for this run'
              : `no single receiver (code ${model.chosenCode ?? 'none'})`
          )
        : `GPU receiver: ${D8_DIRECTIONS[model.chosenDirection].label}.`
    ),
    h('p', {class: 'lens-caption'}, getAdapterLabel(context.record))
  );
}

/** Datum view for D8 routing (misconception M14: diagonal moves are √2 cells long). */
export const D8_STENCIL_VIEW: ViewRegistration<D8StencilModel> = {
  id: 'terrain-d8-stencil',
  label: 'D8 stencil',
  rung: 'datum',
  recordType: 'datum',
  order: 10,
  signature: {binds: ['entity'], free: ['neighbour'], layout: {neighbour: 'position'}},
  descriptor: {audience: 'analyst', task: 'explain', interaction: 'low'},
  level: 'realized',
  grain: 'datum',
  isActive: context => getDirectionCapture(context) !== undefined,
  satisfies: context => {
    const entity = getPinnedEntity(context);
    if (entity === undefined)
      return {ok: false, reason: 'pin a cell (datum=<n>) to open its D8 stencil'};
    const surface = getSurfaceCapture(context);
    if (!surface) return {ok: false, reason: 'no elevation surface captured as a raster'};
    return getCaptureValue(surface, entity) === undefined
      ? {ok: false, reason: `cell ${entity} is outside the captured raster`}
      : {ok: true};
  },
  build: buildD8StencilModel,
  render: renderD8Stencil,
  receipt: model => ({
    columns: ['neighbour', 'elevation', 'distance', 'drop / distance', 'chosen'],
    rows: model.neighbours.map(neighbour => [
      neighbour.label,
      neighbour.elevation === undefined ? 'outside grid' : formatViewNumber(neighbour.elevation),
      `${neighbour.distanceFactor === 1 ? '1' : '1.414'} × ${formatViewNumber(model.cellSize)} ${model.unit}`,
      neighbour.gradient === undefined ? '—' : formatViewNumber(neighbour.gradient),
      neighbour.direction === model.chosenDirection ? 'receiver' : 'no'
    ]),
    provenance: model.neighbours.map(() => 'derived' as const)
  }),
  describe: (model, context) => ({
    subject: `D8 stencil of cell ${model.entity} (column ${model.column}, row ${model.row})`,
    keyNumber: {
      label: 'receiver',
      value:
        model.chosenDirection === undefined ? 'unknown' : D8_DIRECTIONS[model.chosenDirection].label
    },
    facts: [
      `diagonal neighbours are 1.414 cell sizes away (${formatViewNumber(model.cellSize * Math.SQRT2)} ${model.unit})`,
      `drops are taken on ${model.surfaceName}`,
      'D8 assumes all input becomes runoff to a single neighbour'
    ],
    provenance: model.surfaceProvenance,
    adapterLabel: getAdapterLabel(context.record)
  })
};
