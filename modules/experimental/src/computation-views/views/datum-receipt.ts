// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import type {
  LensRecord,
  ViewContext,
  ViewRegistration,
  ViewNeighbourhoodFamily
} from '../registry/view-registration';
import type {MarkProvenance} from '../types';
import {h, rawHtml, viewNodeToDom, type ViewElementNode} from '../view-node';
import {
  provenanceBadge,
  provenanceMarkAttributes,
  provenancePatternDefs
} from '../encoding/provenance';
import {coarsenValues, crispnessAttributes, crispnessDefs} from '../encoding/crispness';
import {
  findNode,
  getAdapterLabel,
  getAllCaptures,
  getCaptureProvenance,
  getCaptureValue,
  getCapturesForKeySpace,
  getDatumKeySpace,
  getFormula,
  getNamespaceSegments,
  getPinnedEntity
} from './record-queries';
import {getFinalStateCapture, resolveValueScale} from './value-strip';
import {formatTexValue, getSymbolValueAt, substituteTexTerms} from './formula-tex';

type ReceiptRow = {
  captureId: string;
  name: string;
  stage: string;
  iteration?: number;
  afterNode?: string;
  value: string;
  provenance: MarkProvenance;
  meaning?: string;
  provisional: boolean;
};
type Neighbour = {key: number; value?: number};
type NeighbourhoodSlot = {
  kind: ViewNeighbourhoodFamily['kind'] | 'unknown';
  reason?: string;
  neighbours?: Neighbour[];
  cells?: {dx: number; dy: number; key?: number; value?: number; diagonalCost?: string}[];
  values?: {name: string; value: string; provenance: MarkProvenance}[];
};

/** Captured quantities of one datum, ordered by analysis stage and scheduled node. */
export type DatumReceiptModel = {
  entity: number;
  keySpace: string;
  keySpaceLabel: string;
  rows: ReceiptRow[];
  formula?: {formulaId: string; workedTex: string; ascii: string};
  neighbourhoods: NeighbourhoodSlot[];
};

function getComponents(capture: GPUComputationCapture, entity: number): number[] {
  const components = capture.components ?? 1;
  return Array.from(
    {length: Math.min(components, capture.values.length - entity * components)},
    (_, index) => Number(capture.values[entity * components + index])
  );
}

function getCaptureNode(record: LensRecord, capture: GPUComputationCapture) {
  return capture.afterNode
    ? findNode(
        record,
        capture.graph && !capture.afterNode.includes(':')
          ? `${capture.graph}:${capture.afterNode}`
          : capture.afterNode
      )
    : undefined;
}

function buildNeighbourhood(
  family: ViewNeighbourhoodFamily,
  context: ViewContext,
  entity: number,
  mainCapture: GPUComputationCapture,
  diagonalCost: string | number | boolean | undefined
): NeighbourhoodSlot {
  const captures = getAllCaptures(context.record);
  const findCapture = (reference: string) =>
    captures.find(capture => capture.id === reference) ??
    getFinalStateCapture(context.record, reference);
  const unknown = (reason: string): NeighbourhoodSlot => ({kind: 'unknown', reason});
  if (family.kind === 'none') return {kind: 'none'};
  if (family.kind === 'flag-offset-slot') {
    return {
      kind: family.kind,
      values: family.captures.map(name => {
        const capture = findCapture(name);
        const value = capture && getCaptureValue(capture, entity);
        return {
          name,
          value:
            value === undefined
              ? 'unknown — value not captured'
              : getComponents(capture!, entity).join(', '),
          provenance: capture && value !== undefined ? getCaptureProvenance(capture) : 'unknown'
        };
      })
    };
  }
  if (family.kind === 'stencil-3x3') {
    const shape = context.record.axes.keySpaces.find(
      keySpace => keySpace.id === family.keySpace
    )?.shape;
    if (!shape || !(shape[0] > 0 && shape[1] > 0)) return unknown('raster shape not captured');
    const [width, height] = shape;
    const column = entity % width;
    const row = Math.floor(entity / width);
    const cells: NonNullable<NeighbourhoodSlot['cells']> = [];
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const outside =
          column + dx < 0 || column + dx >= width || row + dy < 0 || row + dy >= height;
        const key = outside ? undefined : (row + dy) * width + column + dx;
        cells.push({
          dx,
          dy,
          key,
          value: key === undefined ? undefined : getCaptureValue(mainCapture, key),
          diagonalCost:
            dx !== 0 &&
            dy !== 0 &&
            (diagonalCost === true ||
              diagonalCost === 1.414 ||
              diagonalCost === '1.414' ||
              diagonalCost === Math.SQRT2)
              ? '1.414'
              : undefined
        });
      }
    }
    return {kind: family.kind, cells};
  }
  const relation = context.record.axes.relations.find(
    candidate => candidate.id === family.relation
  );
  if (!relation || relation.kind === 'unknown') return unknown('relation is unknown');
  const reference = relation.ref;
  if (!reference || !('layout' in reference) || !reference.capture)
    return unknown('relation values not captured');
  const targets = findCapture(reference.capture);
  if (!targets) return unknown('relation values not captured');
  let keys: number[];
  if (family.kind === 'csr-star') {
    const offsets = findCapture(`${reference.capture}-offsets`);
    if (reference.layout !== 'csr' || !offsets) return unknown('relation values not captured');
    const start = getCaptureValue(offsets, entity);
    const end = getCaptureValue(offsets, entity + 1);
    if (
      start === undefined ||
      end === undefined ||
      !Number.isInteger(start) ||
      !Number.isInteger(end) ||
      start < 0 ||
      end < start ||
      end > targets.values.length
    )
      return unknown('relation values not captured');
    keys = Array.from({length: end - start}, (_, index) => Number(targets.values[start + index]));
  } else {
    if (reference.layout !== 'index' || getCaptureValue(targets, entity) === undefined)
      return unknown('relation values not captured');
    keys = getComponents(targets, entity);
  }
  return {
    kind: family.kind,
    neighbours: keys.map(key => ({key, value: getCaptureValue(mainCapture, key)}))
  };
}

/** Collects only captured values, plus an optional formula and declared neighbourhood families. */
export function buildDatumReceiptModel(context: ViewContext): DatumReceiptModel {
  const entity = getPinnedEntity(context);
  const keySpace = getDatumKeySpace(context);
  // The registration requires a pinned datum with captured values in its key space.
  assert(entity !== undefined && keySpace !== undefined);
  const captures = getCapturesForKeySpace(context.record, keySpace).filter(
    capture => getCaptureValue(capture, entity) !== undefined
  );
  const stages = [...new Set((context.packs ?? []).flatMap(pack => pack.analysisStages ?? []))];
  const rows: ReceiptRow[] = captures.map(capture => {
    const node = getCaptureNode(context.record, capture)?.node;
    const stage =
      (context.packs ?? []).map(pack => pack.captureStages?.[capture.name]).find(Boolean) ??
      (node && getNamespaceSegments(node.namespace)[0]) ??
      'result';
    if (!stages.includes(stage)) stages.push(stage);
    return {
      captureId: capture.id,
      name: capture.name,
      stage,
      iteration: capture.iteration,
      afterNode: capture.afterNode,
      value: getComponents(capture, entity).join(', '),
      provenance: getCaptureProvenance(capture),
      meaning: capture.meaning,
      provisional:
        capture.iteration !== undefined &&
        (getFinalStateCapture(context.record, capture.name)?.iteration ?? Infinity) >
          capture.iteration
    };
  });
  const nodeOrder = (row: ReceiptRow) => {
    const capture = captures.find(candidate => candidate.id === row.captureId)!;
    const found = getCaptureNode(context.record, capture);
    return found ? found.graph.nodes.indexOf(found.node) : Number.MAX_SAFE_INTEGER;
  };
  rows.sort(
    (first, second) =>
      stages.indexOf(first.stage) - stages.indexOf(second.stage) ||
      nodeOrder(first) - nodeOrder(second) ||
      (first.iteration ?? Infinity) - (second.iteration ?? Infinity)
  );
  const writtenResources = new Set(
    captures
      .map(capture => capture.resource)
      .filter((resource): resource is string => Boolean(resource))
  );
  const formulaNode = context.record.graphs
    .flatMap(graph => graph.nodes)
    .find(
      node =>
        node.formula &&
        getFormula(context.record, node.formula) &&
        node.writes.some(resource => writtenResources.has(resource))
    );
  const formula = getFormula(context.record, formulaNode?.formula);
  const substitutions = new Map<string, string>();
  for (const symbol of formula?.symbols ?? []) {
    const captured = getSymbolValueAt(context.record, symbol, entity);
    if (captured) substitutions.set(symbol.id, formatTexValue(captured.value, symbol.role));
  }
  const mainCapture = captures.find(capture => capture.id === rows[0]?.captureId);
  // satisfies guarantees at least one captured value.
  assert(mainCapture);
  const neighbourhoods = (context.packs ?? []).flatMap(pack => {
    const family = pack.neighbourhoods?.[keySpace];
    return family
      ? [
          buildNeighbourhood(
            family,
            context,
            entity,
            mainCapture,
            pack.metadata?.['stencilDiagonalCost']
          )
        ]
      : [];
  });
  return {
    entity,
    keySpace,
    keySpaceLabel:
      context.record.axes.keySpaces.find(candidate => candidate.id === keySpace)?.label ?? keySpace,
    rows,
    formula: formula
      ? {
          formulaId: formula.id,
          workedTex: substituteTexTerms(formula.tex, substitutions),
          ascii: formula.ascii
        }
      : undefined,
    neighbourhoods: neighbourhoods.length ? neighbourhoods : [{kind: 'none'}]
  };
}

function renderNeighbourhood(
  slot: NeighbourhoodSlot,
  model: DatumReceiptModel,
  context: ViewContext,
  index: number
): ViewElementNode {
  if (slot.kind === 'none') return h('p', null, 'no neighbourhood declared by the domain pack');
  if (slot.kind === 'unknown')
    return h('span', {class: 'lens-unknown'}, `unknown — ${slot.reason}`);
  if (slot.values)
    return h(
      'table',
      {class: 'lens-flag-offset-slot'},
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          slot.values.map(value => h('th', {scope: 'col'}, value.name))
        )
      ),
      h(
        'tbody',
        null,
        h(
          'tr',
          null,
          slot.values.map(value => h('td', null, value.value, provenanceBadge(value.provenance)))
        )
      )
    );
  if (slot.kind === 'candidate-list')
    return h(
      'ul',
      {class: 'lens-candidate-list'},
      (slot.neighbours ?? []).map(neighbour =>
        h(
          'li',
          null,
          h(
            'button',
            {
              type: 'button',
              'data-action': 'pick',
              'data-axis': 'entity',
              'data-key': neighbour.key
            },
            `${neighbour.key}: ${neighbour.value ?? 'unknown — value not captured'}`
          ),
          provenanceBadge(neighbour.value === undefined ? 'unknown' : model.rows[0].provenance)
        )
      )
    );
  const idPrefix = `${context.idPrefix}-neighbourhood-${index}`;
  const scale = resolveValueScale(context, model.rows[0].name)?.scale;
  const attributes = (value: number | undefined) =>
    provenanceMarkAttributes(
      value === undefined ? 'unknown' : model.rows[0].provenance,
      value === undefined
        ? 'currentColor'
        : Number.isFinite(value)
          ? (scale?.colorFor(value) ?? '#808080')
          : (scale?.noDataColor ?? '#808080'),
      idPrefix
    );
  const marks: ViewElementNode[] = [];
  if (slot.cells) {
    for (const cell of slot.cells) {
      const outside = cell.key === undefined;
      marks.push(
        h(
          'g',
          {
            'data-action': outside ? undefined : 'pick',
            'data-axis': outside ? undefined : 'entity',
            'data-key': cell.key
          },
          h('rect', {
            x: (cell.dx + 1) * 70 + 1,
            y: (cell.dy + 1) * 70 + 1,
            width: 68,
            height: 68,
            ...(outside
              ? {
                  ...provenanceMarkAttributes('metadata', 'currentColor', idPrefix),
                  fill: `url(#${idPrefix}-hatched)`,
                  'stroke-dasharray': '3 2'
                }
              : attributes(cell.value))
          }),
          h(
            'title',
            null,
            outside
              ? 'outside raster (edge truncated)'
              : `${cell.key}: ${cell.value ?? 'unknown — value not captured'}`
          ),
          outside
            ? null
            : h(
                'text',
                {x: (cell.dx + 1) * 70 + 6, y: (cell.dy + 1) * 70 + 25, fill: 'currentColor'},
                String(cell.key)
              ),
          !outside && cell.diagonalCost
            ? h(
                'text',
                {x: (cell.dx + 1) * 70 + 6, y: (cell.dy + 1) * 70 + 50, fill: 'currentColor'},
                cell.diagonalCost
              )
            : null
        )
      );
    }
  } else {
    const neighbours = slot.neighbours ?? [];
    const shown = neighbours.slice(0, 16);
    for (let position = 0; position < shown.length; position++) {
      const neighbour = shown[position];
      const angle = (position / shown.length) * Math.PI * 2;
      const horizontal = 105 + Math.cos(angle) * 78;
      const vertical = 105 + Math.sin(angle) * 78;
      marks.push(
        h('line', {
          x1: 105,
          y1: 105,
          x2: horizontal,
          y2: vertical,
          stroke: 'currentColor',
          opacity: 0.55
        }),
        h(
          'circle',
          {
            cx: horizontal,
            cy: vertical,
            r: 9,
            ...attributes(neighbour.value),
            'data-action': 'pick',
            'data-axis': 'entity',
            'data-key': neighbour.key
          },
          h('title', null, `${neighbour.key}: ${neighbour.value ?? 'unknown — value not captured'}`)
        )
      );
    }
    const centreValue = Number(model.rows[0].value.split(', ')[0]);
    marks.push(
      h(
        'circle',
        {cx: 105, cy: 105, r: 12, ...attributes(centreValue)},
        h('title', null, `${model.entity}: ${centreValue}`)
      )
    );
    if (neighbours.length > 16)
      marks.push(
        h('text', {x: 5, y: 225, fill: 'currentColor'}, `+${neighbours.length - 16} more`)
      );
  }
  return h(
    'svg',
    {
      class: `lens-neighbourhood lens-${slot.kind}`,
      width: 230,
      height: 230,
      viewBox: '0 0 230 230',
      role: 'img',
      'aria-label': `${slot.kind} neighbourhood`
    },
    provenancePatternDefs(idPrefix, context.ground),
    crispnessDefs(idPrefix),
    h(
      'g',
      crispnessAttributes(model.rows[0].provisional ? 'provisional' : 'final', idPrefix),
      marks
    )
  );
}

/** Renders staged receipt rows, the captured worked formula and neighbourhood evidence. */
export function renderDatumReceipt(
  model: DatumReceiptModel,
  context: ViewContext
): ViewElementNode {
  return h(
    'section',
    {class: 'lens-view lens-datum-receipt'},
    h('h3', null, `Receipt of ${model.keySpaceLabel} ${model.entity}`),
    h(
      'table',
      {class: 'lens-receipt'},
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          ['stage', 'quantity', 'value', 'iteration', 'provenance'].map(column =>
            h('th', {scope: 'col'}, column)
          )
        )
      ),
      [...new Set(model.rows.map(row => row.stage))].map(stage =>
        h(
          'tbody',
          {'data-stage': stage},
          h('tr', null, h('th', {scope: 'rowgroup', colspan: 5}, stage)),
          model.rows
            .filter(row => row.stage === stage)
            .map(row =>
              h(
                'tr',
                {
                  'data-provisional': row.provisional ? 'provisional' : undefined,
                  'data-provenance': row.provenance
                },
                h('td', null, stage),
                h(
                  'td',
                  null,
                  h(
                    'button',
                    {
                      type: 'button',
                      'data-action': 'bind',
                      'data-axis': 'buffer',
                      'data-key': row.captureId,
                      title: row.meaning
                    },
                    row.name
                  )
                ),
                h(
                  'td',
                  null,
                  row.provisional
                    ? coarsenValues(row.value.split(', ').map(Number), 'provisional').join(', ')
                    : row.value
                ),
                h('td', null, row.iteration === undefined ? '—' : String(row.iteration)),
                h('td', null, provenanceBadge(row.provenance))
              )
            )
        )
      )
    ),
    model.formula
      ? h(
          'div',
          {class: 'lens-receipt-formula'},
          h('h4', null, model.formula.formulaId),
          context.renderTeX
            ? rawHtml(
                'div',
                {class: 'lens-formula-tex'},
                context.renderTeX(model.formula.workedTex, {displayMode: true})
              )
            : h(
                'div',
                null,
                h('pre', {class: 'lens-tex'}, model.formula.workedTex),
                h('p', {class: 'lens-ascii'}, model.formula.ascii)
              )
        )
      : null,
    model.neighbourhoods.map((slot, index) => renderNeighbourhood(slot, model, context, index)),
    h('p', null, getAdapterLabel(context.record))
  );
}

/** Generic pinned-datum receipt registration. */
export const DATUM_RECEIPT_VIEW: ViewRegistration<DatumReceiptModel> = {
  id: 'datum-receipt',
  label: 'Datum receipt',
  rung: 'datum',
  recordType: 'datum',
  order: 50,
  signature: {
    binds: ['entity'],
    free: ['term', 'buffer', 'neighbour'],
    layout: {term: 'position', buffer: 'position', neighbour: 'position'}
  },
  descriptor: {audience: 'analyst', task: 'explain', interaction: 'medium'},
  level: 'realized',
  grain: 'datum',
  satisfies: context => {
    const entity = getPinnedEntity(context);
    if (entity === undefined)
      return {ok: false, reason: 'pin a datum (datum=<n>) to open its receipt'};
    const keySpace = getDatumKeySpace(context);
    return keySpace &&
      getCapturesForKeySpace(context.record, keySpace).some(
        capture => getCaptureValue(capture, entity) !== undefined
      )
      ? {ok: true}
      : {ok: false, reason: `no captured values for ${keySpace ?? 'unknown'} ${entity}`};
  },
  build: buildDatumReceiptModel,
  render: renderDatumReceipt,
  receipt: model => ({
    columns: ['stage', 'quantity', 'value', 'iteration', 'provenance'],
    rows: model.rows.map(row => [
      row.stage,
      row.name,
      row.provisional
        ? coarsenValues(row.value.split(', ').map(Number), 'provisional').join(', ')
        : row.value,
      row.iteration === undefined ? '—' : String(row.iteration),
      row.provenance
    ]),
    provenance: model.rows.map(row => row.provenance)
  }),
  describe: (model, context) => {
    const last = model.rows[model.rows.length - 1];
    return {
      subject: `Receipt of ${model.keySpaceLabel} ${model.entity}`,
      keyNumber: {label: last.name, value: last.value},
      facts: [...new Set(model.rows.map(row => row.stage))].map(
        stage => `${stage}: ${model.rows.filter(row => row.stage === stage).length} quantities`
      ),
      provenance: last.provenance,
      adapterLabel: getAdapterLabel(context.record)
    };
  }
};

let mountCounter = 0;

/** Mounts this view into a host-owned element; update replaces its tree and destroy clears it. */
export function mountDatumReceipt(
  element: HTMLElement,
  record: LensRecord,
  options: Partial<Omit<ViewContext, 'record'>> = {}
): {update(record: LensRecord): void; destroy(): void} {
  const defaults = {
    bindings: {},
    ground: 'light' as const,
    idPrefix: `lens-datum-receipt-${++mountCounter}`,
    reducedMotion: false,
    ...options
  };
  const update = (nextRecord: LensRecord) => {
    const context: ViewContext = {...defaults, record: nextRecord};
    const satisfaction = DATUM_RECEIPT_VIEW.satisfies(context);
    const tree = !('reason' in satisfaction)
      ? renderDatumReceipt(buildDatumReceiptModel(context), context)
      : h('p', {class: 'lens-disabled'}, satisfaction.reason);
    element.replaceChildren(viewNodeToDom(tree, element.ownerDocument));
  };
  update(record);
  return {update, destroy: () => element.replaceChildren()};
}
