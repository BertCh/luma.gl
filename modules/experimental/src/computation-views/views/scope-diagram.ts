// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {LensRecord, ViewContext, ViewRegistration} from '../registry/view-registration';
import type {OperatorDescriptor, OperatorReduction} from '../types';
import {h, viewNodeToDom, type ViewElementNode} from '../view-node';
import {provenanceMarkAttributes, provenancePatternDefs} from '../encoding/provenance';
import {crispnessDefs} from '../encoding/crispness';
import {
  getBoundNode,
  getDatumKeySpace,
  getMostConcreteGraph,
  getPinnedEntity
} from './record-queries';

/** Schematic scope geometry derived solely from an operator descriptor. */
export type ScopeDiagramModel = OperatorDescriptor & {
  op: string;
  grid: {size: 7; cells: {dx: number; dy: number; role: 'subject' | 'input' | 'context'}[]};
  caption: string;
};

const REDUCTION_GLYPHS: Record<OperatorReduction, string> = {
  none: '—',
  count: '#',
  sum: 'Σ',
  'weighted-mean': 'Σw·x / Σw',
  argmin: 'argmin',
  histogram: 'histogram',
  mode: 'mode',
  select: 'select',
  'scan-sort': 'scan / sort',
  solve: 'solve'
};

function chooseOperator(context: ViewContext): ({op: string} & OperatorDescriptor) | undefined {
  const bound = getBoundNode(context);
  const nodes =
    context.bindings.node !== undefined
      ? bound
        ? [bound.node]
        : []
      : (getMostConcreteGraph(context.record)?.nodes ?? []);
  for (const node of nodes) {
    if (!node.op) continue;
    for (const pack of context.packs ?? []) {
      const descriptor = pack.operators?.[node.op];
      if (descriptor) return {op: node.op, ...descriptor};
    }
  }
  return undefined;
}

/** Builds a seven-by-seven schematic, preserving the subject as a separate role. */
export function buildScopeDiagramModel(context: ViewContext): ScopeDiagramModel {
  const operator = chooseOperator(context);
  // A scope descriptor is required by satisfies.
  assert(operator);
  const cells: ScopeDiagramModel['grid']['cells'] = [];
  for (let dy = -3; dy <= 3; dy++) {
    for (let dx = -3; dx <= 3; dx++) {
      const distance = Math.max(Math.abs(dx), Math.abs(dy));
      let input = false;
      switch (operator.scope) {
        case 'local':
          break;
        case 'focal':
          input = distance <= 1;
          break;
        case 'block':
          input = dx >= 0 && dx < 3 && dy >= 0 && dy < 3;
          break;
        case 'zonal':
          input = dx + dy >= 0;
          break;
        case 'global':
        case 'solver':
          input = true;
          break;
        case 'frontier':
          input = distance === 1 || distance === 2;
          break;
        case 'pairwise':
          input = dx === 0 || dy === 0;
          break;
      }
      cells.push({dx, dy, role: distance === 0 ? 'subject' : input ? 'input' : 'context'});
    }
  }
  const detail =
    operator.scope === 'zonal'
      ? ' schematic zones.'
      : operator.scope === 'solver'
        ? ' iterates to a fixed point.'
        : '';
  return {
    ...operator,
    grid: {size: 7, cells},
    caption: `Schematic generated from scope × reduction, not this run's data.${detail}`
  };
}

/** Renders achromatic scope tiers and the declared reduction glyph. */
export function renderScopeDiagram(
  model: ScopeDiagramModel,
  context: ViewContext
): ViewElementNode {
  const keySpace = context.record.axes.keySpaces.find(
    candidate => candidate.id === getDatumKeySpace(context)
  );
  const entity = getPinnedEntity(context);
  const width = keySpace?.shape?.[0];
  return h(
    'section',
    {class: 'lens-view lens-scope-diagram'},
    h('h3', null, model.op),
    h(
      'svg',
      {
        viewBox: '0 0 210 210',
        width: 210,
        height: 210,
        role: 'img',
        class: 'lens-provenance--metadata',
        'aria-label': `scope ${model.scope} × reduction ${model.reduction}`
      },
      provenancePatternDefs(context.idPrefix, context.ground),
      crispnessDefs(context.idPrefix),
      h('title', null, `scope ${model.scope} × reduction ${model.reduction}`),
      model.grid.cells.map(cell =>
        h('rect', {
          x: (cell.dx + 3) * 30 + 1,
          y: (cell.dy + 3) * 30 + 1,
          width: 28,
          height: 28,
          ...provenanceMarkAttributes('metadata', 'currentColor', context.idPrefix),
          opacity: cell.role === 'subject' ? 1 : cell.role === 'input' ? 0.55 : 0.2,
          'stroke-width': cell.role === 'subject' ? 4 : 2,
          'data-role': cell.role,
          'data-ring':
            model.scope === 'frontier' && cell.role === 'input'
              ? Math.max(Math.abs(cell.dx), Math.abs(cell.dy))
              : undefined
        })
      )
    ),
    h('p', {class: 'lens-reduction'}, REDUCTION_GLYPHS[model.reduction]),
    h('p', null, model.caption),
    entity !== undefined && keySpace?.indexSpace === '2d' && width
      ? h('p', null, `Applies to cell (${entity % width}, ${Math.floor(entity / width)})`)
      : null
  );
}

/** Generic metadata-only scope registration. */
export const SCOPE_DIAGRAM_VIEW: ViewRegistration<ScopeDiagramModel> = {
  id: 'scope-diagram',
  label: 'Scope',
  rung: 'scope',
  recordType: 'operator',
  order: 50,
  signature: {binds: [], free: ['input'], layout: {input: 'position'}},
  descriptor: {audience: 'analyst', task: 'explain', interaction: 'low'},
  level: 'specification',
  grain: 'datum',
  satisfies: context =>
    chooseOperator(context)
      ? {ok: true}
      : {
          ok: false,
          reason: 'no operator with a scope × reduction descriptor (domain pack operators table)'
        },
  build: buildScopeDiagramModel,
  render: renderScopeDiagram,
  describe: model => ({
    subject: `${model.op}: ${model.scope} scope, ${model.reduction} reduction`,
    keyNumber: {
      label: 'input cells',
      value: model.grid.cells.filter(cell => cell.role === 'input').length
    },
    facts: [model.caption],
    provenance: 'metadata'
  })
};

let mountCounter = 0;

/** Mounts this view into a host-owned element; update replaces its tree and destroy clears it. */
export function mountScopeDiagram(
  element: HTMLElement,
  record: LensRecord,
  options: Partial<Omit<ViewContext, 'record'>> = {}
): {update(record: LensRecord): void; destroy(): void} {
  const defaults = {
    bindings: {},
    ground: 'light' as const,
    idPrefix: `lens-scope-diagram-${++mountCounter}`,
    reducedMotion: false,
    ...options
  };
  const update = (nextRecord: LensRecord) => {
    const context: ViewContext = {...defaults, record: nextRecord};
    const satisfaction = SCOPE_DIAGRAM_VIEW.satisfies(context);
    const tree = !('reason' in satisfaction)
      ? renderScopeDiagram(buildScopeDiagramModel(context), context)
      : h('p', {class: 'lens-disabled'}, satisfaction.reason);
    element.replaceChildren(viewNodeToDom(tree, element.ownerDocument));
  };
  update(record);
  return {update, destroy: () => element.replaceChildren()};
}
