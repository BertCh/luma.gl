// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUComputationEdgeKind} from '@luma.gl/gpgpu';
import type {
  LensReceiptTable,
  LensRecord,
  ViewContext,
  ViewRegistration
} from '../registry/view-registration';
import {h, viewNodeToDom, type ViewElementNode} from '../view-node';
import {
  provenanceBadge,
  provenanceMarkAttributes,
  provenancePatternDefs
} from '../encoding/provenance';
import {crispnessDefs} from '../encoding/crispness';
import {
  getAdapterLabel,
  getAllCaptures,
  getCaptureProvenance,
  getGraphByForm
} from './record-queries';
import {renderValueStrip, resolveValueScale} from './value-strip';
import {foldGraphByDegreeOfInterest, type FoldedGraph, type GraphUnit} from './doi-folding';
import {
  buildMemoryAliasing,
  buildPassResourceLadder,
  type CommandGraphPreset,
  type MemoryAliasing,
  type PassResourceLadder
} from './graph-presets';

/** Plain-data projection of one selected command graph form and presentation preset. */
export type CommandGraphViewModel = {
  graphId: string;
  graphLabel: string;
  form: 'compiled' | 'logical';
  level: 'execution' | 'skeleton';
  preset: CommandGraphPreset;
  folded: FoldedGraph | undefined;
  ladder?: PassResourceLadder;
  aliasing?: MemoryAliasing;
  focus?: {unitId: string; derivedFrom: string[]; realizedBy: string[]};
  previews: {
    unitId: string;
    scalars: {name: string; value: number}[];
    thumbnail?: {captureId: string};
    inspector: {unitId: string; label: string};
    hasCaptures: boolean;
  }[];
  hasLogical: boolean;
  hasCompiled: boolean;
  counts: {nodes: number; edges: number; resources: number};
  edgeKinds: Partial<Record<GPUComputationEdgeKind, number>>;
};

const PRESETS: readonly CommandGraphPreset[] = ['algorithm', 'pass-resource', 'memory-aliasing'];
const TIER_OPACITY = {subject: 1, input: 0.55, context: 0.2};
let mountCount = 0;

/** Builds a graph view without modifying the record or choosing one member of a cross-form mapping. */
export function buildCommandGraphViewModel(context: ViewContext): CommandGraphViewModel {
  const logical = getGraphByForm(context.record, ['intent', 'lowered']);
  const compiled = getGraphByForm(context.record, ['compiled', 'encoded']);
  const graph =
    context.options?.['form'] === 'logical' ? (logical ?? compiled) : (compiled ?? logical);
  // The registration's satisfaction guard requires at least one graph.
  assert(graph);
  const form = graph.form === 'intent' || graph.form === 'lowered' ? 'logical' : 'compiled';
  const preset =
    PRESETS.find(candidate => candidate === context.options?.['preset']) ?? 'algorithm';
  const focusOption = context.bindings.node ?? context.options?.['focus'];
  const focus =
    typeof focusOption === 'string'
      ? focusOption.startsWith(`${graph.id}:`)
        ? focusOption.slice(graph.id.length + 1)
        : focusOption
      : undefined;
  const budget = context.options?.['budget'];
  // ViewContext options are scalar; hosts serialize the expanded path list as JSON.
  const expanded = parseExpanded(context.options?.['expanded']);
  const folded = foldGraphByDegreeOfInterest(graph, {
    focus,
    budget: typeof budget === 'number' ? budget : undefined,
    foldLoops: context.options?.['foldLoops'] !== false,
    expanded
  });
  const focusUnit = folded.units.find(unit => unit.id === folded.focusUnitId);
  const focusMembers = new Set(focusUnit?.memberIds);
  const references = new Set([...focusMembers].map(id => `${graph.id}:${id}`));
  const captures = getAllCaptures(context.record).filter(
    capture =>
      capture.graph === graph.id ||
      (!capture.graph &&
        context.record.graphs.filter(candidate =>
          candidate.nodes.some(node => node.id === capture.afterNode)
        ).length === 1)
  );
  const edgeKinds: CommandGraphViewModel['edgeKinds'] = {};
  for (const edge of graph.edges) edgeKinds[edge.kind] = (edgeKinds[edge.kind] ?? 0) + 1;
  return {
    graphId: graph.id,
    graphLabel: graph.label ?? graph.id,
    form,
    level: form === 'logical' ? 'skeleton' : 'execution',
    preset,
    folded,
    ladder: preset === 'pass-resource' ? buildPassResourceLadder(graph) : undefined,
    aliasing: preset === 'memory-aliasing' ? buildMemoryAliasing(graph) : undefined,
    focus: focusUnit
      ? {
          unitId: focusUnit.id,
          derivedFrom: [
            ...new Set(
              graph.nodes
                .filter(node => focusMembers.has(node.id))
                .flatMap(node => node.derivedFrom ?? [])
            )
          ],
          realizedBy: context.record.graphs
            .filter(candidate => candidate.form !== graph.form)
            .flatMap(candidate =>
              candidate.nodes
                .filter(node => node.derivedFrom?.some(reference => references.has(reference)))
                .map(node => `${candidate.id}:${node.id}`)
            )
        }
      : undefined,
    previews: folded.units.map(unit => {
      const members = new Set(unit.memberIds);
      const unitCaptures = captures.filter(
        capture => capture.afterNode && members.has(capture.afterNode)
      );
      const scalars = graph.nodes
        .filter(node => members.has(node.id))
        .flatMap(node =>
          [...Object.entries(node.attrs), ...Object.entries(node.workload ?? {})].flatMap(
            ([name, value]) =>
              typeof value === 'number'
                ? [{name: unit.memberIds.length > 1 ? `${node.label}: ${name}` : name, value}]
                : []
          )
        )
        .slice(0, 3);
      return {
        unitId: unit.id,
        scalars,
        thumbnail:
          unit.id === focusUnit?.id && unitCaptures.length
            ? {captureId: unitCaptures[0].id}
            : undefined,
        inspector: {unitId: unit.id, label: `Inspect ${unit.label}`},
        hasCaptures: unitCaptures.length > 0
      };
    }),
    hasLogical: Boolean(logical),
    hasCompiled: Boolean(compiled),
    counts: {
      nodes: graph.nodes.length,
      edges: graph.edges.length,
      resources: graph.resources?.length ?? 0
    },
    edgeKinds
  };
}

/** Renders a single compiler-order column, resource ladder, or allocation lifetime chart. */
export function renderCommandGraphView(
  model: CommandGraphViewModel,
  context: ViewContext
): ViewElementNode {
  return h(
    'section',
    {class: 'lens-view lens-command-graph-view', 'data-level': model.level},
    h(
      'div',
      {class: 'lens-graph-toolbar'},
      PRESETS.map(preset =>
        h(
          'button',
          {
            type: 'button',
            'data-action': 'toggle',
            'data-axis': 'preset',
            'data-value': preset,
            'aria-pressed': String(model.preset === preset)
          },
          preset
        )
      ),
      (['logical', 'compiled'] as const).map(form =>
        h(
          'button',
          {
            type: 'button',
            'data-action': 'toggle',
            'data-axis': 'form',
            'data-value': form,
            'aria-pressed': String(model.form === form),
            disabled: form === 'logical' ? !model.hasLogical : !model.hasCompiled,
            title: (form === 'logical' ? !model.hasLogical : !model.hasCompiled)
              ? `No ${form} graph is recorded`
              : undefined
          },
          form
        )
      )
    ),
    h(
      'p',
      {class: 'lens-caption'},
      model.form === 'compiled'
        ? `Compiled schedule on ${getAdapterLabel(context.record)} (retrospective).`
        : 'Logical graph (prospective): what was declared, before scheduling.'
    ),
    model.preset === 'algorithm'
      ? renderAlgorithm(model, context)
      : model.preset === 'pass-resource'
        ? renderLadder(model, context)
        : renderAliasing(model, context),
    model.focus
      ? h(
          'p',
          {class: 'lens-graph-lineage'},
          `Derived from: ${model.focus.derivedFrom.join(', ') || 'unknown'}. Realized by: ${model.focus.realizedBy.join(', ') || 'unknown'}.`
        )
      : undefined
  );
}

function renderAlgorithm(model: CommandGraphViewModel, context: ViewContext): ViewElementNode {
  const folded = model.folded!;
  const positions = new Map(folded.units.map((unit, index) => [unit.id, index * 64 + 24]));
  const units = new Map(folded.units.map(unit => [unit.id, unit]));
  const marks = folded.edges.flatMap(edge =>
    Object.entries(edge.kinds).flatMap(([kind, count], kindIndex) => {
      const from = positions.get(edge.from)!;
      const to = positions.get(edge.to)!;
      const reach = Math.min(96, 18 + Math.abs(to - from) / 6 + kindIndex * 5);
      const opacity = Math.min(
        TIER_OPACITY[units.get(edge.from)!.tier],
        TIER_OPACITY[units.get(edge.to)!.tier]
      );
      return Array.from({length: kind === 'indirect' ? 2 : 1}, (_, lineIndex) => {
        const start = 110 - lineIndex * 4;
        return h(
          'path',
          {
            ...provenanceMarkAttributes('metadata', 'currentColor', context.idPrefix),
            d: `M ${start} ${from} C ${start - reach} ${from}, ${start - reach} ${to}, ${start} ${to}`,
            opacity,
            'stroke-width': kind === 'dataflow' ? Math.min(3, count) * 0.75 : Math.min(3, count),
            'stroke-dasharray':
              kind === 'war' || kind === 'waw' ? '5 3' : kind === 'explicit' ? '1 3' : undefined,
            'data-kind': kind,
            'data-count': count
          },
          h(
            'title',
            null,
            `${edge.from} → ${edge.to}: ${kind} ×${count}; ${edge.resources.join(', ') || '—'}`
          )
        );
      });
    })
  );
  return h(
    'div',
    {
      class: 'lens-graph-algorithm',
      style: 'display:grid;grid-template-columns:120px minmax(0,1fr)'
    },
    h(
      'svg',
      {
        width: 120,
        height: Math.max(1, folded.units.length) * 64,
        style: `grid-column:1;grid-row:1 / span ${Math.max(1, folded.units.length)}`,
        role: 'img',
        'aria-label': 'Dependencies in graph node order'
      },
      provenancePatternDefs(context.idPrefix, context.ground),
      crispnessDefs(context.idPrefix),
      marks
    ),
    folded.units.map((unit, index) => renderUnit(unit, model, context, index)),
    !folded.units.length ? h('p', null, 'No nodes') : undefined
  );
}

function renderUnit(
  unit: GraphUnit,
  model: CommandGraphViewModel,
  context: ViewContext,
  index: number
): ViewElementNode {
  const preview = model.previews.find(candidate => candidate.unitId === unit.id)!;
  const capture = preview.thumbnail
    ? getAllCaptures(context.record).find(
        candidate => candidate.id === preview.thumbnail!.captureId
      )
    : undefined;
  let thumbnail: ViewElementNode | undefined;
  if (capture) {
    const prefix = `${context.idPrefix}-thumbnail-${index}`;
    thumbnail = renderValueStrip({
      values: capture.values,
      scale: resolveValueScale(context, capture.name)?.scale,
      provenance: getCaptureProvenance(capture),
      idPrefix: prefix,
      ground: context.ground,
      width: 160,
      height: 12,
      maxCells: 64,
      label: capture.name
    });
    thumbnail.children.unshift(
      provenancePatternDefs(prefix, context.ground),
      crispnessDefs(prefix)
    );
  }
  return h(
    'div',
    {
      class: `lens-graph-unit lens-graph-unit--${unit.kind}`,
      'data-unit': unit.id,
      'data-tier': unit.tier,
      style: `grid-column:2;grid-row:${index + 1};height:64px;opacity:${TIER_OPACITY[unit.tier]}`
    },
    h(
      'button',
      {
        type: 'button',
        'data-action': 'bind',
        'data-axis': 'node',
        'data-key': unit.kind === 'group' ? unit.id : unit.memberIds[0],
        ...provenanceMarkAttributes('metadata', 'currentColor', context.idPrefix),
        style: `color:currentColor;background:transparent;border:${unit.kind === 'node' ? '0' : '1px solid currentColor'}`
      },
      unit.label
    ),
    unit.kind !== 'node'
      ? h(
          'button',
          {
            type: 'button',
            'data-action': 'toggle',
            'data-axis': 'node',
            'data-key': unit.id,
            'aria-label': `Expand ${unit.label}`
          },
          '+'
        )
      : undefined,
    h(
      'span',
      provenanceMarkAttributes('metadata', 'currentColor', context.idPrefix),
      preview.scalars.map(scalar => `${scalar.name}=${scalar.value}`).join(' · ')
    ),
    preview.hasCaptures
      ? h(
          'button',
          {
            type: 'button',
            'data-action': 'pin-inspector',
            'data-key': preview.inspector.unitId,
            'aria-label': preview.inspector.label
          },
          'inspect'
        )
      : undefined,
    thumbnail
  );
}

function renderLadder(model: CommandGraphViewModel, context: ViewContext): ViewElementNode {
  const ladder = model.ladder!;
  return h(
    'div',
    {class: 'lens-graph-ladder', style: 'overflow:auto'},
    h('p', null, 'Read: outline dot · write: solid dot · read-write: both'),
    !ladder.resources.length
      ? h('p', {class: 'lens-unknown'}, 'unknown — no resource metadata or accesses')
      : undefined,
    h(
      'table',
      null,
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          h('th', {scope: 'col'}, 'Pass'),
          ladder.resources.map(resource =>
            h(
              'th',
              {
                scope: 'col',
                title: `${resource.ownership}; allocation ${resource.allocation ?? 'unknown'}`
              },
              resource.id
            )
          )
        )
      ),
      h(
        'tbody',
        null,
        ladder.passes.map((pass, passIndex) =>
          h(
            'tr',
            null,
            h(
              'th',
              {scope: 'row'},
              h(
                'button',
                {
                  type: 'button',
                  'data-action': 'bind',
                  'data-axis': 'node',
                  'data-key': pass.nodeId
                },
                pass.label
              ),
              provenanceBadge('metadata')
            ),
            ladder.resources.map((resource, resourceIndex) => {
              const cell = ladder.cells.find(
                candidate =>
                  candidate.nodeId === pass.nodeId && candidate.resourceId === resource.id
              );
              if (!cell) return h('td', null, '—');
              const prefix = `${context.idPrefix}-cell-${passIndex}-${resourceIndex}`;
              return h(
                'td',
                {'data-access': cell.access},
                h(
                  'svg',
                  {width: 30, height: 18, role: 'img', 'aria-label': cell.access},
                  provenancePatternDefs(prefix, context.ground),
                  crispnessDefs(prefix),
                  cell.access !== 'write'
                    ? h('circle', {
                        ...provenanceMarkAttributes('metadata', 'currentColor', prefix),
                        cx: 8,
                        cy: 9,
                        r: 4
                      })
                    : undefined,
                  cell.access !== 'read'
                    ? h('circle', {
                        ...provenanceMarkAttributes('metadata', 'currentColor', prefix),
                        fill: 'currentColor',
                        cx: 21,
                        cy: 9,
                        r: 4
                      })
                    : undefined
                )
              );
            })
          )
        )
      )
    )
  );
}

function renderAliasing(model: CommandGraphViewModel, context: ViewContext): ViewElementNode {
  const aliasing = model.aliasing!;
  const width = Math.max(320, context.width ?? 640);
  const left = 160;
  const stepWidth = (width - left - 12) / Math.max(1, aliasing.scheduleLength);
  return h(
    'div',
    {class: 'lens-graph-aliasing'},
    h(
      'p',
      null,
      `Peak transient bytes: ${aliasing.totalsKnown ? aliasing.peakTransientBytes : 'unknown'}. Aliased savings: ${aliasing.savingsKnown ? `${aliasing.aliasedSavingsBytes} bytes` : 'unknown'}.`,
      !aliasing.totalsKnown ? ' Allocation or lifetime metadata missing.' : undefined
    ),
    h(
      'svg',
      {
        width,
        height: aliasing.allocations.length * 56 + 40,
        role: 'img',
        'aria-label': 'Allocation lifetimes over compiler schedule'
      },
      provenancePatternDefs(context.idPrefix, context.ground),
      crispnessDefs(context.idPrefix),
      h(
        'text',
        {x: left, y: 14, fill: 'currentColor'},
        `Schedule index 0–${Math.max(0, aliasing.scheduleLength - 1)}`
      ),
      aliasing.allocations.map((allocation, index) =>
        h(
          'g',
          {'data-allocation': allocation.allocation},
          h(
            'text',
            {x: 0, y: index * 56 + 40, fill: 'currentColor', 'font-size': 11},
            allocation.allocation,
            h('title', null, `${allocation.allocationKnown ? allocation.bytes : 'unknown'} bytes`)
          ),
          allocation.resources.some(resource => resource.lifetime === 'unknown')
            ? h(
                'text',
                {
                  x: left,
                  y: index * 56 + 58,
                  fill: 'currentColor',
                  'data-lifetime': 'unknown',
                  'font-size': 11
                },
                'unknown lifetime — see resource table'
              )
            : undefined,
          allocation.resources.map(resource =>
            resource.lifetime === 'unknown'
              ? undefined
              : h(
                  'rect',
                  {
                    ...provenanceMarkAttributes('metadata', 'currentColor', context.idPrefix),
                    x: left + resource.firstIndex! * stepWidth,
                    y: index * 56 + 25,
                    width: (resource.lastIndex! - resource.firstIndex! + 1) * stepWidth,
                    height: 20,
                    fill: resource.retained ? `url(#${context.idPrefix}-hatched)` : 'none',
                    'data-resource': resource.id,
                    'data-retained': String(resource.retained)
                  },
                  h(
                    'title',
                    null,
                    `${resource.id}: ${resource.firstIndex}–${resource.lastIndex}, ${resource.bytes} bytes${resource.retained ? '; retained for observation (lens twin)' : ''}`
                  )
                )
          )
        )
      )
    ),
    !aliasing.allocations.length ? h('p', null, 'No recorded transient allocations') : undefined,
    h(
      'p',
      null,
      `Imported: ${aliasing.imported.map(resource => `${resource.id} (${resource.bytes} bytes)`).join(', ') || '—'}`
    ),
    renderReceipt(getReceipt(model)!)
  );
}

function getReceipt(model: CommandGraphViewModel): LensReceiptTable | undefined {
  if (model.ladder) {
    const ladder = model.ladder;
    return {
      columns: ['Pass', ...ladder.resources.map(resource => resource.id)],
      rows: ladder.passes.map(pass => [
        pass.label,
        ...ladder.resources.map(
          resource =>
            ladder.cells.find(
              cell => cell.nodeId === pass.nodeId && cell.resourceId === resource.id
            )?.access ?? '—'
        )
      ]),
      provenance: ladder.passes.map(() => 'metadata')
    };
  }
  if (model.aliasing) {
    const rows = model.aliasing.allocations.flatMap(allocation =>
      allocation.resources.map(resource => [
        allocation.allocation,
        resource.id,
        resource.lifetime === 'unknown' ? 'unknown' : String(resource.firstIndex),
        resource.lifetime === 'unknown' ? 'unknown' : String(resource.lastIndex),
        String(resource.bytes),
        resource.retained ? 'retained for observation (lens twin)' : '—'
      ])
    );
    rows.push(
      ...model.aliasing.imported.map(resource => [
        'imported',
        resource.id,
        '—',
        '—',
        String(resource.bytes),
        '—'
      ])
    );
    return {
      columns: ['Allocation', 'Resource', 'First index', 'Last index', 'Bytes', 'Observation'],
      rows,
      provenance: rows.map(() => 'metadata')
    };
  }
  return undefined;
}

function renderReceipt(receipt: LensReceiptTable): ViewElementNode {
  return h(
    'table',
    null,
    h(
      'thead',
      null,
      h(
        'tr',
        null,
        receipt.columns.map(column => h('th', {scope: 'col'}, column)),
        h('th', {scope: 'col'}, 'Provenance')
      )
    ),
    h(
      'tbody',
      null,
      receipt.rows.map(row =>
        h(
          'tr',
          null,
          row.map(value => h('td', null, value || '—')),
          h('td', null, provenanceBadge('metadata'))
        )
      )
    )
  );
}

function parseExpanded(value: string | number | boolean | undefined): string[] {
  if (typeof value !== 'string') return [];
  try {
    const paths: unknown = JSON.parse(value);
    return Array.isArray(paths)
      ? paths.filter((path): path is string => typeof path === 'string')
      : [value];
  } catch {
    return [value];
  }
}

/** Generic graph rung: execution metadata, with logical forms reported at skeleton level. */
export const COMMAND_GRAPH_VIEW: ViewRegistration<CommandGraphViewModel> = {
  id: 'command-graph-view',
  label: 'Command graph',
  rung: 'graph',
  recordType: 'graph',
  order: 50,
  signature: {binds: [], free: ['node'], layout: {node: 'position'}},
  descriptor: {audience: 'expert', task: 'explore', interaction: 'high'},
  level: 'execution',
  grain: 'contributor',
  satisfies: context =>
    context.record.graphs.length ? {ok: true} : {ok: false, reason: 'record has no graph'},
  build: buildCommandGraphViewModel,
  render: renderCommandGraphView,
  describe: model => ({
    subject: `${model.form} command graph of ${model.graphLabel}`,
    keyNumber: {label: 'nodes', value: model.counts.nodes},
    facts: [
      `${model.folded?.units.length ?? 0} visible units vs budget ${model.folded?.budget ?? 24}`,
      `${model.folded?.hiddenNodeCount ?? 0} hidden nodes`,
      `Edges by kind: ${
        Object.entries(model.edgeKinds)
          .map(([kind, count]) => `${kind} ${count}`)
          .join(', ') || 'none'
      }`,
      ...(model.aliasing
        ? [
            `Aliased savings: ${model.aliasing.savingsKnown ? `${model.aliasing.aliasedSavingsBytes} bytes` : 'unknown'}`
          ]
        : [])
    ],
    provenance: 'metadata'
  }),
  receipt: getReceipt
};

/** Mounts the pure graph view in a host document; interaction is delegated through data attributes. */
export function mountCommandGraphView(
  element: HTMLElement,
  record: LensRecord,
  options: Partial<Omit<ViewContext, 'record'>> = {}
): {update(record: LensRecord): void; destroy(): void} {
  const defaults = {
    bindings: {},
    ground: 'light' as const,
    idPrefix: `lens-command-graph-${++mountCount}`,
    reducedMotion: false
  };
  const update = (nextRecord: LensRecord): void => {
    const context: ViewContext = {...defaults, ...options, record: nextRecord};
    const satisfaction = COMMAND_GRAPH_VIEW.satisfies(context);
    const tree = !('reason' in satisfaction)
      ? renderCommandGraphView(buildCommandGraphViewModel(context), context)
      : h('p', {class: 'lens-disabled'}, satisfaction.reason);
    element.replaceChildren(viewNodeToDom(tree, element.ownerDocument));
  };
  update(record);
  return {update, destroy: () => element.replaceChildren()};
}
