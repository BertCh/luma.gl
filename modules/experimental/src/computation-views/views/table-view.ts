// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import type {LensReceiptTable, ViewContext, ViewRegistration} from '../registry/view-registration';
import type {MarkProvenance} from '../types';
import {h} from '../view-node';
import type {ViewElementNode} from '../view-node';
import {getAdapterLabel, getMostConcreteGraph, getRecordCaptures} from './view-utils';

/** A display table with inferred column types and bounded row count. */
export type TypedTable = {
  columns: {name: string; type: 'number' | 'string' | 'boolean' | 'array' | 'mixed'}[];
  rows: string[][];
  truncatedCount: number;
};

/** A named section of the complete record inspector. */
export type TableViewSection = {id: string; label: string; table: TypedTable};

/** Pure table-view model, including per-capture lineage for the receipt. */
export type TableViewModel = {sections: TableViewSection[]; captureProvenance: MarkProvenance[]};

/** Infers types across all rows and formats missing values explicitly. */
export function buildTypedTable(
  rows: readonly Record<string, unknown>[],
  options: {maxRows?: number} = {}
): TypedTable {
  const names = [...new Set(rows.flatMap(row => Object.keys(row)))];
  const columns = names.map(name => {
    const types = new Set(
      rows
        .map(row => row[name])
        .filter(value => value !== undefined && value !== null)
        .map(getCellType)
    );
    return {name, type: types.size === 1 ? [...types][0] : ('mixed' as const)};
  });
  const maxRows = Math.max(0, Math.floor(options.maxRows ?? 200));
  return {
    columns,
    rows: rows.slice(0, maxRows).map(row => names.map(name => formatCell(row[name]))),
    truncatedCount: Math.max(0, rows.length - maxRows)
  };
}

function getCellType(value: unknown): TypedTable['columns'][number]['type'] {
  if (Array.isArray(value) || ArrayBuffer.isView(value)) return 'array';
  if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'string')
    return typeof value as 'number' | 'boolean' | 'string';
  return 'mixed';
}

function formatCell(value: unknown): string {
  if (value === undefined || value === null || value === '') return '—';
  if (Array.isArray(value)) return value.length ? value.map(formatCell).join(', ') : '—';
  if (typeof value === 'object')
    return (
      JSON.stringify(value, (_key, item) =>
        typeof item === 'number' && !Number.isFinite(item) ? String(item) : item
      ) || '—'
    );
  return String(value);
}

function summarizeCapture(
  capture: GPUComputationCapture,
  context: ViewContext
): Record<string, unknown> {
  let minimum = Infinity;
  let maximum = -Infinity;
  let finiteCount = 0;
  let nanCount = 0;
  let infinityCount = 0;
  for (let index = 0; index < capture.values.length; index++) {
    const value = capture.values[index];
    if (Number.isNaN(value)) nanCount++;
    else if (!Number.isFinite(value)) infinityCount++;
    else {
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
      finiteCount++;
    }
  }
  const row: Record<string, unknown> = {
    id: capture.id,
    name: capture.name,
    keySpace: capture.keySpace,
    iteration: capture.iteration,
    elementType: capture.elementType,
    count: capture.values.length,
    min: finiteCount ? minimum : undefined,
    max: finiteCount ? maximum : undefined,
    'NaN count': nanCount,
    'Inf count': infinityCount,
    provenance: capture.provenance === 'gpu-replay' ? 'GPU replay' : 'GPU readback'
  };
  const entity = context.bindings.entity;
  if (entity !== undefined) {
    const components = capture.components ?? 1;
    const keySpace = context.record.axes.axes.find(axis => axis.id === 'entity')?.keySpace;
    const index =
      typeof entity === 'number' &&
      Number.isInteger(entity) &&
      entity >= 0 &&
      (!keySpace || keySpace === capture.keySpace)
        ? entity * components
        : -1;
    row[`value at ${Array.isArray(entity) ? `{${entity.length} keys}` : entity}`] =
      index >= 0 && index + components <= capture.values.length
        ? components === 1
          ? capture.values[index]
          : Array.from({length: components}, (_, offset) => capture.values[index + offset])
        : undefined;
  }
  return row;
}

function buildTableView(context: ViewContext): TableViewModel {
  const {record, bindings} = context;
  const sections: TableViewSection[] = [];
  const addSection = (id: string, label: string, rows: Record<string, unknown>[]) =>
    sections.push({id, label, table: buildTypedTable(rows)});
  for (const graph of record.graphs) {
    const nodes = graph.nodes.filter(
      node => bindings.node === undefined || node.id === bindings.node
    );
    addSection(
      `graph:${graph.id}`,
      graph.label ?? graph.id,
      nodes.map(node => ({
        id: node.id,
        label: node.label,
        namespace: node.namespace,
        kind: node.kind,
        op: node.op,
        reads: node.reads,
        writes: node.writes,
        loop: node.loop ? `${node.loop.id}#${node.loop.iteration}` : undefined,
        dispatch: node.dispatch?.workgroups.join('×'),
        formula: node.formula,
        source: node.source,
        bindings: node.bindings
      }))
    );
    if (bindings.node !== undefined)
      addSection(
        `attributes:${graph.id}`,
        'node attributes',
        nodes.flatMap(node =>
          Object.entries(node.attrs).map(([name, value]) => ({node: node.id, name, value}))
        )
      );
  }
  addSection(
    'resources',
    'resources',
    record.graphs.flatMap(graph =>
      (graph.resources ?? []).map(resource => ({
        graph: graph.id,
        id: resource.id,
        kind: resource.kind,
        bytes: resource.bytes,
        ownership: resource.ownership,
        allocation: resource.allocation,
        firstNode: resource.firstNode,
        lastNode: resource.lastNode,
        retained: resource.retained
      }))
    )
  );
  const captures = getRecordCaptures(record);
  addSection(
    'captures',
    'captures',
    captures.map(capture => summarizeCapture(capture, context))
  );
  addSection(
    'decisions',
    'decisions',
    record.decisions.map(decision => ({
      subject: decision.subject,
      kind: decision.kind,
      chosen: decision.chosen,
      candidates: decision.candidates.map(
        candidate =>
          `${candidate.id}(${candidate.supported ? (candidate.score ?? '—') : `unsupported: ${candidate.reason}`})`
      )
    }))
  );
  addSection(
    'choices',
    'choices',
    record.choices.map(choice => ({
      stage: choice.stage,
      name: choice.name,
      value: choice.value,
      alternatives: choice.alternatives
    }))
  );
  addSection(
    'metrics',
    'metrics',
    (record.metrics ?? []).map(metrics => ({
      graph: metrics.graph,
      timebase: metrics.timebase,
      ...metrics.encoding,
      'nodes with timestamps': Object.values(metrics.nodes).filter(
        node => node.gpuBegin !== undefined && node.gpuEnd !== undefined
      ).length
    }))
  );
  // These reference tables keep the fallback useful even without a source or formula view.
  addSection(
    'edges',
    'edges',
    record.graphs.flatMap(graph => graph.edges.map(edge => ({graph: graph.id, ...edge})))
  );
  addSection(
    'formulas',
    'formulas',
    record.refs.formulas.map(formula => ({...formula}))
  );
  addSection(
    'sources',
    'sources',
    record.refs.sources.map(source => ({...source}))
  );
  addSection(
    'axes',
    'axes',
    record.axes.axes.map(axis => ({...axis}))
  );
  addSection(
    'key-spaces',
    'key spaces',
    record.axes.keySpaces.map(keySpace => ({...keySpace}))
  );
  addSection(
    'relations',
    'relations',
    record.axes.relations.map(relation => ({...relation}))
  );
  return {
    sections,
    captureProvenance: captures
      .slice(0, 200)
      .map(capture => (capture.provenance === 'gpu-replay' ? 'replay' : 'readback'))
  };
}

function renderTableView(model: TableViewModel): ViewElementNode {
  return h(
    'section',
    {class: 'lens-table'},
    model.sections.map(section =>
      h(
        'div',
        {'data-section': section.id},
        h(
          'table',
          null,
          h('caption', null, section.label),
          h(
            'thead',
            null,
            h(
              'tr',
              null,
              section.table.columns.map(column => h('th', {scope: 'col'}, column.name))
            )
          ),
          h(
            'tbody',
            null,
            section.table.rows.map(row =>
              h(
                'tr',
                null,
                row.map((cell, index) =>
                  h(
                    'td',
                    {
                      class:
                        section.table.columns[index].type === 'number'
                          ? 'lens-table-number'
                          : undefined,
                      style:
                        section.table.columns[index].type === 'number'
                          ? 'text-align: right'
                          : undefined
                    },
                    section.id === 'captures' && section.table.columns[index].name === 'provenance'
                      ? h(
                          'span',
                          {
                            class: 'lens-provenance',
                            'data-provenance': cell === 'GPU replay' ? 'replay' : 'readback',
                            'data-texture': cell === 'GPU replay' ? 'hatched' : 'solid',
                            style:
                              cell === 'GPU replay'
                                ? 'background-image: repeating-linear-gradient(135deg, transparent 0 3px, currentColor 3px 4px); padding: 2px'
                                : 'border-bottom: 3px solid currentColor'
                          },
                          h('span', {style: 'background: Canvas; color: CanvasText'}, cell)
                        )
                      : cell
                  )
                )
              )
            )
          )
        ),
        section.table.truncatedCount
          ? h('p', {class: 'lens-table-truncated'}, `${section.table.truncatedCount} more rows`)
          : null
      )
    )
  );
}

function getReceipt(model: TableViewModel): LensReceiptTable | undefined {
  const table = model.sections.find(section => section.id === 'captures')?.table;
  return table
    ? {
        columns: table.columns.map(column => column.name),
        rows: table.rows,
        provenance: model.captureProvenance
      }
    : undefined;
}

/** Universal typed inspector: no signature requirements, no fabricated values. */
export const TABLE_VIEW: ViewRegistration<TableViewModel> = {
  id: 'table',
  label: 'Table',
  rung: 'table',
  recordType: 'record',
  order: 1000,
  signature: {binds: [], free: ['node', 'buffer'], layout: {node: 'position', buffer: 'position'}},
  descriptor: {audience: 'expert', task: 'explore', interaction: 'high'},
  level: 'execution',
  grain: 'contributor',
  satisfies: () => ({ok: true}),
  build: buildTableView,
  render: renderTableView,
  receipt: getReceipt,
  formula: (_model, context) => {
    const node = context.record.graphs
      .flatMap(graph => graph.nodes)
      .find(candidate => candidate.id === context.bindings.node);
    const formula = context.record.refs.formulas.find(candidate =>
      node ? candidate.id === node.formula : true
    );
    return formula && {tex: formula.tex, ascii: formula.ascii, derivation: formula.derivation};
  },
  describe: (model, context) => ({
    subject: `Typed tables of record ${context.record.id}`,
    keyNumber: {label: 'nodes', value: getMostConcreteGraph(context.record)?.nodes.length ?? 0},
    facts: model.sections.map(
      section =>
        `${section.label}: ${section.table.rows.length + section.table.truncatedCount} rows`
    ),
    provenance: getRecordCaptures(context.record).length ? 'readback' : 'metadata',
    adapterLabel: getAdapterLabel(context.record)
  })
};
