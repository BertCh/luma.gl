// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUComputationFormulaRef, GPUComputationFormulaSymbol} from '@luma.gl/gpgpu';
import type {LensRecord, ViewContext, ViewRegistration} from '../registry/view-registration';
import type {MarkProvenance} from '../types';
import {h, rawHtml, viewNodeToDom, type ViewElementNode} from '../view-node';
import {provenanceBadge} from '../encoding/provenance';
import {getAdapterLabel, getBoundNode, getFormula, getPinnedEntity} from './record-queries';
import {
  getSymbolValueAt,
  listTexTerms,
  resolveSymbolBinding,
  formatTexValue,
  substituteTexTerms
} from './formula-tex';

/** Formula statement and its optional progressively faded worked example. */
export type FormulaCardModel = {
  formulaId: string;
  tex: string;
  ascii: string;
  derivation: GPUComputationFormulaRef['derivation'];
  symbols: {
    id: string;
    label: string;
    role: GPUComputationFormulaSymbol['role'];
    binding: ReturnType<typeof resolveSymbolBinding>;
    value?: number;
    provenance?: MarkProvenance;
  }[];
  nodes: {id: string; label: string}[];
  entity?: number;
  workedTex?: string;
  fadeStep: number;
  fadedSymbolIds: string[];
};

/**
 * The pack's formula entry for an operator (`GPULocalMoran`), searched across the record's packs:
 * a formula ref, an explicit `{formula: 'none', reason}`, or undefined when no pack declares one.
 */
function getPackFormula(
  context: ViewContext,
  op: string | undefined
): GPUComputationFormulaRef | {formula: 'none'; reason: string} | undefined {
  if (!op) {
    return undefined;
  }
  for (const pack of context.packs ?? []) {
    const entry = pack.formulas?.[op];
    if (entry) {
      return entry;
    }
  }
  return undefined;
}

function chooseFormula(
  context: ViewContext
): GPUComputationFormulaRef | {formula: 'none'; reason: string} | undefined {
  const boundNode = getBoundNode(context)?.node;
  const formulaId = boundNode?.formula ?? context.options?.['formula'];
  if (formulaId !== undefined) {
    return getFormula(context.record, String(formulaId));
  }
  const packFormula = getPackFormula(context, boundNode?.op ?? context.options?.['op']?.toString());
  return packFormula ?? context.record.refs.formulas[0];
}

function chooseFormulaRef(context: ViewContext): GPUComputationFormulaRef | undefined {
  const formula = chooseFormula(context);
  return formula && 'tex' in formula ? formula : undefined;
}

/** Projects a formula and captured symbol values without evaluating its expression. */
export function buildFormulaCardModel(context: ViewContext): FormulaCardModel {
  const formula = chooseFormulaRef(context);
  // The registration checks that a formula exists before building.
  assert(formula);
  const entity = getPinnedEntity(context);
  const symbols = formula.symbols.map(symbol => ({
    id: symbol.id,
    label: symbol.label ?? symbol.id,
    role: symbol.role,
    binding: resolveSymbolBinding(context.record, symbol),
    ...(entity === undefined ? {} : getSymbolValueAt(context.record, symbol, entity))
  }));
  const requestedStep = context.options?.['fadeStep'];
  const fadeStep =
    typeof requestedStep === 'number' && Number.isInteger(requestedStep) && requestedStep >= 0
      ? requestedStep
      : 0;
  const roleOrder = {output: 0, intermediate: 1, parameter: 2, input: 3, index: 4, constant: 5};
  const taggedSymbolIds = new Set(listTexTerms(formula.tex).map(term => term.symbolId));
  const fadedSymbolIds = symbols
    .filter(symbol => symbol.value !== undefined && taggedSymbolIds.has(symbol.id))
    .sort((first, second) => roleOrder[first.role] - roleOrder[second.role])
    .slice(0, fadeStep)
    .map(symbol => symbol.id);
  const substitutions = new Map(
    symbols
      .filter(symbol => symbol.value !== undefined && !fadedSymbolIds.includes(symbol.id))
      .map(symbol => [symbol.id, formatTexValue(Number(symbol.value), symbol.role)])
  );
  let workedTex = entity === undefined ? undefined : substituteTexTerms(formula.tex, substitutions);
  if (workedTex !== undefined) {
    // Faded terms retain their authored body and expose an explicit, trusted placeholder class.
    const placeholders = new Map<string, string>();
    for (const term of listTexTerms(formula.tex).filter(term =>
      fadedSymbolIds.includes(term.symbolId)
    ))
      placeholders.set(term.symbolId, `\\htmlClass{lens-formula-placeholder}{${term.body}\\;?}`);
    workedTex = substituteTexTerms(workedTex, placeholders);
  }
  return {
    formulaId: formula.id,
    tex: formula.tex,
    ascii: formula.ascii,
    derivation: formula.derivation,
    symbols,
    nodes: context.record.graphs.flatMap(graph =>
      graph.nodes
        .filter(node => node.formula === formula.id)
        .map(node => ({id: `${graph.id}:${node.id}`, label: node.label ?? node.id}))
    ),
    entity,
    workedTex,
    fadeStep,
    fadedSymbolIds
  };
}

function renderFormula(tex: string, ascii: string, context: ViewContext): ViewElementNode {
  return context.renderTeX
    ? rawHtml('div', {class: 'lens-formula-tex'}, context.renderTeX(tex, {displayMode: true}))
    : h('div', null, h('pre', {class: 'lens-tex'}, tex), h('p', {class: 'lens-ascii'}, ascii));
}

/** Renders the statement, worked example, symbol bindings and kernel navigation chips. */
export function renderFormulaCard(model: FormulaCardModel, context: ViewContext): ViewElementNode {
  return h(
    'section',
    {class: 'lens-view lens-formula-card'},
    h(
      'header',
      null,
      h('h3', null, model.formulaId),
      h(
        'span',
        {
          class: `lens-provenance lens-provenance--${model.derivation === 'curated' ? 'derived' : 'metadata'}`,
          'data-provenance': model.derivation === 'curated' ? 'derived' : 'metadata'
        },
        model.derivation === 'derived' ? 'derived from the AST' : 'curated formula'
      )
    ),
    renderFormula(model.tex, model.ascii, context),
    model.workedTex === undefined
      ? null
      : h(
          'div',
          {class: 'lens-worked-example'},
          h('h4', null, `Entity ${model.entity}`),
          renderFormula(model.workedTex, model.ascii, context)
        ),
    h(
      'div',
      {class: 'lens-symbols'},
      model.symbols.map(symbol => {
        const binding = symbol.binding;
        const target =
          binding.status === 'bound'
            ? binding.target
            : binding.status === 'unknown'
              ? `unknown — ${binding.reason}`
              : 'unbound — no binding declared';
        return h(
          binding.status === 'bound' ? 'button' : 'span',
          {
            type: binding.status === 'bound' ? 'button' : undefined,
            class: `lens-symbol-chip lens-symbol-chip--${binding.status}${model.fadedSymbolIds.includes(symbol.id) ? ' lens-formula-placeholder' : ''}`,
            'data-action': binding.status === 'bound' ? 'bind' : undefined,
            'data-axis': binding.status === 'bound' ? 'node' : undefined,
            'data-key': binding.status === 'bound' ? binding.nodeIds[0] : undefined,
            'data-value': symbol.id
          },
          `${symbol.label} · ${symbol.role} · ${target}`,
          symbol.provenance ? provenanceBadge(symbol.provenance) : null
        );
      })
    ),
    h(
      'div',
      {class: 'lens-kernels'},
      model.nodes.map(node =>
        h(
          'button',
          {type: 'button', 'data-action': 'bind', 'data-axis': 'node', 'data-key': node.id},
          node.label
        )
      )
    ),
    h('p', null, getAdapterLabel(context.record))
  );
}

/** Generic formula registration, including machine-readable receipts and descriptions. */
export const FORMULA_CARD_VIEW: ViewRegistration<FormulaCardModel> = {
  id: 'formula-card',
  label: 'Formula',
  rung: 'formula',
  recordType: 'formula',
  order: 50,
  signature: {binds: [], free: ['term'], layout: {term: 'position'}},
  descriptor: {audience: 'analyst', task: 'explain', interaction: 'medium'},
  level: 'specification',
  grain: 'kernel',
  satisfies: context => {
    const formula = chooseFormula(context);
    if (formula && 'tex' in formula) {
      return {ok: true};
    }
    return {
      ok: false,
      reason:
        formula && 'reason' in formula
          ? `no closed-form formula: ${formula.reason} (see the iterations rung)`
          : 'no formula ref in this record (operators without a closed form declare formula: none and show the SkeletonView)'
    };
  },
  build: buildFormulaCardModel,
  render: renderFormulaCard,
  formula: model => ({tex: model.tex, ascii: model.ascii, derivation: model.derivation}),
  receipt: model => ({
    columns: ['symbol', 'role', 'binding', 'value', 'provenance'],
    rows: model.symbols.map(symbol => [
      symbol.label,
      symbol.role,
      symbol.binding.status === 'bound' ? symbol.binding.target : symbol.binding.status,
      symbol.value === undefined ? '—' : String(symbol.value),
      symbol.provenance ?? 'metadata'
    ]),
    provenance: model.symbols.map(symbol => symbol.provenance ?? 'metadata')
  }),
  describe: (model, context) => {
    const taggedSymbolIds = new Set(listTexTerms(model.tex).map(term => term.symbolId));
    const substituted = model.symbols.filter(
      symbol =>
        symbol.value !== undefined &&
        taggedSymbolIds.has(symbol.id) &&
        !model.fadedSymbolIds.includes(symbol.id)
    );
    const output = substituted.find(symbol => symbol.role === 'output');
    return {
      subject: `Formula ${model.ascii}`,
      keyNumber: output
        ? {label: output.label, value: output.value!}
        : {
            label: 'bound symbols',
            value: `${model.symbols.filter(symbol => symbol.binding.status === 'bound').length}/${model.symbols.length}`
          },
      facts: [],
      provenance: substituted.some(symbol => symbol.provenance === 'replay')
        ? 'replay'
        : (substituted[0]?.provenance ?? 'metadata'),
      adapterLabel: getAdapterLabel(context.record)
    };
  }
};

let mountCounter = 0;

/** Mounts this view into a host-owned element; update replaces its tree and destroy clears it. */
export function mountFormulaCard(
  element: HTMLElement,
  record: LensRecord,
  options: Partial<Omit<ViewContext, 'record'>> = {}
): {update(record: LensRecord): void; destroy(): void} {
  const defaults = {
    bindings: {},
    ground: 'light' as const,
    idPrefix: `lens-formula-card-${++mountCounter}`,
    reducedMotion: false,
    ...options
  };
  const update = (nextRecord: LensRecord) => {
    const context: ViewContext = {...defaults, record: nextRecord};
    const satisfaction = FORMULA_CARD_VIEW.satisfies(context);
    const tree = !('reason' in satisfaction)
      ? renderFormulaCard(buildFormulaCardModel(context), context)
      : h('p', {class: 'lens-disabled'}, satisfaction.reason);
    element.replaceChildren(viewNodeToDom(tree, element.ownerDocument));
  };
  update(record);
  return {update, destroy: () => element.replaceChildren()};
}
