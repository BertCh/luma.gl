// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {extractLensExample, type LensExample} from '../../computation-lenses/lens-example';
import {
  fromRecordedLensFile,
  getLensScenario,
  isLensScenarioId
} from '../../computation-lenses/registry';
import type {RecordedLensFile} from '../../computation-lenses/types';
import {createDefaultViewRegistry} from '../registry/default-registry';
import {createViewContext} from '../registry/view-registry';
import type {ViewRegistry} from '../registry/view-registry';
import {createGuardContext, evaluateGuards} from '../guards/guards';
import type {GuardSpec, GuardVerdict} from '../guards/guards';
import {renderVerdictChip} from '../views/verdict-chip';
import {LENS_RUNG_IDS} from '../types';
import type {AxisBindings, LensRungId, LensUrlState, RenderTeX, ViewGround} from '../types';
import {findViewNodes, h, rawHtml, viewNodeToHtml} from '../view-node';
import {getObserverEffects} from '../views/record-queries';
import {generateLensDescription, formatKeyNumber} from './description';

/** Directive attributes exactly as authored in `::lens{...}`. */
export type LensDirective = {
  scenario: string;
  rung: string;
  bind?: string;
  datum?: string;
  iteration?: string;
  caption?: string;
  view?: string;
};

/** Pure, serializable static twin consumed by documentation hosts. */
export type LensTwin = {
  size: 'full' | 'thumbnail';
  example?: LensExample;
  scenarioId: string;
  rung: LensRungId;
  viewId: string;
  disabled?: {reason: string};
  caption: string;
  description: string;
  llms: string;
  keyNumber: {label: string; value: string; unit?: string};
  figure: {html: string; width: number; height: number};
  receipt?: {columns: string[]; rows: string[][]; provenance?: string[]};
  formula?: {tex: string; ascii: string; derivation: 'derived' | 'curated'};
  /**
   * Verdicts of the guards the element's packs apply (`ViewPackRegistration.guardsFor`), printed
   * as chips under the caption. Omitted when no guard applies.
   */
  verdicts?: GuardVerdict[];
  provenance: {
    adapter: string;
    date: string;
    variant?: string;
    runId: string;
    observerEffects: string[];
  };
  state: LensUrlState;
};

/** Validates authored attributes; datum and iteration are integer pins, bind is a coordinator name. */
export function parseLensDirective(directive: LensDirective): {
  scenarioId: string;
  rung: LensRungId;
  bindings: AxisBindings;
  caption?: string;
  bind?: string;
  viewId?: string;
} {
  if (typeof directive.scenario !== 'string' || !isLensScenarioId(directive.scenario))
    throw new Error(`Invalid scenario attribute: ${directive.scenario}`);
  const rung = LENS_RUNG_IDS.find(candidate => candidate === directive.rung);
  if (!rung) throw new Error(`Invalid rung attribute: ${directive.rung}`);
  const bindings: {entity?: number; iteration?: number} = {};
  if (directive.datum !== undefined)
    bindings.entity = parseIntegerAttribute('datum', directive.datum);
  if (directive.iteration !== undefined)
    bindings.iteration = parseIntegerAttribute('iteration', directive.iteration);
  return {
    scenarioId: directive.scenario,
    rung,
    bindings,
    caption: directive.caption,
    bind: directive.bind,
    viewId: directive.view
  };
}

function parseIntegerAttribute(name: string, value: string): number {
  if (!/^-?\d+$/.test(value) || !Number.isSafeInteger(Number(value)))
    throw new Error(`Invalid ${name} attribute: ${value}; expected an integer`);
  return Number(value);
}

/** Builds a twin from a recording without a DOM, device or scenario-code import. */
export function buildLensTwin(
  recorded: RecordedLensFile,
  directive: LensDirective,
  options: {
    registry?: ViewRegistry;
    ground?: ViewGround;
    renderTeX?: RenderTeX;
    size?: 'full' | 'thumbnail';
    example?: string;
    /** Guards to evaluate; defaults to the record packs' guards for the scenario's element. */
    guards?: readonly GuardSpec[];
  } = {}
): LensTwin {
  const parsed = parseLensDirective(directive);
  const size = options.size ?? 'full';
  const width = size === 'thumbnail' ? 160 : 640;
  if (parsed.scenarioId !== recorded.scenarioId)
    throw new Error('scenario attribute does not match recorded scenarioId');
  const recording = fromRecordedLensFile(recorded);
  const record = recording.record;
  const variant = recording.variant ?? record.adapter.variant;
  const registry = options.registry ?? createDefaultViewRegistry();
  const context = createViewContext(record, {
    bindings: parsed.bindings,
    ground: options.ground ?? 'light',
    idPrefix: `lens-${parsed.scenarioId.replace(/[^a-zA-Z0-9_-]/g, '-')}-${parsed.rung}`,
    packs: registry.getPacksFor(record),
    renderTeX: options.renderTeX,
    width,
    options: size === 'thumbnail' ? {size: 'thumbnail'} : undefined
  });
  const resolution = registry.resolveRung(context, parsed.rung, parsed.viewId);
  const view = resolution.status === 'enabled' ? resolution.view : resolution.fallback;
  const disabled = resolution.status === 'disabled' ? {reason: resolution.reason} : undefined;
  const model = view.build(context);
  const tree = view.render(model, context);
  const observerEffects = getObserverEffects(record);
  const facts = view.describe(model, context);
  const description = generateLensDescription(
    {
      ...facts,
      adapterLabel: `${recording.adapterLabel}${variant ? ` · ${variant}` : ''}`,
      caveats: [...(facts.caveats ?? []), ...observerEffects]
    },
    {rung: parsed.rung, caption: parsed.caption, disabledReason: disabled?.reason}
  );
  const receipt = size === 'full' ? view.receipt?.(model, context) : undefined;
  const guards =
    options.guards ??
    registry.getGuardsFor({
      record,
      scenarioId: parsed.scenarioId,
      element: getLensScenario(parsed.scenarioId)?.element
    });
  const verdicts =
    size === 'full' && guards.length
      ? evaluateGuards(guards, createGuardContext(record, context.packs ?? [])).map(verdict => ({
          ...verdict,
          blocks: [...verdict.blocks]
        }))
      : undefined;
  const firstSvg = tree.tag === 'svg' ? tree : findViewNodes(tree, node => node.tag === 'svg')[0];
  const keyNumberValue =
    typeof facts.keyNumber.value === 'number'
      ? facts.keyNumber.value.toLocaleString(
          'en-US',
          Number.isInteger(facts.keyNumber.value) ? {} : {maximumSignificantDigits: 4}
        )
      : facts.keyNumber.value;
  return {
    size,
    ...(options.example !== undefined ? {example: extractLensExample(options.example)} : {}),
    scenarioId: parsed.scenarioId,
    rung: parsed.rung,
    viewId: view.id,
    disabled,
    caption:
      size === 'thumbnail' ? firstSentence(description.alt) : (parsed.caption ?? description.alt),
    description: description.alt,
    llms: description.llms,
    keyNumber: {...facts.keyNumber, value: keyNumberValue},
    figure: {
      html: viewNodeToHtml(tree),
      width:
        size === 'thumbnail' || view.id === 'result-view'
          ? width
          : firstSvg
            ? getDimension(firstSvg.attributes['width'], width)
            : width,
      height: firstSvg ? getDimension(firstSvg.attributes['height'], 0) : 0
    },
    ...(size === 'full'
      ? {
          receipt: receipt && {
            columns: [...receipt.columns],
            rows: receipt.rows.map(row => [...row]),
            provenance: receipt.provenance && [...receipt.provenance]
          },
          formula: view.formula?.(model, context),
          ...(verdicts ? {verdicts} : {})
        }
      : {}),
    provenance: {
      adapter: recording.adapterLabel,
      date: recording.recordedAt,
      variant,
      runId: record.runId,
      observerEffects
    },
    state: {
      runId: record.runId,
      rung: parsed.rung,
      bindings: parsed.bindings,
      clauses: [],
      params: {}
    }
  };
}

function firstSentence(description: string): string {
  const sentence = description.match(/^.*?[.!?](?=\s|$)/)?.[0] ?? description;
  return sentence.length <= 80 ? sentence : `${sentence.slice(0, 79).trimEnd()}…`;
}

function getDimension(value: unknown, fallback: number): number {
  const number =
    typeof value === 'number' || (typeof value === 'string' && /^\d+(?:\.\d+)?(?:px)?$/.test(value))
      ? Number.parseFloat(String(value))
      : NaN;
  return Number.isFinite(number) && number >= 0 ? number : fallback;
}

/** Checks the required documentation-twin contract. */
export function validateLensTwin(twin: LensTwin): string[] {
  const errors: string[] = [];
  if (!twin.caption.trim()) errors.push('caption must be non-empty');
  if (!twin.keyNumber.value.trim()) errors.push('key number value must be present');
  if (!twin.figure.html.trim()) errors.push('figure html must be non-empty');
  if (twin.provenance.runId !== twin.state.runId)
    errors.push('provenance.runId must equal state.runId');
  return errors;
}

/** Prints a reference figure with accessible description, inline number, receipt and formula. */
export function renderLensTwinHtml(twin: LensTwin, options: {renderTeX?: RenderTeX} = {}): string {
  const provenance = `Recorded on ${twin.provenance.adapter}${twin.provenance.variant ? ` · ${twin.provenance.variant}` : ''} · ${twin.provenance.date} · run ${twin.provenance.runId}`;
  return viewNodeToHtml(
    h(
      'figure',
      {
        class: 'ladder-lens-twin',
        'data-scenario': twin.scenarioId,
        'data-rung': twin.rung,
        'data-run-id': twin.provenance.runId
      },
      rawHtml('div', {role: 'img', 'aria-label': twin.description}, twin.figure.html),
      h(
        'figcaption',
        null,
        twin.caption,
        ' ',
        h('strong', {class: 'lens-key-number'}, formatKeyNumber(twin.keyNumber)),
        h(
          'p',
          {class: 'lens-recording-provenance'},
          provenance,
          twin.provenance.observerEffects.length
            ? ` · ${twin.provenance.observerEffects.join(' · ')}`
            : ''
        ),
        twin.disabled
          ? h(
              'p',
              {class: 'lens-disabled', role: 'note'},
              `${twin.rung} unavailable: ${twin.disabled.reason}`
            )
          : null,
        twin.verdicts?.length
          ? h('p', {class: 'lens-verdicts'}, twin.verdicts.map(renderVerdictChip))
          : null
      ),
      twin.rung === 'result' && twin.example
        ? h('pre', {class: 'lens-example'}, h('code', {class: 'language-ts'}, twin.example.code))
        : null,
      twin.receipt
        ? h(
            'table',
            {class: 'lens-receipt'},
            h('caption', null, 'Recorded receipt'),
            h(
              'thead',
              null,
              h(
                'tr',
                null,
                twin.receipt.columns.map(column => h('th', {scope: 'col'}, column))
              )
            ),
            h(
              'tbody',
              null,
              twin.receipt.rows.map((row, index) =>
                h(
                  'tr',
                  {'data-provenance': twin.receipt?.provenance?.[index]},
                  row.map(cell => h('td', null, cell || '—'))
                )
              )
            )
          )
        : null,
      twin.formula
        ? h(
            'div',
            {class: 'lens-formula', 'data-derivation': twin.formula.derivation},
            options.renderTeX
              ? rawHtml(
                  'div',
                  {class: 'lens-tex-rendered'},
                  options.renderTeX(twin.formula.tex, {displayMode: true})
                )
              : h('pre', {class: 'lens-tex'}, twin.formula.tex),
            h('p', {class: 'lens-ascii'}, twin.formula.ascii)
          )
        : null
    )
  );
}
