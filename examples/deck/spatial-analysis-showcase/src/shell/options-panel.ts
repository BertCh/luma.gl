// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OptionsStore} from '../scenes/options-state';
import type {OptionSpec, OptionState} from '../scenes/scene';
import {h, icon} from './dom';

/** One generated control. */
export type OptionControl = {
  element: HTMLElement;
  spec: OptionSpec<never>;
  /** Syncs the control with the store. */
  refresh: (state: OptionState) => void;
};

/**
 * Calls `refresh` when the store changes: immediately for user writes, once per frame for
 * scene-driven writes (animated sliders can write every frame). Returns a cleanup function.
 */
function subscribeCoalesced(store: OptionsStore, refresh: () => void): () => void {
  let frame = 0;
  const unsubscribe = store.subscribe((_id, _value, source) => {
    if (source === 'user') {
      refresh();
    } else if (!frame) {
      frame = requestAnimationFrame(() => {
        frame = 0;
        refresh();
      });
    }
  });
  return () => {
    cancelAnimationFrame(frame);
    unsubscribe();
  };
}

const FILTER_THRESHOLD = 8;
const DEFAULT_GROUP = 'General';

type PanelEntry = {
  control: OptionControl;
  item: HTMLElement;
  resetButton: HTMLButtonElement;
  searchText: string;
  group: string;
};

/**
 * The full, grouped options list of a scene: filter, per-control reset and "this step" marks.
 * Every control reads and writes the shared {@link OptionsStore}; the panel never talks to the
 * scene directly.
 */
export class OptionsPanel {
  readonly element: HTMLElement;
  private readonly entries: PanelEntry[] = [];
  private readonly groups: {element: HTMLDetailsElement; entries: PanelEntry[]}[] = [];
  private readonly store: OptionsStore;
  private readonly resetAllButton: HTMLButtonElement;
  private readonly emptyMessage: HTMLElement;
  private readonly onModifiedChange?: (modified: boolean) => void;
  private readonly unsubscribe: () => void;
  private filterText = '';
  private lastModified: boolean | null = null;

  /**
   * @param store the scene's option store.
   * @param onAction called when an action button is pressed.
   * @param onModifiedChange called when "any option differs from its default" changes.
   */
  constructor(
    store: OptionsStore,
    onAction: (id: string) => void,
    onModifiedChange?: (modified: boolean) => void
  ) {
    this.store = store;
    this.onModifiedChange = onModifiedChange;
    this.resetAllButton = h(
      'button',
      {
        class: 'btn btn-ghost btn-small',
        type: 'button',
        title: 'Reset every option to its default'
      },
      icon('reset', 14),
      'Reset all'
    );
    this.resetAllButton.addEventListener('click', () => store.reset());

    const groupNames: string[] = [];
    const byGroup = new Map<string, PanelEntry[]>();
    for (const spec of store.specs) {
      const group = (spec as {group?: string}).group || DEFAULT_GROUP;
      const control = createOptionControl(spec, store, onAction, 'all-');
      const resetButton = h(
        'button',
        {
          class: 'icon-reset',
          type: 'button',
          title: 'Reset to default',
          'aria-label': `Reset ${spec.label} to default`
        },
        icon('reset', 13)
      );
      resetButton.addEventListener('click', () => store.set(spec.id, store.getDefault(spec.id)));
      const item = h(
        'div',
        {class: 'control-item'},
        h('span', {class: 'step-tag'}, 'this step'),
        control.element,
        h('div', {class: 'control-side'}, h('span', {class: 'modified-dot'}), resetButton)
      );
      const help = (spec as {help?: string}).help ?? '';
      const entry: PanelEntry = {
        control,
        item,
        resetButton,
        group,
        searchText: `${spec.label} ${help} ${group} ${spec.id}`.toLowerCase()
      };
      this.entries.push(entry);
      if (!byGroup.has(group)) {
        byGroup.set(group, []);
        groupNames.push(group);
      }
      byGroup.get(group)!.push(entry);
    }
    // Options without a group come first, under "General".
    groupNames.sort((a, b) => Number(b === DEFAULT_GROUP) - Number(a === DEFAULT_GROUP));
    const body = h('div', {class: 'options-body'});
    for (const name of groupNames) {
      const entries = byGroup.get(name)!;
      const element = h(
        'details',
        {class: 'options-group', open: true},
        h(
          'summary',
          {},
          h('span', {class: 'options-group-title'}, name),
          h('span', {class: 'count'}, String(entries.length)),
          icon('chevronDown', 14)
        ),
        h(
          'div',
          {class: 'options-group-body'},
          entries.map(entry => entry.item)
        )
      );
      this.groups.push({element, entries});
      body.append(element);
    }
    this.emptyMessage = h('p', {class: 'options-empty muted', hidden: true}, 'No controls match');
    body.append(this.emptyMessage);

    const toolbar = h('div', {class: 'options-toolbar'});
    if (this.entries.length > FILTER_THRESHOLD) {
      const filter = h('input', {
        class: 'options-filter',
        type: 'search',
        placeholder: 'Filter controls',
        'aria-label': 'Filter controls'
      });
      filter.addEventListener('input', () => {
        this.filterText = filter.value.trim().toLowerCase();
        this.applyFilter();
      });
      toolbar.append(h('div', {class: 'options-search'}, icon('search', 14), filter));
    }
    toolbar.append(
      h(
        'div',
        {class: 'options-legend'},
        h('span', {class: 'badge badge-compile'}, 'rebuild'),
        h('span', {class: 'muted small'}, 'rebuilds a compiled graph')
      ),
      this.resetAllButton
    );
    this.element = h('div', {class: 'options'}, toolbar, body);
    this.refresh();
    this.unsubscribe = subscribeCoalesced(store, () => this.refresh());
  }

  /** Number of options in the schema, for the tab badge. */
  get count(): number {
    return this.entries.length;
  }

  /** Marks the controls that belong to the current step. */
  setStepControls(ids: readonly string[]): void {
    const wanted = new Set(ids);
    for (const entry of this.entries) {
      entry.item.classList.toggle('is-step-control', wanted.has(entry.control.spec.id));
    }
  }

  destroy(): void {
    this.unsubscribe();
  }

  private applyFilter(): void {
    let visible = 0;
    for (const entry of this.entries) {
      const matches = !this.filterText || entry.searchText.includes(this.filterText);
      entry.item.hidden = !matches;
      if (matches) visible++;
    }
    for (const group of this.groups) {
      group.element.hidden = group.entries.every(entry => entry.item.hidden);
      // Keep matching groups open while filtering.
      if (this.filterText && !group.element.hidden) group.element.open = true;
    }
    this.emptyMessage.hidden = visible > 0;
  }

  private refresh(): void {
    const state = this.store.snapshot();
    for (const entry of this.entries) {
      entry.control.refresh(state);
      const modified = this.store.isOptionModified(entry.control.spec.id);
      entry.item.classList.toggle('is-modified', modified);
      entry.resetButton.disabled = !modified;
    }
    const modified = this.store.isModified();
    this.resetAllButton.disabled = !modified;
    if (modified !== this.lastModified) {
      this.lastModified = modified;
      this.onModifiedChange?.(modified);
    }
  }
}

/** The live controls for one story step ("Adjust in this step"), in the order the step lists them. */
export class StepControls {
  readonly element: HTMLElement;
  private readonly controls: OptionControl[];
  private readonly store: OptionsStore;
  private readonly unsubscribe: () => void;

  constructor(
    store: OptionsStore,
    specs: readonly OptionSpec<never>[],
    onAction: (id: string) => void
  ) {
    this.store = store;
    this.controls = specs.map(spec => createOptionControl(spec, store, onAction, 'step-'));
    this.element = h(
      'div',
      {class: 'step-controls'},
      this.controls.map(control => control.element)
    );
    this.refresh();
    this.unsubscribe = subscribeCoalesced(store, () => this.refresh());
  }

  destroy(): void {
    this.unsubscribe();
  }

  private refresh(): void {
    const state = this.store.snapshot();
    for (const control of this.controls) control.refresh(state);
  }
}

/**
 * Creates the control for one option. The same factory serves the full options list and the
 * per-step controls, so `idPrefix` keeps the element ids of the two copies unique.
 */
export function createOptionControl(
  spec: OptionSpec<never>,
  store: OptionsStore,
  onAction: (id: string) => void,
  idPrefix: string
): OptionControl {
  const compileBadge =
    spec.apply === 'compile'
      ? h(
          'span',
          {class: 'badge badge-compile', title: 'Changing this rebuilds a compiled graph'},
          'rebuild'
        )
      : null;
  const help = (spec as {help?: string}).help;
  const helpElement = help ? h('p', {class: 'control-help'}, help) : null;
  const disabledWhen = (spec as {disabledWhen?: (state: OptionState) => boolean}).disabledWhen;
  const id = `${idPrefix}option-${spec.id}`;

  if (spec.kind === 'button') {
    const button = h('button', {class: 'btn', type: 'button', id}, spec.label);
    button.addEventListener('click', () => onAction(spec.id));
    const element = h('div', {class: 'control control-button'}, button, helpElement);
    return {
      element,
      spec,
      refresh: state => {
        button.disabled = Boolean(disabledWhen?.(state));
      }
    };
  }

  if (spec.kind === 'slider') {
    const input = h('input', {type: 'range', id, min: spec.min, max: spec.max, step: spec.step});
    const value = h('output', {class: 'control-value', htmlFor: id});
    const format = (number: number) =>
      spec.format
        ? spec.format(number)
        : `${formatNumber(number, spec.step)}${spec.unit ? ` ${spec.unit}` : ''}`;
    input.addEventListener('input', () => store.set(spec.id, Number(input.value)));
    const element = h(
      'div',
      {class: 'control'},
      h('div', {class: 'control-row'}, h('label', {for: id}, spec.label, compileBadge), value),
      input,
      helpElement
    );
    return {
      element,
      spec,
      refresh: state => {
        const current = state[spec.id] as number;
        input.value = String(current);
        value.textContent = format(current);
        input.disabled = Boolean(disabledWhen?.(state));
        element.classList.toggle('is-disabled', input.disabled);
      }
    };
  }

  if (spec.kind === 'range') {
    const low = h('input', {
      type: 'range',
      min: spec.min,
      max: spec.max,
      step: spec.step,
      'aria-label': `${spec.label} minimum`
    });
    const high = h('input', {
      type: 'range',
      min: spec.min,
      max: spec.max,
      step: spec.step,
      'aria-label': `${spec.label} maximum`
    });
    const value = h('output', {class: 'control-value'});
    const format = (number: number) =>
      spec.format
        ? spec.format(number)
        : `${formatNumber(number, spec.step)}${spec.unit ? ` ${spec.unit}` : ''}`;
    const commit = () => {
      const a = Number(low.value);
      const b = Number(high.value);
      store.set(spec.id, [Math.min(a, b), Math.max(a, b)] as const);
    };
    low.addEventListener('input', commit);
    high.addEventListener('input', commit);
    const element = h(
      'div',
      {class: 'control'},
      h('div', {class: 'control-row'}, h('label', {}, spec.label, compileBadge), value),
      h('div', {class: 'range-pair'}, low, high),
      helpElement
    );
    return {
      element,
      spec,
      refresh: state => {
        const [a, b] = state[spec.id] as readonly [number, number];
        low.value = String(a);
        high.value = String(b);
        value.textContent = `${format(a)} to ${format(b)}`;
        low.disabled = high.disabled = Boolean(disabledWhen?.(state));
        element.classList.toggle('is-disabled', low.disabled);
      }
    };
  }

  if (spec.kind === 'select') {
    const select = h(
      'select',
      {id},
      spec.options.map(option => h('option', {value: option.value}, option.label))
    );
    const optionHelp = h('p', {class: 'control-help'});
    select.addEventListener('change', () => store.set(spec.id, select.value));
    const element = h(
      'div',
      {class: 'control'},
      h('div', {class: 'control-row'}, h('label', {for: id}, spec.label, compileBadge)),
      select,
      optionHelp,
      helpElement
    );
    return {
      element,
      spec,
      refresh: state => {
        const current = state[spec.id] as string;
        select.value = current;
        const selected = spec.options.find(option => option.value === current);
        optionHelp.textContent = selected?.help ?? '';
        optionHelp.hidden = !selected?.help;
        select.disabled = Boolean(disabledWhen?.(state));
        element.classList.toggle('is-disabled', select.disabled);
      }
    };
  }

  // Toggle.
  const checkbox = h('input', {type: 'checkbox', id, role: 'switch'});
  checkbox.addEventListener('change', () => store.set(spec.id, checkbox.checked));
  const element = h(
    'div',
    {class: 'control control-toggle'},
    h(
      'label',
      {class: 'switch', for: id},
      checkbox,
      h('span', {class: 'switch-track'}),
      h('span', {class: 'switch-label'}, spec.label, compileBadge)
    ),
    helpElement
  );
  return {
    element,
    spec,
    refresh: state => {
      checkbox.checked = Boolean(state[spec.id]);
      checkbox.disabled = Boolean(disabledWhen?.(state));
      element.classList.toggle('is-disabled', checkbox.disabled);
    }
  };
}

function formatNumber(value: number, step: number): string {
  const decimals = step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step)));
  return value.toFixed(decimals);
}
