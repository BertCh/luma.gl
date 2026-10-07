// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OptionsStore} from '../scenes/options-state';
import type {OptionSpec, OptionState} from '../scenes/scene';
import {h, icon} from './dom';
import './controls.css';

/** One generated control. */
export type OptionControl = {
  element: HTMLElement;
  spec: OptionSpec<never>;
  /** Syncs the control with the store. */
  refresh: (state: OptionState) => void;
  /** Stops animations and listeners the control owns (the `autoSweep` loop). */
  destroy?: () => void;
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
const EXPERT_GROUP = 'Expert';

type PanelEntry = {
  control: OptionControl;
  item: HTMLElement;
  /** Null for options without state (buttons, presets). */
  resetButton: HTMLButtonElement | null;
  searchText: string;
  group: string;
};

/**
 * The full, grouped options list of a scene: filter, per-control reset and "this step" marks.
 * Options marked `expert` fold into a closed "Expert" group at the end (the filter still finds
 * them). Every control reads and writes the shared {@link OptionsStore}; the panel never talks to the
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
      const isExpert = Boolean((spec as {expert?: boolean}).expert);
      const group = isExpert ? EXPERT_GROUP : (spec as {group?: string}).group || DEFAULT_GROUP;
      const control = createOptionControl(spec, store, onAction, 'all-');
      const hasState = spec.kind !== 'button' && spec.kind !== 'preset';
      let resetButton: HTMLButtonElement | null = null;
      if (hasState) {
        resetButton = h(
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
      }
      const item = h(
        'div',
        {class: 'control-item'},
        h('span', {class: 'step-tag'}, 'this step'),
        control.element,
        hasState
          ? h('div', {class: 'control-side'}, h('span', {class: 'modified-dot'}), resetButton)
          : null
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
    // Options without a group come first, under "General"; "Expert" comes last and starts closed.
    const rank = (name: string) => (name === DEFAULT_GROUP ? 0 : name === EXPERT_GROUP ? 2 : 1);
    groupNames.sort((a, b) => rank(a) - rank(b));
    const body = h('div', {class: 'options-body'});
    for (const name of groupNames) {
      const entries = byGroup.get(name)!;
      const element = h(
        'details',
        {class: 'options-group', open: name !== EXPERT_GROUP},
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
    this.element = h(
      'div',
      {class: 'options', tabindex: '-1', 'data-controls-root': ''},
      toolbar,
      body
    );
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
    for (const entry of this.entries) entry.control.destroy?.();
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
      if (entry.resetButton) entry.resetButton.disabled = !modified;
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
      {class: 'step-controls', tabindex: '-1', 'data-controls-root': ''},
      this.controls.map(control => control.element)
    );
    this.refresh();
    this.unsubscribe = subscribeCoalesced(store, () => this.refresh());
  }

  destroy(): void {
    this.unsubscribe();
    for (const control of this.controls) control.destroy?.();
  }

  private refresh(): void {
    const state = this.store.snapshot();
    for (const control of this.controls) control.refresh(state);
  }
}

type SliderSpec = Extract<OptionSpec<never>, {kind: 'slider'}>;
type SelectSpec = Extract<OptionSpec<never>, {kind: 'select'}>;
type RangeSpec = Extract<OptionSpec<never>, {kind: 'range'}>;
type PresetSpec = Extract<OptionSpec<never>, {kind: 'preset'}>;

/** What every control kind builds: pieces assembled by {@link createOptionControl}. */
type ControlParts = {
  label: string;
  /** Rebuild badge for `apply: 'compile'` options. */
  badge: HTMLElement | null;
  help: HTMLElement | null;
  isDisabled: (state: OptionState) => boolean;
  id: string;
};

/**
 * Creates the control for one option. The same factory serves the full options list and the
 * per-step controls, so `idPrefix` keeps the element ids of the two copies unique.
 *
 * - `slider`: filled track, value readout, optional `marks`, `danger` range, `describe` line,
 *   `display: 'stepper'` and `autoSweep`.
 * - `select`: dropdown, `display: 'segmented'` (radio group with arrow keys) or `'chips'`.
 * - `preset`: chips that write several options at once.
 * - `range`, `toggle` and `button` as before.
 */
export function createOptionControl(
  spec: OptionSpec<never>,
  store: OptionsStore,
  onAction: (id: string) => void,
  idPrefix: string
): OptionControl {
  const apply = (spec as {apply?: string}).apply;
  const parts: ControlParts = {
    label: spec.label,
    badge:
      apply === 'compile'
        ? h(
            'span',
            {class: 'badge badge-compile', title: 'Changing this rebuilds a compiled graph'},
            'rebuild'
          )
        : null,
    help: (spec as {help?: string}).help
      ? h('p', {class: 'control-help'}, (spec as {help?: string}).help)
      : null,
    isDisabled: state =>
      Boolean((spec as {disabledWhen?: (state: OptionState) => boolean}).disabledWhen?.(state)),
    id: `${idPrefix}option-${spec.id}`
  };

  switch (spec.kind) {
    case 'button':
      return createButtonControl(spec, parts, onAction);
    case 'slider':
      return spec.display === 'stepper'
        ? createStepperControl(spec, store, parts)
        : createSliderControl(spec, store, parts);
    case 'range':
      return createRangeControl(spec, store, parts);
    case 'select':
      return spec.display === 'segmented'
        ? createSegmentedControl(spec, store, parts)
        : spec.display === 'chips'
          ? createChipsControl(spec, store, parts)
          : createDropdownControl(spec, store, parts);
    case 'preset':
      return createPresetControl(spec, store, parts);
    default:
      return createToggleControl(spec, store, parts);
  }
}

// -- Buttons, toggles, dropdowns

function createButtonControl(
  spec: Extract<OptionSpec<never>, {kind: 'button'}>,
  parts: ControlParts,
  onAction: (id: string) => void
): OptionControl {
  const button = h('button', {class: 'btn', type: 'button', id: parts.id}, spec.label);
  button.addEventListener('click', () => onAction(spec.id));
  const element = h('div', {class: 'control control-button'}, button, parts.help);
  return {
    element,
    spec,
    refresh: state => {
      button.disabled = parts.isDisabled(state);
    }
  };
}

function createToggleControl(
  spec: Extract<OptionSpec<never>, {kind: 'toggle'}>,
  store: OptionsStore,
  parts: ControlParts
): OptionControl {
  const checkbox = h('input', {type: 'checkbox', id: parts.id, role: 'switch'});
  checkbox.addEventListener('change', () => store.set(spec.id, checkbox.checked));
  const element = h(
    'div',
    {class: 'control control-toggle'},
    h(
      'label',
      {class: 'switch', for: parts.id},
      checkbox,
      h('span', {class: 'switch-track'}),
      h('span', {class: 'switch-label'}, spec.label, parts.badge)
    ),
    parts.help
  );
  return {
    element,
    spec,
    refresh: state => {
      checkbox.checked = Boolean(state[spec.id]);
      checkbox.disabled = parts.isDisabled(state);
      element.classList.toggle('is-disabled', checkbox.disabled);
    }
  };
}

function createDropdownControl(
  spec: SelectSpec,
  store: OptionsStore,
  parts: ControlParts
): OptionControl {
  const select = h(
    'select',
    {id: parts.id},
    spec.options.map(option => h('option', {value: option.value}, option.label))
  );
  const optionHelp = h('p', {class: 'control-help'});
  select.addEventListener('change', () => store.set(spec.id, select.value));
  select.addEventListener('keydown', event => {
    if (event.key === 'Escape') returnFocusToPanel(event, element);
  });
  const element = h(
    'div',
    {class: 'control control-select'},
    h('div', {class: 'control-row'}, h('label', {for: parts.id}, spec.label, parts.badge)),
    h('div', {class: 'select-wrap'}, select, icon('chevronDown', 14)),
    optionHelp,
    parts.help
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
      select.disabled = parts.isDisabled(state);
      element.classList.toggle('is-disabled', select.disabled);
    }
  };
}

/** Escape inside a select or segmented control hands focus back to the panel (and swallows the key). */
function returnFocusToPanel(event: KeyboardEvent, element: HTMLElement): void {
  const root = element.closest<HTMLElement>('[data-controls-root]');
  if (!root) return;
  event.preventDefault();
  event.stopPropagation();
  root.focus({preventScroll: true});
}

// -- Segmented, chips, presets

function createSegmentedControl(
  spec: SelectSpec,
  store: OptionsStore,
  parts: ControlParts
): OptionControl {
  const labelId = `${parts.id}-label`;
  const buttons = spec.options.map(option =>
    h(
      'button',
      {type: 'button', class: 'segment', role: 'radio', 'aria-checked': 'false', tabindex: '-1'},
      option.label
    )
  );
  const group = h(
    'div',
    {class: 'segmented', role: 'radiogroup', 'aria-labelledby': labelId},
    buttons
  );
  group.style.setProperty('--segments', String(Math.max(1, buttons.length)));
  const optionHelp = h('p', {class: 'control-help'});
  buttons.forEach((button, index) => {
    button.addEventListener('click', () => store.set(spec.id, spec.options[index].value));
  });
  group.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      returnFocusToPanel(event, element);
      return;
    }
    const current = Math.max(0, buttons.indexOf(event.target as HTMLButtonElement));
    const count = buttons.length;
    let next = -1;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (current + 1) % count;
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp')
      next = (current - 1 + count) % count;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = count - 1;
    if (next < 0) return;
    // The story view steps with the arrow keys; keep them for the radio group.
    event.preventDefault();
    event.stopPropagation();
    buttons[next].focus();
    store.set(spec.id, spec.options[next].value);
  });
  const element = h(
    'div',
    {class: 'control control-segmented'},
    h(
      'div',
      {class: 'control-row'},
      h('span', {class: 'control-label', id: labelId}, spec.label, parts.badge)
    ),
    group,
    optionHelp,
    parts.help
  );
  return {
    element,
    spec,
    refresh: state => {
      const current = state[spec.id] as string;
      const selectedIndex = spec.options.findIndex(option => option.value === current);
      const disabled = parts.isDisabled(state);
      buttons.forEach((button, index) => {
        const selected = index === selectedIndex;
        button.setAttribute('aria-checked', String(selected));
        button.classList.toggle('is-selected', selected);
        // Roving tabindex: the selected segment (or the first, when none is) takes the tab stop.
        button.tabIndex = selected || (selectedIndex < 0 && index === 0) ? 0 : -1;
        button.disabled = disabled;
      });
      const selected = spec.options[selectedIndex];
      optionHelp.textContent = selected?.help ?? '';
      optionHelp.hidden = !selected?.help;
      element.classList.toggle('is-disabled', disabled);
    }
  };
}

function createChipsControl(
  spec: SelectSpec,
  store: OptionsStore,
  parts: ControlParts
): OptionControl {
  const labelId = `${parts.id}-label`;
  const buttons = spec.options.map((option, index) => {
    const button = h(
      'button',
      {type: 'button', class: 'pill', 'aria-pressed': 'false'},
      option.label
    );
    button.addEventListener('click', () => store.set(spec.id, spec.options[index].value));
    return button;
  });
  const group = h('div', {class: 'pills', role: 'group', 'aria-labelledby': labelId}, buttons);
  group.addEventListener('keydown', event => {
    if (event.key === 'Escape') returnFocusToPanel(event, element);
  });
  const optionHelp = h('p', {class: 'control-help'});
  const element = h(
    'div',
    {class: 'control control-chips'},
    h(
      'div',
      {class: 'control-row'},
      h('span', {class: 'control-label', id: labelId}, spec.label, parts.badge)
    ),
    group,
    optionHelp,
    parts.help
  );
  return {
    element,
    spec,
    refresh: state => {
      const current = state[spec.id] as string;
      const disabled = parts.isDisabled(state);
      buttons.forEach((button, index) => {
        const selected = spec.options[index].value === current;
        button.setAttribute('aria-pressed', String(selected));
        button.classList.toggle('is-selected', selected);
        button.disabled = disabled;
      });
      const selected = spec.options.find(option => option.value === current);
      optionHelp.textContent = selected?.help ?? '';
      optionHelp.hidden = !selected?.help;
      element.classList.toggle('is-disabled', disabled);
    }
  };
}

function createPresetControl(
  spec: PresetSpec,
  store: OptionsStore,
  parts: ControlParts
): OptionControl {
  const labelId = `${parts.id}-label`;
  // The help line follows the hovered or focused chip, then the active one, then the option's own.
  let hovered = -1;
  let active = -1;
  const presetHelp = h('p', {class: 'control-help'});
  const showHelp = () => {
    const shown = hovered >= 0 ? hovered : active;
    const text = (shown >= 0 ? spec.presets[shown].help : undefined) ?? spec.help ?? '';
    presetHelp.textContent = text;
    presetHelp.hidden = !text;
  };
  const buttons = spec.presets.map((preset, index) => {
    const button = h(
      'button',
      {type: 'button', class: 'pill', 'aria-pressed': 'false'},
      preset.label
    );
    button.addEventListener('click', () => store.applyPreset(spec, index));
    for (const [enter, leave] of [
      ['mouseenter', 'mouseleave'],
      ['focus', 'blur']
    ] as const) {
      button.addEventListener(enter, () => {
        hovered = index;
        showHelp();
      });
      button.addEventListener(leave, () => {
        hovered = -1;
        showHelp();
      });
    }
    return button;
  });
  const element = h(
    'div',
    {class: 'control control-preset'},
    h('div', {class: 'control-row'}, h('span', {class: 'control-label', id: labelId}, spec.label)),
    h('div', {class: 'pills', role: 'group', 'aria-labelledby': labelId}, buttons),
    presetHelp
  );
  return {
    element,
    spec,
    refresh: () => {
      active = store.getActivePresetIndex(spec);
      buttons.forEach((button, index) => {
        button.setAttribute('aria-pressed', String(index === active));
        button.classList.toggle('is-selected', index === active);
      });
      showHelp();
    }
  };
}

// -- Sliders

function formatSliderValue(spec: SliderSpec | RangeSpec, value: number): string {
  if (spec.format) return spec.format(value);
  return `${formatNumber(value, spec.step)}${spec.unit ? ` ${spec.unit}` : ''}`;
}

/** Position of `value` along the track, 0 to 1. */
function getFraction(value: number, min: number, max: number): number {
  return max > min ? Math.min(1, Math.max(0, (value - min) / (max - min))) : 0;
}

function createSliderControl(
  spec: SliderSpec,
  store: OptionsStore,
  parts: ControlParts
): OptionControl {
  const input = h('input', {
    type: 'range',
    id: parts.id,
    min: spec.min,
    max: spec.max,
    step: spec.step
  });
  const value = h('output', {class: 'control-value', for: parts.id});
  const describeLine = h('p', {class: 'control-describe', 'aria-live': 'polite', hidden: true});
  const sweep = spec.autoSweep ? createSweep(spec, store, spec.autoSweep) : null;

  // Anything the reader does to the slider ends a running sweep.
  for (const type of ['pointerdown', 'keydown', 'input'] as const) {
    input.addEventListener(type, () => sweep?.stop());
  }
  input.addEventListener('input', () => store.set(spec.id, Number(input.value)));

  const track = h('div', {class: 'slider-track'}, input);
  if (spec.danger) {
    const [from, to] = spec.danger;
    const fromFraction = getFraction(Math.min(from, to), spec.min, spec.max);
    const toFraction = getFraction(Math.max(from, to), spec.min, spec.max);
    const range = `${formatSliderValue(spec, Math.min(from, to))} to ${formatSliderValue(spec, Math.max(from, to))}`;
    for (const className of ['slider-danger-tint', 'slider-danger']) {
      const zone = h('span', {
        class: className,
        title: className === 'slider-danger' ? `Caution: ${range}` : undefined
      });
      zone.style.setProperty('--from', String(fromFraction));
      zone.style.setProperty('--to', String(toFraction));
      track.append(zone);
    }
  }
  const marks = spec.marks?.length
    ? h(
        'div',
        {
          class: spec.marks.some(mark => mark.label) ? 'slider-marks has-labels' : 'slider-marks',
          'aria-hidden': 'true'
        },
        spec.marks.map(mark => {
          const element = h(
            'span',
            {class: mark.label ? 'slider-mark has-label' : 'slider-mark'},
            mark.label ? h('span', {class: 'slider-mark-label'}, mark.label) : null
          );
          element.style.setProperty('--p', String(getFraction(mark.value, spec.min, spec.max)));
          return element;
        })
      )
    : null;

  const element = h(
    'div',
    {class: 'control control-slider'},
    h(
      'div',
      {class: 'control-row'},
      h('label', {for: parts.id}, spec.label, parts.badge),
      sweep?.element,
      value
    ),
    track,
    marks,
    describeLine,
    parts.help
  );
  return {
    element,
    spec,
    destroy: () => sweep?.destroy(),
    refresh: state => {
      const current = state[spec.id] as number;
      const fraction = getFraction(current, spec.min, spec.max);
      input.value = String(current);
      input.style.setProperty('--p', String(fraction));
      input.style.setProperty('--fill', `${fraction * 100}%`);
      value.textContent = formatSliderValue(spec, current);
      const inDanger = Boolean(spec.danger) && isWithin(current, spec.danger!);
      element.classList.toggle('is-danger', inDanger);
      const description = spec.describe?.(current, state as never) ?? '';
      if (describeLine.textContent !== description) describeLine.textContent = description;
      describeLine.hidden = !description;
      // A running sweep rewrites the line every frame; announcing each write would be noise.
      describeLine.setAttribute('aria-live', sweep?.isPlaying() ? 'off' : 'polite');
      input.disabled = parts.isDisabled(state);
      sweep?.setDisabled(input.disabled);
      element.classList.toggle('is-disabled', input.disabled);
    }
  };
}

function isWithin(value: number, [a, b]: readonly [number, number]): boolean {
  return value >= Math.min(a, b) && value <= Math.max(a, b);
}

function createStepperControl(
  spec: SliderSpec,
  store: OptionsStore,
  parts: ControlParts
): OptionControl {
  const labelId = `${parts.id}-label`;
  const count = Math.max(1, Math.round((spec.max - spec.min) / spec.step) + 1);
  const previous = h(
    'button',
    {type: 'button', class: 'stepper-button', 'aria-label': `Previous ${spec.label}`},
    icon('chevronLeft', 16)
  );
  const next = h(
    'button',
    {type: 'button', class: 'stepper-button', 'aria-label': `Next ${spec.label}`},
    icon('chevronRight', 16)
  );
  const value = h('output', {
    class: 'stepper-value',
    id: parts.id,
    role: 'spinbutton',
    tabindex: '0',
    'aria-labelledby': labelId,
    'aria-valuemin': String(spec.min),
    'aria-valuemax': String(spec.max)
  });
  const describeLine = h('p', {class: 'control-describe', 'aria-live': 'polite', hidden: true});
  const sweep = spec.autoSweep ? createSweep(spec, store, spec.autoSweep) : null;

  const move = (direction: number) => {
    sweep?.stop();
    const current = store.get(spec.id) as number;
    const target = snapToStep(spec, current + direction * spec.step);
    store.set(spec.id, target);
  };
  previous.addEventListener('click', () => move(-1));
  next.addEventListener('click', () => move(1));
  const stepper = h(
    'div',
    {class: 'stepper', role: 'group', 'aria-labelledby': labelId},
    previous,
    value,
    next
  );
  stepper.addEventListener('keydown', event => {
    let direction = 0;
    if (event.key === 'ArrowRight' || event.key === 'ArrowUp') direction = 1;
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') direction = -1;
    else if (event.key === 'Home' || event.key === 'End') {
      event.preventDefault();
      event.stopPropagation();
      sweep?.stop();
      store.set(spec.id, event.key === 'Home' ? spec.min : snapToStep(spec, spec.max));
      return;
    }
    if (!direction) return;
    // The story view steps with the arrow keys; keep them for the stepper.
    event.preventDefault();
    event.stopPropagation();
    move(direction);
  });

  const element = h(
    'div',
    {class: 'control control-stepper'},
    h(
      'div',
      {class: 'control-row'},
      h('span', {class: 'control-label', id: labelId}, spec.label, parts.badge),
      sweep?.element
    ),
    stepper,
    describeLine,
    parts.help
  );
  return {
    element,
    spec,
    destroy: () => sweep?.destroy(),
    refresh: state => {
      const current = state[spec.id] as number;
      const position = Math.round((current - spec.min) / spec.step) + 1;
      const text = spec.format ? spec.format(current) : `${position} of ${count}`;
      value.textContent = text;
      value.setAttribute('aria-valuenow', String(current));
      value.setAttribute('aria-valuetext', text);
      const description = spec.describe?.(current, state as never) ?? '';
      if (describeLine.textContent !== description) describeLine.textContent = description;
      describeLine.hidden = !description;
      describeLine.setAttribute('aria-live', sweep?.isPlaying() ? 'off' : 'polite');
      const disabled = parts.isDisabled(state);
      const wasFocused = document.activeElement;
      previous.disabled = disabled || current <= spec.min;
      next.disabled = disabled || current >= spec.max;
      // A button that disables itself at the end of the ladder would drop keyboard focus.
      if (
        (wasFocused === previous && previous.disabled) ||
        (wasFocused === next && next.disabled)
      ) {
        value.focus({preventScroll: true});
      }
      value.tabIndex = disabled ? -1 : 0;
      sweep?.setDisabled(disabled);
      element.classList.toggle('is-disabled', disabled);
    }
  };
}

function countDecimals(number: number): number {
  const [, fraction = ''] = String(number).split('.');
  return Math.min(8, fraction.length);
}

/** Clamps `value` to the slider range and snaps it to its step grid. */
function snapToStep(spec: SliderSpec, value: number): number {
  const stepped = spec.min + Math.round((value - spec.min) / spec.step) * spec.step;
  const clamped = Math.min(spec.max, Math.max(spec.min, stepped));
  return Number(clamped.toFixed(Math.max(countDecimals(spec.step), countDecimals(spec.min))));
}

// -- autoSweep

type Sweep = {
  element: HTMLElement;
  isPlaying: () => boolean;
  stop: () => void;
  setDisabled: (disabled: boolean) => void;
  destroy: () => void;
};

const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';

/** A filled 16px transport glyph (the shared icon set is stroke based and has none). */
function createGlyph(kind: 'play' | 'pause' | 'step'): SVGSVGElement {
  const svg = document.createElementNS(SVG_NAMESPACE, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('width', '14');
  svg.setAttribute('height', '14');
  svg.setAttribute('fill', 'currentColor');
  svg.setAttribute('aria-hidden', 'true');
  const path = document.createElementNS(SVG_NAMESPACE, 'path');
  path.setAttribute(
    'd',
    kind === 'play'
      ? 'M7 4.5v15a1 1 0 0 0 1.5.86l12-7.5a1 1 0 0 0 0-1.72l-12-7.5A1 1 0 0 0 7 4.5z'
      : kind === 'pause'
        ? 'M6 4h4v16H6zM14 4h4v16h-4z'
        : 'M5 4.5v15a1 1 0 0 0 1.5.86L16 14.4V19.5h3v-15h-3V9.6L6.5 3.64A1 1 0 0 0 5 4.5z'
  );
  svg.append(path);
  return svg;
}

/**
 * Play, Pause and step buttons that sweep a slider from `from` to `to` over `durationMs` with
 * `requestAnimationFrame`, writing every frame through `store.set` snapped to the slider step.
 * Under `prefers-reduced-motion` Play advances by one step per press instead.
 */
function createSweep(
  spec: SliderSpec,
  store: OptionsStore,
  sweep: NonNullable<SliderSpec['autoSweep']>
): Sweep {
  const from = sweep.from ?? spec.min;
  const to = sweep.to ?? spec.max;
  const direction = to >= from ? 1 : -1;
  let playing = false;
  let frame = 0;
  let startTime: number | null = null;
  let startProgress = 0;

  const ease = (t: number) => (sweep.ease === 'linear' ? t : 0.5 - Math.cos(Math.PI * t) / 2);
  const invertEase = (p: number) =>
    sweep.ease === 'linear' ? p : Math.acos(1 - 2 * Math.min(1, Math.max(0, p))) / Math.PI;

  const playButton = h('button', {type: 'button', class: 'sweep-button'});
  const stepButton = h(
    'button',
    {
      type: 'button',
      class: 'sweep-button',
      title: 'Step',
      'aria-label': `Step ${spec.label}`,
      hidden: true
    },
    createGlyph('step')
  );
  const element = h('span', {class: 'sweep'}, playButton, stepButton);

  const renderButtons = () => {
    const reduced = prefersReducedMotion();
    playButton.replaceChildren(createGlyph(playing ? 'pause' : reduced ? 'step' : 'play'));
    const label = playing ? 'Pause' : reduced ? 'Step' : 'Play';
    playButton.title = label;
    playButton.setAttribute('aria-label', `${label} ${spec.label}`);
    playButton.setAttribute('aria-pressed', String(playing));
    stepButton.hidden = !playing;
    element.classList.toggle('is-playing', playing);
  };

  const stop = () => {
    if (!playing) return;
    playing = false;
    cancelAnimationFrame(frame);
    frame = 0;
    renderButtons();
  };

  const write = (value: number) => store.set(spec.id, snapToStep(spec, value));

  const tick = (now: number) => {
    if (!playing) return;
    startTime ??= now;
    const progress = Math.min(1, startProgress + (now - startTime) / sweep.durationMs);
    write(from + (to - from) * ease(progress));
    if (progress < 1) {
      frame = requestAnimationFrame(tick);
    } else if (sweep.loop) {
      startTime = null;
      startProgress = 0;
      frame = requestAnimationFrame(tick);
    } else {
      stop();
    }
  };

  const stepOnce = () => {
    const current = store.get(spec.id) as number;
    const atEnd = Math.abs(current - to) < spec.step / 2 || (current - to) * direction > 0;
    const target = atEnd ? from : current + direction * spec.step;
    // Never step past the end of the sweep.
    write(direction > 0 ? Math.min(to, target) : Math.max(to, target));
  };

  const start = () => {
    const current = store.get(spec.id) as number;
    const fraction = to === from ? 0 : (current - from) / (to - from);
    // Resume from the current position when it lies inside the sweep, else start over.
    startProgress = fraction > 0 && fraction < 1 ? invertEase(fraction) : 0;
    startTime = null;
    playing = true;
    renderButtons();
    frame = requestAnimationFrame(tick);
  };

  playButton.addEventListener('click', () => {
    if (playing) stop();
    else if (prefersReducedMotion()) stepOnce();
    else start();
  });
  stepButton.addEventListener('click', () => {
    stop();
    stepOnce();
  });
  renderButtons();

  return {
    element,
    isPlaying: () => playing,
    stop,
    setDisabled: disabled => {
      playButton.disabled = stepButton.disabled = disabled;
      if (disabled) stop();
    },
    destroy: stop
  };
}

function prefersReducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

// -- Range

function createRangeControl(
  spec: RangeSpec,
  store: OptionsStore,
  parts: ControlParts
): OptionControl {
  const create = (suffix: string, label: string) =>
    h('input', {
      type: 'range',
      id: `${parts.id}-${suffix}`,
      min: spec.min,
      max: spec.max,
      step: spec.step,
      'aria-label': `${spec.label} ${label}`
    });
  const low = create('low', 'minimum');
  const high = create('high', 'maximum');
  const value = h('output', {class: 'control-value'});
  const commit = (dragged: HTMLInputElement) => {
    // The thumbs may touch but not cross, so the stored pair stays ordered.
    if (Number(low.value) > Number(high.value)) {
      (dragged === low ? low : high).value = (dragged === low ? high : low).value;
    }
    store.set(spec.id, [Number(low.value), Number(high.value)] as const);
  };
  low.addEventListener('input', () => commit(low));
  high.addEventListener('input', () => commit(high));
  const fill = h('span', {class: 'range-fill'});
  const pair = h('div', {class: 'range-pair'}, h('span', {class: 'range-track'}), fill, low, high);
  const element = h(
    'div',
    {class: 'control control-range'},
    h(
      'div',
      {class: 'control-row'},
      h('span', {class: 'control-label'}, spec.label, parts.badge),
      value
    ),
    pair,
    parts.help
  );
  return {
    element,
    spec,
    refresh: state => {
      const [a, b] = state[spec.id] as readonly [number, number];
      low.value = String(a);
      high.value = String(b);
      pair.style.setProperty('--p-low', String(getFraction(a, spec.min, spec.max)));
      pair.style.setProperty('--p-high', String(getFraction(b, spec.min, spec.max)));
      // Whichever thumb sits at the shared end must stay reachable.
      const crowdedHigh = getFraction(a, spec.min, spec.max) > 0.5;
      low.style.zIndex = crowdedHigh ? '2' : '1';
      high.style.zIndex = crowdedHigh ? '1' : '2';
      value.textContent = `${formatSliderValue(spec, a)} to ${formatSliderValue(spec, b)}`;
      low.disabled = high.disabled = parts.isDisabled(state);
      element.classList.toggle('is-disabled', low.disabled);
    }
  };
}

function formatNumber(value: number, step: number): string {
  const decimals = step >= 1 && Number.isInteger(step) ? 0 : Math.min(4, countDecimals(step));
  return value.toFixed(decimals);
}
