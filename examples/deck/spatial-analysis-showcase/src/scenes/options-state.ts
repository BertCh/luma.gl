// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OptionSpec, OptionState, OptionValue} from './scene';

/** Who changed an option: the user (panel, URL, story step, reset) or the scene through `ctx.setOptions`. */
export type OptionChangeSource = 'user' | 'scene';

type Listener = (id: string, value: OptionValue, source: OptionChangeSource) => void;

/** Returns the stateful options (everything except buttons). */
export function getStatefulOptions(options: readonly OptionSpec<never>[]) {
  return options.filter(option => option.kind !== 'button') as Exclude<
    OptionSpec<never>,
    {kind: 'button'}
  >[];
}

/** Default state of a schema. */
export function getDefaultState(options: readonly OptionSpec<never>[]): OptionState {
  const state: OptionState = {};
  for (const option of getStatefulOptions(options)) {
    state[option.id] = Array.isArray(option.default)
      ? ([...option.default] as unknown as [number, number])
      : (option.default as OptionValue);
  }
  return state;
}

/** Whether two option values are equal (ranges compare element-wise). */
export function areValuesEqual(a: OptionValue, b: OptionValue): boolean {
  if (Array.isArray(a) && Array.isArray(b)) return a[0] === b[0] && a[1] === b[1];
  return a === b;
}

/**
 * Mutable option state of the active scene with change notification and URL serialization.
 * The panel, the story steps, the URL and the scene instance all go through one store.
 */
export class OptionsStore {
  readonly specs: readonly OptionSpec<never>[];
  private readonly defaults: OptionState;
  private values: OptionState;
  private readonly listeners = new Set<Listener>();

  constructor(specs: readonly OptionSpec<never>[]) {
    this.specs = specs;
    this.defaults = getDefaultState(specs);
    this.values = {...this.defaults};
  }

  /** Current state as a plain object (a snapshot). */
  snapshot(): OptionState {
    return {...this.values};
  }

  /** Live read-only view for `ctx.options`. */
  get view(): Readonly<OptionState> {
    return this.values;
  }

  get(id: string): OptionValue {
    return this.values[id];
  }

  getDefault(id: string): OptionValue {
    return this.defaults[id];
  }

  /** Whether any option differs from its default. */
  isModified(): boolean {
    return Object.keys(this.defaults).some(
      id => !areValuesEqual(this.values[id], this.defaults[id])
    );
  }

  /** Whether one option differs from its default. Buttons and unknown ids are never modified. */
  isOptionModified(id: string): boolean {
    if (!(id in this.defaults)) return false;
    return !areValuesEqual(this.values[id], this.defaults[id]);
  }

  /** Sets one option and notifies listeners when the value changed. */
  set(id: string, value: OptionValue, source: OptionChangeSource = 'user'): void {
    if (!(id in this.defaults)) return;
    const next = Array.isArray(value) ? ([value[0], value[1]] as const) : value;
    if (areValuesEqual(this.values[id], next)) return;
    this.values[id] = next;
    for (const listener of this.listeners) listener(id, next, source);
  }

  /** Sets several options in order. */
  setMany(values: Partial<OptionState>, source: OptionChangeSource = 'user'): void {
    for (const [id, value] of Object.entries(values)) {
      if (value !== undefined) this.set(id, value, source);
    }
  }

  /** Restores every option to its default. */
  reset(): void {
    this.setMany(this.defaults);
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Serializes non-default values as `id:value~id:value` (no leading `o=`). */
  serialize(): string {
    const parts: string[] = [];
    for (const [id, value] of Object.entries(this.values)) {
      if (areValuesEqual(value, this.defaults[id])) continue;
      parts.push(`${id}:${encodeValue(value)}`);
    }
    return parts.join('~');
  }

  /** Applies a string produced by {@link serialize}; unknown ids and malformed values are ignored. */
  deserialize(text: string): void {
    for (const part of text.split('~')) {
      const separator = part.indexOf(':');
      if (separator < 0) continue;
      const id = part.slice(0, separator);
      const fallback = this.defaults[id];
      if (fallback === undefined) continue;
      const value = decodeValue(part.slice(separator + 1), fallback);
      if (value !== undefined) this.set(id, value);
    }
  }
}

function encodeValue(value: OptionValue): string {
  if (typeof value === 'boolean') return value ? '1' : '0';
  if (Array.isArray(value)) return `${value[0]}..${value[1]}`;
  return encodeURIComponent(String(value));
}

function decodeValue(text: string, fallback: OptionValue): OptionValue | undefined {
  if (typeof fallback === 'boolean') return text === '1' || text === 'true';
  if (typeof fallback === 'number') {
    const number = Number(text);
    return Number.isFinite(number) ? number : undefined;
  }
  if (Array.isArray(fallback)) {
    const [low, high] = text.split('..').map(Number);
    return Number.isFinite(low) && Number.isFinite(high) ? ([low, high] as const) : undefined;
  }
  return decodeURIComponent(text);
}
