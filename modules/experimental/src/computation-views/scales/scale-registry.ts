// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {VariableKind, ViewGround} from '../types';
import {
  type ClassificationMethod as CartographyClassificationMethod,
  formatBreakLabels,
  getClassIndex,
  getExtent,
  getQuantileBreaks
} from './breaks';
import {rgbToHex} from './class-table';
import {directionFor, getClassColors, type RampName, sampleRamp} from './ramps';
import type {ScaleRuleId} from './scale-validator';

/** Classification vocabulary, including the short Jenks spelling used by view specs. */
export type ClassificationMethod = CartographyClassificationMethod | 'natural';
/** Measurement family of a shared colour scale. */
export type ScaleKind = 'sequential' | 'diverging' | 'cyclic' | 'qualitative';
/** Fixed classification shared by every view in a run. */
export type ScaleClassification =
  | {
      kind: 'classed';
      method: ClassificationMethod;
      breaks: readonly number[];
      source: 'published-table' | 'computed';
    }
  | {kind: 'unclassed'; purpose: 'continuous-inspection'};
/** Evidence used to establish a scale's fixed domain. */
export type ScaleDomainSource = 'final-state' | 'declared' | 'data';
/** An explicitly reasoned departure from a composition rule. */
export type ScaleRuleException = {rule: ScaleRuleId; reason: string};
/** Complete colour mapping for one variable in one run. */
export type ScaleSpec = {
  id: string;
  variable: string;
  variableKind: VariableKind;
  kind: ScaleKind;
  ramp?: RampName;
  palette?: readonly string[];
  domain: readonly [number, number];
  domainSource: ScaleDomainSource;
  classification: ScaleClassification;
  midpoint?: number;
  beyondRange?: {label: string};
  noDataColor: string;
  unit?: string;
  ground: ViewGround;
  reverse?: boolean;
  exceptions?: readonly ScaleRuleException[];
};
/** Reference scoped to the run that owns the scale. */
export type ScaleRef = {readonly scaleId: string; readonly runId: string};
/** One labelled class; outside-domain entries retain their sentinel class index. */
export type ScaleLegendClass = {index: number; color: string; label: string; beyondRange?: boolean};
/** Legend derived from the same mapping used to colour marks. */
export type ScaleLegend = {
  scaleId: string;
  classes: readonly ScaleLegendClass[];
  unit?: string;
  midpoint?: number;
  midpointText?: string;
  noData: {color: string; label: string};
};
/** Frozen scale with pure colour, classification and legend accessors. */
export type ResolvedScale = Readonly<ScaleSpec> & {
  readonly runId: string;
  colorFor(value: number | null | undefined): string;
  /** Below domain or missing is -1; above domain is classCount when beyondRange is declared. */
  classFor(value: number): number;
  legend(): ScaleLegend;
};

function freezeDeep<Value>(value: Value): Value {
  if (value && typeof value === 'object') {
    for (const child of Object.values(value)) freezeDeep(child);
    Object.freeze(value);
  }
  return value;
}

function fingerprint(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(fingerprint).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.entries(value)
      .filter(([, child]) => child !== undefined)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => `${JSON.stringify(key)}:${fingerprint(child)}`)
      .join(',')}}`;
  }
  return JSON.stringify(value) ?? 'undefined';
}

function resolveScale(spec: ScaleSpec, runId: string): ResolvedScale {
  const frozen = freezeDeep(structuredClone(spec));
  const breaks = frozen.classification.kind === 'classed' ? frozen.classification.breaks : [];
  const classCount = breaks.length + 1;
  const ramp = frozen.ramp;
  // Every scale supplies its own colour source; never infer a palette for missing metadata.
  assert(frozen.kind === 'qualitative' ? Boolean(frozen.palette?.length) : Boolean(ramp));
  // A scale's domain is finite and ordered, including constant final states.
  assert(frozen.domain.every(Number.isFinite) && frozen.domain[0] <= frozen.domain[1]);
  const colors =
    frozen.kind === 'qualitative'
      ? Array.from(
          {length: classCount},
          (_, index) => frozen.palette![index % frozen.palette!.length]
        )
      : getClassColors(ramp!, classCount, frozen.reverse).map(rgbToHex);
  if (frozen.kind === 'diverging' && frozen.midpoint !== undefined && ramp) {
    const middleClass = getClassIndex(frozen.midpoint, breaks);
    for (let index = 0; index < classCount; index++) {
      const position =
        index === middleClass
          ? 0.5
          : index < middleClass
            ? (0.5 * index) / middleClass
            : 0.5 + (0.5 * (index - middleClass)) / (classCount - 1 - middleClass);
      colors[index] = rgbToHex(sampleRamp(ramp, position, frozen.reverse));
    }
  }
  const classFor = (value: number): number => {
    if (Number.isNaN(value) || value < frozen.domain[0]) return -1;
    if (value > frozen.domain[1] && frozen.beyondRange) return classCount;
    return getClassIndex(Math.min(value, frozen.domain[1]), breaks);
  };
  const colorFor = (value: number | null | undefined): string => {
    if (value === null || value === undefined || Number.isNaN(value)) return frozen.noDataColor;
    const clamped = Math.max(frozen.domain[0], Math.min(frozen.domain[1], value));
    if (frozen.classification.kind === 'classed' || frozen.kind === 'qualitative') {
      return colors[Math.max(0, Math.min(classCount - 1, classFor(clamped)))];
    }
    const [minimum, maximum] = frozen.domain;
    let position = maximum === minimum ? 0.5 : (clamped - minimum) / (maximum - minimum);
    if (frozen.kind === 'diverging' && frozen.midpoint !== undefined) {
      const midpoint = frozen.midpoint;
      position =
        clamped === midpoint
          ? 0.5
          : clamped < midpoint
            ? (0.5 * (clamped - minimum)) / (midpoint - minimum)
            : 0.5 + (0.5 * (clamped - midpoint)) / (maximum - midpoint);
    }
    return rgbToHex(sampleRamp(ramp!, position, frozen.reverse));
  };
  const labels = formatBreakLabels(breaks, frozen.domain, String);
  const classes: ScaleLegendClass[] = labels.map((label, index) => ({
    index,
    label,
    color: colors[index]
  }));
  if (frozen.beyondRange) {
    classes.unshift({
      index: -1,
      label: `${frozen.beyondRange.label}: < ${frozen.domain[0]}`,
      color: colorFor(-Infinity),
      beyondRange: true
    });
    classes.push({
      index: classCount,
      label: `${frozen.beyondRange.label}: > ${frozen.domain[1]}`,
      color: colorFor(Infinity),
      beyondRange: true
    });
  }
  const legend = freezeDeep<ScaleLegend>({
    scaleId: frozen.id,
    classes,
    unit: frozen.unit,
    midpoint: frozen.midpoint,
    midpointText:
      frozen.midpoint === undefined
        ? undefined
        : `Midpoint: ${frozen.midpoint}${frozen.unit ? ` ${frozen.unit}` : ''}`,
    noData: {color: frozen.noDataColor, label: 'No data'}
  });
  return Object.freeze({...frozen, runId, colorFor, classFor, legend: () => legend});
}

/** Owns one immutable mapping per variable; freeze seals the set of variables for a run. */
export class ScaleRegistry {
  readonly runId: string;
  private scales = new Map<string, ResolvedScale>();
  private variables = new Map<string, ScaleRef>();
  private fingerprints = new Map<string, string>();
  private frozen = false;

  constructor(props: {runId: string}) {
    this.runId = props.runId;
  }

  /** Registers a detached frozen copy, or returns the existing reference for an identical spec. */
  register(spec: ScaleSpec): ScaleRef {
    const existing = this.variables.get(spec.variable);
    const signature = fingerprint(spec);
    if (existing) {
      // A variable's mapping cannot change during a run, including after freeze.
      assert(this.fingerprints.get(spec.variable) === signature);
      return existing;
    }
    // Frozen runs cannot gain variables, and ids cannot refer to two variables.
    assert(!this.frozen && !this.scales.has(spec.id));
    const scale = resolveScale(spec, this.runId);
    const reference = Object.freeze({scaleId: spec.id, runId: this.runId});
    this.scales.set(spec.id, scale);
    this.variables.set(spec.variable, reference);
    this.fingerprints.set(spec.variable, signature);
    return reference;
  }

  /** Resolves only references owned by this registry's run. */
  get(ref: ScaleRef): ResolvedScale {
    // A reference from another run must never silently resolve to a same-named scale.
    assert(ref.runId === this.runId);
    const scale = this.scales.get(ref.scaleId);
    // References must name a registered scale.
    assert(scale);
    return scale;
  }

  /** Finds the one registered reference for a variable. */
  getByVariable(variable: string): ScaleRef | undefined {
    return this.variables.get(variable);
  }

  /** Seals registration; identical registrations remain idempotent. */
  freeze(): void {
    this.frozen = true;
  }

  /** Returns a fresh list containing the shared frozen scale objects. */
  list(): ResolvedScale[] {
    return [...this.scales.values()];
  }
}

/** Inputs for a final-state quantile scale; published-table refers to authored specs, not computed breaks. */
export type QuantileScaleOptions = Omit<
  ScaleSpec,
  'domain' | 'domainSource' | 'classification' | 'reverse' | 'kind'
> & {
  finalStateValues: ArrayLike<number>;
  classCount?: number;
};

/** Computes fixed final-state quantiles and chooses ramp direction for the host ground. */
export function createQuantileScaleSpec(options: QuantileScaleOptions): ScaleSpec {
  const {finalStateValues, classCount = 5, ...spec} = options;
  const domain = getExtent(finalStateValues);
  // A final-state scale needs finite evidence and a positive integer class count.
  assert(domain.every(Number.isFinite) && Number.isInteger(classCount) && classCount > 0);
  const ramp = spec.ramp ?? 'magma';
  return {
    ...spec,
    ramp,
    kind: 'sequential',
    domain,
    domainSource: 'final-state',
    classification: {
      kind: 'classed',
      method: 'quantile',
      breaks: getQuantileBreaks(finalStateValues, classCount),
      source: 'computed'
    },
    reverse: directionFor(spec.ground, ramp).reverse,
    beyondRange: spec.beyondRange ?? {label: 'Beyond final-state range'}
  };
}

/** Computes final-state quantiles with an explicit meaningful diverging midpoint. */
export function createDivergingScaleSpec(
  options: QuantileScaleOptions & {midpoint: number}
): ScaleSpec {
  const spec = createQuantileScaleSpec({...options, ramp: options.ramp ?? 'rdbu'});
  // The midpoint must be finite and lie in the evidence domain.
  assert(
    Number.isFinite(options.midpoint) &&
      options.midpoint >= spec.domain[0] &&
      options.midpoint <= spec.domain[1]
  );
  return {...spec, kind: 'diverging'};
}
