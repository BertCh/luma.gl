// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {AxisId} from '../types';
import {hexToRgba} from './class-table';
import {directionFor, RAMP_INFO, type RampName} from './ramps';
import type {
  ResolvedScale,
  ScaleRef,
  ScaleRegistry,
  ScaleRuleException,
  ScaleSpec
} from './scale-registry';

/** Composition rules from roadmap §E plus cross-view consistency constraints. */
export type ScaleRuleId =
  | 'one-scale-per-variable'
  | 'iteration-domain'
  | 'classed-default'
  | 'hue-only-nominal'
  | 'no-rainbow-ordered'
  | 'iteration-own-channel'
  | 'separate-variables'
  | 'ramp-direction-ground'
  | 'spatial-support-drawn'
  | 'structure-no-hue'
  | 'labels-above-data'
  | 'animation-not-evidence'
  | 'consistent-mapping'
  | 'distinct-variables-distinct-palettes';

/** Channel assignments of one view; a validation call represents views sharing a page. */
export type ViewEncodingSpec = {
  viewId: string;
  color?: ScaleRef;
  structureColor?: string;
  iterationChannel?: 'position' | 'small-multiple' | 'animate' | 'key-frame-strip' | 'color';
  provenanceChannel: 'texture' | 'saturation' | 'alpha' | 'none';
  provisionalChannel: 'crispness' | 'saturation' | 'alpha' | 'none';
  uncertaintyChannel?: 'crispness' | 'saturation' | 'alpha' | 'none';
  animated?: {
    staticCompanion: 'key-frame-strip' | 'chart' | 'none';
    reducedMotion: 'stepped' | 'continuous';
  };
  geometry?: {spatial: boolean; supportDrawn: boolean};
  labelPlacement?: 'above-data' | 'below-data';
  freeAxes?: readonly AxisId[];
  exceptions?: readonly ScaleRuleException[];
};

/** Actionable diagnostic, retaining any declared exception reason. */
export type ScaleValidationIssue = {
  rule: ScaleRuleId;
  severity: 'error' | 'warning' | 'info';
  message: string;
  scaleId?: string;
  viewId?: string;
  exceptionReason?: string;
};

/** Numbered composition rules and the two page consistency constraints. */
export const SCALE_RULES: Record<ScaleRuleId, {number?: number; summary: string}> = {
  'one-scale-per-variable': {
    number: 1,
    summary: 'One scale object per variable. It never changes per frame, filter or view.'
  },
  'iteration-domain': {
    number: 2,
    summary:
      'Iteration domains come from the final state or a declared reference. Values that overshoot go in a stated "beyond range" class.'
  },
  'classed-default': {
    number: 3,
    summary:
      'Classed by default. Use quantile classes from published tables, not Jenks, unless argued otherwise. Use an unclassed fixed domain only for continuous inspection.'
  },
  'hue-only-nominal': {
    number: 4,
    summary:
      'Hue only for nominal data. Use lightness for ordered data. Use diverging scales only with a stated meaningful midpoint.'
  },
  'no-rainbow-ordered': {
    number: 5,
    summary: 'No turbo or rainbow for ordered data. Cyclic ramps are only for periodic quantities.'
  },
  'iteration-own-channel': {
    number: 6,
    summary: "Iteration time gets its own channel, never the variable's ramp."
  },
  'separate-variables': {
    number: 7,
    summary:
      'Provenance, provisional state and value use separate visual variables: lineage uses texture, provisional state uses crispness, value uses lightness. Uncertainty is never saturation, and value-by-alpha never doubles as compositing.'
  },
  'ramp-direction-ground': {
    number: 8,
    summary:
      'Ramp direction follows the ground. Views placed beside a night map inherit the night tables.'
  },
  'spatial-support-drawn': {
    number: 9,
    summary:
      'Spatial support is always drawn: cell size, bandwidth or neighbourhood in true metres.'
  },
  'structure-no-hue': {
    number: 10,
    summary:
      'Structure carries no hue. Hierarchy tiers apply to graphs too: subject at full ink, inputs at half, context faint.'
  },
  'labels-above-data': {
    number: 11,
    summary: 'Labels go above the data, with greedy priority placement.'
  },
  'animation-not-evidence': {
    number: 12,
    summary:
      'Animation is never the evidence. It is user-paced, always paired with a static strip or chart, and turned into stepped frames under reduced motion.'
  },
  'consistent-mapping': {summary: 'Views of the same variable share the same scale reference.'},
  'distinct-variables-distinct-palettes': {
    summary: 'Different variables on a shared page use distinguishable palettes.'
  }
};

/** Rainbow families forbidden for ordered quantities. */
export const RAINBOW_RAMPS: readonly RampName[] = ['spectral', 'rdylbu'];

/** True for valid hex colours whose HSL saturation is at most 0.12. */
export function isAchromatic(hex: string): boolean {
  if (!/^#(?:[\da-f]{3}|[\da-f]{4}|[\da-f]{6}|[\da-f]{8})$/i.test(hex)) return false;
  const channels = hexToRgba(hex)
    .slice(0, 3)
    .map(channel => channel / 255);
  const minimum = Math.min(...channels);
  const maximum = Math.max(...channels);
  const difference = maximum - minimum;
  return difference === 0 || difference / (1 - Math.abs(maximum + minimum - 1)) <= 0.12;
}

function issue(
  rule: ScaleRuleId,
  severity: ScaleValidationIssue['severity'],
  message: string,
  context: {scaleId?: string; viewId?: string; exceptions?: readonly ScaleRuleException[]}
): ScaleValidationIssue {
  const exception = context.exceptions?.find(
    candidate => candidate.rule === rule && candidate.reason.trim()
  );
  return {
    rule,
    severity: exception ? 'info' : severity,
    message,
    ...(context.scaleId ? {scaleId: context.scaleId} : {}),
    ...(context.viewId ? {viewId: context.viewId} : {}),
    ...(exception ? {exceptionReason: exception.reason} : {})
  };
}

/** Checks the rules determined by a scale alone. */
export function validateScaleSpec(spec: ScaleSpec): ScaleValidationIssue[] {
  const issues: ScaleValidationIssue[] = [];
  const add = (rule: ScaleRuleId, severity: ScaleValidationIssue['severity'], message: string) => {
    issues.push(issue(rule, severity, message, {scaleId: spec.id, exceptions: spec.exceptions}));
  };
  if (spec.domainSource === 'final-state' && !spec.beyondRange) {
    add('iteration-domain', 'warning', 'Final-state domains need a stated beyond-range class.');
  }
  if (spec.classification.kind === 'unclassed') {
    if (spec.classification.purpose !== 'continuous-inspection')
      add('classed-default', 'error', 'Unclassed scales require continuous-inspection purpose.');
  } else {
    const {method, source} = spec.classification;
    if (method === 'natural' || method === 'natural-breaks')
      add(
        'classed-default',
        'warning',
        'Jenks breaks require an explicit argument for comparable views.'
      );
    if (source === 'computed' && method !== 'quantile' && method !== 'equal-interval')
      add(
        'classed-default',
        'info',
        'Computed non-quantile classification departs from the classed default.'
      );
  }
  const level = spec.variableKind.level;
  if ((spec.kind === 'qualitative') !== (level === 'nominal'))
    add(
      'hue-only-nominal',
      'error',
      'Qualitative palettes are reserved for nominal variables, which require qualitative scales.'
    );
  if (spec.kind === 'diverging' && (spec.midpoint === undefined || !Number.isFinite(spec.midpoint)))
    add('hue-only-nominal', 'error', 'Diverging scales require a meaningful finite midpoint.');
  if (spec.ramp) {
    if (['ordinal', 'interval', 'ratio'].includes(level) && RAINBOW_RAMPS.includes(spec.ramp))
      add('no-rainbow-ordered', 'error', 'Rainbow ramps cannot encode ordered values.');
    if (RAMP_INFO[spec.ramp].kind === 'cyclic' && level !== 'cyclic')
      add('no-rainbow-ordered', 'error', 'Cyclic ramps require periodic quantities.');
    if (level === 'cyclic' && RAMP_INFO[spec.ramp].kind !== 'cyclic')
      add('no-rainbow-ordered', 'warning', 'A periodic quantity should use a cyclic ramp.');
    if (spec.reverse !== directionFor(spec.ground, spec.ramp).reverse)
      add('ramp-direction-ground', 'error', 'Ramp reversal must follow the host ground.');
  }
  return issues;
}

/** Checks scales together, including duplicate variable registrations before registry construction. */
export function validateScaleSpecs(specs: readonly ScaleSpec[]): ScaleValidationIssue[] {
  const issues = specs.flatMap(validateScaleSpec);
  const variables = new Map<string, ScaleSpec>();
  for (const spec of specs) {
    const previous = variables.get(spec.variable);
    if (previous && previous.id !== spec.id) {
      issues.push(
        issue(
          'one-scale-per-variable',
          'error',
          `Variable ${spec.variable} has more than one scale id.`,
          {
            scaleId: spec.id,
            exceptions: [...(spec.exceptions ?? []), ...(previous.exceptions ?? [])]
          }
        )
      );
    }
    variables.set(spec.variable, spec);
  }
  return issues;
}

/** Validates a page's scale mappings and channel assignments; unresolved refs are diagnostics. */
export function validateViewEncodings(
  registry: ScaleRegistry,
  encodings: readonly ViewEncodingSpec[]
): ScaleValidationIssue[] {
  const issues = validateScaleSpecs(registry.list());
  const resolved: {encoding: ViewEncodingSpec; scale: ResolvedScale}[] = [];
  for (const encoding of encodings) {
    let scale: ResolvedScale | undefined;
    if (encoding.color) {
      try {
        scale = registry.get(encoding.color);
        resolved.push({encoding, scale});
      } catch {
        issues.push(
          issue(
            'consistent-mapping',
            'error',
            'Colour reference is missing or belongs to another run.',
            {
              viewId: encoding.viewId,
              scaleId: encoding.color.scaleId,
              exceptions: encoding.exceptions
            }
          )
        );
      }
    }
    const add = (
      rule: ScaleRuleId,
      severity: ScaleValidationIssue['severity'],
      message: string
    ) => {
      issues.push(
        issue(rule, severity, message, {
          viewId: encoding.viewId,
          scaleId: scale?.id,
          exceptions: [...(encoding.exceptions ?? []), ...(scale?.exceptions ?? [])]
        })
      );
    };
    if (encoding.freeAxes?.includes('iteration') && scale?.domainSource === 'data')
      add('iteration-domain', 'error', 'Iteration views require final-state or declared domains.');
    if (encoding.iterationChannel === 'color' && encoding.color)
      add('iteration-own-channel', 'error', 'Iteration cannot reuse the value colour channel.');
    if (
      encoding.provenanceChannel !== 'texture' ||
      !['crispness', 'none'].includes(encoding.provisionalChannel) ||
      encoding.uncertaintyChannel === 'saturation'
    )
      add(
        'separate-variables',
        'error',
        'Use texture for provenance, crispness for provisional state, and never saturation for uncertainty.'
      );
    if (encoding.uncertaintyChannel === 'alpha')
      add('separate-variables', 'warning', 'Uncertainty alpha must not double as compositing.');
    if (encoding.geometry?.spatial && !encoding.geometry.supportDrawn)
      add('spatial-support-drawn', 'error', 'Spatial views must draw their metric support.');
    if (encoding.structureColor && !isAchromatic(encoding.structureColor))
      add('structure-no-hue', 'error', 'Structural marks must have HSL saturation at most 0.12.');
    if (encoding.labelPlacement === 'below-data')
      add('labels-above-data', 'warning', 'Labels should be above the data.');
    if (encoding.animated?.staticCompanion === 'none')
      add('animation-not-evidence', 'error', 'Animation needs a static strip or chart companion.');
    if (encoding.animated?.reducedMotion === 'continuous')
      add('animation-not-evidence', 'error', 'Reduced motion requires stepped frames.');
  }
  for (let index = 0; index < resolved.length; index++) {
    const current = resolved[index];
    for (const previous of resolved.slice(0, index)) {
      const context = {
        viewId: current.encoding.viewId,
        scaleId: current.scale.id,
        exceptions: [
          ...(current.encoding.exceptions ?? []),
          ...(current.scale.exceptions ?? []),
          ...(previous.encoding.exceptions ?? []),
          ...(previous.scale.exceptions ?? [])
        ]
      };
      if (
        current.scale.variable === previous.scale.variable &&
        (current.encoding.color!.scaleId !== previous.encoding.color!.scaleId ||
          current.encoding.color!.runId !== previous.encoding.color!.runId)
      ) {
        issues.push(
          issue(
            'consistent-mapping',
            'error',
            'Views of one variable must share the same scale reference.',
            context
          )
        );
      }
      const samePalette =
        current.scale.ramp !== undefined
          ? current.scale.ramp === previous.scale.ramp
          : Boolean(
              current.scale.palette &&
                previous.scale.palette &&
                current.scale.palette.length === previous.scale.palette.length &&
                current.scale.palette.every(
                  (color, paletteIndex) => color === previous.scale.palette![paletteIndex]
                )
            );
      if (
        current.scale.variable !== previous.scale.variable &&
        samePalette &&
        current.scale.domain.every(
          (value, domainIndex) => value === previous.scale.domain[domainIndex]
        )
      ) {
        issues.push(
          issue(
            'distinct-variables-distinct-palettes',
            'warning',
            'Different variables on this page share a palette and domain.',
            context
          )
        );
      }
    }
  }
  return issues;
}
