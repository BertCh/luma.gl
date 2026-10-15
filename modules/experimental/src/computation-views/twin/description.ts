// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  DescriptionFacts,
  DescriptionKeyNumber,
  LensDescription,
  LensRungId,
  MarkProvenance
} from '../types';
import type {ViewContext, ViewRegistration} from '../registry/view-registration';

const PROVENANCE_PHRASES: Record<MarkProvenance, string> = {
  readback: 'values read back from the GPU run',
  replay: 'values recomputed by a GPU replay',
  derived: 'derived values',
  metadata: 'structure metadata only',
  unknown: 'provenance unknown'
};

/** Optional authored caption and explicit unavailable-rung explanation. */
export type LensDescriptionOptions = {rung?: LensRungId; caption?: string; disabledReason?: string};

/** Formats a key number's value with its unit only (`4.2 ms`, `1,024`). */
export function formatKeyNumberValue(keyNumber: DescriptionKeyNumber): string {
  const value =
    typeof keyNumber.value === 'number'
      ? keyNumber.value.toLocaleString(
          'en-US',
          Number.isInteger(keyNumber.value) ? {} : {maximumSignificantDigits: 4}
        )
      : keyNumber.value;
  return `${value}${keyNumber.unit ? ` ${keyNumber.unit}` : ''}`;
}

/** Formats a key number with its unit, or its label when no unit was supplied. */
export function formatKeyNumber(keyNumber: DescriptionKeyNumber): string {
  const value =
    typeof keyNumber.value === 'number'
      ? keyNumber.value.toLocaleString(
          'en-US',
          Number.isInteger(keyNumber.value) ? {} : {maximumSignificantDigits: 4}
        )
      : keyNumber.value;
  return `${value} ${keyNumber.unit ?? keyNumber.label}`.trim();
}

/** Generates alt, live-region and Markdown descriptions from the same facts. */
export function generateLensDescription(
  facts: DescriptionFacts,
  options: LensDescriptionOptions = {}
): LensDescription {
  const formatted = formatKeyNumberValue(facts.keyNumber);
  const provenance = PROVENANCE_PHRASES[facts.provenance];
  const unavailable = options.disabledReason
    ? `${options.rung ?? 'Requested rung'} unavailable: ${options.disabledReason}`
    : undefined;
  const firstFact = facts.facts[0];
  const alt = [
    unavailable,
    facts.subject,
    `${facts.keyNumber.label}: ${formatted}`,
    provenance,
    firstFact
  ]
    .filter(Boolean)
    .map(sentence => `${sentence!.replace(/[.]+$/, '')}.`)
    .join(' ');
  const live =
    [`${facts.keyNumber.label} ${formatted}`, unavailable, firstFact]
      .filter(Boolean)
      .join('; ')
      .replace(/[.]+$/, '') + '.';
  const bullets = [
    `${facts.keyNumber.label}: ${formatted}`,
    ...facts.facts,
    ...(facts.caveats ?? []),
    ...(unavailable ? [unavailable] : [])
  ];
  return {
    alt: alt.length > 250 ? `${alt.slice(0, 249).trimEnd()}…` : alt,
    live,
    llms: [
      options.caption ? `**${options.caption}**` : '',
      facts.subject,
      '',
      ...bullets.map(fact => `- ${fact}`),
      '',
      `Provenance: ${provenance}.`,
      `Adapter: ${facts.adapterLabel ?? 'unknown'}.`
    ]
      .filter((line, index) => index > 0 || line !== '')
      .join('\n')
  };
}

/** Uses a view's facts as the sole source of all textual twins. */
export function describeLensView<Model>(
  view: ViewRegistration<Model>,
  model: Model,
  context: ViewContext,
  options: LensDescriptionOptions = {}
): LensDescription {
  return generateLensDescription(view.describe(model, context), options);
}
