// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LensRecord, ViewPackData} from '../registry/view-registration';
import type {VariableKind} from '../types';

/** A missing premise stays unknown rather than silently passing. */
export type GuardStatus = 'pass' | 'fail' | 'unknown';
/** An evaluated premise and the glyphs it gates. */
export type GuardVerdict = {
  guardId: string;
  label: string;
  status: GuardStatus;
  reason: string;
  blocks: readonly string[];
};
/** Metadata supplied explicitly by records and domain packs. */
export type GuardContext = {
  metadata: Readonly<Record<string, string | number | boolean | undefined>>;
  variableKinds?: Readonly<Record<string, VariableKind>>;
  record?: LensRecord;
};
/** A named, metadata-driven condition for drawing a glyph honestly. */
export type GuardSpec = {
  id: string;
  label: string;
  blocks: readonly string[];
  evaluate(context: GuardContext): {status: GuardStatus; reason: string};
};

/** Evaluates guards in declared order. */
export function evaluateGuards(specs: readonly GuardSpec[], context: GuardContext): GuardVerdict[] {
  return specs.map(spec => ({
    guardId: spec.id,
    label: spec.label,
    blocks: spec.blocks,
    ...spec.evaluate(context)
  }));
}

/** Finds the first failing or unknown premise for a glyph. */
export function getBlockingVerdict(
  verdicts: readonly GuardVerdict[],
  glyphId: string
): GuardVerdict | undefined {
  return verdicts.find(verdict => verdict.status !== 'pass' && verdict.blocks.includes(glyphId));
}

/**
 * Gives a guard fixed fallback metadata (for example an operator's declared coordinate model);
 * metadata supplied by packs or records at evaluation time takes precedence.
 */
export function withGuardMetadata(
  guard: GuardSpec,
  metadata: Readonly<Record<string, string | number | boolean>> | undefined
): GuardSpec {
  return metadata && Object.keys(metadata).length
    ? {
        ...guard,
        evaluate: context =>
          guard.evaluate({...context, metadata: {...metadata, ...context.metadata}})
      }
    : guard;
}

/** Merges packs in order; later explicitly supplied metadata takes precedence. */
export function createGuardContext(
  record: LensRecord,
  packs: readonly ViewPackData[]
): GuardContext {
  return {
    record,
    metadata: Object.assign({}, ...packs.map(pack => pack.metadata)),
    variableKinds: Object.assign({}, ...packs.map(pack => pack.variableKinds))
  };
}
