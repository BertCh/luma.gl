// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {
  GPUDecisionRecord,
  GPUDecisionCandidate
} from '../computation-record/computation-record-types';

export type GPUStrategyDecision<Id extends string = string, Details = unknown> = {
  /** Stable implementation identifier. */
  id: Id;
  /** Higher scores are preferred. */
  score: number;
  /** Human-readable explanation for inspector/debug output. */
  reason: string;
  /** Operation-specific execution details. */
  details: Details;
};

export type GPUStrategyCandidate<Id extends string, Workload, Details> = {
  id: Id;
  isSupported?: (context: {device: Device; workload: Workload}) => boolean;
  /** Explains an unsupported candidate when decision recording is enabled. */
  unsupportedReason?: (context: {device: Device; workload: Workload}) => string;
  /** Identifies its first unsupported feature or limit. */
  failingFeature?: (context: {device: Device; workload: Workload}) => string;
  score: (context: {device: Device; workload: Workload}) => number;
  reason: (context: {device: Device; workload: Workload}) => string;
  createDetails: (context: {device: Device; workload: Workload}) => Details;
};

/**
 * Deterministically selects one implementation strategy for a GPU workload.
 *
 * Strategies expose policy without leaking it into public algorithm APIs. Today scores are
 * heuristic; later the same candidate contract can be fed by cached autotuning measurements.
 */
export function selectGPUStrategy<Id extends string, Workload, Details>(props: {
  device: Device;
  workload: Workload;
  candidates: readonly GPUStrategyCandidate<Id, Workload, Details>[];
  preferredId?: Id;
  /** Receives the complete candidate table; absent callbacks incur no recording work. */
  onDecision?: (decision: GPUDecisionRecord) => void;
  /** Stable subject for recorded decisions. */
  subject?: string;
}): GPUStrategyDecision<Id, Details> {
  if (props.candidates.length === 0)
    throw new Error('GPU strategy candidate list must not be empty');
  const context = {device: props.device, workload: props.workload};
  if (props.onDecision) {
    const candidates: GPUDecisionCandidate[] = props.candidates.map(candidate => {
      const supported = candidate.isSupported?.(context) ?? true;
      return supported
        ? {
            id: candidate.id,
            supported,
            score: candidate.score(context),
            reason: candidate.reason(context)
          }
        : {
            id: candidate.id,
            supported,
            reason: candidate.unsupportedReason?.(context) ?? 'unsupported',
            ...(candidate.failingFeature ? {failingFeature: candidate.failingFeature(context)} : {})
          };
    });
    let selectedIndex = -1;
    for (let index = 0; index < candidates.length; index++) {
      const candidate = candidates[index];
      if (!candidate.supported) continue;
      if (props.preferredId) {
        if (candidate.id === props.preferredId) {
          selectedIndex = index;
          break;
        }
      } else if (selectedIndex < 0 || candidate.score! > candidates[selectedIndex].score!) {
        selectedIndex = index;
      }
    }
    if (!candidates.some(candidate => candidate.supported))
      throw new Error('no supported GPU strategy candidate');
    if (selectedIndex < 0)
      throw new Error(`preferred GPU strategy "${props.preferredId}" is not supported`);
    const candidate = candidates[selectedIndex];
    const selected = props.candidates[selectedIndex];
    const decision = Object.freeze({
      id: selected.id,
      score: candidate.score!,
      reason: candidate.reason,
      details: Object.freeze(selected.createDetails(context)) as Details
    });
    const subject = props.subject ?? 'gpu-strategy';
    props.onDecision({
      id: `strategy:${subject}`,
      subject,
      kind: 'strategy',
      chosen: decision.id,
      reason: decision.reason,
      method: props.preferredId ? 'preferred' : 'score',
      candidates
    });
    return decision;
  }
  const supported = props.candidates.filter(candidate => candidate.isSupported?.(context) ?? true);
  if (supported.length === 0) throw new Error('no supported GPU strategy candidate');

  if (props.preferredId) {
    const preferred = supported.find(candidate => candidate.id === props.preferredId);
    if (!preferred)
      throw new Error(`preferred GPU strategy "${props.preferredId}" is not supported`);
    return freezeDecision(preferred, context);
  }

  let best = supported[0];
  let bestScore = best.score(context);
  for (let index = 1; index < supported.length; index++) {
    const candidate = supported[index];
    const score = candidate.score(context);
    if (score > bestScore) {
      best = candidate;
      bestScore = score;
    }
  }
  return freezeDecision(best, context, bestScore);
}

function freezeDecision<Id extends string, Workload, Details>(
  candidate: GPUStrategyCandidate<Id, Workload, Details>,
  context: {device: Device; workload: Workload},
  score = candidate.score(context)
): GPUStrategyDecision<Id, Details> {
  return Object.freeze({
    id: candidate.id,
    score,
    reason: candidate.reason(context),
    details: Object.freeze(candidate.createDetails(context)) as Details
  });
}
