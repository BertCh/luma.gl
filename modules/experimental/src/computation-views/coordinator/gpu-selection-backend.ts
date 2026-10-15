// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeySpaceId, SelectionClause} from '../types';
import type {SelectionBackend} from './selection-backend';

/** GPU-resident selection mask (one u32 bit or byte per key), never read back per frame. */
export type GPUSelectionMaskHandle = {
  keySpace: KeySpaceId;
  keyCount: number;
  buffer: unknown;
  generation: number;
};

/** Wave-2 contract for GPUCrossfilter external-mask dimensions and relation transfer kernels. */
export interface GPUSelectionBackend extends SelectionBackend {
  readonly kind: 'gpu';
  /** Sets an external-mask or range dimension of the underlying GPUCrossfilter. */
  setClause(clause: SelectionClause): void;
  /** Clears the dimension owned by this source. */
  clearSource(source: string): void;
  /** AND of dimensions except the supplied source, resident on the GPU. */
  getMask(keySpace: KeySpaceId, options?: {excludeSource?: string}): GPUSelectionMaskHandle;
  /** Gathers a mask across a buffer relation without readback. */
  transferMask(mask: GPUSelectionMaskHandle, relationId: string): GPUSelectionMaskHandle;
  /** One packed counts readback per frame, for off-view cues. */
  readCounts(): Promise<Record<KeySpaceId, number>>;
}

/** Narrows a backend to the GPU contract. */
export function isGPUSelectionBackend(backend: SelectionBackend): backend is GPUSelectionBackend {
  return backend.kind === 'gpu';
}
