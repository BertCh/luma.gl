// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** A case above a contributor's documented maximum that must be refused while it is declared. */
export type KernelAuditRejection = {
  /** Barrel export of the case. */
  exportName: string;
  /** Case name. */
  name: string;
  /**
   * `'binding'`: a prop whose largest binding would exceed the core `maxStorageBufferBindingSize`
   * (128 MiB). `'dispatch'`: `capacity x lanes` or a 1-D dispatch above its limit.
   */
  kind: 'binding' | 'dispatch';
  /**
   * The documented maximum the refusal names. The error message must contain it, and the
   * contributor's companion case at exactly this value must pass.
   */
  limit?: number;
  /**
   * Error the declaration must throw: the contributor's own `${id}`-prefixed prop check, never a
   * generic graph or kernel guard (see `GENERIC_REFUSAL_PATTERNS`).
   */
  rejects: RegExp;
};
