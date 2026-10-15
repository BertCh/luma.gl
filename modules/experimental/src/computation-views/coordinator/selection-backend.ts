// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeySpaceId, SelectionClause} from '../types';

/** Backend ownership policy for a declared key space. */
export interface SelectionBackend {
  readonly kind: 'cpu' | 'gpu';
  accepts(keySpace: KeySpaceId, keyCount: number): boolean;
}

/** CPU evaluation for small key spaces, with inclusive numeric ranges. */
export class CPUSelectionBackend implements SelectionBackend {
  readonly kind = 'cpu';
  private maxKeyCount: number;

  constructor(props: {maxKeyCount?: number} = {}) {
    this.maxKeyCount = props.maxKeyCount ?? 65536;
  }

  /** Accepts finite, nonnegative key counts within the configured budget. */
  accepts(keySpace: KeySpaceId, keyCount: number): boolean {
    return Number.isInteger(keyCount) && keyCount >= 0 && keyCount <= this.maxKeyCount;
  }

  /** Returns unknown for missing fields or key spaces that exceed the CPU budget. */
  evaluate(
    clause: SelectionClause,
    keyCount: number,
    fieldValues?: (field: string) => ArrayLike<number> | undefined
  ): Set<number> | 'unknown' {
    if (!this.accepts(clause.keySpace, keyCount)) return 'unknown';
    const {predicate} = clause;
    if (predicate.kind === 'keys') {
      return new Set(
        predicate.keys.filter(key => Number.isInteger(key) && key >= 0 && key < keyCount)
      );
    }
    const values = predicate.kind === 'range' ? fieldValues?.(predicate.field) : undefined;
    if (predicate.kind === 'range' && (!values || values.length < keyCount)) return 'unknown';
    const keys = new Set<number>();
    for (let key = 0; key < keyCount; key++) {
      if (
        predicate.kind === 'all' ||
        (values && values[key] >= predicate.range[0] && values[key] <= predicate.range[1])
      )
        keys.add(key);
    }
    return keys;
  }
}
