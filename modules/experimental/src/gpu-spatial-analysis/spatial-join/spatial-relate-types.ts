// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Cell order of a packed DE-9IM matrix, row-major: interior, boundary and exterior of the left
 * geometry against interior, boundary and exterior of the right geometry.
 */
export const GPU_SPATIAL_RELATE_CELLS = [
  'II',
  'IB',
  'IE',
  'BI',
  'BB',
  'BE',
  'EI',
  'EB',
  'EE'
] as const;

/** Bit that marks a relate value whose orientation tests could not all be certified. @internal */
export const GPU_SPATIAL_RELATE_UNCERTAIN_BIT = 0x80000000;

/**
 * Pattern (or any-of list of patterns) over the nine DE-9IM cells, in the OGC text form: nine
 * characters from `T` (any non-empty intersection), `F` (empty), `*` (don't care) and `0`, `1`,
 * `2` (exact dimension). A list matches when any pattern matches.
 */
export type GPUSpatialRelatePattern = string | readonly string[];

const DIMENSION_TEXT = ['F', '0', '1', '2'] as const;

/**
 * Packs a nine-character DE-9IM string of `F`, `0`, `1`, `2` into the GPU layout: two bits per
 * cell, cell `k` of {@link GPU_SPATIAL_RELATE_CELLS} at bits `[2k, 2k + 1]`, holding the
 * intersection dimension plus one (`0` empty, `1` points, `2` lines, `3` areas). The layout fits 18
 * bits; the output of `relate` never sets the upper bits.
 */
export function packGPUSpatialRelate(matrix: string): number {
  if (!/^[F012]{9}$/.test(matrix)) {
    throw new Error(`DE-9IM matrix must be nine characters of F, 0, 1 or 2, got "${matrix}"`);
  }
  let packed = 0;
  for (let cell = 0; cell < 9; cell++) {
    packed |= (DIMENSION_TEXT as readonly string[]).indexOf(matrix[cell]) << (2 * cell);
  }
  return packed;
}

/** Formats a packed relate value as the nine-character OGC DE-9IM string, such as `FF2F11212`. */
export function formatGPUSpatialRelate(packed: number): string {
  let text = '';
  for (let cell = 0; cell < 9; cell++) {
    text += DIMENSION_TEXT[(packed >>> (2 * cell)) & 3];
  }
  return text;
}

/** Returns the pattern list of a user pattern. @internal */
export function getRelatePatterns(pattern: GPUSpatialRelatePattern): readonly string[] {
  return typeof pattern === 'string' ? [pattern] : pattern;
}

/**
 * Returns, per cell, the set of allowed encoded values as a 4-bit mask (bit `v` allows encoded
 * value `v`, that is dimension `v - 1`). `*` allows all, `T` allows 1 to 3, `F` allows 0.
 * @internal
 */
export function parseRelatePattern(pattern: string): number[] {
  if (!/^[TF012*]{9}$/.test(pattern)) {
    throw new Error(
      `DE-9IM pattern must be nine characters of T, F, 0, 1, 2 or *, got "${pattern}"`
    );
  }
  const masks: Record<string, number> = {'*': 0b1111, T: 0b1110, F: 0b0001, '0': 0b0010};
  masks['1'] = 0b0100;
  masks['2'] = 0b1000;
  return [...pattern].map(character => masks[character]);
}

/**
 * Returns whether any pattern can match a pair of disjoint geometries. Such a pattern would need
 * pairs that are not bounding-box candidates, so the join rejects it (use `how: 'anti'`).
 * @internal
 */
export function doesRelatePatternAdmitDisjoint(patterns: readonly string[]): boolean {
  return patterns.some(pattern => {
    const masks = parseRelatePattern(pattern);
    return [0, 1, 3, 4].every(cell => (masks[cell] & 1) !== 0) && (masks[8] & 0b1000) !== 0;
  });
}

/** Topological dimension of a geometry kind. @internal */
export function getSpatialKindDimension(kind: 'points' | 'lines' | 'polygons'): 0 | 1 | 2 {
  return kind === 'points' ? 0 : kind === 'lines' ? 1 : 2;
}

/**
 * OGC / JTS DE-9IM pattern definitions of the named predicates, as lists of alternative patterns.
 * Returns an empty list for a pair of dimensions where the predicate is never true (for example
 * `crosses` between two polygons, as in Shapely).
 *
 * @internal
 */
export function getPredicateRelatePatterns(
  predicate:
    | 'intersects'
    | 'contains'
    | 'within'
    | 'covers'
    | 'coveredBy'
    | 'touches'
    | 'crosses'
    | 'overlaps'
    | 'equals'
    | 'containsProperly',
  leftDimension: number,
  rightDimension: number
): readonly string[] {
  switch (predicate) {
    case 'intersects':
      return ['T********', '*T*******', '***T*****', '****T****'];
    case 'contains':
      return ['T*****FF*'];
    case 'within':
      return ['T*F**F***'];
    case 'covers':
      return ['T*****FF*', '*T****FF*', '***T**FF*', '****T*FF*'];
    case 'coveredBy':
      return ['T*F**F***', '*TF**F***', '**FT*F***', '**F*TF***'];
    case 'touches':
      return leftDimension === 0 && rightDimension === 0
        ? []
        : ['FT*******', 'F**T*****', 'F***T****'];
    case 'crosses':
      if (leftDimension < rightDimension) {
        return ['T*T******'];
      }
      if (leftDimension > rightDimension) {
        return ['T*****T**'];
      }
      return leftDimension === 1 ? ['0********'] : [];
    case 'overlaps':
      if (leftDimension !== rightDimension) {
        return [];
      }
      return leftDimension === 1 ? ['1*T***T**'] : ['T*T***T**'];
    case 'equals':
      return ['T*F**FFF*'];
    case 'containsProperly':
      return ['T**FF*FF*'];
  }
}

/**
 * Returns WGSL `fn relateMatches(m: u32) -> bool` that is true when the packed matrix `m` matches
 * any pattern. An empty list never matches. @internal
 */
export function getRelateMatchesWGSL(patterns: readonly string[]): string {
  const clauses = patterns.map(pattern => {
    const masks = parseRelatePattern(pattern);
    const terms = masks
      .map((mask, cell) =>
        mask === 0b1111 ? undefined : `((${mask}u >> ((m >> ${2 * cell}u) & 3u)) & 1u) != 0u`
      )
      .filter(Boolean);
    return terms.length ? `(${terms.join(' && ')})` : 'true';
  });
  return `fn relateMatches(m: u32) -> bool { return ${clauses.length ? clauses.join(' || ') : 'false'}; }`;
}
