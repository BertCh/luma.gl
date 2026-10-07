// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Draw-order helpers for the spatial-analysis layers (SYNTHESIS G20). Later instances draw on
 * top, so the order decides who wins where marks overlap. Upload the result as the layer's `ids`
 * (`resources.createBuffer(id, order)`): instance `i` then draws row `order[i]`.
 */

/**
 * Row indices ordered by value, so high values (`'ascending'`: low drawn first, high on top) or low
 * values (`'descending'`) end up on top. NaN and infinite values always come last, that is, on
 * top of everything; filter them out of the layer (alpha 0 or `discardAtOrBelow`) if that is not
 * wanted. Ties keep their row order, so the result is deterministic.
 */
export function getSortedOrder(
  values: ArrayLike<number>,
  direction: 'ascending' | 'descending'
): Uint32Array {
  const count = values.length;
  const order = new Uint32Array(count);
  for (let index = 0; index < count; index++) order[index] = index;
  const sign = direction === 'ascending' ? 1 : -1;
  order.sort((first, second) => {
    const a = values[first];
    const b = values[second];
    const aFinite = Number.isFinite(a);
    const bFinite = Number.isFinite(b);
    if (!aFinite || !bFinite) {
      if (aFinite === bFinite) return first - second;
      return aFinite ? -1 : 1;
    }
    return a === b ? first - second : sign * (a - b);
  });
  return order;
}

/**
 * A deterministic shuffle of `0..count-1` (seeded Fisher-Yates over a 32-bit generator). Drawing
 * dense points in file order lets the last category in the file win every overlap ("last drawn
 * wins" bias); a shuffled order spreads that bias evenly. The same `count` and `seed` always give
 * the same order.
 */
export function getShuffledOrder(count: number, seed = 1): Uint32Array {
  const size = Math.max(0, Math.floor(count));
  const order = new Uint32Array(size);
  for (let index = 0; index < size; index++) order[index] = index;
  let state = (Math.floor(seed) >>> 0) + 0x6d2b79f5;
  const next = () => {
    // mulberry32
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
  for (let index = size - 1; index > 0; index--) {
    const other = Math.floor(next() * (index + 1));
    const held = order[index];
    order[index] = order[other];
    order[other] = held;
  }
  return order;
}
