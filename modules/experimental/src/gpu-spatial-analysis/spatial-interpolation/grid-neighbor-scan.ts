// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL generators for the two ways a raster cell walks a `GPUGridIndex`: a row-major sweep of the
 * search square, and an expanding-ring sweep that stops as soon as the `k` nearest are final.
 *
 * Both expect these names in scope (declared by `GPUInverseDistanceWeighting` and `GPUKriging`):
 * `center`, `columnLow`, `columnHigh`, `rowLow`, `rowHigh` (inclusive, already widened by one cell),
 * `cellOffsets`, `sortedIds`, `DOMAIN_MIN`, `DOMAIN_MAX`, `INDEX_WIDTH`, `INDEX_HEIGHT` and
 * `getCoordinate`. The ring sweep also reads `nearCount`, `neighborLimit` and `nearDistances`.
 * `visitSample` runs with `sample` bound and may `continue` to skip it.
 *
 * @internal
 */

/** Visits the index rows overlapping the search square in index row, index column, slot order. */
export function getRectangleScanWGSL(visitSample: string): string {
  return /* wgsl */ `
    for (var indexRow = rowLow; indexRow <= rowHigh; indexRow++) {
      // Cells of one index row are contiguous in cellOffsets, so the row span is one slot range.
      let start = cellOffsets[cellOffsetsOffset + indexRow * INDEX_WIDTH + columnLow];
      let end = cellOffsets[cellOffsetsOffset + indexRow * INDEX_WIDTH + columnHigh + 1u];
      for (var slot = start; slot < end; slot++) {
        let sample = sortedIds[sortedIdsOffset + slot];
        ${visitSample}
      }
    }`;
}

/**
 * Visits index cells in rings of growing Chebyshev radius around the cell holding `center`.
 *
 * After ring `r` every sample inside the visited block is seen, so any unvisited sample is farther
 * than the distance from `center` to the block border (less one cell of slack for the f32 rounding
 * of cell coordinates, the same slack the rectangle sweep absorbs by widening). Once `k` samples
 * are held and the `k`-th distance is below that bound, no unvisited sample can enter the list
 * under the `(distance, row)` order, so the sweep stops. A cell then costs `O(k)` candidates
 * instead of every sample inside the radius, which matters when the radius is large relative to
 * the sample spacing (an unbounded radius is `O(samples)` per cell on the rectangle sweep).
 *
 * The result is independent of visiting order because the near list is keyed by the total order
 * `(d^2, row)`; exact hits are always inside the stopping bound, so they are always visited.
 */
export function getRingScanWGSL(visitSample: string): string {
  return /* wgsl */ `
    let cellExtent = (DOMAIN_MAX - DOMAIN_MIN) / vec2f(f32(INDEX_WIDTH), f32(INDEX_HEIGHT));
    let cellSlack = min(cellExtent.x, cellExtent.y);
    let centerPoint = clamp(center, DOMAIN_MIN, DOMAIN_MAX);
    let lowColumn = i32(columnLow);
    let highColumn = i32(columnHigh);
    let lowRow = i32(rowLow);
    let highRow = i32(rowHigh);
    let centerColumn = clamp(i32(getCoordinate(centerPoint.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH)), lowColumn, highColumn);
    let centerRow = clamp(i32(getCoordinate(centerPoint.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT)), lowRow, highRow);
    let lastRing = max(max(centerColumn - lowColumn, highColumn - centerColumn), max(centerRow - lowRow, highRow - centerRow));
    for (var ring = 0; ring <= lastRing; ring++) {
      let ringColumnLow = max(centerColumn - ring, lowColumn);
      let ringColumnHigh = min(centerColumn + ring, highColumn);
      for (var cellRow = max(centerRow - ring, lowRow); cellRow <= min(centerRow + ring, highRow); cellRow++) {
        let isEdgeRow = cellRow == centerRow - ring || cellRow == centerRow + ring;
        for (var side = 0; side < 2; side++) {
          var firstColumn = ringColumnLow;
          var lastColumn = ringColumnHigh;
          if (isEdgeRow) {
            // Top and bottom rows of the ring are one contiguous slot range.
            if (side == 1) { continue; }
          } else {
            // Interior rows of the ring touch only its left and right cells.
            let column = select(centerColumn + ring, centerColumn - ring, side == 0);
            if (column < lowColumn || column > highColumn) { continue; }
            firstColumn = column;
            lastColumn = column;
          }
          let rowBase = u32(cellRow) * INDEX_WIDTH;
          let start = cellOffsets[cellOffsetsOffset + rowBase + u32(firstColumn)];
          let end = cellOffsets[cellOffsetsOffset + rowBase + u32(lastColumn) + 1u];
          for (var slot = start; slot < end; slot++) {
            let sample = sortedIds[sortedIdsOffset + slot];
            ${visitSample}
          }
        }
      }
      if (nearCount >= neighborLimit) {
        // Distance from the center to the nearest border of the visited block that still has
        // unvisited cells behind it.
        var gap = 3.0e38;
        if (centerColumn - ring > lowColumn) {
          gap = min(gap, center.x - (DOMAIN_MIN.x + f32(centerColumn - ring) * cellExtent.x));
        }
        if (centerColumn + ring < highColumn) {
          gap = min(gap, DOMAIN_MIN.x + f32(centerColumn + ring + 1) * cellExtent.x - center.x);
        }
        if (centerRow - ring > lowRow) {
          gap = min(gap, center.y - (DOMAIN_MIN.y + f32(centerRow - ring) * cellExtent.y));
        }
        if (centerRow + ring < highRow) {
          gap = min(gap, DOMAIN_MIN.y + f32(centerRow + ring + 1) * cellExtent.y - center.y);
        }
        gap = gap - cellSlack;
        if (gap > 0.0 && nearDistances[neighborLimit - 1u] < gap * gap) {
          break;
        }
      }
    }`;
}
