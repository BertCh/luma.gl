// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {rectangle} from './spatial-predicate-oracle';

// rectangle takes (x, y, width, height), not opposite corners. Source order deliberately
// differs from spatial order to exercise heap replacement in a truncated candidate row.
export const PREPARED_SELF_JOIN_FEATURES = [
  rectangle(4, 0, 10, 4), // x = [4, 14]
  rectangle(8, 0, 12, 4), // x = [8, 20]
  rectangle(0, 0, 4, 4), // x = [0, 4]
  rectangle(2, 0, 6, 4), // x = [2, 8]
  rectangle(30, 0, 32, 2), // x = [30, 62], isolated
  rectangle(6, 0, 8, 4) // x = [6, 14], overlaps rows 1 and 3
];

// Interior intersection is supported by the bounding-box candidate broad phase.
export const PREPARED_SELF_JOIN_PATTERN = 'T********';

export const PREPARED_SELF_JOIN_OVERLAPS = [
  [0, 1],
  [0, 3],
  [1, 0],
  [1, 5],
  [2, 3],
  [3, 0],
  [3, 2],
  [3, 5],
  [5, 1],
  [5, 3]
];
