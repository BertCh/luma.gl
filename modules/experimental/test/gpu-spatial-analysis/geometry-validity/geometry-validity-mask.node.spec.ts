// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPU_GEOMETRY_VALIDITY_BIT,
  GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK
} from '../../../src/gpu-spatial-analysis/geometry-validity';

it('review D-20: repeatedVertex is report-only, like the orientation convention', () => {
  const {repeatedVertex, badOrientation, ...structuralBits} = GPU_GEOMETRY_VALIDITY_BIT;
  // GEOS and OGC accept repeated vertices.
  expect(GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK & repeatedVertex).toBe(0);
  expect(GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK & badOrientation).toBe(0);
  for (const [name, bit] of Object.entries(structuralBits)) {
    expect(GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK & bit, name).toBe(bit);
  }
});
