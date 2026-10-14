// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the statistics cases. */
export const STATISTICS_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUGlobalPermutationTest',
    name: 'moran permutations x row blocks above the maximum',
    kind: 'dispatch',
    limit: 16777215,
    rejects: /^Error: global-moran permutations x row blocks must be at most 16777215$/
  },
  {
    exportName: 'GPUGlobalPermutationTest',
    name: 'geary permutations x row blocks above the maximum',
    kind: 'dispatch',
    limit: 16777215,
    rejects: /^Error: global-geary permutations x row blocks must be at most 16777215$/
  },
  {
    exportName: 'GPUGlobalPermutationTest',
    name: 'getisOrdG permutations x row blocks above the maximum',
    kind: 'dispatch',
    limit: 16777215,
    rejects: /^Error: global-getisOrdG permutations x row blocks must be at most 16777215$/
  },
  {
    exportName: 'GPUGlobalPermutationTest',
    name: 'bivariateMoran permutations x row blocks above the maximum',
    kind: 'dispatch',
    limit: 16777215,
    rejects: /^Error: global-bivariateMoran permutations x row blocks must be at most 16777215$/
  },
  {
    exportName: 'GPUGlobalPermutationTest',
    name: 'leesL permutations x row blocks above the maximum',
    kind: 'dispatch',
    limit: 16777215,
    rejects: /^Error: global-leesL permutations x row blocks must be at most 16777215$/
  },
  {
    exportName: 'GPUSpatialRegressionDiagnostics',
    name: 'restandardize false rows above the maximum',
    kind: 'binding',
    limit: 33554432,
    rejects:
      /^Error: spatial-regression-diagnostics rows x \(predictorCount \+ 3\) must be at most 33554432$/
  },
  {
    exportName: 'GPUSpatialRegressionDiagnostics',
    name: 'restandardize true rows above the maximum',
    kind: 'binding',
    limit: 33554432,
    rejects:
      /^Error: spatial-regression-diagnostics rows x \(predictorCount \+ 3\) must be at most 33554432$/
  }
];
