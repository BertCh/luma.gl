// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OperatorDescriptor} from '../../types';

/** GPU Core element scopes and reductions, keyed by public exported identifier. */
export const GPU_CORE_OPERATORS: Record<string, OperatorDescriptor> = {
  GPUScan: {scope: 'global', reduction: 'scan-sort'},
  GPUFlagOffsets: {scope: 'global', reduction: 'scan-sort'},
  GPUCompaction: {scope: 'global', reduction: 'select'},
  GPUMask: {scope: 'local', reduction: 'none'},
  GPUReduction: {scope: 'global', reduction: 'sum'},
  GPUSegmentedReduction: {scope: 'zonal', reduction: 'sum'},
  GPUSort: {scope: 'global', reduction: 'scan-sort'},
  GPUSegmentedSort: {scope: 'zonal', reduction: 'scan-sort'},
  GPUGather: {scope: 'local', reduction: 'select'},
  GPUHistogram: {scope: 'global', reduction: 'histogram'},
  GPUElementwise: {scope: 'local', reduction: 'none'},
  GPUGridIndex: {scope: 'zonal', reduction: 'scan-sort'},
  GPUGridBinning: {scope: 'zonal', reduction: 'count'},
  GPUGridAggregation: {scope: 'zonal', reduction: 'sum'},
  GPUGroupAggregation: {scope: 'zonal', reduction: 'sum'},
  GPUHashIndex: {scope: 'global', reduction: 'none'},
  GPUHashJoin: {scope: 'pairwise', reduction: 'select'},
  GPUMatVec: {scope: 'zonal', reduction: 'sum'},
  GPUMatMul: {scope: 'block', reduction: 'sum'},
  GPUConvolution: {scope: 'focal', reduction: 'sum'},
  GPUFFT1D: {scope: 'global', reduction: 'sum'},
  GPUFFT2D: {scope: 'global', reduction: 'sum'},
  createGPUConjugateGradientProgram: {scope: 'solver', reduction: 'solve'}
};
