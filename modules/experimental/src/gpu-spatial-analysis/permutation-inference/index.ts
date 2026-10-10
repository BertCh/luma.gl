// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPULocalPermutationTest,
  GPU_LOCAL_PERMUTATION_MAXIMUM_NEIGHBORS,
  GPU_LOCAL_PERMUTATION_NOT_TESTED
} from './gpu-local-permutation-test';
export type {
  GPULocalPermutationStatistic,
  GPULocalPermutationTestProps
} from './gpu-local-permutation-test';
export {
  GPUGlobalPermutationTest,
  GPU_GLOBAL_PERMUTATION_RESULT
} from './gpu-global-permutation-test';
export type {
  GPUGlobalPermutationStatistic,
  GPUGlobalPermutationTestProps
} from './gpu-global-permutation-test';
export {
  getGPUPermutationMetadata,
  getGPUPermutationParameterValues,
  GPU_PERMUTATION_PARAMETER_LENGTH
} from './permutation-parameters';
export type {GPUPermutationParameters} from './permutation-parameters';
export type {GPUPermutationAlternative} from './permutation-alternative';
