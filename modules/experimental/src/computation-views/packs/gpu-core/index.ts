// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerDefaultViewPack} from '../../registry/default-registry';
import type {ViewPackRegistration} from '../../registry/view-registration';
import {GPU_CORE_ARRAY_RESULT_VIEW} from './array-result-view';
import {loadGPUCorePackData} from './pack-data';

export {GPU_CORE_OPERATORS} from './operators';
export {
  GPU_CORE_FORMULAS,
  GPU_CORE_FORMULA_REFS,
  mergeGPUCoreFormulaRefs,
  mergeGPUCoreFormulaRefsForGraphs
} from './formulas';
export {loadGPUCorePackData} from './pack-data';
export {
  GPU_CORE_ARRAY_RESULT_VIEW,
  buildGPUCoreArrayResultModel,
  renderGPUCoreArrayResult,
  describeGPUCoreArrayResult,
  renderArrayTable
} from './array-result-view';

/** GPU Core data and result view; import this entry to register it in new default registries. */
export const GPU_CORE_VIEW_PACK: ViewPackRegistration = {
  ...loadGPUCorePackData(),
  views: [GPU_CORE_ARRAY_RESULT_VIEW]
};
registerDefaultViewPack(GPU_CORE_VIEW_PACK);
