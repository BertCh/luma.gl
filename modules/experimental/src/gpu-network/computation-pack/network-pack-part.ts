// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';
import type {ViewPackData} from '../../computation-views/registry/view-registration';

/** One element family's share of the network pack; `network-pack-data.ts` spreads every part in. */
export type NetworkPackPart = {
  /** Curated formulas keyed by the formula id recorded on kernel nodes. */
  formulaRefs: Readonly<Record<string, GPUComputationFormulaRef>>;
  /** Primary formula (or `{formula: 'none', reason}`) per exported element name. */
  formulas: NonNullable<ViewPackData['formulas']>;
  /** Analysis stage per capture name. */
  captureStages: Readonly<Record<string, string>>;
  /** Variable kind per capture name. */
  variableKinds: NonNullable<ViewPackData['variableKinds']>;
};
