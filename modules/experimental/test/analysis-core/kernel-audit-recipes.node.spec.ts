// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {RECIPES_CASES} from './kernel-audit/cases-recipes';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('recipes', RECIPES_CASES);
