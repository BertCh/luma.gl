// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CELLS_AND_JOINS_CASES} from './kernel-audit/cases-cells-and-joins';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('cells-and-joins', CELLS_AND_JOINS_CASES);
