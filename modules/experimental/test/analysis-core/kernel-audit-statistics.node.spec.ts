// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {STATISTICS_CASES} from './kernel-audit/cases-statistics';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('statistics', STATISTICS_CASES);
