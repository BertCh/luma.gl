// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {DATAFRAME_CASES} from './kernel-audit/cases-dataframe';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('dataframe', DATAFRAME_CASES);
