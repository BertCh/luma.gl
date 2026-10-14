// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {RASTER_OPERATIONS_CASES} from './kernel-audit/cases-raster-operations';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('raster-operations', RASTER_OPERATIONS_CASES);
