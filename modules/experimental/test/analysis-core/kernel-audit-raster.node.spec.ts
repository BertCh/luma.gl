// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {RASTER_CASES} from './kernel-audit/cases-raster';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('raster', RASTER_CASES);
