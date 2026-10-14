// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GEOSPATIAL_CASES} from './kernel-audit/cases-geospatial';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('geospatial', GEOSPATIAL_CASES);
