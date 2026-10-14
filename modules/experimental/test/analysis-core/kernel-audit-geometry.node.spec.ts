// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GEOMETRY_CASES} from './kernel-audit/cases-geometry';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('geometry', GEOMETRY_CASES);
