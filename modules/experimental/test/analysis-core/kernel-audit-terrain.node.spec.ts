// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {TERRAIN_CASES} from './kernel-audit/cases-terrain';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('terrain', TERRAIN_CASES);
