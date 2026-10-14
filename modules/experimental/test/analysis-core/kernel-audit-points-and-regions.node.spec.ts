// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {POINTS_AND_REGIONS_CASES} from './kernel-audit/cases-points-and-regions';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('points-and-regions', POINTS_AND_REGIONS_CASES);
