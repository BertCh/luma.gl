// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NETWORK_CASES} from './kernel-audit/cases-network';
import {registerKernelAudit} from './kernel-audit/register-kernel-audit';

registerKernelAudit('network', NETWORK_CASES);
