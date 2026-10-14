// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {IS_FINITE_FLOAT_WGSL} from '../../utils/wgsl-common';

/** WGSL finite-float test shared by the neighbor-search kernels. @internal */
export const NEIGHBOR_SEARCH_FLOAT_WGSL = /* wgsl */ IS_FINITE_FLOAT_WGSL;
