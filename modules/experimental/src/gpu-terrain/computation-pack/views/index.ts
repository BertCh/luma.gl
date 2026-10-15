// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ViewRegistration} from '../../../computation-views/registry/view-registration';
import {D8_STENCIL_VIEW} from './d8-stencil-view';

export {D8_STENCIL_VIEW} from './d8-stencil-view';

/** Terrain domain views (order 0–49, so they win over the generic views of their rung). */
export const TERRAIN_VIEWS: ViewRegistration<unknown>[] = [
  D8_STENCIL_VIEW as ViewRegistration<unknown>
];
