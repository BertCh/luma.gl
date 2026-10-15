// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerDefaultViewPack} from '../../computation-views/registry/default-registry';
import {GPU_NETWORK_PACK_DATA} from './network-pack-data';
import {NETWORK_WAVEFRONT_VIEW} from './views/network-wavefront-view';

registerDefaultViewPack({...GPU_NETWORK_PACK_DATA, views: [NETWORK_WAVEFRONT_VIEW]});

export {GPU_NETWORK_PACK_DATA} from './network-pack-data';
export {NETWORK_OPERATORS} from './network-operators';
export {NETWORK_FORMULAS, NETWORK_FORMULA_REFS} from './network-formulas';
export {NETWORK_WAVEFRONT_VIEW} from './views/network-wavefront-view';
