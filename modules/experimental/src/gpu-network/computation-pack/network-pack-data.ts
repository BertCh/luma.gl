// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ViewPackData} from '../../computation-views/registry/view-registration';
import {NETWORK_FORMULAS} from './network-formulas';
import {NETWORK_OPERATORS} from './network-operators';
import {NETWORK_CONSTRUCTION_PACK} from './network-pack-construction';
import {NETWORK_SEARCH_PACK} from './network-pack-search';
import {NETWORK_STRUCTURE_PACK} from './network-pack-structure';

/** Pure network vocabulary; importing this data never constructs a GPU contributor. */
export const GPU_NETWORK_PACK_DATA: ViewPackData = {
  id: 'gpu-network',
  operators: NETWORK_OPERATORS,
  formulas: NETWORK_FORMULAS,
  analysisStages: [
    'representation',
    'impedance',
    'search',
    'assignment',
    'summary',
    'presentation'
  ],
  captureStages: {
    ...NETWORK_SEARCH_PACK.captureStages,
    ...NETWORK_STRUCTURE_PACK.captureStages,
    ...NETWORK_CONSTRUCTION_PACK.captureStages,
    costs: 'search',
    bands: 'assignment',
    predecessors: 'search',
    bandCounts: 'summary',
    lastImprovedRound: 'search',
    lastQueuedRound: 'search'
  },
  variableKinds: {
    ...NETWORK_SEARCH_PACK.variableKinds,
    ...NETWORK_STRUCTURE_PACK.variableKinds,
    ...NETWORK_CONSTRUCTION_PACK.variableKinds,
    costs: {level: 'ratio', extent: 'intensive'},
    bands: {level: 'ordinal'},
    facility: {level: 'nominal'},
    assignment: {level: 'nominal'},
    component: {level: 'nominal'},
    community: {level: 'nominal'},
    lastImprovedRound: {level: 'ordinal'},
    lastQueuedRound: {level: 'ordinal'},
    bandCounts: {level: 'ratio', extent: 'extensive'},
    'frontier-size': {level: 'ratio', extent: 'extensive'}
  },
  neighbourhoods: {node: {kind: 'csr-star', relation: 'node-out-edges'}},
  metadata: {coordinateModel: 'geographic', executionOrder: 'chaotic'}
};
