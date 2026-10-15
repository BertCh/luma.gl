// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerDefaultViewPack} from '../../computation-views/registry/default-registry';
import type {ViewPackRegistration} from '../../computation-views/registry/view-registration';
import {
  createRasterViewPack,
  mergeRasterOperatorTables
} from '../../gpu-raster/computation-pack/create-raster-pack';
import type {RasterOperatorEntry} from '../../gpu-raster/computation-pack/types';
import {TERRAIN_ANALYSIS_OPERATORS} from './operators/analysis';
import {TERRAIN_DECODE_OPERATORS} from './operators/decode';
import {TERRAIN_FEATURES_OPERATORS} from './operators/features';
import {TERRAIN_FLOW_FIELD_OPERATORS} from './operators/flow-field';
import {TERRAIN_GEOMORPHOMETRY_OPERATORS} from './operators/geomorphometry';
import {TERRAIN_HYDROLOGY_OPERATORS} from './operators/hydrology';
import {TERRAIN_ILLUMINATION_OPERATORS} from './operators/illumination';
import {TERRAIN_POINT_HORIZON_OPERATORS} from './operators/point-horizon';
import {TERRAIN_RELIEF_OPERATORS} from './operators/relief';
import {TERRAIN_VIEWS} from './views/index';

/** Pack id; also the lens scenario entry. */
export const TERRAIN_COMPUTATION_PACK_ID = 'gpu-terrain';

/** Every family table, in docs order. Each family file is filled by one implementer batch. */
export const TERRAIN_FAMILY_OPERATORS = [
  TERRAIN_ANALYSIS_OPERATORS,
  TERRAIN_POINT_HORIZON_OPERATORS,
  TERRAIN_DECODE_OPERATORS,
  TERRAIN_FEATURES_OPERATORS,
  TERRAIN_GEOMORPHOMETRY_OPERATORS,
  TERRAIN_ILLUMINATION_OPERATORS,
  TERRAIN_RELIEF_OPERATORS,
  TERRAIN_HYDROLOGY_OPERATORS,
  TERRAIN_FLOW_FIELD_OPERATORS
] as const;

/** All gpu-terrain contributor entries keyed by exported class name. */
export function getTerrainOperatorEntries(): Readonly<Record<string, RasterOperatorEntry>> {
  return mergeRasterOperatorTables(TERRAIN_COMPUTATION_PACK_ID, TERRAIN_FAMILY_OPERATORS);
}

/** The gpu-terrain view pack (plain data plus the D8 stencil view). */
export function getTerrainViewPack(): ViewPackRegistration {
  return createRasterViewPack(
    TERRAIN_COMPUTATION_PACK_ID,
    getTerrainOperatorEntries(),
    TERRAIN_VIEWS
  );
}

registerDefaultViewPack(getTerrainViewPack());
