// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditCase} from './kernel-audit-context';
import {GEOSPATIAL_CASES} from './cases-geospatial';
import {DATAFRAME_CASES} from './cases-dataframe';
import {RASTER_CASES} from './cases-raster';
import {RASTER_OPERATIONS_CASES} from './cases-raster-operations';
import {TERRAIN_CASES} from './cases-terrain';
import {NETWORK_CASES} from './cases-network';
import {STATISTICS_CASES} from './cases-statistics';
import {RECIPES_CASES} from './cases-recipes';
import {GEOMETRY_CASES} from './cases-geometry';
import {CELLS_AND_JOINS_CASES} from './cases-cells-and-joins';
import {POINTS_AND_REGIONS_CASES} from './cases-points-and-regions';

/** Every kernel-audit case, by area file. */
export const KERNEL_AUDIT_CASES: Readonly<Record<string, readonly KernelAuditCase[]>> = {
  geospatial: GEOSPATIAL_CASES,
  dataframe: DATAFRAME_CASES,
  raster: RASTER_CASES,
  'raster-operations': RASTER_OPERATIONS_CASES,
  terrain: TERRAIN_CASES,
  network: NETWORK_CASES,
  statistics: STATISTICS_CASES,
  recipes: RECIPES_CASES,
  geometry: GEOMETRY_CASES,
  'cells-and-joins': CELLS_AND_JOINS_CASES,
  'points-and-regions': POINTS_AND_REGIONS_CASES
};
