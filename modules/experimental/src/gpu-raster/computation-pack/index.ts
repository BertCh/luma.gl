// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerDefaultViewPack} from '../../computation-views/registry/default-registry';
import type {ViewPackRegistration} from '../../computation-views/registry/view-registration';
import {createRasterViewPack, mergeRasterOperatorTables} from './create-raster-pack';
import {RASTER_ALGEBRA_OPERATORS} from './operators/algebra';
import {RASTER_COST_DISTANCE_OPERATORS} from './operators/cost-distance';
import {RASTER_ISOLINES_OPERATORS} from './operators/isolines';
import {RASTER_REGIONS_OPERATORS} from './operators/regions';
import {RASTER_SAMPLING_OPERATORS} from './operators/sampling';
import {RASTER_VECTOR_FIELD_OPERATORS} from './operators/vector-field';
import {RASTER_ZONES_OPERATORS} from './operators/zones';
import type {RasterOperatorEntry} from './types';

export * from './create-raster-pack';
export * from './guards';
export * from './types';

/** Pack id; also the lens scenario entry. */
export const RASTER_COMPUTATION_PACK_ID = 'gpu-raster';

/** Every family table, in docs order. Each family file is filled by one implementer batch. */
export const RASTER_FAMILY_OPERATORS = [
  RASTER_COST_DISTANCE_OPERATORS,
  RASTER_ZONES_OPERATORS,
  RASTER_ALGEBRA_OPERATORS,
  RASTER_ISOLINES_OPERATORS,
  RASTER_SAMPLING_OPERATORS,
  RASTER_VECTOR_FIELD_OPERATORS,
  RASTER_REGIONS_OPERATORS
] as const;

/** All gpu-raster analysis contributor entries keyed by exported class name. */
export function getRasterOperatorEntries(): Readonly<Record<string, RasterOperatorEntry>> {
  return mergeRasterOperatorTables(RASTER_COMPUTATION_PACK_ID, RASTER_FAMILY_OPERATORS);
}

/** The gpu-raster view pack (plain data plus domain views). */
export function getRasterViewPack(): ViewPackRegistration {
  return createRasterViewPack(RASTER_COMPUTATION_PACK_ID, getRasterOperatorEntries());
}

registerDefaultViewPack(getRasterViewPack());
