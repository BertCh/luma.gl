// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';
import {CELLS_AND_JOINS_REJECTIONS} from './rejections-cells-and-joins';
import {DATAFRAME_REJECTIONS} from './rejections-dataframe';
import {GEOMETRY_REJECTIONS} from './rejections-geometry';
import {GEOMETRY_LINES_REJECTIONS} from './rejections-geometry-lines';
import {NETWORK_REJECTIONS} from './rejections-network';
import {POINTS_AND_REGIONS_REJECTIONS} from './rejections-points-and-regions';
import {RASTER_REJECTIONS} from './rejections-raster';
import {RASTER_OPERATIONS_REJECTIONS} from './rejections-raster-operations';
import {STATISTICS_REJECTIONS} from './rejections-statistics';
import {TERRAIN_REJECTIONS} from './rejections-terrain';

export type {KernelAuditRejection} from './kernel-audit-rejection-type';

/**
 * Cases one step above a contributor's documented maximum: rule 16 asks that they fail at
 * declaration with a short `${id}`-prefixed message naming the prop and the limit, before anything
 * compiles. Each has a passing companion case at exactly the documented maximum. The maxima are
 * sized to the core limits (128 MiB storage bindings, `capacity x lanes <= 2^32 - 1`).
 */
export const KERNEL_AUDIT_REJECTIONS: readonly KernelAuditRejection[] = [
  ...DATAFRAME_REJECTIONS,
  ...RASTER_REJECTIONS,
  ...RASTER_OPERATIONS_REJECTIONS,
  ...TERRAIN_REJECTIONS,
  ...NETWORK_REJECTIONS,
  ...STATISTICS_REJECTIONS,
  ...GEOMETRY_REJECTIONS,
  ...GEOMETRY_LINES_REJECTIONS,
  ...CELLS_AND_JOINS_REJECTIONS,
  ...POINTS_AND_REGIONS_REJECTIONS
];

/**
 * Messages of the generic declaration guards (`createWGSLKernelNode`, `GPUCommandGraph` buffers,
 * dispatch layout). They are backstops: a documented maximum that only they refuse is a finding,
 * because their message names neither the contributor ID nor the prop.
 */
export const GENERIC_REFUSAL_PATTERNS: readonly RegExp[] = [
  /: kernel \S+ binding \w+ needs \d+ bytes/,
  /GPUCommandGraph buffer "[^"]+" exceeds/,
  /element count must be a non-negative uint32/,
  /exceeds (one bounded|bounded one-dimensional|device dispatch)/
];
