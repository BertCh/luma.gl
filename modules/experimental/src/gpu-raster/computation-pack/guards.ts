// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GuardSpec} from '../../computation-views/guards/guards';

/**
 * Metric derivatives (slope, distance, cost per metre) need metric cells. Reads the run's
 * `cellSizeMode` choice and the input raster's coordinate model from the record: `uniform` cells
 * on a geographic grid would treat degrees as metres (misconception M8 for rasters).
 */
export const RASTER_METRIC_CELL_SIZE_GUARD: GuardSpec = {
  id: 'metric-cell-size',
  label: 'Metric cell size',
  blocks: ['metric-derivative', 'metric-distance'],
  evaluate: ({record}) => {
    const mode = record?.choices.find(choice => choice.name === 'cellSizeMode')?.value;
    const raster = record?.captures?.snapshots.find(
      capture => capture.role === 'input' && capture.geometry?.kind === 'raster'
    )?.geometry;
    const model = raster?.kind === 'raster' ? raster.coordinateModel : undefined;
    if (typeof mode !== 'string') return {status: 'unknown', reason: 'missing cellSizeMode choice'};
    if (mode !== 'uniform') {
      return {status: 'pass', reason: `${mode} cell sizes are converted to ground metres`};
    }
    if (model === 'projected') return {status: 'pass', reason: 'uniform metric cells'};
    if (model === 'geographic') {
      return {
        status: 'fail',
        reason: 'uniform cell size on a lon/lat grid reads degrees as metres'
      };
    }
    return {status: 'unknown', reason: 'input raster has no stated coordinate model'};
  }
};

/**
 * A gated fixpoint loop must have converged before its values are drawn as final. Reads the
 * capture whose meaning starts with `converged` (the contributor's own GPU status word): 1 passes,
 * 0 fails, absent is unknown.
 */
export const RASTER_CONVERGENCE_GUARD: GuardSpec = {
  id: 'convergence',
  label: 'Converged',
  blocks: ['final-value', 'wavefront'],
  evaluate: ({record}) => {
    const flags =
      record?.captures?.snapshots.filter(capture => capture.meaning?.startsWith('converged')) ?? [];
    if (!flags.length) return {status: 'unknown', reason: 'no convergence word captured'};
    const failed = flags.filter(capture => capture.values[0] !== 1);
    return failed.length
      ? {
          status: 'fail',
          reason: `${failed.map(capture => capture.name).join(', ')} hit the iteration limit`
        }
      : {status: 'pass', reason: 'every gated loop reached its fixpoint'};
  }
};

/** Guards the raster and terrain packs can apply, by id. */
export const RASTER_GUARDS = {
  'metric-cell-size': RASTER_METRIC_CELL_SIZE_GUARD,
  convergence: RASTER_CONVERGENCE_GUARD
} as const satisfies Readonly<Record<string, GuardSpec>>;

/** Id of a raster/terrain guard. */
export type RasterGuardId = keyof typeof RASTER_GUARDS;

/** Resolves guard ids to specs in the declared order; unknown ids throw. */
export function getRasterGuards(ids: readonly string[]): GuardSpec[] {
  return ids.map(id => {
    const guard = (RASTER_GUARDS as Readonly<Record<string, GuardSpec>>)[id];
    if (!guard) throw new Error(`Unknown raster guard ${id}`);
    return guard;
  });
}
