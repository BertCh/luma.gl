// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RampName} from '../../engine/ramps';

/** Station variable that is interpolated. */
export type RainfallVariable = 'prcp' | 'prcpPrevDay' | 'tmax';

/** Surface drawn by the rainfall scene. */
export type RainfallSurface = 'idw' | 'kriging' | 'difference' | 'error' | 'support';

/** Focal statistic applied to the interpolated surface. */
export type RainfallFocalStatistic = 'mean' | 'min' | 'max' | 'range' | 'standardDeviation';

/** Static facts about the three interpolated variables of `ghcn-stations`. */
export const RAINFALL_VARIABLES: Record<
  RainfallVariable,
  {
    label: string;
    short: string;
    unit: string;
    squaredUnit: string;
    /** Color range `[low, high]`, about the 1st and 99th percentile of the stations. */
    range: readonly [number, number];
  }
> = {
  prcp: {
    label: 'Rain on 27 Sep 2024 (mm)',
    short: 'Rain, 27 Sep',
    unit: 'mm',
    squaredUnit: 'mm²',
    range: [0, 220]
  },
  prcpPrevDay: {
    label: 'Rain on 26 Sep 2024 (mm)',
    short: 'Rain, 26 Sep',
    unit: 'mm',
    squaredUnit: 'mm²',
    range: [0, 125]
  },
  tmax: {
    label: 'Daily high temperature, 27 Sep (°C)',
    short: 'High temperature',
    unit: '°C',
    squaredUnit: '°C²',
    range: [16, 34]
  }
};

const SURFACE_LABELS: Record<RainfallSurface, string> = {
  idw: 'Inverse distance weighting',
  kriging: 'Ordinary kriging',
  difference: 'Kriging minus IDW',
  error: 'Kriging standard error',
  support: 'Stations used per cell (IDW)'
};

const STATISTIC_LABELS: Record<RainfallFocalStatistic, string> = {
  mean: 'mean',
  min: 'minimum',
  max: 'maximum',
  range: 'range',
  standardDeviation: 'standard deviation'
};

/** How the surface is colored, derived from the option state. Shared by layers and legends. */
export type RainfallSurfaceStyle = {
  ramp: RampName;
  title: string;
  unit: string;
  /** Fixed `[low, high]`, or `null` when the scene reports the extent. */
  range: readonly [number, number] | null;
  /** True when the focal statistic applies to this surface. */
  focal: boolean;
};

/** Returns the color style of the drawn surface. */
export function getRainfallSurfaceStyle(state: {
  variable: RainfallVariable;
  surface: RainfallSurface;
  focalStatistic: RainfallFocalStatistic;
  focalRadius: number;
  ramp: RampName;
  neighborCount: number;
}): RainfallSurfaceStyle {
  const meta = RAINFALL_VARIABLES[state.variable];
  const span = meta.range[1] - meta.range[0];
  if (state.surface === 'error') {
    return {
      ramp: 'magma',
      title: `${SURFACE_LABELS.error}: ${meta.short}`,
      unit: meta.unit,
      range: null,
      focal: false
    };
  }
  if (state.surface === 'support') {
    return {
      ramp: 'cividis',
      title: SURFACE_LABELS.support,
      unit: 'stations',
      range: [0, state.neighborCount > 0 ? state.neighborCount : 32],
      focal: false
    };
  }
  const statistic = state.focalRadius > 0 ? state.focalStatistic : 'mean';
  const spread = statistic === 'range' || statistic === 'standardDeviation';
  const statisticNote = statistic === 'mean' ? '' : `, window ${STATISTIC_LABELS[statistic]}`;
  const base = `${meta.short}: ${SURFACE_LABELS[state.surface].toLowerCase()}${statisticNote}`;
  if (state.surface === 'difference') {
    return spread
      ? {ramp: 'magma', title: base, unit: meta.unit, range: [0, 0.2 * span], focal: true}
      : {
          ramp: 'diverging',
          title: base,
          unit: meta.unit,
          range: [-0.2 * span, 0.2 * span],
          focal: true
        };
  }
  return spread
    ? {
        ramp: 'magma',
        title: base,
        unit: meta.unit,
        range: [0, statistic === 'range' ? 0.5 * span : 0.25 * span],
        focal: true
      }
    : {ramp: state.ramp, title: base, unit: meta.unit, range: meta.range, focal: true};
}
