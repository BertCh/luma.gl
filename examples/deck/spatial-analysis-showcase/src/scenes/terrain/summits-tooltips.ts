// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Structured tooltips of the summits story: a summit, a rejected disc maximum, the disc-and-ring
 * test of any cell, a catalogue peak and its snap, and a plain elevation. The mapped value comes
 * first with its class swatch, as the cartography guide asks.
 */

import type {LngLat} from '../../cartography/types';
import type {TooltipContent} from '../scene';
import type {DiscAndRing} from './cpu-dem';
import type {CataloguePeak, SummitCandidate} from './summits-analysis';
import {formatMeters, getSummitColors} from './summits-style';
import {SNAP_STATUS, type SnapStatusId, type TerrainGroundTone} from './terrain-palettes';
import {getSnapStatusColor} from './terrain-palettes';

type Colors = ReturnType<typeof getSummitColors>;

/** A candidate summit that passed the drop test. */
export function getSummitTooltip(options: {
  candidate: SummitCandidate;
  peak: CataloguePeak | null;
  radiusMeters: number;
  colors: Colors;
}): TooltipContent {
  const {candidate, peak, radiusMeters, colors} = options;
  return {
    title: peak?.name ?? 'Summit',
    subtitle: `Highest within ${formatMeters(radiusMeters)} m`,
    rows: [
      {
        label: 'Drop to the ring',
        value: formatMeters(candidate.drop),
        unit: 'm',
        swatch: colors.summit,
        emphasis: true
      },
      {label: 'DEM cell, analysis grid', value: formatMeters(candidate.elevation), unit: 'm'},
      ...(peak?.elevationMeters
        ? [{label: 'Published elevation', value: formatMeters(peak.elevationMeters), unit: 'm'}]
        : [])
    ],
    note: 'A lower bound of prominence: the way out may drop further.'
  };
}

/** A disc maximum that fails the drop test (a ghost). */
export function getRejectedTooltip(options: {
  candidate: SummitCandidate;
  minimumDrop: number;
  peak: CataloguePeak | null;
  colors: Colors;
}): TooltipContent {
  const {candidate, minimumDrop, peak, colors} = options;
  return {
    title: peak?.name ?? 'Highest in its disc, not a summit',
    subtitle: 'Fails the drop test',
    rows: [
      {
        label: 'Drop to the ring',
        value: formatMeters(candidate.drop),
        unit: 'm',
        swatch: colors.rejected,
        emphasis: true
      },
      {label: 'Needed', value: formatMeters(minimumDrop), unit: 'm'},
      {label: 'DEM cell, analysis grid', value: formatMeters(candidate.elevation), unit: 'm'}
    ]
  };
}

/** The disc-and-ring test of one cell, as `GPUTerrainSummits` evaluates it. */
export function getProbeTooltip(options: {
  test: DiscAndRing;
  elevation: number;
  radiusMeters: number;
  minimumDrop: number;
  center: LngLat;
  distanceToHigherMeters: number | null;
  colors: Colors;
}): TooltipContent {
  const {test, elevation, radiusMeters, minimumDrop, center, colors} = options;
  const passes = test.isMax && test.drop >= minimumDrop;
  return {
    title: passes ? 'A summit' : test.isMax ? 'Highest, but the drop is too small' : 'Not a summit',
    subtitle: `Disc of ${formatMeters(radiusMeters)} m around this cell`,
    rows: [
      {
        label: 'Highest in disc',
        value: test.isMax ? 'yes' : 'no',
        swatch: test.isMax ? colors.summit : colors.rejected,
        emphasis: true
      },
      {label: 'This cell', value: formatMeters(elevation), unit: 'm'},
      ...(Number.isFinite(test.ringMax)
        ? [{label: 'Highest ring cell', value: formatMeters(test.ringMax), unit: 'm'}]
        : []),
      ...(test.isMax && Number.isFinite(test.drop)
        ? [{label: 'Drop to the ring', value: formatMeters(test.drop), unit: 'm'}]
        : []),
      ...(!test.isMax && options.distanceToHigherMeters !== null
        ? [
            {
              label: 'Higher ground',
              value: formatMeters(options.distanceToHigherMeters),
              unit: 'm away'
            }
          ]
        : [])
    ],
    ...(test.incomplete ? {note: 'The disc leaves the DEM here.'} : {}),
    highlight: {kind: 'circle', coordinate: center, radiusMeters}
  };
}

/** One catalogue point after the snap, with the three heights a reader should compare. */
export function getSnapTooltip(options: {
  peak: CataloguePeak;
  status: SnapStatusId;
  moveMeters: number;
  errorMeters: number;
  analysisCellHeight: number;
  fullCellHeight: number | null;
  analysisCellMeters: number;
  fullCellMeters: number;
  ground: TerrainGroundTone;
}): TooltipContent {
  const {peak, status, ground} = options;
  const label = SNAP_STATUS.find(entry => entry.id === status)?.label ?? status;
  return {
    title: peak.name,
    subtitle: 'OpenStreetMap peak',
    rows: [
      {
        label: 'Snap result',
        value: label,
        swatch: getSnapStatusColor(status, ground),
        emphasis: true
      },
      {
        label: 'Catalogue elevation',
        value: peak.elevationMeters === null ? 'none' : formatMeters(peak.elevationMeters),
        ...(peak.elevationMeters === null ? {} : {unit: 'm'})
      },
      {
        label: `DEM cell, ${options.analysisCellMeters.toFixed(1)} m grid`,
        value: Number.isFinite(options.analysisCellHeight)
          ? formatMeters(options.analysisCellHeight)
          : 'none',
        ...(Number.isFinite(options.analysisCellHeight) ? {unit: 'm'} : {})
      },
      ...(options.fullCellHeight !== null
        ? [
            {
              label: `DEM cell, ${options.fullCellMeters.toFixed(1)} m grid`,
              value: formatMeters(options.fullCellHeight),
              unit: 'm'
            }
          ]
        : []),
      {label: 'Moved by the snap', value: formatMeters(options.moveMeters), unit: 'm'},
      ...(options.errorMeters > 0
        ? [{label: 'Error added', value: formatMeters(options.errorMeters), unit: 'm'}]
        : [])
    ],
    note: 'The published height and the DEM cell differ: cells average the summit with its flanks.'
  };
}

/** A plain elevation, for cells where nothing else is being asked. */
export function getElevationTooltip(options: {
  elevation: number;
  cellMeters: number;
  colors: Colors;
}): TooltipContent {
  return {
    title: 'Elevation',
    rows: [
      {label: 'DEM cell', value: formatMeters(options.elevation), unit: 'm', emphasis: true},
      {label: 'Cell size', value: options.cellMeters.toFixed(1), unit: 'm'}
    ]
  };
}
