// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  ViewContext,
  ViewRegistration
} from '../../../computation-views/registry/view-registration';
import {h} from '../../../computation-views/view-node';
import {
  getCardMarkAttributes,
  CARD_NOMINAL_COLORS
} from '../../../computation-views/encoding/card-tokens';
import {
  provenanceBadge,
  provenanceMarkAttributes,
  provenancePatternDefs
} from '../../../computation-views/encoding/provenance';
import {getSelectionInk} from '../../../computation-views/encoding/selection-ink';
import {
  getPinnedEntity,
  getAdapterLabel,
  getCaptureProvenance,
  formatViewNumber
} from '../../../computation-views/views/record-queries';
import {evaluateGuards, createGuardContext} from '../../../computation-views/guards/guards';
import {renderVerdictChip} from '../../../computation-views/views/verdict-chip';
import {SPATIAL_GUARDS} from '../guards';
import {
  findSpatialCapture,
  getSpatialValues,
  getSpatialSignificanceLevel,
  getSpatialNeighbors,
  requireSpatialCaptures,
  SPATIAL_CSR_CAPTURES
} from './spatial-record-queries';

/** Builds a Moran scatter from the recorded run's reference distribution and adjusted inference. */
export function buildMoranScatterModel(context: ViewContext) {
  const values = getSpatialValues(context, 'values');
  const statistics = findSpatialCapture(context, 'globalStatistics');
  const finite = values.filter(Number.isFinite);
  const mean = statistics
    ? Number(statistics.values[1])
    : finite.reduce((sum, value) => sum + value, 0) / finite.length;
  const deviation = statistics ? Number(statistics.values[3]) : undefined;
  const standardised = deviation !== undefined && Number.isFinite(deviation) && deviation > 0;
  const lag = getSpatialValues(context, 'spatialLag');
  const adjusted = getSpatialValues(context, 'adjustedPValues');
  const quadrants = getSpatialValues(context, 'quadrants');
  const significanceLevel = getSpatialSignificanceLevel(context);
  const points = values.map((value, key) => ({
    key,
    horizontal: (value - mean) / (standardised ? deviation : 1),
    // The kernel publishes centred lag, so standardise both axes with the same run reference.
    vertical: lag[key] / (standardised ? deviation : 1),
    quadrant: quadrants[key],
    significant:
      significanceLevel !== undefined && adjusted[key] >= 0 && adjusted[key] <= significanceLevel
  }));
  const visible = points.filter(
    point => Number.isFinite(point.horizontal) && Number.isFinite(point.vertical)
  );
  const denominator = visible.reduce((sum, point) => sum + point.horizontal ** 2, 0);
  const slope =
    denominator > 0
      ? visible.reduce((sum, point) => sum + point.horizontal * point.vertical, 0) / denominator
      : undefined;
  const verdict = evaluateGuards(
    [SPATIAL_GUARDS['row-standardisation']],
    createGuardContext(context.record, context.packs ?? [])
  )[0];
  return {
    points: visible,
    omitted: points.length - visible.length,
    standardised,
    significanceLevel,
    slope,
    verdict,
    significantCount: visible.filter(point => point.significant).length,
    core: getPinnedEntity(context),
    neighbors: getSpatialNeighbors(context, getPinnedEntity(context)),
    provenance: getCaptureProvenance(findSpatialCapture(context, 'spatialLag')!)
  };
}

/** Draws only inference-supported quadrant hues, with an achromatic core and outlined neighbours. */
export function renderMoranScatter(
  model: ReturnType<typeof buildMoranScatterModel>,
  context: ViewContext
) {
  return h(
    'section',
    {class: 'lens-view lens-moran-scatter'},
    renderMoranFigure(model, context),
    model.verdict.status === 'pass' && model.slope !== undefined
      ? h('p', null, `slope = I: ${formatViewNumber(model.slope)}`)
      : renderVerdictChip(model.verdict),
    model.slope === undefined
      ? h('span', {class: 'lens-unknown'}, 'unknown — zero scatter variance')
      : null,
    model.significanceLevel === undefined
      ? h(
          'span',
          {class: 'lens-unknown'},
          'unknown — significanceLevel missing; no quadrant colouring'
        )
      : h(
          'p',
          {class: 'cv-note'},
          `Adjusted p ≤ ${model.significanceLevel}; `,
          ['HH', 'LH', 'LL', 'HL'].map((label, index) =>
            h(
              'span',
              null,
              h('span', {
                'aria-hidden': true,
                style: `display:inline-block;width:12px;height:12px;margin:0 4px;background:var(--cv-nom-${index + 1},${CARD_NOMINAL_COLORS[index]})`
              }),
              label
            )
          )
        ),
    h(
      'p',
      {class: 'lens-caption'},
      `${model.significanceLevel === undefined ? `${model.points.length} rows; significance unknown` : `${model.significantCount} significant rows`}; core solid, neighbours outlined; ${model.omitted} nonfinite rows omitted.`
    ),
    h('p', {class: 'cv-note'}, getMoranSamplingNote(model, context)),
    provenanceBadge(model.standardised ? 'derived' : model.provenance),
    h('p', null, getAdapterLabel(context.record))
  );
}

function getMoranDisplayPoints(
  model: ReturnType<typeof buildMoranScatterModel>,
  context: ViewContext
) {
  const limit = context.options?.['size'] === 'thumbnail' ? 24 : 160;
  if (model.points.length <= limit) return model.points;
  const retainedKeys = new Set([
    model.core,
    ...model.neighbors.slice(0, 16).map(neighbor => neighbor.key)
  ]);
  const retained = new Set(
    model.points.flatMap((point, index) => (retainedKeys.has(point.key) ? [index] : []))
  );
  const sampleCount = limit - retained.size;
  for (let index = 0; index < sampleCount; index++) {
    retained.add(Math.round((index * (model.points.length - 1)) / (sampleCount - 1)));
  }
  return [...retained].sort((first, second) => first - second).map(index => model.points[index]);
}

function getMoranSamplingNote(
  model: ReturnType<typeof buildMoranScatterModel>,
  context: ViewContext
): string {
  const shown = getMoranDisplayPoints(model, context).length;
  return shown < model.points.length
    ? `${shown} of ${model.points.length} scatter points shown: evenly sampled with the pinned core and up to 16 neighbours retained. Counts, slope and axis extents use all finite rows.`
    : 'All finite scatter points shown.';
}

function renderMoranFigure(model: ReturnType<typeof buildMoranScatterModel>, context: ViewContext) {
  const horizontalExtent = model.points.reduce(
    (extent, point) => Math.max(extent, Math.abs(point.horizontal)),
    1
  );
  const verticalExtent = model.points.reduce(
    (extent, point) => Math.max(extent, Math.abs(point.vertical)),
    1
  );
  const horizontal = (value: number) => 330 + (value / horizontalExtent) * 270;
  const vertical = (value: number) => 160 - (value / verticalExtent) * 126;
  const ink = getSelectionInk(context.ground);
  const neighbors = new Set(model.neighbors.map(neighbor => neighbor.key));
  const lineExtent = model.slope
    ? Math.min(horizontalExtent, verticalExtent / Math.abs(model.slope))
    : horizontalExtent;

  return h(
    'svg',
    {
      viewBox: '0 0 640 340',
      role: 'img',
      'aria-label': 'Moran scatter',
      style: 'width:100%;height:auto;max-width:720px'
    },
    provenancePatternDefs(context.idPrefix, context.ground),
    h('path', {
      ...getCardMarkAttributes('context', 'line'),
      d: 'M60 160H600 M330 34V286',
      stroke: 'currentColor',
      style: 'stroke:currentColor;fill:none;stroke-width:0.75px',
      opacity: 0.2
    }),
    h(
      'text',
      {x: 330, y: 320, 'text-anchor': 'middle', class: 'cv-label'},
      model.standardised ? 'z(x_i)' : 'centred, not standardised'
    ),
    h(
      'text',
      {
        x: 12,
        y: 160,
        transform: 'rotate(-90 12 160)',
        'text-anchor': 'middle',
        class: 'cv-label'
      },
      model.standardised ? 'spatialLag / sd' : 'spatialLag'
    ),
    model.verdict.status === 'pass' && model.slope !== undefined
      ? h('line', {
          ...getCardMarkAttributes('output', 'line'),
          ...provenanceMarkAttributes('derived', 'currentColor', context.idPrefix),
          x1: horizontal(-lineExtent),
          y1: vertical(-lineExtent * model.slope),
          x2: horizontal(lineExtent),
          y2: vertical(lineExtent * model.slope),
          stroke: 'currentColor',
          style: 'stroke:currentColor;fill:none',
          opacity: 0.55
        })
      : null,
    getMoranDisplayPoints(model, context).map(point => {
      const nominal = point.significant && point.quadrant >= 1 && point.quadrant <= 4;
      const attributes = getCardMarkAttributes(
        'output',
        'point',
        nominal ? {palette: 'nom', classIndex: point.quadrant - 1} : {}
      );
      const color = nominal
        ? `var(--cv-nom-${point.quadrant}, ${CARD_NOMINAL_COLORS[point.quadrant - 1]})`
        : 'currentColor';
      const provenance = provenanceMarkAttributes(
        model.standardised ? 'derived' : model.provenance,
        color,
        context.idPrefix
      );
      const core = point.key === model.core;
      return h(
        'g',
        {'data-action': 'pick', 'data-axis': 'entity', 'data-key': point.key},
        core
          ? h('circle', {
              cx: horizontal(point.horizontal),
              cy: vertical(point.vertical),
              r: 6,
              fill: 'none',
              stroke: ink.halo,
              'stroke-width': 4,
              'data-state': 'selection'
            })
          : null,
        h(
          'circle',
          {
            ...attributes,
            ...provenance,
            style: `fill:${provenance['fill']};stroke:${color}`,
            cx: horizontal(point.horizontal),
            cy: vertical(point.vertical),
            r: core ? 4.5 : 3,
            ...(core
              ? {
                  fill: ink.core,
                  stroke: ink.core,
                  'stroke-width': 2,
                  style: `fill:${ink.core};stroke:${ink.core};stroke-width:2px`
                }
              : neighbors.has(point.key)
                ? {
                    fill: 'none',
                    stroke: ink.core,
                    'stroke-width': 2,
                    style: `fill:none;stroke:${ink.core};stroke-width:2px`
                  }
                : {}),
            opacity: nominal || core || neighbors.has(point.key) ? 1 : 0.55
          },
          h('title', null, `row ${point.key}${core ? ' core' : ''}`)
        )
      );
    })
  );
}

/** Optional datum view for local Moran inference and the slope premise (M1, M4, M5, M7). */
export const MORAN_SCATTER_VIEW: ViewRegistration<ReturnType<typeof buildMoranScatterModel>> = {
  id: 'moran-scatter',
  label: 'MoranScatter',
  rung: 'datum',
  recordType: 'record',
  order: 61,
  signature: {binds: [], free: ['entity'], layout: {entity: 'position'}},
  descriptor: {audience: 'analyst', task: 'explore', interaction: 'high'},
  level: 'realized',
  grain: 'contributor',
  satisfies: context =>
    requireSpatialCaptures(context, [
      'values',
      'spatialLag',
      'adjustedPValues',
      'quadrants',
      ...SPATIAL_CSR_CAPTURES
    ]),
  build: buildMoranScatterModel,
  render: renderMoranScatter,
  describe: (model, context) => ({
    subject: 'Moran scatter',
    keyNumber:
      model.verdict.status === 'pass' && model.slope !== undefined
        ? {label: 'slope = I', value: model.slope}
        : model.significanceLevel === undefined
          ? {label: 'rows; significance unknown', value: model.points.length}
          : {label: 'significant rows', value: model.significantCount},
    facts: [
      model.standardised ? 'both centred axes divided by run sd' : 'centred, not standardised',
      'core solid; neighbours outlined',
      getMoranSamplingNote(model, context)
    ],
    caveats:
      model.significanceLevel === undefined
        ? ['unknown significance level; no quadrant colouring']
        : [],
    provenance: 'derived',
    adapterLabel: getAdapterLabel(context.record)
  })
};
