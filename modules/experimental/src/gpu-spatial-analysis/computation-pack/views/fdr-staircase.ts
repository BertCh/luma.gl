// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  ViewContext,
  ViewRegistration
} from '../../../computation-views/registry/view-registration';
import {h} from '../../../computation-views/view-node';
import {getCardMarkAttributes} from '../../../computation-views/encoding/card-tokens';
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
import {
  findSpatialCapture,
  getSpatialValues,
  getSpatialChoices,
  getSpatialSignificanceLevel,
  requireSpatialCaptures
} from './spatial-record-queries';

/** Sorts the captured family, preserving original keys and the exact BH crossing before decimation. */
export function buildFdrStaircaseModel(context: ViewContext) {
  const values = getSpatialValues(context, 'pValues');
  const sorted = values
    .map((value, key) => ({value, key}))
    .filter(point => Number.isFinite(point.value) && point.value >= 0 && point.value <= 1)
    .sort((first, second) => first.value - second.value || first.key - second.key);
  const significanceLevel = getSpatialSignificanceLevel(context);
  let rejectionCount = 0;
  if (significanceLevel !== undefined)
    sorted.forEach((point, index) => {
      if (point.value <= ((index + 1) * significanceLevel) / sorted.length)
        rejectionCount = index + 1;
    });
  const choices = getSpatialChoices(context.record);
  const permutations = choices['permutations'];
  const floor =
    typeof permutations === 'number' && Number.isInteger(permutations) && permutations > 0
      ? 1 / (permutations + 1)
      : undefined;
  const maximum =
    significanceLevel === undefined
      ? 1
      : Math.min(1, Math.max(significanceLevel * 2, (floor ?? 0) * 1.2));
  const core = getPinnedEntity(context);
  const thumbnail = context.options?.['size'] === 'thumbnail';
  const stride = Math.max(1, Math.ceil(sorted.length / (thumbnail ? 24 : 160)));
  const points = sorted.flatMap((point, index) =>
    index % stride === 0 ||
    index === sorted.length - 1 ||
    Math.abs(index + 1 - rejectionCount) <= 1 ||
    point.key === core
      ? [{...point, rank: index + 1}]
      : []
  );
  const gridCount = floor ? Math.floor(maximum / floor) : 0;
  const gridStride = Math.max(1, Math.ceil(gridCount / (thumbnail ? 8 : 32)));
  const grid = floor
    ? Array.from(
        {length: Math.ceil(gridCount / gridStride)},
        (_, index) => (1 + index * gridStride) * floor
      ).filter(value => value <= maximum)
    : [];
  return {
    points,
    familySize: sorted.length,
    omitted: values.length - sorted.length,
    rejectionCount,
    significanceLevel,
    floor,
    grid,
    gridStride,
    maximum,
    core,
    aboveWindow: sorted.filter(point => point.value > maximum).length,
    correction: choices['pCorrection'] ?? choices['multipleTesting'],
    uncorrected: (choices['pCorrection'] ?? choices['multipleTesting']) === 'none',
    provenance: getCaptureProvenance(findSpatialCapture(context, 'pValues')!)
  };
}

/** Draws the readable linear alpha window, BH boundary and achievable permutation grid. */
export function renderFdrStaircase(
  model: ReturnType<typeof buildFdrStaircaseModel>,
  context: ViewContext
) {
  return h(
    'section',
    {class: 'lens-view lens-fdr-staircase'},
    model.uncorrected ? h('span', {class: 'lens-badge'}, 'uncorrected') : null,
    model.correction === undefined
      ? h('span', {class: 'lens-unknown'}, 'unknown — pCorrection missing; BH shown as comparison')
      : null,
    renderFdrFigure(model, context),
    model.significanceLevel === undefined
      ? h(
          'span',
          {class: 'lens-unknown'},
          'unknown — significanceLevel missing; BH rejection count unavailable'
        )
      : h(
          'p',
          {class: 'lens-caption'},
          `${model.rejectionCount} BH rejections among ${model.familySize} tests; largest crossing marked.`
        ),
    model.floor === undefined
      ? h(
          'span',
          {class: 'lens-unknown'},
          'unknown — permutations missing; achievable grid unavailable'
        )
      : h(
          'p',
          {class: 'cv-note'},
          `Grid k/(M+1); ${model.gridStride > 1 ? `every ${model.gridStride}th level shown, including floor` : 'all levels in window shown'}`
        ),
    h(
      'p',
      {class: 'cv-note'},
      `Linear window 0–${formatViewNumber(model.maximum)}; ${model.aboveWindow} p-values above window; ${model.omitted} invalid values omitted.${model.familySize > model.points.length ? ` ${model.points.length} sampled ranks; crossing and pinned rank retained.` : ''}`
    ),
    provenanceBadge(model.provenance),
    h('p', null, getAdapterLabel(context.record))
  );
}

function renderFdrFigure(model: ReturnType<typeof buildFdrStaircaseModel>, context: ViewContext) {
  const horizontal = (rank: number) => 52 + (rank / Math.max(1, model.familySize)) * 550;
  const vertical = (value: number) => 280 - (value / model.maximum) * 250;
  const ink = getSelectionInk(context.ground);

  return h(
    'svg',
    {
      viewBox: '0 0 640 330',
      role: 'img',
      'aria-label': 'Sorted p-values and Benjamini–Hochberg boundary',
      style: 'width:100%;height:auto;max-width:720px'
    },
    provenancePatternDefs(context.idPrefix, context.ground),
    h('path', {
      ...getCardMarkAttributes('context', 'line'),
      d: 'M52 30V280H602',
      stroke: 'currentColor',
      style: 'stroke:currentColor;fill:none;stroke-width:0.75px',
      opacity: 0.2
    }),
    h('text', {x: 330, y: 315, class: 'cv-label'}, 'rank i'),
    h('text', {x: 12, y: 25, class: 'cv-label'}, 'p'),
    model.grid.map(value =>
      h('line', {
        ...getCardMarkAttributes('context', 'line'),
        ...provenanceMarkAttributes('metadata', 'currentColor', context.idPrefix),
        x1: 52,
        x2: 602,
        y1: vertical(value),
        y2: vertical(value),
        style: 'stroke:currentColor;fill:none;stroke-width:0.75px',
        opacity: 0.2
      })
    ),
    model.floor !== undefined
      ? h(
          'text',
          {x: 55, y: vertical(model.floor) - 4, class: 'cv-label'},
          `floor 1/(M+1) = ${formatViewNumber(model.floor)}`
        )
      : null,
    model.significanceLevel !== undefined
      ? h(
          'g',
          null,
          h('line', {
            ...getCardMarkAttributes('output', 'line'),
            ...provenanceMarkAttributes('metadata', 'currentColor', context.idPrefix),
            x1: horizontal(0),
            y1: vertical(0),
            x2: horizontal(model.familySize),
            y2: vertical(model.significanceLevel),
            style: 'stroke:currentColor;fill:none',
            opacity: 0.55
          }),
          h(
            'text',
            {x: 430, y: vertical(model.significanceLevel) - 8, class: 'cv-label'},
            'BH: i·α/m'
          )
        )
      : null,
    h('path', {
      ...getCardMarkAttributes('output', 'line'),
      ...provenanceMarkAttributes('derived', 'currentColor', context.idPrefix),
      d: model.points
        .filter(point => point.value <= model.maximum)
        .map(
          (point, index) =>
            `${index ? 'H' : 'M'}${horizontal(point.rank)}${index ? 'V' : ','}${vertical(Math.min(point.value, model.maximum))}`
        )
        .join(' '),
      style: 'fill:none;stroke:currentColor',
      opacity: 0.55
    }),
    model.points
      .filter(point => point.value <= model.maximum || point.key === model.core)
      .map(point =>
        h(
          'g',
          {'data-action': 'pick', 'data-axis': 'entity', 'data-key': point.key},
          point.key === model.core
            ? h('circle', {
                cx: horizontal(point.rank),
                cy: vertical(Math.min(point.value, model.maximum)),
                r: 6,
                fill: 'none',
                stroke: ink.halo,
                'stroke-width': 4
              })
            : null,
          h(
            'circle',
            {
              ...getCardMarkAttributes('output', 'point'),
              ...provenanceMarkAttributes(model.provenance, 'currentColor', context.idPrefix),
              cx: horizontal(point.rank),
              cy: vertical(Math.min(point.value, model.maximum)),
              r: point.key === model.core ? 4.5 : 3,
              style: `fill:${provenanceMarkAttributes(model.provenance, 'currentColor', context.idPrefix)['fill']};stroke:${point.key === model.core ? ink.core : 'currentColor'};stroke-width:${point.key === model.core ? 2 : 0.75}px`,
              stroke: point.key === model.core ? ink.core : 'currentColor',
              'stroke-width': point.key === model.core ? 2 : 0.75
            },
            h('title', null, `row ${point.key}; rank ${point.rank}; p ${point.value}`)
          ),
          point.key === model.core && point.value > model.maximum
            ? h(
                'text',
                {x: horizontal(point.rank), y: 18, class: 'cv-label'},
                `↑ pinned p ${formatViewNumber(point.value)}`
              )
            : null,
          point.rank === model.rejectionCount
            ? h(
                'text',
                {
                  x: horizontal(point.rank) + 5,
                  y: vertical(Math.min(point.value, model.maximum)) - 8,
                  class: 'cv-label'
                },
                `${model.rejectionCount} rejected`
              )
            : null
        )
      )
  );
}

/** Optional datum view for multiple testing and permutation resolution (M2, M3). */
export const FDR_STAIRCASE_VIEW: ViewRegistration<ReturnType<typeof buildFdrStaircaseModel>> = {
  id: 'fdr-staircase',
  label: 'FDR staircase',
  rung: 'datum',
  recordType: 'record',
  order: 62,
  signature: {binds: [], free: ['entity'], layout: {entity: 'position'}},
  descriptor: {audience: 'analyst', task: 'explore', interaction: 'high'},
  level: 'realized',
  grain: 'contributor',
  satisfies: context => requireSpatialCaptures(context, ['pValues']),
  build: buildFdrStaircaseModel,
  render: renderFdrStaircase,
  describe: (model, context) => ({
    subject: 'Benjamini–Hochberg staircase',
    keyNumber:
      model.significanceLevel === undefined
        ? {label: 'tests; rejections unknown', value: model.familySize}
        : {label: 'BH rejections', value: model.rejectionCount},
    facts: [
      `${model.familySize} valid tests`,
      model.floor === undefined ? 'unknown permutation grid' : `achievable floor ${model.floor}`,
      ...(model.uncorrected ? ['uncorrected run; BH boundary is a comparison'] : [])
    ],
    provenance: model.provenance,
    adapterLabel: getAdapterLabel(context.record)
  })
};
