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
import {
  getPinnedEntity,
  getAdapterLabel,
  getCaptureProvenance
} from '../../../computation-views/views/record-queries';
import {evaluateGuards, createGuardContext} from '../../../computation-views/guards/guards';
import {renderVerdictChip} from '../../../computation-views/views/verdict-chip';
import {SPATIAL_GUARDS} from '../guards';
import {
  findSpatialCapture,
  getSpatialValues,
  getSpatialChoices,
  getSpatialNeighbors,
  requireSpatialCaptures,
  SPATIAL_CSR_CAPTURES
} from './spatial-record-queries';

/** Projects captured connectivity and the pinned CSR row into a compact receipt. */
export function buildNeighbourhoodViewModel(context: ViewContext) {
  const counts = getSpatialValues(context, 'neighborCounts');
  const maximum = counts.reduce((largest, count) => Math.max(largest, count), 0);
  const lastBin = Math.min(16, maximum);
  const histogram = Array.from({length: lastBin + 1}, (_, count) => ({
    label: count === 16 ? '≥16' : String(count),
    count: 0
  }));
  counts.forEach(count => histogram[Math.min(lastBin, count)].count++);
  const islands = counts.flatMap((count, key) => (count === 0 ? [key] : []));
  const core = getPinnedEntity(context);
  const neighbors = getSpatialNeighbors(context, core);
  const overflow =
    findSpatialCapture(context, 'weightsOverflow') ??
    findSpatialCapture(context, 'overflow') ??
    findSpatialCapture(context, 'edgeOverflow') ??
    findSpatialCapture(context, 'capacityOverflow');
  const guardContext = createGuardContext(context.record, context.packs ?? []);
  const verdict = evaluateGuards([SPATIAL_GUARDS['islands-disclosed']], {
    ...guardContext,
    metadata: {...guardContext.metadata, islandCount: islands.length}
  })[0];
  return {
    histogram,
    islands,
    core,
    neighbors,
    verdict,
    rowSum: neighbors.reduce((sum, neighbor) => sum + neighbor.weight, 0),
    standardisation: getSpatialChoices(context.record)['weightsStandardisation'],
    overflow: overflow ? Array.from(overflow.values).some(value => value !== 0) : undefined,
    provenance: getCaptureProvenance(findSpatialCapture(context, 'neighborCounts')!)
  };
}

/** Renders connectivity, hatched islands and the core's weighted star without map geometry. */
export function renderNeighbourhoodView(
  model: ReturnType<typeof buildNeighbourhoodViewModel>,
  context: ViewContext
) {
  return h(
    'section',
    {class: 'lens-view lens-neighbourhood-view'},
    renderConnectivityHistogram(model, context),
    h(
      'p',
      {class: 'lens-caption'},
      `${model.islands.length} islands, hatched; core ${model.core ?? 'unpinned'} has ${model.neighbors.length} neighbours.`
    ),
    renderVerdictChip(model.verdict),
    provenanceBadge(model.provenance),
    h('p', null, `Islands: ${model.islands.length ? model.islands.join(', ') : 'none'}`),
    model.core === undefined
      ? h('span', {class: 'lens-unknown'}, 'unknown — pin a core for its star')
      : h(
          'div',
          null,
          h(
            'p',
            null,
            `Core ${model.core}; row sum ${model.rowSum}${model.standardisation === 'row' ? '; row-standardised weights' : ''}`
          ),
          h(
            'ul',
            null,
            model.neighbors.map(neighbor =>
              h(
                'li',
                null,
                h(
                  'button',
                  {'data-action': 'pick', 'data-axis': 'entity', 'data-key': neighbor.key},
                  `${neighbor.key}: weight ${neighbor.weight}`
                ),
                provenanceBadge(getCaptureProvenance(findSpatialCapture(context, 'weightsValues')!))
              )
            )
          )
        ),
    model.overflow === undefined
      ? null
      : h(
          'p',
          null,
          `Edge truncation: ${model.overflow ? 'capacity exceeded; edges truncated' : 'none (overflow flag 0)'}`
        ),
    h('p', null, getAdapterLabel(context.record))
  );
}

function renderConnectivityHistogram(
  model: ReturnType<typeof buildNeighbourhoodViewModel>,
  context: ViewContext
) {
  const maximum = Math.max(1, ...model.histogram.map(bin => bin.count));
  const step = 592 / model.histogram.length;

  return h(
    'svg',
    {
      viewBox: '0 0 640 180',
      role: 'img',
      'aria-label': 'Rows by number of neighbours',
      style: 'width:100%;height:auto;max-width:720px'
    },
    provenancePatternDefs(context.idPrefix, context.ground),
    model.histogram.map((bin, index) =>
      h(
        'g',
        null,
        h(
          'rect',
          {
            ...getCardMarkAttributes('output', 'cell'),
            ...provenanceMarkAttributes('derived', 'currentColor', context.idPrefix),
            x: 24 + index * step,
            y: 144 - (bin.count / maximum) * 120,
            width: step - 3,
            height: (bin.count / maximum) * 120,
            style: `fill:url(#${context.idPrefix}-hatched);stroke:currentColor`,
            opacity: index === 0 ? 1 : 0.55,
            'data-islands': index === 0 ? bin.count : undefined
          },
          h('title', null, `${bin.label} neighbours: ${bin.count} rows`)
        ),
        h('text', {x: 24 + index * step, y: 164, class: 'cv-label'}, bin.label)
      )
    )
  );
}

/** Optional datum view for the weights choice and its islands (M6). */
export const NEIGHBOURHOOD_VIEW: ViewRegistration<ReturnType<typeof buildNeighbourhoodViewModel>> =
  {
    id: 'neighbourhood-view',
    label: 'NeighbourhoodView',
    rung: 'datum',
    recordType: 'record',
    order: 60,
    signature: {binds: [], free: ['entity'], layout: {entity: 'position'}},
    descriptor: {audience: 'analyst', task: 'explore', interaction: 'high'},
    level: 'realized',
    grain: 'contributor',
    satisfies: context =>
      requireSpatialCaptures(context, ['neighborCounts', ...SPATIAL_CSR_CAPTURES]),
    build: buildNeighbourhoodViewModel,
    render: renderNeighbourhoodView,
    describe: (model, context) => ({
      subject: 'Neighbourhood connectivity',
      keyNumber: {label: 'islands', value: model.islands.length},
      facts: [
        model.core === undefined
          ? 'core unpinned; row sum unknown'
          : `core ${model.core}; row sum ${model.rowSum}`,
        `${model.islands.length} islands, hatched`
      ],
      provenance: 'derived',
      adapterLabel: getAdapterLabel(context.record)
    })
  };
