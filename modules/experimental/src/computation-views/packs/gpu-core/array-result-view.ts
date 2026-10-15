// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ViewContext, ViewRegistration} from '../../registry/view-registration';
import type {DescriptionFacts} from '../../types';
import {h, type ViewElementNode} from '../../view-node';
import {provenancePatternDefs} from '../../encoding/provenance';
import {getAdapterLabel} from '../../views/record-queries';
import {
  buildGPUCoreArrayResultModel,
  getArrayResultCaptures,
  getArrayTokens,
  getArrayTokenValue,
  type GPUCoreArrayResultModel
} from './array-result-model';
import {renderArrayStrip, renderArrayConnections} from './array-strip-render';
import {renderArrayTile} from './array-tile-render';
import {renderArrayTable} from './array-table-render';

export {buildGPUCoreArrayResultModel} from './array-result-model';
export {renderArrayTable} from './array-table-render';

/** Renders the pure result projection; all interaction uses delegated pick attributes. */
export function renderGPUCoreArrayResult(
  model: GPUCoreArrayResultModel,
  context: ViewContext
): ViewElementNode {
  const tablePanels = model.panels.filter(
    panel => getArrayTokenValue(panel.tokens, 'column') !== undefined
  );
  const panels = model.panels.filter(panel => !tablePanels.includes(panel));
  const tableGroups = ['input', 'context', 'output']
    .map(role => tablePanels.filter(panel => panel.role === role))
    .filter(columns => columns.length);
  const classed = model.panels.filter(
    panel => panel.role === 'output' && (panel.shape || panel.tokens.includes('stencil'))
  );
  return h(
    'section',
    {class: 'lens-view lens-gpu-core-array-result'},
    h(
      'svg',
      {
        class: 'cv-figure',
        width: model.width,
        height: model.height,
        viewBox: `0 0 ${model.width} ${model.height}`,
        role: 'img',
        'aria-label': 'Recorded array inputs and outputs'
      },
      provenancePatternDefs(context.idPrefix, context.ground),
      renderArrayConnections(model, context),
      panels.map(panel =>
        panel.shape || panel.tokens.includes('stencil')
          ? renderArrayTile(panel, model, context)
          : renderArrayStrip(panel, model, context)
      ),
      model.thumbnail
        ? tablePanels.map(panel => renderArrayStrip(panel, model, context))
        : tableGroups.map((columns, index) =>
            renderArrayTable(columns, context, {
              x: 24 + (index * 592) / tableGroups.length,
              y: 48,
              width: 592 / tableGroups.length - 16
            })
          )
    ),
    !model.thumbnail && classed.length
      ? h(
          'p',
          {class: 'cv-legend'},
          classed
            .map(
              panel =>
                `${panel.capture.name}: ${panel.palette === 'div' ? `diverging, midpoint ${panel.scale?.midpoint ?? 'unknown'}` : 'ordered lightness'}, ${panel.scale?.legend().classes.filter(entry => !entry.beyondRange).length ?? 5} classes`
            )
            .join(' · ')
        )
      : null,
    !model.thumbnail && model.notes.some(note => note.includes('unknown'))
      ? h(
          'p',
          {class: 'lens-unknown'},
          model.notes.filter(note => note.includes('unknown')).join('; ')
        )
      : null
  );
}

/** One facts object feeds the caption, accessible description, and llms twin. */
export function describeGPUCoreArrayResult(
  model: GPUCoreArrayResultModel,
  context: ViewContext
): DescriptionFacts {
  return {
    subject: 'Read captured inputs into recorded array outputs',
    keyNumber: model.keyNumber,
    facts: [
      `${model.panels.filter(panel => panel.role === 'output').length} output panels from the recorded run`,
      ...model.notes
    ],
    provenance: model.panels.some(panel => panel.provenance === 'derived')
      ? 'derived'
      : model.panels.some(panel => panel.provenance === 'replay')
        ? 'replay'
        : 'readback',
    adapterLabel: getAdapterLabel(context.record)
  };
}

/** GPU Core's role-tagged array result grammar, preferred over the generic result fallback. */
export const GPU_CORE_ARRAY_RESULT_VIEW: ViewRegistration<GPUCoreArrayResultModel> = {
  id: 'gpu-core-array-result',
  label: 'Array result',
  rung: 'result',
  recordType: 'record',
  order: 10,
  signature: {binds: [], free: ['entity'], layout: {entity: 'position'}},
  descriptor: {audience: 'expert', task: 'present', interaction: 'low'},
  level: 'realized',
  grain: 'contributor',
  satisfies(context) {
    const captures = getArrayResultCaptures(context);
    if (!captures.some(capture => capture.role === 'output'))
      return {
        ok: false,
        reason: 'Array result needs at least one final capture explicitly tagged role=output'
      };
    const hasGrid = captures.some(
      capture => capture.role === 'output' && getArrayTokens(capture).includes('grid')
    );
    if (
      captures.some(
        capture =>
          capture.geometry &&
          !(hasGrid && capture.role === 'input' && capture.geometry.kind === 'points')
      )
    )
      return {
        ok: false,
        reason:
          'Role-tagged geometry belongs to ResultView; only a points input accompanying an array grid is supported here'
      };
    return {ok: true};
  },
  build: buildGPUCoreArrayResultModel,
  render: renderGPUCoreArrayResult,
  describe: describeGPUCoreArrayResult
};
