// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {renderRasterImage, getRasterSamplingNote} from '../encoding/raster-image';
import {assert} from '@luma.gl/core';
import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import type {LensRecord, ViewContext, ViewRegistration} from '../registry/view-registration';
import type {MarkProvenance} from '../types';
import {h, viewNodeToDom, type ViewElementNode} from '../view-node';
import {
  provenanceBadge,
  provenanceMarkAttributes,
  provenancePatternDefs
} from '../encoding/provenance';
import {coarsenValues, crispnessAttributes, crispnessDefs} from '../encoding/crispness';
import {getSelectionInk} from '../encoding/selection-ink';
import {
  getAdapterLabel,
  getAllCaptures,
  getCaptureProvenance,
  getCaptureValue,
  getPinnedEntity
} from './record-queries';
import {getFinalStateCapture, renderValueStrip, resolveValueScale} from './value-strip';

/** Buffer summary, bounded inspection table and optional iteration evidence. */
export type BufferStripModel = {
  captureId: string;
  name: string;
  keySpace: string;
  indexSpace: '1d' | '2d' | 'csr';
  shape?: readonly number[];
  count: number;
  elementType: string;
  provenance: MarkProvenance;
  stats: {min: number; max: number; nanCount: number; infCount: number; finiteCount: number};
  histogram: {bins: 16; counts: number[]; edges: number[]};
  envelope?: {min: number[]; max: number[]};
  diff?: {againstCaptureId: string; changedCount: number; maxAbsDelta: number};
  table: {key: number; value: number | string; delta?: number}[];
};

function chooseCapture(context: ViewContext): GPUComputationCapture | undefined {
  const captures = getAllCaptures(context.record);
  const requested = context.bindings.buffer ?? context.options?.['capture'];
  if (requested !== undefined)
    return (
      captures.find(capture => capture.id === requested) ??
      getFinalStateCapture(context.record, String(requested))
    );
  const snapshots = context.record.captures?.snapshots ?? [];
  const last = snapshots[snapshots.length - 1] ?? captures[captures.length - 1];
  return last && getFinalStateCapture(context.record, last.name);
}

function getIterationCaptures(context: ViewContext, name: string): GPUComputationCapture[] {
  return getAllCaptures(context.record)
    .filter(capture => capture.name === name && capture.iteration !== undefined)
    .sort((first, second) => first.iteration! - second.iteration!);
}

/** Computes finite-value statistics and differences while preserving recorded key order. */
export function buildBufferStripModel(context: ViewContext): BufferStripModel {
  const capture = chooseCapture(context);
  // A captured buffer is required by satisfies.
  assert(capture);
  const components = capture.components ?? 1;
  const count = Math.floor(capture.values.length / components);
  const keySpace = context.record.axes.keySpaces.find(
    candidate => candidate.id === capture.keySpace
  );
  const stats = {min: Infinity, max: -Infinity, nanCount: 0, infCount: 0, finiteCount: 0};
  for (let index = 0; index < capture.values.length; index++) {
    const value = Number(capture.values[index]);
    if (Number.isNaN(value)) stats.nanCount++;
    else if (!Number.isFinite(value)) stats.infCount++;
    else {
      stats.finiteCount++;
      stats.min = Math.min(stats.min, value);
      stats.max = Math.max(stats.max, value);
    }
  }
  if (!stats.finiteCount) {
    stats.min = NaN;
    stats.max = NaN;
  }
  const minimum = stats.finiteCount ? stats.min : 0;
  const maximum = stats.finiteCount ? stats.max : 0;
  const counts = new Array<number>(16).fill(0);
  const edges = Array.from(
    {length: 17},
    (_, index) => minimum + ((maximum - minimum) * index) / 16
  );
  for (let index = 0; index < capture.values.length; index++) {
    const value = Number(capture.values[index]);
    if (Number.isFinite(value))
      counts[
        maximum === minimum
          ? 0
          : Math.min(15, Math.floor(((value - minimum) / (maximum - minimum)) * 16))
      ]++;
  }
  const iterations = getIterationCaptures(context, capture.name);
  const previous = iterations
    .filter(candidate => candidate.iteration! < (capture.iteration ?? Infinity))
    .at(-1);
  let diff: BufferStripModel['diff'];
  if (previous) {
    let changedCount = 0;
    let maxAbsDelta = 0;
    for (let key = 0; key < count; key++) {
      let changed = false;
      for (let component = 0; component < components; component++) {
        const index = key * components + component;
        const value = Number(capture.values[index]);
        const prior = previous.values[index];
        if (!Object.is(value, prior) && value !== prior) changed = true;
        const delta = value - prior;
        if (!Number.isNaN(delta)) maxAbsDelta = Math.max(maxAbsDelta, Math.abs(delta));
      }
      if (changed) changedCount++;
    }
    diff = {againstCaptureId: previous.id, changedCount, maxAbsDelta};
  }
  let envelope: BufferStripModel['envelope'];
  if (new Set(iterations.map(candidate => candidate.iteration)).size >= 2 && count <= 4096) {
    envelope = {min: [], max: []};
    for (let key = 0; key < count; key++) {
      const values = iterations
        .map(candidate => getCaptureValue(candidate, key))
        .filter((value): value is number => value !== undefined && Number.isFinite(value));
      envelope.min.push(values.length ? Math.min(...values) : NaN);
      envelope.max.push(values.length ? Math.max(...values) : NaN);
    }
  }
  const start = Math.max(
    0,
    Math.min(Math.max(0, count - 64), (getPinnedEntity(context) ?? 32) - 32)
  );
  const table = Array.from({length: Math.min(64, count - start)}, (_, offset) => {
    const key = start + offset;
    const values = Array.from({length: components}, (_, component) =>
      Number(capture.values[key * components + component])
    );
    const prior = previous && getCaptureValue(previous, key);
    return {
      key,
      value: components === 1 ? values[0] : values.join(', '),
      delta: prior === undefined ? undefined : values[0] - prior
    };
  });
  return {
    captureId: capture.id,
    name: capture.name,
    keySpace: capture.keySpace ?? 'unknown',
    indexSpace:
      keySpace?.indexSpace === '2d' ? '2d' : keySpace?.indexSpace === 'csr' ? 'csr' : '1d',
    shape: keySpace?.shape,
    count,
    elementType: capture.elementType,
    provenance: getCaptureProvenance(capture),
    stats,
    histogram: {bins: 16, counts, edges},
    envelope,
    diff,
    table
  };
}

function renderIndexSpace(
  model: BufferStripModel,
  capture: GPUComputationCapture,
  context: ViewContext
): ViewElementNode {
  const scale = resolveValueScale(context, model.name)?.scale;
  const values = Array.from({length: model.count}, (_, key) => getCaptureValue(capture, key)!);
  const provisional =
    capture.iteration !== undefined &&
    getFinalStateCapture(context.record, model.name)?.id !== capture.id
      ? 'provisional'
      : 'final';
  const width = context.width ?? 640;
  let tree: ViewElementNode;
  if (model.indexSpace === '2d' && model.shape?.[0] && model.shape[1]) {
    const [columnCount, rowCount] = model.shape;
    const pinned = getPinnedEntity(context);
    const marks = [
      renderRasterImage({
        columns: columnCount,
        rows: rowCount,
        limit: context.options?.['size'] === 'thumbnail' ? 64 : 160,
        valueAt: key => values[key],
        categorical: scale?.variableKind.level === 'nominal',
        colorFor: value =>
          Number.isFinite(value)
            ? (scale?.colorFor(value) ?? '#808080')
            : (scale?.noDataColor ?? '#808080'),
        attributes: {'data-provenance': model.provenance}
      })
    ];
    if (pinned !== undefined && pinned >= 0 && pinned < model.count)
      marks.push(
        h('rect', {
          x: pinned % columnCount,
          y: Math.floor(pinned / columnCount),
          width: 1,
          height: 1,
          fill: 'none',
          stroke: getSelectionInk(context.ground).core,
          'stroke-width': 0.2,
          'data-state': 'selection'
        })
      );
    tree = h(
      'svg',
      {
        class: 'lens-buffer-grid',
        viewBox: `0 0 ${columnCount} ${rowCount}`,
        width,
        height: Math.min(640, (width * rowCount) / columnCount),
        role: 'img',
        'aria-label': model.name
      },
      h('g', crispnessAttributes(provisional, context.idPrefix), marks)
    );
  } else {
    tree = renderValueStrip({
      values,
      scale,
      provenance: model.provenance,
      idPrefix: context.idPrefix,
      ground: context.ground,
      width,
      pinnedKey: getPinnedEntity(context),
      provisional,
      label: `${model.name} by ${model.keySpace}`
    });
    if (model.indexSpace === 'csr') {
      const offsets = getAllCaptures(context.record).find(
        candidate => candidate.name === `${model.name}-offsets`
      );
      if (offsets && model.count) {
        tree.children.push(
          h(
            'g',
            {class: 'lens-csr-brackets'},
            Array.from(offsets.values, offset =>
              h('path', {
                d: `M ${(offset / model.count) * width} 4 v 20 h 3`,
                ...provenanceMarkAttributes(
                  getCaptureProvenance(offsets),
                  'currentColor',
                  context.idPrefix
                )
              })
            )
          )
        );
      }
    }
  }
  tree.children.unshift(
    provenancePatternDefs(context.idPrefix, context.ground),
    crispnessDefs(context.idPrefix)
  );
  return tree;
}

/** Renders a bounded index-space view with histogram, envelope and typed table. */
export function renderBufferStrip(model: BufferStripModel, context: ViewContext): ViewElementNode {
  const capture = getAllCaptures(context.record).find(
    candidate => candidate.id === model.captureId
  );
  // The model names a capture from this context.
  assert(capture);
  const provisional =
    capture.iteration !== undefined &&
    getFinalStateCapture(context.record, model.name)?.id !== capture.id;
  const histogramMaximum = Math.max(1, ...model.histogram.counts);
  const envelope = model.envelope;
  const envelopeValues = envelope ? [...envelope.min, ...envelope.max].filter(Number.isFinite) : [];
  const minimum = envelopeValues.length ? Math.min(...envelopeValues) : 0;
  const maximum = envelopeValues.length ? Math.max(...envelopeValues) : 1;
  const envelopePath = (values: number[]) => {
    let restart = true;
    return values
      .map((value, index) => {
        if (!Number.isFinite(value)) {
          restart = true;
          return '';
        }
        const command = restart ? 'M' : 'L';
        restart = false;
        return `${command}${(index / Math.max(1, values.length - 1)) * 640},${70 - ((value - minimum) / (maximum - minimum || 1)) * 60}`;
      })
      .join(' ');
  };
  return h(
    'section',
    {class: 'lens-view lens-buffer-strip'},
    h(
      'header',
      null,
      h('h3', null, model.name),
      h('span', null, `${model.keySpace} · ${model.count} × ${model.elementType}`),
      provenanceBadge(model.provenance),
      model.stats.nanCount
        ? h('span', {class: 'lens-badge lens-badge--nan'}, `${model.stats.nanCount} NaN`)
        : null,
      model.stats.infCount
        ? h('span', {class: 'lens-badge lens-badge--inf'}, `${model.stats.infCount} Inf`)
        : null
    ),
    renderIndexSpace(model, capture, context),
    model.indexSpace === '2d' && !(model.shape?.[0] && model.shape[1])
      ? h('p', {class: 'lens-unknown'}, 'unknown — raster shape not captured; showing linear slots')
      : null,
    model.indexSpace === 'csr' &&
      !getAllCaptures(context.record).some(candidate => candidate.name === `${model.name}-offsets`)
      ? h('p', {class: 'lens-unknown'}, 'CSR offsets not captured')
      : null,
    h(
      'svg',
      {
        class: 'lens-buffer-histogram',
        viewBox: '0 0 320 84',
        width: 320,
        height: 84,
        role: 'img',
        'aria-label': '16-bin histogram'
      },
      provenancePatternDefs(`${context.idPrefix}-histogram`, context.ground),
      crispnessDefs(`${context.idPrefix}-histogram`),
      model.histogram.counts.map((count, index) =>
        h(
          'rect',
          {
            x: index * 20,
            y: 80 - (count / histogramMaximum) * 76,
            width: 18,
            height: (count / histogramMaximum) * 76,
            ...provenanceMarkAttributes('derived', 'currentColor', `${context.idPrefix}-histogram`)
          },
          h(
            'title',
            null,
            `${model.histogram.edges[index]}–${model.histogram.edges[index + 1]}: ${count}`
          )
        )
      )
    ),
    envelope
      ? h(
          'svg',
          {
            class: 'lens-buffer-envelope',
            viewBox: '0 0 640 80',
            width: context.width ?? 640,
            height: 80,
            role: 'img',
            'aria-label': 'Min/max per key across captured iterations'
          },
          provenancePatternDefs(`${context.idPrefix}-envelope`, context.ground),
          crispnessDefs(`${context.idPrefix}-envelope`),
          ['min', 'max'].map(bound =>
            h('path', {
              ...provenanceMarkAttributes(
                'derived',
                'currentColor',
                `${context.idPrefix}-envelope`
              ),
              fill: 'none',
              'stroke-dasharray': bound === 'min' ? '3 2' : undefined,
              d: envelopePath(bound === 'min' ? envelope.min : envelope.max)
            })
          )
        )
      : null,
    h(
      'table',
      {class: 'lens-buffer-table'},
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          ['key', `value (${model.elementType})`, 'delta', 'provenance'].map(label =>
            h('th', {scope: 'col'}, label)
          )
        )
      ),
      h(
        'tbody',
        null,
        model.table.map(row =>
          h(
            'tr',
            {'data-provisional': provisional ? 'provisional' : undefined},
            h(
              'td',
              null,
              h(
                'button',
                {type: 'button', 'data-action': 'pick', 'data-axis': 'entity', 'data-key': row.key},
                String(row.key)
              )
            ),
            h(
              'td',
              null,
              provisional
                ? coarsenValues(String(row.value).split(', ').map(Number), 'provisional').join(', ')
                : String(row.value)
            ),
            h(
              'td',
              null,
              row.delta === undefined
                ? '—'
                : String(provisional ? coarsenValues([row.delta], 'provisional')[0] : row.delta)
            ),
            h('td', null, provenanceBadge(model.provenance))
          )
        )
      )
    ),
    h(
      'p',
      null,
      `min ${model.stats.min} · max ${model.stats.max}${model.diff ? ` · ${model.diff.changedCount} keys changed vs ${model.diff.againstCaptureId}` : ''}`
    ),
    h('p', null, getAdapterLabel(context.record))
  );
}

/** Generic captured-buffer registration, after the pinned datum receipt. */
export const BUFFER_STRIP_VIEW: ViewRegistration<BufferStripModel> = {
  id: 'buffer-strip',
  label: 'Buffer',
  rung: 'datum',
  recordType: 'buffer',
  order: 70,
  signature: {binds: ['buffer'], free: ['slot'], layout: {slot: 'position'}},
  descriptor: {audience: 'expert', task: 'explore', interaction: 'high'},
  level: 'realized',
  grain: 'kernel',
  satisfies: context =>
    chooseCapture(context) ? {ok: true} : {ok: false, reason: 'no captured buffers in this record'},
  build: buildBufferStripModel,
  render: renderBufferStrip,
  describe: (model, context) => ({
    subject: `Buffer ${model.name} (${model.count} ${model.elementType})`,
    keyNumber:
      model.stats.nanCount > 0 && model.stats.finiteCount === 0 && model.stats.infCount === 0
        ? {label: 'NaN count', value: model.stats.nanCount}
        : {label: 'max', value: model.stats.max},
    facts: [
      `min ${model.stats.min}`,
      ...(model.shape?.length === 2
        ? [
            getRasterSamplingNote(
              model.shape[0],
              model.shape[1],
              context.options?.['size'] === 'thumbnail' ? 64 : 160
            )
          ]
        : []),
      `${model.stats.nanCount} NaN, ${model.stats.infCount} Inf`,
      ...(model.diff ? [`${model.diff.changedCount} keys changed vs previous iteration`] : [])
    ],
    provenance: model.provenance,
    adapterLabel: getAdapterLabel(context.record)
  })
};

let mountCounter = 0;

/** Mounts this view into a host-owned element; update replaces its tree and destroy clears it. */
export function mountBufferStrip(
  element: HTMLElement,
  record: LensRecord,
  options: Partial<Omit<ViewContext, 'record'>> = {}
): {update(record: LensRecord): void; destroy(): void} {
  const defaults = {
    bindings: {},
    ground: 'light' as const,
    idPrefix: `lens-buffer-strip-${++mountCounter}`,
    reducedMotion: false,
    ...options
  };
  const update = (nextRecord: LensRecord) => {
    const context: ViewContext = {...defaults, record: nextRecord};
    const satisfaction = BUFFER_STRIP_VIEW.satisfies(context);
    const tree = !('reason' in satisfaction)
      ? renderBufferStrip(buildBufferStripModel(context), context)
      : h('p', {class: 'lens-disabled'}, satisfaction.reason);
    element.replaceChildren(viewNodeToDom(tree, element.ownerDocument));
  };
  update(record);
  return {update, destroy: () => element.replaceChildren()};
}
