// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUComputationEncodingFlags, GPUComputationMetrics} from '@luma.gl/gpgpu';
import type {LensRecord, ViewContext, ViewRegistration} from '../registry/view-registration';
import {h, viewNodeToDom, type ViewElementNode} from '../view-node';
import {
  provenanceBadge,
  provenanceMarkAttributes,
  provenancePatternDefs
} from '../encoding/provenance';
import {
  getBoundNode,
  getGraph,
  getGraphByForm,
  getMetricsForGraph,
  getNamespaceSegments,
  getObserverEffects,
  getAdapterLabel,
  formatViewNumber
} from './record-queries';

type TimingSpan = {
  id: string;
  label: string;
  begin: number;
  end: number;
  nodeIds: string[];
  provenance: 'readback' | 'derived';
};

/** Timestamp tracks in display units, with encoding effects kept explicit. */
export type TimingViewModel = {
  graphId: string;
  label: string;
  timebase: 'ns' | 'ms';
  unitLabel: 'µs' | 'ms';
  encodingFlags: GPUComputationEncodingFlags;
  observerEffects: string[];
  tracks: {level: number; spans: TimingSpan[]}[];
  untimed: string[];
  total: {begin: number; end: number};
  passes?: {passIndex: number; nodeIds: string[]}[];
  cpuEncodeMsTotal: number;
};

function getTimingMetrics(context: ViewContext): GPUComputationMetrics | undefined {
  const bound = getBoundNode(context);
  const compiled = getGraphByForm(context.record, ['compiled']);
  return (
    (bound && getMetricsForGraph(context.record, bound.graph.id)) ??
    (compiled && getMetricsForGraph(context.record, compiled.id)) ??
    context.record.metrics?.[0]
  );
}

/** Derives ancestor namespace extents from recorded leaf timestamps. */
export function buildTimingViewModel(context: ViewContext): TimingViewModel {
  const metrics = getTimingMetrics(context);
  // The registration requires a timing sidecar.
  assert(metrics);
  const graph = getGraph(context.record, metrics.graph);
  const divisor = metrics.timebase === 'ns' ? 1000 : 1;
  const tracks = new Map<number, TimingSpan[]>();
  const groups = new Map<string, TimingSpan>();
  const passes = new Map<number, string[]>();
  const untimed: string[] = [];
  let cpuEncodeMsTotal = 0;
  const nodeIds = [
    ...new Set([...(graph?.nodes.map(node => node.id) ?? []), ...Object.keys(metrics.nodes)])
  ];
  for (const nodeId of nodeIds) {
    const node = graph?.nodes.find(candidate => candidate.id === nodeId);
    const measurement = metrics.nodes[nodeId];
    cpuEncodeMsTotal += measurement?.cpuEncodeMs ?? 0;
    if (measurement?.passIndex !== undefined) {
      const members = passes.get(measurement.passIndex) ?? [];
      members.push(nodeId);
      passes.set(measurement.passIndex, members);
    }
    if (measurement?.gpuBegin === undefined || measurement.gpuEnd === undefined) {
      untimed.push(nodeId);
      continue;
    }
    const namespace = getNamespaceSegments(node?.namespace ?? '');
    const span: TimingSpan = {
      id: nodeId,
      label: node?.label ?? nodeId,
      begin: measurement.gpuBegin / divisor,
      end: measurement.gpuEnd / divisor,
      nodeIds: [nodeId],
      provenance: 'readback'
    };
    const leaves = tracks.get(namespace.length) ?? [];
    leaves.push(span);
    tracks.set(namespace.length, leaves);
    for (let level = 0; level < namespace.length; level++) {
      const prefix = namespace.slice(0, level + 1).join('/');
      let group = groups.get(prefix);
      if (!group) {
        group = {
          id: `namespace:${prefix}`,
          label: prefix,
          begin: span.begin,
          end: span.end,
          nodeIds: [],
          provenance: 'derived'
        };
        groups.set(prefix, group);
        const ancestors = tracks.get(level) ?? [];
        ancestors.push(group);
        tracks.set(level, ancestors);
      }
      group.begin = Math.min(group.begin, span.begin);
      group.end = Math.max(group.end, span.end);
      group.nodeIds.push(nodeId);
    }
  }
  const orderedTracks = [...tracks]
    .sort(([first], [second]) => first - second)
    .map(([level, spans]) => ({level, spans}));
  const leaves = orderedTracks
    .flatMap(track => track.spans)
    .filter(span => span.provenance === 'readback');
  return {
    graphId: metrics.graph,
    label: graph?.label ?? metrics.graph,
    timebase: metrics.timebase,
    unitLabel: metrics.timebase === 'ns' ? 'µs' : 'ms',
    encodingFlags: metrics.encoding,
    observerEffects: getObserverEffects(context.record),
    tracks: orderedTracks,
    untimed,
    total: {
      begin: leaves.length ? Math.min(...leaves.map(span => span.begin)) : 0,
      end: leaves.length ? Math.max(...leaves.map(span => span.end)) : 0
    },
    passes: passes.size
      ? [...passes]
          .sort(([first], [second]) => first - second)
          .map(([passIndex, members]) => ({passIndex, nodeIds: members}))
      : undefined,
    cpuEncodeMsTotal
  };
}

/** Renders hierarchical timestamp tracks and the observer effects of this encoding. */
export function renderTimingView(model: TimingViewModel, context: ViewContext): ViewElementNode {
  const flags = model.encodingFlags;
  const width = Math.max(240, context.width ?? 640);
  const duration = model.total.end - model.total.begin;
  const position = (time: number) =>
    100 + ((time - model.total.begin) / (duration || 1)) * (width - 120);
  return h(
    'section',
    {class: 'lens-view lens-timing-view'},
    h('h3', null, `GPU timing · ${model.label}`),
    h(
      'p',
      {class: 'lens-encoding-flags'},
      `coalesced: ${flags.coalesced ? 'yes' : 'no'}${flags.timestamped ? ' (timestamps on)' : ''} · timestamped: ${flags.timestamped ? 'yes' : 'no'} · taps ${flags.taps} · lens twin: ${flags.lensTwin ? 'yes' : 'no'} · retained: ${flags.retained?.join(', ') || 'none'}`
    ),
    model.observerEffects.map(effect => h('p', {class: 'lens-observer-effect'}, `${effect}.`)),
    h(
      'figure',
      null,
      h(
        'svg',
        {
          viewBox: `0 0 ${width} ${model.tracks.length * 48 + 60}`,
          role: 'img',
          'aria-label': `GPU time in ${model.unitLabel}`
        },
        provenancePatternDefs(context.idPrefix, context.ground),
        model.tracks.map((track, index) =>
          h(
            'g',
            {'data-level': track.level},
            h('text', {x: 0, y: index * 48 + 20, fill: 'currentColor'}, `level ${track.level}`),
            track.spans.map(span =>
              h(
                'g',
                {'data-action': 'bind', 'data-axis': 'node', 'data-key': span.id},
                h(
                  'rect',
                  {
                    x: position(span.begin),
                    y: index * 48 + 4,
                    width: Math.max(1, position(span.end) - position(span.begin)),
                    height: 26,
                    ...provenanceMarkAttributes(span.provenance, 'currentColor', context.idPrefix)
                  },
                  h(
                    'title',
                    null,
                    `${span.label}: ${formatViewNumber(span.end - span.begin)} ${model.unitLabel}`
                  )
                ),
                h(
                  'text',
                  {x: position(span.begin), y: index * 48 + 43, fill: 'currentColor'},
                  span.label
                )
              )
            )
          )
        ),
        Array.from({length: 5}, (_, index) => {
          const time = model.total.begin + (duration * index) / 4;
          const baseline = model.tracks.length * 48 + 5;
          return h(
            'g',
            null,
            h('line', {
              x1: position(time),
              x2: position(time),
              y1: baseline,
              y2: baseline + 5,
              stroke: 'currentColor'
            }),
            h(
              'text',
              {x: position(time), y: baseline + 20, 'text-anchor': 'middle', fill: 'currentColor'},
              `${formatViewNumber(time)} ${model.unitLabel}`
            )
          );
        })
      ),
      h(
        'figcaption',
        null,
        `${getAdapterLabel(context.record)} · CPU encode ${formatViewNumber(model.cpuEncodeMsTotal)} ms${model.tracks.length ? '' : ' · GPU timestamps unknown'}`
      )
    ),
    h('p', null, `Untimed nodes: ${model.untimed.join(', ') || 'none'}`),
    model.passes &&
      h(
        'table',
        {class: 'lens-passes'},
        h('thead', null, h('tr', null, h('th', null, 'Pass index'), h('th', null, 'Nodes'))),
        h(
          'tbody',
          null,
          model.passes.map(pass =>
            h(
              'tr',
              null,
              h('td', null, String(pass.passIndex), provenanceBadge('metadata')),
              h('td', null, pass.nodeIds.join(', ') || '—')
            )
          )
        )
      )
  );
}

/** Generic execution timing view. */
export const TIMING_VIEW: ViewRegistration<TimingViewModel> = {
  id: 'timing-view',
  label: 'Timing',
  rung: 'timing',
  recordType: 'timing',
  order: 50,
  signature: {binds: ['run'], free: ['node', 'time'], layout: {node: 'position', time: 'position'}},
  descriptor: {audience: 'expert', task: 'explore', interaction: 'high'},
  level: 'execution',
  grain: 'contributor',
  satisfies: context => {
    const metrics = getTimingMetrics(context);
    return metrics &&
      Object.values(metrics.nodes).some(
        node =>
          (node.gpuBegin !== undefined && node.gpuEnd !== undefined) ||
          node.cpuEncodeMs !== undefined
      )
      ? {ok: true}
      : {ok: false, reason: 'no timing sidecar (timestamps need timestamp-query)'};
  },
  build: buildTimingViewModel,
  render: renderTimingView,
  describe: (model, context) => {
    const slowest = model.tracks
      .flatMap(track => track.spans)
      .filter(span => span.provenance === 'readback')
      .sort((first, second) => second.end - second.begin - (first.end - first.begin))[0];
    return {
      subject: `GPU timing of ${model.label}`,
      keyNumber: {
        label: 'GPU time',
        value: slowest ? model.total.end - model.total.begin : 'unknown',
        unit: model.unitLabel
      },
      facts: [
        slowest
          ? `slowest node ${slowest.label}: ${formatViewNumber(slowest.end - slowest.begin)} ${model.unitLabel}`
          : 'GPU timestamps unknown',
        `untimed nodes ${model.untimed.length}`
      ],
      caveats: model.observerEffects,
      provenance: 'readback',
      adapterLabel: getAdapterLabel(context.record)
    };
  }
};

let mountCount = 0;

/** Mounts the timing view in a host-owned element; no DOM is accessed until called. */
export function mountTimingView(
  element: HTMLElement,
  record: LensRecord,
  options: Partial<Omit<ViewContext, 'record'>> = {}
): {update(record: LensRecord): void; destroy(): void} {
  const defaults = {
    bindings: {},
    ground: 'light' as const,
    idPrefix: `lens-timing-${++mountCount}`,
    reducedMotion: false,
    ...options
  };
  let destroyed = false;
  const update = (nextRecord: LensRecord) => {
    if (destroyed) return;
    const context: ViewContext = {...defaults, record: nextRecord};
    const satisfaction = TIMING_VIEW.satisfies(context);
    const tree = !('reason' in satisfaction)
      ? renderTimingView(buildTimingViewModel(context), context)
      : h('p', {class: 'lens-disabled'}, satisfaction.reason);
    element.replaceChildren(viewNodeToDom(tree, element.ownerDocument));
  };
  update(record);
  return {
    update,
    destroy: () => {
      destroyed = true;
      element.replaceChildren();
    }
  };
}
