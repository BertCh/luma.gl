// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {renderRasterImage, getRasterSamplingNote} from '../encoding/raster-image';
import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import type {LensRecord, ViewContext, ViewRegistration} from '../registry/view-registration';
import type {ProvisionalState} from '../types';
import {h, viewNodeToDom, type ViewElementNode} from '../view-node';
import {
  provenanceBadge,
  provenanceMarkAttributes,
  provenancePatternDefs
} from '../encoding/provenance';
import {crispnessDefs, crispnessAttributes, coarsenValues} from '../encoding/crispness';
import {
  getAdapterLabel,
  getBoundIteration,
  getPinnedEntity,
  getDatumKeySpace,
  getCaptureValue,
  getCaptureProvenance,
  getMetricsForGraph,
  formatViewNumber
} from './record-queries';
import {resolveValueScale} from './value-strip';

/** Recorded loop evidence, convergence counters, and static key frames. */
export type RealizedViewModel = {
  loops: {
    loopId: string;
    label: string;
    maximumIterations?: number;
    executedIterations?: number;
    gated: boolean;
  }[];
  series: {name: string; unit?: string; values: number[]; logScale: boolean}[];
  keyFrames: {iteration: number; captureId: string; provisional: ProvisionalState}[];
  boundIteration?: number;
  pinned?: {
    entity: number;
    trajectory: {iteration: number; value: number}[];
    settledAt?: {value: number; meaning: string} | {unknown: string};
  };
  iterationRange: {min: number; max: number};
  caption: string;
};

function getIterationCaptures(context: ViewContext): GPUComputationCapture[] {
  const snapshots =
    context.record.captures?.snapshots.filter(capture => capture.iteration !== undefined) ?? [];
  const counts = new Map<string, Set<number>>();
  for (const capture of snapshots) {
    const iterations = counts.get(capture.name) ?? new Set<number>();
    iterations.add(capture.iteration!);
    counts.set(capture.name, iterations);
  }
  const name =
    typeof context.options?.['capture'] === 'string'
      ? context.options['capture']
      : [...counts].sort((first, second) => second[1].size - first[1].size)[0]?.[0];
  const byIteration = new Map<number, GPUComputationCapture>();
  for (const capture of snapshots)
    if (capture.name === name) byIteration.set(capture.iteration!, capture);
  return [...byIteration.values()].sort((first, second) => first.iteration! - second.iteration!);
}

/** Builds a static trace using recorded iteration evidence only. */
export function buildRealizedViewModel(context: ViewContext): RealizedViewModel {
  const loopMap = new Map<string, RealizedViewModel['loops'][number]>();
  const executions = new Map<string, Set<number>>();
  const measuredLoops = new Set<string>();
  for (const graph of context.record.graphs) {
    const metrics = getMetricsForGraph(context.record, graph.id);
    for (const node of graph.nodes) {
      if (!node.loop && node.kind !== 'loop') continue;
      const loopId = node.loop?.id ?? node.id;
      const loop = loopMap.get(loopId) ?? {
        loopId,
        label: node.dedupName ?? node.label,
        gated: false
      };
      loop.gated ||= node.loop?.phase === 'gated' || Boolean(node.condition);
      const maximum =
        node.loop?.maximumIterations ??
        (typeof node.attrs['maximumIterations'] === 'number'
          ? node.attrs['maximumIterations']
          : undefined);
      if (maximum !== undefined)
        loop.maximumIterations = Math.max(loop.maximumIterations ?? 0, maximum);
      if (metrics) {
        const iterations = executions.get(loopId) ?? new Set<number>();
        const measurement = metrics.nodes[node.id];
        if (node.loop && measurement) measuredLoops.add(loopId);
        if (
          node.loop &&
          measurement &&
          (measurement.outcome === 'executed' ||
            (measurement.outcome === undefined &&
              measurement.gpuBegin !== undefined &&
              measurement.gpuEnd !== undefined))
        )
          iterations.add(node.loop.iteration);
        executions.set(loopId, iterations);
      }
      loopMap.set(loopId, loop);
    }
  }
  // Without per-node measurements an empty set means "not measured", not "zero iterations".
  for (const [loopId, iterations] of executions)
    if (measuredLoops.has(loopId)) loopMap.get(loopId)!.executedIterations = iterations.size;
  // A contributor's own GPU iteration counter, read back as `executed iterations: <loop id>`,
  // is direct evidence and wins over counting measured nodes.
  for (const capture of [
    ...(context.record.captures?.snapshots ?? []),
    ...(context.record.captures?.stamps ?? [])
  ]) {
    const loopReference = capture.meaning?.match(/^executed iterations:\s*(.+)$/)?.[1]?.trim();
    const count = Number(capture.values[0]);
    if (!loopReference || !Number.isFinite(count)) continue;
    const loopId =
      [...loopMap.keys()].find(id => id === loopReference) ??
      [...loopMap.keys()].find(id => id.endsWith(loopReference)) ??
      loopReference;
    const loop = loopMap.get(loopId) ?? {loopId, label: loopReference, gated: true};
    loop.executedIterations = count;
    loopMap.set(loopId, loop);
  }
  const series = (context.record.metrics ?? [])
    .flatMap(metrics => metrics.series ?? [])
    .filter(item => item.axis === 'iteration')
    .map(item => {
      const values = Array.from(item.values);
      return {
        name: item.name,
        unit: item.unit,
        values,
        logScale:
          values.length > 0 &&
          values.every(value => Number.isFinite(value) && value > 0) &&
          Math.max(...values) / Math.min(...values) > 100
      };
    });
  const captures = getIterationCaptures(context);
  const last = captures.at(-1)?.iteration;
  const candidates = new Set([0, 1, 2]);
  for (let iteration = 4; last !== undefined && iteration <= last; iteration *= 2)
    candidates.add(iteration);
  const selected = captures
    .filter(capture => candidates.has(capture.iteration!) && capture.iteration !== last)
    .slice(0, 7);
  if (captures.length) selected.push(captures[captures.length - 1]);
  const keyFrames = selected.map(capture => ({
    iteration: capture.iteration!,
    captureId: capture.id,
    provisional: capture.iteration === last ? ('final' as const) : ('provisional' as const)
  }));
  const entity = getPinnedEntity(context);
  const keySpace = getDatumKeySpace(context);
  let pinned: RealizedViewModel['pinned'];
  if (entity !== undefined) {
    const stamp = context.record.captures?.stamps.find(
      capture =>
        keySpace !== undefined &&
        capture.keySpace === keySpace &&
        getCaptureValue(capture, entity) !== undefined
    );
    pinned = {
      entity,
      trajectory: captures
        .filter(capture => keySpace !== undefined && capture.keySpace === keySpace)
        .flatMap(capture => {
          const value = getCaptureValue(capture, entity);
          return value === undefined ? [] : [{iteration: capture.iteration!, value}];
        }),
      settledAt: stamp
        ? {value: getCaptureValue(stamp, entity)!, meaning: stamp.meaning ?? stamp.name}
        : {unknown: 'no stamp for the datum key space'}
    };
  }
  const iterations = [
    ...captures.map(capture => capture.iteration!),
    ...series.flatMap(item => item.values.map((_, index) => index)),
    ...[...executions.values()].flatMap(iterations => [...iterations])
  ];
  const chaotic =
    context.record.graphs.some(graph => graph.nodes.some(node => node.attrs['chaotic'] === true)) ||
    context.packs?.some(pack => pack.metadata?.['executionOrder'] === 'chaotic');
  return {
    loops: [...loopMap.values()],
    series,
    keyFrames,
    boundIteration: getBoundIteration(context),
    pinned,
    iterationRange: {
      min: iterations.length ? Math.min(...iterations) : 0,
      max: iterations.length ? Math.max(...iterations) : 0
    },
    caption: `${
      chaotic
        ? `Each frame is a valid execution prefix on ${getAdapterLabel(context.record)}, not the only possible one.`
        : `State after each recorded iteration on ${getAdapterLabel(context.record)}.`
    }${getDecimationNote(captures)} ${captures
      .slice(0, 1)
      .map(capture => {
        const shape =
          capture.geometry?.kind === 'raster'
            ? capture.geometry.shape
            : context.record.axes.keySpaces.find(space => space.id === capture.keySpace)?.shape;
        return getRasterSamplingNote(
          shape?.[0] ?? Math.ceil(capture.values.length / (capture.components ?? 1)),
          shape?.[1] ?? 1,
          96,
          resolveValueScale(context, capture.name)?.scale.variableKind.level === 'nominal'
        );
      })
      .join('')}`.trim()
  };
}

/** States that captured iterations are key frames when executed iterations between them are missing. */
function getDecimationNote(captures: readonly GPUComputationCapture[]): string {
  const iterations = [...new Set(captures.map(capture => capture.iteration!))].sort(
    (first, second) => first - second
  );
  if (iterations.length < 2 || iterations.at(-1)! - iterations[0] + 1 === iterations.length)
    return '';
  return ` Key frames only: iterations ${iterations.join(', ')} were captured; the iterations between them ran but were not read back.`;
}

/** Draws convergence in achromatic ink and key frames on one final-state value scale. */
export function renderRealizedView(
  model: RealizedViewModel,
  context: ViewContext
): ViewElementNode {
  const width = Math.max(240, context.width ?? 640);
  const plotWidth = width - 100;
  const range = model.iterationRange;
  const position = (iteration: number) =>
    60 + ((iteration - range.min) / (range.max - range.min || 1)) * plotWidth;
  const captures = getIterationCaptures(context);
  return h(
    'section',
    {class: 'lens-view lens-realized-view'},
    h('h3', null, model.loops.map(loop => loop.label).join(', ')),
    h('input', {
      type: 'range',
      class: 'lens-scrubber',
      min: range.min,
      max: range.max,
      step: 1,
      value: model.boundIteration ?? range.max,
      'data-action': 'scrub',
      'data-axis': 'iteration',
      'aria-label': 'Iteration'
    }),
    model.series.length > 0 &&
      h(
        'svg',
        {
          class: 'lens-convergence',
          viewBox: `0 0 ${width} ${Math.max(1, model.series.length) * 160 + 35}`,
          role: 'img',
          'aria-label': 'Convergence by iteration'
        },
        provenancePatternDefs(context.idPrefix, context.ground),
        model.series.map((series, seriesIndex) => {
          const transform = (value: number) => (series.logScale ? Math.log10(value) : value);
          const finite = series.values.filter(Number.isFinite).map(transform);
          const minimum = finite.length ? Math.min(...finite) : 0;
          const maximum = finite.length ? Math.max(...finite) : 1;
          const top = seriesIndex * 160 + 25;
          const ordinate = (value: number) =>
            top + 100 - ((transform(value) - minimum) / (maximum - minimum || 1)) * 100;
          let connected = false;
          const commands = series.values.map((value, iteration) => {
            if (!Number.isFinite(value)) {
              connected = false;
              return '';
            }
            const command = `${connected ? 'L' : 'M'} ${position(iteration)} ${ordinate(value)}`;
            connected = true;
            return command;
          });
          return h(
            'g',
            {'data-scale': series.logScale ? 'log' : 'linear'},
            h(
              'text',
              {x: 60, y: top - 10, fill: 'currentColor'},
              `${series.name}${series.unit ? ` (${series.unit})` : ''}${series.logScale ? ' · log y' : ''}`
            ),
            h('path', {
              d: commands.join(' '),
              ...provenanceMarkAttributes('readback', 'currentColor', context.idPrefix),
              fill: 'none',
              stroke: 'currentColor',
              'stroke-width': 1.5
            }),
            h('line', {
              x1: 60,
              x2: width - 40,
              y1: top + 100,
              y2: top + 100,
              stroke: 'currentColor',
              opacity: 0.55
            }),
            h(
              'text',
              {x: 4, y: top + 10, fill: 'currentColor'},
              formatViewNumber(series.logScale ? 10 ** maximum : maximum)
            ),
            h(
              'text',
              {x: 4, y: top + 100, fill: 'currentColor'},
              formatViewNumber(series.logScale ? 10 ** minimum : minimum)
            ),
            model.boundIteration !== undefined &&
              h('line', {
                x1: position(model.boundIteration),
                x2: position(model.boundIteration),
                y1: top,
                y2: top + 100,
                stroke: 'currentColor',
                'data-bound-iteration': model.boundIteration
              }),
            h('text', {x: 60, y: top + 120, fill: 'currentColor'}, String(range.min)),
            h('text', {x: width - 40, y: top + 120, fill: 'currentColor'}, String(range.max))
          );
        }),
        h(
          'text',
          {x: width / 2, y: Math.max(1, model.series.length) * 160 + 20, fill: 'currentColor'},
          'iteration'
        )
      ),
    model.keyFrames.length > 0 &&
      h(
        'div',
        {class: 'lens-key-frames'},
        model.keyFrames.map(frame => {
          const capture = captures.find(candidate => candidate.id === frame.captureId);
          if (!capture) return h('p', {class: 'lens-unknown'}, 'unknown — capture missing');
          const idPrefix = `${context.idPrefix}-frame-${frame.iteration}`;
          const values = Array.from(
            {length: Math.ceil(capture.values.length / (capture.components ?? 1))},
            (_, index) => getCaptureValue(capture, index) ?? NaN
          );
          const shape =
            capture.geometry?.kind === 'raster'
              ? capture.geometry.shape
              : context.record.axes.keySpaces.find(space => space.id === capture.keySpace)?.shape;
          const columns = shape?.[0] ?? Math.max(1, values.length);
          const rows = shape?.[1] ?? 1;
          const scale = resolveValueScale(context, capture.name)?.scale;
          const strip = h(
            'svg',
            {
              class: 'lens-value-strip',
              viewBox: `0 0 ${columns} ${rows}`,
              width: 96,
              height: shape ? 96 : 28,
              preserveAspectRatio: shape ? 'xMidYMid meet' : 'none',
              role: 'img',
              'aria-label': `iteration ${frame.iteration}`
            },
            renderRasterImage({
              columns,
              rows,
              limit: 96,
              valueAt: key => values[key],
              categorical: scale?.variableKind.level === 'nominal',
              colorFor: value =>
                Number.isFinite(value)
                  ? (scale?.colorFor(value) ?? '#808080')
                  : (scale?.noDataColor ?? '#d9d9d9'),
              attributes: {
                'data-provenance': getCaptureProvenance(capture),
                ...crispnessAttributes(frame.provisional, idPrefix)
              }
            })
          );
          const pinned =
            capture.keySpace === getDatumKeySpace(context) ? model.pinned?.entity : undefined;
          if (pinned !== undefined && pinned < values.length)
            strip.children.push(
              h('rect', {
                x: pinned % columns,
                y: Math.floor(pinned / columns),
                width: 1,
                height: 1,
                fill: 'none',
                stroke: 'currentColor',
                'stroke-width': 0.2,
                'data-state': 'selection'
              })
            );
          strip.children.unshift(
            provenancePatternDefs(idPrefix, context.ground),
            crispnessDefs(idPrefix)
          );
          return h(
            'figure',
            {'data-iteration': frame.iteration},
            h(
              'figcaption',
              null,
              `iteration ${frame.iteration}`,
              provenanceBadge(getCaptureProvenance(capture))
            ),
            strip
          );
        })
      ),
    model.pinned &&
      model.pinned.trajectory.length > 0 &&
      h(
        'table',
        {class: 'lens-pinned-trajectory'},
        h('caption', null, `Entity ${model.pinned.entity} trajectory`),
        h('thead', null, h('tr', null, h('th', null, 'Iteration'), h('th', null, 'Value'))),
        h(
          'tbody',
          null,
          model.pinned.trajectory.map(point =>
            h(
              'tr',
              null,
              h(
                'td',
                null,
                String(point.iteration),
                provenanceBadge(
                  getCaptureProvenance(
                    captures.find(capture => capture.iteration === point.iteration)!
                  )
                )
              ),
              h(
                'td',
                null,
                String(
                  coarsenValues(
                    [point.value],
                    point.iteration === captures.at(-1)?.iteration ? 'final' : 'provisional'
                  )[0]
                )
              )
            )
          )
        )
      ),
    model.pinned?.settledAt &&
      h(
        'p',
        {class: 'unknown' in model.pinned.settledAt ? 'lens-unknown' : undefined},
        'unknown' in model.pinned.settledAt
          ? `unknown — ${model.pinned.settledAt.unknown}`
          : `${model.pinned.settledAt.meaning}: ${model.pinned.settledAt.value}`
      ),
    h('p', {class: 'lens-caption'}, model.caption)
  );
}

/** Generic realized-iteration view. */
export const REALIZED_VIEW: ViewRegistration<RealizedViewModel> = {
  id: 'realized-view',
  label: 'Realized iterations',
  rung: 'iterations',
  recordType: 'loop',
  order: 50,
  signature: {binds: [], free: ['iteration'], layout: {iteration: 'small-multiple'}},
  descriptor: {audience: 'analyst', task: 'explore', interaction: 'medium'},
  level: 'realized',
  grain: 'contributor',
  satisfies: context => {
    const record = context.record;
    const hasLoop = record.graphs.some(graph =>
      graph.nodes.some(node => node.loop || node.kind === 'loop')
    );
    const hasTrace =
      record.metrics?.some(metrics =>
        metrics.series?.some(series => series.axis === 'iteration')
      ) ||
      record.captures?.snapshots.some(capture => capture.iteration !== undefined) ||
      Boolean(record.captures?.stamps.length);
    return hasLoop && hasTrace
      ? {ok: true}
      : {
          ok: false,
          reason:
            'no realized iterations recorded (needs a loop trace: series, per-iteration snapshots or stamps)'
        };
  },
  build: buildRealizedViewModel,
  render: renderRealizedView,
  describe: (model, context) => ({
    subject: `Realized iterations of ${model.loops.map(loop => loop.label).join(', ')}`,
    keyNumber: {
      label: 'iterations executed',
      value: model.loops.some(loop => loop.executedIterations !== undefined)
        ? model.loops.reduce((sum, loop) => sum + (loop.executedIterations ?? 0), 0)
        : model.keyFrames.length
    },
    facts: [
      ...model.series.map(
        series =>
          `${series.name}: ${series.values.length ? formatViewNumber(series.values[series.values.length - 1]) : 'unknown'}`
      ),
      ...(model.pinned?.settledAt
        ? [
            'unknown' in model.pinned.settledAt
              ? `settle iteration unknown: ${model.pinned.settledAt.unknown}`
              : `${model.pinned.settledAt.meaning}: ${model.pinned.settledAt.value}`
          ]
        : [])
    ],
    provenance: 'readback',
    adapterLabel: getAdapterLabel(context.record)
  })
};

let mountCount = 0;

/** Mounts the realized view in a host-owned element; no DOM is accessed until called. */
export function mountRealizedView(
  element: HTMLElement,
  record: LensRecord,
  options: Partial<Omit<ViewContext, 'record'>> = {}
): {update(record: LensRecord): void; destroy(): void} {
  const defaults = {
    bindings: {},
    ground: 'light' as const,
    idPrefix: `lens-realized-${++mountCount}`,
    reducedMotion: false,
    ...options
  };
  let destroyed = false;
  const update = (nextRecord: LensRecord) => {
    if (destroyed) return;
    const context: ViewContext = {...defaults, record: nextRecord};
    const satisfaction = REALIZED_VIEW.satisfies(context);
    const tree = !('reason' in satisfaction)
      ? renderRealizedView(buildRealizedViewModel(context), context)
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
