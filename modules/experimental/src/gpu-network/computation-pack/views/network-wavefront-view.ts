// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import type {
  ViewContext,
  ViewRegistration
} from '../../../computation-views/registry/view-registration';
import {h, type ViewElementNode} from '../../../computation-views/view-node';
import {
  getAllCaptures,
  getAdapterLabel,
  getGraphByForm,
  getMetricsForGraph,
  getCaptureProvenance
} from '../../../computation-views/views/record-queries';
import {
  createGeometryProjection,
  type GeometryProjection
} from '../../../computation-views/views/geometry-projection';
import {
  CARD_COLORS,
  getCardMarkAttributes,
  getCardPaletteColors
} from '../../../computation-views/encoding/card-tokens';
import {encodeBase64, encodePNG} from '../../../computation-views/encoding/png';
import {
  PROVENANCE_HATCH_STROKE,
  provenanceMarkAttributes,
  provenancePatternDefs
} from '../../../computation-views/encoding/provenance';

function getEvidence(context: ViewContext) {
  const captures = getAllCaptures(context.record);
  const roadGraph = captures.find(
    capture => capture.role === 'context' && capture.geometry?.kind === 'graph'
  );
  const geometry = roadGraph?.geometry;
  const resolve = (reference?: string) =>
    captures.find(capture => capture.id === reference || capture.name === reference);
  const layout = geometry?.kind === 'graph' ? resolve(geometry.layoutCapture) : undefined;
  const offsets = geometry?.kind === 'graph' ? resolve(geometry.offsetsCapture) : undefined;
  const targets = geometry?.kind === 'graph' ? resolve(geometry.targetsCapture) : undefined;
  const groups = new Map<string, GPUComputationCapture[]>();
  for (const capture of captures) {
    if (
      capture.keySpace !== 'node' ||
      capture.iteration === undefined ||
      capture.provenance !== 'gpu-readback' ||
      (capture.components && capture.components !== 1)
    )
      continue;
    const group = groups.get(capture.name) ?? [];
    group.push(capture);
    groups.set(capture.name, group);
  }
  const snapshots = [...groups.values()].sort((left, right) => right.length - left.length)[0] ?? [];
  snapshots.sort((left, right) => left.iteration! - right.iteration!);
  const replays = captures.filter(
    capture =>
      capture.provenance === 'gpu-replay' &&
      capture.name === snapshots[0]?.name &&
      capture.iteration !== undefined
  );
  return {layout, offsets, targets, snapshots, replays, geometry};
}

function sampleEvenly<Value>(values: Value[], limit: number): Value[] {
  return values.length <= limit
    ? values
    : Array.from(
        {length: limit},
        (_, index) => values[Math.round((index * (values.length - 1)) / (limit - 1))]
      );
}

/** Recorded wavefronts in native network coordinates, with first-finite round colouring. */
export type NetworkWavefrontModel = {
  points: [number, number][];
  edges: [number, number][];
  arrivals: number[];
  maximumRound: number;
  frames: GPUComputationCapture[];
  snapshotCount: number;
  replays: GPUComputationCapture[];
  frontier: number[];
  projection: GeometryProjection;
  caption: string;
};

/** Builds geometry and arrival rounds only from captured buffers; never simulates the search. */
export function buildNetworkWavefrontModel(context: ViewContext): NetworkWavefrontModel {
  const {layout, offsets, targets, snapshots, replays, geometry} = getEvidence(context);
  // satisfies requires the layout, CSR and at least two iteration snapshots.
  assert(layout && offsets && targets && snapshots.length >= 2);
  const points: [number, number][] = Array.from({length: layout.values.length / 2}, (_, index) => [
    layout.values[2 * index],
    layout.values[2 * index + 1]
  ]);
  const edges: [number, number][] = [];
  for (let node = 0; node < points.length; node++) {
    for (
      let edge = offsets.values[node];
      edge < Math.min(offsets.values[node + 1], targets.values.length);
      edge++
    ) {
      const target = targets.values[edge];
      if (target < points.length) edges.push([node, target]);
    }
  }
  const arrivals = points.map(
    (_, node) =>
      snapshots.find(capture => Number.isFinite(capture.values[node]))?.iteration ?? Infinity
  );
  const graph = getGraphByForm(context.record);
  const frontier = graph
    ? Array.from(
        getMetricsForGraph(context.record, graph.id)?.series?.find(
          series => series.name === 'frontier-size' && series.axis === 'iteration'
        )?.values ?? []
      )
    : [];
  const chaotic = graph?.nodes.some(node => node.attrs['chaotic'] === true);
  return {
    points,
    edges,
    arrivals,
    maximumRound: snapshots.at(-1)!.iteration!,
    frames: sampleEvenly(snapshots, 8),
    snapshotCount: snapshots.length,
    replays,
    frontier,
    projection: createGeometryProjection({
      coordinateModel: geometry?.coordinateModel ?? 'unknown',
      points,
      width: 640,
      height: 360,
      padding: 24
    }),
    caption: chaotic
      ? `Each frame is a valid execution prefix on ${getAdapterLabel(context.record)}, not the only possible one.`
      : `Recorded wavefront on ${getAdapterLabel(context.record)}.`
  };
}

function renderFrame(
  model: NetworkWavefrontModel,
  capture: GPUComputationCapture,
  context: ViewContext
): ViewElementNode {
  if (model.points.length + model.edges.length > 32) {
    return renderRasterFrame(model, capture, context);
  }
  return h(
    'g',
    null,
    model.edges.map(([source, target]) => {
      const [firstX, firstY] = model.projection.project(...model.points[source]);
      const [lastX, lastY] = model.projection.project(...model.points[target]);
      return h('line', {
        ...getCardMarkAttributes('context', 'line'),
        ...provenanceMarkAttributes('readback', 'none', context.idPrefix),
        x1: firstX,
        y1: firstY,
        x2: lastX,
        y2: lastY
      });
    }),
    model.points.map((point, node) => {
      const reached = Number.isFinite(capture.values[node]);
      const arrival = model.arrivals[node];
      const attributes = getCardMarkAttributes(
        reached ? 'output' : 'context',
        'point',
        reached
          ? {
              palette: 'time',
              classIndex: Math.min(4, Math.floor((arrival / Math.max(1, model.maximumRound)) * 5))
            }
          : {}
      );
      const [horizontal, vertical] = model.projection.project(...point);
      const circle = h('circle', {
        ...attributes,
        ...provenanceMarkAttributes(
          getCaptureProvenance(capture),
          attributes['fill'],
          context.idPrefix
        ),
        cx: horizontal,
        cy: vertical,
        ...(reached ? {'data-action': 'pick', 'data-axis': 'entity', 'data-key': node} : {})
      });
      // Overlay the hatch without value classes, which would override its pattern fill in CSS.
      return capture.provenance === 'gpu-replay' && reached
        ? h(
            'g',
            null,
            h('circle', {...attributes, cx: horizontal, cy: vertical}),
            h('circle', {...circle.attributes, class: 'cv-replay-hatch'})
          )
        : circle;
    })
  );
}

function getSamplingNote(model: NetworkWavefrontModel, context: ViewContext): string {
  const thumbnail = context.options?.['size'] === 'thumbnail';
  return [
    thumbnail
      ? 'Final recorded frame only.'
      : `${model.frames.length} of ${model.snapshotCount} recorded frames, evenly sampled with first and final retained.`,
    model.points.length + model.edges.length > 32
      ? `All ${model.points.length} nodes and ${model.edges.length} edges rasterized at ${thumbnail ? '96×54' : '160×90 (key frames 96×54)'}; overlapping pixels use draw order. Node picking decimated to at most 8 evenly sampled keys plus the pinned key.`
      : 'All nodes and edges shown as vectors.',
    !thumbnail && model.frontier.length > 256
      ? `Frontier sparkline decimated to 256 evenly sampled rounds, including first and final.`
      : ''
  ]
    .filter(Boolean)
    .join(' ');
}

// Only rasterizes recorded geometry and values; this is display work, not a search replay.
function renderRasterFrame(
  model: NetworkWavefrontModel,
  capture: GPUComputationCapture,
  context: ViewContext
): ViewElementNode {
  const width = context.options?.['size'] === 'thumbnail' ? 96 : 160;
  const height = (width * 9) / 16;
  const colors = CARD_COLORS[context.ground];
  const palette = [
    colors['paper'],
    colors['context-line'],
    colors['context-fill'],
    ...getCardPaletteColors('time', context.ground),
    PROVENANCE_HATCH_STROKE[context.ground]
  ];
  const pixels = new Uint8Array(width * height);
  const points = model.points.map(point => {
    const [horizontal, vertical] = model.projection.project(...point);
    return [Math.round((horizontal * width) / 640), Math.round((vertical * height) / 360)];
  });
  const paintPixel = (horizontal: number, vertical: number, color: number) => {
    if (horizontal >= 0 && horizontal < width && vertical >= 0 && vertical < height) {
      pixels[vertical * width + horizontal] = color;
    }
  };
  for (const [source, target] of model.edges) {
    const [firstX, firstY] = points[source];
    const [lastX, lastY] = points[target];
    const steps = Math.max(Math.abs(lastX - firstX), Math.abs(lastY - firstY), 1);
    if (!Number.isFinite(steps)) continue;
    for (let step = 0; step <= steps; step++) {
      paintPixel(
        Math.round(firstX + ((lastX - firstX) * step) / steps),
        Math.round(firstY + ((lastY - firstY) * step) / steps),
        1
      );
    }
  }
  // Paint reached nodes last so context cannot cover the recorded wavefront.
  for (const reached of [false, true]) {
    points.forEach(([horizontal, vertical], node) => {
      if (Number.isFinite(capture.values[node]) !== reached) return;
      const color = reached
        ? 3 + Math.min(4, Math.floor((model.arrivals[node] / Math.max(1, model.maximumRound)) * 5))
        : 2;
      for (let offsetY = -1; offsetY <= 1; offsetY++) {
        for (let offsetX = -1; offsetX <= 1; offsetX++) {
          const hatched =
            reached &&
            capture.provenance === 'gpu-replay' &&
            (horizontal + offsetX + vertical + offsetY) % 4 === 0;
          paintPixel(horizontal + offsetX, vertical + offsetY, hatched ? 8 : color);
        }
      }
    });
  }
  const pickKeys = new Set(
    sampleEvenly(
      model.points.map((_, node) => node),
      8
    )
  );
  const pinned = context.bindings['entity'];
  if (typeof pinned === 'number' && model.points[pinned]) pickKeys.add(pinned);
  return h(
    'g',
    null,
    h('image', {
      width: 640,
      height: 360,
      'image-rendering': 'pixelated',
      'data-raster-width': width,
      'data-raster-height': height,
      'data-provenance': getCaptureProvenance(capture),
      class: capture.provenance === 'gpu-replay' ? 'cv-replay-hatch' : undefined,
      href: `data:image/png;base64,${encodeBase64(encodePNG(width, height, pixels, palette))}`
    }),
    [...pickKeys]
      .filter(node => Number.isFinite(capture.values[node]))
      .map(node => {
        const [horizontal, vertical] = model.projection.project(...model.points[node]);
        return h('circle', {
          cx: horizontal,
          cy: vertical,
          r: 4,
          fill: 'transparent',
          'data-provenance': getCaptureProvenance(capture),
          'data-action': 'pick',
          'data-axis': 'entity',
          'data-key': node
        });
      })
  );
}

/** Renders static final state, recorded frames, replay evidence and a frontier sparkline. */
export function renderNetworkWavefront(
  model: NetworkWavefrontModel,
  context: ViewContext
): ViewElementNode {
  const thumbnail = context.options?.['size'] === 'thumbnail';
  const frames = thumbnail ? model.frames.slice(-1) : model.frames;
  const frameLabel = (capture: GPUComputationCapture) =>
    capture.provenance === 'gpu-replay'
      ? `replay · round ${capture.iteration}`
      : capture.iteration === model.maximumRound
        ? 'final'
        : `round ${capture.iteration}`;
  const frontierMaximum = model.frontier.reduce((maximum, value) => Math.max(maximum, value), 0);
  const positiveMinimum = model.frontier.reduce(
    (minimum, value) => (value > 0 ? Math.min(minimum, value) : minimum),
    Infinity
  );
  const logarithmic = frontierMaximum / positiveMinimum > 100;
  const transform = (value: number) => (logarithmic ? Math.log1p(value) : value);
  const sparkline = sampleEvenly(
    model.frontier.map((value, index) => ({value, index})),
    256
  )
    .map(
      ({value, index}) =>
        `${index ? 'L' : 'M'} ${(8 + (index / Math.max(1, model.frontier.length - 1)) * 500).toFixed(1)} ${(38 - (transform(value) / Math.max(1, transform(frontierMaximum))) * 30).toFixed(1)}`
    )
    .join(' ');
  return h(
    'section',
    {class: 'lens-view lens-network-wavefront'},
    h(
      'svg',
      {
        class: 'cv-figure',
        viewBox: '0 0 640 360',
        role: 'img',
        'aria-label': 'Recorded network wavefront'
      },
      provenancePatternDefs(context.idPrefix, context.ground),
      frames.map((capture, index) =>
        h(
          'g',
          {
            class: 'cv-frame',
            'data-cv-frame': index,
            'data-iteration': capture.iteration,
            hidden: index !== frames.length - 1
          },
          renderFrame(model, capture, context)
        )
      )
    ),
    !thumbnail &&
      h(
        'button',
        {type: 'button', class: 'cv-play', 'data-cv-play': true},
        context.reducedMotion ? 'Step ▸' : '▶ Play'
      ),
    h('p', {'aria-live': 'polite'}, `round ${model.maximumRound} of ${model.maximumRound}`),
    !thumbnail &&
      h(
        'div',
        {class: 'cv-key-frames'},
        sampleEvenly(model.frames, model.replays.length ? 5 : 6)
          .concat(model.replays.slice(0, 1))
          .map((capture, index) => {
            const thumbnailContext = {
              ...context,
              idPrefix: `${context.idPrefix}-thumbnail-${index}`,
              options: {...context.options, size: 'thumbnail'}
            };
            return h(
              'figure',
              null,
              h(
                'svg',
                {viewBox: '0 0 640 360', width: 96, role: 'img', 'aria-label': frameLabel(capture)},
                provenancePatternDefs(thumbnailContext.idPrefix, context.ground),
                renderFrame(model, capture, thumbnailContext)
              ),
              h('figcaption', null, frameLabel(capture))
            );
          })
      ),
    !thumbnail &&
      model.frontier.length > 0 &&
      h(
        'svg',
        {
          class: 'cv-frontier-sparkline',
          viewBox: '0 0 640 48',
          role: 'img',
          'aria-label': `Frontier size in nodes${logarithmic ? ' (log1p scale; zero retained)' : ''}`,
          'data-scale': logarithmic ? 'log1p' : 'linear'
        },
        h('path', {
          d: sparkline,
          fill: 'none',
          stroke: 'currentColor',
          'data-provenance': 'readback'
        }),
        h('text', {x: 520, y: 30, fill: 'currentColor'}, `${model.frontier.at(-1)} nodes`)
      ),
    h('p', null, model.caption),
    h('p', {class: 'cv-note'}, getSamplingNote(model, context)),
    h('p', null, model.projection.statement),
    h(
      'p',
      null,
      'Colour: first finite recorded round. Step capture splits queue submissions; replay reruns the compiled graph with a round limit.'
    )
  );
}

/** Domain-specific iterations view; generic RealizedView remains the fallback for end-only runs. */
export const NETWORK_WAVEFRONT_VIEW: ViewRegistration<NetworkWavefrontModel> = {
  id: 'network-wavefront-view',
  label: 'Network wavefront',
  rung: 'iterations',
  recordType: 'loop',
  order: 10,
  signature: {
    binds: [],
    free: ['entity', 'iteration'],
    layout: {entity: 'position', iteration: 'animate'}
  },
  descriptor: {audience: 'analyst', task: 'explore', interaction: 'medium'},
  level: 'realized',
  grain: 'contributor',
  satisfies: context => {
    const evidence = getEvidence(context);
    if (!evidence.layout)
      return {ok: false, reason: 'needs a context graph with a captured node layout'};
    if (!evidence.offsets || !evidence.targets)
      return {ok: false, reason: 'needs context graph CSR offsets and targets captures'};
    if (new Set(evidence.snapshots.map(capture => capture.iteration)).size < 2)
      return {
        ok: false,
        reason: 'needs at least two iteration snapshots of one capture on key space node'
      };
    return {ok: true};
  },
  build: buildNetworkWavefrontModel,
  render: renderNetworkWavefront,
  describe: (model, context) => ({
    subject: 'Network reachability wavefront',
    keyNumber: {label: 'reached nodes', value: model.arrivals.filter(Number.isFinite).length},
    facts: [
      context.options?.['size'] === 'thumbnail'
        ? 'Final recorded frame'
        : `${model.frames.length} recorded key frames`,
      'Colour encodes first finite round'
    ],
    provenance: 'readback',
    caveats: [
      model.caption,
      getSamplingNote(model, context),
      'Step capture splits queue submissions',
      ...(model.replays.length
        ? ['Hatched replay re-runs the graph with activeIterations limited']
        : [])
    ],
    adapterLabel: getAdapterLabel(context.record)
  })
};
