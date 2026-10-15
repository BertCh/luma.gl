// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUComputationBinding,
  GPUComputationNode,
  GPUComputationSourceRef
} from '@luma.gl/gpgpu';
import type {LensRecord} from '../registry/view-registration';
import {h, type ViewElementNode} from '../view-node';
import {provenanceBadge} from '../encoding/provenance';
import {getSource} from './record-queries';

/** Retained kernel source and its metadata, or an explicit unavailable reason. */
export type KernelSourceModel =
  | {
      sourceId: string;
      language: GPUComputationSourceRef['language'];
      moduleId: string;
      entryPoint?: string;
      lines: {
        number: number;
        text: string;
        marks: ('binding' | 'workgroup-size' | 'entry-point')[];
      }[];
      truncatedLineCount: number;
      bindings: GPUComputationBinding[];
      constants?: Readonly<Record<string, number>>;
      hashLabel: string;
    }
  | {status: 'unavailable'; reason: string};

/** Projects retained source without inventing code from a hash. */
export function buildKernelSourceModel(
  record: LensRecord,
  node: GPUComputationNode
): KernelSourceModel {
  const source = getSource(record, node.source);
  if (!source) return {status: 'unavailable', reason: 'kernel source not recorded'};
  if (source.text === undefined) {
    return {status: 'unavailable', reason: `source text not retained (only hash ${source.hash})`};
  }
  const lines = source.text.split(/\r?\n/);
  return {
    sourceId: source.id,
    language: source.language,
    moduleId: source.moduleId,
    entryPoint: source.entryPoint,
    hashLabel: source.hash,
    constants: source.constants,
    bindings: [...(node.bindings ?? [])],
    truncatedLineCount: Math.max(0, lines.length - 200),
    lines: lines.slice(0, 200).map((text, index) => {
      const marks: ('binding' | 'workgroup-size' | 'entry-point')[] = [];
      const words: string[] = text.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      if (
        /\bvar\s*<\s*(storage|uniform)\b/.test(text) &&
        node.bindings?.some(binding => words.includes(binding.name))
      )
        marks.push('binding');
      if (text.includes('@workgroup_size')) marks.push('workgroup-size');
      if (
        source.entryPoint &&
        text.match(/\bfn\s+([A-Za-z_][A-Za-z_0-9]*)/)?.[1] === source.entryPoint
      )
        marks.push('entry-point');
      return {number: index + 1, text, marks};
    })
  };
}

/** Renders escaped code lines and resource links for the source inspector. */
export function renderKernelSource(model: KernelSourceModel): ViewElementNode {
  if ('status' in model)
    return h(
      'figure',
      {class: 'lens-kernel-source'},
      h('span', {class: 'lens-unknown'}, model.reason)
    );
  return h(
    'figure',
    {class: 'lens-kernel-source'},
    h(
      'pre',
      null,
      h(
        'code',
        {'data-language': model.language},
        model.lines.map(line =>
          h(
            'span',
            {
              class: [
                'lens-source-line',
                ...line.marks.map(mark => `lens-source-line--${mark}`)
              ].join(' '),
              'data-line': line.number
            },
            line.text,
            '\n'
          )
        )
      )
    ),
    model.truncatedLineCount > 0 &&
      h('p', null, `${model.truncatedLineCount} source lines omitted`),
    h(
      'table',
      null,
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          ['Name', 'Resource', 'Access', 'Element type'].map(label => h('th', null, label))
        )
      ),
      h(
        'tbody',
        null,
        model.bindings.map(binding =>
          h(
            'tr',
            null,
            h('td', null, binding.name || '—', provenanceBadge('metadata')),
            h(
              'td',
              {'data-action': 'bind', 'data-axis': 'buffer', 'data-key': binding.resource},
              binding.resource || '—'
            ),
            h('td', null, binding.access),
            h('td', null, binding.elementType ?? '—')
          )
        )
      )
    ),
    h(
      'figcaption',
      null,
      `${model.moduleId} · ${model.entryPoint ?? 'entry point unknown'} · ${model.hashLabel}`
    )
  );
}
