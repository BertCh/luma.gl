// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {MarkProvenance, ProvenanceTexture, ViewGround} from '../types';
import {h, type ViewElementNode} from '../view-node';
import {hexToRgba} from '../scales/class-table';

/** Texture is lineage; it does not alter value lightness or provisional crispness. */
export const PROVENANCE_TEXTURE: Record<MarkProvenance, ProvenanceTexture> = {
  readback: 'solid',
  replay: 'hatched',
  derived: 'hatched',
  metadata: 'outline',
  unknown: 'outline'
};
/** Plain-language lineage labels, including an explicit unknown state. */
export const PROVENANCE_LABEL: Record<MarkProvenance, string> = {
  readback: 'GPU readback',
  replay: 'GPU replay',
  derived: 'derived',
  metadata: 'metadata',
  unknown: 'unknown'
};
/** Reference grounds for contrast checks; dark matches the copied selection halo. */
export const VIEW_GROUND_COLOR: Record<ViewGround, string> = {light: '#ffffff', dark: '#13171c'};
/** Neutral hatch strokes with at least 3:1 contrast against each reference ground. */
export const PROVENANCE_HATCH_STROKE: Record<ViewGround, string> = {
  light: '#595959',
  dark: '#b3b3b3'
};

/** WCAG contrast of opaque sRGB hex colours, using relative luminance. */
export function contrastRatio(hexA: string, hexB: string): number {
  const luminance = (hex: string) => {
    const channels = hexToRgba(hex)
      .slice(0, 3)
      .map(channel => {
        const normalized = channel / 255;
        return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
      });
    return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
  };
  const first = luminance(hexA);
  const second = luminance(hexB);
  return (Math.max(first, second) + 0.05) / (Math.min(first, second) + 0.05);
}

/** SVG definitions for a ground-contrasting 45-degree lineage hatch. */
export function provenancePatternDefs(idPrefix: string, ground: ViewGround): ViewElementNode {
  return h(
    'defs',
    null,
    h(
      'pattern',
      {
        id: `${idPrefix}-hatched`,
        patternUnits: 'userSpaceOnUse',
        width: 6,
        height: 6,
        patternTransform: 'rotate(45)'
      },
      h('line', {
        x1: 0,
        y1: 0,
        x2: 0,
        y2: 6,
        stroke: PROVENANCE_HATCH_STROKE[ground],
        'stroke-width': 1
      })
    )
  );
}

/** Mark attributes preserving a solid value-coloured edge around replay/derived hatching. */
export function provenanceMarkAttributes(
  provenance: MarkProvenance,
  fillHex: string,
  idPrefix: string
): Record<string, string> {
  const attributes: Record<string, string> = {'data-provenance': provenance};
  switch (PROVENANCE_TEXTURE[provenance]) {
    case 'solid':
      return {...attributes, fill: fillHex};
    case 'hatched':
      return {
        ...attributes,
        fill: `url(#${idPrefix}-hatched)`,
        stroke: fillHex,
        'stroke-width': '1'
      };
    case 'outline':
      return {
        ...attributes,
        fill: 'none',
        stroke: fillHex,
        ...(provenance === 'unknown' ? {'stroke-dasharray': '3 2'} : {})
      };
  }
}

/** Inline lineage label with explanatory accessible title text. */
export function provenanceBadge(provenance: MarkProvenance): ViewElementNode {
  const titles: Record<MarkProvenance, string> = {
    readback: 'Values read back from GPU execution.',
    replay: 'Values reconstructed by GPU replay.',
    derived: 'Values derived from recorded evidence.',
    metadata: 'Static metadata, not an execution measurement.',
    unknown: 'Value lineage is unknown; no provenance has been inferred.'
  };
  return h(
    'span',
    {class: `lens-provenance lens-provenance--${provenance}`, title: titles[provenance]},
    PROVENANCE_LABEL[provenance]
  );
}
