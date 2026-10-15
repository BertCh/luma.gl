// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LensRecord} from '../registry/view-registration';
import {
  KeyRelationGraph,
  createFunctionRelation,
  createCSRRelation,
  createUnavailableRelation,
  type KeyRelation
} from '../coordinator/key-relations';
import {getAllCaptures} from '../views/record-queries';

/** Builds only declared, captured or evaluable relations; missing evidence stays unknown. */
export function createKeyRelationGraphFromRecord(record: LensRecord): KeyRelationGraph {
  const graph = new KeyRelationGraph();
  const captures = getAllCaptures(record);
  for (const relation of record.axes.relations) {
    const reference = relation.ref;
    const fromCount = record.axes.keySpaces.find(space => space.id === relation.from)?.size;
    const unavailable = (reason: string) => createUnavailableRelation({...relation, reason});
    let mapping: KeyRelation;
    const capture =
      reference && 'layout' in reference && reference.capture
        ? (captures.find(candidate => candidate.id === reference.capture) ??
          captures.find(candidate => candidate.name === reference.capture))
        : undefined;
    if (relation.kind === 'unknown' || (relation.timeDependent && !capture)) {
      mapping = unavailable('time-dependent relation not captured');
    } else if (relation.kind === 'function' && reference && 'function' in reference) {
      const name = reference.function;
      const parameters = reference.params;
      const divisor = parameters?.['divisor'];
      const width = parameters?.['width'];
      const tileWidth = parameters?.['tileWidth'];
      const tileHeight = parameters?.['tileHeight'];
      let map: ((key: number) => number) | undefined;
      if (name === 'identity') map = key => key;
      if (name === 'divide' && divisor && divisor > 0) map = key => Math.floor(key / divisor);
      if (
        name === 'tile-2d' &&
        width &&
        tileWidth &&
        tileHeight &&
        width > 0 &&
        tileWidth > 0 &&
        tileHeight > 0
      ) {
        map = key =>
          Math.floor(Math.floor(key / width) / tileHeight) * Math.ceil(width / tileWidth) +
          Math.floor((key % width) / tileWidth);
      }
      mapping =
        fromCount === undefined
          ? unavailable('key space size unknown')
          : map
            ? createFunctionRelation({
                ...relation,
                cardinality: name === 'identity' ? '1:1' : relation.cardinality,
                fromCount,
                map
              })
            : unavailable(`function ${name} not evaluable in views`);
    } else if (relation.kind === 'buffer' && reference && 'layout' in reference && capture) {
      if (reference.layout === 'index') {
        mapping = createFunctionRelation({
          ...relation,
          fromCount: fromCount ?? capture.values.length,
          map: key => (key < capture.values.length ? capture.values[key] : [])
        });
      } else {
        const offsets =
          (reference.offsetsResource
            ? captures.find(candidate => candidate.resource === reference.offsetsResource)
            : undefined) ??
          captures.find(candidate => candidate.name === `${capture.name}-offsets`);
        const toCount = record.axes.keySpaces.find(space => space.id === relation.to)?.size;
        mapping = !offsets
          ? unavailable('relation values not captured')
          : createCSRRelation({
              ...relation,
              offsets: offsets.values,
              targets: capture.values,
              toCount: toCount ?? Infinity
            });
      }
    } else {
      mapping = unavailable('relation values not captured');
    }
    graph.add(mapping);
  }
  return graph;
}
