// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createSeededRandom, type OracleCSR} from './spatial-weights-oracle';

/** Row `row` of a CSR as `[neighbor, weight, distance]` triples. */
function getRow(csr: OracleCSR, row: number): [number, number, number][] {
  const entries: [number, number, number][] = [];
  for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
    entries.push([csr.neighbors[slot], csr.weights[slot], csr.distances[slot] ?? 0]);
  }
  return entries;
}

/** Builds a CSR from per-row entries sorted ascending by neighbor. */
function buildCSR(rowEntries: [number, number, number][][], withDistances: boolean): OracleCSR {
  const csr: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
  for (const entries of rowEntries) {
    for (const [neighbor, weight, distance] of entries) {
      csr.neighbors.push(neighbor);
      csr.weights.push(weight);
      if (withDistances) csr.distances.push(distance);
    }
    csr.offsets.push(csr.neighbors.length);
  }
  return csr;
}

/** Random directed weights: no self entries, ascending rows, positive weights and distances. */
export function createRandomWeights(
  rows: number,
  density: number,
  seed: number,
  symmetric = false
): OracleCSR {
  const random = createSeededRandom(seed);
  const entries: [number, number, number][][] = Array.from({length: rows}, () => []);
  for (let row = 0; row < rows; row++) {
    for (let column = symmetric ? row + 1 : 0; column < rows; column++) {
      if (column !== row && random() < density) {
        const weight = 0.5 + random() * 3;
        const distance = 0.1 + random() * 5;
        entries[row].push([column, weight, distance]);
        if (symmetric) entries[column].push([row, weight, distance]);
      }
    }
  }
  for (const row of entries) row.sort((a, b) => a[0] - b[0]);
  return buildCSR(entries, true);
}

/** Weight rule for neighbors both operands list. */
export type OracleCombineRule = 'left' | 'right' | 'sum' | 'min' | 'max' | 'product' | 'binary';

/** CPU union, intersection, difference or symmetric difference of two CSRs. */
export function computeBinaryOracle(
  operation: 'union' | 'intersection' | 'difference' | 'symmetricDifference',
  left: OracleCSR,
  right: OracleCSR,
  rule: OracleCombineRule = 'left',
  withDistances = true
): OracleCSR {
  const rows = left.offsets.length - 1;
  const combine = (a: number, b: number) =>
    ({
      left: a,
      right: b,
      sum: a + b,
      min: Math.min(a, b),
      max: Math.max(a, b),
      product: a * b,
      binary: 1
    })[rule];
  const result: [number, number, number][][] = [];
  for (let row = 0; row < rows; row++) {
    const leftMap = new Map(getRow(left, row).map(entry => [entry[0], entry]));
    const rightMap = new Map(getRow(right, row).map(entry => [entry[0], entry]));
    const ids = [...new Set([...leftMap.keys(), ...rightMap.keys()])].sort((a, b) => a - b);
    const entries: [number, number, number][] = [];
    for (const id of ids) {
      const a = leftMap.get(id);
      const b = rightMap.get(id);
      const keep = {
        union: true,
        intersection: Boolean(a && b),
        difference: Boolean(a && !b),
        symmetricDifference: Boolean(a) !== Boolean(b)
      }[operation];
      if (!keep) continue;
      const weight = rule === 'binary' ? 1 : a && b ? combine(a[1], b[1]) : (a ?? b)![1];
      entries.push([id, weight, (a ?? b)![2]]);
    }
    result.push(entries);
  }
  return buildCSR(result, withDistances);
}

/** CPU exactly-k-step (or cumulative) neighbors by breadth-first search; weights 1. */
export function computeHigherOrderOracle(
  csr: OracleCSR,
  order: number,
  cumulative = false
): OracleCSR {
  const rows = csr.offsets.length - 1;
  const result: [number, number, number][][] = [];
  for (let source = 0; source < rows; source++) {
    const depth = new Map<number, number>([[source, 0]]);
    let frontier = [source];
    for (let level = 1; level <= order; level++) {
      const next: number[] = [];
      for (const node of frontier) {
        for (const [neighbor] of getRow(csr, node)) {
          if (!depth.has(neighbor)) {
            depth.set(neighbor, level);
            next.push(neighbor);
          }
        }
      }
      frontier = next;
    }
    const ids = [...depth.entries()]
      .filter(([id, level]) => id !== source && (cumulative ? level <= order : level === order))
      .map(([id]) => id)
      .sort((a, b) => a - b);
    result.push(ids.map(id => [id, 1, 0]));
  }
  return buildCSR(result, false);
}

/** CPU self-weight insertion or replacement; the self distance is 0. */
export function computeSelfWeightOracle(
  csr: OracleCSR,
  selfWeight: number | ArrayLike<number>
): OracleCSR {
  const rows = csr.offsets.length - 1;
  const result: [number, number, number][][] = [];
  for (let row = 0; row < rows; row++) {
    const value = typeof selfWeight === 'number' ? selfWeight : selfWeight[row];
    const entries = getRow(csr, row).filter(entry => entry[0] !== row);
    entries.push([row, value, 0]);
    entries.sort((a, b) => a[0] - b[0]);
    result.push(entries);
  }
  return buildCSR(result, csr.distances.length > 0);
}

/** CPU subgraph: masked-out rows are empty and slots to masked-out rows are dropped. */
export function computeSubgraphOracle(csr: OracleCSR, mask: ArrayLike<number>): OracleCSR {
  const rows = csr.offsets.length - 1;
  const result: [number, number, number][][] = [];
  for (let row = 0; row < rows; row++) {
    result.push(mask[row] ? getRow(csr, row).filter(entry => mask[entry[0]]) : []);
  }
  return buildCSR(result, csr.distances.length > 0);
}

/** CPU block weights: every other row with the same group (IDs `>= groupCount` have none). */
export function computeBlockOracle(groupIds: ArrayLike<number>, groupCount: number): OracleCSR {
  const rows = groupIds.length;
  const result: [number, number, number][][] = [];
  for (let row = 0; row < rows; row++) {
    const entries: [number, number, number][] = [];
    if (groupIds[row] < groupCount) {
      for (let other = 0; other < rows; other++) {
        if (other !== row && groupIds[other] === groupIds[row]) entries.push([other, 1, 0]);
      }
    }
    result.push(entries);
  }
  return buildCSR(result, false);
}

/** CPU weights summary in double precision. */
export function computeSummaryOracle(csr: OracleCSR, tolerance = 0) {
  const rows = csr.offsets.length - 1;
  const find = (row: number, column: number): number | undefined => {
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
      if (csr.neighbors[slot] === column) return csr.weights[slot];
    }
    return undefined;
  };
  let s0 = 0;
  let s1 = 0;
  let asymmetricSlots = 0;
  const rowSums = new Array<number>(rows).fill(0);
  const columnSums = new Array<number>(rows).fill(0);
  const cardinality: number[] = [];
  for (let row = 0; row < rows; row++) {
    cardinality.push(csr.offsets[row + 1] - csr.offsets[row]);
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
      const column = csr.neighbors[slot];
      const weight = csr.weights[slot];
      const reverse = find(column, row);
      s0 += weight;
      rowSums[row] += weight;
      columnSums[column] += weight;
      s1 += reverse === undefined ? weight * weight : 0.5 * (weight + reverse) ** 2;
      if (reverse === undefined || Math.abs(weight - reverse) > tolerance) asymmetricSlots++;
    }
  }
  let s2 = 0;
  for (let row = 0; row < rows; row++) s2 += (rowSums[row] + columnSums[row]) ** 2;
  return {
    s0,
    s1,
    s2,
    slots: csr.offsets[rows],
    asymmetricSlots,
    isolates: cardinality.filter(count => count === 0).length,
    minimumCardinality: Math.min(...cardinality),
    maximumCardinality: Math.max(...cardinality),
    cardinality
  };
}

/** Truncates a CSR to `capacity` slots the way the GPU clamps offsets. */
export function truncateCSR(csr: OracleCSR, capacity: number): OracleCSR {
  const offsets = csr.offsets.map(offset => Math.min(offset, capacity));
  const used = offsets[offsets.length - 1];
  return {
    offsets,
    neighbors: csr.neighbors.slice(0, used),
    weights: csr.weights.slice(0, used),
    distances: csr.distances.slice(0, used)
  };
}
