// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialWeightsCombineRule} from '../../src/gpu-spatial-analysis/spatial-weights';

export type SparseMatrix = {
  offsets: number[];
  neighbors: number[];
  weights: number[];
  distances: number[];
};
export type SparseCase = {
  name: string;
  reference: string;
  n: number;
};
export type SparseProvenance = {
  provenance: {sourceSha256?: Record<string, string>; sourceTracts?: number; sourcePlaces?: number};
  inputFloat32Sha256: string;
};
export type AlgebraDataset = SparseProvenance & {
  name: string;
  n: number;
  left: SparseMatrix;
  right: SparseMatrix;
  mask: number[];
  groups: number[];
  groupCount: number;
};
export type AlgebraCase = SparseCase & {
  datasetIndex: number;
  operation:
    | 'union'
    | 'intersection'
    | 'difference'
    | 'symmetricDifference'
    | 'subgraph'
    | 'selfWeight'
    | 'higherOrder'
    | 'block';
  weightRule?: GPUSpatialWeightsCombineRule;
  selfWeight?: number;
  order?: number;
  cumulative?: boolean;
  workCapacity?: number;
  expected: SparseMatrix;
  rowSums: number[];
};
export type AlgebraFixture = {datasets: AlgebraDataset[]; cases: AlgebraCase[]};
export type TransposeCase = SparseCase &
  SparseProvenance & {
    columnCount: number;
    input: SparseMatrix;
    expected: SparseMatrix;
    /** NaN for rectangular matrices: the shared decoder maps JSON null to NaN. */
    asymmetricSlots: number;
    rowSums: number[];
  };
export type ForestCase = SparseCase &
  SparseProvenance & {
    input: SparseMatrix;
    values: number[];
    columnCount: number;
    standardize: boolean;
    exactEdges: boolean;
    expected: {
      flags: number[];
      componentLabels: number[];
      componentCount: number;
      edgeCount: number;
      slotCosts: number[];
      totalCost: number;
      standardizedValues: number[];
    };
  };

/** Readback invariant, not a numerical reference: every uploaded CSR weight is compared too. */
export function sumSparseRows(matrix: SparseMatrix): number[] {
  return matrix.offsets
    .slice(0, -1)
    .map((begin, row) =>
      matrix.weights.slice(begin, matrix.offsets[row + 1]).reduce((sum, value) => sum + value, 0)
    );
}
