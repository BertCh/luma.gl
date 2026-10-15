// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUNeighborhoodSummaryStatistic} from '../../src/gpu-spatial-analysis/neighborhood-summary';
import type {Round4Case} from './parity-wp1-round4-fixture';

export type Round5Weights = {offsets: number[]; neighbors: number[]; weights: number[]};

export type NeighborhoodCase = Round4Case &
  Round5Weights & {
    values: number[];
    categories: number[];
    mask: number[];
    normalize: boolean;
    includeFocal: boolean;
    focalWeight: number;
    maximumNeighbors: number;
    truncated: boolean;
    integerValues: boolean;
    statistics: GPUNeighborhoodSummaryStatistic[];
    expected: {table: number[]; modes: number[]; entropy: number[]; overflow: number[]};
  };

export type PartitionCase = Round4Case & {
  columns: number;
  values: number[];
  labels: number[];
  labelCapacity: number;
  graph?: Round5Weights;
  expected: {summary: number[]; regionSizes: number[]; regionWithinSumOfSquares: number[]};
};

export type RegionQuery = {
  name: string;
  kind: 'rectangle' | 'polygon' | 'radius';
  coordinates: number[];
  binCount: number;
  domain?: [number, number];
  expected: {
    mask: number[];
    ids: number[];
    counts: number[];
    statistics: number[];
    histogram: number[];
  };
};

export type RegionCase = Round4Case & {
  positions: number[];
  values: number[];
  sourceIds: number[];
  queries: RegionQuery[];
};
