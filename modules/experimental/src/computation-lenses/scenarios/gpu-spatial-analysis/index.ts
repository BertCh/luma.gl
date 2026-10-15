// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registers the gpu-spatial-analysis lens scenarios (side-effect imports). Append-only: S1, S2 and
// S3 each add one import line per scenario file, grouped by family.
import '../../../gpu-spatial-analysis/computation-pack/index';
// S2 (dynamics, density, accessibility, joins, validity, recipes): one import in s2-index.ts per scenario.
import './s2-index';
import './local-moran.lens';
// S3 (cells, geometry, lines): one import in s3-index.ts per scenario.
import './s3-index';
// S1 weights family (contiguity, lattice, transform, lag, neighbour search, algebra, summary, transpose, validation, map coloring).
import './contiguity-weights.lens';
import './lattice-weights.lens';
import './spatial-weights-transform.lens';
import './spatial-lag.lens';
import './neighbor-search.lens';
import './spatial-weights-algebra.lens';
import './spatial-weights-summary.lens';
import './spatial-weights-transpose.lens';
import './spatial-weights-validation.lens';
import './map-coloring.lens';
// S1 statistics and regression families: one import per scenario in each s1-*-index.ts.
import './s1-inference-index';
import './s1-summaries-index';
import './s1-regions-index';
import './s1-clustering-index';
import './s1-regression-index';
