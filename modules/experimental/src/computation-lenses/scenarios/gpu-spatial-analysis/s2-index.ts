// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registers the S2 gpu-spatial-analysis lens scenarios (side-effect imports), grouped by family.

// density
import './point-density.lens';
import './study-window-preparation.lens';
import './inverse-distance-weighting.lens';
import './kriging.lens';
import './dot-density.lens';
import './random-points-in-polygon.lens';
import './random-points-on-line.lens';
import './kernel-density.lens';

// joins
import './spatial-predicate-join.lens';
import './spatial-join-prepared.lens';
import './spatial-join-candidates.lens';
import './point-in-polygon-join.lens';
import './nearest-feature-join.lens';
import './nearest-feature-weights.lens';
import './pair-gather.lens';
import './offset-expansion.lens';
import './bounds-filter.lens';
import './buffer-selection.lens';

// dynamics
import './class-assignment.lens';
import './classification-fit.lens';
import './transition-matrix.lens';
import './transition-diagnostics.lens';
import './spatial-markov.lens';
import './lisa-markov.lens';
import './knox-test.lens';
import './mantel-test.lens';
import './spatial-scan-statistic.lens';

// recipes
import './straight-line-catchments-recipe.lens';
import './cluster-and-outline-recipe.lens';
import './points-in-polygons-choropleth-recipe.lens';
import './period-comparison-recipe.lens';
import './change-of-support-recipe.lens';
import './rate-cluster-map-recipe.lens';
import './hot-spot-analysis-recipe.lens';
import './spatial-regression-recipe.lens';
import './fleet-dwell-recipe.lens';

// accessibility
import './catchment-accessibility.lens';
import './huff-trade-areas.lens';
import './areal-interpolation.lens';
import './pycnophylactic.lens';
import './segregation.lens';

// validity
import './segment-intersection.lens';
import './geometry-validity.lens';
import './geometry-predicates.lens';
import './minimum-clearance.lens';
import './coverage-validity.lens';
