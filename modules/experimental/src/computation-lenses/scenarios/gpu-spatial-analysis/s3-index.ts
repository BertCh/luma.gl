// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registers the S3 gpu-spatial-analysis lens scenarios (side-effect imports), grouped by family:
// cells, geometry, lines (docs pages operators-cells.md, operators-geometry.md, operators-lines.md).

// cells
import './point-to-cell.lens';
import './cell-geometry.lens';
import './cell-aggregation.lens';
import './cell-rollup.lens';
import './cell-pyramid.lens';
import './cell-level-selection.lens';
import './cell-topology.lens';
import './cell-compaction.lens';
import './cell-grid-path.lens';
import './cell-measures.lens';
import './cell-cover.lens';
import './cell-set-outline.lens';
import './segment-ring-assembly.lens';
import './line-noding.lens';
import './polygonize.lens';
import './make-valid.lens';
import './cell-table-compare.lens';

// geometry
import './minimum-bounds.lens';
import './outline-geometry.lens';
import './offset-curve.lens';
import './vertex-snap.lens';
import './polygon-overlay.lens';
import './buffer-surface.lens';
import './affine-transform.lens';
import './geometry-orientation.lens';
import './geometry-cleanup.lens';
import './polygon-triangulation.lens';
import './label-point.lens';
import './shape-descriptors.lens';
import './delaunay-tessellation.lens';
import './voronoi-diagram.lens';
import './line-density.lens';
import './line-length-per-polygon.lens';
import './grid-generator.lens';
import './shape-generator.lens';
import './hilbert-keys.lens';
import './lon-lat-polygon-preparation.lens';
import './rectangle-clip.lens';

// lines
import './line-segmentize.lens';
import './line-smooth.lens';
import './line-chunk.lens';
import './geometry-measures.lens';
import './local-frame-projection.lens';
import './linear-referencing.lens';
import './line-locate.lens';
import './line-split.lens';
import './line-merge.lens';
import './line-clip-by-polygon.lens';
import './shared-paths.lens';
import './line-simplification.lens';
import './trajectory-metrics.lens';
import './trajectory-playhead.lens';
import './trajectory-resample.lens';
import './zone-events.lens';
import './fleet-dwell-zone-events-recipe.lens';
import './trajectory-encounters.lens';
import './trajectory-closest-approach.lens';
import './track-similarity.lens';
import './great-circle-arcs.lens';
import './geodesic-pairs.lens';
import './geodesic-destination.lens';
import './coverage-simplification.lens';
import './coverage-dissolve.lens';
