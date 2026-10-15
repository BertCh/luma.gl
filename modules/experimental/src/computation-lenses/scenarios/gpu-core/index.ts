// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registers the GPU Core view pack (operators, formulas, array result view) before its scenarios.
import '../../../computation-views/packs/gpu-core/index';
import './scan.lens';
import './flag-offsets.lens';
import './compaction.lens';
import './mask.lens';
import './scan-anatomy.lens';
import './reduction.lens';
import './segmented-reduction.lens';
import './sort.lens';
import './segmented-sort.lens';
import './gather.lens';
import './histogram.lens';
import './elementwise.lens';
import './elementwise-id-probe.lens';
import './grid-index.lens';
import './grid-binning.lens';
import './grid-aggregation.lens';
import './group-aggregation.lens';
import './hash-index.lens';
import './hash-join.lens';
import './mat-vec.lens';
import './mat-mul.lens';
import './convolution.lens';
import './fft1d.lens';
import './fft2d.lens';
import './conjugate-gradient-program.lens';
