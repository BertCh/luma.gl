// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getCardTokenCss} from '../encoding/card-tokens';
import {PROVENANCE_HATCH_STROKE, VIEW_GROUND_COLOR} from '../encoding/provenance';
import {getSelectionInk} from '../encoding/selection-ink';

const lightInk = getSelectionInk('light');
const darkInk = getSelectionInk('dark');

/** Shared lens presentation; evidence textures and neutral selection ink retain 3:1 contrast. */
export const LENS_CSS = `${getCardTokenCss()}

.ladder-lens-frame, .ladder-lens-controls, .ladder-lens-message, .lens-twin {
  --lens-ink: #202020;
  --lens-ground: ${VIEW_GROUND_COLOR.light};
  --lens-muted: #595959;
  --lens-selection-core: ${lightInk.core};
  --lens-selection-halo: ${lightInk.halo};
  --lens-hatch: ${PROVENANCE_HATCH_STROKE.light};
  color: var(--lens-ink); background: var(--lens-ground);
  font: inherit; line-height: 1.5; box-sizing: border-box;
}
:root[data-theme='dark'] .ladder-lens-frame, :root[data-theme='dark'] .ladder-lens-controls, :root[data-theme='dark'] .ladder-lens-message, :root[data-theme='dark'] .lens-twin {
  --lens-ink: #f2f2f2; --lens-ground: ${VIEW_GROUND_COLOR.dark}; --lens-muted: #b3b3b3;
  --lens-selection-core: ${darkInk.core}; --lens-selection-halo: ${darkInk.halo};
  --lens-hatch: ${PROVENANCE_HATCH_STROKE.dark};
}
.lens-twin { background: var(--cv-paper); }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme]) .ladder-lens-frame, :root:not([data-theme]) .ladder-lens-controls, :root:not([data-theme]) .ladder-lens-message, :root:not([data-theme]) .lens-twin {
    --lens-ink: #f2f2f2; --lens-ground: ${VIEW_GROUND_COLOR.dark}; --lens-muted: #b3b3b3;
    --lens-selection-core: ${darkInk.core}; --lens-selection-halo: ${darkInk.halo};
    --lens-hatch: ${PROVENANCE_HATCH_STROKE.dark};
  }
}
.ladder-lens-frame[data-ground="dark"] {
  --lens-ink: #f2f2f2; --lens-ground: ${VIEW_GROUND_COLOR.dark}; --lens-muted: #b3b3b3;
  --lens-selection-core: ${darkInk.core}; --lens-selection-halo: ${darkInk.halo};
  --lens-hatch: ${PROVENANCE_HATCH_STROKE.dark};
}
.ladder-lens-frame[data-ground="light"] {
  --lens-ink: #202020; --lens-ground: ${VIEW_GROUND_COLOR.light}; --lens-muted: #595959;
  --lens-selection-core: ${lightInk.core}; --lens-selection-halo: ${lightInk.halo};
  --lens-hatch: ${PROVENANCE_HATCH_STROKE.light};
}
.ladder-lens-mount, .ladder-lens-interactive { display: block; min-width: 0; }
.ladder-lens-frame { border: 1px solid var(--lens-muted); border-radius: .375rem; padding: 1rem; margin: .5rem 0; }
.ladder-lens-frame *, .ladder-lens-controls * { box-sizing: border-box; scroll-margin: 48px; }
.ladder-lens-frame, .ladder-lens-controls { scroll-margin: 48px; }
.ladder-lens-frame:focus-visible, .ladder-lens-frame :focus-visible,
.ladder-lens-controls :focus-visible, .lens-focus-visible, [data-focus="true"] {
  outline: 2px solid var(--lens-selection-core); outline-offset: 2px;
}
.ladder-lens-frame button, .ladder-lens-controls button, .lens-symbol-chip {
  color: inherit; background: var(--lens-ground); border: 1px solid var(--lens-muted);
  border-radius: .25rem; padding: .25rem .5rem; font: inherit; cursor: pointer;
}
.lens-rung-rail, .ladder-lens-controls, .lens-graph-toolbar, .lens-verdicts, .lens-symbols {
  display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; margin-block: .5rem;
}
.lens-rung-rail [aria-current="true"] { border: 2px solid var(--lens-selection-core); font-weight: 700; }
.lens-rung-rail [aria-disabled="true"], .ladder-lens-controls button:disabled { color: var(--lens-muted); border-style: dashed; cursor: default; }
.lens-axis-header { margin-block: .75rem; }
.lens-axis-chip { display: inline-block; }
.lens-axis-chip--bound { font-weight: 600; }
.lens-axis-chip--free, .lens-axis-chip--context, .lens-axis-adapter { color: var(--lens-muted); }
.lens-caption, .lens-recording-provenance, .lens-observer-effect, .lens-encoding-flags { color: var(--lens-muted); font-size: .875em; }
.lens-caption { margin-bottom: 0; overflow-wrap: anywhere; }
.lens-key-number { font-weight: 700; font-variant-numeric: tabular-nums; }
.lens-badge, .lens-verdict, .lens-provenance { display: inline-block; border: 1px solid currentColor; border-radius: .2rem; padding: .05rem .35rem; font-size: .875em; }
.lens-verdict--pass { border-style: solid; }
.lens-verdict--fail { border-width: 2px; font-weight: 700; }
.lens-verdict--unknown, .lens-unknown, .lens-provenance--unknown { border: 1px dashed var(--lens-muted); padding: .25rem; }
.lens-badge--stale, .lens-badge--nan, .lens-badge--inf { border-style: dashed; font-weight: 600; }
.lens-provenance::before { content: ''; display: inline-block; width: .75em; height: .75em; margin-right: .3em; border: 1px solid var(--lens-hatch); }
.lens-provenance--readback::before { background: var(--lens-hatch); }
.lens-provenance--replay::before, .lens-provenance--derived::before {
  background: repeating-linear-gradient(135deg, var(--lens-ground) 0 2px, var(--lens-hatch) 2px 3px, var(--lens-ground) 3px 5px);
}
.lens-provenance--metadata::before, .lens-provenance--unknown::before { background: transparent; }
.lens-decoupled { opacity: .45; filter: grayscale(1); }
.lens-disabled, .lens-disabled-reason, .lens-permalink-warning, .ladder-lens-message, .lens-blocked-glyph { padding: .5rem; border-inline-start: 3px solid currentColor; }
.lens-off-view { border-inline-start: 3px solid var(--lens-selection-core); padding-inline-start: .5rem; }
.lens-inspector { border-top: 2px solid var(--lens-muted); padding-top: 1rem; margin-top: 1rem; }
.lens-view, .lens-table, .lens-receipt, .lens-buffer-table, .lens-graph-ladder, .lens-graph-aliasing { max-width: 100%; overflow: auto; padding: 4px; }
.lens-table table, .lens-receipt table, .lens-buffer-table table, .lens-twin table { border-collapse: collapse; width: 100%; font-size: .875em; }
.lens-table th, .lens-table td, .lens-receipt th, .lens-receipt td, .lens-buffer-table th, .lens-buffer-table td, .lens-twin th, .lens-twin td { text-align: start; padding: .3rem .5rem; border-bottom: 1px solid var(--lens-muted); vertical-align: top; }
.lens-table-number { text-align: end !important; font-variant-numeric: tabular-nums; }
.lens-table-truncated, .lens-formula-placeholder { color: var(--lens-muted); }
.lens-formula-card, .lens-datum-receipt, .lens-buffer-strip, .lens-dispatch-view,
.lens-skeleton-view, .lens-realized-view, .lens-command-graph-view, .lens-timing-view,
.lens-compare-view, .lens-scope-diagram { min-width: 0; max-width: 100%; }
.lens-formula, .lens-formula-tex, .lens-tex, .lens-tex-rendered, .lens-receipt-formula { overflow-x: auto; padding-block: .5rem; }
.lens-ascii, .lens-kernel-source, .lens-source-line { font-family: monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
.lens-source-line--highlight, .lens-symbol-chip--bound { outline: 1px solid var(--lens-selection-core); }
.lens-kernel-source { max-height: 32rem; overflow: auto; }
.lens-graph-algorithm { position: relative; }
.lens-graph-unit { min-width: 0; padding: .25rem; }
.lens-graph-unit--group, .lens-graph-unit--loop { border: 1px solid var(--lens-muted); }
.lens-graph-lineage, .lens-kernels, .lens-passes { margin-block: .5rem; }
.lens-key-frames, .lens-workgroups, .lens-lanes, .lens-candidate-list, .lens-flag-offset-slot { display: flex; flex-wrap: wrap; gap: .5rem; }
.lens-value-strip, .lens-buffer-grid, .lens-buffer-envelope, .lens-buffer-histogram,
.lens-convergence, .lens-csr-brackets, .lens-neighbourhood, .lens-reduction,
.lens-pinned-trajectory { max-width: 100%; height: auto; overflow: visible; }
.lens-scrubber { width: 100%; accent-color: var(--lens-selection-core); }
.lens-worked-example { padding: .5rem; border-inline-start: 2px solid var(--lens-muted); }
[data-hover="true"] { outline: 1px solid var(--lens-selection-core); outline-offset: 1px; }
[data-state="highlight"], [data-state="selection"] { stroke: var(--lens-selection-core); }
[data-action][data-state="highlight"] { outline: 2px solid var(--lens-selection-core); outline-offset: 1px; box-shadow: 0 0 0 3px var(--lens-selection-halo); }
[data-state="selection-halo"] { stroke: var(--lens-selection-halo); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
@media print {
  [data-ladder-twin][hidden] { display: block !important; }
  .ladder-lens-controls, .ladder-lens-interactive, .ladder-lens-message { display: none !important; }
}
`;

/** Installs exactly one style element per owner document, on explicit host activation. */
export function ensureLensStyles(ownerDocument: Document): void {
  if (ownerDocument.querySelector('style[data-luma-lens-styles]')) return;
  const style = ownerDocument.createElement('style');
  style.setAttribute('data-luma-lens-styles', '');
  style.textContent = LENS_CSS;
  (ownerDocument.head ?? ownerDocument.documentElement).append(style);
}
