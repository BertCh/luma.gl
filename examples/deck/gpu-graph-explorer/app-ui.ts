// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUGraphColor,
  type GPUGraphColorScale,
  type GPUGraphDeckEffect,
  type GPUGraphDeckEffectStats
} from '@deck.gl-community/arrow-layers';
import {
  GRAPH_EXPLORER_LINEAR_LAYOUT_VERTEX_COUNT,
  GRAPH_EXPLORER_MAXIMUM_EXACT_VERTEX_COUNT,
  GRAPH_EXPLORER_SHOWCASE_DEFAULT_VERTEX_COUNT,
  GRAPH_EXPLORER_VERTEX_COUNTS,
  type GraphExplorerLayoutMode
} from '../../experimental/gpu-graph-explorer/graph-data';

export const DEFAULT_NEIGHBORHOOD_DEPTH = 2;

/** Analytics column mapped to node color. */
export type ExplorerColorColumn =
  'community' | 'component' | 'degree' | 'pagerank' | 'core' | 'band';

/** Analytics column mapped to node radius. */
export type ExplorerNodeSize = 'uniform' | 'degree' | 'pagerank' | 'core';

/**
 * Seven categorical hues for label columns, legible on the dark canvas and without yellow. The
 * length is deliberately prime: labels are minimum vertex IDs, which are often multiples of a power
 * of two, and `label % 8` would then map every community to the same color.
 */
export const EXPLORER_CATEGORICAL_PALETTE: readonly GPUGraphColor[] = [
  [86, 180, 233],
  [230, 159, 0],
  [0, 200, 150],
  [240, 110, 170],
  [160, 130, 255],
  [170, 220, 70],
  [230, 70, 70]
];

/** Low-to-high sequential ramp (blue to amber) whose low end stays visible on the canvas. */
export const EXPLORER_SEQUENTIAL_PALETTE: readonly GPUGraphColor[] = [
  [72, 84, 170],
  [43, 130, 190],
  [40, 176, 170],
  [110, 208, 120],
  [216, 226, 80],
  [255, 196, 64]
];

/** Color of unreached reachability bands. */
export const EXPLORER_NULL_COLOR: GPUGraphColor = [74, 84, 108];

/** Tint of the A to B shortest path (nodes and edges). */
export const EXPLORER_PATH_COLOR: [number, number, number] = [255, 240, 110];

/**
 * Tint of hover-neighborhood edges. Mutable tuples because deck's own `highlightColor` layer prop
 * is typed `number[]` and intersects with the layer's `GPUGraphColor`.
 */
export const EXPLORER_NEIGHBORHOOD_EDGE_COLOR: [number, number, number] = [160, 200, 255];

/**
 * Domain of a normalized float column. Degree and PageRank are heavy tailed and normalized by their
 * maximum, so a few hubs sit near 1 and nearly every other vertex below 0.2; mapping `[0, 1]`
 * would paint the whole graph one color. Values above the upper bound clamp to the top color.
 */
export function getExplorerValueDomain(
  column: ExplorerColorColumn | ExplorerNodeSize
): readonly [number, number] {
  return column === 'degree' || column === 'pagerank' ? [0, HEAVY_TAIL_DOMAIN_MAXIMUM] : [0, 1];
}

const HEAVY_TAIL_DOMAIN_MAXIMUM = 0.2;

/** Returns the GPU color scale for a color column; changing it only rewrites a uniform block. */
export function getExplorerColorScale(
  column: ExplorerColorColumn,
  bandCount: number
): GPUGraphColorScale {
  switch (column) {
    case 'community':
    case 'component':
      return {
        type: 'categorical',
        palette: EXPLORER_CATEGORICAL_PALETTE,
        nullColor: EXPLORER_NULL_COLOR
      };
    case 'band':
      return {
        type: 'linear',
        domain: [0, Math.max(1, bandCount - 1)],
        palette: EXPLORER_SEQUENTIAL_PALETTE,
        nullColor: EXPLORER_NULL_COLOR
      };
    default:
      return {
        type: 'linear',
        domain: getExplorerValueDomain(column),
        palette: EXPLORER_SEQUENTIAL_PALETTE,
        nullColor: EXPLORER_NULL_COLOR
      };
  }
}

export type GraphExplorerControls = {
  update: () => void;
  destroy: () => void;
};

export type GraphExplorerControlProps = {
  getEffect: () => GPUGraphDeckEffect | null;
  getStats: () => GPUGraphDeckEffectStats | null;
  getPendingVertexCount: () => number | null;
  getLoadingStatus: () => string | null;
  getEdgesVisible: () => boolean;
  getColorColumn: () => ExplorerColorColumn;
  /** CPU-side interaction state: hovered vertex and the clicked path endpoints A and B. */
  getInteraction: () => {
    hoverVertex: number | null;
    pathSource: number | null;
    pathTarget: number | null;
  };
  resize: (vertexCount: number) => void;
  setLayoutMode: (mode: GraphExplorerLayoutMode) => void;
  setColorColumn: (column: ExplorerColorColumn) => void;
  setNodeSize: (size: ExplorerNodeSize) => void;
  setNeighborhoodHops: (hops: number) => void;
  setEdgesVisible: (visible: boolean) => void;
  setPaused: (paused: boolean) => void;
  redraw: (reason: string) => void;
};

export function createExplorerControls(
  container: HTMLDivElement,
  props: GraphExplorerControlProps
): GraphExplorerControls {
  const panel = document.createElement('section');
  panel.setAttribute('aria-label', 'GPU graph explorer controls and live diagnostics');
  panel.setAttribute('data-gpu-graph-inspector', '');
  Object.assign(panel.style, {
    position: 'absolute',
    left: '18px',
    top: '76px',
    zIndex: '2',
    width: 'min(324px, calc(100% - 36px))',
    maxHeight: 'calc(100% - 94px)',
    overflowY: 'auto',
    boxSizing: 'border-box',
    padding: '16px',
    borderRadius: '16px',
    background: 'linear-gradient(160deg, rgba(11, 18, 38, 0.96), rgba(8, 13, 27, 0.92))',
    border: '1px solid rgba(113, 161, 242, 0.24)',
    boxShadow: '0 18px 55px rgba(0, 0, 0, 0.34)',
    backdropFilter: 'blur(16px)',
    pointerEvents: 'auto',
    touchAction: 'auto',
    color: '#edf4ff',
    font: '12px/1.48 system-ui, sans-serif'
  });
  const initialSizeIndex = Math.max(
    0,
    GRAPH_EXPLORER_VERTEX_COUNTS.indexOf(GRAPH_EXPLORER_SHOWCASE_DEFAULT_VERTEX_COUNT)
  );
  panel.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;gap:8px">
      <strong style="font-size:16px;letter-spacing:-.25px">Graph Observatory</strong>
      <span style="padding:3px 8px;border-radius:99px;background:#123a45;color:#80eadb;
        font-size:10px;font-weight:700;letter-spacing:.4px">LIVE GPU</span>
    </div>
    <p style="margin:7px 0 14px;color:#a9b8d0">Resident graph analytics and direct
      deck.gl rendering. No per-frame graph readback.</p>

    <label style="display:block;margin-bottom:13px">
      <span style="display:flex;justify-content:space-between;color:#b9c8df">Graph size
        <strong data-gpu-graph-size-value style="color:#8ccfff">1,024</strong>
      </span>
      <input data-gpu-graph-size aria-label="Graph vertex count" type="range" min="0"
        max="${GRAPH_EXPLORER_VERTEX_COUNTS.length - 1}" step="1"
        value="${initialSizeIndex}" style="width:100%;margin:8px 0 0;accent-color:#76bbff" />
      <span style="display:flex;justify-content:space-between;color:#8798b2;font-size:10px">
        <span>128</span>
        <span>${GRAPH_EXPLORER_VERTEX_COUNTS[GRAPH_EXPLORER_VERTEX_COUNTS.length - 1].toLocaleString()} vertices</span>
      </span>
    </label>

    <div style="display:flex;align-items:center;justify-content:space-between;margin:-5px 0 12px">
      <button data-gpu-graph-size-decrease type="button"
        aria-label="Decrease graph population">−</button>
      <span style="font-size:10px;color:#8798b2">14 actual resident graph populations</span>
      <button data-gpu-graph-size-increase type="button"
        aria-label="Increase graph population">+</button>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-bottom:11px">
      <label>Color by
        <select data-gpu-graph-color aria-label="Node color encoding"
          style="display:block;width:100%;margin-top:4px">
          <option value="community">Community</option>
          <option value="component">Component</option>
          <option value="degree">Degree</option>
          <option value="pagerank">PageRank</option>
          <option value="core">Core number</option>
          <option value="band">Reachability band</option>
        </select>
      </label>
      <label>Node size
        <select data-gpu-graph-node-size aria-label="Node size encoding"
          style="display:block;width:100%;margin-top:4px">
          <option value="pagerank">PageRank</option>
          <option value="degree">Degree</option>
          <option value="core">Core number</option>
          <option value="uniform">Uniform</option>
        </select>
      </label>
    </div>

    <label style="display:block;margin-bottom:10px">Force layout
      <select data-gpu-graph-layout aria-label="GPU force layout algorithm"
        style="display:block;width:100%;margin-top:4px">
        <option value="auto">Adaptive · exact / spatial / sampled</option>
        <option value="exact">Exact · bounded population</option>
        <option value="spatial">Spatial · near / far grid</option>
        <option value="sampled">Linear · four sampled repulsions</option>
      </select>
    </label>

    <label style="display:block;margin-bottom:10px">Hover neighborhood hops
      <input data-gpu-graph-depth aria-label="Hover neighborhood hop radius"
        type="range" min="0" max="8" value="${DEFAULT_NEIGHBORHOOD_DEPTH}"
        style="display:block;width:100%;margin-top:5px;accent-color:#d692ff" />
    </label>

    <label style="display:flex;align-items:center;gap:7px;margin-bottom:12px;color:#c4d0e2">
      <input data-gpu-graph-edges type="checkbox" checked aria-label="Show original graph edges" />
      Show original source-chunk edges <span data-gpu-graph-edge-count></span>
    </label>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:7px;padding:10px;
      border:1px solid rgba(113,161,242,.16);border-radius:11px;background:rgba(4,9,20,.48)">
      <span style="color:#9dafc8">Frame rate</span>
      <strong data-gpu-graph-fps style="text-align:right">—</strong>
      <span style="color:#9dafc8">CPU encode</span>
      <strong data-gpu-graph-encode style="text-align:right">—</strong>
      <span style="color:#9dafc8">Resident memory</span>
      <strong data-gpu-graph-memory style="text-align:right">—</strong>
      <span style="color:#9dafc8">Spatial index</span>
      <strong data-gpu-graph-index style="text-align:right">—</strong>
      <span style="color:#9dafc8">GPU pipeline</span>
      <strong data-gpu-graph-pipeline style="text-align:right">—</strong>
      <span style="color:#9dafc8">Recipe columns</span>
      <strong data-gpu-graph-iterations style="text-align:right">—</strong>
    </div>

    <div data-gpu-graph-legend aria-label="GPU graph visualization legend"
      style="display:flex;gap:6px 10px;align-items:center;flex-wrap:wrap;margin:11px 0;color:#b9c8df">
    </div>

    <div style="display:flex;gap:6px;flex-wrap:wrap">
      <button data-gpu-graph-pause type="button" aria-label="Pause progressive GPU layout">Pause</button>
      <button data-gpu-graph-reset type="button">Reset layout</button>
      <button data-gpu-graph-release type="button">Release pins</button>
    </div>
    <p data-gpu-graph-status role="status" aria-live="polite"
      style="margin:11px 0 3px;color:#b4c5df">Initializing WebGPU graph…</p>
    <p style="margin:4px 0 0;color:#8293ad;font-size:11px">Hover a node for its neighborhood · click two nodes for the shortest path · drag to pin · scroll to zoom</p>`;
  container.appendChild(panel);

  for (const control of panel.querySelectorAll<HTMLElement>('select, button')) {
    Object.assign(control.style, {
      border: '1px solid rgba(136, 170, 222, 0.26)',
      borderRadius: '7px',
      background: '#121d32',
      color: '#e6efff',
      padding: '5px 7px',
      font: '11px system-ui, sans-serif'
    });
  }

  const size = panel.querySelector<HTMLInputElement>('[data-gpu-graph-size]');
  const sizeValue = panel.querySelector<HTMLElement>('[data-gpu-graph-size-value]');
  const decreaseSize = panel.querySelector<HTMLButtonElement>('[data-gpu-graph-size-decrease]');
  const increaseSize = panel.querySelector<HTMLButtonElement>('[data-gpu-graph-size-increase]');
  const layout = panel.querySelector<HTMLSelectElement>('[data-gpu-graph-layout]');
  const color = panel.querySelector<HTMLSelectElement>('[data-gpu-graph-color]');
  const nodeSize = panel.querySelector<HTMLSelectElement>('[data-gpu-graph-node-size]');
  const edges = panel.querySelector<HTMLInputElement>('[data-gpu-graph-edges]');
  const depth = panel.querySelector<HTMLInputElement>('[data-gpu-graph-depth]');
  const edgeCount = panel.querySelector<HTMLElement>('[data-gpu-graph-edge-count]');
  const pause = panel.querySelector<HTMLButtonElement>('[data-gpu-graph-pause]');
  const reset = panel.querySelector<HTMLButtonElement>('[data-gpu-graph-reset]');
  const release = panel.querySelector<HTMLButtonElement>('[data-gpu-graph-release]');
  const status = panel.querySelector<HTMLElement>('[data-gpu-graph-status]');
  const framesPerSecond = panel.querySelector<HTMLElement>('[data-gpu-graph-fps]');
  const encoding = panel.querySelector<HTMLElement>('[data-gpu-graph-encode]');
  const memory = panel.querySelector<HTMLElement>('[data-gpu-graph-memory]');
  const spatialIndex = panel.querySelector<HTMLElement>('[data-gpu-graph-index]');
  const pipeline = panel.querySelector<HTMLElement>('[data-gpu-graph-pipeline]');
  const legend = panel.querySelector<HTMLElement>('[data-gpu-graph-legend]');
  const iterations = panel.querySelector<HTMLElement>('[data-gpu-graph-iterations]');
  const isolatedInteractionEvents = [
    'pointerdown',
    'pointermove',
    'pointerup',
    'mousedown',
    'mouseup',
    'touchstart',
    'touchmove',
    'touchend',
    'keydown',
    'keyup'
  ] as const;
  const stopDeckInteraction = (event: Event): void => {
    event.stopPropagation();
  };
  for (const eventName of isolatedInteractionEvents) {
    panel.addEventListener(eventName, stopDeckInteraction);
  }
  let sizeDebounce: ReturnType<typeof setTimeout> | null = null;
  let pendingVertexCount: number | null = null;
  let activeSizePointer: number | null = null;
  let paused = false;
  let renderedLegendKey = '';

  const update = (): void => {
    const effect = props.getEffect();
    if (!effect || !status) return;
    const statistics = props.getStats();
    const recipes = effect.recipeColumns;
    const interaction = props.getInteraction();
    const hover = interaction.hoverVertex === null ? 'none' : `${interaction.hoverVertex}`;
    const pathState = !recipes
      ? ''
      : interaction.pathSource === null
        ? ' · path: click a node for endpoint A'
        : interaction.pathTarget === null
          ? ` · path A ${interaction.pathSource}, click a node for endpoint B`
          : ` · path A ${interaction.pathSource} to B ${interaction.pathTarget}`;
    const recipeState = recipes
      ? ''
      : ` · analytics columns unavailable (${effect.recipeColumnsSkipReason ?? 'unknown'})`;
    for (const control of [color, nodeSize, depth]) {
      if (control) control.disabled = !recipes;
    }
    if (depth && recipes && document.activeElement !== depth) {
      depth.value = `${recipes.currentNeighborhoodHops}`;
    }
    if (legend) {
      const legendKey = `${recipes ? props.getColorColumn() : 'none'}:${recipes?.bandHopThresholds.length ?? 0}`;
      if (legendKey !== renderedLegendKey) {
        renderedLegendKey = legendKey;
        legend.innerHTML = recipes
          ? getLegendMarkup(props.getColorColumn(), recipes.bandHopThresholds)
          : getLegendMarkup(null, []);
      }
    }
    const vertexCount = effect.graph.vertexCount;
    const sizeIndex = GRAPH_EXPLORER_VERTEX_COUNTS.findIndex(count => count === vertexCount);
    const pendingGraph = props.getPendingVertexCount();
    // Keep the slider interactive so a new selection can supersede an in-flight graph rebuild.
    if (decreaseSize) decreaseSize.disabled = pendingGraph !== null;
    if (increaseSize) increaseSize.disabled = pendingGraph !== null;
    if (pendingVertexCount === null && pendingGraph === null) {
      if (size && sizeIndex >= 0) size.value = `${sizeIndex}`;
      if (sizeValue) sizeValue.textContent = vertexCount.toLocaleString();
    } else if (sizeValue && pendingGraph !== null) {
      sizeValue.textContent = pendingGraph.toLocaleString();
    }
    const exactOption = layout?.querySelector<HTMLOptionElement>('option[value="exact"]');
    if (exactOption) {
      exactOption.disabled = vertexCount > GRAPH_EXPLORER_MAXIMUM_EXACT_VERTEX_COUNT;
    }
    const spatialOption = layout?.querySelector<HTMLOptionElement>('option[value="spatial"]');
    if (spatialOption) {
      spatialOption.disabled = vertexCount >= GRAPH_EXPLORER_LINEAR_LAYOUT_VERTEX_COUNT;
    }
    if (edges) edges.checked = props.getEdgesVisible();
    const visibleEdges = props.getEdgesVisible() ? effect.renderedEdgeCount : 0;
    if (edgeCount) {
      edgeCount.textContent = `${visibleEdges.toLocaleString()} / ${effect.graph.edgeCount.toLocaleString()}`;
    }
    const boundedAnalysis =
      effect.activeLayoutMode === 'sampled' &&
      effect.graph.vertexCount >= GRAPH_EXPLORER_LINEAR_LAYOUT_VERTEX_COUNT
        ? ' · bounded analytics; convergence not sampled'
        : '';
    status.textContent =
      props.getLoadingStatus() ??
      `${vertexCount.toLocaleString()} resident vertices · ${visibleEdges.toLocaleString()} / ${effect.graph.edgeCount.toLocaleString()} original edges drawn · ${effect.activeLayoutMode} GPU layout · ${effect.renderMode} · hover ${hover}${pathState}${recipeState}${boundedAnalysis}`;
    if (framesPerSecond) {
      framesPerSecond.textContent = statistics?.framesPerSecond
        ? `${Math.round(statistics.framesPerSecond)} fps`
        : 'warming up';
    }
    if (encoding) {
      encoding.textContent = statistics
        ? `${statistics.frameEncodeMilliseconds.toFixed(2)} ms`
        : 'pending';
    }
    if (memory) {
      memory.textContent = statistics
        ? formatGraphBytes(statistics.residentBufferBytes + statistics.transientBufferBytes)
        : 'pending';
    }
    if (spatialIndex) {
      spatialIndex.textContent = statistics?.gridCellCount
        ? `${statistics.gridCellCount} cells · ${formatGraphBytes(statistics.spatialIndexBytes)}`
        : effect.activeLayoutMode === 'sampled'
          ? '4 samples / vertex'
          : 'exact';
    }
    if (pipeline) {
      pipeline.textContent = statistics
        ? `${statistics.completedAnalysisStages}/${statistics.totalAnalysisStages} init · ${statistics.frameNodeCount} frame`
        : 'compiling';
    }
    if (iterations) {
      const recipeStats = statistics?.recipeColumns;
      iterations.textContent = !recipes
        ? 'skipped'
        : recipeStats
          ? `${recipeStats.completedStages}/${recipeStats.totalStages} stages · ${recipeStats.interactionEncodeCount} interactions`
          : 'pending';
    }
  };

  const getSelectedVertexCount = (): number =>
    GRAPH_EXPLORER_VERTEX_COUNTS[Number(size?.value ?? initialSizeIndex)] ??
    GRAPH_EXPLORER_SHOWCASE_DEFAULT_VERTEX_COUNT;

  const commitSize = (): void => {
    if (sizeDebounce !== null) {
      clearTimeout(sizeDebounce);
      sizeDebounce = null;
    }
    const vertexCount = pendingVertexCount ?? getSelectedVertexCount();
    pendingVertexCount = null;
    if (
      props.getPendingVertexCount() !== null ||
      props.getEffect()?.graph.vertexCount !== vertexCount
    ) {
      props.resize(vertexCount);
    }
  };

  const previewSize = (): void => {
    pendingVertexCount = getSelectedVertexCount();
    if (sizeValue) sizeValue.textContent = pendingVertexCount.toLocaleString();
    if (sizeDebounce !== null) clearTimeout(sizeDebounce);
    sizeDebounce = setTimeout(commitSize, 140);
  };

  /** Implements native range keyboard semantics before Deck can reinterpret arrows as panning. */
  const stepSize = (step: number): void => {
    if (!size || size.disabled) return;
    const index = Math.max(
      0,
      Math.min(GRAPH_EXPLORER_VERTEX_COUNTS.length - 1, Number(size.value) + step)
    );
    size.value = `${index}`;
    previewSize();
    commitSize();
  };

  const updateSizeFromKeyboard = (event: KeyboardEvent): void => {
    if (!size || size.disabled) return;
    let index = Number(size.value);
    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowUp':
        index++;
        break;
      case 'ArrowLeft':
      case 'ArrowDown':
        index--;
        break;
      case 'Home':
        index = 0;
        break;
      case 'End':
        index = GRAPH_EXPLORER_VERTEX_COUNTS.length - 1;
        break;
      default:
        return;
    }
    event.preventDefault();
    event.stopPropagation();
    const bounded = Math.max(0, Math.min(GRAPH_EXPLORER_VERTEX_COUNTS.length - 1, index));
    size.value = `${bounded}`;
    previewSize();
    commitSize();
  };

  /** Maps real pointer coordinates to the same discrete populations used by keyboard controls. */
  const previewSizeFromPointer = (event: PointerEvent): void => {
    if (!size || size.disabled) return;
    const bounds = size.getBoundingClientRect();
    if (bounds.width <= 0) return;
    const fraction = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
    size.value = `${Math.round(fraction * (GRAPH_EXPLORER_VERTEX_COUNTS.length - 1))}`;
    previewSize();
    if (sizeDebounce !== null) {
      clearTimeout(sizeDebounce);
      sizeDebounce = null;
    }
  };

  const startSizePointer = (event: PointerEvent): void => {
    if (!size || size.disabled || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    activeSizePointer = event.pointerId;
    size.focus();
    size.setPointerCapture(event.pointerId);
    previewSizeFromPointer(event);
  };

  const moveSizePointer = (event: PointerEvent): void => {
    if (event.pointerId !== activeSizePointer) return;
    event.stopPropagation();
    previewSizeFromPointer(event);
  };

  const finishSizePointer = (event: PointerEvent): void => {
    if (!size || event.pointerId !== activeSizePointer) return;
    event.stopPropagation();
    previewSizeFromPointer(event);
    activeSizePointer = null;
    if (size.hasPointerCapture(event.pointerId)) size.releasePointerCapture(event.pointerId);
    commitSize();
  };

  const decreaseGraphSize = (): void => stepSize(-1);
  const increaseGraphSize = (): void => stepSize(1);

  const updateLayoutMode = (): void => {
    const mode = layout?.value;
    if (mode === 'auto' || mode === 'exact' || mode === 'spatial' || mode === 'sampled') {
      props.setLayoutMode(mode);
    }
  };

  const updateColorMode = (): void => {
    const column = color?.value;
    if (
      column === 'community' ||
      column === 'component' ||
      column === 'degree' ||
      column === 'pagerank' ||
      column === 'core' ||
      column === 'band'
    ) {
      props.setColorColumn(column);
      update();
    }
  };

  const updateNodeSizeMode = (): void => {
    const mode = nodeSize?.value;
    if (mode === 'pagerank' || mode === 'degree' || mode === 'core' || mode === 'uniform') {
      props.setNodeSize(mode);
    }
  };

  const updateEdgeVisibility = (): void => {
    props.setEdgesVisible(edges?.checked ?? true);
  };

  const togglePause = (): void => {
    paused = !paused;
    if (pause) {
      pause.textContent = paused ? 'Resume' : 'Pause';
      pause.setAttribute(
        'aria-label',
        paused ? 'Resume progressive GPU layout' : 'Pause progressive GPU layout'
      );
    }
    props.setPaused(paused);
  };

  const updateDepth = (): void => {
    props.setNeighborhoodHops(Number(depth?.value ?? DEFAULT_NEIGHBORHOOD_DEPTH));
    props.redraw('GPU Graph deck neighborhood hops changed');
  };
  const resetLayout = (): void => {
    props.getEffect()?.requestReset();
    props.redraw('GPU Graph deck deterministic layout reset');
  };
  const clearPins = (): void => {
    props.getEffect()?.clearPins();
    update();
    props.redraw('GPU Graph deck pins released');
  };
  size?.addEventListener('input', previewSize);
  size?.addEventListener('change', commitSize);
  size?.addEventListener('keydown', updateSizeFromKeyboard);
  size?.addEventListener('pointerdown', startSizePointer);
  size?.addEventListener('pointermove', moveSizePointer);
  size?.addEventListener('pointerup', finishSizePointer);
  size?.addEventListener('pointercancel', finishSizePointer);
  decreaseSize?.addEventListener('click', decreaseGraphSize);
  increaseSize?.addEventListener('click', increaseGraphSize);
  layout?.addEventListener('change', updateLayoutMode);
  color?.addEventListener('change', updateColorMode);
  nodeSize?.addEventListener('change', updateNodeSizeMode);
  edges?.addEventListener('change', updateEdgeVisibility);
  depth?.addEventListener('input', updateDepth);
  pause?.addEventListener('click', togglePause);
  reset?.addEventListener('click', resetLayout);
  release?.addEventListener('click', clearPins);

  return {
    update,
    destroy: () => {
      if (sizeDebounce !== null) clearTimeout(sizeDebounce);
      for (const eventName of isolatedInteractionEvents) {
        panel.removeEventListener(eventName, stopDeckInteraction);
      }
      size?.removeEventListener('input', previewSize);
      size?.removeEventListener('change', commitSize);
      size?.removeEventListener('keydown', updateSizeFromKeyboard);
      size?.removeEventListener('pointerdown', startSizePointer);
      size?.removeEventListener('pointermove', moveSizePointer);
      size?.removeEventListener('pointerup', finishSizePointer);
      size?.removeEventListener('pointercancel', finishSizePointer);
      decreaseSize?.removeEventListener('click', decreaseGraphSize);
      increaseSize?.removeEventListener('click', increaseGraphSize);
      layout?.removeEventListener('change', updateLayoutMode);
      color?.removeEventListener('change', updateColorMode);
      nodeSize?.removeEventListener('change', updateNodeSizeMode);
      edges?.removeEventListener('change', updateEdgeVisibility);
      depth?.removeEventListener('input', updateDepth);
      pause?.removeEventListener('click', togglePause);
      reset?.removeEventListener('click', resetLayout);
      release?.removeEventListener('click', clearPins);
      panel.remove();
    }
  };
}

export function formatGraphBytes(byteLength: number): string {
  if (byteLength < 1_024) return `${byteLength} B`;
  if (byteLength < 1_048_576) return `${(byteLength / 1_024).toFixed(1)} KB`;
  return `${(byteLength / 1_048_576).toFixed(2)} MB`;
}

export function createStandaloneContainer(): HTMLDivElement {
  document.body.style.margin = '0';
  const container = document.createElement('div');
  Object.assign(container.style, {
    position: 'fixed',
    inset: '0',
    overflow: 'hidden',
    background: '#070d18'
  });
  document.body.appendChild(container);
  return container;
}

function toCssColor(color: GPUGraphColor): string {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

function getSwatchMarkup(color: GPUGraphColor | string, label: string): string {
  const background = typeof color === 'string' ? color : toCssColor(color);
  return `<span style="display:inline-flex;align-items:center;gap:4px"><span style="width:9px;height:9px;border-radius:50%;background:${background}"></span>${label}</span>`;
}

function getGradientMarkup(low: string, high: string): string {
  const stops = EXPLORER_SEQUENTIAL_PALETTE.map(toCssColor).join(', ');
  return `<span style="display:inline-flex;align-items:center;gap:5px">${low}<span style="width:70px;height:8px;border-radius:4px;background:linear-gradient(90deg, ${stops})"></span>${high}</span>`;
}

/** Legend for the active color column plus the fixed hover and path encodings. */
function getLegendMarkup(
  column: ExplorerColorColumn | null,
  bandHopThresholds: readonly number[]
): string {
  const parts: string[] = [];
  switch (column) {
    case 'community':
    case 'component': {
      const swatches = EXPLORER_CATEGORICAL_PALETTE.map(
        entry =>
          `<span style="width:9px;height:9px;border-radius:50%;background:${toCssColor(entry)}"></span>`
      ).join('');
      parts.push(
        `<span style="display:inline-flex;align-items:center;gap:3px">${swatches} ${column} label mod 7</span>`
      );
      break;
    }
    case 'degree':
    case 'pagerank':
    case 'core':
      parts.push(
        getGradientMarkup(
          'low',
          column === 'core'
            ? 'high · core number (normalized)'
            : `high · ${column} (${HEAVY_TAIL_DOMAIN_MAXIMUM * 100}% of max and up)`
        )
      );
      break;
    case 'band': {
      const lastHop = bandHopThresholds[bandHopThresholds.length - 1] ?? 0;
      parts.push(getGradientMarkup('A', `${lastHop} hops from A`));
      parts.push(getSwatchMarkup(EXPLORER_NULL_COLOR, 'unreached / no A'));
      break;
    }
    default:
      parts.push(getSwatchMarkup(EXPLORER_CATEGORICAL_PALETTE[0], 'uniform color'));
  }
  if (column !== null) {
    parts.push(getSwatchMarkup('#ffffff', 'hovered neighborhood (others dim)'));
    parts.push(getSwatchMarkup(EXPLORER_PATH_COLOR, 'shortest path A to B'));
  }
  return parts.join('');
}
