// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUGraphDeckEffect,
  GPUGraphEdgeLayer,
  GPUGraphNodeLayer,
  OrthographicView,
  type GPUGraphDeckEffectStats,
  type GPUGraphNodeColumn,
  type GPUGraphAnalysisColumns,
  type PickingInfo
} from '@deck.gl-community/arrow-layers';
import {Buffer, type Device} from '@luma.gl/core';
import {ArrowDeck} from '../arrow-deck';
import {getDeckExampleProps, type DeckExampleDeviceOptions} from '../deck-example-device';
import {
  GRAPH_EXPLORER_LINEAR_LAYOUT_VERTEX_COUNT,
  GRAPH_EXPLORER_MAX_VISIBLE_EDGES,
  GRAPH_EXPLORER_SHOWCASE_DEFAULT_VERTEX_COUNT,
  makeGraphExplorerDataset,
  type GraphExplorerDataset,
  type GraphExplorerLayoutMode
} from '../../experimental/gpu-graph-explorer/graph-data';
import {addGraphExplorerSampledLayoutToGraph} from '../../experimental/gpu-graph-explorer/graph-scale-layout';
import {
  createExplorerControls,
  createStandaloneContainer,
  DEFAULT_NEIGHBORHOOD_DEPTH,
  EXPLORER_NEIGHBORHOOD_EDGE_COLOR,
  EXPLORER_PATH_COLOR,
  getExplorerColorScale,
  getExplorerValueDomain,
  type ExplorerColorColumn,
  type ExplorerNodeSize
} from './app-ui';

type GPUGraphExplorerDeckOptions = DeckExampleDeviceOptions & {
  dataset?: GraphExplorerDataset;
  layoutMode?: GraphExplorerLayoutMode;
  pointMode?: boolean;
  maxVisibleEdges?: number;
};

/**
 * Creates an optional deck.gl explorer using resident GPU Graph analytics and original edge chunks.
 *
 * Deck owns the WebGPU frame encoder, rendering, controller, and asynchronous node picking. The
 * graph module never depends on deck.gl, and no graph column is downloaded for animation, color,
 * sizing, selection, or dragging.
 */
export function createGPUGraphExplorerDeck(
  parent?: HTMLDivElement,
  options: GPUGraphExplorerDeckOptions = {}
): ArrowDeck<OrthographicView> {
  const {
    dataset,
    layoutMode: initialLayoutMode = 'auto',
    pointMode,
    maxVisibleEdges = GRAPH_EXPLORER_MAX_VISIBLE_EDGES,
    ...deviceOptions
  } = options;
  const initialDataset =
    dataset ?? makeGraphExplorerDataset(GRAPH_EXPLORER_SHOWCASE_DEFAULT_VERTEX_COUNT);
  const ownsContainer = !parent;
  const container = parent ?? createStandaloneContainer();
  if (getComputedStyle(container).position === 'static') container.style.position = 'relative';

  let effect: GPUGraphDeckEffect | null = null;
  let activeDevice: Device | null = null;
  let latestStats: GPUGraphDeckEffectStats | null = null;
  let currentLayoutMode = initialLayoutMode;
  let currentColorColumn: ExplorerColorColumn = 'community';
  let currentNodeSize: ExplorerNodeSize = 'pagerank';
  let currentHops = DEFAULT_NEIGHBORHOOD_DEPTH;
  // CPU-only interaction state. Nothing here is ever read back from the GPU.
  let hoverVertex: number | null = null;
  let pathSource: number | null = null;
  let pathTarget: number | null = null;
  let edgesVisible = initialDataset.vertexCount < GRAPH_EXPLORER_LINEAR_LAYOUT_VERTEX_COUNT;
  let pendingGraphVertexCount: number | null = null;
  let loadingStatus: string | null = null;
  let rebuildGeneration = 0;
  let rebuildFrame: number | null = null;
  let draggedVertex: number | null = null;
  let deck: ArrowDeck<OrthographicView>;
  const controls = createExplorerControls(container, {
    getEffect: () => effect,
    getStats: () => latestStats,
    getPendingVertexCount: () => pendingGraphVertexCount,
    getLoadingStatus: () => loadingStatus,
    getEdgesVisible: () => edgesVisible,
    getColorColumn: () => currentColorColumn,
    getInteraction: () => ({hoverVertex, pathSource, pathTarget}),
    resize: vertexCount => scheduleGraphResize(vertexCount),
    setLayoutMode: mode => {
      currentLayoutMode = mode;
      if (effect) scheduleGraphResize(effect.graph.vertexCount);
    },
    setColorColumn: column => {
      currentColorColumn = column;
      updateLayers('GPU Graph GPU visual color encoding changed');
    },
    setNodeSize: size => {
      currentNodeSize = size;
      updateLayers('GPU Graph GPU node sizing changed');
    },
    setNeighborhoodHops: hops => {
      currentHops = hops;
      effect?.setNeighborhoodHops(hops);
    },
    setEdgesVisible: visible => {
      edgesVisible = visible;
      updateLayers('GPU Graph source-chunk edge visibility changed');
    },
    setPaused: paused => {
      deck?.setProps({_animate: !paused});
      if (!paused) deck?.redraw('GPU Graph progressive GPU layout resumed');
    },
    redraw: reason => deck?.redraw(reason)
  });

  deck = new ArrowDeck<OrthographicView>({
    parent: container,
    ...getDeckExampleProps({...deviceOptions, deviceType: 'webgpu'}),
    views: new OrthographicView({id: 'gpu-graph-orthographic'}),
    initialViewState: {target: [0, 0, 0], zoom: 6.8, minZoom: 4, maxZoom: 12},
    controller: {
      dragPan: true,
      scrollZoom: {smooth: true, speed: 0.02},
      doubleClickZoom: true,
      touchZoom: true
    },
    _animate: true,
    pickAsync: 'auto',
    layers: [],
    effects: [],
    getTooltip: info => getVertexTooltip(info, effect),
    onHover: info => {
      if (draggedVertex !== null) return;
      setHoveredVertex(info.picked && info.index >= 0 ? info.index : null);
    },
    onClick: info => {
      clickVertex(info.picked && info.index >= 0 ? info.index : null);
    },
    onDragStart: (info, event) => {
      if (!effect || !info.picked || info.index < 0) return;
      draggedVertex = info.index;
      effect.setPinnedVertex(draggedVertex, true);
      updateDraggedVertex(effect, draggedVertex, info);
      controls.update();
      event.stopPropagation();
      deck.redraw('GPU Graph vertex drag started');
    },
    onDrag: (info, event) => {
      if (!effect || draggedVertex === null) return;
      updateDraggedVertex(effect, draggedVertex, info);
      event.stopPropagation();
      deck.redraw('GPU Graph vertex dragged');
    },
    onDragEnd: (_info, event) => {
      if (draggedVertex === null) return;
      draggedVertex = null;
      controls.update();
      event.stopPropagation();
      deck.redraw('GPU Graph vertex pinned');
    },
    onLoad: ({deck: loadedDeck, device}) => {
      if (device.type !== 'webgpu') throw new Error('GPU Graph deck explorer requires WebGPU');
      activeDevice = device;
      rebuildGraph(initialDataset, loadedDeck);
    },
    onFinalize: () => {
      activeDevice = null;
      rebuildGeneration++;
      if (rebuildFrame !== null) cancelAnimationFrame(rebuildFrame);
      draggedVertex = null;
      controls.destroy();
      if (ownsContainer) container.remove();
    }
  });

  return deck;

  /** Yields before large CPU generation and GPU allocation so progress remains visible. */
  function scheduleGraphResize(vertexCount: number): void {
    const generation = ++rebuildGeneration;
    pendingGraphVertexCount = vertexCount;
    loadingStatus = `Preparing ${vertexCount.toLocaleString()} resident vertices…`;
    controls.update();
    scheduleAfterPaint(generation, () => {
      let nextDataset: GraphExplorerDataset;
      try {
        nextDataset = makeGraphExplorerDataset(vertexCount);
      } catch (error) {
        loadingStatus = error instanceof Error ? error.message : 'Graph generation failed';
        pendingGraphVertexCount = null;
        controls.update();
        return;
      }
      loadingStatus = `Uploading ${vertexCount.toLocaleString()} vertices and ${nextDataset.sourceChunks.reduce((total, chunk) => total + chunk.length, 0).toLocaleString()} original edges…`;
      controls.update();
      scheduleAfterPaint(generation, () => {
        try {
          rebuildGraph(nextDataset);
          loadingStatus = null;
          pendingGraphVertexCount = null;
          controls.update();
        } catch (error) {
          loadingStatus =
            error instanceof Error ? error.message : 'The current adapter cannot hold this graph';
          pendingGraphVertexCount = null;
          controls.update();
        }
      });
    });
  }

  /** Uses two frame callbacks so the current status is painted before expensive synchronous work. */
  function scheduleAfterPaint(generation: number, callback: () => void): void {
    if (rebuildFrame !== null) cancelAnimationFrame(rebuildFrame);
    rebuildFrame = requestAnimationFrame(() => {
      if (generation !== rebuildGeneration || !activeDevice) return;
      rebuildFrame = requestAnimationFrame(() => {
        rebuildFrame = null;
        if (generation === rebuildGeneration && activeDevice) callback();
      });
    });
  }

  /** Rebuilds resident algorithms and original source layers without changing Deck's camera. */
  function rebuildGraph(
    nextDataset: GraphExplorerDataset,
    targetDeck: ArrowDeck<OrthographicView> = deck
  ): void {
    if (!activeDevice) return;
    latestStats = null;
    draggedVertex = null;
    hoverVertex = null;
    pathSource = null;
    pathTarget = null;
    if (nextDataset.vertexCount >= GRAPH_EXPLORER_LINEAR_LAYOUT_VERTEX_COUNT) {
      edgesVisible = false;
    }
    const nextEffect = new GPUGraphDeckEffect(activeDevice, nextDataset, {
      layoutMode: currentLayoutMode,
      pointMode,
      maxVisibleEdges,
      addSampledLayoutToGraph: addGraphExplorerSampledLayoutToGraph,
      onStats: stats => {
        latestStats = stats;
        controls.update();
      }
    });
    nextEffect.setNeighborhoodHops(currentHops);
    effect = nextEffect;
    targetDeck.setProps({effects: [nextEffect], layers: createLayers(nextEffect)});
    controls.update();
    // Deck updates same-ID layer bindings at the start of its next animation frame. Drawing
    // synchronously here would reuse the previous effect's already-destroyed GPU allocations.
  }

  /** Sends hover to the analysis neighborhood; layers only re-bind a mask, nothing rebuilds. */
  function setHoveredVertex(vertex: number | null): void {
    if (!effect || vertex === hoverVertex) return;
    hoverVertex = vertex;
    effect.setHoverVertex(vertex);
    updateLayers('GPU Graph hover neighborhood changed');
  }

  /** First click sets A, second sets B, third starts a new A; empty space clears. */
  function clickVertex(vertex: number | null): void {
    if (!effect) return;
    if (vertex === null) {
      pathSource = null;
      pathTarget = null;
    } else if (pathSource === null || pathTarget !== null) {
      pathSource = vertex;
      pathTarget = null;
    } else if (vertex !== pathSource) {
      pathTarget = vertex;
    }
    // Reachability bands need a source; A alone is expressed as the trivial path A to A, which
    // the layers ignore until B is set because pathRanks are only bound with both endpoints.
    effect.setPathEndpoints(pathSource, pathSource === null ? null : (pathTarget ?? pathSource));
    updateLayers('GPU Graph shortest path endpoints changed');
  }

  /** Preserves stable layer IDs while custom layers rebind each newly owned physical buffer. */
  function updateLayers(reason: string): void {
    if (!effect || !deck) return;
    deck.setProps({layers: createLayers(effect)});
    controls.update();
    deck.redraw(reason);
  }

  function createLayers(
    graphEffect: GPUGraphDeckEffect
  ): (GPUGraphEdgeLayer | GPUGraphNodeLayer)[] {
    const analysis = graphEffect.analysisColumns;
    const hasPath = Boolean(analysis) && pathSource !== null && pathTarget !== null;
    const highlightMask = analysis && hoverVertex !== null ? analysis.neighborhoodMask : undefined;
    const pathRanks = analysis && hasPath ? analysis.pathRanks : undefined;
    const baseRadius = Math.max(1.4, Math.min(6, 60 / Math.sqrt(graphEffect.graph.vertexCount)));
    const colorColumn = analysis ? getColorColumn(analysis, currentColorColumn) : undefined;
    const sizeColumn =
      analysis && currentNodeSize !== 'uniform'
        ? analysis.columns[SIZE_COLUMN_NAMES[currentNodeSize]]
        : undefined;
    const nonemptyChunkCount = graphEffect.graph.sourceVertices.data.filter(
      chunk => chunk.length > 0
    ).length;
    let remainingVisibleEdges = graphEffect.renderedEdgeCount;
    let remainingVisibleChunks = nonemptyChunkCount;
    const edgeLayers = edgesVisible
      ? graphEffect.graph.sourceVertices.data.flatMap((source, chunkIndex) => {
          if (source.length === 0) return [];
          const target = graphEffect.graph.targetVertices.data[chunkIndex];
          const visibleEdgeCount = Math.min(
            source.length,
            Math.ceil(remainingVisibleEdges / Math.max(remainingVisibleChunks, 1))
          );
          remainingVisibleEdges -= visibleEdgeCount;
          remainingVisibleChunks--;
          if (visibleEdgeCount === 0) return [];
          return [
            new GPUGraphEdgeLayer({
              id: `gpu-graph-edges-${chunkIndex}`,
              data: [],
              pickable: false,
              positions: graphEffect.positions,
              sourceVertices:
                source.buffer instanceof Buffer ? source.buffer : source.buffer.buffer,
              targetVertices:
                target.buffer instanceof Buffer ? target.buffer : target.buffer.buffer,
              highlightMask,
              pathRanks,
              highlightColor: EXPLORER_NEIGHBORHOOD_EDGE_COLOR,
              pathColor: EXPLORER_PATH_COLOR,
              edgeCount: visibleEdgeCount,
              opacity: graphEffect.graph.vertexCount > 1_024 ? 0.22 : 0.55
            })
          ];
        })
      : [];
    const nodeLayer = new GPUGraphNodeLayer({
      id: 'gpu-graph-nodes',
      data: [],
      pickable: true,
      autoHighlight: true,
      positions: graphEffect.positions,
      colorColumn,
      colorScale: getExplorerColorScale(
        currentColorColumn,
        analysis?.bandHopThresholds.length ?? 0
      ),
      sizeColumn,
      sizeScale: {
        domain: getExplorerValueDomain(currentNodeSize),
        range: [baseRadius * 0.7, baseRadius * 2.6]
      },
      radiusPixels: baseRadius,
      highlightMask,
      pathRanks,
      pathColor: EXPLORER_PATH_COLOR,
      pointMode: graphEffect.renderMode === 'points',
      vertexCount: graphEffect.graph.vertexCount,
      opacity: 1
    });
    return [...edgeLayers, nodeLayer];
  }
}

const SIZE_COLUMN_NAMES = {
  degree: 'degree',
  pagerank: 'pageRank',
  core: 'coreNumber'
} as const;

/** Maps an explorer color column to the analysis output that the node layer reads. */
function getColorColumn(
  analysis: GPUGraphAnalysisColumns,
  column: ExplorerColorColumn
): GPUGraphNodeColumn {
  switch (column) {
    case 'community':
      return analysis.columns.community;
    case 'component':
      return analysis.columns.component;
    case 'degree':
      return analysis.columns.degree;
    case 'pagerank':
      return analysis.columns.pageRank;
    case 'core':
      return analysis.columns.coreNumber;
    case 'band':
      return {buffer: analysis.reachabilityBands, format: 'uint32'};
  }
}

/** Updates the same float32x2 allocation bound directly by the node layer's instance attribute. */
function updateDraggedVertex(effect: GPUGraphDeckEffect, vertex: number, info: PickingInfo): void {
  const coordinate = info.coordinate;
  if (!coordinate || coordinate.length < 2) return;
  effect.setVertexPosition(vertex, [coordinate[0], coordinate[1]]);
}

function getVertexTooltip(info: PickingInfo, effect: GPUGraphDeckEffect | null): string | null {
  if (!info.picked || info.index < 0 || !effect) return null;
  const state = effect.isVertexPinned(info.index) ? 'pinned' : 'movable';
  return `Vertex ${info.index} · ${state}\nhover: neighborhood · click: path endpoints A, B`;
}
