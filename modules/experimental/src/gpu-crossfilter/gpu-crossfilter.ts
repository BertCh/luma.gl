// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuXfilter.

import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GraphVectorView,
  type GPUCommandGraphCPUCondition,
  type GPUCommandNode,
  type GPUNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUGroupAggregation, type GPUGroupAggregationOperation} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUHistogram,
  type GPUHistogramDomain,
  type GPUHistogramEdges
} from '@luma.gl/gpgpu/gpu-core';
import {GPUMask} from '@luma.gl/gpgpu/gpu-core';
import {GPUReduction} from '@luma.gl/gpgpu/gpu-core';
import {GPUVisibilityWorkflow} from '@luma.gl/gpgpu/gpu-core';
import {createTransientView} from '@luma.gl/gpgpu/gpu-core';
import {
  createHistogramCubeBuildNodes,
  createHistogramCubeReadoutNode,
  getHistogramCubeCellCount,
  type HistogramCubeBinning
} from './histogram-cube';
import {
  createOwnedView,
  GPUCrossfilterSelection,
  type GPUCrossfilterDimension,
  type GPUCrossfilterMask,
  type GPUCrossfilterScalarFormat,
  type GPUCrossfilterScalarInput
} from './gpu-selection';

/** Shared options for a linked, GPU-resident dashboard view. */
export type GPUCrossfilterViewOptions = {
  /** Stable view identifier used by graph nodes and {@link GPUCrossfilter.getViewMask}. */
  id: string;
  /** Selection dimension controlled by this view, if any. */
  dimension?: string;
  /** Whether this view includes its own selection in its result. */
  includeOwnSelection?: boolean;
};

/** GPU-resident histogram whose selection stays synchronized with linked views. */
export type GPUCrossfilterHistogramView<
  Format extends GPUCrossfilterScalarFormat = GPUCrossfilterScalarFormat
> = GPUCrossfilterViewOptions & {
  /** This view accumulates scalar values into histogram bins. */
  kind: 'histogram';
  /** Scalar source rows sharing the dashboard's chunk topology. */
  input: GPUCrossfilterScalarInput<Format>;
  /** Caller-owned bin counts. */
  output: GraphDataView<'uint32'>;
} & (
    | {
        /** Equal-width literal, GPU-resident, or automatically inferred domain. */
        domain: GPUHistogramDomain<Format>;
        /** Equal-width histograms do not accept explicit boundaries. */
        edges?: never;
      }
    | {
        /** Irregular literal or GPU-resident bin boundaries. */
        edges: GPUHistogramEdges<Format>;
        /** Irregular histograms do not use an equal-width domain. */
        domain?: never;
      }
  );

/** GPU-resident dense group counts or floating-point grouped statistics. */
export type GPUCrossfilterGroupView = GPUCrossfilterViewOptions & {
  /** This view aggregates rows by their dense unsigned group keys. */
  kind: 'group';
  /** One source-aligned group key per row. */
  keys: GPUCrossfilterMask;
} & (
    | {
        /** Caller-owned group counts. */
        output: GraphDataView<'uint32'>;
        /** Counting is the default grouped operation. */
        operation?: 'count';
        /** Counting does not consume floating-point values. */
        values?: never;
      }
    | {
        /** Caller-owned floating-point group statistics. */
        output: GraphDataView<'float32'>;
        /** Floating-point grouped aggregation to compute. */
        operation: Exclude<GPUGroupAggregationOperation, 'count'>;
        /** One source-aligned floating-point contribution per row. */
        values: GPUCrossfilterScalarInput<'float32'>;
      }
  );

/** Stable visible source indices and count for a linked scatterplot, map, or other renderer. */
export type GPUCrossfilterVisibilityView = GPUCrossfilterViewOptions & {
  /** This view publishes stable visible indices for rendering. */
  kind: 'visibility';
  /** Caller-owned destination for stable, compacted source indices. */
  output: GPUCrossfilterMask;
  /** Caller-owned visible count, optionally an indirect draw command word. */
  count: GraphDataView<'uint32'>;
  /** Optional additional caller-owned, canonical source-aligned mask. */
  outputMask?: GPUCrossfilterMask;
  /** Optional explicit stable source identifiers. */
  sourceIds?: GPUCrossfilterMask;
  /** First generated source identifier when explicit IDs are not supplied. */
  firstSourceIndex?: number;
};

/** Source-aligned selection output consumed directly by a linked renderer or compute pass. */
export type GPUCrossfilterMaskView = GPUCrossfilterViewOptions & {
  /** This view publishes a canonical source-aligned selection mask. */
  kind: 'mask';
  /** Caller-owned zero-or-one selection destination. */
  output: GPUCrossfilterMask;
};

/** One-row count of the rows that pass this view's effective selection. */
export type GPUCrossfilterCountView = GPUCrossfilterViewOptions & {
  /** This view publishes how many live rows pass the effective selection. */
  kind: 'count';
  /** Caller-owned one-row `uint32` count; rewritten on every encoding unless nodes are skipped. */
  output: GraphDataView<'uint32'>;
};

/** Dashboard view kept synchronized entirely through GPU command-graph work. */
export type GPUCrossfilterView =
  | GPUCrossfilterHistogramView
  | GPUCrossfilterGroupView
  | GPUCrossfilterVisibilityView
  | GPUCrossfilterMaskView
  | GPUCrossfilterCountView;

/** Source selections, linked views, and optional public mask owned by one dashboard. */
export type GPUCrossfilterProps = {
  /** Prefix for generated graph resources and compute passes. */
  id?: string;
  /** Range and rectangular brush dimensions sharing source row and chunk topology. */
  dimensions: readonly GPUCrossfilterDimension[];
  /** Linked histogram, group, visibility, and mask consumers. */
  views?: readonly GPUCrossfilterView[];
  /** Optional caller-owned destination for the intersection of every selection. */
  outputMask?: GPUCrossfilterMask;
  /**
   * Optional per-row liveness, nonzero = live, for example a time-window or residency mask. Dead
   * rows never enter a histogram, group, count, visibility list, or mask output, including views
   * that exclude their own dimension, and they are skipped when a histogram view uses
   * `domain: 'auto'`. Topology: it must share the dimensions' row count and chunk layout, and its
   * presence is fixed at construction. Its contents may change every frame without recompiling
   * (with {@link GPUCrossfilterProps.skipUnchangedDimensions} or
   * {@link GPUCrossfilterProps.histogramCubes}, call {@link GPUCrossfilter.invalidate} after
   * changing them).
   */
  liveMask?: GPUCrossfilterMask;
  /**
   * Skips every graph node whose inputs did not change since it last ran, so brushing one
   * dimension costs work proportional to the nodes that depend on it (the original crossfilter.js
   * strategy): the dimension's predicate, the composed masks that include it, and the views that
   * include its brush. Views that exclude that dimension, such as its own self-excluding
   * histogram, are not recomputed.
   *
   * Defaults to `false` because the documented contract lets callers rewrite source buffers
   * between encodings. When `true`, source buffers, group keys and values, `liveMask`, and view
   * domains are assumed constant until the application calls {@link GPUCrossfilter.invalidate}.
   * Masks become persistent buffers (they must outlive encodings that skip their writer), and
   * view outputs are rewritten only when they can change.
   *
   * Change tracking is recorded when the graph evaluates each node's condition, that is, when the
   * graph is encoded. If an encoding is never submitted, the skipped nodes' outputs stay stale
   * until the next change; call {@link GPUCrossfilter.invalidate} to force a full refresh.
   */
  skipUnchangedDimensions?: boolean;
  /**
   * Enables Falcon-style data cubes for histogram views (Moritz, Howe, Heer, "Falcon: Balancing
   * Interactive Latency and Resolution Sensitivity for Scalable Linked Visualizations", CHI 2019),
   * for tables with tens of millions of rows. Defaults to `false`.
   *
   * While a histogram view is active ({@link GPUCrossfilter.activateView}), the GPU builds, for
   * every other histogram view, an exact integer cube of rows by (active bin, linked bin) that
   * passes every other selection. A brush on the active dimension then costs `O(bins)` per linked
   * histogram, independent of the row count. Cubes are rebuilt when the activation or a
   * non-active dimension changes. Memory is `activeBins * linkedBins * 4` bytes per linked view,
   * validated against the device's storage-binding limit at construction. Source rows are assumed
   * constant until {@link GPUCrossfilter.invalidate}.
   */
  histogramCubes?: boolean;
};

/** Generation key bumped when a dimension's brush changes. */
const selectionKey = (dimensionId: string): string => `selection:${dimensionId}`;
/** Generation key bumped when a dimension's source contents are reported changed. */
const contentKey = (dimensionId: string): string => `content:${dimensionId}`;
/** Generation key bumped when a view changes which path (exact or cube) produces its output. */
const viewKey = (viewId: string): string => `view:${viewId}`;
const LIVE_KEY = 'live';
const ALL_KEY = 'all';
const CUBE_KEY = 'cube';

/** CPU skip-condition options of one group of graph nodes. */
type ConditionOptions = {
  /** Runs only when a dependency generation changed; defaults to `skipUnchangedDimensions`. */
  track?: boolean;
  /** Returns whether the group is eligible at all this encoding. */
  gate?: () => boolean;
};

/**
 * Coordinates GPU-resident selections across linked histogram, group, and rendering views.
 *
 * Range and rectangular-brush updates write only a small selection-control buffer. Source rows,
 * per-dimension predicates, composed masks, grouped counts, and compacted visible indices never
 * leave the GPU. Compile the command graph once and encode it again after each interaction.
 */
export class GPUCrossfilter<Parameters = void> {
  /** Prefix for graph resources and linked-view passes. */
  readonly id: string;
  /** Graph that owns transient selection masks and linked-view compute nodes. */
  readonly graph: GPUCommandGraph<Parameters>;
  /** Registered dashboard dimensions in selection-composition order. */
  readonly dimensions: readonly GPUCrossfilterDimension[];
  /** Linked dashboard consumers. */
  readonly views: readonly GPUCrossfilterView[];
  /** GPU-resident intersection of every registered dimension. */
  readonly mask: GPUCrossfilterMask;

  private readonly selections = new Map<string, GPUCrossfilterSelection>();
  private readonly viewMasks = new Map<string, GPUCrossfilterMask | undefined>();
  private readonly excludedDimensionMasks = new Map<string, GPUCrossfilterMask | undefined>();
  private readonly liveMask?: GPUCrossfilterMask;
  private readonly skipUnchanged: boolean;
  private readonly cubesEnabled: boolean;
  /** Buffers allocated by this controller for persistent masks, extents, and cubes. */
  private readonly ownedBuffers: Buffer[] = [];
  /** Change counters behind the skip conditions; absent keys are generation 0. */
  private readonly generations = new Map<string, number>();
  /** Whether each dimension currently has an active range or bounds brush. */
  private readonly brushActive = new Map<string, boolean>();
  /** The histogram view whose dimension is the cube's active dimension. */
  private activeViewId?: string;
  /** Linked histogram views currently answered from a cube instead of the exact path. */
  private servedViewIds: ReadonlySet<string> = new Set();
  /** Selection key whose dependent mask passes are idle because only cubes consume the brush. */
  private suppressedSelectionKey?: string;
  /** Whether masks may go stale while cubes serve every consumer of the active brush. */
  private readonly maySuppressMasks: boolean;
  private addedToGraph = false;
  private destroyed = false;

  /** Registers source-aligned selections and reserves GPU control and mask storage. */
  constructor(graph: GPUCommandGraph<Parameters>, props: GPUCrossfilterProps) {
    this.id = props.id ?? 'gpu-crossfilter';
    this.graph = graph;
    this.dimensions = props.dimensions;
    this.views = props.views ?? [];
    this.skipUnchanged = props.skipUnchangedDimensions === true;
    this.cubesEnabled = props.histogramCubes === true;
    // Only histogram views read the brush through a cube. Any other consumer of a selection mask,
    // or a public mask, needs the mask passes to keep running.
    this.maySuppressMasks =
      this.cubesEnabled &&
      props.outputMask === undefined &&
      this.dimensions.every(dimension => dimension.mask === undefined) &&
      this.views.every(view => view.kind === 'histogram');

    if (this.dimensions.length === 0) {
      throw new Error(`${this.id} requires at least one selection dimension`);
    }

    try {
      for (const dimension of this.dimensions) {
        if (this.selections.has(dimension.id)) {
          throw new Error(`${this.id} selection dimensions require unique identifiers`);
        }
        const firstSelection = this.selections.values().next().value;
        const selection = new GPUCrossfilterSelection(graph, dimension, {
          id: `${this.id}-${dimension.id}`,
          persistentMask: this.skipUnchanged
        });
        this.selections.set(dimension.id, selection);
        if (firstSelection) {
          assertMatchingInputs(firstSelection.mask, selection.mask, `${this.id} ${dimension.id}`);
        }
      }
      validateViews(this.views, this.selections, this.id);
      if (props.liveMask) {
        assertMatchingInputs(
          this.selections.values().next().value!.mask,
          props.liveMask,
          `${this.id} liveMask`
        );
        this.liveMask = props.liveMask;
      }

      const firstMask = this.selections.values().next().value!.mask;
      this.mask = props.outputMask ?? this.createMaskLike(`${this.id}-mask`, firstMask);
      assertMatchingInputs(firstMask, this.mask, `${this.id} output mask`);
      if (this.cubesEnabled) this.validateCubeMemory();
    } catch (error) {
      for (const selection of this.selections.values()) selection.destroy();
      for (const buffer of this.ownedBuffers) buffer.destroy();
      throw error;
    }
  }

  /** Updates an inclusive scalar range without reading source data or recompiling the graph. */
  setRange(dimensionId: string, range: readonly [number, number]): this {
    this.getSelection(dimensionId).setRange(range);
    this.noteSelectionChange(dimensionId, true);
    return this;
  }

  /** Updates inclusive `[minX, minY, maxX, maxY]` map or scatterplot brush bounds. */
  setBounds(dimensionId: string, bounds: readonly [number, number, number, number]): this {
    this.getSelection(dimensionId).setBounds(bounds);
    this.noteSelectionChange(dimensionId, true);
    return this;
  }

  /** Disables one selection while retaining its reusable GPU control buffer and graph passes. */
  clear(dimensionId: string): this {
    this.getSelection(dimensionId).clear();
    this.noteSelectionChange(dimensionId, false);
    return this;
  }

  /** Disables every selection without changing source buffers or command-graph topology. */
  clearAll(): this {
    this.assertAvailable();
    for (const [dimensionId, selection] of this.selections) {
      selection.clear();
      this.bumpGeneration(selectionKey(dimensionId));
      this.brushActive.set(dimensionId, false);
    }
    this.updateCubeState();
    return this;
  }

  /**
   * Reports that source contents changed, so skipped nodes must run again.
   *
   * With `skipUnchangedDimensions` or `histogramCubes`, the controller assumes source buffers do
   * not change between encodings. Pass a dimension id after rewriting that dimension's input (its
   * predicate, every view that includes its brush, and its own histogram refresh), or nothing
   * after rewriting any other source: group keys or values, a view input that is not a
   * dimension's input, `liveMask`, or a `'auto'` domain's data. Without either option this does
   * nothing, because every node runs on every encoding.
   *
   * It also recovers from an encoding that was never submitted: change tracking records a node as
   * up to date when the graph evaluates its condition, not when the GPU finishes.
   */
  invalidate(dimensionId?: string): this {
    this.assertAvailable();
    if (dimensionId === undefined) {
      this.bumpGeneration(ALL_KEY);
    } else {
      this.getSelection(dimensionId);
      this.bumpGeneration(selectionKey(dimensionId));
      this.bumpGeneration(contentKey(dimensionId));
    }
    return this;
  }

  /**
   * Makes a histogram view's dimension the active dimension of the Falcon cubes.
   *
   * Requires `histogramCubes: true`. The view must be a histogram of a `range` dimension whose
   * `input` is that dimension's `input` (the same view object), and that dimension must not be
   * `exclusiveMaximum`. While active and brushed, every other histogram view is read from a cube
   * and the active dimension's brush is snapped outward to the active view's bins: the result
   * equals the exact path for brushes whose lower endpoint is a bin's lower boundary and whose
   * upper endpoint lies in the last bin it should include (for integer data, one below the next
   * boundary). Rows outside the active view's domain or edges are not counted, so a brush wider
   * than the domain behaves as the domain. Views that are not histograms, and any histogram view
   * sharing the active dimension without its own selection, keep their exact path.
   *
   * Masks that depend on the brushed dimension stop updating while cubes serve the brush when
   * every view is a histogram and no public mask exists; otherwise they keep updating and only
   * the histogram passes are saved.
   */
  activateView(viewId: string): this {
    this.assertAvailable();
    if (!this.cubesEnabled) {
      throw new Error(`${this.id} activateView requires histogramCubes: true`);
    }
    const view = this.views.find(candidate => candidate.id === viewId);
    if (!view || !this.isCubeCandidate(view)) {
      throw new Error(`${this.id} view "${viewId}" cannot be the active cube view`);
    }
    this.activeViewId = viewId;
    this.bumpGeneration(CUBE_KEY);
    this.updateCubeState();
    return this;
  }

  /** Stops serving histogram views from cubes; every view returns to its exact path. */
  deactivateView(): this {
    this.assertAvailable();
    this.activeViewId = undefined;
    this.bumpGeneration(CUBE_KEY);
    this.updateCubeState();
    return this;
  }

  /** Returns one dimension's GPU-resident predicate mask. */
  getDimensionMask(dimensionId: string): GPUCrossfilterMask {
    return this.getSelection(dimensionId).mask;
  }

  /**
   * Returns a linked view's effective GPU mask after {@link addToGraph}.
   *
   * An own-selection-excluding view with no other dimensions returns `undefined`, indicating that
   * its histogram or grouped aggregation consumes every source row without a mask.
   */
  getViewMask(viewId: string): GPUCrossfilterMask | undefined {
    this.assertAvailable();
    if (!this.views.some(view => view.id === viewId)) {
      throw new Error(`${this.id} does not contain view "${viewId}"`);
    }
    if (!this.addedToGraph) {
      throw new Error(`${this.id} view masks are available after addToGraph()`);
    }
    return this.viewMasks.get(viewId);
  }

  /** Adds reusable selection, composition, aggregation, and visibility passes to the graph. */
  addToGraph(graph: GPUCommandGraph<Parameters> = this.graph): void {
    this.assertAvailable();
    if (graph !== this.graph) {
      throw new Error(`${this.id} selections must be added to their owning graph`);
    }
    if (this.addedToGraph) {
      throw new Error(`${this.id} can only be added to its graph once`);
    }

    for (const [dimensionId, selection] of this.selections) {
      selection.addToGraph(graph, nodeId =>
        this.createCondition(nodeId, [selectionKey(dimensionId), contentKey(dimensionId)])
      );
    }
    this.composeMasks(
      `${this.id}/controller/compose`,
      [
        ...Array.from(this.selections.values(), selection => selection.mask),
        ...(this.liveMask ? [this.liveMask] : [])
      ],
      this.mask,
      this.getMaskDependencies()
    );

    for (const view of this.views) {
      const mask = this.getEffectiveMask(view);
      this.viewMasks.set(view.id, mask);

      switch (view.kind) {
        case 'histogram':
          this.addHistogramView(view, mask);
          break;
        case 'group':
          this.addGroupView(view, mask);
          break;
        case 'visibility':
          this.addVisibilityView(view, mask);
          break;
        case 'mask':
          this.addMaskView(view, mask);
          break;
        case 'count':
          this.addCountView(view, mask);
          break;
      }
    }
    if (this.cubesEnabled) this.addHistogramCubes();
    this.addedToGraph = true;
  }

  /** Releases controller-owned selection controls without destroying imported source buffers. */
  destroy(): void {
    if (this.destroyed) return;
    for (const selection of this.selections.values()) selection.destroy();
    for (const buffer of this.ownedBuffers) buffer.destroy();
    this.destroyed = true;
  }

  /** Resolves one registered range or bounds selection. */
  private getSelection(dimensionId: string): GPUCrossfilterSelection {
    this.assertAvailable();
    const selection = this.selections.get(dimensionId);
    if (!selection) {
      throw new Error(`${this.id} does not contain selection dimension "${dimensionId}"`);
    }
    return selection;
  }

  /** Records a brush change for the skip conditions and the cube routing. */
  private noteSelectionChange(dimensionId: string, active: boolean): void {
    this.bumpGeneration(selectionKey(dimensionId));
    this.brushActive.set(dimensionId, active);
    this.updateCubeState();
  }

  private bumpGeneration(key: string): void {
    this.generations.set(key, (this.generations.get(key) ?? 0) + 1);
  }

  /**
   * Creates the CPU skip condition of one graph node, or `undefined` when the node always runs.
   *
   * The node runs when `gate` allows it, no cube-served brush makes it idle, and (when tracked)
   * a dependency generation changed since it last ran or it never ran. Each node needs its own
   * condition because the "last ran" record is per node; every node of one group sees the same
   * generations, so a group runs or skips together. The record is made when the graph evaluates
   * the condition, before the GPU executes the encoding.
   */
  private createCondition(
    nodeId: string,
    dependencies: readonly string[],
    options: ConditionOptions = {}
  ): GPUCommandGraphCPUCondition<Parameters> | undefined {
    const track = options.track ?? this.skipUnchanged;
    const maySuppress =
      this.maySuppressMasks && dependencies.some(key => key.startsWith('selection:'));
    if (!track && !maySuppress && !options.gate) return undefined;

    const observed = new Map<string, number>();
    let hasRun = false;
    return {
      id: `${nodeId}/skip`,
      source: 'cpu',
      evaluate: () => {
        if (options.gate && !options.gate()) return false;
        if (
          this.suppressedSelectionKey !== undefined &&
          dependencies.includes(this.suppressedSelectionKey)
        ) {
          return false;
        }
        if (!track) return true;
        let changed = !hasRun;
        hasRun = true;
        for (const key of [ALL_KEY, ...dependencies]) {
          const generation = this.generations.get(key) ?? 0;
          if (observed.get(key) !== generation) {
            observed.set(key, generation);
            changed = true;
          }
        }
        return changed;
      }
    };
  }

  /** Adds an operation, attaching one skip condition per command node it expands to. */
  private addConditioned(
    node: GPUNode<Parameters>,
    dependencies: readonly string[],
    options?: ConditionOptions
  ): void {
    addConditionedNodes(this.graph, node, nodeId =>
      this.createCondition(nodeId, dependencies, options)
    );
  }

  /** Generation keys of the effective mask that leaves out one dimension. */
  private getMaskDependencies(excludedDimensionId?: string): string[] {
    const keys = this.dimensions
      .filter(dimension => dimension.id !== excludedDimensionId)
      .map(dimension => selectionKey(dimension.id));
    if (this.liveMask) keys.push(LIVE_KEY);
    return keys;
  }

  /** Generation keys of everything a view's output depends on. */
  private getViewDependencies(view: GPUCrossfilterView): string[] {
    const dependencies = this.getMaskDependencies(
      excludesOwnSelection(view) ? view.dimension : undefined
    );
    if (view.kind === 'histogram') {
      // A histogram also reads its own dimension's source values, and switches between paths.
      if (view.dimension !== undefined) dependencies.push(contentKey(view.dimension));
      dependencies.push(viewKey(view.id));
    }
    return dependencies;
  }

  /** Reserves a source-aligned mask, persistent when nodes that write it may be skipped. */
  private createMaskLike(id: string, template: GPUCrossfilterMask): GPUCrossfilterMask {
    return createMaskLike(
      this.graph,
      id,
      template,
      this.skipUnchanged ? this.ownedBuffers : undefined
    );
  }

  /** Computes and caches leave-one-dimension-out masks for linked distribution views. */
  private getEffectiveMask(view: GPUCrossfilterView): GPUCrossfilterMask | undefined {
    if (!excludesOwnSelection(view) || !view.dimension) return this.mask;
    if (this.excludedDimensionMasks.has(view.dimension)) {
      return this.excludedDimensionMasks.get(view.dimension);
    }

    const inputs = Array.from(this.selections.values())
      .filter(selection => selection.dimension.id !== view.dimension)
      .map(selection => selection.mask);
    if (this.liveMask) inputs.push(this.liveMask);
    if (inputs.length === 0) {
      this.excludedDimensionMasks.set(view.dimension, undefined);
      return undefined;
    }

    const output = this.createMaskLike(
      `${this.id}/controller/mask/without/${view.dimension}`,
      this.mask
    );
    this.composeMasks(
      `${this.id}/controller/compose-without/${view.dimension}`,
      inputs,
      output,
      this.getMaskDependencies(view.dimension)
    );
    this.excludedDimensionMasks.set(view.dimension, output);
    return output;
  }

  /** Adds either an equal-width or an irregular linked histogram. */
  private addHistogramView(
    view: GPUCrossfilterHistogramView,
    mask: GPUCrossfilterMask | undefined
  ): void {
    const id = `${this.id}/view/${view.id.length}:${view.id}`;
    const dependencies = this.getViewDependencies(view);
    // While a cube answers this view, its exact path idles.
    const options: ConditionOptions | undefined = this.cubesEnabled
      ? {gate: () => !this.servedViewIds.has(view.id)}
      : undefined;
    if (view.edges !== undefined) {
      this.addConditioned(
        new GPUHistogram({
          id,
          input: view.input,
          output: view.output,
          edges: view.edges,
          mask
        }),
        dependencies,
        options
      );
      return;
    }
    let domain = view.domain;
    if (domain === 'auto' && this.liveMask) {
      // GPUHistogram's own 'auto' extent spans every input row; restrict it to live rows.
      const extent = createTransientView(
        this.graph,
        `${id}-live-extent`,
        view.input.format,
        2
      ) as GraphDataView<typeof view.input.format>;
      this.addConditioned(
        new GPUReduction({
          id: `${id}/extent`,
          input: view.input,
          mask: this.liveMask,
          output: extent,
          operation: 'extent'
        }),
        dependencies,
        options
      );
      domain = extent;
    }
    this.addConditioned(
      new GPUHistogram({
        id,
        input: view.input,
        output: view.output,
        domain,
        mask
      }),
      dependencies,
      options
    );
  }

  /** Adds a one-row count of rows passing the effective selection. */
  private addCountView(view: GPUCrossfilterCountView, mask: GPUCrossfilterMask | undefined): void {
    const dependencies = this.getViewDependencies(view);
    const selectionMask = mask ?? this.createAllRowsMask(view.id, dependencies);
    this.viewMasks.set(view.id, selectionMask);
    this.addConditioned(
      new GPUReduction({
        id: `${this.id}/view/${view.id.length}:${view.id}`,
        input: selectionMask,
        output: view.output,
        operation: 'sum'
      }),
      dependencies
    );
  }

  /**
   * Intersects masks, splitting the work into intermediate masks when one pass would exceed the
   * device's storage-buffer binding limit (one binding per input plus the output).
   */
  private composeMasks(
    id: string,
    inputs: readonly GPUCrossfilterMask[],
    output: GPUCrossfilterMask,
    dependencies: readonly string[],
    options?: ConditionOptions
  ): void {
    const maximumInputs = Math.max(
      2,
      (this.graph.device.limits.maxStorageBuffersPerShaderStage || 8) - 1
    );
    let remaining = inputs;
    let level = 0;
    while (remaining.length > maximumInputs) {
      const next: GPUCrossfilterMask[] = [];
      for (let first = 0; first < remaining.length; first += maximumInputs) {
        const group = remaining.slice(first, first + maximumInputs);
        if (group.length === 1) {
          next.push(group[0]);
          continue;
        }
        // Intermediate masks live and die within one composition, so transient storage is safe.
        const partial = createMaskLike(this.graph, `${id}/partial-${level}-${first}`, output);
        this.addConditioned(
          new GPUMask({id: `${id}/partial-${level}-${first}`, inputs: group, output: partial}),
          dependencies,
          options
        );
        next.push(partial);
      }
      remaining = next;
      level++;
    }
    this.addConditioned(new GPUMask({id, inputs: remaining, output}), dependencies, options);
  }

  /** Adds source-aligned masked counts or floating-point grouped statistics. */
  private addGroupView(view: GPUCrossfilterGroupView, mask: GPUCrossfilterMask | undefined): void {
    const id = `${this.id}/view/${view.id.length}:${view.id}`;
    const dependencies = this.getViewDependencies(view);
    if (!isGroupStatisticView(view)) {
      this.addConditioned(
        new GPUGroupAggregation({
          id,
          keys: view.keys,
          output: view.output,
          operation: 'count',
          mask
        }),
        dependencies
      );
      return;
    }

    this.addConditioned(
      new GPUGroupAggregation({
        id,
        keys: view.keys,
        values: view.values,
        output: view.output,
        operation: view.operation,
        mask
      }),
      dependencies
    );
  }

  /** Adds stable row-ID compaction and an optional public source-aligned predicate. */
  private addVisibilityView(
    view: GPUCrossfilterVisibilityView,
    mask: GPUCrossfilterMask | undefined
  ): void {
    const dependencies = this.getViewDependencies(view);
    const selectionMask = mask ?? this.createAllRowsMask(view.id, dependencies);
    this.viewMasks.set(view.id, selectionMask);
    this.addConditioned(
      new GPUVisibilityWorkflow({
        id: `${this.id}/view/${view.id.length}:${view.id}`,
        predicates: [{kind: 'selection', mask: selectionMask}],
        output: view.output,
        count: view.count,
        outputMask: view.outputMask,
        sourceIds: view.sourceIds,
        firstSourceIndex: view.firstSourceIndex
      }),
      dependencies
    );
  }

  /** Publishes one effective source-aligned view mask without transferring GPU ownership. */
  private addMaskView(view: GPUCrossfilterMaskView, mask: GPUCrossfilterMask | undefined): void {
    const dependencies = this.getViewDependencies(view);
    const selectionMask = mask ?? this.createAllRowsMask(view.id, dependencies);
    if (selectionMask !== view.output) {
      this.addConditioned(
        new GPUMask({
          id: `${this.id}/view/${view.id.length}:${view.id}`,
          inputs: [selectionMask],
          output: view.output
        }),
        dependencies
      );
    }
    this.viewMasks.set(view.id, view.output);
  }

  /** Creates an all-rows predicate using only source-aligned GPU mask operations. */
  private createAllRowsMask(viewId: string, dependencies: readonly string[]): GPUCrossfilterMask {
    const id = `${this.id}/controller/mask/view/${viewId}/all`;
    const nodeId = `${this.id}/controller/view/${viewId}/all`;
    const firstMask = this.selections.values().next().value!.mask;
    if (this.liveMask) {
      const live = this.createMaskLike(id, firstMask);
      this.addConditioned(
        new GPUMask({id: nodeId, inputs: [this.liveMask], output: live}),
        dependencies
      );
      return live;
    }
    // `m OR NOT m` is all ones for any contents, so this mask depends on no selection.
    const inverse = createMaskLike(this.graph, `${id}-inverse`, firstMask);
    const output = this.createMaskLike(id, firstMask);
    this.addConditioned(
      new GPUMask({
        id: `${nodeId}/not`,
        inputs: [firstMask],
        output: inverse,
        operation: 'not'
      }),
      dependencies
    );
    this.addConditioned(
      new GPUMask({
        id: `${nodeId}/compose`,
        inputs: [firstMask, inverse],
        output,
        operation: 'or'
      }),
      dependencies
    );
    return output;
  }

  /** Whether a view can be the active cube view: a histogram of a range dimension's own input. */
  private isCubeCandidate(view: GPUCrossfilterView): view is GPUCrossfilterHistogramView {
    if (view.kind !== 'histogram' || view.dimension === undefined) return false;
    const dimension = this.dimensions.find(candidate => candidate.id === view.dimension);
    return (
      dimension?.kind === 'range' &&
      dimension.input === view.input &&
      dimension.exclusiveMaximum !== true
    );
  }

  /** Whether `linked` reads the brush of `active` and so is answered from a cube. */
  private isCubeLinked(
    active: GPUCrossfilterHistogramView,
    linked: GPUCrossfilterView
  ): linked is GPUCrossfilterHistogramView {
    return (
      linked.kind === 'histogram' &&
      linked.id !== active.id &&
      !(linked.dimension === active.dimension && excludesOwnSelection(linked))
    );
  }

  /** Rejects cubes that exceed the device's storage binding or buffer size. */
  private validateCubeMemory(): void {
    const {maxStorageBufferBindingSize, maxBufferSize} = this.graph.device.limits;
    for (const active of this.views) {
      if (!this.isCubeCandidate(active)) continue;
      for (const linked of this.views) {
        if (!this.isCubeLinked(active, linked)) continue;
        const byteLength =
          getHistogramCubeCellCount(active.output.length, linked.output.length) *
          Uint32Array.BYTES_PER_ELEMENT;
        if (byteLength > maxStorageBufferBindingSize || byteLength > maxBufferSize) {
          throw new Error(
            `${this.id} cube "${active.id}" x "${linked.id}" needs ${byteLength} bytes, above the device limit`
          );
        }
      }
    }
  }

  /**
   * Recomputes which histogram views a cube serves and whether the brushed dimension's mask passes
   * may idle, then marks views whose output path changed so their next owner rewrites them.
   */
  private updateCubeState(): void {
    if (!this.cubesEnabled) return;
    const served = new Set<string>();
    let suppressedKey: string | undefined;
    const active = this.views.find(view => view.id === this.activeViewId);
    if (
      active &&
      this.isCubeCandidate(active) &&
      active.dimension !== undefined &&
      this.brushActive.get(active.dimension)
    ) {
      for (const view of this.views) {
        if (this.isCubeLinked(active, view)) served.add(view.id);
      }
      // Idle the active dimension's masks only when every consumer of its brush is cube-served.
      const everyConsumerServed = this.views.every(
        view =>
          served.has(view.id) || (view.dimension === active.dimension && excludesOwnSelection(view))
      );
      if (this.maySuppressMasks && everyConsumerServed) {
        suppressedKey = selectionKey(active.dimension);
      }
    }
    for (const view of this.views) {
      if (served.has(view.id) !== this.servedViewIds.has(view.id)) {
        this.bumpGeneration(viewKey(view.id));
      }
    }
    this.servedViewIds = served;
    this.suppressedSelectionKey = suppressedKey;
  }

  /** Adds the cube build and readout nodes for every (active candidate, linked view) pair. */
  private addHistogramCubes(): void {
    const histograms = this.views.filter(
      (view): view is GPUCrossfilterHistogramView => view.kind === 'histogram'
    );
    const candidates = histograms.filter(view => this.isCubeCandidate(view));
    if (candidates.length === 0) return;

    // Automatic domains need an extent the cube kernels and the brush readout can read.
    const extents = new Map<string, GraphDataView>();
    for (const view of histograms) {
      if (view.edges !== undefined || view.domain !== 'auto') continue;
      const extent = createOwnedView(
        this.graph,
        `${this.id}/cube/${view.id.length}:${view.id}/extent`,
        view.input.format,
        2,
        this.ownedBuffers
      );
      extents.set(view.id, extent);
      this.addConditioned(
        new GPUReduction({
          id: `${this.id}/cube/${view.id.length}:${view.id}/extent`,
          input: view.input,
          ...(this.liveMask ? {mask: this.liveMask} : {}),
          output: extent as GraphDataView<typeof view.input.format>,
          operation: 'extent'
        }),
        [
          ...(this.liveMask ? [LIVE_KEY] : []),
          ...(view.dimension !== undefined ? [contentKey(view.dimension)] : [])
        ],
        {track: true, gate: () => this.activeViewId !== undefined}
      );
    }

    const cubeRowCounts = new Map<string, number>();
    for (const active of candidates) {
      for (const linked of histograms) {
        if (!this.isCubeLinked(active, linked)) continue;
        cubeRowCounts.set(
          linked.id,
          Math.max(
            cubeRowCounts.get(linked.id) ?? 0,
            getHistogramCubeCellCount(active.output.length, linked.output.length)
          )
        );
      }
    }
    const cubes = new Map<string, GraphDataView<'uint32'>>();
    for (const [viewId, rowCount] of cubeRowCounts) {
      cubes.set(
        viewId,
        createOwnedView(
          this.graph,
          `${this.id}/cube/${viewId.length}:${viewId}/cells`,
          'uint32',
          rowCount,
          this.ownedBuffers
        )
      );
    }

    for (const active of candidates) {
      for (const linked of histograms) {
        if (!this.isCubeLinked(active, linked)) continue;
        this.addHistogramCubePair(active, linked, cubes.get(linked.id)!, extents);
      }
    }
  }

  /** Adds the mask, build, and readout nodes of one cube. */
  private addHistogramCubePair(
    active: GPUCrossfilterHistogramView,
    linked: GPUCrossfilterHistogramView,
    cube: GraphDataView<'uint32'>,
    extents: ReadonlyMap<string, GraphDataView>
  ): void {
    const activeDimension = active.dimension!;
    const pairId = `${this.id}/cube/${active.id.length}:${active.id}/${linked.id.length}:${linked.id}`;
    // The cube counts rows passing every selection except the active one, and except the linked
    // view's own when the linked view excludes it.
    const excluded = new Set([activeDimension]);
    if (linked.dimension !== undefined && excludesOwnSelection(linked)) {
      excluded.add(linked.dimension);
    }
    const maskDimensions = this.dimensions.filter(dimension => !excluded.has(dimension.id));
    const maskInputs = [
      ...maskDimensions.map(dimension => this.selections.get(dimension.id)!.mask),
      ...(this.liveMask ? [this.liveMask] : [])
    ];
    const dependencies = [
      ...maskDimensions.map(dimension => selectionKey(dimension.id)),
      ...(this.liveMask ? [LIVE_KEY] : []),
      CUBE_KEY,
      contentKey(activeDimension),
      ...(linked.dimension !== undefined ? [contentKey(linked.dimension)] : [])
    ];
    const options: ConditionOptions = {track: true, gate: () => this.activeViewId === active.id};

    let mask: GPUCrossfilterMask | undefined;
    if (maskInputs.length === 1) {
      mask = maskInputs[0];
    } else if (maskInputs.length > 1) {
      // The composed mask is read only by this cube's build, which shares its condition.
      mask = createMaskLike(this.graph, `${pairId}/mask`, this.mask);
      this.composeMasks(`${pairId}/compose`, maskInputs, mask, dependencies, options);
    }

    const activeSide = {
      chunks: getChunks(active.input),
      binning: getCubeBinning(active, extents)
    };
    const linkedSide = {
      chunks: getChunks(linked.input),
      binning: getCubeBinning(linked, extents)
    };
    for (const node of createHistogramCubeBuildNodes(this.graph, {
      id: pairId,
      active: activeSide,
      linked: linkedSide,
      cube,
      masks: mask ? getChunks(mask) : undefined,
      createCondition: nodeId => this.createCondition(nodeId, dependencies, options)
    })) {
      this.graph.add(node);
    }
    this.graph.add(
      createHistogramCubeReadoutNode(this.graph, {
        id: pairId,
        active: activeSide,
        linked: linkedSide,
        cube,
        brushState: this.selections.get(activeDimension)!.getControlView(),
        output: linked.output,
        condition: this.createCondition(`${pairId}/readout`, [], {
          track: false,
          gate: () => this.activeViewId === active.id && this.servedViewIds.has(linked.id)
        })
      })
    );
  }

  /** Rejects interactions after controller-owned selection buffers have been destroyed. */
  private assertAvailable(): void {
    if (this.destroyed) {
      throw new Error(`${this.id} has been destroyed`);
    }
  }
}

/** Whether a distribution view leaves its own dimension's brush out of its result. */
function excludesOwnSelection(view: GPUCrossfilterView): boolean {
  return (
    view.dimension !== undefined &&
    (view.includeOwnSelection === false ||
      (view.includeOwnSelection !== true && (view.kind === 'histogram' || view.kind === 'group')))
  );
}

/** Returns source chunks in their original order. */
function getChunks<Format extends GPUCrossfilterScalarFormat>(
  input: GPUCrossfilterScalarInput<Format>
): readonly GraphDataView<Format>[] {
  return input instanceof GraphVectorView ? input.data : [input];
}

/** Describes how a histogram view bins values, reading automatic domains from their extents. */
function getCubeBinning(
  view: GPUCrossfilterHistogramView,
  extents: ReadonlyMap<string, GraphDataView>
): HistogramCubeBinning {
  const base = {format: view.input.format, binCount: view.output.length};
  if (view.edges !== undefined) return {...base, edges: view.edges};
  return {...base, domain: view.domain === 'auto' ? extents.get(view.id)! : view.domain};
}

/** Expands an operation into command nodes and adds each with its own skip condition. */
function addConditionedNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  node: GPUNode<Parameters>,
  createCondition: (nodeId: string) => GPUCommandGraphCPUCondition<Parameters> | undefined
): void {
  for (const commandNode of expandCommandNodes(graph, node)) {
    if (commandNode.condition || commandNode.publication) {
      throw new Error(`${commandNode.id} cannot be skipped by a crossfilter`);
    }
    const condition = createCondition(commandNode.id);
    graph.add(
      condition ? ({...commandNode, condition} as GPUCommandNode<Parameters>) : commandNode
    );
  }
}

/** Resolves groups and producers to concrete command nodes, as `GPUCommandGraph.add` would. */
function expandCommandNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  node: GPUNode<Parameters>
): GPUCommandNode<Parameters>[] {
  if (Array.isArray(node)) {
    return (node as readonly GPUNode<Parameters>[]).flatMap(child =>
      expandCommandNodes(graph, child)
    );
  }
  if ('getNodes' in node) {
    return node.getNodes().flatMap(child => expandCommandNodes(graph, child));
  }
  if ('getCommandNodes' in node) return [...node.getCommandNodes(graph)];
  return [node as GPUCommandNode<Parameters>];
}

/** Narrows optional count operations away from floating-point grouped-statistic views. */
function isGroupStatisticView(
  view: GPUCrossfilterGroupView
): view is Extract<
  GPUCrossfilterGroupView,
  {operation: Exclude<GPUGroupAggregationOperation, 'count'>}
> {
  return view.operation !== undefined && view.operation !== 'count';
}

/**
 * Reserves graph-owned mask storage while retaining every original vector chunk boundary.
 * With `ownedBuffers` the storage is persistent and zero-initialized instead of transient.
 */
function createMaskLike<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  template: GPUCrossfilterMask,
  ownedBuffers?: Buffer[]
): GPUCrossfilterMask {
  const createView = (viewId: string, length: number): GraphDataView<'uint32'> =>
    ownedBuffers
      ? createOwnedView(graph, viewId, 'uint32', length, ownedBuffers)
      : createTransientView(graph, viewId, 'uint32', length);
  if (!(template instanceof GraphVectorView)) {
    return createView(id, template.length);
  }

  let emptyChunk: GraphDataView<'uint32'> | undefined;
  const data = template.data.map((chunk, chunkIndex) => {
    if (chunk.length === 0) {
      emptyChunk ??= createView(`${id}-empty`, 0);
      return emptyChunk;
    }
    return createView(`${id}-chunk-${chunkIndex}`, chunk.length);
  });

  return new GraphVectorView({
    id,
    name: id,
    format: 'uint32',
    length: template.length,
    valueLength: template.length,
    stride: 1,
    byteStride: Uint32Array.BYTES_PER_ELEMENT,
    rowByteLength: Uint32Array.BYTES_PER_ELEMENT,
    data
  });
}

/** Validates exact source-view kind, row count, and ordered vector chunk topology. */
function assertMatchingInputs(
  first: GPUCrossfilterScalarInput,
  second: GPUCrossfilterScalarInput,
  name: string
): void {
  if (first instanceof GraphVectorView !== second instanceof GraphVectorView) {
    throw new Error(`${name} must preserve the same view kind`);
  }
  if (first.length !== second.length) {
    throw new Error(`${name} must preserve the same row count`);
  }
  if (
    first instanceof GraphVectorView &&
    second instanceof GraphVectorView &&
    (first.data.length !== second.data.length ||
      first.data.some((chunk, chunkIndex) => chunk.length !== second.data[chunkIndex].length))
  ) {
    throw new Error(`${name} must preserve the same chunk topology`);
  }
}

/** Validates stable view identifiers and their optional linked selection dimensions. */
function validateViews(
  views: readonly GPUCrossfilterView[],
  selections: ReadonlyMap<string, GPUCrossfilterSelection>,
  controllerId: string
): void {
  const firstMask = selections.values().next().value?.mask;
  const identifiers = new Set<string>();
  for (const view of views) {
    if (identifiers.has(view.id)) {
      throw new Error(`${controllerId} linked views require unique identifiers`);
    }
    identifiers.add(view.id);
    if (view.dimension !== undefined && !selections.has(view.dimension)) {
      throw new Error(`${controllerId} view "${view.id}" references an unknown dimension`);
    }
    if (!firstMask) continue;

    switch (view.kind) {
      case 'histogram':
        assertMatchingInputs(firstMask, view.input, `${controllerId} view "${view.id}" input`);
        break;
      case 'group':
        assertMatchingInputs(firstMask, view.keys, `${controllerId} view "${view.id}" keys`);
        if (isGroupStatisticView(view)) {
          assertMatchingInputs(firstMask, view.values, `${controllerId} view "${view.id}" values`);
        }
        break;
      case 'mask':
        assertMatchingInputs(firstMask, view.output, `${controllerId} view "${view.id}" output`);
        break;
      case 'count':
        break;
      case 'visibility':
        if (view.outputMask) {
          assertMatchingInputs(
            firstMask,
            view.outputMask,
            `${controllerId} view "${view.id}" output mask`
          );
        }
        if (view.sourceIds) {
          assertMatchingInputs(
            firstMask,
            view.sourceIds,
            `${controllerId} view "${view.id}" source IDs`
          );
        }
        break;
    }
  }
}
