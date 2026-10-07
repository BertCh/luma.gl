// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Shared machinery of the terrain scenes (builder B14a): one elevation raster, one active
 * "product" (a set of compiled contributor graphs writing one raster), one colorize pass, an
 * optional relief underlay, a throttled single-pixel hover probe and a throttled histogram of the
 * displayed raster.
 *
 * Contract with the showcase rules:
 * - Every product graph is compiled once, when the product is first shown or when one of its
 *   compile-time options changes. Switching between built products re-encodes nothing and
 *   rebuilds nothing.
 * - Per-frame work is: write the parameter buffers of the active product when it is dirty (or its
 *   elevation changed), encode its graphs, encode the tiny colorize graph when the paint changed.
 * - Readbacks are tickets of a `GPUReadbackRing`: a histogram 350 ms after the last change and a
 *   4 + 4 byte probe under the pointer. Nothing blocks.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder, Device} from '@luma.gl/core';
import {
  getGPUReliefShadingParameterValues,
  GPU_RELIEF_SHADING_PARAMETER_LENGTH,
  GPUReliefShading
} from '@luma.gl/experimental/gpu-terrain';
import type {GPUParameterBuffer} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import type {SpatialAnalysisResources} from '../../engine/resources';
import {measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, ScenePointerEvent} from '../scene';
import type {AlpsGrid} from './b14a-grid';
import {
  createColorizeGraph,
  PAINT_PARAMETER_LENGTH,
  writePaint,
  type PaintSpec,
  type Rgb
} from './b14a-colorize';
import {ColorRasterLayer} from './b14a-layers';

const HISTOGRAM_BINS = 1024;
const MAXIMUM_CLASSES = 32;
const STATS_SETTLE_MILLISECONDS = 350;

/** Histogram summary of one raster. Class rasters carry counts per class code instead. */
export type ValueStats =
  | {
      kind: 'float';
      count: number;
      min: number;
      max: number;
      mean: number;
      histogram: Uint32Array;
      /** Value below which a share `fraction` in `[0, 1]` of the valid cells lies (bin accuracy). */
      quantile: (fraction: number) => number;
    }
  | {kind: 'classes'; total: number; counts: Float64Array};

/**
 * A unit of GPU work: compiled graphs plus the parameter writes that precede them, re-encoded
 * only when it is dirty, when the elevation changed, or when a stage it depends on re-ran.
 * Heavy stages set `minIntervalMs` so a dragged slider cannot queue more GPU work than the GPU
 * finishes.
 */
export class Stage {
  readonly graphs: CompiledGPUCommandGraph<void>[];
  dirty = true;
  /** Times this stage was encoded. Dependents compare it with the count they last saw. */
  runCount = 0;
  private lastRun = Number.NEGATIVE_INFINITY;
  private seenElevationVersion = -1;
  private seenDependencyRuns: number[];

  constructor(
    readonly id: string,
    graphs: CompiledGPUCommandGraph<void>[],
    private readonly writeParameters: () => void,
    readonly dependencies: readonly Stage[] = [],
    readonly minIntervalMs = 0,
    private readonly releaseResources: () => void = () => {}
  ) {
    this.graphs = graphs;
    this.seenDependencyRuns = dependencies.map(() => -1);
  }

  /** Requests a re-encode. */
  markDirty(): void {
    this.dirty = true;
  }

  /** True while this stage or one it depends on still has work to encode. */
  isPending(elevationVersion: number): boolean {
    return (
      this.dirty ||
      this.seenElevationVersion !== elevationVersion ||
      this.dependencies.some(
        (dependency, index) =>
          dependency.runCount !== this.seenDependencyRuns[index] ||
          dependency.isPending(elevationVersion)
      )
    );
  }

  /** Encodes dependencies first, then this stage if needed. Returns true when it was encoded. */
  run(commandEncoder: CommandEncoder, now: number, elevationVersion: number): boolean {
    for (const dependency of this.dependencies)
      dependency.run(commandEncoder, now, elevationVersion);
    const needsRun =
      this.dirty ||
      this.seenElevationVersion !== elevationVersion ||
      this.dependencies.some(
        (dependency, index) => dependency.runCount !== this.seenDependencyRuns[index]
      );
    if (!needsRun || now - this.lastRun < this.minIntervalMs) return false;
    this.writeParameters();
    for (const graph of this.graphs) graph.encode(commandEncoder, {parameters: undefined});
    this.dirty = false;
    this.seenElevationVersion = elevationVersion;
    this.seenDependencyRuns = this.dependencies.map(dependency => dependency.runCount);
    this.runCount++;
    this.lastRun = now;
    return true;
  }

  /** Frees the stage's own graphs and buffers. */
  release(): void {
    this.releaseResources();
  }
}

/** What the session draws: one raster written by a stage, plus how to release the build. */
export type ProductBuild = {
  id: string;
  /** Compile-time configuration; a different key means a different build. */
  configKey: string;
  /** The stage whose outputs include `value`. May be shared with other builds. */
  stage: Stage;
  /** `rgba8` is a packed color raster that is drawn as is (no colorize pass, histogram or probe). */
  value: {buffer: Buffer; format: 'float32' | 'uint32' | 'rgba8'};
  /** Frees what the build owns (not shared stages). */
  release: () => void;
  palette?: readonly Rgb[];
  paintGraph: CompiledGPUCommandGraph<void> | null;
};

/** Options of {@link ProductBuilder.finish}. */
export type FinishOptions = {
  /** Uncompiled graphs, compiled here in order. Defaults to the builder's own graph. */
  graphs?: GPUCommandGraph<void>[];
  /** Name of a buffer created with `floats()` or `words()`. */
  value: string;
  format: 'float32' | 'uint32' | 'rgba8';
  write: () => void;
  minIntervalMs?: number;
  palette?: readonly Rgb[];
  /** Stages this one reads. */
  dependencies?: readonly Stage[];
};

/** Creates the buffers, parameter buffers and graphs of one product and releases them together. */
export class ProductBuilder {
  readonly graph: GPUCommandGraph<void>;
  private readonly owned: {destroy: () => void}[] = [];
  private readonly buffers = new Map<string, Buffer>();

  constructor(
    private readonly session: TerrainSession,
    readonly id: string
  ) {
    this.graph = new GPUCommandGraph<void>(session.device, {id: `terrain-${id}`});
  }

  /** A second graph of the same product (for example a heavy one-time stage). */
  newGraph(suffix: string): GPUCommandGraph<void> {
    return new GPUCommandGraph<void>(this.session.device, {id: `terrain-${this.id}-${suffix}`});
  }

  /** Elevation and validity of the session, imported into `graph`. */
  elevation(graph: GPUCommandGraph<void> = this.graph) {
    return this.session.importElevation(graph);
  }

  /** Creates a storage buffer owned by this product. */
  buffer(name: string, byteLength: number): Buffer {
    const buffer = this.session.resources.createBuffer(`${this.id}-${name}`, byteLength);
    this.owned.push({destroy: () => this.session.resources.release(buffer)});
    this.buffers.set(name, buffer);
    return buffer;
  }

  /** Creates a float32 buffer of `count` values (default one per pixel) and imports it. */
  floats(name: string, count = this.session.grid.pixelCount, graph = this.graph) {
    return importGraphBuffer(graph, name, this.buffer(name, count * 4), 'float32', count);
  }

  /** Creates a uint32 buffer of `count` values (default one per pixel) and imports it. */
  words(name: string, count = this.session.grid.pixelCount, graph = this.graph) {
    return importGraphBuffer(graph, name, this.buffer(name, count * 4), 'uint32', count);
  }

  /** Imports an existing buffer of this product into another graph. */
  reimport(name: string, format: 'float32' | 'uint32', graph: GPUCommandGraph<void>) {
    const buffer = this.buffers.get(name);
    if (!buffer) throw new Error(`Unknown product buffer ${name}`);
    return importGraphBuffer(graph, name, buffer, format, buffer.byteLength / 4);
  }

  /** Creates a float32 parameter buffer and its view in `graph`. */
  settings(length: number, graph: GPUCommandGraph<void> = this.graph) {
    const parameters = this.session.resources.createParameterBuffer(
      `${this.id}-settings-${this.owned.length}`,
      'float32',
      length
    );
    this.owned.push({destroy: () => this.session.resources.release(parameters)});
    return {parameters, view: parameters.importToGraph(graph) as GraphDataView<'float32'>};
  }

  /** Compiles the graphs into a stage (the stage releases the graphs and this builder's buffers). */
  finishStage(options: Omit<FinishOptions, 'value' | 'format' | 'palette'>): Stage {
    const {resources} = this.session;
    const graphs = (options.graphs ?? [this.graph]).map(graph => resources.track(graph.compile()));
    const stage = new Stage(
      this.id,
      graphs,
      options.write,
      options.dependencies ?? [],
      options.minIntervalMs ?? 0,
      () => {
        for (const graph of graphs) resources.release(graph);
        for (let index = this.owned.length - 1; index >= 0; index--) this.owned[index].destroy();
      }
    );
    this.session.registerStage(stage);
    return stage;
  }

  /** Returns a buffer created by this builder. */
  getBuffer(name: string): Buffer {
    const buffer = this.buffers.get(name);
    if (!buffer) throw new Error(`Unknown product buffer ${name}`);
    return buffer;
  }

  /** Compiles the graphs and returns a build that displays buffer `options.value`. */
  finish(options: FinishOptions): ProductBuild {
    const stage = this.finishStage(options);
    const buffer = this.getBuffer(options.value);
    return {
      id: this.id,
      configKey: '',
      stage,
      value: {buffer, format: options.format},
      palette: options.palette,
      release: () => {
        this.session.unregisterStage(stage);
        stage.release();
      },
      paintGraph: null
    };
  }
}

/** A small readback ring that copies ranges of caller buffers and hands back the bytes. */
class RangeReader {
  private readonly ring: GPUReadbackRing;
  private pendingFlag = false;
  private stopped = false;

  constructor(resources: SpatialAnalysisResources, id: string, byteLength: number) {
    this.ring = resources.track(new GPUReadbackRing(resources.device, {id, byteLength}));
  }

  get pending(): boolean {
    return this.pendingFlag;
  }

  stop(): void {
    this.stopped = true;
  }

  /** Copies the ranges back to back; returns false when a read is in flight or no ticket is free. */
  request(
    commandEncoder: CommandEncoder,
    copies: readonly {buffer: Buffer; offset: number; size: number}[],
    onBytes: (bytes: Uint8Array) => void
  ): boolean {
    if (this.pendingFlag || this.stopped) return false;
    const ticket = this.ring.tryAcquire();
    if (!ticket) return false;
    let destinationOffset = 0;
    for (const copy of copies) {
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: copy.buffer,
        sourceOffset: copy.offset,
        destinationBuffer: ticket.buffer,
        destinationOffset,
        size: copy.size
      });
      destinationOffset += copy.size;
    }
    ticket.markEncoded({byteOffset: 0, byteLength: destinationOffset});
    this.pendingFlag = true;
    void ticket
      .read()
      .then(bytes => {
        if (!this.stopped) onBytes(bytes.slice());
      })
      .catch(() => {
        // The ring or device was destroyed while the read was in flight.
      })
      .finally(() => {
        this.pendingFlag = false;
      });
    return true;
  }
}

/** Layer-side options of {@link TerrainSession.getLayers}. */
export type SessionLayerOptions = {
  /** Draw the relief underlay beneath the product. */
  underlay: boolean;
  underlayAlpha: number;
  alpha: number;
  /** Optional illumination multiplied into the product color. */
  light?: {buffer: Buffer; strength: number; gain: number; floor: number};
  /** Draw the product at all (false shows only the underlay). */
  showProduct?: boolean;
};

/** Describes a hovered cell. */
export type HoverDescriber = (cell: {
  value: number;
  elevation: number;
  column: number;
  row: number;
}) => string | null;

/** The shared engine of the terrain scenes. */
export class TerrainSession {
  readonly device: Device;
  readonly resources: SpatialAnalysisResources;
  readonly grid: AlpsGrid;
  readonly elevationBuffer: Buffer;
  readonly validityBuffer: Buffer;
  readonly colorsBuffer: Buffer;
  /** Called with the histogram of the displayed raster after it settles. */
  onStats: ((buildId: string, stats: ValueStats) => void) | null = null;
  /** Describes the hovered cell for the tooltip. */
  describeHover: HoverDescriber = ({value}) => String(value);

  private readonly ctx: Pick<SceneContext<object>, 'device' | 'signal'>;
  private readonly paintParameters: GPUParameterBuffer<'float32'>;
  private readonly statsReader: RangeReader;
  private readonly hoverReader: RangeReader;
  private bulkReader: RangeReader | null = null;
  private readonly builds = new Map<string, ProductBuild>();
  private readonly stages = new Set<Stage>();
  private current: ProductBuild | null = null;
  private paint: PaintSpec = {mode: 'ramp', ramp: 'viridis', low: 0, high: 1, alpha: 1};
  private paintDirty = true;
  private statsDirty = false;
  private lastComputeTime = 0;
  private elevationVersion = 0;
  private hoverIndex = -1;
  private hoverCache: {
    index: number;
    version: number;
    buildId: string;
    text: string | null;
  } | null = null;
  private underlay: {
    graphs: CompiledGPUCommandGraph<void>[];
    paintGraph: CompiledGPUCommandGraph<void>;
    colors: Buffer;
    version: number;
  } | null = null;
  private destroyed = false;

  constructor(
    ctx: Pick<SceneContext<object>, 'device' | 'signal'>,
    resources: SpatialAnalysisResources,
    grid: AlpsGrid
  ) {
    this.ctx = ctx;
    this.device = ctx.device;
    this.resources = resources;
    this.grid = grid;
    this.elevationBuffer = resources.createBuffer('elevation', grid.pixelCount * 4);
    this.validityBuffer = resources.createBuffer('validity', grid.pixelCount * 4);
    this.colorsBuffer = resources.createBuffer('colors', grid.pixelCount * 4);
    this.paintParameters = resources.createParameterBuffer(
      'paint',
      'float32',
      PAINT_PARAMETER_LENGTH
    );
    this.statsReader = new RangeReader(resources, 'terrain-stats', grid.pixelCount * 4);
    this.hoverReader = new RangeReader(resources, 'terrain-hover', 8);
  }

  /**
   * Copies a whole buffer back through a shared ring (lazily created, one pixel-count float
   * buffer large) for one-off verification. Returns false when a read is in flight; retry later.
   */
  readBulk(
    commandEncoder: CommandEncoder,
    buffer: Buffer,
    onBytes: (bytes: Uint8Array) => void
  ): boolean {
    this.bulkReader ??= new RangeReader(this.resources, 'terrain-bulk', this.grid.pixelCount * 4);
    return this.bulkReader.request(
      commandEncoder,
      [{buffer, offset: 0, size: this.grid.pixelCount * 4}],
      bytes => {
        if (!this.destroyed) onBytes(bytes);
      }
    );
  }

  /** Registers a stage so `markAllDirty` reaches it. */
  registerStage(stage: Stage): void {
    this.stages.add(stage);
  }

  /** Forgets a stage (call before releasing it). */
  unregisterStage(stage: Stage): void {
    this.stages.delete(stage);
  }

  /** A build that displays `buffer` computed by `stage` (shared stages stay owned by the caller). */
  createBuild(
    id: string,
    stage: Stage,
    buffer: Buffer,
    format: 'float32' | 'uint32' | 'rgba8',
    palette?: readonly Rgb[]
  ): ProductBuild {
    return {
      id,
      configKey: '',
      stage,
      value: {buffer, format},
      release: () => {},
      palette,
      paintGraph: null
    };
  }

  /** A product that is a buffer someone else fills (for example the decoded elevation itself). */
  createStaticBuild(
    id: string,
    buffer: Buffer,
    format: 'float32' | 'uint32' | 'rgba8',
    palette?: readonly Rgb[]
  ): ProductBuild {
    return this.createBuild(id, new Stage(`${id}-static`, [], () => {}), buffer, format, palette);
  }

  /** Elevation band (buffer plus validity) imported into `graph`. */
  importElevation(graph: GPUCommandGraph<void>) {
    return {
      id: 'elevation',
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: importGraphBuffer(
          graph,
          'elevation',
          this.elevationBuffer,
          'float32',
          this.grid.pixelCount
        )
      },
      validity: importGraphBuffer(
        graph,
        'validity',
        this.validityBuffer,
        'uint32',
        this.grid.pixelCount
      )
    };
  }

  /** Creates a builder for a product. */
  builder(id: string): ProductBuilder {
    return new ProductBuilder(this, id);
  }

  /** Marks the elevation (and validity) buffers changed: every product recomputes when shown. */
  elevationChanged(): void {
    this.elevationVersion++;
    this.statsDirty = true;
    this.hoverCache = null;
  }

  /** Looks up the build cached for a product id and configuration key. */
  getBuild(id: string, configKey: string): ProductBuild | undefined {
    const build = this.builds.get(id);
    return build && build.configKey === configKey ? build : undefined;
  }

  /** Registers a freshly built product, releasing the previous build of the same product. */
  addBuild(build: ProductBuild, configKey: string): ProductBuild {
    build.configKey = configKey;
    const previous = this.builds.get(build.id);
    if (previous && previous !== build) {
      if (this.current === previous) this.current = null;
      this.releaseBuild(previous);
    }
    this.builds.set(build.id, build);
    return build;
  }

  private releaseBuild(build: ProductBuild): void {
    if (build.paintGraph) this.resources.release(build.paintGraph);
    build.paintGraph = null;
    build.release();
  }

  /** Makes a product the displayed one. */
  activate(build: ProductBuild, paint: Partial<PaintSpec>): void {
    if (!build.paintGraph && build.value.format !== 'rgba8') {
      const format = build.value.format;
      build.paintGraph = createColorizeGraph(
        this.resources,
        this.device,
        build.id,
        {
          buffer: build.value.buffer,
          format,
          length: this.grid.pixelCount
        },
        this.paintParameters,
        this.colorsBuffer,
        build.palette
      );
    }
    this.current = build;
    this.hoverCache = null;
    this.setPaint(paint);
    this.statsDirty = build.value.format !== 'rgba8';
    this.lastComputeTime = performance.now();
  }

  /** Updates the paint (ramp, range, alpha); only the colorize pass re-runs. */
  setPaint(paint: Partial<PaintSpec>): void {
    this.paint = {...this.paint, ...paint};
    writePaint(this.paintParameters, this.paint);
    this.paintDirty = true;
  }

  /** The active build. */
  get active(): ProductBuild | null {
    return this.current;
  }

  /** Marks the active product's parameters changed: it re-encodes (rate limited when heavy). */
  markDirty(build: ProductBuild | null = this.current): void {
    if (!build) return;
    build.stage.markDirty();
    this.statsDirty = true;
    this.lastComputeTime = performance.now();
  }

  /** Marks every cached product dirty, for settings shared by all products (for example zFactor). */
  markAllDirty(): void {
    for (const stage of this.stages) stage.markDirty();
    this.statsDirty = true;
    this.lastComputeTime = performance.now();
  }

  /** Builds the hillshade underlay: relief shading is computed once per elevation version. */
  enableUnderlay(): void {
    if (this.underlay) return;
    const {grid, resources, device} = this;
    const settings = resources.createParameterBuffer(
      'underlay-settings',
      'float32',
      GPU_RELIEF_SHADING_PARAMETER_LENGTH
    );
    settings.write(
      getGPUReliefShadingParameterValues({
        ...grid.cellSettings,
        lights: 'mdow',
        lightWeighting: 'aspect',
        hillshadeStrength: 1,
        skyViewStrength: 0,
        textureShadeStrength: 0,
        tintStrength: 0,
        exposure: 1,
        elevationStops: []
      })
    );
    const hillshade = resources.createBuffer('underlay-hillshade', grid.pixelCount * 4);
    const colors = resources.createBuffer('underlay-colors', grid.pixelCount * 4);
    const paintParameters = resources.createParameterBuffer(
      'underlay-paint',
      'float32',
      PAINT_PARAMETER_LENGTH
    );
    writePaint(paintParameters, {mode: 'ramp', ramp: 'grayscale', low: 0.05, high: 0.95, alpha: 1});
    const graph = new GPUCommandGraph<void>(device, {id: 'terrain-underlay'});
    graph.add(
      new GPUReliefShading({
        id: 'underlay-shading',
        width: grid.width,
        height: grid.height,
        elevation: this.importElevation(graph),
        settings: settings.importToGraph(graph),
        cellSizeMode: 'web-mercator',
        rowDirection: 'south',
        hillshade: importGraphBuffer(graph, 'hillshade', hillshade, 'float32', grid.pixelCount)
      })
    );
    const compiled = resources.track(graph.compile());
    const paintGraph = createColorizeGraph(
      resources,
      device,
      'underlay',
      {buffer: hillshade, format: 'float32', length: grid.pixelCount},
      paintParameters,
      colors
    );
    this.underlay = {graphs: [compiled], paintGraph, colors, version: -1};
  }

  /** Every compiled graph currently alive for the displayed product (for "Under the hood"). */
  getCompiledGraphs(): CompiledGPUCommandGraph<never>[] {
    const graphs: CompiledGPUCommandGraph<void>[] = [];
    if (this.underlay) graphs.push(...this.underlay.graphs, this.underlay.paintGraph);
    for (const stage of this.stages) graphs.push(...stage.graphs);
    for (const build of this.builds.values()) {
      if (build.paintGraph) graphs.push(build.paintGraph);
    }
    return graphs as unknown as CompiledGPUCommandGraph<never>[];
  }

  /** Times the active product's graphs outside the frame; returns milliseconds per graph. */
  async measure(): Promise<{id: string; milliseconds: number; method: string}[]> {
    const build = this.current;
    if (!build) return [];
    const results: {id: string; milliseconds: number; method: string}[] = [];
    const graphs = [...collectStages(build.stage)].flatMap(stage => stage.graphs);
    for (const [index, graph] of graphs.entries()) {
      const timing = await measureCompiledGraph(this.device, graph, {
        parameters: undefined,
        completionBuffer: build.value.buffer,
        runs: 5,
        warmUpRuns: 1,
        repetitions: 2,
        signal: this.ctx.signal
      });
      results.push({
        id: `${build.id}-${index}`,
        milliseconds: timing.milliseconds,
        method: timing.method
      });
    }
    return results;
  }

  /** Writes parameters, encodes the dirty graphs and schedules the throttled readbacks. */
  encode(commandEncoder: CommandEncoder): void {
    if (this.destroyed) return;
    const now = performance.now();
    if (this.underlay && this.underlay.version !== this.elevationVersion) {
      for (const graph of this.underlay.graphs)
        graph.encode(commandEncoder, {parameters: undefined});
      this.underlay.paintGraph.encode(commandEncoder, {parameters: undefined});
      this.underlay.version = this.elevationVersion;
    }
    const build = this.current;
    if (!build) return;
    if (build.stage.run(commandEncoder, now, this.elevationVersion)) {
      this.paintDirty = true;
      this.statsDirty = true;
      this.lastComputeTime = now;
    }
    if (this.paintDirty && build.paintGraph) {
      build.paintGraph.encode(commandEncoder, {parameters: undefined});
      this.paintDirty = false;
    }
    const settled =
      !build.stage.isPending(this.elevationVersion) &&
      now - this.lastComputeTime > STATS_SETTLE_MILLISECONDS;
    if (this.statsDirty && settled && !this.paintDirty && build.value.format !== 'rgba8') {
      const requested = this.statsReader.request(
        commandEncoder,
        [{buffer: build.value.buffer, offset: 0, size: this.grid.pixelCount * 4}],
        bytes => {
          if (this.destroyed || this.current !== build) return;
          if (build.value.format !== 'rgba8') {
            this.onStats?.(build.id, computeStats(bytes, build.value.format));
          }
        }
      );
      if (requested) this.statsDirty = false;
    }
    if (
      settled &&
      this.hoverIndex >= 0 &&
      !this.hoverReader.pending &&
      build.value.format !== 'rgba8'
    ) {
      const cache = this.hoverCache;
      if (!cache || cache.index !== this.hoverIndex || cache.version !== this.elevationVersion) {
        const index = this.hoverIndex;
        const version = this.elevationVersion;
        this.hoverReader.request(
          commandEncoder,
          [
            {buffer: build.value.buffer, offset: index * 4, size: 4},
            {buffer: this.elevationBuffer, offset: index * 4, size: 4}
          ],
          bytes => {
            if (this.current !== build) return;
            const words = new Uint32Array(bytes.buffer, bytes.byteOffset, 2);
            const value =
              build.value.format === 'uint32'
                ? words[0]
                : new Float32Array(bytes.buffer, bytes.byteOffset, 1)[0];
            const elevation = new Float32Array(bytes.buffer, bytes.byteOffset + 4, 1)[0];
            const column = index % this.grid.width;
            const row = Math.floor(index / this.grid.width);
            this.hoverCache = {
              index,
              version,
              buildId: build.id,
              text: this.describeHover({value, elevation, column, row})
            };
          }
        );
      }
    }
  }

  /** Tooltip text for a pointer event (the probe resolves one frame later). */
  getTooltip(event: ScenePointerEvent): string | null {
    if (!event.coordinate) {
      this.hoverIndex = -1;
      return null;
    }
    const pixel = this.grid.getPixel(event.coordinate[0], event.coordinate[1]);
    if (!pixel) {
      this.hoverIndex = -1;
      return null;
    }
    this.hoverIndex = pixel[1] * this.grid.width + pixel[0];
    const cache = this.hoverCache;
    if (cache && cache.version === this.elevationVersion && cache.buildId === this.current?.id) {
      // The probe lags the pointer by a frame; a value from a neighbouring pixel is still honest.
      const columnDelta = Math.abs((cache.index % this.grid.width) - pixel[0]);
      const rowDelta = Math.abs(Math.floor(cache.index / this.grid.width) - pixel[1]);
      return columnDelta <= 3 && rowDelta <= 3 ? cache.text : null;
    }
    return null;
  }

  /** Layers: underlay, then the product (optionally lit). */
  getLayers(options: SessionLayerOptions): Layer[] {
    const {grid} = this;
    const layers: Layer[] = [];
    const common = {
      coordinateOrigin: [grid.origin[0], grid.origin[1], 0] as [number, number, number],
      gridSize: [grid.width, grid.height] as [number, number],
      bounds: grid.bounds
    };
    if (options.underlay && this.underlay) {
      layers.push(
        new ColorRasterLayer({
          ...common,
          id: 'terrain-underlay',
          colors: this.underlay.colors,
          alpha: options.underlayAlpha
        })
      );
    }
    if (this.current && options.showProduct !== false) {
      layers.push(
        new ColorRasterLayer({
          ...common,
          id: 'terrain-product',
          colors:
            this.current.value.format === 'rgba8' ? this.current.value.buffer : this.colorsBuffer,
          alpha: options.alpha,
          light: options.light?.buffer ?? null,
          lightStrength: options.light?.strength ?? 0,
          lightGain: options.light?.gain ?? 1,
          lightFloor: options.light?.floor ?? 0
        })
      );
    }
    return layers;
  }

  /** Stops readbacks; the caller destroys `resources`. */
  destroy(): void {
    this.destroyed = true;
    this.statsReader.stop();
    this.hoverReader.stop();
    this.bulkReader?.stop();
  }
}

/** Histogram and moments of a raster read back as bytes. */
export function computeStats(bytes: Uint8Array, format: 'float32' | 'uint32'): ValueStats {
  if (format === 'uint32') {
    const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2);
    const counts = new Float64Array(MAXIMUM_CLASSES);
    let total = 0;
    for (const word of words) {
      if (word > 0 && word < MAXIMUM_CLASSES) {
        counts[word]++;
        total++;
      }
    }
    return {kind: 'classes', total, counts};
  }
  const values = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2);
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let count = 0;
  for (const value of values) {
    if (Number.isFinite(value)) {
      if (value < min) min = value;
      if (value > max) max = value;
      sum += value;
      count++;
    }
  }
  const histogram = new Uint32Array(HISTOGRAM_BINS);
  if (count > 0 && max > min) {
    const scale = (HISTOGRAM_BINS - 1) / (max - min);
    for (const value of values) {
      if (Number.isFinite(value)) histogram[Math.floor((value - min) * scale)]++;
    }
  } else {
    histogram[0] = count;
  }
  return {
    kind: 'float',
    count,
    min,
    max,
    mean: count > 0 ? sum / count : Number.NaN,
    histogram,
    quantile(fraction) {
      if (count === 0) return Number.NaN;
      const target = Math.min(Math.max(fraction, 0), 1) * count;
      let seen = 0;
      for (let bin = 0; bin < HISTOGRAM_BINS; bin++) {
        seen += histogram[bin];
        if (seen >= target) return min + ((bin + 0.5) / HISTOGRAM_BINS) * (max - min);
      }
      return max;
    }
  };
}

/** A stage and everything it depends on, dependencies first. */
function collectStages(stage: Stage, found: Set<Stage> = new Set()): Set<Stage> {
  for (const dependency of stage.dependencies) collectStages(dependency, found);
  found.add(stage);
  return found;
}
