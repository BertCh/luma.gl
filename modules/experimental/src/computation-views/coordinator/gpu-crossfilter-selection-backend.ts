// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  type GraphDataView,
  GraphVectorView,
  GPUReduction,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUTransferMask} from '@luma.gl/gpgpu';
import type {GPUCrossfilter} from '../../gpu-crossfilter/gpu-crossfilter';
import type {GPUCrossfilterMask} from '../../gpu-crossfilter/gpu-selection';
import type {KeySpaceId, SelectionClause} from '../types';
import type {GPUSelectionBackend, GPUSelectionMaskHandle} from './gpu-selection-backend';

/** Source-to-dimension mapping for one row domain; writes must complete before the hook returns. */
export type GPUCrossfilterSelectionKeySpace = {
  keyCount: number;
  dimensions: Record<string, string>;
  /** Writes the mapped external dimension. The optional source disambiguates multiple brushes. */
  writeMask?: (keySpace: KeySpaceId, keys: readonly number[], source?: string) => void;
};

/** Fixed GPUTransferMask binding; output must be persistent and belong to the crossfilter graph. */
export type GPUCrossfilterSelectionRelation = {
  from: KeySpaceId;
  to: KeySpaceId;
  transfer: GPUTransferMask;
  output: GraphDataView<'uint32'>;
};

/** Construction contract for a dedicated, not-yet-compiled crossfilter graph. */
export type GPUCrossfilterSelectionBackendProps = {
  crossfilter: GPUCrossfilter;
  keySpaces: Record<KeySpaceId, GPUCrossfilterSelectionKeySpace>;
  relations?: Record<string, GPUCrossfilterSelectionRelation>;
  minimumKeyCount?: number;
};

/**
 * GPU brushing for key spaces above the CPU budget (65,536), including spatial concepts pages.
 *
 * Call crossfilter.addToGraph() first, then construct this adapter before compiling. The adapter
 * owns compilation/submission of that dedicated void-parameter graph and its packed count buffer.
 * Publish persistent mask storage (explicit outputMask or skipUnchangedDimensions: true).
 * Call destroy() before destroying the caller-owned crossfilter and imported buffers.
 *
 * A source maps to dimensions[source]. For excludeSource, register a crossfilter view with that
 * dimension and includeOwnSelection: false (the default). Its getViewMask() is the effective
 * mask; undefined means all rows. Without exclusion, the crossfilter.mask is used. Spaces with
 * no dimensions are relation targets and use their unique incoming relation output. One
 * crossfilter shares one row topology; distinct target topologies belong to transfer outputs.
 *
 * Handle.buffer is a GraphDataView or GraphVectorView, preserving offsets/chunk boundaries and
 * allowing downstream graph consumers without readback. Relation sources must bind that exact
 * logical view. Relations execute in declaration order; declare upstream transfers first.
 * Changes upload only flags/control data. readCounts() performs one packed readback for all
 * spaces; no per-frame mask readback occurs. LadderCoordinator currently needs GPU routing.
 */
export class GPUCrossfilterSelectionBackend implements GPUSelectionBackend {
  readonly kind = 'gpu';
  private readonly props: GPUCrossfilterSelectionBackendProps;
  private readonly minimumKeyCount: number;
  private generation = 0;
  private submittedGeneration = -1;
  private compiled?: CompiledGPUCommandGraph;
  private counts?: Buffer;
  private readonly ownedBuffers: Buffer[] = [];
  private readonly allRows = new Map<string, GraphDataView<'uint32'>>();
  private readonly relationSources = new Map<string, unknown>();
  private destroyed = false;

  /** Borrows the crossfilter, source dimensions and fixed relation bindings. */
  constructor(props: GPUCrossfilterSelectionBackendProps) {
    this.props = props;
    this.minimumKeyCount = props.minimumKeyCount ?? 65537;
  }

  /** Accepts declared key spaces with an exact, finite integer size above the GPU threshold. */
  accepts(keySpace: KeySpaceId, keyCount: number): boolean {
    return (
      Number.isInteger(keyCount) &&
      keyCount >= 0 &&
      keyCount >= this.minimumKeyCount &&
      this.props.keySpaces[keySpace]?.keyCount === keyCount
    );
  }

  /** Uploads explicit keys or updates a range dimension; unsupported predicate/dimension pairs throw. */
  setClause(clause: SelectionClause): void {
    this.assertAvailable();
    const space = this.getSpace(clause.keySpace);
    const dimensionId = space.dimensions[clause.source];
    const dimension = this.props.crossfilter.dimensions.find(
      candidate => candidate.id === dimensionId
    );
    if (!dimension)
      throw new Error(`No crossfilter dimension for source ${clause.source} in ${clause.keySpace}`);
    if (clause.predicate.kind === 'keys' && dimension.kind === 'external') {
      const keys = clause.predicate.keys;
      if (keys.some(key => !Number.isInteger(key) || key < 0 || key >= space.keyCount))
        throw new Error(`Selection keys outside ${clause.keySpace}`);
      if (space.writeMask) space.writeMask(clause.keySpace, keys, clause.source);
      else this.writeKeys(this.props.crossfilter.getDimensionMask(dimensionId), keys);
      this.props.crossfilter.setExternalActive(dimensionId, true);
    } else if (clause.predicate.kind === 'range' && dimension.kind === 'range') {
      this.props.crossfilter.setRange(dimensionId, clause.predicate.range);
    } else {
      throw new Error(
        `Unsupported ${clause.predicate.kind} predicate for ${dimension.kind} dimension ${dimensionId}`
      );
    }
    this.props.crossfilter.invalidate(dimensionId);
    this.generation++;
  }

  /** Deactivates every dimension owned by this source across declared spaces. */
  clearSource(source: string): void {
    this.assertAvailable();
    for (const dimensionId of new Set(
      Object.values(this.props.keySpaces).flatMap(space =>
        space.dimensions[source] ? [space.dimensions[source]] : []
      )
    )) {
      const dimension = this.props.crossfilter.dimensions.find(
        candidate => candidate.id === dimensionId
      );
      if (dimension?.kind === 'external')
        this.props.crossfilter.setExternalActive(dimensionId, false);
      else this.props.crossfilter.clear(dimensionId);
      this.props.crossfilter.invalidate(dimensionId);
      this.generation++;
    }
  }

  /** Submits pending work and returns the source-excluding GPU view, without reading any flags. */
  getMask(keySpace: KeySpaceId, options: {excludeSource?: string} = {}): GPUSelectionMaskHandle {
    const buffer = this.resolveMask(keySpace, options.excludeSource);
    this.submit();
    return {
      keySpace,
      keyCount: this.getSpace(keySpace).keyCount,
      buffer,
      generation: this.generation
    };
  }

  /** Runs the declared fixed-binding transfer (index/CSR, any/count) entirely on the GPU. */
  transferMask(mask: GPUSelectionMaskHandle, relationId: string): GPUSelectionMaskHandle {
    this.assertAvailable();
    const relation = this.props.relations?.[relationId];
    if (!relation || relation.from !== mask.keySpace)
      throw new Error(`Unknown or incompatible selection relation ${relationId}`);
    this.prepare();
    if (
      mask.keyCount !== this.getSpace(relation.from).keyCount ||
      this.relationSources.get(relationId) !== mask.buffer
    )
      throw new Error(`Relation ${relationId} requires its declared source mask view`);
    this.generation++;
    this.submit();
    return {
      keySpace: relation.to,
      keyCount: this.getSpace(relation.to).keyCount,
      buffer: relation.output,
      generation: this.generation
    };
  }

  /** Reads all nonzero selection counts in one packed GPU readback. */
  async readCounts(): Promise<Record<KeySpaceId, number>> {
    this.submit();
    const bytes = await this.counts!.readAsync();
    const counts = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    return Object.fromEntries(
      Object.keys(this.props.keySpaces).map((keySpace, index) => [keySpace, counts[index]])
    );
  }

  /** Releases only adapter-owned compilation and buffers; borrowed inputs remain caller-owned. */
  destroy(): void {
    if (this.destroyed) return;
    this.compiled?.destroy();
    for (const buffer of this.ownedBuffers) buffer.destroy();
    this.destroyed = true;
  }

  private getSpace(keySpace: string): GPUCrossfilterSelectionKeySpace {
    const space = this.props.keySpaces[keySpace];
    if (!space) throw new Error(`Unknown selection key space ${keySpace}`);
    return space;
  }

  private resolveMask(keySpace: string, excludeSource?: string): GPUCrossfilterMask {
    this.assertAvailable();
    const space = this.getSpace(keySpace);
    let mask: GPUCrossfilterMask | undefined;
    if (excludeSource) {
      const dimension = space.dimensions[excludeSource];
      const view =
        dimension &&
        this.props.crossfilter.views.find(
          candidate => candidate.dimension === dimension && candidate.includeOwnSelection !== true
        );
      if (!view)
        throw new Error(`No self-excluding crossfilter view for ${excludeSource} in ${keySpace}`);
      mask = this.props.crossfilter.getViewMask(view.id) ?? this.getAllRows(keySpace);
    } else if (Object.keys(space.dimensions).length) mask = this.props.crossfilter.mask;
    else {
      const incoming = Object.values(this.props.relations ?? {}).filter(
        relation => relation.to === keySpace
      );
      if (incoming.length !== 1)
        throw new Error(
          `Selection space ${keySpace} requires exactly one incoming relation or source dimensions`
        );
      mask = incoming[0].output;
    }
    if (mask.length !== space.keyCount)
      throw new Error(`Crossfilter mask length differs from key space ${keySpace}`);
    const chunks = mask instanceof GraphVectorView ? mask.chunks.map(chunk => chunk.data) : [mask];
    if (chunks.some(chunk => chunk.buffer.transient))
      throw new Error(
        'GPU selection handles require persistent masks; set outputMask or skipUnchangedDimensions'
      );
    return mask;
  }

  private writeKeys(mask: GPUCrossfilterMask, keys: readonly number[]): void {
    const chunks = mask instanceof GraphVectorView ? mask.chunks.map(chunk => chunk.data) : [mask];
    const selected = new Set(keys);
    let firstKey = 0;
    for (const chunk of chunks) {
      const buffer = chunk.buffer.defaultBuffer;
      if (!(buffer instanceof Buffer) || chunk.byteStride !== 4)
        throw new Error('External selection needs imported packed buffers or a writeMask hook');
      const values = new Uint32Array(chunk.length);
      for (let index = 0; index < values.length; index++)
        values[index] = selected.has(firstKey + index) ? 1 : 0;
      buffer.write(values, chunk.byteOffset);
      firstKey += chunk.length;
    }
  }

  private getAllRows(keySpace: string): GraphDataView<'uint32'> {
    let view = this.allRows.get(keySpace);
    if (!view) {
      if (this.compiled)
        throw new Error('Register self-excluding views before compiling the selection graph');
      view = this.importValues(
        `all-${keySpace}`,
        new Uint32Array(this.getSpace(keySpace).keyCount).fill(1)
      );
      this.allRows.set(keySpace, view);
    }
    return view;
  }

  private importValues(name: string, values: Uint32Array): GraphDataView<'uint32'> {
    const graph = this.props.crossfilter.graph;
    const buffer = graph.device.createBuffer({
      id: `${this.props.crossfilter.id}-selection-${name}`,
      data: values,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    this.ownedBuffers.push(buffer);
    const handle = graph.importBuffer(
      {id: buffer.id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format: 'uint32', length: values.length});
  }

  private prepare(): void {
    this.assertAvailable();
    if (this.compiled) return;
    const graph = this.props.crossfilter.graph;
    for (const [relationId, relation] of Object.entries(this.props.relations ?? {})) {
      this.getSpace(relation.from);
      if (
        relation.output.length !== this.getSpace(relation.to).keyCount ||
        relation.output.buffer.transient
      )
        throw new Error(
          `Relation ${relationId} needs persistent output matching its target key space`
        );
      const nodes = relation.transfer.getCommandNodes(graph);
      const target = nodes[0]?.resources?.[0];
      if (!target || !('buffer' in target) || target.buffer !== relation.output)
        throw new Error(`Relation ${relationId} output differs from its transfer target`);
      // GPUTransferMask's scatter reads sourceMask first, followed by the relation buffers.
      const source = nodes[1]?.resources?.[0];
      this.relationSources.set(
        relationId,
        source && 'buffer' in source ? source.buffer : undefined
      );
      graph.add(nodes);
    }
    const keySpaces = Object.keys(this.props.keySpaces);
    const packed = this.importValues('counts', new Uint32Array(keySpaces.length));
    this.counts = this.ownedBuffers[this.ownedBuffers.length - 1];
    keySpaces.forEach((keySpace, index) => {
      const allRows = this.getAllRows(keySpace);
      const mask = this.resolveMask(keySpace);
      graph.add(
        new GPUReduction({
          id: `${this.props.crossfilter.id}-selection-count-${keySpace}`,
          input: allRows,
          mask,
          output: graph.createDataView(packed.buffer, {
            format: 'uint32',
            length: 1,
            byteOffset: index * 4
          }),
          operation: 'sum'
        })
      );
    });
    this.compiled = graph.compile();
  }

  private submit(): void {
    this.prepare();
    if (this.submittedGeneration === this.generation) return;
    const device = this.props.crossfilter.graph.device;
    const encoder = device.createCommandEncoder();
    this.compiled!.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
    this.submittedGeneration = this.generation;
  }

  private assertAvailable(): void {
    if (this.destroyed) throw new Error('GPU selection backend has been destroyed');
  }
}
