// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  type DrawCommandBufferView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPURasterContours,
  type GPURasterBand,
  type GPURasterMetadata
} from '../../gpu-raster/index';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  captureGraphCommandNodes,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainGrid
} from './terrain-analysis-utils';

/** One contour level and its caller-owned outputs, mirroring `GPURasterContoursProps`. */
export type GPUTerrainContourLevel = {
  /** Fixed level (topology) or a one-row float32 view rewritten per frame. Equal samples count as high. */
  level: number | GraphDataView<'float32'>;
  /** Two float32x2 local pixel-center positions per segment. */
  vertices: GraphDataView<'float32x2'>;
  /** One-row `min(required, capacity)` segment count. */
  segmentCount: GraphDataView<'uint32'>;
  /** Optional one-row unclamped segment count. */
  requiredSegmentCount?: GraphDataView<'uint32'>;
  /** Optional one-row per-level overflow. A graph transient is used when omitted. */
  overflow?: GraphDataView<'uint32'>;
  /**
   * Optional non-indexed indirect draw record (16 bytes). All four words (`vertexCount`,
   * `instanceCount`, `firstVertex`, `firstInstance`) are rewritten on every encoding from the
   * GPU-side segment count, which is already clamped to capacity. `drawLayout` picks the shape.
   */
  draw?: DrawCommandBufferView;
  /** Draw record index. */
  drawCommandIndex?: number;
  /**
   * Shape of the draw record written to `draw`. Defaults to `'instanced'`.
   *
   * - `'instanced'`: `[verticesPerInstance, segmentCount, 0, 0]`. One instance per segment, for
   *   instanced-segment renderers (a quad per segment uses 4 or 6 vertices) that read endpoints
   *   by instance index from `vertices`.
   * - `'line-list'`: `[2 * segmentCount, 1, 0, 0]`. One non-instanced `line-list` draw over the
   *   `vertices` buffer used as an ordinary vertex buffer (two vertices per segment).
   */
  drawLayout?: 'instanced' | 'line-list';
  /**
   * Vertices per instance for `drawLayout: 'instanced'`. Defaults to 2, matching the original
   * two-vertex record. Must be a positive integer; rejected with `'line-list'`.
   */
  verticesPerInstance?: number;
  /** Optional segment capacity below `vertices.length / 2`. */
  capacity?: number;
};

/**
 * Properties for {@link GPUTerrainContours}.
 *
 * Topology: grid size, elevation format, the number of levels, numeric level values, capacities,
 * and `metadata`. Per-frame: GPU level views and elevation contents.
 */
export type GPUTerrainContoursProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-contours'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture. */
  elevation: GPURasterBand;
  /** At least one contour level. */
  levels: readonly GPUTerrainContourLevel[];
  /** One-row flag: 1 when any level overflowed in this encoding, else 0. */
  overflow: GraphDataView<'uint32'>;
  /** Forwarded to every level. `pixelInterpretation: 'point'` removes the half-pixel vertex offset. */
  metadata?: GPURasterMetadata;
};

/**
 * Extracts marching-squares contour segments for several elevation levels into bounded buffers.
 *
 * Each level is one `GPURasterContours` pipeline (classify, scan, scatter, publish); one
 * contributor-wide overflow flag reports truncation in any level. Per-level layout errors surface from
 * `GPURasterContours` when the contributor is added to a graph.
 */
export class GPUTerrainContours implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainContoursProps;
  /** Marching-squares cells read one pixel right and below. */
  readonly requiredHalo = 1;

  constructor(props: GPUTerrainContoursProps) {
    this.id = props.id ?? 'terrain-contours';
    this.props = props;
    const {id} = this;
    validateTerrainGrid(id, props.width, props.height);
    if (props.levels.length < 1) {
      throw new Error(`${id} requires at least one contour level`);
    }
    for (const [index, level] of props.levels.entries()) {
      validateContourLevelDraw(id, index, level);
    }
    validatePackedUint32View(props.overflow, `${id} overflow`);
    if (props.overflow.length < 1) {
      throw new Error(`${id} overflow must contain one uint32 row`);
    }
    const otherBuffers = [
      ...getTerrainBandViews(props.elevation),
      ...props.levels.flatMap(level => [
        level.vertices,
        level.segmentCount,
        level.requiredSegmentCount,
        level.overflow,
        typeof level.level === 'number' ? undefined : level.level
      ])
    ]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    if (otherBuffers.includes(props.overflow.buffer)) {
      throw new Error(`${id} outputs must not share buffers with each other or with inputs`);
    }
  }

  /** Returns optional elevation gather, then per level the contour pipeline and an overflow OR. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.overflow,
      ...props.levels.flatMap(level => [
        level.vertices,
        level.segmentCount,
        level.requiredSegmentCount,
        level.overflow,
        typeof level.level === 'number' ? undefined : level.level,
        level.draw?.words
      ])
    ]);
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, false);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    for (const [index, level] of props.levels.entries()) {
      // The default two-vertex instanced record is written by the raster publish pass itself;
      // every other shape is written by a contributor node that reads the clamped segment count.
      const customDraw = level.draw !== undefined && !isDefaultContourDraw(level);
      const levelOverflow =
        level.overflow ?? createTransientView(graph, `${id}-level-${index}-overflow`, 'uint32', 1);
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterContours({
            id: `${id}-level-${index}`,
            width,
            height,
            input: source.band,
            level: level.level,
            vertices: level.vertices,
            segmentCount: level.segmentCount,
            overflow: levelOverflow,
            requiredSegmentCount: level.requiredSegmentCount,
            draw: customDraw ? undefined : level.draw,
            drawCommandIndex: customDraw ? undefined : level.drawCommandIndex,
            capacity: level.capacity,
            metadata: props.metadata
          }).addToGraph(graph)
        )
      );
      if (customDraw && level.draw) {
        nodes.push(createContourDrawNode<Parameters>(graph, id, index, level, level.draw));
      }
      // The level-0 node writes without reading, so every encoding resets the contributor flag.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-overflow-${index}`,
          operation: 'GPUTerrainContours',
          variant: 'overflow',
          bindings: [
            {name: 'levelOverflow', view: levelOverflow, type: 'u32', access: 'read'},
            {name: 'outputOverflow', view: props.overflow, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          body: `let previous = ${index === 0 ? '0u' : 'outputOverflow[outputOverflowOffset]'};
  outputOverflow[outputOverflowOffset] =
    select(0u, 1u, previous != 0u || levelOverflow[levelOverflowOffset] != 0u);`
        })
      );
    }
    return nodes;
  }
}

function isDefaultContourDraw(level: GPUTerrainContourLevel): boolean {
  return (
    (level.drawLayout ?? 'instanced') === 'instanced' && (level.verticesPerInstance ?? 2) === 2
  );
}

function validateContourLevelDraw(id: string, index: number, level: GPUTerrainContourLevel): void {
  const label = `${id} level ${index}`;
  const {draw} = level;
  if (!draw) {
    if (
      level.drawCommandIndex !== undefined ||
      level.drawLayout !== undefined ||
      level.verticesPerInstance !== undefined
    ) {
      throw new Error(`${label} drawCommandIndex, drawLayout and verticesPerInstance require draw`);
    }
    return;
  }
  const layout = level.drawLayout ?? 'instanced';
  if (layout !== 'instanced' && layout !== 'line-list') {
    throw new Error(`${label} drawLayout must be 'instanced' or 'line-list'`);
  }
  if (level.verticesPerInstance !== undefined) {
    if (layout !== 'instanced') {
      throw new Error(`${label} verticesPerInstance requires drawLayout 'instanced'`);
    }
    if (!Number.isSafeInteger(level.verticesPerInstance) || level.verticesPerInstance < 1) {
      throw new Error(`${label} verticesPerInstance must be a positive integer`);
    }
  }
  const commandIndex = level.drawCommandIndex ?? 0;
  if (!Number.isSafeInteger(commandIndex) || commandIndex < 0) {
    throw new Error(`${label} drawCommandIndex must be a non-negative integer`);
  }
  if (draw.type !== 'draw' || draw.recordByteLength !== 16) {
    throw new Error(`${label} draw must be a non-indexed indirect draw record view`);
  }
  if (commandIndex >= draw.capacity) {
    throw new Error(`${label} drawCommandIndex exceeds the indirect draw capacity`);
  }
  validatePackedUint32View(draw.words, `${label} draw words`);
  if (draw.words.buffer !== draw.buffer || draw.words.length < draw.capacity * 4) {
    throw new Error(`${label} draw words must span every draw record`);
  }
}

function createContourDrawNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  index: number,
  level: GPUTerrainContourLevel,
  draw: DrawCommandBufferView
): GPUCommandNode<Parameters> {
  const capacity = level.capacity ?? Math.floor(level.vertices.length / 2);
  const wordOffset = (level.drawCommandIndex ?? 0) * 4;
  const isLineList = level.drawLayout === 'line-list';
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-level-${index}-draw`,
    operation: 'GPUTerrainContours',
    variant: 'draw',
    bindings: [
      {name: 'segmentCount', view: level.segmentCount, type: 'u32', access: 'read'},
      {name: 'drawWords', view: draw.words, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    body: `let count = min(segmentCount[segmentCountOffset], ${capacity}u);
  let base = drawWordsOffset + ${wordOffset}u;
  drawWords[base] = ${isLineList ? '2u * count' : `${level.verticesPerInstance ?? 2}u`};
  drawWords[base + 1u] = ${isLineList ? '1u' : 'count'};
  drawWords[base + 2u] = 0u;
  drawWords[base + 3u] = 0u;`
  });
}
