// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUEdgeModel,
  resolveGPUSpatialContext,
  validateGPUStatusPort,
  type GPUPolygonGeometryFields,
  type GPUSpatialContext,
  type GPUStatusScalar
} from '../../analysis-core/contracts/index';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  createLinePrefixNode,
  createSegmentizeCountNode,
  createSegmentizeEmitNode,
  createLineOffsetsPublishNode
} from '../line-segmentize/line-segmentize-kernels';
import {
  createPolygonStageCountNode,
  createPolygonStageEmitNode,
  createPolygonStageOffsetsNode,
  createPolygonPublishNode
} from '../rectangle-clip/rectangle-clip-kernels';
import {GPU_LON_LAT_POLYGON_PREPARATION_PARAMETER_LENGTH} from './lon-lat-polygon-preparation-parameters';
import {
  createPreparationInitializeNode,
  createPreparationClassifyNode,
  createPreparationFeatureNode,
  createPreparationCapNode,
  createPreparationOffsetsNode,
  createPreparationNormalizeNode,
  createPreparationTopologyNode,
  createPreparationMergeCountNode,
  createPreparationMergeEmitNode,
  createPreparationStatusNode
} from './lon-lat-polygon-preparation-kernels';

const OPERATION = 'GPULonLatPolygonPreparation';

/** Fixed doubled topology and bounded, linear-degree polygon positions. */
export type GPULonLatPolygonPreparationOutput = {
  /** Vertex and transient stream capacity; written longitudes are in [-180, 180]. */
  positions: GraphDataView<'float32x2'>;
  /** F + 1 offsets; each input feature retains its row. */
  featureOffsets: GraphDataView<'uint32'>;
  /** 2P + 1 offsets, main polygons then east polygons within each feature. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** 2R + 1 offsets; empty copies preserve ring roles. */
  ringOffsets: GraphDataView<'uint32'>;
  /** Per input ring: crossing, capped, ambiguous, over-wide, invalid coordinate, clamped bits. */
  ringFlags?: GraphDataView<'uint32'>;
  /** GPU-written completeness and validity diagnostics, reset on every encoding. */
  status: {
    /** Vertices written. */
    count?: GPUStatusScalar;
    /** One when any intermediate or final stream exceeded vertex capacity; offsets remain valid. */
    overflow: GPUStatusScalar;
    /**
     * Exact final vertex count unless an intermediate stream overflowed; then a conservative
     * bound of 8 * (unclamped dense count + 3 * input ring count), covering both clipped copies.
     * Size `output.positions` from `requiredCount`; transient memory scales with it.
     */
    requiredCount?: GPUStatusScalar;
    /** Number of features emptied because any ring was invalid or rejected. */
    invalidCount?: GPUStatusScalar;
    /** One for clamped great-circle densification or a disabled great-circle degree step. */
    approximation?: GPUStatusScalar;
  };
};

/** Construction-time topology/options and per-frame GPU inputs for polygon preparation. */
export type GPULonLatPolygonPreparationProps = GPUPolygonGeometryFields & {
  /** Node prefix; defaults to 'lon-lat-polygon-preparation'. */
  id?: string;
  /** Longitude/latitude with metric none or great-circle; omitted edge model means linear. */
  spatialContext?: GPUSpatialContext;
  /** Four float32 words from getGPULonLatPolygonPreparationParameterValues. */
  parameters: GraphDataView<'float32'>;
  /** Maximum pieces per original edge, an integer in [1, 65536]; defaults to 1024. */
  maximumPieceCount?: number;
  /** Split at the antimeridian (default), or reject crossing features. */
  antimeridian?: 'split' | 'reject';
  /** Cap at the nearer pole (default), or reject pole-enclosing features. */
  poleRings?: 'cap' | 'reject';
  /** Caller-owned bounded output and optional diagnostics. */
  output: GPULonLatPolygonPreparationOutput;
};

/**
 * Prepares short-way longitude/latitude polygons entirely on the GPU: densify, classify, cap,
 * normalize and clip. Positions always retain wrapped longitude plus integer turns until final
 * publication. Each feature retains its row; each polygon/ring has main and east copies.
 * Nearer-pole capping is a heuristic; represent the complementary area with a world shell and
 * this ring as a hole. Great-circle arcs through a pole are ambiguous in v1.
 * Size `output.positions` from `requiredCount`; transient memory scales with it. Intermediate
 * overflow reports a conservative resize bound; without intermediate overflow the count is exact.
 */
export class GPULonLatPolygonPreparation implements GPUCommandNodeProducer {
  /** Prefix for command and transient identifiers. */
  readonly id: string;
  /** Validated construction-time properties. */
  readonly props: GPULonLatPolygonPreparationProps;
  /** Output context: input metric, units and datum, with linear edges. */
  readonly spatialContext: GPUSpatialContext;
  private readonly inputContext: GPUSpatialContext;

  constructor(props: GPULonLatPolygonPreparationProps) {
    this.id = props.id ?? 'lon-lat-polygon-preparation';
    this.props = props;
    const {id} = this;
    this.inputContext = resolveGPUSpatialContext(id, props.spatialContext, {
      coordinateSpaces: ['longitude-latitude'],
      metrics: ['none', 'great-circle'],
      edgeModels: ['linear', 'great-circle']
    });
    this.spatialContext = {...this.inputContext, edgeModel: 'linear'};
    const maximumPieceCount = props.maximumPieceCount ?? 1024;
    if (
      !Number.isInteger(maximumPieceCount) ||
      maximumPieceCount < 1 ||
      maximumPieceCount > 65536
    ) {
      throw new Error(`${id} maximumPieceCount must be an integer in [1, 65536]`);
    }
    if (props.antimeridian !== undefined && !['split', 'reject'].includes(props.antimeridian)) {
      throw new Error(`${id} antimeridian must be split or reject`);
    }
    if (props.poleRings !== undefined && !['cap', 'reject'].includes(props.poleRings)) {
      throw new Error(`${id} poleRings must be cap or reject`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    for (const name of ['featureOffsets', 'polygonOffsets', 'ringOffsets'] as const) {
      validatePackedView(props[name], ['uint32'], `${id} ${name}`);
      if (props[name].length < 2) {
        throw new Error(`${id} ${name} needs two rows`);
      }
      validatePackedView(props.output[name], ['uint32'], `${id} output.${name}`);
      const length = name === 'featureOffsets' ? props[name].length : 2 * props[name].length - 1;
      if (props.output[name].length !== length) {
        throw new Error(`${id} output.${name} must hold ${length} rows`);
      }
    }
    if (props.output.ringFlags) {
      validatePackedView(props.output.ringFlags, ['uint32'], `${id} output.ringFlags`);
      if (props.output.ringFlags.length !== props.ringOffsets.length - 1) {
        throw new Error(`${id} output.ringFlags must hold ${props.ringOffsets.length - 1} rows`);
      }
    }
    validatePackedView(props.output.positions, ['float32x2'], `${id} output.positions`);
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LON_LAT_POLYGON_PREPARATION_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters must hold 4 float32 values`);
    }
    validateGPUStatusPort(id, props.output.status);
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
  }

  /**
   * Builds fresh transient storage and deterministic GPU stages without reading input data.
   * Zero output capacity publishes diagnostics without binding the empty output position view.
   */
  getCommandNodes<Parameters>(graph: GPUCommandGraph<Parameters>): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    const maximumPieceCount = props.maximumPieceCount ?? 1024;
    const ringCount = props.ringOffsets.length - 1;
    const capacity = output.positions.length;
    const words = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const positions = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32x2', length);
    const turns = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'sint32', length);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const scan = (
      name: string,
      input: GraphDataView<'uint32'>,
      result: GraphDataView<'uint32'>
    ) => {
      nodes.push(
        ...new GPUScan({
          id: `${id}-${name}`,
          input,
          output: result,
          mode: 'exclusive'
        }).getCommandNodes(graph)
      );
    };
    // Unclamped totals: dense, cap, main left/right, east left/right. The status node
    // repacks these as four publisher totals plus the conservative required count.
    const stageFlags = words('stage-totals', 6);
    const denseTotal = graph.createDataView(stageFlags.buffer, {format: 'uint32', length: 1});
    const diagnostics = words('diagnostics', 2);
    const approximation = graph.createDataView(diagnostics.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: Uint32Array.BYTES_PER_ELEMENT
    });
    const rectangle = createTransientView(graph, `${id}-rectangle`, 'float32', 4);
    const metadata = words('metadata', 4 * ringCount);
    const rowTurns = turns('row-turns', props.positions.length);
    const denseCounts = words('dense-counts', props.positions.length);
    const denseStarts = words('dense-starts', props.positions.length);
    const densePositions = positions('dense-positions', capacity);
    const denseTurns = turns('dense-turns', capacity);
    const denseOffsets = words('dense-offsets', ringCount + 1);
    const edgeModel = getGPUEdgeModel(this.inputContext);
    const interpolation = edgeModel === 'linear' ? 'linear-degrees' : 'great-circle';
    const segmentInputs = {
      operation: OPERATION,
      positions: props.positions,
      lineOffsets: props.ringOffsets,
      coordinateSystem: 'spherical' as const,
      radius: 1,
      closedLines: true,
      polygonEdges: true
    };
    nodes.push(
      createPreparationInitializeNode(graph, {
        id: `${id}-initialize`,
        diagnostics,
        rectangle,
        stageFlags
      }),
      createLinePrefixNode(graph, {...segmentInputs, id: `${id}-prefix`, rowTurns}),
      createSegmentizeCountNode(graph, {
        ...segmentInputs,
        id: `${id}-dense-count`,
        parameters: props.parameters,
        maximumPieceCount,
        pieceCountModel: 'degree-step',
        approximation: edgeModel === 'great-circle' ? approximation : undefined,
        interpolation,
        counts: denseCounts
      })
    );
    scan('dense-scan', denseCounts, denseStarts);
    nodes.push(
      createSegmentizeEmitNode(graph, {
        ...segmentInputs,
        id: `${id}-dense-emit`,
        counts: denseCounts,
        starts: denseStarts,
        rowTurns,
        interpolation,
        outputPositions: densePositions,
        outputTurns: denseTurns
      }),
      createLineOffsetsPublishNode(graph, {
        id: `${id}-dense-offsets-publish`,
        operation: OPERATION,
        lineOffsets: props.ringOffsets,
        lineCount: ringCount,
        rowCount: props.positions.length,
        counts: denseCounts,
        starts: denseStarts,
        outputLineOffsets: denseOffsets,
        capacity,
        count: words('dense-count', 1),
        overflow: words('dense-overflow', 1),
        requiredCount: denseTotal
      })
    );
    const capCounts = words('cap-counts', ringCount);
    const capStarts = words('cap-starts', ringCount);
    const capPositions = positions('cap-positions', capacity);
    const capTurns = turns('cap-turns', capacity);
    const capOffsets = words('cap-offsets', ringCount + 1);
    nodes.push(
      createPreparationClassifyNode(graph, {
        id: `${id}-classify`,
        positions: props.positions,
        ringOffsets: props.ringOffsets,
        densePositions,
        denseOffsets,
        parameters: props.parameters,
        metadata,
        counts: capCounts,
        diagnostics,
        edgeModel,
        maximumPieceCount,
        antimeridian: props.antimeridian ?? 'split',
        poleRings: props.poleRings ?? 'cap'
      }),
      createPreparationFeatureNode(graph, {
        id: `${id}-features`,
        featureOffsets: props.featureOffsets,
        polygonOffsets: props.polygonOffsets,
        metadata,
        counts: capCounts,
        diagnostics
      })
    );
    scan('cap-scan', capCounts, capStarts);
    nodes.push(
      createPreparationCapNode(graph, {
        id: `${id}-cap`,
        positions: densePositions,
        turns: denseTurns,
        offsets: denseOffsets,
        metadata,
        starts: capStarts,
        counts: capCounts,
        outputPositions: capPositions,
        outputTurns: capTurns
      }),
      createPreparationOffsetsNode(graph, {
        id: `${id}-cap-offsets-publish`,
        starts: capStarts,
        counts: capCounts,
        offsets: capOffsets,
        capacity,
        stageFlags,
        stage: 1
      }),
      createPreparationNormalizeNode(graph, {
        id: `${id}-normalize`,
        positions: capPositions,
        turns: capTurns,
        ringOffsets: capOffsets,
        polygonOffsets: props.polygonOffsets
      })
    );
    const copies = [];
    for (let copy = 0; copy < 2; copy++) {
      let inputPositions = capPositions;
      let inputTurns = capTurns;
      let inputOffsets = capOffsets;
      const positionBuffers = [
        positions(`clip-${copy}-positions-a`, capacity),
        positions(`clip-${copy}-positions-b`, capacity)
      ];
      const turnBuffers = [
        turns(`clip-${copy}-turns-a`, capacity),
        turns(`clip-${copy}-turns-b`, capacity)
      ];
      const offsetBuffers = [
        words(`clip-${copy}-offsets-a`, ringCount + 1),
        words(`clip-${copy}-offsets-b`, ringCount + 1)
      ];
      // Classification already validates latitudes; only longitude boundaries can clip.
      for (let stage = 0; stage < (capacity === 0 ? 0 : 2); stage++) {
        const stagePositions = positionBuffers[stage % 2];
        const stageTurns = turnBuffers[stage % 2];
        const stageOffsets = offsetBuffers[stage % 2];
        const counts = words(`clip-${copy}-${stage}-counts`, inputPositions.length);
        const starts = words(`clip-${copy}-${stage}-starts`, inputPositions.length);
        const inputs = {
          operation: OPERATION,
          boundary: stage,
          positions: inputPositions,
          positionTurns: inputTurns,
          longitudeCopy: stage === 0 ? copy : 0,
          ringOffsets: inputOffsets,
          parameters: rectangle
        };
        nodes.push(
          createPolygonStageCountNode(graph, {
            ...inputs,
            id: `${id}-clip-${copy}-${stage}-count`,
            counts
          })
        );
        scan(`clip-${copy}-${stage}-scan`, counts, starts);
        nodes.push(
          createPolygonStageEmitNode(graph, {
            ...inputs,
            id: `${id}-clip-${copy}-${stage}-emit`,
            starts,
            outputPositions: stagePositions,
            outputTurns: stageTurns
          }),
          createPolygonStageOffsetsNode(graph, {
            id: `${id}-clip-${copy}-${stage}-offsets`,
            operation: OPERATION,
            capacity,
            inputOffsets,
            starts,
            counts,
            outputOffsets: stageOffsets,
            stageFlags,
            stage: 2 + copy * 2 + stage
          })
        );
        inputPositions = stagePositions;
        inputTurns = stageTurns;
        inputOffsets = stageOffsets;
      }
      copies.push({positions: inputPositions, turns: inputTurns, offsets: inputOffsets});
    }
    const ringMap = words('ring-map', 2 * ringCount);
    const counts = words('merge-counts', 2 * ringCount);
    const starts = words('merge-starts', 2 * ringCount);
    nodes.push(
      createPreparationTopologyNode(graph, {
        id: `${id}-topology`,
        featureOffsets: props.featureOffsets,
        polygonOffsets: props.polygonOffsets,
        outputFeatureOffsets: output.featureOffsets,
        outputPolygonOffsets: output.polygonOffsets,
        ringMap
      })
    );
    copies.forEach((source, copy) =>
      nodes.push(
        createPreparationMergeCountNode(graph, {
          ...source,
          id: `${id}-merge-${copy}-count`,
          ringMap,
          counts,
          copy
        })
      )
    );
    scan('merge-scan', counts, starts);
    // An empty caller-owned output need not back a full float32x2 storage binding.
    // Counts and status still run so zero capacity reports overflow and the resize bound.
    if (capacity > 0) {
      copies.forEach((source, copy) =>
        nodes.push(
          createPreparationMergeEmitNode(graph, {
            ...source,
            id: `${id}-merge-${copy}-emit`,
            ringMap,
            counts,
            starts,
            output: output.positions,
            copy
          })
        )
      );
    }
    nodes.push(
      createPreparationOffsetsNode(graph, {
        id: `${id}-output-offsets`,
        starts,
        counts,
        offsets: output.ringOffsets,
        capacity: output.positions.length
      }),
      createPreparationStatusNode(graph, {
        id: `${id}-status`,
        counts,
        starts,
        stageFlags,
        capacity,
        metadata,
        diagnostics,
        ringFlags: output.ringFlags,
        invalidCount: output.status.invalidCount,
        approximation: output.status.approximation
      }),
      createPolygonPublishNode(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        capacity: output.positions.length,
        ringCount: 2 * ringCount,
        stageFlags,
        count: output.status.count ?? words('output-count', 1),
        overflow: output.status.overflow,
        requiredCount: output.status.requiredCount,
        requiredCountIndex: 4
      })
    );
    return nodes;
  }

  private getInputViews(): GraphDataView[] {
    const {props} = this;
    return [
      props.positions,
      props.featureOffsets,
      props.polygonOffsets,
      props.ringOffsets,
      props.parameters
    ];
  }

  private getOutputViews(): GraphDataView[] {
    const {output} = this.props;
    return [
      output.positions,
      output.featureOffsets,
      output.polygonOffsets,
      output.ringOffsets,
      output.ringFlags,
      ...Object.values(output.status)
    ].filter((view): view is NonNullable<typeof view> => Boolean(view));
  }
}
