// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  GPULatticeWeights,
  GPUSpatialWeightsAlgebra,
  type GPUNeighborSearchKernel,
  type GPUNeighborSearchWeightKind,
  type GPUSpatialWeightsCombineRule,
  type GPUSpatialWeightsKernel
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import type {RampName} from '../../engine/ramps';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance} from '../scene';
import {FOCUS_COLORS, getLinkColor, getOutlineColor, ISLAND_COLOR, NO_DATA} from './b4-colors';
import {
  getVariableInfo,
  loadGeography,
  type Geography,
  type GeographyId,
  type VariableId
} from './b4-geography';
import {B4PolygonFillLayer, createGeographyBuffers, type GeographyBuffers} from './b4-layers';
import {
  createAnalysisSpace,
  NO_MODE,
  type AnalysisSpace,
  type SpaceDisplay,
  type SpaceSnapshot,
  type SummaryChoice
} from './b4-space';
import {
  createWeightsCore,
  SLOTS_PER_ROW,
  type WeightsConfig,
  type WeightsCore,
  type WeightsSource,
  type WeightsTransform
} from './b4-weights-core';
import {FOCUS_SLOTS, type WeightsKitSummary, type WeightsMatrixChoice} from './b4-weights-kit';

/** Combination applied on top of the chosen neighbourhood. */
export type WeightsCombine =
  | 'none'
  | 'union'
  | 'intersection'
  | 'difference'
  | 'symmetricDifference'
  | 'higherOrder'
  | 'selfWeight'
  | 'subgraph'
  | 'block';

/** Option state of the spatial-weights scene. */
export type SpatialWeightsOptions = {
  geography: GeographyId;
  variable: VariableId;
  focus: 'typical' | 'corner' | 'island';
  source: WeightsSource | 'lattice';
  snapTolerance: string;
  k: number;
  bandFactor: number;
  knnCapFactor: number;
  weightKind: GPUNeighborSearchWeightKind;
  kernel: GPUNeighborSearchKernel;
  power: number;
  distanceFloorFactor: number;
  rowStandardize: boolean;
  transform: WeightsTransform;
  transformKernel: GPUSpatialWeightsKernel;
  bandwidth: string;
  doubleSum: 'one' | 'rows';
  combine: WeightsCombine;
  partnerK: number;
  weightRule: GPUSpatialWeightsCombineRule;
  order: number;
  cumulative: boolean;
  selfWeight: number;
  populationPercentile: number;
  lagNormalize: boolean;
  includeFocal: boolean;
  focalWeight: string;
  display: SpaceDisplay;
  summary: SummaryChoice;
  ramp: RampName;
  matrix: WeightsMatrixChoice;
  showLinks: boolean;
  showOutlines: boolean;
  latticeCriterion: 'rook' | 'queen';
  latticeRadius: number;
  latticeMask: boolean;
};

/** Slot capacity of the analysed CSR: it must hold block weights (sum of squared group sizes). */
const ANALYSIS_SLOTS: Record<GeographyId, number> = {
  'us-counties': 320_000,
  'chicago-tracts': 48_000
};
/** Focus presets: a typical place, a corner case where queen and rook differ, and an island. */
const FOCUS_POINTS: Record<
  GeographyId,
  Record<SpatialWeightsOptions['focus'], [number, number]>
> = {
  'us-counties': {
    typical: [-87.65, 41.84],
    corner: [-108.35, 36.55],
    island: [-70.05, 41.28]
  },
  'chicago-tracts': {
    typical: [-87.63, 41.88],
    corner: [-87.6107, 41.7774],
    island: [-87.9, 41.98]
  }
};
const LATTICE_COLUMNS: Record<GeographyId, number> = {'us-counties': 110, 'chicago-tracts': 70};
const LATTICE_MAXIMUM_RADIUS = 4;

type Lattice = {
  columns: number;
  rows: number;
  cell: number;
  /** `[minX, minY, maxX, maxY]` of the whole grid. */
  bounds: [number, number, number, number];
  featureOfCell: Int32Array;
  space: AnalysisSpace;
  structureMask: Buffer;
  values: Buffer;
  valueMask: Buffer;
  producers: Map<string, CompiledGPUCommandGraph<void>>;
  overflow: Buffer;
  total: Buffer;
  focusRow: number;
  insideMask: Uint32Array;
};

type World = {
  geography: Geography;
  resources: SpatialAnalysisResources;
  core: WeightsCore;
  partner: ReturnType<WeightsCore['createPartner']>;
  space: AnalysisSpace;
  buffers: GeographyBuffers;
  values: Buffer;
  valueMask: Buffer;
  selfWeights: Buffer;
  subgraphMask: Buffer;
  groups: Buffer;
  algebra: Map<string, CompiledGPUCommandGraph<void>>;
  algebraOverflow: Buffer;
  algebraTotal: Buffer;
  lattice: Lattice | null;
  focusRow: number;
  populationSorted: Float32Array;
  snapshot: SpaceSnapshot | null;
  kitSummary: WeightsKitSummary | null;
  latticeSnapshot: SpaceSnapshot | null;
  latticeKitSummary: WeightsKitSummary | null;
};

/** Lattice cell containing the focus centroid of a world. */
function getLatticeCellOf(
  target: {geography: Geography; focusRow: number},
  minX: number,
  minY: number,
  cell: number,
  columns: number,
  rows: number
): number {
  const x = target.geography.centroids[target.focusRow * 2];
  const y = target.geography.centroids[target.focusRow * 2 + 1];
  const column = Math.min(columns - 1, Math.max(0, Math.floor((x - minX) / cell)));
  const row = Math.min(rows - 1, Math.max(0, Math.floor((y - minY) / cell)));
  return row * columns + column;
}

const geographyCache = new Map<GeographyId, Promise<Geography>>();

/**
 * Spatial weights on real polygons. One `GPUContiguityWeights` pass (queen or rook) or one
 * `GPUNeighborSearch` pass (kNN or distance band on centroids) or `GPULatticeWeights` (a raster
 * grid) writes a CSR; `GPUSpatialWeightsTransform` rewrites the weights, `GPUSpatialWeightsAlgebra`
 * combines neighbourhoods, `GPUSpatialWeightsSummary` and `GPUSpatialWeightsTranspose` diagnose
 * them, and `GPUSpatialLag` and `GPUNeighborhoodSummary` use them. Variants compile on first use
 * and are cached, so a compile-time option rebuilds once and a parameter never does.
 */
export async function createSpatialWeights(
  ctx: SceneContext<SpatialWeightsOptions>
): Promise<SceneInstance<SpatialWeightsOptions>> {
  const {device} = ctx;
  let destroyed = false;
  let world: World | null = null;
  let switchToken = 0;
  let dirty = true;
  let readbackStale = true;

  const getGeography = (id: GeographyId) => {
    let promise = geographyCache.get(id);
    if (!promise) {
      promise = loadGeography(id, ctx.datasets, ctx.signal);
      promise.catch(() => geographyCache.delete(id));
      geographyCache.set(id, promise);
    }
    return promise;
  };

  const getConfig = (geography: Geography): WeightsConfig => {
    const o = ctx.options;
    const source = o.source === 'lattice' ? 'queen' : o.source;
    return {
      source,
      k: o.k,
      snapTolerance: Number(o.snapTolerance),
      bandFactor: o.bandFactor,
      knnCapFactor: o.knnCapFactor,
      weightKind: o.weightKind,
      kernel: o.kernel,
      power: o.power,
      distanceFloor: o.distanceFloorFactor * geography.medianSpacing,
      rowStandardize: o.rowStandardize,
      transform: o.transform,
      transformKernel: o.transformKernel,
      bandwidthFactor: o.bandwidth === 'adaptive' ? 0 : Number(o.bandwidth),
      doubleSum: o.doubleSum
    };
  };

  const getAlgebraKey = (): string | null => {
    const o = ctx.options;
    switch (o.combine) {
      case 'none':
        return null;
      case 'union':
      case 'intersection':
      case 'difference':
      case 'symmetricDifference':
        return `${o.combine}-${o.weightRule}`;
      case 'higherOrder':
        return `higherOrder-${o.order}-${o.cumulative}`;
      default:
        return o.combine;
    }
  };

  const buildWorld = (geography: Geography): World => {
    const resources = new SpatialAnalysisResources(device, `weights-${geography.id}`);
    const rowCount = geography.count;
    const create = (name: string, data: number | Float32Array | Uint32Array) =>
      resources.createBuffer(name, data);
    const core = createWeightsCore({device, resources, id: 'weights-core', geography});
    const analysisSlots = ANALYSIS_SLOTS[geography.id];
    const initialValues = geography.getVariable(ctx.options.variable);
    const values = create('values', Float32Array.from(initialValues));
    const valueMask = create('value-mask', new Uint32Array(rowCount));
    const displayMask = create('display-mask', new Uint32Array(rowCount).fill(1));
    const space = createAnalysisSpace({
      device,
      resources,
      id: 'polygons',
      rows: rowCount,
      slots: analysisSlots,
      unionSlots: analysisSlots,
      positions: core.positions,
      values,
      mask: valueMask,
      displayMask,
      categories: geography.groups,
      drawAllLinks: true,
      producer: {overflow: core.overflow, total: core.total}
    });
    const populationSorted = Float32Array.from(geography.population).sort();
    const focusRow = Math.max(0, geography.pick(...FOCUS_POINTS[geography.id][ctx.options.focus]));
    const created: World = {
      geography,
      resources,
      core,
      partner: core.createPartner('partner'),
      space,
      buffers: createGeographyBuffers(resources, geography, 'weights'),
      values,
      valueMask,
      selfWeights: create('self-weights', new Float32Array(rowCount)),
      subgraphMask: create('subgraph-mask', new Uint32Array(rowCount).fill(1)),
      groups: create('group-ids', geography.groups),
      algebra: new Map(),
      algebraOverflow: create('algebra-overflow', 4),
      algebraTotal: create('algebra-total', 4),
      lattice: null,
      focusRow,
      populationSorted,
      snapshot: null,
      kitSummary: null,
      latticeSnapshot: null,
      latticeKitSummary: null
    };
    writeValueMask(created);
    space.kit.setFocusRow(focusRow);
    return created;
  };

  const writeValueMask = (target: World) => {
    const values = target.geography.getVariable(ctx.options.variable);
    target.valueMask.write(Uint32Array.from(values, value => (Number.isFinite(value) ? 1 : 0)));
  };

  const writeVariable = (target: World) => {
    target.values.write(Float32Array.from(target.geography.getVariable(ctx.options.variable)));
    writeValueMask(target);
    if (target.lattice) writeLatticeValues(target, target.lattice);
  };

  const writeFocus = (target: World) => {
    const row = target.geography.pick(...FOCUS_POINTS[target.geography.id][ctx.options.focus]);
    if (row < 0) return;
    target.focusRow = row;
    target.space.kit.setFocusRow(row);
    if (target.lattice) {
      const [minX, minY] = target.lattice.bounds;
      target.lattice.focusRow = getLatticeCellOf(
        target,
        minX,
        minY,
        target.lattice.cell,
        target.lattice.columns,
        target.lattice.rows
      );
      target.lattice.space.kit.setFocusRow(target.lattice.focusRow);
    }
  };

  const writeSelfWeights = (target: World) => {
    target.selfWeights.write(new Float32Array(target.geography.count).fill(ctx.options.selfWeight));
  };

  const writeSubgraphMask = (target: World) => {
    const sorted = target.populationSorted;
    const threshold =
      sorted[
        Math.min(
          sorted.length - 1,
          Math.floor((ctx.options.populationPercentile / 100) * sorted.length)
        )
      ];
    target.subgraphMask.write(
      Uint32Array.from(target.geography.population, value => (value >= threshold ? 1 : 0))
    );
  };

  /** Builds the lattice the first time the lattice source is chosen. */
  const buildLattice = (target: World): Lattice => {
    const {geography, resources} = target;
    const columns = LATTICE_COLUMNS[geography.id];
    const [minX, minY, maxX, maxY] = geography.bounds;
    const cell = (maxX - minX) / columns;
    const rows = Math.ceil((maxY - minY) / cell);
    const count = columns * rows;
    const featureOfCell = new Int32Array(count).fill(-1);
    const positions = new Float32Array(count * 2);
    const categories = new Uint32Array(count);
    const insideMask = new Uint32Array(count);
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        const index = row * columns + column;
        const x = minX + (column + 0.5) * cell;
        const y = minY + (row + 0.5) * cell;
        positions[index * 2] = x;
        positions[index * 2 + 1] = y;
        const feature = geography.pickMeters(x, y);
        featureOfCell[index] = feature;
        if (feature >= 0) {
          insideMask[index] = 1;
          categories[index] = geography.groups[feature];
        }
      }
    }
    const create = (name: string, data: number | Float32Array | Uint32Array) =>
      resources.createBuffer(`lattice-${name}`, data);
    const slots = count * ((2 * LATTICE_MAXIMUM_RADIUS + 1) ** 2 - 1);
    const values = create('values', new Float32Array(count));
    const valueMask = create('value-mask', new Uint32Array(count));
    const structureMask = create('structure-mask', Uint32Array.from(insideMask));
    const overflow = create('overflow', 4);
    const total = create('total', 4);
    const space = createAnalysisSpace({
      device,
      resources,
      id: 'lattice',
      rows: count,
      slots,
      unionSlots: slots,
      positions: create('positions', positions),
      values,
      mask: valueMask,
      displayMask: structureMask,
      categories,
      drawAllLinks: false,
      producer: {overflow, total}
    });
    const lattice: Lattice = {
      columns,
      rows,
      cell,
      bounds: [minX, minY, minX + columns * cell, minY + rows * cell],
      featureOfCell,
      space,
      structureMask,
      values,
      valueMask,
      producers: new Map(),
      overflow,
      total,
      focusRow: getLatticeCellOf(target, minX, minY, cell, columns, rows),
      insideMask
    };
    space.kit.setFocusRow(lattice.focusRow);
    writeLatticeValues(target, lattice);
    writeLatticeStructure(lattice);
    return lattice;
  };
  const writeLatticeValues = (target: World, lattice: Lattice) => {
    const source = target.geography.getVariable(ctx.options.variable);
    const values = new Float32Array(lattice.featureOfCell.length);
    const mask = new Uint32Array(lattice.featureOfCell.length);
    for (let index = 0; index < values.length; index++) {
      const feature = lattice.featureOfCell[index];
      const value = feature >= 0 ? source[feature] : Number.NaN;
      values[index] = value;
      mask[index] = Number.isFinite(value) ? 1 : 0;
    }
    lattice.values.write(values);
    lattice.valueMask.write(mask);
  };

  const writeLatticeStructure = (lattice: Lattice) => {
    lattice.structureMask.write(
      ctx.options.latticeMask
        ? Uint32Array.from(lattice.insideMask)
        : new Uint32Array(lattice.insideMask.length).fill(1)
    );
  };

  const prepareLatticeProducer = (lattice: Lattice): boolean => {
    const {latticeCriterion: criterion, latticeRadius: radius} = ctx.options;
    const key = `${criterion}-${radius}`;
    if (lattice.producers.has(key)) return false;
    const {space} = lattice;
    const graph = new GPUCommandGraph<void>(device, {id: `lattice-${key}`});
    const count = lattice.columns * lattice.rows;
    graph.add(
      new GPULatticeWeights({
        id: 'lattice',
        width: lattice.columns,
        height: lattice.rows,
        criterion,
        radius,
        cellSize: [lattice.cell, lattice.cell],
        mask: importGraphBuffer(graph, 'mask', lattice.structureMask, 'uint32', count),
        weights: {
          offsets: importGraphBuffer(graph, 'offsets', space.csr.offsets, 'uint32', count + 1),
          neighbors: importGraphBuffer(
            graph,
            'neighbors',
            space.csr.neighbors,
            'uint32',
            space.slots
          ),
          weights: importGraphBuffer(graph, 'weights', space.csr.weights, 'float32', space.slots)
        },
        overflow: importGraphBuffer(graph, 'overflow', lattice.overflow, 'uint32', 1),
        totalNeighbors: importGraphBuffer(graph, 'total', lattice.total, 'uint32', 1)
      })
    );
    lattice.producers.set(key, world!.resources.track(graph.compile()));
    return true;
  };

  /** Compiles the algebra variant of the current options. */
  const prepareAlgebra = (target: World): boolean => {
    const key = getAlgebraKey();
    if (!key || target.algebra.has(key)) return false;
    const o = ctx.options;
    const {geography, space, core} = target;
    const rowCount = geography.count;
    const graph = new GPUCommandGraph<void>(device, {id: `weights-algebra-${key}`});
    const importCsr = (
      name: string,
      csr: {offsets: Buffer; neighbors: Buffer; weights: Buffer},
      slots: number
    ) => ({
      offsets: importGraphBuffer(graph, `${name}-offsets`, csr.offsets, 'uint32', rowCount + 1),
      neighbors: importGraphBuffer(graph, `${name}-neighbors`, csr.neighbors, 'uint32', slots),
      weights: importGraphBuffer(graph, `${name}-weights`, csr.weights, 'float32', slots)
    });
    const left = importCsr('a', core.csr, core.slots);
    const common = {
      id: 'algebra',
      output: importCsr('out', space.csr, space.slots),
      overflow: importGraphBuffer(graph, 'overflow', target.algebraOverflow, 'uint32', 1),
      totalNeighbors: importGraphBuffer(graph, 'total', target.algebraTotal, 'uint32', 1)
    };
    switch (o.combine) {
      case 'union':
      case 'intersection':
      case 'difference':
      case 'symmetricDifference':
        graph.add(
          new GPUSpatialWeightsAlgebra({
            ...common,
            operation: o.combine,
            left,
            right: importCsr('b', target.partner.csr, target.partner.slots),
            weightRule: o.weightRule
          })
        );
        break;
      case 'higherOrder':
        graph.add(
          new GPUSpatialWeightsAlgebra({
            ...common,
            operation: 'higherOrder',
            weights: left,
            order: o.order,
            cumulative: o.cumulative
          })
        );
        break;
      case 'selfWeight':
        graph.add(
          new GPUSpatialWeightsAlgebra({
            ...common,
            operation: 'selfWeight',
            weights: left,
            selfWeight: importGraphBuffer(
              graph,
              'self-weights',
              target.selfWeights,
              'float32',
              rowCount
            )
          })
        );
        break;
      case 'subgraph':
        graph.add(
          new GPUSpatialWeightsAlgebra({
            ...common,
            operation: 'subgraph',
            weights: left,
            mask: importGraphBuffer(graph, 'mask', target.subgraphMask, 'uint32', rowCount)
          })
        );
        break;
      case 'block':
        graph.add(
          new GPUSpatialWeightsAlgebra({
            ...common,
            operation: 'block',
            groupIds: importGraphBuffer(graph, 'groups', target.groups, 'uint32', rowCount),
            groupCount: geography.groupCount
          })
        );
        break;
      default:
        return false;
    }
    target.algebra.set(key, target.resources.track(graph.compile()));
    return true;
  };

  const isBinaryCombine = () =>
    ['union', 'intersection', 'difference', 'symmetricDifference'].includes(ctx.options.combine);

  /** Compiles every variant the current options need. Called from `create` and `setOption`. */
  const prepareAll = () => {
    const target = world;
    if (!target) return;
    const o = ctx.options;
    const focalWeight = Number(o.focalWeight);
    if (o.source === 'lattice') {
      target.lattice ??= buildLattice(target);
      prepareLatticeProducer(target.lattice);
      target.lattice.space.prepare(o.includeFocal, focalWeight);
    } else {
      target.core.prepare(getConfig(target.geography));
      if (isBinaryCombine()) target.partner.prepare(o.partnerK);
      prepareAlgebra(target);
      target.space.prepare(o.includeFocal, focalWeight);
    }
    target.space.setDisplayChoice(o.display, o.summary);
    target.lattice?.space.setDisplayChoice(o.display, o.summary);
  };

  const writeParameters = () => {
    const target = world;
    if (!target) return;
    target.core.writeParameters(getConfig(target.geography));
  };

  const readoutSource = () => {
    const target = world;
    if (!target) return null;
    const useLattice = ctx.options.source === 'lattice' && target.lattice;
    return {
      useLattice: Boolean(useLattice),
      snapshot: useLattice ? target.latticeSnapshot : target.snapshot,
      kit: useLattice ? target.latticeKitSummary : target.kitSummary
    };
  };

  const updateReadouts = () => {
    const target = world;
    const source = readoutSource();
    if (!target || !source) return;
    const {geography} = target;
    const o = ctx.options;
    const variable = getVariableInfo(o.variable);
    const rows = source.useLattice
      ? target.lattice!.columns * target.lattice!.rows
      : geography.count;
    const insideRows = source.useLattice
      ? target.lattice!.insideMask.reduce((a, b) => a + b, 0)
      : rows;
    ctx.setReadout(
      'rows',
      source.useLattice
        ? `${formatCount(insideRows)} of ${formatCount(rows)} cells`
        : formatCount(rows)
    );
    const kit = source.kit;
    if (kit) {
      ctx.setReadout('links', formatCount(kit.slots));
      ctx.setReadout(
        'neighbors',
        `${(kit.slots / Math.max(1, rows - kit.isolates)).toFixed(2)} avg, ${kit.minimumCardinality}-${kit.maximumCardinality}`
      );
      ctx.setReadout('asymmetric', formatCount(kit.asymmetricSlots));
      ctx.setReadout(
        'sums',
        `${formatCount(kit.s0)} / ${formatCount(kit.s1)} / ${formatCount(kit.s2)}`
      );
      ctx.setReadout(
        'union',
        kit.unionOverflow ? 'overflow' : `${formatCount(kit.unionSlots)} slots`
      );
      ctx.setReadout(
        'lagRange',
        `${kit.lagMinimum.toFixed(variable.digits)} to ${kit.lagMaximum.toFixed(variable.digits)} ${variable.unit}`
      );
      ctx.setReadout(
        'capacity',
        kit.producerOverflow ? 'producer overflow: some links dropped' : 'ok'
      );
      ctx.setReadout('islands', kit.isolates === 0 ? 'none' : formatCount(kit.isolates));
      if (source.snapshot) {
        const names: string[] = [];
        if (kit.isolates > 0 && !source.useLattice) {
          for (let row = 0; row < rows && names.length < 3; row++) {
            if (source.snapshot.cardinality[row] === 0) names.push(geography.getName(row));
          }
          ctx.setReadout(
            'islands',
            `${kit.isolates}: ${names.join(', ')}${kit.isolates > names.length ? ', ...' : ''}`
          );
        }
        const matrix = o.matrix;
        const ids = source.snapshot.focusIds[matrix];
        const focus = kit.focus[matrix];
        const focusRow = source.useLattice ? target.lattice!.focusRow : target.focusRow;
        if (!source.useLattice) {
          const labels: string[] = [];
          for (let slot = 0; slot < Math.min(focus.degree, 5, FOCUS_SLOTS); slot++) {
            labels.push(geography.getName(ids[slot]).replace(/ County$/, ''));
          }
          ctx.setReadout(
            'focus',
            `${geography.getName(focusRow)}: ${focus.degree} neighbours (weight sum ${focus.weightSum.toFixed(2)})${labels.length ? ` - ${labels.join(', ')}${focus.degree > labels.length ? ', ...' : ''}` : ''}`
          );
        } else {
          ctx.setReadout(
            'focus',
            `cell ${focusRow}: ${focus.degree} neighbours (weight sum ${focus.weightSum.toFixed(2)})`
          );
        }
      }
    }
    if (o.source === 'band') {
      ctx.setReadout(
        'band',
        `${(target.core.getBandMeters(getConfig(geography)) / 1000).toFixed(1)} km`
      );
    } else {
      ctx.setReadout('band', 'band source only');
    }
    ctx.setReadout(
      'spacing',
      `${(geography.medianSpacing / 1000).toFixed(geography.medianSpacing > 5000 ? 0 : 2)} km`
    );
    const algebraOverflow = false;
    void algebraOverflow;
  };

  const getFillExtent = (snapshot: SpaceSnapshot | null) => {
    if (!snapshot) return;
    const [low, high] = snapshot.extent;
    if (Number.isFinite(low) && Number.isFinite(high)) ctx.setLegendExtent('display', [low, high]);
  };

  const requestReadbackIfNeeded = (commandEncoder: CommandEncoder) => {
    const target = world;
    if (!target || !readbackStale) return;
    const useLattice = ctx.options.source === 'lattice' && target.lattice;
    const space = useLattice ? target.lattice!.space : target.space;
    if (space.isReading()) return;
    readbackStale = false;
    space.requestReadback(
      commandEncoder,
      summary => {
        if (useLattice) target.latticeKitSummary = summary;
        else target.kitSummary = summary;
        updateReadouts();
      },
      snapshot => {
        if (useLattice) target.latticeSnapshot = snapshot;
        else target.snapshot = snapshot;
        getFillExtent(snapshot);
        updateReadouts();
      }
    );
  };

  const copyCsr = (
    commandEncoder: CommandEncoder,
    from: {offsets: Buffer; neighbors: Buffer; weights: Buffer},
    to: {offsets: Buffer; neighbors: Buffer; weights: Buffer},
    rowCount: number,
    slots: number
  ) => {
    commandEncoder.copyBufferToBuffer({
      sourceBuffer: from.offsets,
      destinationBuffer: to.offsets,
      size: (rowCount + 1) * 4
    });
    commandEncoder.copyBufferToBuffer({
      sourceBuffer: from.neighbors,
      destinationBuffer: to.neighbors,
      size: slots * 4
    });
    commandEncoder.copyBufferToBuffer({
      sourceBuffer: from.weights,
      destinationBuffer: to.weights,
      size: slots * 4
    });
  };

  const switchWorld = async (id: GeographyId) => {
    const token = ++switchToken;
    ctx.setStatus(`Loading ${id === 'us-counties' ? 'counties' : 'tracts'}...`);
    const geography = await getGeography(id);
    if (token !== switchToken || destroyed) return;
    const previous = world;
    world = null;
    ctx.requestLayers();
    if (previous) setTimeout(() => previous.resources.destroy(), 200);
    world = buildWorld(geography);
    prepareAll();
    writeParameters();
    writeSelfWeights(world);
    writeSubgraphMask(world);
    dirty = true;
    readbackStale = true;
    ctx.setStatus('');
    ctx.requestLayers();
  };

  const initial = await getGeography(ctx.options.geography);
  if (ctx.signal.aborted) throw new Error('aborted');
  world = buildWorld(initial);
  prepareAll();
  writeParameters();
  writeSelfWeights(world);
  writeSubgraphMask(world);
  ctx.setStatus('');

  const getPickedRow = (event: {coordinate: readonly [number, number] | null}): number => {
    const target = world;
    if (!target || !event.coordinate) return -1;
    return target.geography.pick(event.coordinate[0], event.coordinate[1]);
  };

  const getLatticeCell = (event: {coordinate: readonly [number, number] | null}): number => {
    const target = world;
    const lattice = target?.lattice;
    if (!target || !lattice || !event.coordinate) return -1;
    const [x, y] = target.geography.project(event.coordinate[0], event.coordinate[1]);
    const column = Math.floor((x - lattice.bounds[0]) / lattice.cell);
    const row = Math.floor((y - lattice.bounds[1]) / lattice.cell);
    if (column < 0 || row < 0 || column >= lattice.columns || row >= lattice.rows) return -1;
    return row * lattice.columns + column;
  };

  return {
    getCompiledGraphs() {
      const target = world;
      if (!target) return [];
      return [
        ...target.core.getGraphs(),
        ...target.algebra.values(),
        ...target.space.getGraphs(),
        ...(target.lattice
          ? [...target.lattice.producers.values(), ...target.lattice.space.getGraphs()]
          : [])
      ] as CompiledGPUCommandGraph<never>[];
    },

    setOption(id, value) {
      const target = world;
      if (id === 'geography') {
        void switchWorld(value as GeographyId);
        return;
      }
      if (!target) return;
      if (id === 'variable') writeVariable(target);
      if (id === 'focus') writeFocus(target);
      if (id === 'populationPercentile') writeSubgraphMask(target);
      if (id === 'selfWeight') writeSelfWeights(target);
      if (id === 'latticeMask' && target.lattice) writeLatticeStructure(target.lattice);
      prepareAll();
      writeParameters();
      dirty = true;
      readbackStale = true;
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const target = world;
      if (!target) return;
      const o = ctx.options;
      if (dirty || frame.frameIndex < 2) {
        const focalWeight = Number(o.focalWeight);
        if (o.source === 'lattice' && target.lattice) {
          const lattice = target.lattice;
          lattice.producers
            .get(`${o.latticeCriterion}-${o.latticeRadius}`)
            ?.encode(commandEncoder, {parameters: undefined});
          lattice.space.encode(commandEncoder, {
            normalizeLag: o.lagNormalize,
            matrix: o.matrix,
            display: o.display,
            summary: o.summary,
            includeFocal: o.includeFocal,
            focalWeight
          });
        } else {
          target.core.encode(commandEncoder, getConfig(target.geography));
          const algebraKey = getAlgebraKey();
          if (algebraKey) {
            if (isBinaryCombine()) target.partner.encode(commandEncoder, o.partnerK);
            target.algebra.get(algebraKey)?.encode(commandEncoder, {parameters: undefined});
          } else {
            copyCsr(
              commandEncoder,
              target.core.csr,
              target.space.csr,
              target.geography.count,
              target.core.slots
            );
          }
          target.space.encode(commandEncoder, {
            normalizeLag: o.lagNormalize,
            matrix: o.matrix,
            display: o.display,
            summary: o.summary,
            includeFocal: o.includeFocal,
            focalWeight
          });
        }
        dirty = false;
        readbackStale = true;
      }
      if (frame.frameIndex >= 1) requestReadbackIfNeeded(commandEncoder);
    },

    getLayers() {
      const target = world;
      if (!target) return [];
      const o = ctx.options;
      const theme = ctx.theme();
      const {geography, buffers} = target;
      const coordinateOrigin: [number, number, number] = [
        geography.origin[0],
        geography.origin[1],
        0
      ];
      const useLattice = o.source === 'lattice' && target.lattice;
      const space = useLattice ? target.lattice!.space : target.space;
      const isFocus = o.display === 'focus';
      const style = isFocus
        ? {
            values: space.focusClass,
            valueFormat: 'uint32' as const,
            colormap: 'category' as const,
            palette: FOCUS_COLORS
          }
        : {
            values: space.shown,
            valueFormat: 'float32' as const,
            colormap: o.ramp,
            extent: space.shownExtent
          };
      const layers: Layer[] = [];
      if (useLattice) {
        const lattice = target.lattice!;
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'weights-lattice-fill',
            coordinateOrigin,
            gridSize: [lattice.columns, lattice.rows],
            bounds: lattice.bounds,
            ...style,
            noDataColor: NO_DATA,
            opacity: 0.92
          })
        );
      } else {
        layers.push(
          new B4PolygonFillLayer({
            id: 'weights-fill',
            coordinateOrigin,
            triangles: buffers.triangles,
            features: buffers.features,
            triangleVertexCount: buffers.triangleVertexCount,
            ...style,
            noDataColor: NO_DATA,
            opacity: 0.92
          })
        );
      }
      if (o.showOutlines || useLattice) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'weights-outline',
            coordinateOrigin,
            segments: buffers.outline,
            instanceCount: buffers.outlineSegmentCount,
            widthPixels: 0.8,
            color: getOutlineColor(theme)
          })
        );
      }
      if (!useLattice && o.showLinks) {
        const links = target.space.kit.allSegments[o.matrix];
        if (links) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: `weights-links-${o.matrix}`,
              coordinateOrigin,
              segments: links.segments,
              weights: links.fade,
              instanceCount: links.slots,
              widthPixels: geography.id === 'us-counties' ? 0.7 : 1.1,
              color: getLinkColor(theme, geography.id === 'us-counties')
            })
          );
        }
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: `weights-focus-links-${o.matrix}-${useLattice ? 'lattice' : 'polygons'}`,
          coordinateOrigin,
          segments: space.kit.focusSegments[o.matrix],
          instanceCount: FOCUS_SLOTS,
          widthPixels: 2.6,
          color: theme === 'dark' ? [255, 214, 120, 255] : [20, 20, 40, 255]
        })
      );
      if (!useLattice) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'weights-islands',
            coordinateOrigin,
            positions: target.core.positions,
            instanceCount: geography.count,
            values: target.space.isolate,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: ISLAND_COLOR,
            noDataColor: [0, 0, 0, 0],
            radiusPixels: 6
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      const target = world;
      if (!target) return null;
      const o = ctx.options;
      const variable = getVariableInfo(o.variable);
      const useLattice = o.source === 'lattice' && target.lattice;
      const snapshot = useLattice ? target.latticeSnapshot : target.snapshot;
      if (useLattice) {
        const cell = getLatticeCell(event);
        if (cell < 0) return null;
        const feature = target.lattice!.featureOfCell[cell];
        if (feature < 0 && o.latticeMask) return null;
        const lines = [
          feature >= 0 ? target.geography.getName(feature) : 'Outside the map (empty cell)'
        ];
        if (snapshot) lines.push(`Neighbours: ${snapshot.cardinality[cell]}`);
        return lines.join('\n');
      }
      const row = getPickedRow(event);
      if (row < 0) return null;
      const {geography} = target;
      const value = geography.getVariable(o.variable)[row];
      const lines = [
        `${geography.getName(row)} (${geography.getGroupName(row)})`,
        `${variable.label}: ${Number.isFinite(value) ? `${value.toFixed(variable.digits)} ${variable.unit}` : 'no data'}`
      ];
      if (snapshot) {
        lines.push(
          `Neighbours: ${snapshot.cardinality[row]}${snapshot.oneWay[row] > 0 ? ` (${snapshot.oneWay[row]} one-way)` : ''}`
        );
        const lag = snapshot.lag[row];
        if (Number.isFinite(lag)) lines.push(`Spatial lag: ${lag.toFixed(variable.digits)}`);
        const mode = snapshot.modes[row];
        if (mode !== NO_MODE && mode < geography.groupNames.length) {
          lines.push(`Dominant neighbour group: ${geography.groupNames[mode]}`);
        }
      }
      return lines.join('\n');
    },

    onClick(event) {
      const target = world;
      if (!target) return false;
      if (ctx.options.source === 'lattice' && target.lattice) {
        const cell = getLatticeCell(event);
        if (cell < 0) return false;
        target.lattice.focusRow = cell;
        target.lattice.space.kit.setFocusRow(cell);
      } else {
        const row = getPickedRow(event);
        if (row < 0) return false;
        target.focusRow = row;
        target.space.kit.setFocusRow(row);
      }
      dirty = true;
      readbackStale = true;
      return true;
    },

    destroy() {
      destroyed = true;
      switchToken++;
      const target = world;
      world = null;
      if (target) {
        target.space.stop();
        target.lattice?.space.stop();
        target.resources.destroy();
      }
    }
  };
}

void SLOTS_PER_ROW;
