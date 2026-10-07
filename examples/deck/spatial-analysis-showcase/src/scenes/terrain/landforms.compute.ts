// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getClassIndexOf} from '../../cartography/class-table';
import {ALPS} from '../../cartography/gazetteer';
import {formatCount, formatDistance, formatSigned, liveText} from '../../cartography/live-text';
import type {ClassTable, MapAnnotation} from '../../cartography/types';
import {measureCompiledGraph} from '../../engine/vector-timing';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance, TooltipContent, TooltipRow} from '../scene';
import {loadAlpsGrid} from './b14a-grid';
import {ColorRasterLayer} from './b14a-layers';
import {TerrainSession, type Stage} from './b14a-session';
import {createDemProbe, createTerrainDemFromGrid} from './cpu-dem';
import {
  createTablePaintGraph,
  TABLE_PAINT_PARAMETER_LENGTH,
  writeTablePaint
} from './landforms-paint';
import {
  buildAgreement,
  buildCurvature,
  buildGeomorphons,
  buildPosition,
  buildWeiss,
  getCurvatureConfigKey,
  getCurvatureValueId,
  getGeomorphonKey,
  getPositionConfigKey,
  getWeissConfigKey,
  type LandformBuild,
  type ValueRef
} from './landforms-products';
import {createRayOverlay, findCrestCell} from './landforms-rays';
import {
  COMPARE_RADIUS,
  getCurvatureKind,
  getCurvatureName,
  getDilationRadius,
  getScaleLabels,
  getSignedKey,
  getSwatchColor,
  GEOMORPHON_CODES,
  GEOMORPHON_LEGEND_ORDER,
  GEOMORPHON_NAMES,
  makeCurvatureTable,
  makeDeviationTable,
  makeLandformTables,
  makePositionTable,
  PEAK_CODE,
  PIT_CODE,
  SCALE_PRESETS,
  SWEEP_GROUPS,
  SWEEP_RADII,
  type LandformOptions,
  type LandformProduct,
  type LandformTables
} from './landforms.style';
import {demSampleLine} from './terrain-furniture';
import {createTerrainGround, rasterizeGlacierMask} from './terrain-ground';
import {WEISS_LABELS, type TerrainGroundTone} from './terrain-palettes';
import {
  getClippedLabelPoint,
  glacierLabels,
  loadAlpsContext,
  snapPeaksToDem,
  terrainLabel
} from './terrain-places';

export type {LandformOptions, LandformProduct};

const SYNC_DELAY_MILLISECONDS = 80;
const SLIDER_SYNC_DELAY_MILLISECONDS = 250;
const READ_INTERVAL_MILLISECONDS = 220;
const MAXIMUM_GEOMORPHON_BUILDS = 10;
const HISTOGRAM_BINS = 48;
/** The elevation never changes after the first upload, so every stage sees one version. */
const ELEVATION_VERSION = 0;

type TableId = 'quiet' | 'grass' | 'weiss' | 'agreement' | 'scale' | 'curvature' | 'tpi' | 'dev';

/** What is drawn and read now. */
type Display = {
  key: string;
  /** Encoded every frame (its dependencies first). */
  stage: Stage;
  /** The raster painted. */
  value: ValueRef;
  mode: 'codes' | 'breaks';
  tableId: TableId;
  /** Rasters read back, the painted one first. */
  reads: ValueRef[];
  compare: {stage: Stage; value: ValueRef} | null;
  /** Peaks and pits are enlarged at low zoom. */
  dilate: boolean;
  /** The frozen-scale key of the signed table, if the product is signed. */
  signedKey: string | null;
  forms: ValueRef | null;
  agreement: ValueRef | null;
  weiss: ValueRef | null;
  scale: ValueRef | null;
};

type SweepState = {
  phase: 'compiling' | 'running';
  builds: LandformBuild[];
  index: number;
  ran: boolean;
  reading: boolean;
  /** Share of valid cells per group, per radius. */
  shares: number[][];
};

/** Counts the class codes of a raster: index 0 holds code 0, up to code 31. */
function countCodes(words: Uint32Array): Float64Array {
  const counts = new Float64Array(32);
  for (let index = 0; index < words.length; index++) {
    const word = words[index];
    if (word < 32) counts[word]++;
  }
  return counts;
}

/** The value below which `probability` of the absolute finite values lie (sampled, sorted). */
function getAbsolutePercentile(values: Float32Array, probability: number): number {
  const stride = Math.max(1, Math.floor(values.length / 400_000));
  const sampled = new Float32Array(Math.ceil(values.length / stride));
  let count = 0;
  for (let index = 0; index < values.length; index += stride) {
    const value = values[index];
    if (Number.isFinite(value)) sampled[count++] = Math.abs(value);
  }
  if (count === 0) return Number.NaN;
  const used = sampled.subarray(0, count).sort();
  return used[Math.min(count - 1, Math.floor(probability * count))];
}

/** Bin counts of the finite values over `[low, high]`, the tails clipped into the end bins. */
function getHistogram(values: Float32Array, low: number, high: number, bins: number): number[] {
  const counts = new Array<number>(bins).fill(0);
  const scale = bins / (high - low);
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    counts[Math.min(bins - 1, Math.max(0, Math.floor((value - low) * scale)))]++;
  }
  return counts;
}

/** The stage and every stage it depends on, dependencies first. */
function collectStages(stage: Stage, found: Set<Stage> = new Set()): Set<Stage> {
  for (const dependency of stage.dependencies) collectStages(dependency, found);
  found.add(stage);
  return found;
}

/**
 * The landforms story: geomorphons, curvature, topographic position and Weiss landforms of the
 * Matterhorn tile. Every product is compiled when first shown (or when a compile-time option
 * changes) and cached; per-frame options are parameter writes. One table-driven colorize pass
 * paints each raster from the class table the legend reads, over the chapter relief ground.
 */
export async function createLandforms(
  ctx: SceneContext<LandformOptions>
): Promise<SceneInstance<LandformOptions>> {
  const {device} = ctx;
  const grid = await loadAlpsGrid(ctx.datasets.get('alps-dem'), ctx.signal);
  const context = await loadAlpsContext(ctx.datasets.get('alps-context'), ctx.signal);
  ctx.signal.throwIfAborted();
  const {width, height, pixelCount} = grid;
  const dem = createTerrainDemFromGrid(grid);
  const probe = createDemProbe(dem);
  const glacierMask = rasterizeGlacierMask(context.glacierGeoJson, dem);
  const getOptions = () => ctx.options as LandformOptions;

  const resources = new SpatialAnalysisResources(device, 'landforms');
  const session = new TerrainSession(ctx, resources, grid);
  session.elevationBuffer.write(grid.cpuElevation);
  session.validityBuffer.write(new Uint32Array(pixelCount).fill(1));
  const colorsMain = resources.createBuffer('colors-main', pixelCount * 4);
  const colorsCompare = resources.createBuffer('colors-compare', pixelCount * 4);
  const paintParameters = resources.createParameterBuffer(
    'paint',
    'float32',
    TABLE_PAINT_PARAMETER_LENGTH
  );
  const rayOverlay = createRayOverlay(resources, grid);

  let groundTone: TerrainGroundTone = ctx.ground();
  const terrainGround = createTerrainGround({
    dem,
    device,
    ground: groundTone,
    glacierMask
  });
  await terrainGround.prepare();
  ctx.signal.throwIfAborted();

  let destroyed = false;
  let timers: ReturnType<typeof setTimeout>[] = [];
  const cellMeters = grid.groundCellSize;

  // --- Tables ------------------------------------------------------------------------------------
  let tables: LandformTables = makeLandformTables(groundTone);
  let deviationTable: ClassTable = makeDeviationTable(groundTone);
  /** 98th percentile of |value| per signed key, measured once and frozen. */
  const frozenP98 = new Map<string, number>();
  const signedTables = new Map<string, ClassTable>();

  function rebuildSignedTables(): void {
    signedTables.clear();
    for (const [key, p98] of frozenP98) {
      signedTables.set(
        key,
        key.startsWith('curvature:')
          ? makeCurvatureTable(p98, groundTone)
          : makePositionTable(p98, groundTone)
      );
    }
  }

  function getTable(target: Display | null = display): ClassTable | null {
    if (!target) return null;
    switch (target.tableId) {
      case 'quiet':
        return tables.quiet;
      case 'grass':
        return tables.grass;
      case 'weiss':
        return tables.weiss;
      case 'agreement':
        return tables.agreement;
      case 'scale':
        return tables.scale;
      case 'dev':
        return deviationTable;
      default:
        return signedTables.get(target.signedKey ?? '') ?? null;
    }
  }

  // --- Builds ------------------------------------------------------------------------------------
  const geomorphonBuilds = new Map<string, LandformBuild>();
  let curvatureEntry: {key: string; build: LandformBuild} | null = null;
  let positionEntry: {key: string; build: LandformBuild} | null = null;
  let weissEntry: {key: string; build: LandformBuild} | null = null;
  let agreementEntry: {key: string; geomorphonKey: string; build: LandformBuild} | null = null;
  let sweep: SweepState | null = null;
  const latest = new Map<string, Uint32Array | Float32Array>();

  const forEachBuild = (visit: (build: LandformBuild) => void) => {
    for (const build of geomorphonBuilds.values()) visit(build);
    for (const entry of [curvatureEntry, positionEntry, weissEntry, agreementEntry]) {
      if (entry) visit(entry.build);
    }
  };

  const releaseBuild = (build: LandformBuild) => {
    for (const value of Object.values(build.values)) latest.delete(value.id);
    build.release();
  };

  function getProtectedGeomorphonKeys(): Set<string> {
    const state = getOptions();
    const keys = new Set<string>([getGeomorphonKey(state.geomorphonRadius, state)]);
    keys.add(getGeomorphonKey(COMPARE_RADIUS, state));
    if (agreementEntry) keys.add(agreementEntry.geomorphonKey);
    for (const radius of SWEEP_RADII) keys.add(getGeomorphonKey(radius, state));
    return keys;
  }

  function getGeomorphonBuild(radius: number): LandformBuild {
    const key = getGeomorphonKey(radius, getOptions());
    const existing = geomorphonBuilds.get(key);
    if (existing) {
      // Most recently used goes last.
      geomorphonBuilds.delete(key);
      geomorphonBuilds.set(key, existing);
      return existing;
    }
    const build = buildGeomorphons(session, radius, getOptions);
    geomorphonBuilds.set(key, build);
    const keep = getProtectedGeomorphonKeys();
    for (const [candidate, old] of geomorphonBuilds) {
      if (geomorphonBuilds.size <= MAXIMUM_GEOMORPHON_BUILDS) break;
      if (keep.has(candidate)) continue;
      releaseBuild(old);
      geomorphonBuilds.delete(candidate);
    }
    return build;
  }

  function ensurePaint(value: ValueRef, target: 'main' | 'compare'): void {
    if (value.paints.has(target)) return;
    value.paints.set(
      target,
      createTablePaintGraph(
        resources,
        device,
        `${value.id}-${target}`,
        {buffer: value.buffer, format: value.format, width, height},
        paintParameters,
        target === 'main' ? colorsMain : colorsCompare
      )
    );
  }

  /** Builds (or reuses) what the options show. Compiles: call it from an option handler, never a frame. */
  function getDisplayFor(state: LandformOptions): Display {
    const none = {
      forms: null,
      agreement: null,
      weiss: null,
      scale: null,
      compare: null,
      signedKey: null
    };
    switch (state.product) {
      case 'geomorphons': {
        const build = getGeomorphonBuild(state.geomorphonRadius);
        const shortBuild = state.compareRadius ? getGeomorphonBuild(COMPARE_RADIUS) : null;
        const tableId = state.palette === 'grass' ? 'grass' : 'quiet';
        return {
          ...none,
          key: `${build.values.forms.id}|${tableId}|${shortBuild?.values.forms.id ?? ''}`,
          stage: build.stage,
          value: build.values.forms,
          mode: 'codes',
          tableId,
          reads: [build.values.forms],
          compare: shortBuild ? {stage: shortBuild.stage, value: shortBuild.values.forms} : null,
          dilate: true,
          forms: build.values.forms
        };
      }
      case 'curvature': {
        const configKey = getCurvatureConfigKey(state);
        if (curvatureEntry?.key !== configKey) {
          if (curvatureEntry) releaseBuild(curvatureEntry.build);
          curvatureEntry = null;
          curvatureEntry = {key: configKey, build: buildCurvature(session, getOptions)};
        }
        const kind = getCurvatureKind(state);
        const value = curvatureEntry.build.values[getCurvatureValueId(kind)];
        return {
          ...none,
          key: `${value.id}|curvature`,
          stage: curvatureEntry.build.stage,
          value,
          mode: 'breaks',
          tableId: 'curvature',
          reads: [value],
          dilate: false,
          signedKey: getSignedKey(state)
        };
      }
      case 'position': {
        const configKey = getPositionConfigKey(state);
        if (positionEntry?.key !== configKey) {
          if (positionEntry) releaseBuild(positionEntry.build);
          positionEntry = null;
          positionEntry = {
            key: configKey,
            build: buildPosition(session, state.positionProduct, getOptions)
          };
        }
        const build = positionEntry.build;
        if (state.positionProduct === 'scale') {
          return {
            ...none,
            key: `${build.values.codes.id}|scale`,
            stage: build.stage,
            value: build.values.codes,
            mode: 'codes',
            tableId: 'scale',
            reads: [build.values.codes],
            dilate: false,
            scale: build.values.codes
          };
        }
        const tableId = state.positionProduct === 'tpi' ? 'tpi' : 'dev';
        return {
          ...none,
          key: `${build.values.value.id}|${tableId}`,
          stage: build.stage,
          value: build.values.value,
          mode: 'breaks',
          tableId,
          reads: [build.values.value],
          dilate: false,
          signedKey: getSignedKey(state)
        };
      }
      case 'weiss': {
        const weissKey = getWeissConfigKey(state);
        if (weissEntry?.key !== weissKey) {
          if (agreementEntry) releaseBuild(agreementEntry.build);
          agreementEntry = null;
          if (weissEntry) releaseBuild(weissEntry.build);
          weissEntry = null;
          weissEntry = {key: weissKey, build: buildWeiss(session, getOptions)};
        }
        const geomorphonBuild = getGeomorphonBuild(state.geomorphonRadius);
        const geomorphonKey = getGeomorphonKey(state.geomorphonRadius, state);
        if (agreementEntry && agreementEntry.geomorphonKey !== geomorphonKey) {
          releaseBuild(agreementEntry.build);
          agreementEntry = null;
        }
        agreementEntry ??= {
          key: weissKey,
          geomorphonKey,
          build: buildAgreement(session, weissEntry.build, geomorphonBuild)
        };
        const weissValue = weissEntry.build.values.codes;
        const formsValue = geomorphonBuild.values.forms;
        const agreementValue = agreementEntry.build.values.codes;
        const shown =
          state.view === 'agreement'
            ? agreementValue
            : state.view === 'geomorphons'
              ? formsValue
              : weissValue;
        const tableId: TableId =
          state.view === 'agreement'
            ? 'agreement'
            : state.view === 'geomorphons'
              ? state.palette === 'grass'
                ? 'grass'
                : 'quiet'
              : 'weiss';
        return {
          ...none,
          key: `${shown.id}|${tableId}`,
          // The agreement stage runs both classifiers, so every share is always known.
          stage: agreementEntry.build.stage,
          value: shown,
          mode: 'codes',
          tableId,
          reads: [shown, ...[weissValue, formsValue, agreementValue].filter(ref => ref !== shown)],
          dilate: state.view === 'geomorphons',
          forms: formsValue,
          agreement: agreementValue,
          weiss: weissValue
        };
      }
    }
  }

  // --- Display state -----------------------------------------------------------------------------
  let display: Display | null = null;
  let displayGeneration = 0;
  let paintDirty = true;
  let painted = false;
  let dilateRadius = 0;
  let isolate: number[] | null = null;
  let readQueue: ValueRef[] = [];
  let readInFlight = false;
  let readDirty = false;
  let lastReadAt = 0;
  const measuredKeys = new Set<string>();
  let measuring = false;
  let lastFurniture = '';

  function applyPaintParameters(): void {
    const table = getTable();
    if (!table || !display) return;
    writeTablePaint(paintParameters, table, {
      mode: display.mode,
      dilateRadius: display.dilate ? dilateRadius : 0,
      peakCode: PEAK_CODE,
      pitCode: PIT_CODE,
      isolate
    });
    paintDirty = true;
  }

  function publishTables(): void {
    ctx.setLegendData('ground', groundTone);
    ctx.setLegendData('tables', tables);
  }

  function publishSignedTable(): void {
    const table = getTable();
    const target = display;
    ctx.setLegendData('signedFor', target?.signedKey ?? null);
    ctx.setLegendData('signedTable', target?.signedKey && table ? table : null);
  }

  function setDisplay(next: Display): void {
    const changed = !display || display.key !== next.key;
    display = next;
    if (changed) {
      displayGeneration++;
      painted = false;
      isolate = null;
      readQueue = [];
    }
    paintDirty = true;
    readDirty = true;
    dilateRadius = next.dilate ? getDilationRadius(ctx.getViewState().zoom) : 0;
    applyPaintParameters();
    publishSignedTable();
    updateReadoutsOfState();
    ctx.requestLayers();
  }

  function syncDisplay(): void {
    if (destroyed) return;
    const state = getOptions();
    try {
      const next = getDisplayFor(state);
      ensurePaint(next.value, 'main');
      if (next.compare) ensurePaint(next.compare.value, 'compare');
      ctx.setStatus('');
      setDisplay(next);
    } catch (error) {
      ctx.setStatus(`Cannot build ${state.product}: ${(error as Error).message}`);
    }
    updateRays();
    publishFurniture();
    updateGlacierNote();
  }

  let syncTimer: ReturnType<typeof setTimeout> | null = null;

  /** Rebuilds the display after the options settle (a slider drag compiles once, not per tick). */
  function scheduleSync(delay = SYNC_DELAY_MILLISECONDS): void {
    if (syncTimer) clearTimeout(syncTimer);
    syncTimer = setTimeout(() => {
      syncTimer = null;
      syncDisplay();
    }, delay);
  }

  // --- Furniture, readouts, notes -----------------------------------------------------------------
  const getRadiusMeters = (radius: number) => radius * cellMeters;

  function getSubtitle(state: LandformOptions): string {
    const geomorphonLine = `Geomorphons · L = ${state.geomorphonRadius} px = ${Math.round(
      getRadiusMeters(state.geomorphonRadius)
    )} m · flat < ${state.geomorphonFlatAngle.toFixed(1)}°`;
    switch (state.product) {
      case 'geomorphons':
        return state.compareRadius
          ? `${geomorphonLine} · against L = ${COMPARE_RADIUS} px`
          : geomorphonLine;
      case 'curvature':
        return `${getCurvatureName(getCurvatureKind(state))}, 1/m · ${
          state.curvatureMethod === 'florinsky' ? '5 x 5' : '3 x 3'
        } polynomial fit · ${cellMeters.toFixed(1)} m cells`;
      case 'position': {
        if (state.positionProduct === 'scale') {
          return 'Window radius at which a cell stands out most · eight windows';
        }
        const radius = SCALE_PRESETS[state.scalePreset][state.scaleIndex];
        const unit =
          state.positionProduct === 'tpi'
            ? 'Height above the window mean, m'
            : 'Height against the window, SD';
        return `${unit} · window ${Math.round(getRadiusMeters(2 * radius + 1))} m wide`;
      }
      case 'weiss':
        if (state.view === 'geomorphons') return geomorphonLine;
        if (state.view === 'agreement') {
          return `Weiss against geomorphons (L = ${state.geomorphonRadius} px) · convex, neutral, concave`;
        }
        return `Weiss landforms · windows ${state.weissSmall} and ${state.weissLarge} px · ${state.weissStandardization} standardisation`;
    }
  }

  function getScaleTicks(state: LandformOptions): number[] | undefined {
    if (
      state.product === 'geomorphons' ||
      (state.product === 'weiss' && state.view === 'geomorphons')
    ) {
      const ticks = [getRadiusMeters(state.geomorphonRadius)];
      if (state.product === 'geomorphons' && state.compareRadius) {
        ticks.unshift(getRadiusMeters(COMPARE_RADIUS));
      }
      return ticks.map(value => Math.round(value));
    }
    if (state.product === 'position' && state.positionProduct !== 'scale') {
      const radius = SCALE_PRESETS[state.scalePreset][state.scaleIndex];
      return [Math.round(getRadiusMeters(2 * radius + 1))];
    }
    return undefined;
  }

  function publishFurniture(): void {
    const state = getOptions();
    const ticks = getScaleTicks(state);
    const furniture = {
      title: {subtitle: getSubtitle(state), sample: demSampleLine(grid)},
      scaleBar: {units: 'metric' as const, ticks}
    };
    const serialized = JSON.stringify(furniture);
    if (serialized === lastFurniture) return;
    lastFurniture = serialized;
    ctx.setFurniture(furniture);
  }

  function updateReadoutsOfState(): void {
    const state = getOptions();
    const radius = state.geomorphonRadius;
    ctx.setReadout('lookout', `${radius} px = ${Math.round(getRadiusMeters(radius))} m`);
    if (state.product === 'position' && state.positionProduct !== 'scale') {
      const window = 2 * SCALE_PRESETS[state.scalePreset][state.scaleIndex] + 1;
      ctx.setReadout('naiveReads', window * window);
      ctx.setReadout('tableReads', 4);
    }
    const target = display;
    if (target) {
      const passes = collectStages(target.stage).size;
      let graphs = 0;
      for (const stage of collectStages(target.stage)) graphs += stage.graphs.length;
      ctx.setCost({
        records: pixelCount,
        passes: graphs + 1 + (target.compare ? graphs + 1 : 0),
        note: `${passes} contributor stage${passes === 1 ? '' : 's'} · ${formatCount(geomorphonBuilds.size)} geomorphon graphs cached`
      });
    }
  }

  /** The label point of Gornergletscher inside the DEM window, for the finding note. */
  const glacierNames = glacierLabels(context, {
    window: dem.lngLatBounds,
    names: ['Gornergletscher'],
    max: 1
  });
  const gornerPolygon = context.glaciers.find(polygon =>
    polygon.name?.toLowerCase().startsWith('gornergletscher')
  );
  const gornerPoint = gornerPolygon ? getClippedLabelPoint(gornerPolygon, dem.lngLatBounds) : null;

  function updateGlacierNote(): void {
    const state = getOptions();
    const target = display;
    const forms = target?.forms ? latest.get(target.forms.id) : undefined;
    if (
      !target ||
      !forms ||
      !gornerPoint ||
      state.product !== 'geomorphons' ||
      state.showRays ||
      state.compareRadius
    ) {
      ctx.setAnnotations('glacier-note', null);
      return;
    }
    const counts = new Float64Array(11);
    let total = 0;
    for (let index = 0; index < pixelCount; index++) {
      if (glacierMask[index] === 0) continue;
      const code = forms[index];
      if (code >= 1 && code <= 10) {
        counts[code]++;
        total++;
      }
    }
    if (total === 0) {
      ctx.setAnnotations('glacier-note', null);
      return;
    }
    let top = 1;
    for (let code = 2; code <= 10; code++) if (counts[code] > counts[top]) top = code;
    ctx.setAnnotations('glacier-note', [
      {
        kind: 'note',
        id: 'glacier-finding',
        coordinate: gornerPoint.lngLat,
        title: liveText('{share:percent} of glacier cells read {form}', {
          share: counts[top] / total,
          form: GEOMORPHON_NAMES[top]
        }),
        text: 'Ice is not a class: the classifier sees shape, not ice',
        tone: 'accent',
        priority: 5
      }
    ]);
  }

  // --- The pinned cell ---------------------------------------------------------------------------
  const hut = ALPS.places['hornli-hut'];
  const defaultPin = findCrestCell(grid, hut.lngLat, 150, 8) ?? {
    column: width >> 1,
    row: height >> 1
  };
  let pin = defaultPin;

  function updateRays(): void {
    const state = getOptions();
    if (state.product !== 'geomorphons' || !state.showRays) {
      rayOverlay.setCell(null);
      ctx.setAnnotations('pin', null);
      ctx.setReadout('pattern', null);
      ctx.setReadout('readsPerCell', null);
      ctx.requestLayers();
      return;
    }
    const result = probe.geomorphonRays(
      pin.column,
      pin.row,
      state.geomorphonRadius,
      state.geomorphonFlatAngle
    );
    rayOverlay.setCell(result);
    const reads = result.rays.reduce((sum, ray) => sum + ray.cellCount, 0);
    ctx.setReadout('pattern', result.pattern);
    ctx.setReadout('readsPerCell', reads);
    const coordinate = grid.getLongitudeLatitude(pin.column, pin.row);
    ctx.setAnnotations('pin', [
      {
        kind: 'note',
        id: 'pin-note',
        coordinate: [coordinate[0], coordinate[1]],
        title: liveText('{pattern} = {form}', {
          pattern: result.pattern,
          form: GEOMORPHON_NAMES[result.formCode] ?? 'Border'
        }),
        text: liveText('{up:integer} rise, {down:integer} fall', {
          up: result.plusCount,
          down: result.minusCount
        }),
        tone: 'signal',
        priority: 9
      }
    ]);
    ctx.requestLayers();
  }

  // --- Statistics from readbacks -----------------------------------------------------------------
  function publishFormStats(counts: Float64Array): void {
    let total = 0;
    for (let code = 1; code <= 10; code++) total += counts[code];
    if (total === 0) return;
    const shares = Array.from({length: 11}, (_, code) => (code === 0 ? 0 : counts[code] / total));
    ctx.setLegendData('formShares', shares);
    const table = getOptions().palette === 'grass' ? tables.grass : tables.quiet;
    ctx.setChart('classShares', {
      kind: 'stacked',
      description:
        'Share of cells in each landform class, ridge-like forms first, valley-like forms last.',
      segments: GEOMORPHON_LEGEND_ORDER.map(name => {
        const code = GEOMORPHON_CODES[name];
        return {
          label: GEOMORPHON_NAMES[code],
          value: counts[code],
          color: getSwatchColor(table.colors[code], groundTone)
        };
      })
    });
    ctx.setReadout('cells', total);
    ctx.setReadout('peakShare', counts[PEAK_CODE] / total);
  }

  function publishAgreementStats(counts: Float64Array): void {
    const disagree = counts[1];
    const opposite = counts[2];
    const total = counts[0] + disagree + opposite;
    if (total === 0) return;
    ctx.setLegendData('agreementShares', [counts[0] / total, disagree / total, opposite / total]);
    ctx.setReadout('agreeShare', counts[0] / total);
    ctx.setReadout('oppositeShare', opposite / total);
  }

  function publishScaleStats(counts: Float64Array): void {
    let total = 0;
    for (let code = 1; code <= 8; code++) total += counts[code];
    if (total === 0) return;
    ctx.setLegendData(
      'scaleShares',
      Array.from({length: 9}, (_, code) => (code === 0 ? 0 : counts[code] / total))
    );
  }

  /** Freezes the breaks of a signed product at its first reading and draws its histogram. */
  function updateSignedScale(target: Display, values: Float32Array): void {
    const key = target.signedKey;
    if (!key) return;
    if (!frozenP98.has(key) && target.tableId !== 'dev') {
      const p98 = getAbsolutePercentile(values, 0.98);
      if (!(p98 > 0)) return;
      frozenP98.set(key, p98);
      signedTables.set(
        key,
        key.startsWith('curvature:')
          ? makeCurvatureTable(p98, groundTone)
          : makePositionTable(p98, groundTone)
      );
      applyPaintParameters();
      publishSignedTable();
    }
    const table = getTable(target);
    const p98 = frozenP98.get(key);
    if (key.startsWith('curvature:') && table && p98) {
      const high = p98 * 1.2;
      const counts = getHistogram(values, -high, high, HISTOGRAM_BINS);
      ctx.setChart('curvatureHistogram', {
        kind: 'histogram',
        values: counts,
        xDomain: [-high, high],
        breaks: table.breaks,
        classColors: table.colors.map(color => getSwatchColor(color, groundTone)),
        xLabel: 'Curvature (1/m), tails clipped',
        formatX: value => value.toPrecision(2),
        description:
          'Histogram of the curvature of every cell, bars coloured by class, with the frozen class breaks.',
        yLabel: 'Cells'
      });
      const positive = table.breaks.slice(table.breaks.length / 2);
      const format = table.format ?? ((value: number) => value.toPrecision(2));
      ctx.setReadout('breaks', `± ${positive.map(value => format(value)).join(', ')} 1/m`);
      ctx.setReadout('p98', `${p98.toPrecision(2)} 1/m`);
    } else if (p98) {
      ctx.setReadout('p98', `${p98.toPrecision(2)} m`);
    }
  }

  function handleRead(ref: ValueRef, bytes: Uint8Array): void {
    const target = display;
    if (!target) return;
    const values =
      ref.format === 'uint32'
        ? new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2)
        : new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2);
    latest.set(ref.id, values);
    if (values instanceof Uint32Array) {
      if (ref === target.forms) {
        publishFormStats(countCodes(values));
        updateGlacierNote();
      } else if (ref === target.agreement) {
        publishAgreementStats(countCodes(values));
      } else if (ref === target.scale) {
        publishScaleStats(countCodes(values));
      }
    } else if (ref === target.value) {
      updateSignedScale(target, values);
    }
    if (ref === target.value) void measureDisplay(target, false);
    ctx.requestLayers();
  }

  function scheduleReads(
    commandEncoder: Parameters<SceneInstance<LandformOptions>['encode']>[0],
    now: number
  ): void {
    const target = display;
    if (!target || sweep || readInFlight) return;
    if (readQueue.length === 0) {
      if (!readDirty || now - lastReadAt < READ_INTERVAL_MILLISECONDS) return;
      if (target.stage.isPending(ELEVATION_VERSION)) return;
      readQueue = [...target.reads];
      readDirty = false;
    }
    const ref = readQueue[0];
    const generation = displayGeneration;
    const requested = session.readBulk(commandEncoder, ref.buffer, bytes => {
      readInFlight = false;
      lastReadAt = performance.now();
      if (destroyed || generation !== displayGeneration) return;
      handleRead(ref, bytes);
    });
    if (requested) {
      readInFlight = true;
      readQueue.shift();
    }
  }

  /** Times the displayed graphs once per display, outside the frame. */
  async function measureDisplay(target: Display, force: boolean): Promise<void> {
    if (measuring || destroyed) return;
    if (!force && measuredKeys.has(target.key)) return;
    measuredKeys.add(target.key);
    measuring = true;
    ctx.setReadout('timing', 'measuring...');
    try {
      let total = 0;
      let method = 'wall clock';
      for (const stage of collectStages(target.stage)) {
        for (const graph of stage.graphs) {
          const timing = await measureCompiledGraph(
            device,
            graph as CompiledGPUCommandGraph<void>,
            {
              parameters: undefined,
              completionBuffer: target.value.buffer,
              runs: 5,
              warmUpRuns: 1,
              repetitions: 2,
              signal: ctx.signal
            }
          );
          total += timing.milliseconds;
          method = timing.method === 'gpu-timestamps' ? 'GPU' : 'wall clock';
        }
      }
      if (!destroyed) {
        ctx.setReadout(
          'timing',
          `${total.toFixed(1)} ms (${method}), ${formatCount(pixelCount)} cells`
        );
      }
    } catch {
      if (!destroyed) ctx.setReadout('timing', null);
    } finally {
      measuring = false;
    }
  }

  // --- The sweep ---------------------------------------------------------------------------------
  function startSweep(): void {
    if (sweep || getOptions().product !== 'geomorphons') return;
    sweep = {phase: 'compiling', builds: [], index: 0, ran: false, reading: false, shares: []};
    ctx.setStatus(`Compiling ${SWEEP_RADII.length} look-out radii, once each...`);
    compileNextSweepBuild();
  }

  function compileNextSweepBuild(): void {
    const state = sweep;
    if (!state || destroyed) return;
    if (state.builds.length < SWEEP_RADII.length) {
      try {
        state.builds.push(getGeomorphonBuild(SWEEP_RADII[state.builds.length]));
      } catch (error) {
        ctx.setStatus(`Sweep failed: ${(error as Error).message}`);
        sweep = null;
        return;
      }
      timers.push(setTimeout(compileNextSweepBuild, 16));
      return;
    }
    state.phase = 'running';
    state.index = 0;
    ctx.setStatus('Sweeping the look-out radius...');
  }

  function advanceSweep(
    commandEncoder: Parameters<SceneInstance<LandformOptions>['encode']>[0],
    now: number
  ): void {
    const state = sweep;
    if (!state || state.phase !== 'running') return;
    const build = state.builds[state.index];
    if (!build) return;
    if (!state.ran) {
      build.stage.markDirty();
      state.ran = build.stage.run(commandEncoder, now, ELEVATION_VERSION);
    }
    if (!state.ran || state.reading) return;
    const requested = session.readBulk(commandEncoder, build.values.forms.buffer, bytes => {
      if (destroyed || sweep !== state) return;
      const counts = countCodes(
        new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2)
      );
      let total = 0;
      for (let code = 1; code <= 10; code++) total += counts[code];
      state.shares.push(
        SWEEP_GROUPS.map(group =>
          total > 0 ? (group.codes.reduce((sum, code) => sum + counts[code], 0) / total) * 100 : 0
        )
      );
      state.index++;
      state.ran = false;
      state.reading = false;
      if (state.index >= state.builds.length) finishSweep(state);
    });
    if (requested) state.reading = true;
  }

  function finishSweep(state: SweepState): void {
    sweep = null;
    ctx.setStatus('');
    ctx.setChart('sweepChart', {
      kind: 'line',
      series: SWEEP_GROUPS.map((group, groupIndex) => ({
        label: group.label,
        x: SWEEP_RADII,
        y: state.shares.map(row => row[groupIndex]),
        points: true,
        color: groupIndex
      })),
      xScale: 'log',
      xDomain: [SWEEP_RADII[0], SWEEP_RADII[SWEEP_RADII.length - 1]],
      xLabel: 'Look-out radius L (px)',
      yLabel: 'Share of cells (%)',
      formatY: value => `${value.toFixed(0)}`,
      link: {option: 'geomorphonRadius', label: value => `L = ${value} px`},
      description:
        'Share of cells in ridge-like, slope, valley-like and flat forms at five look-out radii.'
    });
    ctx.setCost({
      records: pixelCount,
      passes: SWEEP_RADII.length * 2,
      note: `${SWEEP_RADII.length} radii compiled once each, then cached`
    });
    readDirty = true;
    updateReadoutsOfState();
  }

  // --- Layers ------------------------------------------------------------------------------------
  function getLayerStyle(target: Display): {alpha: number; blendMode: 'normal' | 'multiply'} {
    const scale = getOptions().opacity;
    switch (target.tableId) {
      case 'quiet':
        return groundTone === 'light'
          ? {alpha: 0.85 * scale, blendMode: 'multiply'}
          : {alpha: 0.85 * scale, blendMode: 'normal'};
      case 'grass':
      case 'weiss':
        return {alpha: 0.62 * scale, blendMode: 'normal'};
      case 'agreement':
        return {alpha: 0.9 * scale, blendMode: 'normal'};
      case 'scale':
        return {alpha: 0.95 * scale, blendMode: 'normal'};
      default:
        return {alpha: scale, blendMode: 'normal'};
    }
  }

  function getLayers(): Layer[] {
    const state = getOptions();
    const layers: Layer[] = [];
    if (state.underlay) layers.push(terrainGround.getLayer({opacity: 1}));
    const target = display;
    if (target && painted && getTable(target)) {
      const common = {
        coordinateOrigin: [grid.origin[0], grid.origin[1], 0] as [number, number, number],
        gridSize: [width, height] as [number, number],
        bounds: grid.bounds,
        ...getLayerStyle(target)
      };
      if (target.compare && ctx.getCompare()) {
        layers.push(
          new ColorRasterLayer({
            ...common,
            id: 'landforms-classes-a',
            colors: colorsCompare,
            compareSide: 'a'
          }),
          new ColorRasterLayer({
            ...common,
            id: 'landforms-classes-b',
            colors: colorsMain,
            compareSide: 'b'
          })
        );
      } else {
        layers.push(new ColorRasterLayer({...common, id: 'landforms-classes', colors: colorsMain}));
      }
    }
    if (state.product === 'geomorphons' && state.showRays) {
      layers.push(...rayOverlay.getLayers(groundTone));
    }
    return layers;
  }

  // --- Places ------------------------------------------------------------------------------------
  // Six place labels in all (the overlay draws the ones inside the frame): the summits are snapped
  // to the highest cell of this DEM, the glacier name sits inside the window.
  ctx.setAnnotations('places', [
    ...snapPeaksToDem(
      [terrainLabel('matterhorn', {priority: 6}), terrainLabel('gornergrat', {priority: 4})],
      dem
    ),
    terrainLabel('hornli-hut', {priority: 5, minZoom: 11}),
    terrainLabel('riffelsee', {priority: 3, minZoom: 11}),
    terrainLabel('zermatt', {priority: 2}),
    ...glacierNames
  ] as MapAnnotation[]);

  // --- Tooltip -----------------------------------------------------------------------------------
  function getTooltip(
    event: Parameters<NonNullable<SceneInstance<LandformOptions>['getTooltip']>>[0]
  ): TooltipContent | null {
    const target = display;
    if (!target || !event.coordinate) return null;
    const pixel = grid.getPixel(event.coordinate[0], event.coordinate[1]);
    if (!pixel) return null;
    const [column, row] = pixel;
    const index = row * width + column;
    const state = getOptions();
    const elevation = grid.cpuElevation[index];
    const horn = probe.hornAt(column, row);
    const common: TooltipRow[] = [
      {label: 'Elevation', value: Math.round(elevation).toLocaleString('en-US'), unit: 'm'},
      {label: 'Slope', value: Math.round(horn.slopeDeg), unit: '°'}
    ];
    const swatchOf = (table: ClassTable | null, classIndex: number) =>
      table && classIndex >= 0 && classIndex < table.colors.length
        ? getSwatchColor(table.colors[classIndex], groundTone)
        : undefined;
    const table = getTable(target);

    const getFormContent = (): TooltipContent => {
      const result = probe.geomorphonRays(
        column,
        row,
        state.geomorphonRadius,
        state.geomorphonFlatAngle
      );
      const forms = target.forms ? latest.get(target.forms.id) : undefined;
      const code = forms ? forms[index] : result.formCode;
      const formTable = state.palette === 'grass' ? tables.grass : tables.quiet;
      return {
        title: GEOMORPHON_NAMES[code] ?? 'Border',
        subtitle: `${result.plusCount} rays rise, ${result.minusCount} fall`,
        rows: [
          {
            label: 'Landform',
            value: GEOMORPHON_NAMES[code] ?? 'Border',
            swatch: swatchOf(formTable, code),
            emphasis: true
          },
          {label: 'Rays', value: result.pattern},
          ...common
        ]
      };
    };

    if (target.tableId === 'quiet' || target.tableId === 'grass') return getFormContent();

    const value = latest.get(target.value.id)?.[index];
    if (value === undefined) return {title: 'Reading the map...', rows: common};

    switch (target.tableId) {
      case 'weiss': {
        const label = WEISS_LABELS[value - 1];
        return {
          title: label ?? 'No class',
          rows: [
            {
              label: 'Landform',
              value: label ?? 'No class',
              swatch: swatchOf(table, value),
              emphasis: true
            },
            ...common
          ]
        };
      }
      case 'agreement': {
        const weissCode = target.weiss ? latest.get(target.weiss.id)?.[index] : undefined;
        const labels = ['Agree', 'Disagree', 'Opposite', 'Not compared'];
        const formCode = target.forms ? latest.get(target.forms.id)?.[index] : undefined;
        return {
          title: labels[value] ?? 'Not compared',
          rows: [
            {
              label: 'Weiss and geomorphons',
              value: labels[value] ?? 'Not compared',
              swatch: swatchOf(table, value),
              emphasis: true
            },
            ...(weissCode !== undefined
              ? [{label: 'Weiss', value: WEISS_LABELS[weissCode - 1] ?? 'No class'}]
              : []),
            ...(formCode !== undefined
              ? [{label: 'Geomorphon', value: GEOMORPHON_NAMES[formCode] ?? 'Border'}]
              : []),
            ...common
          ]
        };
      }
      case 'scale': {
        const labels = getScaleLabels(SCALE_PRESETS[state.scalePreset]);
        const label = labels[value - 1];
        return {
          title: label ? `Strongest at a ${label} window radius` : 'No dominant scale',
          rows: [
            {
              label: 'Strongest window radius',
              value: label ?? 'none',
              swatch: swatchOf(table, value),
              emphasis: true
            },
            ...common
          ]
        };
      }
      case 'curvature': {
        const number = value as number;
        const kind = getCurvatureKind(state);
        const classIndex = table ? getClassIndexOf(table, number) : -1;
        const middle = table ? Math.floor(table.colors.length / 2) : -1;
        const reading =
          classIndex === middle
            ? 'near flat'
            : number > 0
              ? kind === 'profile' || kind === 'plan'
                ? 'convex'
                : 'positive'
              : kind === 'profile' || kind === 'plan'
                ? 'concave'
                : 'negative';
        return {
          title: `${getCurvatureName(kind)}: ${reading}`,
          rows: [
            {
              label: getCurvatureName(kind),
              value: Number.isFinite(number) ? number.toPrecision(2) : 'no value',
              unit: '1/m',
              swatch: swatchOf(table, classIndex),
              emphasis: true
            },
            {label: 'Reading', value: reading},
            ...common
          ]
        };
      }
      default: {
        // tpi and dev.
        const number = value as number;
        const radius = SCALE_PRESETS[state.scalePreset][state.scaleIndex];
        const windowCells = 2 * radius + 1;
        const windowMeters = windowCells * cellMeters;
        const classIndex = table ? getClassIndexOf(table, number) : -1;
        const west = Math.max(0, column - radius);
        const east = Math.min(width - 1, column + radius);
        const north = Math.max(0, row - radius);
        const south = Math.min(height - 1, row + radius);
        const northWest = grid.getLongitudeLatitude(west - 0.5, north - 0.5);
        const southEast = grid.getLongitudeLatitude(east + 0.5, south + 0.5);
        const isTpi = target.tableId === 'tpi';
        const direction = number >= 0 ? 'above' : 'below';
        return {
          title: isTpi
            ? `${Math.abs(number).toFixed(0)} m ${direction} the mean of its ${formatDistance(windowMeters)} window`
            : `${Math.abs(number).toFixed(1)} standard deviations ${direction} its ${formatDistance(windowMeters)} window`,
          rows: [
            {
              label: isTpi ? 'Position' : 'Deviation',
              value: Number.isFinite(number) ? formatSigned(number, isTpi ? 1 : 2) : 'no value',
              unit: isTpi ? 'm' : 'standard deviations',
              swatch: swatchOf(table, classIndex),
              emphasis: true
            },
            {label: 'Window', value: `${windowCells} x ${windowCells}`, unit: 'cells'},
            ...common
          ],
          highlight: {
            kind: 'box',
            bounds: [northWest[0], southEast[1], southEast[0], northWest[1]]
          }
        };
      }
    }
  }

  // --- Option handling ---------------------------------------------------------------------------
  const COMPILE_OPTIONS = new Set<keyof LandformOptions>([
    'product',
    'geomorphonRadius',
    'geomorphonSkip',
    'geomorphonComparison',
    'compareRadius',
    'curvatureMethod',
    'curvatureMoreKinds',
    'curvatureBorder',
    'ringRadiusInner',
    'ringRadiusOuter',
    'ringSquash',
    'positionProduct',
    'scalePreset',
    'innerFraction',
    'quantum',
    'weissSmall',
    'weissLarge',
    'weissStandardization'
  ]);
  const PARAMETER_OPTIONS = new Set<keyof LandformOptions>([
    'geomorphonFlatAngle',
    'geomorphonFlatDistance',
    'flatGradient',
    'zFactor',
    'ringGainInner',
    'ringGainOuter',
    'scaleIndex',
    'weissThreshold',
    'weissSlope'
  ]);
  const SLIDER_OPTIONS = new Set<keyof LandformOptions>([
    'geomorphonRadius',
    'weissSmall',
    'weissLarge',
    'ringRadiusInner',
    'ringRadiusOuter',
    'innerFraction'
  ]);

  const markStagesDirty = () =>
    forEachBuild(build => {
      build.stage.markDirty();
    });

  syncDisplay();
  publishTables();

  return {
    getCompiledGraphs() {
      const graphs: CompiledGPUCommandGraph<never>[] = [...session.getCompiledGraphs()];
      forEachBuild(build => {
        for (const value of Object.values(build.values)) {
          for (const graph of value.paints.values()) {
            graphs.push(graph as unknown as CompiledGPUCommandGraph<never>);
          }
        }
      });
      return graphs;
    },

    setOption(id) {
      const key = id as keyof LandformOptions;
      if (COMPILE_OPTIONS.has(key)) {
        scheduleSync(
          SLIDER_OPTIONS.has(key) ? SLIDER_SYNC_DELAY_MILLISECONDS : SYNC_DELAY_MILLISECONDS
        );
      } else if (PARAMETER_OPTIONS.has(key)) {
        markStagesDirty();
        if (key === 'geomorphonFlatAngle') updateRays();
        updateReadoutsOfState();
        publishFurniture();
      } else if (key === 'curvatureKind' || key === 'view' || key === 'palette') {
        scheduleSync(0);
      } else if (key === 'showRays') {
        updateRays();
        updateGlacierNote();
      } else {
        ctx.requestLayers();
      }
      if (key === 'scaleIndex' || key === 'geomorphonFlatAngle') publishFurniture();
    },

    onAction(id) {
      if (id === 'sweep') startSweep();
      else if (id === 'measure' && display) void measureDisplay(display, true);
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    onGroundChange(nextGround) {
      groundTone = nextGround;
      terrainGround.setGround(nextGround);
      tables = makeLandformTables(nextGround);
      deviationTable = makeDeviationTable(nextGround);
      rebuildSignedTables();
      publishTables();
      publishSignedTable();
      applyPaintParameters();
      const forms = display?.forms ? latest.get(display.forms.id) : undefined;
      if (forms instanceof Uint32Array) publishFormStats(countCodes(forms));
      ctx.requestLayers();
    },

    onLegendFilter(legendId, classes) {
      const target = display;
      if (!target) return;
      if (!classes) isolate = null;
      else if (legendId === 'landform') {
        isolate = classes
          .filter(entry => entry < GEOMORPHON_LEGEND_ORDER.length)
          .map(entry => GEOMORPHON_CODES[GEOMORPHON_LEGEND_ORDER[entry]]);
      } else if (legendId === 'scale') {
        isolate = classes.map(entry => entry + 1);
      } else {
        isolate = [...classes];
      }
      applyPaintParameters();
      ctx.requestLayers();
    },

    onCompareChange() {
      ctx.requestLayers();
    },

    onClick(event) {
      const state = getOptions();
      if (state.product !== 'geomorphons' || !state.showRays || !event.coordinate) return false;
      const pixel = grid.getPixel(event.coordinate[0], event.coordinate[1]);
      if (!pixel) return false;
      pin = {column: pixel[0], row: pixel[1]};
      updateRays();
      return true;
    },

    getTooltip,

    encode(commandEncoder) {
      if (destroyed) return;
      const now = performance.now();
      advanceSweep(commandEncoder, now);
      const target = display;
      if (!target) return;
      const radius = target.dilate ? getDilationRadius(ctx.getViewState().zoom) : 0;
      if (radius !== dilateRadius) {
        dilateRadius = radius;
        applyPaintParameters();
      }
      let ran = target.stage.run(commandEncoder, now, ELEVATION_VERSION);
      if (target.compare)
        ran = target.compare.stage.run(commandEncoder, now, ELEVATION_VERSION) || ran;
      if (ran) {
        paintDirty = true;
        readDirty = true;
      }
      if (paintDirty && getTable(target)) {
        target.value.paints.get('main')?.encode(commandEncoder, {parameters: undefined});
        if (target.compare) {
          target.compare.value.paints
            .get('compare')
            ?.encode(commandEncoder, {parameters: undefined});
        }
        paintDirty = false;
        painted = true;
      }
      scheduleReads(commandEncoder, now);
    },

    getLayers,

    destroy() {
      destroyed = true;
      if (syncTimer) clearTimeout(syncTimer);
      for (const timer of timers) clearTimeout(timer);
      timers = [];
      // Dependents first: the agreement graph reads the Weiss and geomorphon rasters.
      for (const entry of [agreementEntry, weissEntry, positionEntry, curvatureEntry]) {
        if (entry) entry.build.release();
      }
      for (const build of geomorphonBuilds.values()) build.release();
      geomorphonBuilds.clear();
      session.destroy();
      terrainGround.destroy();
      resources.destroy();
    }
  };
}
