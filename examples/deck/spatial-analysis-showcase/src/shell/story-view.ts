// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import './story-view.css';
import type {DataCatalog, LoadedDataset} from '../data/catalog';
import {loadDatasetInfo} from '../data/catalog';
import type {DatasetInfo} from '../data/dataset-types';
import type {DeckHost, HostServices, StageRect, ViewPadding} from '../engine/deck-host';
import {setCompareState} from '../engine/layers';
import {prefersReducedMotion} from '../engine/motion';
import {areValuesEqual, OptionsStore, type OptionChangeSource} from '../scenes/options-state';
import {getChapter} from '../scenes/chapters';
import {loadScene} from '../scenes/registry';
import type {BasemapSpec, FurnitureSpec, GroundPalette} from '../cartography/types';
import type {
  AnyScene,
  ChartData,
  LegendSpec,
  OptionSpec,
  OptionState,
  ReadoutSpec,
  SceneContext,
  SceneInstance,
  StepCamera,
  StoryStep,
  ViewState
} from '../scenes/scene';
import {renderChart, updateChartLink} from './chart';
import {CompareDivider, type CompareState} from './compare-divider';
import {h, clearElement, copyText, formatBytes, icon} from './dom';
import {highlightCode} from './highlight';
import {renderLegends, renderLegendThumbnail} from './legend';
import {MapChips} from './map-chips';
import {MapFurniture} from './map-furniture';
import {MapInset} from './map-inset';
import {MapTooltip} from './map-tooltip';
import {parseActionLink, renderMarkdown} from './markdown';
import {OptionsPanel, StepControls} from './options-panel';
import {getReferenceHash, getStoryHash, navigate, replaceHash, type Route} from './router';
import {openShortcutSheet, type Shortcut} from './shortcut-sheet';
import {CostRow, createPipelineStrip, type StepCost} from './story-card-parts';
import {createUnderTheHood, measureActiveScene, type UnderTheHood} from './story-engine';
import {
  formatReadout,
  getChartLink,
  getInterpolatedReadoutIds,
  getStepControlIds,
  getStepReadoutIds,
  parseOptionText,
  renderStepBody,
  snapToSlider
} from './story-readouts';
import {SheetController} from './story-sheet';
import {getTheme, onThemeChange} from './theme';
import {TimeBar, type TimeBarData} from './time-bar';

/** A scene instance with its option type erased. */
type ShellInstance = SceneInstance<OptionState>;

type StoryRoute = Extract<Route, {name: 'story'}>;

type PanelTab = 'story' | 'controls' | 'details';

const PANEL_TABS: readonly {id: PanelTab; label: string}[] = [
  {id: 'story', label: 'Story'},
  {id: 'controls', label: 'All controls'},
  {id: 'details', label: 'Details'}
];

/** Viewport width at or below which the panel is a bottom sheet. */
const MOBILE_QUERY = '(max-width: 900px)';

/** One rendered readout row; a readout can appear in the full list and inside the active step. */
type ReadoutView = {
  value: HTMLElement;
  row: HTMLElement;
  chart: HTMLElement | null;
  /** The element `renderChart` returned (the target of `updateChartLink`). */
  chartElement: HTMLElement | null;
  /** Option the chart's link marker follows. */
  linkOption: string | null;
};

/** A readout chart drawn in a map inset while the active step lists it. */
type InsetView = {inset: MapInset; chartElement: HTMLElement | null; linkOption: string | null};

const DEFAULT_FLY_MILLISECONDS = 1600;
/** Pause between steps of the auto-play tour. */
const AUTOPLAY_MILLISECONDS = 9000;
const STATS_INTERVAL_MILLISECONDS = 500;
/** Scene-driven option writes refresh legends and code at most this often. */
const DERIVED_REFRESH_MILLISECONDS = 250;
/** How long a "highlight" action link rings a place. */
const PING_MILLISECONDS = 2600;
/** Gap kept between the phone sheet and the framed subject. */
const SHEET_CLEARANCE_PIXELS = 16;
/** Camera drift (pixels of pan, zoom levels, degrees) after which "Modified" shows. */
const MOVED_PIXELS = 80;
const MOVED_ZOOM = 0.4;
const MOVED_DEGREES = 8;
const TILE_BASEMAP_CREDIT = 'Basemap © CARTO © OpenStreetMap contributors';

const SHORTCUTS: readonly Shortcut[] = [
  {keys: 'Left Right', action: 'Previous or next step (focus in the story panel)'},
  {keys: '[ ]', action: 'Previous or next step'},
  {keys: 'R', action: 'Reset the options to this step'},
  {keys: 'Esc', action: 'Return to the step view; close a pinned card'},
  {keys: '?', action: 'Show this list'}
];

/** Dependencies the story view needs from the application. */
export type StoryViewDependencies = {
  getHost: () => DeckHost;
  catalog: DataCatalog;
  hasWebGPU: boolean;
};

type MapCorners = {
  topLeft: HTMLElement;
  topRight: HTMLElement;
  bottomLeft: HTMLElement;
  bottomRight: HTMLElement;
};

/**
 * The story page: narrative panel on the left (a bottom sheet on phones), the persistent Deck map
 * on the right. It owns one scene at a time and wires the option store, the URL, the legends, the
 * readouts and every map overlay (tooltip, chips, time bar, compare divider, insets, furniture)
 * to it.
 */
export class StoryView {
  private readonly dependencies: StoryViewDependencies;
  private root: HTMLElement | null = null;
  private scene: AnyScene | null = null;
  /** Furniture fields a scene set at runtime (`ctx.setFurniture`), merged last. */
  private runtimeFurniture: FurnitureSpec | null = null;
  private store: OptionsStore | null = null;
  private instance: ShellInstance | null = null;
  private stepIndex = 0;
  private readonly gpuExtents = new Map<string, readonly [number, number]>();
  private readonly legendData: Record<string, unknown> = {};
  private readonly readoutSpecs = new Map<string, ReadoutSpec>();
  /** Latest value and chart per readout: the single source of truth for every rendered copy. */
  private readonly readoutValues = new Map<string, string | number | null>();
  private readonly readoutCharts = new Map<string, ChartData | null>();
  private readonly allReadoutViews = new Map<string, ReadoutView>();
  private stepReadoutViews = new Map<string, ReadoutView>();
  /** `{{readoutId}}` slots in the active step's body. */
  private liveReadoutSpans = new Map<string, HTMLElement[]>();
  private readonly insetViews = new Map<string, InsetView>();
  private stepControls: StepControls | null = null;
  private optionsPanel: OptionsPanel | null = null;
  private activeTab: PanelTab = 'story';
  /** Whether scene-originated option writes are currently forwarded to `instance.setOption`. */
  private notifySceneWrites = false;
  private derivedFrame = 0;
  private urlTimer = 0;
  private cleanups: (() => void)[] = [];
  private mountGeneration = 0;

  // Step state.
  /** True while `applyStepState` writes, so those writes are not counted as the reader's. */
  private applyingStepState = false;
  private appliedOptions: OptionState = {};
  /** Options the reader changed since the step was applied. */
  private readonly touchedOptions = new Set<string>();
  /** Camera the step settled on; `null` while it flies (the "Modified" check is off). */
  private stepViewReference: ViewState | null = null;
  private cameraMoved = false;
  private cameraSettleTimer = 0;
  private pingTimer = 0;
  private autoplayTimer = 0;
  private autoplayCleanup: (() => void) | null = null;

  // Cost chips and engine statistics.
  private cost: StepCost | null = null;
  private costRow: CostRow | null = null;
  private gpuMilliseconds: number | null = null;
  private measuring = false;
  private lastRebuildCount: number | null = null;
  private hood: UnderTheHood | null = null;

  // Panel parts that change while the story runs.
  private stepRows: {
    item: HTMLElement;
    row: HTMLButtonElement;
    badge: HTMLElement;
    detail: HTMLElement;
  }[] = [];
  private panel: HTMLElement | null = null;
  private progressSegments: HTMLButtonElement[] = [];
  private previousButtons: HTMLButtonElement[] = [];
  private nextButtons: HTMLButtonElement[] = [];
  private autoplayButton!: HTMLButtonElement;
  private sheetTitle!: HTMLElement;
  private announcer!: HTMLElement;
  private legendContainer!: HTMLElement;
  private legendOverlay!: HTMLElement;
  private legendToggle!: HTMLElement;
  private legendSpecs: readonly LegendSpec[] = [];
  private statusLine!: HTMLElement;
  private snippetCode!: HTMLElement;
  private snippetText = '';
  private tabButtons = new Map<PanelTab, HTMLButtonElement>();
  private tabPanels = new Map<PanelTab, HTMLElement>();
  private controlsDot!: HTMLElement;

  // Map slot parts.
  private furniture: MapFurniture | null = null;
  private mapSlot: HTMLElement | null = null;
  private ground: 'light' | 'dark' | null = null;
  /** Attribution of the scene's datasets, for the furniture credit line. */
  private credits: string[] = [];
  private mapCorners: MapCorners | null = null;
  private creditSlot: HTMLElement | null = null;
  private tooltip: MapTooltip | null = null;
  private tooltipHasContent = false;
  private chips: MapChips | null = null;
  private timeBar: TimeBar | null = null;
  private timeBarData: TimeBarData | null = null;
  private compareDivider: CompareDivider | null = null;
  private compareActive = false;
  private compareState: CompareState | null = null;
  private sheet: SheetController | null = null;
  private obstacleObserver: ResizeObserver | null = null;
  private obstacleFrame = 0;
  /** JSON key of the legends last drawn, so unchanged legends keep their DOM (and click state). */
  private lastLegendKey = '';
  private shiftHeld = false;
  private navigatingBack = false;
  private paddingTimer = 0;

  constructor(dependencies: StoryViewDependencies) {
    this.dependencies = dependencies;
  }

  /** The scene id shown, if any. */
  get sceneId(): string | null {
    return this.scene?.id ?? null;
  }

  async mount(root: HTMLElement, route: StoryRoute): Promise<void> {
    this.root = root;
    const generation = ++this.mountGeneration;
    clearElement(root);
    const scene = await loadScene(route.sceneId);
    if (generation !== this.mountGeneration) return;
    if (!scene) {
      clearElement(root);
      root.append(
        h(
          'div',
          {class: 'page narrow'},
          h('h1', {}, 'Story not found'),
          h('p', {}, `There is no story called "${route.sceneId}".`),
          h('a', {class: 'btn', href: '#/'}, 'Back to all stories')
        )
      );
      return;
    }
    this.scene = scene;
    this.activeTab = 'story';
    this.warnAboutStepReferences(scene);
    this.store = new OptionsStore(scene.options);
    this.stepIndex = this.resolveStepIndex(route.stepId);
    this.applyStepState(this.stepIndex, true);
    if (route.options) this.store.deserialize(route.options);
    this.buildLayout(root, scene);
    this.refreshStep(false);
    this.refreshLegends();
    this.refreshSnippet();
    this.startKeyboard();
    if (!this.dependencies.hasWebGPU) {
      this.statusLine.textContent = '';
      return;
    }
    await this.activate(scene, generation);
  }

  /** Handles a route change inside the same scene (step or option state). */
  update(route: StoryRoute): void {
    if (!this.scene || !this.store) return;
    const index = this.resolveStepIndex(route.stepId);
    if (index !== this.stepIndex) {
      this.navigatingBack = index < this.stepIndex;
      this.stepIndex = index;
      this.applyStepState(index, false);
      // Moving through the story always shows the step, whichever tab was open.
      this.selectTab('story');
      this.refreshStep(true);
    }
    if (route.options) this.store.deserialize(route.options);
  }

  unmount(): void {
    this.mountGeneration++;
    this.runtimeFurniture = null;
    this.stopAutoplay();
    for (const cleanup of this.cleanups) cleanup();
    this.cleanups = [];
    this.instance = null;
    const host = this.dependencies.hasWebGPU ? this.dependencies.getHost() : null;
    if (host) {
      host.onTooltip = null;
      host.onMapClick = null;
      host.onStatus = null;
      host.element.removeAttribute('role');
      host.element.removeAttribute('aria-label');
      host.deactivate();
      host.setBasemap(null);
      host.setViewPadding({});
      host.setActive(false);
    }
    setCompareState(null);
    this.compareDivider?.destroy();
    this.timeBar?.destroy();
    this.chips?.destroy();
    this.tooltip?.destroy();
    this.sheet?.destroy();
    for (const {inset} of this.insetViews.values()) inset.destroy();
    this.insetViews.clear();
    this.obstacleObserver?.disconnect();
    this.obstacleObserver = null;
    cancelAnimationFrame(this.obstacleFrame);
    this.obstacleFrame = 0;
    this.lastLegendKey = '';
    this.furniture?.destroy();
    this.furniture = null;
    this.mapSlot = null;
    this.creditSlot = null;
    this.panel = null;
    this.ground = null;
    this.credits = [];
    this.mapCorners = null;
    this.tooltip = null;
    this.chips = null;
    this.timeBar = null;
    this.timeBarData = null;
    this.compareDivider = null;
    this.compareActive = false;
    this.compareState = null;
    this.sheet = null;
    this.root?.replaceChildren();
    this.scene = null;
    this.store = null;
    this.gpuExtents.clear();
    for (const key of Object.keys(this.legendData)) delete this.legendData[key];
    this.legendSpecs = [];
    this.readoutSpecs.clear();
    this.readoutValues.clear();
    this.readoutCharts.clear();
    this.allReadoutViews.clear();
    this.stepReadoutViews.clear();
    this.liveReadoutSpans.clear();
    this.stepRows = [];
    this.progressSegments = [];
    this.previousButtons = [];
    this.nextButtons = [];
    this.stepControls = null;
    this.optionsPanel = null;
    this.costRow = null;
    this.cost = null;
    this.hood = null;
    this.gpuMilliseconds = null;
    this.measuring = false;
    this.lastRebuildCount = null;
    this.touchedOptions.clear();
    this.stepViewReference = null;
    this.cameraMoved = false;
    this.tooltipHasContent = false;
    window.clearTimeout(this.cameraSettleTimer);
    window.clearTimeout(this.paddingTimer);
    window.clearTimeout(this.pingTimer);
    clearTimeout(this.derivedFrame);
    clearTimeout(this.urlTimer);
    this.derivedFrame = 0;
  }

  // ---------------------------------------------------------------------------------------------
  // Step state and camera
  // ---------------------------------------------------------------------------------------------

  private resolveStepIndex(stepId: string | null): number {
    const scene = this.scene;
    if (!scene || !stepId) return 0;
    const byId = scene.story.findIndex(step => step.id === stepId);
    if (byId >= 0) return byId;
    const numeric = Number(stepId);
    return Number.isInteger(numeric)
      ? Math.min(Math.max(numeric - 1, 0), scene.story.length - 1)
      : 0;
  }

  /**
   * Defaults, then the options of every step up to `index` in order; a step with
   * `optionsMode: 'fresh'` starts again from the defaults, so nothing earlier leaks in.
   */
  private applyStepState(index: number, initial: boolean): void {
    const scene = this.scene;
    const store = this.store;
    if (!scene || !store) return;
    let first = 0;
    for (let step = index; step > 0; step--) {
      if (scene.story[step]?.optionsMode === 'fresh') {
        first = step;
        break;
      }
    }
    this.applyingStepState = true;
    try {
      if (!initial) store.reset();
      for (let step = first; step <= index; step++) {
        const options = scene.story[step]?.options;
        if (options) store.setMany(options as Record<string, never>);
      }
    } finally {
      this.applyingStepState = false;
    }
    this.appliedOptions = store.snapshot();
    this.touchedOptions.clear();
    this.updateModifiedChip();
  }

  /** The camera the step settles on: the last explicit view or bounds up to `index`. */
  private getEffectiveCamera(index: number): {
    view: Partial<ViewState>;
    bounds: readonly [number, number, number, number] | null;
    padding: StepCamera['padding'];
    /** Zoom ceiling and pose set by the step that set the bounds. */
    boundsOptions: {maxZoom?: number; pitch?: number; bearing?: number};
  } {
    const scene = this.scene!;
    let view: Record<string, number | undefined> = {...scene.initialView};
    let bounds: readonly [number, number, number, number] | null = null;
    let padding: StepCamera['padding'];
    let boundsOptions: {maxZoom?: number; pitch?: number; bearing?: number} = {};
    for (let step = 0; step <= index; step++) {
      const camera = scene.story[step]?.camera;
      if (!camera) continue;
      const {
        transitionMs: _transition,
        bounds: stepBounds,
        padding: stepPadding,
        ...fields
      } = camera;
      view = {...view, ...fields};
      if (stepBounds) {
        bounds = stepBounds;
        padding = stepPadding;
        boundsOptions = {maxZoom: fields.zoom, pitch: fields.pitch, bearing: fields.bearing};
      } else if (fields.longitude !== undefined || fields.latitude !== undefined) {
        bounds = null;
        boundsOptions = {};
        padding = stepPadding;
      } else if (stepPadding !== undefined) {
        padding = stepPadding;
      }
    }
    return {view, bounds, padding, boundsOptions};
  }

  private getCumulativeCamera(index: number) {
    return this.getEffectiveCamera(index).view as ViewState;
  }

  /** Scene basemap merged with every step's up to `index` (`null` when none declares one). */
  private getCumulativeBasemap(index: number): BasemapSpec | null {
    const scene = this.scene!;
    let spec: BasemapSpec | null = scene.basemap ? {...scene.basemap} : null;
    for (let step = 0; step <= index; step++) {
      const override = scene.story[step]?.basemap;
      if (override) spec = mergeBasemap(spec, override);
    }
    return spec;
  }

  /** Scene furniture merged with every step's up to `index` (`null` when none declares any). */
  private getCumulativeFurniture(index: number): FurnitureSpec | null {
    const scene = this.scene!;
    let spec: FurnitureSpec | null = scene.furniture ? {...scene.furniture} : null;
    for (let step = 0; step <= index; step++) {
      const override = scene.story[step]?.furniture;
      if (override) spec = mergeDefined(spec, override);
    }
    const runtime = this.runtimeFurniture;
    if (runtime) {
      const merged = mergeDefined(spec, runtime);
      // Runtime cartouche and scale-bar fields refine the step's objects instead of replacing them.
      const title = spec?.title;
      if (typeof runtime.title === 'object') {
        // `title: true` means scene title + step title as subtitle: keep that subtitle.
        const base =
          typeof title === 'object'
            ? title
            : title === true
              ? {subtitle: scene.story[index]?.title ?? ''}
              : {};
        merged.title = title === false ? false : {...base, ...runtime.title};
      }
      const scaleBar = spec?.scaleBar;
      if (typeof runtime.scaleBar === 'object') {
        // A step that turns the scale bar off keeps it off.
        merged.scaleBar =
          scaleBar === false
            ? false
            : {...(typeof scaleBar === 'object' ? scaleBar : {}), ...runtime.scaleBar};
      }
      spec = merged;
    }
    return spec;
  }

  /**
   * Moves the camera to the step's view: bounds are fitted to the free map area, an explicit
   * view goes through `setViewState` with the standing padding (the phone sheet) plus the step's
   * `padding`. Jumps under reduced motion (the host zeroes the flight).
   */
  private moveToStepCamera(animate: boolean): void {
    const scene = this.scene;
    if (!scene || !this.dependencies.hasWebGPU) return;
    const host = this.dependencies.getHost();
    const step = scene.story[this.stepIndex];
    const {view, bounds, padding, boundsOptions} = this.getEffectiveCamera(this.stepIndex);
    const transitionMs = animate ? (step.camera?.transitionMs ?? DEFAULT_FLY_MILLISECONDS) : 0;
    const extra = normalizePadding(padding);
    if (bounds) {
      if (boundsOptions.pitch !== undefined || boundsOptions.bearing !== undefined) {
        host.setViewState({pitch: boundsOptions.pitch, bearing: boundsOptions.bearing});
      }
      host.fitBounds(bounds, {
        padding: extra ?? undefined,
        maxZoom: boundsOptions.maxZoom,
        transitionMs
      });
    } else {
      const standing = host.getViewPadding();
      host.setViewState({
        ...view,
        transitionMs,
        padding: extra ? addPadding(standing, extra) : undefined
      });
    }
    window.clearTimeout(this.cameraSettleTimer);
    this.stepViewReference = null;
    this.setCameraMoved(false);
    const settleWait = prefersReducedMotion() ? 0 : transitionMs;
    this.cameraSettleTimer = window.setTimeout(() => {
      this.stepViewReference = host.getViewState();
    }, settleWait + 120);
  }

  /** Keeps the standing view padding in step with the phone sheet. */
  private updateViewPadding(recenter: boolean): void {
    if (!this.dependencies.hasWebGPU) return;
    const host = this.dependencies.getHost();
    const sheetHeight = this.sheet?.getHeight() ?? 0;
    // Before the slot is attached its height reads 0; the window is a fair stand-in.
    const slotHeight = this.mapSlot?.clientHeight || window.innerHeight;
    const bottom =
      sheetHeight > 0 ? Math.min(sheetHeight + SHEET_CLEARANCE_PIXELS, slotHeight * 0.6) : 0;
    const reference = this.stepViewReference;
    host.setViewPadding({bottom: Math.round(bottom)}, {recenter, transitionMs: 200});
    if (recenter && reference) {
      // The recentre slides the view by design: it is not the reader moving away from the step.
      const wasMoved = this.cameraMoved;
      this.stepViewReference = null;
      window.clearTimeout(this.paddingTimer);
      this.paddingTimer = window.setTimeout(() => {
        this.stepViewReference = wasMoved ? reference : host.getViewState();
      }, 320);
    }
    this.scheduleObstacles();
  }

  /** Checks the live camera against the step's (cheap arithmetic; runs on every view change). */
  private checkCameraMoved(view: {
    longitude: number;
    latitude: number;
    zoom: number;
    pitch?: number;
    bearing?: number;
  }): void {
    const reference = this.stepViewReference;
    if (!reference) return;
    const worldSize = 512 * 2 ** view.zoom;
    const dx = (Math.abs(view.longitude - reference.longitude) / 360) * worldSize;
    const dy = Math.abs(getMercatorY(view.latitude) - getMercatorY(reference.latitude)) * worldSize;
    const moved =
      Math.hypot(dx, dy) > MOVED_PIXELS ||
      Math.abs(view.zoom - reference.zoom) > MOVED_ZOOM ||
      Math.abs((view.pitch ?? 0) - (reference.pitch ?? 0)) > MOVED_DEGREES ||
      Math.abs((view.bearing ?? 0) - (reference.bearing ?? 0)) > MOVED_DEGREES;
    this.setCameraMoved(moved);
  }

  private setCameraMoved(moved: boolean): void {
    if (moved === this.cameraMoved) return;
    this.cameraMoved = moved;
    this.updateModifiedChip();
  }

  /** Option ids that are playback or display state, not the reader's choices. */
  private isTransientOption(id: string): boolean {
    const timeline = this.scene?.timeline;
    const clock = this.getCumulativeFurniture(this.stepIndex)?.clock;
    return (
      (timeline !== undefined &&
        [timeline.time, timeline.play, timeline.speed, timeline.window].includes(id)) ||
      (clock !== undefined && clock !== false && clock.option === id)
    );
  }

  private areOptionsModified(): boolean {
    const store = this.store;
    if (!store) return false;
    for (const id of this.touchedOptions) {
      if (!areValuesEqual(store.get(id), this.appliedOptions[id])) return true;
    }
    return false;
  }

  private updateModifiedChip(): void {
    this.chips?.setModified(this.areOptionsModified() || this.cameraMoved, () =>
      this.resetToStep()
    );
  }

  /** Puts the options and the camera back to the step's. */
  private resetToStep(): void {
    this.applyStepState(this.stepIndex, false);
    this.moveToStepCamera(true);
  }

  // ---------------------------------------------------------------------------------------------
  // Cartography
  // ---------------------------------------------------------------------------------------------

  /** Applies the step's basemap and furniture (they do not need a running scene instance). */
  private applyStepCartography(): void {
    const scene = this.scene;
    if (!scene || !this.dependencies.hasWebGPU) return;
    const host = this.dependencies.getHost();
    host.setBasemap(this.getCumulativeBasemap(this.stepIndex));
    this.updateGround();
    this.furniture?.setSpec(this.getCumulativeFurniture(this.stepIndex), {
      sceneTitle: scene.title,
      stepTitle: scene.story[this.stepIndex]?.title ?? '',
      credits: this.credits,
      basemapCredit: host.basemap.style === 'none' ? undefined : TILE_BASEMAP_CREDIT
    });
    this.syncClock();
    this.scheduleObstacles();
  }

  /** Feeds the furniture clock the current value of its option. */
  private syncClock(): void {
    const clock = this.getCumulativeFurniture(this.stepIndex)?.clock;
    if (!clock || !this.store) {
      this.furniture?.updateClock(null);
      return;
    }
    const value = Number(this.store.get(clock.option));
    this.furniture?.updateClock(Number.isFinite(value) ? value : null);
  }

  /** Mirrors the map ground on the map slot (label tokens follow it) and tells the scene. */
  private updateGround(): void {
    const ground = this.dependencies.getHost().ground;
    if (this.mapSlot) this.mapSlot.dataset['ground'] = ground;
    this.chips?.setGroundNote(ground === getTheme() ? null : `Map: ${ground} ground`);
    if (ground === this.ground) return;
    const changed = this.ground !== null;
    this.ground = ground;
    if (changed) this.instance?.onGroundChange?.(ground);
  }

  /** Moves the legend between corners: bottom-left on desktop, top-left (under the title) on phones. */
  private placeLegend(mobile: boolean): void {
    const corners = this.mapCorners;
    if (!corners) return;
    if (mobile) corners.topLeft.append(this.legendOverlay);
    else corners.bottomLeft.prepend(this.legendOverlay);
    this.scheduleObstacles();
  }

  /** Starts or stops the step's compare divider (state never leaks between steps). */
  private applyStepCompare(): void {
    const divider = this.compareDivider;
    const step = this.scene?.story[this.stepIndex];
    if (!divider || !step) return;
    this.compareActive = Boolean(step.compare);
    divider.setSpec(step.compare ?? null);
    if (!step.compare) {
      setCompareState(null);
      this.compareState = null;
    }
  }

  private handleCompareChange(state: CompareState): void {
    if (!this.compareActive) {
      setCompareState(null);
      this.compareState = null;
      this.instance?.onCompareChange?.({position: 0.5, showing: 'both'});
      return;
    }
    this.compareState = state;
    setCompareState(state);
    this.instance?.onCompareChange?.(state);
  }

  /** Readout charts with `placement: 'map'` listed by the active step get an inset card. */
  private syncInsets(): void {
    const scene = this.scene;
    const corners = this.mapCorners;
    if (!scene || !corners) return;
    const wanted = new Set(
      getStepReadoutIds(scene.story[this.stepIndex]).filter(id =>
        isMapChart(this.readoutSpecs.get(id))
      )
    );
    for (const [id, view] of this.insetViews) {
      if (wanted.has(id)) continue;
      this.obstacleObserver?.unobserve(view.inset.element);
      view.inset.destroy();
      this.insetViews.delete(id);
    }
    for (const id of wanted) {
      const spec = this.readoutSpecs.get(id)!;
      if (!this.insetViews.has(id)) {
        const corner = spec.mapCorner ?? 'top-right';
        const inset = new MapInset({title: spec.label, corner});
        this.insetViews.set(id, {inset, chartElement: null, linkOption: null});
        const stack = {
          'top-right': corners.topRight,
          'top-left': corners.topLeft,
          'bottom-right': corners.bottomRight,
          'bottom-left': corners.bottomLeft
        }[corner];
        stack.append(inset.element);
        this.obstacleObserver?.observe(inset.element);
      }
      this.paintInset(id);
    }
  }

  private paintInset(id: string): void {
    const view = this.insetViews.get(id);
    const spec = this.readoutSpecs.get(id);
    if (!view || !spec) return;
    const chart = this.readoutCharts.get(id);
    if (!chart) {
      view.chartElement = null;
      view.linkOption = null;
      view.inset.setContent(null, spec.label);
      return;
    }
    const rendered = this.renderLinkedChart(chart);
    view.chartElement = rendered.element;
    view.linkOption = rendered.linkOption;
    view.inset.setContent(rendered.element, spec.label);
    this.scheduleObstacles();
  }

  // ---------------------------------------------------------------------------------------------
  // Obstacles for annotation labels
  // ---------------------------------------------------------------------------------------------

  private scheduleObstacles(): void {
    if (this.obstacleFrame || !this.dependencies.hasWebGPU) return;
    this.obstacleFrame = requestAnimationFrame(() => {
      this.obstacleFrame = 0;
      this.reportObstacles();
    });
  }

  /** Tells the annotation placer which map rectangles it must keep clear (event driven). */
  private reportObstacles(): void {
    const slot = this.mapSlot;
    const furniture = this.furniture;
    if (!slot || !furniture || !this.dependencies.hasWebGPU) return;
    const origin = slot.getBoundingClientRect();
    const elements: (HTMLElement | null)[] = [
      furniture.cartouche,
      furniture.northArrow,
      furniture.scaleBar,
      furniture.credit,
      furniture.clock,
      this.legendOverlay,
      ...(this.chips ? [...this.chips.element.children] : []).map(child => child as HTMLElement),
      this.timeBar?.element ?? null,
      ...[...this.insetViews.values()].map(view => view.inset.element)
    ];
    const rects: StageRect[] = [];
    for (const element of elements) {
      if (!element) continue;
      const box = element.getBoundingClientRect();
      if (box.width <= 0 || box.height <= 0) continue;
      rects.push({
        x: box.left - origin.left,
        y: box.top - origin.top,
        width: box.width,
        height: box.height
      });
    }
    const sheetHeight = this.sheet?.getHeight() ?? 0;
    if (sheetHeight > 0) {
      rects.push({x: 0, y: origin.height - sheetHeight, width: origin.width, height: sheetHeight});
    }
    this.dependencies.getHost().setAnnotationObstacles(rects);
  }

  // ---------------------------------------------------------------------------------------------
  // Layout
  // ---------------------------------------------------------------------------------------------

  private buildLayout(root: HTMLElement, scene: AnyScene): void {
    const store = this.store!;
    const chapter = getChapter(scene.chapter);
    const host = this.dependencies.hasWebGPU ? this.dependencies.getHost() : null;

    this.statusLine = h('p', {class: 'status-line', role: 'status'});
    this.announcer = h('div', {
      class: 'visually-hidden',
      'aria-live': 'polite',
      'aria-atomic': 'true'
    });
    const header = h(
      'header',
      {class: 'story-header'},
      h(
        'nav',
        {class: 'breadcrumb', 'aria-label': 'Breadcrumb'},
        h('a', {href: '#/'}, 'Stories'),
        h('span', {}, '/'),
        h('a', {href: `#/?chapter=${scene.chapter}`}, chapter?.title ?? scene.chapter)
      ),
      h('h1', {}, scene.title)
    );

    // Story tab: the step list on its rail, then every readout.
    const accordion = h('ol', {class: 'step-list', 'aria-label': 'Story steps'});
    this.stepRows = scene.story.map((step, index) => {
      const badge = h('span', {class: 'step-badge'}, String(index + 1));
      const detailId = `step-detail-${index}`;
      const row = h(
        'button',
        {
          class: 'step-row',
          type: 'button',
          'aria-controls': detailId,
          'aria-expanded': 'false'
        },
        badge,
        h('span', {class: 'step-row-title'}, step.title)
      );
      row.addEventListener('click', () => this.goToStep(index));
      const detail = h('div', {class: 'step-detail', id: detailId, hidden: true});
      const item = h('li', {class: 'step-item'}, row, detail);
      accordion.append(item);
      return {item, row, badge, detail};
    });

    const readoutList = h('dl', {class: 'readouts'});
    const hoodReadoutList = h('dl', {class: 'readouts hood-readouts'});
    for (const spec of scene.readouts ?? []) {
      this.readoutSpecs.set(spec.id, spec);
      // Map insets draw their own copy while a step lists them; they have no row.
      if (isMapChart(spec)) continue;
      const view = this.createReadoutView(spec, false);
      this.allReadoutViews.set(spec.id, view);
      (spec.hood ? hoodReadoutList : readoutList).append(view.row);
    }
    const allReadouts = readoutList.childElementCount
      ? collapsible(
          'All readouts',
          readoutList,
          false,
          h('span', {class: 'count'}, String(readoutList.childElementCount))
        )
      : null;
    const storyPanel = h('div', {class: 'tab-panel'}, accordion, allReadouts);

    // All controls tab.
    this.controlsDot = h('span', {
      class: 'modified-dot',
      hidden: true,
      title: 'Changed from defaults'
    });
    const optionsPanel = new OptionsPanel(
      store,
      id => this.instance?.onAction?.(id, store.snapshot()),
      modified => {
        this.controlsDot.hidden = !modified;
      }
    );
    this.optionsPanel = optionsPanel;
    this.cleanups.push(() => optionsPanel.destroy());
    this.cleanups.push(() => this.stepControls?.destroy());
    const controlsPanel = h('div', {class: 'tab-panel'}, optionsPanel.element);

    // Code.
    this.snippetCode = h('code', {});
    const copyButton = h(
      'button',
      {class: 'btn btn-ghost btn-small', type: 'button'},
      icon('copy', 14),
      'Copy'
    );
    copyButton.addEventListener('click', async () => {
      if (await copyText(this.snippetText)) {
        copyButton.replaceChildren(icon('check', 14), 'Copied');
        setTimeout(() => copyButton.replaceChildren(icon('copy', 14), 'Copy'), 1400);
      }
    });
    const codeSection = scene.snippet
      ? collapsible(
          'Code',
          h(
            'div',
            {class: 'code-block'},
            h('div', {class: 'code-toolbar'}, copyButton),
            h('pre', {}, this.snippetCode)
          ),
          false
        )
      : null;

    // About.
    const aboutParts: HTMLElement[] = [];
    for (const [heading, markdown] of [
      ['What it computes', scene.about?.what],
      ['Why it matters', scene.about?.why],
      ['How to read the map', scene.about?.howToRead]
    ] as const) {
      if (!markdown) continue;
      aboutParts.push(
        h(
          'section',
          {},
          h('h4', {}, heading),
          h('div', {class: 'prose', html: renderMarkdown(markdown)})
        )
      );
    }
    const aboutSection = aboutParts.length
      ? collapsible('About this analysis', h('div', {}, aboutParts), false)
      : null;

    // Datasets.
    const datasetList = h('ul', {class: 'dataset-list'});
    const layoutGeneration = this.mountGeneration;
    void Promise.all(scene.datasets.map(ref => loadDatasetInfo(ref.id))).then(infos => {
      if (layoutGeneration !== this.mountGeneration) return;
      scene.datasets.forEach((ref, index) => {
        datasetList.append(this.renderDatasetItem(ref.id, ref.role, infos[index]));
      });
      this.credits = infos.map(info => info?.attribution ?? '').filter(Boolean);
      this.applyStepCartography();
    });
    const dataSection = scene.datasets.length ? collapsible('Data', datasetList, false) : null;

    // Under the hood.
    if (host) {
      const hood = createUnderTheHood({
        getHost: this.dependencies.getHost,
        onMeasure: () => void this.measureGpu(),
        hoodReadouts: hoodReadoutList.childElementCount ? hoodReadoutList : null,
        wrap: content => collapsible('Under the hood', content, false)
      });
      this.hood = hood;
    }

    const detailsPanel = h(
      'div',
      {class: 'tab-panel'},
      h('p', {class: 'lede'}, scene.summary),
      h(
        'div',
        {class: 'chips details-chips'},
        scene.contributors.map(name =>
          name.startsWith('Frontier')
            ? h('span', {class: 'chip chip-static', title: 'Scene-local prototype'}, name)
            : h(
                'a',
                {class: 'chip', href: getReferenceHash(name), title: `Reference: ${name}`},
                name
              )
        )
      ),
      codeSection,
      aboutSection,
      dataSection,
      this.hood?.element
    );

    // Tabs.
    const tabBar = h('div', {class: 'tab-bar', role: 'tablist', 'aria-label': 'Panel sections'});
    const panels = new Map<PanelTab, HTMLElement>([
      ['story', storyPanel],
      ['controls', controlsPanel],
      ['details', detailsPanel]
    ]);
    this.tabButtons.clear();
    this.tabPanels.clear();
    for (const tab of PANEL_TABS) {
      const panelElement = panels.get(tab.id)!;
      panelElement.setAttribute('role', 'tabpanel');
      panelElement.id = `panel-${tab.id}`;
      panelElement.setAttribute('aria-labelledby', `tab-${tab.id}`);
      const button = h(
        'button',
        {
          class: 'tab',
          type: 'button',
          role: 'tab',
          id: `tab-${tab.id}`,
          'aria-controls': panelElement.id
        },
        tab.label,
        tab.id === 'controls'
          ? [h('span', {class: 'count'}, String(optionsPanel.count)), this.controlsDot]
          : null
      );
      button.addEventListener('click', () => this.selectTab(tab.id));
      button.addEventListener('keydown', event => {
        const keys = ['ArrowRight', 'ArrowLeft', 'Home', 'End'];
        if (!keys.includes(event.key)) return;
        // Keep the story-step arrow shortcuts from firing while the tabs are navigated.
        event.preventDefault();
        event.stopPropagation();
        const current = PANEL_TABS.findIndex(entry => entry.id === this.activeTab);
        const next =
          event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? PANEL_TABS.length - 1
              : (current + (event.key === 'ArrowRight' ? 1 : -1) + PANEL_TABS.length) %
                PANEL_TABS.length;
        this.selectTab(PANEL_TABS[next].id);
        this.tabButtons.get(PANEL_TABS[next].id)?.focus();
      });
      this.tabButtons.set(tab.id, button);
      this.tabPanels.set(tab.id, panelElement);
      tabBar.append(button);
    }
    this.selectTab('story');

    // Footer: segmented progress, then Back / tour / shortcuts / Next.
    const progress = h('div', {
      class: 'step-progress',
      role: 'group',
      'aria-label': 'Story progress'
    });
    this.progressSegments = scene.story.map((step, index) => {
      const segment = h('button', {
        class: 'step-segment',
        type: 'button',
        title: step.title,
        'aria-label': `Step ${index + 1} of ${scene.story.length}: ${step.title}`
      });
      segment.addEventListener('click', () => this.goToStep(index));
      progress.append(segment);
      return segment;
    });
    const previousButton = h(
      'button',
      {class: 'btn', type: 'button', 'aria-label': 'Previous step'},
      icon('chevronLeft'),
      'Back'
    );
    const nextButton = h(
      'button',
      {class: 'btn btn-primary', type: 'button', 'aria-label': 'Next step'},
      'Next',
      icon('chevronRight')
    );
    previousButton.addEventListener('click', () => this.goToStep(this.stepIndex - 1));
    nextButton.addEventListener('click', () => this.goToStep(this.stepIndex + 1));
    this.autoplayButton = h(
      'button',
      {class: 'btn btn-ghost btn-small autoplay-button', type: 'button', 'aria-pressed': 'false'},
      'Auto-play tour'
    );
    this.autoplayButton.addEventListener('click', () => {
      if (this.autoplayTimer) this.stopAutoplay();
      else this.startAutoplay();
    });
    const shortcutsButton = h(
      'button',
      {
        class: 'btn btn-ghost btn-small shortcuts-button',
        type: 'button',
        'aria-label': 'Keyboard shortcuts',
        title: 'Keyboard shortcuts (?)'
      },
      '?'
    );
    shortcutsButton.addEventListener('click', () => openShortcutSheet(SHORTCUTS));
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const syncAutoplayAvailability = () => {
      this.autoplayButton.disabled = motionQuery.matches;
      this.autoplayButton.title = motionQuery.matches
        ? 'Auto-play is off because you asked for reduced motion'
        : 'Advance through the steps every few seconds';
      if (motionQuery.matches) this.stopAutoplay();
    };
    syncAutoplayAvailability();
    motionQuery.addEventListener('change', syncAutoplayAvailability);
    this.cleanups.push(() => motionQuery.removeEventListener('change', syncAutoplayAvailability));
    const footer = h(
      'footer',
      {class: 'panel-footer'},
      progress,
      h(
        'div',
        {class: 'panel-footer-row'},
        previousButton,
        h('div', {class: 'panel-footer-tools'}, this.autoplayButton, shortcutsButton),
        nextButton
      )
    );
    this.previousButtons.push(previousButton);
    this.nextButtons.push(nextButton);

    // The phone sheet: handle, then a peek row with the step title and Back / Next.
    const sheetHandle = h(
      'button',
      {
        class: 'sheet-handle',
        type: 'button',
        'aria-label': 'Expand or collapse the story panel',
        'aria-expanded': 'true'
      },
      h('span', {})
    );
    this.sheetTitle = h('span', {class: 'sheet-peek-title'});
    const peekPrevious = h(
      'button',
      {class: 'sheet-peek-button', type: 'button', 'aria-label': 'Previous step'},
      icon('chevronLeft', 18)
    );
    const peekNext = h(
      'button',
      {class: 'sheet-peek-button', type: 'button', 'aria-label': 'Next step'},
      icon('chevronRight', 18)
    );
    peekPrevious.addEventListener('click', () => this.goToStep(this.stepIndex - 1));
    peekNext.addEventListener('click', () => this.goToStep(this.stepIndex + 1));
    this.previousButtons.push(peekPrevious);
    this.nextButtons.push(peekNext);
    const peekRow = h('div', {class: 'sheet-peek'}, peekPrevious, this.sheetTitle, peekNext);

    const panel = h(
      'aside',
      {class: 'story-panel', id: 'story-panel', tabindex: '-1', 'aria-label': 'Story'},
      sheetHandle,
      peekRow,
      h(
        'div',
        {class: 'story-panel-scroll'},
        header,
        this.statusLine,
        tabBar,
        storyPanel,
        controlsPanel,
        detailsPanel
      ),
      footer,
      this.announcer
    );
    this.panel = panel;
    // Action links in step prose (and anywhere in the panel) are inert anchors; run their verb.
    panel.addEventListener('click', event => {
      const action = event.target instanceof HTMLElement ? parseActionLink(event.target) : null;
      if (!action) return;
      event.preventDefault();
      this.runAction(action.verb, action.params);
    });

    // Legends float over the map.
    this.legendContainer = h('div', {class: 'legends'});
    const legendBody = h('div', {class: 'map-legend-body'}, this.legendContainer);
    this.legendToggle = h(
      'button',
      {class: 'map-legend-toggle', type: 'button', 'aria-expanded': 'true'},
      icon('layers', 14),
      'Legend',
      icon('chevronDown', 14)
    );
    this.legendOverlay = h(
      'div',
      {class: 'map-legend', hidden: true},
      this.legendToggle,
      legendBody
    );
    const setLegendCollapsed = (collapsed: boolean) => {
      this.legendOverlay.classList.toggle('is-collapsed', collapsed);
      this.legendToggle.setAttribute('aria-expanded', String(!collapsed));
      this.scheduleObstacles();
    };
    setLegendCollapsed(window.matchMedia(MOBILE_QUERY).matches);
    this.legendToggle.addEventListener('click', () =>
      setLegendCollapsed(!this.legendOverlay.classList.contains('is-collapsed'))
    );

    const mapSlot = h('div', {
      class: 'story-map',
      id: 'story-map',
      tabindex: '-1',
      'aria-label': 'Map'
    });
    this.mapSlot = mapSlot;
    if (host) {
      mapSlot.append(host.element);
      host.setActive(true);
      host.element.setAttribute('role', 'img');
      this.setupMapSlot(mapSlot, scene, host);
    } else {
      mapSlot.append(
        h(
          'div',
          {class: 'unsupported'},
          h('h2', {}, 'WebGPU is not available in this browser'),
          h(
            'p',
            {},
            'This analysis runs on the GPU with WebGPU compute shaders. Open the showcase in a recent Chrome, Edge or Safari (with WebGPU enabled) to see the map. The narrative, the reference docs and the data catalog still work here.'
          ),
          h('a', {class: 'btn', href: '#/reference'}, 'Browse the reference')
        )
      );
    }
    if (this.mapCorners) {
      const mobileQuery = window.matchMedia(MOBILE_QUERY);
      this.placeLegend(mobileQuery.matches);
      const sheet = new SheetController({
        panel,
        handle: sheetHandle,
        mapSlot,
        onSettle: () => {
          this.updateViewPadding(true);
        }
      });
      this.sheet = sheet;
      sheet.setEnabled(mobileQuery.matches);
      this.updateViewPadding(false);
      const onMobileChange = (event: MediaQueryListEvent) => {
        this.placeLegend(event.matches);
        sheet.setEnabled(event.matches);
      };
      const onResize = () => sheet.refresh();
      mobileQuery.addEventListener('change', onMobileChange);
      window.addEventListener('resize', onResize);
      this.cleanups.push(() => {
        mobileQuery.removeEventListener('change', onMobileChange);
        window.removeEventListener('resize', onResize);
      });
    } else {
      mapSlot.append(this.legendOverlay);
    }

    // Skip links: the router uses the hash, so they move focus instead of changing it.
    const skipTo = (label: string, target: HTMLElement) => {
      const link = h('a', {class: 'skip-link', href: '#'}, label);
      link.addEventListener('click', event => {
        event.preventDefault();
        target.focus();
      });
      return link;
    };
    clearElement(root);
    root.append(
      skipTo('Skip to map', mapSlot),
      skipTo('Skip to story', panel),
      h('div', {class: 'story-layout'}, panel, mapSlot)
    );
    this.subscribeStore(store);
    if (host) {
      host.start();
      this.startStatsTimer();
    }
  }

  /** Builds the map overlays: corner stacks, tooltip, chips, time bar and compare divider. */
  private setupMapSlot(mapSlot: HTMLElement, scene: AnyScene, host: DeckHost): void {
    // Furniture, chips, insets and the legend share four corner stacks over the map.
    const furniture = new MapFurniture();
    this.furniture = furniture;
    const chips = new MapChips();
    this.chips = chips;
    const corners: MapCorners = {
      topLeft: h('div', {class: 'map-corner map-corner-top-left'}, furniture.cartouche),
      topRight: h(
        'div',
        {class: 'map-corner map-corner-top-right'},
        furniture.northArrow,
        furniture.clock,
        chips.element
      ),
      bottomLeft: h('div', {class: 'map-corner map-corner-bottom-left'}),
      bottomRight: h('div', {class: 'map-corner map-corner-bottom-right'}, furniture.scaleBar)
    };
    this.mapCorners = corners;
    const creditSlot = h('div', {class: 'map-credit-slot'}, furniture.credit);
    this.creditSlot = creditSlot;
    furniture.onNorthClick = () => host.setViewState({bearing: 0, transitionMs: 400});

    const divider = new CompareDivider(state => this.handleCompareChange(state));
    this.compareDivider = divider;
    const tooltip = new MapTooltip();
    this.tooltip = tooltip;

    mapSlot.append(
      divider.element,
      corners.topLeft,
      corners.topRight,
      corners.bottomLeft,
      corners.bottomRight,
      creditSlot
    );
    if (scene.timeline) {
      const timeBar = new TimeBar(this.store!, scene.timeline);
      this.timeBar = timeBar;
      if (this.timeBarData) timeBar.setData(this.timeBarData);
      mapSlot.classList.add('has-time-bar');
      mapSlot.append(timeBar.element);
    }
    mapSlot.append(tooltip.element);

    this.setupTooltipEvents(host);

    // Obstacles are reported when a stack resizes (legend, chips, insets), never per frame.
    const observer = new ResizeObserver(() => this.scheduleObstacles());
    this.obstacleObserver = observer;
    for (const element of [
      corners.topLeft,
      corners.topRight,
      corners.bottomLeft,
      corners.bottomRight,
      creditSlot,
      mapSlot,
      ...(this.timeBar ? [this.timeBar.element] : [])
    ]) {
      observer.observe(element);
    }

    this.cleanups.push(
      host.addViewListener(view => {
        furniture.updateView({
          latitude: view.latitude,
          zoom: view.zoom,
          bearing: view.bearing ?? 0,
          pitch: view.pitch ?? 0
        });
        this.checkCameraMoved(view);
      })
    );
  }

  /** Hover tooltip, pinning and the hover highlight. */
  private setupTooltipEvents(host: DeckHost): void {
    const tooltip = this.tooltip!;
    host.onTooltip = (content, position) => {
      const isTouch = position.pointerType === 'touch';
      if (!content) {
        this.tooltipHasContent = false;
        tooltip.hide();
        if (!tooltip.pinned) host.setHighlightGroup('hover', null);
        return;
      }
      // A pinned card holds still for the mouse; a touch replaces it.
      if (tooltip.pinned && !isTouch) return;
      this.tooltipHasContent = true;
      const data = typeof content === 'string' ? {title: content} : content;
      const anchor = data.anchor ? this.projectToSlot(data.anchor) : undefined;
      tooltip.show(
        data,
        {x: position.x, y: position.y},
        {anchor, pointerType: position.pointerType}
      );
      host.setHighlightGroup('hover', data.highlight ? [data.highlight] : null);
    };
    tooltip.onClose = () => this.releaseTooltip();
    const trackShift = (event: Event) => {
      this.shiftHeld = (event as KeyboardEvent | PointerEvent).shiftKey === true;
    };
    for (const type of ['keydown', 'keyup', 'pointerdown', 'pointermove'] as const) {
      window.addEventListener(type, trackShift, {capture: true, passive: true});
    }
    this.cleanups.push(() => {
      for (const type of ['keydown', 'keyup', 'pointerdown', 'pointermove'] as const) {
        window.removeEventListener(type, trackShift, {capture: true});
      }
    });
    host.onMapClick = position => {
      if (position.pointerType === 'touch') {
        // The tap's own tooltip already pinned itself; a tap on nothing releases the card.
        if (tooltip.pinned && !this.tooltipHasContent) this.releaseTooltip();
        return;
      }
      if (tooltip.pinned) this.releaseTooltip();
      else if (this.tooltipHasContent && this.shiftHeld) tooltip.pin();
    };
    this.cleanups.push(() => {
      host.setHighlightGroup('hover', null);
      host.setHighlightGroup('action', null);
    });
  }

  private releaseTooltip(): void {
    this.tooltip?.unpin();
    this.tooltip?.hide();
    this.tooltipHasContent = false;
    if (this.dependencies.hasWebGPU) this.dependencies.getHost().setHighlightGroup('hover', null);
  }

  /** Screen position of a coordinate in map slot pixels, or `undefined` when it cannot be projected. */
  private projectToSlot(coordinate: readonly [number, number]): {x: number; y: number} | undefined {
    const viewport = this.dependencies.getHost().viewport;
    if (!viewport) return undefined;
    const [x, y] = viewport.project([coordinate[0], coordinate[1]]);
    return Number.isFinite(x) && Number.isFinite(y) ? {x, y} : undefined;
  }

  /** One store listener for everything that follows options regardless of the scene instance. */
  private subscribeStore(store: OptionsStore): void {
    const unsubscribe = store.subscribe((id, value, source: OptionChangeSource) => {
      if (source === 'user' && !this.applyingStepState && !this.isTransientOption(id)) {
        this.touchedOptions.add(id);
        this.updateModifiedChip();
      }
      const clock = this.getCumulativeFurniture(this.stepIndex)?.clock;
      if (clock && clock.option === id) this.syncClock();
      if (typeof value === 'number') this.updateLinkedCharts(id, value);
    });
    this.cleanups.push(unsubscribe);
  }

  private updateLinkedCharts(option: string, value: number): void {
    const views: {linkOption: string | null; chartElement: HTMLElement | null}[] = [
      ...this.allReadoutViews.values(),
      ...this.stepReadoutViews.values(),
      ...this.insetViews.values()
    ];
    for (const view of views) {
      if (view.linkOption === option && view.chartElement) {
        updateChartLink(view.chartElement, value);
      }
    }
  }

  private renderDatasetItem(
    id: string,
    role: string | undefined,
    info: DatasetInfo | undefined
  ): HTMLElement {
    if (!info)
      return h(
        'li',
        {},
        h('strong', {}, id),
        h('div', {class: 'muted small'}, 'Not in the data catalog yet.')
      );
    return h(
      'li',
      {},
      h('strong', {}, info.title),
      role ? h('span', {class: 'muted'}, ` — ${role}`) : null,
      h('div', {class: 'small'}, info.attribution),
      h(
        'div',
        {class: 'muted small'},
        `${info.license} · ${formatBytes(info.approxBytes)} · `,
        h(
          'a',
          {href: info.sourceUrl, target: '_blank', rel: 'noreferrer noopener'},
          'source',
          icon('external', 11)
        )
      )
    );
  }

  // ---------------------------------------------------------------------------------------------
  // Engine statistics and cost chips
  // ---------------------------------------------------------------------------------------------

  private startStatsTimer(): void {
    const timer = window.setInterval(() => {
      if (!this.dependencies.hasWebGPU) return;
      this.hood?.update();
      this.refreshCost();
    }, STATS_INTERVAL_MILLISECONDS);
    this.cleanups.push(() => clearInterval(timer));
  }

  private refreshCost(): void {
    if (!this.costRow || !this.dependencies.hasWebGPU) return;
    const {rebuildCount} = this.dependencies.getHost().pollRebuilds();
    // A compile-time rebuild invalidates the cached GPU time.
    if (this.lastRebuildCount !== null && rebuildCount !== this.lastRebuildCount) {
      this.gpuMilliseconds = null;
    }
    this.lastRebuildCount = rebuildCount;
    this.costRow.update(this.cost, {
      rebuilds: rebuildCount,
      gpuMilliseconds: this.gpuMilliseconds,
      measuring: this.measuring
    });
  }

  /** Measures every compiled graph once; the number is cached in the cost chip and the hood. */
  private async measureGpu(): Promise<void> {
    if (this.measuring || !this.dependencies.hasWebGPU) return;
    const scene = this.scene;
    this.measuring = true;
    this.hood?.setMeasuring(true);
    this.refreshCost();
    const result = await measureActiveScene(this.dependencies.getHost());
    if (this.scene !== scene) return;
    this.measuring = false;
    if (result) this.gpuMilliseconds = result.milliseconds;
    this.hood?.showMeasurement(result);
    this.hood?.setMeasuring(false);
    this.refreshCost();
  }

  // ---------------------------------------------------------------------------------------------
  // Scene activation
  // ---------------------------------------------------------------------------------------------

  private async activate(scene: AnyScene, generation: number): Promise<void> {
    const host = this.dependencies.getHost();
    const store = this.store!;
    const {catalog} = this.dependencies;
    host.onStatus = message => {
      if (generation === this.mountGeneration) this.statusLine.textContent = message;
    };
    const snapshotAtStart = store.snapshot();
    const stepAtStart = this.stepIndex;
    const createScene = scene.create as (
      ctx: SceneContext<OptionState>
    ) => Promise<SceneInstance<OptionState>>;
    const activation = host.activate<ShellInstance>({
      id: scene.id,
      view: this.getCumulativeCamera(this.stepIndex),
      create: async (services: HostServices) => {
        const loaded = new Map<string, LoadedDataset>();
        await Promise.all(
          scene.datasets.map(async ref => {
            const dataset = await catalog.load(ref.id, services.signal);
            loaded.set(ref.id, dataset);
          })
        );
        services.signal.throwIfAborted();
        if (generation === this.mountGeneration) {
          const synthetic = [...loaded.values()].some(dataset => dataset.origin === 'synthetic');
          if (synthetic) {
            this.statusLine.textContent =
              'Using deterministic synthetic data (offline or ?data=synthetic).';
            this.chips?.setStatus('synthetic', 'Synthetic data', 'warn');
          }
        }
        const guard = () => generation === this.mountGeneration && !services.signal.aborted;
        return createScene({
          device: services.device,
          options: store.view,
          datasets: {
            get: id => {
              const dataset = loaded.get(id);
              if (!dataset) throw new Error(`Scene "${scene.id}" did not declare dataset "${id}"`);
              return dataset;
            }
          },
          signal: services.signal,
          forceSynthetic: catalog.forceSynthetic,
          theme: getTheme,
          ground: () => host.ground,
          getViewport: services.getViewport,
          requestLayers: services.updateLayers,
          setReadout: (id, value) => {
            if (generation === this.mountGeneration) this.setReadout(id, value);
          },
          setLegendExtent: (id, extent) => {
            if (generation !== this.mountGeneration) return;
            this.gpuExtents.set(id, extent);
            this.refreshLegends();
          },
          refreshLegends: () => {
            if (generation === this.mountGeneration) this.refreshLegends();
          },
          setLegendData: (key, value) => {
            if (generation !== this.mountGeneration) return;
            this.legendData[key] = value;
            this.refreshLegends();
          },
          setOptions: (values, options) => {
            if (generation !== this.mountGeneration) return;
            this.notifySceneWrites = Boolean(options?.notify);
            try {
              store.setMany(values as Partial<OptionState>, 'scene');
            } finally {
              this.notifySceneWrites = false;
            }
          },
          flyTo: services.flyTo,
          getViewState: services.getViewState,
          setChart: (id, chart) => {
            if (generation === this.mountGeneration) this.setChart(id, chart);
          },
          refreshTooltip: services.refreshTooltip,
          limits: services.limits,
          setStatus: message => {
            if (generation === this.mountGeneration) this.statusLine.textContent = message;
          },
          setMapDragEnabled: services.setMapDragEnabled,
          setAnnotations: (key, annotations) => {
            if (!guard()) return;
            host.setAnnotationGroup(`data:${key}`, annotations);
          },
          setHighlight: highlight => {
            if (!guard()) return;
            host.setHighlightGroup(
              'scene',
              highlight === null ? null : Array.isArray(highlight) ? highlight : [highlight]
            );
          },
          setAnnotationTime: time => {
            if (guard()) host.setAnnotationTime(time);
          },
          fitBounds: services.fitBounds,
          setCost: cost => {
            if (generation !== this.mountGeneration) return;
            this.cost = cost;
            this.refreshCost();
          },
          setTimelineData: data => {
            if (generation !== this.mountGeneration) return;
            this.timeBarData = data;
            this.timeBar?.setData(data);
          },
          setFurniture: overrides => {
            if (generation !== this.mountGeneration) return;
            this.runtimeFurniture = overrides;
            this.applyStepCartography();
          },
          setAnnotationHalo: weight => {
            if (guard()) host.setAnnotationHalo(weight);
          },
          reducedMotion: prefersReducedMotion,
          getMetersPerPixel: services.getMetersPerPixel,
          getCompare: () => this.compareState
        }) as Promise<ShellInstance>;
      }
    });
    const instance = await activation;
    if (!instance || generation !== this.mountGeneration) {
      if (generation === this.mountGeneration && host.state.error) {
        this.statusLine.textContent = `This analysis could not start: ${host.state.error}`;
      }
      return;
    }
    this.instance = instance;
    if (this.statusLine.textContent && !this.statusLine.textContent.startsWith('Using')) {
      this.statusLine.textContent = '';
    }
    // Replay options that changed while the scene was being created.
    const current = store.snapshot();
    for (const [id, value] of Object.entries(current)) {
      if (JSON.stringify(snapshotAtStart[id]) !== JSON.stringify(value)) {
        instance.setOption?.(id, value, current);
      }
    }
    const unsubscribe = store.subscribe((id, value, source: OptionChangeSource) => {
      if (source === 'user' || this.notifySceneWrites) {
        this.instance?.setOption?.(id, value, store.snapshot());
      }
      if (source === 'user') {
        this.refreshLegends();
        this.refreshSnippet();
        this.scheduleUrlSync();
      } else {
        this.scheduleDerivedRefresh();
      }
    });
    this.cleanups.push(unsubscribe);
    this.cleanups.push(
      onThemeChange(theme => {
        host.setTheme(theme);
        this.instance?.onThemeChange?.(theme);
        this.updateGround();
      })
    );
    host.setTheme(getTheme());
    this.updateGround();
    this.applyStepSideEffects(false);
    // The activation view carries no bounds or padding, and the reader may have stepped while the
    // scene loaded: frame the current step now, without a flight.
    const effective = this.getEffectiveCamera(this.stepIndex);
    if (effective.bounds || effective.padding !== undefined || this.stepIndex !== stepAtStart) {
      this.moveToStepCamera(false);
    } else {
      // The camera the story opened on is the reference for the "Modified" check.
      this.stepViewReference = host.getViewState();
    }
    // Tell the scene the state of a compare step that was already open when it appeared.
    if (this.compareDivider?.state) this.handleCompareChange(this.compareDivider.state);
  }

  private goToStep(index: number): void {
    const scene = this.scene;
    if (!scene || index < 0 || index >= scene.story.length) return;
    navigate(getStoryHash(scene.id, scene.story[index].id));
  }

  // ---------------------------------------------------------------------------------------------
  // Readouts
  // ---------------------------------------------------------------------------------------------

  private createReadoutView(spec: ReadoutSpec, tile: boolean): ReadoutView {
    const isChart = spec.kind === 'chart';
    const value = h('dd', {}, isChart ? null : '—');
    const chart = isChart ? h('div', {class: 'readout-chart'}) : null;
    const row = h(
      'div',
      {
        class: `readout${isChart ? ' readout-chart-row' : ''}${tile && !isChart ? ' is-tile' : ''}${spec.layout === 'block' ? ' is-block' : ''}`,
        title: spec.help ?? ''
      },
      h('dt', {}, spec.label),
      isChart ? chart : value
    );
    const view: ReadoutView = {value, row, chart, chartElement: null, linkOption: null};
    this.paintReadout(spec, view, tile);
    return view;
  }

  /** Renders a chart with its linked option marker wired both ways. */
  private renderLinkedChart(chart: ChartData): {element: HTMLElement; linkOption: string | null} {
    const store = this.store!;
    const link = getChartLink(chart);
    if (!link) return {element: renderChart(chart), linkOption: null};
    const current = Number(store.get(link.option));
    const element = renderChart(chart, {
      linkValue: Number.isFinite(current) ? current : undefined,
      onLinkInput: value => this.writeLinkedOption(link.option, value)
    });
    return {element, linkOption: link.option};
  }

  /** A chart pointer wrote a value: snap it to the option's step, clamp it, write it. */
  private writeLinkedOption(option: string, value: number): void {
    const store = this.store;
    const spec = store?.specs.find(entry => entry.id === option);
    if (!store || !spec) return;
    store.set(option, snapToSlider(spec, value));
  }

  /** Copies the latest value or chart of a readout into one rendered copy. */
  private paintReadout(spec: ReadoutSpec, view: ReadoutView, tile = false): void {
    if (view.chart) {
      const chart = this.readoutCharts.get(spec.id);
      if (!chart) {
        view.chartElement = null;
        view.linkOption = null;
        view.chart.replaceChildren();
        return;
      }
      const rendered = this.renderLinkedChart(chart);
      view.chartElement = rendered.element;
      view.linkOption = rendered.linkOption;
      view.chart.replaceChildren(rendered.element);
      return;
    }
    const text = formatReadout(spec, this.readoutValues.get(spec.id) ?? null);
    if (view.value.textContent !== text) view.value.textContent = text;
    const block = !tile && (spec.layout ? spec.layout === 'block' : text.includes('\n'));
    view.row.classList.toggle('is-block', block);
  }

  private getReadoutViews(id: string): ReadoutView[] {
    return [this.allReadoutViews.get(id), this.stepReadoutViews.get(id)].filter(
      (view): view is ReadoutView => Boolean(view)
    );
  }

  private setReadout(id: string, value: string | number | null): void {
    const spec = this.readoutSpecs.get(id);
    if (!spec) return;
    this.readoutValues.set(id, value);
    for (const view of this.getReadoutViews(id)) {
      this.paintReadout(spec, view, view.row.classList.contains('is-tile'));
    }
    // Live `{{id}}` slots in the step text change in place; the card is not re-rendered.
    const spans = this.liveReadoutSpans.get(id);
    if (spans) {
      const text = formatReadout(spec, value);
      for (const span of spans) if (span.textContent !== text) span.textContent = text;
    }
  }

  private setChart(id: string, chart: ChartData | null): void {
    const spec = this.readoutSpecs.get(id);
    if (!spec) return;
    this.readoutCharts.set(id, chart);
    if (isMapChart(spec)) {
      this.paintInset(id);
      return;
    }
    for (const view of this.getReadoutViews(id)) this.paintReadout(spec, view);
  }

  private selectTab(tab: PanelTab): void {
    this.activeTab = tab;
    for (const [id, button] of this.tabButtons) {
      const selected = id === tab;
      button.classList.toggle('is-active', selected);
      button.setAttribute('aria-selected', String(selected));
      button.tabIndex = selected ? 0 : -1;
      this.tabPanels.get(id)!.hidden = !selected;
    }
  }

  /** Warns once per scene about step control, readout or stage ids that do not exist. */
  private warnAboutStepReferences(scene: AnyScene): void {
    const optionIds = new Set(scene.options.map(option => option.id));
    const readoutIds = new Set((scene.readouts ?? []).map(readout => readout.id));
    const stageIds = new Set((scene.pipeline ?? []).map(stage => stage.id));
    const problems: string[] = [];
    for (const step of scene.story) {
      for (const id of getStepControlIds(step)) {
        if (!optionIds.has(id)) problems.push(`${step.id}: control "${id}"`);
      }
      for (const id of getStepReadoutIds(step)) {
        if (!readoutIds.has(id)) problems.push(`${step.id}: readout "${id}"`);
      }
      for (const narrative of [step.body, step.evidence, step.caveat]) {
        if (!narrative) continue;
        for (const id of getInterpolatedReadoutIds(narrative)) {
          if (!readoutIds.has(id)) problems.push(`${step.id}: {{${id}}}`);
        }
      }
      if (step.stage && !stageIds.has(step.stage))
        problems.push(`${step.id}: stage "${step.stage}"`);
    }
    if (problems.length) {
      // biome-ignore lint/suspicious/noConsole: authoring aid for scene authors
      console.warn(
        `Scene "${scene.id}" has story steps that reference unknown ids (skipped): ${problems.join(', ')}`
      );
    }
  }

  // ---------------------------------------------------------------------------------------------
  // The step card
  // ---------------------------------------------------------------------------------------------

  /**
   * Fills the active step's card: headline, cost chips, pipeline strip, body, evidence and caveat
   * with live readouts, diagram, "Adjust" controls and readouts (key figures as tiles).
   */
  private renderStepDetail(index: number): void {
    const scene = this.scene!;
    const store = this.store!;
    const step = scene.story[index];
    const detail = this.stepRows[index].detail;
    this.stepControls?.destroy();
    this.stepControls = null;
    this.stepReadoutViews = new Map();
    this.liveReadoutSpans = new Map();
    this.costRow = null;
    const parts: (HTMLElement | null)[] = [];
    const formatStepReadout = (readoutId: string): string => {
      const spec = this.readoutSpecs.get(readoutId);
      return spec ? formatReadout(spec, this.readoutValues.get(readoutId) ?? null) : '—';
    };

    if (step.headline) parts.push(h('p', {class: 'step-headline'}, step.headline));

    if (this.dependencies.hasWebGPU) {
      const costRow = new CostRow(() => void this.measureGpu());
      this.costRow = costRow;
      parts.push(costRow.element);
      this.lastRebuildCount = null;
      this.refreshCost();
    }

    if (scene.pipeline?.length) {
      parts.push(
        createPipelineStrip(scene.pipeline, step.stage, (option, value) => store.set(option, value))
      );
    }

    const body = h('div', {
      class: 'prose step-body',
      html: renderStepBody(step.body, formatStepReadout)
    });
    parts.push(body);

    const narrativeNotes: {kind: 'evidence' | 'caveat'; label: string; text: string}[] = [];
    if (step.evidence) {
      narrativeNotes.push({kind: 'evidence', label: 'Evidence', text: step.evidence});
    }
    if (step.caveat) {
      narrativeNotes.push({kind: 'caveat', label: 'Caveat', text: step.caveat});
    }
    if (narrativeNotes.length) {
      const noteList = h('dl', {class: 'step-note-list'});
      for (const note of narrativeNotes) {
        noteList.append(
          h(
            'div',
            {class: `step-note step-note-${note.kind}`},
            h('dt', {}, note.label),
            h('dd', {
              class: 'prose',
              html: renderStepBody(note.text, formatStepReadout)
            })
          )
        );
      }
      parts.push(h('aside', {class: 'step-notes', 'aria-label': 'Interpretation notes'}, noteList));
    }

    if (step.diagram) {
      parts.push(h('figure', {class: 'step-diagram'}, renderChart(step.diagram)));
    }

    const specs = getStepControlIds(step)
      .map(id => store.specs.find(spec => spec.id === id))
      .filter((spec): spec is OptionSpec<never> => Boolean(spec));
    if (specs.length) {
      const controls = new StepControls(store, specs, id =>
        this.instance?.onAction?.(id, store.snapshot())
      );
      this.stepControls = controls;
      parts.push(
        h(
          'section',
          {class: 'step-adjust', 'aria-label': 'Adjust in this step'},
          h('h4', {class: 'step-adjust-title'}, 'Adjust'),
          controls.element
        )
      );
    }

    // Readouts: key figures as tiles, the rest as rows. Engine detail stays in "Under the hood"
    // unless the step lists it by name, and map insets draw their own chart.
    const listed = step.readouts ? new Set(step.readouts) : new Set<string>();
    const readoutSpecs = getStepReadoutIds(step)
      .map(id => this.readoutSpecs.get(id))
      .filter(
        (spec): spec is ReadoutSpec =>
          Boolean(spec) && !isMapChart(spec) && (!spec!.hood || listed.has(spec!.id))
      );
    const tiles = readoutSpecs.filter(spec => spec.emphasis === 'tile' && spec.kind !== 'chart');
    const rows = readoutSpecs.filter(spec => !tiles.includes(spec));
    for (const [specsOfKind, className, isTile] of [
      [tiles, 'readouts readout-tiles', true],
      [rows, 'readouts step-readouts', false]
    ] as const) {
      if (!specsOfKind.length) continue;
      const list = h('dl', {class: className});
      for (const spec of specsOfKind) {
        const view = this.createReadoutView(spec, isTile);
        view.row.classList.toggle('is-highlighted', step.highlight?.readout === spec.id);
        this.stepReadoutViews.set(spec.id, view);
        list.append(view.row);
      }
      parts.push(list);
    }
    detail.replaceChildren(...parts.filter((part): part is HTMLElement => Boolean(part)));
    for (const span of detail.querySelectorAll<HTMLElement>('.live-readout')) {
      const id = span.dataset['readout'] ?? '';
      const spans = this.liveReadoutSpans.get(id) ?? [];
      spans.push(span);
      this.liveReadoutSpans.set(id, spans);
    }
    this.optionsPanel?.setStepControls(specs.map(spec => spec.id));
  }

  /** Scene-driven option writes can come every frame; refresh legends, code and URL at most once per frame / 300 ms. */
  private scheduleDerivedRefresh(): void {
    if (!this.derivedFrame) {
      this.derivedFrame = window.setTimeout(() => {
        this.derivedFrame = 0;
        this.refreshLegends();
        this.refreshSnippet();
      }, DERIVED_REFRESH_MILLISECONDS);
    }
    this.scheduleUrlSync();
  }

  /** Writes the URL at most once per 300 ms (options can change every frame during playback). */
  private scheduleUrlSync(): void {
    clearTimeout(this.urlTimer);
    this.urlTimer = window.setTimeout(() => this.syncUrl(), 300);
  }

  private refreshStep(animate: boolean): void {
    const scene = this.scene!;
    const step: StoryStep = scene.story[this.stepIndex];
    const last = scene.story.length - 1;
    for (const button of this.previousButtons) button.disabled = this.stepIndex === 0;
    for (const button of this.nextButtons) button.disabled = this.stepIndex === last;
    this.sheetTitle.textContent = step.title;
    this.progressSegments.forEach((segment, index) => {
      segment.classList.toggle('is-done', index < this.stepIndex);
      segment.classList.toggle('is-active', index === this.stepIndex);
      if (index === this.stepIndex) segment.setAttribute('aria-current', 'step');
      else segment.removeAttribute('aria-current');
    });
    this.stepRows.forEach((entry, index) => {
      const active = index === this.stepIndex;
      const done = index < this.stepIndex;
      entry.item.classList.toggle('is-active', active);
      entry.item.classList.toggle('is-done', done);
      entry.row.setAttribute('aria-expanded', String(active));
      if (active) entry.row.setAttribute('aria-current', 'step');
      else entry.row.removeAttribute('aria-current');
      entry.detail.hidden = !active;
      if (!active) entry.detail.replaceChildren();
      entry.badge.replaceChildren(done ? icon('check', 13) : String(index + 1));
    });
    this.renderStepDetail(this.stepIndex);
    for (const [id, view] of this.allReadoutViews) {
      view.row.classList.toggle('is-highlighted', step.highlight?.readout === id);
    }
    // Title and finding are announced together, once per step change.
    this.announcer.textContent = animate
      ? [step.title, step.headline].filter(Boolean).join('. ')
      : '';
    if (animate) {
      this.stepRows[this.stepIndex].item.scrollIntoView({
        behavior: prefersReducedMotion() ? 'auto' : 'smooth',
        block: 'nearest'
      });
    }
    this.refreshLegends();
    this.refreshSnippet();
    this.syncUrl();
    this.applyStepCartography();
    this.applyStepCompare();
    this.syncInsets();
    this.updateMapLabel();
    if (this.instance || this.dependencies.hasWebGPU) this.applyStepSideEffects(animate);
  }

  private applyStepSideEffects(animate: boolean): void {
    const scene = this.scene;
    if (!scene || !this.dependencies.hasWebGPU || !this.instance) return;
    const host = this.dependencies.getHost();
    const step = scene.story[this.stepIndex];
    if (animate) {
      this.releaseTooltip();
      host.setHighlightGroup('action', null);
    }
    // Stepping back to a step with no camera of its own returns to the view it inherits.
    if (animate && (step.camera || this.navigatingBack)) this.moveToStepCamera(true);
    host.setCallout(step.callout ?? null);
    const annotations = [...(scene.annotations ?? []), ...(step.annotations ?? [])];
    host.setAnnotationGroup('story', annotations.length ? annotations : null);
    this.updateModifiedChip();
    this.scheduleObstacles();
  }

  /** The map's text alternative: the step's, else its title and the first legend title. */
  private updateMapLabel(): void {
    const scene = this.scene;
    if (!scene || !this.dependencies.hasWebGPU) return;
    const step = scene.story[this.stepIndex];
    const firstLegend = this.legendSpecs.find(spec => 'title' in spec && spec.title);
    const legendTitle = firstLegend && 'title' in firstLegend ? firstLegend.title : '';
    const label = step.textAlternative ?? [step.title, legendTitle].filter(Boolean).join('. ');
    this.dependencies.getHost().element.setAttribute('aria-label', label);
  }

  private refreshLegends(): void {
    if (!this.scene || !this.store) return;
    this.legendSpecs = this.scene.legends(this.store.snapshot() as never, this.legendData);
    let key = '';
    try {
      key = JSON.stringify([this.legendSpecs, [...this.gpuExtents]]);
    } catch {
      key = '';
    }
    // Identical legends keep their DOM, so playback does not reset interactive selections.
    if (key && key === this.lastLegendKey) return;
    this.lastLegendKey = key;
    renderLegends(this.legendContainer, this.legendSpecs, this.gpuExtents, {
      onFilter: (id, classes) => this.instance?.onLegendFilter?.(id, classes)
    });
    this.legendOverlay.hidden = this.legendContainer.childElementCount === 0;
    // The collapsed phone pill carries a thumbnail of the first legend.
    this.legendToggle.querySelector('.legend-thumb')?.remove();
    const thumbnail = this.legendSpecs[0] ? renderLegendThumbnail(this.legendSpecs[0]) : null;
    if (thumbnail) this.legendToggle.append(thumbnail);
    this.scheduleObstacles();
  }

  private refreshSnippet(): void {
    const scene = this.scene;
    if (!scene?.snippet || !this.store) return;
    const text =
      typeof scene.snippet === 'string'
        ? scene.snippet
        : scene.snippet(this.store.snapshot() as never);
    if (text === this.snippetText && this.snippetCode.firstChild) return;
    this.snippetText = text;
    this.snippetCode.innerHTML = highlightCode(this.snippetText);
  }

  private syncUrl(): void {
    const scene = this.scene;
    const store = this.store;
    if (!scene || !store) return;
    replaceHash(getStoryHash(scene.id, scene.story[this.stepIndex].id, store.serialize()));
  }

  // ---------------------------------------------------------------------------------------------
  // Action links, keyboard, auto-play
  // ---------------------------------------------------------------------------------------------

  /** Runs the verb of an `action:` link in step prose. */
  private runAction(verb: string, params: Record<string, string>): void {
    const scene = this.scene;
    const store = this.store;
    if (!scene || !store) return;
    const host = this.dependencies.hasWebGPU ? this.dependencies.getHost() : null;
    const number = (key: string) => {
      const value = Number(params[key]);
      return params[key] !== undefined && Number.isFinite(value) ? value : undefined;
    };
    switch (verb) {
      case 'fly': {
        const longitude = number('lng');
        const latitude = number('lat');
        if (!host || longitude === undefined || latitude === undefined) return;
        host.setViewState({
          longitude,
          latitude,
          zoom: number('z') ?? number('zoom'),
          pitch: number('pitch'),
          bearing: number('bearing'),
          transitionMs: number('ms') ?? DEFAULT_FLY_MILLISECONDS
        });
        return;
      }
      case 'set':
        for (const [id, text] of Object.entries(params)) {
          const spec = store.specs.find(entry => entry.id === id);
          const value = spec ? parseOptionText(spec, text) : undefined;
          if (value !== undefined) store.set(id, value);
        }
        return;
      case 'reset':
        this.applyStepState(this.stepIndex, false);
        return;
      case 'camera':
        this.moveToStepCamera(true);
        return;
      case 'step': {
        const target = scene.story.findIndex(step => step.id === params['id']);
        if (target >= 0) this.goToStep(target);
        return;
      }
      case 'highlight': {
        const lng = number('lng');
        const lat = number('lat');
        if (!host || lng === undefined || lat === undefined) return;
        host.setHighlightGroup('action', [{kind: 'point', coordinate: [lng, lat], pulse: true}]);
        window.clearTimeout(this.pingTimer);
        this.pingTimer = window.setTimeout(
          () => host.setHighlightGroup('action', null),
          PING_MILLISECONDS
        );
      }
    }
  }

  private startKeyboard(): void {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
      if (document.querySelector('dialog[open]')) return;
      const target = event.target instanceof HTMLElement ? event.target : null;
      if (target && isEditable(target)) return;
      switch (event.key) {
        case '[':
          this.goToStep(this.stepIndex - 1);
          break;
        case ']':
          this.goToStep(this.stepIndex + 1);
          break;
        case 'r':
        case 'R':
          this.applyStepState(this.stepIndex, false);
          break;
        case '?':
          event.preventDefault();
          openShortcutSheet(SHORTCUTS);
          break;
        case 'ArrowRight':
        case 'ArrowLeft':
          // Arrows belong to the page elsewhere (sliders, tabs, the time bar, the map).
          if (!target || !this.panel?.contains(target) || ownsArrowKeys(target)) break;
          this.goToStep(this.stepIndex + (event.key === 'ArrowRight' ? 1 : -1));
          break;
      }
    };
    // Capture phase: a pinned card must be released before the camera reacts to Escape.
    const onEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented) return;
      if (document.querySelector('dialog[open]')) return;
      if (this.tooltip?.pinned) {
        this.releaseTooltip();
        return;
      }
      const target = event.target instanceof HTMLElement ? event.target : null;
      if (target && isEditable(target)) return;
      this.moveToStepCamera(true);
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keydown', onEscape, {capture: true});
    this.cleanups.push(() => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keydown', onEscape, {capture: true});
    });
  }

  private startAutoplay(): void {
    const scene = this.scene;
    if (!scene || prefersReducedMotion() || this.autoplayTimer) return;
    this.autoplayButton.setAttribute('aria-pressed', 'true');
    this.autoplayButton.textContent = 'Pause tour';
    this.autoplayTimer = window.setInterval(() => {
      if (this.stepIndex >= scene.story.length - 1) this.stopAutoplay();
      else this.goToStep(this.stepIndex + 1);
    }, AUTOPLAY_MILLISECONDS);
    // Any interaction with the page (outside the tour button) hands control back to the reader.
    const pause = (event: Event) => {
      if (event.target instanceof Node && this.autoplayButton.contains(event.target)) return;
      this.stopAutoplay();
    };
    const root = this.root;
    const events = ['pointerdown', 'keydown', 'wheel'] as const;
    for (const type of events) root?.addEventListener(type, pause, {capture: true, passive: true});
    this.autoplayCleanup = () => {
      for (const type of events) root?.removeEventListener(type, pause, {capture: true});
    };
  }

  private stopAutoplay(): void {
    window.clearInterval(this.autoplayTimer);
    this.autoplayTimer = 0;
    this.autoplayCleanup?.();
    this.autoplayCleanup = null;
    if (this.autoplayButton) {
      this.autoplayButton.setAttribute('aria-pressed', 'false');
      this.autoplayButton.textContent = 'Auto-play tour';
    }
  }
}

/** A `<details>` section with a title and optional trailing badge. */
function collapsible(
  title: string,
  content: HTMLElement,
  open: boolean,
  badge?: HTMLElement
): HTMLElement {
  return h(
    'details',
    {class: 'card collapsible', open},
    h('summary', {}, h('span', {class: 'section-title'}, title), badge, icon('chevronDown', 16)),
    h('div', {class: 'collapsible-body'}, content)
  );
}

/** Whether a readout is a chart drawn in a map inset rather than in the panel. */
function isMapChart(spec: ReadoutSpec | undefined): boolean {
  return spec?.kind === 'chart' && spec.placement === 'map';
}

/** Text fields and other elements that consume typed characters. */
function isEditable(element: HTMLElement): boolean {
  return /^(INPUT|SELECT|TEXTAREA)$/.test(element.tagName) || element.isContentEditable;
}

/** Controls whose arrow keys do something else (sliders, radio groups, tabs). */
function ownsArrowKeys(element: HTMLElement): boolean {
  return Boolean(
    element.closest(
      '[role="slider"], [role="radio"], [role="radiogroup"], [role="tab"], [role="tablist"], [role="listbox"]'
    )
  );
}

type PaddingSides = Partial<ViewPadding>;

/** A step camera's `padding` as per-side pixels, or `null` when it asks for none. */
function normalizePadding(padding: StepCamera['padding']): PaddingSides | null {
  if (padding === undefined) return null;
  if (typeof padding === 'number') {
    return {top: padding, right: padding, bottom: padding, left: padding};
  }
  return padding;
}

/** Sums two paddings side by side. */
function addPadding(a: ViewPadding, b: PaddingSides): ViewPadding {
  return {
    top: a.top + (b.top ?? 0),
    right: a.right + (b.right ?? 0),
    bottom: a.bottom + (b.bottom ?? 0),
    left: a.left + (b.left ?? 0)
  };
}

/** Shallow merge that skips `undefined` fields of the override, so they do not erase the base. */
function mergeDefined<T extends object>(base: T | null, override: T): T {
  const merged = {...base} as T;
  for (const [key, value] of Object.entries(override)) {
    if (value !== undefined) (merged as Record<string, unknown>)[key] = value;
  }
  return merged;
}

/** Merges a step's basemap over the earlier one; `palette` merges per theme. */
function mergeBasemap(base: BasemapSpec | null, override: BasemapSpec): BasemapSpec {
  const merged = mergeDefined(base, override);
  if (base?.palette && override.palette) {
    const split = (palette: NonNullable<BasemapSpec['palette']>) =>
      'light' in palette || 'dark' in palette
        ? (palette as {light?: GroundPalette; dark?: GroundPalette})
        : {light: palette as GroundPalette, dark: palette as GroundPalette};
    const before = split(base.palette);
    const after = split(override.palette);
    merged.palette = {
      light: {...before.light, ...after.light},
      dark: {...before.dark, ...after.dark}
    };
  }
  return merged;
}

/** Web Mercator y of a latitude as a fraction of the world height (north is 0). */
function getMercatorY(latitude: number): number {
  const clamped = Math.min(Math.max(latitude, -85), 85);
  return (1 - Math.log(Math.tan(Math.PI / 4 + (clamped * Math.PI) / 360)) / Math.PI) / 2;
}
