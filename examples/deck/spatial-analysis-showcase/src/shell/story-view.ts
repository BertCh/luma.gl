// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import type {DataCatalog, LoadedDataset} from '../data/catalog';
import {loadDatasetInfo} from '../data/catalog';
import type {DatasetInfo} from '../data/dataset-types';
import type {DeckHost, HostServices} from '../engine/deck-host';
import {
  formatCompiledGraphTiming,
  hasGPUTimestamps,
  measureCompiledGraph
} from '../engine/vector-timing';
import {OptionsStore, type OptionChangeSource} from '../scenes/options-state';
import {getChapter} from '../scenes/chapters';
import {loadScene} from '../scenes/registry';
import type {
  AnyScene,
  ChartData,
  OptionSpec,
  OptionState,
  ReadoutSpec,
  SceneContext,
  SceneInstance,
  StoryStep
} from '../scenes/scene';
import {renderChart} from './chart';
import {renderMarkdown} from './markdown';
import {h, clearElement, copyText, formatBytes, icon} from './dom';
import {highlightCode} from './highlight';
import {renderLegends} from './legend';
import {OptionsPanel, StepControls} from './options-panel';
import {getReferenceHash, getStoryHash, navigate, replaceHash, type Route} from './router';
import {getTheme, onThemeChange} from './theme';

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
type ReadoutView = {value: HTMLElement; row: HTMLElement; chart: HTMLElement | null};

const DEFAULT_FLY_MILLISECONDS = 1600;

/** Dependencies the story view needs from the application. */
export type StoryViewDependencies = {
  getHost: () => DeckHost;
  catalog: DataCatalog;
  hasWebGPU: boolean;
};

/**
 * The story page: narrative panel on the left, the persistent Deck map on the right. It owns one
 * scene at a time and wires the option store, the URL, the legends and readouts to it.
 */
export class StoryView {
  private readonly dependencies: StoryViewDependencies;
  private root: HTMLElement | null = null;
  private scene: AnyScene | null = null;
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
  private stepControls: StepControls | null = null;
  private optionsPanel: OptionsPanel | null = null;
  private activeTab: PanelTab = 'story';
  /** Whether scene-originated option writes are currently forwarded to `instance.setOption`. */
  private notifySceneWrites = false;
  private derivedFrame = 0;
  private urlTimer = 0;
  private cleanups: (() => void)[] = [];
  private mountGeneration = 0;

  // Panel parts that change while the story runs.
  private stepRows: {
    item: HTMLElement;
    row: HTMLButtonElement;
    badge: HTMLElement;
    detail: HTMLElement;
  }[] = [];
  private stepCounter!: HTMLElement;
  private previousButton!: HTMLButtonElement;
  private nextButton!: HTMLButtonElement;
  private legendContainer!: HTMLElement;
  private legendOverlay!: HTMLElement;
  private statusLine!: HTMLElement;
  private snippetCode!: HTMLElement;
  private snippetText = '';
  private tabButtons = new Map<PanelTab, HTMLButtonElement>();
  private tabPanels = new Map<PanelTab, HTMLElement>();
  private controlsDot!: HTMLElement;

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
    root.append(h('div', {class: 'page-loading'}, 'Loading story…'));
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
    for (const cleanup of this.cleanups) cleanup();
    this.cleanups = [];
    this.instance = null;
    this.dependencies.getHost().deactivate();
    this.dependencies.getHost().setActive(false);
    this.root?.replaceChildren();
    this.scene = null;
    this.store = null;
    this.gpuExtents.clear();
    for (const key of Object.keys(this.legendData)) delete this.legendData[key];
    this.readoutSpecs.clear();
    this.readoutValues.clear();
    this.readoutCharts.clear();
    this.allReadoutViews.clear();
    this.stepReadoutViews.clear();
    this.stepRows = [];
    this.stepControls = null;
    this.optionsPanel = null;
    cancelAnimationFrame(this.derivedFrame);
    clearTimeout(this.urlTimer);
    this.derivedFrame = 0;
  }

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

  /** Defaults, then every step's options up to `index` in order. */
  private applyStepState(index: number, initial: boolean): void {
    const scene = this.scene;
    const store = this.store;
    if (!scene || !store) return;
    if (!initial) store.reset();
    for (let step = 0; step <= index; step++) {
      const options = scene.story[step]?.options;
      if (options) store.setMany(options as Record<string, never>);
    }
  }

  private getCumulativeCamera(index: number) {
    const scene = this.scene!;
    let view = {...scene.initialView};
    for (let step = 0; step <= index; step++) {
      const {transitionMs: _ignored, ...camera} = scene.story[step]?.camera ?? {};
      view = {...view, ...camera};
    }
    return view;
  }

  private buildLayout(root: HTMLElement, scene: AnyScene): void {
    const store = this.store!;
    const chapter = getChapter(scene.chapter);
    const host = this.dependencies.hasWebGPU ? this.dependencies.getHost() : null;

    this.statusLine = h(
      'p',
      {class: 'status-line', role: 'status'},
      this.dependencies.hasWebGPU ? 'Loading data and compiling GPU graphs…' : ''
    );
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

    // Story tab: the step accordion, then every readout.
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
    for (const spec of scene.readouts ?? []) {
      this.readoutSpecs.set(spec.id, spec);
      const view = this.createReadoutView(spec);
      this.allReadoutViews.set(spec.id, view);
      readoutList.append(view.row);
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
    if (scene.about?.what)
      aboutParts.push(
        h(
          'section',
          {},
          h('h4', {}, 'What it computes'),
          h('div', {class: 'prose', html: renderMarkdown(scene.about.what)})
        )
      );
    if (scene.about?.why)
      aboutParts.push(
        h(
          'section',
          {},
          h('h4', {}, 'Why it matters'),
          h('div', {class: 'prose', html: renderMarkdown(scene.about.why)})
        )
      );
    if (scene.about?.howToRead)
      aboutParts.push(
        h(
          'section',
          {},
          h('h4', {}, 'How to read the map'),
          h('div', {class: 'prose', html: renderMarkdown(scene.about.howToRead)})
        )
      );
    const aboutSection = aboutParts.length
      ? collapsible('About this analysis', h('div', {}, aboutParts), false)
      : null;

    // Datasets.
    const datasetList = h('ul', {class: 'dataset-list'});
    void Promise.all(scene.datasets.map(ref => loadDatasetInfo(ref.id))).then(infos => {
      scene.datasets.forEach((ref, index) =>
        datasetList.append(this.renderDatasetItem(ref.id, ref.role, infos[index]))
      );
    });
    const dataSection = scene.datasets.length ? collapsible('Data', datasetList, false) : null;

    // Under the hood.
    const hood = this.createUnderTheHood();
    this.cleanups.push(hood.destroy);

    const detailsPanel = h(
      'div',
      {class: 'tab-panel'},
      h('p', {class: 'lede'}, scene.summary),
      h(
        'div',
        {class: 'chips details-chips'},
        scene.contributors.map(name =>
          h('a', {class: 'chip', href: getReferenceHash(name), title: `Reference: ${name}`}, name)
        )
      ),
      codeSection,
      aboutSection,
      dataSection,
      hood.element
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

    // Footer navigation.
    this.stepCounter = h('span', {class: 'step-counter', 'aria-live': 'polite'});
    this.previousButton = h(
      'button',
      {class: 'btn', type: 'button', 'aria-label': 'Previous step'},
      icon('chevronLeft'),
      'Back'
    );
    this.nextButton = h(
      'button',
      {class: 'btn btn-primary', type: 'button', 'aria-label': 'Next step'},
      'Next',
      icon('chevronRight')
    );
    this.previousButton.addEventListener('click', () => this.goToStep(this.stepIndex - 1));
    this.nextButton.addEventListener('click', () => this.goToStep(this.stepIndex + 1));
    const footer = h(
      'footer',
      {class: 'panel-footer'},
      this.previousButton,
      this.stepCounter,
      this.nextButton
    );

    const sheetHandle = h(
      'button',
      {class: 'sheet-handle', type: 'button', 'aria-label': 'Expand or collapse the panel'},
      h('span', {})
    );
    const panel = h(
      'aside',
      {class: 'story-panel', 'aria-label': 'Story'},
      sheetHandle,
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
      footer
    );
    sheetHandle.addEventListener('click', () => panel.classList.toggle('is-expanded'));

    // Legends float over the map.
    this.legendContainer = h('div', {class: 'legends'});
    const legendBody = h('div', {class: 'map-legend-body'}, this.legendContainer);
    const legendToggle = h(
      'button',
      {class: 'map-legend-toggle', type: 'button', 'aria-expanded': 'true'},
      icon('layers', 14),
      'Legend',
      icon('chevronDown', 14)
    );
    this.legendOverlay = h('div', {class: 'map-legend', hidden: true}, legendToggle, legendBody);
    const setLegendCollapsed = (collapsed: boolean) => {
      this.legendOverlay.classList.toggle('is-collapsed', collapsed);
      legendToggle.setAttribute('aria-expanded', String(!collapsed));
    };
    setLegendCollapsed(window.matchMedia(MOBILE_QUERY).matches);
    legendToggle.addEventListener('click', () =>
      setLegendCollapsed(!this.legendOverlay.classList.contains('is-collapsed'))
    );

    const mapSlot = h('div', {class: 'story-map'});
    if (host) {
      mapSlot.append(host.element);
      host.setActive(true);
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
    mapSlot.append(this.legendOverlay);
    clearElement(root);
    root.append(h('div', {class: 'story-layout'}, panel, mapSlot));
    if (host) host.start();
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

  private createUnderTheHood() {
    const lines = h('dl', {class: 'hood-grid'});
    const cells = new Map<string, HTMLElement>();
    for (const [key, label] of [
      ['graphs', 'Compiled graphs'],
      ['nodes', 'Graph nodes'],
      ['rebuilds', 'Rebuilds'],
      ['frames', 'Frames encoded'],
      ['fps', 'Frame rate'],
      ['encode', 'CPU encode'],
      ['gpu', 'GPU time']
    ] as const) {
      const value = h('dd', {}, '—');
      cells.set(key, value);
      lines.append(h('div', {}, h('dt', {}, label), value));
    }
    const note = h(
      'p',
      {class: 'muted small'},
      'Compile once, rewrite parameters each frame: the rebuild counter only moves when a compile-time option changes. Compute and drawing share one command buffer per frame.'
    );
    const measureButton = h('button', {class: 'btn btn-small', type: 'button'}, 'Measure GPU time');
    const measureResult = h('div', {class: 'muted small'});
    let measuring = false;
    measureButton.addEventListener('click', async () => {
      const host = this.dependencies.getHost();
      const device = host.gpuDevice;
      const instance = host.activeInstance;
      if (!device || !instance || measuring) return;
      measuring = true;
      measureButton.disabled = true;
      measureResult.textContent = 'Measuring…';
      const fence = device.createBuffer({
        byteLength: 4,
        usage: Buffer.COPY_SRC | Buffer.COPY_DST | Buffer.STORAGE
      });
      try {
        let total = 0;
        let cpu = 0;
        let method = 'wall-clock' as 'wall-clock' | 'gpu-timestamps';
        for (const graph of instance.getCompiledGraphs()) {
          const timing = await measureCompiledGraph(device, graph as never, {
            parameters: undefined as never,
            completionBuffer: fence,
            runs: 5
          });
          total += timing.milliseconds;
          cpu += timing.cpuEncodeMilliseconds;
          method = timing.method;
        }
        const summary = formatCompiledGraphTiming({
          milliseconds: total,
          cpuEncodeMilliseconds: cpu,
          method
        });
        cells.get('gpu')!.textContent = `${total.toFixed(2)} ms`;
        measureResult.textContent = `All graphs once: ${summary}`;
      } catch {
        measureResult.textContent = 'Measurement was interrupted.';
      } finally {
        fence.destroy();
        measuring = false;
        measureButton.disabled = false;
      }
    });
    const update = () => {
      const host = this.dependencies.getHost();
      const {graphCount, nodeCount, rebuildCount} = host.pollRebuilds();
      const stats = host.stats;
      cells.get('graphs')!.textContent = String(graphCount);
      cells.get('nodes')!.textContent = String(nodeCount);
      cells.get('rebuilds')!.textContent = String(rebuildCount);
      cells.get('frames')!.textContent = String(stats.frameCount);
      cells.get('fps')!.textContent = `${stats.framesPerSecond.toFixed(0)} fps`;
      cells.get('encode')!.textContent = `${stats.encodeMilliseconds.toFixed(2)} ms`;
      if (!measuring && cells.get('gpu')!.textContent === '—' && host.gpuDevice) {
        measureResult.textContent = hasGPUTimestamps(host.gpuDevice)
          ? 'GPU timestamp queries available.'
          : 'timestamp-query unavailable: timings use wall clock.';
      }
    };
    const timer = window.setInterval(() => {
      if (this.dependencies.hasWebGPU) update();
    }, 500);
    const element = collapsible(
      'Under the hood',
      h('div', {}, lines, h('div', {class: 'hood-actions'}, measureButton, measureResult), note),
      false
    );
    return {element, destroy: () => clearInterval(timer)};
  }

  private async activate(scene: AnyScene, generation: number): Promise<void> {
    const host = this.dependencies.getHost();
    const store = this.store!;
    const {catalog} = this.dependencies;
    host.onStatus = message => {
      if (generation === this.mountGeneration) this.statusLine.textContent = message;
    };
    const snapshotAtStart = store.snapshot();
    const createScene = scene.create as (
      ctx: SceneContext<OptionState>
    ) => Promise<SceneInstance<OptionState>>;
    const instance = await host.activate<ShellInstance>({
      id: scene.id,
      view: this.getCumulativeCamera(this.stepIndex),
      create: async (services: HostServices) => {
        const loaded = new Map<string, LoadedDataset>();
        await Promise.all(
          scene.datasets.map(async ref => {
            loaded.set(ref.id, await catalog.load(ref.id, services.signal));
          })
        );
        services.signal.throwIfAborted();
        if (generation === this.mountGeneration) {
          const synthetic = [...loaded.values()].some(dataset => dataset.origin === 'synthetic');
          if (synthetic)
            this.statusLine.textContent =
              'Using deterministic synthetic data (offline or ?data=synthetic).';
        }
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
          getViewport: services.getViewport,
          requestLayers: services.updateLayers,
          setReadout: (id, value) => this.setReadout(id, value),
          setLegendExtent: (id, extent) => {
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
          setChart: (id, chart) => this.setChart(id, chart),
          refreshTooltip: services.refreshTooltip,
          limits: services.limits,
          setStatus: message => {
            if (generation === this.mountGeneration) this.statusLine.textContent = message;
          },
          setMapDragEnabled: services.setMapDragEnabled
        }) as Promise<ShellInstance>;
      }
    });
    if (!instance || generation !== this.mountGeneration) return;
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
        this.syncUrl();
      } else {
        this.scheduleDerivedRefresh();
      }
    });
    this.cleanups.push(unsubscribe);
    this.cleanups.push(
      onThemeChange(theme => {
        host.setTheme(theme);
        this.instance?.onThemeChange?.(theme);
      })
    );
    host.setTheme(getTheme());
    this.applyStepSideEffects(false);
  }

  private goToStep(index: number): void {
    const scene = this.scene;
    if (!scene || index < 0 || index >= scene.story.length) return;
    navigate(getStoryHash(scene.id, scene.story[index].id));
  }

  private createReadoutView(spec: ReadoutSpec): ReadoutView {
    const isChart = spec.kind === 'chart';
    const value = h('dd', {}, isChart ? null : '—');
    const chart = isChart ? h('div', {class: 'readout-chart'}) : null;
    const row = h(
      'div',
      {
        class: `readout${isChart ? ' readout-chart-row' : ''}${spec.layout === 'block' ? ' is-block' : ''}`,
        title: spec.help ?? ''
      },
      h('dt', {}, spec.label),
      isChart ? chart : value
    );
    const view = {value, row, chart};
    this.paintReadout(spec, view);
    return view;
  }

  /** Copies the latest value or chart of a readout into one rendered copy. */
  private paintReadout(spec: ReadoutSpec, view: ReadoutView): void {
    if (view.chart) {
      const chart = this.readoutCharts.get(spec.id);
      view.chart.replaceChildren(...(chart ? [renderChart(chart)] : []));
      return;
    }
    const text = formatReadout(spec, this.readoutValues.get(spec.id) ?? null);
    view.value.textContent = text;
    const block = spec.layout ? spec.layout === 'block' : text.includes('\n');
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
    for (const view of this.getReadoutViews(id)) this.paintReadout(spec, view);
  }

  private setChart(id: string, chart: ChartData | null): void {
    const spec = this.readoutSpecs.get(id);
    if (!spec) return;
    this.readoutCharts.set(id, chart);
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

  /** Warns once per scene about step control or readout ids that do not exist. */
  private warnAboutStepReferences(scene: AnyScene): void {
    const optionIds = new Set(scene.options.map(option => option.id));
    const readoutIds = new Set((scene.readouts ?? []).map(readout => readout.id));
    const problems: string[] = [];
    for (const step of scene.story) {
      for (const id of getStepControlIds(step)) {
        if (!optionIds.has(id)) problems.push(`${step.id}: control "${id}"`);
      }
      for (const id of getStepReadoutIds(step)) {
        if (!readoutIds.has(id)) problems.push(`${step.id}: readout "${id}"`);
      }
    }
    if (problems.length) {
      // biome-ignore lint/suspicious/noConsole: authoring aid for scene authors
      console.warn(
        `Scene "${scene.id}" has story steps that reference unknown ids (skipped): ${problems.join(', ')}`
      );
    }
  }

  /** Fills the active step's detail: text, "Adjust in this step" controls and readouts. */
  private renderStepDetail(index: number): void {
    const scene = this.scene!;
    const store = this.store!;
    const step = scene.story[index];
    const detail = this.stepRows[index].detail;
    this.stepControls?.destroy();
    this.stepControls = null;
    this.stepReadoutViews = new Map();
    const parts: (HTMLElement | null)[] = [
      h('div', {class: 'prose step-body', html: renderMarkdown(step.body)})
    ];

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
          h('h4', {class: 'step-adjust-title'}, 'Adjust in this step'),
          controls.element
        )
      );
    }

    const readoutSpecs = getStepReadoutIds(step)
      .map(id => this.readoutSpecs.get(id))
      .filter((spec): spec is ReadoutSpec => Boolean(spec));
    if (readoutSpecs.length) {
      const list = h('dl', {class: 'readouts step-readouts'});
      for (const spec of readoutSpecs) {
        const view = this.createReadoutView(spec);
        view.row.classList.toggle('is-highlighted', step.highlight?.readout === spec.id);
        this.stepReadoutViews.set(spec.id, view);
        list.append(view.row);
      }
      parts.push(list);
    }
    detail.replaceChildren(...parts.filter((part): part is HTMLElement => Boolean(part)));
    this.optionsPanel?.setStepControls(specs.map(spec => spec.id));
  }

  /** Scene-driven option writes can come every frame; refresh legends, code and URL at most once per frame / 300 ms. */
  private scheduleDerivedRefresh(): void {
    if (!this.derivedFrame) {
      this.derivedFrame = requestAnimationFrame(() => {
        this.derivedFrame = 0;
        this.refreshLegends();
        this.refreshSnippet();
      });
    }
    clearTimeout(this.urlTimer);
    this.urlTimer = window.setTimeout(() => this.syncUrl(), 300);
  }

  private refreshStep(animate: boolean): void {
    const scene = this.scene!;
    const step: StoryStep = scene.story[this.stepIndex];
    this.stepCounter.textContent = `Step ${this.stepIndex + 1} of ${scene.story.length}`;
    this.previousButton.disabled = this.stepIndex === 0;
    this.nextButton.disabled = this.stepIndex === scene.story.length - 1;
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
    if (animate) {
      this.stepRows[this.stepIndex].item.scrollIntoView({behavior: 'smooth', block: 'nearest'});
    }
    this.refreshLegends();
    this.refreshSnippet();
    this.syncUrl();
    if (this.instance || this.dependencies.hasWebGPU) this.applyStepSideEffects(animate);
  }

  private applyStepSideEffects(animate: boolean): void {
    const scene = this.scene;
    if (!scene || !this.dependencies.hasWebGPU || !this.instance) return;
    const host = this.dependencies.getHost();
    const step = scene.story[this.stepIndex];
    if (animate && step.camera) {
      host.setViewState({
        ...this.getCumulativeCamera(this.stepIndex),
        transitionMs: step.camera.transitionMs ?? DEFAULT_FLY_MILLISECONDS
      });
    }
    host.setCallout(step.callout ?? null);
  }

  private refreshLegends(): void {
    if (!this.scene || !this.store) return;
    renderLegends(
      this.legendContainer,
      this.scene.legends(this.store.snapshot() as never, this.legendData),
      this.gpuExtents
    );
    this.legendOverlay.hidden = this.legendContainer.childElementCount === 0;
  }

  private refreshSnippet(): void {
    const scene = this.scene;
    if (!scene?.snippet || !this.store) return;
    this.snippetText =
      typeof scene.snippet === 'string'
        ? scene.snippet
        : scene.snippet(this.store.snapshot() as never);
    this.snippetCode.innerHTML = highlightCode(this.snippetText);
  }

  private syncUrl(): void {
    const scene = this.scene;
    const store = this.store;
    if (!scene || !store) return;
    replaceHash(getStoryHash(scene.id, scene.story[this.stepIndex].id, store.serialize()));
  }

  private startKeyboard(): void {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName)) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === 'ArrowRight') this.goToStep(this.stepIndex + 1);
      else if (event.key === 'ArrowLeft') this.goToStep(this.stepIndex - 1);
    };
    window.addEventListener('keydown', handler);
    this.cleanups.push(() => window.removeEventListener('keydown', handler));
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

/** Option and button ids a step shows: its own list, else the options it sets. */
function getStepControlIds(step: StoryStep): readonly string[] {
  return step.controls ?? Object.keys(step.options ?? {});
}

/** Readout ids a step shows: its own list, else the highlighted readout. */
function getStepReadoutIds(step: StoryStep): readonly string[] {
  return step.readouts ?? (step.highlight?.readout ? [step.highlight.readout] : []);
}

function formatReadout(spec: ReadoutSpec, value: string | number | null): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'string') return value;
  if (!Number.isFinite(value)) return '—';
  switch (spec.format) {
    case 'integer':
      return Math.round(value).toLocaleString('en-US');
    case 'decimal':
      return value.toLocaleString('en-US', {maximumFractionDigits: 3});
    case 'percent':
      return `${(value * 100).toFixed(1)}%`;
    case 'milliseconds':
      return `${value.toFixed(2)} ms`;
    case 'bytes':
      return formatBytes(value);
    case 'meters':
      return value >= 1000 ? `${(value / 1000).toFixed(2)} km` : `${value.toFixed(0)} m`;
    default:
      return String(value);
  }
}
