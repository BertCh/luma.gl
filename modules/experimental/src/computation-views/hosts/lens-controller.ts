// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  AxisBindingValue,
  AxisBindings,
  AxisId,
  LensDescription,
  LensRungId,
  LensUrlState,
  RenderTeX,
  ViewGround
} from '../types';
import type {LensRecord, ViewContext, ViewSelectionState} from '../registry/view-registration';
import {ViewRegistry, createViewContext} from '../registry/view-registry';
import {getAxisHeader} from '../registry/default-axis-header';
import {LadderCoordinator} from '../coordinator/ladder-coordinator';
import {createGuardContext, evaluateGuards, type GuardSpec} from '../guards/guards';
import {describeLensView} from '../twin/description';
import {h, findViewNodes, type ViewElementNode} from '../view-node';
import {renderAxisHeader} from '../views/axis-header-view';
import {renderVerdictChip} from '../views/verdict-chip';
import {getAdapterLabel, getDatumKeySpace, getObserverEffects} from '../views/record-queries';
import {buildCommandGraphViewModel} from '../views/command-graph-view';
import {ScaleRegistry} from '../scales/scale-registry';
import {createKeyRelationGraphFromRecord} from './record-relations';
import {createLadderNavigation, navigateLadder, type LadderNavigationKey} from './navigation-graph';
import {getLensHistoryAction} from './url-state';

/** Pure lens inputs; DOM and device ownership belong to the host. */
export type LensControllerOptions = {
  record: LensRecord;
  registry: ViewRegistry;
  rung: LensRungId;
  viewId?: string;
  bindings?: AxisBindings;
  params?: LensUrlState['params'];
  ground?: ViewGround;
  reducedMotion?: boolean;
  renderTeX?: RenderTeX;
  coordinator?: LadderCoordinator;
  lensId: string;
  caption?: string;
  guards?: readonly GuardSpec[];
  now?: () => number;
};

/** Semantic actions accepted by both DOM hosts and programmatic readers. */
export type LensAction =
  | {type: 'set-rung'; rung: LensRungId}
  | {type: 'bind'; axis: AxisId; value: AxisBindingValue}
  | {type: 'free'; axis: AxisId}
  | {type: 'pick'; axis: AxisId; keys: readonly number[]; label?: string}
  | {type: 'hover'; axis: AxisId; keys: readonly number[]}
  | {type: 'hover-end'}
  | {type: 'scrub-preview'; value: number}
  | {type: 'scrub-commit'}
  | {type: 'toggle'; axis: string; value: string}
  | {type: 'pin-inspector'; unitId: string}
  | {type: 'close-inspector'}
  | {type: 'navigate'; key: LadderNavigationKey}
  | {type: 'set-record'; record: LensRecord}
  | {type: 'mark-stale'}
  | {type: 'apply-url-state'; state: LensUrlState}
  | {type: 'request-update'};

/** Complete host-independent render, description and history decision. */
export type LensFrame = {
  tree: ViewElementNode;
  description: LensDescription;
  announcement?: string;
  state: LensUrlState;
  historyAction: 'push' | 'replace' | 'none';
  rungs: Array<{rung: LensRungId; status: 'enabled' | 'disabled'; reason?: string}>;
};

// Hosts share scales while they retain a record. Releasing the last host drops its cache.
const scaleCaches = new Map<LensRecord, {registry: ScaleRegistry; users: number}>();

/** DOM-free state machine for one lens and its participation in a shared coordinator. */
export class LensController {
  private options: LensControllerOptions;
  private record: LensRecord;
  private state: LensUrlState;
  private coordinator: LadderCoordinator;
  private registeredRelationIds: string[] = [];
  private scales: ScaleRegistry;
  private unsubscribe: () => void;
  private listeners = new Set<() => void>();
  private frame?: LensFrame;
  private pendingAnnouncement = false;
  private cursor = 0;
  private cursorVisible = false;
  private inspector?: string;
  private stale = false;
  private live = false;
  private dispatching = false;
  private rendering = false;
  private destroyed = false;

  constructor(options: LensControllerOptions) {
    this.options = options;
    this.record = options.record;
    this.coordinator = options.coordinator ?? new LadderCoordinator();
    this.state = {
      runId: this.record.runId,
      rung: options.rung,
      bindings: {...options.bindings},
      clauses: [],
      params: {...options.params}
    };
    this.cursor = typeof this.state.bindings.entity === 'number' ? this.state.bindings.entity : 0;
    this.scales = this.retainScales();
    this.registerRecordRelations();
    this.unsubscribe = this.coordinator.subscribe(
      event => {
        if (this.dispatching || this.rendering || this.destroyed) return;
        if (
          event.type !== 'selection' &&
          event.type !== 'param' &&
          !(event.type === 'coupling' && event.viewId === options.lensId)
        )
          return;
        if (
          event.type === 'param' &&
          event.name === 'iteration' &&
          typeof event.value === 'number'
        ) {
          this.state = {...this.state, bindings: {...this.state.bindings, iteration: event.value}};
        }
        this.pendingAnnouncement =
          (event.type === 'selection' && event.selectionId === 'selection') ||
          (event.type === 'param' && event.heat !== 'hot');
        this.frame = undefined;
        for (const listener of [...this.listeners]) listener();
      },
      {viewId: options.lensId}
    );
  }

  /** Returns the current complete frame without changing history. */
  getFrame(): LensFrame {
    this.frame ??= this.renderFrame('none', this.pendingAnnouncement);
    this.pendingAnnouncement = false;
    return this.frame;
  }

  /** Applies one semantic action and returns its frame and history decision. */
  dispatch(action: LensAction): LensFrame {
    if (this.destroyed) return this.getFrame();
    const previous = this.getState();
    let announcement: string | boolean = !['hover', 'hover-end', 'scrub-preview'].includes(
      action.type
    );
    this.dispatching = true;
    try {
      switch (action.type) {
        case 'set-rung':
          this.state = {...this.state, rung: action.rung};
          break;
        case 'bind':
          this.bind(action.axis, action.value);
          break;
        case 'free':
          this.free(action.axis);
          break;
        case 'pick':
        case 'hover': {
          const keySpace = getDatumKeySpace(this.createContext());
          if (keySpace)
            this.coordinator.pick(
              action.type === 'pick' ? 'selection' : 'hover',
              {viewId: this.options.lensId, keySpace},
              action.keys,
              {writer: 'pointer', label: action.type === 'pick' ? action.label : undefined}
            );
          break;
        }
        case 'hover-end':
          this.coordinator.clearClauses('hover', this.options.lensId, {
            writer: 'pointer',
            origin: this.options.lensId
          });
          break;
        case 'scrub-preview':
          this.coordinator.previewParam('iteration', action.value, {
            writer: 'pointer',
            origin: this.options.lensId
          });
          this.bind('iteration', action.value);
          break;
        case 'scrub-commit':
          this.coordinator.commitParam('iteration', {
            writer: 'pointer',
            origin: this.options.lensId
          });
          break;
        case 'toggle': {
          let value = action.value;
          const axis = action.axis === 'node' ? 'expanded' : action.axis;
          if (axis === 'expanded') {
            const previousValue = this.state.params['expanded'];
            let expanded: string[] = [];
            try {
              const parsed: unknown = JSON.parse(String(previousValue ?? '[]'));
              if (Array.isArray(parsed))
                expanded = parsed.filter((entry): entry is string => typeof entry === 'string');
            } catch {
              /* Missing expansion state starts closed. */
            }
            value = JSON.stringify(
              expanded.includes(value)
                ? expanded.filter(entry => entry !== value)
                : [...expanded, value]
            );
          }
          this.state = {...this.state, params: {...this.state.params, [axis]: value}};
          break;
        }
        case 'pin-inspector':
          this.inspector = action.unitId;
          break;
        case 'close-inspector':
          this.inspector = undefined;
          break;
        case 'navigate': {
          const context = this.createContext();
          const navigation = createLadderNavigation({
            rungs: this.options.registry
              .listRungs(context)
              .filter(rung => rung.status === 'enabled')
              .map(rung => rung.rung),
            rung: this.state.rung,
            entityCount: this.getKeyCount(context)
          });
          navigation.entityIndex = Math.min(this.cursor, Math.max(0, navigation.entityCount - 1));
          // Enter always pins; the navigation helper's toggle starts from an unpinned cursor.
          const result = navigateLadder(navigation, action.key);
          this.cursor = result.state.entityIndex;
          this.cursorVisible = true;
          if (action.key === 'ArrowUp' || action.key === 'ArrowDown')
            this.state = {...this.state, rung: result.state.rungs[result.state.rungIndex]};
          if (action.key === 'Enter' && navigation.entityCount > 0)
            this.bind('entity', this.cursor);
          if (action.key === 'Escape') this.free('entity');
          announcement = result.announcement;
          break;
        }
        case 'mark-stale':
          this.stale = true;
          this.coordinator.markStale([this.options.lensId]);
          break;
        case 'set-record':
          this.releaseScales();
          this.record = action.record;
          this.scales = this.retainScales();
          this.registerRecordRelations();
          this.state = {...this.state, runId: action.record.runId};
          this.stale = false;
          this.live = true;
          this.coordinator.setRecord(action.record.runId);
          break;
        case 'apply-url-state':
          this.state = {
            ...action.state,
            runId: this.record.runId,
            bindings: {...action.state.bindings},
            params: {...action.state.params}
          };
          this.cursor =
            typeof this.state.bindings.entity === 'number' ? this.state.bindings.entity : 0;
          this.coordinator.clearClauses('selection', undefined, {
            writer: 'url',
            origin: this.options.lensId
          });
          for (const clause of action.state.clauses)
            this.coordinator.addClause('selection', clause, {
              writer: 'url',
              origin: this.options.lensId
            });
          break;
        case 'request-update':
          this.coordinator.requestUpdate(this.options.lensId);
          break;
      }
      const next = this.getState();
      const historyAction =
        action.type === 'scrub-preview'
          ? 'replace'
          : action.type === 'apply-url-state'
            ? 'none'
            : getLensHistoryAction(previous, next);
      this.frame = this.renderFrame(historyAction, announcement);
      return this.frame;
    } finally {
      this.dispatching = false;
    }
  }

  /** Returns a detached URL snapshot, including current shared selection clauses. */
  getState(): LensUrlState {
    return {
      ...this.state,
      bindings: Object.fromEntries(
        Object.entries(this.state.bindings).map(([axis, value]) => [
          axis,
          Array.isArray(value) ? [...value] : value
        ])
      ),
      params: {...this.state.params},
      clauses: this.coordinator.getClauses('selection')
    };
  }

  /** Releases this controller; shared page coordinators remain owned by the page. */
  destroy(): void {
    if (this.destroyed) return;
    this.getFrame();
    this.destroyed = true;
    this.unsubscribe();
    this.listeners.clear();
    if (
      this.coordinator.getClauses('hover').some(clause => clause.source === this.options.lensId)
    ) {
      this.coordinator.clearClauses('hover', this.options.lensId, {origin: this.options.lensId});
    }
    this.releaseScales();
    this.coordinator.removeRelations(this.registeredRelationIds);
    if (!this.options.coordinator) this.coordinator.destroy();
  }

  /** Subscribes to changes from other views, without echoing this controller's actions. */
  onCoordinatorChange(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private bind(axis: AxisId, value: AxisBindingValue): void {
    this.state = {...this.state, bindings: {...this.state.bindings, [axis]: value}};
    if (axis === 'entity' && typeof value === 'number') this.cursor = value;
  }

  private free(axis: AxisId): void {
    const bindings = {...this.state.bindings};
    delete bindings[axis];
    this.state = {...this.state, bindings};
  }

  private createContext(): ViewContext {
    return createViewContext(this.record, {
      bindings: this.state.bindings,
      ground: this.options.ground,
      idPrefix: this.options.lensId,
      reducedMotion: this.options.reducedMotion,
      renderTeX: this.options.renderTeX,
      packs: this.options.registry.getPacksFor(this.record),
      options: this.state.params,
      scales: this.scales
    });
  }

  private getKeyCount(context: ViewContext): number {
    return (
      this.record.axes.keySpaces.find(space => space.id === getDatumKeySpace(context))?.size ?? 0
    );
  }

  /**
   * Registers this record's declared relations on the shared page coordinator (ids prefixed by
   * run, so two lenses never replace each other's relations) and its key-space sizes. A re-run
   * replaces the previous run's relations.
   */
  private registerRecordRelations(): void {
    this.coordinator.removeRelations(this.registeredRelationIds);
    const relations = createKeyRelationGraphFromRecord(this.record)
      .list()
      .map(relation => ({...relation, id: `${this.record.runId}/${relation.id}`}));
    this.coordinator.addRelations(relations);
    this.registeredRelationIds = relations.map(relation => relation.id);
    for (const space of this.record.axes.keySpaces) {
      if (space.size !== undefined) this.coordinator.registerKeySpace(space.id, space.size);
    }
  }

  private resolveSelection(context: ViewContext): ViewSelectionState {
    const keySpace = getDatumKeySpace(context);
    if (!keySpace)
      return this.coordinator.getClauses('selection').length
        ? {unknownReason: 'datum key space unknown'}
        : {};
    const keyCount = this.record.axes.keySpaces.find(space => space.id === keySpace)?.size;
    if (keyCount === undefined)
      return this.coordinator.getClauses('selection').length
        ? {unknownReason: 'key space size unknown'}
        : {};
    // No viewId: a view highlights its own picks as well as linked picks.
    const resolve = (selectionId: string) => ({
      result: this.coordinator.resolve(selectionId, keySpace, {keyCount}),
      coordinator: this.coordinator
    });
    const {result, coordinator} = resolve('selection');
    const hover = resolve('hover').result;
    const selection: ViewSelectionState = {hovered: hover.status === 'ok' ? hover.keys : undefined};
    if (result.status === 'ok') {
      selection.highlighted = new Map(
        [...result.keys]
          .filter(key => key >= 0 && key < keyCount)
          .map(key => [key, {covered: 1, total: 1}])
      );
      // A sentinel absent from clause sources disables highlight's self-exclusion.
      let highlightId = `${this.options.lensId}:highlight`;
      const clauses = coordinator.getClauses('selection');
      while (clauses.some(clause => clause.source === highlightId)) highlightId += ':';
      const highlight = coordinator.highlight('selection', {
        viewId: highlightId,
        keySpace,
        keyCount
      });
      selection.offViewCount = highlight.status === 'ok' ? highlight.offView.count : 0;
    } else if (result.status !== 'empty')
      selection.unknownReason = result.status === 'unknown' ? result.reason : 'unlinked key spaces';
    return selection;
  }

  private renderFrame(
    historyAction: LensFrame['historyAction'],
    announce: string | boolean = false
  ): LensFrame {
    const now = this.options.now ?? Date.now;
    const start = now();
    const context = this.createContext();
    context.selection = this.resolveSelection(context);
    const resolution = this.options.registry.resolveRung(
      context,
      this.state.rung,
      this.options.viewId
    );
    const view = resolution.status === 'enabled' ? resolution.view : resolution.fallback;
    const model = view.build(context);
    const description = describeLensView(view, model, context, {
      rung: this.state.rung,
      caption: this.options.caption,
      disabledReason: resolution.status === 'disabled' ? resolution.reason : undefined
    });
    const header = getAxisHeader(view, model, context);
    // Explicit pins remain visible even if the fallback's signature does not mention the axis.
    const chips = [...header.chips];
    for (const [axis, value] of Object.entries(this.state.bindings)) {
      if (value === undefined) continue;
      const chip = {
        axis,
        label: this.record.axes.axes.find(candidate => candidate.id === axis)?.label ?? axis,
        state: 'bound' as const,
        valueText: Array.isArray(value) ? value.join(', ') : String(value),
        action: 'free' as const
      };
      const index = chips.findIndex(candidate => candidate.axis === axis);
      if (index < 0) chips.push(chip);
      else chips[index] = chip;
    }
    const rungs = this.options.registry.listRungs(context);
    const content = view.render(model, context);
    for (const mark of findViewNodes(
      content,
      node => node.attributes['data-axis'] === 'entity' && node.attributes['data-key'] !== undefined
    )) {
      const first = Number(mark.attributes['data-key']);
      const last = Number(mark.attributes['data-key-end'] ?? first);
      if (this.cursorVisible && first <= this.cursor && last >= this.cursor)
        mark.attributes['data-focus'] = 'true';
      const covered = [...(context.selection.highlighted?.keys() ?? [])].filter(
        key => key >= first && key <= last
      ).length;
      if (covered) {
        mark.attributes['data-state'] = 'highlight';
        mark.attributes['data-covered'] = covered;
        mark.attributes['data-total'] = last - first + 1;
      }
      if ([...(context.selection.hovered ?? [])].some(key => key >= first && key <= last))
        mark.attributes['data-hover'] = 'true';
    }
    const viewContainer = h('div', {class: 'lens-view'}, content);
    let inspector: ViewElementNode | undefined;
    if (this.inspector) {
      const unit = this.record.graphs.length
        ? buildCommandGraphViewModel(context).folded?.units.find(
            candidate => candidate.id === this.inspector
          )
        : undefined;
      const member = unit?.memberIds[0];
      const table = this.options.registry.getTableView();
      const inspectorContext = {
        ...context,
        idPrefix: `${context.idPrefix}-inspector`,
        bindings: {...context.bindings, node: member ?? this.inspector}
      };
      inspector = h(
        'aside',
        {class: 'lens-inspector', 'aria-label': 'Pinned inspector'},
        h('button', {type: 'button', 'data-action': 'close-inspector'}, 'Close inspector'),
        table.render(table.build(inspectorContext), inspectorContext)
      );
    }
    const tree = h(
      'div',
      {
        class: 'ladder-lens-frame',
        tabindex: 0,
        role: 'group',
        'aria-roledescription': 'computation lens',
        'aria-label': description.alt,
        'data-ground': context.ground
      },
      h(
        'nav',
        {class: 'lens-rung-rail', 'aria-label': 'Ladder'},
        rungs.map(({rung, status, reason}) =>
          h(
            'button',
            {
              type: 'button',
              tabindex: -1,
              'data-action': 'rung',
              'data-value': rung,
              'aria-current': rung === this.state.rung ? 'true' : undefined,
              'aria-disabled': status === 'disabled' ? 'true' : undefined,
              title: reason
            },
            rung
          )
        )
      ),
      renderAxisHeader({...header, chips}),
      h(
        'div',
        {class: 'lens-verdicts'},
        evaluateGuards(
          this.options.guards ?? [],
          createGuardContext(this.record, context.packs ?? [])
        ).map(renderVerdictChip)
      ),
      resolution.status === 'disabled'
        ? h('p', {class: 'lens-disabled-reason'}, resolution.reason)
        : undefined,
      this.stale || this.coordinator.getCoupling(this.options.lensId) === 'stale'
        ? h('span', {class: 'lens-badge lens-badge--stale'}, 'stale — re-running on this GPU')
        : undefined,
      viewContainer,
      context.selection.unknownReason
        ? h('p', {class: 'lens-unknown'}, `Selection unknown: ${context.selection.unknownReason}`)
        : undefined,
      context.selection.offViewCount
        ? h(
            'p',
            {class: 'lens-off-view'},
            `↗ ${context.selection.offViewCount} selected keys off view`
          )
        : undefined,
      h(
        'p',
        {class: 'lens-caption'},
        this.options.caption ?? description.alt,
        ` · ${getAdapterLabel(this.record)} · ${this.live ? 'live' : 'recorded'} · run ${this.record.runId}`,
        getObserverEffects(this.record).map(effect =>
          h('span', {class: 'lens-observer-effect'}, ` · ${effect}`)
        )
      ),
      inspector
    );
    this.rendering = true;
    let coupling;
    try {
      coupling = this.coordinator.reportUpdate(this.options.lensId, now() - start);
    } finally {
      this.rendering = false;
    }
    if (coupling === 'decoupled') {
      viewContainer.attributes['class'] = 'lens-view lens-decoupled';
      viewContainer.attributes['aria-disabled'] = 'true';
      tree.children.push(h('button', {type: 'button', 'data-action': 'request-update'}, 'update'));
    }
    // All descendants participate through the frame's single tab stop and delegated navigation.
    for (const element of findViewNodes(tree, node =>
      ['button', 'input', 'select', 'a'].includes(node.tag)
    ))
      element.attributes['tabindex'] = -1;
    return {
      tree,
      description,
      announcement:
        typeof announce === 'string' ? announce : announce ? description.live : undefined,
      state: this.getState(),
      historyAction,
      rungs
    };
  }

  private retainScales(): ScaleRegistry {
    let entry = scaleCaches.get(this.record);
    if (!entry) {
      entry = {registry: new ScaleRegistry({runId: this.record.runId}), users: 0};
      scaleCaches.set(this.record, entry);
    }
    entry.users++;
    return entry.registry;
  }

  private releaseScales(): void {
    const entry = scaleCaches.get(this.record);
    if (entry && --entry.users === 0) scaleCaches.delete(this.record);
  }
}
