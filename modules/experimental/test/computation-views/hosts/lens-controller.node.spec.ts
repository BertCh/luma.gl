// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, describe, expect, test, vi} from 'vitest';
import {
  LensController,
  type LensControllerOptions
} from '../../../src/computation-views/hosts/lens-controller';
import {LadderCoordinator} from '../../../src/computation-views/coordinator/ladder-coordinator';
import {ViewRegistry} from '../../../src/computation-views/registry/view-registry';
import {FORMULA_CARD_VIEW} from '../../../src/computation-views/views/formula-card';
import {BUFFER_STRIP_VIEW} from '../../../src/computation-views/views/buffer-strip';
import {COMMAND_GRAPH_VIEW} from '../../../src/computation-views/views/command-graph-view';
import {
  findViewNodes,
  getViewNodeText,
  viewNodeToHtml
} from '../../../src/computation-views/view-node';
import {makeV7Record} from '../fixtures/v7-lens-record';

const controllers: LensController[] = [];
function makeController(options: Partial<LensControllerOptions> = {}) {
  const registry = new ViewRegistry();
  registry.register(FORMULA_CARD_VIEW);
  registry.register(BUFFER_STRIP_VIEW);
  registry.register(COMMAND_GRAPH_VIEW);
  const controller = new LensController({
    record: makeV7Record(),
    registry,
    lensId: `test-${controllers.length}`,
    rung: 'table',
    ...options
  });
  controllers.push(controller);
  return controller;
}
afterEach(() => {
  for (const controller of controllers) controller.destroy();
  controllers.length = 0;
});

describe('LensController', () => {
  test('renders table, an honest disabled fallback and a single generated description', () => {
    const controller = makeController();
    const frame = controller.getFrame();
    expect(frame.tree.attributes['aria-label']).toBe(frame.description.alt);
    expect(frame.tree.attributes['tabindex']).toBe(0);
    expect(viewNodeToHtml(frame.tree)).toContain('lens-table');
    expect(frame.historyAction).toBe('none');
    const disabled = controller.dispatch({type: 'set-rung', rung: 'compare'});
    expect(viewNodeToHtml(disabled.tree)).toContain('lens-disabled-reason');
    expect(viewNodeToHtml(disabled.tree)).toContain('lens-table');
    expect(getViewNodeText(disabled.tree)).toContain('no view registered for rung compare');
    expect(disabled.rungs.find(rung => rung.rung === 'compare')?.status).toBe('disabled');
    expect(disabled.historyAction).toBe('push');
    expect(getViewNodeText(frame.tree)).toContain('Fixture GPU · recorded · run v7-run');
    expect(getViewNodeText(frame.tree)).toContain('timestamps disabled');
  });
  test('entity binding pushes history and remains explicit in the fallback header; freeing removes it', () => {
    const controller = makeController();
    const bound = controller.dispatch({type: 'bind', axis: 'entity', value: 3});
    expect(bound.historyAction).toBe('push');
    expect(getViewNodeText(bound.tree)).toContain('row 3 pinned');
    const freed = controller.dispatch({type: 'free', axis: 'entity'});
    expect(freed.state.bindings.entity).toBeUndefined();
    expect(getViewNodeText(freed.tree)).not.toContain('row 3 pinned');
    expect(freed.historyAction).toBe('push');
  });
  test('picks highlight their own captured marks and hover changes no URL state', () => {
    const coordinator = new LadderCoordinator();
    const controller = makeController({
      coordinator,
      rung: BUFFER_STRIP_VIEW.rung,
      viewId: BUFFER_STRIP_VIEW.id
    });
    const picked = controller.dispatch({type: 'pick', axis: 'entity', keys: [2], label: 'row two'});
    expect(coordinator.getClauses('selection')).toMatchObject([
      {source: 'test-0', predicate: {kind: 'keys', keys: [2]}}
    ]);
    expect(viewNodeToHtml(picked.tree)).toMatch(/data-state="(selection|highlight)"/);
    expect(picked.state.bindings.entity).toBeUndefined();
    const hovered = controller.dispatch({type: 'hover', axis: 'entity', keys: [3]});
    expect(hovered.historyAction).toBe('none');
    expect(hovered.announcement).toBeUndefined();
    expect(coordinator.getClauses('hover')).toHaveLength(1);
    controller.dispatch({type: 'hover-end'});
    expect(coordinator.getClauses('hover')).toHaveLength(0);
  });
  test('scrub previews replace history silently and commits announce once', () => {
    const coordinator = new LadderCoordinator();
    const controller = makeController({coordinator});
    const preview = controller.dispatch({type: 'scrub-preview', value: 2});
    expect(preview.historyAction).toBe('replace');
    expect(preview.announcement).toBeUndefined();
    expect(preview.state.bindings.iteration).toBe(2);
    expect(coordinator.getActionLog()).toHaveLength(0);
    const commit = controller.dispatch({type: 'scrub-commit'});
    expect(commit.announcement).toBe(commit.description.live);
    expect(coordinator.getActionLog()).toHaveLength(1);
  });
  test('navigates enabled rungs, moves an unpinned cursor, then Enter pins and Escape frees', () => {
    const controller = makeController();
    const enabled = controller
      .getFrame()
      .rungs.filter(rung => rung.status === 'enabled')
      .map(rung => rung.rung);
    const up = controller.dispatch({type: 'navigate', key: 'ArrowUp'});
    expect(up.state.rung).toBe(enabled[enabled.indexOf('table') - 1]);
    expect(up.announcement).toContain('Rung');
    expect(controller.dispatch({type: 'navigate', key: 'ArrowDown'}).state.rung).toBe('table');
    controller.dispatch({type: 'navigate', key: 'ArrowRight'});
    expect(controller.getState().bindings.entity).toBeUndefined();
    const pinned = controller.dispatch({type: 'navigate', key: 'Enter'});
    expect(pinned.state.bindings.entity).toBe(1);
    expect(pinned.historyAction).toBe('push');
    expect(controller.dispatch({type: 'navigate', key: 'Enter'}).state.bindings.entity).toBe(1);
    expect(
      controller.dispatch({type: 'navigate', key: 'Escape'}).state.bindings.entity
    ).toBeUndefined();
    controller.dispatch({type: 'navigate', key: 'End'});
    expect(controller.dispatch({type: 'navigate', key: 'Enter'}).state.bindings.entity).toBe(15);
  });
  test('shows a focus outline on the cursor without pinning the entity', () => {
    const controller = makeController({rung: BUFFER_STRIP_VIEW.rung, viewId: BUFFER_STRIP_VIEW.id});
    const frame = controller.dispatch({type: 'navigate', key: 'ArrowRight'});
    expect(frame.state.bindings.entity).toBeUndefined();
    expect(
      findViewNodes(frame.tree, node => node.attributes['data-focus'] === 'true')
    ).not.toHaveLength(0);
  });
  test('stale badges last until the new run lands', () => {
    const controller = makeController();
    expect(viewNodeToHtml(controller.dispatch({type: 'mark-stale'}).tree)).toContain(
      'stale — re-running on this GPU'
    );
    const record = {...makeV7Record(), runId: 'next-run'};
    const updated = controller.dispatch({type: 'set-record', record});
    expect(viewNodeToHtml(updated.tree)).not.toContain('lens-badge--stale');
    expect(updated.state.runId).toBe('next-run');
    expect(getViewNodeText(updated.tree)).toContain('live · run next-run');
  });
  test('decouples a slow render and only explicitly relinks on request-update', () => {
    let time = 0;
    let slow = true;
    const controller = makeController({
      now: () => {
        time += slow ? 101 : 1;
        return time;
      }
    });
    expect(viewNodeToHtml(controller.getFrame().tree)).toContain('lens-decoupled');
    slow = false;
    expect(
      viewNodeToHtml(controller.dispatch({type: 'toggle', axis: 'preset', value: 'algorithm'}).tree)
    ).toContain('lens-decoupled');
    expect(viewNodeToHtml(controller.dispatch({type: 'request-update'}).tree)).not.toContain(
      'lens-decoupled'
    );
  });
  test('two controllers share picks without echoing the source or notifying destroyed subscribers', () => {
    const coordinator = new LadderCoordinator();
    const first = makeController({coordinator, lensId: 'first'});
    const second = makeController({coordinator, lensId: 'second'});
    const firstChanged = vi.fn();
    const secondChanged = vi.fn();
    first.onCoordinatorChange(firstChanged);
    const unsubscribe = second.onCoordinatorChange(secondChanged);
    first.dispatch({type: 'pick', axis: 'entity', keys: [4]});
    expect(firstChanged).not.toHaveBeenCalled();
    expect(secondChanged).toHaveBeenCalledTimes(1);
    expect(second.getFrame().state.clauses[0].source).toBe('first');
    unsubscribe();
    second.destroy();
    first.dispatch({type: 'pick', axis: 'entity', keys: [5]});
    expect(secondChanged).toHaveBeenCalledTimes(1);
  });
  test('runs only explicitly supplied guards and renders a pinned table inspector', () => {
    const evaluate = vi.fn(() => ({status: 'unknown' as const, reason: 'missing premise'}));
    const controller = makeController({
      guards: [{id: 'premise', label: 'Premise', blocks: [], evaluate}]
    });
    expect(viewNodeToHtml(controller.getFrame().tree)).toContain('lens-verdict--unknown');
    expect(evaluate).toHaveBeenCalledTimes(1);
    expect(
      viewNodeToHtml(controller.dispatch({type: 'pin-inspector', unitId: 'double'}).tree)
    ).toContain('lens-inspector');
    expect(viewNodeToHtml(controller.dispatch({type: 'close-inspector'}).tree)).not.toContain(
      'lens-inspector'
    );
    expect(
      findViewNodes(makeController().getFrame().tree, node =>
        String(node.attributes['class']).includes('lens-verdict--')
      )
    ).toHaveLength(0);
  });
  test('URL state restores clauses with no history echo and state snapshots are detached', () => {
    const controller = makeController();
    const state = {
      ...controller.getState(),
      rung: 'formula' as const,
      bindings: {entity: 4},
      params: {preset: 'algorithm'},
      clauses: [
        {
          source: 'link',
          keySpace: 'row',
          predicate: {kind: 'keys' as const, keys: [4]},
          label: 'row 4'
        }
      ]
    };
    const frame = controller.dispatch({type: 'apply-url-state', state});
    expect(frame.historyAction).toBe('none');
    expect(frame.state.bindings.entity).toBe(4);
    expect(frame.state.clauses).toHaveLength(1);
    state.bindings.entity = 8;
    expect(controller.getState().bindings.entity).toBe(4);
  });
  test('shared coordinator selections use declared record relations and report unknown mappings', () => {
    const record = makeV7Record();
    record.axes = {
      ...record.axes,
      keySpaces: [...record.axes.keySpaces, {id: 'external', label: 'external', size: 16}],
      relations: [
        {
          id: 'external-row',
          from: 'external',
          to: 'row',
          cardinality: '1:1',
          kind: 'function',
          ref: {function: 'identity'},
          timeDependent: false
        }
      ]
    };
    const coordinator = new LadderCoordinator();
    const controller = makeController({
      record,
      coordinator,
      rung: BUFFER_STRIP_VIEW.rung,
      viewId: BUFFER_STRIP_VIEW.id
    });
    coordinator.pick('selection', {viewId: 'external', keySpace: 'external'}, [4]);
    expect(viewNodeToHtml(controller.getFrame().tree)).toContain('data-state="highlight"');
    const unknown = {
      ...record,
      axes: {
        ...record.axes,
        relations: record.axes.relations.map(relation => ({...relation, kind: 'unknown' as const}))
      }
    };
    expect(
      getViewNodeText(controller.dispatch({type: 'set-record', record: unknown}).tree)
    ).toContain('Selection unknown: time-dependent relation not captured');
  });
  test('keeps off-view counts explicit when mapped keys exceed the displayed domain', () => {
    const record = makeV7Record();
    record.axes = {
      ...record.axes,
      keySpaces: [
        {id: 'row', label: 'row', size: 8},
        {id: 'source', label: 'source', size: 16}
      ],
      relations: [
        {
          id: 'source-row',
          from: 'source',
          to: 'row',
          cardinality: '1:1',
          kind: 'function',
          ref: {function: 'identity'},
          timeDependent: false
        }
      ]
    };
    const coordinator = new LadderCoordinator();
    const controller = makeController({record, coordinator});
    coordinator.pick('selection', {viewId: 'source', keySpace: 'source'}, [12]);
    expect(getViewNodeText(controller.getFrame().tree)).toContain('1 selected keys off view');
  });

  test('expanded graph actions maintain a serializable set of group ids', () => {
    const controller = makeController();
    controller.dispatch({type: 'toggle', axis: 'node', value: 'group:rows'});
    expect(controller.getState().params['expanded']).toBe('["group:rows"]');
    controller.dispatch({type: 'toggle', axis: 'node', value: 'group:rows'});
    expect(controller.getState().params['expanded']).toBe('[]');
  });

  test('shared scrub previews never announce, while discrete linked picks do', () => {
    const coordinator = new LadderCoordinator();
    const first = makeController({coordinator});
    const second = makeController({coordinator});
    second.getFrame();
    first.dispatch({type: 'scrub-preview', value: 3});
    expect(second.getFrame().announcement).toBeUndefined();
    expect(second.getState().bindings.iteration).toBe(3);
    first.dispatch({type: 'pick', axis: 'entity', keys: [3]});
    expect(second.getFrame().announcement).toBe(second.getFrame().description.live);
  });

  test('unknown key-space sizes do not silently discard a selection', () => {
    const record = makeV7Record();
    record.axes = {...record.axes, keySpaces: [{id: 'row', label: 'row'}]};
    const controller = makeController({record});
    expect(
      getViewNodeText(controller.dispatch({type: 'pick', axis: 'entity', keys: [3]}).tree)
    ).toContain('Selection unknown: key space size unknown');
  });
  test('replacing a record invalidates relations retained by a supplied coordinator', () => {
    const record = makeV7Record();
    record.axes = {
      ...record.axes,
      keySpaces: [...record.axes.keySpaces, {id: 'external', label: 'external', size: 16}],
      relations: [
        {
          id: 'external-row',
          from: 'external',
          to: 'row',
          cardinality: '1:1',
          kind: 'function',
          ref: {function: 'identity'},
          timeDependent: false
        }
      ]
    };
    // The controller registers the record's relations on the supplied page coordinator.
    const coordinator = new LadderCoordinator();
    const controller = makeController({record, coordinator});
    coordinator.pick('selection', {viewId: 'external', keySpace: 'external'}, [4]);
    expect(getViewNodeText(controller.getFrame().tree)).not.toContain('Selection unknown');
    const updated = {
      ...record,
      axes: {
        ...record.axes,
        relations: record.axes.relations.map(relation => ({...relation, kind: 'unknown' as const}))
      }
    };
    expect(
      getViewNodeText(controller.dispatch({type: 'set-record', record: updated}).tree)
    ).toContain('Selection unknown: time-dependent relation not captured');
  });
});
