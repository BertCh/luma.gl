// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  ViewRegistry,
  createViewContext
} from '../../../src/computation-views/registry/view-registry';
import {createDefaultViewRegistry} from '../../../src/computation-views/registry/default-registry';
import {TABLE_VIEW} from '../../../src/computation-views/views/table-view';
import {LENS_RUNG_IDS} from '../../../src/computation-views/types';
import {makeScanRecord} from '../fixtures/scan-record';

const context = createViewContext(makeScanRecord());

describe('ViewRegistry', () => {
  test('defaults and table are always available', () => {
    const registry = createDefaultViewRegistry();
    expect(context).toMatchObject({
      bindings: {},
      ground: 'light',
      idPrefix: 'lens',
      reducedMotion: false
    });
    expect(registry.getTableView()).toBe(TABLE_VIEW);
    expect(registry.resolveRung(context, 'table').status).toBe('enabled');
    expect(registry.listRungs(context).map(item => item.rung)).toEqual(LENS_RUNG_IDS);
    expect(registry.listRungs(context).find(item => item.rung === 'kernel')).toEqual({
      rung: 'kernel',
      status: 'disabled',
      reason: 'no compute node with a recorded dispatch shape and workgroup size'
    });
  });
  test('orders candidates and keeps first failure, excluding inactive views', () => {
    const registry = new ViewRegistry();
    registry.register({
      ...TABLE_VIEW,
      id: 'z',
      rung: 'kernel',
      order: 5,
      satisfies: () => ({ok: false, reason: 'z reason'})
    });
    registry.register({
      ...TABLE_VIEW,
      id: 'a',
      rung: 'kernel',
      order: 5,
      satisfies: () => ({ok: false, reason: 'a reason'})
    });
    registry.register({
      ...TABLE_VIEW,
      id: 'inactive',
      rung: 'kernel',
      order: 0,
      isActive: () => false
    });
    expect(registry.viewsForRung('kernel').map(view => view.id)).toEqual(['inactive', 'a', 'z']);
    expect(registry.resolveRung(context, 'kernel')).toMatchObject({
      status: 'disabled',
      reason: 'a reason',
      fallback: TABLE_VIEW
    });
    expect(() => registry.register(TABLE_VIEW)).toThrow();
    registry.register({...TABLE_VIEW, id: 'working', rung: 'kernel', order: 10});
    expect(registry.resolveRung(context, 'kernel', 'a')).toMatchObject({
      status: 'enabled',
      view: {id: 'working'}
    });
  });
  test('pack views win by order, with explicit satisfied preference', () => {
    const registry = new ViewRegistry();
    registry.register({...TABLE_VIEW, id: 'generic', rung: 'graph', order: 50});
    registry.registerPack({
      id: 'gpu-core',
      metadata: {coordinateModel: 'projected'},
      views: [{...TABLE_VIEW, id: 'domain', rung: 'graph', order: 1}]
    });
    registry.registerPack({id: 'other'});
    expect(registry.resolveRung(context, 'graph')).toMatchObject({
      view: {id: 'domain'},
      alternatives: [{id: 'generic'}]
    });
    expect(registry.resolveRung(context, 'graph', 'generic')).toMatchObject({
      view: {id: 'generic'},
      alternatives: [{id: 'domain'}]
    });
    expect(registry.getPacksFor(context.record)).toEqual([registry.getPack('gpu-core')]);
    expect(registry.get('missing')).toBeUndefined();
    expect(registry.get('domain')?.id).toBe('domain');
    expect(() => registry.registerPack({id: 'gpu-core'})).toThrow();
  });
});
