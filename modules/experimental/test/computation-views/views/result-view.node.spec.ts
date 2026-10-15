// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {decodeRasterImage} from '../fixtures/png';
import {describe, expect, test} from 'vitest';
import {
  buildResultViewModel,
  renderResultView,
  RESULT_VIEW
} from '../../../src/computation-views/views/result-view';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {
  findViewNodes,
  getViewNodeText,
  viewNodeToHtml
} from '../../../src/computation-views/view-node';
import {getSelectionInk} from '../../../src/computation-views/encoding/selection-ink';
import {resolveValueScale} from '../../../src/computation-views/views/value-strip';
import {
  CARD_COLORS,
  getCardPaletteColors
} from '../../../src/computation-views/encoding/card-tokens';
import {
  createDivergingScaleSpec,
  ScaleRegistry
} from '../../../src/computation-views/scales/scale-registry';
import {
  makeY2Points,
  makeY2Arrays,
  makeY2Frames,
  makeY2Capture,
  makeY2Record
} from '../fixtures/y2-card-records';
import {makeV8Capture, makeV8ResultRecord} from '../fixtures/v8-result-record';

function render(kind: Parameters<typeof makeV8ResultRecord>[0] = 'polygons', options = {}) {
  const context = createViewContext(makeV8ResultRecord(kind), {idPrefix: 'v8', ...options});
  const model = buildResultViewModel(context);
  const tree = renderResultView(model, context);
  return {context, model, tree, html: viewNodeToHtml(tree)};
}

describe('result view', () => {
  test('requires an explicitly tagged output', () => {
    const record = makeV8ResultRecord();
    expect(RESULT_VIEW.satisfies(createViewContext(record))).toEqual({ok: true});
    for (const capture of record.captures!.snapshots) capture.role = undefined;
    expect(RESULT_VIEW.satisfies(createViewContext(record))).toEqual({
      ok: false,
      reason:
        'no captures tagged role: output (set capture.role, and capture.geometry for spatial results, when recording)'
    });
    expect(RESULT_VIEW.signature).toEqual({
      binds: [],
      free: ['entity'],
      layout: {entity: 'position'}
    });
  });
  test('shares extent and projection, counts features and describes output provenance', () => {
    const {tree, model, context, html} = render('polygons', {options: {layout: 'side-by-side'}});
    const figures = findViewNodes(tree, node => node.tag === 'figure');
    expect(figures.map(figure => figure.attributes['data-role'])).toEqual(['input', 'output']);
    const drawings = findViewNodes(
      tree,
      node => node.tag === 'svg' && Boolean(node.attributes['data-extent'])
    );
    expect(new Set(drawings.map(drawing => drawing.attributes['data-extent'])).size).toBe(1);
    expect(drawings.map(drawing => drawing.attributes['width'])).toEqual([312, 312]);
    expect(model.panels.map(panel => panel.count)).toEqual([6, 2]);
    expect(RESULT_VIEW.describe(model, context)).toMatchObject({
      subject: 'Result of gpu-core.geometry: input → polygons',
      keyNumber: {label: 'polygons', value: 2},
      provenance: 'readback'
    });
    expect(html).toContain('fill-rule="evenodd"');
    expect(html).toContain('data-action="pick" data-axis="entity"');
    expect(html).toContain('data-provenance="readback"');
    expect(html).toContain('cv-legend');
    // The adapter is stated once by the host caption and in describe(), not inside the figure.
    expect(RESULT_VIEW.describe(model, context).adapterLabel).toContain('Fixture adapter');
    expect(html).not.toContain('<td></td>');
  });
  test('uses final iterations and resolves capture ids before names', () => {
    const record = makeV8ResultRecord();
    const output = record.captures!.snapshots[1];
    record.captures = {
      ...record.captures!,
      snapshots: [
        ...record.captures!.snapshots,
        {...output, id: 'earlier', iteration: 1, values: []},
        makeV8Capture('offsets-id', [0, 1])
      ]
    };
    const model = buildResultViewModel(createViewContext(record));
    expect(model.output.capture.id).toBe('polygons-id');
    expect(model.output.count).toBe(2);
  });
  test('overlay keeps full input role styling below the output', () => {
    const {tree, model} = render('polygons', {options: {layout: 'overlay'}});
    expect(model.width).toBe(640);
    const input = findViewNodes(
      tree,
      node => node.tag === 'g' && node.attributes['data-role'] === 'input'
    )[0];
    expect(
      findViewNodes(input, node => node.tag === 'g' && node.attributes['opacity'] === 0.2)
    ).toHaveLength(0);
    expect(
      findViewNodes(input, node => node.tag === 'circle').every(
        node => node.attributes['fill'] === CARD_COLORS.light['input']
      )
    ).toBe(true);
  });
  test('unvalued geometry uses role colours and pins use both selection inks', () => {
    const {tree, html} = render('polygons', {bindings: {entity: 1}});
    const ink = getSelectionInk('light');
    expect(html).toContain(`fill="${CARD_COLORS.light['input']}"`);
    for (const [state, color] of [
      ['selection', ink.core],
      ['selection-halo', ink.halo]
    ]) {
      const selected = findViewNodes(tree, node => node.attributes['data-state'] === state);
      expect(selected).toHaveLength(2);
      expect(selected.every(node => node.attributes['stroke'] === color)).toBe(true);
    }
  });
  test('missing path offsets produce unknown and no polygon; lines remain open', () => {
    const record = makeV8ResultRecord();
    record.captures = {
      ...record.captures!,
      snapshots: record.captures!.snapshots.filter(capture => capture.name !== 'offsets')
    };
    let context = createViewContext(record);
    let tree = renderResultView(buildResultViewModel(context), context);
    expect(viewNodeToHtml(tree)).toContain('unknown — offsets capture missing');
    expect(findViewNodes(tree, node => node.tag === 'path')).toHaveLength(0);
    const lines = makeV8ResultRecord();
    lines.captures!.snapshots[1].geometry = {
      kind: 'lines',
      coordinateModel: 'geographic',
      offsetsCapture: 'offsets'
    };
    context = createViewContext(lines);
    tree = renderResultView(buildResultViewModel(context), context);
    const paths = findViewNodes(tree, node => node.tag === 'path');
    expect(paths).toHaveLength(2);
    expect(
      paths.every(
        node => node.attributes['fill'] === 'none' && !String(node.attributes['d']).includes('Z')
      )
    ).toBe(true);
  });
  test('raster averages finite values in square blocks, preserves support and selects containing block', () => {
    const record = makeV8ResultRecord('raster');
    record.captures!.snapshots[0].values = Array.from({length: 20000}, (_, index) =>
      index === 0 ? NaN : index === 1 ? Infinity : index
    );
    const context = createViewContext(record, {bindings: {entity: 201}});
    const model = buildResultViewModel(context);
    const tree = renderResultView(model, context);
    const cells = findViewNodes(
      tree,
      node => node.tag === 'rect' && node.attributes['data-action'] === 'pick'
    );
    expect(cells).toHaveLength(1); // Only the pinned cell remains a vector overlay.
    const image = findViewNodes(tree, node => node.tag === 'image')[0];
    const decoded = decodeRasterImage(image);
    expect([decoded.width, decoded.height]).toEqual([160, 80]);
    expect(decoded.palette.slice(0, 3)).toEqual([217, 217, 217]); // First sample is NaN.
    expect(cells[0].attributes['data-key']).toBe(201);
    expect(RESULT_VIEW.describe(model, context).facts?.join(' ')).toContain(
      'downsampled from 200×100'
    );
    expect(viewNodeToHtml(tree)).toContain('cell size 10 m');
    expect(findViewNodes(tree, node => node.attributes['data-state'] === 'selection')).toHaveLength(
      1
    );
    const thumbnail = render('raster', {width: 160, options: {size: 'thumbnail'}});
    expect(thumbnail.html).toContain('downsampled from 200×100');
    expect(thumbnail.model.output.blockSize).toBe(4);
    expect(thumbnail.model.height).toBe(96);
  });
  test('non-finite raster blocks use the no-data colour', () => {
    const record = makeV8ResultRecord('raster');
    record.captures!.snapshots[0].values = Array.from({length: 20000}, () => NaN);
    const context = createViewContext(record);
    const tree = renderResultView(buildResultViewModel(context), context);
    const image = findViewNodes(tree, node => node.tag === 'image')[0];
    expect(decodeRasterImage(image).palette).toEqual([217, 217, 217]);
    expect(new Set(decodeRasterImage(image).pixels)).toEqual(new Set([0]));
  });
  test('graphs use bounded adjacency tables without a layout and recorded coordinates with one', () => {
    const table = render('graph');
    expect(table.html).toContain(
      'no layout recorded: adjacency table instead of a node-link drawing'
    );
    expect(table.html).toContain('8 more rows');
    expect(findViewNodes(table.tree, node => node.tag === 'circle')).toHaveLength(0);
    expect(findViewNodes(table.tree, node => node.tag === 'tr')).toHaveLength(33);
    const thumbnail = render('graph', {options: {size: 'thumbnail'}, width: 160});
    expect(thumbnail.html).not.toContain('more rows');
    expect(thumbnail.html).not.toContain('<text');
    const layout = render('layout');
    expect(findViewNodes(layout.tree, node => node.tag === 'circle')).toHaveLength(40);
    expect(
      findViewNodes(
        layout.tree,
        node => node.tag === 'line' && node.attributes['data-action'] === 'pick'
      )
    ).toHaveLength(40);
    expect(RESULT_VIEW.describe(layout.model, layout.context).keyNumber).toEqual({
      label: 'nodes',
      value: 40
    });
  });
  test('caps graph edges at 2000 and declares omitted edges', () => {
    const record = makeV8ResultRecord('layout');
    record.captures!.snapshots[1].values = [0, 2003];
    record.captures!.snapshots[2].values = Array.from({length: 2003}, () => 1);
    const context = createViewContext(record);
    const tree = renderResultView(buildResultViewModel(context), context);
    expect(
      findViewNodes(tree, node => node.tag === 'line' && node.attributes['data-action'] === 'pick')
    ).toHaveLength(183);
    expect(getViewNodeText(tree)).toContain('stride');
  });
  test('large point and path captures stay within total figure budgets and disclose stride sampling', () => {
    for (const kind of ['points', 'lines', 'polygons'] as const) {
      const coordinates = Array.from({length: 12000}, (_, index) => index % 97);
      const record = makeY2Record([
        makeY2Capture('geometry', coordinates, {
          role: 'output',
          components: 2,
          geometry:
            kind === 'points'
              ? {kind, coordinateModel: 'projected'}
              : {kind, coordinateModel: 'projected', offsetsCapture: 'offsets'}
        }),
        makeY2Capture('offsets', [0, 3000, 6000])
      ]);
      for (const thumbnail of [false, true]) {
        const context = createViewContext(record, {options: thumbnail ? {size: 'thumbnail'} : {}});
        const model = buildResultViewModel(context);
        const tree = renderResultView(model, context);
        const vertices =
          findViewNodes(tree, node => node.tag === 'circle').length +
          findViewNodes(tree, node => node.tag === 'path').reduce(
            (count, path) => count + (String(path.attributes['d']).match(/[ML]/g)?.length ?? 0),
            0
          );
        expect(vertices).toBeGreaterThan(0);
        expect(vertices).toBeLessThanOrEqual(thumbnail ? 400 : 4000);
        expect(RESULT_VIEW.describe(model, context).facts?.join(' ')).toContain(
          'stride decimation'
        );
      }
    }
  });
  test('thumbnails reduce point radius and renders remain deterministic', () => {
    const full = render();
    const thumbnail = render('layout', {width: 160, options: {size: 'thumbnail'}});
    expect(
      Number(findViewNodes(thumbnail.tree, node => node.tag === 'circle')[0].attributes['r'])
    ).toBe(3 * 0.6);
    expect(Number(findViewNodes(full.tree, node => node.tag === 'circle')[0].attributes['r'])).toBe(
      2.5
    );
    expect(full.html).toBe(
      viewNodeToHtml(renderResultView(buildResultViewModel(full.context), full.context))
    );
  });
});

function renderCard(
  record = makeY2Points(),
  options: Parameters<typeof createViewContext>[1] = {}
) {
  const context = createViewContext(record, {idPrefix: 'y2', ...options});
  const model = buildResultViewModel(context);
  const tree = renderResultView(model, context);
  return {context, model, tree, html: viewNodeToHtml(tree)};
}

function cardMarks(tree: Parameters<typeof findViewNodes>[0], role: string) {
  return findViewNodes(tree, node =>
    String(node.attributes['class'] ?? '')
      .split(' ')
      .includes(`cv-mark--${role}`)
  );
}

describe('card result grammar', () => {
  test('draws context first and role colours as light presentation attributes on either ground', () => {
    for (const ground of ['light', 'dark'] as const) {
      const {tree, model, context, html} = renderCard(makeY2Points(), {ground});
      const drawings = findViewNodes(tree, node => node.attributes['class'] === 'cv-figure');
      expect(drawings).toHaveLength(1);
      expect(drawings[0].attributes).toMatchObject({
        width: 640,
        height: 360,
        viewBox: '0 0 640 360',
        preserveAspectRatio: 'xMidYMid meet'
      });
      const groups = drawings[0].children.filter(
        child => typeof child !== 'string' && child.tag === 'g'
      );
      expect(
        groups.map(group => typeof group !== 'string' && group.attributes['data-role'])
      ).toEqual(['context', 'input', 'output', 'highlight']);
      for (const role of ['context', 'input', 'output'] as const) {
        const marks = cardMarks(tree, role);
        expect(marks.length).toBeGreaterThan(0);
        expect(
          marks.every(
            mark =>
              mark.attributes['fill'] ===
              CARD_COLORS.light[role === 'context' ? 'context-fill' : role]
          )
        ).toBe(true);
        if (role === 'context')
          expect(
            marks.every(
              mark => mark.attributes['aria-hidden'] === 'true' && !mark.attributes['data-action']
            )
          ).toBe(true);
      }
      expect(RESULT_VIEW.describe(model, context).subject).not.toContain('context');
      expect(RESULT_VIEW.describe(model, context).facts.join(' ')).not.toContain('input context');
      expect(html).toContain('grey: input input · blue: output output');
      expect(html).not.toContain('position: absolute');
    }
    // The pre-schema `meaning: 'context'` form is no longer read; inputs stay inputs.
    const legacy = makeY2Points();
    legacy.captures!.snapshots[0].role = 'input';
    legacy.captures!.snapshots[0].meaning = 'context';
    expect(cardMarks(renderCard(legacy).tree, 'context')).toHaveLength(0);
  });
  test('uses same-support side-by-side panels and keeps only output and context in thumbnails', () => {
    const record = makeY2Points([1, 3, 8]);
    const full = renderCard(record);
    expect(full.model.layout).toBe('side-by-side');
    expect(full.html).toContain('class="cv-panels"');
    const panels = findViewNodes(
      full.tree,
      node => node.attributes['class'] === 'cv-figure cv-panel'
    );
    expect(panels).toHaveLength(2);
    expect(panels.every(panel => panel.attributes['viewBox'] === '0 0 312 360')).toBe(true);
    expect(new Set(panels.map(panel => panel.attributes['data-extent'])).size).toBe(1);
    expect(cardMarks(full.tree, 'context')).toHaveLength(4);
    expect(full.html).toContain('class="cv-panel-caption">input: input');
    const thumbnail = renderCard(record, {options: {size: 'thumbnail'}});
    expect(thumbnail.html).toContain('viewBox="0 0 160 96"');
    expect(cardMarks(thumbnail.tree, 'input')).toHaveLength(0);
    expect(cardMarks(thumbnail.tree, 'context')).toHaveLength(2);
    expect(findViewNodes(thumbnail.tree, node => node.tag === 'svg')).toHaveLength(1);
    for (const forbidden of ['<text', 'cv-legend', 'cv-note', '<figcaption'])
      expect(thumbnail.html).not.toContain(forbidden);
    record.captures!.snapshots[1].values = [0, 0];
    expect(renderCard(record).model.layout).toBe('overlay');
    expect(renderCard(makeY2Points()).model.layout).toBe('overlay');
  });
  test('classifies polygons with the shared scale and preserves readback and replay layering', () => {
    const record = makeV8ResultRecord();
    const rendered = renderCard(record);
    const paths = cardMarks(rendered.tree, 'output');
    expect(paths).toHaveLength(2);
    paths.forEach((path, index) => {
      const classIndex = resolveValueScale(rendered.context, 'values')!.scale.classFor(
        [1, 5][index]
      );
      expect(path.attributes['class']).toContain(`cv-v cv-seq-${classIndex + 1}`);
      expect(path.attributes['fill']).toBe(getCardPaletteColors('seq', 'light')[classIndex]);
      expect(path.attributes['data-provenance']).toBe('readback');
    });
    record.captures!.snapshots[3].provenance = 'gpu-replay';
    const replay = renderCard(record);
    expect(
      cardMarks(replay.tree, 'output').every(
        mark => !String(mark.attributes['fill']).startsWith('url')
      )
    ).toBe(true);
    const hatch = findViewNodes(
      replay.tree,
      node => node.tag === 'path' && node.attributes['opacity'] === 0.55
    );
    expect(hatch).toHaveLength(2);
    expect(hatch.every(mark => mark.attributes['fill'] === 'url(#y2-hatched)')).toBe(true);
  });
  test('nominal classes follow first appearance, with the five largest categories only', () => {
    const packs = [{id: 'categories', variableKinds: {values: {level: 'nominal' as const}}}];
    const small = renderCard(makeY2Points([9, 4, 9, 7]), {packs});
    expect(cardMarks(small.tree, 'output').map(mark => mark.attributes['class'])).toEqual([
      'cv-mark cv-mark--output cv-point cv-v cv-nom-1',
      'cv-mark cv-mark--output cv-point cv-v cv-nom-2',
      'cv-mark cv-mark--output cv-point cv-v cv-nom-1',
      'cv-mark cv-mark--output cv-point cv-v cv-nom-3'
    ]);
    const large = renderCard(makeY2Points([1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3]), {packs});
    expect(large.html).toContain('7 categories; 5 largest coloured');
    expect(large.model.output.categories).toEqual([3, 4, 5, 6, 7]);
    expect(cardMarks(large.tree, 'output')).toHaveLength(10);
    expect(large.html).not.toContain('cv-nom-6');
  });
  test('declared highlight is drawn above data, with a full-size label, unless the palette diverges', () => {
    const record = makeY2Points([-2, 0, 2]);
    record.captures!.snapshots[2].highlight = {key: 1, label: 'Chosen subject'};
    const full = renderCard(record, {options: {layout: 'overlay'}});
    expect(cardMarks(full.tree, 'highlight')).toHaveLength(1);
    expect(cardMarks(full.tree, 'highlight')[0].attributes['r']).toBe('4.5');
    expect(full.html).toContain('class="cv-label cv-label--subject"');
    expect(full.html).toContain('Chosen subject');
    const thumbnail = renderCard(record, {options: {size: 'thumbnail'}});
    expect(cardMarks(thumbnail.tree, 'highlight')).toHaveLength(1);
    expect(thumbnail.html).not.toContain('<text');
    const scales = new ScaleRegistry({runId: record.runId});
    scales.register(
      createDivergingScaleSpec({
        id: 'signed',
        variable: 'values',
        variableKind: {level: 'interval'},
        midpoint: 0,
        ground: 'light',
        noDataColor: '#d9d9d9',
        finalStateValues: [-2, 0, 2]
      })
    );
    const diverging = renderCard(record, {scales});
    expect(cardMarks(diverging.tree, 'highlight')).toHaveLength(0);
    expect(cardMarks(diverging.tree, 'output')).toHaveLength(3);
    expect(diverging.html).toContain('cv-div-');
    expect(diverging.html).not.toContain('Chosen subject');
  });
  test('raster context forces overlay and full-size known support gets a scale tick', () => {
    const raster = makeY2Frames(1).captures!.snapshots[0];
    const record = makeY2Record([
      makeY2Capture('hillshade', [1, 2, 3, 4], {role: 'context', geometry: raster.geometry}),
      makeY2Capture('elevation', [1, 2, 3, 4], {role: 'input', geometry: raster.geometry}),
      raster
    ]);
    const full = renderCard(record);
    expect(full.model.layout).toBe('overlay');
    expect(full.html).toContain('30 m</text>');
    expect(full.html).toContain('cell size 30 m');
    const withoutContext = makeY2Record(record.captures!.snapshots.slice(1));
    expect(renderCard(withoutContext).model.layout).toBe('side-by-side');
  });
  test('arrays use bounded labelled strips and pure-picture thumbnails', () => {
    const record = makeY2Arrays();
    const full = renderCard(record);
    expect(full.model.grammar).toBe('strips');
    expect(full.model.height).toBe(100);
    expect(full.html).toContain('20 values, 16 shown');
    expect(full.html).toContain('>… 4 more …</text>');
    expect(full.html).toContain('class="cv-cell-text"');
    expect(cardMarks(full.tree, 'output')).toHaveLength(16);
    expect(RESULT_VIEW.describe(full.model, full.context).keyNumber).toEqual({
      label: 'values',
      value: 20
    });
    const thumbnail = renderCard(record, {options: {size: 'thumbnail'}});
    expect(cardMarks(thumbnail.tree, 'input')).toHaveLength(8);
    expect(cardMarks(thumbnail.tree, 'output')).toHaveLength(8);
    expect(thumbnail.html).not.toContain('<text');
    expect(thumbnail.html).not.toContain('cv-note');
  });
  test('shared keys or column meanings produce a six-row table with output columns and missing cells', () => {
    const record = makeY2Arrays(true);
    const full = renderCard(record);
    expect(full.html).toContain('<table class="cv-table">');
    expect(full.html).toContain('<th class="cv-col--output"');
    expect(full.html).toContain('<td class="cv-col--output"');
    expect(full.html).toContain('>—</td>');
    expect(full.html).toContain('… 14 more rows');
    expect(full.html).not.toContain('<td></td>');
    expect(RESULT_VIEW.describe(full.model, full.context).keyNumber).toEqual({
      label: 'rows',
      value: 4
    });
    const thumbnail = renderCard(record, {options: {size: 'thumbnail'}});
    expect(thumbnail.html).not.toContain('<table');
    expect(thumbnail.html).not.toContain('<text');
    expect(cardMarks(thumbnail.tree, 'output')).toHaveLength(6);
    expect(
      cardMarks(thumbnail.tree, 'output').every(
        mark => mark.attributes['fill'] === CARD_COLORS.light['output']
      )
    ).toBe(true);
    record.captures!.snapshots.forEach(capture => {
      capture.keySpace = undefined;
      capture.meaning = 'column value';
    });
    expect(renderCard(record).model.grammar).toBe('table');
  });
  test('records frames with only the final visible and a bounded static evidence strip', () => {
    const record = makeY2Frames();
    const full = renderCard(record);
    const frames = findViewNodes(full.tree, node => node.attributes['class'] === 'cv-frame');
    expect(frames).toHaveLength(3);
    expect(frames.map(frame => frame.attributes['hidden'])).toEqual([true, true, false]);
    expect(frames.map(frame => frame.attributes['data-cv-frame'])).toEqual([0, 1, 2]);
    expect(full.html).toContain('data-cv-play');
    expect(full.html).toContain('class="cv-key-frames"');
    expect(full.html).toContain('round 0');
    expect(full.html).toContain('cv-time-');
    expect(full.html).toBe(renderCard(record).html);
    expect(renderCard(record, {reducedMotion: true}).html).toContain('Step ▸');
    expect(renderCard(makeY2Frames(1)).html).not.toContain('cv-frame');
    const identical = makeY2Frames();
    identical.captures!.snapshots.forEach(capture => {
      capture.values = [1, 2, 3, 4];
    });
    expect(renderCard(identical).html).not.toContain('data-cv-play');
    const long = renderCard(makeY2Frames(20));
    expect(long.model.frames).toHaveLength(8);
    expect(long.model.frames.at(-1)!.capture.iteration).toBe(19);
    expect(findViewNodes(long.tree, node => node.tag === 'figure')).toHaveLength(6);
    expect(renderCard(record, {options: {size: 'thumbnail'}}).html).not.toContain('data-cv-play');
  });
});

describe('card evidence edge cases', () => {
  test('classed no-data marks get a swatch only when a non-finite value is actually drawn', () => {
    const record = makeY2Points([1, NaN, Infinity]);
    const full = renderCard(record);
    expect(
      cardMarks(full.tree, 'output').filter(mark =>
        String(mark.attributes['class']).includes('cv-no-data')
      )
    ).toHaveLength(2);
    expect(full.html).toContain('No data');
    const arrays = makeY2Arrays();
    arrays.captures!.snapshots[1].values = Array.from(
      arrays.captures!.snapshots[1].values,
      (value, index) => (index === 10 ? NaN : value)
    );
    const omitted = renderCard(arrays);
    expect(omitted.html).not.toContain('No data');
    expect(
      cardMarks(omitted.tree, 'input').every(
        mark => mark.attributes['fill'] === CARD_COLORS.light['input']
      )
    ).toBe(true);
  });
  test('input graph edges are unpickable context and output node values use classes', () => {
    const record = makeV8ResultRecord('layout');
    const graph = record.captures!.snapshots[0];
    const input = {...graph, id: 'input-graph-id', name: 'input-graph', role: 'input' as const};
    graph.values = Array.from({length: 40}, (_, index) => index);
    record.captures = {...record.captures!, snapshots: [input, ...record.captures!.snapshots]};
    const full = renderCard(record, {options: {layout: 'overlay'}});
    const edges = cardMarks(full.tree, 'context').filter(mark => mark.tag === 'line');
    expect(edges).toHaveLength(40);
    expect(edges.every(mark => !mark.attributes['data-action'])).toBe(true);
    expect(
      cardMarks(full.tree, 'output')
        .filter(mark => mark.tag === 'circle')
        .every(mark => String(mark.attributes['class']).includes('cv-seq-'))
    ).toBe(true);
  });
  test('raster origin and row direction place cells under vector inputs', () => {
    const geometry = {
      kind: 'raster' as const,
      shape: [2, 2] as [number, number],
      cellSize: 1,
      unit: 'm',
      coordinateModel: 'projected' as const,
      origin: [100, 202] as [number, number]
    };
    const point = makeV8Capture('station', [100.5, 201.5], {
      components: 2,
      role: 'input',
      geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
    });
    const cellRectangles = (rowDirection: 'south' | 'north') => {
      const record = makeV8ResultRecord('raster');
      record.captures!.snapshots = [
        point,
        makeV8Capture('surface', [1, 2, 3, 4], {
          role: 'output',
          geometry: {
            ...geometry,
            ...(rowDirection === 'north' ? {origin: [100, 200], rowDirection} : {})
          }
        })
      ];
      const context = createViewContext(record, {idPrefix: 'placement'});
      const placed = buildResultViewModel(context);
      const drawn = renderResultView(placed, context);
      const cells = findViewNodes(drawn, node => node.tag === 'image');
      return {placed, cells, html: viewNodeToHtml(drawn)};
    };
    for (const rowDirection of ['south', 'north'] as const) {
      const {placed, cells, html} = cellRectangles(rowDirection);
      expect(html).not.toContain('origin not recorded');
      const [stationX, stationY] = placed.projection.project(100.5, 201.5);
      const image = cells[0];
      expect(stationX).toBeGreaterThan(Number(image.attributes['x']));
      expect(stationY).toBeGreaterThan(Number(image.attributes['y']));
      expect(stationX).toBeLessThan(
        Number(image.attributes['x']) + Number(image.attributes['width'])
      );
      const decoded = decodeRasterImage(image);
      expect(decoded.pixels[0]).toBe(rowDirection === 'south' ? 0 : 2);
    }
  });
  test('graph valuesCapture colours nodes on the run scale', () => {
    const record = makeV8ResultRecord('layout');
    const graph = record.captures!.snapshots[0];
    graph.geometry = {...graph.geometry!, valuesCapture: 'rank'} as typeof graph.geometry;
    record.captures = {
      ...record.captures!,
      snapshots: [
        ...record.captures!.snapshots,
        makeV8Capture(
          'rank',
          Array.from({length: 40}, (_, index) => index)
        )
      ]
    };
    const full = renderCard(record);
    expect(full.model.output.values?.name).toBe('rank');
    const nodes = cardMarks(full.tree, 'output').filter(mark => mark.tag === 'circle');
    expect(nodes).toHaveLength(40);
    expect(nodes.every(mark => String(mark.attributes['class']).includes('cv-seq-'))).toBe(true);
  });
  test('array snapshots retain the final state, frame evidence and explicit highlights', () => {
    const snapshots = [0, 1, 2].map(iteration =>
      makeY2Capture('output', [iteration, 2, 3], {
        id: `output-${iteration}`,
        role: 'output',
        iteration,
        highlight: {key: 0, label: 'answer'}
      })
    );
    const full = renderCard(makeY2Record(snapshots));
    expect(findViewNodes(full.tree, node => node.attributes['class'] === 'cv-frame')).toHaveLength(
      3
    );
    expect(full.html).toContain('data-cv-play');
    expect(full.html).toContain('cv-label--subject');
    const picture = renderCard(makeY2Record(snapshots), {options: {size: 'thumbnail'}});
    expect(cardMarks(picture.tree, 'highlight')).toHaveLength(1);
    expect(picture.html).not.toContain('<text');
  });
});
