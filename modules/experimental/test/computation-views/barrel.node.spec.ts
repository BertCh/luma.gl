// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';

test('computation-views and computation-lenses import in Node without a DOM', async () => {
  expect(typeof (globalThis as {document?: unknown}).document).toBe('undefined');
  const views = await import('@luma.gl/experimental/computation-views');
  const lenses = await import('@luma.gl/experimental/computation-lenses');
  for (const name of [
    'buildLensTwin',
    'renderLensTwinHtml',
    'generateLensDescription',
    'createDefaultViewRegistry',
    'registerDefaultViewPack',
    'ViewRegistry',
    'LadderCoordinator',
    'getPageCoordinator',
    'ScaleRegistry',
    'validateViewEncodings',
    'mountLadderLens',
    'defineLadderLensElement',
    'LensController',
    'encodeLensUrlState',
    'navigateLadder',
    'PageScheduler',
    'getAnalysisDevice',
    'mountFormulaCard',
    'mountDatumReceipt',
    'mountBufferStrip',
    'mountDispatchView',
    'mountSkeletonView',
    'mountRealizedView',
    'mountCommandGraphView',
    'mountTimingView',
    'mountCompareView',
    'mountScopeDiagram'
  ]) {
    expect(typeof (views as Record<string, unknown>)[name], name).toBe('function');
  }
  for (const name of ['registerLensScenario', 'fromRecordedLensFile', 'toRecordedLensFile']) {
    expect(typeof (lenses as Record<string, unknown>)[name], name).toBe('function');
  }
  const root = await import('@luma.gl/experimental');
  expect('buildLensTwin' in root).toBe(false);
});
