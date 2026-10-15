// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Lazily imported `computation-views` and `computation-lenses` modules. */
export type LiveLensModules = {views: any; lenses: any};
/** Directive selectors read back from the `<ladder-lens>` element. */
export type LiveLensAttributes = {
  scenario: string;
  rung: string;
  bind: string;
  datum?: number;
  iteration?: number;
  view?: string;
};

const RECORD_FILES = import.meta.glob<unknown>('../../../../../docs/lens-records/*.json', {
  import: 'default'
});

/** Loads `docs/lens-records/<scenario>.json` lazily; null when no record exists. */
async function loadRecordedLens(scenario: string): Promise<unknown> {
  const key = Object.keys(RECORD_FILES).find(path => path.endsWith(`/${scenario}.json`));
  return key ? RECORD_FILES[key]() : null;
}

/** Renders TeX with the static twins' KaTeX options (MathML for assistive technology). */
async function createRenderTeX(): Promise<
  (tex: string, options: {displayMode: boolean}) => string
> {
  const katex = (await import('katex')).default;
  return (tex, options) =>
    katex.renderToString(tex, {
      displayMode: options.displayMode,
      output: 'htmlAndMathml',
      throwOnError: false,
      strict: 'ignore',
      trust: ({command}: {command: string}) =>
        command === '\\htmlClass' || command === '\\htmlData',
      macros: {'\\term': '\\htmlClass{lens-term}{#1}'}
    });
}

/** Creates the page's one WebGPU analysis device when the reader asks for a live run. */
async function createAnalysisDevice(props: {requiredLimits?: Record<string, number>}) {
  const [{luma}, {webgpuAdapter}] = await Promise.all([
    import('@luma.gl/core'),
    import('@luma.gl/webgpu')
  ]);
  return luma.createDevice({
    type: 'webgpu',
    adapters: [webgpuAdapter],
    requiredLimits: props.requiredLimits
  });
}

/**
 * Upgrades every `<ladder-lens>` on the page to the computation-views element (idempotent), which
 * keeps the static twin until the reader explores the recorded run or runs it on this GPU, and
 * opens the clicked lens on its recorded run when one exists. Resolves to a no-op dispose: the
 * element cleans up when it leaves the page.
 */
export async function mountLiveLens(
  element: HTMLElement,
  attributes: LiveLensAttributes,
  modules: LiveLensModules
): Promise<() => void> {
  const define = modules.views.defineLadderLensElement;
  if (typeof define !== 'function') {
    throw new Error('Live lenses are not available in this build.');
  }
  define({
    loadRecord: async (scenarioId: string) => {
      const json = await loadRecordedLens(scenarioId);
      if (!json) throw new Error(`No recorded run of ${scenarioId} yet.`);
      return json;
    },
    renderTeX: await createRenderTeX(),
    createDevice: createAnalysisDevice
  });
  // The element now owns the controls; the host buttons are no longer needed anywhere on the page.
  for (const button of element.ownerDocument.querySelectorAll<HTMLElement>(
    '.lens-twin__live-button'
  )) {
    button.hidden = true;
  }
  const action = element.getAttribute('recorded') === 'true' ? 'explore' : 'live';
  element.querySelector<HTMLButtonElement>(`[data-ladder="${action}"]`)?.click();
  return () => {};
}
