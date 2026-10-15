// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import type {LensRecord, ViewContext, ViewRegistration} from '../registry/view-registration';
import {h, viewNodeToDom, type ViewElementNode} from '../view-node';
import {provenancePatternDefs} from '../encoding/provenance';
import {crispnessDefs} from '../encoding/crispness';
import {createDivergingScaleSpec} from '../scales/scale-registry';
import {
  getAdapterLabel,
  getAllCaptures,
  getCaptureProvenance,
  getCaptureValue,
  getPinnedEntity
} from './record-queries';
import {
  getFinalStateCapture,
  getRunScaleRegistry,
  renderValueStrip,
  resolveValueScale
} from './value-strip';

/** Explicit B-minus-A comparison over the shared keys of two references. */
export type CompareViewModel = {
  name: string;
  labelA: string;
  labelB: string;
  mode: 'delta' | 'juxtapose';
  count: number;
  delta: number[];
  changedCount: number;
  maxAbsDelta: number;
  equalCount: number;
  midpoint: 0;
  provenance: 'derived';
};

type References = {
  first: GPUComputationCapture;
  second: GPUComputationCapture;
  labelA: string;
  labelB: string;
};
function chooseReferences(context: ViewContext): References | undefined {
  const requested = context.bindings.buffer ?? context.options?.['capture'];
  const allCaptures = getAllCaptures(context.record);
  const selectedName = allCaptures.find(
    capture => capture.id === requested || capture.name === requested
  )?.name;
  const captures =
    requested === undefined
      ? allCaptures
      : allCaptures.filter(capture => capture.name === selectedName);
  if (context.compare) {
    for (const capture of captures) {
      const first = getFinalStateCapture(context.record, capture.name)!;
      const second = getFinalStateCapture(context.compare.record, capture.name);
      if (second)
        return {
          first,
          second,
          labelA: context.compare.label,
          labelB: context.compare.referenceLabel
        };
    }
  }
  for (const first of captures.filter(capture => capture.provenance === 'gpu-replay')) {
    const second = captures.find(
      capture => capture.name === first.name && capture.provenance === 'gpu-readback'
    );
    if (second) return {first, second, labelA: 'GPU replay', labelB: 'run'};
  }
  for (const name of new Set(captures.map(capture => capture.name))) {
    const iterations = captures
      .filter(capture => capture.name === name && capture.iteration !== undefined)
      .sort((first, second) => first.iteration! - second.iteration!);
    const distinct = iterations.filter(
      (capture, index) =>
        index === iterations.length - 1 || capture.iteration !== iterations[index + 1].iteration
    );
    const first =
      context.options?.['iterationA'] === undefined
        ? distinct[distinct.length - 2]
        : distinct.find(capture => capture.iteration === context.options?.['iterationA']);
    const second =
      context.options?.['iterationB'] === undefined
        ? distinct[distinct.length - 1]
        : distinct.find(capture => capture.iteration === context.options?.['iterationB']);
    if (first && second && first.iteration !== second.iteration)
      return {
        first,
        second,
        labelA: `iteration ${first.iteration}`,
        labelB: `iteration ${second.iteration}`
      };
  }
  return undefined;
}

/** Computes deltas from captured values, without replaying the computation on the CPU. */
export function buildCompareViewModel(context: ViewContext): CompareViewModel {
  const references = chooseReferences(context);
  // Two references are required by satisfies.
  assert(references);
  const {first, second, labelA, labelB} = references;
  const count = Math.min(
    Math.floor(first.values.length / (first.components ?? 1)),
    Math.floor(second.values.length / (second.components ?? 1))
  );
  const delta = Array.from(
    {length: count},
    (_, key) => getCaptureValue(second, key)! - getCaptureValue(first, key)!
  );
  let equalCount = 0;
  let maxAbsDelta = 0;
  for (let key = 0; key < count; key++) {
    if (
      Object.is(getCaptureValue(first, key), getCaptureValue(second, key)) ||
      getCaptureValue(first, key) === getCaptureValue(second, key)
    )
      equalCount++;
    if (!Number.isNaN(delta[key])) maxAbsDelta = Math.max(maxAbsDelta, Math.abs(delta[key]));
  }
  return {
    name: first.name,
    labelA,
    labelB,
    mode: context.options?.['mode'] === 'juxtapose' ? 'juxtapose' : 'delta',
    count,
    delta,
    changedCount: count - equalCount,
    maxAbsDelta,
    equalCount,
    midpoint: 0,
    provenance: 'derived'
  };
}

/** Renders a diverging delta strip or two strips using exactly the same run scale. */
export function renderCompareView(model: CompareViewModel, context: ViewContext): ViewElementNode {
  const references = chooseReferences(context);
  // Rendering follows successful model construction with the same context.
  assert(references);
  const registry = getRunScaleRegistry(context);
  const variable = `delta:${model.name}`;
  let deltaReference = registry.getByVariable(variable);
  if (!deltaReference && model.mode === 'delta') {
    const finite = model.delta.filter(Number.isFinite);
    let minimum = 0;
    let maximum = 0;
    for (const value of finite) {
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
    const extent = Math.max(Math.abs(minimum), Math.abs(maximum));
    const evidence = minimum === 0 || maximum === 0 ? [...finite, -extent, extent] : finite;
    deltaReference = registry.register(
      createDivergingScaleSpec({
        id: `${context.record.runId}:${variable}`,
        variable,
        variableKind: {level: 'interval'},
        ramp: 'rdbu',
        noDataColor: '#808080',
        ground: context.ground,
        finalStateValues: evidence.length ? evidence : [0],
        midpoint: 0
      })
    );
  }
  const sharedScale =
    model.mode === 'juxtapose' ? resolveValueScale(context, model.name)?.scale : undefined;
  const strips =
    model.mode === 'delta'
      ? [
          {
            values: model.delta,
            provenance: model.provenance,
            label: `B − A: ${model.name}`,
            scale: deltaReference ? registry.get(deltaReference) : undefined
          }
        ]
      : [references.first, references.second].map((capture, index) => ({
          values: Array.from({length: model.count}, (_, key) => getCaptureValue(capture, key)!),
          provenance: getCaptureProvenance(capture),
          label: index === 0 ? model.labelA : model.labelB,
          scale: sharedScale
        }));
  return h(
    'section',
    {class: 'lens-view lens-compare-view'},
    h('h3', null, `B − A: ${model.labelB} minus ${model.labelA}`),
    strips.map((strip, index) => {
      const idPrefix = `${context.idPrefix}-${index}`;
      const tree = renderValueStrip({
        ...strip,
        idPrefix,
        ground: context.ground,
        width: context.width,
        pinnedKey: getPinnedEntity(context)
      });
      tree.children.unshift(
        provenancePatternDefs(idPrefix, context.ground),
        crispnessDefs(idPrefix)
      );
      return h('div', null, h('h4', null, strip.label), tree);
    }),
    h('p', null, 'midpoint 0 = unchanged'),
    h(
      'p',
      null,
      `${model.changedCount} keys changed · max |Δ| ${model.maxAbsDelta} · ${model.equalCount} equal`
    ),
    h('p', null, getAdapterLabel(context.record))
  );
}

/** Generic comparison registration with delta provenance. */
export const COMPARE_VIEW: ViewRegistration<CompareViewModel> = {
  id: 'compare-view',
  label: 'Compare',
  rung: 'compare',
  recordType: 'comparison',
  order: 50,
  signature: {binds: [], free: [], layout: {}},
  descriptor: {audience: 'analyst', task: 'explore', interaction: 'medium'},
  level: 'realized',
  grain: 'contributor',
  satisfies: context =>
    chooseReferences(context)
      ? {ok: true}
      : {ok: false, reason: 'needs two references (k vs k′, run A vs B, or GPU vs replay)'},
  build: buildCompareViewModel,
  render: renderCompareView,
  describe: (model, context) => ({
    subject: `Comparison of ${model.name}: ${model.labelB} minus ${model.labelA}`,
    keyNumber: {label: 'keys changed', value: model.changedCount},
    facts: [`max |Δ| ${model.maxAbsDelta}`, `${model.equalCount} equal`],
    provenance: 'derived',
    adapterLabel: getAdapterLabel(context.record)
  })
};

let mountCounter = 0;

/** Mounts this view into a host-owned element; update replaces its tree and destroy clears it. */
export function mountCompareView(
  element: HTMLElement,
  record: LensRecord,
  options: Partial<Omit<ViewContext, 'record'>> = {}
): {update(record: LensRecord): void; destroy(): void} {
  const defaults = {
    bindings: {},
    ground: 'light' as const,
    idPrefix: `lens-compare-view-${++mountCounter}`,
    reducedMotion: false,
    ...options
  };
  const update = (nextRecord: LensRecord) => {
    const context: ViewContext = {...defaults, record: nextRecord};
    const satisfaction = COMPARE_VIEW.satisfies(context);
    const tree = !('reason' in satisfaction)
      ? renderCompareView(buildCompareViewModel(context), context)
      : h('p', {class: 'lens-disabled'}, satisfaction.reason);
    element.replaceChildren(viewNodeToDom(tree, element.ownerDocument));
  };
  update(record);
  return {update, destroy: () => element.replaceChildren()};
}
