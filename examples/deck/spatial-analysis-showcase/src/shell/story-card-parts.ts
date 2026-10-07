// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OptionValue, PipelineStage} from '../scenes/scene';
import {h} from './dom';

const COUNT_FORMAT = new Intl.NumberFormat('en-US', {
  notation: 'compact',
  maximumFractionDigits: 1
});

/** What a scene reports through `ctx.setCost`. */
export type StepCost = {records?: number; passes?: number; note?: string};

/** What the shell adds to the scene's cost. */
export type EngineCost = {
  /** Compiled-graph rebuilds since the scene opened. */
  rebuilds: number;
  /** Cached GPU time of all graphs once, `null` before it was measured. */
  gpuMilliseconds: number | null;
  /** A measurement is running. */
  measuring: boolean;
};

/**
 * The cost chip row under the step headline: `records · passes · GPU ms · rebuilds`. Records and
 * passes come from the scene (`ctx.setCost`), rebuilds from the host; the GPU chip is a button
 * that measures once on demand and keeps the number. The rebuild chip pulses when it increments
 * (two alternating keyframe names restart the animation without a layout read).
 */
export class CostRow {
  /** The `<ul>` to place in the card. */
  readonly element: HTMLElement;
  private readonly records = h('li', {class: 'cost-chip', hidden: true});
  private readonly passes = h('li', {class: 'cost-chip', hidden: true});
  private readonly gpuButton: HTMLButtonElement;
  private readonly rebuilds = h('li', {class: 'cost-chip'});
  private lastRebuilds: number | null = null;
  private pulseToggle = false;

  /** @param onMeasure Called when the GPU chip is pressed. */
  constructor(onMeasure: () => void) {
    this.gpuButton = h('button', {
      type: 'button',
      class: 'cost-gpu',
      title: 'Measure the GPU time of every graph once',
      on: {click: onMeasure}
    });
    this.element = h(
      'ul',
      {class: 'cost-row', 'aria-label': 'Cost of this step'},
      this.records,
      this.passes,
      h('li', {class: 'cost-chip'}, this.gpuButton),
      this.rebuilds
    );
  }

  /** Rewrites the chips. Cheap: text is assigned only when it changed. */
  update(cost: StepCost | null, engine: EngineCost): void {
    this.setChip(this.records, cost?.records, value => `${COUNT_FORMAT.format(value)} records`);
    this.setChip(this.passes, cost?.passes, value => `${value} ${value === 1 ? 'pass' : 'passes'}`);
    if (cost?.note) this.element.title = cost.note;
    else this.element.removeAttribute('title');
    const gpuText = engine.measuring
      ? 'Measuring…'
      : engine.gpuMilliseconds === null
        ? 'GPU ms: measure'
        : `${engine.gpuMilliseconds.toFixed(2)} ms GPU`;
    if (this.gpuButton.textContent !== gpuText) this.gpuButton.textContent = gpuText;
    this.gpuButton.disabled = engine.measuring;
    const rebuildText = `${engine.rebuilds} ${engine.rebuilds === 1 ? 'rebuild' : 'rebuilds'}`;
    if (this.rebuilds.textContent !== rebuildText) this.rebuilds.textContent = rebuildText;
    if (this.lastRebuilds !== null && engine.rebuilds > this.lastRebuilds) {
      this.pulseToggle = !this.pulseToggle;
      this.rebuilds.classList.remove('is-pulse-a', 'is-pulse-b');
      this.rebuilds.classList.add(this.pulseToggle ? 'is-pulse-a' : 'is-pulse-b');
    }
    this.lastRebuilds = engine.rebuilds;
  }

  private setChip(
    chip: HTMLElement,
    value: number | undefined,
    format: (value: number) => string
  ): void {
    const visible = value !== undefined && Number.isFinite(value);
    chip.hidden = !visible;
    if (!visible) return;
    const text = format(value);
    if (chip.textContent !== text) chip.textContent = text;
  }
}

/**
 * The pipeline strip: one chip per stage of the computation, the step's `stage` highlighted with
 * its detail line under the strip. A stage with `show` is a button that writes that option, so
 * the map shows the intermediate product.
 */
export function createPipelineStrip(
  stages: readonly PipelineStage[],
  activeStageId: string | undefined,
  onShow: (option: string, value: OptionValue) => void
): HTMLElement {
  const active = stages.find(stage => stage.id === activeStageId);
  const chips = stages.map((stage, index) => {
    const isActive = stage.id === activeStageId;
    const label = [h('span', {class: 'pipeline-index'}, String(index + 1)), stage.label];
    const attributes = {
      class: `pipeline-chip${isActive ? ' is-active' : ''}`,
      title: stage.detail ?? '',
      'aria-current': isActive ? 'step' : undefined
    };
    const chip = stage.show
      ? h('button', {...attributes, type: 'button', 'data-show': ''}, label)
      : h('span', attributes, label);
    const {show} = stage;
    if (show) chip.addEventListener('click', () => onShow(show.option, show.value));
    return h('li', {class: 'pipeline-item'}, chip);
  });
  return h(
    'div',
    {class: 'pipeline'},
    h('ol', {class: 'pipeline-strip', 'aria-label': 'Computation stages'}, chips),
    active?.detail ? h('p', {class: 'pipeline-detail'}, active.detail) : null
  );
}
