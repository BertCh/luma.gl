// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadProgress} from '../data/catalog';
import {formatBytes, h} from './dom';

/** Phases of opening a story: download, GPU graph compile, first frame. */
export type LoadingPhase = 'fetch' | 'compile' | 'frame';

const FADE_MILLISECONDS = 220;

type DatasetRow = {
  name: HTMLElement;
  bar: HTMLElement;
  size: HTMLElement;
  row: HTMLElement;
};

/**
 * The three-phase loading cover of the map slot. Fetch: one row per dataset with a 2 px progress
 * bar and the megabytes received; compile: a shimmer over the map ground with the compile note;
 * first frame: a 220 ms cross-fade when {@link LoadingOverlay.finish} is called. Append
 * {@link LoadingOverlay.element} to the map slot; it ignores the pointer.
 */
export class LoadingOverlay {
  /** Cover element. */
  readonly element: HTMLElement;
  private readonly rows = new Map<string, DatasetRow>();
  private readonly list: HTMLElement;
  private readonly message: HTMLElement;
  private finished = false;

  constructor() {
    this.list = h('ul', {class: 'map-loading-datasets'});
    this.message = h('p', {class: 'map-loading-message', role: 'status'});
    this.element = h(
      'div',
      {class: 'map-loading', dataset: {phase: 'fetch'}},
      h(
        'div',
        {class: 'map-loading-card'},
        h('p', {class: 'map-loading-title'}, 'Opening story'),
        this.list,
        this.message
      )
    );
  }

  /** Lists the datasets that are about to download. */
  setDatasets(datasets: readonly {id: string; title: string}[]): void {
    this.rows.clear();
    this.list.replaceChildren();
    for (const dataset of datasets) {
      const bar = h('span', {class: 'map-loading-bar-fill'});
      const size = h('span', {class: 'map-loading-size'}, '…');
      const name = h('span', {class: 'map-loading-name'}, dataset.title);
      const row = h(
        'li',
        {class: 'map-loading-row'},
        name,
        size,
        h('span', {class: 'map-loading-bar', 'aria-hidden': 'true'}, bar)
      );
      this.rows.set(dataset.id, {name, bar, size, row});
      this.list.append(row);
    }
  }

  /** Replaces the label of one dataset row (the catalog title arrives after the row exists). */
  setTitle(datasetId: string, title: string): void {
    const row = this.rows.get(datasetId);
    if (row) row.name.textContent = title;
  }

  /** Updates one dataset's bar. Unknown totals show the received megabytes only. */
  setProgress(datasetId: string, progress: LoadProgress): void {
    const row = this.rows.get(datasetId);
    if (!row) return;
    row.bar.style.transform = `scaleX(${progress.fraction ?? 0})`;
    row.size.textContent =
      progress.totalBytes && progress.totalBytes > 0
        ? `${formatBytes(progress.loadedBytes)} of ${formatBytes(progress.totalBytes)}`
        : formatBytes(progress.loadedBytes);
  }

  /** Marks one dataset as done. */
  setDatasetDone(datasetId: string): void {
    const row = this.rows.get(datasetId);
    if (!row) return;
    row.bar.style.transform = 'scaleX(1)';
    row.row.classList.add('is-done');
  }

  /** Switches the phase: `compile` shows the shimmer and the compile note. */
  setPhase(phase: LoadingPhase): void {
    if (this.finished) return;
    this.element.dataset['phase'] = phase;
    if (phase === 'compile') this.message.textContent = 'Compiling GPU graph (first run 1-2 s)';
    else if (phase === 'fetch') this.message.textContent = '';
  }

  /** Shows a scene status line (`ctx.setStatus`) or an error in the card. */
  setMessage(text: string, isError = false): void {
    if (this.finished) return;
    this.message.textContent = text;
    this.element.classList.toggle('is-error', isError);
    if (isError) this.element.dataset['phase'] = 'error';
  }

  /** Cross-fades the cover out (the first frame is on screen) and removes it. */
  finish(): void {
    if (this.finished) return;
    this.finished = true;
    this.element.classList.add('is-leaving');
    window.setTimeout(() => this.element.remove(), FADE_MILLISECONDS + 40);
  }

  /** Removes the cover at once. */
  destroy(): void {
    this.finished = true;
    this.element.remove();
  }
}
