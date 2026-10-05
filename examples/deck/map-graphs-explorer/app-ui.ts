// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Handle returned by control builders so modes can update a control programmatically. */
export type MapGraphsControlHandle<T> = {
  setValue: (value: T) => void;
  setDisabled: (disabled: boolean) => void;
};

/** Builders a mode uses to populate its own section of the panel. */
export type MapGraphsControlSection = {
  addSlider: (props: {
    label: string;
    min: number;
    max: number;
    step: number;
    value: number;
    format?: (value: number) => string;
    onChange: (value: number) => void;
  }) => MapGraphsControlHandle<number>;
  addSelect: <T extends string>(props: {
    label: string;
    options: readonly {value: T; label: string}[];
    value: T;
    onChange: (value: T) => void;
  }) => MapGraphsControlHandle<T>;
  addToggle: (props: {
    label: string;
    value: boolean;
    onChange: (value: boolean) => void;
  }) => MapGraphsControlHandle<boolean>;
  addButton: (props: {label: string; onClick: () => void}) => MapGraphsControlHandle<string>;
  /** Adds a `label: value` readout line in the summary block. */
  addReadout: (label: string, initialValue?: string) => MapGraphsControlHandle<string>;
  /** Adds a color legend: discrete swatches, or a gradient when `gradient` is set. */
  addLegend: (props: {
    title: string;
    entries?: readonly {color: readonly number[]; label: string}[];
    gradient?: {colors: readonly (readonly number[])[]; minimumLabel: string; maximumLabel: string};
  }) => void;
  /** Adds explanatory text. */
  addNote: (text: string) => MapGraphsControlHandle<string>;
};

/** Mode summary shown in the tab strip. */
export type MapGraphsModeTab = {id: string; title: string};

/** The explorer panel. */
export type MapGraphsPanel = {
  /** Rebuilds the per-mode area and returns a fresh section. */
  beginMode: (props: {
    title: string;
    recipes: readonly string[];
    description: string;
  }) => MapGraphsControlSection;
  setActiveTab: (id: string) => void;
  setStatus: (message: string) => void;
  setFooter: (lines: readonly string[]) => void;
  setBasemapStatus: (message: string) => void;
  destroy: () => void;
};

const PANEL_STYLE: Partial<CSSStyleDeclaration> = {
  position: 'absolute',
  left: '14px',
  top: '14px',
  zIndex: '2',
  width: 'min(330px, calc(100% - 28px))',
  maxHeight: 'calc(100% - 28px)',
  overflowY: 'auto',
  boxSizing: 'border-box',
  padding: '14px',
  borderRadius: '14px',
  background: 'linear-gradient(160deg, rgba(11, 18, 38, 0.95), rgba(8, 13, 27, 0.9))',
  border: '1px solid rgba(113, 161, 242, 0.24)',
  boxShadow: '0 18px 55px rgba(0, 0, 0, 0.34)',
  color: '#edf4ff',
  font: '12px/1.45 system-ui, sans-serif',
  pointerEvents: 'auto'
};

const CONTROL_LABEL_STYLE =
  'display:flex;justify-content:space-between;gap:8px;color:#a9b8d0;margin-top:9px';
const INPUT_STYLE =
  'width:100%;box-sizing:border-box;margin-top:4px;background:#0d1730;color:#edf4ff;border:1px solid #2a3c66;border-radius:6px;padding:4px 6px;font:inherit';

/** Creates a full-size container when the example runs standalone. */
export function createStandaloneContainer(): HTMLDivElement {
  const container = document.createElement('div');
  Object.assign(container.style, {
    position: 'fixed',
    inset: '0',
    overflow: 'hidden',
    background: '#0b1020'
  });
  document.body.appendChild(container);
  return container;
}

/** Creates the MapLibre basemap container beneath the deck canvas. */
export function createBasemapContainer(container: HTMLDivElement): HTMLDivElement {
  const basemap = document.createElement('div');
  Object.assign(basemap.style, {position: 'absolute', inset: '0', zIndex: '0'});
  container.appendChild(basemap);
  return basemap;
}

/** Creates the panel with one tab per registered mode. */
export function createMapGraphsPanel(
  container: HTMLDivElement,
  tabs: readonly MapGraphsModeTab[],
  onSelectMode: (id: string) => void
): MapGraphsPanel {
  const panel = document.createElement('section');
  panel.setAttribute('aria-label', 'Map graphs explorer controls');
  panel.dataset.mapGraphsPanel = '';
  Object.assign(panel.style, PANEL_STYLE);
  panel.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;gap:8px">
      <strong style="font-size:15px;letter-spacing:-.2px">Map Graphs</strong>
      <span style="padding:2px 8px;border-radius:99px;background:#123a45;color:#80eadb;font-size:10px;font-weight:700;letter-spacing:.4px">WEBGPU</span>
    </div>
    <p style="margin:5px 0 10px;color:#a9b8d0">luma.gl map-graph recipes compiled once per mode,
      parameters rewritten every frame, outputs drawn by deck.gl straight from GPU buffers.</p>
    <div data-tabs style="display:flex;flex-wrap:wrap;gap:4px"></div>
    <div data-mode style="margin-top:12px"></div>
    <div data-footer style="margin-top:12px;padding-top:8px;border-top:1px solid #22314f;color:#7f90ad;font-size:11px"></div>
    <div data-basemap style="color:#d9a35f;font-size:11px"></div>
  `;
  container.appendChild(panel);
  const tabStrip = panel.querySelector<HTMLDivElement>('[data-tabs]')!;
  const modeArea = panel.querySelector<HTMLDivElement>('[data-mode]')!;
  const footer = panel.querySelector<HTMLDivElement>('[data-footer]')!;
  const basemapStatus = panel.querySelector<HTMLDivElement>('[data-basemap]')!;
  const tabButtons = new Map<string, HTMLButtonElement>();
  for (const tab of tabs) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = tab.title;
    button.dataset.modeTab = tab.id;
    button.style.cssText =
      'background:#13203f;color:#c8d6f0;border:1px solid #2a3c66;border-radius:99px;padding:3px 9px;font:inherit;font-size:11px;cursor:pointer';
    button.addEventListener('click', () => onSelectMode(tab.id));
    tabStrip.appendChild(button);
    tabButtons.set(tab.id, button);
  }
  let statusElement: HTMLDivElement | null = null;

  return {
    beginMode({title, recipes, description}) {
      modeArea.innerHTML = '';
      const header = document.createElement('div');
      header.innerHTML = `
        <div style="font-size:14px;font-weight:650">${title}</div>
        <div style="margin-top:2px;color:#80eadb;font:11px ui-monospace,monospace">${recipes.join(' · ')}</div>
        <p style="margin:6px 0 0;color:#a9b8d0">${description}</p>`;
      modeArea.appendChild(header);
      statusElement = document.createElement('div');
      statusElement.dataset.modeStatus = '';
      statusElement.style.cssText = 'margin-top:6px;color:#f1c96b;min-height:1em';
      modeArea.appendChild(statusElement);
      const controls = document.createElement('div');
      const readouts = document.createElement('div');
      readouts.dataset.modeReadouts = '';
      readouts.style.cssText =
        'margin-top:10px;padding:8px;border-radius:8px;background:rgba(19,32,63,.7);font:11px/1.6 ui-monospace,monospace;display:none';
      modeArea.appendChild(controls);
      modeArea.appendChild(readouts);
      return createControlSection(controls, readouts);
    },
    setActiveTab(id) {
      for (const [tabId, button] of tabButtons) {
        const active = tabId === id;
        button.style.background = active ? '#2b5bd7' : '#13203f';
        button.style.color = active ? '#ffffff' : '#c8d6f0';
        button.setAttribute('aria-pressed', String(active));
      }
    },
    setStatus(message) {
      if (statusElement) statusElement.textContent = message;
    },
    setFooter(lines) {
      footer.innerHTML = lines.map(line => `<div>${line}</div>`).join('');
    },
    setBasemapStatus(message) {
      basemapStatus.textContent = message;
    },
    destroy() {
      panel.remove();
    }
  };
}

function createControlSection(
  controls: HTMLDivElement,
  readouts: HTMLDivElement
): MapGraphsControlSection {
  function addLabeledRow(label: string): {row: HTMLLabelElement; valueElement: HTMLSpanElement} {
    const row = document.createElement('label');
    row.style.cssText = 'display:block';
    const caption = document.createElement('div');
    caption.style.cssText = CONTROL_LABEL_STYLE;
    const name = document.createElement('span');
    name.textContent = label;
    const valueElement = document.createElement('span');
    valueElement.style.color = '#edf4ff';
    caption.append(name, valueElement);
    row.appendChild(caption);
    controls.appendChild(row);
    return {row, valueElement};
  }

  return {
    addSlider({label, min, max, step, value, format = String, onChange}) {
      const {row, valueElement} = addLabeledRow(label);
      const input = document.createElement('input');
      input.type = 'range';
      Object.assign(input, {
        min: String(min),
        max: String(max),
        step: String(step),
        value: String(value)
      });
      input.style.cssText = 'width:100%;margin-top:4px';
      input.setAttribute('aria-label', label);
      valueElement.textContent = format(value);
      input.addEventListener('input', () => {
        const next = Number(input.value);
        valueElement.textContent = format(next);
        onChange(next);
      });
      row.appendChild(input);
      return {
        setValue: next => {
          input.value = String(next);
          valueElement.textContent = format(next);
        },
        setDisabled: disabled => {
          input.disabled = disabled;
        }
      };
    },
    addSelect({label, options, value, onChange}) {
      const {row} = addLabeledRow(label);
      const select = document.createElement('select');
      select.style.cssText = INPUT_STYLE;
      select.setAttribute('aria-label', label);
      for (const option of options) {
        const element = document.createElement('option');
        element.value = option.value;
        element.textContent = option.label;
        select.appendChild(element);
      }
      select.value = value;
      select.addEventListener('change', () => onChange(select.value as typeof value));
      row.appendChild(select);
      return {
        setValue: next => {
          select.value = next;
        },
        setDisabled: disabled => {
          select.disabled = disabled;
        }
      };
    },
    addToggle({label, value, onChange}) {
      const row = document.createElement('label');
      row.style.cssText =
        'display:flex;align-items:center;gap:6px;margin-top:9px;color:#a9b8d0;cursor:pointer';
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = value;
      input.addEventListener('change', () => onChange(input.checked));
      const text = document.createElement('span');
      text.textContent = label;
      row.append(input, text);
      controls.appendChild(row);
      return {
        setValue: next => {
          input.checked = next;
        },
        setDisabled: disabled => {
          input.disabled = disabled;
        }
      };
    },
    addButton({label, onClick}) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = label;
      button.style.cssText =
        'margin:9px 6px 0 0;background:#1d3a7a;color:#fff;border:1px solid #3b63c4;border-radius:6px;padding:4px 10px;font:inherit;cursor:pointer';
      button.addEventListener('click', onClick);
      controls.appendChild(button);
      return {
        setValue: next => {
          button.textContent = next;
        },
        setDisabled: disabled => {
          button.disabled = disabled;
          button.style.opacity = disabled ? '0.5' : '1';
        }
      };
    },
    addReadout(label, initialValue = '–') {
      readouts.style.display = 'block';
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;justify-content:space-between;gap:8px';
      const name = document.createElement('span');
      name.style.color = '#7f90ad';
      name.textContent = label;
      const value = document.createElement('span');
      value.textContent = initialValue;
      row.append(name, value);
      readouts.appendChild(row);
      return {
        setValue: next => {
          value.textContent = next;
        },
        setDisabled: () => {}
      };
    },
    addLegend({title, entries, gradient}) {
      const legend = document.createElement('div');
      legend.style.cssText = 'margin-top:10px;color:#a9b8d0;font-size:11px';
      const heading = document.createElement('div');
      heading.textContent = title;
      legend.appendChild(heading);
      const toCss = (color: readonly number[]) =>
        `rgba(${color[0]},${color[1]},${color[2]},${(color[3] ?? 255) / 255})`;
      if (gradient) {
        const bar = document.createElement('div');
        bar.style.cssText = `height:8px;border-radius:4px;margin-top:4px;background:linear-gradient(90deg,${gradient.colors.map(toCss).join(',')})`;
        const labels = document.createElement('div');
        labels.style.cssText = 'display:flex;justify-content:space-between';
        labels.innerHTML = `<span>${gradient.minimumLabel}</span><span>${gradient.maximumLabel}</span>`;
        legend.append(bar, labels);
      }
      if (entries) {
        const list = document.createElement('div');
        list.style.cssText = 'display:flex;flex-wrap:wrap;gap:4px 10px;margin-top:4px';
        for (const entry of entries) {
          const item = document.createElement('span');
          item.innerHTML = `<span style="display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;background:${toCss(entry.color)}"></span>${entry.label}`;
          list.appendChild(item);
        }
        legend.appendChild(list);
      }
      controls.appendChild(legend);
    },
    addNote(text) {
      const note = document.createElement('p');
      note.style.cssText = 'margin:8px 0 0;color:#7f90ad;font-size:11px';
      note.textContent = text;
      controls.appendChild(note);
      return {
        setValue: next => {
          note.textContent = next;
        },
        setDisabled: () => {}
      };
    }
  };
}
