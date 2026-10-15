// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * A tiny, dependency-free view tree. Every computation view renders a {@link ViewNode} once; the
 * same tree is serialised to an HTML string for build-time static twins (no DOM) and materialised
 * into DOM elements for live lenses. Interaction never lives in the tree as closures: marks carry
 * `data-action`, `data-axis` and `data-key` attributes and the mount function delegates events.
 */

/** Attribute values; `false`, `null` and `undefined` omit the attribute, `true` writes it empty. */
export type ViewNodeAttributeValue = string | number | boolean | null | undefined;

/** An element in a view tree. */
export type ViewElementNode = {
  tag: string;
  attributes: Record<string, ViewNodeAttributeValue>;
  children: ViewNode[];
  /** Pre-rendered trusted markup (for example KaTeX output) placed as the element's content. */
  rawHtml?: string;
};

/** A node in a view tree: an element or a text run. */
export type ViewNode = ViewElementNode | string;

/** Children accepted by {@link h}: nested arrays and empty values are flattened away. */
export type ViewNodeChild = ViewNode | ViewNodeChild[] | null | undefined | false;

const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';
const MATHML_NAMESPACE = 'http://www.w3.org/1998/Math/MathML';
const SVG_TAGS = new Set([
  'svg',
  'image',
  'g',
  'path',
  'rect',
  'circle',
  'line',
  'polyline',
  'polygon',
  'text',
  'tspan',
  'defs',
  'pattern',
  'filter',
  'feGaussianBlur',
  'clipPath',
  'title',
  'desc',
  'marker',
  'use'
]);
const VOID_TAGS = new Set(['br', 'hr', 'img', 'input', 'meta', 'link', 'col', 'wbr']);

/** Creates a view element. */
export function h(
  tag: string,
  attributes: Record<string, ViewNodeAttributeValue> | null = null,
  ...children: ViewNodeChild[]
): ViewElementNode {
  return {tag, attributes: attributes ?? {}, children: flattenChildren(children)};
}

/** Creates an element whose content is trusted pre-rendered markup (KaTeX, MathML). */
export function rawHtml(
  tag: string,
  attributes: Record<string, ViewNodeAttributeValue> | null,
  html: string
): ViewElementNode {
  return {tag, attributes: attributes ?? {}, children: [], rawHtml: html};
}

/** Escapes text for HTML content and attribute values. */
export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** Serialises a view tree to an HTML string. Pure; usable at build time without a DOM. */
export function viewNodeToHtml(node: ViewNode): string {
  if (typeof node === 'string') {
    return escapeHtml(node);
  }
  let html = `<${node.tag}`;
  for (const [name, value] of Object.entries(node.attributes)) {
    if (value === false || value === null || value === undefined) {
      continue;
    }
    html += value === true ? ` ${name}` : ` ${name}="${escapeHtml(String(value))}"`;
  }
  if (VOID_TAGS.has(node.tag)) {
    return `${html}>`;
  }
  html += '>';
  html += node.rawHtml ?? node.children.map(viewNodeToHtml).join('');
  return `${html}</${node.tag}>`;
}

/** Materialises a view tree into DOM nodes owned by `ownerDocument`. */
export function viewNodeToDom(node: ViewNode, ownerDocument: Document, namespace?: string): Node {
  if (typeof node === 'string') {
    return ownerDocument.createTextNode(node);
  }
  const elementNamespace =
    node.tag === 'svg'
      ? SVG_NAMESPACE
      : node.tag === 'math'
        ? MATHML_NAMESPACE
        : (namespace ?? (SVG_TAGS.has(node.tag) ? SVG_NAMESPACE : undefined));
  const element = elementNamespace
    ? ownerDocument.createElementNS(elementNamespace, node.tag)
    : ownerDocument.createElement(node.tag);
  for (const [name, value] of Object.entries(node.attributes)) {
    if (value === false || value === null || value === undefined) {
      continue;
    }
    element.setAttribute(name, value === true ? '' : String(value));
  }
  if (node.rawHtml !== undefined) {
    element.innerHTML = node.rawHtml;
    return element;
  }
  const childNamespace = elementNamespace === SVG_NAMESPACE ? SVG_NAMESPACE : undefined;
  for (const child of node.children) {
    element.appendChild(viewNodeToDom(child, ownerDocument, childNamespace));
  }
  return element;
}

/** Concatenates the text content of a view tree (for descriptions and tests). */
export function getViewNodeText(node: ViewNode): string {
  if (typeof node === 'string') {
    return node;
  }
  return node.children.map(getViewNodeText).join('');
}

/** Finds every element in a view tree matching a predicate, depth first. */
export function findViewNodes(
  node: ViewNode,
  predicate: (element: ViewElementNode) => boolean
): ViewElementNode[] {
  if (typeof node === 'string') {
    return [];
  }
  const matches = predicate(node) ? [node] : [];
  for (const child of node.children) {
    matches.push(...findViewNodes(child, predicate));
  }
  return matches;
}

function flattenChildren(children: ViewNodeChild[]): ViewNode[] {
  const flattened: ViewNode[] = [];
  for (const child of children) {
    if (child === null || child === undefined || child === false) {
      continue;
    }
    if (Array.isArray(child)) {
      flattened.push(...flattenChildren(child));
    } else {
      flattened.push(child);
    }
  }
  return flattened;
}
