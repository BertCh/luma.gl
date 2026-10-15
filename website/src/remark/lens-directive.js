// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const fs = require('node:fs');
const path = require('node:path');
const katex = require('katex');
const {
  buildRecordedTwin,
  buildPendingTwin,
  twinKey,
  parseLensAttributes,
  isMdxDocument,
  setMdxLensHtml
} = require('./lens-twin-html');

/** Attaches static lens HAST to leaf directives for Docusaurus's MDX pipeline. */
function remarkLensDirective({docsDirectory}) {
  return async (tree, file) => {
    const {fromHtml} = await import('hast-util-from-html');
    const {toHtml} = await import('hast-util-to-html');
    const isMdx = isMdxDocument(tree, file);
    const parseHtml = markup => fromHtml(markup, {fragment: true}).children;
    function walk(node) {
      if (node.type === 'leafDirective' && node.name === 'lens') {
        try {
          const directive = parseLensAttributes(node.attributes);
          const twinPath = path.join(
            docsDirectory,
            'lens-records',
            'twins',
            `${twinKey(directive)}.json`
          );
          const figure = fs.existsSync(twinPath)
            ? buildRecordedTwin(
                directive,
                JSON.parse(fs.readFileSync(twinPath, 'utf8')),
                katex.renderToString,
                parseHtml
              )
            : buildPendingTwin(directive);
          if (isMdx) {
            setMdxLensHtml(node, toHtml(figure));
          } else {
            node.data = {
              hName: figure.tagName,
              hProperties: figure.properties,
              hChildren: figure.children
            };
          }
        } catch (error) {
          throw new Error(`${file.path}:${node.position?.start?.line ?? 1}: ${error.message}`);
        }
      }
      for (const child of node.children || []) {
        walk(child);
      }
    }
    walk(tree);
  };
}

module.exports = remarkLensDirective;
