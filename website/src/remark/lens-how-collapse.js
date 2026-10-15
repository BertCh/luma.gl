// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

function countLenses(node) {
  return (
    Number(
      (node.type === 'leafDirective' && node.name === 'lens') || node.data?.lensDirective === 'lens'
    ) + (node.children || []).reduce((count, child) => count + countLenses(child), 0)
  );
}

/** Replaces exact level-four headings with closed, MDX-safe disclosure parents. */
function remarkLensHowCollapse() {
  return tree => {
    function walk(parent) {
      if (!parent.children) return;
      const children = parent.children;
      for (let index = 0; index < children.length; index++) {
        const heading = children[index];
        if (
          heading.type !== 'heading' ||
          heading.depth !== 4 ||
          heading.children?.length !== 1 ||
          heading.children[0].type !== 'text' ||
          heading.children[0].value !== 'How it computes'
        ) {
          walk(heading);
          continue;
        }
        let end = index + 1;
        while (
          end < children.length &&
          children[end].type !== 'thematicBreak' &&
          !(children[end].type === 'heading' && children[end].depth <= 4)
        )
          end++;
        const content = children.slice(index + 1, end);
        const count = content.reduce((total, child) => total + countLenses(child), 0);
        const summary = {
          type: 'paragraph',
          data: {
            hName: 'summary',
            hProperties: {className: ['lens-how__summary']},
            hChildren: [
              {type: 'text', value: 'How it computes'},
              ...(count
                ? [
                    {
                      type: 'element',
                      tagName: 'span',
                      properties: {className: ['lens-how__count']},
                      children: [
                        {type: 'text', value: ` · ${count} ${count === 1 ? 'view' : 'views'}`}
                      ]
                    }
                  ]
                : [])
            ]
          },
          children: []
        };
        const details = {
          type: 'blockquote',
          data: {hName: 'details', hProperties: {className: ['lens-how']}},
          children: [summary, ...content]
        };
        children.splice(index, end - index, details);
        for (const child of content) walk(child);
      }
    }
    walk(tree);
  };
}

module.exports = remarkLensHowCollapse;
