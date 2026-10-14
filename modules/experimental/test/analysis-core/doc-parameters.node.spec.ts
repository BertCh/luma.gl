// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {expect, it} from 'vitest';
import {DOCS_ROOT, readDocPage} from './doc-program';
import {generateDocPages} from './doc-parameters';

// A-22: the generated "Parameters" subsections on the analysis doc pages match the packers and
// their TSDoc. Regenerate with `UPDATE_DOC_PARAMETERS=1 npx vitest run --project node
// modules/experimental/test/analysis-core/doc-parameters.node.spec.ts`.

it('analysis doc "Parameters" subsections are in sync with the packers', () => {
  const {pages, undocumented} = generateDocPages();
  const stale: string[] = [];
  for (const [page, text] of pages) {
    if (text === readDocPage(page)) {
      continue;
    }
    if (process.env['UPDATE_DOC_PARAMETERS']) {
      writeFileSync(join(DOCS_ROOT, page), text);
    } else {
      stale.push(page);
    }
  }
  expect(stale, 'run with UPDATE_DOC_PARAMETERS=1 to regenerate').toEqual([]);
  // Every class with a packer has a section on an analysis page to hold its parameters.
  expect(undocumented).toEqual([]);
}, 600_000);
