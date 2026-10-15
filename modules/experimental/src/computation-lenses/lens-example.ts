// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Maximum number of source lines shown beneath a result figure. */
export const LENS_EXAMPLE_MAX_LINES = 25;

/** Checked source snippet for a scenario's actual execution code. */
export type LensExample = {code: string; language: 'ts'; lineCount: number};

/** Extracts and dedents the unique example region, rejecting missing, empty or oversized blocks. */
export function extractLensExample(sourceText: string): LensExample {
  const lines = sourceText.replace(/\r\n?/g, '\n').split('\n');
  const starts = lines.flatMap((line, index) =>
    /^\s*\/\/\s*#region example\s*$/.test(line) ? [index] : []
  );
  const ends = lines.flatMap((line, index) =>
    /^\s*\/\/\s*#endregion example\s*$/.test(line) ? [index] : []
  );
  if (!starts.length || !ends.length) throw new Error('Missing example region markers');
  if (starts.length !== 1 || ends.length !== 1) throw new Error('Duplicated example region');
  if (ends[0] <= starts[0]) throw new Error('Example region end precedes its start');
  const snippet = lines.slice(starts[0] + 1, ends[0]);
  if (!snippet.length || snippet.every(line => !line.trim()))
    throw new Error('Empty example region');
  if (!snippet[0].trim() || !snippet[snippet.length - 1].trim())
    throw new Error('Example region has blank first or last line');
  if (snippet.length > LENS_EXAMPLE_MAX_LINES)
    throw new Error(`Example region exceeds ${LENS_EXAMPLE_MAX_LINES} lines (${snippet.length})`);
  const indent = Math.min(
    ...snippet.filter(line => line.trim()).map(line => line.match(/^\s*/)![0].length)
  );
  return {
    code: snippet.map(line => line.slice(indent)).join('\n'),
    language: 'ts',
    lineCount: snippet.length
  };
}

/** Lists imports outside the public core, engine, gpgpu and experimental entry points. */
export function findNonPublicImports(sourceText: string): string[] {
  // Tokenize comments and strings first so commented imports and quoted code are ignored.
  const tokens = [
    ...sourceText.matchAll(
      /\/\*[\s\S]*?\*\/|\/\/[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|[A-Za-z_$][\w$]*|[^\s]/g
    )
  ]
    .map(match => match[0])
    .filter(token => !token.startsWith('//') && !token.startsWith('/*'));
  const imports: string[] = [];
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index];
    let specifier: string | undefined;
    if (token === 'from' || token === 'import') specifier = tokens[index + 1];
    if ((token === 'import' || token === 'require') && tokens[index + 1] === '(')
      specifier = tokens[index + 2];
    if (!specifier || !/^["'`]/.test(specifier)) continue;
    const name = specifier.slice(1, -1);
    const publicEntry =
      /^@luma\.gl\/(?:core|engine|gpgpu)$/.test(name) ||
      /^@luma\.gl\/(?:gpgpu|experimental)\/[a-z0-9-]+(?:\/[a-z0-9-]+)*$/.test(name);
    if (!publicEntry || /\/(?:src|dist|test)(?:\/|$)/.test(name)) imports.push(name);
  }
  return [...new Set(imports)];
}

/** Returns the sibling example path relative to the computation-lenses source directory. */
export function getLensExamplePath(scenarioId: string): string {
  if (
    !/^[a-z0-9]+(?:-[a-z0-9]+)*\.[a-z0-9]+(?:-[a-z0-9]+)*(?:\.[a-z0-9]+(?:-[a-z0-9]+)*)?$/.test(
      scenarioId
    )
  ) {
    throw new Error(`Invalid scenario id: ${scenarioId}`);
  }
  const [entry, element] = scenarioId.split('.');
  return `scenarios/${entry}/${element}.example.ts`;
}
