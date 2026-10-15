// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Card additions: result is a rung; every figure carries data-rung. Recorded twins may
// include size (full/thumbnail) and an example (code, language: ts, lineCount), printed
// immediately after the caption. Pending result figures retain the result aspect box.

const ATTRIBUTE_NAMES = ['scenario', 'rung', 'bind', 'datum', 'iteration', 'caption', 'view'];
const RUNG_NAMES = [
  'result',
  'formula',
  'datum',
  'iterations',
  'kernel',
  'graph',
  'timing',
  'compare',
  'scope',
  'table'
];

/** Validates author attributes and normalizes optional integer selectors. */
function parseLensAttributes(attributes = {}) {
  for (const [name, value] of Object.entries(attributes)) {
    if (!ATTRIBUTE_NAMES.includes(name)) {
      throw new Error(`Unknown lens attribute: ${name}`);
    }
    if (typeof value !== 'string') {
      throw new Error(`Lens attribute ${name} must be a string.`);
    }
  }
  if (!/^[a-z0-9-]+(\.[a-z0-9-]+){1,2}$/.test(attributes.scenario || '')) {
    throw new Error('Invalid lens scenario; expected <entry>.<element>[.<variant>].');
  }
  if (!RUNG_NAMES.includes(attributes.rung)) {
    throw new Error(`Invalid lens rung: ${attributes.rung}`);
  }
  if (attributes.view !== undefined && !/^[a-z0-9]+(-[a-z0-9]+)*$/.test(attributes.view)) {
    throw new Error(`Invalid lens view: ${attributes.view}`);
  }
  const directive = {...attributes, bind: attributes.bind ?? 'page'};
  for (const name of ['datum', 'iteration']) {
    if (attributes[name] !== undefined) {
      if (!/^\d+$/.test(attributes[name]) || !Number.isSafeInteger(Number(attributes[name]))) {
        throw new Error(`Lens attribute ${name} must be a non-negative safe integer.`);
      }
      directive[name] = Number(attributes[name]);
    }
  }
  return directive;
}

/** Returns the shared filename key, including zero-valued selectors and the chosen view. */
function twinKey(directive) {
  return `${directive.scenario}~${directive.rung}${directive.datum === undefined ? '' : `~d${directive.datum}`}${directive.iteration === undefined ? '' : `~i${directive.iteration}`}${directive.view === undefined ? '' : `~v${directive.view}`}`;
}

/** Verdict chip text, identical to computation-views `renderVerdictChip`. */
function verdictText(verdict) {
  return verdict.status === 'pass'
    ? `✓ ${verdict.label}`
    : verdict.status === 'fail'
      ? `✕ ${verdict.label}: ${verdict.reason}`
      : `? ${verdict.label}: unknown — ${verdict.reason}`;
}

function text(value) {
  return {type: 'text', value: String(value)};
}

function element(tagName, properties, children = []) {
  return {type: 'element', tagName, properties, children};
}

function buildLens(directive, recorded, description, children) {
  const properties = {
    scenario: directive.scenario,
    rung: directive.rung,
    bind: directive.bind,
    ...(directive.datum === undefined ? {} : {datum: String(directive.datum)}),
    ...(directive.iteration === undefined ? {} : {iteration: String(directive.iteration)}),
    ...(directive.view === undefined ? {} : {view: directive.view}),
    recorded: String(recorded),
    ariaLabel: description
  };
  return element('ladder-lens', properties, children);
}

function buildFigure(directive, pending, children) {
  const key = twinKey(directive);
  return element(
    'figure',
    {
      className: pending ? ['lens-twin', 'lens-twin--pending'] : ['lens-twin'],
      id: `lens-${key.replaceAll('~', '-')}`,
      dataLensKey: key,
      dataRung: directive.rung
    },
    children
  );
}

function buildLiveButton(directive) {
  return element('p', {className: ['lens-twin__live']}, [
    element(
      'button',
      {
        type: 'button',
        className: ['lens-twin__live-button'],
        dataLensKey: twinKey(directive),
        hidden: true
      },
      [text('Run live on this GPU')]
    )
  ]);
}

/** Builds the recorded HTML contract using injected TeX rendering and fragment parsing. */
function buildRecordedTwin(directive, twin, renderTeX, parseHtml) {
  const {keyNumber, provenance} = twin;
  const children = [
    buildLens(directive, true, twin.description, [
      element(
        'div',
        {className: ['lens-twin__figure'], role: 'img', ariaLabel: twin.description},
        twin.figure ? parseHtml(twin.figure.html) : []
      )
    ]),
    element('figcaption', {className: ['lens-twin__caption']}, [
      element('span', {className: ['lens-twin__caption-text']}, [
        text(directive.caption ?? twin.caption)
      ]),
      element('span', {className: ['lens-twin__key-number']}, [
        element('strong', {}, [text(`${keyNumber.label}:`)]),
        text(' '),
        element('data', {value: keyNumber.value}, [text(keyNumber.value)]),
        ...(keyNumber.unit ? [text(` ${keyNumber.unit}`)] : [])
      ]),
      element('span', {className: ['lens-twin__provenance']}, [
        text(
          `Recorded on ${provenance.adapter}${provenance.variant ? `, ${provenance.variant}` : ''}, ${provenance.date}${provenance.observerEffects.length ? `; ${provenance.observerEffects.join(', ')}` : ''}`
        )
      ]),
      ...(twin.verdicts?.length
        ? [
            element(
              'span',
              {className: ['lens-twin__verdicts']},
              twin.verdicts.map(verdict =>
                element(
                  'span',
                  {
                    className: ['lens-verdict', `lens-verdict--${verdict.status}`],
                    role: 'note',
                    title: verdict.reason
                  },
                  [text(verdictText(verdict))]
                )
              )
            )
          ]
        : [])
    ])
  ];
  if (twin.example) {
    children.push(
      element('pre', {className: ['lens-example']}, [
        element('code', {className: ['language-ts']}, [text(twin.example.code)])
      ])
    );
  }
  if (twin.disabled) {
    children.push(
      element('p', {className: ['lens-twin__disabled']}, [
        text(`Rung unavailable: ${twin.disabled.reason}`)
      ])
    );
  }
  if (twin.formula) {
    children.push(
      element(
        'div',
        {
          className: ['lens-twin__formula'],
          dataDerivation: twin.formula.derivation
        },
        parseHtml(
          renderTeX(twin.formula.tex, {
            displayMode: true,
            output: 'htmlAndMathml',
            throwOnError: false,
            strict: 'ignore',
            trust: ({command}) => command === '\\htmlClass' || command === '\\htmlData',
            macros: {'\\term': '\\htmlClass{lens-term}{#1}'}
          })
        )
      )
    );
  }
  if (twin.receipt) {
    children.push(
      element('table', {className: ['lens-twin__receipt']}, [
        element('thead', {}, [
          element(
            'tr',
            {},
            twin.receipt.columns.map(column => element('th', {}, [text(column)]))
          )
        ]),
        element(
          'tbody',
          {},
          twin.receipt.rows.map((row, rowIndex) =>
            element(
              'tr',
              twin.receipt.provenance?.[rowIndex]
                ? {dataProvenance: twin.receipt.provenance[rowIndex]}
                : {},
              row.map(cell => element('td', {}, [text(cell === '' ? '—' : cell)]))
            )
          )
        )
      ])
    );
  }
  children.push(buildLiveButton(directive));
  return buildFigure(directive, false, children);
}

/** Builds the pending HTML contract without reading records or generating a twin. */
function buildPendingTwin(directive) {
  const caption = directive.caption ?? `How ${directive.scenario} computes (${directive.rung})`;
  return buildFigure(directive, true, [
    buildLens(directive, false, caption, [
      element(
        'div',
        {
          className: [
            'lens-twin__figure',
            'lens-twin__figure--pending',
            ...(directive.rung === 'result' ? ['lens-twin__figure--result'] : [])
          ]
        },
        [
          text('Record pending: no recorded run of '),
          element('code', {}, [text(directive.scenario)]),
          text(' yet.')
        ]
      )
    ]),
    element('figcaption', {className: ['lens-twin__caption']}, [
      element('span', {className: ['lens-twin__caption-text']}, [text(caption)]),
      element('span', {className: ['lens-twin__key-number']}, [
        element('strong', {}, [text('Key number:')]),
        text(' pending')
      ])
    ]),
    buildLiveButton(directive)
  ]);
}

/** Detects MDX syntax even when the source uses the documentation's .md extension. */
function isMdxDocument(tree, file) {
  function hasMdxNode(node) {
    return node.type.startsWith('mdx') || (node.children || []).some(hasMdxNode);
  }
  return file.path?.endsWith('.mdx') || hasMdxNode(tree);
}

/** Carries trusted, generated HTML as a JSX string prop, never as MDX source. */
function setMdxLensHtml(node, html) {
  const directiveName = node.name;
  Object.assign(node, {
    type: 'mdxJsxFlowElement',
    name: 'LensHtml',
    attributes: [{type: 'mdxJsxAttribute', name: 'html', value: html}],
    children: [],
    data: {lensDirective: directiveName}
  });
}

module.exports = {
  buildRecordedTwin,
  buildPendingTwin,
  twinKey,
  parseLensAttributes,
  isMdxDocument,
  setMdxLensHtml
};
