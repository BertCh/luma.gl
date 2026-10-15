// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createElement, type ReactNode} from 'react';

/** Renders trusted build-time lens markup without interpreting SVG or MathML as MDX. */
export default function LensHtml({html}: {html: string}): ReactNode {
  return createElement('div', {className: 'lens-html', dangerouslySetInnerHTML: {__html: html}});
}
