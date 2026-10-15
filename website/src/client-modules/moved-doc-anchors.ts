// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ClientModule} from '@docusaurus/types';
import movedDocAnchors from './moved-doc-anchors.json';

type MovedAnchorMap = Record<string, Record<string, string>>;

function decodeAnchor(hash: string): string {
  try {
    return decodeURIComponent(hash.replace(/^#/, ''));
  } catch {
    return hash.replace(/^#/, '');
  }
}

/** Resolves a moved anchor while preserving a deployment's base path. */
export function resolveMovedAnchor(
  pathname: string,
  hash: string,
  map: MovedAnchorMap
): string | null {
  const normalizedPathname = pathname.replace(/\/$/, '');
  const anchor = decodeAnchor(hash);
  for (const [route, anchors] of Object.entries(map)) {
    const normalizedRoute = route.replace(/\/$/, '');
    if (anchor && normalizedPathname.endsWith(normalizedRoute) && Object.hasOwn(anchors, anchor)) {
      return normalizedPathname.slice(0, -normalizedRoute.length) + anchors[anchor];
    }
  }
  return null;
}

const movedDocAnchorsClientModule: ClientModule = {
  onRouteDidUpdate({location}) {
    if (typeof window === 'undefined' || !location.hash) {
      return;
    }
    if (document.getElementById(decodeAnchor(location.hash))) {
      return;
    }
    const target = resolveMovedAnchor(location.pathname, location.hash, movedDocAnchors);
    if (target) {
      window.location.replace(target);
    }
  }
};

export default movedDocAnchorsClientModule;
