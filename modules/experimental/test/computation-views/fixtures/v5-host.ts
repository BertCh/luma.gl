// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Minimal host-owned DOM used to exercise the vanilla mount lifecycle in Node. */
export function makeHost(): {element: HTMLElement; getText(): string; getChildCount(): number} {
  type HostNode = {
    text: string;
    children: HostNode[];
    appendChild(child: HostNode): void;
    setAttribute(name: string, value: string): void;
  };
  const makeElement = (): HostNode => ({
    text: '',
    children: [],
    appendChild(child) {
      this.children.push(child);
    },
    setAttribute() {}
  });
  const children: HostNode[] = [];
  const element = {
    ownerDocument: {
      createElement: makeElement,
      createElementNS: makeElement,
      createTextNode: (text: string) => ({...makeElement(), text})
    },
    replaceChildren(...replacement: HostNode[]) {
      children.splice(0, children.length, ...replacement);
    }
  };
  const getText = (node: HostNode): string => node.text + node.children.map(getText).join('');
  return {
    element: element as unknown as HTMLElement,
    getText: () => children.map(getText).join(''),
    getChildCount: () => children.length
  };
}
