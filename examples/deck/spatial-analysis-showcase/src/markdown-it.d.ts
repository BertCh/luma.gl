// Minimal typings for markdown-it 14 (the package ships none and `@types/markdown-it` is not installed).
declare module 'markdown-it' {
  type Token = {
    type: string;
    tag: string;
    content: string;
    info: string;
    attrs: [string, string][] | null;
    children: Token[] | null;
    attrGet: (name: string) => string | null;
    attrSet: (name: string, value: string) => void;
  };
  type Renderer = {
    rules: Record<
      string,
      | ((tokens: Token[], index: number, options: unknown, env: unknown, self: Renderer) => string)
      | undefined
    >;
    renderToken: (tokens: Token[], index: number, options: unknown) => string;
  };
  type Options = {html?: boolean; linkify?: boolean; typographer?: boolean; breaks?: boolean};
  export default class MarkdownIt {
    constructor(options?: Options);
    renderer: Renderer;
    render(source: string, env?: unknown): string;
    renderInline(source: string, env?: unknown): string;
    utils: {escapeHtml: (text: string) => string};
  }
}
