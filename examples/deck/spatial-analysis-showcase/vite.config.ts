import {defineConfig} from 'vite';

const websiteBaseUrl = process.env['WEBSITE_BASE_URL'] || '/';

const alias = {
  '@luma.gl/core': `${__dirname}/../../../modules/core/src`,
  '@luma.gl/engine': `${__dirname}/../../../modules/engine/src`,
  '@luma.gl/experimental': `${__dirname}/../../../modules/experimental/src`,
  '@luma.gl/gpgpu': `${__dirname}/../../../modules/gpgpu/src`,
  '@luma.gl/shadertools': `${__dirname}/../../../modules/shadertools/src`,
  '@luma.gl/webgl': `${__dirname}/../../../modules/webgl/src`,
  '@luma.gl/webgpu': `${__dirname}/../../../modules/webgpu/src`
};

export default defineConfig({
  base: websiteBaseUrl,
  resolve: {
    alias,
    dedupe: Object.keys(alias)
  },
  optimizeDeps: {exclude: Object.keys(alias)},
  server: {port: 5312, strictPort: true, open: false}
});
