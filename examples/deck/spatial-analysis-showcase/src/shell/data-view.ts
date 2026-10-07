// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {loadDatasetInfos} from '../data/catalog';
import {findScenesByDataset} from '../scenes/registry';
import {formatBytes, h, icon} from './dom';
import {getStoryHash} from './router';

/** The data catalog page: every dataset with licence, size, source and the stories that use it. */
export async function renderData(root: HTMLElement): Promise<void> {
  root.replaceChildren(h('div', {class: 'page-loading'}, 'Loading data catalog…'));
  const infos = await loadDatasetInfos();
  const usage = await Promise.all(infos.map(info => findScenesByDataset(info.id)));
  const total = infos.reduce((sum, info) => sum + info.approxBytes, 0);
  root.replaceChildren(
    h(
      'div',
      {class: 'page'},
      h('h1', {}, 'Data'),
      h(
        'p',
        {class: 'lede'},
        `${infos.length} datasets, about ${formatBytes(total)} in total. Every scene credits the data it uses; add ?data=synthetic to the URL to run on deterministic synthetic data instead of the network.`
      ),
      h(
        'div',
        {class: 'table-wrap'},
        h(
          'table',
          {class: 'data-table'},
          h(
            'thead',
            {},
            h(
              'tr',
              {},
              ['Dataset', 'Licence', 'Size', 'Bounding box', 'Used in'].map(title =>
                h('th', {}, title)
              )
            )
          ),
          h(
            'tbody',
            {},
            infos.map((info, row) =>
              h(
                'tr',
                {},
                h(
                  'td',
                  {},
                  h('strong', {}, info.title),
                  h('div', {class: 'muted small'}, info.id),
                  h('p', {class: 'small'}, info.description),
                  h(
                    'div',
                    {class: 'small'},
                    info.attribution,
                    ' ',
                    h(
                      'a',
                      {href: info.sourceUrl, target: '_blank', rel: 'noreferrer noopener'},
                      'Source',
                      icon('external', 11)
                    )
                  )
                ),
                h('td', {class: 'small'}, info.license),
                h('td', {}, formatBytes(info.approxBytes)),
                h('td', {class: 'small mono'}, info.bbox.map(value => value.toFixed(2)).join(', ')),
                h(
                  'td',
                  {},
                  usage[row].length
                    ? usage[row].map(scene =>
                        h('div', {}, h('a', {href: getStoryHash(scene.id)}, scene.title))
                      )
                    : h('span', {class: 'muted small'}, 'Not used yet')
                )
              )
            )
          )
        )
      )
    )
  );
}
