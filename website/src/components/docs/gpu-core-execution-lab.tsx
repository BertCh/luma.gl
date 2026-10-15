import React, {type ReactNode, useState} from 'react';

import {useGPUCoreAnatomyModel, RecordPendingPanel} from './gpu-core-compiler-anatomy';

const NUMBER_FORMAT = new Intl.NumberFormat('en-US', {notation: 'compact', maximumFractionDigits: 1});

/**
 * Execution-slice lab over the recorded `gpu-core.scan.anatomy` run. It offers only the invocation
 * budgets the run recorded. Optional-branch and undersized-output controls are absent because the
 * run recorded neither.
 */
export function GPUCoreExecutionLab(): ReactNode {
  const model = useGPUCoreAnatomyModel();
  const [selectedBudget, setSelectedBudget] = useState<number | undefined>();

  if (model.status !== 'ready') {
    return <RecordPendingPanel model={model} />;
  }
  const series = model.sliceSeries.find(candidate => candidate.budget === selectedBudget) ?? model.sliceSeries[0];

  return (
    <section className="gpu-core-execution-lab" aria-label="Budgeted execution lab">
      <header>
        <span className="gpu-core-tutorial__eyebrow">Recorded execution policy</span>
        <h3>Slice work at node boundaries before it surprises the frame</h3>
        <p>
          Slices come from the recorded run ({model.provenance.adapterLabel}, run{' '}
          <code>{model.provenance.runId}</code>); changing the budget selects another recorded plan and
          does not rebuild graph topology.
        </p>
      </header>
      {series ? (
        <>
          <div className="gpu-core-execution-lab__controls" role="group" aria-label="Recorded invocation budgets">
            <span>Per-frame invocation budget</span>
            {model.sliceSeries.map(candidate => (
              <button
                aria-pressed={candidate.budget === series.budget}
                key={candidate.budget}
                onClick={() => setSelectedBudget(candidate.budget)}
                type="button"
              >
                {NUMBER_FORMAT.format(candidate.budget)}
              </button>
            ))}
          </div>
          <div className="gpu-core-execution-lab__status">
            <div>
              <span>Execution</span>
              <strong>
                {series.slices.length} frame {series.slices.length === 1 ? 'slice' : 'slices'}
              </strong>
            </div>
            <div>
              <span>Oversized steps</span>
              <strong>{series.slices.filter(slice => slice.oversized).length}</strong>
            </div>
          </div>
          <ol>
            {series.slices.map(slice => (
              <li key={slice.index}>
                <strong>Frame {slice.index + 1}</strong>:{' '}
                {slice.nodeIds.map(nodeIdentifier => model.nodes.find(node => node.id === nodeIdentifier)?.label ?? nodeIdentifier).join(', ')}
                {slice.oversized ? ' (exceeds the budget and runs alone)' : ''}
              </li>
            ))}
          </ol>
        </>
      ) : (
        <p>The run recorded no frame slices.</p>
      )}
    </section>
  );
}
