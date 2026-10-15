import React, {type ReactNode, useEffect, useId, useState} from 'react';

import {
  GPU_CORE_ANATOMY_SCENARIO_ID,
  GPU_CORE_COMPILER_STAGES,
  buildGPUCoreAnatomyModel,
  type GPUCoreAnatomyModel,
  type GPUCoreAnatomyNode,
  type GPUCoreAnatomyUnavailable,
  type GPUCoreCompilerStageId
} from './gpu-core-compiler-model';

const NUMBER_FORMAT = new Intl.NumberFormat('en-US', {notation: 'compact', maximumFractionDigits: 1});

const PENDING_MODEL: GPUCoreAnatomyUnavailable = {
  status: 'pending',
  reason: 'Loading the recorded run.'
};

/**
 * Loads and decodes the recorded `gpu-core.scan.anatomy` run in the browser only. Resolves to a
 * pending state before the load completes (and during SSR) and when the record is missing.
 */
export function useGPUCoreAnatomyModel(): GPUCoreAnatomyModel | GPUCoreAnatomyUnavailable {
  const [model, setModel] = useState<GPUCoreAnatomyModel | GPUCoreAnatomyUnavailable>(PENDING_MODEL);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [module, lenses] = await Promise.all([
          import('../../../../docs/lens-records/gpu-core.scan.anatomy.json'),
          import('@luma.gl/experimental/computation-lenses')
        ]);
        const decoded = lenses.fromRecordedLensFile(module.default ?? module);
        if (!cancelled) setModel(buildGPUCoreAnatomyModel(decoded));
      } catch {
        if (!cancelled) {
          setModel({
            status: 'pending',
            reason: 'The recorded run is not available in this build.'
          });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return model;
}

/** Interactive view of the command-graph compilation lifecycle, rebuilt from a recorded GPU run. */
export function GPUCoreCompilerAnatomy(): ReactNode {
  const model = useGPUCoreAnatomyModel();
  const [activeStageIdentifier, setActiveStageIdentifier] =
    useState<GPUCoreCompilerStageId>('declared');
  const [selectedNodeIdentifier, setSelectedNodeIdentifier] = useState<string | undefined>();
  const [selectedBudget, setSelectedBudget] = useState<number | undefined>();
  const panelIdentifier = useId();

  if (model.status !== 'ready') {
    return <RecordPendingPanel model={model} />;
  }

  const activeStage = GPU_CORE_COMPILER_STAGES.find(stage => stage.id === activeStageIdentifier)!;
  const selectedNode =
    model.nodes.find(node => node.id === selectedNodeIdentifier) ?? model.nodes[0];
  const sliceSeries =
    model.sliceSeries.find(series => series.budget === selectedBudget) ?? model.sliceSeries[0];

  return (
    <section className="gpu-core-anatomy" aria-label="Interactive anatomy of a compiled GPU Core">
      <header>
        <div>
          <span className="gpu-core-tutorial__eyebrow">Interactive compiler anatomy</span>
          <h3>Follow one recorded graph from declaration to encoded work</h3>
          <p>
            Choose a compilation view, then select a node to see its operator, resources,
            dependencies, and dispatch shape exactly as recorded.
          </p>
          <p>
            <small>
              Recorded {model.provenance.recordedAt} on {model.provenance.adapterLabel}, run{' '}
              <code>{model.provenance.runId}</code>, scenario{' '}
              <code>{GPU_CORE_ANATOMY_SCENARIO_ID}</code>.
            </small>
          </p>
        </div>
      </header>

      <nav className="gpu-core-anatomy__stages" aria-label="Compilation views" role="tablist">
        {GPU_CORE_COMPILER_STAGES.map((stage, stageIndex) => (
          <button
            aria-controls={panelIdentifier}
            aria-selected={stage.id === activeStageIdentifier}
            id={`${panelIdentifier}-${stage.id}`}
            key={stage.id}
            onClick={() => setActiveStageIdentifier(stage.id)}
            role="tab"
            type="button"
          >
            <small>{stageIndex + 1}</small>
            {stage.label}
          </button>
        ))}
      </nav>

      <div
        aria-labelledby={`${panelIdentifier}-${activeStageIdentifier}`}
        className="gpu-core-anatomy__stage-panel"
        id={panelIdentifier}
        role="tabpanel"
      >
        <strong>{activeStage.label}</strong>
        <p>{activeStage.explanation}</p>
      </div>

      {activeStageIdentifier === 'lifetimes' || activeStageIdentifier === 'allocations' ? (
        <ResourceLifetimeView allocationView={activeStageIdentifier === 'allocations'} model={model} />
      ) : activeStageIdentifier === 'hazards' ? (
        <HazardView model={model} selectedNodeIdentifier={selectedNode.id} onSelect={setSelectedNodeIdentifier} />
      ) : activeStageIdentifier === 'slices' ? (
        <SliceView
          budget={sliceSeries?.budget}
          budgets={model.sliceSeries.map(series => series.budget)}
          model={model}
          onBudgetChange={setSelectedBudget}
          onSelect={setSelectedNodeIdentifier}
          selectedNodeIdentifier={selectedNode.id}
        />
      ) : activeStageIdentifier === 'declared' ? (
        <DeclaredView model={model} selectedNodeIdentifier={selectedNode.id} onSelect={setSelectedNodeIdentifier} />
      ) : (
        <NodeSequenceView
          mode={activeStageIdentifier}
          model={model}
          selectedNodeIdentifier={selectedNode.id}
          onSelect={setSelectedNodeIdentifier}
        />
      )}

      <NodeInspector node={selectedNode} />
    </section>
  );
}

/** Shown when the recorded run is unavailable; it lists what the panel will show and invents nothing. */
export function RecordPendingPanel({model}: {model: GPUCoreAnatomyUnavailable}): ReactNode {
  return (
    <section className="gpu-core-anatomy" aria-label="Interactive anatomy of a compiled GPU Core">
      <header>
        <div>
          <span className="gpu-core-tutorial__eyebrow">Interactive compiler anatomy</span>
          <h3>Record pending</h3>
          <p>
            {model.reason} This panel shows the compiled graph of a real <code>GPUScan</code> run inside a
            mask, compaction, and masked reduction graph: its nodes, hazard edges, resource lifetimes,
            physical allocations, dispatch shapes, and frame slices. Every item comes from the recorded
            scenario <code>{GPU_CORE_ANATOMY_SCENARIO_ID}</code>.
          </p>
          <p>
            The record is captured on a WebGPU device by running that scenario and saving it to{' '}
            <code>docs/lens-records/{GPU_CORE_ANATOMY_SCENARIO_ID}.json</code>.
          </p>
        </div>
      </header>
    </section>
  );
}

function DeclaredView({
  model,
  onSelect,
  selectedNodeIdentifier
}: {
  model: GPUCoreAnatomyModel;
  onSelect: (identifier: string) => void;
  selectedNodeIdentifier: string;
}): ReactNode {
  return (
    <div className="gpu-core-anatomy__declared">
      {model.contributors.map(contributor => (
        <section key={contributor.label}>
          <strong>{contributor.label}</strong>
          <ol className="gpu-core-anatomy__nodes gpu-core-anatomy__nodes--declared">
            {contributor.nodeIds.map(nodeIdentifier => {
              const node = model.nodes.find(candidate => candidate.id === nodeIdentifier)!;
              return (
                <li key={node.id}>
                  <button
                    aria-pressed={node.id === selectedNodeIdentifier}
                    className={`gpu-core-anatomy__node gpu-core-anatomy__node--${node.kind}`}
                    onClick={() => onSelect(node.id)}
                    type="button"
                  >
                    <small>{node.contributorPath}</small>
                    <strong>{node.label}</strong>
                  </button>
                </li>
              );
            })}
          </ol>
        </section>
      ))}
    </div>
  );
}

function NodeSequenceView({
  mode,
  model,
  onSelect,
  selectedNodeIdentifier
}: {
  mode: GPUCoreCompilerStageId;
  model: GPUCoreAnatomyModel;
  onSelect: (identifier: string) => void;
  selectedNodeIdentifier: string;
}): ReactNode {
  const encodedPassCount = model.encodedPasses?.length;
  return (
    <>
      {mode === 'encoded' ? (
        <p>
          <small>{model.encodedNote}</small>
          {encodedPassCount !== undefined ? <small> {encodedPassCount} passes.</small> : null}
        </p>
      ) : null}
      <ol className={`gpu-core-anatomy__nodes gpu-core-anatomy__nodes--${mode}`}>
        {model.nodes.map((node, nodeIndex) => (
          <li key={node.id}>
            <button
              aria-pressed={node.id === selectedNodeIdentifier}
              className={`gpu-core-anatomy__node gpu-core-anatomy__node--${node.kind}`}
              onClick={() => onSelect(node.id)}
              type="button"
            >
              <small>step {nodeIndex + 1}</small>
              <strong>{node.label}</strong>
              {mode === 'encoded' ? <span>{describeDispatch(node)}</span> : null}
            </button>
          </li>
        ))}
      </ol>
    </>
  );
}

function HazardView({
  model,
  onSelect,
  selectedNodeIdentifier
}: {
  model: GPUCoreAnatomyModel;
  onSelect: (identifier: string) => void;
  selectedNodeIdentifier: string;
}): ReactNode {
  const labelOf = (identifier: string) =>
    model.nodes.find(node => node.id === identifier)?.label ?? identifier;
  return (
    <div className="gpu-core-anatomy__hazards">
      {model.edges.map(edge => (
        <div className="gpu-core-anatomy__hazard" key={`${edge.from}-${edge.to}-${edge.kind}-${edge.resource ?? ''}`}>
          <button aria-pressed={edge.from === selectedNodeIdentifier} onClick={() => onSelect(edge.from)} type="button">
            {labelOf(edge.from)}
          </button>
          <span>
            → <code>{edge.resource ? `${edge.resource} · ${edge.kind}` : edge.kind}</code> →
          </span>
          <button aria-pressed={edge.to === selectedNodeIdentifier} onClick={() => onSelect(edge.to)} type="button">
            {labelOf(edge.to)}
          </button>
        </div>
      ))}
    </div>
  );
}

function ResourceLifetimeView({
  allocationView,
  model
}: {
  allocationView: boolean;
  model: GPUCoreAnatomyModel;
}): ReactNode {
  const allocationNumbers = new Map(model.allocations.map((group, index) => [group.allocation, index + 1]));
  return (
    <div
      className="gpu-core-anatomy__lifetimes"
      role="img"
      aria-label={`Logical resource lifetimes across ${model.nodes.length} scheduled nodes`}
    >
      <div className="gpu-core-anatomy__lifetime-axis" aria-hidden="true">
        <span />
        {model.nodes.map((node, index) => (
          <small key={node.id}>{index + 1}</small>
        ))}
      </div>
      {model.resources.map(resource => (
        <div className="gpu-core-anatomy__lifetime" key={resource.id}>
          <strong>{resource.id}</strong>
          <div className="gpu-core-anatomy__lifetime-track">
            {resource.firstNode !== undefined && resource.lastNode !== undefined ? (
              <span
                className={`gpu-core-anatomy__lifetime-bar gpu-core-anatomy__lifetime-bar--${resource.ownership}`}
                style={{
                  gridColumn: `${resource.firstNode + 1} / ${resource.lastNode + 2}`,
                  ...(allocationView && resource.allocation
                    ? ({'--allocation-index': allocationNumbers.get(resource.allocation)} as React.CSSProperties)
                    : {})
                }}
              >
                {allocationView
                  ? resource.allocation ?? resource.ownership
                  : `${resource.ownership}${resource.byteLength !== undefined ? ` · ${NUMBER_FORMAT.format(resource.byteLength)} B` : ''}`}
              </span>
            ) : (
              <small>lifetime not recorded</small>
            )}
          </div>
        </div>
      ))}
      {allocationView ? <p>{model.allocationSentence}</p> : null}
    </div>
  );
}

function SliceView({
  budget,
  budgets,
  model,
  onBudgetChange,
  onSelect,
  selectedNodeIdentifier
}: {
  budget: number | undefined;
  budgets: readonly number[];
  model: GPUCoreAnatomyModel;
  onBudgetChange: (budget: number) => void;
  onSelect: (identifier: string) => void;
  selectedNodeIdentifier: string;
}): ReactNode {
  const series = model.sliceSeries.find(candidate => candidate.budget === budget);
  if (!series) {
    return <p>The run recorded no frame slices.</p>;
  }
  return (
    <div className="gpu-core-anatomy__slice-view">
      <div role="group" aria-label="Recorded invocation budgets">
        Invocation budget:{' '}
        {budgets.map(candidate => (
          <button
            aria-pressed={candidate === budget}
            key={candidate}
            onClick={() => onBudgetChange(candidate)}
            type="button"
          >
            {NUMBER_FORMAT.format(candidate)}
          </button>
        ))}
        <small> Only the budgets this run recorded.</small>
      </div>
      <div className="gpu-core-anatomy__slices">
        {series.slices.map(slice => (
          <section className={slice.oversized ? 'gpu-core-anatomy__slice--oversized' : undefined} key={slice.index}>
            <header>
              <strong>Frame {slice.index + 1}</strong>
              <span>
                {slice.invocationCount !== undefined
                  ? `${NUMBER_FORMAT.format(slice.invocationCount)} invocations`
                  : 'invocations unknown'}
              </span>
            </header>
            <div>
              {slice.nodeIds.map(nodeIdentifier => {
                const node = model.nodes.find(candidate => candidate.id === nodeIdentifier)!;
                return (
                  <button
                    aria-pressed={nodeIdentifier === selectedNodeIdentifier}
                    key={nodeIdentifier}
                    onClick={() => onSelect(nodeIdentifier)}
                    type="button"
                  >
                    {node.label}
                  </button>
                );
              })}
            </div>
            {slice.oversized ? <small>Indivisible step exceeds the requested budget and runs alone.</small> : null}
          </section>
        ))}
      </div>
    </div>
  );
}

function describeDispatch(node: GPUCoreAnatomyNode): string {
  if (node.workgroups?.length) {
    return `${node.workgroups.join(' × ')} workgroups${node.indirect ? ', indirect' : ''}`;
  }
  return 'dispatch not recorded';
}

function NodeInspector({node}: {node: GPUCoreAnatomyNode}): ReactNode {
  return (
    <section className="gpu-core-anatomy__inspector" aria-live="polite">
      <header>
        <span>Selected node</span>
        <strong>{node.label}</strong>
      </header>
      <div className="gpu-core-anatomy__facts">
        <div><span>Contributor</span><strong>{node.contributorPath}</strong></div>
        <div><span>Operator</span><strong>{node.operator ?? 'unknown'}</strong></div>
        <div><span>Namespace</span><strong>{node.namespace || 'unknown'}</strong></div>
        <div><span>Reads</span><strong>{node.reads.join(', ') || 'none'}</strong></div>
        <div><span>Writes</span><strong>{node.writes.join(', ') || 'none'}</strong></div>
        <div>
          <span>Dispatch</span>
          <strong>
            {describeDispatch(node)}
            {node.invocationCount !== undefined ? `, ${NUMBER_FORMAT.format(node.invocationCount)} invocations` : ''}
          </strong>
        </div>
        <div><span>Source module</span><strong>{node.moduleId ?? 'unknown'}</strong></div>
      </div>
    </section>
  );
}
