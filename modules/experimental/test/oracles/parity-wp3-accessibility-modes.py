# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""PySAL access 2SFCA/3SFCA with scene scoring parameters and output combinations.
Float32 costs and parameters, float64 reference. Filter the cost frame by the
inclusive threshold BEFORE three_stage_fca: access otherwise normalizes choice
weights using out-of-range facilities. This aligns the catchment definition,
not the GPU implementation. 3SFCA ratios use the published Wan, Zou & Sternberg
(2012) equation R_j = S_j / sum_i P_i W_ij^2 / sum_k W_ik, evaluated as a NumPy
matrix expression (composite reference for this intermediate output only).
No zero-denominator convention is attributed to access. All facilities in these
cases have positive catchment demand; disconnected origins are filled with zero.
"""
import importlib.metadata
import json
import math
from pathlib import Path
import numpy
import pandas
from access import Access, fca
HERE=Path(__file__).resolve().parent

def main():
    costs=numpy.asarray([[0,100,300,700,1300,2700,5400,numpy.inf],[500,200,0,600,1800,3000,5000,numpy.inf],[900,500,300,0,1200,2500,4700,numpy.inf]],dtype=numpy.float32).astype(float)
    supply=numpy.asarray([1800,850,1200],dtype=float)
    demand=numpy.asarray([1,3,4,8,5,2,7,6],dtype=float)
    demandFrame=pandas.DataFrame({'demand':demand},index=pandas.Index(range(8),name='geoid'))
    supplyFrame=pandas.DataFrame({'supply':supply},index=pandas.Index(range(3),name='geoid'))
    costFrame=pandas.DataFrame([dict(origin=origin,dest=facility,cost=costs[facility,origin]) for facility in range(3) for origin in range(8) if numpy.isfinite(costs[facility,origin])])
    cases=[]
    for decay in ['none','exponential','power']:
        for threshold in [300,2700,5400]:
            settings=dict(threshold=threshold,decay=decay,beta=float(numpy.float32(1/600 if decay=='exponential' else 1.5 if decay=='power' else 0)),minimumCost=60)
            weight=(lambda cost:1.) if decay=='none' else (lambda cost: math.exp(-settings['beta']*cost)) if decay=='exponential' else (lambda cost:max(cost,60)**(-settings['beta']))
            analysis=Access(demandFrame.copy(),'demand',supplyFrame.copy(),'supply',cost_df=costFrame,cost_origin='origin',cost_dest='dest',cost_name='cost')
            cumulative=analysis.weighted_catchment(name='cumulative',max_cost=threshold,normalize=False).iloc[:,0].reindex(range(8)).fillna(0).to_list()
            gravity=analysis.weighted_catchment(name='gravity',max_cost=threshold,weight_fn=weight,normalize=False).iloc[:,0].reindex(range(8)).fillna(0).to_list()
            for method in ['2sfca','3sfca']:
                filtered=costFrame[costFrame.cost<=threshold].copy()
                operation=fca.two_stage_fca if method=='2sfca' else fca.three_stage_fca
                catchment=operation(demandFrame.reset_index(),supplyFrame.copy(),filtered,max_cost=threshold,weight_fn=weight).reindex(range(8)).fillna(0).to_list()
                if method=='2sfca':
                    totals=fca.weighted_catchment(demandFrame.copy(),filtered,max_cost=threshold,loc_index=True,loc_value='demand',weight_fn=weight).reindex(range(3)).to_numpy()
                else:
                    weights=numpy.asarray([[weight(cost) if cost<=threshold else 0 for cost in row] for row in costs])
                    totals=(weights**2)@numpy.divide(demand,weights.sum(axis=0),out=numpy.zeros(8),where=weights.sum(axis=0)>0)
                assert numpy.all(totals>0)
                expected=dict(cumulative=cumulative,gravity=gravity,catchment=catchment,ratios=(supply/totals).tolist())
                cases.append(dict(case=f'showcase-{decay}-{threshold}-{method}',method=method,settings=settings,expected=expected,
                    reference='access.fca.'+('two_stage_fca' if method=='2sfca' else 'three_stage_fca; composite Wan et al. (2012) facility ratios')+'; access.Access.weighted_catchment',
                    tolerances={name:dict(absolute=0 if name=='cumulative' else 1e-6*max(1,max(map(abs,values))),relative=0 if name=='cumulative' else 1e-4) for name,values in expected.items()}))
    fixture=dict(generator=Path(__file__).name,versions={name:importlib.metadata.version(name) for name in ['access','numpy','pandas']},conventions=__doc__,
        input=dict(case='showcase',reference='externally supplied float32 seconds matrix',versions={name:importlib.metadata.version(name) for name in ['access','numpy','pandas']},originCount=8,facilityNodes=[0,2,3],costs=[value if math.isfinite(value) else 'Infinity' for value in costs.flat],supply=supply.tolist(),demand=demand.tolist(),source={}),cases=cases)
    output=HERE/'fixtures/parity-wp3-accessibility-showcase.json'
    output.write_text(json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n')
    print(output.name,len(cases),output.stat().st_size)
if __name__=='__main__':main()
