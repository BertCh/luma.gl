# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Composite weighted Gi* reference from the published Getis-Ord equation.

Citation: Ord & Getis (1995), Geographical Analysis 27(4), 286-306; equation
reproduced at https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/h-how-hot-spot-analysis-getis-ord-gi-spatial-stati.html
esda.G_Local.calc assumes binary cardinality for its analytic variance, so it is
NOT used to certify nonuniform weights. NumPy evaluates the published weighted
finite-population mean/variance, independently checked against all 720 permutations
of six observations. SciPy sparse products, Kendall and normal survival, and
statsmodels FDR provide the remaining independent composition. The original cited
category adapter is reused. Uploaded weights are f32; all reference math is f64.
"""
import importlib.util,itertools,json
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse import csr_matrix,kron
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau,norm
from statsmodels.stats.multitest import multipletests
DIRECTORY=Path(__file__).parent
spec=importlib.util.spec_from_file_location('reference',DIRECTORY/'parity-wp1-emerging-hot-spots.py');reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
spec=importlib.util.spec_from_file_location('transport',DIRECTORY/'parity-wp1-round5.py');transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
values=np.array([1,2,3,5,8,13]);weights=np.array([1,.5,.25,0,2,1])
draws=np.array([weights@values[list(order)] for order in itertools.permutations(range(6))])
np.testing.assert_allclose(draws.mean(),weights.sum()*values.mean(),atol=1e-12)
np.testing.assert_allclose(draws.var(),values.var()*(6*(weights@weights)-weights.sum()**2)/5,atol=1e-12)
random=np.random.default_rng(8204);cases=[]
positions=np.array([[0,0],[1,0],[3,0],[0,2],[2,3],[5,4]],dtype=float);distances=cdist(positions,positions)
for slices in [12,52,73]:
 cube=random.poisson(30,(6,slices));cube[0]+=np.arange(slices);cube[5]+=np.arange(slices)[::-1]
 for kind in ['inverse-distance','kernel']:
  spatial=np.zeros((6,6));eligible=(distances>0)&(distances<4)
  spatial[eligible]=1/distances[eligible] if kind=='inverse-distance' else 1-distances[eligible]/4
  spatial=spatial.astype(np.float32).astype(float);spatialMatrix=csr_matrix(spatial)
  temporal=csr_matrix(np.tril(np.ones((slices,slices)))-np.tril(np.ones((slices,slices)),-3))
  operator=kron(csr_matrix(spatial+np.eye(6)),temporal,format='csr')
  population=cube.ravel().astype(float);size=len(population)
  sums=np.asarray(operator.sum(axis=1)).ravel();squares=np.asarray(operator.multiply(operator).sum(axis=1)).ravel()
  scores=(operator@population-population.mean()*sums)/(population.std()*np.sqrt((size*squares-sums*sums)/(size-1)))
  probabilities=2*norm.sf(abs(scores));scoreCube=scores.astype(np.float32).reshape(cube.shape)
  for method in ['none','benjamini-hochberg','benjamini-yekutieli']:
   adjusted=probabilities if method=='none' else multipletests(probabilities,method={'benjamini-hochberg':'fdr_bh','benjamini-yekutieli':'fdr_by'}[method])[1]
   for confidence in [.9,.95,.99]:
    expected=dict(giZScores=scores,pValues=probabilities,adjustedPValues=adjusted,globalStatistics=[size,population.mean(),population.var(),population.std()],trendS=[],trendZ=[],trendP=[],category=[],hotSliceCount=[],coldSliceCount=[],lastValidSlice=[])
    for cell in range(6):
     series=scoreCube[cell].astype(float);pairCount=slices*(slices-1)//2;counts=np.unique(series,return_counts=True)[1];tiedPairs=np.sum(counts*(counts-1)//2)
     statistic=int(round(kendalltau(np.arange(slices),series,method='asymptotic').statistic*np.sqrt(pairCount*(pairCount-tiedPairs))))
     variance=(slices*(slices-1)*(2*slices+5)-np.sum(counts*(counts-1)*(2*counts+5)))/18
     trend=(statistic-np.sign(statistic))/np.sqrt(variance) if variance>0 else 0;probability=2*norm.sf(abs(trend))
     significant=adjusted.reshape(cube.shape)[cell]<=float(np.float32(1-confidence));states=np.where(significant,np.sign(series),0).astype(int)
     for key,value in dict(trendS=statistic,trendZ=trend,trendP=probability,category=reference.classify(states,trend,probability),hotSliceCount=int((states==1).sum()),coldSliceCount=int((states==-1).sum()),lastValidSlice=slices-1).items():expected[key].append(value)
    cases.append(dict(name=f'weighted-{kind}-{slices}-{method}-{confidence}',reference='composite Getis-Ord weighted equation (Ord/Getis 1995), exhaustive six-value permutation moment check; scipy Kendall/norm; statsmodels FDR',n=size,width=3,height=2,sliceCount=slices,values=cube.ravel(),mask=[1]*6,temporalWindow=2,multipleTesting=method,finalSlice='require',confidence=confidence,offsets=spatialMatrix.indptr,neighbors=spatialMatrix.indices,weights=spatialMatrix.data,expected=expected))
transport.writeFixture(DIRECTORY/'fixtures','emerging-weighted',dict(cases=cases))
path=DIRECTORY/'fixtures/parity-wp1-emerging-weighted.json';wrapper=json.loads(path.read_text());wrapper.update(generator=Path(__file__).name,command='/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python -I modules/experimental/test/oracles/parity-wp1-emerging-weighted.py',versions={**wrapper['versions'],'statsmodels':reference.statsmodels.__version__});path.write_text(json.dumps(wrapper,separators=(',',':'))+'\n')
