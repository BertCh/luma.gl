# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""SpatialEpi likelihoods on explicitly enumerated space/time windows.

Independent scipy distances and stable numpy argsort enumerate nearest-prefix
windows; circle only ends after a distance tie. Spatial population fractions use
all buckets; cylinders enumerate consecutive time runs. R SpatialEpi supplies
EVERY candidate likelihood: poissonLogLkhd/binomialLogLkhd. Low uses complement
of the 2x2 table; both takes max of high and low. The permutation model supplies
fixed-margin expectations to the Poisson likelihood (Kulldorff et al. 2005).

Supplied null tables: numpy multinomial (Poisson), multivariate_hypergeometric
(Bernoulli), scipy.stats.random_table (space-time permutation). GPU test replaces
ONLY the random-draw graph node with these tables; evaluation, selection and p
kernels run unchanged. This validates conditional Monte Carlo inference exactly,
NOT Philox RNG or sampling correctness. Every candidate value is retained in the
fixture, while the public GPU outputs expose per-center and global maxima only.
SpatialEpi::kulldorff is additionally run for the spatial nearest/high cases.
Its Bernoulli sampler is multinomial, so never use that sampler as fixed-population
parity. No copied TypeScript/WGSL numerical algorithm. Profile S/P.
"""
import argparse
import base64
import zlib
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from scipy.stats import random_table, chi2_contingency
from scipy.special import xlogy

ROOT=Path(__file__).resolve().parents[4]
EVAL=Path('/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval')
R_CODE=r'''
library(SpatialEpi); library(jsonlite)
args <- commandArgs(TRUE); input <- fromJSON(args[1],simplifyVector=FALSE)
results <- lapply(input,function(item) {
  candidate <- do.call(rbind,lapply(item$tables,unlist))
  likelihood <- if(item$model == 'bernoulli') SpatialEpi:::binomialLogLkhd else SpatialEpi:::poissonLogLkhd
  values <- apply(candidate,1,function(row) {
    observed <- row[1]; population <- row[2]; totalPopulation <- row[3]; totalCases <- row[4]
    if(population <= 0 || population >= totalPopulation) return(0)
    high <- likelihood(observed,population,totalPopulation,totalCases)
    low <- likelihood(totalCases-observed,totalPopulation-population,totalPopulation,totalCases)
    if(item$direction == 'high') high else if(item$direction == 'low') low else max(high,low)
  })
  direct <- NULL
  if(item$timeBuckets == 1 && item$direction == 'high' && item$shape == 'nearest') {
    set.seed(8201)
    direct <- kulldorff(matrix(unlist(item$positions),ncol=2,byrow=TRUE),unlist(item$cases),unlist(item$population),if(item$model=='poisson') unlist(item$population)*sum(unlist(item$cases))/sum(unlist(item$population)) else NULL,item$fraction,19,1,FALSE)
  }
  list(values=values,direct=direct)
})
write_json(list(versions=list(R=R.version.string,SpatialEpi=as.character(packageVersion('SpatialEpi')),jsonlite=as.character(packageVersion('jsonlite'))),results=results),args[2],digits=17,auto_unbox=TRUE)
'''

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'fixtures'); parser.add_argument('--showcase',action='store_true'); args=parser.parse_args()
    positions=np.array([[0,0],[2,0],[0,2],[6,1],[11,4],[17,2],[23,7]],dtype=np.float32)
    distance=cdist(positions,positions); order=np.argsort(distance,axis=1,kind='stable')
    random=np.random.default_rng(8201); cases=[]; requests=[]
    configurations = [('poisson',26,baseline) for baseline in ['population','uniform','independence']] if args.showcase else [(model,buckets,'population') for model in ['poisson','bernoulli','space-time-permutation'] for buckets in ([1,3] if model!='space-time-permutation' else [3])]
    for model,buckets,baselineKind in configurations:
      if True:
       observed=np.array([[12,2,1],[9,1,2],[1,1,2],[2,3,1],[1,2,6],[1,2,3],[2,3,2]],dtype=int)[:,:buckets]
       baseline=np.array([[40,50,60],[45,55,65],[60,50,40],[50,55,60],[65,50,55],[60,65,55],[55,60,50]],dtype=float)[:,:buckets]
       if args.showcase:
        observed=np.random.default_rng(8207).poisson(2,(7,26)); observed[:2,8:14]+=6
        baseline=np.repeat(np.array([40,45,60,50,65,60,55],dtype=float)[:,None],26,axis=1)
        if baselineKind=='uniform': baseline=np.ones_like(observed,dtype=float)
        if baselineKind=='independence': baseline=np.outer(observed.sum(axis=1),observed.sum(axis=0))/observed.sum()
        baseline=baseline.astype(np.float32).astype(float)
       maximumTimeBuckets=6 if args.showcase else 2
       total=int(observed.sum()); replicates=19
       if model=='poisson':
        population=baseline; draws=random.multinomial(total,population.ravel()/population.sum(),replicates).reshape(replicates,7,buckets)
       elif model=='bernoulli':
        population=baseline; draws=random.multivariate_hypergeometric(population.ravel().astype(int),total,replicates).reshape(replicates,7,buckets)
       else:
        population=np.outer(observed.sum(axis=1),observed.sum(axis=0))/total
        draws=random_table.rvs(observed.sum(axis=1),observed.sum(axis=0),size=replicates,random_state=random)
       for shape in ['circle','nearest']:
        for direction in (['high'] if args.showcase else ['high','low','both']):
         fraction=.5; windows=[]
         for center,neighbors in enumerate(order):
          for count in range(1,8):
           members=neighbors[:count]
           if population[members].sum()>fraction*population.sum(): break
           if shape=='circle' and count<7 and distance[center,neighbors[count-1]]==distance[center,neighbors[count]]: continue
           for first in range(buckets):
            for last in range(first,min(buckets,first+maximumTimeBuckets)):
             windows.append(dict(center=center,count=count,first=first,last=last,members=members.tolist(),radius=float(distance[center,members[-1]])))
         tables=[]
         for counts in [observed,*draws]:
          for window in windows:
           selection=np.ix_(window['members'],range(window['first'],window['last']+1))
           tables.append([int(counts[selection].sum()),float(population[selection].sum()),float(population.sum()),total])
         scene=dict(name=f'{model}-{buckets}-{shape}-{direction}'+(f'-{baselineKind}' if args.showcase else ''),maximumTimeBuckets=maximumTimeBuckets,baselineKind=baselineKind,reference='SpatialEpi::poissonLogLkhd/binomialLogLkhd + scipy window and conditional-null adapters',model=model,timeBuckets=buckets,positions=positions.ravel().tolist(),cases=observed.ravel().tolist(),baseline=population.ravel().tolist(),controls=(baseline-observed).astype(int).ravel().tolist(),replicates=draws.reshape(replicates,-1).tolist(),shape=shape,direction=direction,fraction=fraction,windows=windows)
         cases.append(scene); requests.append(dict(model=model,timeBuckets=buckets,tables=tables,direction=direction,shape=shape,positions=scene['positions'],cases=scene['cases'],population=scene['baseline'],fraction=fraction))
    with tempfile.TemporaryDirectory(prefix='parity-wp1-') as directory:
      directory=Path(directory); script=directory/'reference.R'; inputs=directory/'input.json'; output=directory/'output.json'
      script.write_text(R_CODE); inputs.write_text(json.dumps(requests))
      subprocess.run([str(EVAL/'refenv/bin/Rscript'),str(script),str(inputs),str(output)],check=True)
      result=json.loads(output.read_text())
    for scene,expected,request in zip(cases,result['results'],requests):
      rawValues=expected['values']
      boundaryAdapters=0
      for index,value in enumerate(rawValues):
        if not isinstance(value,(int,float)):
          observed,population,totalPopulation,totalCases=request['tables'][index]
          expectedCount=population*totalCases/totalPopulation
          eligible=(scene['direction']=='both' or (scene['direction']=='high' and observed>expectedCount) or (scene['direction']=='low' and observed<expectedCount))
          if scene['model']=='bernoulli':
            table=np.array([[observed,population-observed],[totalCases-observed,totalPopulation-population-totalCases+observed]])
            statistic=chi2_contingency(table,correction=False,lambda_='log-likelihood').statistic/2
          else:
            statistic=xlogy(observed,observed/expectedCount)+xlogy(totalCases-observed,(totalCases-observed)/(totalCases-expectedCount))
          rawValues[index]=float(statistic) if eligible else 0.0
          boundaryAdapters+=1
      scene['zeroCellScipyAdapters']=boundaryAdapters
      values=np.array(rawValues,dtype=float).reshape(20,-1); assert np.all(np.isfinite(values))
      maxima=values.max(axis=1); primary=int(values[0].argmax()); window=scene['windows'][primary]
      scene['expected']=dict(candidateLikelihoods=values.tolist(),maxima=maxima.tolist(),zoneStatistics=[float(values[0,[index for index,window in enumerate(scene['windows']) if window['center']==center]].max(initial=0)) for center in range(7)],indices=[window['center'],window['count'],window['first'],window['last']],members=window['members'],observed=request['tables'][primary][0],expected=request['tables'][primary][1]*sum(scene['cases'])/sum(scene['baseline']),radius=window['radius'],pValue=float((1+(maxima[1:]>=maxima[0]).sum())/20))
      scene['expected']['optimalWindows']=[dict(indices=[candidate['center'],candidate['count'],candidate['first'],candidate['last']],members=candidate['members'],observed=request['tables'][index][0],expected=request['tables'][index][1]*sum(scene['cases'])/sum(scene['baseline']),radius=candidate['radius']) for index,candidate in enumerate(scene['windows']) if values[0,index]==maxima[0]]
      if expected['direct']:
        direct=expected['direct']; np.testing.assert_allclose(maxima[0],direct['most.likely.cluster']['log.likelihood.ratio'],rtol=1e-12,atol=1e-10)
        scene['directKulldorff']=direct
    fixture=dict(generator=Path(__file__).name,command=f'{EVAL}/venv/bin/python -I {Path(__file__).relative_to(ROOT)}'+(' --showcase' if args.showcase else ''),rCommand=f'{EVAL}/refenv/bin/Rscript reference.R input.json output.json',versions={**result['versions'],'numpy':np.__version__,'scipy':scipy.__version__},seed=8201,adapter=R_CODE,installation=dict(lock=str(EVAL/'venv.lock'),command=f'R_MAKEVARS_USER=/private/tmp/parity-wp1-r8-Makevars {EVAL}/refenv/bin/Rscript -e '+"'options(repos=c(CRAN=\"https://cloud.r-project.org\")); install.packages(\"SpatialEpi\",lib=.libPaths()[1],Ncpus=2)'",makevars=Path('/private/tmp/parity-wp1-r8-Makevars').read_text(),compiler='Apple clang version 15.0.0 (clang-1500.3.9.4)',log='/private/tmp/parity-wp1-r8-install-cxx17.log'),cases=cases)
    args.output_dir.mkdir(exist_ok=True,parents=True)
    if args.showcase:
      fixture=dict(generator=fixture['generator'],command=fixture['command'],versions=fixture['versions'],encoding='zlib-base64-json',sha256=hashlib.sha256(json.dumps(fixture,separators=(',',':')).encode()).hexdigest(),data=base64.b64encode(zlib.compress(json.dumps(fixture,separators=(',',':')).encode(),9)).decode())
    path=args.output_dir/('parity-wp1-scan-showcase.json' if args.showcase else 'parity-wp1-scan.json'); path.write_text(json.dumps(fixture,separators=(',',':'))+'\n'); assert path.stat().st_size<1500000; print(path,len(cases),path.stat().st_size)
if __name__=='__main__': main()
