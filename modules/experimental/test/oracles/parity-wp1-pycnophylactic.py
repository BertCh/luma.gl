# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""R pycno 1.4.1 core, with explicit boundary/kernel/stopping adapters.

The installed pycno:::.pycno.core (called by pycno::pycno) supplies populate,
additive correction, nonnegative clamp and multiplicative correction unchanged
for positive-mass zones. We set r=0 (no damping), replace its mean-padded rook
smoother with a normalized finite-support stencil (rook or box), and instrument
fixed iteration counts and zone-relative stopping. The zero-total multiplicative
0/0 is defined as zero. These are declared API convention adapters, NOT a claim
that unmodified pycno defaults equal this contributor. Store original R function
hash, complete adapter text and R command. No GPU/TS implementation is executed.
L profile: full cells and zone totals, 1e-6 scale + 1e-4 relative; exact stop count.
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
EVAL = Path('/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval')
RSCRIPT = EVAL / 'refenv/bin/Rscript'
R_CODE = r'''
library(jsonlite)
library(pycno)
args <- commandArgs(TRUE)
inputs <- fromJSON(args[1], simplifyVector=FALSE)
original <- paste(deparse(pycno:::.pycno.core), collapse="\n")
outputs <- lapply(inputs, function(item) {
  zones <- matrix(unlist(item$zones)+1L, nrow=item$height, byrow=TRUE)
  pops <- unlist(item$totals)
  # R package's finite support adapter; absent/dummy zero zones do not smooth.
  support <- matrix(pops[zones] > 0, nrow=nrow(zones))
  stencil <- if(item$kernel == "rook") rbind(c(-1,0),c(1,0),c(0,-1),c(0,1)) else as.matrix(expand.grid(-1:1,-1:1))
  smoothAdapter <- function(x) {
    numerator <- denominator <- x*0
    for (offset in seq_len(nrow(stencil))) {
      rows <- which(seq_len(nrow(x))+stencil[offset,1] >= 1 & seq_len(nrow(x))+stencil[offset,1] <= nrow(x))
      columns <- which(seq_len(ncol(x))+stencil[offset,2] >= 1 & seq_len(ncol(x))+stencil[offset,2] <= ncol(x))
      adjacentRows <- rows+stencil[offset,1]; adjacentColumns <- columns+stencil[offset,2]
      numerator[rows,columns] <- numerator[rows,columns] + x[adjacentRows,adjacentColumns]*support[adjacentRows,adjacentColumns]
      denominator[rows,columns] <- denominator[rows,columns] + support[adjacentRows,adjacentColumns]
    }
    result <- numerator/pmax(denominator,1)
    result[denominator == 0] <- x[denominator == 0]
    result
  }
  code <- original
  code <- sub('x <- populate(zones, pops)', 'smooth2D <- smoothAdapter; x <- populate(zones, pops); initial <- x; iteration <- 0L; converged <- FALSE',code,fixed=TRUE)
  code <- sub('old.x <- x','old.x <- x; iteration <- iteration+1L',code,fixed=TRUE)
  code <- sub('correct <- pops[item]/sum(x[zone.set])','correct <- if(pops[item] == 0) 0 else pops[item]/sum(x[zone.set])',code,fixed=TRUE)
  code <- sub('if (max(abs(old.x - x)) < stopper)', 'converged <- item$tolerance >= 0 && max(abs(old.x-x)/pmax(initial,1e-30)) <= item$tolerance; if (converged || iteration >= item$iterations)',code,fixed=TRUE)
  adapted <- eval(parse(text=code))
  body(adapted)[[length(body(adapted))]] <- quote(return(list(mass=as.vector(t(x)), iterations=iteration, converged=as.integer(converged))))
  if(item$iterations == 0) {
    values <- pops[zones]/tabulate(zones,nbins=length(pops))[zones]
    result <- list(mass=as.vector(t(matrix(values,nrow=nrow(zones)))),iterations=0,converged=0)
  } else result <- adapted(zones,pops,r=0,verbose=FALSE)
  result
})
write_json(list(versions=list(R=R.version.string,pycno=as.character(packageVersion('pycno')),jsonlite=as.character(packageVersion('jsonlite'))), original=original, results=outputs),args[2],digits=17,auto_unbox=TRUE)
'''

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent/'fixtures'); args=parser.parse_args()
    zones = np.tile(np.repeat(np.arange(3), 4), (8,1))
    cases=[]
    for kernel in ['rook','box']:
        for iterations in [0,2,8,32,128,512]:
            cases.append(dict(name=f'{kernel}-{iterations}',width=12,height=8,zones=zones.ravel().tolist(),totals=[8,256,32],kernel=kernel,iterations=iterations,tolerance=-1))
        cases.append(dict(name=f'{kernel}-convergence',width=12,height=8,zones=zones.ravel().tolist(),totals=[8,256,32],kernel=kernel,iterations=512,tolerance=float(np.float32(.001))))
    sourcePath=Path(__file__).parent/'fixtures/parity-wp1-areal-interpolation.json'
    source=json.loads(sourcePath.read_text())['cases'][0]
    crop=np.array(source['sourceZones']).reshape(source['height'],source['width'])[48:72,48:72]
    ids=np.unique(crop); ids=ids[ids < source['sourceZoneCount']]
    totals=np.array(source['sourceValues'])[ids].astype(float)
    # Drop zero-mass and exterior cells from the support, retaining a dummy zero zone.
    ids=ids[totals>0]; totals=totals[totals>0]
    mapped=np.full(crop.shape,len(ids),dtype=int)
    for index,zone in enumerate(ids): mapped[crop==zone]=index
    cases.append(dict(name='chicago-tract-crop-rook-32',width=24,height=24,zones=mapped.ravel().tolist(),totals=totals.tolist()+[0],kernel='rook',iterations=32,tolerance=-1,provenance={'sourceFixtureSha256':hashlib.sha256(sourcePath.read_bytes()).hexdigest(),'crop':[48,72,48,72],'originalZoneIds':ids.tolist(),'source':source['provenance']}))
    # Dummy cells must be outside the GPU support, not a valid zero-total zone.
    with tempfile.TemporaryDirectory(prefix='parity-wp1-') as directory:
        directory=Path(directory); script=directory/'reference.R'; inputs=directory/'input.json'; output=directory/'output.json'
        script.write_text(R_CODE); inputs.write_text(json.dumps(cases))
        subprocess.run([str(RSCRIPT),str(script),str(inputs),str(output)],check=True)
        result=json.loads(output.read_text())
    for case, expected in zip(cases,result['results']):
        case['expected']=expected
        case['reference']='R pycno:::.pycno.core via explicit finite-support/kernel/stopping adapter'
        if case['totals'][-1]==0:
            dummy=len(case['totals'])-1
            case['zones']=[4294967295 if zone==dummy else zone for zone in case['zones']]
            case['totals']=case['totals'][:-1]
        mass=np.array(expected['mass'])
        case['zoneSums']=[float(mass[np.array(case['zones'])==zone].sum()) for zone in range(len(case['totals']))]
        assert np.all(mass>=0)
        np.testing.assert_allclose(case['zoneSums'],case['totals'],atol=1e-9,rtol=1e-12)
    fixture=dict(generator=Path(__file__).name, command=f'{EVAL}/venv/bin/python -I {Path(__file__).relative_to(ROOT)}', rCommand=f'{RSCRIPT} reference.R input.json output.json',versions=result['versions'],originalFunctionSha256=hashlib.sha256(result['original'].encode()).hexdigest(),adapter=R_CODE,cases=cases)
    args.output_dir.mkdir(exist_ok=True,parents=True); path=args.output_dir/'parity-wp1-pycnophylactic.json'; path.write_text(json.dumps(fixture,separators=(',',':'))+'\n'); assert path.stat().st_size<1500000
    print(path,len(cases),path.stat().st_size)
if __name__=='__main__': main()
