// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// Reference-only bridge: WebR is installed in the locked reference venv, never the repo.
import {readFileSync, writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
const [runtime, input, output] = process.argv.slice(2);
const {WebR} = await import(pathToFileURL(`${runtime}/node_modules/webr/dist/webr.mjs`));
const webR = new WebR();
await webR.init();
await webR.installPackages(['spatstat.explore', 'jsonlite']);
await webR.FS.writeFile('/input.json', new Uint8Array(readFileSync(input)));
const result = await webR.evalRString(`
  library(spatstat.explore)
  input <- jsonlite::fromJSON('/input.json', simplifyVector=FALSE)
  results <- lapply(input, function(scene) {
    if (scene$name == 'rectangle') {
      W <- owin(xrange=unlist(scene$bounds)[c(1,3)], yrange=unlist(scene$bounds)[c(2,4)])
    } else {
      W <- owin(poly=lapply(scene$rings, function(ring) {
        vertices <- matrix(unlist(ring), ncol=2, byrow=TRUE)
        list(x=vertices[,1], y=vertices[,2])
      }))
    }
    points <- matrix(unlist(scene$points), ncol=2, byrow=TRUE)
    X <- ppp(points[,1], points[,2], window=W, checkdup=FALSE)
    curves <- lapply(c(20L,30L,60L), function(count) {
      radii <- seq(0,scene$maximumDistance,length.out=count+1)
      K <- Kest(X, r=radii, correction=c('none','border','isotropic'), nlarge=Inf)
      # A separate border-only call selects Kborder.engine, whose exact censoring
      # thresholds match the reduced-sample definition. The generic Kount path
      # bins both pair and censor distances and shifts border eligibility.
      border <- Kest(X, r=radii, correction='border', nlarge=Inf)
      list(radiusCount=count, none=unname(K$un[-1]), border=unname(border$border[-1]), isotropic=unname(K$iso[-1]))
    })
    list(name=scene$name, area=area.owin(W), nearest=unname(nndist(X)), border=unname(bdist.points(X)),
      guardRatio=unname(clarkevans(X, correction='guard', clipregion=erosion(W, 4))),
      donnellyRatio=if(scene$name=='rectangle') unname(clarkevans(X, correction='Donnelly')) else NA_real_, curves=curves)
  })
  jsonlite::toJSON(list(versions=list(R=R.version.string, spatstat.explore=as.character(packageVersion('spatstat.explore')),
    spatstat.geom=as.character(packageVersion('spatstat.geom'))), results=results), auto_unbox=TRUE, digits=17, na='null')
`);
writeFileSync(output, result + '\n');
await webR.close();
