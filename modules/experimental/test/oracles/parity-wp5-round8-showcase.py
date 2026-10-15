# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Read-only AST inventory of WP5 constructor/packer calls and scene option schemas.

This is a reachability REVIEW INPUT, not an interprocedural coverage proof.
It preserves exact object expressions (including helpers) and all slider/select
values; it must not count arbitrary scalar/coordinate inputs as a tested mode.
"""
import hashlib
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
PROGRAM=r'''
const fs=require('fs');const path=require('path');const ts=require('typescript');
const root=process.cwd();const folder=path.join(root,'examples/deck/spatial-analysis-showcase/src');
const classNames=new Set(['GPUTerrainHorizon','GPUTerrainCurvature','GPUGeomorphons','GPUPointHorizonProfile','GPUPointHorizonVisibility','GPUSolarIrradiance','GPUSolarPosition','GPUSolarShadowMask','GPUTerrainCastShadow','GPUTerrainCumulativeViewshed','GPUTerrainHeightAboveDrainage','GPUTerrainHydrologicIndices','GPUTerrainLineOfSight','GPUTerrainStreamOrder','GPUTerrainViewshed','GPUTerrainWatersheds','GPULocalDominance','GPUMultiScaleRelief','GPUProfilePeaks','GPUReliefBlend','GPUSimpleLocalRelief','GPUTerrainContours','GPUTerrainCriticalPoints','GPUTerrainFlowField','GPUTerrainPeakSnap','GPUTerrainSpikeRepair','GPUTerrainSummits','GPUTerrainTopographicPosition','GPUTerrainVectorRuggedness','GPUTerrainWeissLandforms','GPUTextureShading']);
const calls=[];const schemas=[];
function files(directory){return fs.readdirSync(directory,{withFileTypes:true}).flatMap(entry=>entry.isDirectory()?files(path.join(directory,entry.name)):[path.join(directory,entry.name)]);}
for(const filename of files(folder).filter(value=>value.endsWith('.ts'))){
 const text=fs.readFileSync(filename,'utf8');const source=ts.createSourceFile(filename,text,ts.ScriptTarget.Latest,true);
 function walk(node){
  if((ts.isNewExpression(node)||ts.isCallExpression(node))&&node.expression){
   const name=node.expression.getText(source);const matched=[...classNames].find(value=>name===value||name.startsWith('get'+value));
   if(matched)calls.push({class:matched,file:path.relative(root,filename),line:source.getLineAndCharacterOfPosition(node.getStart()).line+1,expression:node.getText(source)});
  }
  if(['horizon','landforms','sun-and-shadow','viewshed','terrain-basics','relief-visualization','summits','watersheds','terrain-flow'].some(name=>filename.endsWith('/'+name+'.scene.ts'))&&ts.isObjectLiteralExpression(node)){
   const properties=Object.fromEntries(node.properties.filter(ts.isPropertyAssignment).map(property=>[property.name.getText(source).replace(/['"]/g,''),property.initializer]));
   if(properties.id&&properties.kind){const entry={file:path.relative(root,filename),line:source.getLineAndCharacterOfPosition(node.getStart()).line+1};for(const key of ['id','kind','min','max','step','default','apply','options','disabledWhen','presets'])if(properties[key])entry[key]=properties[key].getText(source);schemas.push(entry);}
  }
  ts.forEachChild(node,walk);
 }walk(source);
}
process.stdout.write(JSON.stringify({calls,schemas}));
'''
result=subprocess.run(['node','-e',PROGRAM],cwd=ROOT,text=True,capture_output=True,check=True)
data=json.loads(result.stdout)
data['generatorSha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
path=Path('/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/handoffs/parity-wp5-round8-showcase.json')
path.write_text(json.dumps(data,indent=2)+'\n')
print(path,len(data['calls']),'calls',len(data['schemas']),'option declarations')
# Focused human-readable exact source excerpts: no claims of transitive closure.
lines=['# WP5 round 8 exact showcase call inventory','',__doc__]
for className in dict.fromkeys(call['class'] for call in data['calls']):
    lines.extend(['',f'## {className}'])
    for call in data['calls']:
        if call['class']==className:lines.extend(['',f"{call['file']}:{call['line']}",'```ts',call['expression'],'```'])
path.with_suffix('.md').write_text('\n'.join(lines)+'\n')
