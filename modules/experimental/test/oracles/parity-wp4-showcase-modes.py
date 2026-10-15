# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reproduce the Round 8 call-site/options audit with the installed TypeScript AST.

This is provenance, not a numerical reference. Excludes scene code-snippet
strings; records real constructors and parameter packer calls, including helpers.
Each associated scene option schema is retained verbatim, with source hashes,
so finite UI alternatives and numeric domains can be independently reviewed.
"""
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
SCRIPT=r'''
const ts=require('typescript'),fs=require('fs'),path=require('path'),crypto=require('crypto');
const names=['ParticleAdvection','ChangeDetection','CostDistance','CostDistancePath','LineIntegralConvolution','RasterConnectedComponents','RasterDenseComponents','Streamlines','DistanceField','PolygonRasterization','RasterZonalStatistics','Isobands','RasterArithmetic','RasterExtremaPyramid','IsobandRings','Isolines','RasterCellStatistics','RasterJoin','RasterPatchMetrics','RasterSampling','RasterSieve','RasterStatistics','WeightedOverlay'].map(name=>'GPU'+name);
const records=Object.fromEntries(names.map(name=>[name,[]]));const sources={};const schemas={};
function digest(text){return crypto.createHash('sha256').update(text).digest('hex');}
function walk(directory){for(const file of fs.readdirSync(directory)){const filename=path.join(directory,file);if(fs.statSync(filename).isDirectory()){walk(filename);continue;}if(!filename.endsWith('.ts'))continue;
 const contents=fs.readFileSync(filename,'utf8');const source=ts.createSourceFile(filename,contents,ts.ScriptTarget.Latest,true);
 function visit(node){
  if((ts.isNewExpression(node)||ts.isCallExpression(node))){const expression=node.expression.getText(source);const name=names.find(name=>expression===name||expression==='get'+name+'ParameterValues'||expression==='get'+name+'Uint32ParameterValues');if(name){records[name].push({file:filename,line:source.getLineAndCharacterOfPosition(node.getStart()).line+1,expression:node.getText(source)});sources[filename]=digest(contents);}}
  if(filename.endsWith('.scene.ts')&&ts.isObjectLiteralExpression(node)) {const members=node.properties.filter(ts.isPropertyAssignment);if(members.some(member=>member.name.getText(source)==='kind')&&members.some(member=>member.name.getText(source)==='id')){(schemas[filename]??=[]).push(node.getText(source));sources[filename]=digest(contents);}}
  ts.forEachChild(node,visit);
 }visit(source);
}}
walk('examples/deck/spatial-analysis-showcase/src');
const usedSchemas={};for(const calls of Object.values(records))for(const call of calls){const filename=call.file.replace(/\.compute\.ts$/,'.scene.ts');if(schemas[filename])usedSchemas[filename]=schemas[filename];}
fs.writeFileSync('modules/experimental/test/oracles/parity-wp4-showcase-modes.json',JSON.stringify({conventions:'Actual AST calls; display snippets excluded. Finite mode coverage is distinguished from exhaustive slider/camera/data Cartesian products in handoff.',sources,calls:records,optionSchemas:usedSchemas},null,2)+'\n');
console.log(names.length+' classes; '+Object.values(records).flat().length+' calls; '+Object.keys(usedSchemas).length+' scene option schemas');
'''
subprocess.run(['node','-e',SCRIPT],cwd=ROOT,check=True)
