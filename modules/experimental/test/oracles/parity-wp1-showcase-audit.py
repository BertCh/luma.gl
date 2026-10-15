# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Snapshot constructor, parameter and options-schema expressions with TypeScript's AST.

Read-only provenance audit, not a numerical oracle. Records the exact source text
and SHA256 so the handoff's mode inventory can be checked against scene changes.
UI presentation selectors and upstream input-data choices are retained in schemas;
the handoff distinguishes those from contributor branches and numeric sweeps.
"""
import hashlib
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
EVAL=Path('/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval')
classes=[line[2:].split(':')[0] for line in (EVAL/'prompts/wp1.md').read_text().splitlines() if line.startswith('- GPU')]
script=r'''
const fs=require('node:fs'),path=require('node:path'),ts=require('typescript');
const classes=JSON.parse(process.argv[1]);const results=[];
function visitFiles(directory){for(const entry of fs.readdirSync(directory,{withFileTypes:true})){
 const file=path.join(directory,entry.name);if(entry.isDirectory())visitFiles(file);
 else if(file.endsWith('.ts')){const text=fs.readFileSync(file,'utf8');
 if(!classes.some(name=>text.includes(name))&&!file.endsWith('.scene.ts')&&!text.includes('addChangeOfSupportRecipe'))continue;
 const source=ts.createSourceFile(file,text,ts.ScriptTarget.Latest,true);const expressions=[];
 function visit(node){const value=node.getText(source);let kind;
 if(ts.isNewExpression(node)&&classes.includes(node.expression.getText(source)))kind='constructor';
 if(ts.isCallExpression(node)&&(classes.some(name=>node.expression.getText(source).includes(name))||node.expression.getText(source).includes('addChangeOfSupportRecipe')))kind='parameters-or-recipe';
 if(ts.isPropertyAssignment(node)&&node.name.getText(source)==='options'&&ts.isArrayLiteralExpression(node.initializer))kind='options-schema';
 if(ts.isVariableDeclaration(node)&&/ITERATION_LADDER|CUBE_SLICE_COUNTS|BUCKET_COUNT|MAXIMUM_CLUSTERS|MAXIMUM_PERMUTATIONS/.test(node.name.getText(source)))kind='capacity';
 if(kind)expressions.push({kind,line:source.getLineAndCharacterOfPosition(node.getStart(source)).line+1,text:value});
 ts.forEachChild(node,visit);}
 visit(source);if(expressions.length)results.push({file,expressions});
 }} }
visitFiles('examples/deck/spatial-analysis-showcase/src/scenes');
process.stdout.write(JSON.stringify({versions:{node:process.version,typescript:ts.version},files:results}));
'''
result=json.loads(subprocess.check_output(['node','-e',script,json.dumps(classes)],cwd=ROOT))
for item in result['files']:
 item['sha256']=hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest()
result.update(generator=Path(__file__).name,command=f'{EVAL}/venv/bin/python -I {Path(__file__).relative_to(ROOT)}',classes=classes)
output=EVAL/'audit/parity-wp1-showcase-modes.json';output.write_text(json.dumps(result,indent=2)+'\n');print(output,len(result['files']))
