function e(e,t={}){let n=[],r=null,i=()=>{r&&n.push({id:r.id,title:r.title,body:r.lines.join(`
`).trim(),...t[r.id]})};for(let t of e.split(`
`)){let e=/^## (.+?)(?:\s*\{#([\w-]+)\})?\s*$/.exec(t);if(e){i();let t=e[1];r={id:e[2]??t.toLowerCase().replace(/[^a-z0-9]+/g,`-`).replace(/^-+|-+$/g,``),title:t,lines:[]}}else r&&r.lines.push(t)}return i(),n}export{e as t};