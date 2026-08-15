"""Self-contained interactive HTML rendering of the memory-op graph.

One file, no external resources: the graph JSON is embedded and rendered with
vanilla JS + inline SVG. Lanes (one per trajectory, x = agent step) hold
extraction (▲) and recall (▼) ops; edges link each memory's creation to every
recall that delivered it, across lanes, and dashed lifecycle edges link a
memory's creation → updates → deletion. Memories recalled but never created
in the loaded set anchor a dashed ghost ▲ at the left edge of their first
recall's lane. Consecutive identical recall injections collapse into ×n spans;
consecutive history rewrites collapse into shaded bands (translucent with a
dashed border when they span nearly the whole lane). Ops sharing one (lane, kind, step) dodge
sideways instead of stacking; a/p/r count labels stagger between two heights;
create→use edges stay faint until hovered or selected (their arrowhead only
appears then). Step-less ops park in a gutter right of the last step; a tick
marks each lane's own last step. Zoom controls rescale the step axis; lane
labels stay pinned while scrolling. Selecting a memory outlines the recall
spans that delivered it.
"""

from __future__ import annotations

import html
import json
import re

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { --fg:#1c2430; --muted:#667085; --line:#d6dbe3; --bg:#ffffff; --panel:#f6f8fa;
          --extract:#2e9e5b; --recall:#3567d6; --reject:#98a2b3; --flag:#d64545; }
  * { box-sizing: border-box; }
  body { margin:0; font:13px/1.45 -apple-system, "Helvetica Neue", Arial, sans-serif; color:var(--fg); background:var(--bg); }
  header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex; flex-wrap:wrap; gap:8px 18px; align-items:center; }
  header h1 { font-size:15px; margin:0; }
  header .stats { color:var(--muted); }
  header label { color:var(--muted); user-select:none; cursor:pointer; }
  header input[type=checkbox] { vertical-align:-2px; }
  header .zoom { display:inline-flex; gap:4px; align-items:center; color:var(--muted); }
  header .zoom button { padding:1px 9px; border:1px solid var(--line); background:#fff; border-radius:4px; cursor:pointer; font:inherit; color:var(--fg); }
  header .zoom button:hover { background:var(--panel); }
  #memfilter { padding:2px 6px; border:1px solid var(--line); border-radius:4px; }
  #legend { border-bottom:1px solid var(--line); padding:4px 16px 6px; background:var(--panel); }
  #legend summary { cursor:pointer; color:var(--muted); font-size:12px; }
  #legend .note { color:var(--muted); margin:4px 0 2px; }
  #legend .groups { display:grid; grid-template-columns:repeat(auto-fit,minmax(255px,1fr)); gap:4px 28px; padding:4px 0 2px; }
  #legend .group h4 { margin:4px 0 2px; font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); font-weight:600; }
  #legend .group h4 span { text-transform:none; letter-spacing:0; font-weight:400; }
  #legend .item { display:flex; align-items:center; gap:7px; margin:2.5px 0; }
  #legend .item svg { flex:0 0 auto; }
  #main { display:flex; min-height:calc(100vh - 110px); }
  #canvas { flex:1; overflow:auto; padding:8px; }
  #sidebar { width:360px; border-left:1px solid var(--line); background:var(--panel); display:flex; flex-direction:column; }
  #sidebar h2 { font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:10px 12px 4px; }
  #memlist { list-style:none; margin:0; padding:0 6px; overflow-y:auto; max-height:38vh; }
  #memlist li { display:flex; align-items:center; padding:4px 8px; border-radius:5px; cursor:pointer; }
  #memlist li:hover { background:#e9eef5; }
  #memlist li.sel { background:#dbe6fb; }
  #memlist .badge { display:inline-block; min-width:10px; height:10px; border-radius:5px; margin-right:6px; flex:0 0 auto; }
  #memlist .mkey { flex:1 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  #memlist .st { color:var(--muted); font-size:11px; margin-left:6px; flex:0 0 auto; }
  #memlist li.placeholder { cursor:default; color:var(--muted); font-style:italic; }
  #memlist li.placeholder:hover { background:none; }
  #memlist .act { font-weight:600; }
  #memhdr button { font-size:10px; margin-left:6px; padding:0 6px; border:1px solid var(--line); background:#fff; border-radius:4px; cursor:pointer; color:var(--muted); }
  #detail { flex:1; margin:8px; padding:10px; background:#fff; border:1px solid var(--line); border-radius:6px; overflow:auto; }
  #detail h3 { margin:0 0 6px; font-size:13px; word-break:break-all; }
  #detail .kv { color:var(--muted); margin:2px 0; }
  #detail pre { background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:8px; font-size:11px; white-space:pre-wrap; word-break:break-word; max-height:40vh; overflow:auto; }
  #detail .chip { display:inline-block; padding:0 6px; border-radius:8px; font-size:11px; margin-right:4px; background:#eef2f7; }
  #detail h4 { margin:8px 0 3px; font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
  #detail .mrow { margin:3px 0 6px; }
  #detail .mrow .vline { color:var(--muted); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:1px; }
  #detail .mrow .meta { color:var(--muted); font-size:11px; margin-left:2px; }
  #detail .flagbox { background:#fdeeee; border:1px solid #f3c6c6; color:var(--flag); border-radius:4px; padding:4px 8px; margin:4px 0; font-size:11px; }
  #detail .memlink { cursor:pointer; }
  #detail .memlink:hover { text-decoration:underline; }
  #detail .rawjson summary { cursor:pointer; color:var(--muted); font-size:11px; margin-top:6px; }
  footer { border-top:1px solid var(--line); padding:6px 16px; color:var(--muted); }
  footer details { display:inline-block; }
  .opnode { cursor:pointer; }
  .opnode.located path, .opnode.located rect { stroke:#101828; stroke-width:2.5; }
  .ghostnode { cursor:pointer; }
  .ghostnode.located path { stroke-width:3; }
  .op-recall.hot rect, .op-recall.hot path { stroke:#101828; stroke-width:1.1; fill-opacity:1; filter:drop-shadow(0 1px 2.5px rgba(16,24,40,.4)); }
  #guide { pointer-events:none; }
  svg text { user-select:none; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="stats" id="stats"></span>
  <label><input type="checkbox" id="f-extract" checked> extractions</label>
  <label><input type="checkbox" id="f-recall" checked> recalls</label>
  <label><input type="checkbox" id="f-edges" checked> edges</label>
  <label><input type="checkbox" id="f-rejonly" checked title="hide extractions that changed nothing (no candidates and no deletions)"> hide quiet ▲</label>
  <span class="zoom">zoom <button id="z-out" title="zoom out (−)">−</button><button id="z-fit" title="fit width (0)">fit</button><button id="z-in" title="zoom in (+)">+</button></span>
  <input id="memfilter" placeholder="filter memories…" size="18">
</header>
<details id="legend" open>
  <summary>legend — what each symbol means</summary>
  <div class="note">Each lane is one trajectory, x-axis = agent step (per lane — a cross-lane edge may point left and is still causal). <b>Click a lane (or any op node) to inspect the memory actions at that step</b> in the Memories panel; hover a memory or edge to preview its path. +/− zoom, 0 fits width, ←/→ nudge the step inspector, Esc clears.</div>
  <div class="groups">
    <div class="group">
      <h4>▲ extraction <span>— memory write decision</span></h4>
      <div class="item"><svg width="14" height="14"><path d="M7,2 L2,12 L12,12 Z" fill="#2e9e5b"/></svg><span><b>approved</b> candidates — a memory enters the store</span></div>
      <div class="item"><svg width="14" height="14"><path d="M7,2 L2,12 L12,12 Z" fill="#e2a93b"/></svg><span><b>pending</b> candidates only — not injectable yet</span></div>
      <div class="item"><svg width="14" height="14"><path d="M7,2 L2,12 L12,12 Z" fill="#98a2b3"/></svg><span>all rejected — nothing stored</span></div>
      <div class="item"><svg width="14" height="14"><path d="M7,2 L2,12 L12,12 Z" fill="#d64545"/></svg><span>payload unparseable (flagged)</span></div>
      <div class="item"><svg width="34" height="14"><text x="0" y="11" font-size="9" fill="#475467">a/p/r</text></svg><span>counts above ▲ = approved / pending / rejected</span></div>
      <div class="item"><svg width="16" height="14"><text x="0" y="11" font-size="9" fill="#d64545">✕n</text></svg><span>also <b>deleted</b> n memories at this extraction</span></div>
    </div>
    <div class="group">
      <h4>▼ recall <span>— memories delivered to the agent</span></h4>
      <div class="item"><svg width="14" height="14"><path d="M7,12 L2,2 L12,2 Z" fill="#3567d6"/></svg><span>one injection at this step; number = distinct memories delivered</span></div>
      <div class="item"><svg width="32" height="14"><rect x="1" y="2" width="30" height="10" rx="5" fill="#3567d6" fill-opacity="0.85"/><text x="16" y="11" font-size="8" fill="#fff" text-anchor="middle">×n</text></svg><span>span of n consecutive steps with identical injections</span></div>
      <div class="item"><svg width="32" height="14"><rect x="1" y="2" width="30" height="10" rx="5" fill="#3567d6" stroke="#101828" stroke-width="1"/></svg><span>outlined span = delivered the selected/hovered memory</span></div>
    </div>
    <div class="group" id="legend-verdicts" style="display:none">
      <h4>▼ verdicts <span>— injudge outcome per injection span</span></h4>
      <div class="item"><svg width="14" height="14"><path d="M7,12 L2,2 L12,2 Z" fill="#2e9e5b"/></svg><span><b>positive</b> — the injection measurably helped later work</span></div>
      <div class="item"><svg width="14" height="14"><path d="M7,12 L2,2 L12,2 Z" fill="#d64545"/></svg><span><b>negative</b> — it measurably hurt</span></div>
      <div class="item"><svg width="14" height="14"><path d="M7,12 L2,2 L12,2 Z" fill="#98a2b3"/></svg><span><b>neutral</b> — no observable footprint</span></div>
      <div class="item"><svg width="14" height="14"><path d="M7,12 L2,2 L12,2 Z" fill="#e2a93b"/></svg><span><b>error</b> — judge failed (excluded from rates)</span></div>
      <div class="item"><svg width="14" height="14"><path d="M7,12 L2,2 L12,2 Z" fill="#3567d6"/></svg><span>blue = not judged (no verdicts merged)</span></div>
      <div class="note">click a recall node for the judge's analysis, cited evidence, and per-memory usage</div>
      <div class="note">rubric marks: <b>R</b> relevance (bool ✓/✗), <b>A</b> accuracy (✓ corroborated · ≈ uncontradicted · ✗ contradicted), <b>N</b> novelty (✓ novel · ≈ confirming · ✗ redundant), <b>I</b> influence (— none · ◔ referenced · ◑ acted-on · ● decisive), <b>P</b> effect (+ improved · 0 no effect · ✗ harmed)</div>
    </div>
    <div class="group">
      <h4>edges &amp; markers</h4>
      <div class="item"><svg width="30" height="14"><path d="M2,12 C10,12 10,2 28,2" fill="none" stroke="#b06ab3" stroke-width="1.8"/></svg><span>one memory's creation → each use; color matches its sidebar dot; direction arrow shows on hover/selection</span></div>
      <div class="item"><svg width="30" height="14"><path d="M2,12 Q15,0 28,12" fill="none" stroke="#b06ab3" stroke-width="1.6" stroke-dasharray="4 3"/></svg><span>lifecycle: creation → update → deletion of one memory</span></div>
      <div class="item"><svg width="14" height="14"><path d="M7,2 L2,12 L12,12 Z" fill="none" stroke="#b06ab3" stroke-width="1.5" stroke-dasharray="3 2"/></svg><span>ghost creation — origin not observed (predates the loaded set)</span></div>
      <div class="item"><svg width="12" height="16"><line x1="6" y1="1" x2="6" y2="15" stroke="#c3cbd8" stroke-width="2" stroke-dasharray="3 3"/></svg><span>history rewritten (context spliced/compacted here)</span></div>
      <div class="item"><svg width="18" height="16"><rect x="1" y="1" width="16" height="14" rx="3" fill="#edf0f5" stroke="#c3cbd8"/></svg><span>band = rewrites at n consecutive steps (count ↻n); translucent + dashed border = rewrites at nearly every step</span></div>
    </div>
  </div>
</details>
<div id="main">
  <div id="canvas"></div>
  <div id="sidebar">
    <h2 id="memhdr">Memories</h2>
    <ul id="memlist"></ul>
    <h2>Detail</h2>
    <div id="detail"><span style="color:var(--muted)">click a memory or an op node</span></div>
  </div>
</div>
<footer>
  <details><summary>validation (<span id="vcount"></span>)</summary><ul id="vlist"></ul></details>
</footer>
<script>
"use strict";
const G = __GRAPH_JSON__;

// ---------- helpers ----------
function esc(s){ return String(s==null?"":s).replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
function hue(key){ let h=0; for(const c of key) h=(h*31+c.charCodeAt(0))>>>0; return h%360; }
function memColor(id){ return `hsl(${hue(id)},65%,42%)`; }
// Memory identity is the (namespace, key) pair — the same identity the graph
// registry keys on; a bare key would collapse two stores' memories into one.
// The id string is JSON: no separator to collide with key/namespace content,
// null and "" stay distinct. A non-string namespace names no store (the same
// rule as the registry's _namespace_of). Display text stays the bare key
// everywhere; the namespace only disambiguates on demand (detail panel, list
// tooltip).
const memId=m=>JSON.stringify([typeof m.namespace==="string"?m.namespace:null,m.key]);
const keyOf=id=>JSON.parse(id)[1];
const memById=Object.fromEntries(G.memories.map(m=>[memId(m),m]));
// Bare-key contexts (injudge rubric keys, namespace-less deletion targets)
// link only when exactly one loaded memory carries the key — never guess.
// Null-prototype maps: memory keys are trajectory-controlled, so a key like
// "__proto__" must count and link as data, never touch the prototype.
const idByKey=Object.create(null);
{
  const n=Object.create(null);
  for(const m of G.memories) n[m.key]=(n[m.key]||0)+1;
  for(const m of G.memories) if(n[m.key]===1) idByKey[m.key]=memId(m);
}
// injudge verdict colors (ops carry verdicts only when merged via --verdicts)
const VD_COLOR={positive:"#2e9e5b",negative:"#d64545",neutral:"#98a2b3",error:"#e2a93b"};
function shorten(s,n){ s=String(s||""); return s.length>n? s.slice(0,n-1)+"…" : s; }
// "20260727-065319.391Z-cure-v2-9eb25cc9" → {main:"cure-v2 · #9eb25cc9", sub:"2026-07-27 06:53Z"}
// run_id is stringified first: a hand-edited run.json can carry a non-string,
// and String.prototype.match on it would throw and kill the whole page.
function prettyLabel(t){
  const m=String(t.run_id||"").match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})\.\d+Z-(.+)-([0-9a-f]{6,})$/);
  if(!m) return {main:shorten(t.run_id,34), sub:""};
  return {main:`${m[7]} · #${m[8]}`, sub:`${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}Z`};
}

// ---------- zoom-independent derived data ----------
const LABEL_W=250, TOP=34, LANE_H=64, LANE_GAP=30, RIGHT=50, BOTTOM=30;
const trajs=G.trajectories, maxSteps=Math.max(10, ...trajs.map(t=>t.n_steps||0));
const height=TOP+trajs.length*(LANE_H+LANE_GAP)+BOTTOM;
const laneY=i=> TOP+i*(LANE_H+LANE_GAP)+LANE_H/2;
const trajIdx=Object.fromEntries(trajs.map((t,i)=>[t.id,i]));
const opsById=Object.fromEntries(G.ops.map(o=>[o.id,o]));
const opOrder=Object.fromEntries(G.ops.map((o,i)=>[o.id,i]));

// collapse consecutive recalls (per traj) with identical key sets into spans;
// step-less recalls (truncated trajectories) become singleton spans, never merged
const recallSpans={}; // firstOpId -> {ids:[], memIds:[], keySig, steps:[a,b]}
const spansByTraj={}; // traj id -> span groups in step order
{
  for(const t of trajs){
    const rs=G.ops.filter(o=>o.traj===t.id&&o.kind==="recall")
                  .sort((a,b)=>(a.step==null?1e15:a.step)-(b.step==null?1e15:b.step));
    let grp=null;
    // memIds are JSON strings (no literal control chars) — the joined
    // signature stays unambiguous
    const keySig=ids=>ids.slice().sort().join("\u001f");
    const close=()=>{ if(!grp) return; recallSpans[grp.ids[0]]=grp; (spansByTraj[t.id]=spansByTraj[t.id]||[]).push(grp); };
    for(const o of rs){
      const ids=o.memories.map(memId);
      if(grp && o.step!=null && grp.steps[1]!=null && o.step===grp.steps[1]+1 && keySig(ids)===grp.keySig) { grp.ids.push(o.id); grp.steps[1]=o.step; continue; }
      close();
      grp={ids:[o.id], memIds:ids, keySig:keySig(ids), steps:[o.step,o.step]};
    }
    close();
  }
}
const spanLeader={}; // any recall op id -> leader id
for(const [lead,g] of Object.entries(recallSpans)) for(const id of g.ids) spanLeader[id]=lead;

// key-set diff at each span start — why the previous span broke
const spanDiff={}; // leadId -> {added:[], removed:[]}
for(const t of trajs){
  let prev=null;
  for(const g of (spansByTraj[t.id]||[])){
    const cur=new Set(g.memIds);
    if(prev){
      const added=[...cur].filter(k=>!prev.has(k)), removed=[...prev].filter(k=>!cur.has(k));
      if(added.length||removed.length) spanDiff[g.ids[0]]={added,removed};
    }
    prev=cur;
  }
}

// memory ids with an edge into each recall span → stagger converging arrowheads
// (each op appears at most once in a memory's recalls, so a span leader is
// pushed exactly once; non-leader ops share their leader's key set by
// construction, so the leader carries the memory whenever any member does)
const edgeTargets={}; // leadId -> [memory id]
for(const m of G.memories){
  if(!(m.created&&m.created.op)) continue;
  for(const r of (m.recalls||[])){
    const lead=spanLeader[r.op]||r.op;
    if(lead!==r.op) continue; // one entry per span
    (edgeTargets[lead]=edgeTargets[lead]||[]).push(memId(m));
  }
}

// memory ids created at each op → fan out parallel curves leaving one ▲
const edgeSources={}; // creation opId -> [memory id]
for(const m of G.memories){
  const c=m.created&&m.created.op;
  if(c&&(m.recalls||[]).length) (edgeSources[c]=edgeSources[c]||[]).push(memId(m));
}

// collapse rewrites at consecutive steps into bands (per traj)
const rewriteBands={}; // traj id -> [{a,b,count,items}] (count>=2)
const rewriteSingles={}; // traj id -> [rw]
for(const t of trajs){
  // rewrites always carry an int step (parse_trajectory computes it), unlike op steps
  const rws=(t.rewrites||[]).slice().sort((a,b)=>a.step-b.step);
  const bands=[], singles=[];
  let grp=null;
  const close=()=>{ if(!grp) return;
    if(grp.items.length>1) bands.push({a:grp.a,b:grp.b,count:grp.items.length,items:grp.items});
    else singles.push(grp.items[0]);
    grp=null; };
  for(const rw of rws){
    const s=rw.step;
    if(grp && s<=grp.b+1){ grp.b=Math.max(grp.b,s); grp.items.push(rw); continue; }
    close();
    grp={a:s,b:s,items:[rw]};
  }
  close();
  rewriteBands[t.id]=bands;
  rewriteSingles[t.id]=singles;
}

// ---------- svg build (zoom-dependent) ----------
let PX=20; // px per agent step
let nDrawnEdges=0, nLifeEdges=0;
// step-less ops park in a gutter right of the last step instead of piling onto it
const anyUnattributed=G.ops.some(o=>o.step==null);
const GUTTER=anyUnattributed?64:0;
const plotW=width=>width-LABEL_W-RIGHT-GUTTER;
function buildSVG(){
  const width=Math.max(Math.max(640,canvas.clientWidth-20), LABEL_W+RIGHT+GUTTER+maxSteps*PX);
  const X=s=> LABEL_W + (Math.max(0,s)/maxSteps)*plotW(width);
  const gutterX=X(maxSteps)+34;
  const opPos={};
  // ops sharing one (lane, kind, step) dodge sideways instead of stacking
  {
    const slots={};
    for(const o of G.ops){
      if(trajIdx[o.traj]==null) continue;
      const k=o.traj+"|"+o.kind+"|"+(o.step==null?"g":o.step);
      (slots[k]=slots[k]||[]).push(o);
    }
    for(const slot of Object.values(slots)){
      const n=slot.length;
      slot.forEach((o,j)=>{
        opPos[o.id]={x:(o.step==null?gutterX:X(o.step))+(n>1?(j-(n-1)/2)*12:0),
                     y:laneY(trajIdx[o.traj])+(o.kind==="extract"?-13:13)};
      });
    }
  }
  // ghost creation markers: memories recalled but never created in the loaded
  // set anchor a dashed ▲ at the left edge of their first recall's lane,
  // staggered so several ghosts don't pile onto the same x
  const ghostPos={}; // memory id -> {x,y}
  {
    const perLane={};
    for(const m of G.memories){
      if((m.created&&m.created.op)||!(m.recalls||[]).length) continue;
      const first=[...m.recalls].sort((a,b)=>
        (trajIdx[a.traj]-trajIdx[b.traj])||((a.step==null?1e15:a.step)-(b.step==null?1e15:b.step)))[0];
      const gi=trajIdx[first.traj]; if(gi==null) continue;
      const k=perLane[gi]||0; perLane[gi]=k+1;
      ghostPos[memId(m)]={x:X(0)+k*15, y:laneY(gi)-13};
    }
  }
  // stagger a/p/r count labels within each lane so neighbors don't collide
  const extLabelY={};
  {
    const per={};
    for(const o of G.ops) if(o.kind==="extract"&&opPos[o.id]) (per[o.traj]=per[o.traj]||[]).push(o);
    for(const os of Object.values(per)){
      os.sort((a,b)=>opPos[a.id].x-opPos[b.id].x);
      os.forEach((o,j)=>extLabelY[o.id]=j%2?-19:-10);
    }
  }

  let svg=[`<svg id="graph" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`];
  // open (unfilled) arrowhead — visually distinct from the ▲/▼ op glyphs
  svg.push(`<defs><marker id="arr" viewBox="0 0 8 8" refX="6.6" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0.8,0.8 L7,4 L0.8,7.2" fill="none" style="stroke:#475467;stroke:context-stroke" stroke-width="1.5"/></marker></defs>`);
  // step axis ticks
  for(let s=0;s<=maxSteps;s+=Math.ceil(maxSteps/20)){
    const x=X(s);
    svg.push(`<line x1="${x}" y1="${TOP-14}" x2="${x}" y2="${height-BOTTOM+6}" stroke="#eef1f5"/>`);
    svg.push(`<text x="${x}" y="${TOP-18}" font-size="9" fill="#98a2b3" text-anchor="middle">${s}</text>`);
  }
  // lanes
  trajs.forEach((t,i)=>{
    const y=laneY(i);
    for(const band of rewriteBands[t.id]){
      const x0=X(band.a)-8, x1=X(band.b)+8;
      const srcs=[...new Set(band.items.map(r=>r.proxy_source||"?"))];
      const tip=`${band.count} history rewrites at steps ${band.a}–${band.b} (${srcs.map(s=>esc(s)).join(" + ")})`;
      const fill=srcs.length===1&&srcs[0]===t.memory_lane? "#ece4f7" : "#edf0f5";
      if(t.n_steps>0 && band.b-band.a+1>=0.8*t.n_steps){
        // rewrites at nearly every step — same band shape, but translucent with
        // a dashed border so a lane-long box stays in the background
        svg.push(`<rect x="${x0}" y="${y-LANE_H/2+2}" width="${x1-x0}" height="${LANE_H-4}" rx="4" fill="${fill}" fill-opacity="0.6" stroke="#ccd4e0" stroke-dasharray="4 3"><title>${tip}</title></rect>`);
        svg.push(`<text x="${x0+5}" y="${y-LANE_H/2+12}" font-size="8.5" fill="#98a2b3">↻${band.count}</text>`);
      } else {
        svg.push(`<rect x="${x0}" y="${y-LANE_H/2+2}" width="${x1-x0}" height="${LANE_H-4}" rx="4" fill="${fill}" stroke="#dde3ec"><title>${tip}</title></rect>`);
        svg.push(`<text x="${x0+5}" y="${y-LANE_H/2+12}" font-size="8.5" fill="#98a2b3">↻${band.count}</text>`);
      }
    }
    svg.push(`<line x1="${LABEL_W-8}" y1="${y}" x2="${width-RIGHT+20}" y2="${y}" stroke="#e4e7ec"/>`);
    if(t.n_steps>0){ // lane-end tick: lanes share the axis but end at different steps
      const xe=X(t.n_steps);
      svg.push(`<line x1="${xe}" y1="${y-LANE_H/2+2}" x2="${xe}" y2="${y+LANE_H/2-2}" stroke="#c3cbd8" stroke-width="1.2"><title>${esc(t.id)} ends at step ${t.n_steps}</title></line>`);
    }
    for(const rw of rewriteSingles[t.id]){
      const x=X(rw.step);
      const rwCol=rw.proxy_source===t.memory_lane? "#b06ab3" : "#c3cbd8";
      svg.push(`<line x1="${x}" y1="${y-LANE_H/2+4}" x2="${x}" y2="${y+LANE_H/2-4}" stroke="${rwCol}" stroke-dasharray="3 3"><title>history rewritten (kept ${esc(rw.kept)}, dropped ${esc(rw.dropped)}${rw.proxy_source?", "+esc(rw.proxy_source):""})</title></line>`);
    }
  });
  // edges (before nodes so nodes sit on top): creation op → each recall span;
  // memories with no observed creation originate at their ghost marker
  const edgeEls=[];
  nDrawnEdges=0;
  nLifeEdges=0;
  for(const m of G.memories){
    const fromId=m.created&&m.created.op, mid=memId(m);
    for(const r of (m.recalls||[])){
      const toLead=spanLeader[r.op]||r.op;
      if(toLead!==r.op) continue; // one edge per span
      const a=(fromId&&opPos[fromId])||ghostPos[mid], b=opPos[toLead];
      if(!a||!b) continue;
      const g2=recallSpans[toLead];
      // a span pill is drawn at its true start step even when its leader op was
      // dodged sideways (a co-located singleton recall) — arrows land there too
      const bx=(g2&&g2.ids.length>1&&g2.steps[0]!=null)? X(g2.steps[0]) : b.x;
      const tg=edgeTargets[toLead]||[], ti=tg.indexOf(mid);
      // arrowheads converge on the exact step x; fan them vertically instead of
      // horizontally so every tip stays on the step gridline at any zoom
      const tyOff=(tg.length>1&&ti>=0? (ti-(tg.length-1)/2)*4 : 0);
      const sg=(fromId&&edgeSources[fromId])||[], si=sg.indexOf(mid);
      const sx=a.x+(sg.length>1&&si>=0? (si-(sg.length-1)/2)*5 : 0);
      const y0=a.y+9, y1=b.y-9+tyOff; // ▲ bottom → ▼/pill top
      const dy=Math.max(20, Math.min(80, Math.abs(y1-y0)*0.45));
      const col=memColor(mid);
      nDrawnEdges++;
      edgeEls.push(`<path class="edge" data-mem="${esc(mid)}" d="M${sx},${y0} C${sx},${y0+dy} ${bx},${y1-dy} ${bx},${y1}" fill="none" stroke="${col}" stroke-opacity="0.2" stroke-width="1.5"><title>${esc(m.key)}</title></path>`);
    }
    // lifecycle: creation → each update → deletion, dashed in the memory color
    const chain=[...new Set([fromId, ...(m.updates||[]).map(u=>u.op), m.deleted_by&&m.deleted_by.op]
      .filter(id=>id&&opPos[id]))].sort((p,q)=>opOrder[p]-opOrder[q]);
    for(let i=0;i+1<chain.length;i++){
      const a=opPos[chain[i]], b=opPos[chain[i+1]], col=memColor(mid);
      const mid2=(a.x+b.x)/2, top=Math.min(a.y,b.y)-24;
      nLifeEdges++;
      edgeEls.push(`<path class="edge life" data-mem="${esc(mid)}" d="M${a.x},${a.y-9} Q${mid2},${top} ${b.x},${b.y-9}" fill="none" stroke="${col}" stroke-opacity="0.35" stroke-width="1.5" stroke-dasharray="5 4"><title>${esc(m.key)} lifecycle: ${chain[i]} → ${chain[i+1]}</title></path>`);
    }
  }
  svg.push(`<g id="edges">${edgeEls.join("")}</g>`);
  // op nodes
  function tri(cx,cy,up,r){ const s=up?1:-1; return `M${cx},${cy-s*r} L${cx-r},${cy+s*r} L${cx+r},${cy+s*r} Z`; }
  for(const o of G.ops){
    const p=opPos[o.id]; if(!p) continue;
    if(o.kind==="extract"){
      const appr=o.candidates.filter(c=>(c.review_status||"")==="approved").length;
      const pend=o.candidates.length-appr;
      const rej=o.rejections?o.rejections.count:0;
      const ndel=(o.deletions||[]).length;
      const fill=o.flagged? "var(--flag)" : appr>0? "var(--extract)" : pend>0? "#e2a93b" : "var(--reject)";
      const label=o.flagged? "!" : `${appr}/${pend}/${rej}`;
      svg.push(`<g class="opnode op-extract" data-op="${o.id}" transform="translate(${p.x},${p.y})">
        <path d="${tri(0,0,true,7)}" fill="${fill}"><title>${o.id} extract @step ${o.step} (approved ${appr}, pending ${pend}, rejected ${rej}${ndel?`, deleted ${ndel}`:""}${o.flagged?", FLAGGED: "+esc(o.flagged):""})</title></path>
        <text y="${extLabelY[o.id]||-10}" font-size="8.5" text-anchor="middle" fill="#475467">${label}</text>
        ${ndel? `<text x="9" y="4" font-size="8.5" fill="var(--flag)">✕${ndel}</text>` : ""}</g>`);
    } else {
      const lead=spanLeader[o.id], g=recallSpans[lead];
      if(o.id!==lead||!g) continue;
      const n=g.ids.length, keys=new Set(g.memIds).size;
      // spans always sit at their true start step; only singleton ▼s dodge
      const px=(n>1&&g.steps[0]!=null)? X(g.steps[0]) : p.x;
      let w=n>1? Math.max(14,(X(g.steps[1])-X(g.steps[0]))+10):14;
      // a span ending at the lane's last step keeps its cap clear of the end tick
      const lt=trajs[trajIdx[o.traj]];
      if(n>1&&lt&&lt.n_steps>0&&g.steps[1]!=null) w=Math.max(14,Math.min(w, X(lt.n_steps)+2-px));
      const stepTxt=g.steps[0]==null? "step ?" : `steps ${g.steps[0]}${g.steps[1]>g.steps[0]?"–"+g.steps[1]:""}`;
      const df=spanDiff[lead];
      const diffTxt=df? ` — set change: ${df.added.map(k=>"+"+esc(keyOf(k))).concat(df.removed.map(k=>"−"+esc(keyOf(k)))).join(", ")}` : "";
      // verdict coloring: a pill can cover several judged spans — it takes a
      // verdict color only when every judged span it covers agrees (verdicts
      // always follow the judged span, never the pill); a mixed pill stays
      // uncolored and says so, the popover lists each span's verdict
      const spanIds=new Set(g.ids.map(id=>opsById[id]&&opsById[id].verdict_span).filter(Boolean));
      const svd=[...new Set(g.ids.map(id=>opsById[id]&&opsById[id].span_verdict).filter(Boolean))];
      const vcol=svd.length===1? (VD_COLOR[svd[0]]||null) : null;
      const fillNow=vcol||"var(--recall)";
      const vdTxt=svd.length===1
        ? ` — verdict ${esc(svd[0])}${spanIds.size>1?` (covers ${spanIds.size} judged spans; click for each)`:""}`
        : (svd.length>1? ` — mixed verdicts across ${spanIds.size} judged spans (click for each)` : "");
      svg.push(`<g class="opnode op-recall" data-op="${lead}" transform="translate(${px},${p.y})">
        ${n>1? `<rect x="0" y="-7" width="${w}" height="14" rx="7" fill="${fillNow}" fill-opacity="0.85"/>` : `<path d="${tri(0,0,false,7)}" fill="${fillNow}"/>`}
        <text y="${n>1?3:-10}" x="${n>1?w/2:0}" font-size="8.5" text-anchor="middle" fill="${n>1?"#fff":"#475467"}">${n>1?`×${n}`:keys}</text>
        <title>${lead} recall @${stepTxt} (${n} injection${n>1?"s":""}, ${keys} memor${keys>1?"ies":"y"})${diffTxt}${vdTxt}</title></g>`);
    }
  }
  // ghost creation markers (origin not observed) — dashed ▲ with "?"
  for(const [mid,p] of Object.entries(ghostPos)){
    svg.push(`<g class="ghostnode" data-mem="${esc(mid)}" transform="translate(${p.x},${p.y})">
      <path d="${tri(0,0,true,7)}" fill="none" stroke="${memColor(mid)}" stroke-width="1.6" stroke-dasharray="3 2.4"><title>${esc(keyOf(mid))} — origin not observed; creation predates the loaded trajectories (first seen recalled)</title></path>
      <text y="-10" font-size="8.5" text-anchor="middle" fill="${memColor(mid)}">?</text></g>`);
  }
  // gutter for step-less ops (unattributed — truncated or unclassifiable)
  if(GUTTER){
    const gx=X(maxSteps)+12;
    svg.push(`<line x1="${gx}" y1="${TOP-14}" x2="${gx}" y2="${height-BOTTOM+6}" stroke="#e4e7ec" stroke-dasharray="4 3"/>`);
    svg.push(`<text x="${gx+4}" y="${TOP-18}" font-size="8.5" fill="#98a2b3">no step</text>`);
  }
  // step guide (crosshair), moved on mousemove
  svg.push(`<g id="guide" style="display:none"><line y1="${TOP-14}" y2="${height-BOTTOM+6}" stroke="#98a2b3" stroke-width="1"/><text y="${height-8}" font-size="9.5" fill="#667085" text-anchor="middle"></text></g>`);
  // pinned step marker (step inspector), drawn when focus is set
  if(focus){
    const fi=trajIdx[focus.traj];
    if(fi!=null){
      const fy=laneY(fi), fx=X(focus.step);
      svg.push(`<g id="focusmark" pointer-events="none"><line x1="${fx}" y1="${fy-LANE_H/2}" x2="${fx}" y2="${fy+LANE_H/2}" stroke="#101828" stroke-width="1.6"/><text x="${fx+4}" y="${fy-LANE_H/2+11}" font-size="9.5" font-weight="600" fill="#101828">@${focus.step}</text></g>`);
    }
  }
  // lane labels, kept pinned on horizontal scroll via #canvas scroll handler
  const lab=[`<g id="labels"><rect x="0" y="0" width="${LABEL_W-6}" height="${height}" fill="#fff" opacity="0.93"/><line x1="${LABEL_W-6}" y1="0" x2="${LABEL_W-6}" y2="${height}" stroke="#eef1f5"/><text x="8" y="${TOP-18}" font-size="10" fill="#667085">agent step →</text>`];
  trajs.forEach((t,i)=>{
    const y=laneY(i), pl=prettyLabel(t);
    // the recorded instance id (memory_session) leads the label when present
    // — ignore a corrupt non-string one like any other drifted datum
    const inst=typeof t.instance_id==="string"&&t.instance_id?t.instance_id:null;
    const sub=[inst?pl.main:null, pl.sub, `${t.n_steps} steps`, t.status, t.model].filter(Boolean).join(" · ");
    lab.push(`<text x="8" y="${y-6}" font-size="11" font-weight="600"><title>${esc(t.run_id)}</title>${esc(inst||pl.main)}</text>`);
    lab.push(`<text x="8" y="${y+8}" font-size="9" fill="#667085"><title>${esc(sub)}</title>${esc(shorten(sub,44))}</text>`);
  });
  lab.push(`</g>`);
  svg.push(lab.join(""));
  svg.push(`</svg>`);
  return svg.join("");
}

// ---------- selection / hover / step focus ----------
const canvas=document.getElementById("canvas");
const detail=document.getElementById("detail");
let selected=null, hoverMem=null, focus=null; // focus={traj,step} — step inspector
// memory actions at (traj, step): memory id -> Set("created"|"updated"|"deleted"|"recalled")
function actionsAt(trajId, step){
  const acts={};
  const add=(id,kind)=>{(acts[id]=acts[id]||new Set()).add(kind);};
  for(const o of G.ops){
    if(o.traj!==trajId||o.step!==step) continue;
    if(o.kind==="extract"){
      for(const c of o.candidates){
        if(!c.key) continue;
        const id=memId(c), m=memById[id];
        add(id, m&&m.created&&m.created.op===o.id ? "created" : "updated");
      }
      for(const m of G.memories) if(m.deleted_by&&m.deleted_by.op===o.id) add(memId(m),"deleted");
    } else {
      for(const dm of o.memories) if(dm.key) add(memId(dm),"recalled");
    }
  }
  return acts;
}
function collapseUses(recalls){
  const byT={};
  for(const r of recalls){ (byT[r.traj]=byT[r.traj]||[]).push(r.step); }
  return Object.entries(byT).map(([t,ss])=>{
    ss=[...new Set(ss.filter(s=>s!=null))].sort((a,b)=>a-b);
    const ranges=[]; let a=null,b=null;
    for(const s of ss){ if(a==null){a=b=s;continue;} if(s===b+1){b=s;continue;} ranges.push([a,b]); a=b=s; }
    if(a!=null) ranges.push([a,b]);
    return `${esc(t)}@`+(ranges.map(([x,y])=>x===y?`${x}`:`${x}–${y}`).join(",")||"?");
  }).join(" · ");
}
function paintEdges(){
  document.querySelectorAll(".edge").forEach(p=>{
    const hot=p.dataset.mem===selected||p.dataset.mem===hoverMem;
    p.setAttribute("stroke-opacity", hot?"0.95":(selected?"0.07":(p.classList.contains("life")?"0.35":"0.2")));
    p.setAttribute("stroke-width", hot?"2.6":"1.5");
    // arrowheads only on the highlighted edge — opaque markers on every curve
    // are pure clutter in the resting state
    if(hot) p.setAttribute("marker-end","url(#arr)"); else p.removeAttribute("marker-end");
  });
  // outline the recall spans that delivered the selected/hovered memory
  const hotKey=selected||hoverMem;
  document.querySelectorAll(".op-recall").forEach(n=>{
    const g=recallSpans[n.dataset.op];
    n.classList.toggle("hot", !!(hotKey&&g&&g.memIds.includes(hotKey)));
  });
}
function clearSel(){
  selected=null;
  document.querySelectorAll("#memlist li.sel").forEach(li=>li.classList.remove("sel"));
  paintEdges();
  detail.innerHTML=`<span style="color:var(--muted)">click a memory or an op node</span>`;
}
function locateOp(opId){
  const el=document.querySelector(`.opnode[data-op="${CSS.escape(opId)}"]`);
  if(!el) return;
  el.scrollIntoView({behavior:"smooth", block:"nearest", inline:"center"});
  el.classList.add("located");
  setTimeout(()=>el.classList.remove("located"), 1700);
}
function selectMemory(id){
  clearSel(); selected=id; paintEdges();
  const li=document.querySelector(`#memlist li[data-mem="${CSS.escape(id)}"]`);
  if(li) li.classList.add("sel");
  const m=memById[id]; if(!m) return;
  const c=m.created||{};
  if(c.op) locateOp(c.op);
  else {
    const gh=document.querySelector(`.ghostnode[data-mem="${CSS.escape(id)}"]`);
    if(gh){ gh.scrollIntoView({behavior:"smooth", block:"nearest", inline:"center"});
      gh.classList.add("located"); setTimeout(()=>gh.classList.remove("located"), 1700); }
  }
  // lifecycle trail: creating status → later review statuses → deletion
  const trail=[];
  const pushT=(st,op)=>{ if(st&&(!trail.length||trail[trail.length-1].st!==st)) trail.push({st,op}); };
  if(c.op) pushT(c.status||m.status, c.op);
  for(const u of (m.updates||[])) pushT(u.review_status, u.op);
  if(m.deleted_by) pushT("deleted", m.deleted_by.op);
  const trailHtml=trail.length>1
    ? `<div class="kv">lifecycle: ${trail.map(x=>`<span class="chip" style="color:${x.st==="deleted"?"var(--flag)":(STATUS_COLOR[x.st]||"var(--muted)")}" title="${esc(x.op||"")}">${esc(x.st)}</span>`).join(" → ")}</div>`
    : "";
  detail.innerHTML=`<h3 style="color:${memColor(id)}">${esc(m.key)}</h3>
    <div class="kv"><span class="chip">${esc(m.memory_type||"?")}</span><span class="chip">${esc(m.status)}</span>${m.deleted_by?`<span class="chip" style="color:var(--flag)">deleted</span>`:""}${m.confidence!=null?`<span class="chip">conf ${esc(m.confidence)}</span>`:""}${m.origin==="not_observed"?`<span class="chip">origin not observed</span>`:""}${m.namespace?`<span class="chip" title="store namespace ${esc(m.namespace)}">store ${esc(shorten(m.namespace,16))}</span>`:""}</div>
    <div class="kv">created: ${c.op?`${esc(c.op)} @ ${esc(c.traj)} step ${c.step}`:"— (origin not observed)"}</div>
    ${trailHtml}
    ${m.recalled_before_creation&&m.recalled_before_creation.length?`<div class="kv" style="color:var(--flag)">⚠ ${m.recalled_before_creation.length} recall(s) precede this creation — the store entry likely predates the loaded trajectories</div>`:""}
    <div class="kv">used (${m.recalls.length}): ${collapseUses(m.recalls)||"—"}</div>
    ${m.updates&&m.updates.length?`<div class="kv">updates: ${m.updates.map(u=>`${esc(u.op)}@${u.step}${u.duplicate?" (duplicate)":""}`).join(", ")}</div>`:""}
    ${m.deleted_by?`<div class="kv">deleted by ${esc(m.deleted_by.op)} @ step ${m.deleted_by.step}</div>`:""}
    <pre>${esc(m.value||"")}</pre>`;
}
function memChip(id,sign,label){
  return `<span class="chip memlink" data-mem="${esc(id)}" title="select memory">${sign||""}${esc(label||keyOf(id))}</span>`;
}
// a bare key links only when it names exactly one loaded memory (an ambiguous
// or unknown key renders as plain text, never a wrong link)
function chipForKey(key,sign){
  const id=idByKey[key];
  return id? memChip(id,sign,key) : `<span class="chip">${sign||""}${esc(key)}</span>`;
}
function bindMemLinks(){
  detail.querySelectorAll(".memlink").forEach(el=>el.addEventListener("click",ev=>{
    ev.stopPropagation(); selectMemory(el.dataset.mem);
  }));
}
function showOp(opId){
  const o=opsById[opId]; if(!o) return;
  clearSel();
  let h=`<h3>${esc(o.id)} <span style="color:var(--muted);font-weight:400">${o.kind}</span></h3>
    <div class="kv">${o.step!=null?`step ${o.step}`:"step unattributed"}${o.turn!=null?` · turn ${o.turn}`:""}${o.at?` · ${esc(o.at)}`:""}</div>`;
  let body;
  if(o.kind==="extract"){
    body={step:o.step,turn:o.turn,status:o.status,trigger:o.trigger,flagged:o.flagged,candidates:o.candidates,deletions:o.deletions,rejections:o.rejections};
    if(o.status||o.trigger) h+=`<div class="kv">${o.status?`status ${esc(o.status)}`:""}${o.trigger?` · trigger ${esc(o.trigger)}`:""}${o.declared_step!=null?` · declared step ${esc(o.declared_step)}`:""}</div>`;
    if(o.flagged) h+=`<div class="flagbox">⚠ ${esc(o.flagged)}</div>`;
    if(o.candidates.length){
      h+=`<h4>candidates (${o.candidates.length})</h4>`;
      for(const c of o.candidates){
        const stCol=STATUS_COLOR[c.review_status]||"var(--muted)";
        h+=`<div class="mrow"><div>${c.key?memChip(memId(c)):"—"}<span class="chip" style="color:${stCol}">${esc(c.review_status||"?")}</span>${c.confidence!=null?`<span class="meta">conf ${esc(c.confidence)}</span>`:""}</div><div class="vline" title="${esc(c.value||"")}">${esc(shorten(c.value,150))}</div></div>`;
      }
    }
    if(o.deletions&&o.deletions.length){
      h+=`<h4>deletions (${o.deletions.length})</h4>`;
      for(const d of o.deletions){
        const tgt=d.target||"";
        // the before-ref pins the exact (namespace, key); a ref-less target
        // resolves by bare key only when unambiguous, else plain text
        const id=d.ref? memId(d.ref) : idByKey[tgt];
        h+=`<div class="mrow"><div>${id&&memById[id]?memChip(id,"",tgt):esc(tgt)}</div></div>`;
      }
    }
    if(o.rejections&&o.rejections.count){
      const by=o.rejections.by_reason||{};
      h+=`<h4>rejections (${o.rejections.count}) — ${Object.entries(by).map(([r,n])=>`${esc(r)} ×${n}`).join(", ")}</h4>`;
    }
  } else {
    const g=recallSpans[opId];
    const n=g?g.ids.length:1, steps=g?g.steps:[o.step];
    const keys=new Set(o.memories.map(memId)).size;
    body={steps,injections:g?g.ids:[o.id],memories:o.memories,
          search_operation_id:o.search_operation_id,query:o.query,matched:o.matched,binding_status:o.binding_status};
    h+=`<div class="kv">${n} injection${n>1?"s":""} · steps ${steps[0]==null?"?":steps[0]}${steps[1]>steps[0]?"–"+steps[1]:""} · ${keys} distinct memor${keys>1?"ies":"y"}</div>`;
    if(o.matched!=null||o.binding_status||o.query!=null)
      h+=`<div class="kv">${o.matched!=null?`search returned ${o.matched}`:""}${o.binding_status?` · binding ${esc(o.binding_status)}`:""}${o.query!=null?` · query “${esc(shorten(o.query,120))}”`:""}</div>`;
    const df=spanDiff[opId];
    if(df) h+=`<div class="kv">set change: ${df.added.map(id=>memChip(id,"+",keyOf(id))).join(" ")}${df.added.length&&df.removed.length?" ":""}${df.removed.map(id=>memChip(id,"−",keyOf(id))).join(" ")}</div>`;
    if(o.memories.length){
      h+=`<h4>delivered (${o.memories.length})</h4>`;
      for(const dm of o.memories){
        h+=`<div class="mrow"><div>${memChip(memId(dm))}<span class="chip">${esc(dm.memory_type||"")}${dm.needs_verification?" ⚠":""}</span>${dm.confidence!=null?`<span class="meta">conf ${esc(dm.confidence)}</span>`:""}</div><div class="vline" title="${esc(dm.value||"")}">${esc(shorten(dm.value,150))}</div></div>`;
      }
    }
    // injudge verdicts for every judged span this memgraph span covers
    const vops=(g? g.ids:[o.id]).map(id=>opsById[id]).filter(x=>x&&x.verdict_span);
    if(vops.length){
      h+=`<h4>injection verdicts (injudge)</h4>`;
      const seen=new Set();
      // enum glyphs per signal; relevance is the only bool signal
      const SIG_ENUM={
        accuracy:{corroborated:["✓","#2e9e5b"],uncontradicted:["≈","#5b8dd9"],contradicted:["✗","#d64545"]},
        novelty:{novel:["✓","#2e9e5b"],confirming:["≈","#5b8dd9"],redundant:["✗","#d64545"]},
        influence:{none:["—","var(--muted)"],referenced:["◔","#5b8dd9"],acted_on:["◑","#2e9e5b"],decisive:["●","#2e9e5b"]},
        positive_effect:{improved:["+","#2e9e5b"],no_effect:["0","#b8860b"],harmed:["✗","#d64545"]},
      };
      const sigCell=(sig,s,abbr)=>{
        const e=(sig&&sig[s])||{}, v=e.value;
        let mark="—", col="var(--muted)";
        if(typeof v==="boolean"){
          mark=v?"✓":"✗"; col=v?"#2e9e5b":"#d64545";
        } else if(v!=null){
          const cell=(SIG_ENUM[s]||{})[v];
          // a foreign/hand-edited verdicts artifact can carry an off-enum
          // value — escape it like every other interpolated datum
          if(cell){ mark=cell[0]; col=cell[1]; } else { mark=esc(String(v)); }
        }
        return `<span title="${esc(s)}: ${esc(v==null?"null":String(v))}${e.note?" — "+esc(e.note):""}" style="color:${col}">${abbr}${mark}</span>`;
      };
      for(const x of vops){
        if(seen.has(x.verdict_span)) continue;
        seen.add(x.verdict_span);
        const d=x.verdict_detail||{}, v=d.verdict||x.span_verdict||"unjudged";
        const dc=d.delivery_cost||{};
        // artifact-controlled numbers render escaped like any other datum;
        // the thousands-formatted token count only when it really is a number —
        // 0 included: a redacted/unsized placement counts 0 tokens, and hiding
        // the line would misread as "no deliveries" (injudge warns about the
        // understatement separately). A missing deliveries count renders "?",
        // never an empty blank.
        const dcTxt=typeof dc.approx_tokens_total==="number"?`<span class="meta">delivered ${esc(dc.deliveries==null?"?":dc.deliveries)}× ≈${esc(dc.approx_tokens_total.toLocaleString())} tokens</span>`:"";
        h+=`<div class="mrow"><div><span class="chip" style="color:${VD_COLOR[v]||"var(--muted)"}">${esc(v)}</span>${d.confidence!=null?`<span class="meta">conf ${esc(d.confidence)}</span>`:""}<span class="meta">${esc(x.verdict_span)}</span>${d.auto?`<span class="meta">auto: ${esc(d.auto)}</span>`:""}${dcTxt}</div>`;
        if(d.error) h+=`<div class="flagbox">judge error: ${esc(d.error)}</div>`;
        if(d.analysis) h+=`<div class="vline" style="white-space:normal">${esc(d.analysis)}</div>`;
        if(d.rubric&&Object.keys(d.rubric).length){
          // per-memory rubric marks: R(elevance) A(ccuracy) N(ovelty)
          // I(nfluence) P(ositive effect)
          for(const [mk,sig] of Object.entries(d.rubric))
            h+=`<div class="vline">${chipForKey(mk)} ${sigCell(sig,"relevance","R")} ${sigCell(sig,"accuracy","A")} ${sigCell(sig,"novelty","N")} ${sigCell(sig,"influence","I")} ${sigCell(sig,"positive_effect","P")}</div>`;
        }
        // a hand-edited verdicts artifact can carry null/non-object entries
        // or non-array step lists — skip them like any other drifted datum,
        // never dead-panel the op on a TypeError
        for(const [k,u] of Object.entries(d.memory_usage||{})){
          if(!u||typeof u!=="object") continue;
          h+=`<div class="vline" title="${esc(u.evidence||"")}">${esc(k)} → ${esc(u.usage)}${u.reason?` <span class="meta">(${esc(u.reason)})</span>`:""}</div>`;
        }
        for(const evv of (Array.isArray(d.evidence)?d.evidence:[])){
          if(!evv||typeof evv!=="object") continue;
          const evSteps=Array.isArray(evv.steps)?evv.steps:[], evSeqs=Array.isArray(evv.seqs)?evv.seqs:[];
          h+=`<div class="vline" style="white-space:normal">• ${esc(evv.claim||"")} (steps ${esc(evSteps.join(",")||"?")}; seqs ${esc(evSeqs.join(",")||"?")})</div>`;
        }
        h+=`</div>`;
      }
    }
  }
  h+=`<details class="rawjson"><summary>raw JSON</summary><pre>${esc(JSON.stringify(body,null,1))}</pre></details>`;
  detail.innerHTML=h;
  bindMemLinks();
  // sync the Memories panel to this op's step
  if(o.step!=null){ focus={traj:o.traj, step:o.step}; refreshFocus(); }
}

// ---------- mount / zoom ----------
function mount(keepScroll){
  const frac=keepScroll && canvas.scrollWidth? canvas.scrollLeft/canvas.scrollWidth : 0;
  canvas.innerHTML=buildSVG();
  if(keepScroll) canvas.scrollLeft=frac*canvas.scrollWidth;
  document.querySelectorAll(".opnode").forEach(n=>n.addEventListener("click",ev=>{ev.stopPropagation();showOp(n.dataset.op);}));
  document.querySelectorAll(".ghostnode").forEach(n=>{
    n.addEventListener("click",ev=>{ev.stopPropagation();selectMemory(n.dataset.mem);});
    n.addEventListener("mouseenter",()=>{hoverMem=n.dataset.mem;paintEdges();});
    n.addEventListener("mouseleave",()=>{hoverMem=null;paintEdges();});
  });
  document.querySelectorAll(".edge").forEach(p=>{
    p.addEventListener("click",ev=>{ev.stopPropagation();selectMemory(p.dataset.mem);});
    p.addEventListener("mouseenter",()=>{hoverMem=p.dataset.mem;paintEdges();});
    p.addEventListener("mouseleave",()=>{hoverMem=null;paintEdges();});
  });
  const svg=document.getElementById("graph"), guide=document.getElementById("guide");
  const width=+svg.getAttribute("width");
  svg.addEventListener("mousemove",ev=>{
    const r=svg.getBoundingClientRect();
    const x=ev.clientX-r.left;
    const step=Math.round((x-LABEL_W)/plotW(width)*maxSteps);
    if(step<0||step>maxSteps){ guide.style.display="none"; return; }
    const gx=LABEL_W+(step/maxSteps)*plotW(width);
    guide.style.display="";
    guide.setAttribute("transform",`translate(${gx},0)`);
    guide.querySelector("text").textContent=`step ${step}`;
  });
  svg.addEventListener("mouseleave",()=>guide.style.display="none");
  applyFilters();
  paintEdges();
  const nE=G.ops.filter(o=>o.kind==="extract").length, nR=G.ops.filter(o=>o.kind==="recall").length;
  const nDeliv=G.memories.reduce((s,m)=>s+(m.recalls||[]).length,0);
  let vdStats="";
  if(G.tool&&G.tool.verdicts){
    document.getElementById("legend-verdicts").style.display="";
    const seenS=new Set(), vc={};
    for(const o of G.ops){
      if(o.kind!=="recall"||!o.verdict_span||seenS.has(o.verdict_span)) continue;
      seenS.add(o.verdict_span);
      const v=o.span_verdict||"unjudged"; vc[v]=(vc[v]||0)+1;
    }
    vdStats=` · verdicts +${vc.positive||0}/−${vc.negative||0}/○${vc.neutral||0}`
      +(vc.error?` !${vc.error}`:"")+(vc.unjudged?` ?${vc.unjudged}`:"");
  }
  document.getElementById("stats").textContent=
    `${trajs.length} trajectories · ${nE} extractions · ${nR} recalls · ${G.memories.length} memories · ${nDeliv} deliveries along ${nDrawnEdges} create→use paths${nLifeEdges?` · ${nLifeEdges} lifecycle links`:""}${vdStats}`;
}
canvas.addEventListener("scroll",()=>{
  const lab=document.getElementById("labels");
  if(lab) lab.setAttribute("transform",`translate(${canvas.scrollLeft},0)`);
});
function refreshFocus(){
  mount(true); // redraws the pinned marker (SVG is rebuilt)
  renderMemList(document.getElementById("memfilter").value.trim().toLowerCase());
  updateMemHeader();
}
function clearFocus(){ if(focus){ focus=null; refreshFocus(); } }
canvas.addEventListener("click",ev=>{
  // lane click → step inspector; background click → clear everything
  const svg=document.getElementById("graph");
  if(svg){
    const r=svg.getBoundingClientRect();
    const x=ev.clientX-r.left, y=ev.clientY-r.top;
    const width=+svg.getAttribute("width");
    const step=Math.round((x-LABEL_W)/plotW(width)*maxSteps);
    const li=Math.floor((y-TOP)/(LANE_H+LANE_GAP));
    const t=trajs[li];
    if(t && Math.abs(y-laneY(li))<=LANE_H/2+6 && step>=0 && step<=maxSteps){
      clearSel();
      if(focus&&focus.traj===t.id&&focus.step===step) focus=null; // click same step → unfocus
      else focus={traj:t.id, step};
      refreshFocus();
      return;
    }
  }
  focus=null; clearSel(); refreshFocus();
});
document.addEventListener("keydown",ev=>{
  if(ev.target&&ev.target.tagName==="INPUT") return;
  if(ev.key==="Escape"){ clearSel(); clearFocus(); }
  else if(ev.key==="+"||ev.key==="=") zoom(1.25);
  else if(ev.key==="-") zoom(1/1.25);
  else if(ev.key==="0") zoomFit();
  else if((ev.key==="ArrowLeft"||ev.key==="ArrowRight")&&focus){
    focus={traj:focus.traj, step:Math.min(maxSteps,Math.max(0,focus.step+(ev.key==="ArrowRight"?1:-1)))};
    refreshFocus();
  }
});
function zoom(f){ PX=Math.min(60,Math.max(6,PX*f)); mount(true); }
function zoomFit(){ PX=Math.min(30,Math.max(4,(canvas.clientWidth-16-LABEL_W-RIGHT-GUTTER)/maxSteps)); mount(true); }
document.getElementById("z-in").onclick=()=>zoom(1.25);
document.getElementById("z-out").onclick=()=>zoom(1/1.25);
document.getElementById("z-fit").onclick=zoomFit;

// ---------- sidebar ----------
const memlist=document.getElementById("memlist");
const STATUS_COLOR={"approved":"var(--extract)","pending":"#e2a93b","pending_review":"#e2a93b","unknown":"var(--muted)"};
const ACTION_LABEL={created:"▲ created",updated:"↻ updated",deleted:"✕ deleted",recalled:"▼ recalled"};
const ACTION_COLOR={created:"var(--extract)",updated:"#e2a93b",deleted:"var(--flag)",recalled:"var(--recall)"};
function updateMemHeader(){
  const h=document.getElementById("memhdr");
  h.textContent=focus? `Memories @ ${focus.traj} step ${focus.step}` : "Memories";
  if(focus){
    const b=document.createElement("button");
    b.textContent="✕ clear";
    b.onclick=()=>{ focus=null; refreshFocus(); };
    h.appendChild(b);
  }
}
function renderMemList(filter){
  memlist.innerHTML="";
  const acts=focus? actionsAt(focus.traj, focus.step):null;
  let shown=0;
  for(const m of G.memories){
    if(filter && !m.key.toLowerCase().includes(filter)) continue;
    const mid=memId(m);
    const kinds=acts? acts[mid]:null;
    if(acts && !kinds) continue; // step inspector: only memories acting at this step
    shown++;
    const li=document.createElement("li");
    li.dataset.mem=mid;
    const used=m.recalls.length;
    const created=(m.created&&m.created.op)?1:0;
    // terminal state wins: a deleted memory shows "deleted", not its last review status
    const effStatus=m.deleted_by? "deleted" : m.status;
    const stCol=m.deleted_by? "var(--flag)" : (STATUS_COLOR[m.status]||"var(--muted)");
    li.title=`${m.key}${m.namespace?` (store ${m.namespace})`:""} — status: ${effStatus}${m.deleted_by?` (was ${m.status}; deleted by ${m.deleted_by.op} @ step ${m.deleted_by.step})`:""}; created ${created?"at "+m.created.traj+" step "+m.created.step:"not observed"}; recalled ${used}×. Dot color matches its edges.`;
    const stHtml=kinds
      ? [...kinds].map(k=>`<span class="act" style="color:${ACTION_COLOR[k]}">${ACTION_LABEL[k]}</span>`).join(" ")
      : `<span style="color:${stCol}">${esc(effStatus)}</span>${m.origin==="not_observed"?" · origin?":""} · ▲${created} ▼${used}`;
    li.innerHTML=`<span class="badge" style="background:${memColor(mid)}"></span><span class="mkey">${esc(m.key)}</span><span class="st">${stHtml}</span>`;
    li.onclick=()=>selectMemory(mid);
    li.onmouseenter=()=>{hoverMem=mid;paintEdges();};
    li.onmouseleave=()=>{hoverMem=null;paintEdges();};
    memlist.appendChild(li);
  }
  if(acts && shown===0){
    const li=document.createElement("li");
    li.className="placeholder";
    li.textContent="no memory actions at this step";
    memlist.appendChild(li);
  }
}
document.getElementById("memfilter").oninput=e=>renderMemList(e.target.value.trim().toLowerCase());
updateMemHeader();
renderMemList("");

// ---------- filters / validation ----------
function applyFilters(){
  const fe=document.getElementById("f-extract").checked, fr=document.getElementById("f-recall").checked,
        fed=document.getElementById("f-edges").checked, frej=document.getElementById("f-rejonly").checked;
  document.querySelectorAll(".op-extract").forEach(n=>{
    const o=opsById[n.dataset.op];
    const quiet=!o.flagged && o.candidates.length===0 && !(o.deletions||[]).length;
    n.style.display=(!fe||(frej&&quiet))?"none":"";
  });
  document.querySelectorAll(".op-recall").forEach(n=>n.style.display=fr?"":"none");
  document.getElementById("edges").style.display=fed?"":"none";
}
for(const id of ["f-extract","f-recall","f-edges","f-rejonly"]) document.getElementById(id).onchange=applyFilters;

const vl=document.getElementById("vlist");
document.getElementById("vcount").textContent=G.validation.length;
for(const v of G.validation){
  const li=document.createElement("li"); li.textContent=`[${v.level}] ${v.msg}`; vl.appendChild(li);
}

mount(false);
// default view: fit width when the graph would overflow the canvas
if(canvas.scrollWidth>canvas.clientWidth+2) zoomFit();
</script>
</body>
</html>
"""


def render_html(graph: dict, title: str = "memgraph — memory operation relationships") -> str:
    """Render the graph dict as one self-contained HTML document."""
    # every "<" in the payload becomes its unicode escape (valid in both
    # JSON and JS string literals): trajectory-controlled values can neither
    # close the script element ("</script>") nor open a script-data comment
    # ("<!--"), which would swallow the rest of the document. U+2028/U+2029
    # get the same treatment: line terminators a pre-ES2019 JS parser
    # rejects inside string literals.
    payload = (
        json.dumps(graph, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    # one substitution pass: replacements are never re-scanned, so a graph
    # datum carrying the literal __TITLE__ survives, a title carrying the
    # literal __GRAPH_JSON__ survives, and no ordering between the two
    # placeholders matters. The title lands in <title>/<h1> text — escape it
    # like any other interpolated string.
    escaped = html.escape(title)
    return re.sub(
        "__TITLE__|__GRAPH_JSON__",
        lambda m: escaped if m.group(0) == "__TITLE__" else payload,
        _TEMPLATE,
    )
