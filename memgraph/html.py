"""Self-contained interactive HTML rendering of the memory-op graph.

One file, no external resources: the graph JSON is embedded and drawn with
vanilla JS + inline SVG. Rows follow the recording's structure: a run that
records episodes gets a header, then a foldable header per sequence and one
row per episode (x = step within the episode, so rows of different episodes
line up); a run naming no episode is one row on its own steps, and runs
without memory operations are only counted in the page header. Writes (▲)
sit above a row's line, recalls (▼) below, and consecutive identical recalls
collapse into ×n pills (never across episodes). At rest only the faint
creation → recall links of memories created in the loaded set are drawn.
Memories created earlier share one dashed ▲ per run, and their links (plus
every memory's update/deletion chain) appear only for the hovered or
selected memory: thousands of pre-existing memories would otherwise bury the
graph. The step axis fits the window by default; zoom rescales it and row
labels stay pinned while scrolling.
"""

from __future__ import annotations

import html
import json
import re

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --fg:#1c2430; --muted:#667085; --line:#e4e7ec; --panel:#f7f8fa;
          --write:#2e9e5b; --pending:#e2a93b; --reject:#98a2b3; --recall:#3567d6; --flag:#d64545; }
  * { box-sizing:border-box; }
  [hidden] { display:none !important; }
  html, body { height:100%; }
  body { margin:0; display:flex; flex-direction:column; font:13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; color:var(--fg); background:#fff; }
  header { display:flex; flex-wrap:wrap; align-items:center; gap:6px 18px; padding:10px 16px 8px; border-bottom:1px solid var(--line); }
  h1 { font-size:15px; margin:0; }
  #summary, #summary .vd { display:flex; flex-wrap:wrap; align-items:center; gap:4px 12px; color:var(--muted); }
  #summary b { color:var(--fg); }
  .controls { margin-left:auto; display:flex; align-items:center; gap:12px; color:var(--muted); }
  .controls label { cursor:pointer; user-select:none; }
  button { font:inherit; color:var(--fg); background:#fff; border:1px solid var(--line); border-radius:5px; padding:0 8px; cursor:pointer; }
  button:hover { background:var(--panel); }
  #search { width:210px; padding:3px 8px; border:1px solid var(--line); border-radius:5px; font:inherit; }
  #key { display:flex; flex-wrap:wrap; align-items:center; gap:3px 16px; padding:5px 16px; border-bottom:1px solid var(--line); background:var(--panel); color:var(--muted); font-size:12px; }
  #key > span, #key-verdicts > span { display:inline-flex; align-items:center; gap:5px; cursor:help; }
  #key-verdicts { display:inline-flex; gap:12px; }
  #key .hint { margin-left:auto; cursor:default; }
  kbd { font:11px ui-monospace, Menlo, monospace; border:1px solid var(--line); border-radius:3px; padding:0 3px; background:#fff; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; flex:0 0 auto; }
  .muted { color:var(--muted); font-weight:400; }
  .warn { color:#b54708; }
  .link { cursor:pointer; }
  .link:hover { text-decoration:underline; }
  #main { flex:1; display:flex; min-height:0; }
  #canvas { flex:1; overflow:auto; scrollbar-gutter:stable; }
  #canvas .none { padding:24px; color:var(--muted); }
  #sidebar { width:390px; flex:0 0 auto; display:flex; flex-direction:column; min-height:0; border-left:1px solid var(--line); background:var(--panel); }
  #memhdr { display:flex; align-items:center; gap:8px; margin:10px 14px 4px; font-size:12px; font-weight:600; }
  #memlist { list-style:none; margin:0; padding:0 6px; overflow-y:auto; flex:0 1 42%; min-height:90px; }
  #memlist li { display:flex; gap:8px; align-items:baseline; padding:4px 8px; border-radius:5px; cursor:pointer; }
  #memlist li:hover { background:#eaeef4; }
  #memlist li.sel { background:#dbe6fb; }
  #memlist li.empty { color:var(--muted); font-style:italic; cursor:default; background:none; }
  #memlist .txt { flex:1; min-width:0; }
  #memlist .k, #memlist .v, .row .v { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  #memlist .v, .row .v { color:var(--muted); font-size:12px; }
  #memlist .n { flex:0 0 auto; color:var(--muted); font-size:11px; }
  #detail { flex:1; overflow:auto; margin:6px 8px 8px; padding:10px 12px; background:#fff; border:1px solid var(--line); border-radius:6px; }
  #detail h3 { margin:0 0 4px; font-size:14px; word-break:break-all; }
  #detail h4 { margin:14px 0 4px; font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
  #detail p { margin:6px 0; }
  .help p { color:var(--muted); }
  .sub { color:var(--muted); margin:2px 0; }
  .tag { display:inline-block; padding:0 6px; margin:0 4px 2px 0; border-radius:8px; font-size:11px; background:#eef2f7; }
  .row { margin:3px 0 7px; }
  .marks { font-size:12px; margin-left:4px; }
  .marks span { margin-right:6px; cursor:help; }
  .verdict { border-left:3px solid var(--c); padding:2px 0 2px 10px; margin:8px 0 12px; }
  .flag { background:#fdeeee; border:1px solid #f3c6c6; color:var(--flag); border-radius:4px; padding:4px 8px; margin:6px 0; font-size:12px; }
  pre { background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:8px; font-size:11.5px; white-space:pre-wrap; word-break:break-word; max-height:45vh; overflow:auto; }
  details > summary { cursor:pointer; color:var(--muted); font-size:12px; margin-top:6px; }
  footer { border-top:1px solid var(--line); padding:5px 16px; font-size:12px; max-height:30vh; overflow:auto; }
  footer ul { margin:4px 0; padding-left:18px; color:var(--muted); }
  footer li.warn { color:#b54708; }
  /* graph */
  svg text { user-select:none; font-size:9px; fill:var(--muted); }
  .tick { fill:#98a2b3; text-anchor:middle; }
  .tick.l { text-anchor:start; }
  .tick.r { text-anchor:end; }
  .cnt, .gcnt { fill:#475467; text-anchor:middle; }
  .gcnt { text-anchor:start; }
  .cnt.del { fill:var(--flag); text-anchor:start; }
  .pill { fill:#fff; text-anchor:middle; font-size:8.5px; }
  .lname { font-size:11.5px; font-weight:600; fill:var(--fg); }
  .lname tspan { fill:var(--muted); font-weight:400; }
  .lsub { font-size:10px; }
  .lsub.warn { fill:#b54708; }
  .lep { font-size:11px; font-weight:600; fill:var(--fg); }
  .lep.none { font-weight:400; fill:var(--muted); }
  .lseq { font-size:11px; font-weight:600; fill:#344054; }
  .hstat { font-size:10px; }
  .hrun { fill:#e9edf3; }
  .hseq { fill:#f2f4f7; }
  .seqhead { cursor:pointer; }
  .fstep { font-size:9.5px; font-weight:600; fill:#101828; }
  .tiny .cnt, .tiny .pill { display:none; }
  .edge { fill:none; stroke-width:1.5; stroke-opacity:.22; }
  .edge:hover { stroke-opacity:.9; }
  #edges.dim { opacity:.35; }
  .hide-links #edges { display:none; }
  .noop { display:none; }
  .show-noop .noop { display:inline; }
  .opnode { cursor:pointer; }
  .opnode.located path, .opnode.located rect { stroke:#101828; stroke-width:2.5; }
  .op-recall.hot path, .op-recall.hot rect { stroke:#101828; stroke-width:1.2; }
  .ghost { cursor:help; }
  #hot, #guide, #focusmark { pointer-events:none; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div id="summary"></div>
  <div class="controls">
    <input id="search" type="search" placeholder="Search memories  ( / )">
    <label title="show every faint link from a memory's write to its recalls (hovering a node or memory traces its own links either way)"><input type="checkbox" id="f-links"> all links</label>
    <label title="writes that stored and deleted nothing (e.g. every candidate rejected)"><input type="checkbox" id="f-noop"> no-op writes</label>
    <span><button id="z-out" title="zoom out (−)">−</button> <button id="z-fit" title="fit to window (0)">fit</button> <button id="z-in" title="zoom in (+)">+</button></span>
  </div>
</header>
<div id="key">
  <span title="Write (extraction). Green: memories stored · amber: pending review · gray: nothing stored · red: unparseable. The number counts memories written; ✕n counts memories deleted."><svg width="12" height="12"><path d="M6,1 L1,11 L11,11 Z" fill="#2e9e5b"/></svg>write</span>
  <span title="Recall: memories delivered to the agent (hover for the count); hollow = nothing delivered"><svg width="12" height="12"><path d="M6,11 L1,1 L11,1 Z" fill="#3567d6"/></svg>recall</span>
  <span title="The same memory set recalled at n consecutive steps (×n)"><svg width="24" height="12"><rect x="1" y="1" width="22" height="10" rx="5" fill="#3567d6"/></svg>repeated recall</span>
  <span title="Memories first seen in recalls: they were created before the loaded runs. The number counts them; select one to trace its recalls."><svg width="12" height="12"><path d="M6,1 L1,11 L11,11 Z" fill="none" stroke="#98a2b3" stroke-width="1.4" stroke-dasharray="2.5 2"/></svg>created earlier</span>
  <span title="A memory's write → each of its recalls. Hover a node to trace its memories; hover or select a memory to trace it, including updates and deletions (dashed). Tick “all links” to draw every link."><svg width="22" height="12"><path d="M1,2 C10,2 12,10 21,10" fill="none" stroke="#b06ab3" stroke-width="1.6"/></svg>memory link</span>
  <span title="Context history rewritten (spliced or compacted); violet when on the memory lane"><svg width="12" height="12"><rect x="1" y="1" width="10" height="10" rx="2" fill="#e4e8ee"/></svg>history rewrite</span>
  <span id="key-verdicts" hidden></span>
  <span class="hint">click a node, memory, or lane step · <kbd>+</kbd>/<kbd>−</kbd> zoom · <kbd>Esc</kbd> clear</span>
</div>
<div id="main">
  <div id="canvas"></div>
  <aside id="sidebar">
    <div id="memhdr"></div>
    <ul id="memlist"></ul>
    <div id="detail"></div>
  </aside>
</div>
<footer><details><summary id="vsum"></summary><ul id="vlist"></ul></details></footer>
<script>
"use strict";
const G = __GRAPH_JSON__;

// ---------- helpers ----------
function esc(s){ return String(s==null?"":s).replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
function hue(key){ let h=0; for(const c of key) h=(h*31+c.charCodeAt(0))>>>0; return h%360; }
const memColor=id=>`hsl(${hue(id)},65%,42%)`;
function shorten(s,n){ s=String(s||"").replace(/\s+/g," ").trim(); return s.length>n? s.slice(0,n-1)+"…" : s; }
const obj=x=>x&&typeof x==="object"?x:{};
// lookup tables indexed by artifact-controlled strings: null-prototype, so a
// "constructor" verdict or status resolves to nothing, never to a member
const table=o=>Object.assign(Object.create(null),o);
// Memory identity is the (namespace, item) pair — the same identity the graph
// registry keys on; a bare key would collapse two stores' (or two same-store
// items') memories into one. The id string is JSON: no separator to collide
// with identity content, null and "" stay distinct. A non-string namespace
// names no store (the same rule as the registry's namespace_of). Display text
// stays the bare key everywhere; the namespace only disambiguates on demand.
const memId=m=>JSON.stringify([typeof m.namespace==="string"?m.namespace:null, m.grouping_id||m.item_id||m.key]);
const keyOf=id=>JSON.parse(id)[1];
const memById=Object.fromEntries(G.memories.map(m=>[memId(m),m]));
// display text for a memory id: the registered memory's bare key (the item
// identity itself only when the memory is not in the registry)
const dispKey=id=>{const m=memById[id]; return m?m.key:keyOf(id);};
// Bare-key contexts (injudge memory labels, namespace-less deletion targets)
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
const VD_COLOR=table({positive:"#2e9e5b",negative:"#d64545",unresolved:"#7a5af8",neutral:"#98a2b3",error:"#e2a93b"});
const TALLY=["positive","negative","unresolved","neutral","error","unjudged","unassessable","empty"];
// with verdicts merged, unjudged recalls fade so the judged ones stand out
const RECALL_FILL=G.tool&&G.tool.verdicts? "#aebfee" : "var(--recall)";
const TALLY_COLOR={...VD_COLOR, unjudged:"#aebfee", unassessable:"#d0d5dd", empty:"#eceff3"};
const VD_TIP={positive:"a memory was used and improved later work (judge assessment)",
  negative:"a memory was acted on without benefit (judge assessment; not proven harm)",
  unresolved:"a memory was acted on, but its effect could not be determined (uncertainty, not a null effect)",
  neutral:"no influence established (includes referenced-but-unacted)",
  error:"the judge failed (excluded from rates)", unjudged:"not judged in the merged report",
  unassessable:"no observed later turn (outside every rate)", empty:"nothing delivered (outside every rate)"};
const USAGE_COLOR=table({used:"#2e9e5b",noted:"#3567d6",ignored:"#98a2b3",misused:"#d64545",unclear:"#7a5af8"});
const STATUS_COLOR=table({approved:"var(--write)",pending:"var(--pending)",pending_review:"var(--pending)"});
const ACTION={created:["▲ created","var(--write)"],updated:["↻ updated","var(--pending)"],deleted:["✕ deleted","var(--flag)"],recalled:["▼ recalled","var(--recall)"]};
// "20260727-065319.391Z-cure-v2-9eb25cc9" → {main:"cure-v2 · #9eb25cc9", sub:"2026-07-27 06:53Z"}
// run_id is stringified first: a hand-edited run.json can carry a non-string,
// and String.prototype.match on it would throw and kill the whole page.
function prettyLabel(t){
  const m=String(t.run_id||"").match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})\.\d+Z-(.+)-([0-9a-f]{6,})$/);
  if(!m) return {main:shorten(t.run_id,34), sub:""};
  return {main:`${m[7]} · #${m[8]}`, sub:`${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}Z`};
}
// the recorded instance id (memory_session) names a run when present — a
// corrupt non-string one is ignored like any other drifted datum
const runName=t=>(typeof t.instance_id==="string"&&t.instance_id)||prettyLabel(t).main;

// ---------- rows: run → sequence → episode ----------
const opsById=Object.fromEntries(G.ops.map(o=>[o.id,o]));
const opOrder=Object.fromEntries(G.ops.map((o,i)=>[o.id,i]));
const busy=new Set(G.ops.map(o=>o.traj));
const idle=G.trajectories.filter(t=>!busy.has(t.id));
// Display order: a run that records several episodes gets a header, then a
// header per sequence and one row per episode (x = step within the episode);
// a run of one episode or none is a single row on its own steps. Ops join
// their recorded episode's row; any outside every episode join a trailing
// row of that run.
const outline=[], rowOf={};
for(const t of G.trajectories){
  if(!busy.has(t.id)) continue;
  const ops=G.ops.filter(o=>o.traj===t.id), eps=Array.isArray(t.episodes)? t.episodes : [];
  if(eps.length<2){
    const r={kind:"row", t, ep:null, base:0, len:t.n_steps||0, ops};
    outline.push(r); for(const o of ops) rowOf[o.id]=r;
    continue;
  }
  outline.push({kind:"run", t, n:eps.length});
  const seqs=new Map(), epRow=[];
  eps.forEach((e,i)=>{ e=obj(e); const k=typeof e.sequence_id==="string"? e.sequence_id : null; if(!seqs.has(k)) seqs.set(k,[]); seqs.get(k).push([i,e]); });
  for(const [name,members] of seqs){
    const seq={kind:"seq", t, name, key:JSON.stringify([t.id,name]), n:members.length};
    outline.push(seq);
    for(const [i,e] of members){
      const ok=Number.isInteger(e.first_step)&&Number.isInteger(e.last_step);
      outline.push(epRow[i]={kind:"row", t, ep:e, seq, base:ok? e.first_step-1 : 0, len:ok? e.last_step-e.first_step+1 : 0, ops:[]});
    }
  }
  let loose=null;
  for(const o of ops){
    const r=(Number.isInteger(o.episode)&&epRow[o.episode])||(loose=loose||{kind:"row", t, ep:null, loose:true, base:0, len:0, ops:[]});
    r.ops.push(o); rowOf[o.id]=r;
  }
  if(loose) outline.push(loose);
}
const rows=outline.filter(e=>e.kind==="row");
rows.forEach((r,i)=>r.i=i);
// an op's x is its step within its row; null parks it in the no-step gutter
const localStep=o=>{ const r=rowOf[o.id]; return !r||r.loose||o.step==null? null : Math.max(0,o.step-r.base); };
const maxSteps=G.ops.reduce((a,o)=>Math.max(a,localStep(o)||0), rows.reduce((a,r)=>Math.max(a,r.len),10));
const rowName=r=>r.ep? (typeof r.ep.episode_id==="string"&&r.ep.episode_id||`${r.t.id} unnamed episode`)
  : r.loose? `${r.t.id} outside episodes` : `${r.t.id} ${runName(r.t)}`;
// episode rows count steps from the episode start; the global step stays
// alongside — injudge's analyses cite global steps
const runTip=t=>esc([t.run_id, t.status, t.adapter].filter(Boolean).join(" · "));
const where=o=>{
  const r=rowOf[o.id], L=localStep(o);
  if(!r||!r.ep) return `${o.traj} ${o.step==null?"(no step)":`step ${o.step}`}`;
  return `${rowName(r)} ${L==null?"(no step)":`step ${L} · global ${o.step}`}`;
};
const stepOrd=s=>s==null?1e15:s;
// most-recalled first: the memories that reached the most turns lead the list
const memOrder=G.memories.map(m=>[memId(m),m]).sort((a,b)=>(b[1].recalls||[]).length-(a[1].recalls||[]).length);

// collapse consecutive recalls (per row, so never across episodes) with
// identical memory sets into spans; step-less recalls stay singletons
const recallSpans={}; // leader op id -> {ids:[], memIds:[], sig, steps:[a,b]}
const spanLeader={};  // any recall op id -> leader id
const spanDiff={};    // leader id -> {added:[], removed:[]}: why the previous span broke
for(const r of rows){
  const rs=r.ops.filter(o=>o.kind==="recall").sort((a,b)=>stepOrd(a.step)-stepOrd(b.step));
  // memIds are JSON strings (no literal control chars) — the joined
  // signature stays unambiguous
  const sig=ids=>ids.slice().sort().join("\u001f");
  let grp=null, prev=null;
  const close=()=>{
    if(!grp) return;
    recallSpans[grp.ids[0]]=grp;
    for(const id of grp.ids) spanLeader[id]=grp.ids[0];
    const cur=new Set(grp.memIds);
    if(prev){
      const added=[...cur].filter(k=>!prev.has(k)), removed=[...prev].filter(k=>!cur.has(k));
      if(added.length||removed.length) spanDiff[grp.ids[0]]={added,removed};
    }
    prev=cur;
  };
  for(const o of rs){
    const ids=o.memories.map(memId);
    if(grp&&o.step!=null&&grp.steps[1]!=null&&o.step===grp.steps[1]+1&&sig(ids)===grp.sig){ grp.ids.push(o.id); grp.steps[1]=o.step; continue; }
    close();
    grp={ids:[o.id], memIds:ids, sig:sig(ids), steps:[o.step,o.step]};
  }
  close();
}

// history rewrites at consecutive steps merge into one band per row (an
// episode row takes those inside its step range), in row-local steps;
// rewrites always carry an int step (parse_trajectory computes it)
for(const r of rows){
  r.bands=[];
  if(r.loose) continue;
  const inside=rw=>!r.ep||(rw.step>r.base&&rw.step<=r.base+r.len);
  for(const rw of (r.t.rewrites||[]).filter(inside).sort((a,b)=>a.step-b.step)){
    const s=rw.step-r.base, g=r.bands[r.bands.length-1];
    if(g&&s<=g.b+1){ g.b=Math.max(g.b,s); g.items.push(rw); }
    else r.bands.push({a:s,b:s,items:[rw]});
  }
}

// memories recalled but never created in the loaded set share one dashed ▲
// per run — the run of their first recall (on its header, or its lone row)
const runOrder=Object.fromEntries(G.trajectories.map((t,i)=>[t.id,i]));
const ghostRun={}, ghostCount={};
for(const m of G.memories){
  if((m.created&&m.created.op)||!(m.recalls||[]).length) continue;
  const first=m.recalls.reduce((a,r)=>(runOrder[r.traj]??Infinity)<(runOrder[a.traj]??Infinity)? r : a);
  if(!busy.has(first.traj)) continue;
  ghostRun[memId(m)]=first.traj; ghostCount[first.traj]=(ghostCount[first.traj]||0)+1;
}

// ---------- svg build (zoom-dependent) ----------
const LABEL_W=230, PAD=22, TOP=26, RIGHT=24, BOTTOM=18, GAP=3;
const H={run:28, seq:24, row:40, empty:20};
// step-less ops park in a gutter right of the last step instead of piling onto it
const GUTTER=G.ops.some(o=>localStep(o)==null)?64:0;
const collapsed=new Set(); // sequence keys whose episode rows are hidden
let PX=1, W=0, height=0, fitted=true, pos={}, ghostPos={}; // pos: drawn op id -> {x,y}
const plotW=()=>W-LABEL_W-PAD-RIGHT-GUTTER;
const X=s=>LABEL_W+PAD+(Math.max(0,s)/maxSteps)*plotW();
const stepAt=x=>Math.round((x-LABEL_W-PAD)/plotW()*maxSteps);
const mid=e=>e.y+e.h/2;
function tri(cx,cy,up,r){ const s=up?1:-1; return `M${cx},${cy-s*r} L${cx-r},${cy+s*r} L${cx+r},${cy+s*r} Z`; }
function niceStep(raw){ const p=10**Math.floor(Math.log10(Math.max(1,raw))); return [1,2,5,10].map(m=>m*p).find(m=>m>=raw); }
// ▲ bottom → ▼/pill top
function curve(a,b){
  const y0=a.y+8, y1=b.y-8, dy=Math.max(18,Math.min(80,Math.abs(y1-y0)*0.45));
  return `M${a.x},${y0} C${a.x},${y0+dy} ${b.x},${y1-dy} ${b.x},${y1}`;
}
// one link per recall span that delivered the memory (a span's members share
// its leader's memory set, so the leader carries the memory whenever any does)
function links(m){
  const id=memId(m), c=m.created&&m.created.op;
  const from=c? pos[c] : ghostPos[ghostRun[id]];
  if(!from) return [];
  return (m.recalls||[]).filter(r=>spanLeader[r.op]===r.op&&pos[r.op]).map(r=>curve(from,pos[r.op]));
}
// y and height per outline entry; rows of a collapsed sequence get y=null
function layout(){
  let y=TOP;
  for(const e of outline){
    e.y=null;
    if(e.kind==="row"&&e.seq&&collapsed.has(e.seq.key)) continue;
    e.h=e.kind==="row"? (e.ops.length? H.row : H.empty) : H[e.kind];
    e.y=y; y+=e.h+GAP;
  }
  height=y+BOTTOM;
}
function buildSVG(){
  layout();
  W=Math.max(canvas.clientWidth, LABEL_W+PAD+RIGHT+GUTTER+maxSteps*PX);
  pos={}; ghostPos={};
  // drawn ops sharing one (row, kind, step) dodge sideways instead of stacking
  const slots={}, gx=X(maxSteps)+GUTTER/2+6;
  for(const o of G.ops){
    const r=rowOf[o.id];
    if(!r||r.y==null||(o.kind==="recall"&&!recallSpans[o.id])) continue;
    const k=r.i+"|"+o.kind+"|"+localStep(o);
    (slots[k]=slots[k]||[]).push(o);
  }
  for(const slot of Object.values(slots)) slot.forEach((o,j)=>{
    const L=localStep(o);
    pos[o.id]={x:(L==null?gx:X(L))+(j-(slot.length-1)/2)*12, y:mid(rowOf[o.id])+(o.kind==="extract"?-10:10)};
  });
  // a pill sits at its span's true start step, never dodged
  for(const [lead,g] of Object.entries(recallSpans)) if(g.ids.length>1&&g.steps[0]!=null&&pos[lead]) pos[lead].x=X(g.steps[0]-rowOf[lead].base);

  const s=[`<svg id="graph" width="${W}" height="${height}" xmlns="http://www.w3.org/2000/svg">`,
    // open arrowhead — visually distinct from the ▲/▼ op glyphs
    `<defs><marker id="arr" viewBox="0 0 8 8" refX="6.6" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0.8,0.8 L7,4 L0.8,7.2" fill="none" style="stroke:#475467;stroke:context-stroke" stroke-width="1.5"/></marker></defs>`];
  const every=niceStep(70*maxSteps/plotW());
  for(let st=0;st<=maxSteps;st+=every){
    const x=X(st);
    s.push(`<line x1="${x}" y1="${TOP-10}" x2="${x}" y2="${height-BOTTOM}" stroke="#f0f2f5"/><text x="${x}" y="${TOP-14}" class="tick">${st}</text>`);
  }
  const ghost=(t,x,y)=>{
    const n=ghostCount[t.id]; if(!n) return;
    ghostPos[t.id]={x,y};
    s.push(`<g class="ghost" transform="translate(${x},${y})"><path d="${tri(0,0,true,7)}" fill="#fff" stroke="#98a2b3" stroke-width="1.5" stroke-dasharray="3 2"/><text x="-6" y="-10" class="gcnt">${n}</text><title>${n} memor${n===1?"y":"ies"} first seen in recalls — created before the loaded runs; select one to trace it</title></g>`);
  };
  for(const e of outline){
    if(e.y==null) continue;
    if(e.kind==="run"){
      s.push(`<rect y="${e.y}" width="${W}" height="${e.h}" class="hrun"/><text x="${LABEL_W+(ghostCount[e.t.id]?34:PAD)}" y="${mid(e)+4}" class="hstat">${esc([`${e.t.n_steps} steps`, `${e.n} episodes`, prettyLabel(e.t).sub, e.t.model].filter(Boolean).join(" · "))}</text>`);
      ghost(e.t, LABEL_W+12, mid(e)-1);
      continue;
    }
    if(e.kind==="seq"){ s.push(`<rect y="${e.y}" width="${W}" height="${e.h}" class="hseq"/>`); continue; }
    const y=mid(e), x0=X(0)-6, end=X(e.len)+6;
    s.push(`<rect x="${x0}" y="${e.y}" width="${end-x0}" height="${e.h}" rx="5" fill="#f8f9fb"/>`);
    for(const b of e.bands){
      const h=Math.max(1.5,Math.min(PX/2,6)), bx0=X(b.a)-h, bx1=X(b.b)+h, rw=b.items[0];
      const srcs=[...new Set(b.items.map(r=>r.proxy_source||"?"))].map(s=>esc(s)).join(" + ");
      const tip=b.items.length>1? `${b.items.length} history rewrites, steps ${b.a}–${b.b} (${srcs})`
        : `history rewritten at step ${b.a}: kept ${esc(rw.kept)}, dropped ${esc(rw.dropped)} (${srcs})`;
      s.push(`<rect x="${bx0}" y="${e.y}" width="${bx1-bx0}" height="${e.h}" fill="${b.items.every(r=>r.on_memory_lane)?"#efe4f8":"#e9ecf1"}"><title>${tip}</title></rect>`);
      if(b.items.length>1&&bx1-bx0>=22) s.push(`<text x="${bx0+4}" y="${e.y+10}" class="tick l">↻${b.items.length}</text>`);
    }
    if(e.ops.length) s.push(`<line x1="${x0}" y1="${y}" x2="${end}" y2="${y}" stroke="#d5dae1"/>`);
    if(!e.ep&&!e.loose) ghost(e.t, LABEL_W+10, y-10);
  }
  // faint links for memories created in the loaded set; pre-existing
  // memories' links are drawn only when traced (#hot)
  const lk=[];
  for(const m of G.memories){
    if(!(m.created&&m.created.op)) continue;
    const id=memId(m);
    for(const d of links(m)) lk.push(`<path class="edge" data-mem="${esc(id)}" d="${d}" stroke="${memColor(id)}"><title>${esc(m.key)}</title></path>`);
  }
  s.push(`<g id="edges">${lk.join("")}</g><g id="hot"></g>`);
  const late=[];
  for(const o of G.ops){
    const p=pos[o.id]; if(!p) continue;
    if(o.kind==="extract"){
      // a recorded non-approved review status is pending; no status means
      // the adapter stores without review
      const pend=o.candidates.filter(c=>c.review_status&&c.review_status!=="approved").length;
      const stored=o.candidates.length-pend, rej=o.rejections?o.rejections.count:0, ndel=(o.deletions||[]).length;
      const fill=o.flagged? "var(--flag)" : stored? "var(--write)" : pend? "var(--pending)" : "var(--reject)";
      const noop=!o.flagged&&!o.candidates.length&&!ndel;
      const tip=`${o.id} · write at ${esc(where(o))}: ${stored} stored${pend?`, ${pend} pending review`:""}${rej?`, ${rej} rejected`:""}${ndel?`, ${ndel} deleted`:""}${o.flagged?` — FLAGGED: ${esc(o.flagged)}`:""}`;
      const label=o.flagged? "!" : o.candidates.length||"";
      s.push(`<g class="opnode op-extract${noop?" noop":""}" data-op="${o.id}" transform="translate(${p.x},${p.y})"><path d="${tri(0,0,true,7)}" fill="${fill}"/>${label!==""?`<text y="-10" class="cnt">${label}</text>`:""}${ndel?`<text x="9" y="4" class="cnt del">✕${ndel}</text>`:""}<title>${tip}</title></g>`);
    } else {
      const g=recallSpans[o.id], n=g.ids.length, keys=new Set(g.memIds).size, base=rowOf[o.id].base;
      const stepTxt=g.steps[0]==null? "no step" : `step ${g.steps[0]-base}${g.steps[1]>g.steps[0]?"–"+(g.steps[1]-base):""}`;
      // verdict coloring: a pill can cover several judged spans — it takes a
      // verdict color only when every judged span it covers agrees (verdicts
      // always follow the judged span, never the pill); a mixed pill stays
      // uncolored and says so, the detail panel lists each span's verdict
      const spanIds=new Set(g.ids.map(id=>opsById[id].verdict_span).filter(Boolean));
      const svd=[...new Set(g.ids.map(id=>opsById[id].span_verdict).filter(Boolean))];
      const vcol=svd.length===1&&VD_COLOR[svd[0]], fill=vcol||RECALL_FILL;
      const vdTxt=svd.length===1? ` — verdict ${esc(svd[0])}${spanIds.size>1?` (${spanIds.size} judged spans)`:""}`
        : svd.length>1? ` — mixed verdicts across ${spanIds.size} judged spans` : "";
      const df=spanDiff[o.id];
      const tip=`${o.id} · recall at ${stepTxt}: ${keys} memor${keys===1?"y":"ies"}${n>1?`, same set ${n}× in a row`:""}${df?` (+${df.added.length} −${df.removed.length} vs previous recall)`:""}${vdTxt}`;
      const w=n>1? Math.max(12, X(g.steps[1]-base)-X(g.steps[0]-base)+12) : 0;
      const shape=n>1? `<rect x="-6" y="-6" width="${w}" height="12" rx="6" fill="${fill}"/>${w>=26?`<text x="${w/2-6}" y="3" class="pill">×${n}</text>`:""}`
        : keys? `<path d="${tri(0,0,false,7)}" fill="${fill}"/>` : `<path d="${tri(0,0,false,6)}" fill="#fff" stroke="${fill}" stroke-width="1.5"/>`;
      // judged recalls paint last so neighbors never cover their color
      (vcol?late:s).push(`<g class="opnode op-recall" data-op="${o.id}" transform="translate(${p.x},${p.y})">${shape}<title>${tip}</title></g>`);
    }
  }
  s.push(...late);
  if(GUTTER){
    const x=X(maxSteps)+12;
    s.push(`<line x1="${x}" y1="${TOP-10}" x2="${x}" y2="${height-BOTTOM}" stroke="#e4e7ec" stroke-dasharray="4 3"/><text x="${x+6}" y="${TOP-14}" class="tick l">no step</text>`);
  }
  s.push(`<g id="focusmark"></g><g id="guide" style="display:none"><line y1="${TOP-10}" y2="${height-BOTTOM}" stroke="#98a2b3"/><text class="tick"></text></g>`);
  // row labels and headers, kept pinned on horizontal scroll (pinLabels)
  const anyEp=rows.some(r=>r.ep);
  s.push(`<g id="labels"><rect width="${LABEL_W}" height="${height}" fill="#fff"/><text x="${LABEL_W-8}" y="${TOP-14}" class="tick r">${anyEp?"step in episode":"agent step"} →</text>`);
  for(const e of outline){
    if(e.y==null) continue;
    const y=mid(e), t=e.t;
    if(e.kind==="run"){
      s.push(`<g><rect y="${e.y}" width="${LABEL_W}" height="${e.h}" class="hrun"/><title>${runTip(t)}</title><text x="12" y="${y+4}" class="lname"><tspan>${esc(t.id)}</tspan> ${esc(shorten(runName(t),26))}</text></g>`);
    } else if(e.kind==="seq"){
      s.push(`<g class="seqhead" data-seq="${esc(e.key)}"><rect y="${e.y}" width="${LABEL_W}" height="${e.h}" class="hseq"/><text x="12" y="${y+4}" class="lseq">${collapsed.has(e.key)?"▸":"▾"} ${esc(shorten(e.name??"no sequence",20))} · ${e.n} episode${e.n===1?"":"s"}</text></g>`);
    } else if(e.ep){
      const st=typeof e.ep.status==="string"&&e.ep.status!=="prepared"? ` · ${e.ep.status}` : "";
      const sub=(e.len? `global steps ${e.base+1}–${e.base+e.len}` : "no agent step")+st;
      s.push(`<g><title>${esc(rowName(e))}${esc(st)}</title><text x="22" y="${e.ops.length?y-2:y+4}" class="lep${e.ops.length?"":" none"}">${esc(shorten(rowName(e),30))}</text>${e.ops.length?`<text x="22" y="${y+11}" class="lsub${st?" warn":""}">${esc(sub)}</text>`:""}</g>`);
    } else {
      const sub=e.loose? "ops outside every recorded episode" : [`${t.n_steps} steps`, prettyLabel(t).sub, t.model].filter(Boolean).join(" · ");
      s.push(`<g><title>${runTip(t)}</title><text x="12" y="${y-2}" class="lname"><tspan>${esc(t.id)}</tspan> ${esc(shorten(e.loose?"outside episodes":runName(t),26))}</text><text x="12" y="${y+11}" class="lsub">${esc(shorten(sub,40))}</text></g>`);
    }
  }
  s.push(`</g></svg>`);
  return s.join("");
}

// ---------- state: traced memory, step inspector ----------
const canvas=document.getElementById("canvas"), detail=document.getElementById("detail"),
      memlist=document.getElementById("memlist"), memhdr=document.getElementById("memhdr"),
      search=document.getElementById("search");
let selected=null, hoverMem=null, hoverOp=null, focus=null; // focus={row,step}: the step inspector (row index, row-local step)
const HELP=`<div class="help"><p><b>Select a memory</b> in the list or <b>click a node</b> to see its details here.</p>
  <p>Rows are episodes, grouped by run and sequence (click a sequence header to fold it); x = step within the episode. Writes ▲ sit above the line, recalls ▼ below. Hover a node to see where its memories came from or went; hovering or selecting a memory traces its full path.</p>
  <p>Click empty row space to list the memory actions at that step; ←/→ moves the step.</p></div>`;

// the traced memory: every link (pre-existing memories start at their lane's
// dashed ▲) plus its creation → update → deletion chain, dashed
function hotSvg(id,m){
  const col=memColor(id);
  let s=links(m).map(d=>`<path d="${d}" fill="none" stroke="${col}" stroke-width="2.4" marker-end="url(#arr)"/>`).join("");
  const chain=[...new Set([m.created&&m.created.op, ...(m.updates||[]).map(u=>u.op), ...(m.deletions||[]).map(d=>d.op)])]
    .filter(op=>op&&pos[op]).sort((p,q)=>opOrder[p]-opOrder[q]);
  for(let i=0;i+1<chain.length;i++){
    const a=pos[chain[i]], b=pos[chain[i+1]];
    s+=`<path d="M${a.x},${a.y-9} Q${(a.x+b.x)/2},${Math.min(a.y,b.y)-28} ${b.x},${b.y-9}" fill="none" stroke="${col}" stroke-width="2" stroke-dasharray="5 4"/>`;
  }
  return s;
}
// a hovered node's own links: where a recall's memories came from, or where
// a write's memories were recalled afterwards
function opLinks(opId){
  const o=opsById[opId], p=pos[opId]; if(!o||!p) return "";
  const path=(d,id)=>`<path d="${d}" fill="none" stroke="${memColor(id)}" stroke-width="2" marker-end="url(#arr)"/>`;
  let s="";
  if(o.kind==="recall"){
    for(const id of new Set(recallSpans[opId].memIds)){
      const m=memById[id], c=m&&m.created&&m.created.op, from=c? pos[c] : ghostPos[ghostRun[id]];
      if(from&&from!==p) s+=path(curve(from,p),id);
    }
  } else {
    for(const c of o.candidates){
      const id=memId(c), m=c.key&&memById[id];
      if(m) for(const r of m.recalls||[]) if(spanLeader[r.op]===r.op&&pos[r.op]&&opOrder[r.op]>opOrder[opId]) s+=path(curve(p,pos[r.op]),id);
    }
  }
  return s;
}
function paint(){
  const hot=document.getElementById("hot"); if(!hot) return;
  const id=selected||hoverMem, m=id&&memById[id];
  hot.innerHTML=m? hotSvg(id,m) : hoverOp? opLinks(hoverOp) : "";
  document.getElementById("edges").classList.toggle("dim", !!(m||hoverOp));
  // outline the recall spans that delivered the traced memory
  document.querySelectorAll(".op-recall").forEach(n=>n.classList.toggle("hot", !!(m&&recallSpans[n.dataset.op].memIds.includes(id))));
}
function drawFocus(){
  const g=document.getElementById("focusmark"); if(!g) return;
  const r=focus&&rows[focus.row];
  if(!r||r.y==null){ g.innerHTML=""; return; }
  const x=X(focus.step);
  g.innerHTML=`<line x1="${x}" y1="${r.y}" x2="${x}" y2="${r.y+r.h}" stroke="#101828" stroke-width="1.5"/><text x="${x+4}" y="${r.y+10}" class="fstep">@${focus.step}</text>`;
}
// unfold the sequence holding an op so it can be drawn and scrolled to
function reveal(opId){
  const r=rowOf[opId];
  if(r&&r.seq&&collapsed.delete(r.seq.key)) mount();
}
function locate(opId){
  if(opId) reveal(opId);
  const el=opId&&document.querySelector(`.opnode[data-op="${CSS.escape(spanLeader[opId]||opId)}"]`);
  if(!el) return;
  el.scrollIntoView({behavior:"smooth", block:"center", inline:"center"});
  el.classList.add("located");
  setTimeout(()=>el.classList.remove("located"), 1700);
}
// memory actions at (row, step): memory id -> Set("created"|"updated"|"deleted"|"recalled")
const opsAt=(row,step)=>rows[row].ops.filter(o=>localStep(o)===step);
function actionsAt(row, step){
  const acts={};
  const add=(id,kind)=>{(acts[id]=acts[id]||new Set()).add(kind);};
  for(const o of opsAt(row,step)){
    if(o.kind==="extract"){
      for(const c of o.candidates){
        if(!c.key) continue;
        const id=memId(c), m=memById[id];
        add(id, m&&m.created&&m.created.op===o.id ? "created" : "updated");
      }
      for(const m of G.memories) if((m.deletions||[]).some(d=>d.op===o.id)) add(memId(m),"deleted");
    } else {
      for(const dm of o.memories) if(dm.key) add(memId(dm),"recalled");
    }
  }
  return acts;
}
function setFocus(f){
  focus=f; selected=null; paint(); drawFocus(); renderList();
  if(!focus){ detail.innerHTML=HELP; return; }
  const ops=opsAt(focus.row,focus.step), r=rows[focus.row];
  detail.innerHTML=`<h3>${esc(rowName(r))} step ${focus.step}</h3>${r.ep&&focus.step?`<div class="sub">global step ${r.base+focus.step}</div>`:""}`+(ops.length
    ? ops.map(o=>`<div class="row">${opLink(o.id, `${o.kind==="extract"?"▲ write":"▼ recall"} ${o.id}`)}</div>`).join("")
    : `<div class="sub">No memory operation at this step.</div>`);
}
function clearAll(){ hoverMem=null; setFocus(null); }

// ---------- sidebar: memory list ----------
function renderList(){
  const q=search.value.trim().toLowerCase(), acts=focus? actionsAt(focus.row, focus.step) : null;
  const items=[];
  for(const [mid,m] of memOrder){
    const kinds=acts&&acts[mid];
    if(acts&&!kinds) continue; // step inspector: only memories acting at this step
    if(q&&!String(m.key).toLowerCase().includes(q)&&!String(m.value||"").toLowerCase().includes(q)) continue;
    const used=(m.recalls||[]).length, val=shorten(m.value,140);
    // terminal state wins: a deleted memory shows "deleted", not its last review status
    const badge=kinds? [...kinds].map(k=>`<b style="color:${ACTION[k][1]}">${ACTION[k][0]}</b>`).join(" ")
      : (m.deleted? tag("deleted","var(--flag)") : m.status!=="approved"&&STATUS_COLOR[m.status]? tag(m.status,STATUS_COLOR[m.status]) : "")
        +`<span title="recalled ${used}×">▼${used}</span>`;
    items.push(`<li data-mem="${esc(mid)}"${mid===selected?' class="sel"':""}><i class="dot" style="background:${memColor(mid)}"></i><div class="txt"><div class="k">${esc(m.key)}</div>${val&&val!==m.key?`<div class="v">${esc(val)}</div>`:""}</div><span class="n">${badge}</span></li>`);
  }
  memlist.innerHTML=items.join("")||`<li class="empty">${acts?"no memory actions at this step":"no matching memories"}</li>`;
  memhdr.innerHTML=focus? `Memories at ${esc(rowName(rows[focus.row]))} step ${focus.step} <button id="unfocus" title="show all memories (Esc)">✕</button>`
    : `Memories <span class="muted">${items.length<memOrder.length?`${items.length} of `:""}${memOrder.length} · most recalled first</span>`;
}
memlist.addEventListener("click",ev=>{ const li=ev.target.closest("li[data-mem]"); if(li) selectMemory(li.dataset.mem); });
memlist.addEventListener("mouseover",ev=>{ const li=ev.target.closest("li[data-mem]"), id=li?li.dataset.mem:null; if(id!==hoverMem){ hoverMem=id; paint(); } });
memlist.addEventListener("mouseleave",()=>{ hoverMem=null; paint(); });
memhdr.addEventListener("click",ev=>{ if(ev.target.id==="unfocus") clearAll(); });
search.addEventListener("input",renderList);

// ---------- detail panel ----------
const tag=(t,col)=>`<span class="tag"${col?` style="color:${col}"`:""}>${esc(t)}</span>`;
function memChip(id,sign,label){
  return `<span class="tag link" data-mem="${esc(id)}" title="select memory">${sign||""}${esc(label||dispKey(id))}</span>`;
}
// a bare key links only when it names exactly one loaded memory (an ambiguous
// or unknown key renders as plain text, never a wrong link); `text` is the
// display label when it differs from the looked-up key (an injudge span-local
// id chip showing its memory's key)
function chipForKey(key,sign,text){
  const id=idByKey[key], label=text||key;
  return id? memChip(id,sign,label) : `<span class="tag">${sign||""}${esc(label)}</span>`;
}
const opLink=(id,text)=>`<span class="link" data-op="${esc(id)}">${esc(text)}</span>`;
// recalls per row, as row-local step ranges
function collapseUses(recalls){
  const byRow=new Map();
  for(const r of recalls){
    const o=opsById[r.op], row=o&&rowOf[o.id], k=row? rowName(row) : r.traj;
    if(!byRow.has(k)) byRow.set(k,[]);
    byRow.get(k).push(o? localStep(o) : r.step);
  }
  return [...byRow].map(([t,ss])=>{
    ss=[...new Set(ss.filter(s=>s!=null))].sort((a,b)=>a-b);
    const ranges=[]; let a=null,b=null;
    for(const s of ss){ if(a==null){a=b=s;continue;} if(s===b+1){b=s;continue;} ranges.push([a,b]); a=b=s; }
    if(a!=null) ranges.push([a,b]);
    return `${esc(t)} @ `+(ranges.map(([x,y])=>x===y?`${x}`:`${x}–${y}`).join(", ")||"?");
  }).join(" · ");
}
function selectMemory(id){
  selected=id; hoverMem=null; paint();
  memlist.querySelectorAll("li").forEach(li=>li.classList.toggle("sel", li.dataset.mem===id));
  const li=memlist.querySelector(`li[data-mem="${CSS.escape(id)}"]`);
  if(li) li.scrollIntoView({block:"nearest"});
  const m=memById[id]; if(!m){ detail.innerHTML=HELP; return; }
  const c=m.created||{};
  locate(c.op||((m.recalls||[])[0]||{}).op);
  // one history line: creation → updates → deletions in op order (a revived
  // memory keeps its deletions — the deletes edges carry the same record)
  const hist=[...(c.op?[{op:c.op,what:"created",st:c.status}]:[]),
    ...(m.updates||[]).map(u=>({op:u.op,what:u.duplicate?"re-written":"updated",st:u.review_status})),
    ...(m.deletions||[]).map(d=>({op:d.op,what:"deleted"}))].sort((a,b)=>(opOrder[a.op]??0)-(opOrder[b.op]??0));
  const at=x=>{ const o=opsById[x.op]; return o? where(o) : x.op; };
  const histHtml=hist.length>1? `<div class="sub">History: ${hist.map(x=>opLink(x.op,`${x.what} @ ${at(x)}`)+(x.st?" "+tag(x.st,STATUS_COLOR[x.st]):"")).join(" → ")}</div>` : "";
  // tags follow the list's terminal-state-wins rule: a revived memory deleted
  // again shows "deleted", not "revived"
  detail.innerHTML=`<h3 style="color:${memColor(id)}">${esc(m.key)}</h3>
    <div>${m.deleted?tag("deleted","var(--flag)"):""}${m.revived&&!m.deleted?tag("revived"):""}${STATUS_COLOR[m.status]?tag(m.status,STATUS_COLOR[m.status]):""}${m.memory_type?tag(m.memory_type):""}${m.confidence!=null?`<span class="tag">conf ${esc(m.confidence)}</span>`:""}${m.namespace?`<span class="tag" title="store namespace ${esc(m.namespace)}">store ${esc(shorten(m.namespace,12))}</span>`:""}</div>
    <div class="sub">Created: ${c.op? opLink(c.op, opsById[c.op]? where(opsById[c.op]) : c.op) : "before the loaded runs (origin not observed)"}</div>
    <div class="sub">Recalled: ${m.recalls.length? `${m.recalls.length}× — ${collapseUses(m.recalls)}` : "never"}</div>
    ${histHtml}
    ${(m.recalled_before_creation||[]).length?`<div class="flag">⚠ ${m.recalled_before_creation.length} recall(s) precede this creation — the store entry likely predates the loaded runs</div>`:""}
    <pre>${esc(m.value||"")}</pre>`;
}
// enum glyphs per rubric signal; relevance is the only bool signal
const SIG_ENUM={
  accuracy:{corroborated:["✓","#2e9e5b"],uncontradicted:["≈","#5b8dd9"],contradicted:["✗","#d64545"]},
  novelty:{novel:["✓","#2e9e5b"],confirming:["≈","#5b8dd9"],redundant:["✗","#d64545"]},
  influence:{none:["—","var(--muted)"],referenced:["◔","#5b8dd9"],acted_on:["◑","#2e9e5b"],decisive:["●","#2e9e5b"]},
  positive_effect:{improved:["+","#2e9e5b"],no_effect:["0","#b8860b"],harmed:["✗","#d64545"]},
};
function sigCell(sig,s,abbr){
  const e=obj(obj(sig)[s]), v=e.value;
  let mark="—", col="var(--muted)";
  if(typeof v==="boolean"){
    mark=v?"✓":"✗"; col=v?"#2e9e5b":"#d64545";
  } else if(v!=null){
    const cell=Object.hasOwn(SIG_ENUM[s]||{},v)? SIG_ENUM[s][v] : null;
    // a foreign/hand-edited verdicts artifact can carry an off-enum
    // value — escape it like every other interpolated datum
    if(cell){ mark=cell[0]; col=cell[1]; } else { mark=esc(String(v)); }
  }
  return `<span title="${esc(s)}: ${esc(v==null?"null":String(v))}${e.note?" — "+esc(e.note):""}" style="color:${col}">${abbr}${mark}</span>`;
}
// injudge verdicts for every judged span a drawn recall covers
function verdictsHtml(vops){
  let h="";
  const seen=new Set();
  for(const x of vops){
    if(seen.has(x.verdict_span)) continue;
    seen.add(x.verdict_span);
    const d=x.verdict_detail||{}, v=d.verdict||x.span_verdict||"unjudged", col=VD_COLOR[v]||"var(--muted)";
    const dc=d.delivery_cost||{}, inp=d.input||{};
    // artifact-controlled numbers render escaped like any other datum; the
    // token count is formatted only when it really is a number — 0 included:
    // a redacted/unsized placement counts 0 tokens, and hiding the line would
    // misread as "no deliveries". A missing deliveries count renders "?".
    const meta=[
      d.confidence!=null?`conf ${esc(d.confidence)}`:"",
      typeof dc.approx_tokens_total==="number"?`delivered ${esc(dc.deliveries==null?"?":dc.deliveries)}× ≈${esc(dc.approx_tokens_total.toLocaleString())} tokens`:"",
      // the agent-run record: what judged the span (path is artifact text;
      // the markers only select static suffixes)
      inp.path?`judged by ${esc(inp.path)}${inp.gold?" · gold":""}${inp.forced?" · forced":""}${typeof inp.attempts==="number"?` · ${esc(inp.attempts)} attempt${inp.attempts===1?"":"s"}`:""}`:"",
      d.auto?`auto: ${esc(d.auto)}`:"",
    ].filter(Boolean).join(" · ");
    h+=`<div class="verdict" style="--c:${col}"><div><b style="color:${col}">${esc(v)}</b> <span class="sub">${meta}</span></div>
      <div class="sub">span ${esc(x.verdict_span)}${d.episode_execution_id?` · episode ${esc(d.episode_execution_id)}`:""}</div>`;
    if(d.error) h+=`<div class="flag">judge error: ${esc(d.error)}</div>`;
    if(d.analysis) h+=`<p>${esc(d.analysis)}</p>`;
    // rubric and usage key on span-local entry ids (m1, m2, …): one row per
    // entry, labelled with its memory's display key off memory_labels
    const mLabels=d.memory_labels||{}, rubric=obj(d.rubric), usage=obj(d.memory_usage);
    const ids=[...new Set([...Object.keys(rubric), ...Object.keys(usage)])];
    for(const k of ids){
      const u=obj(usage[k]);
      const lab=typeof mLabels[k]==="string"&&mLabels[k]&&mLabels[k]!==k? mLabels[k] : null;
      h+=`<div class="row">${chipForKey(lab||k,"",lab?`${k} · ${lab}`:k)}${u.usage!=null?`<b style="color:${USAGE_COLOR[u.usage]||"var(--muted)"}" title="${esc(u.evidence||"")}">${esc(u.usage)}</b>`:""}${u.reason?` <span class="sub">(${esc(u.reason)})</span>`:""}`
        +(Object.hasOwn(rubric,k)?`<span class="marks">${sigCell(rubric[k],"relevance","R")}${sigCell(rubric[k],"accuracy","A")}${sigCell(rubric[k],"novelty","N")}${sigCell(rubric[k],"influence","I")}${sigCell(rubric[k],"positive_effect","P")}</span>`:"")+`</div>`;
    }
    if(ids.length) h+=`<div class="sub" style="font-size:11px">R relevant · A accurate · N novel · I influence · P effect — hover a mark for its value</div>`;
    // a hand-edited verdicts artifact can carry null evidence entries or
    // non-array step lists — skip them, never dead-panel the op on a TypeError
    const evs=(Array.isArray(d.evidence)?d.evidence:[]).filter(evv=>evv&&typeof evv==="object");
    if(evs.length){
      h+=`<details><summary>evidence (${evs.length})</summary>`;
      for(const evv of evs){
        const evSteps=Array.isArray(evv.steps)?evv.steps:[], evSeqs=Array.isArray(evv.seqs)?evv.seqs:[];
        h+=`<p>• ${esc(evv.claim||"")} <span class="sub">(steps ${esc(evSteps.join(",")||"?")}; seqs ${esc(evSeqs.join(",")||"?")})</span></p>`;
      }
      h+=`</details>`;
    }
    h+=`</div>`;
  }
  return h;
}
function showOp(opId){
  const o=opsById[opId]; if(!o) return;
  selected=null; paint();
  const L=localStep(o);
  if(L!=null){ focus={row:rowOf[o.id].i, step:L}; drawFocus(); renderList(); } // sync the step inspector
  let h, body;
  if(o.kind==="extract"){
    body={step:o.step,turn:o.turn,status:o.status,trigger:o.trigger,flagged:o.flagged,candidates:o.candidates,deletions:o.deletions,rejections:o.rejections};
    h=`<h3>▲ Write <span class="muted">${esc(o.id)}</span></h3>
      <div class="sub">${esc(where(o))}${o.turn!=null?` · turn ${esc(o.turn)}`:""}${o.status?` · ${esc(o.status)}`:""}${o.trigger?` · trigger ${esc(o.trigger)}`:""}${o.declared_step!=null?` · declared step ${esc(o.declared_step)}`:""}${o.at?` · ${esc(o.at)}`:""}</div>`;
    if(o.flagged) h+=`<div class="flag">⚠ ${esc(o.flagged)}</div>`;
    if(o.candidates.length){
      h+=`<h4>Written (${o.candidates.length})</h4>`;
      for(const c of o.candidates)
        h+=`<div class="row">${c.key?memChip(memId(c)):"—"}${c.review_status?tag(c.review_status,STATUS_COLOR[c.review_status]):""}${c.confidence!=null?`<span class="sub">conf ${esc(c.confidence)}</span>`:""}<div class="v" title="${esc(c.value||"")}">${esc(shorten(c.value,160))}</div></div>`;
    }
    if((o.deletions||[]).length){
      h+=`<h4>Deleted (${o.deletions.length})</h4>`;
      for(const d of o.deletions){
        const tgt=d.target||"";
        // the before-ref pins the exact (namespace, item); a ref-less target
        // resolves by bare key only when unambiguous, else plain text
        const id=d.ref? memId(d.ref) : idByKey[tgt];
        h+=`<div class="row">${id&&memById[id]?memChip(id,"",tgt):esc(tgt)}</div>`;
      }
    }
    if(o.rejections&&o.rejections.count){
      h+=`<h4>Rejected (${esc(o.rejections.count)})</h4><div class="sub">${Object.entries(obj(o.rejections.by_reason)).map(([r,n])=>`${esc(r)} ×${esc(n)}`).join(", ")}</div>`;
    }
  } else {
    const g=recallSpans[opId], ids=g?g.ids:[o.id], steps=g?g.steps:[o.step,o.step];
    body={steps,injections:ids,memories:o.memories,searches:o.searches,queries:o.queries,matched:o.matched,binding_status:o.binding_status};
    const qs=o.queries||[];
    const how=[o.trigger==="explicit"?"agent-directed search":"", o.matched!=null?`search returned ${esc(o.matched)}`:"", o.binding_status?`binding ${esc(o.binding_status)}`:""].filter(Boolean).join(" · ");
    h=`<h3>▼ Recall <span class="muted">${esc(o.id)}</span></h3>
      <div class="sub">${esc(where(o))}${steps[1]>steps[0]?` · same set ${ids.length}× in a row, through step ${steps[1]-rowOf[o.id].base}`:""}${o.turn!=null?` · turn ${esc(o.turn)}`:""}</div>
      ${how?`<div class="sub">${how}</div>`:""}
      ${qs.length?`<div class="sub">Quer${qs.length>1?"ies":"y"}: ${qs.map(q=>`“${esc(shorten(q,160))}”`).join("; ")}</div>`:""}`;
    h+=verdictsHtml(ids.map(id=>opsById[id]).filter(x=>x&&x.verdict_span));
    h+=`<h4>Delivered (${o.memories.length})</h4>`;
    for(const dm of o.memories)
      h+=`<div class="row">${memChip(memId(dm))}${dm.memory_type?tag(dm.memory_type):""}${dm.needs_verification?tag("needs verification","var(--pending)"):""}${dm.confidence!=null?`<span class="sub">conf ${esc(dm.confidence)}</span>`:""}<div class="v" title="${esc(dm.value||"")}">${esc(shorten(dm.value,160))}</div></div>`;
    const df=spanDiff[opId];
    if(df) h+=`<details><summary>vs previous recall: +${df.added.length} new, −${df.removed.length} dropped</summary><div>${df.added.map(id=>memChip(id,"+")).join("")}${df.removed.map(id=>memChip(id,"−")).join("")}</div></details>`;
  }
  if(o.cost) body.cost=o.cost;
  h+=`<details><summary>raw JSON</summary><pre>${esc(JSON.stringify(body,null,1))}</pre></details>`;
  detail.innerHTML=h;
}
detail.addEventListener("click",ev=>{
  const l=ev.target.closest(".link"); if(!l) return;
  if(l.dataset.mem!=null) selectMemory(l.dataset.mem);
  else if(l.dataset.op){ showOp(l.dataset.op); locate(l.dataset.op); }
});

// ---------- canvas: mount, zoom, pointer ----------
function pinLabels(){ const l=document.getElementById("labels"); if(l) l.setAttribute("transform",`translate(${canvas.scrollLeft},0)`); }
function mount(){
  const frac=canvas.scrollWidth? canvas.scrollLeft/canvas.scrollWidth : 0;
  canvas.innerHTML=rows.length? buildSVG() : `<p class="none">No memory operations were recorded in the loaded trajectories.</p>`;
  canvas.scrollLeft=frac*canvas.scrollWidth;
  canvas.classList.toggle("tiny", plotW()/maxSteps<6); // count labels need room
  pinLabels(); paint(); drawFocus();
}
const fitPX=()=>(canvas.clientWidth-LABEL_W-PAD-RIGHT-GUTTER)/maxSteps;
// f=0 fits the window; zooming out never goes below fit
function zoom(f){ fitted=!f; PX=f? Math.min(60,Math.max(fitPX(),PX*f)) : fitPX(); mount(); }
document.getElementById("z-in").onclick=()=>zoom(1.5);
document.getElementById("z-out").onclick=()=>zoom(1/1.5);
document.getElementById("z-fit").onclick=()=>zoom(0);
window.addEventListener("resize",()=>{ if(fitted) zoom(0); });
canvas.addEventListener("scroll",pinLabels);
const overLabels=ev=>ev.clientX-canvas.getBoundingClientRect().left<LABEL_W;
canvas.addEventListener("click",ev=>{
  const node=ev.target.closest(".opnode"); if(node) return showOp(node.dataset.op);
  const edge=ev.target.closest(".edge"); if(edge) return selectMemory(edge.dataset.mem);
  // sequence header → fold/unfold; row click → step inspector; anywhere else clears
  const svg=document.getElementById("graph");
  if(svg){
    const b=svg.getBoundingClientRect(), y=ev.clientY-b.top, step=stepAt(ev.clientX-b.left);
    const e=outline.find(e=>e.y!=null&&y>=e.y&&y<e.y+e.h);
    if(e&&e.kind==="seq"){ collapsed.has(e.key)? collapsed.delete(e.key) : collapsed.add(e.key); return mount(); }
    if(e&&e.kind==="row"&&!overLabels(ev)&&step>=0&&step<=maxSteps)
      return setFocus(focus&&focus.row===e.i&&focus.step===step? null : {row:e.i, step});
  }
  clearAll();
});
canvas.addEventListener("mousemove",ev=>{
  const svg=document.getElementById("graph"), guide=document.getElementById("guide"); if(!svg) return;
  const b=svg.getBoundingClientRect(), step=stepAt(ev.clientX-b.left), on=step>=0&&step<=maxSteps&&!overLabels(ev);
  guide.style.display=on?"":"none";
  if(!on) return;
  // the readout follows the pointer: rows run far below the top axis
  const label=guide.querySelector("text");
  guide.setAttribute("transform",`translate(${X(step)},0)`);
  label.setAttribute("y",ev.clientY-b.top-8);
  label.textContent=`step ${step}`;
});
canvas.addEventListener("mouseleave",()=>{ const g=document.getElementById("guide"); if(g) g.style.display="none"; });
canvas.addEventListener("mouseover",ev=>{
  const e=ev.target.closest(".edge"), n=ev.target.closest(".opnode"), id=e?e.dataset.mem:null, op=n?n.dataset.op:null;
  if(id!==hoverMem||op!==hoverOp){ hoverMem=id; hoverOp=op; paint(); }
});
const fLinks=document.getElementById("f-links"), fNoop=document.getElementById("f-noop");
fLinks.onchange=fNoop.onchange=()=>{ canvas.classList.toggle("hide-links",!fLinks.checked); canvas.classList.toggle("show-noop",fNoop.checked); };
fLinks.onchange(); // a reload can restore checkbox state
document.addEventListener("keydown",ev=>{
  if(ev.target.tagName==="INPUT"){ if(ev.key==="Escape") ev.target.blur(); return; }
  if(ev.key==="Escape") clearAll();
  else if(ev.key==="+"||ev.key==="=") zoom(1.5);
  else if(ev.key==="-") zoom(1/1.5);
  else if(ev.key==="0") zoom(0);
  else if(ev.key==="/"){ ev.preventDefault(); search.focus(); }
  else if((ev.key==="ArrowLeft"||ev.key==="ArrowRight")&&focus){
    ev.preventDefault();
    setFocus({row:focus.row, step:Math.min(maxSteps,Math.max(0,focus.step+(ev.key==="ArrowRight"?1:-1)))});
  }
});

// ---------- header summary, verdict key, validation ----------
{
  const count=k=>G.ops.filter(o=>o.kind===k).length, nT=G.trajectories.length;
  const out=[`<span><b>${nT}</b> run${nT===1?"":"s"}${idle.length?` <span title="${esc(idle.map(t=>`${t.id} ${runName(t)}`).join("\n"))}">(${idle.length} without memory operations)</span>`:""}</span>`,
    `<span><b>${G.memories.length}</b> memor${G.memories.length===1?"y":"ies"}</span>`, ...["extract","recall"].map((k,i)=>`<span><b>${count(k)}</b> ${["write","recall"][i]}${count(k)===1?"":"s"}</span>`)];
  const nEp=rows.filter(r=>r.ep).length, nSeq=outline.filter(e=>e.kind==="seq").length;
  if(nEp) out.splice(1,0,`<span><b>${nEp}</b> episode${nEp===1?"":"s"} in ${nSeq} sequence${nSeq===1?"":"s"}</span>`);
  if(G.tool&&G.tool.verdicts){
    const kv=document.getElementById("key-verdicts");
    kv.hidden=false;
    kv.innerHTML=TALLY.slice(0,6).map(v=>`<span title="${VD_TIP[v]}"><i class="dot" style="background:${TALLY_COLOR[v]}"></i>${v}</span>`).join("");
    const seen=new Set(), vc=Object.create(null);
    for(const o of G.ops){
      if(o.kind!=="recall"||!o.verdict_span||seen.has(o.verdict_span)) continue;
      seen.add(o.verdict_span);
      // mechanical skips (empty delivery, unobserved consumer) stay apart
      // from judged verdicts: they sit outside every rate
      const auto=(o.verdict_detail||{}).auto;
      const v=auto==="empty_delivery"?"empty":auto?"unassessable":(o.span_verdict||"unjudged");
      vc[v]=(vc[v]||0)+1;
    }
    // a judged verdict's count jumps through its spans (VD_COLOR names exactly
    // the judged verdicts; unjudged and mechanical skips are not targets)
    out.push(`<span class="vd">${TALLY.filter(v=>vc[v]).map(v=>`<span title="${VD_TIP[v]}${VD_COLOR[v]?" — click to jump to each":""}"${VD_COLOR[v]?` class="link" data-vd="${v}"`:""}><i class="dot" style="background:${TALLY_COLOR[v]}"></i> ${vc[v]} ${v}</span>`).join("")}</span>`);
  }
  const nWarn=G.validation.filter(v=>v.level==="warn").length;
  if(nWarn) out.push(`<span class="link warn" id="vjump">⚠ ${nWarn} warning${nWarn===1?"":"s"}</span>`);
  const summary=document.getElementById("summary");
  summary.innerHTML=out.join("");
  // each click selects the next drawn recall with that verdict, zoomed in far
  // enough to tell neighboring steps apart
  const nextAt=Object.create(null);
  summary.addEventListener("click",ev=>{
    const chip=ev.target.closest("[data-vd]"); if(!chip) return;
    const v=chip.dataset.vd;
    const leads=[...new Set(G.ops.filter(o=>o.kind==="recall"&&o.span_verdict===v&&!(o.verdict_detail||{}).auto).map(o=>spanLeader[o.id]))];
    if(!leads.length) return;
    const lead=leads[nextAt[v]=((nextAt[v]??-1)+1)%leads.length];
    if(PX<16){ fitted=false; PX=16; mount(); }
    showOp(lead); locate(lead);
  });
  const foot=document.querySelector("footer details");
  if(nWarn) document.getElementById("vjump").onclick=()=>{ foot.open=true; foot.scrollIntoView(); };
  document.getElementById("vsum").textContent=`Validation notes (${G.validation.length}${nWarn?`, ${nWarn} warnings`:""})`;
  document.getElementById("vlist").innerHTML=[...G.validation].sort((a,b)=>(b.level==="warn")-(a.level==="warn"))
    .map(v=>`<li${v.level==="warn"?' class="warn"':""}>${esc(v.msg)}</li>`).join("");
}

zoom(0);
setFocus(null);
</script>
</body>
</html>
"""


def render_html(graph: dict, title: str = "memgraph — memory operations") -> str:
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
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
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
