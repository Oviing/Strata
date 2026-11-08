/** DiffusionGemma (Mac, one request at a time): catalog group, Mode setting, the gateway's pass-through, and the canvas preview that is never saved. */
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import {readFileSync,writeFileSync,readdirSync} from 'node:fs';
import {join} from 'node:path';
import {build} from 'esbuild';
import {nativeFixture,finishJob} from './helpers/native-fixture.mjs';
import {nativeGateway} from '../server/native-gateway.mjs';
import {proxyCompletion} from '../server/inference.mjs';
import {NATIVE_FIELDS,validateNativeSettings} from '../server/native-settings.mjs';

const compiled=await build({stdin:{contents:"export * from './src/lib/native-catalog';export * from './src/lib/native-names';export * from './src/lib/canvas-draft';",resolveDir:process.cwd(),loader:'ts'},bundle:true,write:false,format:'esm',platform:'node',loader:{'.json':'json'}});
const {nativeCatalogModels,isDiffusionGemma,nativeShortName,setDraft,clearDraft,draftOf,DRAFT_LIMIT}=await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
const listen=server=>new Promise(r=>server.listen(0,'127.0.0.1',()=>r(server.address().port)));
const close=server=>new Promise(r=>{server.closeAllConnections();server.close(r);});
const size=(id,model)=>({id,model,about:'',download_gb:11,ram_needed_gb:12,experts_gb:9,on_this_pc:'fits',installed:false});
const family=(name,title,sizes)=>({family:name,title,about:'',experimental:false,images:false,sizes});

test('catalog: DiffusionGemma is its own group (it was dropped before) and the Qwen groups are unchanged',()=>{
 const raw=[family('qwen','Qwen3.8-Flash-Next',[size('q2_0','Q2_0')]),family('diffusiongemma','DiffusionGemma-26B-A4B',[size('dgemma-mlx-q3','mlx-q3'),size('dgemma-mlx-q4','mlx-q4')])];
 const groups=nativeCatalogModels(raw);
 assert.deepEqual(groups.map(g=>g.title),['Qwen3.8-Flash-Next','Qwen3.6-35B-A3B','DiffusionGemma-26B-A4B']);
 const g=groups[2];assert.equal(g.family,'diffusiongemma');assert.deepEqual(g.sizes.map(s=>s.id),['dgemma-mlx-q3','dgemma-mlx-q4']);
 assert.ok(g.sizes.every(s=>s.sourceFamily==='diffusiongemma'&&s.images===false));assert.equal(g.images,false);
 assert.equal(groups[0].sizes.length,1);
 assert.equal(isDiffusionGemma({family:'diffusiongemma'}),true);assert.equal(isDiffusionGemma({family:'qwen'}),false);
 assert.equal(nativeShortName('dgemma-mlx-q3'),'DiffusionGemma MLX-Q3');
});

test('settings: Mode is offered for DiffusionGemma only, takes diffusion or ar, and the Qwen engine fields are hidden for it',()=>{
 const mode=NATIVE_FIELDS.find(f=>f.id==='mode');
 assert.deepEqual(mode.choices,['diffusion','ar']);assert.equal(mode.forModel,'dgemma');
 assert.match(mode.hint,/not measured/);assert.doesNotMatch(mode.hint,/\d+\s*(tok|tokens?\/s|x faster|%)/i);
 assert.deepEqual(validateNativeSettings({mode:'ar'}),{mode:'ar'});assert.deepEqual(validateNativeSettings({mode:null}),{mode:null});
 assert.throws(()=>validateNativeSettings({mode:'fast'}),/supported option/);
 for(const id of ['parallel','mtp','drafting','batchMtp','layer_split','gpu_order','vram_reserve_mib','vision_tokens'])assert.equal(NATIVE_FIELDS.find(f=>f.id===id).hideFor,'dgemma',id);
});

const addDiffusion=f=>{
 f.catalog.families.push({...family('diffusiongemma','DiffusionGemma-26B-A4B',[{...size('dgemma-mlx-q3','mlx-q3'),installed:true}])});
 f.installed.models.push({model:'dgemma-mlx-q3',config:'strata-dgemma-mlx-q3.json',model_name:'DiffusionGemma mlx-q3',context:32768,ready:true});
 writeFileSync(join(f.repo,'strata-dgemma-mlx-q3.json'),JSON.stringify({arch:'diffusion_gemma',model:'mlx-q3',args:['--max-context','32768'],unknown_native_key:{preserve:true}}));
};

test('settings: saving Mode writes "mode" into the run config, keeps the rest, and Default removes it; a load carries it into the instance',async()=>{
 const f=nativeFixture();try{
  addDiffusion(f);await f.control.refresh(true);
  const path=join(f.repo,'strata-dgemma-mlx-q3.json'),original=f.control.settings('dgemma-mlx-q3');
  assert.equal(original.values.mode,null);
  f.control.save('dgemma-mlx-q3',{values:{...original.values,mode:'ar'},mcp:original.mcp});
  let saved=JSON.parse(readFileSync(path,'utf8'));assert.equal(saved.mode,'ar');assert.equal(saved.arch,'diffusion_gemma');assert.deepEqual(saved.unknown_native_key,{preserve:true});
  assert.equal(f.control.settings('dgemma-mlx-q3').values.mode,'ar');
  const job=f.control.operation('load',{model:'dgemma-mlx-q3'});const done=await finishJob(f.control,job.id);assert.equal(done.state,'done',done.error);
  const instance=f.control.resolveInstance(done.result.instanceId);assert.equal(instance.arch,'diffusion_gemma');assert.equal(instance.mode,'ar');
  const written=readdirSync(join(f.directory,'native-instances')).map(n=>JSON.parse(readFileSync(join(f.directory,'native-instances',n),'utf8')));
  assert.equal(written.at(-1).mode,'ar');
  assert.ok(!written.at(-1).parallel&&!written.at(-1).vision,'no parallel slots or picture reading for this family');
  await f.control.unload(instance.id);
  f.control.save('dgemma-mlx-q3',{values:{...f.control.settings('dgemma-mlx-q3').values,mode:null},mcp:{}});
  saved=JSON.parse(readFileSync(path,'utf8'));assert.equal('mode' in saved,false);
 }finally{await f.cleanup();}
});

test('install: the setup arguments allow --mode',()=>{
 const source=readFileSync(new URL('../server/native-control.mjs',import.meta.url),'utf8');
 assert.ok(source.includes("const allowed=['family','mode','model',"));
});

test('gateway: the "strata" request field and the strata_draft delta pass through unchanged',{timeout:10000},async()=>{
 let seen;const chunk='data: {"choices":[{"delta":{"strata_draft":"The quick brown ___"}}]}\n\n';
 const engine=http.createServer(async(req,res)=>{let text='';for await(const c of req)text+=c;seen=JSON.parse(text);res.writeHead(200,{'content-type':'text/event-stream'});res.write(chunk);res.write('data: {"choices":[{"delta":{"content":"The quick"}}]}\n\n');res.end('data: [DONE]\n\n');});
 const enginePort=await listen(engine);const records=[];
 const serving={open:v=>{const r={...v,id:String(records.length)};records.push(r);return r;},mark:(id,v)=>Object.assign(records[Number(id)],v),finish:(id,v)=>Object.assign(records[Number(id)],v)};
 const control={resolveInstance:()=>({id:'d',apiModelId:'dgemma-mlx-q3',port:enginePort,key:'k',gpus:[]})};
 const gateway=http.createServer(async(req,res)=>{let text='';for await(const c of req)text+=c;await nativeGateway({req,res,path:req.url,body:JSON.parse(text),control,serving});});const port=await listen(gateway);
 try{
  const response=await fetch(`http://127.0.0.1:${port}/v1/chat/completions`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({model:'dgemma-mlx-q3',stream:true,messages:[],strata:{drafts:true}})});
  const body=await response.text();assert.deepEqual(seen.strata,{drafts:true});assert.ok(body.includes(chunk),'the draft chunk arrives byte for byte');
 }finally{await close(gateway);await close(engine);}
});

const streamFrom=async(chunks,extra)=>{
 const engine=http.createServer(async(req,res)=>{for await(const _ of req);res.writeHead(200,{'content-type':'text/event-stream'});for(const c of chunks)res.write(`data: ${JSON.stringify(c)}\n\n`);res.end('data: [DONE]\n\n');});
 const port=await listen(engine),frames=[];
 try{await proxyCompletion({baseUrl:`http://127.0.0.1:${port}/v1`,message:'hi',onFrame:f=>frames.push(f),extra});}finally{await close(engine);}
 return frames;
};
const delta=d=>({choices:[{delta:d}]});

test('host stream: strata_draft becomes replacing "draft" frames, never text, and an empty one ends them before done',async()=>{
 const frames=await streamFrom([delta({strata_draft:'The ___ ___'}),delta({strata_draft:'The quick ___'}),delta({content:'The quick'}),delta({strata_draft:'brown ___'}),{choices:[],usage:{prompt_tokens:3,completion_tokens:2}}],{strata:{drafts:true}});
 const drafts=frames.filter(f=>f.type==='draft');assert.deepEqual(drafts.map(f=>f.text),['The ___ ___','The quick ___','brown ___','']);
 assert.deepEqual(frames.filter(f=>f.type==='delta').map(f=>f.text),['The quick']);
 assert.ok(frames.findIndex(f=>f.type==='draft'&&f.text==='')<frames.findIndex(f=>f.type==='done'));
});
test('host stream: without the request for drafts a draft field is ignored (other models are untouched)',async()=>{
 const frames=await streamFrom([delta({strata_draft:'x'}),delta({content:'ok'})],{});
 assert.equal(frames.some(f=>f.type==='draft'),false);
});
test('host chat asks for drafts only for a DiffusionGemma instance',()=>{
 const index=readFileSync(new URL('../server/index.mjs',import.meta.url),'utf8');
 assert.match(index,/\.\.\.\(instance\.arch==='diffusion_gemma'\?\{strata:\{drafts:true\}\}:\{\}\)/);
});

test('page: the canvas is kept outside the chat, replaced each time, bounded, and cleared; the chat history never gets it',()=>{
 clearDraft('s1');assert.equal(draftOf('s1'),'');
 setDraft('s1','one ___');setDraft('s1','one two ___');assert.equal(draftOf('s1'),'one two ___');assert.equal(draftOf('s2'),'');
 setDraft('s1','x'.repeat(DRAFT_LIMIT+50));assert.equal(draftOf('s1').length,DRAFT_LIMIT);
 setDraft('s1',undefined);assert.equal(draftOf('s1'),'');
 const store=readFileSync(new URL('../src/lib/store.ts',import.meta.url),'utf8');
 assert.match(store,/if \(type === 'draft'\) \{ setDraft\(sessionId, frame\.text\); return; \}/);
 const handler=store.indexOf("if (type === 'draft')");assert.ok(handler>0&&handler<store.indexOf("if (type === 'delta' || type === 'reasoning') { queueText"));
 assert.match(store,/\} finally \{\s*flushWaiting\(\);\s*clearDraft\(sessionId\);/);
 assert.doesNotMatch(readFileSync(new URL('../src/lib/types.ts',import.meta.url),'utf8'),/strata_draft|canvasDraft/);
});
test('page: the preview is faded, not selectable, hidden from screen readers and has no motion',()=>{
 const component=readFileSync(new URL('../src/components/DraftCanvas.tsx',import.meta.url),'utf8'),css=readFileSync(new URL('../src/styles/app.css',import.meta.url),'utf8');
 assert.match(component,/aria-hidden="true"/);assert.match(component,/useDraft\(sessionId\)/);
 const rule=css.slice(css.indexOf('.st-canvas-draft {'));const block=rule.slice(0,rule.indexOf('}'));
 assert.match(block,/user-select: none/);assert.match(block,/color: var\(--st-ink-muted\)/);assert.doesNotMatch(block,/animation|transition|transform/);
 assert.match(readFileSync(new URL('../src/screens/Chat.tsx',import.meta.url),'utf8'),/m\.streaming && isLast && m\.role === 'assistant' \? <DraftCanvas/);
});
