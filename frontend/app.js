(()=>{"use strict";
const API=location.hostname==="localhost"||location.hostname==="127.0.0.1"?"http://localhost:8000":"https://the-harsh-vardhan-tidal-api.hf.space";
const $=id=>document.getElementById(id);
const ua=$("uploadArea"),uc=$("uploadContent"),up=$("uploadPreview"),fi=$("fileInput"),
  bb=$("browseBtn"),cb=$("clearBtn"),pi=$("previewImage"),ra=$("resultsArea"),
  si=$("statusIndicator"),st=$("statusText"),rv=$("resultVerdict"),vi=$("verdictIcon"),
  vt=$("verdictText"),vd=$("verdictDetail"),mc=$("metricConfidence"),mr=$("metricRatio"),
  mt=$("metricTime"),mi=$("maskImage");

async function checkHealth(){
  try{const r=await fetch(`${API}/health`,{signal:AbortSignal.timeout(5000)});
    if(r.ok){si.className="status-indicator online";st.textContent="API connected";return true}}
  catch{}si.className="status-indicator offline";st.textContent="API offline";return false}
checkHealth();setInterval(checkHealth,15000);

ua.addEventListener("dragover",e=>{e.preventDefault();ua.classList.add("drag-active")});
ua.addEventListener("dragleave",()=>ua.classList.remove("drag-active"));
ua.addEventListener("drop",e=>{e.preventDefault();ua.classList.remove("drag-active");if(e.dataTransfer.files.length)handleFile(e.dataTransfer.files[0])});
bb.addEventListener("click",e=>{e.stopPropagation();fi.click()});
fi.addEventListener("change",()=>{if(fi.files.length)handleFile(fi.files[0])});
cb.addEventListener("click",e=>{e.stopPropagation();clear()});

function handleFile(f){
  if(!["image/jpeg","image/png","image/webp"].includes(f.type))return alert("Upload JPEG/PNG/WebP");
  if(f.size>20*1024*1024)return alert("Max 20MB");
  const r=new FileReader();r.onload=e=>{pi.src=e.target.result;uc.hidden=true;up.hidden=false};r.readAsDataURL(f);
  submit(f)}
function clear(){fi.value="";uc.hidden=false;up.hidden=true;ra.hidden=true}

async function submit(f){
  ra.hidden=false;vt.textContent="Analyzing...";vi.textContent="⏳";mc.textContent=mr.textContent=mt.textContent="—";mi.src="";
  const fd=new FormData();fd.append("file",f);
  try{const r=await fetch(`${API}/infer`,{method:"POST",body:fd});
    if(!r.ok){const e=await r.json().catch(()=>({}));throw new Error(e.detail||`HTTP ${r.status}`)}
    show(await r.json())}catch(e){vi.textContent="⚠";vt.textContent="Error";vd.textContent=e.message}}

function show(d){
  const t=d.is_tampered;rv.className=`result-verdict ${t?"verdict-tampered":"verdict-authentic"}`;
  vi.textContent=t?"✗":"✓";vt.textContent=t?"Tampered":"Authentic";
  vd.textContent=t?`${(d.tampered_ratio*100).toFixed(1)}% tampered`:"No tampering found";
  mc.textContent=`${(d.confidence*100).toFixed(1)}%`;mr.textContent=`${(d.tampered_ratio*100).toFixed(2)}%`;
  mt.textContent=`${d.inference_time_ms.toFixed(0)}ms`;
  if(d.mask_base64)mi.src=`data:image/png;base64,${d.mask_base64}`}
})();
