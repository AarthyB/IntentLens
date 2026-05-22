const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

// Markdown-lite: **bold**
function mdBold(s){return s.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')}

// Conversation history sent to backend for context
let convHistory = [];
let sessionStats = {total:0, totalMs:0, totalConf:0, labels:{Platonic:0,Romantic:0,Ambiguous:0}};

function resize(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,160)+'px'}
function scrollBot(){const c=document.getElementById('chat');c.scrollTo({top:c.scrollHeight,behavior:'smooth'})}
function onKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}}
function useEx(el){
  const t=el.textContent.replace(/\s*[📱🌀💭]\s*/gu,'').trim();
  document.getElementById('inp').value=t;
  resize(document.getElementById('inp'));
  send();
}

function hideWelcome(){
  const w=document.getElementById('welcome');
  if(w){w.style.transition='opacity .2s';w.style.opacity='0';setTimeout(()=>w.remove(),200)}
}

function clearChat(){
  document.getElementById('chat').innerHTML='';
  convHistory=[];
  sessionStats={total:0,totalMs:0,totalConf:0,labels:{Platonic:0,Romantic:0,Ambiguous:0}};
  updateStats();
  document.getElementById('ctxCount').textContent='0 turns';
  // re-add welcome
  const w=document.createElement('div');w.className='welcome';w.id='welcome';
  w.innerHTML=`<div class="welcome-icon">💬</div>
    <h2>What does this message really mean?</h2>
    <p>Paste any message and I'll analyze the relational intent — then ask follow-up questions and I'll factor in the whole conversation.</p>
    <div class="ex-grid">
      <span class="ex-pill" onclick="useEx(this)">Why didn't you message me today? 📱</span>
      <span class="ex-pill" onclick="useEx(this)">I can't stop thinking about you</span>
      <span class="ex-pill" onclick="useEx(this)">Want to grab lunch with the crew?</span>
      <span class="ex-pill" onclick="useEx(this)">I feel something I can't explain 🌀</span>
      <span class="ex-pill" onclick="useEx(this)">But even a friend can say that, right?</span>
      <span class="ex-pill" onclick="useEx(this)">Would you want to go on a date with me?</span>
    </div>`;
  document.getElementById('chat').appendChild(w);
  fetch('/api/reset',{method:'POST'});
}

//  render one full exchange 
function appendExchange(userText, data, isError){
  const chat = document.getElementById('chat');
  const group = document.createElement('div');
  group.className = 'msg-group';

  // user bubble
  group.innerHTML = `<div class="user-turn"><div class="user-bubble">${esc(userText)}</div></div>`;

  // ai turn
  const aiTurn = document.createElement('div');
  aiTurn.className = 'ai-turn';
  aiTurn.innerHTML = `<div class="ai-avatar">🔍</div><div class="ai-content" id="ai-content-pending"></div>`;
  group.appendChild(aiTurn);
  chat.appendChild(group);
  scrollBot();

  const content = aiTurn.querySelector('#ai-content-pending');
  content.removeAttribute('id');

  if(isError){
    content.innerHTML = `<div class="err-txt">⚠️ ${esc(data)}</div>`;
    return;
  }

  const cls = data.label.toLowerCase();
  const toggleId = 'tog-'+Date.now();
  const barsId   = 'bars-'+Date.now();

  // Build analyzed-message chip if we extracted a specific quote
  const speakerHTML = data.speaker
    ? `<div class="speaker-tag">👤 ${esc(data.speaker)}</div>` : '';
  const quoteHTML = data.analyzed_text
    ? `<div class="analyzed-chip">
        <span class="analyzed-chip-label">analyzing:</span>
        <span class="analyzed-chip-text">"${esc(data.analyzed_text)}"</span>
       </div>` : '';

  content.innerHTML = `
    ${speakerHTML}
    ${quoteHTML}
    <div class="reply-text">${mdBold(esc(data.reply))}</div>
    <div class="verdict-row">
      <span class="verdict-pill ${cls}">${data.emoji} ${data.label}</span>
      <span class="conf-txt">confidence <span>${data.confidence}%</span></span>
      <span class="certainty-txt">· ${data.certainty}</span>
    </div>
    <div class="prob-toggle" id="${toggleId}" onclick="toggleBars('${toggleId}','${barsId}')">
      <span class="prob-toggle-icon">▶</span>
      <span class="prob-toggle-lbl">Show probability breakdown</span>
    </div>
    <div class="prob-bars" id="${barsId}">
      <div class="prob-bars-inner">
        ${data.probabilities.map(p=>`
          <div class="pb-row">
            <span class="pb-lbl">${p.emoji} ${p.label}</span>
            <div class="pb-track"><div class="pb-fill ${p.label.toLowerCase()}" data-w="${p.prob}"></div></div>
            <span class="pb-pct ${p.label.toLowerCase()}">${p.prob}%</span>
          </div>`).join('')}
      </div>
    </div>
    <div class="ai-meta">
      <span>${data.model_name || 'Transformer'}</span>
      ${data.context_used>0?`<span>context: ${data.context_used} prior turns</span>`:'<span>no prior context</span>'}
      <span>${data.inference_ms}ms</span>
    </div>`;

  scrollBot();

  // animate bars when opened
  content._probData = data.probabilities;
}

function toggleBars(toggleId, barsId){
  const tog  = document.getElementById(toggleId);
  const bars = document.getElementById(barsId);
  const opening = !bars.classList.contains('open');
  tog.classList.toggle('open', opening);
  bars.classList.toggle('open', opening);
  tog.querySelector('.prob-toggle-lbl').textContent = opening ? 'Hide breakdown' : 'Show probability breakdown';
  if(opening){
    requestAnimationFrame(()=>requestAnimationFrame(()=>{
      bars.querySelectorAll('.pb-fill').forEach(b=>{b.style.width=b.dataset.w+'%'});
    }));
  }
}

//  thinking placeholder 
function showThinking(userText){
  const chat = document.getElementById('chat');
  const group = document.createElement('div');
  group.className = 'msg-group'; group.id = 'pending-group';
  group.innerHTML = `
    <div class="user-turn"><div class="user-bubble">${esc(userText)}</div></div>
    <div class="ai-turn">
      <div class="ai-avatar">🔍</div>
      <div class="ai-content"><div class="thinking"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div>
    </div>`;
  chat.appendChild(group);
  scrollBot();
}
function removePending(){const p=document.getElementById('pending-group');if(p)p.remove()}

//  stats 
function updateStats(data){
  document.getElementById('sTotal').textContent = sessionStats.total;
  document.getElementById('sAvgMs').textContent = sessionStats.total
    ? Math.round(sessionStats.totalMs/sessionStats.total)+'ms' : '—';
  document.getElementById('sAvgConf').textContent = sessionStats.total
    ? Math.round(sessionStats.totalConf/sessionStats.total)+'%' : '—';
  const emojis={Platonic:'🤝',Romantic:'💕',Ambiguous:'🤔'};
  const top = Object.entries(sessionStats.labels).sort((a,b)=>b[1]-a[1])[0];
  document.getElementById('sTopLabel').textContent = (top&&top[1]>0) ? emojis[top[0]] : '—';

  // breakdown bars
  const tot = sessionStats.total || 1;
  ['Platonic','Romantic','Ambiguous'].forEach(l=>{
    const n = sessionStats.labels[l];
    const pct = Math.round(n/tot*100);
    const key = l.toLowerCase().slice(0,4);
    const idMap = {Plat:'bdPlat',Romantic:'bdRom',Ambiguous:'bdAmb'};
    document.getElementById('bd'+l.slice(0,4)+(l==='Romantic'?'':''))
    const fillEl = document.getElementById('bd'+({'Platonic':'Plat','Romantic':'Rom','Ambiguous':'Amb'}[l]));
    const numEl  = document.getElementById('bd'+({'Platonic':'Plat','Romantic':'Rom','Ambiguous':'Amb'}[l])+'N');
    if(fillEl) fillEl.style.width = pct+'%';
    if(numEl)  numEl.textContent  = n;
  });
  document.getElementById('ctxCount').textContent = convHistory.length+' turn'+(convHistory.length!==1?'s':'');
}

//  main send 
async function send(){
  const inp  = document.getElementById('inp');
  const text = inp.value.trim();
  if(!text) return;

  hideWelcome();
  inp.value=''; resize(inp);
  document.getElementById('sendBtn').disabled=true;

  showThinking(text);

  try{
    const res  = await fetch('/api/analyze',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text, history: convHistory}),
    });
    const data = await res.json();
    removePending();

    if(!res.ok){
      appendExchange(text, data.error||'Server error', true);
    } else {
      appendExchange(text, data, false);
      // update history for next turn
      convHistory.push({text, label_id: data.label_id, label: data.label, analyzed_text: data.analyzed_text});
      if(convHistory.length>20) convHistory.shift();

      // update stats
      sessionStats.total++;
      sessionStats.totalMs   += data.inference_ms||0;
      sessionStats.totalConf += data.confidence||0;
      sessionStats.labels[data.label] = (sessionStats.labels[data.label]||0)+1;
      updateStats(data);
    }
  } catch(e){
    removePending();
    appendExchange(text,'Could not reach the server. Is Flask running on port 5000?',true);
  }

  document.getElementById('sendBtn').disabled=false;
  inp.focus();
}

// load model info
async function loadInfo(){
  try{
    const d = await (await fetch('/api/model/info')).json();
    const accEl = document.getElementById('mAcc');
    const f1El  = document.getElementById('mF1');
    if(d.accuracy != null && d.accuracy > 0){
      accEl.textContent = (d.accuracy * 100).toFixed(1) + '%';
      accEl.style.color = 'var(--txt)';
    } else {
      accEl.textContent = 'evaluating…';
      accEl.style.color = 'var(--muted)';
      // retry once after 4 seconds (model may still be evaluating on load)
      setTimeout(async () => {
        try {
          const d2 = await (await fetch('/api/model/info')).json();
          if(d2.accuracy != null && d2.accuracy > 0){
            accEl.textContent = (d2.accuracy * 100).toFixed(1) + '%';
            accEl.style.color = 'var(--txt)';
          } else {
            accEl.textContent = 'N/A';
          }
          if(d2.f1 != null && d2.f1 > 0){
            f1El.textContent = d2.f1.toFixed(3);
          }
        } catch(e2){}
      }, 4000);
    }
    if(d.f1 != null && d.f1 > 0){
      f1El.textContent = d.f1.toFixed(3);
      f1El.style.color = 'var(--txt)';
    }
    // Update model name tag
    const tagEl = document.getElementById('modelTag');
    if(tagEl && d.model){
      const name = d.model.includes('DistilBERT') ? 'DistilBERT · 6-layer · 12-head' : 'Transformer · 3-layer · 4-head attn';
      tagEl.textContent = name;
    }
  } catch(e){
    document.getElementById('mAcc').textContent = 'offline';
    document.getElementById('mF1').textContent  = '—';
  }
}
window.addEventListener('DOMContentLoaded', loadInfo);