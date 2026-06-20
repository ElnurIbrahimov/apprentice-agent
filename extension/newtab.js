(function(){"use strict";var Q,X;const nt="https://aura-elnur.duckdns.org",E="aura_conversations",$=nt;let b=$;const it=5*60*1e3,x=["The best way to predict the future is to invent it. — Alan Kay","Simplicity is the ultimate sophistication. — Leonardo da Vinci","First, solve the problem. Then, write the code. — John Johnson","Code is like humor. When you have to explain it, it is bad. — Cory House","The only way to do great work is to love what you do. — Steve Jobs","Stay hungry, stay foolish. — Steve Jobs","Think different.","Move fast and break things. Then fix them.","The computer was born to solve problems that did not exist before. — Bill Gates","Any sufficiently advanced technology is indistinguishable from magic. — Arthur C. Clarke","Intelligence is the ability to adapt to change. — Stephen Hawking","Not all those who wander are lost. — J.R.R. Tolkien","Imagination is more important than knowledge. — Albert Einstein","Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away. — Saint-Exupéry","The question of whether a computer can think is no more interesting than the question of whether a submarine can swim. — Dijkstra"],H={neutral:{core:"#7c3aed",glow:"#a78bfa",highlight:"#e0d6ff"},calm:{core:"#7c3aed",glow:"#a78bfa",highlight:"#e0d6ff"},content:{core:"#7c3aed",glow:"#a78bfa",highlight:"#e0d6ff"},thoughtful:{core:"#7c3aed",glow:"#a78bfa",highlight:"#e0d6ff"},happy:{core:"#f59e0b",glow:"#fbbf24",highlight:"#fef3c7"},excited:{core:"#f59e0b",glow:"#fbbf24",highlight:"#fef3c7"},playful:{core:"#f59e0b",glow:"#fbbf24",highlight:"#fef3c7"},gratification:{core:"#f59e0b",glow:"#fbbf24",highlight:"#fef3c7"},joy:{core:"#f59e0b",glow:"#fbbf24",highlight:"#fef3c7"},curious:{core:"#3b82f6",glow:"#60a5fa",highlight:"#dbeafe"},surprised:{core:"#3b82f6",glow:"#60a5fa",highlight:"#dbeafe"},engaged:{core:"#3b82f6",glow:"#60a5fa",highlight:"#dbeafe"},admiration:{core:"#3b82f6",glow:"#60a5fa",highlight:"#dbeafe"},confident:{core:"#10b981",glow:"#34d399",highlight:"#d1fae5"},satisfaction:{core:"#10b981",glow:"#34d399",highlight:"#d1fae5"},pride:{core:"#10b981",glow:"#34d399",highlight:"#d1fae5"},gratitude:{core:"#10b981",glow:"#34d399",highlight:"#d1fae5"},sad:{core:"#64748b",glow:"#94a3b8",highlight:"#cbd5e1"},distress:{core:"#64748b",glow:"#94a3b8",highlight:"#cbd5e1"},disappointment:{core:"#64748b",glow:"#94a3b8",highlight:"#cbd5e1"},sorry_for:{core:"#64748b",glow:"#94a3b8",highlight:"#cbd5e1"},remorse:{core:"#64748b",glow:"#94a3b8",highlight:"#cbd5e1"},anxious:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},frustrated:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},angry:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},fearful:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},fear:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},hate:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},empathetic:{core:"#8b5cf6",glow:"#a78bfa",highlight:"#ede9fe"},concerned:{core:"#8b5cf6",glow:"#a78bfa",highlight:"#ede9fe"}};function S(t){return H[t]||H.neutral}function D(t){const e=document.documentElement;e.style.setProperty("--dot-core",t.core),e.style.setProperty("--dot-glow",t.glow),e.style.setProperty("--dot-highlight",t.highlight)}let T=null;async function ot(){var n,i,a;if(T&&Date.now()-T.ts<it){D(S(T.emotion));return}const t=await y(["apiKey"]),e=((i=(n=t==null?void 0:t.apiKey)==null?void 0:n.trim)==null?void 0:i.call(n))||"";if(e)try{const o={"X-API-Key":e},d=await fetch(`${b}/api/status`,{signal:AbortSignal.timeout(4e3),headers:o});if(!d.ok)return;const p=await d.json(),v=((a=p==null?void 0:p.mood)==null?void 0:a.emotion)||"neutral";T={emotion:v,ts:Date.now()},D(S(v))}catch{}}function R(t){const e=String(t.getHours()).padStart(2,"0"),n=String(t.getMinutes()).padStart(2,"0");return`${e}<span class="nt-clock-colon">:</span>${n}`}function _(t){return t.toLocaleDateString(void 0,{weekday:"long",month:"long",day:"numeric"})}function N(t){const e=t.getHours();return e>=5&&e<12?"Good morning":e>=12&&e<17?"Good afternoon":e>=17&&e<21?"Good evening":"Good night"}const L="aura_weather_cache",at=30*60*1e3;function st(t){const e=t.toLowerCase();return e.includes("thunder")?"⛈️":e.includes("snow")||e.includes("blizzard")?"❄️":e.includes("rain")?"🌧️":e.includes("drizzle")?"🌦️":e.includes("mist")||e.includes("fog")?"🌫️":e.includes("overcast")||e.includes("cloudy")?"☁️":e.includes("partly")?"⛅":e.includes("sunny")||e.includes("clear")?"☀️":"🌤️"}async function rt(){var n,i,a,o,d,p,v,C,M;const e=(await y([L]))[L];if(e&&Date.now()-e.cachedAt<at)return e;try{const s=new AbortController;setTimeout(()=>s.abort(),5e3);const r=await fetch("https://wttr.in/?format=j1",{signal:s.signal});if(!r.ok)return null;const u=await r.json(),h=(n=u.current_condition)==null?void 0:n[0],m=(i=u.nearest_area)==null?void 0:i[0];if(!h)return null;const g={temp:h.temp_C,condition:((o=(a=h.weatherDesc)==null?void 0:a[0])==null?void 0:o.value)||"",icon:st(((p=(d=h.weatherDesc)==null?void 0:d[0])==null?void 0:p.value)||""),city:((C=(v=m==null?void 0:m.areaName)==null?void 0:v[0])==null?void 0:C.value)||"",cachedAt:Date.now()};return(M=c==null?void 0:c.storage)!=null&&M.local&&c.storage.local.set({[L]:g}),g}catch{return null}}function ct(t){const e=Date.now()-t,n=Math.floor(e/6e4);if(n<1)return"just now";if(n<60)return n+"m ago";const i=Math.floor(n/60);if(i<24)return i+"h ago";const a=Math.floor(i/24);return a<7?a+"d ago":new Date(t).toLocaleDateString(void 0,{month:"short",day:"numeric"})}function lt(){const t=Math.floor(Date.now()/864e5)%x.length;return x[t]}function l(t){return String(t??"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;")}function B(t){return typeof t!="string"||!t||/[\x00-\x1f\s]/.test(t)?!1:/^https?:\/\/[^\s]+$/i.test(t)}const dt=/^(#[0-9a-fA-F]{3,8}|rgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*(?:,\s*(?:\d*\.)?\d+\s*)?\))$/;function ht(t){return typeof t=="string"&&dt.test(t)}const c=typeof chrome<"u"&&(chrome!=null&&chrome.storage)?chrome:null;function y(t){return new Promise(e=>{var n;(n=c==null?void 0:c.storage)!=null&&n.local?c.storage.local.get(t,i=>e(i||{})):e({})})}function A(t){var e;(e=c==null?void 0:c.runtime)!=null&&e.sendMessage&&c.runtime.sendMessage(t)}function ut(){var t;(t=c==null?void 0:c.topSites)!=null&&t.get&&c.topSites.get(e=>{if(!e||e.length===0)return;const n=document.getElementById("topsites-section"),i=document.getElementById("topsites-row");!n||!i||(n.style.display="",i.innerHTML=e.slice(0,8).map(a=>{let o;try{o=new URL(a.url).hostname.replace("www.","")}catch{o=a.url}const d=o.split(".")[0]||o;return`<a class="nt-topsite" href="${l(a.url)}" title="${l(a.title||o)}">
        <div class="nt-topsite-icon">
          <img src="https://www.google.com/s2/favicons?domain=${l(o)}&sz=32" alt="" width="20" height="20" loading="lazy" />
        </div>
        <span class="nt-topsite-label">${l(d)}</span>
      </a>`}).join(""))})}const f={chat:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>',search:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>',translate:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 8l6 6"></path><path d="M4 14l6-6 2-3"></path><path d="M2 5h12"></path><path d="M7 2v3"></path><path d="M22 22l-5-10-5 10"></path><path d="M14 18h6"></path></svg>',write:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"></path><path d="m15 5 4 4"></path></svg>',code:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>',research:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"></path><circle cx="11" cy="10" r="3"></circle><path d="m14 13-1.5-1.5"></path></svg>',ocr:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 2H2v5"></path><path d="M17 2h5v5"></path><path d="M7 22H2v-5"></path><path d="M17 22h5v-5"></path><line x1="5" y1="12" x2="19" y2="12"></line></svg>',grammar:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline><line x1="4" y1="21" x2="20" y2="21"></line></svg>',chatBubble:'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>',plus:'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>',x:'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>'};function U(){var M;const t=document.getElementById("root"),e=new Date;t.innerHTML=`
    <div class="nt-clock nt-fade nt-fade-d1">
      <div class="nt-greeting" id="clock-greeting">${l(N(e))}</div>
      <div class="nt-clock-time" id="clock-time">${R(e)}</div>
      <div class="nt-clock-date" id="clock-date">${l(_(e))}<span class="nt-weather" id="weather"></span></div>
    </div>

    <div class="nt-logo nt-fade nt-fade-d2">
      <div class="nt-logo-dot"></div>
      <span class="nt-logo-text">AURA</span>
    </div>

    <form class="nt-search-wrap nt-fade nt-fade-d3" id="search-form">
      <button type="button" class="nt-mode-toggle disabled" id="mode-toggle" title="Click or press Tab to switch mode">
        <span class="nt-mode-icon"><span class="nt-mode-icon-google">G</span></span>
        <span class="nt-mode-label">Google</span>
      </button>
      <input class="nt-search" id="search-input" type="text" placeholder="Search Google..." autofocus />
      <svg class="nt-search-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"></circle>
        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
      </svg>
      <div class="nt-mode-hint" id="mode-hint">Tab to switch to AURA</div>
    </form>

    <div class="nt-actions nt-fade nt-fade-d4">
      <button class="nt-action" data-panel="chat">
        <span class="nt-action-icon">${f.chat}</span>Chat
      </button>
      <button class="nt-action" data-panel="search">
        <span class="nt-action-icon">${f.search}</span>Search
      </button>
      <button class="nt-action" data-panel="translate">
        <span class="nt-action-icon">${f.translate}</span>Translate
      </button>
      <button class="nt-action" data-panel="write">
        <span class="nt-action-icon">${f.write}</span>Write
      </button>
      <button class="nt-action" data-panel="code">
        <span class="nt-action-icon">${f.code}</span>Code
      </button>
      <button class="nt-action" data-panel="research">
        <span class="nt-action-icon">${f.research}</span>Research
      </button>
      <button class="nt-action" data-panel="ocr">
        <span class="nt-action-icon">${f.ocr}</span>OCR
      </button>
      <button class="nt-action" data-panel="grammar">
        <span class="nt-action-icon">${f.grammar}</span>Grammar
      </button>
    </div>

    <div class="nt-cockpit nt-fade nt-fade-d5" id="cockpit" style="display:none">
      <div class="nt-card nt-card-span2" id="card-ticker">
        <div class="nt-card-title">💭 Aura is thinking</div>
        <div class="nt-card-body"><div class="nt-ticker" id="ticker-text">—</div><div class="nt-ticker-meta" id="ticker-meta"></div></div>
      </div>
      <div class="nt-card nt-card-span2" id="card-heatmap">
        <div class="nt-card-title">🔥 Context focus</div>
        <div class="nt-card-body"><div class="nt-heatmap" id="heatmap-chips"><span class="nt-card-empty">No active topics</span></div></div>
      </div>
      <div class="nt-card" id="card-starter">
        <div class="nt-card-title">✨ Starter</div>
        <div class="nt-card-body" id="starter-body"><span class="nt-card-empty">Quiet.</span></div>
      </div>
      <div class="nt-card" id="card-hands">
        <div class="nt-card-title">🤖 Hands</div>
        <div class="nt-card-body" id="hands-body"><span class="nt-card-empty">No hands running</span></div>
      </div>
      <div class="nt-card" id="card-activity">
        <div class="nt-card-title">📊 Activity</div>
        <div class="nt-card-body"><div class="nt-list" id="activity-list"><span class="nt-card-empty">No events</span></div></div>
      </div>
      <div class="nt-card" id="card-memories">
        <div class="nt-card-title">🧠 Memories</div>
        <div class="nt-card-body"><div class="nt-list" id="memories-list"><span class="nt-card-empty">No memories</span></div></div>
      </div>
      <div class="nt-card nt-card-span2" id="card-feed">
        <div class="nt-card-title">📎 Recent captures</div>
        <div class="nt-card-body"><div class="nt-feed" id="feed-row"><span class="nt-card-empty">No captures</span></div></div>
      </div>
    </div>

    <div class="nt-topsites nt-fade nt-fade-d5" id="topsites-section" style="display:none">
      <div class="nt-topsites-row" id="topsites-row"></div>
    </div>

    <div class="nt-recent nt-fade nt-fade-d6" id="recent-section">
      <div class="nt-recent-header">
        <div class="nt-recent-title">Recent conversations</div>
        <button class="nt-new-chat-btn" id="new-chat-btn">
          ${f.plus} New chat
        </button>
      </div>
      <div class="nt-recent-list" id="recent-list">
        <div class="nt-recent-empty">No conversations yet. Click Chat or switch to AURA mode to start.</div>
      </div>
    </div>

    <div class="nt-quote">
      <div class="nt-quote-text">${l(lt())}</div>
    </div>

    <div class="nt-brand">AURA</div>

    <div class="nt-status" id="status-indicator">
      <div class="nt-status-dot"></div>
      AURA offline
    </div>
  `,setInterval(()=>{const s=new Date,r=document.getElementById("clock-time");r&&(r.innerHTML=R(s))},1e3),setInterval(()=>{const s=new Date,r=document.getElementById("clock-greeting"),u=document.getElementById("clock-date");if(r&&(r.textContent=N(s)),u){const h=document.getElementById("weather"),m=h?h.outerHTML:'<span class="nt-weather" id="weather"></span>';u.innerHTML=l(_(s))+m}},6e4),rt().then(s=>{if(!s)return;const r=document.getElementById("weather");r&&(r.innerHTML=` &middot; ${s.icon} ${l(s.temp)}°C ${l(s.condition)}`)});const n=document.getElementById("search-form"),i=document.getElementById("search-input"),a=document.getElementById("mode-toggle"),o=document.getElementById("mode-hint");let d="google",p=!1;function v(s){d=s,s==="aura"?(a.classList.add("aura-mode"),a.innerHTML='<span class="nt-mode-icon"><span class="nt-mode-icon-aura"></span></span><span class="nt-mode-label">AURA</span>',i.placeholder="Ask AURA anything...",o.textContent="Tab to switch to Google"):(a.classList.remove("aura-mode"),a.innerHTML='<span class="nt-mode-icon"><span class="nt-mode-icon-google">G</span></span><span class="nt-mode-label">Google</span>',i.placeholder="Search Google...",o.textContent=p?"Tab to switch to AURA":"AURA offline"),i.focus()}a.addEventListener("click",()=>{p&&v(d==="google"?"aura":"google")}),i.addEventListener("focus",()=>n.classList.add("focused")),i.addEventListener("blur",()=>n.classList.remove("focused")),n.addEventListener("submit",s=>{s.preventDefault();const r=i.value.trim();r&&(d==="aura"?(A({type:"OPEN_SIDEBAR",panel:"chat",message:r}),i.value=""):window.location.href=`https://www.google.com/search?q=${encodeURIComponent(r)}`)}),document.addEventListener("keydown",s=>{s.key==="Tab"&&document.activeElement===i&&p&&(s.preventDefault(),v(d==="google"?"aura":"google")),s.key==="/"&&document.activeElement!==i&&(s.preventDefault(),i.focus()),s.key==="Escape"&&document.activeElement===i&&(i.value="",i.blur())}),document.querySelectorAll(".nt-action[data-panel]").forEach(s=>{s.addEventListener("click",()=>{let r=s.dataset.panel||"";(r==="write"||r==="code")&&(r="artifacts"),r&&A({type:"OPEN_SIDEBAR",panel:r})})}),(M=document.getElementById("new-chat-btn"))==null||M.addEventListener("click",()=>{A({type:"OPEN_SIDEBAR",panel:"chat",newConversation:!0})}),ut(),y([E]).then(s=>{const r=(s[E]||[]).slice(0,5);if(r.length===0)return;const u=document.getElementById("recent-list");u.innerHTML=r.map(h=>`<button class="nt-recent-item" data-conv-id="${l(h.id)}">
            <span class="nt-recent-item-icon">${f.chatBubble}</span>
            <span class="nt-recent-item-text">${l(h.title)}</span>
            <span class="nt-recent-item-time">${l(ct(h.timestamp))}</span>
            <button class="nt-recent-item-delete" title="Remove">${f.x}</button>
          </button>`).join(""),u.querySelectorAll(".nt-recent-item").forEach(h=>{h.addEventListener("click",m=>{if(m.target.closest(".nt-recent-item-delete"))return;const g=h.dataset.convId;g&&A({type:"OPEN_SIDEBAR",panel:"chat",conversationId:g})})}),u.querySelectorAll(".nt-recent-item-delete").forEach(h=>{h.addEventListener("click",m=>{var Z;m.stopPropagation();const g=h.closest(".nt-recent-item"),Y=(Z=g==null?void 0:g.dataset)==null?void 0:Z.convId;Y&&(g.remove(),y([E]).then(yt=>{var et;const tt=(yt[E]||[]).filter(wt=>wt.id!==Y);(et=c==null?void 0:c.storage)!=null&&et.local&&c.storage.local.set({[E]:tt}),tt.length===0&&(u.innerHTML='<div class="nt-recent-empty">No conversations yet. Click Chat or switch to AURA mode to start.</div>')}))})})});const C=new AbortController;fetch(`${b}/api/health`,{signal:C.signal,method:"GET"}).then(s=>{if(s.ok){const r=document.getElementById("status-indicator");r&&(r.innerHTML='<div class="nt-status-dot online"></div>AURA online'),p=!0,a.classList.remove("disabled"),o.textContent="Tab to switch to AURA",ot(),y(["cockpitDisabled"]).then(u=>{u!=null&&u.cockpitDisabled||mt()})}}).catch(()=>{}),setTimeout(()=>C.abort(),3e3)}async function j(){var n,i;const t=await y(["apiKey"]),e=((i=(n=t==null?void 0:t.apiKey)==null?void 0:n.trim)==null?void 0:i.call(n))||"";return e?{"X-API-Key":e}:{}}async function w(t,e=4e3){try{const n=await j(),i=new AbortController,a=setTimeout(()=>i.abort(),e),o=await fetch(`${b}${t}`,{headers:n,signal:i.signal});return clearTimeout(a),o.ok?await o.json():null}catch{return null}}function pt(t){return new Date(t*1e3).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"})}function ft(t){if(typeof t=="number"&&Number.isFinite(t))return t>1e12?Math.floor(t/1e3):t;if(typeof t=="string"&&t){const e=Date.parse(t);if(!Number.isNaN(e))return Math.floor(e/1e3)}return 0}function k(t,e={}){A({type:"OPEN_SIDEBAR",panel:t,...e})}let I=[],O=!1;function P(){I.length>0||(I=[window.setInterval(q,8e3),window.setInterval(K,3e4),window.setInterval(z,2e4),window.setInterval(F,15e3),window.setInterval(J,6e4),window.setInterval(V,6e4),window.setInterval(W,6e4)])}function gt(){for(const t of I)clearInterval(t);I=[]}function G(){q(),K(),z(),F(),J(),V(),W()}function mt(){const t=document.getElementById("cockpit");t&&(t.style.display="",O=!0,G(),P(),document.addEventListener("visibilitychange",()=>{O&&(document.hidden?gt():(G(),P()))}))}async function q(){var o;const t=await w("/api/thinking/teaser"),e=document.getElementById("ticker-text"),n=document.getElementById("ticker-meta");if(!e||!n)return;if(!(t!=null&&t.has_teaser)||!t.teaser){e.textContent="—",n.textContent="";return}e.textContent=t.teaser.content;const i=((o=t.teaser.topics)==null?void 0:o.slice(0,4).join(" · "))||"";n.textContent=i;const a=document.getElementById("card-ticker");a&&(a.onclick=()=>k("multi-agent"),a.style.cursor="pointer")}async function K(){const t=await w("/api/context/heatmap"),e=document.getElementById("heatmap-chips");if(!e)return;if(!(t!=null&&t.items)||t.items.length===0){e.innerHTML='<span class="nt-card-empty">No active topics</span>';return}e.innerHTML=t.items.slice(0,12).map(i=>{const a=Math.max(.35,Number(i.opacity)||.5),o=Math.round(9+(Number(i.size)||0)*3);return`<span class="nt-heatchip" style="background:${ht(i.color)?i.color:"#7c3aed"};opacity:${a};font-size:${o}px">${l(i.name)}</span>`}).join("");const n=document.getElementById("card-heatmap");n&&(n.onclick=()=>k("context-heatmap"),n.style.cursor="pointer")}async function z(){var a,o;const t=await w("/api/conversation/starter/pending"),e=document.getElementById("starter-body");if(!e)return;if(!(t!=null&&t.has_starter)||!t.starter){e.innerHTML='<span class="nt-card-empty">Quiet.</span>';return}const n=t.starter.content,i=t.starter.topic||"";e.innerHTML=`
    <div class="nt-starter">${l(n)}</div>
    <div class="nt-starter-actions">
      <button class="nt-starter-btn primary" id="starter-accept">Reply</button>
      <button class="nt-starter-btn" id="starter-dismiss">Dismiss</button>
    </div>
  `,(a=document.getElementById("starter-accept"))==null||a.addEventListener("click",()=>{k("chat",{message:n})}),(o=document.getElementById("starter-dismiss"))==null||o.addEventListener("click",async()=>{if(i){const d=await j();fetch(`${b}/api/conversation/starter/dismiss?topic=${encodeURIComponent(i)}`,{method:"POST",headers:d}).catch(()=>{})}e.innerHTML='<span class="nt-card-empty">Dismissed.</span>'})}async function F(){const t=await w("/api/hands"),e=document.getElementById("hands-body");if(!e)return;const n=(t==null?void 0:t.hands)??[];if(n.length===0){e.innerHTML='<span class="nt-card-empty">No hands running</span>';return}const i=n.filter(o=>o.state==="running"||o.state==="active"||o.state==="paused").slice(0,6);if(i.length===0){e.innerHTML='<span class="nt-card-empty">All hands idle</span>';return}e.innerHTML=i.map(o=>`<div class="nt-hand"><span class="nt-hand-dot ${o.state==="running"||o.state==="active"?"running":o.state==="paused"?"paused":"idle"}"></span>${l(o.name)}</div>`).join("");const a=document.getElementById("card-hands");a&&(a.onclick=()=>k("hands"),a.style.cursor="pointer")}async function J(){const t=Math.floor(Date.now()/1e3)-86400,e=await w(`/api/activity/events?limit=15&after=${t}`),n=document.getElementById("activity-list");if(!n)return;const a=(Array.isArray(e==null?void 0:e.events)?e.events:[]).map(o=>({timestamp:ft(o==null?void 0:o.timestamp),title:(o==null?void 0:o.title)??(o==null?void 0:o.summary)??(o==null?void 0:o.description)??"",url:(o==null?void 0:o.url)??""}));if(a.length===0){n.innerHTML='<span class="nt-card-empty">No events</span>';return}n.innerHTML=a.slice(0,10).map(o=>{const d=o.title||"(event)",p=B(o.url)?o.url:"";return`<div class="nt-li" data-url="${l(p)}">
      <span class="nt-li-time">${pt(o.timestamp)}</span>
      <span class="nt-li-text">${l(d)}</span>
    </div>`}).join(""),n.querySelectorAll(".nt-li").forEach(o=>{const d=o.dataset.url;d&&B(d)&&(o.onclick=()=>window.open(d,"_blank","noopener"))})}async function V(){const t=await w("/api/memory/recent?limit=8"),e=document.getElementById("memories-list");if(!e)return;const n=Array.isArray(t==null?void 0:t.memories)?t.memories:[];if(n.length===0){e.innerHTML='<span class="nt-card-empty">No memories</span>';return}e.innerHTML=n.slice(0,8).map(a=>{const o=String((a==null?void 0:a.content)??"").slice(0,80);return`<div class="nt-li"><span class="nt-li-text">${l(o)}</span></div>`}).join("");const i=document.getElementById("card-memories");i&&(i.onclick=()=>k("memory-browser"),i.style.cursor="pointer")}async function W(){const t=await w("/api/feed/list?limit=6&offset=0"),e=document.getElementById("feed-row");if(!e)return;const n=(t==null?void 0:t.items)??[];if(n.length===0){e.innerHTML='<span class="nt-card-empty">No captures</span>';return}e.innerHTML=n.map(i=>{const a=i.title||"Capture",o=B(i.thumbnail)?`<img src="${l(i.thumbnail)}" alt="">`:"";return`<div class="nt-feed-item" title="${l(a)}" data-id="${l(i.id)}">
      ${o}
    </div>`}).join(""),e.querySelectorAll(".nt-feed-item").forEach(i=>{i.onclick=()=>k("feed")})}async function vt(){var n,i;const t=await y(["backendUrl"]),e=(i=(n=t==null?void 0:t.backendUrl)==null?void 0:n.trim)==null?void 0:i.call(n);e&&(b=e.replace(/\/+$/,""))}(X=(Q=c==null?void 0:c.storage)==null?void 0:Q.onChanged)==null||X.addListener((t,e)=>{if(e==="local"&&t.backendUrl){const n=t.backendUrl.newValue;b=typeof n=="string"&&n.trim()?n.trim().replace(/\/+$/,""):$}}),(async()=>(await vt(),document.readyState==="loading"?document.addEventListener("DOMContentLoaded",U):U()))()})();
