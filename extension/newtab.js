(function(){"use strict";const k="aura_conversations",A="https://aura-elnur.duckdns.org",L=["The best way to predict the future is to invent it. — Alan Kay","Simplicity is the ultimate sophistication. — Leonardo da Vinci","First, solve the problem. Then, write the code. — John Johnson","Code is like humor. When you have to explain it, it is bad. — Cory House","The only way to do great work is to love what you do. — Steve Jobs","Stay hungry, stay foolish. — Steve Jobs","Think different.","Move fast and break things. Then fix them.","The computer was born to solve problems that did not exist before. — Bill Gates","Any sufficiently advanced technology is indistinguishable from magic. — Arthur C. Clarke","Intelligence is the ability to adapt to change. — Stephen Hawking","Not all those who wander are lost. — J.R.R. Tolkien","Imagination is more important than knowledge. — Albert Einstein","Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away. — Saint-Exupéry","The question of whether a computer can think is no more interesting than the question of whether a submarine can swim. — Dijkstra"],x={neutral:{core:"#7c3aed",glow:"#a78bfa",highlight:"#e0d6ff"},calm:{core:"#7c3aed",glow:"#a78bfa",highlight:"#e0d6ff"},content:{core:"#7c3aed",glow:"#a78bfa",highlight:"#e0d6ff"},thoughtful:{core:"#7c3aed",glow:"#a78bfa",highlight:"#e0d6ff"},happy:{core:"#f59e0b",glow:"#fbbf24",highlight:"#fef3c7"},excited:{core:"#f59e0b",glow:"#fbbf24",highlight:"#fef3c7"},playful:{core:"#f59e0b",glow:"#fbbf24",highlight:"#fef3c7"},gratification:{core:"#f59e0b",glow:"#fbbf24",highlight:"#fef3c7"},joy:{core:"#f59e0b",glow:"#fbbf24",highlight:"#fef3c7"},curious:{core:"#3b82f6",glow:"#60a5fa",highlight:"#dbeafe"},surprised:{core:"#3b82f6",glow:"#60a5fa",highlight:"#dbeafe"},engaged:{core:"#3b82f6",glow:"#60a5fa",highlight:"#dbeafe"},admiration:{core:"#3b82f6",glow:"#60a5fa",highlight:"#dbeafe"},confident:{core:"#10b981",glow:"#34d399",highlight:"#d1fae5"},satisfaction:{core:"#10b981",glow:"#34d399",highlight:"#d1fae5"},pride:{core:"#10b981",glow:"#34d399",highlight:"#d1fae5"},gratitude:{core:"#10b981",glow:"#34d399",highlight:"#d1fae5"},sad:{core:"#64748b",glow:"#94a3b8",highlight:"#cbd5e1"},distress:{core:"#64748b",glow:"#94a3b8",highlight:"#cbd5e1"},disappointment:{core:"#64748b",glow:"#94a3b8",highlight:"#cbd5e1"},sorry_for:{core:"#64748b",glow:"#94a3b8",highlight:"#cbd5e1"},remorse:{core:"#64748b",glow:"#94a3b8",highlight:"#cbd5e1"},anxious:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},frustrated:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},angry:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},fearful:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},fear:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},hate:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},empathetic:{core:"#8b5cf6",glow:"#a78bfa",highlight:"#ede9fe"},concerned:{core:"#8b5cf6",glow:"#a78bfa",highlight:"#ede9fe"}};function B(e){return x[e]||x.neutral}function $(e){const t=document.documentElement;t.style.setProperty("--dot-core",e.core),t.style.setProperty("--dot-glow",e.glow),t.style.setProperty("--dot-highlight",e.highlight)}let C=null;async function Q(){var o,n,a;if(C&&Date.now()-C.ts<3e5){$(B(C.emotion));return}const e=await w(["apiKey"]),t=((n=(o=e==null?void 0:e.apiKey)==null?void 0:o.trim)==null?void 0:n.call(o))||"";if(t)try{const i={"X-API-Key":t},h=await fetch(`${A}/api/status`,{signal:AbortSignal.timeout(4e3),headers:i});if(!h.ok)return;const f=await h.json(),v=((a=f==null?void 0:f.mood)==null?void 0:a.emotion)||"neutral";C={emotion:v,ts:Date.now()},$(B(v))}catch{}}function H(e){const t=String(e.getHours()).padStart(2,"0"),o=String(e.getMinutes()).padStart(2,"0");return`${t}<span class="nt-clock-colon">:</span>${o}`}function S(e){return e.toLocaleDateString(void 0,{weekday:"long",month:"long",day:"numeric"})}function D(e){const t=e.getHours();return t>=5&&t<12?"Good morning":t>=12&&t<17?"Good afternoon":t>=17&&t<21?"Good evening":"Good night"}const I="aura_weather_cache",V=30*60*1e3;function F(e){const t=e.toLowerCase();return t.includes("thunder")?"⛈️":t.includes("snow")||t.includes("blizzard")?"❄️":t.includes("rain")?"🌧️":t.includes("drizzle")?"🌦️":t.includes("mist")||t.includes("fog")?"🌫️":t.includes("overcast")||t.includes("cloudy")?"☁️":t.includes("partly")?"⛅":t.includes("sunny")||t.includes("clear")?"☀️":"🌤️"}async function X(){var o,n,a,i,h,f,v,T,M;const t=(await w([I]))[I];if(t&&Date.now()-t.cachedAt<V)return t;try{const s=new AbortController;setTimeout(()=>s.abort(),5e3);const r=await fetch("https://wttr.in/?format=j1",{signal:s.signal});if(!r.ok)return null;const u=await r.json(),d=(o=u.current_condition)==null?void 0:o[0],m=(n=u.nearest_area)==null?void 0:n[0];if(!d)return null;const p={temp:d.temp_C,condition:((i=(a=d.weatherDesc)==null?void 0:a[0])==null?void 0:i.value)||"",icon:F(((f=(h=d.weatherDesc)==null?void 0:h[0])==null?void 0:f.value)||""),city:((T=(v=m==null?void 0:m.areaName)==null?void 0:v[0])==null?void 0:T.value)||"",cachedAt:Date.now()};return(M=l==null?void 0:l.storage)!=null&&M.local&&l.storage.local.set({[I]:p}),p}catch{return null}}function Y(e){const t=Date.now()-e,o=Math.floor(t/6e4);if(o<1)return"just now";if(o<60)return o+"m ago";const n=Math.floor(o/60);if(n<24)return n+"h ago";const a=Math.floor(n/24);return a<7?a+"d ago":new Date(e).toLocaleDateString(void 0,{month:"short",day:"numeric"})}function Z(){const e=Math.floor(Date.now()/864e5)%L.length;return L[e]}function c(e){const t=document.createElement("span");return t.textContent=e,t.innerHTML}const l=typeof chrome<"u"&&(chrome!=null&&chrome.storage)?chrome:null;function w(e){return new Promise(t=>{var o;(o=l==null?void 0:l.storage)!=null&&o.local?l.storage.local.get(e,n=>t(n||{})):t({})})}function E(e){var t;(t=l==null?void 0:l.runtime)!=null&&t.sendMessage&&l.runtime.sendMessage(e)}function tt(){var e;(e=l==null?void 0:l.topSites)!=null&&e.get&&l.topSites.get(t=>{if(!t||t.length===0)return;const o=document.getElementById("topsites-section"),n=document.getElementById("topsites-row");!o||!n||(o.style.display="",n.innerHTML=t.slice(0,8).map(a=>{let i;try{i=new URL(a.url).hostname.replace("www.","")}catch{i=a.url}const h=i.split(".")[0]||i;return`<a class="nt-topsite" href="${c(a.url)}" title="${c(a.title||i)}">
        <div class="nt-topsite-icon">
          <img src="https://www.google.com/s2/favicons?domain=${c(i)}&sz=32" alt="" width="20" height="20" loading="lazy" />
        </div>
        <span class="nt-topsite-label">${c(h)}</span>
      </a>`}).join(""))})}const g={chat:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>',search:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>',translate:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 8l6 6"></path><path d="M4 14l6-6 2-3"></path><path d="M2 5h12"></path><path d="M7 2v3"></path><path d="M22 22l-5-10-5 10"></path><path d="M14 18h6"></path></svg>',write:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"></path><path d="m15 5 4 4"></path></svg>',code:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>',research:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"></path><circle cx="11" cy="10" r="3"></circle><path d="m14 13-1.5-1.5"></path></svg>',ocr:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 2H2v5"></path><path d="M17 2h5v5"></path><path d="M7 22H2v-5"></path><path d="M17 22h5v-5"></path><line x1="5" y1="12" x2="19" y2="12"></line></svg>',grammar:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline><line x1="4" y1="21" x2="20" y2="21"></line></svg>',chatBubble:'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>',plus:'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>',x:'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>'};function R(){var M;const e=document.getElementById("root"),t=new Date;e.innerHTML=`
    <div class="nt-clock nt-fade nt-fade-d1">
      <div class="nt-greeting" id="clock-greeting">${c(D(t))}</div>
      <div class="nt-clock-time" id="clock-time">${H(t)}</div>
      <div class="nt-clock-date" id="clock-date">${c(S(t))}<span class="nt-weather" id="weather"></span></div>
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
        <span class="nt-action-icon">${g.chat}</span>Chat
      </button>
      <button class="nt-action" data-panel="search">
        <span class="nt-action-icon">${g.search}</span>Search
      </button>
      <button class="nt-action" data-panel="translate">
        <span class="nt-action-icon">${g.translate}</span>Translate
      </button>
      <button class="nt-action" data-panel="write">
        <span class="nt-action-icon">${g.write}</span>Write
      </button>
      <button class="nt-action" data-panel="code">
        <span class="nt-action-icon">${g.code}</span>Code
      </button>
      <button class="nt-action" data-panel="research">
        <span class="nt-action-icon">${g.research}</span>Research
      </button>
      <button class="nt-action" data-panel="ocr">
        <span class="nt-action-icon">${g.ocr}</span>OCR
      </button>
      <button class="nt-action" data-panel="grammar">
        <span class="nt-action-icon">${g.grammar}</span>Grammar
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
          ${g.plus} New chat
        </button>
      </div>
      <div class="nt-recent-list" id="recent-list">
        <div class="nt-recent-empty">No conversations yet. Click Chat or switch to AURA mode to start.</div>
      </div>
    </div>

    <div class="nt-quote">
      <div class="nt-quote-text">${c(Z())}</div>
    </div>

    <div class="nt-brand">AURA</div>

    <div class="nt-status" id="status-indicator">
      <div class="nt-status-dot"></div>
      AURA offline
    </div>
  `,setInterval(()=>{const s=new Date,r=document.getElementById("clock-time"),u=document.getElementById("clock-greeting"),d=document.getElementById("clock-date");if(r&&(r.innerHTML=H(s)),u&&(u.textContent=D(s)),d){const m=document.getElementById("weather"),p=m?m.outerHTML:'<span class="nt-weather" id="weather"></span>';d.innerHTML=c(S(s))+p}},1e3),X().then(s=>{if(!s)return;const r=document.getElementById("weather");r&&(r.innerHTML=` &middot; ${s.icon} ${c(s.temp)}°C ${c(s.condition)}`)});const o=document.getElementById("search-form"),n=document.getElementById("search-input"),a=document.getElementById("mode-toggle"),i=document.getElementById("mode-hint");let h="google",f=!1;function v(s){h=s,s==="aura"?(a.classList.add("aura-mode"),a.innerHTML='<span class="nt-mode-icon"><span class="nt-mode-icon-aura"></span></span><span class="nt-mode-label">AURA</span>',n.placeholder="Ask AURA anything...",i.textContent="Tab to switch to Google"):(a.classList.remove("aura-mode"),a.innerHTML='<span class="nt-mode-icon"><span class="nt-mode-icon-google">G</span></span><span class="nt-mode-label">Google</span>',n.placeholder="Search Google...",i.textContent=f?"Tab to switch to AURA":"AURA offline"),n.focus()}a.addEventListener("click",()=>{f&&v(h==="google"?"aura":"google")}),n.addEventListener("focus",()=>o.classList.add("focused")),n.addEventListener("blur",()=>o.classList.remove("focused")),o.addEventListener("submit",s=>{s.preventDefault();const r=n.value.trim();r&&(h==="aura"?(E({type:"OPEN_SIDEBAR",panel:"chat",message:r}),n.value=""):window.location.href=`https://www.google.com/search?q=${encodeURIComponent(r)}`)}),document.addEventListener("keydown",s=>{s.key==="Tab"&&document.activeElement===n&&f&&(s.preventDefault(),v(h==="google"?"aura":"google")),s.key==="/"&&document.activeElement!==n&&(s.preventDefault(),n.focus()),s.key==="Escape"&&document.activeElement===n&&(n.value="",n.blur())}),document.querySelectorAll(".nt-action[data-panel]").forEach(s=>{s.addEventListener("click",()=>{const r=s.dataset.panel;r&&E({type:"OPEN_SIDEBAR",panel:r})})}),(M=document.getElementById("new-chat-btn"))==null||M.addEventListener("click",()=>{E({type:"OPEN_SIDEBAR",panel:"chat",newConversation:!0})}),tt(),w([k]).then(s=>{const r=(s[k]||[]).slice(0,5);if(r.length===0)return;const u=document.getElementById("recent-list");u.innerHTML=r.map(d=>`<button class="nt-recent-item" data-conv-id="${c(d.id)}">
            <span class="nt-recent-item-icon">${g.chatBubble}</span>
            <span class="nt-recent-item-text">${c(d.title)}</span>
            <span class="nt-recent-item-time">${c(Y(d.timestamp))}</span>
            <button class="nt-recent-item-delete" title="Remove">${g.x}</button>
          </button>`).join(""),u.querySelectorAll(".nt-recent-item").forEach(d=>{d.addEventListener("click",m=>{if(m.target.closest(".nt-recent-item-delete"))return;const p=d.dataset.convId;p&&E({type:"OPEN_SIDEBAR",panel:"chat",conversationId:p})})}),u.querySelectorAll(".nt-recent-item-delete").forEach(d=>{d.addEventListener("click",m=>{var z;m.stopPropagation();const p=d.closest(".nt-recent-item"),K=(z=p==null?void 0:p.dataset)==null?void 0:z.convId;K&&(p.remove(),w([k]).then(ot=>{var W;const J=(ot[k]||[]).filter(at=>at.id!==K);(W=l==null?void 0:l.storage)!=null&&W.local&&l.storage.local.set({[k]:J}),J.length===0&&(u.innerHTML='<div class="nt-recent-empty">No conversations yet. Click Chat or switch to AURA mode to start.</div>')}))})})});const T=new AbortController;fetch(`${A}/api/health`,{signal:T.signal,method:"GET"}).then(s=>{if(s.ok){const r=document.getElementById("status-indicator");r&&(r.innerHTML='<div class="nt-status-dot online"></div>AURA online'),f=!0,a.classList.remove("disabled"),i.textContent="Tab to switch to AURA",Q(),w(["cockpitDisabled"]).then(u=>{u!=null&&u.cockpitDisabled||nt()})}}).catch(()=>{}),setTimeout(()=>T.abort(),3e3)}async function _(){var o,n;const e=await w(["apiKey"]),t=((n=(o=e==null?void 0:e.apiKey)==null?void 0:o.trim)==null?void 0:n.call(o))||"";return t?{"X-API-Key":t}:{}}async function y(e,t=4e3){try{const o=await _(),n=new AbortController,a=setTimeout(()=>n.abort(),t),i=await fetch(`${A}${e}`,{headers:o,signal:n.signal});return clearTimeout(a),i.ok?await i.json():null}catch{return null}}function et(e){return new Date(e*1e3).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"})}function b(e,t={}){E({type:"OPEN_SIDEBAR",panel:e,...t})}function nt(){const e=document.getElementById("cockpit");e&&(e.style.display="",j(),N(),O(),U(),G(),P(),q(),setInterval(j,8e3),setInterval(N,3e4),setInterval(O,2e4),setInterval(U,15e3),setInterval(G,6e4),setInterval(P,6e4),setInterval(q,6e4))}async function j(){var i;const e=await y("/api/thinking/teaser"),t=document.getElementById("ticker-text"),o=document.getElementById("ticker-meta");if(!t||!o)return;if(!(e!=null&&e.has_teaser)||!e.teaser){t.textContent="—",o.textContent="";return}t.textContent=e.teaser.content;const n=((i=e.teaser.topics)==null?void 0:i.slice(0,4).join(" · "))||"";o.textContent=n;const a=document.getElementById("card-ticker");a&&(a.onclick=()=>b("multi-agent"),a.style.cursor="pointer")}async function N(){const e=await y("/api/context/heatmap"),t=document.getElementById("heatmap-chips");if(!t)return;if(!(e!=null&&e.items)||e.items.length===0){t.innerHTML='<span class="nt-card-empty">No active topics</span>';return}t.innerHTML=e.items.slice(0,12).map(n=>{const a=Math.max(.35,n.opacity),i=Math.round(9+n.size*3);return`<span class="nt-heatchip" style="background:${c(n.color)};opacity:${a};font-size:${i}px">${c(n.name)}</span>`}).join("");const o=document.getElementById("card-heatmap");o&&(o.onclick=()=>b("context-heatmap"),o.style.cursor="pointer")}async function O(){var a,i;const e=await y("/api/conversation/starter/pending"),t=document.getElementById("starter-body");if(!t)return;if(!(e!=null&&e.has_starter)||!e.starter){t.innerHTML='<span class="nt-card-empty">Quiet.</span>';return}const o=e.starter.content,n=e.starter.topic||"";t.innerHTML=`
    <div class="nt-starter">${c(o)}</div>
    <div class="nt-starter-actions">
      <button class="nt-starter-btn primary" id="starter-accept">Reply</button>
      <button class="nt-starter-btn" id="starter-dismiss">Dismiss</button>
    </div>
  `,(a=document.getElementById("starter-accept"))==null||a.addEventListener("click",()=>{b("chat",{message:o})}),(i=document.getElementById("starter-dismiss"))==null||i.addEventListener("click",async()=>{if(n){const h=await _();fetch(`${A}/api/conversation/starter/dismiss?topic=${encodeURIComponent(n)}`,{method:"POST",headers:h}).catch(()=>{})}t.innerHTML='<span class="nt-card-empty">Dismissed.</span>'})}async function U(){const e=await y("/api/hands"),t=document.getElementById("hands-body");if(!t)return;const o=(e==null?void 0:e.hands)??[];if(o.length===0){t.innerHTML='<span class="nt-card-empty">No hands running</span>';return}const n=o.filter(i=>i.state==="running"||i.state==="active"||i.state==="paused").slice(0,6);if(n.length===0){t.innerHTML='<span class="nt-card-empty">All hands idle</span>';return}t.innerHTML=n.map(i=>`<div class="nt-hand"><span class="nt-hand-dot ${i.state==="running"||i.state==="active"?"running":i.state==="paused"?"paused":"idle"}"></span>${c(i.name)}</div>`).join("");const a=document.getElementById("card-hands");a&&(a.onclick=()=>b("hands"),a.style.cursor="pointer")}async function G(){const e=Math.floor(Date.now()/1e3)-86400,t=await y(`/api/activity/events?limit=15&after=${e}`),o=document.getElementById("activity-list");if(!o)return;const n=(t==null?void 0:t.events)??[];if(n.length===0){o.innerHTML='<span class="nt-card-empty">No events</span>';return}o.innerHTML=n.slice(0,10).map(a=>{const i=a.title||a.description||"(event)";return`<div class="nt-li" data-url="${c(a.url||"")}">
      <span class="nt-li-time">${et(a.timestamp)}</span>
      <span class="nt-li-text">${c(i)}</span>
    </div>`}).join(""),o.querySelectorAll(".nt-li").forEach(a=>{const i=a.dataset.url;i&&(a.onclick=()=>window.open(i,"_blank","noopener"))})}async function P(){const e=await y("/api/memory/recent?limit=8"),t=document.getElementById("memories-list");if(!t)return;const o=(e==null?void 0:e.memories)??[];if(o.length===0){t.innerHTML='<span class="nt-card-empty">No memories</span>';return}t.innerHTML=o.slice(0,8).map(a=>{const i=(a.content||"").slice(0,80);return`<div class="nt-li"><span class="nt-li-text">${c(i)}</span></div>`}).join("");const n=document.getElementById("card-memories");n&&(n.onclick=()=>b("memory-browser"),n.style.cursor="pointer")}async function q(){const e=await y("/api/feed/list?limit=6&offset=0"),t=document.getElementById("feed-row");if(!t)return;const o=(e==null?void 0:e.items)??[];if(o.length===0){t.innerHTML='<span class="nt-card-empty">No captures</span>';return}t.innerHTML=o.map(n=>{const a=n.title||"Capture";return`<div class="nt-feed-item" title="${c(a)}" data-id="${c(n.id)}">
      ${n.thumbnail?`<img src="${c(n.thumbnail)}" alt="">`:""}
    </div>`}).join(""),t.querySelectorAll(".nt-feed-item").forEach(n=>{n.onclick=()=>b("feed")})}document.readyState==="loading"?document.addEventListener("DOMContentLoaded",R):R()})();
