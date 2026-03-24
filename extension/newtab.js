(function(){"use strict";const v="aura_conversations",T="https://aura-elnur.duckdns.org",O="i-L5ShpMkY2B7loNb8VS4EAAT-Ronh-K8cIgRILGjnQ",M=["The best way to predict the future is to invent it. — Alan Kay","Simplicity is the ultimate sophistication. — Leonardo da Vinci","First, solve the problem. Then, write the code. — John Johnson","Code is like humor. When you have to explain it, it is bad. — Cory House","The only way to do great work is to love what you do. — Steve Jobs","Stay hungry, stay foolish. — Steve Jobs","Think different.","Move fast and break things. Then fix them.","The computer was born to solve problems that did not exist before. — Bill Gates","Any sufficiently advanced technology is indistinguishable from magic. — Arthur C. Clarke","Intelligence is the ability to adapt to change. — Stephen Hawking","Not all those who wander are lost. — J.R.R. Tolkien","Imagination is more important than knowledge. — Albert Einstein","Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away. — Saint-Exupéry","The question of whether a computer can think is no more interesting than the question of whether a submarine can swim. — Dijkstra"],L={neutral:{core:"#7c3aed",glow:"#a78bfa",highlight:"#e0d6ff"},calm:{core:"#7c3aed",glow:"#a78bfa",highlight:"#e0d6ff"},content:{core:"#7c3aed",glow:"#a78bfa",highlight:"#e0d6ff"},thoughtful:{core:"#7c3aed",glow:"#a78bfa",highlight:"#e0d6ff"},happy:{core:"#f59e0b",glow:"#fbbf24",highlight:"#fef3c7"},excited:{core:"#f59e0b",glow:"#fbbf24",highlight:"#fef3c7"},playful:{core:"#f59e0b",glow:"#fbbf24",highlight:"#fef3c7"},gratification:{core:"#f59e0b",glow:"#fbbf24",highlight:"#fef3c7"},joy:{core:"#f59e0b",glow:"#fbbf24",highlight:"#fef3c7"},curious:{core:"#3b82f6",glow:"#60a5fa",highlight:"#dbeafe"},surprised:{core:"#3b82f6",glow:"#60a5fa",highlight:"#dbeafe"},engaged:{core:"#3b82f6",glow:"#60a5fa",highlight:"#dbeafe"},admiration:{core:"#3b82f6",glow:"#60a5fa",highlight:"#dbeafe"},confident:{core:"#10b981",glow:"#34d399",highlight:"#d1fae5"},satisfaction:{core:"#10b981",glow:"#34d399",highlight:"#d1fae5"},pride:{core:"#10b981",glow:"#34d399",highlight:"#d1fae5"},gratitude:{core:"#10b981",glow:"#34d399",highlight:"#d1fae5"},sad:{core:"#64748b",glow:"#94a3b8",highlight:"#cbd5e1"},distress:{core:"#64748b",glow:"#94a3b8",highlight:"#cbd5e1"},disappointment:{core:"#64748b",glow:"#94a3b8",highlight:"#cbd5e1"},sorry_for:{core:"#64748b",glow:"#94a3b8",highlight:"#cbd5e1"},remorse:{core:"#64748b",glow:"#94a3b8",highlight:"#cbd5e1"},anxious:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},frustrated:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},angry:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},fearful:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},fear:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},hate:{core:"#ef4444",glow:"#f87171",highlight:"#fecaca"},empathetic:{core:"#8b5cf6",glow:"#a78bfa",highlight:"#ede9fe"},concerned:{core:"#8b5cf6",glow:"#a78bfa",highlight:"#ede9fe"}};function x(e){return L[e]||L.neutral}function B(e){const t=document.documentElement;t.style.setProperty("--dot-core",e.core),t.style.setProperty("--dot-glow",e.glow),t.style.setProperty("--dot-highlight",e.highlight)}let k=null;async function G(){var a,o,c;if(k&&Date.now()-k.ts<3e5){B(x(k.emotion));return}const e=await E(["apiKey"]),t=((o=(a=e==null?void 0:e.apiKey)==null?void 0:a.trim)==null?void 0:o.call(a))||O;try{const l={"X-API-Key":t},h=await fetch(`${T}/api/status`,{signal:AbortSignal.timeout(4e3),headers:l});if(!h.ok)return;const f=await h.json(),w=((c=f==null?void 0:f.mood)==null?void 0:c.emotion)||"neutral";k={emotion:w,ts:Date.now()},B(x(w))}catch{}}function I(e){const t=String(e.getHours()).padStart(2,"0"),a=String(e.getMinutes()).padStart(2,"0");return`${t}<span class="nt-clock-colon">:</span>${a}`}function S(e){return e.toLocaleDateString(void 0,{weekday:"long",month:"long",day:"numeric"})}function H(e){const t=e.getHours();return t>=5&&t<12?"Good morning":t>=12&&t<17?"Good afternoon":t>=17&&t<21?"Good evening":"Good night"}const C="aura_weather_cache",U=30*60*1e3;function N(e){const t=e.toLowerCase();return t.includes("thunder")?"⛈️":t.includes("snow")||t.includes("blizzard")?"❄️":t.includes("rain")?"🌧️":t.includes("drizzle")?"🌦️":t.includes("mist")||t.includes("fog")?"🌫️":t.includes("overcast")||t.includes("cloudy")?"☁️":t.includes("partly")?"⛅":t.includes("sunny")||t.includes("clear")?"☀️":"🌤️"}async function P(){var a,o,c,l,h,f,w,b,y;const t=(await E([C]))[C];if(t&&Date.now()-t.cachedAt<U)return t;try{const n=new AbortController;setTimeout(()=>n.abort(),5e3);const i=await fetch("https://wttr.in/?format=j1",{signal:n.signal});if(!i.ok)return null;const p=await i.json(),r=(a=p.current_condition)==null?void 0:a[0],m=(o=p.nearest_area)==null?void 0:o[0];if(!r)return null;const u={temp:r.temp_C,condition:((l=(c=r.weatherDesc)==null?void 0:c[0])==null?void 0:l.value)||"",icon:N(((f=(h=r.weatherDesc)==null?void 0:h[0])==null?void 0:f.value)||""),city:((b=(w=m==null?void 0:m.areaName)==null?void 0:w[0])==null?void 0:b.value)||"",cachedAt:Date.now()};return(y=s==null?void 0:s.storage)!=null&&y.local&&s.storage.local.set({[C]:u}),u}catch{return null}}function K(e){const t=Date.now()-e,a=Math.floor(t/6e4);if(a<1)return"just now";if(a<60)return a+"m ago";const o=Math.floor(a/60);if(o<24)return o+"h ago";const c=Math.floor(o/24);return c<7?c+"d ago":new Date(e).toLocaleDateString(void 0,{month:"short",day:"numeric"})}function q(){const e=Math.floor(Date.now()/864e5)%M.length;return M[e]}function d(e){const t=document.createElement("span");return t.textContent=e,t.innerHTML}const s=typeof chrome<"u"&&(chrome!=null&&chrome.storage)?chrome:null;function E(e){return new Promise(t=>{var a;(a=s==null?void 0:s.storage)!=null&&a.local?s.storage.local.get(e,o=>t(o||{})):t({})})}function A(e){var t;(t=s==null?void 0:s.runtime)!=null&&t.sendMessage&&s.runtime.sendMessage(e)}function z(){var e;(e=s==null?void 0:s.topSites)!=null&&e.get&&s.topSites.get(t=>{if(!t||t.length===0)return;const a=document.getElementById("topsites-section"),o=document.getElementById("topsites-row");!a||!o||(a.style.display="",o.innerHTML=t.slice(0,8).map(c=>{let l;try{l=new URL(c.url).hostname.replace("www.","")}catch{l=c.url}const h=l.split(".")[0]||l;return`<a class="nt-topsite" href="${d(c.url)}" title="${d(c.title||l)}">
        <div class="nt-topsite-icon">
          <img src="https://www.google.com/s2/favicons?domain=${d(l)}&sz=32" alt="" width="20" height="20" loading="lazy" />
        </div>
        <span class="nt-topsite-label">${d(h)}</span>
      </a>`}).join(""))})}const g={chat:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>',search:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>',translate:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 8l6 6"></path><path d="M4 14l6-6 2-3"></path><path d="M2 5h12"></path><path d="M7 2v3"></path><path d="M22 22l-5-10-5 10"></path><path d="M14 18h6"></path></svg>',write:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"></path><path d="m15 5 4 4"></path></svg>',code:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>',research:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"></path><circle cx="11" cy="10" r="3"></circle><path d="m14 13-1.5-1.5"></path></svg>',ocr:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 2H2v5"></path><path d="M17 2h5v5"></path><path d="M7 22H2v-5"></path><path d="M17 22h5v-5"></path><line x1="5" y1="12" x2="19" y2="12"></line></svg>',grammar:'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline><line x1="4" y1="21" x2="20" y2="21"></line></svg>',chatBubble:'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>',plus:'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>',x:'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>'};function R(){var y;const e=document.getElementById("root"),t=new Date;e.innerHTML=`
    <div class="nt-clock nt-fade nt-fade-d1">
      <div class="nt-greeting" id="clock-greeting">${d(H(t))}</div>
      <div class="nt-clock-time" id="clock-time">${I(t)}</div>
      <div class="nt-clock-date" id="clock-date">${d(S(t))}<span class="nt-weather" id="weather"></span></div>
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
      <div class="nt-quote-text">${d(q())}</div>
    </div>

    <div class="nt-brand">AURA</div>

    <div class="nt-status" id="status-indicator">
      <div class="nt-status-dot"></div>
      AURA offline
    </div>
  `,setInterval(()=>{const n=new Date,i=document.getElementById("clock-time"),p=document.getElementById("clock-greeting"),r=document.getElementById("clock-date");if(i&&(i.innerHTML=I(n)),p&&(p.textContent=H(n)),r){const m=document.getElementById("weather"),u=m?m.outerHTML:'<span class="nt-weather" id="weather"></span>';r.innerHTML=d(S(n))+u}},1e3),P().then(n=>{if(!n)return;const i=document.getElementById("weather");i&&(i.innerHTML=` &middot; ${n.icon} ${d(n.temp)}°C ${d(n.condition)}`)});const a=document.getElementById("search-form"),o=document.getElementById("search-input"),c=document.getElementById("mode-toggle"),l=document.getElementById("mode-hint");let h="google",f=!1;function w(n){h=n,n==="aura"?(c.classList.add("aura-mode"),c.innerHTML='<span class="nt-mode-icon"><span class="nt-mode-icon-aura"></span></span><span class="nt-mode-label">AURA</span>',o.placeholder="Ask AURA anything...",l.textContent="Tab to switch to Google"):(c.classList.remove("aura-mode"),c.innerHTML='<span class="nt-mode-icon"><span class="nt-mode-icon-google">G</span></span><span class="nt-mode-label">Google</span>',o.placeholder="Search Google...",l.textContent=f?"Tab to switch to AURA":"AURA offline"),o.focus()}c.addEventListener("click",()=>{f&&w(h==="google"?"aura":"google")}),o.addEventListener("focus",()=>a.classList.add("focused")),o.addEventListener("blur",()=>a.classList.remove("focused")),a.addEventListener("submit",n=>{n.preventDefault();const i=o.value.trim();i&&(h==="aura"?(A({type:"OPEN_SIDEBAR",panel:"chat",message:i}),o.value=""):window.location.href=`https://www.google.com/search?q=${encodeURIComponent(i)}`)}),document.addEventListener("keydown",n=>{n.key==="Tab"&&document.activeElement===o&&f&&(n.preventDefault(),w(h==="google"?"aura":"google")),n.key==="/"&&document.activeElement!==o&&(n.preventDefault(),o.focus()),n.key==="Escape"&&document.activeElement===o&&(o.value="",o.blur())}),document.querySelectorAll(".nt-action[data-panel]").forEach(n=>{n.addEventListener("click",()=>{const i=n.dataset.panel;i&&A({type:"OPEN_SIDEBAR",panel:i})})}),(y=document.getElementById("new-chat-btn"))==null||y.addEventListener("click",()=>{A({type:"OPEN_SIDEBAR",panel:"chat",newConversation:!0})}),z(),E([v]).then(n=>{const i=(n[v]||[]).slice(0,5);if(i.length===0)return;const p=document.getElementById("recent-list");p.innerHTML=i.map(r=>`<button class="nt-recent-item" data-conv-id="${d(r.id)}">
            <span class="nt-recent-item-icon">${g.chatBubble}</span>
            <span class="nt-recent-item-text">${d(r.title)}</span>
            <span class="nt-recent-item-time">${d(K(r.timestamp))}</span>
            <button class="nt-recent-item-delete" title="Remove">${g.x}</button>
          </button>`).join(""),p.querySelectorAll(".nt-recent-item").forEach(r=>{r.addEventListener("click",m=>{if(m.target.closest(".nt-recent-item-delete"))return;const u=r.dataset.convId;u&&A({type:"OPEN_SIDEBAR",panel:"chat",conversationId:u})})}),p.querySelectorAll(".nt-recent-item-delete").forEach(r=>{r.addEventListener("click",m=>{var D;m.stopPropagation();const u=r.closest(".nt-recent-item"),$=(D=u==null?void 0:u.dataset)==null?void 0:D.convId;$&&(u.remove(),E([v]).then(J=>{var j;const _=(J[v]||[]).filter(V=>V.id!==$);(j=s==null?void 0:s.storage)!=null&&j.local&&s.storage.local.set({[v]:_}),_.length===0&&(p.innerHTML='<div class="nt-recent-empty">No conversations yet. Click Chat or switch to AURA mode to start.</div>')}))})})});const b=new AbortController;fetch(`${T}/api/health`,{signal:b.signal,method:"GET"}).then(n=>{if(n.ok){const i=document.getElementById("status-indicator");i&&(i.innerHTML='<div class="nt-status-dot online"></div>AURA online'),f=!0,c.classList.remove("disabled"),l.textContent="Tab to switch to AURA",G()}}).catch(()=>{}),setTimeout(()=>b.abort(),3e3)}document.readyState==="loading"?document.addEventListener("DOMContentLoaded",R):R()})();
