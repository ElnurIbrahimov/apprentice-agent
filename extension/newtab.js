(function(){"use strict";const h="aura_conversations",w="http://89.167.107.134",m=["The best way to predict the future is to invent it. — Alan Kay","Simplicity is the ultimate sophistication. — Leonardo da Vinci","First, solve the problem. Then, write the code. — John Johnson","Code is like humor. When you have to explain it, it is bad. — Cory House","The only way to do great work is to love what you do. — Steve Jobs","Stay hungry, stay foolish. — Steve Jobs","Think different.","Move fast and break things. Then fix them.","The computer was born to solve problems that did not exist before. — Bill Gates","Any sufficiently advanced technology is indistinguishable from magic. — Arthur C. Clarke","Intelligence is the ability to adapt to change. — Stephen Hawking","Not all those who wander are lost. — J.R.R. Tolkien","Imagination is more important than knowledge. — Albert Einstein","Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away. — Saint-Exupéry","The question of whether a computer can think is no more interesting than the question of whether a submarine can swim. — Dijkstra"];function f(t){const n=String(t.getHours()).padStart(2,"0"),o=String(t.getMinutes()).padStart(2,"0");return`${n}:${o}`}function p(t){return t.toLocaleDateString(void 0,{weekday:"long",month:"long",day:"numeric"})}function E(t){const n=Date.now()-t,o=Math.floor(n/6e4);if(o<1)return"just now";if(o<60)return o+"m ago";const s=Math.floor(o/60);if(s<24)return s+"h ago";const l=Math.floor(s/24);return l<7?l+"d ago":new Date(t).toLocaleDateString(void 0,{month:"short",day:"numeric"})}function k(){const t=Math.floor(Date.now()/864e5)%m.length;return m[t]}function c(t){const n=document.createElement("span");return n.textContent=t,n.innerHTML}const i=typeof chrome<"u"&&(chrome!=null&&chrome.storage)?chrome:null;function b(t){return new Promise(n=>{var o;(o=i==null?void 0:i.storage)!=null&&o.local?i.storage.local.get(t,s=>n(s||{})):n({})})}function u(t){var n;(n=i==null?void 0:i.runtime)!=null&&n.sendMessage&&i.runtime.sendMessage(t)}function g(){const t=document.getElementById("root"),n=new Date;t.innerHTML=`
    <div class="nt-clock">
      <div class="nt-clock-time" id="clock-time">${c(f(n))}</div>
      <div class="nt-clock-date" id="clock-date">${c(p(n))}</div>
    </div>

    <div class="nt-logo">
      <div class="nt-logo-dot"></div>
      <span class="nt-logo-text">AURA</span>
    </div>

    <form class="nt-search-wrap" id="search-form">
      <input class="nt-search" id="search-input" type="text" placeholder="Search Google..." autofocus />
      <svg class="nt-search-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"></circle>
        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
      </svg>
    </form>

    <div class="nt-actions">
      <button class="nt-action" data-panel="chat">
        <span class="nt-action-icon">💬</span>Chat
      </button>
      <button class="nt-action" data-panel="search">
        <span class="nt-action-icon">🔍</span>Search
      </button>
      <button class="nt-action" data-panel="translate">
        <span class="nt-action-icon">🌐</span>Translate
      </button>
      <button class="nt-action" data-panel="write">
        <span class="nt-action-icon">✍️</span>Write
      </button>
    </div>

    <div class="nt-recent" id="recent-section" style="display:none">
      <div class="nt-recent-title">Recent conversations</div>
      <div class="nt-recent-list" id="recent-list"></div>
    </div>

    <div class="nt-quote">
      <div class="nt-quote-text">${c(k())}</div>
    </div>

    <div class="nt-status" id="status-indicator">
      <div class="nt-status-dot"></div>
      AURA offline
    </div>
  `,setInterval(()=>{const a=new Date,e=document.getElementById("clock-time"),d=document.getElementById("clock-date");e&&(e.textContent=f(a)),d&&(d.textContent=p(a))},1e4);const o=document.getElementById("search-form"),s=document.getElementById("search-input");o.addEventListener("submit",a=>{a.preventDefault();const e=s.value.trim();e&&(e.startsWith("!")?u({type:"OPEN_SIDEBAR",panel:"chat",message:e.slice(1).trim()}):window.location.href=`https://www.google.com/search?q=${encodeURIComponent(e)}`)}),document.querySelectorAll(".nt-action[data-panel]").forEach(a=>{a.addEventListener("click",()=>{const e=a.dataset.panel;e&&u({type:"OPEN_SIDEBAR",panel:e})})}),b([h]).then(a=>{const e=(a[h]||[]).slice(0,5);if(e.length===0)return;const d=document.getElementById("recent-section"),v=document.getElementById("recent-list");d.style.display="",v.innerHTML=e.map(r=>`<button class="nt-recent-item" data-conv-id="${c(r.id)}">
            <span class="nt-recent-item-text">${c(r.title)}</span>
            <span class="nt-recent-item-time">${c(E(r.timestamp))}</span>
          </button>`).join(""),v.querySelectorAll(".nt-recent-item").forEach(r=>{r.addEventListener("click",()=>{const y=r.dataset.convId;y&&u({type:"OPEN_SIDEBAR",panel:"chat",conversationId:y})})})});const l=new AbortController;fetch(`${w}/api/health`,{signal:l.signal,method:"GET"}).then(a=>{if(a.ok){const e=document.getElementById("status-indicator");e&&(e.innerHTML='<div class="nt-status-dot online"></div>AURA online'),s.placeholder="Search Google or type ! to ask AURA..."}}).catch(()=>{}),setTimeout(()=>l.abort(),3e3)}document.readyState==="loading"?document.addEventListener("DOMContentLoaded",g):g()})();
