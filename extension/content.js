(function(){"use strict";const ve={general:{accent:"#7c3aed",glow:"rgba(124, 58, 237, 0.35)",icon:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M8 1L10 6H15L11 9.5L12.5 14.5L8 11.5L3.5 14.5L5 9.5L1 6H6L8 1Z"/>
    </svg>`},article:{accent:"#3b82f6",glow:"rgba(59, 130, 246, 0.35)",icon:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M3 2h10a1 1 0 011 1v10a1 1 0 01-1 1H3a1 1 0 01-1-1V3a1 1 0 011-1zm1 3v1h8V5H4zm0 3v1h8V8H4zm0 3v1h5v-1H4z"/>
    </svg>`},media:{accent:"#f59e0b",glow:"rgba(245, 158, 11, 0.35)",icon:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M2 4h1v8H2V4zm2 2h1v4H4V6zm2-1h1v6H6V5zm2 2h1v2H8V7zm2-2h1v6h-1V5zm2 1h1v4h-1V6z"/>
    </svg>`},code:{accent:"#10b981",glow:"rgba(16, 185, 129, 0.35)",icon:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M5.5 3.5L1 8l4.5 4.5 1-1L3 8l3.5-3.5-1-1zm5 0l-1 1L13 8l-3.5 3.5 1 1L15 8l-4.5-4.5z"/>
    </svg>`},email:{accent:"#6366f1",glow:"rgba(99, 102, 241, 0.35)",icon:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M8 9.5c-.3 0-.6-.1-.8-.3L2 4.5V12h12V4.5l-5.2 4.7c-.2.2-.5.3-.8.3zM2 3h12l-6 5.4L2 3z"/>
    </svg>`},shopping:{accent:"#ec4899",glow:"rgba(236, 72, 153, 0.35)",icon:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M6 1l-1.5 4H1l3 2.5-1 4L6 9l2 1.5L10 9l3 3-1-4L15 5h-3.5L10 1H6zm0 1.5h4l1 2.5h2l-2 1.5.7 2.8L10 8l-2 1.5L6 8 4.3 9.3 5 6.5 3 5h2L6 2.5z"/>
    </svg>`}},Ee={general:["ask","summarize","explain","translate","save","copy"],article:["ask","summarize","highlight","translate","save","explain","copy"],media:["ask","describe","transcript","summarize","translate","save"],code:["ask","explain","review","debug","refactor","copy","save"],email:["ask","summarize","reply","translate","save","action-items"],shopping:["ask","compare","summarize","pros-cons","save","price-history"]},U={morphDuration:350,morphEasing:"cubic-bezier(0.4, 0, 0.0, 1)",flowDuration:500,glowPulse:3e3,sequentialStagger:40,dismissDelay:400,crossFadeDuration:400,selectionDelay:300,imageHoverDelay:800},z={bg:"rgba(10, 8, 24, 0.88)",bgHeavy:"rgba(10, 8, 24, 0.75)",backdrop:"blur(20px) saturate(1.5)",borderOpacity:.25,shadowBase:"0 8px 32px rgba(0,0,0,0.4)"},K={height:28,iconSize:15,imageIconSize:16,imageBarHeight:32,maxActionsPerRow:7},J={maxWidth:520,maxHeight:480,previewMaxLines:6,previewMaxChars:2e3,imagePreviewMaxHeight:200},V={pillPadding:"6px 10px",glowIntensityMin:.15,glowIntensityMax:.35,logoSize:20,expandDuration:220,dragThreshold:4,edgeMargin:12},W="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",Z=2147483647,Ie=2147483645;function nt(){const e=ve.general;return{type:"general",cadence:"engaged",suppressGhostBars:!1,readingProgress:0,actions:Ee.general,accent:e.accent,glow:e.glow,icon:e.icon,sessionActions:[]}}function ot(){let e=nt();const t=new Set;return{get(){return e},subscribe(n){return t.add(n),()=>t.delete(n)},update(n){e={...e,...n};for(const o of t)o(e)}}}function rt(e,t){var d;let n="";try{n=new URL(e).hostname.replace(/^www\./,"")}catch{}if(n==="github.com"||n==="gitlab.com")return"code";if(n==="youtube.com"||n==="netflix.com")return"media";if(n==="mail.google.com"||n==="outlook.live.com")return"email";if(n==="amazon.com"||n==="ebay.com"||n==="etsy.com"||e.includes("/product/")||e.includes("/cart/"))return"shopping";if(t.querySelector('article, [role="article"]'))return"article";if(t.querySelectorAll("pre, code").length>=3)return"code";if(t.querySelector("video, audio"))return"media";const p=t.querySelectorAll('script[type="application/ld+json"]');for(const l of p)try{const u=JSON.parse(l.textContent??""),k=(d=Array.isArray(u)?u[0]:u)==null?void 0:d["@type"];if(typeof k=="string"&&k.toLowerCase().includes("product"))return"shopping"}catch{}return"general"}function it(){const e=[],t=[];let l="engaged",u=0;function k(h){const c=h-1e4,g=h-3e4;for(;e.length&&e[0].ts<c;)e.shift();for(;t.length&&t[0]<g;)t.shift()}function b(h){return t.length>=3?"active":e.length>=3&&e.filter(g=>g.velocity>=300).length/e.length>=.6?"passive":"engaged"}function m(h){if(h-u<3e3)return;const c=b();c!==l&&(l=c,u=h)}return{getCadence(){return l},recordScroll(h){const c=Date.now();e.push({ts:c,velocity:Math.abs(h)}),k(c),m(c)},recordSelection(){const h=Date.now();t.push(h),k(h),b()==="active"?(l="active",u=h):m(h)},recordInput(){const h=Date.now();t.push(h),k(h),b()==="active"?(l="active",u=h):m(h)}}}function at(){let e=0;const t=new Set;let n=0;return{recordAction(o,r){t.add(r)},recordDismissal(){e++},getExtraDelay(){return Math.min(e*200,2e3)},shouldPromoteContinue(o){return t.has(o)},getSessionActions(){const o=[];return n>=2&&o.push("review-highlights"),o},recordHighlight(){n++}}}function st(e,t){const n=[],o=it(),r=at();function i(){const f=location.href,N=rt(f,document),L=ve[N];e.update({type:N,accent:L.accent,glow:L.glow,icon:L.icon,actions:[...Ee[N],...r.getSessionActions()],sessionActions:r.getSessionActions()})}const p=f=>{if(typeof requestIdleCallback<"u"){const N=requestIdleCallback(f,{timeout:2e3});n.push(()=>cancelIdleCallback(N))}else{const N=setTimeout(f,200);n.push(()=>clearTimeout(N))}};p(i);const d=()=>p(i);window.addEventListener("popstate",d),n.push(()=>window.removeEventListener("popstate",d));const l=history.pushState.bind(history),u=history.replaceState.bind(history);history.pushState=function(...f){l(...f),d()},history.replaceState=function(...f){u(...f),d()},n.push(()=>{history.pushState=l,history.replaceState=u});let k=null;const b=new MutationObserver(()=>{k&&clearTimeout(k),k=setTimeout(()=>p(i),2e3)});b.observe(document.body,{childList:!0,subtree:!0}),n.push(()=>{b.disconnect(),k&&clearTimeout(k)});let m=window.scrollY,h=Date.now();const c=()=>{const f=Date.now(),N=Math.max(f-h,1),M=Math.abs(window.scrollY-m)/N*1e3;o.recordScroll(M),m=window.scrollY,h=f,e.update({cadence:o.getCadence()})};window.addEventListener("scroll",c,{passive:!0}),n.push(()=>window.removeEventListener("scroll",c));const g=f=>{f.target.matches("input, textarea, [contenteditable]")&&(o.recordInput(),e.update({suppressGhostBars:!0,cadence:o.getCadence()}))},S=f=>{f.target.matches("input, textarea, [contenteditable]")&&e.update({suppressGhostBars:!1})};document.addEventListener("focusin",g),document.addEventListener("focusout",S),n.push(()=>{document.removeEventListener("focusin",g),document.removeEventListener("focusout",S)});const T=()=>{const f=window.getSelection();f&&f.toString().length>0&&(o.recordSelection(),e.update({cadence:o.getCadence()}))};document.addEventListener("selectionchange",T),n.push(()=>document.removeEventListener("selectionchange",T));const A=document.querySelector("article")??document.querySelector("main")??document.querySelector('[role="main"]');if(A){let f=0,N=0;p(()=>{const E=A.getBoundingClientRect();f=E.top+window.scrollY,N=E.bottom+window.scrollY});const M=()=>{const E=window.scrollY+window.innerHeight,a=N-f;if(a<=0)return;const s=Math.min(Math.max((E-f)/a,0),1);e.update({readingProgress:s})};window.addEventListener("scroll",M,{passive:!0}),n.push(()=>window.removeEventListener("scroll",M))}try{const f=t.storage.session;if(f){f.get(["contextType"],M=>{if(M!=null&&M.contextType&&e.get().type==="general"){const E=M.contextType,a=ve[E];e.update({type:E,accent:a.accent,glow:a.glow,icon:a.icon,actions:Ee[E]})}});let N=null;const L=e.subscribe(M=>{M.type!==N&&(N=M.type,f.set({contextType:M.type}))});n.push(L)}}catch{}return()=>{for(const f of n)f()}}function lt(){return`
/* ── Host / Root ─────────────────────────────────────────────────────────── */
:host {
  --aura-accent: #7c3aed;
  --aura-glow: rgba(124, 58, 237, 0.3);
  all: initial;
  pointer-events: none;
  font-family: ${W};
}

/* ── FAB ──────────────────────────────────────────────────────────────────── */
.aura-fab {
  position: fixed;
  right: 0;
  bottom: 30px;
  z-index: ${Z};
  transform: translateX(100%);
  transition: transform ${U.morphDuration}ms ${U.morphEasing};
  pointer-events: none;
  will-change: transform, opacity;
}

.aura-fab.show {
  transform: translateX(0);
}

.aura-fab.left {
  left: 0;
  right: auto;
  transform: translateX(-100%);
}

.aura-fab.left.show {
  transform: translateX(0);
}

.fab-pill {
  position: relative;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: ${V.pillPadding};
  background: ${z.bg};
  backdrop-filter: ${z.backdrop};
  -webkit-backdrop-filter: ${z.backdrop};
  border: 1px solid rgba(255, 255, 255, ${z.borderOpacity});
  border-radius: 20px;
  box-shadow: ${z.shadowBase};
  cursor: pointer;
  pointer-events: auto;
  transition: padding ${V.expandDuration}ms ${U.morphEasing},
              border-radius ${V.expandDuration}ms ${U.morphEasing};
  will-change: transform, opacity;
  font-family: ${W};
  color: #fff;
  user-select: none;
}

.fab-pill.hover {
  padding: ${V.pillPadding};
}

.fab-pill.dragging {
  border-radius: 50%;
  cursor: move;
  padding: ${V.pillPadding};
}

.fab-glow {
  position: absolute;
  inset: -6px;
  border-radius: inherit;
  background: var(--aura-glow);
  filter: blur(10px);
  animation: aura-glow-pulse ${U.glowPulse}ms ease-in-out infinite;
  pointer-events: none;
  z-index: -1;
  will-change: transform, opacity;
}

.fab-logo {
  width: ${V.logoSize}px;
  height: ${V.logoSize}px;
  pointer-events: none;
  flex-shrink: 0;
}

.fab-close {
  position: absolute;
  bottom: -18px;
  left: 50%;
  transform: translateX(-50%);
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.2s ease;
  pointer-events: auto;
  font-size: 9px;
  color: rgba(255, 255, 255, 0.7);
}

.fab-pill:hover .fab-close,
.aura-fab:hover .fab-close {
  opacity: 1;
}

.fab-popout {
  position: absolute;
  bottom: calc(100% + 8px);
  right: 0;
  min-width: 160px;
  background: ${z.bg};
  backdrop-filter: ${z.backdrop};
  -webkit-backdrop-filter: ${z.backdrop};
  border: 1px solid rgba(255, 255, 255, ${z.borderOpacity});
  border-radius: 12px;
  box-shadow: ${z.shadowBase};
  padding: 6px;
  opacity: 0;
  pointer-events: none;
  transform: translateY(4px);
  transition: opacity 0.2s ease, transform 0.2s ease;
  will-change: transform, opacity;
}

.fab-popout.visible {
  opacity: 1;
  pointer-events: auto;
  transform: translateY(0);
}

.aura-fab.left .fab-popout {
  right: auto;
  left: 0;
}

.fab-action-btn {
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: transparent;
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: rgba(255, 255, 255, 0.8);
  transition: background 0.15s ease, color 0.15s ease;
  pointer-events: auto;
  font-family: ${W};
}

.fab-action-btn:hover {
  background: var(--aura-accent);
  color: #fff;
}

/* ── Ghost Bar ────────────────────────────────────────────────────────────── */
.ghost-bar {
  position: fixed;
  z-index: ${Ie};
  display: flex;
  align-items: center;
  background: ${z.bg};
  backdrop-filter: ${z.backdrop};
  -webkit-backdrop-filter: ${z.backdrop};
  border: 1px solid rgba(255, 255, 255, ${z.borderOpacity});
  box-shadow: ${z.shadowBase};
  pointer-events: auto;
  will-change: transform, opacity;
  font-family: ${W};
  color: #fff;
}

.ghost-bar-text {
  height: ${K.height}px;
  padding: 0 8px;
  border-radius: 0 0 8px 8px;
  display: flex;
  align-items: center;
  gap: 2px;
}

.ghost-bar-image {
  height: ${K.imageBarHeight}px;
  padding: 0 10px;
  background: ${z.bgHeavy};
  border-radius: 6px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.gb-action {
  width: ${K.iconSize}px;
  height: ${K.iconSize}px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: rgba(255, 255, 255, 0.75);
  border-radius: 4px;
  transition: color 0.15s ease, background 0.15s ease;
  pointer-events: auto;
  background: transparent;
  border: none;
  padding: 0;
}

.gb-action:hover {
  color: var(--aura-accent);
  background: rgba(255, 255, 255, 0.08);
}

.gb-separator {
  width: 1px;
  height: 12px;
  background: rgba(255, 255, 255, 0.2);
  flex-shrink: 0;
  margin: 0 2px;
}

.gb-extended {
  display: none;
  align-items: center;
  gap: 2px;
  padding-top: 4px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  margin-top: 4px;
}

.gb-extended.visible {
  display: flex;
}

/* ── Modal ────────────────────────────────────────────────────────────────── */
.aura-modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.3);
  z-index: ${Ie};
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: auto;
}

.aura-modal {
  max-width: ${J.maxWidth}px;
  max-height: ${J.maxHeight}px;
  width: 90vw;
  background: ${z.bg};
  backdrop-filter: ${z.backdrop};
  -webkit-backdrop-filter: ${z.backdrop};
  border: 1px solid rgba(255, 255, 255, ${z.borderOpacity});
  border-radius: 16px;
  box-shadow: ${z.shadowBase};
  overflow: hidden;
  display: flex;
  flex-direction: column;
  will-change: transform, opacity;
  font-family: ${W};
  color: #fff;
}

.modal-preview {
  padding: 12px 16px;
  max-height: calc(${J.previewMaxLines} * 1.5em + 24px);
  overflow-y: auto;
  font-size: 13px;
  line-height: 1.5;
  color: rgba(255, 255, 255, 0.75);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.modal-input {
  width: 100%;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 8px;
  padding: 10px 12px;
  color: #fff;
  font-family: ${W};
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s ease;
  box-sizing: border-box;
}

.modal-input:focus {
  border-color: var(--aura-accent);
}

.modal-action-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 16px;
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  background: transparent;
  color: rgba(255, 255, 255, 0.85);
  font-family: ${W};
  font-size: 13px;
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}

.modal-action-btn:hover {
  background: var(--aura-accent);
  border-color: var(--aura-accent);
  color: #fff;
}

.modal-model-select {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 8px;
  padding: 6px 10px;
  color: #fff;
  font-family: ${W};
  font-size: 13px;
  outline: none;
  cursor: pointer;
  transition: border-color 0.2s ease;
}

.modal-model-select:focus {
  border-color: var(--aura-accent);
}

/* ── Highlights ───────────────────────────────────────────────────────────── */
.hl-tooltip {
  position: fixed;
  z-index: ${Z};
  background: ${z.bg};
  backdrop-filter: ${z.backdrop};
  -webkit-backdrop-filter: ${z.backdrop};
  border: 1px solid rgba(255, 255, 255, ${z.borderOpacity});
  border-radius: 8px;
  box-shadow: ${z.shadowBase};
  padding: 6px 10px;
  font-family: ${W};
  font-size: 12px;
  color: rgba(255, 255, 255, 0.9);
  pointer-events: none;
  will-change: transform, opacity;
}

/* ── Toast ────────────────────────────────────────────────────────────────── */
.aura-toast {
  position: fixed;
  bottom: 80px;
  left: 50%;
  transform: translateX(-50%);
  z-index: ${Z};
  background: rgba(16, 185, 129, 0.9);
  backdrop-filter: ${z.backdrop};
  -webkit-backdrop-filter: ${z.backdrop};
  border: 1px solid rgba(255, 255, 255, ${z.borderOpacity});
  border-radius: 20px;
  box-shadow: ${z.shadowBase};
  padding: 8px 18px;
  font-family: ${W};
  font-size: 13px;
  color: #fff;
  pointer-events: none;
  will-change: transform, opacity;
}

/* ── Keyframes ────────────────────────────────────────────────────────────── */
@keyframes aura-glow-pulse {
  0%   { opacity: ${V.glowIntensityMin}; }
  50%  { opacity: ${V.glowIntensityMax}; }
  100% { opacity: ${V.glowIntensityMin}; }
}
`.trim()}const se="forwards";function re(e){try{e.commitStyles()}catch{}e.cancel()}async function Ce(e,t){const n=e.offsetHeight||0,o=[{height:"0px",opacity:0},{height:`${n}px`,opacity:1}],r=t.direction==="down"?o:[...o].reverse(),i=e.animate(r,{duration:t.duration,easing:t.easing,delay:t.delay??0,fill:se});await i.finished,re(i)}async function ct(e,t){const n=e.animate([{opacity:1},{opacity:0}],{duration:t.duration,easing:t.easing,delay:t.delay??0,fill:se});await n.finished,re(n)}async function Oe(e,t){const n=e.animate([{opacity:0},{opacity:1}],{duration:t.duration,easing:t.easing,delay:t.delay??0,fill:se});await n.finished,re(n)}async function dt(e,t,n){const o={duration:n.duration,easing:n.easing,delay:n.delay??0,fill:se},r=e.animate([{opacity:1},{opacity:0}],o),i=t.animate([{opacity:0},{opacity:1}],o);await Promise.all([r.finished,i.finished]),re(r),re(i)}async function $e(e,t,n,o){const r=e.animate([{width:`${t.width}px`,height:`${t.height}px`,transform:`translate(${t.left}px, ${t.top}px)`},{width:`${n.width}px`,height:`${n.height}px`,transform:`translate(${n.left}px, ${n.top}px)`}],{duration:o.duration,easing:o.easing,delay:o.delay??0,fill:se});await r.finished,re(r)}const ut=[{action:"chat",tip:"Chat",svg:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M2 2h12a1 1 0 011 1v8a1 1 0 01-1 1H5l-3 3V3a1 1 0 011-1z"/>
    </svg>`},{action:"search",tip:"Search",svg:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M6.5 1a5.5 5.5 0 014.23 9.02l3.12 3.12-1.06 1.06-3.12-3.12A5.5 5.5 0 116.5 1zm0 1.5a4 4 0 100 8 4 4 0 000-8z"/>
    </svg>`},{action:"page",tip:"This Page",svg:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M3 2h10a1 1 0 011 1v10a1 1 0 01-1 1H3a1 1 0 01-1-1V3a1 1 0 011-1zm1 3v1h8V5H4zm0 3v1h8V8H4zm0 3v1h5v-1H4z"/>
    </svg>`},{action:"translate",tip:"Translate",svg:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M1 2h7v1.5H5.5v1H8v1.5H5.5c-.2 1-.7 2-1.5 2.7l1.7 1.8-1.1 1-1.6-1.8C2.5 10 2 10.2 1.5 10.3L1 8.8c.5-.1.9-.3 1.3-.5L1 6.8l1.1-1 1.2 1.4c.5-.5.9-1.1 1.1-1.7H1V2zm10 3l3 8h-1.5l-.6-1.7h-2.8L8.5 13H7l3-8h1zm-.5 2.5l-1 2.8h2l-1-2.8z"/>
    </svg>`},{action:"save",tip:"Save to Memory",svg:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M3 1h10a1 1 0 011 1v13l-6-3-6 3V2a1 1 0 011-1z"/>
    </svg>`}];function pt(){let e=null,t=null,n=null,o=null,r=null,i=null,p=null,d=null,l="right",u=40,k=0,b=0,m=!1,h=0,c=null,g=!1;function S(){const v=document.createElement("div");return v.className="fab-glow",Object.assign(v.style,{position:"absolute",inset:"-8px",borderRadius:"50px",background:"var(--aura-glow)",filter:"blur(12px)",animation:"aura-glow-pulse 3s ease-in-out infinite",pointerEvents:"none",zIndex:"-1"}),v}function T(v){const w=document.createElement("div");w.className="fab-logo",Object.assign(w.style,{width:`${V.logoSize}px`,height:`${V.logoSize}px`,color:"var(--aura-accent)",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:"0",transition:"color 0.3s ease"}),w.innerHTML=v;const O=w.querySelector("svg");return O&&(O.style.width="100%",O.style.height="100%"),w}function A(){const v=document.createElement("button");return v.className="fab-close",v.setAttribute("aria-label","Close Aura"),v.innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 12" fill="currentColor" width="10" height="10">
      <path d="M1 1l10 10M11 1L1 11" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    </svg>`,Object.assign(v.style,{position:"absolute",top:"-6px",right:"-6px",width:"16px",height:"16px",borderRadius:"50%",background:"rgba(10,8,24,0.9)",border:"1px solid rgba(255,255,255,0.15)",color:"rgba(255,255,255,0.6)",cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center",padding:"0",opacity:"0",transition:"opacity 0.2s",pointerEvents:"all"}),v}function f(v){const w=document.createElement("button");w.className="fab-action-btn",w.dataset.action=v.action,w.setAttribute("aria-label",v.tip),w.innerHTML=v.svg,Object.assign(w.style,{display:"flex",flexDirection:"column",alignItems:"center",gap:"4px",background:"transparent",border:"none",color:"rgba(255,255,255,0.75)",cursor:"pointer",padding:"6px 8px",borderRadius:"8px",fontSize:"10px",fontFamily:W,transition:"background 0.15s, color 0.15s"});const O=w.querySelector("svg");O&&(O.setAttribute("width","16"),O.setAttribute("height","16"));const I=document.createElement("span");return I.textContent=v.tip,I.style.fontSize="10px",w.appendChild(I),w.addEventListener("mouseenter",()=>{w.style.background="rgba(255,255,255,0.08)",w.style.color="var(--aura-accent)"}),w.addEventListener("mouseleave",()=>{w.style.background="transparent",w.style.color="rgba(255,255,255,0.75)"}),w}function N(){const v=document.createElement("div");v.className="fab-popout hidden",Object.assign(v.style,{position:"absolute",display:"flex",flexDirection:"row",gap:"4px",padding:"8px",background:"rgba(10,8,24,0.92)",backdropFilter:"blur(20px) saturate(1.5)",border:"1px solid rgba(255,255,255,0.12)",borderRadius:"14px",boxShadow:"0 8px 32px rgba(0,0,0,0.4)",zIndex:String(Z),transition:"opacity 0.2s, transform 0.2s",opacity:"0",pointerEvents:"none"});for(const w of ut)v.appendChild(f(w));return v}function L(v){const w=document.createElement("div");return w.className="fab-pill",Object.assign(w.style,{display:"flex",alignItems:"center",justifyContent:"center",padding:V.pillPadding,background:"rgba(10,8,24,0.88)",backdropFilter:"blur(20px) saturate(1.5)",border:"1px solid rgba(255,255,255,0.12)",borderRadius:"50px",cursor:"pointer",position:"relative",boxShadow:"0 4px 20px rgba(0,0,0,0.3)",transition:`padding ${V.expandDuration}ms ease, border-radius ${V.expandDuration}ms ease`,userSelect:"none",touchAction:"none"}),n=S(),w.appendChild(n),o=T(v.icon),w.appendChild(o),i=A(),w.appendChild(i),w}function M(){if(!e||!t)return;const v=V.edgeMargin;Object.assign(e.style,{position:"fixed",top:`${u}%`,[l==="right"?"right":"left"]:`${v}px`,[l==="right"?"left":"right"]:"auto",zIndex:String(Z),transform:""}),E()}function E(){if(!r||!e)return;const v=l==="right";Object.assign(r.style,{top:"50%",transform:"translateY(-50%)",[v?"right":"left"]:"calc(100% + 8px)",[v?"left":"right"]:"auto"})}function a(){!r||!t||(c&&(clearTimeout(c),c=null),r.classList.remove("hidden"),r.style.opacity="1",r.style.pointerEvents="all",t.style.borderBottomRightRadius="50px",i&&(i.style.opacity="1"))}function s(){!r||!t||(r.style.opacity="0",r.style.pointerEvents="none",c=setTimeout(()=>{r.classList.add("hidden"),i&&(i.style.opacity="0")},200))}function C(v,w){let O=!1,I=!1;function P(){setTimeout(()=>{!O&&!I&&s()},0)}v.addEventListener("mouseenter",()=>{O=!0,a()}),v.addEventListener("mouseleave",()=>{O=!1,P()}),w.addEventListener("mouseenter",()=>{I=!0,a()}),w.addEventListener("mouseleave",()=>{I=!1,P()})}function x(v){v.addEventListener("pointerdown",w=>{w.target.closest(".fab-close")||(k=w.clientX,b=w.clientY,m=!1,h=0,v.setPointerCapture(w.pointerId))}),v.addEventListener("pointermove",w=>{if(!v.hasPointerCapture(w.pointerId))return;const O=w.clientX-k,I=w.clientY-b;if(h=Math.sqrt(O*O+I*I),h>V.dragThreshold){m=!0,v.classList.add("dragging"),v.style.borderRadius="50%";const P=window.innerHeight,D=w.clientY,j=Math.min(Math.max(D/P*100,5),90);e&&(e.style.top=`${j}%`)}}),v.addEventListener("pointerup",w=>{if(!m)return;v.classList.remove("dragging"),v.style.borderRadius="50px";const O=window.innerWidth;l=w.clientX>O/2?"right":"left";const I=window.innerHeight;u=Math.min(Math.max(w.clientY/I*100,5),90),m=!1,M(),B()})}function y(v,w){v.addEventListener("click",O=>{O.target.closest(".fab-close")||m||h>V.dragThreshold||d&&d.runtime.sendMessage({type:"OPEN_PANEL",panel:"chat"})}),i==null||i.addEventListener("click",O=>{O.stopPropagation(),e&&(e.style.display="none")}),w.addEventListener("click",O=>{const I=O.target.closest(".fab-action-btn");if(!I||!d)return;const P=I.dataset.action??"";$(P)})}function _(){const v=window.getSelection();return v?v.toString().trim():""}function $(v){if(!d)return;const w=location.href,O=document.title;switch(v){case"chat":d.runtime.sendMessage({type:"OPEN_PANEL",panel:"chat"});break;case"search":d.runtime.sendMessage({type:"OPEN_PANEL",panel:"search"});break;case"page":d.runtime.sendMessage({type:"OPEN_PANEL",panel:"ask"});break;case"translate":d.runtime.sendMessage({type:"OPEN_PANEL",panel:"translate"});break;case"save":{const P=_()||`${O}
${w}`;d.runtime.sendMessage({type:"SAVE_KNOWLEDGE",text:P,url:w,title:O},D=>{});break}}}function R(v){var w;if(!(!e||!o)&&(e.style.setProperty("--aura-accent",v.accent),e.style.setProperty("--aura-glow",v.glow),!g&&o)){g=!0;const O=o,I=T(v.icon);I.style.position="absolute",I.style.inset="0",I.style.opacity="0",(w=O.parentElement)==null||w.appendChild(I),dt(O,I,{duration:U.crossFadeDuration,easing:"ease"}).then(()=>{O.remove(),I.style.position="",I.style.inset="",I.style.opacity="1",o=I,g=!1})}}function H(){d&&d.storage.local.get(["auraFabSide","auraFabOffset"],v=>{(v.auraFabSide==="left"||v.auraFabSide==="right")&&(l=v.auraFabSide),typeof v.auraFabOffset=="number"&&(u=v.auraFabOffset),M()})}function B(){d&&d.storage.local.set({auraFabSide:l,auraFabOffset:u})}return{init(v,w,O){d=O;const I=w.get(),P=document.createElement("div");P.className="aura-fab",Object.assign(P.style,{position:"fixed",zIndex:String(Z),fontFamily:W,"--aura-accent":I.accent,"--aura-glow":I.glow}),e=P;const D=L(I);t=D,P.appendChild(D);const j=N();r=j,P.appendChild(j),v.appendChild(P),C(D,j),x(D),y(D,j),H(),p=w.subscribe(R)},destroy(){p&&(p(),p=null),c&&(clearTimeout(c),c=null),e==null||e.remove(),e=null,t=null,n=null,o=null,r=null,i=null,d=null},showDock(){e&&(e.style.display="")}}}const He={ask:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M8 1L10 6H15L11 9.5L12.5 14.5L8 11.5L3.5 14.5L5 9.5L1 6H6L8 1Z"/>
  </svg>`,copy:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M4 4h7a1 1 0 011 1v7a1 1 0 01-1 1H4a1 1 0 01-1-1V5a1 1 0 011-1zm0-2a3 3 0 00-3 3v7a3 3 0 003 3h7a3 3 0 003-3V5a3 3 0 00-3-3H4z"/>
    <path d="M7 1h5a3 3 0 013 3v5h-2V4a1 1 0 00-1-1H7V1z"/>
  </svg>`,explain:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M8 1l1.5 3 3.5.5-2.5 2.5.6 3.5L8 9 4.9 10.5l.6-3.5L3 4.5 6.5 4z"/>
    <path d="M2 13h12v1.5H2z" opacity=".5"/>
  </svg>`,summarize:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M2 3h12v2H2V3zm0 4h12v2H2V7zm0 4h8v2H2v-2z"/>
  </svg>`,translate:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.5"/>
    <path d="M8 1C5.5 4 5.5 12 8 15M8 1c2.5 3 2.5 11 0 14M1 8h14M2 5h12M2 11h12"/>
  </svg>`,highlight:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M5 1h6v10l-3 3-3-3V1zm1 1v9l2 2 2-2V2H6z"/>
  </svg>`,more:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <circle cx="3" cy="8" r="1.5"/>
    <circle cx="8" cy="8" r="1.5"/>
    <circle cx="13" cy="8" r="1.5"/>
  </svg>`,describe:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M2 2h12a1 1 0 011 1v10a1 1 0 01-1 1H2a1 1 0 01-1-1V3a1 1 0 011-1zm1 2v8h10V4H3zm2 2a1 1 0 110 2 1 1 0 010-2zm7 4H4l2-3 1.5 2 2-3L12 10z"/>
  </svg>`,edit:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M11.5 1.5l3 3-8 8H3.5v-3l8-8zM10 3L13 6l-7 7H4v-2L10 3z"/>
  </svg>`,save:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M8 10L4.5 6.5l1-1L7 7V1h2v6l1.5-1.5 1 1L8 10zm-5 2h10v2H3v-2z"/>
  </svg>`,rewrite:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M2 8a6 6 0 1110.76-3H10v2h5V2h-2v2.5A8 8 0 102 8h2z"/>
  </svg>`,grammar:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M1 2h14v2H1V2zm0 4h10v2H1V6zm0 4h14v2H1v-2zm0 4h6v2H1v-2z"/>
  </svg>`,define:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M7 1H2v14h12V6l-5-5H7zm0 1.5L11.5 7H7V2.5zM4 4h2v2H4V4zm0 3h8v2H4V7zm0 3h8v2H4v-2z"/>
  </svg>`,"read-aloud":`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M7 3v10L3.5 10H1V6h2.5L7 3zm2 2a4 4 0 010 6V9.5a2 2 0 000-3V5zm2-2a7 7 0 010 10V12.5a5 5 0 000-9V3z"/>
  </svg>`,review:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M8 2a6 6 0 100 12A6 6 0 008 2zm0 2a4 4 0 110 8A4 4 0 018 4zm0 2a2 2 0 100 4 2 2 0 000-4z"/>
  </svg>`,debug:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M8 1a4 4 0 00-4 4v1H2v2h2v1a4 4 0 004 4 4 4 0 004-4V8h2V6h-2V5a4 4 0 00-4-4zm0 2a2 2 0 012 2v6a2 2 0 01-4 0V5a2 2 0 012-2z"/>
  </svg>`,refactor:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M2 2h5v2H4v2h3v2H4v2h3v2H2V2zm7 0h5v12H9V2zm2 2v8h1V4h-1z"/>
  </svg>`};function gt(e){return He[e]??He.more}function ze(e,t,n,o,r){Object.assign(e.style,{position:"fixed",left:`${t.left}px`,top:`${n}px`,width:`${t.width}px`,height:`${o}px`,background:r,backdropFilter:z.backdrop,WebkitBackdropFilter:z.backdrop,border:`1px solid rgba(255,255,255,${z.borderOpacity})`,boxShadow:z.shadowBase,borderRadius:"6px",display:"flex",alignItems:"center",gap:"2px",padding:"0 6px",overflow:"hidden",boxSizing:"border-box",zIndex:String(Z),userSelect:"none"})}function me(e){const t=document.createElement("button");return t.className="gb-action",t.dataset.action=e,t.title=e,t.innerHTML=gt(e),Object.assign(t.style,{background:"none",border:"none",cursor:"pointer",color:"rgba(255,255,255,0.85)",padding:"3px",display:"flex",alignItems:"center",justifyContent:"center",borderRadius:"4px",flexShrink:"0"}),t}function mt(){let e=null,t=null,n="",o="",r=null,i=document.body,p=null,d=null,l=null,u=null,k=null,b=null,m=null;const h=[];function c(){e&&(e.remove(),e=null,t=null,r=null)}async function g(){if(!e)return;const s=e;e=null,t=null,r=null;try{await Ce(s,{direction:"up",duration:U.morphDuration,easing:U.morphEasing})}catch{}s.remove()}function S(s){var x;if(!d)return;if(s==="ask"){if(l){const y={type:t==="image"?"image":"text",text:n,imageUrl:o,rect:e?e.getBoundingClientRect():r??new DOMRect};l(y)}return}if(s==="copy"){(x=navigator.clipboard)==null||x.writeText(n).catch(()=>{});return}if(s==="highlight"){d.runtime.sendMessage({type:"SAVE_KNOWLEDGE",text:n,url:location.href,title:document.title});return}if(s==="describe"){d.runtime.sendMessage({type:"IMAGE_DESCRIBE",imageUrl:o});return}if(s==="edit"){d.runtime.sendMessage({type:"IMAGE_EDIT_OPEN",imageUrl:o});return}if(s==="save"){d.runtime.sendMessage({type:"IMAGE_SAVE",imageUrl:o});return}const C={type:"QUICK_ACTION",action:s,text:n};d.runtime.sendMessage(C)}function T(s){s.addEventListener("click",C=>{const x=C.target.closest(".gb-action");if(!x)return;const y=x.dataset.action??"";if(y==="more"){const _=s.querySelector(".gb-extended");_&&(_.style.display=_.style.display==="none"?"flex":"none");return}S(y)})}function A(s){const C=s.trim();return/^(https?:\/\/|www\.)\S+$/.test(C)?"url":/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(C)?"email":[/[{}\[\]();]/.test(C),/\b(function|const|let|var|def|class|import|return|if|for|while)\b/.test(C),/=>|->|::/.test(C),/^\s{2,}/m.test(C)].filter(Boolean).length>=2?"code":"text"}function f(s,C){if(!p)return;const x=p.get();if(x.suppressGhostBars)return;c(),n=C,o="",r=s,t="text";const y=document.createElement("div");y.className="ghost-bar ghost-bar-text";const _=s.bottom;ze(y,s,_,K.height,z.bg);const $=A(C),R={code:["explain","ask","copy"],url:["ask","summarize","copy"],email:["ask","copy"],text:[]},B=(R[$].length?R[$]:x.actions).slice(0,K.maxActionsPerRow-1);for(const I of B)y.appendChild(me(I));const v=me("more");y.appendChild(v);const w=document.createElement("div");w.className="gb-extended";const O=["rewrite","grammar","define","read-aloud"];for(const I of O)w.appendChild(me(I));Object.assign(w.style,{display:"none",position:"absolute",top:`${K.height}px`,left:"0",right:"0",background:z.bg,borderRadius:"0 0 6px 6px",padding:"2px 6px",gap:"2px"}),y.style.position="fixed",y.appendChild(w),T(y),i.appendChild(y),e=y,Ce(y,{direction:"down",duration:U.morphDuration,easing:U.morphEasing}).catch(()=>{})}function N(s){if(!p||p.get().suppressGhostBars)return;c();const x=s.getBoundingClientRect();n="",o=s.src??s.currentSrc??"",r=x,t="image";const y=document.createElement("div");y.className="ghost-bar ghost-bar-image";const _=x.bottom-K.imageBarHeight;ze(y,x,_,K.imageBarHeight,z.bgHeavy);const $=["describe","edit","save","ask"];for(const R of $){const H=me(R);H.style.width=`${K.imageIconSize+8}px`,H.style.height=`${K.imageIconSize+8}px`,y.appendChild(H)}T(y),i.appendChild(y),e=y,Ce(y,{direction:"down",duration:U.morphDuration,easing:U.morphEasing}).catch(()=>{})}function L(){return e?e.getBoundingClientRect():null}function M(s){l=s}function E(s,C,x){i=s,p=C,d=x;const y=()=>{u&&clearTimeout(u),u=setTimeout(()=>{const H=window.getSelection();if(!H||H.rangeCount===0||H.toString().trim().length===0){g().catch(()=>{});return}const v=H.getRangeAt(0).getBoundingClientRect();v.width===0&&v.height===0||f(v,H.toString())},U.selectionDelay)};document.addEventListener("selectionchange",y);const _=H=>{const B=H.target;if(B.tagName!=="IMG")return;const v=B,w=v.getBoundingClientRect();w.width<80||w.height<80||(b&&(clearTimeout(b),b=null),k&&clearTimeout(k),m=v,k=setTimeout(()=>{m===v&&N(v)},U.imageHoverDelay))},$=H=>{const B=H.target,v=H.relatedTarget,w=B.tagName==="IMG",O=e&&(B===e||e.contains(B));!w&&!O||v&&e&&(v===e||e.contains(v))||v&&v.tagName==="IMG"&&v===m||(k&&(clearTimeout(k),k=null),t==="image"&&(b&&clearTimeout(b),b=setTimeout(()=>{g().catch(()=>{})},U.dismissDelay)))};document.addEventListener("mouseover",_,!0),document.addEventListener("mouseout",$,!0),h.push(()=>document.removeEventListener("selectionchange",y),()=>document.removeEventListener("mouseover",_,!0),()=>document.removeEventListener("mouseout",$,!0));const R=()=>{if(!e||!r)return;const H=window.innerHeight,B=window.innerWidth;if(t==="text"){const v=window.getSelection();if(!v||v.rangeCount===0){g().catch(()=>{});return}const w=v.getRangeAt(0).getBoundingClientRect();if(w.bottom<0||w.top>H||w.right<0||w.left>B){g().catch(()=>{});return}e.style.top=`${w.bottom}px`,e.style.left=`${w.left}px`,e.style.width=`${w.width}px`,r=w}else if(t==="image"&&m){const v=m.getBoundingClientRect();if(v.bottom<0||v.top>H||v.right<0||v.left>B){g().catch(()=>{});return}const w=v.bottom-K.imageBarHeight;e.style.top=`${w}px`,e.style.left=`${v.left}px`,e.style.width=`${v.width}px`,r=v}};window.addEventListener("scroll",R,{passive:!0}),h.push(()=>window.removeEventListener("scroll",R))}function a(){u&&clearTimeout(u),k&&clearTimeout(k),b&&clearTimeout(b);for(const s of h)s();h.length=0,c()}return{init:E,destroy:a,showTextBar:f,showImageBar:N,hideBar:g,getBarRect:L,onAskClicked:M}}function ft(e){if(e.length<=J.previewMaxChars)return e;const t=e.length-J.previewMaxChars;return e.slice(0,J.previewMaxChars)+`... (${t} more chars)`}function ht(e){switch(e){case"article":return"Ask about this article...";case"code":return"Ask about this code...";default:return"Ask anything about this text..."}}function Re(){const e=Math.min(J.maxWidth,window.innerWidth-32),t=Math.min(J.maxHeight,window.innerHeight-32),n=(window.innerWidth-e)/2,o=(window.innerHeight-t)/2;return{left:n,top:o,right:n+e,bottom:o+t,width:e,height:t,x:n,y:o,toJSON:()=>({})}}function Be(e,t){Object.assign(e.style,{position:"fixed",left:"0",top:"0",width:`${t.width}px`,height:`${t.height}px`,transform:`translate(${t.left}px, ${t.top}px)`,background:z.bg,backdropFilter:z.backdrop,WebkitBackdropFilter:z.backdrop,border:`1px solid rgba(255,255,255,${z.borderOpacity})`,borderRadius:"16px",boxShadow:z.shadowBase,fontFamily:W,color:"#e5e7eb",overflow:"hidden",zIndex:String(Z),boxSizing:"border-box"})}function bt(e,t){const n=document.createElement("div");n.className="modal-content-wrap",Object.assign(n.style,{display:"flex",flexDirection:"column",gap:"12px",padding:"16px",height:"100%",boxSizing:"border-box",opacity:"0"});const o=document.createElement("div");o.className="modal-preview",Object.assign(o.style,{fontSize:"13px",lineHeight:"1.5",color:"rgba(229,231,235,0.75)",overflow:"hidden",display:"-webkit-box",WebkitLineClamp:String(J.previewMaxLines),WebkitBoxOrient:"vertical",maxHeight:`${J.previewMaxLines*20}px`,flexShrink:"0"}),o.textContent=ft(e);const r=document.createElement("input");r.type="text",r.className="modal-input",r.placeholder=t,Object.assign(r.style,{background:"rgba(255,255,255,0.07)",border:"1px solid rgba(255,255,255,0.15)",borderRadius:"8px",padding:"8px 12px",color:"#e5e7eb",fontSize:"14px",fontFamily:W,outline:"none",flexShrink:"0"});const i=document.createElement("div");i.className="modal-actions",Object.assign(i.style,{display:"flex",flexWrap:"wrap",gap:"6px",flexShrink:"0"});const p=[{label:"Explain",value:"explain"},{label:"Summarize",value:"summarize"},{label:"Chat with AURA",value:"chat"},{label:"Save to Memory",value:"save"},{label:"Translate",value:"translate"}];for(const b of p){const m=document.createElement("button");m.className="modal-action-btn",m.textContent=b.label,m.dataset.action=b.value,Object.assign(m.style,{background:"rgba(255,255,255,0.08)",border:"1px solid rgba(255,255,255,0.12)",borderRadius:"6px",padding:"5px 10px",color:"#e5e7eb",fontSize:"12px",fontFamily:W,cursor:"pointer"}),i.appendChild(m)}const d=document.createElement("div");d.className="modal-model-row",Object.assign(d.style,{display:"flex",alignItems:"center",gap:"8px",marginTop:"auto",flexShrink:"0"});const l=document.createElement("span");l.textContent="Model",Object.assign(l.style,{fontSize:"12px",color:"rgba(229,231,235,0.5)"});const u=document.createElement("select");u.className="modal-model-select",Object.assign(u.style,{background:"rgba(255,255,255,0.07)",border:"1px solid rgba(255,255,255,0.15)",borderRadius:"6px",padding:"4px 8px",color:"#e5e7eb",fontSize:"12px",fontFamily:W,cursor:"pointer"});const k=[{label:"Auto",value:"auto"},{label:"Fast",value:"fast"},{label:"Balanced",value:"balanced"},{label:"Powerful",value:"powerful"}];for(const b of k){const m=document.createElement("option");m.value=b.value,m.textContent=b.label,u.appendChild(m)}return d.appendChild(l),d.appendChild(u),n.appendChild(o),n.appendChild(r),n.appendChild(i),n.appendChild(d),n}function xt(e){const t=document.createElement("div");t.className="modal-content-wrap",Object.assign(t.style,{display:"flex",flexDirection:"column",gap:"12px",padding:"16px",height:"100%",boxSizing:"border-box",opacity:"0"});const n=document.createElement("div");n.className="modal-preview",Object.assign(n.style,{flexShrink:"0",overflow:"hidden",borderRadius:"8px"});const o=document.createElement("img");o.src=e,Object.assign(o.style,{maxWidth:"100%",maxHeight:`${J.imagePreviewMaxHeight}px`,objectFit:"contain",display:"block"}),n.appendChild(o);const r=document.createElement("div");r.className="modal-actions",Object.assign(r.style,{display:"flex",flexWrap:"wrap",gap:"6px",flexShrink:"0"});const i=[{label:"Describe",value:"describe"},{label:"Summarize",value:"summarize"},{label:"Chat with AURA",value:"chat"},{label:"Save to Memory",value:"save"},{label:"Translate",value:"translate"}];for(const k of i){const b=document.createElement("button");b.className="modal-action-btn",b.textContent=k.label,b.dataset.action=k.value,Object.assign(b.style,{background:"rgba(255,255,255,0.08)",border:"1px solid rgba(255,255,255,0.12)",borderRadius:"6px",padding:"5px 10px",color:"#e5e7eb",fontSize:"12px",fontFamily:W,cursor:"pointer"}),r.appendChild(b)}const p=document.createElement("input");p.type="text",p.className="modal-input",p.placeholder="Ask about this image...",Object.assign(p.style,{background:"rgba(255,255,255,0.07)",border:"1px solid rgba(255,255,255,0.15)",borderRadius:"8px",padding:"8px 12px",color:"#e5e7eb",fontSize:"14px",fontFamily:W,outline:"none",flexShrink:"0"});const d=document.createElement("div");d.className="modal-model-row",Object.assign(d.style,{display:"flex",alignItems:"center",gap:"8px",marginTop:"auto",flexShrink:"0"});const l=document.createElement("span");l.textContent="Model",Object.assign(l.style,{fontSize:"12px",color:"rgba(229,231,235,0.5)"});const u=document.createElement("select");u.className="modal-model-select",Object.assign(u.style,{background:"rgba(255,255,255,0.07)",border:"1px solid rgba(255,255,255,0.15)",borderRadius:"6px",padding:"4px 8px",color:"#e5e7eb",fontSize:"12px",fontFamily:W,cursor:"pointer"});for(const k of[{label:"Auto",value:"auto"},{label:"Fast",value:"fast"},{label:"Balanced",value:"balanced"},{label:"Powerful",value:"powerful"}]){const b=document.createElement("option");b.value=k.value,b.textContent=k.label,u.appendChild(b)}return d.appendChild(l),d.appendChild(u),t.appendChild(n),t.appendChild(r),t.appendChild(p),t.appendChild(d),t}function yt(){let e={overlay:null,modal:null,originRect:null,content:"",isOpen:!1,closing:!1,opening:!1},t=null,n=null;const o=p=>{p.key==="Escape"&&e.isOpen&&i()};async function r(p,d,l){if(e.opening)return;e.isOpen&&await i(),e.opening=!0,e.originRect=d,e.content=l,e.isOpen=!0;const u=document.createElement("div");u.className="aura-modal-overlay",Object.assign(u.style,{position:"fixed",inset:"0",background:"rgba(0,0,0,0.3)",zIndex:String(Z-1),opacity:"0"}),document.body.appendChild(u),e.overlay=u;const k=document.createElement("div");k.className="aura-modal",Be(k,d),document.body.appendChild(k),e.modal=k,Oe(u,{duration:U.flowDuration,easing:"ease-out"}).then(()=>{u.style.opacity="1"});const b=Re();await $e(k,d,b,{duration:U.morphDuration,easing:U.morphEasing}),Be(k,b),k.appendChild(p),Oe(p,{duration:U.crossFadeDuration,easing:"ease-out"}).then(()=>{p.style.opacity="1"}),k.querySelectorAll(".modal-action-btn").forEach(h=>{h.addEventListener("click",()=>{const c=h.dataset.action??"ask",g=k.querySelector(".modal-model-select"),S=(g==null?void 0:g.value)??"auto";n==null||n(c,e.content,S)})});const m=k.querySelector(".modal-input");m&&m.addEventListener("keydown",h=>{if(h.key==="Enter"){const c=k.querySelector(".modal-model-select"),g=(c==null?void 0:c.value)??"auto";n==null||n("ask",m.value,g)}}),u.addEventListener("click",()=>i()),document.addEventListener("keydown",o),e.opening=!1}async function i(){if(!e.isOpen||e.closing)return;e.closing=!0;const{modal:p,overlay:d,originRect:l}=e;document.removeEventListener("keydown",o);const u=[];if(p&&l){const k=Re();u.push($e(p,k,l,{duration:U.morphDuration,easing:U.morphEasing}).catch(()=>{}))}d&&u.push(ct(d,{duration:U.morphDuration,easing:"ease-in"}).catch(()=>{})),await Promise.all(u),p==null||p.remove(),d==null||d.remove(),e={overlay:null,modal:null,originRect:null,content:"",isOpen:!1,closing:!1,opening:!1}}return{init(p,d,l){t=d},destroy(){i()},openWithText(p,d){const l=ht((t==null?void 0:t.get().type)??"general"),u=bt(p,l);r(u,d,p)},openWithImage(p,d){const l=xt(p);r(l,d,p)},close:i,onAction(p){n=p}}}function wt(){let e,t=()=>{};function n(E,a){try{a?e.runtime.sendMessage(E,a):e.runtime.sendMessage(E)}catch{}}const o=document.createElement("div");o.id="aura-highlight-host",Object.assign(o.style,{position:"fixed",top:"0",left:"0",zIndex:"2147483646",pointerEvents:"none"}),document.documentElement.appendChild(o);const r=o.attachShadow({mode:"closed"}),i=document.createElement("style");i.textContent=`
    @keyframes hl-tooltip-in {
      from { opacity: 0; transform: translateY(4px) scale(0.95); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    .hl-tooltip {
      position: fixed;
      background: rgba(10, 8, 24, 0.92);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(124, 58, 237, 0.3);
      border-radius: 8px;
      padding: 5px 10px;
      display: flex;
      align-items: center;
      gap: 8px;
      pointer-events: auto;
      animation: hl-tooltip-in 0.15s ease forwards;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
      z-index: 2147483647;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
    }
    .hl-tooltip-text {
      color: rgba(226, 232, 240, 0.9);
      font-size: 11px;
      font-weight: 500;
      white-space: nowrap;
    }
    .hl-tooltip-delete {
      width: 18px; height: 18px; border-radius: 4px;
      background: transparent; border: none;
      color: rgba(226, 232, 240, 0.5);
      cursor: pointer; display: flex; align-items: center; justify-content: center;
      padding: 0; transition: background 0.12s, color 0.12s;
    }
    .hl-tooltip-delete:hover {
      background: rgba(239, 68, 68, 0.25);
      color: rgba(239, 68, 68, 1);
    }
  `,r.appendChild(i);const p=document.createElement("div");r.appendChild(p);const d=document.createElement("style");d.textContent=`
    mark[data-aura-hl] {
      background: rgba(124, 58, 237, 0.15);
      border-bottom: 2px solid rgba(124, 58, 237, 0.5);
      border-radius: 2px;
      cursor: pointer;
      transition: background 0.15s ease;
    }
    mark[data-aura-hl]:hover {
      background: rgba(124, 58, 237, 0.28);
    }
    mark[data-aura-hl].aura-hl-stale {
      background: rgba(124, 58, 237, 0.08);
      border-bottom: 2px dashed rgba(124, 58, 237, 0.35);
    }
    mark[data-aura-hl].aura-hl-flash {
      background: rgba(124, 58, 237, 0.45) !important;
      transition: background 0.3s ease;
    }
  `,document.head.appendChild(d);let l=null,u=null;function k(){u&&(clearTimeout(u),u=null),l&&(l.remove(),l=null)}function b(E,a){k();const s=E.getBoundingClientRect();l=document.createElement("div"),l.className="hl-tooltip";const C=document.createElement("span");C.className="hl-tooltip-text",C.textContent="Saved to AURA",l.appendChild(C);const x=document.createElement("button");x.className="hl-tooltip-delete",x.title="Remove highlight",x.innerHTML='<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',x.addEventListener("click",y=>{y.stopPropagation(),A(a),k()}),l.appendChild(x),l.style.top=`${Math.round(s.top-34)}px`,l.style.left=`${Math.round(s.left+s.width/2-60)}px`,p.appendChild(l)}function m(E){if(E.nodeType===Node.DOCUMENT_NODE)return"/";const a=[];let s=E;for(;s&&s!==document;){if(s.nodeType===Node.ELEMENT_NODE){const C=s;let x=C.tagName.toLowerCase();const y=C.parentNode;if(y){const _=Array.from(y.childNodes).filter($=>$.nodeType===Node.ELEMENT_NODE&&$.tagName===C.tagName);if(_.length>1){const $=_.indexOf(C)+1;x+=`[${$}]`}}a.unshift(x)}else if(s.nodeType===Node.TEXT_NODE){const C=s.parentNode;if(C){const x=Array.from(C.childNodes).filter(y=>y.nodeType===Node.TEXT_NODE);if(x.length>1){const y=x.indexOf(s)+1;a.unshift(`text()[${y}]`)}else a.unshift("text()")}}s=s.parentNode}return"/"+a.join("/")}function h(E){const a=E.commonAncestorContainer,s=(a.nodeType===Node.TEXT_NODE,a.textContent||""),C=E.toString(),x=s.indexOf(C);if(x===-1)return"";const y=s.slice(Math.max(0,x-50),x),_=s.slice(x+C.length,x+C.length+50);return y+"|||"+_}function c(){return"hl_"+Date.now().toString(36)+"_"+Math.random().toString(36).slice(2,8)}function g(E){const a=E.getAttribute("data-aura-hl")||"";E.addEventListener("mouseenter",()=>b(E,a)),E.addEventListener("mouseleave",()=>{u=setTimeout(k,300)})}function S(E){const a=window.getSelection();if(!a||a.rangeCount===0)return null;const s=a.getRangeAt(0);if(s.collapsed)return null;try{const C=document.createElement("mark");return C.setAttribute("data-aura-hl",E),s.surroundContents(C),a.removeAllRanges(),g(C),C}catch{try{const y=s.cloneContents().textContent||"";if(!y.trim())return null;s.deleteContents();const _=document.createElement("mark");return _.setAttribute("data-aura-hl",E),_.textContent=y,s.insertNode(_),a.removeAllRanges(),g(_),_}catch{return null}}}function T(){const E=window.getSelection();if(!E||E.rangeCount===0||E.isCollapsed)return!1;const a=E.getRangeAt(0),s=a.toString().trim();if(!s)return!1;const C=c(),x=m(a.startContainer),y=h(a);if(!S(C))return!1;const $={id:C,url:window.location.href,text:s,xpath:x,context:y,timestamp:Date.now(),color:"purple",pageTitle:document.title};return n({type:"SAVE_HIGHLIGHT",highlight:$},R=>{R&&R.ok?t("Highlight saved to AURA"):t((R==null?void 0:R.error)||"Failed to save highlight",3e3)}),!0}function A(E){const a=document.querySelector(`mark[data-aura-hl="${E}"]`);if(a){const s=a.parentNode;for(;a.firstChild;)s==null||s.insertBefore(a.firstChild,a);a.remove(),s==null||s.normalize()}n({type:"DELETE_HIGHLIGHT",id:E,url:window.location.href},s=>{t("Highlight removed")})}function f(E,a,s){try{const B=document.evaluate(E,document,null,XPathResult.FIRST_ORDERED_NODE_TYPE,null).singleNodeValue;if(B&&B.textContent&&B.textContent.includes(a)){const v=document.createRange(),w=B.textContent.indexOf(a);if(w>=0)return v.setStart(B,w),v.setEnd(B,w+a.length),v}}catch{}const C=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT,null),[x,y]=s.split("|||");let _=null,$=-1,R=0;for(;C.nextNode();){const H=C.currentNode,B=H.textContent||"",v=B.indexOf(a);if(v===-1)continue;let w=1;x&&B.slice(Math.max(0,v-50),v).includes(x.slice(-20))&&(w+=2),y&&B.slice(v+a.length,v+a.length+50).includes(y.slice(0,20))&&(w+=2),w>R&&(R=w,_=H,$=v)}if(_&&$>=0){const H=document.createRange();return H.setStart(_,$),H.setEnd(_,$+a.length),H}return null}function N(E){if(document.querySelector(`mark[data-aura-hl="${E.id}"]`))return!0;const a=f(E.xpath,E.text,E.context);if(!a)return!1;try{const s=document.createElement("mark");return s.setAttribute("data-aura-hl",E.id),E.stale&&s.classList.add("aura-hl-stale"),a.surroundContents(s),g(s),!0}catch{try{const C=a.toString();a.deleteContents();const x=document.createElement("mark");return x.setAttribute("data-aura-hl",E.id),E.stale&&x.classList.add("aura-hl-stale"),x.textContent=C,a.insertNode(x),g(x),!0}catch{return!1}}}function L(){n({type:"GET_HIGHLIGHTS",url:window.location.href},E=>{if(!(!E||!E.ok||!E.highlights))for(const a of E.highlights)N(a)||(a.stale=!0,N(a))})}function M(E){const a=document.querySelector(`mark[data-aura-hl="${E}"]`);a&&(a.scrollIntoView({behavior:"smooth",block:"center"}),a.classList.add("aura-hl-flash"),setTimeout(()=>a.classList.remove("aura-hl-flash"),1500))}return{init(E,a,s){e=s,setTimeout(L,1500)},destroy(){o.remove(),d.remove()},scrollTo:M,saveHighlight:T,setShowToast(E){t=E}}}const vt="mail.google.com";function Et(){let e,t=()=>{};const n=new Map;function o(){return window.location.hostname===vt}function r(){const m=document.querySelectorAll(".a3s.aiL");if(m.length===0)return"";const h=[];return m.forEach(c=>{var S;const g=(S=c.innerText)==null?void 0:S.trim();g&&h.push(g)}),h.join(`

---

`).slice(0,2e4)}function i(m){const c=["Message Body","Nachrichtentext","Corps du message","Cuerpo del mensaje","Corpo da mensagem","Corpo del messaggio","Текст сообщения","Mesaj Metni","メッセージ本文","메시지 본문","邮件正文","نص الرسالة","Berichttekst","Treść wiadomości","संदेश का मुख्य भाग","Mesaj mətni"].map(S=>`div[aria-label="${S}"]`).join(", "),g=m.querySelector(c+', div[g_editable="true"][contenteditable="true"], div.editable[contenteditable="true"]');return g||m.querySelector('div[contenteditable="true"][role="textbox"]')}function p(m){var c;const h=i(m);return h&&((c=h.innerText)==null?void 0:c.trim())||""}function d(m){return m.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;")}function l(m,h){const c=i(m);if(!c)return;c.focus();const g=window.getSelection();if(g){const T=document.createRange();T.selectNodeContents(c),g.removeAllRanges(),g.addRange(T)}document.execCommand("insertText",!1,h)||(c.innerHTML=h.split(`
`).map(T=>`<div>${d(T)||"<br>"}</div>`).join("")),c.dispatchEvent(new Event("input",{bubbles:!0})),c.dispatchEvent(new Event("change",{bubbles:!0}))}function u(m){if(n.has(m))return;const h=m.querySelector('div[aria-label*="Send"], div[data-tooltip*="Send"], div[aria-label*="Enviar"], div[aria-label*="Envoyer"], div[aria-label*="Senden"], div[aria-label*="Отправить"]'),c=m.querySelector(".btC, .bAK, tr.btC, .IZ");if(!((h==null?void 0:h.parentElement)||c))return;const S=document.createElement("div");S.className="aura-gmail-ai-host",Object.assign(S.style,{display:"inline-flex",alignItems:"center",verticalAlign:"middle",marginLeft:"8px",position:"relative",zIndex:"1"});const T=S.attachShadow({mode:"closed"}),A=document.createElement("style");A.textContent=`
      @keyframes gmail-aura-in {
        from { opacity: 0; transform: scale(0.85); }
        to   { opacity: 1; transform: scale(1); }
      }
      @keyframes gmail-aura-spin {
        to { transform: rotate(360deg); }
      }
      @keyframes gmail-aura-menu-in {
        from { opacity: 0; transform: translateY(4px) scale(0.95); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
      }

      :host {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      }

      .gmail-ai-btn {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 18px;
        border: 1px solid rgba(124, 58, 237, 0.35);
        background: rgba(124, 58, 237, 0.08);
        color: #7c3aed;
        font-size: 12px;
        font-weight: 600;
        font-family: inherit;
        cursor: pointer;
        white-space: nowrap;
        transition: all 0.15s ease;
        animation: gmail-aura-in 0.25s ease forwards;
        line-height: 1.4;
        letter-spacing: 0.01em;
      }
      .gmail-ai-btn:hover {
        background: rgba(124, 58, 237, 0.15);
        border-color: rgba(124, 58, 237, 0.5);
        box-shadow: 0 0 12px rgba(124, 58, 237, 0.15);
      }
      .gmail-ai-btn:active {
        transform: scale(0.97);
      }
      .gmail-ai-btn .sparkle {
        font-size: 13px;
        line-height: 1;
      }

      .gmail-ai-menu {
        position: absolute;
        bottom: calc(100% + 6px);
        left: 0;
        background: rgba(10, 8, 24, 0.94);
        backdrop-filter: blur(20px) saturate(1.5);
        -webkit-backdrop-filter: blur(20px) saturate(1.5);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 10px;
        padding: 4px;
        min-width: 180px;
        box-shadow: 0 -8px 32px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.05) inset;
        animation: gmail-aura-menu-in 0.18s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        z-index: 10000;
      }

      .gmail-ai-menu-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 7px;
        background: transparent;
        border: none;
        color: rgba(226, 232, 240, 0.92);
        font-size: 12.5px;
        font-weight: 500;
        font-family: inherit;
        cursor: pointer;
        white-space: nowrap;
        width: 100%;
        text-align: left;
        transition: background 0.12s, color 0.12s;
        line-height: 1;
        box-sizing: border-box;
      }
      .gmail-ai-menu-item:hover {
        background: rgba(124, 58, 237, 0.25);
        color: #fff;
      }
      .gmail-ai-menu-item:active {
        background: rgba(124, 58, 237, 0.4);
      }
      .gmail-ai-menu-item .item-icon {
        font-size: 14px;
        width: 18px;
        text-align: center;
        flex-shrink: 0;
      }
      .gmail-ai-menu-item.loading {
        opacity: 0.55;
        pointer-events: none;
      }

      .gmail-ai-sep {
        height: 1px;
        background: rgba(255,255,255,0.08);
        margin: 3px 8px;
      }

      .gmail-ai-sub {
        padding: 2px 0 2px 4px;
      }
      .gmail-ai-sub .gmail-ai-menu-item {
        font-size: 11.5px;
        padding: 6px 12px 6px 26px;
      }

      .gmail-ai-spinner {
        display: inline-block;
        width: 14px;
        height: 14px;
        border: 2px solid rgba(124, 58, 237, 0.3);
        border-top-color: rgba(160, 148, 210, 0.9);
        border-radius: 50%;
        animation: gmail-aura-spin 0.6s linear infinite;
        flex-shrink: 0;
      }

      .gmail-ai-toast {
        position: absolute;
        bottom: calc(100% + 6px);
        left: 50%;
        transform: translateX(-50%);
        background: rgba(5, 150, 105, 0.92);
        backdrop-filter: blur(12px);
        color: #fff;
        font-size: 11.5px;
        font-weight: 500;
        font-family: inherit;
        padding: 5px 12px;
        border-radius: 6px;
        white-space: nowrap;
        pointer-events: none;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        animation: gmail-aura-menu-in 0.15s ease forwards;
        z-index: 10001;
      }
    `,T.appendChild(A);const f=document.createElement("div");f.style.position="relative",f.style.display="inline-flex",f.style.alignItems="center",T.appendChild(f);const N=document.createElement("button");N.className="gmail-ai-btn",N.innerHTML='<span class="sparkle">✦</span> AI',f.appendChild(N);let L=null,M=null,E=null,a=null;function s(w,O=2500){E&&E.remove(),a&&clearTimeout(a),E=document.createElement("div"),E.className="gmail-ai-toast",E.textContent=w,f.appendChild(E),a=setTimeout(()=>{E&&(E.remove(),E=null),a=null},O)}function C(){L&&(L.remove(),L=null),M=null}function x(w){L&&L.querySelectorAll(".gmail-ai-menu-item").forEach(O=>{O.classList.add("loading")})}function y(w,O){const I=p(m),P=r();if(w==="draft_reply"&&!I&&!P){s("No email thread found",3e3),C();return}if(w!=="draft_reply"&&!I){s("Compose body is empty",3e3),C();return}x();const D={type:"QUICK_ACTION",action:w,text:I||"(empty — draft a new reply)",...w==="draft_reply"?{threadContext:P}:{},...O?{language:O}:{}};t(D,j=>{j&&j.ok&&j.result?(l(m,j.result),s("Updated by AURA")):s((j==null?void 0:j.error)||"Action failed",3e3),C()})}const _=[{icon:"✍️",label:"Draft reply",action:"draft_reply"},{icon:"✨",label:"Improve",action:"improve"},{icon:"🏢",label:"Make formal",action:"make_formal",separator:!0},{icon:"😊",label:"Make casual",action:"make_casual"},{icon:"✂️",label:"Shorten",action:"shorten"},{icon:"🌐",label:"Translate to...",action:"translate_menu",separator:!0}],$=["English","Spanish","French","German","Chinese"];function R(){C(),L=document.createElement("div"),L.className="gmail-ai-menu",_.forEach(w=>{if(w.separator){const I=document.createElement("div");I.className="gmail-ai-sep",L.appendChild(I)}const O=document.createElement("button");O.className="gmail-ai-menu-item",O.innerHTML=`<span class="item-icon">${w.icon}</span><span>${w.label}</span>`,O.addEventListener("click",I=>{I.preventDefault(),I.stopPropagation(),w.action==="translate_menu"?H(O):y(w.action)}),L.appendChild(O)}),f.appendChild(L)}function H(w){if(M){M.remove(),M=null;return}M=document.createElement("div"),M.className="gmail-ai-sub",$.forEach(O=>{const I=document.createElement("button");I.className="gmail-ai-menu-item",I.textContent=O,I.addEventListener("click",P=>{P.preventDefault(),P.stopPropagation(),y("gmail_translate",O)}),M.appendChild(I)}),L&&w.parentNode===L&&w.after(M)}N.addEventListener("click",w=>{w.preventDefault(),w.stopPropagation(),L?C():R()});const B=w=>{if(!L)return;w.composedPath().includes(S)||C()};document.addEventListener("mousedown",B,!0),h!=null&&h.parentElement?h.parentElement.insertBefore(S,h.nextSibling):c&&c.appendChild(S);const v=new MutationObserver(()=>{document.body.contains(m)||(v.disconnect(),document.removeEventListener("mousedown",B,!0),S.remove(),n.delete(m))});v.observe(document.body,{childList:!0,subtree:!0}),n.set(m,{composeEl:m,buttonHost:S,shadow:T,observer:v,outsideHandler:B})}function k(){['div[role="dialog"]',"div.ip.iq","div.nH.nn"].forEach(h=>{document.querySelectorAll(h).forEach(c=>{i(c)&&(n.has(c)||u(c))})})}function b(){if(!o())return;k();const m=new MutationObserver(g=>{var T,A,f;let S=!1;for(const N of g){if(N.addedNodes.length>0)for(const L of N.addedNodes){if(L.nodeType!==Node.ELEMENT_NODE)continue;const M=L;if((T=M.matches)!=null&&T.call(M,'div[role="dialog"]')||(A=M.querySelector)!=null&&A.call(M,'div[role="dialog"]')||(f=M.querySelector)!=null&&f.call(M,'div[contenteditable="true"]')){S=!0;break}}if(S)break}S&&setTimeout(k,300)}),h=document.querySelector('div[role="main"]')||document.body;m.observe(h,{childList:!0,subtree:!0});let c=null;c=setInterval(()=>{if(!o()){c&&clearInterval(c);return}k()},3e3)}return{init(m,h,c){e=c,t=(g,S)=>{try{S?e.runtime.sendMessage(g,S):e.runtime.sendMessage(g)}catch{}},b()},destroy(){for(const m of n.values())m.observer.disconnect(),document.removeEventListener("mousedown",m.outsideHandler,!0),m.buttonHost.remove();n.clear()}}}const Ct=["display","position","flex-direction","align-items","justify-content","gap","flex-wrap","flex","flex-grow","flex-shrink","width","height","min-width","min-height","max-width","max-height","padding","padding-top","padding-right","padding-bottom","padding-left","margin","margin-top","margin-right","margin-bottom","margin-left","border","border-radius","border-color","border-width","border-style","background","background-color","background-image","background-size","color","font-size","font-weight","font-family","line-height","letter-spacing","text-align","text-decoration","text-transform","box-shadow","opacity","overflow","z-index","grid-template-columns","grid-template-rows","grid-gap","transform","transition"];function kt(){let e;const t=document.createElement("div");t.id="aura-capture-host",Object.assign(t.style,{position:"fixed",top:"0",left:"0",width:"0",height:"0",zIndex:"2147483647",pointerEvents:"none"}),document.documentElement.appendChild(t);const n=t.attachShadow({mode:"closed"}),o=document.createElement("style");o.textContent=`
    @keyframes capture-pulse {
      0%, 100% { opacity: 0.6; }
      50% { opacity: 1; }
    }
    .capture-overlay {
      position: fixed;
      pointer-events: none;
      border: 2px solid rgba(124, 58, 237, 0.8);
      background: rgba(124, 58, 237, 0.08);
      border-radius: 3px;
      transition: top 0.05s ease, left 0.05s ease, width 0.05s ease, height 0.05s ease;
      box-shadow: 0 0 0 1px rgba(124, 58, 237, 0.2),
                  0 0 20px rgba(124, 58, 237, 0.15),
                  inset 0 0 20px rgba(124, 58, 237, 0.05);
      z-index: 2147483647;
    }
    .capture-tooltip {
      position: fixed;
      background: rgba(10, 8, 24, 0.92);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(124, 58, 237, 0.35);
      border-radius: 6px;
      padding: 4px 10px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      font-size: 11px;
      color: rgba(226, 232, 240, 0.9);
      white-space: nowrap;
      pointer-events: none;
      z-index: 2147483647;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .capture-tooltip .tag {
      color: #a78bfa;
      font-weight: 600;
    }
    .capture-tooltip .cls {
      color: rgba(167, 139, 250, 0.6);
      max-width: 200px;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .capture-tooltip .dims {
      color: rgba(226, 232, 240, 0.5);
      font-size: 10px;
    }
    .capture-banner {
      position: fixed;
      top: 8px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(10, 8, 24, 0.92);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(124, 58, 237, 0.4);
      border-radius: 10px;
      padding: 8px 16px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      font-size: 12px;
      color: rgba(226, 232, 240, 0.9);
      z-index: 2147483647;
      pointer-events: none;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
      display: flex;
      align-items: center;
      gap: 8px;
      animation: capture-pulse 2s ease-in-out infinite;
    }
    .capture-banner .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #a78bfa;
      box-shadow: 0 0 8px rgba(124, 58, 237, 0.6);
    }
  `,n.appendChild(o);const r=document.createElement("div");n.appendChild(r);let i=!1,p=null,d=null,l=null,u=null;function k(A){const f=window.getComputedStyle(A),N={};for(const L of Ct){const M=f.getPropertyValue(L);M&&M!=="none"&&M!=="normal"&&M!=="auto"&&M!=="0px"&&M!=="rgba(0, 0, 0, 0)"&&(N[L]=M)}return N}function b(A){const f=A.tagName.toLowerCase(),N=A.className&&typeof A.className=="string"?"."+A.className.trim().split(/\s+/).slice(0,2).join("."):"";return f+N}function m(A){const f=A.getBoundingClientRect(),N=window.getComputedStyle(A),L=A.outerHTML,M={};M[b(A)]=k(A);const E=A.querySelectorAll("*");let a=0;for(const s of E){if(a>=50)break;const C=k(s);if(Object.keys(C).length>0){const x=b(s),y=M[x]?`${x}:nth(${a})`:x;M[y]=C}a++}return{html:L,css:M,dimensions:{width:f.width,height:f.height,padding:`${N.paddingTop} ${N.paddingRight} ${N.paddingBottom} ${N.paddingLeft}`,margin:`${N.marginTop} ${N.marginRight} ${N.marginBottom} ${N.marginLeft}`},textContent:(A.textContent||"").slice(0,2e3).trim(),tagName:A.tagName.toLowerCase(),className:(typeof A.className=="string"?A.className:"").trim()}}function h(){i||(i=!0,l=document.createElement("div"),l.className="capture-banner",l.innerHTML='<span class="dot"></span> AURA Capture Mode — Click any element • Esc to exit',r.appendChild(l),p=document.createElement("div"),p.className="capture-overlay",p.style.display="none",r.appendChild(p),d=document.createElement("div"),d.className="capture-tooltip",d.style.display="none",r.appendChild(d),t.style.width="100vw",t.style.height="100vh",document.addEventListener("mousemove",g,!0),document.addEventListener("click",S,!0),document.addEventListener("keydown",T,!0))}function c(){if(i){i=!1,u=null,p&&(p.remove(),p=null),d&&(d.remove(),d=null),l&&(l.remove(),l=null),t.style.width="0",t.style.height="0",document.removeEventListener("mousemove",g,!0),document.removeEventListener("click",S,!0),document.removeEventListener("keydown",T,!0);try{e.runtime.sendMessage({type:"OPEN_PANEL",panel:"capture"})}catch{}}}function g(A){if(!i)return;const f=document.elementsFromPoint(A.clientX,A.clientY);let N=null;for(const M of f)if(!(M===t||t.contains(M))&&!(M.id==="aura-host"||M.id==="aura-dock-shadow"||M.id==="aura-quick-action-host"||M.id==="aura-highlight-host"||M.id==="aura-img-toolbar-host"||M.id==="aura-capture-host")&&!(M===document.documentElement||M===document.body)){N=M;break}if(!N){p&&(p.style.display="none"),d&&(d.style.display="none"),u=null;return}u=N;const L=N.getBoundingClientRect();if(p&&(p.style.display="block",p.style.top=L.top+"px",p.style.left=L.left+"px",p.style.width=L.width+"px",p.style.height=L.height+"px"),d){const M=N.tagName.toLowerCase(),E=N.className&&typeof N.className=="string"?N.className.trim().split(/\s+/).slice(0,3).join(" "):"",a=Math.round(L.width),s=Math.round(L.height);d.textContent="";const C=document.createElement("span");if(C.className="tag",C.textContent=`<${M}>`,d.appendChild(C),E){const $=document.createElement("span");$.className="cls",$.textContent="."+E.split(" ").join("."),d.appendChild($)}const x=document.createElement("span");x.className="dims",x.textContent=`${a}x${s}`,d.appendChild(x);let y=L.top-30;y<4&&(y=L.bottom+6);let _=L.left;_<4&&(_=4),d.style.display="flex",d.style.top=y+"px",d.style.left=_+"px"}}function S(A){if(!i||!u)return;A.preventDefault(),A.stopPropagation(),A.stopImmediatePropagation();const f=u,N=f.getBoundingClientRect(),L=m(f);try{e.runtime.sendMessage({type:"CAPTURE_ELEMENT",rect:{x:Math.round(N.left),y:Math.round(N.top),w:Math.round(N.width),h:Math.round(N.height)},elementData:L},M=>{})}catch{}c()}function T(A){if(A.key==="Escape"&&i){A.preventDefault(),A.stopPropagation(),c();try{e.runtime.sendMessage({type:"CAPTURE_MODE_EXITED"}).catch(()=>{})}catch{}}}return{init(A,f,N){e=N},destroy(){i&&c(),t.remove()},start:h,stop:c}}const St=50;function Tt(){let e;const t=new Map;function n(x,y){if(t.size>=St){const _=t.keys().next().value;_&&t.delete(_)}t.set(x,y)}function o(x){const y=t.get(x);return y&&(t.delete(x),t.set(x,y)),y}const r=document.createElement("div");r.id="aura-link-preview-host",Object.assign(r.style,{position:"fixed",top:"0",left:"0",zIndex:"2147483646",pointerEvents:"none"}),document.documentElement.appendChild(r);const i=r.attachShadow({mode:"closed"}),p=document.createElement("style");p.textContent=["@keyframes lp-in { from { opacity:0; transform:translateY(4px) scale(0.96); } to { opacity:1; transform:translateY(0) scale(1); } }","@keyframes lp-shimmer { 0% { background-position:-200px 0; } 100% { background-position:200px 0; } }",'.lp-popup { position:fixed; width:320px; max-height:280px; background:rgba(10,8,24,0.92); backdrop-filter:blur(20px) saturate(1.5); -webkit-backdrop-filter:blur(20px) saturate(1.5); border:1px solid rgba(124,58,237,0.25); border-radius:12px; padding:14px 16px 12px; pointer-events:auto; animation:lp-in 0.2s cubic-bezier(0.16,1,0.3,1) forwards; box-shadow:0 8px 32px rgba(0,0,0,0.5),0 0 0 1px rgba(255,255,255,0.05) inset; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Inter",system-ui,sans-serif; box-sizing:border-box; overflow:hidden; display:flex; flex-direction:column; gap:8px; }',".lp-domain { display:inline-block; background:rgba(124,58,237,0.15); border:1px solid rgba(124,58,237,0.25); border-radius:4px; padding:2px 7px; font-size:10.5px; font-weight:600; color:rgba(160,148,210,0.9); letter-spacing:0.3px; max-width:fit-content; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }",".lp-title { font-size:13px; font-weight:600; color:rgba(226,232,240,0.95); line-height:1.35; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; margin:0; }",".lp-description { font-size:12px; font-weight:400; color:rgba(226,232,240,0.65); line-height:1.45; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; margin:0; }",".lp-shimmer { height:12px; border-radius:4px; background:linear-gradient(90deg,rgba(124,58,237,0.08) 25%,rgba(124,58,237,0.18) 50%,rgba(124,58,237,0.08) 75%); background-size:400px 100%; animation:lp-shimmer 1.5s infinite linear; }",".lp-shimmer.short { width:60%; } .lp-shimmer.long { width:90%; } .lp-shimmer+.lp-shimmer { margin-top:6px; }",".lp-loading-label { font-size:11px; color:rgba(160,148,210,0.5); margin-bottom:4px; }",".lp-actions { display:flex; gap:6px; margin-top:4px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.06); }",".lp-btn { background:rgba(124,58,237,0.12); border:1px solid rgba(124,58,237,0.2); border-radius:6px; padding:4px 10px; font-size:11px; font-weight:500; font-family:inherit; color:rgba(200,180,255,0.9); cursor:pointer; transition:background 0.15s,border-color 0.15s,color 0.15s; white-space:nowrap; }",".lp-btn:hover { background:rgba(124,58,237,0.25); border-color:rgba(124,58,237,0.4); color:#fff; }",".lp-btn:active { background:rgba(124,58,237,0.35); }"].join(`
`),i.appendChild(p);const d=document.createElement("div");i.appendChild(d);let l=null,u=null,k=null,b=null,m=!1;const h=()=>{m=!0},c=()=>{m=!1};function g(x){try{return new URL(x.href,location.href).hostname!==location.hostname}catch{return!1}}function S(x){const y=x.href||"";if(!y.startsWith("http://")&&!y.startsWith("https://"))return!1;try{const _=new URL(y,location.href);if(_.hostname===location.hostname&&_.pathname===location.pathname&&_.hash)return!1}catch{return!1}return(x.textContent||"").trim().length<10?!1:g(x)}function T(){l&&(l.remove(),l=null),b=null}function A(){u&&(clearTimeout(u),u=null),k&&(clearTimeout(k),k=null)}function f(){k&&clearTimeout(k),k=setTimeout(()=>{T(),k=null},300)}function N(){k&&(clearTimeout(k),k=null)}function L(x){if(!l)return;const y=x.getBoundingClientRect();l.style.visibility="hidden",l.style.display="flex";const _=l.offsetHeight||180;l.style.visibility="";let $=y.left+y.width/2-160;$<8&&($=8),$+320>window.innerWidth-8&&($=window.innerWidth-328);let R=y.bottom+8;R+_>window.innerHeight-8&&(R=y.top-_-8,R<8&&(R=8)),l.style.top=Math.round(R)+"px",l.style.left=Math.round($)+"px"}function M(x,y,_){if(x.innerHTML="",x.style.display="none",_.title&&_.title!==y.textContent&&(y.textContent=_.title),_.description){const $=document.createElement("div");$.className="lp-description",$.textContent=_.description,y.after($)}l&&b&&L(b)}function E(x,y){T(),b=x;let _="";try{_=new URL(y).hostname}catch{_=y}const $=(x.textContent||"").trim();l=document.createElement("div"),l.className="lp-popup";const R=document.createElement("div");R.className="lp-domain",R.textContent=_,l.appendChild(R);const H=document.createElement("div");H.className="lp-title",H.textContent=$,l.appendChild(H);const B=document.createElement("div"),v=document.createElement("div");v.className="lp-loading-label",v.textContent="Loading preview…";const w=document.createElement("div");w.className="lp-shimmer long";const O=document.createElement("div");O.className="lp-shimmer short",B.appendChild(v),B.appendChild(w),B.appendChild(O),l.appendChild(B);const I=document.createElement("div");I.className="lp-actions";const P=document.createElement("button");P.className="lp-btn",P.textContent="Open",P.addEventListener("click",G=>{G.preventDefault(),G.stopPropagation(),window.open(y,"_blank","noopener"),T()});const D=document.createElement("button");D.className="lp-btn",D.textContent="Summarize in AURA",D.addEventListener("click",G=>{G.preventDefault(),G.stopPropagation();try{e.runtime.sendMessage({type:"OPEN_WITH_TEXT",action:"summarize",text:"Summarize this page: "+y,url:y,title:$})}catch{}T()}),I.appendChild(P),I.appendChild(D),l.appendChild(I),l.addEventListener("mouseenter",N),l.addEventListener("mouseleave",f),d.appendChild(l),L(x);const j=o(y);if(j){M(B,H,j);return}try{e.runtime.sendMessage({type:"LINK_PREVIEW",url:y},G=>{if(e.runtime.lastError||!G||!l||b!==x)return;const Y={title:G.title||$,description:G.description||"",domain:G.domain||_};n(y,Y),M(B,H,Y)})}catch{}}const a=x=>{if(m)return;const y=x.target.closest("a");if(!(!y||!S(y))){if(b===y&&l){N();return}A(),u=setTimeout(()=>{m||(E(y,y.href),u=null)},800)}},s=x=>{const y=x.target.closest("a");if(y&&y===b){const _=x.relatedTarget;if(_&&r.contains(_))return;f()}y&&u&&A()},C=()=>{if(l&&b){const x=b.getBoundingClientRect();x.bottom<0||x.top>window.innerHeight?(A(),T()):L(b)}};return{init(x,y,_){e=_,document.addEventListener("mousedown",h,!0),document.addEventListener("mouseup",c,!0),document.addEventListener("mouseover",a,!0),document.addEventListener("mouseout",s,!0),window.addEventListener("scroll",C,{passive:!0})},destroy(){document.removeEventListener("mousedown",h,!0),document.removeEventListener("mouseup",c,!0),document.removeEventListener("mouseover",a,!0),document.removeEventListener("mouseout",s,!0),window.removeEventListener("scroll",C),A(),T(),r.remove()}}}function Lt(e,t){var h;let n="https://aura-elnur.duckdns.org",o="";function r(){return new Promise(c=>{var g;if(!((g=e==null?void 0:e.storage)!=null&&g.local)){c();return}e.storage.local.get(["backendUrl","apiKey"],S=>{var T,A;(T=S==null?void 0:S.backendUrl)!=null&&T.trim()&&(n=S.backendUrl.trim().replace(/\/+$/,"")),(A=S==null?void 0:S.apiKey)!=null&&A.trim()&&(o=S.apiKey.trim()),c()})})}r(),(h=e==null?void 0:e.storage)!=null&&h.onChanged&&e.storage.onChanged.addListener((c,g)=>{var S,T,A;g==="local"&&((S=c.backendUrl)!=null&&S.newValue&&(n=c.backendUrl.newValue.trim().replace(/\/+$/,"")),((T=c.apiKey)==null?void 0:T.newValue)!==void 0&&(o=((A=c.apiKey.newValue)==null?void 0:A.trim())||""))});function i(){const c=window.location.hostname,g=window.location.pathname,S=new URLSearchParams(window.location.search);if(!c.match(/^(www\.)?google\./)||g!=="/search"||!S.get("q"))return!1;const T=S.get("tbm");if(T&&["isch","lcl","vid","shop","nws","bks","fin"].includes(T))return!1;const A=S.get("udm");return!(A&&["2","14"].includes(A))}function p(){const g=new URLSearchParams(window.location.search).get("q")||"";if(g)return g;const S=document.querySelector('input[name="q"]');return(S==null?void 0:S.value)||""}function d(){const c=window.getComputedStyle(document.body).backgroundColor;if(!c||c==="transparent")return"light";const g=c.match(/\d+/g);if(g&&g.length>=3){const[S,T,A]=g.map(Number);return .299*S+.587*T+.114*A<128?"dark":"light"}return"light"}function l(c){return c.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;")}function u(c){let g=l(c);return g=g.replace(/\*\*(.+?)\*\*/g,"<strong>$1</strong>"),g=g.replace(/__(.+?)__/g,"<strong>$1</strong>"),g=g.replace(new RegExp("(?<!\\*)\\*(?!\\*)(.+?)(?<!\\*)\\*(?!\\*)","g"),"<em>$1</em>"),g=g.replace(/`([^`]+)`/g,"<code>$1</code>"),g=g.replace(/\[([^\]]+)\]\((https?:\/\/[^)"]+)\)/g,'<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'),g=g.replace(/^[\s]*[-*]\s+(.+)$/gm,"<li>$1</li>"),g=g.replace(/((?:<li>.*<\/li>\n?)+)/g,"<ul>$1</ul>"),g=g.replace(/^[\s]*\d+\.\s+(.+)$/gm,"<li>$1</li>"),g=g.replace(/\n\n+/g,"</p><p>"),g="<p>"+g+"</p>",g=g.replace(/\n/g,"<br>"),g=g.replace(/<p>\s*<\/p>/g,""),g=g.replace(/<\/?(?!(?:strong|em|code|a|li|ul|ol|p|br)\b)[^>]*>/gi,""),g}function k(c,g){const S=/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,T=[];let A;for(;(A=S.exec(g))!==null;)T.push({title:A[1],url:A[2]});if(T.length===0)return;const f=document.createElement("div");f.className="serp-citations";const N=document.createElement("div");N.className="serp-citations-label",N.textContent="Sources",f.appendChild(N);const L=document.createElement("div");L.className="serp-citation-list",T.forEach((M,E)=>{const a=document.createElement("a");a.className="serp-citation-chip",a.href=M.url,a.target="_blank",a.rel="noopener noreferrer";const s=document.createElement("span");s.className="serp-citation-num",s.textContent=String(E+1),a.appendChild(s);const C=document.createTextNode(" "+M.title);a.appendChild(C),L.appendChild(a)}),f.appendChild(L),c.appendChild(f)}function b(c,g,S){const T=document.createElement("div");T.className="serp-footer";const A=document.createElement("button");A.className="serp-followup-btn",A.innerHTML='<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg> Ask follow-up',A.addEventListener("click",()=>{t({type:"OPEN_WITH_TEXT",action:"ask",text:`I searched for "${g}" and got the following AI answer:

${S}

I have a follow-up question: `,url:window.location.href,title:document.title})});const f=document.createElement("span");f.className="serp-powered",f.textContent="Powered by AURA",T.appendChild(A),T.appendChild(f),c.appendChild(T)}async function m(){if(!i()||(await new Promise(O=>{e.storage.local.get(["aura_serp_hidden"],O)})).aura_serp_hidden)return;const g=p();if(!g)return;const S=document.createElement("div");S.id="aura-serp-host",Object.assign(S.style,{position:"fixed",top:"80px",right:"16px",width:"340px",maxHeight:"calc(100vh - 100px)",zIndex:"2147483640",pointerEvents:"auto"}),document.documentElement.appendChild(S);const T=S.attachShadow({mode:"closed"}),f=d()==="dark",N=document.createElement("style");N.textContent=`
      @keyframes serp-fade-in {
        from { opacity: 0; transform: translateY(-8px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @keyframes serp-pulse {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 1; }
      }

      *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

      :host {
        display: block;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      }

      .serp-card {
        background: ${f?"rgba(30, 27, 48, 0.92)":"rgba(255, 255, 255, 0.95)"};
        backdrop-filter: blur(24px) saturate(1.4);
        -webkit-backdrop-filter: blur(24px) saturate(1.4);
        border-radius: 16px;
        overflow-y: auto;
        max-height: calc(100vh - 120px);
        box-shadow: ${f?"0 8px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.06)":"0 8px 40px rgba(0,0,0,0.12), 0 0 0 1px rgba(0,0,0,0.06)"};
        border: 1px solid ${f?"rgba(124, 58, 237, 0.2)":"rgba(124, 58, 237, 0.15)"};
        padding: 20px 24px 16px;
        animation: serp-fade-in 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        position: relative;
        overflow: hidden;
        transition: border-color 0.25s ease;
      }
      .serp-card:hover {
        border-color: ${f?"rgba(124, 58, 237, 0.35)":"rgba(124, 58, 237, 0.3)"};
      }

      .serp-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
      }
      .serp-header-left {
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .serp-logo {
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: ${f?"rgba(160, 148, 210, 0.9)":"rgba(124, 58, 237, 0.85)"};
        background: ${f?"rgba(124, 58, 237, 0.12)":"rgba(124, 58, 237, 0.08)"};
        border-radius: 8px;
        flex-shrink: 0;
      }
      .serp-title {
        font-size: 14px;
        font-weight: 600;
        color: ${f?"rgba(226, 232, 240, 0.9)":"rgba(30, 27, 48, 0.9)"};
        letter-spacing: -0.01em;
      }
      .serp-title-sub {
        font-size: 11px;
        font-weight: 400;
        color: ${f?"rgba(160, 148, 210, 0.5)":"rgba(100, 90, 140, 0.6)"};
        margin-left: 6px;
      }

      .serp-controls {
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .serp-ctrl-btn {
        width: 28px;
        height: 28px;
        border-radius: 8px;
        border: none;
        background: transparent;
        color: ${f?"rgba(160, 148, 210, 0.5)":"rgba(100, 90, 140, 0.5)"};
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.15s, color 0.15s;
        padding: 0;
      }
      .serp-ctrl-btn:hover {
        background: ${f?"rgba(124, 58, 237, 0.15)":"rgba(124, 58, 237, 0.1)"};
        color: ${f?"rgba(224, 214, 255, 1)":"rgba(124, 58, 237, 0.9)"};
      }
      .serp-ctrl-btn[title="Hide AURA answers"]:hover {
        background: rgba(239, 68, 68, 0.12);
        color: rgba(239, 68, 68, 0.9);
      }

      .serp-body {
        font-size: 14px;
        line-height: 1.7;
        color: ${f?"rgba(226, 232, 240, 0.85)":"rgba(30, 27, 48, 0.85)"};
        overflow: hidden;
        transition: max-height 0.3s ease;
      }
      .serp-body.collapsed {
        max-height: 0 !important;
        margin: 0;
        padding: 0;
      }

      .serp-loading {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0;
      }
      .serp-loading-dots {
        display: flex;
        gap: 4px;
      }
      .serp-loading-dots span {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: ${f?"rgba(124, 58, 237, 0.6)":"rgba(124, 58, 237, 0.5)"};
        animation: serp-pulse 1.2s ease-in-out infinite;
      }
      .serp-loading-dots span:nth-child(2) { animation-delay: 0.2s; }
      .serp-loading-dots span:nth-child(3) { animation-delay: 0.4s; }
      .serp-loading-text {
        font-size: 13px;
        color: ${f?"rgba(160, 148, 210, 0.6)":"rgba(100, 90, 140, 0.6)"};
      }

      .serp-answer {
        white-space: pre-wrap;
        word-break: break-word;
      }
      .serp-answer p { margin-bottom: 8px; }
      .serp-answer p:last-child { margin-bottom: 0; }
      .serp-answer strong, .serp-answer b {
        font-weight: 600;
        color: ${f?"rgba(226, 232, 240, 0.95)":"rgba(30, 27, 48, 0.95)"};
      }
      .serp-answer code {
        background: ${f?"rgba(124, 58, 237, 0.1)":"rgba(124, 58, 237, 0.06)"};
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 13px;
        font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
      }
      .serp-answer ul, .serp-answer ol {
        padding-left: 20px;
        margin-bottom: 8px;
      }
      .serp-answer li { margin-bottom: 4px; }
      .serp-answer a {
        color: ${f?"rgba(160, 148, 255, 0.9)":"rgba(100, 58, 237, 0.9)"};
        text-decoration: none;
      }
      .serp-answer a:hover { text-decoration: underline; }

      .serp-citations {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid ${f?"rgba(255,255,255,0.06)":"rgba(0,0,0,0.06)"};
      }
      .serp-citations-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: ${f?"rgba(160, 148, 210, 0.5)":"rgba(100, 90, 140, 0.5)"};
        margin-bottom: 6px;
      }
      .serp-citation-list {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }
      .serp-citation-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: ${f?"rgba(124, 58, 237, 0.1)":"rgba(124, 58, 237, 0.06)"};
        border: 1px solid ${f?"rgba(124, 58, 237, 0.15)":"rgba(124, 58, 237, 0.1)"};
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        color: ${f?"rgba(200, 180, 255, 0.8)":"rgba(100, 58, 237, 0.8)"};
        text-decoration: none;
        transition: background 0.15s, border-color 0.15s;
        max-width: 280px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .serp-citation-chip:hover {
        background: ${f?"rgba(124, 58, 237, 0.2)":"rgba(124, 58, 237, 0.12)"};
        border-color: ${f?"rgba(124, 58, 237, 0.3)":"rgba(124, 58, 237, 0.2)"};
      }
      .serp-citation-num {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        background: ${f?"rgba(124, 58, 237, 0.2)":"rgba(124, 58, 237, 0.1)"};
        font-size: 10px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
      }

      .serp-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 14px;
        padding-top: 10px;
        border-top: 1px solid ${f?"rgba(255,255,255,0.06)":"rgba(0,0,0,0.06)"};
      }
      .serp-followup-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: ${f?"rgba(124, 58, 237, 0.12)":"rgba(124, 58, 237, 0.08)"};
        border: 1px solid ${f?"rgba(124, 58, 237, 0.2)":"rgba(124, 58, 237, 0.15)"};
        border-radius: 8px;
        padding: 7px 14px;
        font-size: 12.5px;
        font-weight: 500;
        font-family: inherit;
        color: ${f?"rgba(200, 180, 255, 0.9)":"rgba(100, 58, 237, 0.9)"};
        cursor: pointer;
        transition: background 0.15s, border-color 0.15s, color 0.15s, transform 0.15s;
      }
      .serp-followup-btn:hover {
        background: ${f?"rgba(124, 58, 237, 0.22)":"rgba(124, 58, 237, 0.15)"};
        border-color: ${f?"rgba(124, 58, 237, 0.35)":"rgba(124, 58, 237, 0.3)"};
        transform: scale(1.01);
      }
      .serp-followup-btn:active { transform: scale(0.98); }
      .serp-powered {
        font-size: 11px;
        color: ${f?"rgba(160, 148, 210, 0.35)":"rgba(100, 90, 140, 0.35)"};
      }

      .serp-offline {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
      }
      .serp-offline-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: rgba(239, 68, 68, 0.6);
        flex-shrink: 0;
      }
      .serp-offline-text {
        font-size: 13px;
        color: ${f?"rgba(226, 232, 240, 0.5)":"rgba(30, 27, 48, 0.5)"};
      }

      .serp-error {
        font-size: 13px;
        color: ${f?"rgba(239, 150, 150, 0.8)":"rgba(200, 50, 50, 0.7)"};
        padding: 4px 0;
      }
    `,T.appendChild(N);const L=document.createElement("div");L.className="serp-card";const M=document.createElement("div");M.className="serp-header";const E=document.createElement("div");E.className="serp-header-left";const a=document.createElement("div");a.className="serp-logo",a.innerHTML='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3L2 21M12 3L22 21M5.8 14.2L18.2 14.2"/></svg>';const s=document.createElement("div"),C=document.createElement("span");C.className="serp-title",C.textContent="AI Answer";const x=document.createElement("span");x.className="serp-title-sub",x.textContent="by AURA",s.appendChild(C),s.appendChild(x),E.appendChild(a),E.appendChild(s);const y=document.createElement("div");y.className="serp-controls";const _=document.createElement("button");_.className="serp-ctrl-btn",_.title="Collapse",_.innerHTML='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>';const $=document.createElement("button");$.className="serp-ctrl-btn",$.title="Hide AURA answers",$.innerHTML='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',y.appendChild(_),y.appendChild($),M.appendChild(E),M.appendChild(y),L.appendChild(M);const R=document.createElement("div");R.className="serp-body";const H=document.createElement("div");H.className="serp-loading";const B=document.createElement("div");B.className="serp-loading-dots",B.innerHTML="<span></span><span></span><span></span>";const v=document.createElement("span");v.className="serp-loading-text",v.textContent=`Thinking about "${g.slice(0,60)}${g.length>60?"...":""}"`,H.appendChild(B),H.appendChild(v),R.appendChild(H),L.appendChild(R),T.appendChild(L);let w=!1;_.addEventListener("click",()=>{w=!w,w?(R.classList.add("collapsed"),_.title="Expand",_.innerHTML='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>'):(R.classList.remove("collapsed"),_.title="Collapse",_.innerHTML='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>')}),$.addEventListener("click",()=>{e.storage.local.set({aura_serp_hidden:!0}),S.remove()});try{const O=JSON.stringify({message:g,conversation_id:"__serp_answer__",stream:!1,system_context:`The user searched Google for: "${g}". Provide a concise, direct answer to their query. Be helpful and factual. Use markdown formatting sparingly — bold for emphasis, lists where appropriate. If you reference sources, format them as [Source Title](URL) and they will be rendered as citation chips. Keep the answer focused and under 200 words unless the topic requires more detail.`});let I=null;(async()=>{try{I=await new Promise((Y,Q)=>{e.runtime.sendMessage({type:"SERP_FETCH",url:`${n}/api/chat`,body:O,apiKey:o},fo=>{e.runtime.lastError?Q(new Error(e.runtime.lastError.message)):Y(fo)})})}catch{const Y={"Content-Type":"application/json"};o&&(Y["X-API-Key"]=o);const Q=await fetch(`${n}/api/chat`,{method:"POST",headers:Y,body:O,signal:AbortSignal.timeout(3e4)});if(!Q.ok)throw new Error(`HTTP ${Q.status}`);I={ok:!0,text:await Q.text()}}if(!(I!=null&&I.ok))throw new Error((I==null?void 0:I.error)||"Backend unreachable");H.remove();const P=document.createElement("div");P.className="serp-answer",R.appendChild(P);let D="";const j=I.text||"",G=j.split(`
`).filter(Y=>Y.trim());for(const Y of G)try{const Q=JSON.parse(Y);Q.chunk?D+=Q.chunk:Q.response?D=Q.response:Q.content&&(D=Q.content)}catch{D+=Y}if(!D.trim()&&j.trim()&&(D=j),P.innerHTML=u(D),!D.trim()){P.innerHTML='<span class="serp-error">No response from AI.</span>';return}k(R,D),b(L,g,D)})().catch(P=>{H.remove();const D=document.createElement("div");D.className="serp-offline";const j=document.createElement("div");j.className="serp-offline-dot";const G=document.createElement("span");G.className="serp-offline-text",G.textContent=`AURA is offline — backend did not respond (${(P==null?void 0:P.message)||"timeout"})`,D.appendChild(j),D.appendChild(G),R.appendChild(D)})}catch{H.remove();const I=document.createElement("div");I.className="serp-offline";const P=document.createElement("div");P.className="serp-offline-dot";const D=document.createElement("span");D.className="serp-offline-text",D.textContent="AURA is offline — backend did not respond",I.appendChild(P),I.appendChild(D),R.appendChild(I)}}m()}function Mt(e){var r;if(e.id)return"#"+CSS.escape(e.id);const t=e.getAttribute("aria-label");if(t)return`[aria-label="${t}"]`;const n=[];let o=e;for(let i=0;i<4&&o&&o!==document.body;i++,o=o.parentElement){const p=o.tagName.toLowerCase();if(o.id){n.unshift("#"+CSS.escape(o.id));break}const l=[...((r=o.parentElement)==null?void 0:r.children)||[]].indexOf(o)+1;n.unshift(p+":nth-child("+l+")")}return n.join(">")}function At(){const e=[],t=document.querySelectorAll('a,button,input,textarea,select,[role="button"],[onclick]');let n=0;for(const o of t){if(e.length>=80)break;const r=o.getBoundingClientRect();if(r.width===0||r.height===0)continue;const i=o,p=o;e.push({index:n++,type:o.tagName.toLowerCase(),text:(i.innerText||p.value||p.placeholder||i.title||"").slice(0,80).trim(),selector:Mt(i),href:o.href||""})}return e}function Nt(e){if(e.action==="scroll")return window.scrollBy(0,e.amount||300),{ok:!0};let t;try{t=document.querySelector(e.selector)}catch{return{ok:!1,error:"Invalid selector: "+e.selector}}if(!t)return{ok:!1,error:"Element not found: "+e.selector};if(e.action==="click")return t.click(),{ok:!0};if(e.action==="type")return t.focus(),t.value=e.text||"",t.dispatchEvent(new Event("input",{bubbles:!0})),t.dispatchEvent(new Event("change",{bubbles:!0})),{ok:!0};if(e.action==="selectOption"){if(t.tagName.toLowerCase()!=="select")return{ok:!1,error:"Element is not a <select>"};const n=t,o=[...n.options].find(r=>r.value===e.value||r.text===e.value);return o?(n.value=o.value,n.dispatchEvent(new Event("change",{bubbles:!0})),{ok:!0}):{ok:!1,error:"Option not found: "+e.value}}return{ok:!1,error:"Unknown action: "+e.action}}function _t(e,t){const n=document.createElement("div");Object.assign(n.style,{position:"fixed",top:"0",left:"0",width:"100vw",height:"100vh",zIndex:"2147483646",cursor:"crosshair",background:"rgba(0,0,0,0.4)"});const o=new Image;o.src=e,o.style.cssText="position:fixed;top:0;left:0;width:100%;height:100%;opacity:0.7;pointer-events:none;",n.appendChild(o);const r=document.createElement("canvas");r.width=window.innerWidth,r.height=window.innerHeight,Object.assign(r.style,{position:"absolute",top:"0",left:"0",width:"100%",height:"100%"}),n.appendChild(r);const i=r.getContext("2d"),p=document.createElement("div");Object.assign(p.style,{position:"fixed",top:"12px",left:"50%",transform:"translateX(-50%)",background:"rgba(0,0,0,0.75)",color:"#fff",padding:"6px 14px",borderRadius:"6px",fontSize:"13px",pointerEvents:"none"}),p.textContent="Drag to select region • Press Esc to cancel",n.appendChild(p),document.body.appendChild(n);let d=0,l=0,u=!1;const k=window.devicePixelRatio||1;function b(c,g,S,T){i&&(i.clearRect(0,0,r.width,r.height),i.strokeStyle="#7c3aed",i.lineWidth=2,i.strokeRect(c,g,S,T),i.fillStyle="rgba(124,58,237,0.12)",i.fillRect(c,g,S,T))}n.addEventListener("mousedown",c=>{d=c.clientX,l=c.clientY,u=!0}),n.addEventListener("mousemove",c=>{u&&b(d,l,c.clientX-d,c.clientY-l)});function m(c){c.key==="Escape"&&(document.body.contains(n)&&document.body.removeChild(n),document.removeEventListener("keydown",m),t({ok:!1}))}n.addEventListener("mouseup",c=>{u=!1;const g=Math.min(d,c.clientX),S=Math.min(l,c.clientY),T=Math.abs(c.clientX-d),A=Math.abs(c.clientY-l);if(document.removeEventListener("keydown",m),document.body.contains(n)&&document.body.removeChild(n),T<5||A<5){t({ok:!1});return}t({ok:!0,x:g,y:S,w:T,h:A,dpr:k})}),document.addEventListener("keydown",m);const h=new MutationObserver(()=>{document.body.contains(n)||(document.removeEventListener("keydown",m),h.disconnect())});h.observe(document.body,{childList:!0})}const ie=5e4,It=["article","main",'[role="main"]',".post-content",".article-body",".entry-content",".post-body",".article-content",".story-body",".content-body","#article-body","#content",".markdown-body",".wiki-content"],Ot=["nav","header","footer","aside","script","style","noscript","iframe",".sidebar",".menu",".nav",".navigation",".cookie",".cookie-banner",".cookie-consent",".popup",".modal",".overlay",".ad",".ads",".advert",".advertisement",".social-share",".share-buttons",".social-buttons",".comments",".comment-section","#comments",".related-posts",".recommended",".newsletter",".subscribe",'[role="navigation"]','[role="banner"]','[role="contentinfo"]','[role="complementary"]','[aria-hidden="true"]',".sr-only",".visually-hidden"];function $t(){for(const e of It){const t=document.querySelector(e);if(t&&t.textContent&&t.textContent.trim().length>200)return t}return document.body}function Ht(e){const t=e.cloneNode(!0);for(const n of Ot)t.querySelectorAll(n).forEach(o=>o.remove());return t}function zt(e){const t=[],n=new Set(["P","DIV","SECTION","ARTICLE","BLOCKQUOTE","PRE","H1","H2","H3","H4","H5","H6","UL","OL","LI","TABLE","TR","DT","DD","FIGURE","FIGCAPTION","HR","BR"]);function o(r){var l;if(r.nodeType===Node.TEXT_NODE){const u=(r.textContent||"").replace(/\s+/g," ");u.trim()&&t.push(u);return}if(r.nodeType!==Node.ELEMENT_NODE)return;const i=r,p=i.tagName;if(i.hasAttribute("hidden")||((l=i.style)==null?void 0:l.display)==="none")return;if(/^H[1-6]$/.test(p)){const u=parseInt(p[1]),k="#".repeat(Math.min(u,3))+" ",b=(i.textContent||"").trim();b&&t.push(`

`+k+b+`
`);return}if(p==="LI"){const u=(i.textContent||"").trim();u&&t.push(`
- `+u);return}if(p==="A"){const u=i.href,k=(i.textContent||"").trim();k&&u&&!u.startsWith("javascript:")?t.push(k+" ("+u+")"):k&&t.push(k);return}if(p==="HR"){t.push(`

---

`);return}if(p==="BR"){t.push(`
`);return}if(p==="PRE"){const u=(i.textContent||"").trim();u&&t.push("\n\n```\n"+u+"\n```\n\n");return}const d=n.has(p);d&&t.push(`

`);for(const u of i.childNodes)o(u);d&&t.push(`
`)}return o(e),t.join("").replace(/\n{3,}/g,`

`).replace(/[ \t]+/g," ").trim()}function Rt(){var b,m;const e=window.location.href,t=document.querySelector("h1.ytd-watch-metadata, h1.ytd-video-primary-info-renderer, #title h1"),n=((b=t==null?void 0:t.textContent)==null?void 0:b.trim())||document.title.replace(/ - YouTube$/,"").trim();let o="";const r=document.querySelectorAll("ytd-transcript-segment-renderer .segment-text, yt-formatted-string.ytd-transcript-segment-renderer, #segments-container ytd-transcript-segment-renderer");if(r.length>0){const h=[];r.forEach(c=>{var S;const g=(S=c.textContent)==null?void 0:S.trim();g&&h.push(g)}),o=h.join(" ")}let i="";const p=document.querySelector("ytd-text-inline-expander #plain-snippet-text, #description-inline-expander, ytd-expander .content, #description .content");p&&(i=((m=p.textContent)==null?void 0:m.trim())||"");const d=document.querySelectorAll("ytd-comment-thread-renderer #content-text");let l="";if(d.length>0){const h=[];d.forEach((c,g)=>{var T;if(g>=10)return;const S=(T=c.textContent)==null?void 0:T.trim();S&&h.push("- "+S)}),h.length>0&&(l=`

## Top Comments
`+h.join(`
`))}let u=`# ${n}

`;o&&(u+=`## Transcript
${o}

`),i&&(u+=`## Description
${i}

`),u+=l,u.length>ie&&(u=u.slice(0,ie)+`

[...truncated]`);const k=u.split(/\s+/).filter(Boolean).length;return{text:u,title:n,url:e,wordCount:k,isYouTube:!0,videoTitle:n,transcript:o||void 0}}function Bt(){var e,t,n,o;try{const r=window.location.href,i=document.title;if(r.match(/\.pdf($|\?|#)/i)||document.contentType==="application/pdf")return{text:((t=(e=document.body)==null?void 0:e.innerText)==null?void 0:t.slice(0,ie))||"[PDF document]",title:i,url:r,wordCount:0,isPdf:!0};if(r.includes("youtube.com/watch")||r.includes("youtu.be/"))return Rt();const p=$t(),d=Ht(p);let l=zt(d);l.length<100&&(l=((n=document.body)==null?void 0:n.innerText)||""),l.length>ie&&(l=l.slice(0,ie)+`

[...truncated]`);const u=l.split(/\s+/).filter(Boolean).length;return{text:l,title:i,url:r,wordCount:u}}catch{const i=(((o=document.body)==null?void 0:o.innerText)||"").slice(0,ie);return{text:i,title:document.title,url:window.location.href,wordCount:i.split(/\s+/).filter(Boolean).length}}}const Pt=[{label:"Improve",icon:'<path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/>',action:"improve"},{label:"Expand",icon:'<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>',action:"expand"},{label:"Shorten",icon:'<polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/>',action:"shorten"},{label:"Fix grammar",icon:'<polyline points="20 6 9 17 4 12"/>',action:"fix_grammar"},{label:"Translate",icon:'<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/>',action:"translate"}],Dt=new Set(["password","hidden","file","checkbox","radio","range","color","date","datetime-local","month","week","time","submit","reset","button","image"]),Pe=200;function qt(e,t,n){const o=document.createElement("div");o.id="aura-quick-action-host",Object.assign(o.style,{position:"fixed",top:"0",left:"0",zIndex:"2147483646",pointerEvents:"none"}),document.documentElement.appendChild(o);const r=o.attachShadow({mode:"closed"}),i=document.createElement("style");i.textContent=`
    @keyframes qa-icon-in {
      from { opacity: 0; transform: scale(0.7); }
      to   { opacity: 1; transform: scale(1); }
    }
    @keyframes qa-menu-in {
      from { opacity: 0; transform: translateY(4px) scale(0.95); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes qa-spin {
      to { transform: rotate(360deg); }
    }

    .qa-trigger {
      position: fixed;
      width: 20px;
      height: 20px;
      border-radius: 5px;
      background: rgba(10, 8, 24, 0.75);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(124, 58, 237, 0.3);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      pointer-events: auto;
      animation: qa-icon-in 0.2s ease forwards;
      transition: border-color 0.15s, background 0.15s, box-shadow 0.15s;
      padding: 0;
      box-sizing: border-box;
    }
    .qa-trigger:hover {
      border-color: rgba(124, 58, 237, 0.6);
      background: rgba(124, 58, 237, 0.18);
      box-shadow: 0 0 10px rgba(124, 58, 237, 0.25);
    }
    .qa-trigger svg {
      width: 12px;
      height: 12px;
      color: rgba(160, 148, 210, 0.8);
    }

    .qa-menu {
      position: fixed;
      background: rgba(10, 8, 24, 0.92);
      backdrop-filter: blur(20px) saturate(1.5);
      -webkit-backdrop-filter: blur(20px) saturate(1.5);
      border: 1px solid rgba(124, 58, 237, 0.25);
      border-radius: 10px;
      padding: 4px;
      pointer-events: auto;
      animation: qa-menu-in 0.18s cubic-bezier(0.16, 1, 0.3, 1) forwards;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05) inset;
      min-width: 140px;
    }
    .qa-menu-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 7px 12px;
      border-radius: 7px;
      background: transparent;
      border: none;
      color: rgba(226, 232, 240, 0.9);
      font-size: 12px;
      font-weight: 500;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      cursor: pointer;
      white-space: nowrap;
      width: 100%;
      text-align: left;
      transition: background 0.12s, color 0.12s;
      line-height: 1;
      box-sizing: border-box;
    }
    .qa-menu-item:hover {
      background: rgba(124, 58, 237, 0.25);
      color: #fff;
    }
    .qa-menu-item:active {
      background: rgba(124, 58, 237, 0.4);
    }
    .qa-menu-item svg {
      width: 14px;
      height: 14px;
      flex-shrink: 0;
      color: rgba(160, 148, 210, 0.7);
    }
    .qa-menu-item:hover svg {
      color: rgba(200, 180, 255, 1);
    }
    .qa-menu-item.loading {
      opacity: 0.6;
      pointer-events: none;
    }
    .qa-menu-item .qa-spinner {
      width: 14px;
      height: 14px;
      border: 2px solid rgba(124, 58, 237, 0.3);
      border-top-color: rgba(160, 148, 210, 0.9);
      border-radius: 50%;
      animation: qa-spin 0.6s linear infinite;
      flex-shrink: 0;
    }

    .qa-translate-sub {
      padding: 2px 4px 4px 4px;
    }
    .qa-translate-sub .qa-menu-item {
      font-size: 11.5px;
      padding: 5px 10px 5px 22px;
    }
  `,r.appendChild(i);const p=document.createElement("div");r.appendChild(p);let d=null,l=null,u=null,k=null;function b(a){if(a.tagName==="TEXTAREA")return!0;if(a.tagName==="INPUT"){const C=(a.type||"text").toLowerCase();return!Dt.has(C)}return!!(a.isContentEditable&&a.getAttribute("contenteditable")==="true")}function m(){u&&(u.remove(),u=null),k&&(k.remove(),k=null)}function h(){l&&(l.remove(),l=null),m(),d=null}function c(a){const s=a.getBoundingClientRect();if(s.width<Pe){h();return}const C=s.bottom<0||s.top>window.innerHeight||s.right<0||s.left>window.innerWidth;if(l||(l=document.createElement("div"),l.className="qa-trigger",l.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/></svg>',l.addEventListener("click",_=>{if(_.preventDefault(),_.stopPropagation(),u){m();return}f()}),p.appendChild(l)),l.style.display=C?"none":"",C){m();return}const x=20,y=6;l.style.top=`${Math.round(s.top+(s.height-x)/2)}px`,l.style.left=`${Math.round(s.right-x-y)}px`}function g(a){return a.isContentEditable?a.innerText||"":a.value||""}function S(a,s){a.isContentEditable?a.innerText=s:a.value=s,a.dispatchEvent(new Event("input",{bubbles:!0})),a.dispatchEvent(new Event("change",{bubbles:!0}))}function T(a,s){if(!d)return;const C=g(d);if(!C.trim()){m();return}u&&u.querySelectorAll(".qa-menu-item").forEach(y=>{y.classList.add("loading")});const x=d;t({type:"QUICK_ACTION",action:a,text:C,language:s},y=>{y&&y.ok&&y.result?(S(x,y.result),n("Text updated by AURA")):n((y==null?void 0:y.error)||"Quick action failed",3e3),m()})}function A(a){if(k){k.remove(),k=null;return}const s=["English","Spanish","French","German","Chinese","Russian","Japanese","Arabic","Portuguese","Azerbaijani"];k=document.createElement("div"),k.className="qa-translate-sub",s.forEach(C=>{const x=document.createElement("button");x.className="qa-menu-item",x.textContent=C,x.addEventListener("click",y=>{y.preventDefault(),y.stopPropagation(),T("translate",C)}),k.appendChild(x)}),u&&a.parentNode===u&&a.after(k)}function f(){if(!l||!d)return;m(),u=document.createElement("div"),u.className="qa-menu",Pt.forEach(C=>{const x=document.createElement("button");x.className="qa-menu-item",x.innerHTML=`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${C.icon}</svg><span>${C.label}</span>`,x.addEventListener("click",y=>{y.preventDefault(),y.stopPropagation(),C.action==="translate"?A(x):T(C.action)}),u.appendChild(x)}),p.appendChild(u);const a=l.getBoundingClientRect(),s=6;u.style.top=`${Math.round(a.bottom+s)}px`,u.style.left=`${Math.round(a.right-150)}px`,requestAnimationFrame(()=>{if(!u)return;const C=u.getBoundingClientRect();C.right>window.innerWidth-8&&(u.style.left=`${Math.round(window.innerWidth-C.width-8)}px`),C.left<8&&(u.style.left="8px"),C.bottom>window.innerHeight-8&&(u.style.top=`${Math.round(a.top-C.height-s)}px`)})}function N(a){const s=a.target;!s||!b(s)||s.getBoundingClientRect().width<Pe||(d=s,c(s))}function L(a){setTimeout(()=>{if(u||k)return;const s=document.activeElement;s&&s===d||h()},200)}function M(){d&&c(d)}document.addEventListener("mousedown",a=>{!u&&!l||a.composedPath().includes(o)||m()},!0),document.addEventListener("focusin",N,!0),document.addEventListener("focusout",L,!0),window.addEventListener("scroll",M,{passive:!0}),window.addEventListener("resize",M,{passive:!0}),new MutationObserver(()=>{d&&!document.body.contains(d)&&h()}).observe(document.body,{childList:!0,subtree:!0})}function Ft(e){document.addEventListener("aura-yt-subtitles",t=>{try{const n=t.detail;e({type:"YT_SUBTITLES",videoId:n.videoId||"",lang:n.lang||"",segments:n.segments||[]})}catch{}}),document.addEventListener("aura-yt-metadata",t=>{try{const n=t.detail;e({type:"YT_METADATA",videoId:n.videoId||"",title:n.title||"",duration:n.duration||0,description:n.description||"",channelName:n.channelName||"",chapters:n.chapters||[],captionTracks:n.captionTracks||[]})}catch{}})}function jt(e){document.addEventListener("aura-netflix-subtitles",t=>{try{const n=t.detail;e({type:"NETFLIX_SUBTITLES",movieId:n.movieId||"",lang:n.lang||"",trackId:n.trackId||"",segments:n.segments||[]})}catch{}}),document.addEventListener("aura-netflix-metadata",t=>{try{const n=t.detail;e({type:"NETFLIX_METADATA",movieId:n.movieId||"",title:n.title||"",episodeTitle:n.episodeTitle||"",seasonNumber:n.seasonNumber||0,episodeNumber:n.episodeNumber||0,duration:n.duration||0})}catch{}})}const Ut="p, h1, h2, h3, h4, h5, h6, li, td, th, blockquote, figcaption",ke="data-aura-translated",De=10,Wt=10;function Vt(e){const t={mode:"bilingual",targetLang:"English",active:!1,badge:null,elements:[],activeCount:0};function n(){const b=document.querySelectorAll(Ut),m=[];for(const h of b){if(h.hasAttribute(ke))continue;const c=h.getBoundingClientRect();c.width===0&&c.height===0||h.closest("#aura-host, #aura-dock-shadow, #aura-quick-action-host, .aura-translate-badge")||h.tagName==="SPAN"&&(h.textContent||"").trim().length<=20||(h.textContent||"").trim().length<5||m.push(h)}return m}function o(b){const m=document.createElement("div");return m.className="aura-page-translation",m.setAttribute("data-aura-translation","true"),Object.assign(m.style,{borderLeft:"2px solid rgba(124, 58, 237, 0.6)",background:"rgba(124, 58, 237, 0.05)",padding:"6px 10px",marginTop:"4px",marginBottom:"4px",fontSize:"0.95em",color:"inherit",opacity:"0",fontFamily:"inherit",lineHeight:"1.5",borderRadius:"0 4px 4px 0",transition:"opacity 0.3s ease",fontStyle:"italic"}),m.textContent="Translating...",m.style.color="rgba(124, 58, 237, 0.5)",b.setAttribute(ke,"true"),b.after(m),requestAnimationFrame(()=>{m.style.opacity="0.6"}),m}function r(b,m){b.style.opacity="0",b.textContent=m,b.style.fontStyle="normal",b.style.color="inherit",requestAnimationFrame(()=>{b.style.opacity="0.85"})}function i(b,m){return new Promise(h=>{try{e.runtime.sendMessage({type:"TRANSLATE_BATCH",texts:b,targetLang:m},c=>{if(e.runtime.lastError){h(b.map(()=>"[Translation failed]"));return}c!=null&&c.ok&&c.translations?h(c.translations):h(b.map(()=>(c==null?void 0:c.error)||"[Translation failed]"))})}catch{h(b.map(()=>"[Translation failed]"))}})}function p(){if(!t.badge)return;const b=t.badge.querySelector("[data-badge-mode]");b&&(b.textContent=t.mode==="bilingual"?"Bilingual":"Translated")}function d(b){t.mode=b;for(const m of t.elements)b==="translated"?(m.original.style.display="none",m.translation.style.marginTop="0"):(m.original.style.display="",m.translation.style.marginTop="4px");p()}function l(){t.active=!1;for(const b of t.elements)b.translation.remove(),b.original.removeAttribute(ke),b.original.style.display="";t.elements=[],t.badge&&(t.badge.remove(),t.badge=null)}function u(){t.badge&&(t.badge.remove(),t.badge=null),t.badge=document.createElement("div"),t.badge.className="aura-translate-badge",Object.assign(t.badge.style,{position:"fixed",bottom:"20px",right:"20px",zIndex:"2147483646",background:"rgba(10, 8, 24, 0.92)",backdropFilter:"blur(20px) saturate(1.5)",WebkitBackdropFilter:"blur(20px) saturate(1.5)",border:"1px solid rgba(124, 58, 237, 0.35)",borderRadius:"12px",padding:"8px 12px",display:"flex",alignItems:"center",gap:"8px",boxShadow:"0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05) inset",fontFamily:"-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif",fontSize:"12px",color:"rgba(226, 232, 240, 0.9)"});const b=document.createElement("span");Object.assign(b.style,{width:"6px",height:"6px",borderRadius:"50%",background:"#7c3aed",flexShrink:"0"}),t.badge.appendChild(b);const m=document.createElement("span");m.style.color="rgba(160, 148, 210, 0.8)",m.textContent="Translation active",t.badge.appendChild(m);const h=document.createElement("span");Object.assign(h.style,{width:"1px",height:"14px",background:"rgba(255,255,255,0.1)",flexShrink:"0"}),t.badge.appendChild(h);const c=document.createElement("span");c.setAttribute("data-badge-lang",""),c.textContent=t.targetLang,c.style.color="rgba(124, 58, 237, 0.9)",c.style.fontWeight="600",t.badge.appendChild(c);const g={background:"rgba(124, 58, 237, 0.15)",border:"1px solid rgba(124, 58, 237, 0.3)",borderRadius:"6px",color:"rgba(226, 232, 240, 0.9)",padding:"3px 8px",cursor:"pointer",fontSize:"11px",fontFamily:"inherit",transition:"background 0.15s, border-color 0.15s"},S=document.createElement("button");S.setAttribute("data-badge-mode",""),S.textContent="Bilingual",Object.assign(S.style,g),S.addEventListener("mouseenter",()=>{S.style.background="rgba(124, 58, 237, 0.3)"}),S.addEventListener("mouseleave",()=>{S.style.background="rgba(124, 58, 237, 0.15)"}),S.addEventListener("click",()=>{d(t.mode==="bilingual"?"translated":"bilingual")}),t.badge.appendChild(S);const T=document.createElement("button");T.textContent="✕",Object.assign(T.style,{...g,padding:"3px 6px",color:"rgba(226, 232, 240, 0.6)"}),T.title="Remove translation",T.addEventListener("mouseenter",()=>{T.style.background="rgba(239, 68, 68, 0.2)",T.style.borderColor="rgba(239, 68, 68, 0.4)",T.style.color="rgba(239, 68, 68, 0.9)"}),T.addEventListener("mouseleave",()=>{T.style.background="rgba(124, 58, 237, 0.15)",T.style.borderColor="rgba(124, 58, 237, 0.3)",T.style.color="rgba(226, 232, 240, 0.6)"}),T.addEventListener("click",()=>{l()}),t.badge.appendChild(T),document.body.appendChild(t.badge)}async function k(b){t.targetLang=b,t.active=!0,t.mode="bilingual",t.elements=[],t.activeCount=0,u();const m=n();if(m.length===0)return;const h=[];for(const T of m){const A=(T.textContent||"").trim();if(!A)continue;const f=o(T);t.elements.push({original:T,translation:f}),h.push({original:T,translation:f,text:A})}const c=[];for(let T=0;T<h.length;T+=De)c.push(h.slice(T,T+De));const g=async T=>{for(;t.activeCount>=Wt;)await new Promise(A=>setTimeout(A,100));if(t.active){t.activeCount++;try{const A=T.map(N=>N.text),f=await i(A,t.targetLang);if(!t.active)return;T.forEach((N,L)=>{t.active&&(r(N.translation,f[L]||"[No translation]"),t.mode==="translated"&&(N.original.style.display="none"))})}finally{t.activeCount--}}},S=c.map(T=>g(T));await Promise.all(S)}return{start:k,remove:l,setMode:d}}const Gt=["display","position","flex-direction","align-items","justify-content","gap","flex-wrap","flex","flex-grow","flex-shrink","width","height","min-width","min-height","max-width","max-height","padding","padding-top","padding-right","padding-bottom","padding-left","margin","margin-top","margin-right","margin-bottom","margin-left","border","border-radius","border-color","border-width","border-style","background","background-color","background-image","background-size","color","font-size","font-weight","font-family","line-height","letter-spacing","text-align","text-decoration","text-transform","box-shadow","opacity","overflow","z-index","grid-template-columns","grid-template-rows","grid-gap","transform","transition"];function Xt(e){const t=window.getComputedStyle(e),n={};for(const o of Gt){const r=t.getPropertyValue(o);r&&r!=="none"&&r!=="normal"&&r!=="auto"&&r!=="0px"&&r!=="rgba(0, 0, 0, 0)"&&(n[o]=r)}return n}function Yt(e){const t=e.tagName.toLowerCase(),n=e.className&&typeof e.className=="string"?"."+e.className.trim().split(/\s+/).slice(0,2).join("."):"";return t+n}function Kt(){const e=document.documentElement.cloneNode(!0),t=["script","noscript",'iframe[src*="ads"]','iframe[src*="track"]','iframe[src*="pixel"]','iframe[width="0"]','iframe[height="0"]','img[src*="pixel"]','img[src*="track"]','img[width="1"]','img[height="1"]','[id*="cookie"]','[class*="cookie"]','[id*="consent"]','[class*="consent"]','[id*="gdpr"]','[class*="gdpr"]','[id*="onetrust"]','[class*="onetrust"]','[id*="CybotCookiebot"]','[data-testid*="cookie"]','[id*="ad-"]','[class*="ad-container"]','[class*="ad-wrapper"]','link[rel="preconnect"]','link[rel="dns-prefetch"]','meta[http-equiv="Content-Security-Policy"]',"style[data-emotion]"];for(const E of t)try{e.querySelectorAll(E).forEach(a=>a.remove())}catch{}e.querySelectorAll("*").forEach(E=>{const a=E.getAttributeNames();for(const s of a)(s.startsWith("on")||s==="data-analytics"||s==="data-tracking")&&E.removeAttribute(s)});const n=e.outerHTML,o={},r=["body","header","nav","main","footer","aside","section","article","h1","h2","h3","h4","h5","h6","p","a","button","input","textarea","ul","ol","li","img","form","table","th","td",'[class*="hero"]','[class*="card"]','[class*="btn"]','[class*="nav"]','[class*="header"]','[class*="footer"]','[class*="sidebar"]','[class*="container"]','[class*="wrapper"]','[class*="grid"]','[class*="flex"]','[class*="modal"]','[class*="banner"]'];let i=0;for(const E of r){if(i>=200)break;try{const a=document.querySelectorAll(E);for(const s of a){if(i>=200)break;const C=Xt(s);if(Object.keys(C).length>0){const x=Yt(s),y=o[x]?`${x}:nth(${i})`:x;o[y]=C,i++}}}catch{}}const p=[];for(const[E,a]of Object.entries(o)){p.push(`${E} {`);for(const[s,C]of Object.entries(a))p.push(`  ${s}: ${C};`);p.push("}"),p.push("")}const d=p.join(`
`),l=new Set,u=["color","background-color","border-color","outline-color"],k=document.querySelectorAll("*");let b=0;for(const E of k){if(b>=500)break;const a=window.getComputedStyle(E);for(const s of u){const C=a.getPropertyValue(s);C&&C!=="rgba(0, 0, 0, 0)"&&C!=="transparent"&&C!=="inherit"&&C!=="initial"&&l.add(C)}b++}const m=Array.from(l).slice(0,50),h=new Set;for(const E of k){if(h.size>=20)break;const s=window.getComputedStyle(E).getPropertyValue("font-family");if(s){const C=s.split(",").map(x=>x.trim().replace(/^["']|["']$/g,""));for(const x of C)x&&!x.includes("inherit")&&!x.includes("initial")&&x.length<50&&h.add(x)}}const c=Array.from(h).slice(0,20),g=E=>{const a=document.querySelector(`meta[property="${E}"], meta[name="${E}"]`);return(a==null?void 0:a.getAttribute("content"))||""},S=document.querySelector('link[rel="icon"], link[rel="shortcut icon"]'),T={title:document.title||"",description:g("description"),og_image:g("og:image"),og_title:g("og:title"),og_description:g("og:description"),og_type:g("og:type"),og_site_name:g("og:site_name"),favicon:(S==null?void 0:S.getAttribute("href"))||""},A={width:window.innerWidth,height:window.innerHeight},f=[];try{for(const E of document.styleSheets){try{const a=E.cssRules||E.rules;if(!a)continue;for(const s of a)if(s instanceof CSSMediaRule&&s.conditionText&&(f.includes(s.conditionText)||f.push(s.conditionText),f.length>=20))break}catch{}if(f.length>=20)break}}catch{}const N=[];document.querySelectorAll("img[src]").forEach(E=>{const a=E.getAttribute("src");if(a&&!a.startsWith("data:")&&N.length<50)try{N.push(new URL(a,location.href).href)}catch{N.push(a)}});const L=[];document.querySelectorAll('link[rel="stylesheet"][href]').forEach(E=>{const a=E.getAttribute("href");if(a&&L.length<20)try{L.push(new URL(a,location.href).href)}catch{L.push(a)}});const M=document.querySelectorAll("*").length;return{html:n,css:d,css_map:o,colors:m,fonts:c,metadata:T,source_url:location.href,viewport:A,asset_urls:{images:N,stylesheets:L},responsive_info:{viewport_width:A.width,media_queries:f},element_count:M}}function Qt(e,t){e.runtime.onMessage.addListener((n,o,r)=>{if(n.type==="EXTRACT_PAGE")return r(t.extractMainContent()),!1;if(n.type==="GET_DOM")return r({ok:!0,dom:t.serializeDOM(),url:location.href,title:document.title}),!1;if(n.type==="EXEC_ACTION")return r(t.execAction(n.action)),!1;if(n.type==="FILL_FORM"){const i=n.fields;let p=0;for(const d of i||[])t.execAction({action:"type",selector:d.selector,text:d.value}).ok&&p++;return r({ok:!0,filled:p,total:(i==null?void 0:i.length)||0}),!1}if(n.type==="SHOW_OCR_OVERLAY")return t.showOcrOverlay(n.dataUrl,r),!0;if(n.type==="PAGE_TRANSLATE")return t.translateActive&&t.removePageTranslation(),t.startPageTranslation(n.targetLang).then(()=>{r({ok:!0})}).catch(i=>{r({ok:!1,error:i.message})}),!0;if(n.type==="TRANSLATE_TOGGLE_MODE")return t.setTranslateMode(n.mode),r({ok:!0}),!1;if(n.type==="TRANSLATE_REMOVE")return t.removePageTranslation(),r({ok:!0}),!1;if(n.type==="TRANSLATE_CHANGE_LANG")return t.translateActive?(t.removePageTranslation(),t.startPageTranslation(n.targetLang).then(()=>{r({ok:!0})}).catch(i=>{r({ok:!1,error:i.message})}),!0):(r({ok:!0}),!1);if(n.type==="SCROLL_TO_HIGHLIGHT")return t.scrollToHighlight(n.id),r({ok:!0}),!1;if(n.type==="SHOW_DOCK")return t.showDock(),r({ok:!0}),!1;if(n.type==="START_CAPTURE_MODE")return t.startCaptureMode(),r({ok:!0}),!1;if(n.type==="STOP_CAPTURE_MODE")return t.stopCaptureMode(),r({ok:!0}),!1;if(n.type==="EXTRACT_FULL_PAGE"){try{const i=t.extractFullPageData();r({ok:!0,data:i})}catch(i){r({ok:!1,error:i.message||"Extraction failed"})}return!1}})}const fe=typeof browser<"u"?browser:chrome,Jt=5*60*1e3,Zt=48*60*60*1e3,en=10*60*1e3,tn=3,nn=3,on=2,rn=1e4,qe=2e3;function an(e){try{const t=new URL(e);return`${t.protocol}//${t.host}${t.pathname}`}catch{return e}}let Fe=0;function sn(){return location.protocol==="http:"||location.protocol==="https:"}function Se(e,t={}){const n=Date.now();if(!(n-Fe<Jt)){Fe=n;try{fe.runtime.sendMessage({type:"STUCK_SIGNAL",kind:e,url:location.href,title:document.title,...t}).catch(()=>{})}catch{}}}async function ln(e){const t=an(e);try{const o=(await fe.storage.local.get(["aura_visit_log"])).aura_visit_log||{},r=Date.now(),i=o[t]||{ts:[]};i.ts=i.ts.filter(l=>r-l<Zt);const p=i.ts[i.ts.length-1]||0;if(!(r-p<en)){i.ts.push(r),o[t]=i;for(const u of Object.keys(o))o[u].ts.length||delete o[u];const l=Object.keys(o);if(l.length>qe){const u=l.sort((k,b)=>(o[k].ts[0]||0)-(o[b].ts[0]||0));for(const k of u.slice(0,l.length-qe))delete o[k]}await fe.storage.local.set({aura_visit_log:o})}i.ts.length>=tn&&Se("tab_revisit",{count:i.ts.length})}catch{}}const je=new WeakMap;let le=null;function cn(){if(!(typeof IntersectionObserver>"u"))try{le=new IntersectionObserver(n=>{for(const o of n){if(!o.isIntersecting)continue;const r=(je.get(o.target)||0)+1;if(je.set(o.target,r),r>=nn){const i=(o.target.textContent||"").slice(0,140);Se("reread",{snippet:i}),le==null||le.unobserve(o.target)}}},{threshold:.6,rootMargin:"0px"});const e=document.querySelectorAll("p, article, section, blockquote");let t=0;for(const n of Array.from(e))if((n.textContent||"").trim().length>=120&&(le.observe(n),t++,t>=200))break}catch{}}const Ue=new WeakMap;function dn(){const e=t=>{const n=t.target;if(!n||!("value"in n)||n.type==="password"||n.type==="hidden")return;const o=Ue.get(n)||{lastEdit:0,emptyAt:[]},r=Date.now();if(n.value===""){if(o.emptyAt=o.emptyAt.filter(i=>r-i<rn),o.emptyAt.push(r),o.emptyAt.length>=on){const i=n.getAttribute("name")||n.getAttribute("id")||"field";Se("form_flipflop",{field:i}),o.emptyAt=[]}}else o.lastEdit=r;Ue.set(n,o)};document.addEventListener("input",e,{capture:!0,passive:!0})}function un(){window.addEventListener("beforeunload",()=>{if(document.querySelector('textarea:focus, [contenteditable="true"]:focus'))try{fe.runtime.sendMessage({type:"STUCK_SIGNAL",kind:"workflow_boundary_unsaved",url:location.href,title:document.title})}catch{}})}function pn(){if(!sn())return;ln(location.href);const e=()=>{cn(),dn(),un()};document.readyState==="complete"?setTimeout(e,500):window.addEventListener("load",()=>setTimeout(e,500),{once:!0})}const Te=typeof browser<"u"?browser:chrome,gn=800,mn=3,fn=2e3,hn=200,bn="aura-ghost-chip";let F=null,ee=null,ae=!0;const xn=["accounts.google.com","login.microsoftonline.com","okta.com","1password.com","lastpass.com","bitwarden.com","dashlane.com","mail.google.com","outlook.live.com","outlook.office.com","mail.yahoo.com","mail.proton.me"];function yn(){const e=location.hostname;return!!(xn.some(t=>e===t||e.endsWith("."+t))||/\bbank\b|\bpayment\b|\bcheckout\b/i.test(e))}function wn(){if(ee&&document.body.contains(ee))return ee;const e=document.createElement("div");return e.id=bn,e.setAttribute("role","tooltip"),e.setAttribute("aria-live","polite"),e.style.cssText=["position:fixed","z-index:2147483646","pointer-events:none","max-width:480px","padding:6px 10px","border-radius:8px","background:rgba(20,20,28,0.92)","color:rgba(220,220,235,0.92)",'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif',"font-size:12px","line-height:1.4","box-shadow:0 6px 24px rgba(0,0,0,0.35), 0 0 0 1px rgba(124,58,237,0.35)","opacity:0","transform:translateY(-4px)","transition:opacity 0.12s ease, transform 0.12s ease","display:none"].join(";"),document.body.appendChild(e),ee=e,e}function ne(){ee&&(ee.style.opacity="0",ee.style.transform="translateY(-4px)",ee.style.display="none",ee.textContent=""),F&&(F.suggestion="")}function vn(e,t){const n=wn();n.textContent="";const o=document.createElement("span");o.textContent="Aura · Tab to accept",o.style.cssText="color:rgba(167,139,250,0.85);margin-right:8px;font-size:10.5px;text-transform:uppercase;letter-spacing:0.05em",n.appendChild(o);const r=document.createElement("span");r.textContent=t,n.appendChild(r);const i=e.getBoundingClientRect(),p=Math.max(8,window.innerWidth-500),d=Math.min(Math.max(8,i.left),p),l=Math.min(i.bottom+6,window.innerHeight-40);n.style.left=`${d}px`,n.style.top=`${l}px`,n.style.display="block",n.offsetHeight,n.style.opacity="1",n.style.transform="translateY(0)"}function We(e){return e instanceof HTMLTextAreaElement||e instanceof HTMLInputElement?e.value||"":e.textContent||""}function En(e,t){if(e instanceof HTMLTextAreaElement||e instanceof HTMLInputElement){const n=e.selectionStart??e.value.length;e.value=e.value.slice(0,n)+t+e.value.slice(n);const o=n+t.length;try{e.setSelectionRange(o,o)}catch{}e.dispatchEvent(new Event("input",{bubbles:!0}))}else if(e.isContentEditable){const n=window.getSelection();if(n&&n.rangeCount>0){const o=n.getRangeAt(0),r=document.createTextNode(t);o.insertNode(r),o.setStartAfter(r),o.setEndAfter(r),n.removeAllRanges(),n.addRange(o),e.dispatchEvent(new Event("input",{bubbles:!0}))}}ne(),F=null}async function Cn(e,t){if(!F)return;F.abortCtrl&&F.abortCtrl.abort();const n=new AbortController;F.abortCtrl=n;try{const o=await Te.runtime.sendMessage({type:"GHOST_COMPLETE",text:t.slice(-fn),url:location.href,title:document.title});if(!o||typeof o.continuation!="string")return;const r=String(o.continuation||"").trim();if(!r||r.length>hn)return;F&&F.el===e&&F.text===t&&(F.suggestion=r,vn(e,r))}catch{}}function kn(e){!F||F.el!==e||(F.debounceTimer&&window.clearTimeout(F.debounceTimer),F.debounceTimer=window.setTimeout(()=>{if(!F||F.el!==e)return;const t=We(e);F.text=t,!(t.length<mn)&&Cn(e,t)},gn))}function Sn(e){return!e||!(e instanceof HTMLElement)?!1:e instanceof HTMLInputElement?!(["password","email","tel","number","date","url"].includes(e.type)||e.getAttribute("autocomplete")==="off"):e instanceof HTMLTextAreaElement?e.getAttribute("autocomplete")!=="off":e.isContentEditable}function Tn(e){if(!ae||yn()||!Sn(e.target))return;const t=e.target;F={el:t,text:We(t),suggestion:"",debounceTimer:null,abortCtrl:null},ne(),kn(t)}function Ln(e){if(F){if(e.key==="Tab"&&!e.shiftKey&&F.suggestion){e.preventDefault(),En(F.el,F.suggestion);return}F.suggestion&&ne()}}function Ve(){ne()}function Mn(){var e,t,n,o;if(!(location.protocol!=="http:"&&location.protocol!=="https:")){try{(t=(e=Te.storage)==null?void 0:e.local)==null||t.get(["ghostTextMode"],r=>{const i=r==null?void 0:r.ghostTextMode;(i==="inline"||i==="off")&&(ae=!1)}),(o=(n=Te.storage)==null?void 0:n.onChanged)==null||o.addListener((r,i)=>{i==="local"&&r.ghostTextMode&&(ae=r.ghostTextMode.newValue==="chip",ae||ne())})}catch{}document.addEventListener("input",Tn,{capture:!0,passive:!0}),document.addEventListener("keydown",Ln,{capture:!0}),window.addEventListener("scroll",Ve,{capture:!0,passive:!0}),window.addEventListener("resize",Ve,{passive:!0}),window.addEventListener("blur",ne)}}window.addEventListener("aura-ghost-disable",()=>{ae=!1,ne()}),window.addEventListener("aura-ghost-enable",()=>{ae=!0});const Le=typeof browser<"u"?browser:chrome,An=600,Nn=8,_n=2e3,In=3,On="aura-ghost-inline",$n="aura-ghost-mirror",Hn=["accounts.google.com","login.microsoftonline.com","okta.com","1password.com","lastpass.com","bitwarden.com","dashlane.com","mail.google.com","outlook.live.com","outlook.office.com","mail.yahoo.com","mail.proton.me"];function zn(){const e=location.hostname;return!!(Hn.some(t=>e===t||e.endsWith("."+t))||/\bbank\b|\bpayment\b|\bcheckout\b/i.test(e))}let q=null,oe=null,he=null,Ge=!0,be="inline",Xe=null,xe=null;function Rn(){if(xe)return xe;try{return Xe=document.createElement("canvas"),xe=Xe.getContext("2d"),xe}catch{return null}}function Me(){if(oe&&document.body.contains(oe))return oe;const e=document.createElement("div");return e.id=On,e.style.cssText=["position:fixed","z-index:2147483646","pointer-events:none","color:rgba(167,139,250,0.55)","font-family:inherit","white-space:pre-wrap","overflow:hidden","display:none"].join(";"),document.body.appendChild(e),oe=e,e}function Bn(){if(he&&document.body.contains(he))return he;const e=document.createElement("div");return e.id=$n,e.style.cssText=["position:absolute","top:0","left:-9999px","visibility:hidden","white-space:pre-wrap","word-wrap:break-word","overflow-wrap:break-word","pointer-events:none"].join(";"),document.body.appendChild(e),he=e,e}function Pn(e,t){const n=window.getComputedStyle(e),o=["boxSizing","width","height","overflow","fontFamily","fontSize","fontWeight","fontStyle","fontVariant","letterSpacing","textTransform","textIndent","lineHeight","paddingTop","paddingRight","paddingBottom","paddingLeft","borderTopWidth","borderRightWidth","borderBottomWidth","borderLeftWidth","borderTopStyle","borderRightStyle","borderBottomStyle","borderLeftStyle"];for(const r of o)t.style[r]=n[r]}function Dn(e,t){try{const n=Bn(),o=e.getBoundingClientRect();Pn(e,n),n.style.width=`${o.width}px`,n.style.height="auto";const r=e.selectionStart??e.value.length,i=e.value.substring(0,r).replace(/\n$/,`
 `);n.textContent=i;const p=document.createElement("span");p.textContent="​",n.appendChild(p);const d=p.getBoundingClientRect(),l=n.getBoundingClientRect(),u=d.left-l.left,k=d.top-l.top,b=o.left+u-e.scrollLeft,m=o.top+k-e.scrollTop;if(b<o.left||b>o.right||m<o.top||m>o.bottom)return!1;const h=Me(),c=window.getComputedStyle(e);return h.style.font=c.font,h.style.fontSize=c.fontSize,h.style.fontFamily=c.fontFamily,h.style.lineHeight=c.lineHeight,h.style.letterSpacing=c.letterSpacing,h.style.left=`${b}px`,h.style.top=`${m}px`,h.style.maxWidth=`${o.right-b-4}px`,h.textContent=t,h.style.display="",!0}catch{return!1}}function qn(e){var t;try{const n=window.getSelection();if(!n||n.rangeCount===0)return!1;const o=n.getRangeAt(0).cloneRange();o.collapse(!1);const r=document.createElement("span");r.textContent="​",o.insertNode(r);const i=r.getBoundingClientRect(),p=r.parentNode;if(p==null||p.removeChild(r),i.width===0&&i.height===0)return!1;const d=Me(),l=(t=n.anchorNode)==null?void 0:t.parentElement;if(l){const u=window.getComputedStyle(l);d.style.font=u.font,d.style.fontSize=u.fontSize,d.style.fontFamily=u.fontFamily,d.style.lineHeight=u.lineHeight,d.style.letterSpacing=u.letterSpacing}return d.style.left=`${i.left}px`,d.style.top=`${i.top}px`,d.style.maxWidth=`${Math.max(240,window.innerWidth-i.left-20)}px`,d.textContent=e,d.style.display="",!0}catch{return!1}}function Fn(e,t){try{const n=e.getBoundingClientRect(),o=window.getComputedStyle(e),r=Rn();if(!r)return!1;r.font=o.font;const i=e.selectionStart??e.value.length,p=r.measureText(e.value.substring(0,i)).width,d=parseFloat(o.paddingLeft)||0,l=parseFloat(o.borderLeftWidth)||0,u=n.left+l+d+p-e.scrollLeft,k=n.top+(n.height-parseFloat(o.fontSize))/2;if(u>n.right-8)return!1;const b=Me();return b.style.font=o.font,b.style.fontSize=o.fontSize,b.style.fontFamily=o.fontFamily,b.style.letterSpacing=o.letterSpacing,b.style.left=`${u}px`,b.style.top=`${k}px`,b.style.maxWidth=`${n.right-u-6}px`,b.textContent=t,b.style.display="",!0}catch{return!1}}function Ye(e,t){return e.tagName==="TEXTAREA"?Dn(e,t):e.tagName==="INPUT"?Fn(e,t):e.isContentEditable?qn(t):!1}function te(){oe&&(oe.style.display="none",oe.textContent=""),q&&(q.suggestion="")}function jn(e){return e.tagName==="TEXTAREA"||e.tagName==="INPUT"?e.value||"":e.isContentEditable&&e.textContent||""}function Un(e){return(e.trim().match(/\S+/g)||[]).length}function Wn(e){if(!(e instanceof HTMLElement))return!1;if(e.tagName==="INPUT"){const t=(e.type||"").toLowerCase();if(["password","email","tel","number","search","url","hidden"].includes(t)||e.getAttribute("autocomplete")==="off")return!1}return!!(e.tagName==="TEXTAREA"||e.isContentEditable||e.tagName==="INPUT")}async function Vn(e,t){return new Promise(n=>{try{Le.runtime.sendMessage({type:"GHOST_COMPLETE",text:e,url:location.href,title:document.title},o=>{var i,p;if(t.aborted){n(null);return}const r=((p=(i=o==null?void 0:o.continuation)==null?void 0:i.trim)==null?void 0:p.call(i))||"";n(r||null)})}catch{n(null)}})}function Gn(e){var o;if(!Ge||be!=="inline")return;const t=e.target;if(!Wn(t))return;const n=jn(t);if(n.length<Nn||n.length>_n){te();return}if(Un(n)<In){te();return}q!=null&&q.debounceTimer&&clearTimeout(q.debounceTimer),(o=q==null?void 0:q.abortCtrl)==null||o.abort(),q={el:t,text:n,suggestion:"",debounceTimer:null,abortCtrl:null},q.debounceTimer=window.setTimeout(async()=>{if(!q||q.el!==t)return;const r=new AbortController;q.abortCtrl=r;const i=await Vn(n,r.signal);if(!i||r.signal.aborted){te();return}if(!q||q.el!==t)return;q.suggestion=i,Ye(t,i)||te()},An)}function Xn(e){if(!(!q||!q.suggestion)&&e.target===q.el){if(e.key==="Tab"){e.preventDefault();const t=q.el,n=q.suggestion;if(t.tagName==="TEXTAREA"||t.tagName==="INPUT"){const o=t,r=o.selectionStart??o.value.length,i=o.value.substring(0,r),p=o.value.substring(r);o.value=i+n+p,o.selectionStart=o.selectionEnd=r+n.length,o.dispatchEvent(new Event("input",{bubbles:!0}))}else if(t.isContentEditable){const o=window.getSelection();if(o&&o.rangeCount>0){const r=o.getRangeAt(0);r.deleteContents(),r.insertNode(document.createTextNode(n)),r.collapse(!1)}}te(),q=null;return}e.key==="Escape"&&(te(),q=null)}}function Yn(){te()}function Ke(){q!=null&&q.suggestion&&q.el&&Ye(q.el,q.suggestion)}function Kn(){var e,t;try{(t=(e=Le.storage)==null?void 0:e.local)==null||t.get(["ghostTextMode"],n=>{const o=n==null?void 0:n.ghostTextMode;(o==="chip"||o==="off"||o==="inline")&&(be=o)})}catch{}}function Qn(){var e,t;if(zn()){Ge=!1;return}Kn();try{(t=(e=Le.storage)==null?void 0:e.onChanged)==null||t.addListener((n,o)=>{if(o==="local"&&n.ghostTextMode){const r=n.ghostTextMode.newValue;(r==="chip"||r==="off"||r==="inline")&&(be=r),be!=="inline"&&te()}})}catch{}document.addEventListener("input",Gn,!0),document.addEventListener("keydown",Xn,!0),document.addEventListener("blur",Yn,!0),window.addEventListener("scroll",Ke,!0),window.addEventListener("resize",Ke)}const Ae=typeof browser<"u"?browser:chrome;let ce=!1,de=0,ye=0,ue="",Ne=!1;const Jn=["accounts.google.com","login.microsoftonline.com","okta.com","1password.com","lastpass.com","bitwarden.com","dashlane.com"];function Zn(){const e=location.hostname;return!!(Jn.some(t=>e.endsWith(t))||/\bbank|\bpayment|\bcheckout\b/i.test(e))}function Qe(){return!(location.protocol!=="http:"&&location.protocol!=="https:"||Zn())}function eo(){const e=document.documentElement,t=window.innerHeight,n=Math.max(1,(e.scrollHeight||t)-t),o=window.scrollY,r=Math.round(o/n*100);r>ye&&(ye=r)}function to(){var e;try{const t=((e=window.getSelection())==null?void 0:e.toString())||"";t.length>=20&&t.length<=500&&t.length>ue.length&&(ue=t)}catch{}}function no(){if(!ce||!Qe()||!de)return null;const e=Date.now()-de;if(e<3e3)return null;const t={url:location.href,title:document.title,dwell_ms:e,scroll_max_pct:ye,timestamp:Date.now()};return ue&&(t.selection=ue),t}function _e(){if(Ne)return;const e=no();if(e){Ne=!0;try{Ae.runtime.sendMessage({type:"LIFELOG_EVENT",event:e}).catch(()=>{})}catch{}}}async function oo(){try{const e=await Ae.storage.local.get(["lifelogEnabled"]);ce=!!(e!=null&&e.lifelogEnabled)}catch{}}function ro(){var e,t;Qe()&&(oo().then(()=>{ce&&(de=Date.now(),ye=0,ue="",Ne=!1,window.addEventListener("scroll",eo,{passive:!0}),document.addEventListener("selectionchange",to,{passive:!0}),document.addEventListener("visibilitychange",()=>{document.visibilityState==="hidden"&&_e()}),window.addEventListener("pagehide",_e),window.addEventListener("beforeunload",_e))}),(t=(e=Ae.storage)==null?void 0:e.onChanged)==null||t.addListener((n,o)=>{o!=="local"||!n.lifelogEnabled||(ce=!!n.lifelogEnabled.newValue,ce&&!de&&(de=Date.now()))}))}const pe=typeof browser<"u"?browser:chrome,io=60*60*1e3,ao=2500,so=["accounts.google.com","login.microsoftonline.com","okta.com","1password.com","lastpass.com","bitwarden.com","dashlane.com","mail.google.com","outlook.live.com","outlook.office.com","mail.yahoo.com","mail.proton.me"];function lo(){const e=location.hostname;return!!(so.some(t=>e===t||e.endsWith("."+t))||/\bbank\b|\bpayment\b|\bcheckout\b/i.test(e))}function Je(){return!(location.protocol!=="http:"&&location.protocol!=="https:"||lo())}function Ze(){try{const e=new URL(location.href);return e.origin+e.pathname}catch{return location.href}}async function co(){var e,t;try{const n=await((t=(e=pe.storage)==null?void 0:e.local)==null?void 0:t.get(["ambientSurfaceLog"])),o=(n==null?void 0:n.ambientSurfaceLog)||{},r=Ze(),i=o[r];return i?Date.now()-i<io:!1}catch{return!0}}async function uo(){var e,t,n,o;try{const r=await((t=(e=pe.storage)==null?void 0:e.local)==null?void 0:t.get(["ambientSurfaceLog"])),i=(r==null?void 0:r.ambientSurfaceLog)||{};i[Ze()]=Date.now();const p=Date.now()-24*60*60*1e3;for(const d in i)i[d]<p&&delete i[d];(o=(n=pe.storage)==null?void 0:n.local)==null||o.set({ambientSurfaceLog:i})}catch{}}async function po(){if(!Je()||await co())return;const e=document.title||"";if(e.trim())try{pe.runtime.sendMessage({type:"AMBIENT_SURFACE_REQUEST",url:location.href,title:e,host:location.hostname}),uo()}catch{}}function go(){var e,t;if(Je())try{(t=(e=pe.storage)==null?void 0:e.local)==null||t.get(["ambientSurfaceEnabled"],n=>{n!=null&&n.ambientSurfaceEnabled&&setTimeout(()=>{document.visibilityState==="visible"&&po()},ao)})}catch{}}const X=typeof browser<"u"?browser:chrome,et=["aura-shadow-host","aura-dock-shadow","aura-host","aura-quick-action-host","aura-highlight-host","aura-img-toolbar-host","aura-capture-host"];let we=null;function ge(e,t){var n;try{t?X.runtime.sendMessage(e,t):X.runtime.sendMessage(e)}catch(o){if(((o==null?void 0:o.message)??"").includes("Extension context invalidated")){we==null||we.remove();for(const i of et)(n=document.getElementById(i))==null||n.remove()}}}function tt(e,t=2e3){const n=document.createElement("div");Object.assign(n.style,{position:"fixed",top:"16px",left:"50%",transform:"translateX(-50%)",background:"rgba(10,8,24,0.92)",backdropFilter:"blur(16px)",WebkitBackdropFilter:"blur(16px)",border:"1px solid rgba(124,58,237,0.35)",borderRadius:"8px",padding:"8px 16px",color:"rgba(226,232,240,0.92)",fontSize:"13px",fontFamily:'-apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", system-ui, sans-serif',fontWeight:"500",zIndex:"2147483647",pointerEvents:"none",boxShadow:"0 4px 16px rgba(0,0,0,0.4)",whiteSpace:"nowrap"}),n.textContent=e,document.documentElement.appendChild(n),setTimeout(()=>n.remove(),t)}function mo(){var N;if(window.__auraToolbarMounted)return;window.__auraToolbarMounted=!0;for(const L of et)(N=document.getElementById(L))==null||N.remove();const e=document.createElement("div");e.id="aura-shadow-host",Object.assign(e.style,{position:"fixed",top:"0",left:"0",width:"0",height:"0",zIndex:"2147483647",pointerEvents:"none",overflow:"visible"}),document.documentElement.appendChild(e),we=e;const t=e.attachShadow({mode:"open"}),n=document.createElement("style");n.textContent=lt(),t.appendChild(n);const o=ot();st(o,X),o.subscribe(L=>{e.style.setProperty("--aura-accent",L.accent),e.style.setProperty("--aura-glow",L.glow)});function r(L){const M=document.createElement("div");return M.dataset.auraModule=L,Object.assign(M.style,{all:"unset",pointerEvents:"none"}),t.appendChild(M),M}const i=r("fab"),p=r("ghost-bar"),d=r("modal"),l=r("highlights"),u=r("capture"),k=r("link-preview"),b=pt(),m=mt(),h=yt(),c=wt(),g=Et(),S=kt(),T=Tt();b.init(i,o,X),m.init(p,o,X),h.init(d,o,X),c.init(l,o,X),g.init(document.body,o,X),S.init(u,o,X),T.init(k,o,X),m.onAskClicked(L=>{L.type==="text"?h.openWithText(L.text,L.rect):h.openWithImage(L.imageUrl,L.rect)}),h.onAction((L,M,E)=>{ge({type:"OPEN_WITH_TEXT",action:L,text:M,url:location.href,title:document.title})}),c.setShowToast(tt);const A=Vt(X);let f=!1;qt(X,ge,tt),Ft(ge),jt(ge),Lt(X,ge),pn(),Mn(),Qn(),ro(),go(),Qt(X,{extractMainContent:Bt,serializeDOM:At,execAction:Nt,showOcrOverlay:_t,startPageTranslation:async L=>{await A.start(L),f=!0},removePageTranslation:()=>{A.remove(),f=!1},setTranslateMode:L=>A.setMode(L),scrollToHighlight:L=>c.scrollTo(L),showDock:()=>b.showDock(),startCaptureMode:()=>S.start(),stopCaptureMode:()=>S.stop(),extractFullPageData:Kt,get translateActive(){return f}})}mo()})();
