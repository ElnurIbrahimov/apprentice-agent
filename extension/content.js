(function(){"use strict";const ke={general:{accent:"#7c3aed",glow:"rgba(124, 58, 237, 0.35)",icon:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
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
    </svg>`}},Se={general:["ask","summarize","explain","translate","save","copy"],article:["ask","summarize","highlight","translate","save","explain","copy"],media:["ask","describe","transcript","summarize","translate","save"],code:["ask","explain","review","debug","refactor","copy","save"],email:["ask","summarize","reply","translate","save","action-items"],shopping:["ask","compare","summarize","pros-cons","save","price-history"]},U={morphDuration:350,morphEasing:"cubic-bezier(0.4, 0, 0.0, 1)",flowDuration:500,glowPulse:3e3,sequentialStagger:40,dismissDelay:400,crossFadeDuration:400,selectionDelay:300,imageHoverDelay:800},R={bg:"rgba(10, 8, 24, 0.88)",bgHeavy:"rgba(10, 8, 24, 0.75)",backdrop:"blur(20px) saturate(1.5)",borderOpacity:.25,shadowBase:"0 8px 32px rgba(0,0,0,0.4)"},K={height:28,iconSize:15,imageIconSize:16,imageBarHeight:32,maxActionsPerRow:7},J={maxWidth:520,maxHeight:480,previewMaxLines:6,previewMaxChars:2e3,imagePreviewMaxHeight:200},V={pillPadding:"6px 10px",glowIntensityMin:.15,glowIntensityMax:.35,logoSize:20,expandDuration:220,dragThreshold:4,edgeMargin:12},W="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",Z=2147483647,He=2147483645;function at(){const e=ke.general;return{type:"general",cadence:"engaged",suppressGhostBars:!1,readingProgress:0,actions:Se.general,accent:e.accent,glow:e.glow,icon:e.icon,sessionActions:[]}}function st(){let e=at();const t=new Set;return{get(){return e},subscribe(n){return t.add(n),()=>t.delete(n)},update(n){e={...e,...n};for(const o of t)o(e)}}}function lt(e,t){var c;let n="";try{n=new URL(e).hostname.replace(/^www\./,"")}catch{}if(n==="github.com"||n==="gitlab.com")return"code";if(n==="youtube.com"||n==="netflix.com")return"media";if(n==="mail.google.com"||n==="outlook.live.com")return"email";if(n==="amazon.com"||n==="ebay.com"||n==="etsy.com"||e.includes("/product/")||e.includes("/cart/"))return"shopping";if(t.querySelector('article, [role="article"]'))return"article";if(t.querySelectorAll("pre, code").length>=3)return"code";if(t.querySelector("video, audio"))return"media";const u=t.querySelectorAll('script[type="application/ld+json"]');for(const l of u)try{const d=JSON.parse(l.textContent??""),S=(c=Array.isArray(d)?d[0]:d)==null?void 0:c["@type"];if(typeof S=="string"&&S.toLowerCase().includes("product"))return"shopping"}catch{}return"general"}function ct(){const e=[],t=[];let l="engaged",d=0;function S(f){const p=f-1e4,T=f-3e4;for(;e.length&&e[0].ts<p;)e.shift();for(;t.length&&t[0]<T;)t.shift()}function x(f){return t.length>=3?"active":e.length>=3&&e.filter(T=>T.velocity>=300).length/e.length>=.6?"passive":"engaged"}function g(f){if(f-d<3e3)return;const p=x();p!==l&&(l=p,d=f)}return{getCadence(){return l},recordScroll(f){const p=Date.now();e.push({ts:p,velocity:Math.abs(f)}),S(p),g(p)},recordSelection(){const f=Date.now();t.push(f),S(f),x()==="active"?(l="active",d=f):g(f)},recordInput(){const f=Date.now();t.push(f),S(f),x()==="active"?(l="active",d=f):g(f)}}}function dt(){let e=0;const t=new Set;let n=0;return{recordAction(o,r){t.add(r)},recordDismissal(){e++},getExtraDelay(){return Math.min(e*200,2e3)},shouldPromoteContinue(o){return t.has(o)},getSessionActions(){const o=[];return n>=2&&o.push("review-highlights"),o},recordHighlight(){n++}}}function ut(e,t){const n=[],o=ct(),r=dt();function i(){const L=location.href,m=lt(L,document),N=ke[m];e.update({type:m,accent:N.accent,glow:N.glow,icon:N.icon,actions:[...Se[m],...r.getSessionActions()],sessionActions:r.getSessionActions()})}const u=L=>{if(typeof requestIdleCallback<"u"){const m=requestIdleCallback(L,{timeout:2e3});n.push(()=>cancelIdleCallback(m))}else{const m=setTimeout(L,200);n.push(()=>clearTimeout(m))}};u(i);const c=()=>u(i);window.addEventListener("popstate",c),n.push(()=>window.removeEventListener("popstate",c));const l=history.pushState.bind(history),d=history.replaceState.bind(history);history.pushState=function(...L){l(...L),c()},history.replaceState=function(...L){d(...L),c()},n.push(()=>{history.pushState=l,history.replaceState=d});let S=null;const x=new MutationObserver(()=>{S&&clearTimeout(S),S=setTimeout(()=>u(i),2e3)});x.observe(document.body,{childList:!0,subtree:!0}),n.push(()=>{x.disconnect(),S&&clearTimeout(S)});let g=window.scrollY,f=Date.now();const p=()=>{const L=Date.now(),m=Math.max(L-f,1),M=Math.abs(window.scrollY-g)/m*1e3;o.recordScroll(M),g=window.scrollY,f=L,e.update({cadence:o.getCadence()})};window.addEventListener("scroll",p,{passive:!0}),n.push(()=>window.removeEventListener("scroll",p));const T=L=>{L.target.matches("input, textarea, [contenteditable]")&&(o.recordInput(),e.update({suppressGhostBars:!0,cadence:o.getCadence()}))},h=L=>{L.target.matches("input, textarea, [contenteditable]")&&e.update({suppressGhostBars:!1})};document.addEventListener("focusin",T),document.addEventListener("focusout",h),n.push(()=>{document.removeEventListener("focusin",T),document.removeEventListener("focusout",h)});const k=()=>{const L=window.getSelection();L&&L.toString().length>0&&(o.recordSelection(),e.update({cadence:o.getCadence()}))};document.addEventListener("selectionchange",k),n.push(()=>document.removeEventListener("selectionchange",k));const A=document.querySelector("article")??document.querySelector("main")??document.querySelector('[role="main"]');if(A){let L=0,m=0;u(()=>{const E=A.getBoundingClientRect();L=E.top+window.scrollY,m=E.bottom+window.scrollY});const M=()=>{const E=window.scrollY+window.innerHeight,s=m-L;if(s<=0)return;const a=Math.min(Math.max((E-L)/s,0),1);e.update({readingProgress:a})};window.addEventListener("scroll",M,{passive:!0}),n.push(()=>window.removeEventListener("scroll",M))}try{const L=t.storage.session;if(L){L.get(["contextType"],M=>{if(M!=null&&M.contextType&&e.get().type==="general"){const E=M.contextType,s=ke[E];e.update({type:E,accent:s.accent,glow:s.glow,icon:s.icon,actions:Se[E]})}});let m=null;const N=e.subscribe(M=>{M.type!==m&&(m=M.type,L.set({contextType:M.type}))});n.push(N)}}catch{}return()=>{for(const L of n)L()}}function pt(){return`
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
  background: ${R.bg};
  backdrop-filter: ${R.backdrop};
  -webkit-backdrop-filter: ${R.backdrop};
  border: 1px solid rgba(255, 255, 255, ${R.borderOpacity});
  border-radius: 20px;
  box-shadow: ${R.shadowBase};
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
  background: ${R.bg};
  backdrop-filter: ${R.backdrop};
  -webkit-backdrop-filter: ${R.backdrop};
  border: 1px solid rgba(255, 255, 255, ${R.borderOpacity});
  border-radius: 12px;
  box-shadow: ${R.shadowBase};
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
  z-index: ${He};
  display: flex;
  align-items: center;
  background: ${R.bg};
  backdrop-filter: ${R.backdrop};
  -webkit-backdrop-filter: ${R.backdrop};
  border: 1px solid rgba(255, 255, 255, ${R.borderOpacity});
  box-shadow: ${R.shadowBase};
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
  background: ${R.bgHeavy};
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
  z-index: ${He};
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: auto;
}

.aura-modal {
  max-width: ${J.maxWidth}px;
  max-height: ${J.maxHeight}px;
  width: 90vw;
  background: ${R.bg};
  backdrop-filter: ${R.backdrop};
  -webkit-backdrop-filter: ${R.backdrop};
  border: 1px solid rgba(255, 255, 255, ${R.borderOpacity});
  border-radius: 16px;
  box-shadow: ${R.shadowBase};
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
  background: ${R.bg};
  backdrop-filter: ${R.backdrop};
  -webkit-backdrop-filter: ${R.backdrop};
  border: 1px solid rgba(255, 255, 255, ${R.borderOpacity});
  border-radius: 8px;
  box-shadow: ${R.shadowBase};
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
  backdrop-filter: ${R.backdrop};
  -webkit-backdrop-filter: ${R.backdrop};
  border: 1px solid rgba(255, 255, 255, ${R.borderOpacity});
  border-radius: 20px;
  box-shadow: ${R.shadowBase};
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
`.trim()}const de="forwards";function se(e){try{e.commitStyles()}catch{}e.cancel()}async function Te(e,t){const n=e.offsetHeight||0,o=[{height:"0px",opacity:0},{height:`${n}px`,opacity:1}],r=t.direction==="down"?o:[...o].reverse(),i=e.animate(r,{duration:t.duration,easing:t.easing,delay:t.delay??0,fill:de});await i.finished,se(i)}async function gt(e,t){const n=e.animate([{opacity:1},{opacity:0}],{duration:t.duration,easing:t.easing,delay:t.delay??0,fill:de});await n.finished,se(n)}async function ze(e,t){const n=e.animate([{opacity:0},{opacity:1}],{duration:t.duration,easing:t.easing,delay:t.delay??0,fill:de});await n.finished,se(n)}async function mt(e,t,n){const o={duration:n.duration,easing:n.easing,delay:n.delay??0,fill:de},r=e.animate([{opacity:1},{opacity:0}],o),i=t.animate([{opacity:0},{opacity:1}],o);await Promise.all([r.finished,i.finished]),se(r),se(i)}async function Re(e,t,n,o){const r=e.animate([{width:`${t.width}px`,height:`${t.height}px`,transform:`translate(${t.left}px, ${t.top}px)`},{width:`${n.width}px`,height:`${n.height}px`,transform:`translate(${n.left}px, ${n.top}px)`}],{duration:o.duration,easing:o.easing,delay:o.delay??0,fill:de});await r.finished,se(r)}const ft=[{action:"chat",tip:"Chat",svg:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M2 2h12a1 1 0 011 1v8a1 1 0 01-1 1H5l-3 3V3a1 1 0 011-1z"/>
    </svg>`},{action:"search",tip:"Search",svg:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M6.5 1a5.5 5.5 0 014.23 9.02l3.12 3.12-1.06 1.06-3.12-3.12A5.5 5.5 0 116.5 1zm0 1.5a4 4 0 100 8 4 4 0 000-8z"/>
    </svg>`},{action:"page",tip:"This Page",svg:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M3 2h10a1 1 0 011 1v10a1 1 0 01-1 1H3a1 1 0 01-1-1V3a1 1 0 011-1zm1 3v1h8V5H4zm0 3v1h8V8H4zm0 3v1h5v-1H4z"/>
    </svg>`},{action:"translate",tip:"Translate",svg:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M1 2h7v1.5H5.5v1H8v1.5H5.5c-.2 1-.7 2-1.5 2.7l1.7 1.8-1.1 1-1.6-1.8C2.5 10 2 10.2 1.5 10.3L1 8.8c.5-.1.9-.3 1.3-.5L1 6.8l1.1-1 1.2 1.4c.5-.5.9-1.1 1.1-1.7H1V2zm10 3l3 8h-1.5l-.6-1.7h-2.8L8.5 13H7l3-8h1zm-.5 2.5l-1 2.8h2l-1-2.8z"/>
    </svg>`},{action:"save",tip:"Save to Memory",svg:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M3 1h10a1 1 0 011 1v13l-6-3-6 3V2a1 1 0 011-1z"/>
    </svg>`}];function ht(){let e=null,t=null,n=null,o=null,r=null,i=null,u=null,c=null,l="right",d=40,S=0,x=0,g=!1,f=0,p=null,T=!1;function h(){const v=document.createElement("div");return v.className="fab-glow",Object.assign(v.style,{position:"absolute",inset:"-8px",borderRadius:"50px",background:"var(--aura-glow)",filter:"blur(12px)",animation:"aura-glow-pulse 3s ease-in-out infinite",pointerEvents:"none",zIndex:"-1"}),v}function k(v){const w=document.createElement("div");w.className="fab-logo",Object.assign(w.style,{width:`${V.logoSize}px`,height:`${V.logoSize}px`,color:"var(--aura-accent)",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:"0",transition:"color 0.3s ease"}),w.innerHTML=v;const I=w.querySelector("svg");return I&&(I.style.width="100%",I.style.height="100%"),w}function A(){const v=document.createElement("button");return v.className="fab-close",v.setAttribute("aria-label","Close Aura"),v.innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 12" fill="currentColor" width="10" height="10">
      <path d="M1 1l10 10M11 1L1 11" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    </svg>`,Object.assign(v.style,{position:"absolute",top:"-6px",right:"-6px",width:"16px",height:"16px",borderRadius:"50%",background:"rgba(10,8,24,0.9)",border:"1px solid rgba(255,255,255,0.15)",color:"rgba(255,255,255,0.6)",cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center",padding:"0",opacity:"0",transition:"opacity 0.2s",pointerEvents:"all"}),v}function L(v){const w=document.createElement("button");w.className="fab-action-btn",w.dataset.action=v.action,w.setAttribute("aria-label",v.tip),w.innerHTML=v.svg,Object.assign(w.style,{display:"flex",flexDirection:"column",alignItems:"center",gap:"4px",background:"transparent",border:"none",color:"rgba(255,255,255,0.75)",cursor:"pointer",padding:"6px 8px",borderRadius:"8px",fontSize:"10px",fontFamily:W,transition:"background 0.15s, color 0.15s"});const I=w.querySelector("svg");I&&(I.setAttribute("width","16"),I.setAttribute("height","16"));const O=document.createElement("span");return O.textContent=v.tip,O.style.fontSize="10px",w.appendChild(O),w.addEventListener("mouseenter",()=>{w.style.background="rgba(255,255,255,0.08)",w.style.color="var(--aura-accent)"}),w.addEventListener("mouseleave",()=>{w.style.background="transparent",w.style.color="rgba(255,255,255,0.75)"}),w}function m(){const v=document.createElement("div");v.className="fab-popout hidden",Object.assign(v.style,{position:"absolute",display:"flex",flexDirection:"row",gap:"4px",padding:"8px",background:"rgba(10,8,24,0.92)",backdropFilter:"blur(20px) saturate(1.5)",border:"1px solid rgba(255,255,255,0.12)",borderRadius:"14px",boxShadow:"0 8px 32px rgba(0,0,0,0.4)",zIndex:String(Z),transition:"opacity 0.2s, transform 0.2s",opacity:"0",pointerEvents:"none"});for(const w of ft)v.appendChild(L(w));return v}function N(v){const w=document.createElement("div");return w.className="fab-pill",Object.assign(w.style,{display:"flex",alignItems:"center",justifyContent:"center",padding:V.pillPadding,background:"rgba(10,8,24,0.88)",backdropFilter:"blur(20px) saturate(1.5)",border:"1px solid rgba(255,255,255,0.12)",borderRadius:"50px",cursor:"pointer",position:"relative",boxShadow:"0 4px 20px rgba(0,0,0,0.3)",transition:`padding ${V.expandDuration}ms ease, border-radius ${V.expandDuration}ms ease`,userSelect:"none",touchAction:"none"}),n=h(),w.appendChild(n),o=k(v.icon),w.appendChild(o),i=A(),w.appendChild(i),w}function M(){if(!e||!t)return;const v=V.edgeMargin;Object.assign(e.style,{position:"fixed",top:`${d}%`,[l==="right"?"right":"left"]:`${v}px`,[l==="right"?"left":"right"]:"auto",zIndex:String(Z),transform:""}),E()}function E(){if(!r||!e)return;const v=l==="right";Object.assign(r.style,{top:"50%",transform:"translateY(-50%)",[v?"right":"left"]:"calc(100% + 8px)",[v?"left":"right"]:"auto"})}function s(){!r||!t||(p&&(clearTimeout(p),p=null),r.classList.remove("hidden"),r.style.opacity="1",r.style.pointerEvents="all",t.style.borderBottomRightRadius="50px",i&&(i.style.opacity="1"))}function a(){!r||!t||(r.style.opacity="0",r.style.pointerEvents="none",p=setTimeout(()=>{r.classList.add("hidden"),i&&(i.style.opacity="0")},200))}function C(v,w){let I=!1,O=!1;function B(){setTimeout(()=>{!I&&!O&&a()},0)}v.addEventListener("mouseenter",()=>{I=!0,s()}),v.addEventListener("mouseleave",()=>{I=!1,B()}),w.addEventListener("mouseenter",()=>{O=!0,s()}),w.addEventListener("mouseleave",()=>{O=!1,B()})}function b(v){v.addEventListener("pointerdown",w=>{w.target.closest(".fab-close")||(S=w.clientX,x=w.clientY,g=!1,f=0,v.setPointerCapture(w.pointerId))}),v.addEventListener("pointermove",w=>{if(!v.hasPointerCapture(w.pointerId))return;const I=w.clientX-S,O=w.clientY-x;if(f=Math.sqrt(I*I+O*O),f>V.dragThreshold){g=!0,v.classList.add("dragging"),v.style.borderRadius="50%";const B=window.innerHeight,q=w.clientY,P=Math.min(Math.max(q/B*100,5),90);e&&(e.style.top=`${P}%`)}}),v.addEventListener("pointerup",w=>{if(!g)return;v.classList.remove("dragging"),v.style.borderRadius="50px";const I=window.innerWidth;l=w.clientX>I/2?"right":"left";const O=window.innerHeight;d=Math.min(Math.max(w.clientY/O*100,5),90),g=!1,M(),z()})}function y(v,w){v.addEventListener("click",I=>{I.target.closest(".fab-close")||g||f>V.dragThreshold||c&&c.runtime.sendMessage({type:"OPEN_PANEL",panel:"chat"})}),i==null||i.addEventListener("click",I=>{I.stopPropagation(),e&&(e.style.display="none")}),w.addEventListener("click",I=>{const O=I.target.closest(".fab-action-btn");if(!O||!c)return;const B=O.dataset.action??"";$(B)})}function _(){const v=window.getSelection();return v?v.toString().trim():""}function $(v){if(!c)return;const w=location.href,I=document.title;switch(v){case"chat":c.runtime.sendMessage({type:"OPEN_PANEL",panel:"chat"});break;case"search":c.runtime.sendMessage({type:"OPEN_PANEL",panel:"search"});break;case"page":c.runtime.sendMessage({type:"OPEN_PANEL",panel:"ask"});break;case"translate":c.runtime.sendMessage({type:"OPEN_PANEL",panel:"translate"});break;case"save":{const B=_()||`${I}
${w}`;c.runtime.sendMessage({type:"SAVE_KNOWLEDGE",text:B,url:w,title:I},q=>{});break}}}function D(v){var w;if(!(!e||!o)&&(e.style.setProperty("--aura-accent",v.accent),e.style.setProperty("--aura-glow",v.glow),!T&&o)){T=!0;const I=o,O=k(v.icon);O.style.position="absolute",O.style.inset="0",O.style.opacity="0",(w=I.parentElement)==null||w.appendChild(O),mt(I,O,{duration:U.crossFadeDuration,easing:"ease"}).then(()=>{I.remove(),O.style.position="",O.style.inset="",O.style.opacity="1",o=O,T=!1})}}function H(){c&&c.storage.local.get(["auraFabSide","auraFabOffset"],v=>{(v.auraFabSide==="left"||v.auraFabSide==="right")&&(l=v.auraFabSide),typeof v.auraFabOffset=="number"&&(d=v.auraFabOffset),M()})}function z(){c&&c.storage.local.set({auraFabSide:l,auraFabOffset:d})}return{init(v,w,I){c=I;const O=w.get(),B=document.createElement("div");B.className="aura-fab",Object.assign(B.style,{position:"fixed",zIndex:String(Z),fontFamily:W,"--aura-accent":O.accent,"--aura-glow":O.glow}),e=B;const q=N(O);t=q,B.appendChild(q);const P=m();r=P,B.appendChild(P),v.appendChild(B),C(q,P),b(q),y(q,P),H(),u=w.subscribe(D)},destroy(){u&&(u(),u=null),p&&(clearTimeout(p),p=null),e==null||e.remove(),e=null,t=null,n=null,o=null,r=null,i=null,c=null},showDock(){e&&(e.style.display="")}}}const Be={ask:`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
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
  </svg>`};function bt(e){return Be[e]??Be.more}function Pe(e,t,n,o,r){Object.assign(e.style,{position:"fixed",left:`${t.left}px`,top:`${n}px`,width:`${t.width}px`,height:`${o}px`,background:r,backdropFilter:R.backdrop,WebkitBackdropFilter:R.backdrop,border:`1px solid rgba(255,255,255,${R.borderOpacity})`,boxShadow:R.shadowBase,borderRadius:"6px",display:"flex",alignItems:"center",gap:"2px",padding:"0 6px",overflow:"hidden",boxSizing:"border-box",zIndex:String(Z),userSelect:"none"})}function be(e){const t=document.createElement("button");return t.className="gb-action",t.dataset.action=e,t.title=e,t.innerHTML=bt(e),Object.assign(t.style,{background:"none",border:"none",cursor:"pointer",color:"rgba(255,255,255,0.85)",padding:"3px",display:"flex",alignItems:"center",justifyContent:"center",borderRadius:"4px",flexShrink:"0"}),t}function xt(){let e=null,t=null,n="",o="",r=null,i=document.body,u=null,c=null,l=null,d=null,S=null,x=null,g=null;const f=[];function p(){e&&(e.remove(),e=null,t=null,r=null)}async function T(){if(!e)return;const a=e;e=null,t=null,r=null;try{await Te(a,{direction:"up",duration:U.morphDuration,easing:U.morphEasing})}catch{}a.remove()}function h(a){var b;if(!c)return;if(a==="ask"){if(l){const y={type:t==="image"?"image":"text",text:n,imageUrl:o,rect:e?e.getBoundingClientRect():r??new DOMRect};l(y)}return}if(a==="copy"){(b=navigator.clipboard)==null||b.writeText(n).catch(()=>{});return}if(a==="highlight"){c.runtime.sendMessage({type:"SAVE_KNOWLEDGE",text:n,url:location.href,title:document.title});return}if(a==="describe"){c.runtime.sendMessage({type:"IMAGE_DESCRIBE",imageUrl:o});return}if(a==="edit"){c.runtime.sendMessage({type:"IMAGE_EDIT_OPEN",imageUrl:o});return}if(a==="save"){c.runtime.sendMessage({type:"IMAGE_SAVE",imageUrl:o});return}const C={type:"QUICK_ACTION",action:a,text:n};c.runtime.sendMessage(C)}function k(a){a.addEventListener("click",C=>{const b=C.target.closest(".gb-action");if(!b)return;const y=b.dataset.action??"";if(y==="more"){const _=a.querySelector(".gb-extended");_&&(_.style.display=_.style.display==="none"?"flex":"none");return}h(y)})}function A(a){const C=a.trim();return/^(https?:\/\/|www\.)\S+$/.test(C)?"url":/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(C)?"email":[/[{}\[\]();]/.test(C),/\b(function|const|let|var|def|class|import|return|if|for|while)\b/.test(C),/=>|->|::/.test(C),/^\s{2,}/m.test(C)].filter(Boolean).length>=2?"code":"text"}function L(a,C){if(!u)return;const b=u.get();if(b.suppressGhostBars)return;p(),n=C,o="",r=a,t="text";const y=document.createElement("div");y.className="ghost-bar ghost-bar-text";const _=a.bottom;Pe(y,a,_,K.height,R.bg);const $=A(C),D={code:["explain","ask","copy"],url:["ask","summarize","copy"],email:["ask","copy"],text:[]},z=(D[$].length?D[$]:b.actions).slice(0,K.maxActionsPerRow-1);for(const O of z)y.appendChild(be(O));const v=be("more");y.appendChild(v);const w=document.createElement("div");w.className="gb-extended";const I=["rewrite","grammar","define","read-aloud"];for(const O of I)w.appendChild(be(O));Object.assign(w.style,{display:"none",position:"absolute",top:`${K.height}px`,left:"0",right:"0",background:R.bg,borderRadius:"0 0 6px 6px",padding:"2px 6px",gap:"2px"}),y.style.position="fixed",y.appendChild(w),k(y),i.appendChild(y),e=y,Te(y,{direction:"down",duration:U.morphDuration,easing:U.morphEasing}).catch(()=>{})}function m(a){if(!u||u.get().suppressGhostBars)return;p();const b=a.getBoundingClientRect();n="",o=a.src??a.currentSrc??"",r=b,t="image";const y=document.createElement("div");y.className="ghost-bar ghost-bar-image";const _=b.bottom-K.imageBarHeight;Pe(y,b,_,K.imageBarHeight,R.bgHeavy);const $=["describe","edit","save","ask"];for(const D of $){const H=be(D);H.style.width=`${K.imageIconSize+8}px`,H.style.height=`${K.imageIconSize+8}px`,y.appendChild(H)}k(y),i.appendChild(y),e=y,Te(y,{direction:"down",duration:U.morphDuration,easing:U.morphEasing}).catch(()=>{})}function N(){return e?e.getBoundingClientRect():null}function M(a){l=a}function E(a,C,b){i=a,u=C,c=b;const y=()=>{d&&clearTimeout(d),d=setTimeout(()=>{const H=window.getSelection();if(!H||H.rangeCount===0||H.toString().trim().length===0){T().catch(()=>{});return}const v=H.getRangeAt(0).getBoundingClientRect();v.width===0&&v.height===0||L(v,H.toString())},U.selectionDelay)};document.addEventListener("selectionchange",y);const _=H=>{const z=H.target;if(z.tagName!=="IMG")return;const v=z,w=v.getBoundingClientRect();w.width<80||w.height<80||(x&&(clearTimeout(x),x=null),S&&clearTimeout(S),g=v,S=setTimeout(()=>{g===v&&m(v)},U.imageHoverDelay))},$=H=>{const z=H.target,v=H.relatedTarget,w=z.tagName==="IMG",I=e&&(z===e||e.contains(z));!w&&!I||v&&e&&(v===e||e.contains(v))||v&&v.tagName==="IMG"&&v===g||(S&&(clearTimeout(S),S=null),t==="image"&&(x&&clearTimeout(x),x=setTimeout(()=>{T().catch(()=>{})},U.dismissDelay)))};document.addEventListener("mouseover",_,!0),document.addEventListener("mouseout",$,!0),f.push(()=>document.removeEventListener("selectionchange",y),()=>document.removeEventListener("mouseover",_,!0),()=>document.removeEventListener("mouseout",$,!0));const D=()=>{if(!e||!r)return;const H=window.innerHeight,z=window.innerWidth;if(t==="text"){const v=window.getSelection();if(!v||v.rangeCount===0){T().catch(()=>{});return}const w=v.getRangeAt(0).getBoundingClientRect();if(w.bottom<0||w.top>H||w.right<0||w.left>z){T().catch(()=>{});return}e.style.top=`${w.bottom}px`,e.style.left=`${w.left}px`,e.style.width=`${w.width}px`,r=w}else if(t==="image"&&g){const v=g.getBoundingClientRect();if(v.bottom<0||v.top>H||v.right<0||v.left>z){T().catch(()=>{});return}const w=v.bottom-K.imageBarHeight;e.style.top=`${w}px`,e.style.left=`${v.left}px`,e.style.width=`${v.width}px`,r=v}};window.addEventListener("scroll",D,{passive:!0}),f.push(()=>window.removeEventListener("scroll",D))}function s(){d&&clearTimeout(d),S&&clearTimeout(S),x&&clearTimeout(x);for(const a of f)a();f.length=0,p()}return{init:E,destroy:s,showTextBar:L,showImageBar:m,hideBar:T,getBarRect:N,onAskClicked:M}}function yt(e){if(e.length<=J.previewMaxChars)return e;const t=e.length-J.previewMaxChars;return e.slice(0,J.previewMaxChars)+`... (${t} more chars)`}function wt(e){switch(e){case"article":return"Ask about this article...";case"code":return"Ask about this code...";default:return"Ask anything about this text..."}}function De(){const e=Math.min(J.maxWidth,window.innerWidth-32),t=Math.min(J.maxHeight,window.innerHeight-32),n=(window.innerWidth-e)/2,o=(window.innerHeight-t)/2;return{left:n,top:o,right:n+e,bottom:o+t,width:e,height:t,x:n,y:o,toJSON:()=>({})}}function Fe(e,t){Object.assign(e.style,{position:"fixed",left:"0",top:"0",width:`${t.width}px`,height:`${t.height}px`,transform:`translate(${t.left}px, ${t.top}px)`,background:R.bg,backdropFilter:R.backdrop,WebkitBackdropFilter:R.backdrop,border:`1px solid rgba(255,255,255,${R.borderOpacity})`,borderRadius:"16px",boxShadow:R.shadowBase,fontFamily:W,color:"#e5e7eb",overflow:"hidden",zIndex:String(Z),boxSizing:"border-box"})}function vt(e,t){const n=document.createElement("div");n.className="modal-content-wrap",Object.assign(n.style,{display:"flex",flexDirection:"column",gap:"12px",padding:"16px",height:"100%",boxSizing:"border-box",opacity:"0"});const o=document.createElement("div");o.className="modal-preview",Object.assign(o.style,{fontSize:"13px",lineHeight:"1.5",color:"rgba(229,231,235,0.75)",overflow:"hidden",display:"-webkit-box",WebkitLineClamp:String(J.previewMaxLines),WebkitBoxOrient:"vertical",maxHeight:`${J.previewMaxLines*20}px`,flexShrink:"0"}),o.textContent=yt(e);const r=document.createElement("input");r.type="text",r.className="modal-input",r.placeholder=t,Object.assign(r.style,{background:"rgba(255,255,255,0.07)",border:"1px solid rgba(255,255,255,0.15)",borderRadius:"8px",padding:"8px 12px",color:"#e5e7eb",fontSize:"14px",fontFamily:W,outline:"none",flexShrink:"0"});const i=document.createElement("div");i.className="modal-actions",Object.assign(i.style,{display:"flex",flexWrap:"wrap",gap:"6px",flexShrink:"0"});const u=[{label:"Explain",value:"explain"},{label:"Summarize",value:"summarize"},{label:"Chat with AURA",value:"chat"},{label:"Save to Memory",value:"save"},{label:"Translate",value:"translate"}];for(const x of u){const g=document.createElement("button");g.className="modal-action-btn",g.textContent=x.label,g.dataset.action=x.value,Object.assign(g.style,{background:"rgba(255,255,255,0.08)",border:"1px solid rgba(255,255,255,0.12)",borderRadius:"6px",padding:"5px 10px",color:"#e5e7eb",fontSize:"12px",fontFamily:W,cursor:"pointer"}),i.appendChild(g)}const c=document.createElement("div");c.className="modal-model-row",Object.assign(c.style,{display:"flex",alignItems:"center",gap:"8px",marginTop:"auto",flexShrink:"0"});const l=document.createElement("span");l.textContent="Model",Object.assign(l.style,{fontSize:"12px",color:"rgba(229,231,235,0.5)"});const d=document.createElement("select");d.className="modal-model-select",Object.assign(d.style,{background:"rgba(255,255,255,0.07)",border:"1px solid rgba(255,255,255,0.15)",borderRadius:"6px",padding:"4px 8px",color:"#e5e7eb",fontSize:"12px",fontFamily:W,cursor:"pointer"});const S=[{label:"Auto",value:"auto"},{label:"Fast",value:"fast"},{label:"Balanced",value:"balanced"},{label:"Powerful",value:"powerful"}];for(const x of S){const g=document.createElement("option");g.value=x.value,g.textContent=x.label,d.appendChild(g)}return c.appendChild(l),c.appendChild(d),n.appendChild(o),n.appendChild(r),n.appendChild(i),n.appendChild(c),n}function Et(e){const t=document.createElement("div");t.className="modal-content-wrap",Object.assign(t.style,{display:"flex",flexDirection:"column",gap:"12px",padding:"16px",height:"100%",boxSizing:"border-box",opacity:"0"});const n=document.createElement("div");n.className="modal-preview",Object.assign(n.style,{flexShrink:"0",overflow:"hidden",borderRadius:"8px"});const o=document.createElement("img");o.src=e,Object.assign(o.style,{maxWidth:"100%",maxHeight:`${J.imagePreviewMaxHeight}px`,objectFit:"contain",display:"block"}),n.appendChild(o);const r=document.createElement("div");r.className="modal-actions",Object.assign(r.style,{display:"flex",flexWrap:"wrap",gap:"6px",flexShrink:"0"});const i=[{label:"Describe",value:"describe"},{label:"Summarize",value:"summarize"},{label:"Chat with AURA",value:"chat"},{label:"Save to Memory",value:"save"},{label:"Translate",value:"translate"}];for(const S of i){const x=document.createElement("button");x.className="modal-action-btn",x.textContent=S.label,x.dataset.action=S.value,Object.assign(x.style,{background:"rgba(255,255,255,0.08)",border:"1px solid rgba(255,255,255,0.12)",borderRadius:"6px",padding:"5px 10px",color:"#e5e7eb",fontSize:"12px",fontFamily:W,cursor:"pointer"}),r.appendChild(x)}const u=document.createElement("input");u.type="text",u.className="modal-input",u.placeholder="Ask about this image...",Object.assign(u.style,{background:"rgba(255,255,255,0.07)",border:"1px solid rgba(255,255,255,0.15)",borderRadius:"8px",padding:"8px 12px",color:"#e5e7eb",fontSize:"14px",fontFamily:W,outline:"none",flexShrink:"0"});const c=document.createElement("div");c.className="modal-model-row",Object.assign(c.style,{display:"flex",alignItems:"center",gap:"8px",marginTop:"auto",flexShrink:"0"});const l=document.createElement("span");l.textContent="Model",Object.assign(l.style,{fontSize:"12px",color:"rgba(229,231,235,0.5)"});const d=document.createElement("select");d.className="modal-model-select",Object.assign(d.style,{background:"rgba(255,255,255,0.07)",border:"1px solid rgba(255,255,255,0.15)",borderRadius:"6px",padding:"4px 8px",color:"#e5e7eb",fontSize:"12px",fontFamily:W,cursor:"pointer"});for(const S of[{label:"Auto",value:"auto"},{label:"Fast",value:"fast"},{label:"Balanced",value:"balanced"},{label:"Powerful",value:"powerful"}]){const x=document.createElement("option");x.value=S.value,x.textContent=S.label,d.appendChild(x)}return c.appendChild(l),c.appendChild(d),t.appendChild(n),t.appendChild(r),t.appendChild(u),t.appendChild(c),t}function Ct(){let e={overlay:null,modal:null,originRect:null,content:"",isOpen:!1,closing:!1,opening:!1},t=null,n=null;const o=u=>{u.key==="Escape"&&e.isOpen&&i()};async function r(u,c,l){if(e.opening)return;e.isOpen&&await i(),e.opening=!0,e.originRect=c,e.content=l,e.isOpen=!0;const d=document.createElement("div");d.className="aura-modal-overlay",Object.assign(d.style,{position:"fixed",inset:"0",background:"rgba(0,0,0,0.3)",zIndex:String(Z-1),opacity:"0"}),document.body.appendChild(d),e.overlay=d;const S=document.createElement("div");S.className="aura-modal",Fe(S,c),document.body.appendChild(S),e.modal=S,ze(d,{duration:U.flowDuration,easing:"ease-out"}).then(()=>{d.style.opacity="1"});const x=De();await Re(S,c,x,{duration:U.morphDuration,easing:U.morphEasing}),Fe(S,x),S.appendChild(u),ze(u,{duration:U.crossFadeDuration,easing:"ease-out"}).then(()=>{u.style.opacity="1"}),S.querySelectorAll(".modal-action-btn").forEach(f=>{f.addEventListener("click",()=>{const p=f.dataset.action??"ask",T=S.querySelector(".modal-model-select"),h=(T==null?void 0:T.value)??"auto";n==null||n(p,e.content,h)})});const g=S.querySelector(".modal-input");g&&g.addEventListener("keydown",f=>{if(f.key==="Enter"){const p=S.querySelector(".modal-model-select"),T=(p==null?void 0:p.value)??"auto";n==null||n("ask",g.value,T)}}),d.addEventListener("click",()=>i()),document.addEventListener("keydown",o),e.opening=!1}async function i(){if(!e.isOpen||e.closing)return;e.closing=!0;const{modal:u,overlay:c,originRect:l}=e;document.removeEventListener("keydown",o);const d=[];if(u&&l){const S=De();d.push(Re(u,S,l,{duration:U.morphDuration,easing:U.morphEasing}).catch(()=>{}))}c&&d.push(gt(c,{duration:U.morphDuration,easing:"ease-in"}).catch(()=>{})),await Promise.all(d),u==null||u.remove(),c==null||c.remove(),e={overlay:null,modal:null,originRect:null,content:"",isOpen:!1,closing:!1,opening:!1}}return{init(u,c,l){t=c},destroy(){i()},openWithText(u,c){const l=wt((t==null?void 0:t.get().type)??"general"),d=vt(u,l);r(d,c,u)},openWithImage(u,c){const l=Et(u);r(l,c,u)},close:i,onAction(u){n=u}}}function kt(){let e,t=()=>{};function n(E,s){try{s?e.runtime.sendMessage(E,s):e.runtime.sendMessage(E)}catch{}}const o=document.createElement("div");o.id="aura-highlight-host",Object.assign(o.style,{position:"fixed",top:"0",left:"0",zIndex:"2147483646",pointerEvents:"none"}),document.documentElement.appendChild(o);const r=o.attachShadow({mode:"closed"}),i=document.createElement("style");i.textContent=`
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
  `,r.appendChild(i);const u=document.createElement("div");r.appendChild(u);const c=document.createElement("style");c.textContent=`
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
  `,document.head.appendChild(c);let l=null,d=null;function S(){d&&(clearTimeout(d),d=null),l&&(l.remove(),l=null)}function x(E,s){S();const a=E.getBoundingClientRect();l=document.createElement("div"),l.className="hl-tooltip";const C=document.createElement("span");C.className="hl-tooltip-text",C.textContent="Saved to AURA",l.appendChild(C);const b=document.createElement("button");b.className="hl-tooltip-delete",b.title="Remove highlight",b.innerHTML='<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',b.addEventListener("click",y=>{y.stopPropagation(),A(s),S()}),l.appendChild(b),l.style.top=`${Math.round(a.top-34)}px`,l.style.left=`${Math.round(a.left+a.width/2-60)}px`,u.appendChild(l)}function g(E){if(E.nodeType===Node.DOCUMENT_NODE)return"/";const s=[];let a=E;for(;a&&a!==document;){if(a.nodeType===Node.ELEMENT_NODE){const C=a;let b=C.tagName.toLowerCase();const y=C.parentNode;if(y){const _=Array.from(y.childNodes).filter($=>$.nodeType===Node.ELEMENT_NODE&&$.tagName===C.tagName);if(_.length>1){const $=_.indexOf(C)+1;b+=`[${$}]`}}s.unshift(b)}else if(a.nodeType===Node.TEXT_NODE){const C=a.parentNode;if(C){const b=Array.from(C.childNodes).filter(y=>y.nodeType===Node.TEXT_NODE);if(b.length>1){const y=b.indexOf(a)+1;s.unshift(`text()[${y}]`)}else s.unshift("text()")}}a=a.parentNode}return"/"+s.join("/")}function f(E){const s=E.commonAncestorContainer,a=(s.nodeType===Node.TEXT_NODE,s.textContent||""),C=E.toString(),b=a.indexOf(C);if(b===-1)return"";const y=a.slice(Math.max(0,b-50),b),_=a.slice(b+C.length,b+C.length+50);return y+"|||"+_}function p(){return"hl_"+Date.now().toString(36)+"_"+Math.random().toString(36).slice(2,8)}function T(E){const s=E.getAttribute("data-aura-hl")||"";E.addEventListener("mouseenter",()=>x(E,s)),E.addEventListener("mouseleave",()=>{d=setTimeout(S,300)})}function h(E){const s=window.getSelection();if(!s||s.rangeCount===0)return null;const a=s.getRangeAt(0);if(a.collapsed)return null;try{const C=document.createElement("mark");return C.setAttribute("data-aura-hl",E),a.surroundContents(C),s.removeAllRanges(),T(C),C}catch{try{const y=a.cloneContents().textContent||"";if(!y.trim())return null;a.deleteContents();const _=document.createElement("mark");return _.setAttribute("data-aura-hl",E),_.textContent=y,a.insertNode(_),s.removeAllRanges(),T(_),_}catch{return null}}}function k(){const E=window.getSelection();if(!E||E.rangeCount===0||E.isCollapsed)return!1;const s=E.getRangeAt(0),a=s.toString().trim();if(!a)return!1;const C=p(),b=g(s.startContainer),y=f(s);if(!h(C))return!1;const $={id:C,url:window.location.href,text:a,xpath:b,context:y,timestamp:Date.now(),color:"purple",pageTitle:document.title};return n({type:"SAVE_HIGHLIGHT",highlight:$},D=>{D&&D.ok?t("Highlight saved to AURA"):t((D==null?void 0:D.error)||"Failed to save highlight",3e3)}),!0}function A(E){const s=document.querySelector(`mark[data-aura-hl="${E}"]`);if(s){const a=s.parentNode;for(;s.firstChild;)a==null||a.insertBefore(s.firstChild,s);s.remove(),a==null||a.normalize()}n({type:"DELETE_HIGHLIGHT",id:E,url:window.location.href},a=>{t("Highlight removed")})}function L(E,s,a){try{const z=document.evaluate(E,document,null,XPathResult.FIRST_ORDERED_NODE_TYPE,null).singleNodeValue;if(z&&z.textContent&&z.textContent.includes(s)){const v=document.createRange(),w=z.textContent.indexOf(s);if(w>=0)return v.setStart(z,w),v.setEnd(z,w+s.length),v}}catch{}const C=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT,null),[b,y]=a.split("|||");let _=null,$=-1,D=0;for(;C.nextNode();){const H=C.currentNode,z=H.textContent||"",v=z.indexOf(s);if(v===-1)continue;let w=1;b&&z.slice(Math.max(0,v-50),v).includes(b.slice(-20))&&(w+=2),y&&z.slice(v+s.length,v+s.length+50).includes(y.slice(0,20))&&(w+=2),w>D&&(D=w,_=H,$=v)}if(_&&$>=0){const H=document.createRange();return H.setStart(_,$),H.setEnd(_,$+s.length),H}return null}function m(E){if(document.querySelector(`mark[data-aura-hl="${E.id}"]`))return!0;const s=L(E.xpath,E.text,E.context);if(!s)return!1;try{const a=document.createElement("mark");return a.setAttribute("data-aura-hl",E.id),E.stale&&a.classList.add("aura-hl-stale"),s.surroundContents(a),T(a),!0}catch{try{const C=s.toString();s.deleteContents();const b=document.createElement("mark");return b.setAttribute("data-aura-hl",E.id),E.stale&&b.classList.add("aura-hl-stale"),b.textContent=C,s.insertNode(b),T(b),!0}catch{return!1}}}function N(){n({type:"GET_HIGHLIGHTS",url:window.location.href},E=>{if(!(!E||!E.ok||!E.highlights))for(const s of E.highlights)m(s)||(s.stale=!0,m(s))})}function M(E){const s=document.querySelector(`mark[data-aura-hl="${E}"]`);s&&(s.scrollIntoView({behavior:"smooth",block:"center"}),s.classList.add("aura-hl-flash"),setTimeout(()=>s.classList.remove("aura-hl-flash"),1500))}return{init(E,s,a){e=a,setTimeout(N,1500)},destroy(){o.remove(),c.remove()},scrollTo:M,saveHighlight:k,setShowToast(E){t=E}}}const St="mail.google.com";function Tt(){let e,t=()=>{};const n=new Map;function o(){return window.location.hostname===St}function r(){const g=document.querySelectorAll(".a3s.aiL");if(g.length===0)return"";const f=[];return g.forEach(p=>{var h;const T=(h=p.innerText)==null?void 0:h.trim();T&&f.push(T)}),f.join(`

---

`).slice(0,2e4)}function i(g){const p=["Message Body","Nachrichtentext","Corps du message","Cuerpo del mensaje","Corpo da mensagem","Corpo del messaggio","Текст сообщения","Mesaj Metni","メッセージ本文","메시지 본문","邮件正文","نص الرسالة","Berichttekst","Treść wiadomości","संदेश का मुख्य भाग","Mesaj mətni"].map(h=>`div[aria-label="${h}"]`).join(", "),T=g.querySelector(p+', div[g_editable="true"][contenteditable="true"], div.editable[contenteditable="true"]');return T||g.querySelector('div[contenteditable="true"][role="textbox"]')}function u(g){var p;const f=i(g);return f&&((p=f.innerText)==null?void 0:p.trim())||""}function c(g){return g.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;")}function l(g,f){const p=i(g);if(!p)return;p.focus();const T=window.getSelection();if(T){const k=document.createRange();k.selectNodeContents(p),T.removeAllRanges(),T.addRange(k)}document.execCommand("insertText",!1,f)||(p.innerHTML=f.split(`
`).map(k=>`<div>${c(k)||"<br>"}</div>`).join("")),p.dispatchEvent(new Event("input",{bubbles:!0})),p.dispatchEvent(new Event("change",{bubbles:!0}))}function d(g){if(n.has(g))return;const f=g.querySelector('div[aria-label*="Send"], div[data-tooltip*="Send"], div[aria-label*="Enviar"], div[aria-label*="Envoyer"], div[aria-label*="Senden"], div[aria-label*="Отправить"]'),p=g.querySelector(".btC, .bAK, tr.btC, .IZ");if(!((f==null?void 0:f.parentElement)||p))return;const h=document.createElement("div");h.className="aura-gmail-ai-host",Object.assign(h.style,{display:"inline-flex",alignItems:"center",verticalAlign:"middle",marginLeft:"8px",position:"relative",zIndex:"1"});const k=h.attachShadow({mode:"closed"}),A=document.createElement("style");A.textContent=`
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
    `,k.appendChild(A);const L=document.createElement("div");L.style.position="relative",L.style.display="inline-flex",L.style.alignItems="center",k.appendChild(L);const m=document.createElement("button");m.className="gmail-ai-btn",m.innerHTML='<span class="sparkle">✦</span> AI',L.appendChild(m);let N=null,M=null,E=null,s=null;function a(w,I=2500){E&&E.remove(),s&&clearTimeout(s),E=document.createElement("div"),E.className="gmail-ai-toast",E.textContent=w,L.appendChild(E),s=setTimeout(()=>{E&&(E.remove(),E=null),s=null},I)}function C(){N&&(N.remove(),N=null),M=null}function b(w){N&&N.querySelectorAll(".gmail-ai-menu-item").forEach(I=>{I.classList.add("loading")})}function y(w,I){const O=u(g),B=r();if(w==="draft_reply"&&!O&&!B){a("No email thread found",3e3),C();return}if(w!=="draft_reply"&&!O){a("Compose body is empty",3e3),C();return}b();const q={type:"QUICK_ACTION",action:w,text:O||"(empty — draft a new reply)",...w==="draft_reply"?{threadContext:B}:{},...I?{language:I}:{}};t(q,P=>{P&&P.ok&&P.result?(l(g,P.result),a("Updated by AURA")):a((P==null?void 0:P.error)||"Action failed",3e3),C()})}const _=[{icon:"✍️",label:"Draft reply",action:"draft_reply"},{icon:"✨",label:"Improve",action:"improve"},{icon:"🏢",label:"Make formal",action:"make_formal",separator:!0},{icon:"😊",label:"Make casual",action:"make_casual"},{icon:"✂️",label:"Shorten",action:"shorten"},{icon:"🌐",label:"Translate to...",action:"translate_menu",separator:!0}],$=["English","Spanish","French","German","Chinese"];function D(){C(),N=document.createElement("div"),N.className="gmail-ai-menu",_.forEach(w=>{if(w.separator){const O=document.createElement("div");O.className="gmail-ai-sep",N.appendChild(O)}const I=document.createElement("button");I.className="gmail-ai-menu-item",I.innerHTML=`<span class="item-icon">${w.icon}</span><span>${w.label}</span>`,I.addEventListener("click",O=>{O.preventDefault(),O.stopPropagation(),w.action==="translate_menu"?H(I):y(w.action)}),N.appendChild(I)}),L.appendChild(N)}function H(w){if(M){M.remove(),M=null;return}M=document.createElement("div"),M.className="gmail-ai-sub",$.forEach(I=>{const O=document.createElement("button");O.className="gmail-ai-menu-item",O.textContent=I,O.addEventListener("click",B=>{B.preventDefault(),B.stopPropagation(),y("gmail_translate",I)}),M.appendChild(O)}),N&&w.parentNode===N&&w.after(M)}m.addEventListener("click",w=>{w.preventDefault(),w.stopPropagation(),N?C():D()});const z=w=>{if(!N)return;w.composedPath().includes(h)||C()};document.addEventListener("mousedown",z,!0),f!=null&&f.parentElement?f.parentElement.insertBefore(h,f.nextSibling):p&&p.appendChild(h);const v=new MutationObserver(()=>{document.body.contains(g)||(v.disconnect(),document.removeEventListener("mousedown",z,!0),h.remove(),n.delete(g))});v.observe(document.body,{childList:!0,subtree:!0}),n.set(g,{composeEl:g,buttonHost:h,shadow:k,observer:v,outsideHandler:z})}function S(){['div[role="dialog"]',"div.ip.iq","div.nH.nn"].forEach(f=>{document.querySelectorAll(f).forEach(p=>{i(p)&&(n.has(p)||d(p))})})}function x(){if(!o())return;S();const g=new MutationObserver(T=>{var k,A,L;let h=!1;for(const m of T){if(m.addedNodes.length>0)for(const N of m.addedNodes){if(N.nodeType!==Node.ELEMENT_NODE)continue;const M=N;if((k=M.matches)!=null&&k.call(M,'div[role="dialog"]')||(A=M.querySelector)!=null&&A.call(M,'div[role="dialog"]')||(L=M.querySelector)!=null&&L.call(M,'div[contenteditable="true"]')){h=!0;break}}if(h)break}h&&setTimeout(S,300)}),f=document.querySelector('div[role="main"]')||document.body;g.observe(f,{childList:!0,subtree:!0});let p=null;p=setInterval(()=>{if(!o()){p&&clearInterval(p);return}S()},3e3)}return{init(g,f,p){e=p,t=(T,h)=>{try{h?e.runtime.sendMessage(T,h):e.runtime.sendMessage(T)}catch{}},x()},destroy(){for(const g of n.values())g.observer.disconnect(),document.removeEventListener("mousedown",g.outsideHandler,!0),g.buttonHost.remove();n.clear()}}}const Lt=["display","position","flex-direction","align-items","justify-content","gap","flex-wrap","flex","flex-grow","flex-shrink","width","height","min-width","min-height","max-width","max-height","padding","padding-top","padding-right","padding-bottom","padding-left","margin","margin-top","margin-right","margin-bottom","margin-left","border","border-radius","border-color","border-width","border-style","background","background-color","background-image","background-size","color","font-size","font-weight","font-family","line-height","letter-spacing","text-align","text-decoration","text-transform","box-shadow","opacity","overflow","z-index","grid-template-columns","grid-template-rows","grid-gap","transform","transition"];function At(){let e;const t=document.createElement("div");t.id="aura-capture-host",Object.assign(t.style,{position:"fixed",top:"0",left:"0",width:"0",height:"0",zIndex:"2147483647",pointerEvents:"none"}),document.documentElement.appendChild(t);const n=t.attachShadow({mode:"closed"}),o=document.createElement("style");o.textContent=`
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
  `,n.appendChild(o);const r=document.createElement("div");n.appendChild(r);let i=!1,u=null,c=null,l=null,d=null;function S(A){const L=window.getComputedStyle(A),m={};for(const N of Lt){const M=L.getPropertyValue(N);M&&M!=="none"&&M!=="normal"&&M!=="auto"&&M!=="0px"&&M!=="rgba(0, 0, 0, 0)"&&(m[N]=M)}return m}function x(A){const L=A.tagName.toLowerCase(),m=A.className&&typeof A.className=="string"?"."+A.className.trim().split(/\s+/).slice(0,2).join("."):"";return L+m}function g(A){const L=A.getBoundingClientRect(),m=window.getComputedStyle(A),N=A.outerHTML,M={};M[x(A)]=S(A);const E=A.querySelectorAll("*");let s=0;for(const a of E){if(s>=50)break;const C=S(a);if(Object.keys(C).length>0){const b=x(a),y=M[b]?`${b}:nth(${s})`:b;M[y]=C}s++}return{html:N,css:M,dimensions:{width:L.width,height:L.height,padding:`${m.paddingTop} ${m.paddingRight} ${m.paddingBottom} ${m.paddingLeft}`,margin:`${m.marginTop} ${m.marginRight} ${m.marginBottom} ${m.marginLeft}`},textContent:(A.textContent||"").slice(0,2e3).trim(),tagName:A.tagName.toLowerCase(),className:(typeof A.className=="string"?A.className:"").trim()}}function f(){i||(i=!0,l=document.createElement("div"),l.className="capture-banner",l.innerHTML='<span class="dot"></span> AURA Capture Mode — Click any element • Esc to exit',r.appendChild(l),u=document.createElement("div"),u.className="capture-overlay",u.style.display="none",r.appendChild(u),c=document.createElement("div"),c.className="capture-tooltip",c.style.display="none",r.appendChild(c),t.style.width="100vw",t.style.height="100vh",document.addEventListener("mousemove",T,!0),document.addEventListener("click",h,!0),document.addEventListener("keydown",k,!0))}function p(){if(i){i=!1,d=null,u&&(u.remove(),u=null),c&&(c.remove(),c=null),l&&(l.remove(),l=null),t.style.width="0",t.style.height="0",document.removeEventListener("mousemove",T,!0),document.removeEventListener("click",h,!0),document.removeEventListener("keydown",k,!0);try{e.runtime.sendMessage({type:"OPEN_PANEL",panel:"capture"})}catch{}}}function T(A){if(!i)return;const L=document.elementsFromPoint(A.clientX,A.clientY);let m=null;for(const M of L)if(!(M===t||t.contains(M))&&!(M.id==="aura-host"||M.id==="aura-dock-shadow"||M.id==="aura-quick-action-host"||M.id==="aura-highlight-host"||M.id==="aura-img-toolbar-host"||M.id==="aura-capture-host")&&!(M===document.documentElement||M===document.body)){m=M;break}if(!m){u&&(u.style.display="none"),c&&(c.style.display="none"),d=null;return}d=m;const N=m.getBoundingClientRect();if(u&&(u.style.display="block",u.style.top=N.top+"px",u.style.left=N.left+"px",u.style.width=N.width+"px",u.style.height=N.height+"px"),c){const M=m.tagName.toLowerCase(),E=m.className&&typeof m.className=="string"?m.className.trim().split(/\s+/).slice(0,3).join(" "):"",s=Math.round(N.width),a=Math.round(N.height);c.textContent="";const C=document.createElement("span");if(C.className="tag",C.textContent=`<${M}>`,c.appendChild(C),E){const $=document.createElement("span");$.className="cls",$.textContent="."+E.split(" ").join("."),c.appendChild($)}const b=document.createElement("span");b.className="dims",b.textContent=`${s}x${a}`,c.appendChild(b);let y=N.top-30;y<4&&(y=N.bottom+6);let _=N.left;_<4&&(_=4),c.style.display="flex",c.style.top=y+"px",c.style.left=_+"px"}}function h(A){if(!i||!d)return;A.preventDefault(),A.stopPropagation(),A.stopImmediatePropagation();const L=d,m=L.getBoundingClientRect(),N=g(L);try{e.runtime.sendMessage({type:"CAPTURE_ELEMENT",rect:{x:Math.round(m.left),y:Math.round(m.top),w:Math.round(m.width),h:Math.round(m.height)},elementData:N},M=>{})}catch{}p()}function k(A){if(A.key==="Escape"&&i){A.preventDefault(),A.stopPropagation(),p();try{e.runtime.sendMessage({type:"CAPTURE_MODE_EXITED"}).catch(()=>{})}catch{}}}return{init(A,L,m){e=m},destroy(){i&&p(),t.remove()},start:f,stop:p}}const Mt=50;function Nt(){let e;const t=new Map;function n(b,y){if(t.size>=Mt){const _=t.keys().next().value;_&&t.delete(_)}t.set(b,y)}function o(b){const y=t.get(b);return y&&(t.delete(b),t.set(b,y)),y}const r=document.createElement("div");r.id="aura-link-preview-host",Object.assign(r.style,{position:"fixed",top:"0",left:"0",zIndex:"2147483646",pointerEvents:"none"}),document.documentElement.appendChild(r);const i=r.attachShadow({mode:"closed"}),u=document.createElement("style");u.textContent=["@keyframes lp-in { from { opacity:0; transform:translateY(4px) scale(0.96); } to { opacity:1; transform:translateY(0) scale(1); } }","@keyframes lp-shimmer { 0% { background-position:-200px 0; } 100% { background-position:200px 0; } }",'.lp-popup { position:fixed; width:320px; max-height:280px; background:rgba(10,8,24,0.92); backdrop-filter:blur(20px) saturate(1.5); -webkit-backdrop-filter:blur(20px) saturate(1.5); border:1px solid rgba(124,58,237,0.25); border-radius:12px; padding:14px 16px 12px; pointer-events:auto; animation:lp-in 0.2s cubic-bezier(0.16,1,0.3,1) forwards; box-shadow:0 8px 32px rgba(0,0,0,0.5),0 0 0 1px rgba(255,255,255,0.05) inset; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Inter",system-ui,sans-serif; box-sizing:border-box; overflow:hidden; display:flex; flex-direction:column; gap:8px; }',".lp-domain { display:inline-block; background:rgba(124,58,237,0.15); border:1px solid rgba(124,58,237,0.25); border-radius:4px; padding:2px 7px; font-size:10.5px; font-weight:600; color:rgba(160,148,210,0.9); letter-spacing:0.3px; max-width:fit-content; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }",".lp-title { font-size:13px; font-weight:600; color:rgba(226,232,240,0.95); line-height:1.35; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; margin:0; }",".lp-description { font-size:12px; font-weight:400; color:rgba(226,232,240,0.65); line-height:1.45; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; margin:0; }",".lp-shimmer { height:12px; border-radius:4px; background:linear-gradient(90deg,rgba(124,58,237,0.08) 25%,rgba(124,58,237,0.18) 50%,rgba(124,58,237,0.08) 75%); background-size:400px 100%; animation:lp-shimmer 1.5s infinite linear; }",".lp-shimmer.short { width:60%; } .lp-shimmer.long { width:90%; } .lp-shimmer+.lp-shimmer { margin-top:6px; }",".lp-loading-label { font-size:11px; color:rgba(160,148,210,0.5); margin-bottom:4px; }",".lp-actions { display:flex; gap:6px; margin-top:4px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.06); }",".lp-btn { background:rgba(124,58,237,0.12); border:1px solid rgba(124,58,237,0.2); border-radius:6px; padding:4px 10px; font-size:11px; font-weight:500; font-family:inherit; color:rgba(200,180,255,0.9); cursor:pointer; transition:background 0.15s,border-color 0.15s,color 0.15s; white-space:nowrap; }",".lp-btn:hover { background:rgba(124,58,237,0.25); border-color:rgba(124,58,237,0.4); color:#fff; }",".lp-btn:active { background:rgba(124,58,237,0.35); }"].join(`
`),i.appendChild(u);const c=document.createElement("div");i.appendChild(c);let l=null,d=null,S=null,x=null,g=!1;const f=()=>{g=!0},p=()=>{g=!1};function T(b){try{return new URL(b.href,location.href).hostname!==location.hostname}catch{return!1}}function h(b){const y=b.href||"";if(!y.startsWith("http://")&&!y.startsWith("https://"))return!1;try{const _=new URL(y,location.href);if(_.hostname===location.hostname&&_.pathname===location.pathname&&_.hash)return!1}catch{return!1}return(b.textContent||"").trim().length<10?!1:T(b)}function k(){l&&(l.remove(),l=null),x=null}function A(){d&&(clearTimeout(d),d=null),S&&(clearTimeout(S),S=null)}function L(){S&&clearTimeout(S),S=setTimeout(()=>{k(),S=null},300)}function m(){S&&(clearTimeout(S),S=null)}function N(b){if(!l)return;const y=b.getBoundingClientRect();l.style.visibility="hidden",l.style.display="flex";const _=l.offsetHeight||180;l.style.visibility="";let $=y.left+y.width/2-160;$<8&&($=8),$+320>window.innerWidth-8&&($=window.innerWidth-328);let D=y.bottom+8;D+_>window.innerHeight-8&&(D=y.top-_-8,D<8&&(D=8)),l.style.top=Math.round(D)+"px",l.style.left=Math.round($)+"px"}function M(b,y,_){if(b.innerHTML="",b.style.display="none",_.title&&_.title!==y.textContent&&(y.textContent=_.title),_.description){const $=document.createElement("div");$.className="lp-description",$.textContent=_.description,y.after($)}l&&x&&N(x)}function E(b,y){k(),x=b;let _="";try{_=new URL(y).hostname}catch{_=y}const $=(b.textContent||"").trim();l=document.createElement("div"),l.className="lp-popup";const D=document.createElement("div");D.className="lp-domain",D.textContent=_,l.appendChild(D);const H=document.createElement("div");H.className="lp-title",H.textContent=$,l.appendChild(H);const z=document.createElement("div"),v=document.createElement("div");v.className="lp-loading-label",v.textContent="Loading preview…";const w=document.createElement("div");w.className="lp-shimmer long";const I=document.createElement("div");I.className="lp-shimmer short",z.appendChild(v),z.appendChild(w),z.appendChild(I),l.appendChild(z);const O=document.createElement("div");O.className="lp-actions";const B=document.createElement("button");B.className="lp-btn",B.textContent="Open",B.addEventListener("click",G=>{G.preventDefault(),G.stopPropagation(),window.open(y,"_blank","noopener"),k()});const q=document.createElement("button");q.className="lp-btn",q.textContent="Summarize in AURA",q.addEventListener("click",G=>{G.preventDefault(),G.stopPropagation();try{e.runtime.sendMessage({type:"OPEN_WITH_TEXT",action:"summarize",text:"Summarize this page: "+y,url:y,title:$})}catch{}k()}),O.appendChild(B),O.appendChild(q),l.appendChild(O),l.addEventListener("mouseenter",m),l.addEventListener("mouseleave",L),c.appendChild(l),N(b);const P=o(y);if(P){M(z,H,P);return}try{e.runtime.sendMessage({type:"LINK_PREVIEW",url:y},G=>{if(e.runtime.lastError||!G||!l||x!==b)return;const te={title:G.title||$,description:G.description||"",domain:G.domain||_};n(y,te),M(z,H,te)})}catch{}}const s=b=>{if(g)return;const y=b.target.closest("a");if(!(!y||!h(y))){if(x===y&&l){m();return}A(),d=setTimeout(()=>{g||(E(y,y.href),d=null)},800)}},a=b=>{const y=b.target.closest("a");if(y&&y===x){const _=b.relatedTarget;if(_&&r.contains(_))return;L()}y&&d&&A()},C=()=>{if(l&&x){const b=x.getBoundingClientRect();b.bottom<0||b.top>window.innerHeight?(A(),k()):N(x)}};return{init(b,y,_){e=_,document.addEventListener("mousedown",f,!0),document.addEventListener("mouseup",p,!0),document.addEventListener("mouseover",s,!0),document.addEventListener("mouseout",a,!0),window.addEventListener("scroll",C,{passive:!0})},destroy(){document.removeEventListener("mousedown",f,!0),document.removeEventListener("mouseup",p,!0),document.removeEventListener("mouseover",s,!0),document.removeEventListener("mouseout",a,!0),window.removeEventListener("scroll",C),A(),k(),r.remove()}}}function _t(e){var r;if(e.id)return"#"+CSS.escape(e.id);const t=e.getAttribute("aria-label");if(t)return`[aria-label="${t}"]`;const n=[];let o=e;for(let i=0;i<4&&o&&o!==document.body;i++,o=o.parentElement){const u=o.tagName.toLowerCase();if(o.id){n.unshift("#"+CSS.escape(o.id));break}const l=[...((r=o.parentElement)==null?void 0:r.children)||[]].indexOf(o)+1;n.unshift(u+":nth-child("+l+")")}return n.join(">")}function It(){const e=[],t=document.querySelectorAll('a,button,input,textarea,select,[role="button"],[onclick]');let n=0;for(const o of t){if(e.length>=80)break;const r=o.getBoundingClientRect();if(r.width===0||r.height===0)continue;const i=o,u=o;e.push({index:n++,type:o.tagName.toLowerCase(),text:(i.innerText||u.value||u.placeholder||i.title||"").slice(0,80).trim(),selector:_t(i),href:o.href||""})}return e}const Ot=new Set(["scroll","click","type","selectOption"]);function $t(e){const t=e&&typeof e.action=="string"?e.action:"";if(!Ot.has(t))return{ok:!1,error:"Unknown action: "+t};if(t==="scroll")return window.scrollBy(0,e.amount||300),{ok:!0};let n;try{n=document.querySelector(e.selector)}catch{return{ok:!1,error:"Invalid selector: "+e.selector}}if(!n)return{ok:!1,error:"Element not found: "+e.selector};if(t==="click")return n.click(),{ok:!0};if(t==="type")return n.focus(),n.value=e.text||"",n.dispatchEvent(new Event("input",{bubbles:!0})),n.dispatchEvent(new Event("change",{bubbles:!0})),{ok:!0};if(t==="selectOption"){if(n.tagName.toLowerCase()!=="select")return{ok:!1,error:"Element is not a <select>"};const o=n,r=[...o.options].find(i=>i.value===e.value||i.text===e.value);return r?(o.value=r.value,o.dispatchEvent(new Event("change",{bubbles:!0})),{ok:!0}):{ok:!1,error:"Option not found: "+e.value}}return{ok:!1,error:"Unhandled action: "+t}}function Ht(e,t){const n=document.createElement("div");Object.assign(n.style,{position:"fixed",top:"0",left:"0",width:"100vw",height:"100vh",zIndex:"2147483646",cursor:"crosshair",background:"rgba(0,0,0,0.4)"});const o=new Image;o.src=e,o.style.cssText="position:fixed;top:0;left:0;width:100%;height:100%;opacity:0.7;pointer-events:none;",n.appendChild(o);const r=document.createElement("canvas");r.width=window.innerWidth,r.height=window.innerHeight,Object.assign(r.style,{position:"absolute",top:"0",left:"0",width:"100%",height:"100%"}),n.appendChild(r);const i=r.getContext("2d"),u=document.createElement("div");Object.assign(u.style,{position:"fixed",top:"12px",left:"50%",transform:"translateX(-50%)",background:"rgba(0,0,0,0.75)",color:"#fff",padding:"6px 14px",borderRadius:"6px",fontSize:"13px",pointerEvents:"none"}),u.textContent="Drag to select region • Press Esc to cancel",n.appendChild(u),document.body.appendChild(n);let c=0,l=0,d=!1;const S=window.devicePixelRatio||1;function x(p,T,h,k){i&&(i.clearRect(0,0,r.width,r.height),i.strokeStyle="#7c3aed",i.lineWidth=2,i.strokeRect(p,T,h,k),i.fillStyle="rgba(124,58,237,0.12)",i.fillRect(p,T,h,k))}n.addEventListener("mousedown",p=>{c=p.clientX,l=p.clientY,d=!0}),n.addEventListener("mousemove",p=>{d&&x(c,l,p.clientX-c,p.clientY-l)});function g(p){p.key==="Escape"&&(document.body.contains(n)&&document.body.removeChild(n),document.removeEventListener("keydown",g),t({ok:!1}))}n.addEventListener("mouseup",p=>{d=!1;const T=Math.min(c,p.clientX),h=Math.min(l,p.clientY),k=Math.abs(p.clientX-c),A=Math.abs(p.clientY-l);if(document.removeEventListener("keydown",g),document.body.contains(n)&&document.body.removeChild(n),k<5||A<5){t({ok:!1});return}t({ok:!0,x:T,y:h,w:k,h:A,dpr:S})}),document.addEventListener("keydown",g);const f=new MutationObserver(()=>{document.body.contains(n)||(document.removeEventListener("keydown",g),f.disconnect())});f.observe(document.body,{childList:!0})}const le=5e4,zt=["article","main",'[role="main"]',".post-content",".article-body",".entry-content",".post-body",".article-content",".story-body",".content-body","#article-body","#content",".markdown-body",".wiki-content"],Rt=["nav","header","footer","aside","script","style","noscript","iframe",".sidebar",".menu",".nav",".navigation",".cookie",".cookie-banner",".cookie-consent",".popup",".modal",".overlay",".ad",".ads",".advert",".advertisement",".social-share",".share-buttons",".social-buttons",".comments",".comment-section","#comments",".related-posts",".recommended",".newsletter",".subscribe",'[role="navigation"]','[role="banner"]','[role="contentinfo"]','[role="complementary"]','[aria-hidden="true"]',".sr-only",".visually-hidden"];function Bt(){for(const e of zt){const t=document.querySelector(e);if(t&&t.textContent&&t.textContent.trim().length>200)return t}return document.body}function Pt(e){const t=e.cloneNode(!0);for(const n of Rt)t.querySelectorAll(n).forEach(o=>o.remove());return t}function Dt(e){const t=[],n=new Set(["P","DIV","SECTION","ARTICLE","BLOCKQUOTE","PRE","H1","H2","H3","H4","H5","H6","UL","OL","LI","TABLE","TR","DT","DD","FIGURE","FIGCAPTION","HR","BR"]);function o(r){var l;if(r.nodeType===Node.TEXT_NODE){const d=(r.textContent||"").replace(/\s+/g," ");d.trim()&&t.push(d);return}if(r.nodeType!==Node.ELEMENT_NODE)return;const i=r,u=i.tagName;if(i.hasAttribute("hidden")||((l=i.style)==null?void 0:l.display)==="none")return;if(/^H[1-6]$/.test(u)){const d=parseInt(u[1]),S="#".repeat(Math.min(d,3))+" ",x=(i.textContent||"").trim();x&&t.push(`

`+S+x+`
`);return}if(u==="LI"){const d=(i.textContent||"").trim();d&&t.push(`
- `+d);return}if(u==="A"){const d=i.href,S=(i.textContent||"").trim();S&&d&&!d.startsWith("javascript:")?t.push(S+" ("+d+")"):S&&t.push(S);return}if(u==="HR"){t.push(`

---

`);return}if(u==="BR"){t.push(`
`);return}if(u==="PRE"){const d=(i.textContent||"").trim();d&&t.push("\n\n```\n"+d+"\n```\n\n");return}const c=n.has(u);c&&t.push(`

`);for(const d of i.childNodes)o(d);c&&t.push(`
`)}return o(e),t.join("").replace(/\n{3,}/g,`

`).replace(/[ \t]+/g," ").trim()}function Ft(){var x,g;const e=window.location.href,t=document.querySelector("h1.ytd-watch-metadata, h1.ytd-video-primary-info-renderer, #title h1"),n=((x=t==null?void 0:t.textContent)==null?void 0:x.trim())||document.title.replace(/ - YouTube$/,"").trim();let o="";const r=document.querySelectorAll("ytd-transcript-segment-renderer .segment-text, yt-formatted-string.ytd-transcript-segment-renderer, #segments-container ytd-transcript-segment-renderer");if(r.length>0){const f=[];r.forEach(p=>{var h;const T=(h=p.textContent)==null?void 0:h.trim();T&&f.push(T)}),o=f.join(" ")}let i="";const u=document.querySelector("ytd-text-inline-expander #plain-snippet-text, #description-inline-expander, ytd-expander .content, #description .content");u&&(i=((g=u.textContent)==null?void 0:g.trim())||"");const c=document.querySelectorAll("ytd-comment-thread-renderer #content-text");let l="";if(c.length>0){const f=[];c.forEach((p,T)=>{var k;if(T>=10)return;const h=(k=p.textContent)==null?void 0:k.trim();h&&f.push("- "+h)}),f.length>0&&(l=`

## Top Comments
`+f.join(`
`))}let d=`# ${n}

`;o&&(d+=`## Transcript
${o}

`),i&&(d+=`## Description
${i}

`),d+=l,d.length>le&&(d=d.slice(0,le)+`

[...truncated]`);const S=d.split(/\s+/).filter(Boolean).length;return{text:d,title:n,url:e,wordCount:S,isYouTube:!0,videoTitle:n,transcript:o||void 0}}function qt(){var e,t,n,o;try{const r=window.location.href,i=document.title;if(r.match(/\.pdf($|\?|#)/i)||document.contentType==="application/pdf")return{text:((t=(e=document.body)==null?void 0:e.innerText)==null?void 0:t.slice(0,le))||"[PDF document]",title:i,url:r,wordCount:0,isPdf:!0};if(r.includes("youtube.com/watch")||r.includes("youtu.be/"))return Ft();const u=Bt(),c=Pt(u);let l=Dt(c);l.length<100&&(l=((n=document.body)==null?void 0:n.innerText)||""),l.length>le&&(l=l.slice(0,le)+`

[...truncated]`);const d=l.split(/\s+/).filter(Boolean).length;return{text:l,title:i,url:r,wordCount:d}}catch{const i=(((o=document.body)==null?void 0:o.innerText)||"").slice(0,le);return{text:i,title:document.title,url:window.location.href,wordCount:i.split(/\s+/).filter(Boolean).length}}}const jt=[{label:"Improve",icon:'<path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/>',action:"improve"},{label:"Expand",icon:'<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>',action:"expand"},{label:"Shorten",icon:'<polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/>',action:"shorten"},{label:"Fix grammar",icon:'<polyline points="20 6 9 17 4 12"/>',action:"fix_grammar"},{label:"Translate",icon:'<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/>',action:"translate"}],Ut=new Set(["password","hidden","file","checkbox","radio","range","color","date","datetime-local","month","week","time","submit","reset","button","image"]),qe=200;function Wt(e,t,n){const o=document.createElement("div");o.id="aura-quick-action-host",Object.assign(o.style,{position:"fixed",top:"0",left:"0",zIndex:"2147483646",pointerEvents:"none"}),document.documentElement.appendChild(o);const r=o.attachShadow({mode:"closed"}),i=document.createElement("style");i.textContent=`
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
  `,r.appendChild(i);const u=document.createElement("div");r.appendChild(u);let c=null,l=null,d=null,S=null;function x(s){if(s.tagName==="TEXTAREA")return!0;if(s.tagName==="INPUT"){const C=(s.type||"text").toLowerCase();return!Ut.has(C)}return!!(s.isContentEditable&&s.getAttribute("contenteditable")==="true")}function g(){d&&(d.remove(),d=null),S&&(S.remove(),S=null)}function f(){l&&(l.remove(),l=null),g(),c=null}function p(s){const a=s.getBoundingClientRect();if(a.width<qe){f();return}const C=a.bottom<0||a.top>window.innerHeight||a.right<0||a.left>window.innerWidth;if(l||(l=document.createElement("div"),l.className="qa-trigger",l.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/></svg>',l.addEventListener("click",_=>{if(_.preventDefault(),_.stopPropagation(),d){g();return}L()}),u.appendChild(l)),l.style.display=C?"none":"",C){g();return}const b=20,y=6;l.style.top=`${Math.round(a.top+(a.height-b)/2)}px`,l.style.left=`${Math.round(a.right-b-y)}px`}function T(s){return s.isContentEditable?s.innerText||"":s.value||""}function h(s,a){s.isContentEditable?s.innerText=a:s.value=a,s.dispatchEvent(new Event("input",{bubbles:!0})),s.dispatchEvent(new Event("change",{bubbles:!0}))}function k(s,a){if(!c)return;const C=T(c);if(!C.trim()){g();return}d&&d.querySelectorAll(".qa-menu-item").forEach(y=>{y.classList.add("loading")});const b=c;t({type:"QUICK_ACTION",action:s,text:C,language:a},y=>{y&&y.ok&&y.result?(h(b,y.result),n("Text updated by AURA")):n((y==null?void 0:y.error)||"Quick action failed",3e3),g()})}function A(s){if(S){S.remove(),S=null;return}const a=["English","Spanish","French","German","Chinese","Russian","Japanese","Arabic","Portuguese","Azerbaijani"];S=document.createElement("div"),S.className="qa-translate-sub",a.forEach(C=>{const b=document.createElement("button");b.className="qa-menu-item",b.textContent=C,b.addEventListener("click",y=>{y.preventDefault(),y.stopPropagation(),k("translate",C)}),S.appendChild(b)}),d&&s.parentNode===d&&s.after(S)}function L(){if(!l||!c)return;g(),d=document.createElement("div"),d.className="qa-menu",jt.forEach(C=>{const b=document.createElement("button");b.className="qa-menu-item",b.innerHTML=`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${C.icon}</svg><span>${C.label}</span>`,b.addEventListener("click",y=>{y.preventDefault(),y.stopPropagation(),C.action==="translate"?A(b):k(C.action)}),d.appendChild(b)}),u.appendChild(d);const s=l.getBoundingClientRect(),a=6;d.style.top=`${Math.round(s.bottom+a)}px`,d.style.left=`${Math.round(s.right-150)}px`,requestAnimationFrame(()=>{if(!d)return;const C=d.getBoundingClientRect();C.right>window.innerWidth-8&&(d.style.left=`${Math.round(window.innerWidth-C.width-8)}px`),C.left<8&&(d.style.left="8px"),C.bottom>window.innerHeight-8&&(d.style.top=`${Math.round(s.top-C.height-a)}px`)})}function m(s){const a=s.target;!a||!x(a)||a.getBoundingClientRect().width<qe||(c=a,p(a))}function N(s){setTimeout(()=>{if(d||S)return;const a=document.activeElement;a&&a===c||f()},200)}function M(){c&&p(c)}document.addEventListener("mousedown",s=>{!d&&!l||s.composedPath().includes(o)||g()},!0),document.addEventListener("focusin",m,!0),document.addEventListener("focusout",N,!0),window.addEventListener("scroll",M,{passive:!0}),window.addEventListener("resize",M,{passive:!0}),new MutationObserver(()=>{c&&!document.body.contains(c)&&f()}).observe(document.body,{childList:!0,subtree:!0})}function X(e,t=500){return typeof e=="string"?e.slice(0,t):""}function re(e,t=0,n=1e9){const o=Number(e);return Number.isFinite(o)?Math.min(Math.max(Math.round(o),t),n):0}function je(e,t=500,n=5e3){if(!Array.isArray(e))return[];const o=[];for(let r=0;r<Math.min(e.length,n);r++){const i=e[r];!i||typeof i!="object"||Array.isArray(i)||o.push({start:re(i.start,0,1e8),dur:re(i.dur,0,1e8),text:X(i.text,t)})}return o}function Vt(e){document.addEventListener("aura-yt-subtitles",t=>{try{const n=t.detail;if(!n||typeof n!="object")return;e({type:"YT_SUBTITLES",videoId:X(n.videoId,20),lang:X(n.lang,10),segments:je(n.segments)})}catch{}}),document.addEventListener("aura-yt-metadata",t=>{try{const n=t.detail;if(!n||typeof n!="object")return;e({type:"YT_METADATA",videoId:X(n.videoId,20),title:X(n.title,500),duration:re(n.duration,0,86400),description:X(n.description,5e3),channelName:X(n.channelName,200),chapters:Array.isArray(n.chapters)?n.chapters.slice(0,200).map(o=>({title:X(o==null?void 0:o.title,200),start:re(o==null?void 0:o.start,0,86400)})):[],captionTracks:Array.isArray(n.captionTracks)?n.captionTracks.slice(0,50).map(o=>({lang:X(o==null?void 0:o.lang,10),url:X(o==null?void 0:o.url,2e3)})):[]})}catch{}})}function Gt(e){document.addEventListener("aura-netflix-subtitles",t=>{try{const n=t.detail;if(!n||typeof n!="object")return;e({type:"NETFLIX_SUBTITLES",movieId:X(n.movieId,20),lang:X(n.lang,10),trackId:X(n.trackId,50),segments:je(n.segments)})}catch{}}),document.addEventListener("aura-netflix-metadata",t=>{try{const n=t.detail;if(!n||typeof n!="object")return;e({type:"NETFLIX_METADATA",movieId:X(n.movieId,20),title:X(n.title,500),episodeTitle:X(n.episodeTitle,500),seasonNumber:re(n.seasonNumber,0,100),episodeNumber:re(n.episodeNumber,0,100),duration:re(n.duration,0,86400)})}catch{}})}const Xt="p, h1, h2, h3, h4, h5, h6, li, td, th, blockquote, figcaption",Le="data-aura-translated",Ue=10,Yt=10;function Kt(e){const t={mode:"bilingual",targetLang:"English",active:!1,badge:null,elements:[],activeCount:0};function n(){const x=document.querySelectorAll(Xt),g=[];for(const f of x){if(f.hasAttribute(Le))continue;const p=f.getBoundingClientRect();p.width===0&&p.height===0||f.closest("#aura-host, #aura-dock-shadow, #aura-quick-action-host, .aura-translate-badge")||f.tagName==="SPAN"&&(f.textContent||"").trim().length<=20||(f.textContent||"").trim().length<5||g.push(f)}return g}function o(x){const g=document.createElement("div");return g.className="aura-page-translation",g.setAttribute("data-aura-translation","true"),Object.assign(g.style,{borderLeft:"2px solid rgba(124, 58, 237, 0.6)",background:"rgba(124, 58, 237, 0.05)",padding:"6px 10px",marginTop:"4px",marginBottom:"4px",fontSize:"0.95em",color:"inherit",opacity:"0",fontFamily:"inherit",lineHeight:"1.5",borderRadius:"0 4px 4px 0",transition:"opacity 0.3s ease",fontStyle:"italic"}),g.textContent="Translating...",g.style.color="rgba(124, 58, 237, 0.5)",x.setAttribute(Le,"true"),x.after(g),requestAnimationFrame(()=>{g.style.opacity="0.6"}),g}function r(x,g){x.style.opacity="0",x.textContent=g,x.style.fontStyle="normal",x.style.color="inherit",requestAnimationFrame(()=>{x.style.opacity="0.85"})}function i(x,g){return new Promise(f=>{try{e.runtime.sendMessage({type:"TRANSLATE_BATCH",texts:x,targetLang:g},p=>{if(e.runtime.lastError){f(x.map(()=>"[Translation failed]"));return}p!=null&&p.ok&&p.translations?f(p.translations):f(x.map(()=>(p==null?void 0:p.error)||"[Translation failed]"))})}catch{f(x.map(()=>"[Translation failed]"))}})}function u(){if(!t.badge)return;const x=t.badge.querySelector("[data-badge-mode]");x&&(x.textContent=t.mode==="bilingual"?"Bilingual":"Translated")}function c(x){t.mode=x;for(const g of t.elements)x==="translated"?(g.original.style.display="none",g.translation.style.marginTop="0"):(g.original.style.display="",g.translation.style.marginTop="4px");u()}function l(){t.active=!1;for(const x of t.elements)x.translation.remove(),x.original.removeAttribute(Le),x.original.style.display="";t.elements=[],t.badge&&(t.badge.remove(),t.badge=null)}function d(){t.badge&&(t.badge.remove(),t.badge=null),t.badge=document.createElement("div"),t.badge.className="aura-translate-badge",Object.assign(t.badge.style,{position:"fixed",bottom:"20px",right:"20px",zIndex:"2147483646",background:"rgba(10, 8, 24, 0.92)",backdropFilter:"blur(20px) saturate(1.5)",WebkitBackdropFilter:"blur(20px) saturate(1.5)",border:"1px solid rgba(124, 58, 237, 0.35)",borderRadius:"12px",padding:"8px 12px",display:"flex",alignItems:"center",gap:"8px",boxShadow:"0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05) inset",fontFamily:"-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif",fontSize:"12px",color:"rgba(226, 232, 240, 0.9)"});const x=document.createElement("span");Object.assign(x.style,{width:"6px",height:"6px",borderRadius:"50%",background:"#7c3aed",flexShrink:"0"}),t.badge.appendChild(x);const g=document.createElement("span");g.style.color="rgba(160, 148, 210, 0.8)",g.textContent="Translation active",t.badge.appendChild(g);const f=document.createElement("span");Object.assign(f.style,{width:"1px",height:"14px",background:"rgba(255,255,255,0.1)",flexShrink:"0"}),t.badge.appendChild(f);const p=document.createElement("span");p.setAttribute("data-badge-lang",""),p.textContent=t.targetLang,p.style.color="rgba(124, 58, 237, 0.9)",p.style.fontWeight="600",t.badge.appendChild(p);const T={background:"rgba(124, 58, 237, 0.15)",border:"1px solid rgba(124, 58, 237, 0.3)",borderRadius:"6px",color:"rgba(226, 232, 240, 0.9)",padding:"3px 8px",cursor:"pointer",fontSize:"11px",fontFamily:"inherit",transition:"background 0.15s, border-color 0.15s"},h=document.createElement("button");h.setAttribute("data-badge-mode",""),h.textContent="Bilingual",Object.assign(h.style,T),h.addEventListener("mouseenter",()=>{h.style.background="rgba(124, 58, 237, 0.3)"}),h.addEventListener("mouseleave",()=>{h.style.background="rgba(124, 58, 237, 0.15)"}),h.addEventListener("click",()=>{c(t.mode==="bilingual"?"translated":"bilingual")}),t.badge.appendChild(h);const k=document.createElement("button");k.textContent="✕",Object.assign(k.style,{...T,padding:"3px 6px",color:"rgba(226, 232, 240, 0.6)"}),k.title="Remove translation",k.addEventListener("mouseenter",()=>{k.style.background="rgba(239, 68, 68, 0.2)",k.style.borderColor="rgba(239, 68, 68, 0.4)",k.style.color="rgba(239, 68, 68, 0.9)"}),k.addEventListener("mouseleave",()=>{k.style.background="rgba(124, 58, 237, 0.15)",k.style.borderColor="rgba(124, 58, 237, 0.3)",k.style.color="rgba(226, 232, 240, 0.6)"}),k.addEventListener("click",()=>{l()}),t.badge.appendChild(k),document.body.appendChild(t.badge)}async function S(x){t.targetLang=x,t.active=!0,t.mode="bilingual",t.elements=[],t.activeCount=0,d();const g=n();if(g.length===0)return;const f=[];for(const k of g){const A=(k.textContent||"").trim();if(!A)continue;const L=o(k);t.elements.push({original:k,translation:L}),f.push({original:k,translation:L,text:A})}const p=[];for(let k=0;k<f.length;k+=Ue)p.push(f.slice(k,k+Ue));const T=async k=>{for(;t.activeCount>=Yt;)await new Promise(A=>setTimeout(A,100));if(t.active){t.activeCount++;try{const A=k.map(m=>m.text),L=await i(A,t.targetLang);if(!t.active)return;k.forEach((m,N)=>{t.active&&(r(m.translation,L[N]||"[No translation]"),t.mode==="translated"&&(m.original.style.display="none"))})}finally{t.activeCount--}}},h=p.map(k=>T(k));await Promise.all(h)}return{start:S,remove:l,setMode:c}}const Qt=["display","position","flex-direction","align-items","justify-content","gap","flex-wrap","flex","flex-grow","flex-shrink","width","height","min-width","min-height","max-width","max-height","padding","padding-top","padding-right","padding-bottom","padding-left","margin","margin-top","margin-right","margin-bottom","margin-left","border","border-radius","border-color","border-width","border-style","background","background-color","background-image","background-size","color","font-size","font-weight","font-family","line-height","letter-spacing","text-align","text-decoration","text-transform","box-shadow","opacity","overflow","z-index","grid-template-columns","grid-template-rows","grid-gap","transform","transition"];function Jt(e){const t=window.getComputedStyle(e),n={};for(const o of Qt){const r=t.getPropertyValue(o);r&&r!=="none"&&r!=="normal"&&r!=="auto"&&r!=="0px"&&r!=="rgba(0, 0, 0, 0)"&&(n[o]=r)}return n}function Zt(e){const t=e.tagName.toLowerCase(),n=e.className&&typeof e.className=="string"?"."+e.className.trim().split(/\s+/).slice(0,2).join("."):"";return t+n}function en(){const e=document.documentElement.cloneNode(!0),t=["script","noscript",'iframe[src*="ads"]','iframe[src*="track"]','iframe[src*="pixel"]','iframe[width="0"]','iframe[height="0"]','img[src*="pixel"]','img[src*="track"]','img[width="1"]','img[height="1"]','[id*="cookie"]','[class*="cookie"]','[id*="consent"]','[class*="consent"]','[id*="gdpr"]','[class*="gdpr"]','[id*="onetrust"]','[class*="onetrust"]','[id*="CybotCookiebot"]','[data-testid*="cookie"]','[id*="ad-"]','[class*="ad-container"]','[class*="ad-wrapper"]','link[rel="preconnect"]','link[rel="dns-prefetch"]','meta[http-equiv="Content-Security-Policy"]',"style[data-emotion]"];for(const E of t)try{e.querySelectorAll(E).forEach(s=>s.remove())}catch{}e.querySelectorAll("*").forEach(E=>{const s=E.getAttributeNames();for(const a of s)(a.startsWith("on")||a==="data-analytics"||a==="data-tracking")&&E.removeAttribute(a)});const n=e.outerHTML,o={},r=["body","header","nav","main","footer","aside","section","article","h1","h2","h3","h4","h5","h6","p","a","button","input","textarea","ul","ol","li","img","form","table","th","td",'[class*="hero"]','[class*="card"]','[class*="btn"]','[class*="nav"]','[class*="header"]','[class*="footer"]','[class*="sidebar"]','[class*="container"]','[class*="wrapper"]','[class*="grid"]','[class*="flex"]','[class*="modal"]','[class*="banner"]'];let i=0;for(const E of r){if(i>=200)break;try{const s=document.querySelectorAll(E);for(const a of s){if(i>=200)break;const C=Jt(a);if(Object.keys(C).length>0){const b=Zt(a),y=o[b]?`${b}:nth(${i})`:b;o[y]=C,i++}}}catch{}}const u=[];for(const[E,s]of Object.entries(o)){u.push(`${E} {`);for(const[a,C]of Object.entries(s))u.push(`  ${a}: ${C};`);u.push("}"),u.push("")}const c=u.join(`
`),l=new Set,d=["color","background-color","border-color","outline-color"],S=document.querySelectorAll("*");let x=0;for(const E of S){if(x>=500)break;const s=window.getComputedStyle(E);for(const a of d){const C=s.getPropertyValue(a);C&&C!=="rgba(0, 0, 0, 0)"&&C!=="transparent"&&C!=="inherit"&&C!=="initial"&&l.add(C)}x++}const g=Array.from(l).slice(0,50),f=new Set;for(const E of S){if(f.size>=20)break;const a=window.getComputedStyle(E).getPropertyValue("font-family");if(a){const C=a.split(",").map(b=>b.trim().replace(/^["']|["']$/g,""));for(const b of C)b&&!b.includes("inherit")&&!b.includes("initial")&&b.length<50&&f.add(b)}}const p=Array.from(f).slice(0,20),T=E=>{const s=document.querySelector(`meta[property="${E}"], meta[name="${E}"]`);return(s==null?void 0:s.getAttribute("content"))||""},h=document.querySelector('link[rel="icon"], link[rel="shortcut icon"]'),k={title:document.title||"",description:T("description"),og_image:T("og:image"),og_title:T("og:title"),og_description:T("og:description"),og_type:T("og:type"),og_site_name:T("og:site_name"),favicon:(h==null?void 0:h.getAttribute("href"))||""},A={width:window.innerWidth,height:window.innerHeight},L=[];try{for(const E of document.styleSheets){try{const s=E.cssRules||E.rules;if(!s)continue;for(const a of s)if(a instanceof CSSMediaRule&&a.conditionText&&(L.includes(a.conditionText)||L.push(a.conditionText),L.length>=20))break}catch{}if(L.length>=20)break}}catch{}const m=[];document.querySelectorAll("img[src]").forEach(E=>{const s=E.getAttribute("src");if(s&&!s.startsWith("data:")&&m.length<50)try{m.push(new URL(s,location.href).href)}catch{m.push(s)}});const N=[];document.querySelectorAll('link[rel="stylesheet"][href]').forEach(E=>{const s=E.getAttribute("href");if(s&&N.length<20)try{N.push(new URL(s,location.href).href)}catch{N.push(s)}});const M=document.querySelectorAll("*").length;return{html:n,css:c,css_map:o,colors:g,fonts:p,metadata:k,source_url:location.href,viewport:A,asset_urls:{images:m,stylesheets:N},responsive_info:{viewport_width:A.width,media_queries:L},element_count:M}}const tn="https://aura-elnur.duckdns.org";function nn(e,t){var p;const n=tn;let o=n,r="";function i(){return new Promise(T=>{var h;if(!((h=e==null?void 0:e.storage)!=null&&h.local)){T();return}e.storage.local.get(["backendUrl","apiKey"],k=>{var A,L;(A=k==null?void 0:k.backendUrl)!=null&&A.trim()&&(o=k.backendUrl.trim().replace(/\/+$/,"")),(L=k==null?void 0:k.apiKey)!=null&&L.trim()&&(r=k.apiKey.trim()),T()})})}i(),(p=e==null?void 0:e.storage)!=null&&p.onChanged&&e.storage.onChanged.addListener((T,h)=>{var k;if(h==="local"){if(T.backendUrl){const A=T.backendUrl.newValue;o=typeof A=="string"&&A.trim()?A.trim().replace(/\/+$/,""):n}T.apiKey&&(r=((k=T.apiKey.newValue)==null?void 0:k.trim())??"")}});function u(){const T=window.location.hostname,h=window.location.pathname,k=new URLSearchParams(window.location.search);if(!T.match(/^(www\.)?google\./)||h!=="/search"||!k.get("q"))return!1;const A=k.get("tbm");if(A&&["isch","lcl","vid","shop","nws","bks","fin"].includes(A))return!1;const L=k.get("udm");return!(L&&["2","14"].includes(L))}function c(){const h=new URLSearchParams(window.location.search).get("q")||"";if(h)return h;const k=document.querySelector('input[name="q"]');return(k==null?void 0:k.value)||""}function l(){const T=window.getComputedStyle(document.body).backgroundColor;if(!T||T==="transparent")return"light";const h=T.match(/\d+/g);if(h&&h.length>=3){const[k,A,L]=h.map(Number);return .299*k+.587*A+.114*L<128?"dark":"light"}return"light"}function d(T){return T.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;")}function S(T){let h=d(T);return h=h.replace(/\*\*(.+?)\*\*/g,"<strong>$1</strong>"),h=h.replace(/__(.+?)__/g,"<strong>$1</strong>"),h=h.replace(new RegExp("(?<!\\*)\\*(?!\\*)(.+?)(?<!\\*)\\*(?!\\*)","g"),"<em>$1</em>"),h=h.replace(/`([^`]+)`/g,"<code>$1</code>"),h=h.replace(/\[([^\]]+)\]\((https?:\/\/[^)"]+)\)/g,'<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'),h=h.replace(/^[\s]*[-*]\s+(.+)$/gm,"<li>$1</li>"),h=h.replace(/((?:<li>.*<\/li>\n?)+)/g,"<ul>$1</ul>"),h=h.replace(/^[\s]*\d+\.\s+(.+)$/gm,"<li>$1</li>"),h=h.replace(/\n\n+/g,"</p><p>"),h="<p>"+h+"</p>",h=h.replace(/\n/g,"<br>"),h=h.replace(/<p>\s*<\/p>/g,""),h=h.replace(/<\/?(?!(?:strong|em|code|a|li|ul|ol|p|br)\b)[^>]*>/gi,""),h}function x(T,h){const k=/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,A=[];let L;for(;(L=k.exec(h))!==null;)A.push({title:L[1],url:L[2]});if(A.length===0)return;const m=document.createElement("div");m.className="serp-citations";const N=document.createElement("div");N.className="serp-citations-label",N.textContent="Sources",m.appendChild(N);const M=document.createElement("div");M.className="serp-citation-list",A.forEach((E,s)=>{const a=document.createElement("a");a.className="serp-citation-chip",a.href=E.url,a.target="_blank",a.rel="noopener noreferrer";const C=document.createElement("span");C.className="serp-citation-num",C.textContent=String(s+1),a.appendChild(C);const b=document.createTextNode(" "+E.title);a.appendChild(b),M.appendChild(a)}),m.appendChild(M),T.appendChild(m)}function g(T,h,k){const A=document.createElement("div");A.className="serp-footer";const L=document.createElement("button");L.className="serp-followup-btn",L.innerHTML='<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg> Ask follow-up',L.addEventListener("click",()=>{t({type:"OPEN_WITH_TEXT",action:"ask",text:`I searched for "${h}" and got the following AI answer:

${k}

I have a follow-up question: `,url:window.location.href,title:document.title})});const m=document.createElement("span");m.className="serp-powered",m.textContent="Powered by AURA",A.appendChild(L),A.appendChild(m),T.appendChild(A)}async function f(){if(!u()||(await new Promise(O=>{e.storage.local.get(["aura_serp_hidden"],O)})).aura_serp_hidden)return;const h=c();if(!h)return;const k=document.createElement("div");k.id="aura-serp-host",Object.assign(k.style,{position:"fixed",top:"80px",right:"16px",width:"340px",maxHeight:"calc(100vh - 100px)",zIndex:"2147483640",pointerEvents:"auto"}),document.documentElement.appendChild(k);const A=k.attachShadow({mode:"closed"}),m=l()==="dark",N=document.createElement("style");N.textContent=`
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
        background: ${m?"rgba(30, 27, 48, 0.92)":"rgba(255, 255, 255, 0.95)"};
        backdrop-filter: blur(24px) saturate(1.4);
        -webkit-backdrop-filter: blur(24px) saturate(1.4);
        border-radius: 16px;
        overflow-y: auto;
        max-height: calc(100vh - 120px);
        box-shadow: ${m?"0 8px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.06)":"0 8px 40px rgba(0,0,0,0.12), 0 0 0 1px rgba(0,0,0,0.06)"};
        border: 1px solid ${m?"rgba(124, 58, 237, 0.2)":"rgba(124, 58, 237, 0.15)"};
        padding: 20px 24px 16px;
        animation: serp-fade-in 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        position: relative;
        overflow: hidden;
        transition: border-color 0.25s ease;
      }
      .serp-card:hover {
        border-color: ${m?"rgba(124, 58, 237, 0.35)":"rgba(124, 58, 237, 0.3)"};
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
        color: ${m?"rgba(160, 148, 210, 0.9)":"rgba(124, 58, 237, 0.85)"};
        background: ${m?"rgba(124, 58, 237, 0.12)":"rgba(124, 58, 237, 0.08)"};
        border-radius: 8px;
        flex-shrink: 0;
      }
      .serp-title {
        font-size: 14px;
        font-weight: 600;
        color: ${m?"rgba(226, 232, 240, 0.9)":"rgba(30, 27, 48, 0.9)"};
        letter-spacing: -0.01em;
      }
      .serp-title-sub {
        font-size: 11px;
        font-weight: 400;
        color: ${m?"rgba(160, 148, 210, 0.5)":"rgba(100, 90, 140, 0.6)"};
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
        color: ${m?"rgba(160, 148, 210, 0.5)":"rgba(100, 90, 140, 0.5)"};
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.15s, color 0.15s;
        padding: 0;
      }
      .serp-ctrl-btn:hover {
        background: ${m?"rgba(124, 58, 237, 0.15)":"rgba(124, 58, 237, 0.1)"};
        color: ${m?"rgba(224, 214, 255, 1)":"rgba(124, 58, 237, 0.9)"};
      }
      .serp-ctrl-btn[title="Hide AURA answers"]:hover {
        background: rgba(239, 68, 68, 0.12);
        color: rgba(239, 68, 68, 0.9);
      }

      .serp-body {
        font-size: 14px;
        line-height: 1.7;
        color: ${m?"rgba(226, 232, 240, 0.85)":"rgba(30, 27, 48, 0.85)"};
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
        background: ${m?"rgba(124, 58, 237, 0.6)":"rgba(124, 58, 237, 0.5)"};
        animation: serp-pulse 1.2s ease-in-out infinite;
      }
      .serp-loading-dots span:nth-child(2) { animation-delay: 0.2s; }
      .serp-loading-dots span:nth-child(3) { animation-delay: 0.4s; }
      .serp-loading-text {
        font-size: 13px;
        color: ${m?"rgba(160, 148, 210, 0.6)":"rgba(100, 90, 140, 0.6)"};
      }

      .serp-answer {
        white-space: pre-wrap;
        word-break: break-word;
      }
      .serp-answer p { margin-bottom: 8px; }
      .serp-answer p:last-child { margin-bottom: 0; }
      .serp-answer strong, .serp-answer b {
        font-weight: 600;
        color: ${m?"rgba(226, 232, 240, 0.95)":"rgba(30, 27, 48, 0.95)"};
      }
      .serp-answer code {
        background: ${m?"rgba(124, 58, 237, 0.1)":"rgba(124, 58, 237, 0.06)"};
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
        color: ${m?"rgba(160, 148, 255, 0.9)":"rgba(100, 58, 237, 0.9)"};
        text-decoration: none;
      }
      .serp-answer a:hover { text-decoration: underline; }

      .serp-citations {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid ${m?"rgba(255,255,255,0.06)":"rgba(0,0,0,0.06)"};
      }
      .serp-citations-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: ${m?"rgba(160, 148, 210, 0.5)":"rgba(100, 90, 140, 0.5)"};
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
        background: ${m?"rgba(124, 58, 237, 0.1)":"rgba(124, 58, 237, 0.06)"};
        border: 1px solid ${m?"rgba(124, 58, 237, 0.15)":"rgba(124, 58, 237, 0.1)"};
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        color: ${m?"rgba(200, 180, 255, 0.8)":"rgba(100, 58, 237, 0.8)"};
        text-decoration: none;
        transition: background 0.15s, border-color 0.15s;
        max-width: 280px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .serp-citation-chip:hover {
        background: ${m?"rgba(124, 58, 237, 0.2)":"rgba(124, 58, 237, 0.12)"};
        border-color: ${m?"rgba(124, 58, 237, 0.3)":"rgba(124, 58, 237, 0.2)"};
      }
      .serp-citation-num {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        background: ${m?"rgba(124, 58, 237, 0.2)":"rgba(124, 58, 237, 0.1)"};
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
        border-top: 1px solid ${m?"rgba(255,255,255,0.06)":"rgba(0,0,0,0.06)"};
      }
      .serp-followup-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: ${m?"rgba(124, 58, 237, 0.12)":"rgba(124, 58, 237, 0.08)"};
        border: 1px solid ${m?"rgba(124, 58, 237, 0.2)":"rgba(124, 58, 237, 0.15)"};
        border-radius: 8px;
        padding: 7px 14px;
        font-size: 12.5px;
        font-weight: 500;
        font-family: inherit;
        color: ${m?"rgba(200, 180, 255, 0.9)":"rgba(100, 58, 237, 0.9)"};
        cursor: pointer;
        transition: background 0.15s, border-color 0.15s, color 0.15s, transform 0.15s;
      }
      .serp-followup-btn:hover {
        background: ${m?"rgba(124, 58, 237, 0.22)":"rgba(124, 58, 237, 0.15)"};
        border-color: ${m?"rgba(124, 58, 237, 0.35)":"rgba(124, 58, 237, 0.3)"};
        transform: scale(1.01);
      }
      .serp-followup-btn:active { transform: scale(0.98); }
      .serp-powered {
        font-size: 11px;
        color: ${m?"rgba(160, 148, 210, 0.35)":"rgba(100, 90, 140, 0.35)"};
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
        color: ${m?"rgba(226, 232, 240, 0.5)":"rgba(30, 27, 48, 0.5)"};
      }

      .serp-error {
        font-size: 13px;
        color: ${m?"rgba(239, 150, 150, 0.8)":"rgba(200, 50, 50, 0.7)"};
        padding: 4px 0;
      }
    `,A.appendChild(N);const M=document.createElement("div");M.className="serp-card";const E=document.createElement("div");E.className="serp-header";const s=document.createElement("div");s.className="serp-header-left";const a=document.createElement("div");a.className="serp-logo",a.innerHTML='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3L2 21M12 3L22 21M5.8 14.2L18.2 14.2"/></svg>';const C=document.createElement("div"),b=document.createElement("span");b.className="serp-title",b.textContent="AI Answer";const y=document.createElement("span");y.className="serp-title-sub",y.textContent="by AURA",C.appendChild(b),C.appendChild(y),s.appendChild(a),s.appendChild(C);const _=document.createElement("div");_.className="serp-controls";const $=document.createElement("button");$.className="serp-ctrl-btn",$.title="Collapse",$.innerHTML='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>';const D=document.createElement("button");D.className="serp-ctrl-btn",D.title="Hide AURA answers",D.innerHTML='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',_.appendChild($),_.appendChild(D),E.appendChild(s),E.appendChild(_),M.appendChild(E);const H=document.createElement("div");H.className="serp-body";const z=document.createElement("div");z.className="serp-loading";const v=document.createElement("div");v.className="serp-loading-dots",v.innerHTML="<span></span><span></span><span></span>";const w=document.createElement("span");w.className="serp-loading-text",w.textContent=`Thinking about "${h.slice(0,60)}${h.length>60?"...":""}"`,z.appendChild(v),z.appendChild(w),H.appendChild(z),M.appendChild(H),A.appendChild(M);let I=!1;$.addEventListener("click",()=>{I=!I,I?(H.classList.add("collapsed"),$.title="Expand",$.innerHTML='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>'):(H.classList.remove("collapsed"),$.title="Collapse",$.innerHTML='<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>')}),D.addEventListener("click",()=>{e.storage.local.set({aura_serp_hidden:!0}),k.remove()});try{const O=JSON.stringify({message:h,conversation_id:"__serp_answer__",stream:!1,system_context:`The user searched Google for: "${h}". Provide a concise, direct answer to their query. Be helpful and factual. Use markdown formatting sparingly — bold for emphasis, lists where appropriate. If you reference sources, format them as [Source Title](URL) and they will be rendered as citation chips. Keep the answer focused and under 200 words unless the topic requires more detail.`});let B=null;(async()=>{try{B=await new Promise((ne,Q)=>{e.runtime.sendMessage({type:"SERP_FETCH",url:`${o}/api/chat`,body:O,apiKey:r},vo=>{e.runtime.lastError?Q(new Error(e.runtime.lastError.message)):ne(vo)})})}catch{const ne={"Content-Type":"application/json"};r&&(ne["X-API-Key"]=r);const Q=await fetch(`${o}/api/chat`,{method:"POST",headers:ne,body:O,signal:AbortSignal.timeout(3e4)});if(!Q.ok)throw new Error(`HTTP ${Q.status}`);B={ok:!0,text:await Q.text()}}if(!(B!=null&&B.ok))throw new Error((B==null?void 0:B.error)||"Backend unreachable");z.remove();const q=document.createElement("div");q.className="serp-answer",H.appendChild(q);let P="";const G=B.text||"",te=G.split(`
`).filter(ne=>ne.trim());for(const ne of te)try{const Q=JSON.parse(ne);Q.chunk?P+=Q.chunk:Q.response?P=Q.response:Q.content&&(P=Q.content)}catch{P+=ne}if(!P.trim()&&G.trim()&&(P=G),q.innerHTML=S(P),!P.trim()){q.innerHTML='<span class="serp-error">No response from AI.</span>';return}x(H,P),g(M,h,P)})().catch(q=>{z.remove();const P=document.createElement("div");P.className="serp-offline";const G=document.createElement("div");G.className="serp-offline-dot";const te=document.createElement("span");te.className="serp-offline-text",te.textContent=`AURA is offline — backend did not respond (${(q==null?void 0:q.message)||"timeout"})`,P.appendChild(G),P.appendChild(te),H.appendChild(P)})}catch{z.remove();const B=document.createElement("div");B.className="serp-offline";const q=document.createElement("div");q.className="serp-offline-dot";const P=document.createElement("span");P.className="serp-offline-text",P.textContent="AURA is offline — backend did not respond",B.appendChild(q),B.appendChild(P),H.appendChild(B)}}f()}function on(e,t){e.runtime.onMessage.addListener((n,o,r)=>{if(o&&o.id&&o.id!==e.runtime.id)return!1;if(n.type==="EXTRACT_PAGE")return r(t.extractMainContent()),!1;if(n.type==="GET_DOM")return r({ok:!0,dom:t.serializeDOM(),url:location.href,title:document.title}),!1;if(n.type==="EXEC_ACTION")return r(t.execAction(n.action)),!1;if(n.type==="FILL_FORM"){const i=n.fields;let u=0;for(const c of i||[])t.execAction({action:"type",selector:c.selector,text:c.value}).ok&&u++;return r({ok:!0,filled:u,total:(i==null?void 0:i.length)||0}),!1}if(n.type==="SHOW_OCR_OVERLAY")return t.showOcrOverlay(n.dataUrl,r),!0;if(n.type==="PAGE_TRANSLATE")return t.translateActive&&t.removePageTranslation(),t.startPageTranslation(n.targetLang).then(()=>{r({ok:!0})}).catch(i=>{r({ok:!1,error:i.message})}),!0;if(n.type==="TRANSLATE_TOGGLE_MODE")return t.setTranslateMode(n.mode),r({ok:!0}),!1;if(n.type==="TRANSLATE_REMOVE")return t.removePageTranslation(),r({ok:!0}),!1;if(n.type==="TRANSLATE_CHANGE_LANG")return t.translateActive?(t.removePageTranslation(),t.startPageTranslation(n.targetLang).then(()=>{r({ok:!0})}).catch(i=>{r({ok:!1,error:i.message})}),!0):(r({ok:!0}),!1);if(n.type==="SCROLL_TO_HIGHLIGHT")return t.scrollToHighlight(n.id),r({ok:!0}),!1;if(n.type==="SHOW_DOCK")return t.showDock(),r({ok:!0}),!1;if(n.type==="START_CAPTURE_MODE")return t.startCaptureMode(),r({ok:!0}),!1;if(n.type==="STOP_CAPTURE_MODE")return t.stopCaptureMode(),r({ok:!0}),!1;if(n.type==="EXTRACT_FULL_PAGE"){try{const i=t.extractFullPageData();r({ok:!0,data:i})}catch(i){r({ok:!1,error:i.message||"Extraction failed"})}return!1}})}const xe=typeof browser<"u"?browser:chrome,rn=5*60*1e3,an=48*60*60*1e3,sn=10*60*1e3,ln=3,cn=3,dn=2,un=1e4,We=2e3;function pn(e){try{const t=new URL(e);return`${t.protocol}//${t.host}${t.pathname}`}catch{return e}}let Ve=0;function gn(){return location.protocol==="http:"||location.protocol==="https:"}function Ae(e,t={}){const n=Date.now();if(!(n-Ve<rn)){Ve=n;try{xe.runtime.sendMessage({type:"STUCK_SIGNAL",kind:e,url:location.href,title:document.title,...t}).catch(()=>{})}catch{}}}async function mn(e){const t=pn(e);try{const o=(await xe.storage.local.get(["aura_visit_log"])).aura_visit_log||{},r=Date.now(),i=o[t]||{ts:[]};i.ts=i.ts.filter(l=>r-l<an);const u=i.ts[i.ts.length-1]||0;if(!(r-u<sn)){i.ts.push(r),o[t]=i;for(const d of Object.keys(o))o[d].ts.length||delete o[d];const l=Object.keys(o);if(l.length>We){const d=l.sort((S,x)=>(o[S].ts[0]||0)-(o[x].ts[0]||0));for(const S of d.slice(0,l.length-We))delete o[S]}await xe.storage.local.set({aura_visit_log:o})}i.ts.length>=ln&&Ae("tab_revisit",{count:i.ts.length})}catch{}}const Ge=new WeakMap;let ue=null;function fn(){if(!(typeof IntersectionObserver>"u"))try{ue=new IntersectionObserver(n=>{for(const o of n){if(!o.isIntersecting)continue;const r=(Ge.get(o.target)||0)+1;if(Ge.set(o.target,r),r>=cn){const i=(o.target.textContent||"").slice(0,140);Ae("reread",{snippet:i}),ue==null||ue.unobserve(o.target)}}},{threshold:.6,rootMargin:"0px"});const e=document.querySelectorAll("p, article, section, blockquote");let t=0;for(const n of Array.from(e))if((n.textContent||"").trim().length>=120&&(ue.observe(n),t++,t>=200))break}catch{}}const Xe=new WeakMap;function hn(){const e=t=>{const n=t.target;if(!n||!("value"in n)||n.type==="password"||n.type==="hidden")return;const o=Xe.get(n)||{lastEdit:0,emptyAt:[]},r=Date.now();if(n.value===""){if(o.emptyAt=o.emptyAt.filter(i=>r-i<un),o.emptyAt.push(r),o.emptyAt.length>=dn){const i=n.getAttribute("name")||n.getAttribute("id")||"field";Ae("form_flipflop",{field:i}),o.emptyAt=[]}}else o.lastEdit=r;Xe.set(n,o)};document.addEventListener("input",e,{capture:!0,passive:!0})}function bn(){window.addEventListener("beforeunload",()=>{if(document.querySelector('textarea:focus, [contenteditable="true"]:focus'))try{xe.runtime.sendMessage({type:"STUCK_SIGNAL",kind:"workflow_boundary_unsaved",url:location.href,title:document.title})}catch{}})}function xn(){if(!gn())return;mn(location.href);const e=()=>{fn(),hn(),bn()};document.readyState==="complete"?setTimeout(e,500):window.addEventListener("load",()=>setTimeout(e,500),{once:!0})}const Me=typeof browser<"u"?browser:chrome,yn=800,wn=3,vn=2e3,En=200,Cn="aura-ghost-chip";let j=null,ee=null,ce=!0;const kn=["accounts.google.com","login.microsoftonline.com","okta.com","1password.com","lastpass.com","bitwarden.com","dashlane.com","mail.google.com","outlook.live.com","outlook.office.com","mail.yahoo.com","mail.proton.me"];function Sn(){const e=location.hostname;return!!(kn.some(t=>e===t||e.endsWith("."+t))||/\bbank\b|\bpayment\b|\bcheckout\b/i.test(e))}function Tn(){if(ee&&document.body.contains(ee))return ee;const e=document.createElement("div");return e.id=Cn,e.setAttribute("role","tooltip"),e.setAttribute("aria-live","polite"),e.style.cssText=["position:fixed","z-index:2147483646","pointer-events:none","max-width:480px","padding:6px 10px","border-radius:8px","background:rgba(20,20,28,0.92)","color:rgba(220,220,235,0.92)",'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif',"font-size:12px","line-height:1.4","box-shadow:0 6px 24px rgba(0,0,0,0.35), 0 0 0 1px rgba(124,58,237,0.35)","opacity:0","transform:translateY(-4px)","transition:opacity 0.12s ease, transform 0.12s ease","display:none"].join(";"),document.body.appendChild(e),ee=e,e}function ie(){ee&&(ee.style.opacity="0",ee.style.transform="translateY(-4px)",ee.style.display="none",ee.textContent=""),j&&(j.suggestion="")}function Ln(e,t){const n=Tn();n.textContent="";const o=document.createElement("span");o.textContent="Aura · Tab to accept",o.style.cssText="color:rgba(167,139,250,0.85);margin-right:8px;font-size:10.5px;text-transform:uppercase;letter-spacing:0.05em",n.appendChild(o);const r=document.createElement("span");r.textContent=t,n.appendChild(r);const i=e.getBoundingClientRect(),u=Math.max(8,window.innerWidth-500),c=Math.min(Math.max(8,i.left),u),l=Math.min(i.bottom+6,window.innerHeight-40);n.style.left=`${c}px`,n.style.top=`${l}px`,n.style.display="block",n.offsetHeight,n.style.opacity="1",n.style.transform="translateY(0)"}function Ye(e){return e instanceof HTMLTextAreaElement||e instanceof HTMLInputElement?e.value||"":e.textContent||""}function An(e,t){if(e instanceof HTMLTextAreaElement||e instanceof HTMLInputElement){const n=e.selectionStart??e.value.length;e.value=e.value.slice(0,n)+t+e.value.slice(n);const o=n+t.length;try{e.setSelectionRange(o,o)}catch{}e.dispatchEvent(new Event("input",{bubbles:!0}))}else if(e.isContentEditable){const n=window.getSelection();if(n&&n.rangeCount>0){const o=n.getRangeAt(0),r=document.createTextNode(t);o.insertNode(r),o.setStartAfter(r),o.setEndAfter(r),n.removeAllRanges(),n.addRange(o),e.dispatchEvent(new Event("input",{bubbles:!0}))}}ie(),j=null}async function Mn(e,t){if(!j)return;j.abortCtrl&&j.abortCtrl.abort();const n=new AbortController;j.abortCtrl=n;try{const o=await Me.runtime.sendMessage({type:"GHOST_COMPLETE",text:t.slice(-vn),url:location.href,title:document.title});if(!o||typeof o.continuation!="string")return;const r=String(o.continuation||"").trim();if(!r||r.length>En)return;j&&j.el===e&&j.text===t&&(j.suggestion=r,Ln(e,r))}catch{}}function Nn(e){!j||j.el!==e||(j.debounceTimer&&window.clearTimeout(j.debounceTimer),j.debounceTimer=window.setTimeout(()=>{if(!j||j.el!==e)return;const t=Ye(e);j.text=t,!(t.length<wn)&&Mn(e,t)},yn))}function _n(e){return!e||!(e instanceof HTMLElement)?!1:e instanceof HTMLInputElement?!(["password","email","tel","number","date","url"].includes(e.type)||e.getAttribute("autocomplete")==="off"):e instanceof HTMLTextAreaElement?e.getAttribute("autocomplete")!=="off":e.isContentEditable}function In(e){if(!ce||Sn()||!_n(e.target))return;const t=e.target;j={el:t,text:Ye(t),suggestion:"",debounceTimer:null,abortCtrl:null},ie(),Nn(t)}function On(e){if(j){if(e.key==="Tab"&&!e.shiftKey&&j.suggestion){e.preventDefault(),An(j.el,j.suggestion);return}j.suggestion&&ie()}}function Ke(){ie()}function $n(){var e,t,n,o;if(!(location.protocol!=="http:"&&location.protocol!=="https:")){try{(t=(e=Me.storage)==null?void 0:e.local)==null||t.get(["ghostTextMode"],r=>{const i=r==null?void 0:r.ghostTextMode;(i==="inline"||i==="off")&&(ce=!1)}),(o=(n=Me.storage)==null?void 0:n.onChanged)==null||o.addListener((r,i)=>{i==="local"&&r.ghostTextMode&&(ce=r.ghostTextMode.newValue==="chip",ce||ie())})}catch{}document.addEventListener("input",In,{capture:!0,passive:!0}),document.addEventListener("keydown",On,{capture:!0}),window.addEventListener("scroll",Ke,{capture:!0,passive:!0}),window.addEventListener("resize",Ke,{passive:!0}),window.addEventListener("blur",ie)}}window.addEventListener("aura-ghost-disable",()=>{ce=!1,ie()}),window.addEventListener("aura-ghost-enable",()=>{ce=!0});const Ne=typeof browser<"u"?browser:chrome,Hn=600,zn=8,Rn=2e3,Bn=3,Pn="aura-ghost-inline",Dn="aura-ghost-mirror",Fn=["accounts.google.com","login.microsoftonline.com","okta.com","1password.com","lastpass.com","bitwarden.com","dashlane.com","mail.google.com","outlook.live.com","outlook.office.com","mail.yahoo.com","mail.proton.me"];function qn(){const e=location.hostname;return!!(Fn.some(t=>e===t||e.endsWith("."+t))||/\bbank\b|\bpayment\b|\bcheckout\b/i.test(e))}let F=null,ae=null,ye=null,Qe=!0,we="inline",Je=null,ve=null;function jn(){if(ve)return ve;try{return Je=document.createElement("canvas"),ve=Je.getContext("2d"),ve}catch{return null}}function _e(){if(ae&&document.body.contains(ae))return ae;const e=document.createElement("div");return e.id=Pn,e.style.cssText=["position:fixed","z-index:2147483646","pointer-events:none","color:rgba(167,139,250,0.55)","font-family:inherit","white-space:pre-wrap","overflow:hidden","display:none"].join(";"),document.body.appendChild(e),ae=e,e}function Un(){if(ye&&document.body.contains(ye))return ye;const e=document.createElement("div");return e.id=Dn,e.style.cssText=["position:absolute","top:0","left:-9999px","visibility:hidden","white-space:pre-wrap","word-wrap:break-word","overflow-wrap:break-word","pointer-events:none"].join(";"),document.body.appendChild(e),ye=e,e}function Wn(e,t){const n=window.getComputedStyle(e),o=["boxSizing","width","height","overflow","fontFamily","fontSize","fontWeight","fontStyle","fontVariant","letterSpacing","textTransform","textIndent","lineHeight","paddingTop","paddingRight","paddingBottom","paddingLeft","borderTopWidth","borderRightWidth","borderBottomWidth","borderLeftWidth","borderTopStyle","borderRightStyle","borderBottomStyle","borderLeftStyle"];for(const r of o)t.style[r]=n[r]}function Vn(e,t){try{const n=Un(),o=e.getBoundingClientRect();Wn(e,n),n.style.width=`${o.width}px`,n.style.height="auto";const r=e.selectionStart??e.value.length,i=e.value.substring(0,r).replace(/\n$/,`
 `);n.textContent=i;const u=document.createElement("span");u.textContent="​",n.appendChild(u);const c=u.getBoundingClientRect(),l=n.getBoundingClientRect(),d=c.left-l.left,S=c.top-l.top,x=o.left+d-e.scrollLeft,g=o.top+S-e.scrollTop;if(x<o.left||x>o.right||g<o.top||g>o.bottom)return!1;const f=_e(),p=window.getComputedStyle(e);return f.style.font=p.font,f.style.fontSize=p.fontSize,f.style.fontFamily=p.fontFamily,f.style.lineHeight=p.lineHeight,f.style.letterSpacing=p.letterSpacing,f.style.left=`${x}px`,f.style.top=`${g}px`,f.style.maxWidth=`${o.right-x-4}px`,f.textContent=t,f.style.display="",!0}catch{return!1}}function Gn(e){var t;try{const n=window.getSelection();if(!n||n.rangeCount===0)return!1;const o=n.getRangeAt(0).cloneRange();o.collapse(!1);const r=document.createElement("span");r.textContent="​",o.insertNode(r);const i=r.getBoundingClientRect(),u=r.parentNode;if(u==null||u.removeChild(r),i.width===0&&i.height===0)return!1;const c=_e(),l=(t=n.anchorNode)==null?void 0:t.parentElement;if(l){const d=window.getComputedStyle(l);c.style.font=d.font,c.style.fontSize=d.fontSize,c.style.fontFamily=d.fontFamily,c.style.lineHeight=d.lineHeight,c.style.letterSpacing=d.letterSpacing}return c.style.left=`${i.left}px`,c.style.top=`${i.top}px`,c.style.maxWidth=`${Math.max(240,window.innerWidth-i.left-20)}px`,c.textContent=e,c.style.display="",!0}catch{return!1}}function Xn(e,t){try{const n=e.getBoundingClientRect(),o=window.getComputedStyle(e),r=jn();if(!r)return!1;r.font=o.font;const i=e.selectionStart??e.value.length,u=r.measureText(e.value.substring(0,i)).width,c=parseFloat(o.paddingLeft)||0,l=parseFloat(o.borderLeftWidth)||0,d=n.left+l+c+u-e.scrollLeft,S=n.top+(n.height-parseFloat(o.fontSize))/2;if(d>n.right-8)return!1;const x=_e();return x.style.font=o.font,x.style.fontSize=o.fontSize,x.style.fontFamily=o.fontFamily,x.style.letterSpacing=o.letterSpacing,x.style.left=`${d}px`,x.style.top=`${S}px`,x.style.maxWidth=`${n.right-d-6}px`,x.textContent=t,x.style.display="",!0}catch{return!1}}function Ze(e,t){return e.tagName==="TEXTAREA"?Vn(e,t):e.tagName==="INPUT"?Xn(e,t):e.isContentEditable?Gn(t):!1}function oe(){ae&&(ae.style.display="none",ae.textContent=""),F&&(F.suggestion="")}function Yn(e){return e.tagName==="TEXTAREA"||e.tagName==="INPUT"?e.value||"":e.isContentEditable&&e.textContent||""}function Kn(e){return(e.trim().match(/\S+/g)||[]).length}function Qn(e){if(!(e instanceof HTMLElement))return!1;if(e.tagName==="INPUT"){const t=(e.type||"").toLowerCase();if(["password","email","tel","number","search","url","hidden"].includes(t)||e.getAttribute("autocomplete")==="off")return!1}return!!(e.tagName==="TEXTAREA"||e.isContentEditable||e.tagName==="INPUT")}async function Jn(e,t){return new Promise(n=>{try{Ne.runtime.sendMessage({type:"GHOST_COMPLETE",text:e,url:location.href,title:document.title},o=>{var i,u;if(t.aborted){n(null);return}const r=((u=(i=o==null?void 0:o.continuation)==null?void 0:i.trim)==null?void 0:u.call(i))||"";n(r||null)})}catch{n(null)}})}function Zn(e){var o;if(!Qe||we!=="inline")return;const t=e.target;if(!Qn(t))return;const n=Yn(t);if(n.length<zn||n.length>Rn){oe();return}if(Kn(n)<Bn){oe();return}F!=null&&F.debounceTimer&&clearTimeout(F.debounceTimer),(o=F==null?void 0:F.abortCtrl)==null||o.abort(),F={el:t,text:n,suggestion:"",debounceTimer:null,abortCtrl:null},F.debounceTimer=window.setTimeout(async()=>{if(!F||F.el!==t)return;const r=new AbortController;F.abortCtrl=r;const i=await Jn(n,r.signal);if(!i||r.signal.aborted){oe();return}if(!F||F.el!==t)return;F.suggestion=i,Ze(t,i)||oe()},Hn)}function eo(e){if(!(!F||!F.suggestion)&&e.target===F.el){if(e.key==="Tab"){e.preventDefault();const t=F.el,n=F.suggestion;if(t.tagName==="TEXTAREA"||t.tagName==="INPUT"){const o=t,r=o.selectionStart??o.value.length,i=o.value.substring(0,r),u=o.value.substring(r);o.value=i+n+u,o.selectionStart=o.selectionEnd=r+n.length,o.dispatchEvent(new Event("input",{bubbles:!0}))}else if(t.isContentEditable){const o=window.getSelection();if(o&&o.rangeCount>0){const r=o.getRangeAt(0);r.deleteContents(),r.insertNode(document.createTextNode(n)),r.collapse(!1)}}oe(),F=null;return}e.key==="Escape"&&(oe(),F=null)}}function to(){oe()}function et(){F!=null&&F.suggestion&&F.el&&Ze(F.el,F.suggestion)}function no(){var e,t;try{(t=(e=Ne.storage)==null?void 0:e.local)==null||t.get(["ghostTextMode"],n=>{const o=n==null?void 0:n.ghostTextMode;(o==="chip"||o==="off"||o==="inline")&&(we=o)})}catch{}}function oo(){var e,t;if(qn()){Qe=!1;return}no();try{(t=(e=Ne.storage)==null?void 0:e.onChanged)==null||t.addListener((n,o)=>{if(o==="local"&&n.ghostTextMode){const r=n.ghostTextMode.newValue;(r==="chip"||r==="off"||r==="inline")&&(we=r),we!=="inline"&&oe()}})}catch{}document.addEventListener("input",Zn,!0),document.addEventListener("keydown",eo,!0),document.addEventListener("blur",to,!0),window.addEventListener("scroll",et,!0),window.addEventListener("resize",et)}const Ie=typeof browser<"u"?browser:chrome;let pe=!1,ge=0,Ee=0,me="",Oe=!1;const ro=["accounts.google.com","login.microsoftonline.com","okta.com","1password.com","lastpass.com","bitwarden.com","dashlane.com"];function io(){const e=location.hostname;return!!(ro.some(t=>e.endsWith(t))||/\bbank|\bpayment|\bcheckout\b/i.test(e))}function tt(){return!(location.protocol!=="http:"&&location.protocol!=="https:"||io())}function ao(){const e=document.documentElement,t=window.innerHeight,n=Math.max(1,(e.scrollHeight||t)-t),o=window.scrollY,r=Math.round(o/n*100);r>Ee&&(Ee=r)}function so(){var e;try{const t=((e=window.getSelection())==null?void 0:e.toString())||"";t.length>=20&&t.length<=500&&t.length>me.length&&(me=t)}catch{}}function lo(){if(!pe||!tt()||!ge)return null;const e=Date.now()-ge;if(e<3e3)return null;const t={url:location.href,title:document.title,dwell_ms:e,scroll_max_pct:Ee,timestamp:Date.now()};return me&&(t.selection=me),t}function $e(){if(Oe)return;const e=lo();if(e){Oe=!0;try{Ie.runtime.sendMessage({type:"LIFELOG_EVENT",event:e}).catch(()=>{})}catch{}}}async function co(){try{const e=await Ie.storage.local.get(["lifelogEnabled"]);pe=!!(e!=null&&e.lifelogEnabled)}catch{}}function uo(){var e,t;tt()&&(co().then(()=>{pe&&(ge=Date.now(),Ee=0,me="",Oe=!1,window.addEventListener("scroll",ao,{passive:!0}),document.addEventListener("selectionchange",so,{passive:!0}),document.addEventListener("visibilitychange",()=>{document.visibilityState==="hidden"&&$e()}),window.addEventListener("pagehide",$e),window.addEventListener("beforeunload",$e))}),(t=(e=Ie.storage)==null?void 0:e.onChanged)==null||t.addListener((n,o)=>{o!=="local"||!n.lifelogEnabled||(pe=!!n.lifelogEnabled.newValue,pe&&!ge&&(ge=Date.now()))}))}const fe=typeof browser<"u"?browser:chrome,po=60*60*1e3,go=2500,mo=["accounts.google.com","login.microsoftonline.com","okta.com","1password.com","lastpass.com","bitwarden.com","dashlane.com","mail.google.com","outlook.live.com","outlook.office.com","mail.yahoo.com","mail.proton.me"];function fo(){const e=location.hostname;return!!(mo.some(t=>e===t||e.endsWith("."+t))||/\bbank\b|\bpayment\b|\bcheckout\b/i.test(e))}function nt(){return!(location.protocol!=="http:"&&location.protocol!=="https:"||fo())}function ot(){try{const e=new URL(location.href);return e.origin+e.pathname}catch{return location.href}}async function ho(){var e,t;try{const n=await((t=(e=fe.storage)==null?void 0:e.local)==null?void 0:t.get(["ambientSurfaceLog"])),o=(n==null?void 0:n.ambientSurfaceLog)||{},r=ot(),i=o[r];return i?Date.now()-i<po:!1}catch{return!0}}async function bo(){var e,t,n,o;try{const r=await((t=(e=fe.storage)==null?void 0:e.local)==null?void 0:t.get(["ambientSurfaceLog"])),i=(r==null?void 0:r.ambientSurfaceLog)||{};i[ot()]=Date.now();const u=Date.now()-24*60*60*1e3;for(const c in i)i[c]<u&&delete i[c];(o=(n=fe.storage)==null?void 0:n.local)==null||o.set({ambientSurfaceLog:i})}catch{}}async function xo(){if(!nt()||await ho())return;const e=document.title||"";if(e.trim())try{fe.runtime.sendMessage({type:"AMBIENT_SURFACE_REQUEST",url:location.href,title:e,host:location.hostname}),bo()}catch{}}function yo(){var e,t;if(nt())try{(t=(e=fe.storage)==null?void 0:e.local)==null||t.get(["ambientSurfaceEnabled"],n=>{n!=null&&n.ambientSurfaceEnabled&&setTimeout(()=>{document.visibilityState==="visible"&&xo()},go)})}catch{}}const Y=typeof browser<"u"?browser:chrome,rt=["aura-shadow-host","aura-dock-shadow","aura-host","aura-quick-action-host","aura-highlight-host","aura-img-toolbar-host","aura-capture-host"];let Ce=null;function he(e,t){var n;try{t?Y.runtime.sendMessage(e,t):Y.runtime.sendMessage(e)}catch(o){if(((o==null?void 0:o.message)??"").includes("Extension context invalidated")){Ce==null||Ce.remove();for(const i of rt)(n=document.getElementById(i))==null||n.remove()}}}function it(e,t=2e3){const n=document.createElement("div");Object.assign(n.style,{position:"fixed",top:"16px",left:"50%",transform:"translateX(-50%)",background:"rgba(10,8,24,0.92)",backdropFilter:"blur(16px)",WebkitBackdropFilter:"blur(16px)",border:"1px solid rgba(124,58,237,0.35)",borderRadius:"8px",padding:"8px 16px",color:"rgba(226,232,240,0.92)",fontSize:"13px",fontFamily:'-apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", system-ui, sans-serif',fontWeight:"500",zIndex:"2147483647",pointerEvents:"none",boxShadow:"0 4px 16px rgba(0,0,0,0.4)",whiteSpace:"nowrap"}),n.textContent=e,document.documentElement.appendChild(n),setTimeout(()=>n.remove(),t)}function wo(){var m;if(window.__auraToolbarMounted)return;window.__auraToolbarMounted=!0;for(const N of rt)(m=document.getElementById(N))==null||m.remove();const e=document.createElement("div");e.id="aura-shadow-host",Object.assign(e.style,{position:"fixed",top:"0",left:"0",width:"0",height:"0",zIndex:"2147483647",pointerEvents:"none",overflow:"visible"}),document.documentElement.appendChild(e),Ce=e;const t=e.attachShadow({mode:"open"}),n=document.createElement("style");n.textContent=pt(),t.appendChild(n);const o=st();ut(o,Y),o.subscribe(N=>{e.style.setProperty("--aura-accent",N.accent),e.style.setProperty("--aura-glow",N.glow)});function r(N){const M=document.createElement("div");return M.dataset.auraModule=N,Object.assign(M.style,{all:"unset",pointerEvents:"none"}),t.appendChild(M),M}const i=r("fab"),u=r("ghost-bar"),c=r("modal"),l=r("highlights"),d=r("capture"),S=r("link-preview"),x=ht(),g=xt(),f=Ct(),p=kt(),T=Tt(),h=At(),k=Nt();x.init(i,o,Y),g.init(u,o,Y),f.init(c,o,Y),p.init(l,o,Y),T.init(document.body,o,Y),h.init(d,o,Y),k.init(S,o,Y),g.onAskClicked(N=>{N.type==="text"?f.openWithText(N.text,N.rect):f.openWithImage(N.imageUrl,N.rect)}),f.onAction((N,M,E)=>{he({type:"OPEN_WITH_TEXT",action:N,text:M,url:location.href,title:document.title})}),p.setShowToast(it);const A=Kt(Y);let L=!1;Wt(Y,he,it),Vt(he),Gt(he),nn(Y,he),xn(),$n(),oo(),uo(),yo(),on(Y,{extractMainContent:qt,serializeDOM:It,execAction:$t,showOcrOverlay:Ht,startPageTranslation:async N=>{await A.start(N),L=!0},removePageTranslation:()=>{A.remove(),L=!1},setTranslateMode:N=>A.setMode(N),scrollToHighlight:N=>p.scrollTo(N),showDock:()=>x.showDock(),startCaptureMode:()=>h.start(),stopCaptureMode:()=>h.stop(),extractFullPageData:en,get translateActive(){return L}})}wo()})();
