import { useState, useRef, useEffect, useCallback } from 'react';
import {
  PaperAirplaneIcon,
  ArrowDownTrayIcon,
  ArrowLeftIcon,
  ArrowRightIcon,
  ArrowsPointingOutIcon,
  ArrowsPointingInIcon,
  StopIcon,
  PresentationChartBarIcon,
} from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

/* ── Types ── */
interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

type SlideCount = 5 | 8 | 10 | 15;
type SlideStyle = 'Professional' | 'Minimal' | 'Creative' | 'Dark' | 'Colorful';

/* ── Constants ── */
const SLIDE_COUNTS: SlideCount[] = [5, 8, 10, 15];

const SLIDE_STYLES: { label: SlideStyle; color: string }[] = [
  { label: 'Professional', color: '#2563eb' },
  { label: 'Minimal',      color: '#6b7280' },
  { label: 'Creative',     color: '#7c3aed' },
  { label: 'Dark',         color: '#1e293b' },
  { label: 'Colorful',     color: '#f59e0b' },
];

const SLIDE_TEMPLATES = [
  { label: 'Pitch Deck', icon: '📊', desc: 'Startup investor pitch', prompt: 'A startup pitch deck: title slide with company name, problem statement with pain point icons, solution overview with product mockup, market size (TAM/SAM/SOM circles), business model with revenue streams, traction metrics with growth chart, competitive landscape 2x2 matrix, team bios with photos, financial projections line chart, and call-to-action with contact info.' },
  { label: 'Quarterly Report', icon: '📈', desc: 'Business performance metrics', prompt: 'A quarterly business report: executive summary, revenue breakdown stacked bar chart, customer acquisition metrics (CAC vs LTV), churn analysis funnel, operational KPIs table, regional performance color-coded, team headcount area chart, strategic initiatives progress cards, risk register, and next quarter priorities.' },
  { label: 'Product Launch', icon: '🚀', desc: 'New product announcement', prompt: 'A product launch presentation: dramatic title with product name, problem journey (3 user pain points), solution reveal with product screenshot, 3 key features (one per slide with demo visuals), comparison table vs competitors, pricing tiers (3 columns, recommended highlighted), customer testimonials, launch timeline, and pre-order CTA.' },
  { label: 'Thesis Defense', icon: '🎓', desc: 'Academic research presentation', prompt: 'An academic thesis defense: title page with university, outline, literature review with key citations, research hypothesis, methodology (data collection, sample, variables), results with statistical charts and error bars, discussion of findings, limitations, conclusions and future work, and references in academic format.' },
  { label: 'Tech Architecture', icon: '⚙️', desc: 'System design overview', prompt: 'A technical architecture presentation: system architecture diagram with component boxes, technology stack logos grid, API data flow with animated arrows, database schema, performance metrics (latency, throughput charts), scalability approach, security layers, CI/CD pipeline stages, monitoring dashboard, and Q&A slide. Developer dark theme with syntax-highlight colors.' },
  { label: 'Sales Demo', icon: '💼', desc: 'Product walkthrough for prospects', prompt: 'A sales demo presentation: product name with main value prop, 3 customer pain points with icons, quick product overview, 4 feature walkthrough slides with screenshots, success metrics (efficiency, cost savings, time saved with before/after numbers), pricing table, customer case study quote, and next steps with booking CTA.' },
  { label: 'Workshop Agenda', icon: '🎯', desc: 'Training session schedule', prompt: 'A workshop agenda: title with date and instructor, learning outcomes checklist, full-day timeline with break times, 4 individual session detail slides (topic, learning points, activities, timing), hands-on exercise instructions, resources and materials list, and feedback/next steps. Warm educational color scheme.' },
  { label: 'Marketing Campaign', icon: '📢', desc: 'Campaign strategy and results', prompt: 'A marketing campaign presentation: campaign name and goal, target audience persona card, KPIs with targets, channel strategy (social, email, display, influencer with budget %), creative assets showcase, campaign timeline gantt chart, results overview with percent changes, detailed performance charts (impressions, CTR, conversion funnel, ROI), learnings, and next campaign recommendations.' },
  { label: 'Case Study', icon: '📋', desc: 'Client success story', prompt: 'A case study presentation: client name and industry, challenge statement, the solution (4 bullet points), implementation timeline with phases, before/after metrics comparison (4 KPIs with % improvement), results showcase (main achievement large number), customer testimonial quote, team collaboration, and ROI summary with contact CTA.' },
  { label: 'Conference Talk', icon: '🎤', desc: 'Public speaking slides', prompt: 'A conference talk: bold title slide with speaker name, speaker bio with headshot, talk outline (5 main points), opening hook with full-screen visual, 5 content sections (each with title, diagram/chart, key bullets), real-world example with screenshot, key takeaways (3 points with icons), resources with QR code, and Q&A. Minimal dark theme with single accent color.' },
  { label: 'Company Overview', icon: '🏢', desc: 'Organization introduction', prompt: 'A company overview: mission statement, company values (4 cards with icons), product/service descriptions with hero images, company timeline milestones, team structure org chart, recognition and awards, client logos, and vision for the future. Minimalist white theme with professional typography.' },
  { label: 'Board Report', icon: '📊', desc: 'Executive summary for leadership', prompt: 'A board report: executive summary bullets, key highlights (3 wins with metrics), strategic progress against annual goals (color-coded status), financial snapshot (revenue, margin, cash), revenue deep-dive by segment, customer metrics (NRR, churn, CAC, LTV), risk register with mitigation, upcoming priorities, and budget forecast chart. Corporate formal design.' },
  { label: 'Research Findings', icon: '🔬', desc: 'Study results presentation', prompt: 'A research study presentation: study title and authors, research question, literature summary, hypothesis, methodology overview, sample characteristics table, 4 results slides (histogram, box plot, correlation heatmap, grouped bar chart with p-values), discussion, limitations, implications, and references. Academic clean design.' },
  { label: 'Brand Guidelines', icon: '🎨', desc: 'Brand identity standards', prompt: 'A brand guidelines presentation: brand logo and variations, color palette (primary, secondary, accent with hex codes), typography system (heading and body fonts with sizes), logo usage rules (dos and donts), imagery style guide with examples, iconography style, social media templates, business card and stationery, voice and tone examples, and brand application examples.' },
  { label: 'Sprint Review', icon: '🔄', desc: 'Agile sprint retrospective', prompt: 'A sprint review presentation: sprint number and dates, sprint goal recap, completed stories (card list with points), demo screenshots of delivered features, velocity chart (last 6 sprints), burndown chart, bugs fixed and technical debt addressed, team feedback (what went well, what to improve), and next sprint planning priorities. Clean dev-friendly design.' },
  { label: 'All-Hands', icon: '👥', desc: 'Company update, wins, priorities', prompt: 'A company all-hands presentation: mission reminder, quarter highlights (3 big wins with metrics), financial snapshot, team spotlights with photos, product roadmap preview, hiring updates, Q&A prompts, and one-shared-goal closing slide. Warm team-focused design.' },
  { label: 'Monthly Update', icon: '📅', desc: 'Progress, blockers, next steps', prompt: 'A monthly team update: goals vs progress (bar chart), key achievements bullet list, active projects status cards (on-track/at-risk/blocked color-coded), team changes (new hires, role changes), lessons learned, next month priorities, and open questions for leadership.' },
  { label: 'Investor Update', icon: '💼', desc: 'Metrics, runway, asks', prompt: 'An investor update deck: TL;DR slide, North Star metric trend, revenue / MRR / ARR charts, customer wins and logos, product shipped this period, team changes, cash/runway card, key risks, and specific asks (intros, hires, advice). Clean professional black/white design.' },
  { label: 'Post-Mortem', icon: '🔥', desc: 'Incident analysis, timeline, actions', prompt: 'A blameless post-mortem presentation: incident summary (what happened, duration, impact, severity), timeline of events with timestamps, root cause analysis (5 whys), what went well vs what went poorly, action items with owners and dates, metrics affected chart, and prevention measures. Somber professional design.' },
  { label: 'Roadmap Review', icon: '🗺️', desc: 'Quarters, themes, commitments', prompt: 'A product roadmap presentation: vision statement, strategic themes (3 pillars), quarter-by-quarter roadmap swim lanes (Q1/Q2/Q3/Q4), committed vs aspirational clearly labeled, dependencies callouts, success metrics per theme, and how we measure progress. Strategic executive design.' },
  { label: 'Competitive Analysis', icon: '⚔️', desc: 'Landscape, positioning, differentiation', prompt: 'A competitive analysis deck: market landscape slide, competitor matrix (us vs 3-4 others on 6 dimensions), positioning map (2x2), feature comparison table, pricing comparison, differentiation strengths, threats to watch, and strategic recommendations. Analytical business design.' },
  { label: 'User Research Readout', icon: '🔍', desc: 'Interviews, themes, insights', prompt: 'A user research presentation: study overview (participants, methodology), participant breakdown (demographics), key themes (3-5 top findings with supporting quotes), behavioral insights, surprising discoveries, opportunity areas, and recommended next steps. Empathetic human-centered design.' },
  { label: 'A/B Test Results', icon: '🧪', desc: 'Variants, metrics, decision', prompt: 'An experiment results presentation: hypothesis and test design, control vs variant metrics comparison, primary metric significance (p-value, confidence interval), secondary metrics table, segment breakdown, surprising findings, decision (ship/kill/iterate), and learnings for future tests. Clean data-driven design.' },
  { label: 'NPS Review', icon: '❤️', desc: 'Score, drivers, verbatims', prompt: 'An NPS quarterly review: current NPS score and trend, score distribution histogram, promoter/passive/detractor breakdown, top themes from verbatims with sample quotes, improvement actions taken from last quarter, category-level NPS (features/support/pricing), and priorities for next quarter.' },
  { label: 'Budget Proposal', icon: '💵', desc: 'Request, rationale, ROI', prompt: 'A budget proposal presentation: executive summary, current-state problem, proposed investment breakdown, ROI projections with sensitivity analysis, implementation timeline Gantt, risks and mitigations, alternatives considered, and requested approval. Finance formal design.' },
  { label: 'OKR Planning', icon: '🎯', desc: 'Objectives, key results, commitments', prompt: 'An OKR planning session deck: company-level objectives, team objective options with trade-offs, draft key results with measurability rubric, dependencies between teams, aspirational vs committed split, capacity check, and decision-making framework. Collaborative strategy design.' },
  { label: 'Hackathon Pitch', icon: '⚡', desc: 'Problem, demo, ask', prompt: 'A hackathon presentation (under 3 minutes): bold problem hook, 30-second live demo description, how it works (3 key technical points with diagram), impact stats, team selfie, and what we need next (hiring, funding, users). Energetic bold design.' },
  { label: 'Keynote', icon: '⭐', desc: 'Big idea, story, call-to-action', prompt: 'A keynote-style presentation with cinematic feel: opening quote full-screen, establishing story (why we are here), three-act structure (tension, reveal, call-to-action), 7-8 visually striking slides with minimal text and large imagery, and memorable closing line. High-contrast bold design.' },
  { label: 'TED-style Talk', icon: '💡', desc: 'Idea worth spreading', prompt: 'A TED-style talk deck: opening hook (personal story or surprising statistic), the idea in one sentence, three supporting arguments with vivid examples, counterargument addressed, call-to-reflection (not just action), closing image full-screen, and 5-word takeaway. Minimal typographic design, one idea per slide.' },
  { label: 'Webinar', icon: '🖥️', desc: 'Learn, demo, Q&A', prompt: 'A webinar presentation: welcome with housekeeping (mute/chat/Q&A), presenter bio, learning objectives, content sections (3-4 topics each with summary slide + deep dive), live demo placeholder slide, poll/interactive slide, Q&A prompt, and next-step CTA with resource links.' },
  { label: 'Workshop Facilitator', icon: '📓', desc: 'Agenda, exercises, debrief', prompt: 'A workshop facilitator deck: welcome and norms, learning objectives, mini-lecture slides (3 concepts), exercise instructions (timing, group size, prompts), debrief discussion questions, synthesis slide, key takeaways, and action planning template. Educational collaborative design.' },
  { label: 'Training Module', icon: '🎓', desc: 'Lesson plan, assessment', prompt: 'A training module presentation: module title and learning outcomes, prerequisite knowledge check, concept explanations with visual aids, worked examples, hands-on exercise instructions, knowledge check quiz questions, summary and key takeaways, and links to further resources. Clean educational design.' },
  { label: 'Book Summary', icon: '📖', desc: 'Key ideas, quotes, application', prompt: 'A book summary presentation: book cover and author, one-sentence thesis, 5-7 key ideas (one per slide with explanation and example), most impactful quotes, how to apply in work/life, recommended chapters to read in full, and connections to other books. Clean literary design.' },
  { label: 'Book Club', icon: '📚', desc: 'Discussion questions, themes', prompt: 'A book club discussion deck: book overview, author background, plot/argument summary (no spoilers version), themes and motifs, favorite passages, discussion questions (10), opposing viewpoints prompts, and next book suggestion. Warm conversational design.' },
  { label: 'Grand Rounds', icon: '🩺', desc: 'Medical case presentation', prompt: 'A medical grand-rounds presentation: case history (anonymized), chief complaint and vitals, examination findings, differential diagnosis, investigations ordered and results, working diagnosis and reasoning, management plan, outcome and follow-up, teaching points, and references. Clean clinical design.' },
  { label: 'Journal Club', icon: '🧠', desc: 'Paper critique, discussion', prompt: 'A journal club presentation: paper citation and significance, background/gap, research question and hypothesis, study design, methods overview, key results with figures, authors conclusions, your critical appraisal (strengths/weaknesses/bias), applicability to practice, and discussion questions. Academic serif design.' },
  { label: 'Scientific Poster', icon: '🖼️', desc: 'Compact research summary', prompt: 'A poster-style presentation (each slide = a poster panel): title and authors banner, background and hypothesis, methods with flow diagram, results with 2-3 key figures, discussion and implications, conclusions bullet list, acknowledgments and funding, and QR code placeholder for full paper. Academic portrait layout.' },
  { label: 'Grant Pitch', icon: '💰', desc: 'Significance, innovation, approach', prompt: 'A research grant pitch: specific aims slide, significance/impact, innovation and novelty, preliminary data, research approach per aim (methods, milestones, risks, alternatives), team and environment, budget justification summary, and timeline. Formal academic design.' },
  { label: 'Legal Opening', icon: '⚖️', desc: 'Mock trial, story, evidence', prompt: 'A legal opening statement deck: case title, one-sentence theory of the case, narrative of events, key evidence preview (3 items with thumbnails), witness list overview, elements the jury must find, closing ask, and thematic tagline. Formal austere design.' },
  { label: 'Campaign Rally', icon: '📣', desc: 'Platform, crowd, call-to-action', prompt: 'A political campaign deck: candidate name and tagline, why Im running (origin story), platform pillars (3-5 policy areas with specific proposals), accomplishments track record, endorsements, volunteer and donate CTAs, event schedule, and closing inspirational slide. Bold patriotic design.' },
  { label: 'Town Hall', icon: '🏛️', desc: 'Community update, Q&A', prompt: 'A town hall presentation: welcome and agenda, community issues addressed (budget, safety, services), key numbers and metrics, completed projects, planned initiatives, how residents can engage, Q&A format, and contact info slide. Civic friendly design.' },
  { label: 'Real Estate Listing', icon: '🏠', desc: 'Property pitch, photos, specs', prompt: 'A real estate listing presentation: property photo hero, address and price, key specs (beds/baths/sqft/year), room-by-room photo slides with captions, neighborhood highlights, school district info, recent comparables, offer process, and agent contact slide. Elegant property-focused design.' },
  { label: 'Travel Itinerary', icon: '✈️', desc: 'Trip plan, day-by-day, map', prompt: 'A travel itinerary presentation: destination hero with dates, flight/transportation summary, day-by-day slides (morning/afternoon/evening activities with photos), accommodations, restaurants to try, budget breakdown, packing checklist, and emergency contacts. Adventurous travel design.' },
  { label: 'Product Teardown', icon: '🔍', desc: 'Competitor deep dive', prompt: 'A product teardown deck: product overview (screenshots, positioning), first-time user experience flow slides, strengths (what they do well), weaknesses (gaps and friction), strategic insights about their approach, implications for our strategy, and what we should steal/avoid. Analytical design.' },
  { label: 'Partnership Pitch', icon: '🤝', desc: 'Why us, mutual value, ask', prompt: 'A partnership pitch deck: our company overview, why we value their brand/audience, proposed partnership structure (3 options), mutual value breakdown with metrics, similar partnerships that worked, timeline and next steps, success metrics, and contact/CTA. Business-forward design.' },
  { label: 'Retrospective', icon: '🔄', desc: 'Start/Stop/Continue format', prompt: 'A team retrospective presentation: period covered, team morale check, wins to celebrate (5 items), Start items (things to begin), Stop items (things to end), Continue items (things working well), action items with owners, and appreciation slide. Warm team design.' },
  { label: 'Onboarding Kickoff', icon: '👋', desc: 'New hire welcome deck', prompt: 'A new-hire onboarding deck: welcome slide with team photos, company mission and values, how we work (norms, tools, communication), meet-the-team spotlights, 30/60/90-day plan, resources and setup checklist, first-week calendar, and who to ask for help. Friendly inclusive design.' },
  { label: 'Reorg Announcement', icon: '🔀', desc: 'Structure change, rationale', prompt: 'A reorganization announcement: why were changing (context), what specifically is changing (old vs new org charts side by side), impact on each team, what is NOT changing (reassurance), reporting changes, timeline and milestones, how we will transition, and Q&A prompts. Direct honest design.' },
  { label: 'Status Update', icon: '📊', desc: 'Project health, risks, next', prompt: 'A project status update: project name and sponsor, overall status (RAG: green/amber/red) with reasoning, key milestones hit/missed, burndown or timeline visual, risks and issues register (with mitigations), asks and decisions needed, and upcoming milestones. Clean PM design.' },
  { label: 'Interview Prep', icon: '💬', desc: 'Candidate assessment framework', prompt: 'An interview kickoff deck for a hiring loop: role summary and success profile, assessment rubric (3-5 competencies with behavioral anchors), interview plan by interviewer (topics, time allocation), common biases to watch for, debrief protocol and calibration, and strong-hire example. Clean HR design.' },
];

const SYSTEM_PROMPT = `You are a senior presentation designer building a production-quality deck in a single HTML file using Reveal.js 5.

OUTPUT FORMAT
- Output ONLY complete HTML starting with <!DOCTYPE html> — no markdown fences, no prose
- If user asks for modifications, return the COMPLETE updated HTML

REQUIRED STACK
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reset.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/theme/black.min.css" id="theme">
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5/plugin/notes/notes.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5/plugin/highlight/highlight.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5/plugin/highlight/monokai.min.css">

Google Fonts: pick a distinctive pair for the deck (serif display + grotesque body works well for editorial; grotesque pair for technical; pick one based on vibe).
Optional: Chart.js v4 CDN for data slides; Lucide icons.

STRUCTURE (this is non-negotiable — Reveal requires it)
<body>
  <div class="reveal">
    <div class="slides">
      <section data-transition="fade">
        <h1>Slide 1</h1>
        <aside class="notes">Speaker notes for slide 1.</aside>
      </section>
      <section>
        <h2>Slide 2</h2>
        <ul>
          <li class="fragment">Appears on next click</li>
          <li class="fragment fade-up">Appears with fade-up</li>
        </ul>
      </section>
      <!-- vertical stack example -->
      <section>
        <section><h2>Parent</h2></section>
        <section><h3>Child 1 (down arrow)</h3></section>
      </section>
    </div>
  </div>
  <script>
    Reveal.initialize({
      hash: true,
      controls: true,
      progress: true,
      slideNumber: 'c/t',
      transition: 'fade', // or 'slide' | 'convex' | 'concave' | 'zoom' | 'none'
      plugins: [RevealNotes, RevealHighlight]
    });
  </script>
</body>

THEME SELECTION
- Dark (default): black.min.css
- Light editorial: white.min.css, simple.min.css
- Vibrant: league.min.css, moon.min.css, night.min.css
- Minimal: beige.min.css, serif.min.css
Or omit the theme CSS entirely and style from scratch with your own tokens.

DESIGN DIRECTIVES (quality bar)
- One dominant color per deck. Pick a register and commit.
- Distinctive typography pair — no Inter/Poppins/Plus-Jakarta defaults. Examples: Fraunces + Inter Tight (editorial), IBM Plex Sans + IBM Plex Serif (technical), Space Grotesk + Space Mono (modern tech).
- Each slide must have ONE clear idea. Don't cram bullet lists — break into multiple slides and use fragments for progressive reveal.
- Big type. Slide H1 should be 72-120px. Body 24-36px minimum.
- Use fragments (class="fragment") for reveals. Variants: fade-in (default), fade-up, fade-down, fade-left, fade-right, grow, shrink, highlight-current-blue, highlight-red.
- Section slides (data-background-color, data-background-image) to break chapters.
- Charts/diagrams get full slides, not tiny corner thumbnails.
- Speaker notes (<aside class="notes">) on every content slide — keep under 40 words.

SLIDE COUNT & STYLE
- Respect the user-requested slide count. Don't pad with filler.
- Respect the user-requested style (Professional/Minimal/Creative/Dark/Colorful) — pick theme + typography accordingly.

AVOID AI-SLOP TELLS
- No purple gradient title slide
- No "Agenda" slide with generic bullet-point icons
- No clipart-style emoji in headings of a professional deck
- No "Thank you!" final slide with rainbow gradient — end on a memorable line, a question, or a CTA`;

/* ── Component ── */
export function SlidesPanel() {
  const [chatMessages, setChatMessages]     = useState<ChatMessage[]>([]);
  const [input, setInput]                   = useState('');
  const [topic, setTopic]                   = useState('');
  const [slideCount, setSlideCount]         = useState<SlideCount>(8);
  const [slideStyle, setSlideStyle]         = useState<SlideStyle>('Professional');
  const [isGenerating, setIsGenerating]     = useState(false);
  const [currentHtml, setCurrentHtml]       = useState('');
  const [streamingCode, setStreamingCode]   = useState('');
  const [currentSlide, setCurrentSlide]     = useState(1);
  const [totalSlides, setTotalSlides]       = useState(0);
  const [isFullscreen, setIsFullscreen]     = useState(false);
  const [selectedModel, setSelectedModel]   = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu]   = useState(false);
  const [presenterMode, setPresenterMode]  = useState(false);
  const [presenterTimer, setPresenterTimer] = useState(0);
  const presenterTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const chatScrollRef  = useRef<HTMLDivElement>(null);
  const abortRef       = useRef<AbortController | null>(null);
  const iframeRef      = useRef<HTMLIFrameElement>(null);
  const modelMenuRef   = useRef<HTMLDivElement>(null);
  const fullscreenRef  = useRef<HTMLDivElement>(null);

  /* ── Presenter timer ── */
  useEffect(() => {
    if (presenterMode) {
      setPresenterTimer(0);
      presenterTimerRef.current = setInterval(() => setPresenterTimer(t => t + 1), 1000);
    } else {
      if (presenterTimerRef.current) clearInterval(presenterTimerRef.current);
    }
    return () => { if (presenterTimerRef.current) clearInterval(presenterTimerRef.current); };
  }, [presenterMode]);

  const formatTimer = (s: number) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

  /* ── Extract speaker notes from HTML ── */
  const getSpeakerNotes = useCallback((): string[] => {
    if (!currentHtml) return [];
    const notes: string[] = [];
    const re = /data-notes=["']([^"']*)["']/gi;
    let m;
    while ((m = re.exec(currentHtml)) !== null) {
      notes.push(m[1].replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'));
    }
    return notes;
  }, [currentHtml]);

  /* ── Auto-scroll chat ── */
  useEffect(() => {
    chatScrollRef.current?.scrollTo({ top: chatScrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [chatMessages]);

  /* ── Cleanup abort on unmount ── */
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  /* ── Fetch models ── */
  useEffect(() => {
    apiFetch('/api/models')
      .then(res => res.json())
      .then(data => {
        const all = [
          ...(data.chatgpt_models    || []),
          ...(data.direct_api_models || []),
          ...(data.cloud_models      || []),
          ...(data.local_models      || []),
        ];
        if (all.length > 0) setAvailableModels(all);
      })
      .catch(() => {});
  }, []);

  /* ── Close model menu on outside click ── */
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setShowModelMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  /* ── Count slides in generated HTML ── */
  const countSlides = useCallback((html: string): number => {
    const matches = html.match(/<section[^>]*class=['"][^'"]*slide[^'"]*['"]/gi);
    return matches ? matches.length : 0;
  }, []);

  /* ── Update slide counter from iframe postMessage ── */
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      if (e.source !== iframeRef.current?.contentWindow) return;
      if (e.data?.type === 'slideChange') {
        setCurrentSlide(e.data.current ?? 1);
        setTotalSlides(e.data.total ?? 0);
      }
    };
    window.addEventListener('message', handler);
    return () => window.removeEventListener('message', handler);
  }, []);

  /* ── Reset slide position when new HTML is set ── */
  useEffect(() => {
    if (currentHtml) {
      const count = countSlides(currentHtml);
      setTotalSlides(count);
      setCurrentSlide(1);
    }
  }, [currentHtml, countSlides]);

  /* ── Navigate via postMessage ── */
  const sendNavMessage = useCallback((direction: 'prev' | 'next') => {
    iframeRef.current?.contentWindow?.postMessage({ type: 'navigate', direction }, '*');
    setCurrentSlide(prev => {
      if (direction === 'prev') return Math.max(1, prev - 1);
      return Math.min(totalSlides || prev, prev + 1);
    });
  }, [totalSlides]);

  /* ── Fullscreen toggle ── */
  const toggleFullscreen = useCallback(() => {
    if (!isFullscreen) {
      fullscreenRef.current?.requestFullscreen?.().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.().catch(() => {});
      setIsFullscreen(false);
    }
  }, [isFullscreen]);

  useEffect(() => {
    const handler = () => {
      if (!document.fullscreenElement) setIsFullscreen(false);
    };
    document.addEventListener('fullscreenchange', handler);
    return () => document.removeEventListener('fullscreenchange', handler);
  }, []);

  /* ── Generate slides ── */
  const handleSend = useCallback(async (message: string) => {
    if (!message.trim() || isGenerating) return;

    const userMsg: ChatMessage = { role: 'user', content: message };
    setChatMessages(prev => [...prev, userMsg]);
    setInput('');
    setTopic('');
    setIsGenerating(true);

    const isFirstGeneration = chatMessages.length === 0;
    const systemCtx = isFirstGeneration
      ? `${SYSTEM_PROMPT}\n\nStyle: ${slideStyle}. Number of slides: ${slideCount}.`
      : currentHtml
        ? `${SYSTEM_PROMPT}\n\nCurrent presentation HTML:\n${currentHtml}`
        : SYSTEM_PROMPT;

    const history = chatMessages.map(m => ({ role: m.role, content: m.content }));
    const controller = new AbortController();
    abortRef.current = controller;
    setStreamingCode('');

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          system_prompt: systemCtx,
          history,
          ...(selectedModel ? { model: selectedModel } : {}),
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      let fullResponse = '';
      const assistantMsg: ChatMessage = { role: 'assistant', content: '' };
      setChatMessages(prev => [...prev, assistantMsg]);

      if (res.body) {
        const reader  = res.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });

          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') continue;
              try {
                const parsed = JSON.parse(data);
                const text = parsed.choices?.[0]?.delta?.content || parsed.content || parsed.chunk || '';
                if (text) {
                  fullResponse += text;
                  setStreamingCode(fullResponse);
                  setChatMessages(prev => {
                    const updated = [...prev];
                    updated[updated.length - 1] = { role: 'assistant', content: fullResponse };
                    return updated;
                  });
                }
              } catch {
                fullResponse += data;
                setStreamingCode(fullResponse);
              }
            } else if (line.trim() && !line.startsWith(':')) {
              fullResponse += line;
              setStreamingCode(fullResponse);
              setChatMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = { role: 'assistant', content: fullResponse };
                return updated;
              });
            }
          }
        }
      } else {
        fullResponse = await res.text();
        setChatMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'assistant', content: fullResponse };
          return updated;
        });
      }

      /* Extract HTML */
      let html = fullResponse.trim();
      const fenceMatch = html.match(/```html?\s*\n([\s\S]*?)```/);
      if (fenceMatch) html = fenceMatch[1].trim();

      if (html.includes('<!DOCTYPE') || html.includes('<html') || html.includes('<section')) {
        setCurrentHtml(html);
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setChatMessages(prev => [
          ...prev,
          { role: 'assistant', content: `Error: ${e.message}. Make sure the backend is running.` },
        ]);
      }
    } finally {
      setIsGenerating(false);
      setStreamingCode('');
      abortRef.current = null;
    }
  }, [chatMessages, currentHtml, isGenerating, selectedModel, slideCount, slideStyle]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsGenerating(false);
  }, []);

  const handleDownload = useCallback(() => {
    if (!currentHtml) return;
    const blob = new Blob([currentHtml], { type: 'text/html' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `aura-slides-${Date.now()}.html`;
    a.click();
    URL.revokeObjectURL(url);
  }, [currentHtml]);

  const handleGenerate = useCallback(() => {
    if (!topic.trim()) return;
    const prompt = `Create a ${slideCount}-slide ${slideStyle.toLowerCase()} presentation about: ${topic}`;
    handleSend(prompt);
  }, [topic, slideCount, slideStyle, handleSend]);

  /* ── Inject postMessage listener into srcdoc ── */
  const buildSrcdoc = (html: string): string => {
    const listenerScript = `
<script>
(function() {
  window.addEventListener('message', function(e) {
    if (!e.data || e.data.type !== 'navigate') return;
    var slides = document.querySelectorAll('.slide');
    if (!slides.length) return;
    var current = parseInt(document.body.dataset.currentSlide || '0', 10);
    if (e.data.direction === 'next') current = Math.min(slides.length - 1, current + 1);
    else current = Math.max(0, current - 1);
    document.body.dataset.currentSlide = current;
    slides.forEach(function(s, i) {
      s.style.display = i === current ? '' : 'none';
    });
    try { window.parent.postMessage({ type: 'slideChange', current: current + 1, total: slides.length }, '*'); } catch(e) {}
  });
  // Report initial slide count after load
  window.addEventListener('load', function() {
    var slides = document.querySelectorAll('.slide');
    if (slides.length) {
      try { window.parent.postMessage({ type: 'slideChange', current: 1, total: slides.length }, '*'); } catch(e) {}
    }
  });
})();
</script>`;
    // Insert before </body> if present, otherwise append
    if (html.includes('</body>')) {
      return html.replace('</body>', listenerScript + '</body>');
    }
    return html + listenerScript;
  };

  const isFirstMessage = chatMessages.length === 0;

  return (
    <div className="flex flex-col md:flex-row h-full overflow-hidden">
      {/* ── Left: Chat panel ── */}
      <div className="flex flex-col md:w-[400px] md:min-w-[300px] md:border-r border-b md:border-b-0 border-chat-border flex-shrink-0 max-md:max-h-[45vh] bg-surface-0">
        {/* Header */}
        <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
          <h2 className="text-sm font-semibold text-chat-text">Slides Builder</h2>
          <p className="text-[10px] text-chat-text-secondary mt-0.5">Describe a topic and Aura will generate a presentation</p>
        </div>

        {/* Chat messages */}
        <div ref={chatScrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
          {/* Initial configuration form */}
          {isFirstMessage && (
            <div className="space-y-3">
              <p className="text-xs text-chat-text-secondary">Configure your presentation:</p>

              {/* Topic input */}
              <div>
                <label className="text-[10px] text-chat-text-secondary uppercase tracking-wide mb-1 block">Topic</label>
                <input
                  type="text"
                  value={topic}
                  onChange={e => setTopic(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') handleGenerate(); }}
                  placeholder="e.g. The Future of Renewable Energy"
                  className="w-full px-3 py-2 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm outline-none focus:border-chat-accent placeholder-chat-text-secondary/50"
                  disabled={isGenerating}
                />
              </div>

              {/* Slide count */}
              <div>
                <label className="text-[10px] text-chat-text-secondary uppercase tracking-wide mb-1 block">Slides</label>
                <div className="flex gap-1.5">
                  {SLIDE_COUNTS.map(n => (
                    <button
                      key={n}
                      onClick={() => setSlideCount(n)}
                      className="flex-1 py-1.5 rounded-lg text-xs font-medium border transition-all"
                      style={{
                        background:   slideCount === n ? 'var(--chat-accent)' : 'var(--surface-1)',
                        borderColor:  slideCount === n ? 'var(--chat-accent)' : 'var(--border-default)',
                        color:        slideCount === n ? '#fff' : 'var(--text-secondary)',
                      }}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>

              {/* Style selector */}
              <div>
                <label className="text-[10px] text-chat-text-secondary uppercase tracking-wide mb-1 block">Style</label>
                <div className="grid grid-cols-2 gap-1.5">
                  {SLIDE_STYLES.map(s => (
                    <button
                      key={s.label}
                      onClick={() => setSlideStyle(s.label)}
                      className="flex items-center gap-2 px-3 py-2 rounded-lg border text-xs font-medium transition-all"
                      style={{
                        background:  slideStyle === s.label ? 'var(--surface-2)' : 'var(--surface-1)',
                        borderColor: slideStyle === s.label ? s.color : 'var(--border-default)',
                        color:       slideStyle === s.label ? 'var(--text-primary)' : 'var(--text-secondary)',
                      }}
                    >
                      <span
                        className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                        style={{ background: s.color }}
                      />
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Quick templates */}
              <div>
                <label className="text-[10px] text-chat-text-secondary uppercase tracking-wide mb-1 block">Quick Start</label>
                <div className="grid grid-cols-2 gap-1 max-h-[180px] overflow-y-auto pr-1">
                  {SLIDE_TEMPLATES.map(t => (
                    <button
                      key={t.label}
                      onClick={() => { setTopic(t.prompt); }}
                      className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg border text-left transition-all hover:border-chat-accent/50"
                      style={{ background: 'var(--surface-1)', borderColor: 'var(--border-default)' }}
                    >
                      <span className="text-sm shrink-0">{t.icon}</span>
                      <div className="min-w-0">
                        <div className="text-[11px] font-medium text-chat-text truncate">{t.label}</div>
                        <div className="text-[9px] text-chat-text-secondary truncate">{t.desc}</div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Generate button */}
              <button
                onClick={handleGenerate}
                disabled={!topic.trim() || isGenerating}
                className="w-full py-2 rounded-lg text-sm font-medium text-white transition-opacity disabled:opacity-40"
                style={{ background: 'var(--chat-accent)' }}
              >
                Generate Presentation
              </button>

              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-chat-border" />
                </div>
                <div className="relative flex justify-center">
                  <span className="px-2 text-[10px] text-chat-text-secondary bg-surface-0">or type a request</span>
                </div>
              </div>
            </div>
          )}

          {/* Messages */}
          {chatMessages.map((msg, i) => (
            <div key={i} className={`text-sm ${msg.role === 'user' ? 'text-right' : ''}`}>
              {msg.role === 'user' ? (
                <div className="inline-block px-3 py-2 rounded-xl bg-chat-accent text-white max-w-[90%] text-left text-xs">
                  {msg.content.length > 200 ? msg.content.slice(0, 200) + '...' : msg.content}
                </div>
              ) : (
                <div className="text-xs text-chat-text-secondary">
                  {msg.content.includes('<!DOCTYPE') || msg.content.includes('<section')
                    ? (
                      <span className="text-green-400">
                        {isGenerating && i === chatMessages.length - 1
                          ? `Building slides... (${Math.round(msg.content.length / 1024)}KB)`
                          : `Generated presentation (${Math.round(msg.content.length / 1024)}KB)`
                        }
                      </span>
                    )
                    : msg.content.length > 300 ? msg.content.slice(0, 300) + '...' : msg.content
                  }
                </div>
              )}
            </div>
          ))}

          {isGenerating && (
            <div className="flex items-center gap-2 text-xs text-purple-400">
              <div className="shimmer-bar h-2 w-20" />
              Generating slides...
            </div>
          )}
        </div>

        {/* Chat input */}
        <div className="p-3 border-t border-chat-border flex-shrink-0">
          <div className="flex gap-2">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend(input);
                }
              }}
              placeholder={currentHtml ? 'e.g. "Make slide 3 more visual"' : 'Describe your presentation...'}
              className="flex-1 p-2.5 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm resize-none outline-none focus:border-chat-accent placeholder-chat-text-secondary/50"
              rows={2}
              disabled={isGenerating}
            />
            <button
              onClick={isGenerating ? handleStop : () => handleSend(input)}
              disabled={!isGenerating && !input.trim()}
              className="self-end p-2.5 rounded-lg bg-chat-accent hover:opacity-90 disabled:opacity-40 text-white transition-opacity"
            >
              {isGenerating
                ? <StopIcon className="w-4 h-4" />
                : <PaperAirplaneIcon className="w-4 h-4" />
              }
            </button>
          </div>

          {/* Model selector */}
          <div className="flex items-center mt-1.5" ref={modelMenuRef}>
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowModelMenu(p => !p)}
                className="flex items-center gap-1 text-[10px] text-chat-text-secondary hover:text-chat-text transition-colors px-2 py-1 rounded-md"
                style={{ background: 'var(--border-subtle)' }}
              >
                <span className="max-w-[140px] truncate">
                  {selectedModel ? selectedModel.split('/').pop() : 'Auto'}
                </span>
                <svg className="w-2.5 h-2.5 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {showModelMenu && availableModels.length > 0 && (
                <div
                  style={{
                    position: 'absolute', bottom: 28, left: 0, width: 220, maxHeight: 280,
                    background: 'var(--surface-1)', border: '1px solid var(--border-default)',
                    borderRadius: 10, overflow: 'hidden', zIndex: 50,
                  }}
                >
                  <div style={{ maxHeight: 280, overflowY: 'auto', padding: 4 }}>
                    <button
                      onClick={() => { setSelectedModel(null); setShowModelMenu(false); }}
                      className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors"
                      style={{
                        color:      !selectedModel ? 'var(--text-primary)' : 'var(--text-secondary)',
                        background: !selectedModel ? 'var(--surface-3)' : 'transparent',
                      }}
                    >
                      Auto (recommended)
                    </button>
                    {availableModels.map(m => (
                      <button
                        key={m}
                        onClick={() => { setSelectedModel(m); setShowModelMenu(false); }}
                        className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors truncate"
                        style={{
                          color:      selectedModel === m ? 'var(--text-primary)' : 'var(--text-secondary)',
                          background: selectedModel === m ? 'var(--surface-3)' : 'transparent',
                        }}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ── Right: Preview ── */}
      <div className="flex-1 flex flex-col min-w-0 bg-surface-0">
        {/* Toolbar */}
        <div className="flex items-center gap-1 px-3 py-2 border-b border-chat-border flex-shrink-0 flex-wrap">
          {/* Slide counter */}
          {totalSlides > 0 && (
            <span className="text-[10px] text-chat-text-secondary px-2 py-1 rounded-md" style={{ background: 'var(--surface-1)' }}>
              {currentSlide} / {totalSlides}
            </span>
          )}

          {/* Navigation */}
          <button
            onClick={() => sendNavMessage('prev')}
            disabled={!currentHtml || currentSlide <= 1}
            className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
            title="Previous slide"
          >
            <ArrowLeftIcon className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => sendNavMessage('next')}
            disabled={!currentHtml || currentSlide >= totalSlides}
            className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
            title="Next slide"
          >
            <ArrowRightIcon className="w-3.5 h-3.5" />
          </button>

          <div className="flex-1" />

          {/* Live streaming indicator */}
          {isGenerating && streamingCode && (
            <span className="text-[10px] text-green-400 px-2">
              Writing... {Math.round(streamingCode.length / 1024)}KB
            </span>
          )}

          {/* Presenter mode */}
          <button
            onClick={() => setPresenterMode(p => !p)}
            disabled={!currentHtml || totalSlides === 0}
            className={`px-2 py-1 rounded text-[10px] font-medium transition-colors disabled:opacity-30 ${
              presenterMode ? 'bg-amber-500/20 text-amber-400' : 'text-chat-text-secondary hover:text-chat-text'
            }`}
            title="Toggle presenter view with notes and timer"
          >
            {presenterMode ? '✕ Exit Presenter' : '🎤 Present'}
          </button>

          {/* Fullscreen */}
          <button
            onClick={toggleFullscreen}
            disabled={!currentHtml}
            className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
            title={isFullscreen ? 'Exit fullscreen' : 'Present fullscreen'}
          >
            {isFullscreen
              ? <ArrowsPointingInIcon className="w-3.5 h-3.5" />
              : <ArrowsPointingOutIcon className="w-3.5 h-3.5" />
            }
          </button>

          {/* Download */}
          <button
            onClick={handleDownload}
            disabled={!currentHtml}
            className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
            title="Download as HTML"
          >
            <ArrowDownTrayIcon className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Preview area */}
        <div ref={fullscreenRef} className={`${presenterMode ? 'flex-[2]' : 'flex-1'} overflow-hidden relative`}>
          {isGenerating && streamingCode && !currentHtml ? (
            /* Live code stream before first render */
            <pre className="p-4 text-xs font-mono text-green-400 whitespace-pre-wrap leading-relaxed h-full overflow-auto bg-surface-1">
              {streamingCode}
              <span className="inline-block w-1.5 h-3.5 bg-green-400 animate-pulse ml-0.5 align-middle" />
            </pre>
          ) : currentHtml ? (
            <iframe
              ref={iframeRef}
              srcDoc={buildSrcdoc(currentHtml)}
              sandbox="allow-scripts"
              className="w-full h-full border-none"
              title="Slides preview"
            />
          ) : (
            <div className="flex items-center justify-center h-full text-chat-text-secondary text-sm">
              <div className="text-center">
                <PresentationChartBarIcon className="w-12 h-12 mx-auto mb-3 opacity-20" />
                <p className="font-medium">Your slides will appear here</p>
                <p className="text-[10px] mt-1 opacity-60">Enter a topic and click Generate</p>
              </div>
            </div>
          )}
        </div>

        {/* Presenter panel — notes + timer */}
        {presenterMode && currentHtml && (
          <div className="border-t border-chat-border flex-shrink-0 flex" style={{ height: 160, background: 'var(--surface-1)' }}>
            {/* Speaker notes */}
            <div className="flex-1 p-3 overflow-y-auto border-r border-chat-border/30">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-chat-text-secondary mb-1.5">Speaker Notes</div>
              {(() => {
                const notes = getSpeakerNotes();
                const note = notes[currentSlide - 1];
                return note ? (
                  <p className="text-xs text-chat-text leading-relaxed">{note}</p>
                ) : (
                  <p className="text-xs text-chat-text-secondary/40 italic">No notes for this slide</p>
                );
              })()}
            </div>
            {/* Timer + slide info */}
            <div className="w-[180px] p-3 flex flex-col items-center justify-center gap-2">
              <div className="text-2xl font-mono font-bold text-chat-text">{formatTimer(presenterTimer)}</div>
              <div className="text-[10px] text-chat-text-secondary">Elapsed Time</div>
              <div className="flex items-center gap-3 mt-1">
                <div className="text-center">
                  <div className="text-lg font-bold text-chat-text">{currentSlide}</div>
                  <div className="text-[9px] text-chat-text-secondary">Current</div>
                </div>
                <div className="text-chat-text-secondary/30">/</div>
                <div className="text-center">
                  <div className="text-lg font-bold text-chat-text-secondary">{totalSlides}</div>
                  <div className="text-[9px] text-chat-text-secondary">Total</div>
                </div>
              </div>
              <button
                onClick={() => setPresenterTimer(0)}
                className="text-[10px] text-chat-text-secondary hover:text-chat-text mt-1"
              >
                Reset Timer
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
