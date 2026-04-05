import { useState, useRef, useEffect, useCallback } from 'react';
import {
  PaperAirplaneIcon, ArrowDownTrayIcon,
  DevicePhoneMobileIcon, DeviceTabletIcon, ComputerDesktopIcon,
  ArrowUturnLeftIcon, ArrowUturnRightIcon,
  ArrowTopRightOnSquareIcon, CodeBracketIcon, EyeIcon,
  StopIcon, CursorArrowRaysIcon, XMarkIcon,
} from '@heroicons/react/24/outline';
import { buildSrcdoc } from '../utils/artifactRenderer';
import { highlightCode } from '../utils/codeHighlighter';

/* ── Types ── */
type DeviceSize = 'desktop' | 'tablet' | 'mobile';
type ViewMode = 'preview' | 'code' | 'split';

const DEVICE_WIDTHS: Record<DeviceSize, string> = {
  desktop: '100%',
  tablet: '768px',
  mobile: '375px',
};

const DEVICE_FRAMES: Record<DeviceSize, { outer: string; inner: string; notch: boolean }> = {
  desktop: { outer: '', inner: '', notch: false },
  tablet:  { outer: 'rounded-[20px] border-[8px] border-gray-700 shadow-2xl', inner: 'rounded-[12px]', notch: false },
  mobile:  { outer: 'rounded-[36px] border-[10px] border-gray-800 shadow-2xl relative', inner: 'rounded-[26px]', notch: true },
};

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  imagePreview?: string;
}

interface Version {
  html: string;
  timestamp: number;
}

/* ── Color Palettes ── */
const COLOR_PALETTES = [
  { name: 'Purple', primary: '#7c3aed', secondary: '#a78bfa', accent: '#c4b5fd', bg: '#0f0a1a' },
  { name: 'Blue', primary: '#3b82f6', secondary: '#60a5fa', accent: '#93c5fd', bg: '#0a1628' },
  { name: 'Green', primary: '#10b981', secondary: '#34d399', accent: '#6ee7b7', bg: '#0a1a14' },
  { name: 'Red', primary: '#ef4444', secondary: '#f87171', accent: '#fca5a5', bg: '#1a0a0a' },
  { name: 'Orange', primary: '#f97316', secondary: '#fb923c', accent: '#fdba74', bg: '#1a110a' },
  { name: 'Pink', primary: '#ec4899', secondary: '#f472b6', accent: '#f9a8d4', bg: '#1a0a14' },
  { name: 'Cyan', primary: '#06b6d4', secondary: '#22d3ee', accent: '#67e8f9', bg: '#0a1a1c' },
  { name: 'Neutral', primary: '#6b7280', secondary: '#9ca3af', accent: '#d1d5db', bg: '#111827' },
  { name: 'Custom', primary: '', secondary: '', accent: '', bg: '' },
];

/* ── Templates ── */
const TEMPLATE_CATEGORIES = [
  {
    category: 'Business',
    templates: [
      { label: 'SaaS Landing', icon: '🚀', desc: 'Hero, features, pricing, testimonials', prompt: 'Create a modern SaaS landing page with hero section, features grid (3 cards with icons), testimonials carousel, pricing table (3 tiers), FAQ accordion, and footer. Professional purple/blue gradient theme.' },
      { label: 'Startup', icon: '💡', desc: 'Bold hero, problem/solution, team', prompt: 'Create a startup landing page with a bold hero with animated gradient text, problem/solution section, how-it-works steps, team section with photo placeholders, investor logos bar, and CTA section.' },
      { label: 'Agency', icon: '🏢', desc: 'Full-screen hero, case studies, portfolio', prompt: 'Create a creative agency website with full-screen hero video placeholder, services grid, case studies gallery, client logos, team carousel, and contact form. Sleek dark theme.' },
      { label: 'Restaurant', icon: '🍽️', desc: 'Menu, reservations, reviews', prompt: 'Create a restaurant website with hero image, menu sections (appetizers, mains, desserts, drinks) with prices, reservation form, location map placeholder, reviews, and footer with hours.' },
      { label: 'Real Estate', icon: '🏠', desc: 'Property search, listings, agent', prompt: 'Create a real estate listing page with property search bar, featured listings grid (6 cards with images, price, beds/baths), neighborhood guide, agent profile, and contact form.' },
    ]
  },
  {
    category: 'Personal',
    templates: [
      { label: 'Portfolio', icon: '👤', desc: 'Projects, skills, about, contact', prompt: 'Create a personal portfolio with hero section, about me, project gallery (6 cards with hover effects), skills progress bars, timeline/experience, testimonials, and contact form. Dark minimal theme.' },
      { label: 'Resume/CV', icon: '📄', desc: 'Experience, education, skills', prompt: 'Create a single-page resume/CV with header (name, title, contact), professional summary, work experience timeline, education, skills bar chart, certifications, and languages. Clean printable design.' },
      { label: 'Blog', icon: '📝', desc: 'Featured post, grid, sidebar', prompt: 'Create a blog homepage with header/nav, featured post hero, 6 post cards in a grid, sidebar with categories and newsletter signup, pagination, and footer.' },
      { label: 'Wedding', icon: '💒', desc: 'Timeline, RSVP, gallery', prompt: 'Create an elegant wedding website with hero (couple names + date), our story timeline, event details, RSVP form, photo gallery, gift registry link, and accommodation info. Romantic soft palette.' },
      { label: 'Link Tree', icon: '🔗', desc: 'Profile, links, social icons', prompt: 'Create a link-in-bio page with profile photo circle, name, bio, 8 stylish link buttons with icons, social media icons at bottom. Gradient background with glassmorphism cards.' },
    ]
  },
  {
    category: 'App & Product',
    templates: [
      { label: 'Dashboard', icon: '📊', desc: 'Stats, charts, data table', prompt: 'Create an analytics dashboard with sidebar nav, top stats row (4 KPI cards), large area chart, data table with sorting, donut chart, and activity feed. Dark theme with purple accents.' },
      { label: 'Pricing', icon: '💎', desc: 'Tier cards, comparison, FAQ', prompt: 'Create a pricing page with monthly/annual toggle, 3 tier cards (Basic/Pro/Enterprise, middle highlighted), feature comparison table, FAQ section, and money-back guarantee badge.' },
      { label: 'Login', icon: '🔐', desc: 'Split layout, social login', prompt: 'Create a login page with split layout: left side gradient with branding/testimonial, right side centered form with email/password inputs, social login buttons (Google, GitHub, Apple), forgot password link.' },
      { label: 'Mobile App', icon: '📱', desc: 'Phone mockup, features, badges', prompt: 'Create a mobile app landing page with phone mockup hero, app store badges, feature sections with phone screenshots, user reviews, download stats counter, and footer.' },
      { label: '404 Page', icon: '🔍', desc: 'Animated 404, search, links', prompt: 'Create a creative 404 page with large animated "404" text, witty message, search bar, popular links, and "Go Home" button. Add floating animated geometric shapes.' },
    ]
  },
  {
    category: 'E-commerce',
    templates: [
      { label: 'Product Page', icon: '🛍️', desc: 'Gallery, details, reviews', prompt: 'Create a product detail page with image gallery (main + thumbnails), product title, price, color/size selectors, add-to-cart button, description tabs, reviews section, and related products.' },
      { label: 'Store Front', icon: '🏪', desc: 'Banner, categories, products', prompt: 'Create an e-commerce homepage with hero banner, category cards, featured products grid (8 items with image/name/price/rating), deals section with countdown timer, and newsletter signup.' },
      { label: 'Checkout', icon: '💳', desc: 'Cart, shipping, payment', prompt: 'Create a checkout page with order summary sidebar, shipping form, payment form with card input, express checkout buttons (Apple Pay, Google Pay), promo code input, and order total breakdown.' },
    ]
  },
  {
    category: 'Creative',
    templates: [
      { label: 'Coming Soon', icon: '⏳', desc: 'Countdown, email signup', prompt: 'Create a coming soon page with animated countdown timer, email signup form, progress bar, social links, and a mesmerizing animated gradient background.' },
      { label: 'Event', icon: '🎪', desc: 'Speakers, schedule, tickets', prompt: 'Create an event/conference landing page with hero with date/location, speaker cards (6), schedule/agenda timeline, ticket tiers, venue map placeholder, sponsors grid, and FAQ.' },
      { label: 'Newsletter', icon: '📬', desc: 'Header, articles, CTA', prompt: 'Create an email newsletter template (HTML email compatible) with header logo, hero image, main article, 3 story cards, CTA button, social icons footer. 600px max-width, table-based layout.' },
    ]
  },
];

const SYSTEM_PROMPT = `You are an expert web designer and developer. Generate a complete, beautiful HTML page with inline CSS and JavaScript.

Rules:
- Output ONLY the complete HTML code starting with <!DOCTYPE html>
- Include ALL CSS in a <style> tag inside <head>
- Include ALL JavaScript in a <script> tag before </body>
- Use modern CSS: flexbox, grid, custom properties, smooth transitions
- Use clean typography with system fonts
- Make it fully responsive
- Use professional color schemes with proper contrast
- Add subtle animations and hover effects
- NO markdown fences, NO explanation text, ONLY the HTML document
- If the user asks for modifications, return the COMPLETE updated HTML`;

const getSystemPrompt = (tailwind: boolean) => tailwind
  ? `You are an expert web designer. Generate a complete, beautiful HTML page using Tailwind CSS.

Rules:
- Output ONLY the complete HTML code starting with <!DOCTYPE html>
- Include <script src="https://cdn.tailwindcss.com"></script> in the <head>
- Use Tailwind utility classes for ALL styling (no custom CSS needed)
- Use modern Tailwind: flex, grid, space, responsive prefixes (md:, lg:)
- Use Tailwind's color palette (slate, purple, blue, etc.)
- Add hover:, focus:, transition classes for interactivity
- Make it fully responsive with Tailwind breakpoints
- NO markdown fences, NO explanation text, ONLY the HTML document
- If the user asks for modifications, return the COMPLETE updated HTML`
  : SYSTEM_PROMPT;

/* ── Main Component ── */
export function WebCreator() {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentHtml, setCurrentHtml] = useState('');
  const [streamingCode, setStreamingCode] = useState('');
  const [versions, setVersions] = useState<Version[]>([]);
  const [versionIndex, setVersionIndex] = useState(-1);
  const [device, setDevice] = useState<DeviceSize>('desktop');
  const [viewMode, setViewMode] = useState<ViewMode>('preview');
  const [preGenViewMode, setPreGenViewMode] = useState<ViewMode | null>(null);
  const [codeHtml, setCodeHtml] = useState('');
  const [showTemplates, setShowTemplates] = useState(true);
  const [useTailwind, setUseTailwind] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const [imageData, setImageData] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [showVersionHistory, setShowVersionHistory] = useState(false);
  const [exportedCode, setExportedCode] = useState('');
  const [isExporting, setIsExporting] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [colorPalette, setColorPalette] = useState<typeof COLOR_PALETTES[0] | null>(COLOR_PALETTES[0]);
  const [customColors, setCustomColors] = useState({ primary: '#7c3aed', secondary: '#a78bfa', accent: '#c4b5fd', bg: '#0f0a1a' });
  const [selectedElement, setSelectedElement] = useState<string | null>(null);
  const [collapsedCategories, setCollapsedCategories] = useState<Record<string, boolean>>({});
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const exportMenuRef = useRef<HTMLDivElement>(null);
  const codeEndRef = useRef<HTMLPreElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const chatScrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Syntax highlight current HTML for code view
  useEffect(() => {
    if (currentHtml) {
      const isDark = !document.documentElement.classList.contains('light');
      highlightCode(currentHtml, 'html', isDark ? 'dark' : 'light')
        .then(setCodeHtml).catch(() => {});
    }
  }, [currentHtml]);

  // Auto-scroll chat
  useEffect(() => {
    chatScrollRef.current?.scrollTo({ top: chatScrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [chatMessages]);

  // Abort in-flight request on unmount
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  // Fetch available models
  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        const all = [
          ...(data.chatgpt_models || []),
          ...(data.direct_api_models || []),
          ...(data.cloud_models || []),
          ...(data.local_models || []),
        ];
        if (all.length > 0) setAvailableModels(all);
      })
      .catch(() => {});
  }, []);

  // Close model menu on click outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setShowModelMenu(false);
      }
      if (exportMenuRef.current && !exportMenuRef.current.contains(e.target as Node)) {
        setShowExportMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Auto-scroll code view during streaming
  useEffect(() => {
    if (isGenerating && codeEndRef.current) {
      codeEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [streamingCode, isGenerating]);

  // Listen for element selections from the preview iframe (click-to-edit)
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      if (e.data?.type === 'elementSelected') {
        const { tag, text, classes } = e.data;
        const desc = text
          ? `the "${text.slice(0, 40)}" ${tag}`
          : `the ${tag}${classes ? `.${String(classes).trim().split(/\s+/)[0]}` : ''}`;
        setSelectedElement(desc);
        setInput(`Change ${desc}: `);
        setTimeout(() => {
          if (textareaRef.current) {
            textareaRef.current.focus();
            const len = textareaRef.current.value.length;
            textareaRef.current.setSelectionRange(len, len);
          }
        }, 0);
      }
    };
    window.addEventListener('message', handler);
    return () => window.removeEventListener('message', handler);
  }, []);

  const processImageFile = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target?.result as string;
      setImagePreview(dataUrl);
      setImageData(dataUrl.split(',')[1]);
    };
    reader.readAsDataURL(file);
  }, []);

  const addVersion = useCallback((html: string) => {
    setVersions((prev) => {
      const next = [...prev, { html, timestamp: Date.now() }].slice(-20);
      setVersionIndex(next.length - 1);
      return next;
    });
    setCurrentHtml(html);
  }, []);

  // Listen for live text edits from the preview iframe
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      if (e.data?.type === 'htmlUpdated') {
        let html = e.data.html as string;
        html = html.replace(/\s*contenteditable="[^"]*"/gi, '');
        html = html.replace(/\s*style="\s*outline-offset:[^"]*"/gi, '');
        setCurrentHtml(html);
        addVersion(html);
      }
    };
    window.addEventListener('message', handler);
    return () => window.removeEventListener('message', handler);
  }, [addVersion]);

  const IMAGE_SYSTEM_PROMPT = `You are an expert web designer and developer. The user has provided a screenshot of a design. Recreate it as a complete, pixel-perfect HTML page with inline CSS and JavaScript.

Rules:
- Output ONLY the complete HTML code starting with <!DOCTYPE html>
- Include ALL CSS in a <style> tag inside <head>
- Include ALL JavaScript in a <script> tag before </body>
- Match the layout, colors, typography, spacing, and content from the screenshot as closely as possible
- Use modern CSS: flexbox, grid, custom properties, smooth transitions
- Make it fully responsive
- Add subtle animations and hover effects where appropriate
- NO markdown fences, NO explanation text, ONLY the HTML document
- If the user asks for modifications, return the COMPLETE updated HTML`;

  const handleSend = useCallback(async (message: string) => {
    if (!message.trim() && !imageData || isGenerating) return;
    setShowTemplates(false);
    setSelectedElement(null);

    // Capture image state before clearing
    const capturedImageData = imageData;
    const capturedImagePreview = imagePreview;

    const effectiveMessage = capturedImageData
      ? `Recreate this design as a complete HTML website. ${message.trim() || 'Match the layout, colors, typography, and content as closely as possible.'}`
      : message;

    const userMsg: ChatMessage = {
      role: 'user',
      content: message || 'Screenshot attached',
      imagePreview: capturedImagePreview || undefined,
    };
    setChatMessages((prev) => [...prev, userMsg]);
    setInput('');
    setImageData(null);
    setImagePreview(null);
    setIsGenerating(true);

    // Build color instruction
    const activePalette = colorPalette?.name === 'Custom'
      ? { ...colorPalette, ...customColors }
      : colorPalette;
    const colorInstruction = activePalette && activePalette.primary
      ? `\n\nUse this color scheme: Primary: ${activePalette.primary}, Secondary: ${activePalette.secondary}, Accent: ${activePalette.accent}, Background: ${activePalette.bg}. Apply these colors consistently throughout the design.`
      : '';

    // Build context with current HTML if editing
    const basePrompt = capturedImageData ? IMAGE_SYSTEM_PROMPT : getSystemPrompt(useTailwind);
    const systemCtx = currentHtml
      ? `${basePrompt}${colorInstruction}\n\nCurrent page HTML:\n${currentHtml}`
      : `${basePrompt}${colorInstruction}`;

    // Build history from prior chat messages
    const history = chatMessages.map((m) => ({ role: m.role, content: m.content }));

    const controller = new AbortController();
    abortRef.current = controller;

    // Switch to split/code view so user sees code being written
    setPreGenViewMode(viewMode);
    if (viewMode === 'preview') setViewMode('split');
    setStreamingCode('');

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: effectiveMessage,
          system_prompt: systemCtx,
          history: history,
          ...(selectedModel ? { model: selectedModel } : {}),
          ...(capturedImageData ? { images: [capturedImageData] } : {}),
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      let fullResponse = '';
      const assistantMsg: ChatMessage = { role: 'assistant', content: '' };
      setChatMessages((prev) => [...prev, assistantMsg]);

      if (res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });

          // Parse SSE or raw chunks
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
                  setChatMessages((prev) => {
                    const updated = [...prev];
                    updated[updated.length - 1] = { role: 'assistant', content: fullResponse };
                    return updated;
                  });
                }
              } catch {
                // Raw text chunk
                fullResponse += data;
                setStreamingCode(fullResponse);
              }
            } else if (line.trim() && !line.startsWith(':')) {
              // Non-SSE — raw streaming
              fullResponse += line;
              setStreamingCode(fullResponse);
              setChatMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = { role: 'assistant', content: fullResponse };
                return updated;
              });
            }
          }
        }
      } else {
        const text = await res.text();
        fullResponse = text;
        setChatMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'assistant', content: fullResponse };
          return updated;
        });
      }

      // Extract HTML from response
      let html = fullResponse.trim();
      // Strip markdown fences if present
      const fenceMatch = html.match(/```html?\s*\n([\s\S]*?)```/);
      if (fenceMatch) html = fenceMatch[1].trim();
      // Auto-inject Tailwind CDN if enabled and not already present
      if (useTailwind && !html.includes('tailwindcss')) {
        html = html.replace('</head>', '<script src="https://cdn.tailwindcss.com"></script>\n</head>');
      }
      // Validate it looks like HTML
      if (html.includes('<!DOCTYPE') || html.includes('<html') || html.includes('<body') || html.includes('<div')) {
        addVersion(html);
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setChatMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `Error: ${e.message}. Make sure the backend is running.` },
        ]);
      }
    } finally {
      setIsGenerating(false);
      setStreamingCode('');
      // Restore previous view mode after generation
      if (preGenViewMode !== null) {
        setViewMode(preGenViewMode);
        setPreGenViewMode(null);
      }
      abortRef.current = null;
    }
  }, [chatMessages, currentHtml, isGenerating, addVersion, colorPalette, customColors, useTailwind, viewMode, selectedModel, preGenViewMode]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsGenerating(false);
  }, []);

  const handleUndo = useCallback(() => {
    if (versionIndex > 0) {
      const i = versionIndex - 1;
      setVersionIndex(i);
      setCurrentHtml(versions[i].html);
    }
  }, [versionIndex, versions]);

  const handleRedo = useCallback(() => {
    if (versionIndex < versions.length - 1) {
      const i = versionIndex + 1;
      setVersionIndex(i);
      setCurrentHtml(versions[i].html);
    }
  }, [versionIndex, versions]);

  const handleDownload = useCallback(() => {
    if (!currentHtml) return;
    const blob = new Blob([currentHtml], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aura-website-${Date.now()}.html`;
    a.click();
    URL.revokeObjectURL(url);
  }, [currentHtml]);

  const handleOpenNewTab = useCallback(() => {
    if (!currentHtml) return;
    const win = window.open('', '_blank');
    if (win) { win.document.write(currentHtml); win.document.close(); }
  }, [currentHtml]);

  const handleExport = useCallback(async (format: 'react' | 'nextjs' | 'tailwind') => {
    if (!currentHtml || isExporting) return;
    setShowExportMenu(false);
    setIsExporting(true);

    const prompts = {
      react: 'Convert this HTML page into a single React functional component (TSX). Use useState for any interactive elements. Export as default. Use inline styles or CSS modules pattern. Output ONLY the code, no markdown fences.',
      nextjs: 'Convert this HTML page into a Next.js App Router page component (page.tsx). Use server/client components appropriately. Add metadata export. Output ONLY the code, no markdown fences.',
      tailwind: 'Convert this HTML page to use Tailwind CSS utility classes instead of inline/custom CSS. Keep it as plain HTML but replace all styles with Tailwind classes. Output ONLY the code, no markdown fences.',
    };

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: `${prompts[format]}\n\nHTML to convert:\n${currentHtml}`,
          system_prompt: 'You are an expert React/Next.js developer. Convert HTML to clean, production-ready code. Output ONLY the code.',
          ...(selectedModel ? { model: selectedModel } : {}),
        }),
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      let fullResponse = '';

      if (res.body) {
        const reader = res.body.getReader();
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
                if (text) fullResponse += text;
              } catch {
                fullResponse += data;
              }
            } else if (line.trim() && !line.startsWith(':')) {
              fullResponse += line;
            }
          }
        }
      } else {
        fullResponse = await res.text();
      }

      // Strip markdown fences
      let code = fullResponse.trim();
      const fenceMatch = code.match(/```(?:tsx?|jsx?|html)?\s*\n([\s\S]*?)```/);
      if (fenceMatch) code = fenceMatch[1].trim();

      setExportedCode(code);
      navigator.clipboard?.writeText(code);
    } catch (e: any) {
      setExportedCode(`Error: ${e.message}`);
    } finally {
      setIsExporting(false);
    }
  }, [currentHtml, isExporting, selectedModel]);

  const CLICK_TO_EDIT_SCRIPT = `<script>(function() {
  var highlighted = null;
  document.addEventListener('mouseover', function(e) {
    if (highlighted) highlighted.style.outline = '';
    highlighted = e.target;
    highlighted.style.outline = '2px solid #7c3aed';
    highlighted.style.outlineOffset = '2px';
  });
  document.addEventListener('mouseout', function(e) {
    if (highlighted) { highlighted.style.outline = ''; highlighted = null; }
  });
  document.addEventListener('click', function(e) {
    e.preventDefault();
    e.stopPropagation();
    var el = e.target;
    var text = (el.textContent || '').trim().slice(0, 50);
    var tag = el.tagName.toLowerCase();
    var classes = el.className || '';
    window.parent.postMessage({
      type: 'elementSelected',
      tag: tag,
      text: text,
      classes: classes,
      innerHTML: (el.innerHTML || '').slice(0, 100)
    }, '*');
  }, true);
})();<\/script>`;

  const EDIT_MODE_SCRIPT = `<script>(function(){
  let editing=null;
  function stopEditing(){
    if(!editing)return;
    editing.contentEditable='false';
    editing.style.outline='';
    window.parent.postMessage({type:'htmlUpdated',html:document.documentElement.outerHTML},'*');
    editing=null;
  }
  document.addEventListener('dblclick',function(e){
    var el=e.target;
    if(['P','H1','H2','H3','H4','H5','H6','SPAN','A','LI','TD','TH','BUTTON','LABEL','FIGCAPTION'].includes(el.tagName)){
      if(editing)stopEditing();
      editing=el;
      el.contentEditable='true';
      el.style.outline='2px solid #7c3aed';
      el.style.outlineOffset='2px';
      el.focus();
      el.addEventListener('blur',function onBlur(){
        el.removeEventListener('blur',onBlur);
        stopEditing();
      });
    }
  });
  document.addEventListener('keydown',function(e){
    if(e.key==='Escape'&&editing){editing.contentEditable='false';editing.style.outline='';editing=null;}
    if(e.key==='Enter'&&editing&&!e.shiftKey){e.preventDefault();editing.blur();}
  });
})();<\/script>`;

  const rawSrcdoc = currentHtml
    ? (() => {
        let h = currentHtml;
        if (editMode) {
          // Edit mode: dblclick for in-place text editing, no click-to-select
          h = h.includes('</body>') ? h.replace('</body>', EDIT_MODE_SCRIPT + '</body>') : h + EDIT_MODE_SCRIPT;
        } else {
          // Normal mode: single-click selects element and pre-fills chat input
          h = h.includes('</body>') ? h.replace('</body>', CLICK_TO_EDIT_SCRIPT + '</body>') : h + CLICK_TO_EDIT_SCRIPT;
        }
        return h;
      })()
    : '';
  const srcdoc = rawSrcdoc ? buildSrcdoc('html', rawSrcdoc) : '';
  const showPreview = viewMode === 'preview' || viewMode === 'split';
  const showCode = viewMode === 'code' || viewMode === 'split';

  return (
    <div className="relative flex flex-col md:flex-row h-full overflow-hidden">
      {/* Left: Chat panel — full width on mobile, fixed width on desktop */}
      <div
        className="flex flex-col md:w-[400px] md:min-w-[300px] md:border-r border-b md:border-b-0 border-chat-border flex-shrink-0 max-md:max-h-[40vh] bg-surface-0"
        onPaste={(e) => {
          const items = e.clipboardData?.items;
          if (items) {
            for (const item of Array.from(items)) {
              if (item.type.startsWith('image/')) {
                const file = item.getAsFile();
                if (file) processImageFile(file);
              }
            }
          }
        }}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          const file = Array.from(e.dataTransfer.files).find(f => f.type.startsWith('image/'));
          if (file) processImageFile(file);
        }}
      >
        {/* Chat header */}
        <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
          <h2 className="text-sm font-semibold text-chat-text">Web Creator</h2>
          <p className="text-[10px] text-chat-text-secondary mt-0.5">Describe a website and Aura will build it</p>
        </div>

        {/* Chat messages */}
        <div ref={chatScrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
          {/* Template picker */}
          {showTemplates && chatMessages.length === 0 && (
            <div className="space-y-2">
              <p className="text-xs text-chat-text-secondary mb-3">Start with a template or describe what you want:</p>
              {TEMPLATE_CATEGORIES.map((cat) => {
                const isCollapsed = collapsedCategories[cat.category];
                return (
                  <div key={cat.category} className="mb-1">
                    <button
                      onClick={() => setCollapsedCategories(prev => ({ ...prev, [cat.category]: !prev[cat.category] }))}
                      className="flex items-center justify-between w-full mb-1.5 group"
                    >
                      <span className="text-[10px] font-semibold text-chat-text-secondary uppercase tracking-wider group-hover:text-chat-text transition-colors">
                        {cat.category}
                      </span>
                      <svg
                        className={`w-3 h-3 text-chat-text-secondary transition-transform ${isCollapsed ? '-rotate-90' : ''}`}
                        fill="none" viewBox="0 0 24 24" stroke="currentColor"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </button>
                    {!isCollapsed && (
                      <div className="grid grid-cols-2 gap-1.5">
                        {cat.templates.map((t) => (
                          <button
                            key={t.label}
                            onClick={() => handleSend(t.prompt)}
                            className="flex flex-col items-start gap-1 px-2.5 py-2 rounded-lg border border-chat-border hover:border-purple-500/40 hover:bg-purple-500/5 text-left transition-all group bg-surface-1"
                          >
                            <div className="flex items-center gap-2 w-full">
                              <span className="flex items-center justify-center w-7 h-7 rounded-md bg-purple-500/10 text-sm flex-shrink-0">{t.icon}</span>
                              <span className="text-[11px] font-medium text-chat-text group-hover:text-white transition-colors leading-tight flex-1">{t.label}</span>
                            </div>
                            <span className="text-[10px] text-chat-text-secondary/50 leading-tight">{t.desc}</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}

              {/* Color palette picker */}
              <div className="pt-2">
                <p className="text-xs text-chat-text-secondary mb-2">Color theme (optional):</p>
                <div className="flex flex-wrap gap-1.5">
                  {COLOR_PALETTES.map((p) => (
                    <button
                      key={p.name}
                      title={p.name}
                      onClick={() => setColorPalette(colorPalette?.name === p.name ? null : p)}
                      className="relative flex items-center justify-center transition-all"
                      style={{
                        width: 28, height: 28, borderRadius: '50%',
                        background: p.name === 'Custom' ? 'conic-gradient(#7c3aed, #3b82f6, #10b981, #ef4444, #ec4899, #7c3aed)' : p.primary,
                        boxShadow: colorPalette?.name === p.name ? `0 0 0 2px var(--surface-0), 0 0 0 4px ${p.primary || '#7c3aed'}` : 'none',
                        border: colorPalette?.name === p.name ? 'none' : '1px solid rgba(255,255,255,0.1)',
                      }}
                    >
                      {p.name === 'Custom' && (
                        <span style={{ fontSize: 10, lineHeight: 1 }}>+</span>
                      )}
                    </button>
                  ))}
                </div>
                {colorPalette && <span className="text-[10px] text-chat-text-secondary mt-1">{colorPalette.name}</span>}
                {/* Custom color inputs */}
                {colorPalette?.name === 'Custom' && (
                  <div className="mt-2 grid grid-cols-2 gap-1.5">
                    {(['primary', 'secondary', 'accent', 'bg'] as const).map((key) => (
                      <label key={key} className="flex items-center gap-1.5 cursor-pointer">
                        <input
                          type="color"
                          value={customColors[key]}
                          onChange={(e) => setCustomColors(c => ({ ...c, [key]: e.target.value }))}
                          className="w-6 h-6 rounded cursor-pointer border-0 bg-transparent"
                        />
                        <span className="text-[10px] text-chat-text-secondary capitalize">{key}</span>
                      </label>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {chatMessages.map((msg, i) => (
            <div
              key={i}
              className={`text-sm ${msg.role === 'user' ? 'text-right' : ''}`}
            >
              {msg.role === 'user' ? (
                <div className="inline-block rounded-xl bg-chat-accent text-white max-w-[90%] text-left overflow-hidden">
                  {msg.imagePreview && (
                    <img
                      src={msg.imagePreview}
                      alt="Attached screenshot"
                      className="w-full max-h-32 object-cover object-top"
                    />
                  )}
                  <div className="px-3 py-2">
                    {msg.content.length > 200 ? msg.content.slice(0, 200) + '...' : msg.content}
                  </div>
                </div>
              ) : (
                <div className="text-xs text-chat-text-secondary">
                  {msg.content.includes('<!DOCTYPE') || msg.content.includes('<html')
                    ? <span className="text-green-400">
                        {isGenerating && i === chatMessages.length - 1
                          ? `Writing code... (${Math.round(msg.content.length / 1024)}KB)`
                          : `Generated website (${Math.round(msg.content.length / 1024)}KB)`
                        }
                      </span>
                    : msg.content.length > 300 ? msg.content.slice(0, 300) + '...' : msg.content
                  }
                </div>
              )}
            </div>
          ))}

          {isGenerating && (
            <div className="flex items-center gap-2 text-xs text-purple-400">
              <div className="shimmer-bar h-2 w-20" />
              Generating...
            </div>
          )}
        </div>

        {/* Chat input */}
        <div className="p-3 border-t border-chat-border flex-shrink-0">
          {/* Selected element badge with actions */}
          {selectedElement && (
            <div className="flex items-center gap-1.5 mb-2 px-2 py-1.5 rounded-md bg-purple-500/10 border border-purple-500/20">
              <CursorArrowRaysIcon className="w-3.5 h-3.5 text-purple-400 flex-shrink-0" />
              <span className="text-[11px] text-purple-300 flex-1 truncate">
                {selectedElement}
              </span>
              <button
                onClick={() => { handleSend(`Redesign ${selectedElement} — make it more modern and visually striking`); setSelectedElement(null); }}
                className="text-[10px] px-1.5 py-0.5 rounded bg-purple-600/40 text-purple-200 hover:bg-purple-600/60 transition-colors flex-shrink-0"
              >Regen</button>
              <button
                onClick={() => { handleSend(`Remove ${selectedElement} from the page completely`); setSelectedElement(null); }}
                className="text-[10px] px-1.5 py-0.5 rounded bg-red-600/30 text-red-300 hover:bg-red-600/50 transition-colors flex-shrink-0"
              >Delete</button>
              <button
                onClick={() => { setInput(`Restyle ${selectedElement}: `); textareaRef.current?.focus(); }}
                className="text-[10px] px-1.5 py-0.5 rounded text-chat-text-secondary hover:text-chat-text transition-colors flex-shrink-0"
                style={{ background: 'var(--surface-2)' }}
              >Style</button>
              <button
                onClick={() => { setSelectedElement(null); setInput(''); }}
                className="text-purple-400/60 hover:text-purple-200 transition-colors flex-shrink-0"
              >
                <XMarkIcon className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
          {/* Image preview thumbnail */}
          {imagePreview && (
            <div className="relative mb-2 inline-block">
              <img src={imagePreview} alt="Screenshot to recreate" className="h-16 w-auto rounded-lg border border-chat-border object-cover" />
              <button
                onClick={() => { setImageData(null); setImagePreview(null); }}
                className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-red-500 text-white text-[10px] flex items-center justify-center hover:bg-red-600 transition-colors"
                title="Remove image"
              >
                ×
              </button>
              <span className="block text-[10px] text-purple-400 mt-0.5">Screenshot ready — send to recreate</span>
            </div>
          )}
          {/* Drop zone hint when dragging */}
          {isDragging && (
            <div className="mb-2 rounded-lg border-2 border-dashed border-purple-500 bg-purple-500/10 p-2 text-center">
              <p className="text-xs text-purple-400">Drop image to recreate as website</p>
            </div>
          )}
          <div className="flex gap-2">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend(input);
                }
              }}
              placeholder={imagePreview ? 'Add instructions (optional)...' : currentHtml ? 'Describe changes or click an element...' : 'Describe your website...'}
              className="flex-1 p-2.5 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm resize-none outline-none focus:border-chat-accent placeholder-chat-text-secondary/70"
              rows={3}
              disabled={isGenerating}
            />
            <button
              onClick={isGenerating ? handleStop : () => handleSend(input)}
              disabled={!isGenerating && !input.trim() && !imageData}
              className="self-end p-2.5 rounded-lg bg-chat-accent hover:opacity-90 disabled:opacity-40 text-white transition-opacity"
            >
              {isGenerating ? <StopIcon className="w-4 h-4" /> : <PaperAirplaneIcon className="w-4 h-4" />}
            </button>
          </div>
          {/* Palette indicator + inline picker (when templates hidden) */}
          <div className="flex items-center gap-2 mt-1.5 flex-wrap">
            {/* Palette swatches — always visible below input */}
            <div className="flex items-center gap-1">
              {COLOR_PALETTES.map((p) => (
                <button
                  key={p.name}
                  title={p.name}
                  onClick={() => setColorPalette(colorPalette?.name === p.name ? null : p)}
                  style={{
                    width: 14, height: 14, borderRadius: '50%', flexShrink: 0,
                    background: p.name === 'Custom' ? 'conic-gradient(#7c3aed, #3b82f6, #10b981, #ef4444, #ec4899, #7c3aed)' : p.primary,
                    boxShadow: colorPalette?.name === p.name ? `0 0 0 1.5px var(--surface-0), 0 0 0 3px ${p.primary || '#7c3aed'}` : 'none',
                    border: colorPalette?.name === p.name ? 'none' : '1px solid rgba(255,255,255,0.08)',
                    transition: 'box-shadow 0.15s',
                  }}
                />
              ))}
            </div>
            {colorPalette && (
              <span className="text-[10px] text-chat-text-secondary flex items-center gap-1">
                <span
                  style={{ width: 8, height: 8, borderRadius: '50%', display: 'inline-block', background: colorPalette.name === 'Custom' ? customColors.primary : colorPalette.primary }}
                />
                {colorPalette.name}
                <button onClick={() => setColorPalette(null)} className="ml-0.5 opacity-50 hover:opacity-100">×</button>
              </span>
            )}
          </div>

          {/* Model selector */}
          <div className="flex items-center mt-1" ref={modelMenuRef}>
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowModelMenu(p => !p)}
                className="flex items-center gap-1 text-[10px] text-chat-text-secondary hover:text-chat-text transition-colors px-2 py-1 rounded-md"
                style={{ background: 'var(--border-subtle)' }}
              >
                <span className="max-w-[140px] truncate">{selectedModel ? selectedModel.split('/').pop() : 'Auto'}</span>
                <svg className="w-2.5 h-2.5 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
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
                      style={{ color: !selectedModel ? 'var(--text-primary)' : 'var(--text-secondary)', background: !selectedModel ? 'var(--surface-3)' : 'transparent' }}
                    >
                      Auto (recommended)
                    </button>
                    {availableModels.map((m) => (
                      <button
                        key={m}
                        onClick={() => { setSelectedModel(m); setShowModelMenu(false); }}
                        className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors truncate"
                        style={{ color: selectedModel === m ? 'var(--text-primary)' : 'var(--text-secondary)', background: selectedModel === m ? 'var(--surface-3)' : 'transparent' }}
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

      {/* Right: Preview + toolbar */}
      <div className="flex-1 flex flex-col min-w-0 bg-surface-0">
        {/* Toolbar */}
        <div className="flex items-center gap-1 px-3 py-2 border-b border-chat-border flex-shrink-0 flex-wrap">
          {/* View modes */}
          <div className="flex rounded-md border border-chat-border overflow-hidden">
            {(['preview', 'code', 'split'] as ViewMode[]).map((m) => (
              <button
                key={m}
                onClick={() => setViewMode(m)}
                className={`px-2 py-1 text-[10px] capitalize transition-colors ${viewMode === m ? 'bg-chat-accent text-white' : 'text-chat-text-secondary hover:text-chat-text'}`}
              >
                {m === 'preview' ? <EyeIcon className="w-3.5 h-3.5 inline" /> : m === 'code' ? <CodeBracketIcon className="w-3.5 h-3.5 inline" /> : 'Split'}
              </button>
            ))}
          </div>

          {/* Device */}
          <div className="flex rounded-md border border-chat-border overflow-hidden ml-1">
            {([['desktop', ComputerDesktopIcon], ['tablet', DeviceTabletIcon], ['mobile', DevicePhoneMobileIcon]] as [DeviceSize, any][]).map(([d, Icon]) => (
              <button
                key={d}
                onClick={() => setDevice(d)}
                className={`p-1 transition-colors ${device === d ? 'bg-chat-accent text-white' : 'text-chat-text-secondary hover:text-chat-text'}`}
              >
                <Icon className="w-3.5 h-3.5" />
              </button>
            ))}
          </div>

          {/* Tailwind toggle */}
          <button
            onClick={() => setUseTailwind(t => !t)}
            className={`px-2 py-1 text-[10px] rounded transition-colors ${useTailwind ? 'bg-blue-500/20 text-blue-400' : 'text-chat-text-secondary'}`}
            title={useTailwind ? 'Tailwind CSS enabled' : 'Tailwind CSS disabled'}
          >
            TW
          </button>

          {/* Edit mode toggle */}
          <button
            onClick={() => setEditMode(m => !m)}
            disabled={!currentHtml}
            className={`px-2 py-1 text-[10px] rounded transition-colors ${editMode ? 'bg-purple-500/20 text-purple-400' : 'text-chat-text-secondary hover:text-chat-text'} disabled:opacity-30`}
            title={editMode ? 'Exit edit mode' : 'Double-click text in preview to edit'}
          >
            ✏️ Edit
          </button>

          <div className="flex-1" />

          {/* Version controls */}
          {versions.length > 0 && (
            <div className="relative flex items-center">
              <button onClick={handleUndo} disabled={versionIndex <= 0} className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30">
                <ArrowUturnLeftIcon className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => setShowVersionHistory(!showVersionHistory)}
                className="text-[10px] text-chat-text-secondary hover:text-chat-text px-1 cursor-pointer"
                title="Version history"
              >
                v{versionIndex + 1}/{versions.length}
              </button>
              <button onClick={handleRedo} disabled={versionIndex >= versions.length - 1} className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30">
                <ArrowUturnRightIcon className="w-3.5 h-3.5" />
              </button>
              {/* Version popover */}
              {showVersionHistory && (
                <div className="absolute bottom-full right-0 mb-2 w-52 max-h-56 overflow-y-auto rounded-lg border border-chat-border shadow-xl z-50 py-1" style={{ background: 'var(--surface-1)' }}>
                  <div className="px-3 py-1.5 text-[10px] text-chat-text-secondary font-medium border-b border-chat-border">Versions</div>
                  {versions.map((v, i) => (
                    <button
                      key={i}
                      onClick={() => { setVersionIndex(i); setCurrentHtml(v.html); setShowVersionHistory(false); }}
                      className={`w-full text-left px-3 py-1.5 text-[11px] hover:bg-white/5 flex items-center justify-between ${i === versionIndex ? 'text-chat-accent' : 'text-chat-text-secondary'}`}
                    >
                      <span>v{i + 1}</span>
                      <span className="text-[9px] text-chat-text-secondary/50">
                        {new Date(v.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        {' · '}{(v.html.length / 1024).toFixed(1)}KB
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Actions */}
          <button
            onClick={handleDownload}
            disabled={!currentHtml}
            className="flex items-center gap-1 px-2 py-1 text-[10px] rounded border border-chat-border text-chat-text-secondary hover:text-chat-text hover:border-purple-500/40 disabled:opacity-30 transition-colors"
            title="Download as HTML file"
          >
            <ArrowDownTrayIcon className="w-3.5 h-3.5" />HTML
          </button>
          <button onClick={handleOpenNewTab} disabled={!currentHtml} className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30" title="Open in new tab">
            <ArrowTopRightOnSquareIcon className="w-3.5 h-3.5" />
          </button>
          {/* Export dropdown */}
          <div className="relative" ref={exportMenuRef}>
            <button
              onClick={() => setShowExportMenu(p => !p)}
              disabled={!currentHtml || isExporting}
              className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30"
              title="Export as..."
            >
              {isExporting
                ? <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" /></svg>
                : <CodeBracketIcon className="w-3.5 h-3.5" />
              }
            </button>
            {showExportMenu && (
              <div className="absolute right-0 top-8 w-48 bg-surface-1 border border-chat-border rounded-lg shadow-lg z-50 py-1">
                <button onClick={() => handleExport('react')} className="w-full px-3 py-2 text-xs text-left text-chat-text hover:bg-surface-2">Export as React Component</button>
                <button onClick={() => handleExport('nextjs')} className="w-full px-3 py-2 text-xs text-left text-chat-text hover:bg-surface-2">Export as Next.js Page</button>
                <button onClick={() => handleExport('tailwind')} className="w-full px-3 py-2 text-xs text-left text-chat-text hover:bg-surface-2">Export with Tailwind</button>
                <div className="border-t border-chat-border my-1" />
                <button onClick={() => { setShowExportMenu(false); handleDownload(); }} className="w-full px-3 py-2 text-xs text-left text-chat-text hover:bg-surface-2">Download as HTML</button>
              </div>
            )}
          </div>
        </div>

        {/* Preview area */}
        <div className={`flex-1 overflow-hidden flex ${viewMode === 'split' ? 'flex-row' : 'flex-col'}`}>
          {showPreview && (
            <div className={`${viewMode === 'split' ? 'w-1/2 border-r border-chat-border' : 'flex-1'} overflow-auto flex justify-center`} style={{ background: currentHtml ? 'white' : 'var(--surface-1)' }}>
              {currentHtml ? (
                <div
                  style={{ width: DEVICE_WIDTHS[device], maxWidth: '100%', height: device === 'desktop' ? '100%' : '90%' }}
                  className={`transition-all duration-300 ${DEVICE_FRAMES[device].outer} ${device !== 'desktop' ? 'my-4 mx-auto' : ''} overflow-hidden`}
                >
                  {DEVICE_FRAMES[device].notch && (
                    <div className="absolute top-1 left-1/2 -translate-x-1/2 w-24 h-5 bg-gray-800 rounded-b-xl z-10" />
                  )}
                  <iframe
                    srcDoc={srcdoc}
                    sandbox="allow-scripts allow-same-origin"
                    className={`w-full h-full border-none ${DEVICE_FRAMES[device].inner}`}
                    title="Website preview"
                  />
                </div>
              ) : (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <div className="empty-state mb-4">
                      <svg className="w-16 h-16 mx-auto text-chat-text-secondary/30 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-5.36-5.364l.707-.707M5.05 12H4m3.979-7.364l.707.707" />
                      </svg>
                    </div>
                    <p className="text-sm text-chat-text font-medium mb-1">Your website will appear here</p>
                    <p className="text-[10px] text-chat-text-secondary mb-4">Pick a template or describe what you want to build</p>
                    <button
                      onClick={() => textareaRef.current?.focus()}
                      className="px-3 py-1.5 text-xs rounded-lg bg-purple-600 hover:bg-purple-700 text-white transition-colors"
                    >
                      Get Started
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {showCode && (
            <div className={`${viewMode === 'split' ? 'w-1/2' : 'flex-1'} overflow-auto bg-surface-1`}>
              {isGenerating && streamingCode ? (
                /* Live streaming code view */
                <pre className="p-4 text-xs font-mono text-green-400 whitespace-pre-wrap leading-relaxed">
                  {streamingCode}
                  <span ref={codeEndRef} className="inline-block w-1.5 h-3.5 bg-green-400 animate-pulse ml-0.5 align-middle" />
                </pre>
              ) : currentHtml ? (
                codeHtml ? (
                  <div
                    dangerouslySetInnerHTML={{ __html: codeHtml }}
                    className="shiki-block p-4 text-sm [&_pre]:!bg-transparent [&_pre]:!m-0 [&_pre]:!p-0 [&_code]:!bg-transparent"
                  />
                ) : (
                  <pre className="p-4 text-sm font-mono text-chat-text whitespace-pre-wrap">{currentHtml}</pre>
                )
              ) : (
                <div className="flex items-center justify-center h-full text-chat-text-secondary text-sm">No code yet</div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Export code modal */}
      {exportedCode && (
        <div className="absolute inset-0 bg-black/80 z-50 flex items-center justify-center p-4" onClick={() => setExportedCode('')}>
          <div className="bg-surface-1 rounded-xl border border-chat-border max-w-2xl w-full max-h-[80vh] flex flex-col" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between px-4 py-3 border-b border-chat-border flex-shrink-0">
              <span className="text-sm font-medium text-chat-text">Exported Code</span>
              <div className="flex gap-3">
                <button onClick={() => navigator.clipboard?.writeText(exportedCode)} className="text-xs text-chat-text-secondary hover:text-chat-text transition-colors">Copy</button>
                <button onClick={() => setExportedCode('')} className="text-xs text-chat-text-secondary hover:text-chat-text transition-colors">Close</button>
              </div>
            </div>
            <pre className="p-4 text-xs font-mono text-green-400 overflow-auto flex-1 whitespace-pre-wrap">{exportedCode}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
