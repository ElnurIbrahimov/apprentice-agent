import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Copy, Download, Maximize2, Minimize2, Code2, Eye,
  Monitor, Tablet, Smartphone, Globe, Layout, Sparkles,
  Send, Trash2, RotateCcw, Upload, User, Bot,
  Undo2, Redo2, MousePointer2, Pencil, Palette, ExternalLink, Square,
  Save, FolderOpen, Search, GitFork, X, FolderTree,
  Hammer, Play, Plus, ChevronUp, ChevronDown, Zap, Server,
} from 'lucide-react';
import { useStore } from '../store';
import CodeEditor, { type CodeEditorDiagnostic } from '../components/CodeEditor';
import FileTree from '../components/FileTree';
import ModelPill from '../components/ModelPill';
import OverlayModal from '../components/OverlayModal';
import { HTTP, getAuthHeaders } from '../api';
import { exportHTML, exportJSON } from '../utils/exportUtils';
import { streamRawGenerate } from '../utils/streamChat';
import { StreamingPreviewController } from '../utils/StreamingPreviewController';
import { useVersionHistory } from '../utils/useVersionHistory';
import OfflineBanner from '../components/OfflineBanner';
import { captureIframeThumbnailDataUrl } from '../utils/captureIframeThumbnail';
import { getDefaultCreationSettings, loadCreationSettings, type CreationSettings } from '../utils/creationSettings';
import { summarizeDiff } from '../utils/diffHeuristics';
import { parseFileOperations } from '../utils/parseFileOperations';
import { formatBuildPlan, parseBuildPlan, type BuildPlanItem } from '../utils/buildPlan';
import { useAgentBuild, type AgentBuildPlanItem } from '../hooks/useAgentBuild';
import {
  createDefaultWebProject,
  detectCodeLanguageFromPath,
  getBaseName,
  getDefaultFileContent,
  getParentDirectory,
  type VirtualFS as VirtualFSClass,
  VirtualFS,
} from '../utils/virtualFS';
import { webCreatorGalleryStore, type SavedWebPage } from '../utils/webCreatorGallery';
import { webCreatorProjectGalleryStore, type SavedWebProject } from '../utils/webCreatorProjectGallery';
import { useWebContainer } from '../hooks/useWebContainer';
import TerminalOutput from '../components/TerminalOutput';
import {
  shareProject, downloadAsZip, deployToGitHubPages,
  saveDeployment, loadGitHubToken, saveGitHubToken,
  type Deployment,
} from '../utils/deployUtils';
import DeployDashboard from '../components/DeployDashboard';
import { useVisualFeedback, type VisualFeedback } from '../hooks/useVisualFeedback';
import { componentLibrary, type UIComponent } from '../utils/componentLibrary';
import ComponentGallery from '../components/ComponentGallery';
import {
  loadDesignTokens, saveDesignTokens, tokensToCssVariables, tokensToSystemPrompt,
  TOKEN_PRESETS, type DesignTokens,
} from '../utils/designTokens';

const DiffEditor = React.lazy(() => import('../components/DiffEditor'));

/* ─── Chrome storage helpers ─── */
declare const browser: any; // Firefox compat
const ext = typeof chrome !== 'undefined' ? chrome : typeof browser !== 'undefined' ? browser : null;
const WC_STORAGE_KEY = 'aura_webcreator_state';

type CreatorMode = 'single' | 'project';
type ProjectWorkflowMode = 'chat' | 'build';
type BuildStage = 'idle' | 'planning' | 'planned' | 'building' | 'completed' | 'error';

interface WebCreatorPersistedState {
  activeFile?: string;
  html?: string;
  history?: Array<{ role: string; content: string }>;
  messages?: ChatMessage[];
  mode?: CreatorMode;
  project?: string;
}

function wcStorageGet(): Promise<WebCreatorPersistedState | null> {
  return new Promise((resolve) => {
    if (!ext?.storage?.local) { resolve(null); return; }
    ext.storage.local.get([WC_STORAGE_KEY], (d: any) => {
      try { resolve(d?.[WC_STORAGE_KEY] ? JSON.parse(d[WC_STORAGE_KEY]) : null); }
      catch { resolve(null); }
    });
  });
}

function wcStorageSave(data: WebCreatorPersistedState) {
  if (!ext?.storage?.local) return;
  const hasProject = !!data.project;
  const hasSinglePage = !!data.html;
  const hasMessages = (data.messages?.length || 0) > 0;
  if (!hasProject && !hasSinglePage && !hasMessages) {
    ext.storage.local.remove([WC_STORAGE_KEY]);
    return;
  }
  ext.storage.local.set({
    [WC_STORAGE_KEY]: JSON.stringify({
      ...data,
      messages: data.messages?.slice(-30) || [],
      history: data.history?.slice(-20) || [],
    }),
  });
}

/* ─── Types ─── */
type ViewMode = 'preview' | 'code';
type DeviceMode = 'desktop' | 'tablet' | 'mobile';

interface ChatMessage {
  id: string;
  role: 'user' | 'ai';
  text: string;
  timestamp: number;
}

interface Template {
  label: string;
  icon: React.ReactNode;
  prompt: string;
  color: string;
}

interface PendingPageDiff {
  modified: string;
  original: string;
  prompt: string;
}

interface ProjectTemplate {
  color: string;
  files: Record<string, string>;
  icon: React.ReactNode;
  label: string;
  prompt: string;
}

interface ProjectDialogState {
  initialValue: string;
  parentPath: string;
  path?: string;
  title: string;
  type: 'create-file' | 'create-folder' | 'rename-file';
}

interface BuildPlanDraft extends BuildPlanItem {
  id: string;
}

function buildRuntimeDiagnostics(message: string, line?: number): CodeEditorDiagnostic[] {
  if (!line || line < 1) return [];
  return [{ line, message, severity: 'error' }];
}

/* ─── Constants ─── */
const TEMPLATES: Template[] = [
  // ─── Business ───
  { label: 'SaaS Landing', icon: <Globe size={16} />, prompt: 'Create a modern SaaS landing page with hero section, features grid (3 cards with icons), testimonials carousel, pricing table (3 tiers), FAQ accordion, and footer. Professional purple/blue gradient theme.', color: '#7c3aed' },
  { label: 'Startup', icon: <Sparkles size={16} />, prompt: 'Create a startup landing page with bold hero with animated gradient text, problem/solution section, how-it-works steps, team section with photo placeholders, investor logos bar, and CTA section.', color: '#6366f1' },
  { label: 'Agency', icon: <Layout size={16} />, prompt: 'Create a creative agency website with full-screen hero video placeholder, services grid, case studies gallery, client logos, team carousel, and contact form. Sleek dark theme.', color: '#8b5cf6' },
  { label: 'Restaurant', icon: <Globe size={16} />, prompt: 'Create a restaurant website with hero image, menu sections (appetizers, mains, desserts, drinks) with prices, reservation form, location map placeholder, reviews, and footer with hours.', color: '#f59e0b' },
  { label: 'Real Estate', icon: <Monitor size={16} />, prompt: 'Create a real estate listing page with property search bar, featured listings grid (6 cards with images, price, beds/baths), neighborhood guide, agent profile, and contact form.', color: '#10b981' },
  { label: 'Law Firm', icon: <Layout size={16} />, prompt: 'Create a law firm website with hero, practice area cards (6), attorney profiles with headshots, case results counter, client testimonials, consultation booking form, and footer with office locations.', color: '#1e40af' },
  { label: 'Consulting', icon: <Monitor size={16} />, prompt: 'Create a consulting firm website with hero, services overview (6 cards), methodology/process steps, case studies with metrics, team carousel, trust badges, and contact CTA.', color: '#0f766e' },
  { label: 'Gym & Fitness', icon: <User size={16} />, prompt: 'Create a gym website with hero, class schedule grid, trainer profiles with specialties, membership tier cards (3), facilities gallery, transformation stories, and free trial CTA.', color: '#dc2626' },
  { label: 'Hotel', icon: <Globe size={16} />, prompt: 'Create a luxury hotel website with full-screen hero, room category cards with prices, amenities grid with icons, photo gallery, guest reviews, location section, and booking date picker CTA.', color: '#b45309' },
  // ─── Personal ───
  { label: 'Portfolio', icon: <User size={16} />, prompt: 'Create a personal portfolio with hero section, about me, project gallery (6 cards with hover effects), skills progress bars, timeline/experience, testimonials, and contact form. Dark minimal theme.', color: '#06b6d4' },
  { label: 'Resume/CV', icon: <Layout size={16} />, prompt: 'Create a single-page resume/CV with header (name, title, contact), professional summary, work experience timeline, education, skills bar chart, certifications, and languages. Clean printable design.', color: '#64748b' },
  { label: 'Blog', icon: <Layout size={16} />, prompt: 'Create a blog homepage with header/nav, featured post hero, 6 post cards in a grid, sidebar with categories and newsletter signup, pagination, and footer.', color: '#10b981' },
  { label: 'Photography', icon: <Monitor size={16} />, prompt: 'Create a photography portfolio with full-width hero image, masonry photo gallery with lightbox modal on click, category filter tabs, about section, and booking form.', color: '#a855f7' },
  { label: 'Podcast', icon: <Globe size={16} />, prompt: 'Create a podcast website with hero (podcast art + subscribe buttons for Apple/Spotify/Google), latest episodes list with play buttons, guest profiles, about the hosts, and sponsor section.', color: '#e11d48' },
  { label: 'Link Tree', icon: <Globe size={16} />, prompt: 'Create a link-in-bio page with profile photo circle, name, bio, 8 stylish link buttons with icons, social media icons at bottom. Gradient background with glassmorphism cards.', color: '#8b5cf6' },
  { label: 'Wedding', icon: <Sparkles size={16} />, prompt: 'Create an elegant wedding website with hero (couple names + date), our story timeline, event details, RSVP form, photo gallery, gift registry link, and accommodation info. Romantic soft palette.', color: '#f472b6' },
  // ─── App & Product ───
  { label: 'Dashboard', icon: <Monitor size={16} />, prompt: 'Create an analytics dashboard with sidebar nav, top stats row (4 KPI cards), large area chart, data table with sorting, donut chart, and activity feed. Dark theme with purple accents.', color: '#f59e0b' },
  { label: 'Pricing', icon: <Sparkles size={16} />, prompt: 'Create a pricing page with monthly/annual toggle, 3 tier cards (Basic/Pro/Enterprise, middle highlighted), feature comparison table, FAQ section, and money-back guarantee badge.', color: '#8b5cf6' },
  { label: 'Login', icon: <User size={16} />, prompt: 'Create a login page with split layout: left side gradient with branding/testimonial, right side centered form with email/password inputs, social login buttons (Google, GitHub, Apple), forgot password link.', color: '#ec4899' },
  { label: 'Mobile App', icon: <Smartphone size={16} />, prompt: 'Create a mobile app landing page with phone mockup hero, app store badges, feature sections with phone screenshots, user reviews, download stats counter, and footer.', color: '#06b6d4' },
  { label: '404 Page', icon: <RotateCcw size={16} />, prompt: 'Create a creative 404 page with large animated "404" text, witty message, search bar, popular links, and "Go Home" button. Add floating animated geometric shapes.', color: '#ef4444' },
  { label: 'Signup Flow', icon: <User size={16} />, prompt: 'Create a multi-step signup form with progress bar (4 steps), animated transitions between steps, inline validation, password strength meter, and success confetti animation.', color: '#10b981' },
  { label: 'Settings Page', icon: <Monitor size={16} />, prompt: 'Create an app settings page with sidebar categories (Profile, Notifications, Security, Billing), toggle switches, input fields, avatar upload, connected accounts, and danger zone. Dark theme.', color: '#64748b' },
  { label: 'Documentation', icon: <Layout size={16} />, prompt: 'Create a documentation page with sidebar table of contents, search bar, breadcrumbs, markdown-style content with code blocks, copy buttons, info/warning callout boxes, and prev/next navigation.', color: '#3b82f6' },
  { label: 'API Reference', icon: <Code2 size={16} />, prompt: 'Create an API reference page with sidebar endpoint list, HTTP method badges (GET green, POST blue, PUT orange, DELETE red), parameter tables, request/response code examples.', color: '#0ea5e9' },
  { label: 'Status Page', icon: <Monitor size={16} />, prompt: 'Create a system status page with overall status banner, service list with uptime bars (90 days), incident history timeline, subscribe to updates form, and uptime percentage badges.', color: '#22c55e' },
  // ─── E-commerce ───
  { label: 'Product Page', icon: <Globe size={16} />, prompt: 'Create a product detail page with image gallery (main + thumbnails), product title, price, color/size selectors, add-to-cart button, description tabs, reviews section, and related products.', color: '#f59e0b' },
  { label: 'Store Front', icon: <Globe size={16} />, prompt: 'Create an e-commerce homepage with hero banner, category cards, featured products grid (8 items with image/name/price/rating), deals section with countdown timer, and newsletter signup.', color: '#10b981' },
  { label: 'Checkout', icon: <Monitor size={16} />, prompt: 'Create a checkout page with order summary sidebar, shipping form, payment form with card input, express checkout buttons (Apple Pay, Google Pay), promo code input, and order total breakdown.', color: '#6366f1' },
  { label: 'Food Delivery', icon: <Globe size={16} />, prompt: 'Create a food delivery app UI with restaurant header, menu categories (horizontal scroll), food items with photos/prices/add buttons, floating cart summary, delivery address input, and order tracking progress bar.', color: '#ef4444' },
  // ─── Creative & Media ───
  { label: 'Coming Soon', icon: <Sparkles size={16} />, prompt: 'Create a coming soon page with animated countdown timer, email signup form, progress bar, social links, and a mesmerizing animated gradient background.', color: '#a855f7' },
  { label: 'Event', icon: <Globe size={16} />, prompt: 'Create an event/conference landing page with hero with date/location, speaker cards (6), schedule/agenda timeline, ticket tiers, venue map placeholder, sponsors grid, and FAQ.', color: '#06b6d4' },
  { label: 'Newsletter', icon: <Layout size={16} />, prompt: 'Create an email newsletter template (HTML email compatible) with header logo, hero image, main article, 3 story cards, CTA button, social icons footer. 600px max-width, table-based layout.', color: '#f97316' },
  { label: 'Magazine', icon: <Layout size={16} />, prompt: 'Create a digital magazine homepage with large editorial hero article, 2-column article grid, breaking news ticker, category tabs, trending sidebar, subscribe CTA, and author bylines with avatars.', color: '#1e40af' },
  // ─── Technology ───
  { label: 'Dev Portfolio', icon: <Code2 size={16} />, prompt: 'Create a developer portfolio with terminal-style hero (typing animation), GitHub stats cards, project showcase (6 repos with stars/forks/language), tech stack icons grid, blog posts, and contact form. Dark hacker theme.', color: '#22c55e' },
  { label: 'AI Product', icon: <Sparkles size={16} />, prompt: 'Create an AI product landing page with hero (animated neural network visualization in CSS), live demo section, feature comparison, integration logos, API code snippet, pricing, and enterprise CTA.', color: '#7c3aed' },
  { label: 'Open Source', icon: <Code2 size={16} />, prompt: 'Create an open source project page with hero (project name + description + GitHub badges), quick start code block, feature list, contributor avatars grid, and "Star on GitHub" CTA. Dark theme.', color: '#64748b' },
  { label: 'CLI Tool', icon: <Code2 size={16} />, prompt: 'Create a CLI tool documentation site with hero (terminal screenshot), one-line install command with copy button, command reference table, usage examples with syntax highlighting, and GitHub link. Monospace dark theme.', color: '#10b981' },
  { label: 'Crypto/Web3', icon: <Monitor size={16} />, prompt: 'Create a DeFi/crypto dashboard with wallet connect button, token balances list, price charts placeholder, swap interface (from/to tokens), transaction history, and portfolio allocation donut chart. Dark cyber theme.', color: '#f59e0b' },
  // ─── Interactive & Fun ───
  { label: 'Quiz/Survey', icon: <Sparkles size={16} />, prompt: 'Create an interactive quiz page with progress bar, question card with 4 answer options (highlight on select), next/back buttons, animated transitions, timer, and results page with score and share buttons.', color: '#ec4899' },
  { label: 'Calculator', icon: <Monitor size={16} />, prompt: 'Create an interactive calculator app (mortgage or BMI calculator) with labeled inputs, sliders for ranges, real-time calculation display, results card with breakdown, and share/print buttons.', color: '#0ea5e9' },
  { label: 'Recipe', icon: <Globe size={16} />, prompt: 'Create a recipe page with hero photo, recipe title/rating/time, ingredient checklist with servings adjuster, step-by-step instructions with photos, nutrition facts table, print recipe button, and related recipes carousel.', color: '#f97316' },
  { label: 'Weather App', icon: <Globe size={16} />, prompt: 'Create a weather app UI with current weather card (temp, icon, condition, location), hourly forecast horizontal scroll, 7-day forecast list, weather details (humidity, wind, UV, pressure), and search location bar.', color: '#3b82f6' },
  { label: 'Music Player', icon: <Sparkles size={16} />, prompt: 'Create a music player UI with album art (large), song title/artist, progress bar with timestamps, playback controls, volume slider, queue/playlist sidebar, and lyrics panel.', color: '#8b5cf6' },
  { label: 'Chat UI', icon: <User size={16} />, prompt: 'Create a messaging app UI with contacts sidebar (avatars, last message, unread badge), chat area with message bubbles, typing indicator, message input with emoji picker, and attachment button.', color: '#06b6d4' },
  // ─── Landing Pages ───
  { label: 'Waitlist', icon: <Sparkles size={16} />, prompt: 'Create a waitlist landing page with bold headline, product teaser (3 feature previews), email signup with referral counter, social proof ticker, and animated background particles.', color: '#a855f7' },
  { label: 'Product Hunt', icon: <Sparkles size={16} />, prompt: 'Create a Product Hunt style launch page with product hero, demo video embed, feature list with icons, founder story, upvote counter, press mentions, and early adopter pricing.', color: '#f97316' },
  { label: 'Comparison', icon: <Layout size={16} />, prompt: 'Create a comparison landing page ("Why switch from X to us") with hero, side-by-side feature comparison table with checkmarks/crosses, pricing comparison, migration guide steps, and "Switch Now" CTA.', color: '#10b981' },
  { label: 'Black Friday', icon: <Sparkles size={16} />, prompt: 'Create a Black Friday deals page with huge countdown timer, deals grid (original/sale price, % off badges), category filter, "Almost Gone" urgency indicators, early access email signup.', color: '#dc2626' },
  // ─── Education ───
  { label: 'Online Course', icon: <Layout size={16} />, prompt: 'Create an online course page with hero (title + instructor), course stats (duration, lessons, level), curriculum accordion, instructor bio, student reviews, certificate preview, and enroll CTA.', color: '#8b5cf6' },
  { label: 'Non-Profit', icon: <Globe size={16} />, prompt: 'Create a non-profit website with hero (mission statement), impact stats, programs section, stories of impact carousel, donation form with preset amounts, volunteer signup, and partner logos.', color: '#059669' },
  // ─── Health ───
  { label: 'Medical Practice', icon: <User size={16} />, prompt: 'Create a medical practice website with hero, services list, doctor profiles with credentials, patient portal login, insurance accepted logos, appointment booking form, and emergency contact info.', color: '#0ea5e9' },
  { label: 'Fitness App', icon: <User size={16} />, prompt: 'Create a fitness app landing page with hero (before/after transformation), workout plan preview, progress tracking charts, meal planning section, app store badges, and free trial CTA.', color: '#ef4444' },
  // ─── Social ───
  { label: 'Social Profile', icon: <User size={16} />, prompt: 'Create a social media profile page with cover photo, profile pic, bio, stats (posts/followers/following), tab bar (Posts/Media/Likes), post feed with like/comment/share buttons, and suggested users sidebar.', color: '#3b82f6' },
  { label: 'Forum', icon: <Layout size={16} />, prompt: 'Create a forum/community page with category cards, latest threads list with avatars/replies/views, pinned announcements, search bar, user leaderboard sidebar, and new thread button.', color: '#f59e0b' },
  // ─── Apps (functional, with JavaScript) ───
  { label: 'Todo App', icon: <Code2 size={16} />, prompt: 'Create a fully functional todo app with: add/edit/delete tasks, mark complete, filter (all/active/completed), drag-to-reorder, local storage persistence, task counter, clear completed button, and dark/light theme toggle. All in one HTML file with working JavaScript.', color: '#22c55e' },
  { label: 'Kanban Board', icon: <Layout size={16} />, prompt: 'Create a Kanban board with 3 columns (To Do, In Progress, Done), draggable cards between columns, add/edit/delete cards, card labels/colors, search filter, and local storage. Include smooth drag animations.', color: '#3b82f6' },
  { label: 'Notes App', icon: <Layout size={16} />, prompt: 'Create a notes app with sidebar folder list, note list, rich text editor (bold, italic, headings, lists, code), search, create/delete/rename notes, auto-save to localStorage, and word count.', color: '#f59e0b' },
  { label: 'Calendar App', icon: <Monitor size={16} />, prompt: 'Create a calendar app with month/week/day views, add/edit/delete events (title, time, color), event dots on dates, event list sidebar, navigate months, today button, and localStorage persistence.', color: '#8b5cf6' },
  { label: 'Expense Tracker', icon: <Monitor size={16} />, prompt: 'Create an expense tracker with add income/expense, transaction list with filters, pie chart by category (CSS-drawn), monthly bar chart, balance card, budget limits, and CSV export button. localStorage persistence.', color: '#10b981' },
  { label: 'Pomodoro Timer', icon: <Sparkles size={16} />, prompt: 'Create a Pomodoro timer with 25min work/5min break cycle, large circular timer, start/pause/reset buttons, session counter, task list, notification sound via Web Audio API, daily stats, and customizable durations.', color: '#ef4444' },
  { label: 'Markdown Editor', icon: <Code2 size={16} />, prompt: 'Create a Markdown editor with split view (editor + live preview), formatting toolbar, word/char count, export as HTML/MD, and fullscreen mode. Support GitHub-flavored markdown. Single HTML file.', color: '#64748b' },
  // ─── Games (playable, with Canvas) ───
  { label: 'Snake Game', icon: <Sparkles size={16} />, prompt: 'Create a Snake game using HTML5 Canvas. Arrow key + touch swipe controls, growing snake, random food, score counter, speed increases, game over with restart, high score in localStorage, grid-based movement.', color: '#22c55e' },
  { label: 'Tetris', icon: <Monitor size={16} />, prompt: 'Create a Tetris game using Canvas. 7 tetromino shapes with colors, rotation, move left/right, soft/hard drop, line clearing animation, score, next piece preview, level progression, ghost piece. Touch controls for mobile.', color: '#3b82f6' },
  { label: '2048', icon: <Sparkles size={16} />, prompt: 'Create a 2048 game. 4x4 grid, slide tiles with arrow keys/swipe, merge same numbers, smooth slide animations, new tile spawning, score tracking, game over detection, undo button, and win animation at 2048.', color: '#f59e0b' },
  { label: 'Wordle', icon: <Code2 size={16} />, prompt: 'Create a Wordle clone. 5-letter word guessing, 6 attempts, color feedback (green/yellow/gray), on-screen keyboard with color updates, word validation, share results as emoji grid, streak tracking, built-in word list.', color: '#22c55e' },
  { label: 'Flappy Bird', icon: <Sparkles size={16} />, prompt: 'Create a Flappy Bird clone using Canvas. Click/tap/space to flap, scrolling pipes with gap, gravity physics, score counter, high score, ground scrolling, death animation, restart button. Touch controls.', color: '#f59e0b' },
  { label: 'Memory Match', icon: <Sparkles size={16} />, prompt: 'Create a Memory Match card game. 4x4 grid face-down cards (8 pairs), click to flip with 3D rotation, match detection, move counter, timer, star rating, win screen, difficulty selector (4x4, 6x6).', color: '#ec4899' },
  { label: 'Minesweeper', icon: <Monitor size={16} />, prompt: 'Create a Minesweeper game. Customizable grid (9x9/16x16), left-click reveal, right-click flag, cascading reveal for empty cells, mine counter, timer, first click safe, win/lose detection.', color: '#64748b' },
  { label: 'Breakout', icon: <Sparkles size={16} />, prompt: 'Create a Breakout/Arkanoid game using Canvas. Movable paddle, bouncing ball, colored brick rows, lives counter, score, level progression, power-ups (multi-ball, wide paddle), game over/win screens.', color: '#8b5cf6' },
  { label: 'Platformer', icon: <Sparkles size={16} />, prompt: 'Create a side-scrolling platformer using Canvas. Player with run/jump, platforms, coins to collect, enemies that patrol, score, lives, scrolling camera, gravity physics, level completion.', color: '#06b6d4' },
  // ─── Dashboards (with Chart.js) ───
  { label: 'Marketing Dash', icon: <Monitor size={16} />, prompt: 'Create a marketing dashboard with KPI cards (visitors, leads, conversion rate, revenue), traffic sources pie chart, campaign bar chart, conversion funnel, top pages table, date range selector. Use Chart.js CDN. Dark theme.', color: '#7c3aed' },
  { label: 'Sales Dashboard', icon: <Monitor size={16} />, prompt: 'Create a sales dashboard with revenue KPI with trend arrow, monthly revenue line chart, sales pipeline bars, top salespeople leaderboard, deals table, win rate gauge. Use Chart.js CDN. Blue theme.', color: '#3b82f6' },
  { label: 'DevOps Monitor', icon: <Monitor size={16} />, prompt: 'Create a DevOps monitoring dashboard with server status cards (CPU, RAM, Disk gauges), deployment timeline, active alerts with severity colors, response time chart, error rate graph, uptime badges. Chart.js CDN. Dark terminal theme.', color: '#22c55e' },
  { label: 'Crypto Tracker', icon: <Monitor size={16} />, prompt: 'Create a crypto portfolio dashboard with top 10 coins table (price, 24h change), portfolio balance card, holdings pie chart, price chart placeholder, fear & greed gauge, watchlist. Chart.js CDN. Dark theme.', color: '#f59e0b' },
  { label: 'Fitness Stats', icon: <User size={16} />, prompt: 'Create a fitness dashboard with weekly activity rings, workout log table, weight trend chart, personal records cards, muscle group radar chart, streak counter, weekly goal bars. Chart.js CDN.', color: '#ef4444' },
];

const SYSTEM_PROMPT = `You are an expert web designer and developer. Generate a complete, beautiful HTML page with inline CSS and JavaScript.

Rules:
- Output ONLY the complete HTML code starting with <!DOCTYPE html>
- Include ALL CSS in a <style> tag inside <head>
- Include ALL JavaScript in a <script> tag before </body>
- Use modern CSS: flexbox, grid, custom properties, smooth transitions
- Use clean typography with system fonts or Google Fonts via CDN
- Make it fully responsive
- Use professional color schemes with proper contrast
- Add subtle animations and hover effects
- NO markdown fences, NO explanation text, ONLY the HTML document
- If the user asks for modifications to a previous page, return the COMPLETE updated HTML (not a diff)`;

const PROJECT_SYSTEM_PROMPT = `You are building a multi-file web project. Respond with file operations only.

Format every file like this:
===FILE: path/to/file.ext===
file contents
===END FILE===

To delete a file:
===DELETE: path/to/file.ext===

Rules:
- You may create, update, or delete multiple files in one response
- Preserve existing files unless the user asked to remove or replace them
- Keep CSS in .css files and JavaScript in .js files when possible
- Prefer editing the smallest set of files needed
- Do not wrap your answer in markdown fences
- Do not add explanation text before or after the file operations`;

const BUILD_PLAN_SYSTEM_PROMPT = `You are planning a multi-file web project build.

Return JSON only. No markdown fences. No explanation text.

Respond with an array like:
[
  { "path": "index.html", "purpose": "Main page shell", "priority": 1 },
  { "path": "styles/main.css", "purpose": "Core styling", "priority": 2 }
]

Rules:
- Use forward slashes in every path
- Order files by dependency: HTML before CSS before JS unless the framework requires otherwise
- Keep the plan concise and practical
- Static projects: max 10 files
- React projects: max 15 files
- Include only files that are genuinely needed`;

const DEVICE_WIDTHS: Record<DeviceMode, string> = {
  desktop: '100%',
  tablet: '768px',
  mobile: '375px',
};
const MAX_AUTO_FIX_ATTEMPTS = 3;

const PROJECT_TEMPLATES: ProjectTemplate[] = [
  {
    label: 'Landing Page',
    icon: <Globe size={16} />,
    prompt: 'Turn this starter into a polished landing page with a hero, features grid, testimonials, pricing, and a footer. Keep the files split cleanly between HTML, CSS, and JavaScript.',
    color: '#7c3aed',
    files: {
      'index.html': `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Landing Page</title>
    <link rel="stylesheet" href="styles/main.css" />
  </head>
  <body>
    <main id="app"></main>
    <script src="scripts/main.js"></script>
  </body>
</html>`,
      'styles/main.css': 'body {\n  margin: 0;\n}\n',
      'scripts/main.js': `document.getElementById('app').innerHTML = '<h1>Landing page starter</h1>';`,
    },
  },
  {
    label: 'Portfolio',
    icon: <User size={16} />,
    prompt: 'Build a portfolio site with a cinematic hero, project cards, an about section, and a contact area. Use multiple files and keep the structure easy to iterate on.',
    color: '#06b6d4',
    files: {
      'index.html': `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Portfolio</title>
    <link rel="stylesheet" href="styles/main.css" />
  </head>
  <body>
    <main id="app"></main>
    <script src="scripts/gallery.js"></script>
  </body>
</html>`,
      'styles/main.css': 'body {\n  margin: 0;\n}\n',
      'scripts/gallery.js': `document.getElementById('app').innerHTML = '<h1>Portfolio starter</h1>';`,
      'pages/about.html': '<section><h2>About</h2><p>Tell your story here.</p></section>\n',
    },
  },
  {
    label: 'Dashboard',
    icon: <Monitor size={16} />,
    prompt: 'Create an analytics dashboard with a stats row, chart area, data table, and filter interactions. Split layout, data, and behavior across files.',
    color: '#f59e0b',
    files: {
      'index.html': `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Dashboard</title>
    <link rel="stylesheet" href="styles/dashboard.css" />
  </head>
  <body>
    <main id="app"></main>
    <script src="scripts/data.js"></script>
    <script src="scripts/main.js"></script>
  </body>
</html>`,
      'styles/dashboard.css': 'body {\n  margin: 0;\n}\n',
      'scripts/data.js': 'window.dashboardData = [];\n',
      'scripts/main.js': `document.getElementById('app').innerHTML = '<h1>Dashboard starter</h1>';`,
    },
  },
  {
    label: 'React App',
    icon: <Zap size={16} />,
    prompt: 'Build a React single-page app with a navbar, hero section, and interactive card grid. Use React 19 with ESM imports from esm.sh. Keep components in separate files with clean JSX.',
    color: '#61dafb',
    files: {
      'index.html': `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>React App</title>
    <script type="importmap">
    {
      "imports": {
        "react": "https://esm.sh/react@19",
        "react-dom/client": "https://esm.sh/react-dom@19/client",
        "react/jsx-runtime": "https://esm.sh/react@19/jsx-runtime"
      }
    }
    </script>
    <link rel="stylesheet" href="styles/app.css" />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="src/main.jsx"></script>
  </body>
</html>`,
      'src/main.jsx': `import { createRoot } from 'react-dom/client';
import App from './App.jsx';
createRoot(document.getElementById('root')).render(<App />);`,
      'src/App.jsx': `export default function App() {
  return <div style={{ fontFamily: 'system-ui', padding: 24 }}><h1>React App</h1><p>Start building!</p></div>;
}`,
      'styles/app.css': `* { margin: 0; box-sizing: border-box; }
body { font-family: system-ui, -apple-system, sans-serif; background: #0a0a0a; color: #ededed; }`,
    },
  },
  {
    label: 'API + Frontend',
    icon: <Server size={16} />,
    prompt: 'Build a frontend that fetches data from a mock API module. Include a data table, search/filter bar, and loading states. Split the API logic, UI rendering, and styles into separate files.',
    color: '#10b981',
    files: {
      'index.html': `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>API Frontend</title>
    <link rel="stylesheet" href="styles/main.css" />
  </head>
  <body>
    <main id="app"></main>
    <script type="module" src="scripts/api.js"></script>
    <script type="module" src="scripts/app.js"></script>
  </body>
</html>`,
      'styles/main.css': `* { margin: 0; box-sizing: border-box; }
body { font-family: system-ui, sans-serif; background: #fafafa; color: #1a1a1a; }`,
      'scripts/api.js': `// Mock API module
export async function fetchData(query = '') {
  await new Promise(r => setTimeout(r, 300));
  const items = [
    { id: 1, name: 'Item A', status: 'active' },
    { id: 2, name: 'Item B', status: 'pending' },
    { id: 3, name: 'Item C', status: 'active' },
  ];
  return query ? items.filter(i => i.name.toLowerCase().includes(query.toLowerCase())) : items;
}`,
      'scripts/app.js': `import { fetchData } from './api.js';
document.getElementById('app').innerHTML = '<h1>Loading...</h1>';
fetchData().then(data => {
  document.getElementById('app').innerHTML = '<h1>Data loaded: ' + data.length + ' items</h1>';
});`,
    },
  },
  {
    label: 'Vite + React',
    icon: <Zap size={16} />,
    prompt: 'Build a modern React app with Vite. Include a navbar, hero section, and card grid. Use Tailwind CSS for styling. Make it look professional and polished.',
    color: '#bd34fe',
    files: {
      'package.json': JSON.stringify({
        name: 'aura-react-project',
        private: true,
        type: 'module',
        scripts: { dev: 'vite', build: 'vite build', preview: 'vite preview' },
        dependencies: { react: '^19.0.0', 'react-dom': '^19.0.0' },
        devDependencies: { '@vitejs/plugin-react': '^4.4.0', vite: '^6.0.0', autoprefixer: '^10.4.0', postcss: '^8.4.0', tailwindcss: '^3.4.0' },
      }, null, 2),
      'vite.config.js': `import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({ plugins: [react()] });`,
      'index.html': `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>React App</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>`,
      'src/main.jsx': `import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
ReactDOM.createRoot(document.getElementById('root')).render(<App />);`,
      'src/App.jsx': `export default function App() {
  return (
    <div style={{ fontFamily: 'system-ui', padding: 24 }}>
      <h1>Vite + React</h1>
      <p>Edit src/App.jsx to get started.</p>
    </div>
  );
}`,
      'src/index.css': `@tailwind base;
@tailwind components;
@tailwind utilities;
body { margin: 0; }`,
      'tailwind.config.js': `export default { content: ['./index.html', './src/**/*.{js,jsx}'], theme: { extend: {} }, plugins: [] };`,
      'postcss.config.js': `export default { plugins: { tailwindcss: {}, autoprefixer: {} } };`,
    },
  },
];

/* ─── Helpers ─── */
function stripFences(s: string): string {
  return s.replace(/^```[\w\-\.]*\r?\n?/, '').replace(/\r?\n?```[\w\-\.]*\s*$/, '').trim();
}

/** Inject error-handler + element-selection scripts into HTML for iframe srcdoc */
function injectIframeScripts(html: string): string {
  const script = `<script>
window.onerror = function(msg, src, line, col, err) {
  parent.postMessage({ type: 'artifact-error', msg: String(msg), line: line, col: col, stack: err ? err.stack : '' }, '*');
};
window.addEventListener('unhandledrejection', function(e) {
  parent.postMessage({ type: 'artifact-error', msg: String(e.reason), line: 0 }, '*');
});
var _auraSelectMode = false;
window.addEventListener('message', function(e) {
  if (e.data && e.data.type === 'toggle-select-mode') _auraSelectMode = e.data.enabled;
});
var _auraStyle = document.createElement('style');
_auraStyle.textContent = '.aura-highlight { outline: 2px solid #3b82f6 !important; outline-offset: 2px; cursor: crosshair !important; }';
document.head.appendChild(_auraStyle);
function _auraGetPath(el) {
  var parts = [];
  while (el && el !== document.body && el !== document.documentElement) {
    var tag = el.tagName.toLowerCase();
    if (el.id) { parts.unshift(tag + '#' + el.id); break; }
    else if (el.className && typeof el.className === 'string') { parts.unshift(tag + '.' + el.className.trim().split(/\\s+/).join('.')); }
    else { parts.unshift(tag); }
    el = el.parentElement;
  }
  return parts.join(' > ');
}
document.addEventListener('mouseover', function(e) {
  if (!_auraSelectMode) return;
  document.querySelectorAll('.aura-highlight').forEach(function(el) { el.classList.remove('aura-highlight'); });
  if (e.target !== document.body && e.target !== document.documentElement) e.target.classList.add('aura-highlight');
}, true);
document.addEventListener('mouseout', function(e) {
  if (!_auraSelectMode) return;
  e.target.classList.remove('aura-highlight');
}, true);
document.addEventListener('click', function(e) {
  if (!_auraSelectMode) return;
  e.preventDefault();
  e.stopPropagation();
  var el = e.target;
  el.classList.remove('aura-highlight');
  parent.postMessage({
    type: 'element-selected',
    tagName: el.tagName,
    classes: (typeof el.className === 'string') ? el.className : '',
    id: el.id || '',
    text: (el.textContent || '').slice(0, 100).trim(),
    outerHTML: el.outerHTML.slice(0, 500),
    path: _auraGetPath(el)
  }, '*');
}, true);
</script>`;
  if (html.includes('</head>')) {
    return html.replace('</head>', script + '</head>');
  }
  return script + html;
}

function buildWebRepairPrompt(currentHtml: string, errorMessage: string, line?: number): string {
  return `The current HTML page has a runtime error in the browser preview.

ERROR: ${errorMessage}
LINE: ${line || 'unknown'}

Current HTML:
\`\`\`html
${currentHtml}
\`\`\`

Return the complete corrected HTML document only. Preserve the user's design intent and existing functionality.`;
}

function buildWebPageName(input: string): string {
  const compact = input.trim().replace(/\s+/g, ' ').slice(0, 42).trim();
  return compact || 'Untitled page';
}

function escapeXml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function buildWebPageThumbnailDataUrl(name: string, prompt: string, html: string): string {
  const summary = (prompt.trim() || html.replace(/\s+/g, ' ').trim()).slice(0, 70);
  const codePreview = html.replace(/\s+/g, ' ').trim();
  const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="640" height="320" viewBox="0 0 640 320">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="#1d4ed8"/>
    </linearGradient>
  </defs>
  <rect width="640" height="320" rx="28" fill="url(#bg)"/>
  <rect x="30" y="28" width="580" height="264" rx="22" fill="rgba(255,255,255,0.08)" stroke="rgba(255,255,255,0.14)"/>
  <rect x="48" y="46" width="120" height="24" rx="12" fill="rgba(15,23,42,0.45)"/>
  <text x="108" y="62" text-anchor="middle" fill="#c4b5fd" font-family="Inter, Arial, sans-serif" font-size="12" font-weight="700" letter-spacing="1.2">WEB PAGE</text>
  <text x="48" y="105" fill="#f8fafc" font-family="Inter, Arial, sans-serif" font-size="28" font-weight="700">${escapeXml(name.slice(0, 28))}</text>
  <text x="48" y="138" fill="rgba(248,250,252,0.84)" font-family="Inter, Arial, sans-serif" font-size="15">${escapeXml(summary)}</text>
  <rect x="48" y="172" width="544" height="94" rx="16" fill="rgba(2,6,23,0.34)"/>
  <text x="66" y="200" fill="rgba(248,250,252,0.7)" font-family="'JetBrains Mono', Consolas, monospace" font-size="12">${escapeXml(codePreview.slice(0, 84) || '<html>')}</text>
  <text x="66" y="224" fill="rgba(248,250,252,0.52)" font-family="'JetBrains Mono', Consolas, monospace" font-size="12">${escapeXml(codePreview.slice(84, 168))}</text>
</svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function buildProjectThumbnailDataUrl(name: string, prompt: string, fileCount: number, entryPoint: string, framework: string): string {
  const summary = (prompt.trim() || `${framework} project`).slice(0, 64);
  const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="640" height="320" viewBox="0 0 640 320">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#0b1120"/>
      <stop offset="100%" stop-color="#0f4c81"/>
    </linearGradient>
  </defs>
  <rect width="640" height="320" rx="28" fill="url(#bg)"/>
  <rect x="28" y="28" width="584" height="264" rx="24" fill="rgba(255,255,255,0.06)" stroke="rgba(255,255,255,0.12)"/>
  <rect x="48" y="50" width="116" height="26" rx="13" fill="rgba(15,23,42,0.48)"/>
  <text x="106" y="67" text-anchor="middle" fill="#7dd3fc" font-family="Inter, Arial, sans-serif" font-size="12" font-weight="700" letter-spacing="1.4">PROJECT</text>
  <text x="48" y="112" fill="#f8fafc" font-family="Inter, Arial, sans-serif" font-size="28" font-weight="700">${escapeXml(name.slice(0, 28))}</text>
  <text x="48" y="144" fill="rgba(248,250,252,0.8)" font-family="Inter, Arial, sans-serif" font-size="14">${escapeXml(summary)}</text>
  <rect x="48" y="174" width="240" height="84" rx="16" fill="rgba(15,23,42,0.38)"/>
  <text x="66" y="202" fill="#bae6fd" font-family="'JetBrains Mono', Consolas, monospace" font-size="12">${escapeXml(entryPoint)}</text>
  <text x="66" y="228" fill="rgba(248,250,252,0.64)" font-family="'JetBrains Mono', Consolas, monospace" font-size="12">${escapeXml(String(fileCount))} files</text>
  <text x="66" y="252" fill="rgba(248,250,252,0.52)" font-family="'JetBrains Mono', Consolas, monospace" font-size="12">${escapeXml(framework)}</text>
  <rect x="322" y="174" width="240" height="84" rx="16" fill="rgba(14,165,233,0.1)" stroke="rgba(125,211,252,0.2)"/>
  <text x="340" y="206" fill="#f8fafc" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="700">Virtual FS Snapshot</text>
  <text x="340" y="232" fill="rgba(248,250,252,0.72)" font-family="Inter, Arial, sans-serif" font-size="12">Reload the full project tree</text>
</svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function buildProjectSummary(fs: VirtualFSClass, activeFile: string): string {
  const files = fs.listFiles();
  return files.map((path) => {
    const content = fs.readFile(path) || '';
    const marker = path === activeFile ? ' [active]' : '';
    return `- ${path}${marker} (${content.length} chars)`;
  }).join('\n');
}

function buildProjectUserMessage(fs: VirtualFSClass, activeFile: string, request: string, selectedElement: {tagName: string, classes: string, id: string, text: string, outerHTML: string, path: string} | null) {
  const activeContent = fs.readFile(activeFile) || '';
  const elementContext = selectedElement
    ? `\nSelected element from preview:\n- Tag: ${selectedElement.tagName}\n- Path: ${selectedElement.path}\n- HTML: ${selectedElement.outerHTML}\n`
    : '';

  return `Current project files:
${buildProjectSummary(fs, activeFile)}

Entry point: ${fs.getProject().entryPoint}
Framework: ${fs.getProject().framework}
Active file:
===FILE: ${activeFile}===
${activeContent}
===END FILE===
${elementContext}
User request: ${request}`;
}

function toBuildPlanDrafts(items: BuildPlanItem[]): BuildPlanDraft[] {
  return items.map((item) => ({
    ...item,
    id: crypto.randomUUID(),
  }));
}

function reprioritizeBuildPlanDrafts(items: BuildPlanDraft[]): BuildPlanDraft[] {
  return items.map((item, index) => ({
    ...item,
    priority: index + 1,
  }));
}

function buildProjectPlanRequest(description: string, framework: string, files: string[]): string {
  const existingFiles = files.length > 0 ? files.map((path) => `- ${path}`).join('\n') : 'None';
  return `Plan a web project for this request:
${description}

Framework: ${framework}
Existing files:
${existingFiles}

Return the essential file plan in dependency order.`;
}

function buildProjectExecutionRequest(
  description: string,
  framework: string,
  planItems: BuildPlanDraft[],
  fs: VirtualFSClass,
) {
  const project = fs.getProject();
  const existingFiles = fs.listFiles().length > 0 ? buildProjectSummary(fs, fs.getProject().entryPoint) : 'None';
  return `Build or refine this ${framework} web project.

Project name: ${project.name}
Entry point: ${project.entryPoint}
Framework: ${framework}

Approved plan:
${formatBuildPlan(planItems)}

Existing project files:
${existingFiles}

User request:
${description}

Follow the approved plan order. Create or update the listed files first. Return file operations only.`;
}

/* ─── Shared styles ─── */
const btnBase: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 5,
  background: 'var(--s2)',
  border: '1px solid var(--b1)',
  borderRadius: 'var(--r-md)',
  color: 'var(--mu)',
  padding: '6px 10px',
  cursor: 'pointer',
  fontSize: '11.5px',
  fontFamily: 'inherit',
  transition: 'all 0.15s ease',
  whiteSpace: 'nowrap',
};

const btnHover: React.CSSProperties = {
  background: 'var(--pg)',
  borderColor: 'rgba(124,58,237,0.2)',
  color: 'var(--pl)',
};

const exportItemStyle: React.CSSProperties = {
  display: 'block', width: '100%', textAlign: 'left', background: 'none',
  border: 'none', color: '#ccc', padding: '6px 12px', fontSize: '11px',
  cursor: 'pointer', fontFamily: 'inherit',
};

/* ─── ActionBtn (outside component to avoid re-creation each render) ─── */
function ActionBtn({ id, icon, label, onClick, hoveredBtn, setHoveredBtn }: {
  id: string; icon: React.ReactNode; label: string; onClick: () => void;
  hoveredBtn: string | null; setHoveredBtn: (v: string | null) => void;
}) {
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHoveredBtn(id)}
      onMouseLeave={() => setHoveredBtn(null)}
      style={{
        ...btnBase,
        ...(hoveredBtn === id ? btnHover : {}),
      }}
    >
      {icon} {label}
    </button>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   WebCreatorPanel
   ═══════════════════════════════════════════════════════════════════════════ */
export default function WebCreatorPanel() {
  const { getModel } = useStore();
  const { versions, currentIdx, pushVersion, goToVersion, undo, redo, canUndo, canRedo, clear: clearVersions } = useVersionHistory(20, 'aura_webcreator_versions');
  const [creationSettings, setCreationSettings] = useState<CreationSettings>(getDefaultCreationSettings());

  /* ─── State ─── */
  const [creatorMode, setCreatorMode] = useState<CreatorMode>('single');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentHtml, setCurrentHtml] = useState('');
  const [projectJson, setProjectJson] = useState('');
  const [activeProjectFile, setActiveProjectFile] = useState('index.html');
  const [viewMode, setViewMode] = useState<ViewMode>('preview');
  const [deviceMode, setDeviceMode] = useState<DeviceMode>('desktop');
  const [fullscreen, setFullscreen] = useState(false);
  const [status, setStatus] = useState('');
  const [iframeError, setIframeError] = useState<string | null>(null);
  const [editorDiagnostics, setEditorDiagnostics] = useState<CodeEditorDiagnostic[]>([]);
  const [hoveredBtn, setHoveredBtn] = useState<string | null>(null);
  const [chatOpen, setChatOpen] = useState(true);
  const [conversationHistory, setConversationHistory] = useState<Array<{ role: string; content: string }>>([]);
  const [selectMode, setSelectMode] = useState(false);
  const [selectedElement, setSelectedElement] = useState<{tagName: string, classes: string, id: string, text: string, outerHTML: string, path: string} | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [detached, setDetached] = useState(false);
  const [themeOpen, setThemeOpen] = useState(false);
  const [galleryItems, setGalleryItems] = useState<SavedWebPage[]>([]);
  const [projectGalleryItems, setProjectGalleryItems] = useState<SavedWebProject[]>([]);
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [saveName, setSaveName] = useState('');
  const [renameDialogPage, setRenameDialogPage] = useState<SavedWebPage | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [deleteDialogPage, setDeleteDialogPage] = useState<SavedWebPage | null>(null);
  const [renameDialogProject, setRenameDialogProject] = useState<SavedWebProject | null>(null);
  const [deleteDialogProject, setDeleteDialogProject] = useState<SavedWebProject | null>(null);
  const [pendingDiff, setPendingDiff] = useState<PendingPageDiff | null>(null);
  const [galleryQuery, setGalleryQuery] = useState('');
  const [projectDialog, setProjectDialog] = useState<ProjectDialogState | null>(null);
  const [projectDialogValue, setProjectDialogValue] = useState('');
  const [projectWorkflowMode, setProjectWorkflowMode] = useState<ProjectWorkflowMode>('chat');
  const [buildStage, setBuildStage] = useState<BuildStage>('idle');
  const [buildPrompt, setBuildPrompt] = useState('');
  const [buildPlanDrafts, setBuildPlanDrafts] = useState<BuildPlanDraft[]>([]);
  const [buildCurrentFile, setBuildCurrentFile] = useState('');
  const [buildCompletedFiles, setBuildCompletedFiles] = useState<string[]>([]);
  const [buildProgressMessage, setBuildProgressMessage] = useState('');
  const [buildPlanError, setBuildPlanError] = useState('');
  const [useAgentBackend, setUseAgentBackend] = useState(false);

  // Agent build hook — connects to backend /api/agent/build
  const handleAgentFileUpdate = useCallback((filename: string, code: string, _artifactType: string) => {
    setProjectJson((prevJson) => {
      if (!prevJson) return prevJson;
      try {
        const fs = VirtualFS.fromJSON(prevJson);
        if (fs.readFile(filename) != null) {
          fs.updateFile(filename, code);
        } else {
          fs.createFile(filename, code);
        }
        const nextJson = fs.toJSON();
        setActiveProjectFile(filename);
        // Update preview
        try {
          const rebuilt = VirtualFS.fromJSON(nextJson);
          const html = rebuilt.buildBundle();
          if (html && iframeRef.current) {
            iframeRef.current.srcdoc = injectIframeScripts(html);
          }
        } catch { /* preview update failed, non-fatal */ }
        return nextJson;
      } catch {
        return prevJson;
      }
    });
  }, []);

  const [agentBuildState, agentBuildActions] = useAgentBuild(handleAgentFileUpdate);

  // Sync agent build state → UI state when using agent backend
  useEffect(() => {
    if (!useAgentBackend) return;
    if (agentBuildState.status === 'building') {
      setBuildStage('building');
      setBuildCurrentFile(agentBuildState.currentFile);
      setBuildProgressMessage(agentBuildState.progressMessage);
      const paths = agentBuildState.filesCreated;
      if (paths.length > 0) setBuildCompletedFiles(paths);
    } else if (agentBuildState.status === 'completed') {
      setBuildStage('completed');
      setBuildCurrentFile('');
      setBuildCompletedFiles(agentBuildState.filesCreated);
      setBuildProgressMessage(agentBuildState.progressMessage);
      setLoading(false);
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'ai',
        text: `Agent build complete. ${agentBuildState.filesCreated.length} files created. You can now refine the project in Chat mode.`,
        timestamp: Date.now(),
      }]);
    } else if (agentBuildState.status === 'error') {
      setBuildStage('error');
      setBuildPlanError(agentBuildState.error);
      setBuildProgressMessage(agentBuildState.error);
      setLoading(false);
    } else if (agentBuildState.status === 'cancelled') {
      setBuildStage('planned');
      setBuildProgressMessage('Build cancelled. You can resume or re-plan.');
      setLoading(false);
    }
  }, [agentBuildState, useAgentBackend]);

  const [themeColors, setThemeColors] = useState({
    primary: '#3b82f6',
    secondary: '#6366f1',
    accent: '#f59e0b',
    background: '#ffffff',
    text: '#111827',
  });
  const [autoFixAttempts, setAutoFixAttempts] = useState(0);
  const [isAutoFixing, setIsAutoFixing] = useState(false);

  /* ─── Visual AI Feedback ─── */
  const visualFeedback = useVisualFeedback();
  const [vfPanelOpen, setVfPanelOpen] = useState(false);

  /* ─── Component Library & Design Tokens ─── */
  const [componentGalleryOpen, setComponentGalleryOpen] = useState(false);
  const [designTokens, setDesignTokens] = useState<DesignTokens | null>(null);

  // Load design tokens on mount
  useEffect(() => {
    loadDesignTokens().then(t => { if (t) setDesignTokens(t); });
  }, []);

  // Consume panel handoff on mount (e.g., from CapturePanel or ArtifactsPanel)
  const { consumePanelHandoff } = useStore();
  useEffect(() => {
    const handoff = consumePanelHandoff();
    if (!handoff) return;
    if (handoff.code && typeof handoff.code === 'string') {
      setCurrentHtml(handoff.code);
      currentHtmlRef.current = handoff.code;
      updatePreview(handoff.code);
      setTimedStatus('Loaded from ' + (handoff.from || 'another panel'), 2000);
    }
    if (handoff.files && typeof handoff.files === 'object') {
      try {
        const fs = createDefaultWebProject(handoff.name || 'Imported Project');
        for (const [p, c] of Object.entries(handoff.files)) {
          if (typeof c === 'string') fs.createFile(p, c);
        }
        setProjectJson(fs.toJSON());
        setCreatorMode('project');
        setTimedStatus('Project loaded from ' + (handoff.from || 'another panel'), 2000);
      } catch { /* ignore */ }
    }
  }, []);

  const projectFs = useMemo(() => {
    if (!projectJson) return null;
    try {
      return VirtualFS.fromJSON(projectJson);
    } catch {
      return null;
    }
  }, [projectJson]);

  const projectFiles = projectFs?.listFiles() || [];
  const projectDirectories = projectFs?.listDirectories() || [];
  const activeProjectContent = projectFs?.readFile(activeProjectFile) || '';
  const activeProjectLanguage = detectCodeLanguageFromPath(activeProjectFile);
  const activeBundleHtml = creatorMode === 'project' ? projectFs?.buildBundle() || '' : currentHtml;

  /* ─── WebContainer for npm-based projects ─── */
  const projectFilesMap = useMemo(() => {
    if (!projectFs) return {};
    const map: Record<string, string> = {};
    for (const path of projectFs.listFiles()) {
      const content = projectFs.readFile(path);
      if (content != null) map[path] = content;
    }
    return map;
  }, [projectFs]);

  const hasPackageJson = !!projectFilesMap['package.json'];
  const [wcEnabled, setWcEnabled] = useState(false);
  const [showTerminal, setShowTerminal] = useState(false);

  const webContainer = useWebContainer({
    enabled: wcEnabled && creatorMode === 'project',
    files: projectFilesMap,
    onServerReady: (url) => {
      if (iframeRef.current) {
        iframeRef.current.src = url;
      }
    },
  });

  // Auto-enable WebContainer when project has package.json
  useEffect(() => {
    if (hasPackageJson && creatorMode === 'project' && !wcEnabled) {
      setWcEnabled(true);
    }
  }, [hasPackageJson, creatorMode, wcEnabled]);

  // Sync file changes to WebContainer when it's running
  useEffect(() => {
    if (!webContainer.serverUrl || !projectFs) return;
    // When a file changes while WebContainer is running, push the update
    // HMR will pick it up automatically
    const filePaths = projectFs.listFiles();
    for (const path of filePaths) {
      const content = projectFs.readFile(path);
      if (content != null) {
        webContainer.writeFile(path, content).catch(() => {});
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectJson]); // Re-run when project JSON changes (proxy for file changes)

  /* ─── Refs ─── */
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const detachedWindowRef = useRef<Window | null>(null);
  const currentHtmlRef = useRef('');
  const conversationHistoryRef = useRef<Array<{ role: string; content: string }>>([]);
  const loadingRef = useRef(false);
  const autoFixAttemptsRef = useRef(0);
  const autoFixingRef = useRef(false);
  const lastAutoFixSignatureRef = useRef<string | null>(null);
  const manualPreviewTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    currentHtmlRef.current = currentHtml;
  }, [currentHtml]);

  useEffect(() => {
    conversationHistoryRef.current = conversationHistory;
  }, [conversationHistory]);

  useEffect(() => {
    loadingRef.current = loading;
  }, [loading]);

  useEffect(() => {
    loadCreationSettings().then(setCreationSettings).catch(() => {});
  }, []);

  const setTimedStatus = useCallback((nextStatus: string, duration = 2000) => {
    setStatus(nextStatus);
    if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
    if (!nextStatus) return;
    statusTimerRef.current = setTimeout(() => setStatus(''), duration);
  }, []);

  const resetBuildWorkflow = useCallback(() => {
    setBuildStage('idle');
    setBuildPrompt('');
    setBuildPlanDrafts([]);
    setBuildCurrentFile('');
    setBuildCompletedFiles([]);
    setBuildProgressMessage('');
    setBuildPlanError('');
    agentBuildActions.reset();
  }, [agentBuildActions]);

  useEffect(() => {
    if (creatorMode !== 'project' || projectWorkflowMode === 'build') return;
    if (buildStage !== 'idle') {
      resetBuildWorkflow();
    }
  }, [buildStage, creatorMode, projectWorkflowMode, resetBuildWorkflow]);

  const refreshGallery = useCallback(async () => {
    const items = await webCreatorGalleryStore.list();
    const missingThumbnails = items.filter((item) => !item.thumbnail);
    if (missingThumbnails.length > 0) {
      await Promise.all(
        missingThumbnails.map((item) => webCreatorGalleryStore.update(item.id, {
          thumbnail: buildWebPageThumbnailDataUrl(item.name, item.prompt || '', item.html),
        })),
      );
      setGalleryItems(await webCreatorGalleryStore.list());
      return;
    }
    setGalleryItems(items);
  }, []);

  const refreshProjectGallery = useCallback(async () => {
    const items = await webCreatorProjectGalleryStore.list();
    const missingThumbnails = items.filter((item) => !item.thumbnail);
    if (missingThumbnails.length > 0) {
      await Promise.all(
        missingThumbnails.map((item) => webCreatorProjectGalleryStore.update(item.id, {
          thumbnail: buildProjectThumbnailDataUrl(
            item.name,
            item.prompt || '',
            item.fileCount,
            item.entryPoint,
            item.framework,
          ),
        })),
      );
      setProjectGalleryItems(await webCreatorProjectGalleryStore.list());
      return;
    }
    setProjectGalleryItems(items);
  }, []);

  useEffect(() => {
    void refreshGallery();
    void refreshProjectGallery();
  }, [refreshGallery, refreshProjectGallery]);

  /* ─── Restore persisted state on mount ─── */
  const restoredRef = useRef(false);
  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;
    wcStorageGet().then(saved => {
      if (!saved) return;
      if (saved.mode) setCreatorMode(saved.mode);
      if (saved.html) {
        setCurrentHtml(saved.html);
      }
      if (saved.project) setProjectJson(saved.project);
      if (saved.activeFile) setActiveProjectFile(saved.activeFile);
      if (saved.messages?.length) setMessages(saved.messages);
      if (saved.history?.length) setConversationHistory(saved.history);
    });
  }, []);

  /* ─── Auto-save state on changes ─── */
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      wcStorageSave({
        mode: creatorMode,
        html: creatorMode === 'single' ? currentHtml : '',
        project: creatorMode === 'project' ? projectJson : '',
        activeFile: creatorMode === 'project' ? activeProjectFile : undefined,
        messages,
        history: conversationHistory,
      });
    }, 1000); // Debounce 1s
    return () => { if (saveTimerRef.current) clearTimeout(saveTimerRef.current); };
  }, [activeProjectFile, conversationHistory, creatorMode, currentHtml, messages, projectJson]);

  /* ─── Cleanup ─── */
  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
      if (manualPreviewTimerRef.current) clearTimeout(manualPreviewTimerRef.current);
      if (detachedWindowRef.current && !detachedWindowRef.current.closed) {
        detachedWindowRef.current.close();
      }
    };
  }, []);

  /* ─── Extract HTML from AI response ─── */
  const extractHtml = useCallback((text: string): string => {
    let cleaned = stripFences(text);
    // If the response contains HTML but also has explanatory text, extract just the HTML
    const docTypeMatch = cleaned.match(/(<!DOCTYPE html[\s\S]*)/i);
    if (docTypeMatch) {
      cleaned = docTypeMatch[1];
    } else {
      const htmlMatch = cleaned.match(/(<html[\s\S]*<\/html>)/i);
      if (htmlMatch) {
        cleaned = htmlMatch[1];
      }
    }
    return cleaned;
  }, []);

  /* ─── iframe error listener ─── */
  /* ─── Auto-scroll chat ─── */
  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [messages]);

  /* ─── Update preview ─── */
  const updatePreview = useCallback((html: string) => {
    if (!iframeRef.current || !html) return;
    setIframeError(null);
    setEditorDiagnostics([]);
    // Inject design token CSS variables if active
    let finalHtml = html;
    if (designTokens) {
      const cssBlock = `<style id="aura-design-tokens">${tokensToCssVariables(designTokens)}</style>`;
      if (finalHtml.includes('</head>')) {
        finalHtml = finalHtml.replace('</head>', `${cssBlock}\n</head>`);
      } else if (finalHtml.includes('<body')) {
        finalHtml = finalHtml.replace('<body', `${cssBlock}\n<body`);
      } else {
        finalHtml = cssBlock + '\n' + finalHtml;
      }
    }
    iframeRef.current.srcdoc = injectIframeScripts(finalHtml);

    // Sync to detached window if open
    if (detachedWindowRef.current && !detachedWindowRef.current.closed) {
      detachedWindowRef.current.document.open();
      detachedWindowRef.current.document.write(html);
      detachedWindowRef.current.document.close();
    }
  }, []);

  const updateProjectPreview = useCallback((fs: VirtualFSClass | null) => {
    if (!fs) return;
    const bundled = fs.buildBundle();
    if (!bundled.trim()) return;
    updatePreview(bundled);
  }, [updatePreview]);

  useEffect(() => {
    if (!projectFs || !projectFiles.includes(activeProjectFile)) {
      if (projectFs) {
        setActiveProjectFile(projectFs.getProject().entryPoint);
      }
      return;
    }
  }, [activeProjectFile, projectFiles, projectFs]);

  const mutateProject = useCallback((mutator: (fs: VirtualFSClass) => void): VirtualFSClass | null => {
    const seed = projectFs || createDefaultWebProject('Aura Project');
    const nextFs = VirtualFS.fromJSON(seed.toJSON());
    mutator(nextFs);
    const serialized = nextFs.toJSON();
    setProjectJson(serialized);
    return nextFs;
  }, [projectFs]);

  const updateBuildPlanDraft = useCallback((id: string, patch: Partial<BuildPlanDraft>) => {
    setBuildPlanDrafts((prev) => reprioritizeBuildPlanDrafts(prev.map((item) => (
      item.id === id
        ? {
            ...item,
            ...patch,
            path: typeof patch.path === 'string'
              ? patch.path.replace(/\\/g, '/').replace(/^\.\//, '').replace(/^\/+/, '').trim()
              : item.path,
            purpose: typeof patch.purpose === 'string' ? patch.purpose : item.purpose,
          }
        : item
    ))));
  }, []);

  const moveBuildPlanDraft = useCallback((id: string, direction: -1 | 1) => {
    setBuildPlanDrafts((prev) => {
      const index = prev.findIndex((item) => item.id === id);
      const nextIndex = index + direction;
      if (index === -1 || nextIndex < 0 || nextIndex >= prev.length) return prev;
      const next = [...prev];
      const [item] = next.splice(index, 1);
      next.splice(nextIndex, 0, item);
      return reprioritizeBuildPlanDrafts(next);
    });
  }, []);

  const removeBuildPlanDraft = useCallback((id: string) => {
    setBuildPlanDrafts((prev) => reprioritizeBuildPlanDrafts(prev.filter((item) => item.id !== id)));
  }, []);

  const addBuildPlanDraft = useCallback(() => {
    setBuildPlanDrafts((prev) => reprioritizeBuildPlanDrafts([
      ...prev,
      {
        id: crypto.randomUUID(),
        path: prev.length === 0 ? 'index.html' : `scripts/file-${prev.length + 1}.js`,
        purpose: 'Additional build step',
        priority: prev.length + 1,
      },
    ]));
  }, []);

  const requestBuildPlan = useCallback(async (description: string) => {
    const trimmed = description.trim();
    if (!trimmed || loading) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      text: trimmed,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setBuildPrompt(trimmed);
    setBuildStage('planning');
    setBuildPlanDrafts([]);
    setBuildCompletedFiles([]);
    setBuildCurrentFile('');
    setBuildPlanError('');
    setBuildProgressMessage('Planning the file list...');
    setLoading(true);
    setStatus('Planning build...');

    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const newHistory = [
      ...conversationHistory,
      { role: 'user', content: trimmed },
    ];
    const model = getModel('webcreator') || undefined;
    const framework = projectFs?.getProject().framework || 'static';
    const files = projectFs?.listFiles() || [];

    try {
      let streamedText = '';
      for await (const chunk of streamRawGenerate(buildProjectPlanRequest(trimmed, framework, files), {
        systemPrompt: BUILD_PLAN_SYSTEM_PROMPT,
        model,
        history: conversationHistory.length > 0 ? conversationHistory : undefined,
        signal: ctrl.signal,
      })) {
        streamedText += chunk;
      }

      const nextPlan = parseBuildPlan(streamedText);
      if (nextPlan.length === 0) {
        throw new Error('Build planner returned no files');
      }

      setBuildPlanDrafts(toBuildPlanDrafts(nextPlan));
      setBuildStage('planned');
      setBuildProgressMessage(`Planned ${nextPlan.length} files. Review and start when ready.`);
      setMessages((prev) => [...prev, {
        id: crypto.randomUUID(),
        role: 'ai',
        text: `Build plan ready: ${nextPlan.map((item) => item.path).slice(0, 4).join(', ')}${nextPlan.length > 4 ? ` +${nextPlan.length - 4} more` : ''}.`,
        timestamp: Date.now(),
      }]);
      setConversationHistory([
        ...newHistory,
        { role: 'assistant', content: `Created a build plan with ${nextPlan.length} files for approval.` },
      ]);
      setTimedStatus('Build plan ready', 1800);
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        setBuildStage('idle');
        setBuildProgressMessage('');
        setTimedStatus('Planning cancelled', 1500);
      } else {
        const message = err?.message || 'Build planning failed';
        setBuildStage('error');
        setBuildPlanError(message);
        setBuildProgressMessage(message);
        setStatus(message);
        setMessages((prev) => [...prev, {
          id: crypto.randomUUID(),
          role: 'ai',
          text: `Error: ${message}`,
          timestamp: Date.now(),
        }]);
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }, [
    conversationHistory,
    getModel,
    loading,
    projectFs,
    setTimedStatus,
  ]);

  const runAgentBuild = useCallback(async () => {
    if (buildPlanDrafts.length === 0 || loading) return;
    const framework = projectFs?.getProject().framework || 'static';

    // Seed files from current project (if any)
    const seedFiles = (projectFs?.listFiles() || []).map(path => ({
      path,
      content: projectFs?.readFile(path) || '',
    })).filter(f => f.content.trim());

    const plan: AgentBuildPlanItem[] = buildPlanDrafts.map(d => ({
      path: d.path,
      purpose: d.purpose,
      priority: d.priority,
    }));

    setLoading(true);
    setStatus('Starting agent build...');
    setMessages(prev => [...prev, {
      id: crypto.randomUUID(),
      role: 'ai',
      text: `Starting agent build with ${plan.length} planned files...`,
      timestamp: Date.now(),
    }]);

    const model = getModel('webcreator') || undefined;
    await agentBuildActions.startBuild(buildPrompt || 'Build the approved project plan.', framework, plan, seedFiles, model);
    // Progress and completion are handled by the useEffect syncing agentBuildState
  }, [agentBuildActions, buildPlanDrafts, buildPrompt, getModel, loading, projectFs]);

  const runApprovedBuild = useCallback(async () => {
    if (buildPlanDrafts.length === 0 || loading) return;

    // Use backend agent if toggled
    if (useAgentBackend) {
      return runAgentBuild();
    }

    const framework = projectFs?.getProject().framework || 'static';
    const seedProject = projectFs || createDefaultWebProject(buildWebPageName(buildPrompt || 'Aura Project'));
    if (!projectFs) {
      setProjectJson(seedProject.toJSON());
      setActiveProjectFile(seedProject.getProject().entryPoint);
      updateProjectPreview(seedProject);
    }
    const workingProject = VirtualFS.fromJSON(seedProject.toJSON());
    const fallbackPath = workingProject.getProject().entryPoint;
    const newHistory = [
      ...conversationHistory,
      { role: 'user', content: buildPrompt || 'Build the approved project plan.' },
      { role: 'assistant', content: `Approved build plan:\n${formatBuildPlan(buildPlanDrafts)}` },
    ];

    setBuildStage('building');
    setBuildPlanError('');
    setBuildCompletedFiles([]);
    setBuildCurrentFile(buildPlanDrafts[0]?.path || '');
    setBuildProgressMessage('Writing files...');
    setViewMode('preview');
    setLoading(true);
    setStatus('Building project...');
    setMessages((prev) => [...prev, {
      id: crypto.randomUUID(),
      role: 'ai',
      text: `Starting build with ${buildPlanDrafts.length} planned files.`,
      timestamp: Date.now(),
    }]);

    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const model = getModel('webcreator') || undefined;

    try {
      let streamedText = '';
      let appliedCount = 0;
      const appliedPaths = new Set<string>();

      for await (const chunk of streamRawGenerate(
        buildProjectExecutionRequest(buildPrompt || 'Build the approved project plan.', framework, buildPlanDrafts, workingProject),
        {
          systemPrompt: PROJECT_SYSTEM_PROMPT,
          model,
          history: conversationHistory.length > 0 ? conversationHistory : undefined,
          signal: ctrl.signal,
        },
      )) {
        streamedText += chunk;
        const readyOperations = parseFileOperations(streamedText, fallbackPath, false);
        const newOperations = readyOperations.slice(appliedCount);
        if (newOperations.length === 0) continue;
        appliedCount = readyOperations.length;

        for (const operation of newOperations) {
          if (operation.type === 'delete') {
            workingProject.deleteFile(operation.path);
          } else if (operation.type === 'create') {
            workingProject.createFile(operation.path, operation.content || '');
          } else {
            workingProject.updateFile(operation.path, operation.content || '');
          }
          appliedPaths.add(operation.path);
          const nextJson = workingProject.toJSON();
          setProjectJson(nextJson);
          if (workingProject.readFile(operation.path) != null) {
            setActiveProjectFile(operation.path);
          }
          updateProjectPreview(workingProject);
          const completed = Array.from(appliedPaths);
          setBuildCompletedFiles(completed);
          const nextPending = buildPlanDrafts.find((item) => !appliedPaths.has(item.path))?.path || '';
          setBuildCurrentFile(nextPending);
          setBuildProgressMessage(`${completed.length}/${buildPlanDrafts.length} files applied`);
        }
      }

      if (appliedCount === 0) {
        const fallbackOperations = parseFileOperations(streamedText, fallbackPath, true);
        for (const operation of fallbackOperations) {
          if (operation.type === 'delete') {
            workingProject.deleteFile(operation.path);
          } else if (operation.type === 'create') {
            workingProject.createFile(operation.path, operation.content || '');
          } else {
            workingProject.updateFile(operation.path, operation.content || '');
          }
          appliedCount += 1;
        }
        if (fallbackOperations.length > 0) {
          setProjectJson(workingProject.toJSON());
          setActiveProjectFile(fallbackOperations.find((operation) => operation.type !== 'delete')?.path || fallbackPath);
          updateProjectPreview(workingProject);
          setBuildCompletedFiles(fallbackOperations.map((operation) => operation.path));
        }
      }

      const completedFiles = Array.from(new Set(workingProject.listFiles().filter((path) => buildPlanDrafts.some((item) => item.path === path))));
      setBuildCompletedFiles(completedFiles);
      setBuildCurrentFile('');
      setBuildStage('completed');
      setBuildProgressMessage(`Build complete: ${completedFiles.length}/${buildPlanDrafts.length} planned files ready.`);
      setMessages((prev) => [...prev, {
        id: crypto.randomUUID(),
        role: 'ai',
        text: `Build complete. ${completedFiles.length} planned file${completedFiles.length === 1 ? '' : 's'} are now in the project tree.`,
        timestamp: Date.now(),
      }]);
      setConversationHistory([
        ...newHistory,
        { role: 'assistant', content: `Completed the approved multi-file build across ${Math.max(completedFiles.length, 1)} file(s).` },
      ]);
      setTimedStatus('Build complete', 1800);
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        setBuildStage('planned');
        setBuildCurrentFile('');
        setBuildProgressMessage('Build stopped. You can resume when ready.');
        setTimedStatus('Build stopped', 1500);
        setMessages((prev) => [...prev, {
          id: crypto.randomUUID(),
          role: 'ai',
          text: 'Build stopped.',
          timestamp: Date.now(),
        }]);
      } else {
        const message = err?.message || 'Build failed';
        setBuildStage('error');
        setBuildPlanError(message);
        setBuildCurrentFile('');
        setBuildProgressMessage(message);
        setStatus(message);
        setMessages((prev) => [...prev, {
          id: crypto.randomUUID(),
          role: 'ai',
          text: `Error: ${message}`,
          timestamp: Date.now(),
        }]);
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }, [
    buildPlanDrafts,
    buildPrompt,
    conversationHistory,
    getModel,
    loading,
    projectFs,
    runAgentBuild,
    setTimedStatus,
    updateProjectPreview,
    useAgentBackend,
  ]);

  const runPreviewAutoFix = useCallback(async (errorMessage: string, line?: number, ignoreLimit = false) => {
    if (!currentHtmlRef.current.trim() || autoFixingRef.current) return;
    if (!ignoreLimit && autoFixAttemptsRef.current >= MAX_AUTO_FIX_ATTEMPTS) return;

    const attempt = autoFixAttemptsRef.current + 1;
    const model = getModel('webcreator') || undefined;

    setIsAutoFixing(true);
    autoFixingRef.current = true;
    setStatus(ignoreLimit ? 'Fixing preview error...' : `Auto-fixing preview error (${attempt}/${MAX_AUTO_FIX_ATTEMPTS})...`);

    try {
      let streamedText = '';
      for await (const chunk of streamRawGenerate(buildWebRepairPrompt(currentHtmlRef.current, errorMessage, line), {
        systemPrompt: SYSTEM_PROMPT,
        model,
        history: conversationHistoryRef.current.length > 0 ? conversationHistoryRef.current : undefined,
      })) {
        streamedText += chunk;
      }

      const finalHtml = extractHtml(streamedText);
      if (!finalHtml.trim()) {
        throw new Error('Auto-fix returned empty HTML');
      }

      autoFixAttemptsRef.current = attempt;
      setAutoFixAttempts(attempt);
      setCurrentHtml(finalHtml);
      setEditorDiagnostics([]);
      updatePreview(finalHtml);
      pushVersion(`Auto-fix attempt ${attempt}: ${errorMessage}`, finalHtml, `Fix ${attempt}`);
      setConversationHistory((prev) => ([
        ...prev,
        { role: 'user', content: `Preview runtime error: ${errorMessage}` },
        { role: 'assistant', content: 'Returned a corrected HTML document that fixes the runtime error.' },
      ].slice(-20)));
      setTimedStatus(ignoreLimit ? 'Preview error fixed' : `Auto-fix applied (${attempt}/${MAX_AUTO_FIX_ATTEMPTS})`, 2400);
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        setTimedStatus(err?.message || 'Auto-fix failed', 3000);
      }
    } finally {
      setIsAutoFixing(false);
      autoFixingRef.current = false;
    }
  }, [extractHtml, getModel, pushVersion, setTimedStatus, updatePreview]);

  /* ─── Detach preview to separate window ─── */
  const detachPreview = useCallback(() => {
    if (detached && detachedWindowRef.current && !detachedWindowRef.current.closed) {
      detachedWindowRef.current.focus();
      return;
    }
    const win = window.open('', 'aura-preview', 'width=1024,height=768');
    if (!win) return;
    detachedWindowRef.current = win;
    setDetached(true);

    // Write current HTML to the new window
    if (activeBundleHtml) {
      win.document.open();
      win.document.write(activeBundleHtml);
      win.document.close();
    }

    // Listen for window close
    const checkClosed = setInterval(() => {
      if (win.closed) {
        clearInterval(checkClosed);
        setDetached(false);
        detachedWindowRef.current = null;
      }
    }, 500);
  }, [activeBundleHtml, detached]);

  /* ─── iframe error listener ─── */
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      if (!e.data || typeof e.data !== 'object') return;
      if (e.source !== iframeRef.current?.contentWindow) return;
      if (e.data.type === 'artifact-error') {
        const msg = e.data.msg || 'Unknown error';
        const lineNumber = typeof e.data.line === 'number' ? e.data.line : 0;
        const line = lineNumber ? ` (line ${lineNumber})` : '';
        setIframeError(`${msg}${line}`);
        setEditorDiagnostics(buildRuntimeDiagnostics(msg, lineNumber));
        if (creationSettings.autoFixErrors && !loadingRef.current && currentHtmlRef.current.trim()) {
          const signature = `${msg}|${lineNumber}|${currentHtmlRef.current}`;
          if (lastAutoFixSignatureRef.current !== signature) {
            lastAutoFixSignatureRef.current = signature;
            void runPreviewAutoFix(msg, lineNumber);
          }
        }
      }
      if (e.data.type === 'element-selected') {
        setSelectedElement(e.data);
        setSelectMode(false);
        if (iframeRef.current?.contentWindow) {
          iframeRef.current.contentWindow.postMessage({ type: 'toggle-select-mode', enabled: false }, '*');
        }
      }
    };
    window.addEventListener('message', handler);
    return () => window.removeEventListener('message', handler);
  }, [creationSettings.autoFixErrors, runPreviewAutoFix]);

  /* ─── Generate / iterate ─── */
  const sendMessage = useCallback(async (overrideText?: string) => {
    const text = (overrideText ?? input).trim();
    if (!text || loading) return;
    if (creatorMode === 'project' && projectWorkflowMode === 'build') {
      await requestBuildPlan(text);
      return;
    }
    const isProjectMode = creatorMode === 'project';
    const workingProject = isProjectMode
      ? (projectFs || createDefaultWebProject(buildWebPageName(text)))
      : null;
    const previousHtml = currentHtml;
    const shouldReviewDiff = !isProjectMode && creationSettings.showDiffBeforeApply && !!previousHtml.trim();

    // Add user message
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      text,
      timestamp: Date.now(),
    };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setAutoFixAttempts(0);
    autoFixAttemptsRef.current = 0;
    setIsAutoFixing(false);
    autoFixingRef.current = false;
    lastAutoFixSignatureRef.current = null;
    setPendingDiff(null);
    setLoading(true);
    setStatus('Generating...');
    setIframeError(null);
    setEditorDiagnostics([]);

    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    // Build conversation history for context continuity
    const newHistory = [
      ...conversationHistory,
      { role: 'user', content: text },
    ];

    // Build the user message (just the request + context, NOT the system prompt)
    let userMessage = text;
    let systemPrompt = SYSTEM_PROMPT;

    // Inject design tokens and component library into system prompt
    const tokenPrompt = designTokens ? tokensToSystemPrompt(designTokens) : '';
    const compSummary = await componentLibrary.getSummaryForLLM();
    systemPrompt += tokenPrompt + compSummary;

    if (isProjectMode && workingProject) {
      userMessage = buildProjectUserMessage(
        workingProject,
        activeProjectFile || workingProject.getProject().entryPoint,
        text,
        selectedElement,
      );
      systemPrompt = PROJECT_SYSTEM_PROMPT + tokenPrompt + compSummary;
    } else if (currentHtml) {
      let elementContext = '';
      if (selectedElement) {
        elementContext = `\n\nThe user has selected a specific element to modify:\nElement: <${selectedElement.tagName.toLowerCase()}>\nCSS path: ${selectedElement.path}\nElement HTML: ${selectedElement.outerHTML}\n\nOnly modify this specific element. Keep all other elements unchanged.`;
      }
      userMessage = `Current HTML page code:\n\`\`\`html\n${currentHtml}\n\`\`\`${elementContext}\n\nUser request: ${text}`;
    }

    const model = getModel('webcreator') || undefined;

    // Set up streaming preview controller
    const previewCtrl = isProjectMode || shouldReviewDiff
      ? null
      : new StreamingPreviewController((html) => {
          if (iframeRef.current) {
            iframeRef.current.srcdoc = injectIframeScripts(html);
          }
        });

    try {
      let streamedText = '';
      let htmlStartIdx = -1; // Track where HTML begins in the stream

      for await (const chunk of streamRawGenerate(userMessage, {
        systemPrompt,
        model,
        history: conversationHistory.length > 0 ? conversationHistory : undefined,
        signal: ctrl.signal,
      })) {
        streamedText += chunk;

        // Progressive preview: detect HTML start and feed ALL content from that point
        if (htmlStartIdx === -1) {
          const docTypeIdx = streamedText.search(/<!DOCTYPE html>/i);
          const htmlIdx = streamedText.search(/<html[\s >]/i);
          const startIdx = docTypeIdx !== -1 ? docTypeIdx : htmlIdx;
          if (!shouldReviewDiff && startIdx !== -1) {
            htmlStartIdx = startIdx;
            // Feed everything from HTML start to preview controller (catches initial chunks)
            previewCtrl?.append(streamedText.slice(htmlStartIdx));
          }
        } else {
          // HTML already started — just append the new chunk
          previewCtrl?.append(chunk);
        }
      }

      // Final extraction
      if (isProjectMode && workingProject) {
        const fallbackPath = activeProjectFile || workingProject.getProject().entryPoint;
        const operations = parseFileOperations(streamedText, fallbackPath);
        const nextFs = VirtualFS.fromJSON(workingProject.toJSON());
        const changedPaths: string[] = [];
        for (const operation of operations) {
          if (operation.type === 'delete') {
            nextFs.deleteFile(operation.path);
            changedPaths.push(operation.path);
            continue;
          }

          const content = operation.content || '';
          if (operation.type === 'create') {
            nextFs.createFile(operation.path, content);
          } else {
            nextFs.updateFile(operation.path, content);
          }
          changedPaths.push(operation.path);
        }

        const nextJson = nextFs.toJSON();
        setProjectJson(nextJson);
        setActiveProjectFile(changedPaths.find((path) => nextFs.readFile(path) != null) || nextFs.getProject().entryPoint);
        updateProjectPreview(nextFs);
        setViewMode('preview');

        const summary = changedPaths.length > 0
          ? `${operations.some((op) => op.type === 'delete') ? 'Updated project files' : 'Applied project updates'}: ${changedPaths.slice(0, 4).join(', ')}${changedPaths.length > 4 ? ` +${changedPaths.length - 4} more` : ''}.`
          : 'Project updated.';

        setMessages(prev => [...prev, {
          id: crypto.randomUUID(),
          role: 'ai',
          text: summary,
          timestamp: Date.now(),
        }]);

        setConversationHistory([
          ...newHistory,
          {
            role: 'assistant',
            content: operations.length > 0
              ? `Applied multi-file project updates across ${Math.max(changedPaths.length, 1)} file(s).`
              : 'Returned a project-mode response with no file operations.',
          },
        ]);

        setStatus(changedPaths.length > 0 ? `Updated ${changedPaths.length} file${changedPaths.length === 1 ? '' : 's'}` : '');
      } else {
        const finalHtml = extractHtml(streamedText);
        const diffMetrics = finalHtml && shouldReviewDiff ? summarizeDiff(previousHtml, finalHtml) : null;
        const shouldAutoAcceptDiff =
          !!diffMetrics &&
          diffMetrics.changedLineCount > 0 &&
          diffMetrics.changedLineCount <= creationSettings.autoAcceptDiffLineThreshold;
        const hasReviewableDiff =
          !!finalHtml &&
          !!diffMetrics &&
          diffMetrics.changedLineCount > 0 &&
          !shouldAutoAcceptDiff;
        if (finalHtml && !hasReviewableDiff) {
          setCurrentHtml(finalHtml);
          updatePreview(finalHtml);
          pushVersion(text, finalHtml);
          setViewMode('preview');
        } else if (finalHtml && hasReviewableDiff) {
          setPendingDiff({
            original: previousHtml,
            modified: finalHtml,
            prompt: text,
          });
          setViewMode('code');
        }

        setMessages(prev => [...prev, {
          id: crypto.randomUUID(),
          role: 'ai',
          text: finalHtml
            ? hasReviewableDiff
              ? 'Update ready for review.'
              : shouldAutoAcceptDiff
                ? 'Small update applied automatically.'
                : 'Page updated. Preview is live below.'
            : streamedText.slice(0, 200),
          timestamp: Date.now(),
        }]);

        setConversationHistory([
          ...newHistory,
          {
            role: 'assistant',
            content: finalHtml
              ? hasReviewableDiff
                ? 'Generated an updated HTML page proposal for review before applying changes.'
                : shouldAutoAcceptDiff
                  ? 'Applied a small HTML update automatically.'
                : 'Generated/updated the HTML page as requested.'
              : streamedText,
          },
        ]);

        setStatus(
          hasReviewableDiff
            ? diffMetrics && diffMetrics.changeRatio >= creationSettings.forceDiffReviewChangePercent / 100
              ? 'Large update ready for review'
              : 'Review changes'
            : shouldAutoAcceptDiff
              ? 'Applied small update automatically'
              : '',
        );
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setStatus(err.message || 'Request failed');
        const errMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'ai',
          text: `Error: ${err.message || 'Request failed'}`,
          timestamp: Date.now(),
        };
        setMessages(prev => [...prev, errMsg]);
      }
    } finally {
      previewCtrl?.dispose();
      setLoading(false);
      abortRef.current = null;
    }
  }, [
    input,
    loading,
    currentHtml,
    creatorMode,
    creationSettings.autoAcceptDiffLineThreshold,
    creationSettings.forceDiffReviewChangePercent,
    creationSettings.showDiffBeforeApply,
    conversationHistory,
    getModel,
    activeProjectFile,
    projectWorkflowMode,
    extractHtml,
    projectFs,
    requestBuildPlan,
    selectedElement,
    updatePreview,
    updateProjectPreview,
    pushVersion,
  ]);

  /* ─── Template click ─── */
  const handleTemplate = useCallback((t: Template) => {
    sendMessage(t.prompt);
  }, [sendMessage]);

  /* ─── Actions ─── */
  const copyCode = useCallback(() => {
    const content = creatorMode === 'project' ? activeProjectContent : currentHtml;
    if (!content) return;
    navigator.clipboard.writeText(content).then(() => {
      setStatus('Copied!');
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
      statusTimerRef.current = setTimeout(() => setStatus(''), 1500);
    });
  }, [activeProjectContent, creatorMode, currentHtml]);

  const downloadHtml = useCallback(() => {
    const isProjectMode = creatorMode === 'project';
    const content = isProjectMode ? activeProjectContent : currentHtml;
    if (!content) return;
    const mimeType = isProjectMode && activeProjectLanguage === 'css'
      ? 'text/css'
      : isProjectMode && activeProjectLanguage === 'json'
        ? 'application/json'
        : 'text/html';
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = isProjectMode ? getBaseName(activeProjectFile) : 'website.html';
    a.click();
    URL.revokeObjectURL(url);
  }, [activeProjectContent, activeProjectFile, activeProjectLanguage, creatorMode, currentHtml]);

  const downloadProjectBundleHtml = useCallback(() => {
    if (!projectFs) return;
    const bundled = projectFs.buildBundle();
    if (!bundled.trim()) return;
    const projectName = projectFs.getProject().name.trim() || 'aura-project';
    const safeName = projectName.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'aura-project';
    exportHTML(bundled, `${safeName}-bundle.html`);
    setTimedStatus('Bundled HTML downloaded', 1600);
  }, [projectFs, setTimedStatus]);

  const downloadProjectSnapshot = useCallback(() => {
    if (!projectFs) return;
    const project = projectFs.getProject();
    const safeName = project.name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'aura-project';
    exportJSON(projectFs.toSerializable(), `${safeName}.aura-project.json`);
    setTimedStatus('Project snapshot downloaded', 1600);
  }, [projectFs, setTimedStatus]);

  const copyDataUrl = useCallback(async () => {
    if (!currentHtml) return;
    const dataUrl = 'data:text/html;charset=utf-8,' + encodeURIComponent(currentHtml);
    try {
      await navigator.clipboard.writeText(dataUrl);
      setStatus('Data URL copied!');
      setTimeout(() => setStatus(''), 1500);
    } catch {
      // fallback
    }
  }, [currentHtml]);

  const openInCodeSandbox = useCallback(() => {
    if (creatorMode === 'project' && projectFs) {
      const files = Object.fromEntries(
        projectFs.listFiles().map((path) => [path, { content: projectFs.readFile(path) || '', isBinary: false }]),
      );
      const params = new URLSearchParams({
        parameters: btoa(JSON.stringify({ files })),
      });
      window.open(`https://codesandbox.io/api/v1/sandboxes/define?${params}`, '_blank');
      return;
    }
    if (!currentHtml) return;
    const params = new URLSearchParams({
      parameters: btoa(JSON.stringify({
        files: {
          'index.html': { content: currentHtml, isBinary: false },
          'package.json': { content: JSON.stringify({ dependencies: {} }), isBinary: false },
        },
      })),
    });
    window.open(`https://codesandbox.io/api/v1/sandboxes/define?${params}`, '_blank');
  }, [creatorMode, currentHtml, projectFs]);

  const openInStackBlitz = useCallback(() => {
    const isProjectMode = creatorMode === 'project';
    if (!currentHtml && !isProjectMode) return;
    const form = document.createElement('form');
    form.method = 'POST';
    form.action = 'https://stackblitz.com/run';
    form.target = '_blank';
    const addField = (name: string, value: string) => {
      const input = document.createElement('input');
      input.type = 'hidden';
      input.name = name;
      input.value = value;
      form.appendChild(input);
    };
    addField('project[template]', 'html');
    if (isProjectMode && projectFs) {
      projectFs.listFiles().forEach((path) => {
        addField(`project[files][${path}]`, projectFs.readFile(path) || '');
      });
    } else {
      addField('project[files][index.html]', currentHtml);
    }
    addField('project[title]', 'AURA WebCreator Export');
    document.body.appendChild(form);
    form.submit();
    document.body.removeChild(form);
  }, [creatorMode, currentHtml, projectFs]);

  /* ─── Deploy & Share ─── */
  const [deployDashboardOpen, setDeployDashboardOpen] = useState(false);
  const [sharing, setSharing] = useState(false);

  const getProjectFilesMap = useCallback((): Record<string, string> => {
    if (creatorMode === 'project' && projectFs) {
      const map: Record<string, string> = {};
      for (const p of projectFs.listFiles()) {
        const c = projectFs.readFile(p);
        if (c != null) map[p] = c;
      }
      return map;
    }
    return { 'index.html': currentHtml };
  }, [creatorMode, currentHtml, projectFs]);

  const handleShare = useCallback(async () => {
    if (sharing) return;
    setSharing(true);
    try {
      const files = getProjectFilesMap();
      const name = creatorMode === 'project' && projectFs
        ? projectFs.getProject().name
        : 'Web Page';
      const entryPoint = creatorMode === 'project' && projectFs
        ? projectFs.getProject().entryPoint
        : 'index.html';

      const result = await shareProject(name, files, entryPoint);
      await saveDeployment({
        id: result.id,
        projectName: result.projectName,
        platform: 'aura',
        url: result.url,
        status: 'live',
        createdAt: Date.now(),
        expiresAt: result.expiresAt * 1000,
      });
      try { await navigator.clipboard.writeText(result.url); } catch { /* ok */ }
      setTimedStatus(`Shared! URL copied: ${result.url}`, 4000);
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(), role: 'ai',
        text: `Project shared! Live at: ${result.url}\n(Expires in 7 days. URL copied to clipboard.)`,
        timestamp: Date.now(),
      }]);
    } catch (err: any) {
      setTimedStatus(err.message || 'Share failed', 3000);
    } finally {
      setSharing(false);
    }
  }, [sharing, getProjectFilesMap, creatorMode, projectFs, setTimedStatus]);

  const handleDownloadZip = useCallback(async () => {
    const files = getProjectFilesMap();
    const name = creatorMode === 'project' && projectFs
      ? projectFs.getProject().name
      : 'web-page';
    await downloadAsZip(name, files);
  }, [getProjectFilesMap, creatorMode, projectFs]);

  const handleGitHubDeploy = useCallback(async () => {
    let token = await loadGitHubToken();
    if (!token) {
      token = prompt('Enter your GitHub Personal Access Token\n(needs repo scope):') || '';
      if (!token) return;
      await saveGitHubToken(token);
    }
    const repoName = prompt('Repository name:', creatorMode === 'project' && projectFs
      ? projectFs.getProject().name
      : 'aura-web-page') || '';
    if (!repoName) return;

    setTimedStatus('Deploying to GitHub Pages...', 30000);
    try {
      const files = getProjectFilesMap();
      const url = await deployToGitHubPages(repoName, files, token);
      await saveDeployment({
        id: `gh-${Date.now()}`,
        projectName: repoName,
        platform: 'github-pages',
        url,
        status: 'live',
        createdAt: Date.now(),
      });
      try { await navigator.clipboard.writeText(url); } catch { /* ok */ }
      setTimedStatus(`Deployed! ${url}`, 5000);
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(), role: 'ai',
        text: `Deployed to GitHub Pages: ${url}\n(May take 1-2 minutes to go live. URL copied.)`,
        timestamp: Date.now(),
      }]);
    } catch (err: any) {
      setTimedStatus(err.message || 'Deploy failed', 3000);
    }
  }, [getProjectFilesMap, creatorMode, projectFs, setTimedStatus]);

  /* ─── Close export dropdown on outside click ─── */
  useEffect(() => {
    if (!exportOpen) return;
    const close = () => setExportOpen(false);
    document.addEventListener('click', close);
    return () => document.removeEventListener('click', close);
  }, [exportOpen]);

  const sendToCli = useCallback(async () => {
    const content = creatorMode === 'project'
      ? JSON.stringify({
          mode: 'project',
          entryPoint: projectFs?.getProject().entryPoint || 'index.html',
          files: projectFs?.listFiles().map((path) => ({ path, content: projectFs.readFile(path) || '' })) || [],
        }, null, 2)
      : currentHtml;
    if (!content) return;
    try {
      await fetch(`${HTTP}/api/feed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({
          type: 'webcreator',
          content,
          title: 'Web Creator export',
        }),
      });
      setStatus('Sent to CLI feed');
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
      statusTimerRef.current = setTimeout(() => setStatus(''), 2000);
    } catch {
      setStatus('Failed to send');
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
      statusTimerRef.current = setTimeout(() => setStatus(''), 2000);
    }
  }, [creatorMode, currentHtml, projectFs]);

  const clearConversation = useCallback(() => {
    setMessages([]);
    setCurrentHtml('');
    setProjectJson('');
    setActiveProjectFile('index.html');
    setConversationHistory([]);
    setStatus('');
    setIframeError(null);
    setEditorDiagnostics([]);
    setAutoFixAttempts(0);
    setIsAutoFixing(false);
    setPendingDiff(null);
    resetBuildWorkflow();
    autoFixAttemptsRef.current = 0;
    autoFixingRef.current = false;
    lastAutoFixSignatureRef.current = null;
    clearVersions();
    if (iframeRef.current) iframeRef.current.srcdoc = '';
  }, [clearVersions, resetBuildWorkflow]);

  const handleHtmlEditorChange = useCallback((nextHtml: string) => {
    if (creatorMode === 'project') {
      const nextFs = mutateProject((fs) => {
        fs.updateFile(activeProjectFile, nextHtml);
      });
      if (nextFs) {
        updateProjectPreview(nextFs);
      }
    } else {
      setCurrentHtml(nextHtml);
      currentHtmlRef.current = nextHtml;
    }
    setIframeError(null);
    setEditorDiagnostics([]);
    lastAutoFixSignatureRef.current = null;
  }, [activeProjectFile, creatorMode, mutateProject, updateProjectPreview]);

  const openSaveDialog = useCallback(() => {
    if (creatorMode === 'project') {
      if (!projectFs || projectFiles.length === 0) return;
      const baseName = buildWebPageName(messages.find((message) => message.role === 'user')?.text || input || projectFs.getProject().name);
      setSaveName(baseName);
      setSaveDialogOpen(true);
      return;
    }
    if (!currentHtml.trim()) return;
    const baseName = buildWebPageName(messages.find((message) => message.role === 'user')?.text || input);
    setSaveName(baseName);
    setSaveDialogOpen(true);
  }, [creatorMode, currentHtml, input, messages, projectFiles.length, projectFs]);

  const saveCurrentPage = useCallback(async () => {
    if (!currentHtml.trim()) return;
    const nextName = saveName.trim() || buildWebPageName(messages.find((message) => message.role === 'user')?.text || input);
    const thumbnail = await captureIframeThumbnailDataUrl({
      iframe: iframeRef.current,
      fallback: { name: nextName, prompt: input, html: currentHtml },
      buildFallback: ({ name, prompt, html }) => buildWebPageThumbnailDataUrl(name, prompt, html),
    });
    await webCreatorGalleryStore.save({
      name: nextName,
      html: currentHtml,
      prompt: messages.find((message) => message.role === 'user')?.text || input,
      thumbnail,
      tags: (messages.find((message) => message.role === 'user')?.text || input)
        .trim()
        .toLowerCase()
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, 6),
    });
    setSaveDialogOpen(false);
    await refreshGallery();
    setTimedStatus(`Saved "${nextName}"`, 1800);
  }, [currentHtml, input, messages, refreshGallery, saveName, setTimedStatus]);

  const saveCurrentProject = useCallback(async () => {
    if (!projectFs || projectFiles.length === 0) return;
    const project = projectFs.getProject();
    const nextName = saveName.trim() || buildWebPageName(messages.find((message) => message.role === 'user')?.text || input || project.name);
    const thumbnail = await captureIframeThumbnailDataUrl({
      iframe: iframeRef.current,
      fallback: {
        name: nextName,
        prompt: messages.find((message) => message.role === 'user')?.text || input,
        fileCount: projectFiles.length,
        entryPoint: project.entryPoint,
        framework: project.framework,
      },
      buildFallback: ({ name, prompt, fileCount, entryPoint, framework }) =>
        buildProjectThumbnailDataUrl(name, prompt, fileCount, entryPoint, framework),
    });
    await webCreatorProjectGalleryStore.save({
      name: nextName,
      project: projectFs.toJSON(),
      entryPoint: project.entryPoint,
      fileCount: projectFiles.length,
      framework: project.framework,
      prompt: messages.find((message) => message.role === 'user')?.text || input,
      thumbnail,
      tags: (messages.find((message) => message.role === 'user')?.text || input)
        .trim()
        .toLowerCase()
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, 6),
    });
    setSaveDialogOpen(false);
    await refreshProjectGallery();
    setTimedStatus(`Saved "${nextName}"`, 1800);
  }, [input, messages, projectFiles.length, projectFs, refreshProjectGallery, saveName, setTimedStatus]);

  const loadSavedPage = useCallback((page: SavedWebPage) => {
    setCreatorMode('single');
    setCurrentHtml(page.html);
    currentHtmlRef.current = page.html;
    setProjectJson('');
    setActiveProjectFile('index.html');
    setInput('');
    setIframeError(null);
    setEditorDiagnostics([]);
    setPendingDiff(null);
    setViewMode('preview');
    setDeviceMode('desktop');
    setMessages(page.prompt ? [{
      id: crypto.randomUUID(),
      role: 'user',
      text: page.prompt,
      timestamp: page.createdAt,
    }, {
      id: crypto.randomUUID(),
      role: 'ai',
      text: 'Page restored from gallery.',
      timestamp: page.updatedAt,
    }] : []);
    setConversationHistory(page.prompt
      ? [
        { role: 'user', content: page.prompt },
        { role: 'assistant', content: 'Restored the saved web page from the gallery.' },
      ]
      : []);
    updatePreview(page.html);
    setGalleryOpen(false);
    setTimedStatus(`Loaded "${page.name}"`, 1800);
  }, [setTimedStatus, updatePreview]);

  const forkSavedPage = useCallback(async (page: SavedWebPage) => {
    const forkName = `${page.name} Copy`;
    const id = await webCreatorGalleryStore.save({
      name: forkName,
      html: page.html,
      prompt: page.prompt,
      thumbnail: page.thumbnail || buildWebPageThumbnailDataUrl(forkName, page.prompt || '', page.html),
      tags: page.tags,
    });
    await refreshGallery();
    const next = await webCreatorGalleryStore.get(id);
    if (next) loadSavedPage(next);
    setTimedStatus(`Forked "${page.name}"`, 1800);
  }, [loadSavedPage, refreshGallery, setTimedStatus]);

  const loadSavedProject = useCallback((projectItem: SavedWebProject) => {
    const fs = VirtualFS.fromJSON(projectItem.project);
    setCreatorMode('project');
    setProjectJson(projectItem.project);
    setActiveProjectFile(fs.getProject().entryPoint);
    setCurrentHtml('');
    currentHtmlRef.current = '';
    setInput('');
    setIframeError(null);
    setEditorDiagnostics([]);
    setPendingDiff(null);
    setViewMode('preview');
    setDeviceMode('desktop');
    setMessages(projectItem.prompt ? [{
      id: crypto.randomUUID(),
      role: 'user',
      text: projectItem.prompt,
      timestamp: projectItem.createdAt,
    }, {
      id: crypto.randomUUID(),
      role: 'ai',
      text: 'Project restored from gallery.',
      timestamp: projectItem.updatedAt,
    }] : []);
    setConversationHistory(projectItem.prompt
      ? [
        { role: 'user', content: projectItem.prompt },
        { role: 'assistant', content: 'Restored the saved project from the gallery.' },
      ]
      : []);
    updateProjectPreview(fs);
    setGalleryOpen(false);
    setTimedStatus(`Loaded "${projectItem.name}"`, 1800);
  }, [setTimedStatus, updateProjectPreview]);

  const forkSavedProject = useCallback(async (projectItem: SavedWebProject) => {
    const forkName = `${projectItem.name} Copy`;
    const id = await webCreatorProjectGalleryStore.save({
      name: forkName,
      project: projectItem.project,
      entryPoint: projectItem.entryPoint,
      fileCount: projectItem.fileCount,
      framework: projectItem.framework,
      prompt: projectItem.prompt,
      thumbnail: projectItem.thumbnail || buildProjectThumbnailDataUrl(
        forkName,
        projectItem.prompt || '',
        projectItem.fileCount,
        projectItem.entryPoint,
        projectItem.framework,
      ),
      tags: projectItem.tags,
    });
    await refreshProjectGallery();
    const next = await webCreatorProjectGalleryStore.get(id);
    if (next) loadSavedProject(next);
    setTimedStatus(`Forked "${projectItem.name}"`, 1800);
  }, [loadSavedProject, refreshProjectGallery, setTimedStatus]);

  const openRenameDialog = useCallback((page: SavedWebPage) => {
    setRenameDialogPage(page);
    setRenameValue(page.name);
  }, []);

  const openRenameProjectDialog = useCallback((projectItem: SavedWebProject) => {
    setRenameDialogProject(projectItem);
    setRenameValue(projectItem.name);
  }, []);

  const submitRenamePage = useCallback(async () => {
    if (!renameDialogPage) return;
    const nextName = renameValue.trim() || renameDialogPage.name;
    await webCreatorGalleryStore.update(renameDialogPage.id, { name: nextName });
    setRenameDialogPage(null);
    setRenameValue('');
    await refreshGallery();
    setTimedStatus('Page renamed', 1500);
  }, [refreshGallery, renameDialogPage, renameValue, setTimedStatus]);

  const openDeleteDialog = useCallback((page: SavedWebPage) => {
    setDeleteDialogPage(page);
  }, []);

  const submitRenameProject = useCallback(async () => {
    if (!renameDialogProject) return;
    const nextName = renameValue.trim() || renameDialogProject.name;
    await webCreatorProjectGalleryStore.update(renameDialogProject.id, { name: nextName });
    setRenameDialogProject(null);
    setRenameValue('');
    await refreshProjectGallery();
    setTimedStatus('Project renamed', 1500);
  }, [refreshProjectGallery, renameDialogProject, renameValue, setTimedStatus]);

  const openDeleteProjectDialog = useCallback((projectItem: SavedWebProject) => {
    setDeleteDialogProject(projectItem);
  }, []);

  const confirmDeletePage = useCallback(async () => {
    if (!deleteDialogPage) return;
    await webCreatorGalleryStore.delete(deleteDialogPage.id);
    setDeleteDialogPage(null);
    await refreshGallery();
    setTimedStatus('Page deleted', 1500);
  }, [deleteDialogPage, refreshGallery, setTimedStatus]);

  const confirmDeleteProject = useCallback(async () => {
    if (!deleteDialogProject) return;
    await webCreatorProjectGalleryStore.delete(deleteDialogProject.id);
    setDeleteDialogProject(null);
    await refreshProjectGallery();
    setTimedStatus('Project deleted', 1500);
  }, [deleteDialogProject, refreshProjectGallery, setTimedStatus]);

  const acceptPendingDiff = useCallback(() => {
    if (!pendingDiff) return;
    setCurrentHtml(pendingDiff.modified);
    currentHtmlRef.current = pendingDiff.modified;
    setIframeError(null);
    setEditorDiagnostics([]);
    updatePreview(pendingDiff.modified);
    pushVersion(pendingDiff.prompt, pendingDiff.modified);
    setPendingDiff(null);
    setViewMode('preview');
    setTimedStatus('Changes applied', 1800);
  }, [pendingDiff, pushVersion, setTimedStatus, updatePreview]);

  const rejectPendingDiff = useCallback(() => {
    setPendingDiff(null);
    setTimedStatus('Kept current page', 1800);
  }, [setTimedStatus]);

  const ensureProjectInitialized = useCallback((name?: string, files?: Record<string, string>) => {
    const nextFs = files
      ? VirtualFS.createProject({
          name: name || 'Aura Project',
          framework: 'static',
          entryPoint: 'index.html',
          files,
        })
      : createDefaultWebProject(name || 'Aura Project');
    const serialized = nextFs.toJSON();
    setProjectJson(serialized);
    setActiveProjectFile(nextFs.getProject().entryPoint);
    updateProjectPreview(nextFs);
    return nextFs;
  }, [updateProjectPreview]);

  const switchCreatorMode = useCallback((nextMode: CreatorMode) => {
    if (nextMode === creatorMode) return;

    if (nextMode === 'project') {
      if (!projectFs) {
        if (currentHtml.trim()) {
          const migrated = VirtualFS.createProject({
            name: buildWebPageName(messages.find((message) => message.role === 'user')?.text || 'Migrated Project'),
            framework: 'static',
            entryPoint: 'index.html',
            files: {
              'index.html': currentHtml,
            },
          });
          setProjectJson(migrated.toJSON());
          setActiveProjectFile('index.html');
          updateProjectPreview(migrated);
        } else {
          ensureProjectInitialized('Aura Project');
        }
      } else {
        updateProjectPreview(projectFs);
      }
    } else if (projectFs && !currentHtml.trim()) {
      const bundled = projectFs.buildBundle();
      setCurrentHtml(bundled);
      currentHtmlRef.current = bundled;
      updatePreview(bundled);
    }

    setCreatorMode(nextMode);
    setViewMode('preview');
    setThemeOpen(false);
    setGalleryOpen(false);
    setPendingDiff(null);
    resetBuildWorkflow();
  }, [creatorMode, currentHtml, ensureProjectInitialized, messages, projectFs, resetBuildWorkflow, updatePreview, updateProjectPreview]);

  const handleProjectTemplate = useCallback((template: ProjectTemplate) => {
    setCreatorMode('project');
    setProjectWorkflowMode('chat');
    ensureProjectInitialized(template.label, template.files);
    setInput('');
    void sendMessage(template.prompt);
  }, [ensureProjectInitialized, sendMessage]);

  const openProjectDialog = useCallback((dialog: ProjectDialogState) => {
    setProjectDialog(dialog);
    setProjectDialogValue(dialog.initialValue);
  }, []);

  const submitProjectDialog = useCallback(() => {
    if (!projectDialog) return;
    const rawValue = projectDialogValue.trim();
    if (!rawValue) return;

    if (projectDialog.type === 'create-file') {
      const targetPath = rawValue.includes('/')
        ? rawValue
        : [projectDialog.parentPath, rawValue].filter(Boolean).join('/');
      const nextFs = mutateProject((fs) => {
        fs.createFile(targetPath, getDefaultFileContent(targetPath));
      });
      if (nextFs) {
        setActiveProjectFile(targetPath);
        updateProjectPreview(nextFs);
      }
      setTimedStatus(`Created ${targetPath}`, 1500);
    } else if (projectDialog.type === 'create-folder') {
      const targetPath = rawValue.includes('/')
        ? rawValue
        : [projectDialog.parentPath, rawValue].filter(Boolean).join('/');
      mutateProject((fs) => {
        fs.createDir(targetPath);
      });
      setTimedStatus(`Created ${targetPath}/`, 1500);
    } else if (projectDialog.type === 'rename-file' && projectDialog.path) {
      const parentDir = getParentDirectory(projectDialog.path);
      const targetPath = rawValue.includes('/')
        ? rawValue
        : [parentDir, rawValue].filter(Boolean).join('/');
      const nextFs = mutateProject((fs) => {
        fs.renameFile(projectDialog.path!, targetPath);
      });
      if (nextFs) {
        setActiveProjectFile(targetPath);
        updateProjectPreview(nextFs);
      }
      setTimedStatus(`Renamed to ${targetPath}`, 1500);
    }

    setProjectDialog(null);
    setProjectDialogValue('');
  }, [mutateProject, projectDialog, projectDialogValue, setTimedStatus, updateProjectPreview]);

  const deleteProjectFile = useCallback((path: string) => {
    const nextFs = mutateProject((fs) => {
      fs.deleteFile(path);
    });
    if (activeProjectFile === path && nextFs) {
      setActiveProjectFile(nextFs.getProject().entryPoint);
      updateProjectPreview(nextFs);
    }
    setTimedStatus(`Deleted ${path}`, 1500);
  }, [activeProjectFile, mutateProject, setTimedStatus, updateProjectPreview]);

  /* ─── Toggle element select mode ─── */
  const toggleSelectMode = useCallback(() => {
    const next = !selectMode;
    setSelectMode(next);
    setSelectedElement(null);
    if (iframeRef.current?.contentWindow) {
      iframeRef.current.contentWindow.postMessage({ type: 'toggle-select-mode', enabled: next }, '*');
    }
  }, [selectMode]);

  /* ─── Apply theme ─── */
  const applyTheme = useCallback((colors: typeof themeColors) => {
    if (!currentHtml) return;
    const cssVars = `<!-- AURA_THEME --><style>:root { --primary: ${colors.primary}; --secondary: ${colors.secondary}; --accent: ${colors.accent}; --bg: ${colors.background}; --text: ${colors.text}; } body { background-color: ${colors.background}; color: ${colors.text}; }</style><!-- /AURA_THEME -->`;
    // Remove previous theme block if exists
    let html = currentHtml.replace(/<!-- AURA_THEME -->[\s\S]*?<!-- \/AURA_THEME -->/g, '');
    if (html.includes('</head>')) {
      html = html.replace('</head>', cssVars + '</head>');
    } else {
      html = cssVars + html;
    }
    setCurrentHtml(html);
    updatePreview(html);
  }, [currentHtml, updatePreview]);

  /* ─── Version restore ─── */
  const restoreVersion = useCallback((idx: number) => {
    const v = goToVersion(idx);
    if (v) {
      setCreatorMode('single');
      setCurrentHtml(v.code);
      updatePreview(v.code);
    }
  }, [goToVersion, updatePreview]);

  useEffect(() => {
    if (loading) return;
    if (manualPreviewTimerRef.current) clearTimeout(manualPreviewTimerRef.current);

    if (creatorMode === 'project') {
      if (!projectFs || !projectFs.buildBundle().trim()) {
        if (iframeRef.current) iframeRef.current.srcdoc = '';
        return;
      }

      manualPreviewTimerRef.current = setTimeout(() => {
        updateProjectPreview(projectFs);
      }, 250);

      return () => {
        if (manualPreviewTimerRef.current) clearTimeout(manualPreviewTimerRef.current);
      };
    }

    if (!currentHtml.trim()) {
      if (iframeRef.current) iframeRef.current.srcdoc = '';
      return;
    }

    manualPreviewTimerRef.current = setTimeout(() => {
      updatePreview(currentHtml);
    }, 250);

    return () => {
      if (manualPreviewTimerRef.current) clearTimeout(manualPreviewTimerRef.current);
    };
  }, [creatorMode, currentHtml, loading, projectFs, updatePreview, updateProjectPreview]);

  /* ─── Device mode tabs ─── */
  const deviceTabs: { mode: DeviceMode; icon: React.ReactNode; label: string }[] = [
    { mode: 'desktop', icon: <Monitor size={13} />, label: 'Desktop' },
    { mode: 'tablet', icon: <Tablet size={13} />, label: 'Tablet' },
    { mode: 'mobile', icon: <Smartphone size={13} />, label: 'Mobile' },
  ];

  /* ─── Panel layout ─── */
  const panelStyle: React.CSSProperties = fullscreen
    ? { position: 'fixed', inset: 0, zIndex: 9999, background: 'var(--bg)', display: 'flex', flexDirection: 'column' }
    : { display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' };

  const hasHtml = !!currentHtml;
  const hasProject = !!projectFs && projectFiles.length > 0;
  const hasContent = creatorMode === 'project' ? hasProject : hasHtml;
  const isEmptyState = messages.length === 0 && !hasContent;
  const copyLabel = creatorMode === 'project' ? `Copy ${getBaseName(activeProjectFile || 'file')}` : 'Copy HTML';
  const saveLabel = creatorMode === 'project' ? 'Save Project' : 'Save Page';
  const galleryCount = creatorMode === 'project' ? projectGalleryItems.length : galleryItems.length;
  const normalizedGalleryQuery = galleryQuery.trim().toLowerCase();
  const filteredGalleryItems = galleryItems.filter((item) => {
    const haystack = `${item.name} ${item.prompt || ''} ${(item.tags || []).join(' ')}`.toLowerCase();
    return !normalizedGalleryQuery || haystack.includes(normalizedGalleryQuery);
  });
  const filteredProjectGalleryItems = projectGalleryItems.filter((item) => {
    const haystack = `${item.name} ${item.prompt || ''} ${item.entryPoint} ${item.framework} ${(item.tags || []).join(' ')}`.toLowerCase();
    return !normalizedGalleryQuery || haystack.includes(normalizedGalleryQuery);
  });
  const buildCompletedCount = buildCompletedFiles.length;
  const buildPlannedCount = buildPlanDrafts.length;
  const buildProgressRatio = buildPlannedCount > 0
    ? Math.min(1, buildCompletedCount / buildPlannedCount)
    : 0;

  return (
    <div style={panelStyle}>
      <OfflineBanner />
      {/* ═══ Top bar ═══ */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', flexShrink: 0,
        borderBottom: '1px solid var(--b1)',
      }}>
        <Layout size={15} style={{ color: 'var(--pl)', flexShrink: 0 }} />
        <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--tx)', letterSpacing: '0.02em' }}>
          Web Creator
        </span>

        <div style={{
          display: 'flex', background: 'var(--s1)', borderRadius: 'var(--r-pill)',
          border: '1px solid var(--b1)', overflow: 'hidden', marginLeft: 4,
        }}>
          <button
            onClick={() => switchCreatorMode('single')}
            style={{
              display: 'flex', alignItems: 'center', gap: 4,
              padding: '3px 8px', border: 'none', cursor: 'pointer',
              fontSize: '10.5px', fontFamily: 'inherit', fontWeight: 500,
              background: creatorMode === 'single' ? 'var(--pg)' : 'transparent',
              color: creatorMode === 'single' ? 'var(--pl)' : 'var(--mu)',
            }}
          >
            <Layout size={11} /> Single Page
          </button>
          <button
            onClick={() => switchCreatorMode('project')}
            style={{
              display: 'flex', alignItems: 'center', gap: 4,
              padding: '3px 8px', border: 'none', cursor: 'pointer',
              fontSize: '10.5px', fontFamily: 'inherit', fontWeight: 500,
              background: creatorMode === 'project' ? 'var(--pg)' : 'transparent',
              color: creatorMode === 'project' ? 'var(--pl)' : 'var(--mu)',
            }}
          >
            <FolderTree size={11} /> Project
          </button>
        </div>

        {creatorMode === 'project' && (
          <div style={{
            display: 'flex', background: 'var(--s1)', borderRadius: 'var(--r-pill)',
            border: '1px solid var(--b1)', overflow: 'hidden',
          }}>
            <button
              onClick={() => setProjectWorkflowMode('chat')}
              style={{
                display: 'flex', alignItems: 'center', gap: 4,
                padding: '3px 8px', border: 'none', cursor: 'pointer',
                fontSize: '10.5px', fontFamily: 'inherit', fontWeight: 500,
                background: projectWorkflowMode === 'chat' ? 'var(--pg)' : 'transparent',
                color: projectWorkflowMode === 'chat' ? 'var(--pl)' : 'var(--mu)',
              }}
            >
              <Send size={11} /> Chat
            </button>
            <button
              onClick={() => setProjectWorkflowMode('build')}
              style={{
                display: 'flex', alignItems: 'center', gap: 4,
                padding: '3px 8px', border: 'none', cursor: 'pointer',
                fontSize: '10.5px', fontFamily: 'inherit', fontWeight: 500,
                background: projectWorkflowMode === 'build' ? 'var(--pg)' : 'transparent',
                color: projectWorkflowMode === 'build' ? 'var(--pl)' : 'var(--mu)',
              }}
            >
              <Hammer size={11} /> Build
            </button>
          </div>
        )}

        {/* Design token preset */}
        <select
          value={designTokens ? TOKEN_PRESETS.find(p => p.tokens.colors.primary === designTokens.colors.primary && p.tokens.darkMode === designTokens.darkMode)?.id || 'custom' : ''}
          onChange={async (e) => {
            const preset = TOKEN_PRESETS.find(p => p.id === e.target.value);
            if (preset) {
              setDesignTokens(preset.tokens);
              await saveDesignTokens(preset.tokens);
              // Re-render preview with new tokens
              if (creatorMode === 'project' && projectFs) {
                updateProjectPreview(projectFs);
              } else if (currentHtml) {
                updatePreview(currentHtml);
              }
            } else if (e.target.value === '') {
              setDesignTokens(null);
            }
          }}
          style={{
            background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
            color: designTokens ? 'var(--tx)' : 'var(--mu)', fontSize: '9.5px', padding: '2px 4px',
            fontFamily: 'inherit', cursor: 'pointer', maxWidth: 80,
          }}
          title="Design theme"
        >
          <option value="">No theme</option>
          {TOKEN_PRESETS.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>

        {/* Device mode toggles (visible when we have content) */}
        {hasContent && viewMode === 'preview' && (
          <div style={{
            display: 'flex', background: 'var(--s1)', borderRadius: 'var(--r-pill)',
            border: '1px solid var(--b1)', overflow: 'hidden', marginLeft: 4,
          }}>
            {deviceTabs.map(d => (
              <button
                key={d.mode}
                onClick={() => setDeviceMode(d.mode)}
                title={d.label}
                style={{
                  display: 'flex', alignItems: 'center', gap: 3,
                  padding: '3px 8px', border: 'none', cursor: 'pointer',
                  fontSize: '10.5px', fontFamily: 'inherit', fontWeight: 500,
                  background: deviceMode === d.mode ? 'var(--pg)' : 'transparent',
                  color: deviceMode === d.mode ? 'var(--pl)' : 'var(--mu)',
                  transition: 'all 0.15s ease',
                }}
              >
                {d.icon}
              </button>
            ))}
          </div>
        )}

        {/* Select element mode */}
        {hasContent && viewMode === 'preview' && (
          <button
            onClick={toggleSelectMode}
            title={selectMode ? 'Exit selection mode' : 'Select element to edit'}
            style={{
              background: selectMode ? 'rgba(59,130,246,0.15)' : 'var(--s2)',
              border: `1px solid ${selectMode ? 'rgba(59,130,246,0.4)' : 'var(--b1)'}`,
              borderRadius: 'var(--r-sm)', color: selectMode ? '#3b82f6' : 'var(--mu)',
              padding: '3px 8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4,
              fontSize: '10px', fontFamily: 'inherit', marginLeft: 4,
            }}
          >
            <MousePointer2 size={12} /> {selectMode ? 'Selecting...' : 'Select'}
          </button>
        )}

        {/* Pop out preview */}
        {activeBundleHtml && viewMode === 'preview' && (
          <button
            onClick={detachPreview}
            title={detached ? 'Preview detached — click to focus' : 'Pop out preview'}
            style={{
              background: detached ? 'rgba(34,197,94,0.15)' : 'var(--s2)',
              border: `1px solid ${detached ? 'rgba(34,197,94,0.4)' : 'var(--b1)'}`,
              borderRadius: 'var(--r-sm)', color: detached ? '#22c55e' : 'var(--mu)',
              padding: '3px 8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4,
              fontSize: '10px', fontFamily: 'inherit',
            }}
          >
            <ExternalLink size={12} /> {detached ? 'Detached' : 'Pop Out'}
          </button>
        )}

        {/* Theme toggle */}
        {creatorMode === 'single' && hasHtml && viewMode === 'preview' && (
          <button
            onClick={() => setThemeOpen(!themeOpen)}
            title="Theme"
            style={{
              background: themeOpen ? 'rgba(245,158,11,0.15)' : 'var(--s2)',
              border: `1px solid ${themeOpen ? 'rgba(245,158,11,0.4)' : 'var(--b1)'}`,
              borderRadius: 'var(--r-sm)', color: themeOpen ? '#f59e0b' : 'var(--mu)',
              padding: '3px 8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4,
              fontSize: '10px', fontFamily: 'inherit',
            }}
          >
            <Palette size={12} /> Theme
          </button>
        )}

        {hasContent && (
          <button
            onClick={() => setGalleryOpen(true)}
            style={{
              ...btnBase,
              padding: '4px 10px',
              fontSize: '10.5px',
              marginLeft: 4,
            }}
          >
            <FolderOpen size={12} /> Gallery
            {galleryCount > 0 && (
              <span style={{
                minWidth: 16,
                height: 16,
                padding: '0 4px',
                borderRadius: 999,
                background: 'rgba(124,58,237,0.18)',
                color: 'var(--pl)',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '9px',
                fontWeight: 700,
              }}>
                {galleryCount}
              </span>
            )}
          </button>
        )}

        {/* View mode: Preview / Code */}
        {hasContent && (
          <div style={{
            display: 'flex', background: 'var(--s1)', borderRadius: 'var(--r-pill)',
            border: '1px solid var(--b1)', overflow: 'hidden', marginLeft: 4,
          }}>
            <button
              onClick={() => setViewMode('preview')}
              style={{
                display: 'flex', alignItems: 'center', gap: 3,
                padding: '3px 8px', border: 'none', cursor: 'pointer',
                fontSize: '10.5px', fontFamily: 'inherit', fontWeight: 500,
                background: viewMode === 'preview' ? 'var(--pg)' : 'transparent',
                color: viewMode === 'preview' ? 'var(--pl)' : 'var(--mu)',
                transition: 'all 0.15s ease',
              }}
            >
              <Eye size={11} /> Preview
            </button>
            <button
              onClick={() => setViewMode('code')}
              style={{
                display: 'flex', alignItems: 'center', gap: 3,
                padding: '3px 8px', border: 'none', cursor: 'pointer',
                fontSize: '10.5px', fontFamily: 'inherit', fontWeight: 500,
                background: viewMode === 'code' ? 'var(--pg)' : 'transparent',
                color: viewMode === 'code' ? 'var(--pl)' : 'var(--mu)',
                transition: 'all 0.15s ease',
              }}
            >
              <Code2 size={11} /> Code
            </button>
          </div>
        )}

        <div style={{ flex: 1 }} />

        {fullscreen && (
          <button onClick={() => setFullscreen(false)} style={{ ...btnBase, padding: '4px 8px' }}>
            <Minimize2 size={13} /> Exit
          </button>
        )}
        <ModelPill featureKey="webcreator" />
      </div>

      {/* ═══ Status bar ═══ */}
      {(status || iframeError || isAutoFixing) && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '5px 12px', fontSize: '11px', flexShrink: 0,
          borderBottom: '1px solid var(--b1)',
          background: isAutoFixing
            ? 'rgba(59,130,246,0.08)'
            : iframeError
              ? 'rgba(239,68,68,0.06)'
              : status === 'Copied!' || status === 'Sent to CLI feed'
                ? 'rgba(16,185,129,0.06)'
                : 'rgba(124,58,237,0.04)',
        }}>
          <span style={{
            color: isAutoFixing
              ? '#60a5fa'
              : iframeError
                ? 'var(--rd)'
                : status === 'Copied!' || status === 'Sent to CLI feed'
                  ? 'var(--gr)'
                  : 'var(--pl)',
            flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {isAutoFixing
              ? `Auto-fixing preview error (${Math.min(autoFixAttempts + 1, MAX_AUTO_FIX_ATTEMPTS)}/${MAX_AUTO_FIX_ATTEMPTS})`
              : iframeError
                ? `Error: ${iframeError}`
                : status}
          </span>
          {iframeError && creatorMode === 'single' && currentHtml && (
            <button
              onClick={() => void runPreviewAutoFix(iframeError, 0, true)}
              disabled={isAutoFixing}
              style={{
                ...btnBase,
                padding: '3px 10px',
                fontSize: '10.5px',
                background: 'rgba(239,68,68,0.1)',
                borderColor: 'rgba(239,68,68,0.2)',
                color: 'var(--rd)',
                opacity: isAutoFixing ? 0.65 : 1,
                cursor: isAutoFixing ? 'wait' : 'pointer',
              }}
            >
              <Sparkles size={11} /> {isAutoFixing ? 'Fixing...' : 'Fix with AI'}
            </button>
          )}
        </div>
      )}

      {/* ═══ Selected element toolbar ═══ */}
      {selectedElement && (
        <div style={{
          padding: '6px 10px', background: 'rgba(59,130,246,0.08)', borderBottom: '1px solid rgba(59,130,246,0.2)',
          display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', flexShrink: 0,
        }}>
          <span style={{ fontSize: '10px', color: '#3b82f6', fontWeight: 600 }}>
            &lt;{selectedElement.tagName.toLowerCase()}&gt;
          </span>
          {selectedElement.text && (
            <span style={{ fontSize: '10px', color: 'var(--mu)', maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              "{selectedElement.text}"
            </span>
          )}
          <span style={{ flex: 1 }} />
          <button
            onClick={() => {
              const desc = selectedElement.text ? `"${selectedElement.text.slice(0, 40)}"` : `<${selectedElement.tagName.toLowerCase()}>`;
              setInput(`Edit the ${selectedElement.tagName.toLowerCase()} element ${desc}: `);
              setSelectedElement(null);
              inputRef.current?.focus();
            }}
            style={{
              background: '#3b82f6', border: 'none', borderRadius: 'var(--r-sm)',
              color: 'white', padding: '3px 10px', fontSize: '10px', cursor: 'pointer',
              fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: 4,
            }}
          >
            <Pencil size={10} /> Edit with AI
          </button>
          <button
            onClick={() => setSelectedElement(null)}
            style={{
              background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
              color: 'var(--mu)', padding: '3px 8px', fontSize: '10px', cursor: 'pointer', fontFamily: 'inherit',
            }}
          >
            Cancel
          </button>
        </div>
      )}

      {creatorMode === 'project' && projectWorkflowMode === 'build' && (
        <div style={{
          padding: '10px 12px',
          borderBottom: '1px solid var(--b1)',
          background: 'linear-gradient(180deg, rgba(124,58,237,0.06), rgba(14,165,233,0.03))',
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
          flexShrink: 0,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              padding: '4px 8px',
              borderRadius: 999,
              background: 'rgba(124,58,237,0.12)',
              color: 'var(--pl)',
              fontSize: '10px',
              fontWeight: 700,
              letterSpacing: '0.04em',
              textTransform: 'uppercase',
            }}>
              <Hammer size={11} />
              Agent Build Mode
            </div>
            <div style={{ fontSize: '11px', color: 'var(--mu)', flex: 1, minWidth: 180 }}>
              {buildPrompt
                ? buildPrompt
                : 'Plan a multi-file build, review the file list, then run the build with live progress.'}
            </div>
            {(buildStage === 'planned' || buildStage === 'completed' || buildStage === 'error') && buildPlanDrafts.length > 0 && (
              <button onClick={resetBuildWorkflow} style={{ ...btnBase, padding: '4px 10px', fontSize: '10.5px' }}>
                <RotateCcw size={11} /> Reset
              </button>
            )}
            {buildStage === 'completed' && (
              <button
                onClick={() => { setProjectWorkflowMode('chat'); }}
                style={{ ...btnBase, padding: '4px 10px', fontSize: '10.5px', background: 'rgba(16,185,129,0.12)', borderColor: 'rgba(16,185,129,0.3)', color: '#10b981' }}
              >
                <Pencil size={11} /> Refine in Chat
              </button>
            )}
          </div>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <span style={{ fontSize: '10px', color: 'var(--mu)' }}>Build engine:</span>
            <button
              onClick={() => setUseAgentBackend(false)}
              disabled={buildStage === 'building'}
              style={{
                ...btnBase, padding: '3px 8px', fontSize: '9.5px',
                background: !useAgentBackend ? 'rgba(124,58,237,0.15)' : 'transparent',
                color: !useAgentBackend ? 'var(--pl)' : 'var(--mu)',
                borderColor: !useAgentBackend ? 'rgba(124,58,237,0.3)' : 'var(--b1)',
                opacity: buildStage === 'building' ? 0.5 : 1,
              }}
            >
              <Zap size={10} /> Quick
            </button>
            <button
              onClick={() => setUseAgentBackend(true)}
              disabled={buildStage === 'building'}
              style={{
                ...btnBase, padding: '3px 8px', fontSize: '9.5px',
                background: useAgentBackend ? 'rgba(16,185,129,0.15)' : 'transparent',
                color: useAgentBackend ? '#10b981' : 'var(--mu)',
                borderColor: useAgentBackend ? 'rgba(16,185,129,0.3)' : 'var(--b1)',
                opacity: buildStage === 'building' ? 0.5 : 1,
              }}
            >
              <Server size={10} /> Agent
            </button>
          </div>

          {(buildStage === 'planning' || buildStage === 'planned' || buildStage === 'building' || buildStage === 'completed' || buildStage === 'error') && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <span style={{ fontSize: '11px', color: 'var(--tx)', fontWeight: 600 }}>
                  {buildStage === 'planning'
                    ? 'Planning files'
                    : buildStage === 'planned'
                      ? 'Plan ready'
                      : buildStage === 'building'
                        ? 'Building project'
                        : buildStage === 'completed'
                          ? 'Build complete'
                          : 'Build issue'}
                </span>
                <span style={{ fontSize: '10.5px', color: 'var(--mu)' }}>
                  {buildProgressMessage || 'Waiting for input'}
                </span>
                {buildCurrentFile && buildStage === 'building' && (
                  <span style={{
                    fontSize: '10px',
                    color: '#7dd3fc',
                    background: 'rgba(14,165,233,0.12)',
                    border: '1px solid rgba(125,211,252,0.18)',
                    borderRadius: 999,
                    padding: '2px 8px',
                  }}>
                    Building {buildCurrentFile}
                  </span>
                )}
              </div>

              <div style={{
                height: 8,
                borderRadius: 999,
                background: 'rgba(148,163,184,0.18)',
                overflow: 'hidden',
              }}>
                <div style={{
                  width: `${buildStage === 'planning' ? 18 : buildProgressRatio * 100}%`,
                  height: '100%',
                  borderRadius: 999,
                  background: buildStage === 'error'
                    ? 'linear-gradient(90deg, #ef4444, #f97316)'
                    : 'linear-gradient(90deg, #7c3aed, #0ea5e9)',
                  transition: 'width 0.2s ease',
                }} />
              </div>

              {buildPlanError && (
                <div style={{ fontSize: '10.5px', color: 'var(--rd)' }}>
                  {buildPlanError}
                </div>
              )}

              {buildPlanDrafts.length > 0 && (
                <div style={{
                  display: 'grid',
                  gap: 8,
                  maxHeight: 220,
                  overflow: 'auto',
                  paddingRight: 2,
                }}>
                  {buildPlanDrafts.map((item, index) => (
                    <div
                      key={item.id}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: '26px minmax(0, 1.1fr) minmax(0, 1.5fr) auto',
                        gap: 8,
                        alignItems: 'center',
                        padding: '8px 10px',
                        borderRadius: 'var(--r-md)',
                        border: '1px solid rgba(148,163,184,0.18)',
                        background: buildCompletedFiles.includes(item.path)
                          ? 'rgba(16,185,129,0.08)'
                          : 'rgba(15,23,42,0.18)',
                      }}
                    >
                      <span style={{ fontSize: '10.5px', color: 'var(--mu)', textAlign: 'center' }}>{index + 1}</span>
                      <input
                        value={item.path}
                        onChange={(event) => updateBuildPlanDraft(item.id, { path: event.target.value })}
                        disabled={buildStage === 'building'}
                        style={{
                          background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
                          color: 'var(--tx)', fontSize: '11px', padding: '7px 8px', outline: 'none', fontFamily: 'inherit',
                        }}
                      />
                      <input
                        value={item.purpose}
                        onChange={(event) => updateBuildPlanDraft(item.id, { purpose: event.target.value })}
                        disabled={buildStage === 'building'}
                        style={{
                          background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
                          color: 'var(--tx)', fontSize: '11px', padding: '7px 8px', outline: 'none', fontFamily: 'inherit',
                        }}
                      />
                      <div style={{ display: 'flex', gap: 4 }}>
                        <button
                          onClick={() => moveBuildPlanDraft(item.id, -1)}
                          disabled={index === 0 || buildStage === 'building'}
                          style={{ ...btnBase, padding: '5px 6px', opacity: index === 0 || buildStage === 'building' ? 0.45 : 1 }}
                        >
                          <ChevronUp size={11} />
                        </button>
                        <button
                          onClick={() => moveBuildPlanDraft(item.id, 1)}
                          disabled={index === buildPlanDrafts.length - 1 || buildStage === 'building'}
                          style={{ ...btnBase, padding: '5px 6px', opacity: index === buildPlanDrafts.length - 1 || buildStage === 'building' ? 0.45 : 1 }}
                        >
                          <ChevronDown size={11} />
                        </button>
                        <button
                          onClick={() => removeBuildPlanDraft(item.id)}
                          disabled={buildStage === 'building'}
                          style={{
                            ...btnBase,
                            padding: '5px 6px',
                            color: 'var(--rd)',
                            borderColor: 'rgba(239,68,68,0.2)',
                            background: 'rgba(239,68,68,0.08)',
                            opacity: buildStage === 'building' ? 0.45 : 1,
                          }}
                        >
                          <X size={11} />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <button onClick={addBuildPlanDraft} disabled={buildStage === 'building'} style={{ ...btnBase, padding: '6px 10px', fontSize: '10.5px', opacity: buildStage === 'building' ? 0.5 : 1 }}>
                    <Plus size={11} /> Add File
                  </button>
                  <button
                    onClick={() => {
                      const prompt = input.trim() || buildPrompt;
                      if (prompt) void requestBuildPlan(prompt);
                    }}
                    disabled={!(input.trim() || buildPrompt) || buildStage === 'building'}
                    style={{ ...btnBase, padding: '6px 10px', fontSize: '10.5px', opacity: !(input.trim() || buildPrompt) || buildStage === 'building' ? 0.5 : 1 }}
                  >
                    <RotateCcw size={11} /> Re-plan From Prompt
                  </button>
                </div>
                {buildStage === 'building' && useAgentBackend ? (
                  <button
                    onClick={() => void agentBuildActions.cancelBuild()}
                    style={{
                      display: 'inline-flex', alignItems: 'center', gap: 6,
                      background: 'rgba(239,68,68,0.12)', color: '#ef4444',
                      border: '1px solid rgba(239,68,68,0.3)', borderRadius: 'var(--r-md)',
                      padding: '7px 12px', cursor: 'pointer', fontSize: '11px', fontWeight: 700, fontFamily: 'inherit',
                    }}
                  >
                    <Square size={11} /> Stop Agent Build
                  </button>
                ) : (
                  <button
                    onClick={() => void runApprovedBuild()}
                    disabled={buildPlanDrafts.length === 0 || buildStage === 'building'}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 6,
                      background: 'var(--p)',
                      color: '#fff',
                      border: 'none',
                      borderRadius: 'var(--r-md)',
                      padding: '7px 12px',
                      cursor: buildPlanDrafts.length === 0 || buildStage === 'building' ? 'not-allowed' : 'pointer',
                      fontSize: '11px',
                      fontWeight: 700,
                      fontFamily: 'inherit',
                      opacity: buildPlanDrafts.length === 0 || buildStage === 'building' ? 0.55 : 1,
                    }}
                  >
                    <Play size={11} /> {buildStage === 'completed' ? 'Run Again' : 'Approve and Build'}
                  </button>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {/* ═══ Theme panel ═══ */}
      {creatorMode === 'single' && themeOpen && currentHtml && (
        <div style={{
          padding: '8px 10px', borderBottom: '1px solid var(--b1)', background: 'var(--s1)',
        }}>
          <div style={{ fontSize: '10px', color: 'var(--mu)', fontWeight: 600, marginBottom: 6 }}>Theme Colors</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {Object.entries(themeColors).map(([key, value]) => (
              <label key={key} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '10px', color: 'var(--mu)' }}>
                <input
                  type="color"
                  value={value}
                  onChange={e => {
                    const newColors = { ...themeColors, [key]: e.target.value };
                    setThemeColors(newColors);
                    applyTheme(newColors);
                  }}
                  style={{ width: 20, height: 20, border: '1px solid var(--b1)', borderRadius: 3, padding: 0, cursor: 'pointer' }}
                />
                {key}
              </label>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
            <button
              onClick={() => {
                const darkColors = { primary: '#3b82f6', secondary: '#8b5cf6', accent: '#f59e0b', background: '#0f172a', text: '#f1f5f9' };
                setThemeColors(darkColors);
                applyTheme(darkColors);
              }}
              style={{ fontSize: '9px', padding: '2px 8px', background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', borderRadius: 'var(--r-sm)', cursor: 'pointer', fontFamily: 'inherit' }}
            >
              Dark
            </button>
            <button
              onClick={() => {
                const lightColors = { primary: '#3b82f6', secondary: '#6366f1', accent: '#f59e0b', background: '#ffffff', text: '#111827' };
                setThemeColors(lightColors);
                applyTheme(lightColors);
              }}
              style={{ fontSize: '9px', padding: '2px 8px', background: '#f8fafc', color: '#475569', border: '1px solid #e2e8f0', borderRadius: 'var(--r-sm)', cursor: 'pointer', fontFamily: 'inherit' }}
            >
              Light
            </button>
            <button
              onClick={() => {
                setInput('Suggest a harmonious color palette for this design and update the CSS accordingly');
              }}
              style={{ fontSize: '9px', padding: '2px 8px', background: 'var(--s2)', color: 'var(--mu)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)', cursor: 'pointer', fontFamily: 'inherit' }}
            >
              AI Suggest
            </button>
          </div>
        </div>
      )}

      {/* ═══ Main content: chat + preview ═══ */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>

        {/* ── Empty state with templates ── */}
        {isEmptyState && (
          <div style={{
            flex: 1, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 20, padding: 24,
            overflow: 'auto',
          }}>
            <div style={{
              width: 56, height: 56, borderRadius: '50%',
              background: 'var(--pg)', border: '1px solid rgba(124,58,237,0.15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Layout size={24} style={{ color: 'var(--pl)' }} />
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--tx)', marginBottom: 4 }}>
                {creatorMode === 'project' ? 'Build multi-file projects with AI' : 'Build websites with AI'}
              </div>
              <div style={{ fontSize: '11.5px', color: 'var(--mu)', maxWidth: 280, lineHeight: 1.5 }}>
                {creatorMode === 'project'
                  ? projectWorkflowMode === 'build'
                    ? 'Describe the project, let Aura draft the file plan, then approve the build and watch files land in real time.'
                    : 'Start from a real file tree, then let Aura create and update multiple files for you.'
                  : 'Describe what you want, iterate with natural language. Pick a template or start from scratch.'}
              </div>
            </div>

            {/* Template grid */}
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))',
              gap: 8, width: '100%', maxWidth: 400,
            }}>
              {(creatorMode === 'project' ? PROJECT_TEMPLATES : TEMPLATES).map(t => (
                <button
                  key={t.label}
                  onClick={() => creatorMode === 'project' ? handleProjectTemplate(t as ProjectTemplate) : handleTemplate(t as Template)}
                  onMouseEnter={() => setHoveredBtn(`tmpl-${t.label}`)}
                  onMouseLeave={() => setHoveredBtn(null)}
                  style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center',
                    gap: 6, padding: '14px 10px',
                    background: hoveredBtn === `tmpl-${t.label}` ? 'var(--s2)' : 'var(--s1)',
                    border: '1px solid var(--b1)',
                    borderColor: hoveredBtn === `tmpl-${t.label}` ? `${t.color}40` : 'var(--b1)',
                    borderRadius: 'var(--r-md)',
                    cursor: 'pointer', transition: 'all 0.2s ease',
                    fontFamily: 'inherit',
                  }}
                >
                  <div style={{
                    width: 36, height: 36, borderRadius: '50%',
                    background: `${t.color}15`,
                    border: `1px solid ${t.color}25`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: t.color,
                    transition: 'all 0.2s ease',
                    transform: hoveredBtn === `tmpl-${t.label}` ? 'scale(1.1)' : 'scale(1)',
                  }}>
                    {t.icon}
                  </div>
                  <span style={{
                    fontSize: '10.5px', fontWeight: 500,
                    color: hoveredBtn === `tmpl-${t.label}` ? 'var(--tx)' : 'var(--mu)',
                    transition: 'color 0.15s ease',
                  }}>
                    {t.label}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ── Content area (chat messages + preview) ── */}
        {!isEmptyState && (
          <>
            {/* Chat messages (collapsible) */}
            {chatOpen && messages.length > 0 && (
              <div
                ref={chatScrollRef}
                style={{
                  maxHeight: hasContent ? 160 : '40%', overflow: 'auto', flexShrink: 0,
                  borderBottom: '1px solid var(--b1)',
                  padding: '8px 12px',
                }}
              >
                {messages.map(m => (
                  <div
                    key={m.id}
                    style={{
                      display: 'flex', gap: 8, marginBottom: 8,
                      alignItems: 'flex-start',
                    }}
                  >
                    <div style={{
                      width: 22, height: 22, borderRadius: '50%', flexShrink: 0,
                      background: m.role === 'user' ? 'var(--pg)' : 'rgba(16,185,129,0.08)',
                      border: `1px solid ${m.role === 'user' ? 'rgba(124,58,237,0.2)' : 'rgba(16,185,129,0.15)'}`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      marginTop: 1,
                    }}>
                      {m.role === 'user'
                        ? <User size={11} style={{ color: 'var(--pl)' }} />
                        : <Bot size={11} style={{ color: '#10b981' }} />
                      }
                    </div>
                    <div style={{
                      fontSize: '11.5px', lineHeight: 1.5,
                      color: m.role === 'user' ? 'var(--tx)' : 'var(--mu)',
                      flex: 1, minWidth: 0,
                      wordBreak: 'break-word',
                    }}>
                      {m.text.length > 200 ? m.text.slice(0, 200) + '...' : m.text}
                    </div>
                  </div>
                ))}
                {loading && (
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
                    <div style={{
                      width: 22, height: 22, borderRadius: '50%', flexShrink: 0,
                      background: 'rgba(16,185,129,0.08)',
                      border: '1px solid rgba(16,185,129,0.15)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>
                      <Bot size={11} style={{ color: '#10b981' }} />
                    </div>
                    <div className="aura-thinking" style={{ transform: 'scale(0.7)', transformOrigin: 'left center' }}>
                      <span /><span /><span />
                    </div>
                    <span style={{
                      fontSize: '10.5px', color: 'var(--pl)', fontWeight: 500,
                      marginLeft: 4, animation: 'streamPulse 1.5s ease-in-out infinite',
                    }}>
                      Generating...
                    </span>
                    <style>{`@keyframes streamPulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }`}</style>
                  </div>
                )}
              </div>
            )}

            {/* Preview / Code area */}
            <div style={{ flex: 1, position: 'relative', overflow: 'hidden', display: 'flex', minHeight: 0 }}>
              {creatorMode === 'project' && hasProject && (
                <div style={{
                  width: 220,
                  borderRight: '1px solid var(--b1)',
                  background: 'rgba(255,255,255,0.02)',
                  flexShrink: 0,
                  minHeight: 0,
                }}>
                  <FileTree
                    files={projectFiles}
                    directories={projectDirectories}
                    activeFile={activeProjectFile}
                    onFileSelect={setActiveProjectFile}
                    onFileCreate={(path) => {
                      openProjectDialog({
                        type: 'create-file',
                        title: 'Create File',
                        parentPath: path,
                        initialValue: path ? `${path}/new-file.html` : 'new-file.html',
                      });
                    }}
                    onFolderCreate={(path) => {
                      openProjectDialog({
                        type: 'create-folder',
                        title: 'Create Folder',
                        parentPath: path,
                        initialValue: path ? `${path}/new-folder` : 'new-folder',
                      });
                    }}
                    onFileRename={(oldPath) => {
                      openProjectDialog({
                        type: 'rename-file',
                        title: 'Rename File',
                        parentPath: getParentDirectory(oldPath),
                        path: oldPath,
                        initialValue: getBaseName(oldPath),
                      });
                    }}
                    onFileDelete={deleteProjectFile}
                  />
                </div>
              )}
              <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
              {/* Preview iframe */}
              {hasContent && viewMode === 'preview' && (
                detached ? (
                  <div style={{
                    flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
                    justifyContent: 'center', gap: 8, color: 'var(--mu)',
                  }}>
                    <ExternalLink size={24} />
                    <div style={{ fontSize: '12px' }}>Preview detached to separate window</div>
                    <button
                      onClick={() => {
                        if (detachedWindowRef.current && !detachedWindowRef.current.closed) {
                          detachedWindowRef.current.close();
                        }
                        setDetached(false);
                        detachedWindowRef.current = null;
                      }}
                      style={{
                        background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
                        color: 'var(--mu)', padding: '4px 12px', fontSize: '11px', cursor: 'pointer', fontFamily: 'inherit',
                      }}
                    >
                      Bring Back
                    </button>
                  </div>
                ) : (
                  <div style={{
                    width: '100%', height: '100%',
                    display: 'flex', justifyContent: 'center',
                    background: deviceMode !== 'desktop' ? 'var(--s1)' : 'transparent',
                    overflow: 'hidden',
                  }}>
                    <div style={{
                      width: DEVICE_WIDTHS[deviceMode],
                      maxWidth: '100%',
                      height: '100%',
                      position: 'relative',
                      transition: 'width 0.3s ease',
                      ...(deviceMode !== 'desktop' ? {
                        border: '1px solid var(--b1)',
                        borderRadius: '8px',
                        overflow: 'hidden',
                        margin: '8px 0',
                        boxShadow: '0 4px 24px rgba(0,0,0,0.3)',
                      } : {}),
                    }}>
                      <iframe
                        ref={iframeRef}
                        sandbox="allow-scripts allow-same-origin"
                        style={{
                          width: '100%', height: '100%', border: 'none',
                          background: '#fff',
                        }}
                      />
                    </div>
                    {/* WebContainer status bar */}
                    {creatorMode === 'project' && wcEnabled && (
                      <div style={{
                        display: 'flex', alignItems: 'center', gap: 6,
                        padding: '4px 10px', background: '#0d1117',
                        borderTop: '1px solid #30363d', flexShrink: 0,
                      }}>
                        <div style={{
                          width: 7, height: 7, borderRadius: '50%',
                          background: webContainer.status === 'running' ? '#3fb950'
                            : webContainer.status === 'error' ? '#f85149'
                            : webContainer.status === 'installing' || webContainer.status === 'booting' ? '#d29922'
                            : '#484f58',
                        }} />
                        <span style={{ fontSize: '9.5px', color: '#8b949e' }}>
                          {webContainer.status === 'running' ? 'Dev server running'
                            : webContainer.status === 'installing' ? 'Installing packages...'
                            : webContainer.status === 'booting' ? 'Booting WebContainer...'
                            : webContainer.status === 'ready' ? 'WebContainer ready'
                            : webContainer.status === 'error' ? 'WebContainer error'
                            : 'WebContainer idle'}
                        </span>
                        <span style={{ flex: 1 }} />
                        {webContainer.status === 'idle' || webContainer.status === 'ready' ? (
                          <button
                            onClick={() => webContainer.fullSetup(projectFilesMap)}
                            style={{
                              background: 'rgba(16,185,129,0.12)', border: '1px solid rgba(16,185,129,0.3)',
                              borderRadius: 'var(--r-sm)', color: '#10b981', padding: '2px 8px',
                              cursor: 'pointer', fontSize: '9px', fontFamily: 'inherit',
                            }}
                          >
                            {webContainer.status === 'ready' ? 'Start Server' : 'Boot & Run'}
                          </button>
                        ) : webContainer.status === 'running' ? (
                          <button
                            onClick={() => webContainer.teardown()}
                            style={{
                              background: 'rgba(248,81,73,0.12)', border: '1px solid rgba(248,81,73,0.3)',
                              borderRadius: 'var(--r-sm)', color: '#f85149', padding: '2px 8px',
                              cursor: 'pointer', fontSize: '9px', fontFamily: 'inherit',
                            }}
                          >
                            Stop
                          </button>
                        ) : null}
                        <button
                          onClick={() => setShowTerminal(!showTerminal)}
                          style={{
                            background: 'none', border: 'none', color: '#8b949e',
                            cursor: 'pointer', fontSize: '9px', fontFamily: 'inherit', padding: '2px 4px',
                          }}
                        >
                          {showTerminal ? 'Hide Terminal' : 'Terminal'}
                          {webContainer.terminalLines.filter(l => l.type === 'stderr').length > 0 && (
                            <span style={{ color: '#f85149', marginLeft: 4 }}>!</span>
                          )}
                        </button>
                      </div>
                    )}
                    {/* Terminal output */}
                    {showTerminal && wcEnabled && (
                      <TerminalOutput
                        lines={webContainer.terminalLines}
                        onClear={webContainer.clearTerminal}
                        title="WebContainer"
                        defaultOpen
                        collapsible={false}
                      />
                    )}
                  </div>
                )
              )}

              {/* Code view */}
              {hasContent && viewMode === 'code' && (
                <div style={{ width: '100%', height: '100%', background: '#0d0d14', display: 'flex', flexDirection: 'column' }}>
                  <div style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '6px 12px',
                    background: 'rgba(13,13,20,0.95)', backdropFilter: 'blur(8px)',
                    borderBottom: '1px solid rgba(255,255,255,0.04)',
                    flexShrink: 0,
                  }}>
                    <span style={{
                      fontSize: '9.5px', fontWeight: 600, letterSpacing: '0.06em',
                      textTransform: 'uppercase', color: '#a78bfa',
                      background: 'rgba(167,139,250,0.1)', padding: '2px 8px', borderRadius: 3,
                    }}>
                      {creatorMode === 'project' ? activeProjectLanguage : 'HTML'}
                    </span>
                    <span style={{ fontSize: '9.5px', color: 'rgba(255,255,255,0.3)', fontVariantNumeric: 'tabular-nums' }}>
                      {(creatorMode === 'project' ? activeProjectContent : currentHtml).length.toLocaleString()} chars
                    </span>
                  </div>
                  <div style={{ flex: 1, minHeight: 0 }}>
                    <CodeEditor
                      code={creatorMode === 'project' ? activeProjectContent : currentHtml}
                      diagnostics={editorDiagnostics}
                      language={creatorMode === 'project' ? activeProjectLanguage as any : 'html'}
                      onChange={handleHtmlEditorChange}
                      readOnly={loading}
                    />
                  </div>
                </div>
              )}

              {/* Loading overlay on preview */}
              {loading && hasContent && viewMode === 'preview' && (
                <div style={{
                  position: 'absolute', inset: 0, display: 'flex',
                  flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12,
                  background: 'rgba(3,3,3,0.5)', backdropFilter: 'blur(4px)',
                  zIndex: 5,
                }}>
                  <div className="aura-thinking">
                    <span /><span /><span />
                  </div>
                  <span style={{ fontSize: '12px', color: 'var(--pl)', fontWeight: 500 }}>
                    {creatorMode === 'project'
                      ? projectWorkflowMode === 'build' && buildStage === 'building'
                        ? 'Building project...'
                        : 'Updating project...'
                      : 'Updating page...'}
                  </span>
                </div>
              )}

              {/* Loading state when no html yet */}
              {loading && !hasContent && (
                <div style={{
                  position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'center', gap: 12,
                }}>
                  <div className="aura-thinking">
                    <span /><span /><span />
                  </div>
                  <span style={{ fontSize: '12px', color: 'var(--pl)', fontWeight: 500 }}>
                    {creatorMode === 'project'
                      ? projectWorkflowMode === 'build'
                        ? buildStage === 'planning'
                          ? 'Planning your project...'
                          : 'Creating your project...'
                        : 'Creating your project...'
                      : 'Creating your website...'}
                  </span>
                </div>
              )}
            </div>
            </div>
          </>
        )}
      </div>

      {/* ═══ Chat input (always visible when not in empty state, or always) ═══ */}
      <div style={{
        padding: '8px 12px', flexShrink: 0,
        borderTop: '1px solid var(--b1)',
        display: 'flex', gap: 8, alignItems: 'flex-end',
      }}>
        {/* Quick actions when we have content */}
        {hasContent && (
          <div style={{ display: 'flex', gap: 4, flexShrink: 0, alignItems: 'center' }}>
            <button
              onClick={clearConversation}
              onMouseEnter={() => setHoveredBtn('clear')}
              onMouseLeave={() => setHoveredBtn(null)}
              title="New page"
              style={{
                ...btnBase, padding: '6px 7px',
                ...(hoveredBtn === 'clear' ? { background: 'rgba(239,68,68,0.1)', borderColor: 'rgba(239,68,68,0.2)', color: 'var(--rd)' } : {}),
              }}
            >
              <Trash2 size={13} />
            </button>
          </div>
        )}

        <textarea
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={
            hasContent
              ? creatorMode === 'project'
                ? projectWorkflowMode === 'build'
                  ? 'Describe the next build goal... "Build a portfolio with blog and contact form"'
                  : 'Describe project changes... "Add a pricing section", "split styles into another file"'
                : 'Describe changes... "Make the header sticky", "Add a contact form"'
              : creatorMode === 'project'
                ? projectWorkflowMode === 'build'
                  ? 'Describe the project you want Aura to build...'
                  : 'Describe the project you want Aura to build...'
                : 'Describe the website you want...'
          }
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
          rows={1}
          style={{
            flex: 1, background: 'var(--s2)', border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)', color: 'var(--tx)', fontSize: '12px',
            padding: '8px 10px', resize: 'none', minHeight: 36, maxHeight: 80,
            outline: 'none', fontFamily: 'inherit', lineHeight: 1.5,
            transition: 'border-color 0.2s ease',
          }}
          onFocus={e => { e.currentTarget.style.borderColor = 'rgba(124,58,237,0.35)'; }}
          onBlur={e => { e.currentTarget.style.borderColor = 'var(--b1)'; }}
          onInput={e => {
            const el = e.currentTarget;
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 80) + 'px';
          }}
        />
        {loading ? (
          <button
            onClick={() => {
              if (abortRef.current) abortRef.current.abort();
              abortRef.current = null;
              setLoading(false);
              setStatus('Cancelled');
              setMessages(prev => [...prev, {
                id: crypto.randomUUID(),
                role: 'ai' as const,
                text: 'Generation stopped.',
                timestamp: Date.now(),
              }]);
            }}
            className="stop-stream-btn"
            aria-label="Stop generating"
            style={{
              alignSelf: 'stretch', minWidth: 40, minHeight: 36,
              padding: '0 12px',
            }}
          >
            <Square size={10} />
            <span>Stop</span>
          </button>
        ) : (
          <button
            onClick={() => sendMessage()}
            disabled={!input.trim()}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: 'var(--p)',
              border: 'none', borderRadius: 'var(--r-md)', color: '#fff',
              padding: '0 14px', cursor: !input.trim() ? 'not-allowed' : 'pointer',
              fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
              alignSelf: 'stretch', minWidth: 40, minHeight: 36,
              opacity: !input.trim() ? 0.5 : 1,
              transition: 'all 0.2s ease',
              boxShadow: '0 2px 10px rgba(124,58,237,0.3)',
            }}
          >
            {creatorMode === 'project' && projectWorkflowMode === 'build' ? <Hammer size={14} /> : <Send size={14} />}
          </button>
        )}
      </div>

      {projectDialog && (
        <OverlayModal
          onClose={() => { setProjectDialog(null); setProjectDialogValue(''); }}
          title={projectDialog.title}
          icon={<FolderTree size={16} style={{ color: 'var(--pl)' }} />}
          zIndex={10009}
        >
          <div style={{ fontSize: '11px', color: 'var(--mu)', lineHeight: 1.5 }}>
            {projectDialog.type === 'create-file'
              ? 'Add a new file to the project.'
              : projectDialog.type === 'create-folder'
                ? 'Create a folder to organize your project files.'
                : 'Rename the selected file.'}
          </div>
          <input
            value={projectDialogValue}
            onChange={(event) => setProjectDialogValue(event.target.value)}
            autoFocus
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                submitProjectDialog();
              }
            }}
            style={{
              background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
              color: 'var(--tx)', fontSize: '12px', padding: '10px 12px', outline: 'none', fontFamily: 'inherit',
            }}
          />
          <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
            <button onClick={() => { setProjectDialog(null); setProjectDialogValue(''); }} style={{ ...btnBase, padding: '8px 12px' }}>
              Cancel
            </button>
            <button
              onClick={submitProjectDialog}
              style={{
                background: 'var(--p)', color: '#fff', border: 'none', borderRadius: 'var(--r-md)',
                padding: '8px 14px', cursor: 'pointer', fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
              }}
            >
              Save
            </button>
          </div>
        </OverlayModal>
      )}

      {saveDialogOpen && (
        <OverlayModal
          onClose={() => setSaveDialogOpen(false)}
          title={saveLabel}
          icon={<Save size={16} style={{ color: 'var(--pl)' }} />}
        >
            <div style={{ fontSize: '11px', color: 'var(--mu)', lineHeight: 1.5 }}>
              {creatorMode === 'project'
                ? 'Save this project snapshot so you can reload the full file tree later from the gallery.'
                : 'Save this page to reload it later from the gallery.'}
            </div>
            <input
              value={saveName}
              onChange={(event) => setSaveName(event.target.value)}
              placeholder={creatorMode === 'project' ? 'Project name' : 'Page name'}
              autoFocus
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault();
                  void (creatorMode === 'project' ? saveCurrentProject() : saveCurrentPage());
                }
              }}
              style={{
                background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
                color: 'var(--tx)', fontSize: '12px', padding: '10px 12px', outline: 'none', fontFamily: 'inherit',
              }}
            />
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button onClick={() => setSaveDialogOpen(false)} style={{ ...btnBase, padding: '8px 12px' }}>
                Cancel
              </button>
              <button
                onClick={() => void (creatorMode === 'project' ? saveCurrentProject() : saveCurrentPage())}
                style={{
                  background: 'var(--p)', color: '#fff', border: 'none', borderRadius: 'var(--r-md)',
                  padding: '8px 14px', cursor: 'pointer', fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
                }}
              >
                Save
              </button>
            </div>
        </OverlayModal>
      )}

      {renameDialogPage && (
        <OverlayModal
          onClose={() => { setRenameDialogPage(null); setRenameValue(''); }}
          title="Rename Page"
          icon={<Pencil size={16} style={{ color: 'var(--pl)' }} />}
          zIndex={10011}
        >
            <input
              value={renameValue}
              onChange={(event) => setRenameValue(event.target.value)}
              placeholder="Page name"
              autoFocus
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault();
                  void submitRenamePage();
                }
              }}
              style={{
                background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
                color: 'var(--tx)', fontSize: '12px', padding: '10px 12px', outline: 'none', fontFamily: 'inherit',
              }}
            />
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button onClick={() => { setRenameDialogPage(null); setRenameValue(''); }} style={{ ...btnBase, padding: '8px 12px' }}>
                Cancel
              </button>
              <button
                onClick={() => void submitRenamePage()}
                style={{
                  background: 'var(--p)', color: '#fff', border: 'none', borderRadius: 'var(--r-md)',
                  padding: '8px 14px', cursor: 'pointer', fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
                }}
              >
                Save Name
              </button>
            </div>
        </OverlayModal>
      )}

      {renameDialogProject && (
        <OverlayModal
          onClose={() => { setRenameDialogProject(null); setRenameValue(''); }}
          title="Rename Project"
          icon={<Pencil size={16} style={{ color: 'var(--pl)' }} />}
          zIndex={10011}
        >
            <input
              value={renameValue}
              onChange={(event) => setRenameValue(event.target.value)}
              placeholder="Project name"
              autoFocus
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault();
                  void submitRenameProject();
                }
              }}
              style={{
                background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
                color: 'var(--tx)', fontSize: '12px', padding: '10px 12px', outline: 'none', fontFamily: 'inherit',
              }}
            />
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button onClick={() => { setRenameDialogProject(null); setRenameValue(''); }} style={{ ...btnBase, padding: '8px 12px' }}>
                Cancel
              </button>
              <button
                onClick={() => void submitRenameProject()}
                style={{
                  background: 'var(--p)', color: '#fff', border: 'none', borderRadius: 'var(--r-md)',
                  padding: '8px 14px', cursor: 'pointer', fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
                }}
              >
                Save Name
              </button>
            </div>
        </OverlayModal>
      )}

      {deleteDialogPage && (
        <OverlayModal
          onClose={() => setDeleteDialogPage(null)}
          title="Delete Page"
          icon={<Trash2 size={16} style={{ color: '#fca5a5' }} />}
          zIndex={10012}
        >
            <div style={{ fontSize: '11px', color: 'var(--mu)', lineHeight: 1.5 }}>
              Permanently remove <span style={{ color: 'var(--tx)', fontWeight: 700 }}>{deleteDialogPage.name}</span> from the gallery?
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button onClick={() => setDeleteDialogPage(null)} style={{ ...btnBase, padding: '8px 12px' }}>
                Cancel
              </button>
              <button
                onClick={() => void confirmDeletePage()}
                style={{
                  background: '#dc2626', color: '#fff', border: 'none', borderRadius: 'var(--r-md)',
                  padding: '8px 14px', cursor: 'pointer', fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
                }}
              >
                Delete
              </button>
            </div>
        </OverlayModal>
      )}

      {deleteDialogProject && (
        <OverlayModal
          onClose={() => setDeleteDialogProject(null)}
          title="Delete Project"
          icon={<Trash2 size={16} style={{ color: '#fca5a5' }} />}
          zIndex={10012}
        >
            <div style={{ fontSize: '11px', color: 'var(--mu)', lineHeight: 1.5 }}>
              Permanently remove <span style={{ color: 'var(--tx)', fontWeight: 700 }}>{deleteDialogProject.name}</span> from the project gallery?
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button onClick={() => setDeleteDialogProject(null)} style={{ ...btnBase, padding: '8px 12px' }}>
                Cancel
              </button>
              <button
                onClick={() => void confirmDeleteProject()}
                style={{
                  background: '#dc2626', color: '#fff', border: 'none', borderRadius: 'var(--r-md)',
                  padding: '8px 14px', cursor: 'pointer', fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
                }}
              >
                Delete
              </button>
            </div>
        </OverlayModal>
      )}

      {pendingDiff && (
        <OverlayModal
          onClose={rejectPendingDiff}
          title="Review Page Changes"
          icon={<GitFork size={16} style={{ color: 'var(--pl)' }} />}
          zIndex={10013}
          contentStyle={{
            width: 'min(1120px, 100%)',
            height: 'min(82vh, 900px)',
            gap: 10,
          }}
        >
          <div style={{ fontSize: '11px', color: 'var(--mu)', lineHeight: 1.5 }}>
            Compare the current page with Aura&apos;s proposed update before it replaces the live preview.
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, fontSize: '10px', color: 'var(--mu)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            <div>Current</div>
            <div>AI Proposal</div>
          </div>
          <React.Suspense
            fallback={
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--mu)', fontSize: 12 }}>
                Loading diff...
              </div>
            }
          >
            <DiffEditor
              original={pendingDiff.original}
              modified={pendingDiff.modified}
              language="html"
            />
          </React.Suspense>
          <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
            <button onClick={rejectPendingDiff} style={{ ...btnBase, padding: '8px 12px' }}>
              Keep Current
            </button>
            <button
              onClick={acceptPendingDiff}
              style={{
                background: 'var(--p)', color: '#fff', border: 'none', borderRadius: 'var(--r-md)',
                padding: '8px 14px', cursor: 'pointer', fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
              }}
            >
              Apply Changes
            </button>
          </div>
        </OverlayModal>
      )}

      {galleryOpen && (
        <div
          onClick={() => setGalleryOpen(false)}
          style={{
            position: 'fixed', inset: 0, zIndex: 10000,
            background: 'rgba(0,0,0,0.48)',
            display: 'flex', justifyContent: 'flex-end',
          }}
        >
          <div
            onClick={(event) => event.stopPropagation()}
            style={{
              width: 'min(520px, 100vw)',
              height: '100%',
              background: '#10111a',
              borderLeft: '1px solid rgba(255,255,255,0.08)',
              boxShadow: '-20px 0 50px rgba(0,0,0,0.35)',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '16px 16px 12px', borderBottom: '1px solid var(--b1)' }}>
              <FolderOpen size={16} style={{ color: 'var(--pl)' }} />
              <div>
                <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--tx)' }}>
                  {creatorMode === 'project' ? 'Project Gallery' : 'Page Gallery'}
                </div>
                <div style={{ fontSize: '10.5px', color: 'var(--mu)' }}>
                  {creatorMode === 'project' ? `${projectGalleryItems.length} saved projects` : `${galleryItems.length} saved pages`}
                </div>
              </div>
              <div style={{ flex: 1 }} />
              <button onClick={() => setGalleryOpen(false)} style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer' }}>
                <X size={16} />
              </button>
            </div>

            <div style={{ padding: '12px 16px', display: 'flex', gap: 8, borderBottom: '1px solid var(--b1)' }}>
              <div style={{ position: 'relative', flex: 1 }}>
                <Search size={12} style={{ position: 'absolute', left: 10, top: 10, color: 'var(--mu)' }} />
                <input
                  value={galleryQuery}
                  onChange={(event) => setGalleryQuery(event.target.value)}
                  placeholder={creatorMode === 'project' ? 'Search saved projects' : 'Search saved pages'}
                  style={{
                    width: '100%',
                    background: 'var(--s2)',
                    border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-md)',
                    color: 'var(--tx)',
                    fontSize: '11.5px',
                    padding: '8px 10px 8px 30px',
                    outline: 'none',
                    fontFamily: 'inherit',
                  }}
                />
              </div>
            </div>

            <div style={{ flex: 1, overflow: 'auto', padding: 16, display: 'flex', flexDirection: 'column', gap: 10 }}>
              {creatorMode === 'project' ? filteredProjectGalleryItems.map((item) => (
                <div
                  key={item.id}
                  style={{
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: 14,
                    background: 'rgba(255,255,255,0.02)',
                    overflow: 'hidden',
                  }}
                >
                  <div style={{
                    height: 112,
                    borderBottom: '1px solid rgba(255,255,255,0.06)',
                    background: item.thumbnail
                      ? `center / cover no-repeat url("${item.thumbnail}")`
                      : 'linear-gradient(135deg, rgba(14,165,233,0.18), rgba(59,130,246,0.12))',
                    position: 'relative',
                    display: 'flex',
                    alignItems: 'flex-end',
                  }}>
                    <div style={{
                      position: 'absolute',
                      inset: 0,
                      background: 'linear-gradient(180deg, rgba(3,7,18,0.04), rgba(3,7,18,0.72))',
                    }} />
                    <div style={{ position: 'relative', zIndex: 1, width: '100%', padding: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{
                          fontSize: '9px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase',
                          color: '#bae6fd', background: 'rgba(15,23,42,0.48)', padding: '3px 7px', borderRadius: 999,
                        }}>
                          {item.framework}
                        </span>
                        <span style={{ fontSize: '10px', color: '#e2e8f0' }}>
                          {new Date(item.updatedAt).toLocaleDateString()}
                        </span>
                      </div>
                      <div style={{ fontSize: '13px', fontWeight: 700, color: '#f8fafc' }}>{item.name}</div>
                    </div>
                  </div>
                  <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
                    <div style={{ fontSize: '11px', color: 'var(--mu)', lineHeight: 1.5 }}>
                      {(item.prompt || `${item.framework} project with ${item.fileCount} files`).slice(0, 140)}
                      {(item.prompt || `${item.framework} project with ${item.fileCount} files`).length > 140 ? '...' : ''}
                    </div>
                    {!!item.tags?.length && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {item.tags.slice(0, 4).map((tag) => (
                          <span
                            key={tag}
                            style={{
                              fontSize: '9px',
                              color: '#7dd3fc',
                              background: 'rgba(14,165,233,0.12)',
                              border: '1px solid rgba(14,165,233,0.18)',
                              padding: '3px 6px',
                              borderRadius: 999,
                            }}
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: '10px', color: 'var(--mu)' }}>
                      <span>{item.fileCount} files</span>
                      <span>{item.entryPoint}</span>
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      <button onClick={() => loadSavedProject(item)} style={{ ...btnBase, padding: '6px 10px', fontSize: '10.5px' }}>
                        <FolderOpen size={12} /> Load
                      </button>
                      <button onClick={() => void forkSavedProject(item)} style={{ ...btnBase, padding: '6px 10px', fontSize: '10.5px' }}>
                        <GitFork size={12} /> Fork
                      </button>
                      <button onClick={() => openRenameProjectDialog(item)} style={{ ...btnBase, padding: '6px 10px', fontSize: '10.5px' }}>
                        <Pencil size={12} /> Rename
                      </button>
                      <button
                        onClick={() => openDeleteProjectDialog(item)}
                        style={{
                          ...btnBase, padding: '6px 10px', fontSize: '10.5px',
                          color: '#fca5a5', borderColor: 'rgba(239,68,68,0.18)',
                        }}
                      >
                        <Trash2 size={12} /> Delete
                      </button>
                    </div>
                  </div>
                </div>
              )) : filteredGalleryItems.map((item) => (
                <div
                  key={item.id}
                  style={{
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: 14,
                    background: 'rgba(255,255,255,0.02)',
                    overflow: 'hidden',
                  }}
                >
                  <div style={{
                    height: 112,
                    borderBottom: '1px solid rgba(255,255,255,0.06)',
                    background: item.thumbnail
                      ? `center / cover no-repeat url("${item.thumbnail}")`
                      : 'linear-gradient(135deg, rgba(124,58,237,0.18), rgba(59,130,246,0.12))',
                    position: 'relative',
                    display: 'flex',
                    alignItems: 'flex-end',
                  }}>
                    <div style={{
                      position: 'absolute',
                      inset: 0,
                      background: 'linear-gradient(180deg, rgba(3,7,18,0.04), rgba(3,7,18,0.72))',
                    }} />
                    <div style={{ position: 'relative', zIndex: 1, width: '100%', padding: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{
                          fontSize: '9px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase',
                          color: '#d8b4fe', background: 'rgba(15,23,42,0.48)', padding: '3px 7px', borderRadius: 999,
                        }}>
                          HTML
                        </span>
                        <span style={{ fontSize: '10px', color: '#e2e8f0' }}>
                          {new Date(item.updatedAt).toLocaleDateString()}
                        </span>
                      </div>
                      <div style={{ fontSize: '13px', fontWeight: 700, color: '#f8fafc' }}>{item.name}</div>
                    </div>
                  </div>
                  <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
                    <div style={{ fontSize: '11px', color: 'var(--mu)', lineHeight: 1.5 }}>
                      {(item.prompt || item.html).slice(0, 140)}{(item.prompt || item.html).length > 140 ? '...' : ''}
                    </div>
                    {!!item.tags?.length && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {item.tags.slice(0, 4).map((tag) => (
                          <span
                            key={tag}
                            style={{
                              fontSize: '9px',
                              color: '#c4b5fd',
                              background: 'rgba(124,58,237,0.12)',
                              border: '1px solid rgba(124,58,237,0.18)',
                              padding: '3px 6px',
                              borderRadius: 999,
                            }}
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: '10px', color: 'var(--mu)' }}>
                      <span>{item.html.length.toLocaleString()} chars</span>
                      <span>Created {new Date(item.createdAt).toLocaleDateString()}</span>
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      <button onClick={() => loadSavedPage(item)} style={{ ...btnBase, padding: '6px 10px', fontSize: '10.5px' }}>
                        <FolderOpen size={12} /> Load
                      </button>
                      <button onClick={() => void forkSavedPage(item)} style={{ ...btnBase, padding: '6px 10px', fontSize: '10.5px' }}>
                        <GitFork size={12} /> Fork
                      </button>
                      <button onClick={() => openRenameDialog(item)} style={{ ...btnBase, padding: '6px 10px', fontSize: '10.5px' }}>
                        <Pencil size={12} /> Rename
                      </button>
                      <button
                        onClick={() => openDeleteDialog(item)}
                        style={{
                          ...btnBase, padding: '6px 10px', fontSize: '10.5px',
                          color: '#fca5a5', borderColor: 'rgba(239,68,68,0.18)',
                        }}
                      >
                        <Trash2 size={12} /> Delete
                      </button>
                    </div>
                  </div>
                </div>
              ))}

              {(creatorMode === 'project' ? filteredProjectGalleryItems.length === 0 : filteredGalleryItems.length === 0) && (
                <div style={{
                  flex: 1,
                  minHeight: 180,
                  border: '1px dashed rgba(255,255,255,0.12)',
                  borderRadius: 16,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 10,
                  color: 'var(--mu)',
                  textAlign: 'center',
                  padding: 24,
                }}>
                  <FolderOpen size={22} style={{ color: 'var(--pl)' }} />
                  <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--tx)' }}>
                    {creatorMode === 'project'
                      ? (projectGalleryItems.length === 0 ? 'No saved projects yet' : 'No matches for this search')
                      : (galleryItems.length === 0 ? 'No saved pages yet' : 'No matches for this search')}
                  </div>
                  <div style={{ fontSize: '10.5px', lineHeight: 1.5, maxWidth: 280 }}>
                    {creatorMode === 'project'
                      ? (projectGalleryItems.length === 0
                        ? 'Save a project here so you can restore the full file tree later.'
                        : 'Try a different search term.')
                      : (galleryItems.length === 0
                        ? 'Save a generated page here so you can reload it later and keep iterating.'
                        : 'Try a different search term.')}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ═══ Version timeline strip ═══ */}
      {creatorMode === 'single' && versions.length > 1 && (
        <div style={{
          display: 'flex', gap: 4, padding: '4px 8px', borderTop: '1px solid var(--b1)',
          overflowX: 'auto', background: 'var(--s1)', flexShrink: 0,
        }}>
          {versions.map((v, i) => (
            <button
              key={v.id}
              onClick={() => restoreVersion(i)}
              title={v.prompt}
              style={{
                flexShrink: 0, padding: '2px 8px', fontSize: '10px',
                background: i === currentIdx ? 'var(--p)' : 'var(--s2)',
                color: i === currentIdx ? 'white' : 'var(--mu)',
                border: '1px solid ' + (i === currentIdx ? 'var(--p)' : 'var(--b1)'),
                borderRadius: 'var(--r-pill)', cursor: 'pointer', fontFamily: 'inherit',
                maxWidth: 100, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}
            >
              v{i + 1}
            </button>
          ))}
        </div>
      )}

      {/* ═══ Footer action bar ═══ */}
      {hasContent && (
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: 6, padding: '6px 10px', flexShrink: 0,
          borderTop: '1px solid var(--b1)',
        }}>
          {creatorMode === 'single' && (
            <>
              <button
                onClick={() => { const v = undo(); if (v) { setCurrentHtml(v.code); updatePreview(v.code); } }}
                disabled={!canUndo}
                title="Undo"
                style={{
                  background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
                  color: canUndo ? 'var(--mu)' : 'var(--s3)', padding: '3px 6px', cursor: canUndo ? 'pointer' : 'not-allowed',
                  display: 'flex', alignItems: 'center', opacity: canUndo ? 1 : 0.4,
                }}
              >
                <Undo2 size={12} />
              </button>
              <button
                onClick={() => { const v = redo(); if (v) { setCurrentHtml(v.code); updatePreview(v.code); } }}
                disabled={!canRedo}
                title="Redo"
                style={{
                  background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
                  color: canRedo ? 'var(--mu)' : 'var(--s3)', padding: '3px 6px', cursor: canRedo ? 'pointer' : 'not-allowed',
                  display: 'flex', alignItems: 'center', opacity: canRedo ? 1 : 0.4,
                }}
              >
                <Redo2 size={12} />
              </button>
            </>
          )}
          {/* Visual Feedback */}
          {hasContent && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
              <button
                onClick={async () => {
                  if (!iframeRef.current) return;
                  const lastPrompt = messages.filter(m => m.role === 'user').pop()?.text || '';
                  if (creatorMode === 'single') {
                    await visualFeedback.analyze(iframeRef.current, lastPrompt);
                  } else {
                    await visualFeedback.analyze(iframeRef.current, lastPrompt);
                  }
                  setVfPanelOpen(true);
                }}
                disabled={visualFeedback.status === 'analyzing' || visualFeedback.status === 'capturing'}
                style={{
                  background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
                  color: 'var(--mu)', padding: '3px 8px', cursor: 'pointer', display: 'flex',
                  alignItems: 'center', gap: 4, fontSize: '10px', fontFamily: 'inherit',
                  opacity: visualFeedback.status === 'analyzing' || visualFeedback.status === 'capturing' ? 0.6 : 1,
                }}
                title="Visual quality check (uses vision model)"
              >
                <Eye size={12} />
                {visualFeedback.status === 'analyzing' || visualFeedback.status === 'capturing' ? 'Analyzing...' : 'Check'}
              </button>
              {visualFeedback.feedback && (
                <button
                  onClick={() => setVfPanelOpen(!vfPanelOpen)}
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer', padding: '2px 6px',
                    borderRadius: 'var(--r-pill)', fontSize: '10px', fontWeight: 700,
                    color: visualFeedback.feedback.score >= 8 ? '#3fb950'
                      : visualFeedback.feedback.score >= 5 ? '#d29922'
                      : '#f85149',
                    backgroundColor: visualFeedback.feedback.score >= 8 ? 'rgba(63,185,80,0.12)'
                      : visualFeedback.feedback.score >= 5 ? 'rgba(210,153,34,0.12)'
                      : 'rgba(248,81,73,0.12)',
                  }}
                >
                  {visualFeedback.feedback.score}/10
                </button>
              )}
              {visualFeedback.status === 'unavailable' && (
                <span style={{ fontSize: '9px', color: 'var(--mu)', fontStyle: 'italic' }}>No vision model</span>
              )}
            </div>
          )}
          <ActionBtn id="save" icon={<Save size={13} />} label={saveLabel} onClick={openSaveDialog} hoveredBtn={hoveredBtn} setHoveredBtn={setHoveredBtn} />
          <ActionBtn id="copy" icon={<Copy size={13} />} label={copyLabel} onClick={copyCode} hoveredBtn={hoveredBtn} setHoveredBtn={setHoveredBtn} />
          <ActionBtn
            id="components"
            icon={<Layout size={13} />}
            label="Components"
            onClick={() => setComponentGalleryOpen(true)}
            hoveredBtn={hoveredBtn}
            setHoveredBtn={setHoveredBtn}
          />
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => setExportOpen(!exportOpen)}
              title="Export"
              style={{
                background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
                color: 'var(--mu)', padding: '3px 8px', cursor: 'pointer', display: 'flex',
                alignItems: 'center', gap: 4, fontSize: '10px', fontFamily: 'inherit',
              }}
            >
              <Download size={12} /> Export
            </button>
            {exportOpen && (
              <div style={{
                position: 'absolute', bottom: '100%', left: 0, marginBottom: 4,
                background: '#1e1e1e', border: '1px solid #333', borderRadius: 'var(--r-md)',
                padding: '4px 0', minWidth: 160, zIndex: 20, boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
              }}>
                <button onClick={() => { downloadHtml(); setExportOpen(false); }} style={exportItemStyle}>
                  {creatorMode === 'project' ? `Download ${getBaseName(activeProjectFile)}` : 'Download HTML'}
                </button>
                {creatorMode === 'project' && (
                  <>
                    <button onClick={() => { downloadProjectBundleHtml(); setExportOpen(false); }} style={exportItemStyle}>
                      Download Bundle HTML
                    </button>
                    <button onClick={() => { downloadProjectSnapshot(); setExportOpen(false); }} style={exportItemStyle}>
                      Download Project JSON
                    </button>
                  </>
                )}
                {creatorMode === 'single' && (
                  <button onClick={() => { copyDataUrl(); setExportOpen(false); }} style={exportItemStyle}>
                    Copy as Data URL
                  </button>
                )}
                <button onClick={() => { openInCodeSandbox(); setExportOpen(false); }} style={exportItemStyle}>
                  Open in CodeSandbox
                </button>
                <button onClick={() => { openInStackBlitz(); setExportOpen(false); }} style={exportItemStyle}>
                  Open in StackBlitz
                </button>
                <div style={{ borderTop: '1px solid #333', margin: '4px 0' }} />
                <button onClick={() => { handleDownloadZip(); setExportOpen(false); }} style={exportItemStyle}>
                  Download as ZIP
                </button>
                <button onClick={() => { handleShare(); setExportOpen(false); }} style={{ ...exportItemStyle, color: '#a78bfa' }}>
                  {sharing ? 'Sharing...' : 'Share (get live URL)'}
                </button>
                <button onClick={() => { handleGitHubDeploy(); setExportOpen(false); }} style={exportItemStyle}>
                  Deploy to GitHub Pages
                </button>
                <div style={{ borderTop: '1px solid #333', margin: '4px 0' }} />
                <button onClick={() => { setDeployDashboardOpen(true); setExportOpen(false); }} style={exportItemStyle}>
                  Manage Deployments
                </button>
                <div style={{ borderTop: '1px solid #333', margin: '4px 0' }} />
                <button onClick={() => {
                  const { handoffToPanel } = useStore.getState();
                  const code = creatorMode === 'project' ? (projectFs?.buildBundle() || currentHtml) : currentHtml;
                  handoffToPanel('slides', { code, from: 'Web Creator' });
                  setExportOpen(false);
                }} style={exportItemStyle}>
                  Create Slides from This
                </button>
                <button onClick={() => {
                  const { handoffToPanel } = useStore.getState();
                  handoffToPanel('code', { context: `Use this HTML in your Python analysis:\n${(creatorMode === 'project' ? projectFs?.buildBundle() : currentHtml)?.slice(0, 5000) || ''}`, from: 'Web Creator' });
                  setExportOpen(false);
                }} style={exportItemStyle}>
                  Process with Python
                </button>
              </div>
            )}
          </div>
          <ActionBtn id="sendcli" icon={<Upload size={13} />} label="Send to CLI" onClick={sendToCli} hoveredBtn={hoveredBtn} setHoveredBtn={setHoveredBtn} />
          <div style={{ flex: 1 }} />
          <ActionBtn
            id="fullscreen"
            icon={fullscreen ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
            label={fullscreen ? 'Exit' : 'Full Screen'}
            onClick={() => setFullscreen(f => !f)}
            hoveredBtn={hoveredBtn}
            setHoveredBtn={setHoveredBtn}
          />
        </div>
      )}

      {/* Visual Feedback Panel */}
      {vfPanelOpen && visualFeedback.feedback && (
        <div style={{
          position: 'absolute', bottom: 48, left: 8, right: 8, zIndex: 50,
          background: 'var(--bg)', border: '1px solid var(--b1)', borderRadius: 'var(--r-lg)',
          boxShadow: '0 -4px 16px rgba(0,0,0,0.3)', maxHeight: 260, overflow: 'auto',
        }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '10px 12px', borderBottom: '1px solid var(--b1)',
          }}>
            <span style={{
              fontSize: '12px', fontWeight: 700,
              color: visualFeedback.feedback.score >= 8 ? '#3fb950'
                : visualFeedback.feedback.score >= 5 ? '#d29922' : '#f85149',
            }}>
              Visual Analysis ({visualFeedback.feedback.score}/10)
            </span>
            <span style={{ fontSize: '10.5px', color: 'var(--mu)', flex: 1 }}>
              {visualFeedback.feedback.overall}
            </span>
            <button onClick={() => setVfPanelOpen(false)} style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer', padding: 0, display: 'flex' }}>
              <X size={14} />
            </button>
          </div>
          {visualFeedback.feedback.issues.length > 0 ? (
            <div style={{ padding: '8px 12px' }}>
              {visualFeedback.feedback.issues.map((issue, i) => (
                <div key={i} style={{
                  display: 'flex', gap: 8, alignItems: 'flex-start',
                  padding: '6px 0', borderBottom: i < visualFeedback.feedback!.issues.length - 1 ? '1px solid var(--b1)' : 'none',
                }}>
                  <span style={{
                    fontSize: '9px', fontWeight: 700, padding: '2px 5px', borderRadius: 3,
                    color: issue.severity === 'high' ? '#f85149' : issue.severity === 'medium' ? '#d29922' : '#8b949e',
                    background: issue.severity === 'high' ? 'rgba(248,81,73,0.12)' : issue.severity === 'medium' ? 'rgba(210,153,34,0.12)' : 'rgba(139,148,163,0.12)',
                    textTransform: 'uppercase', flexShrink: 0,
                  }}>
                    {issue.severity}
                  </span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: '11px', color: 'var(--tx)' }}>{issue.description}</div>
                    {issue.suggestion && (
                      <div style={{ fontSize: '10px', color: 'var(--mu)', marginTop: 2 }}>Fix: {issue.suggestion}</div>
                    )}
                  </div>
                </div>
              ))}
              <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                {visualFeedback.feedback.score < 8 && (
                  <button
                    onClick={async () => {
                      if (!iframeRef.current) return;
                      const lastPrompt = messages.filter(m => m.role === 'user').pop()?.text || '';
                      const code = creatorMode === 'project' ? (projectFs?.buildBundle() || '') : currentHtml;
                      await visualFeedback.autoRefine(iframeRef.current, lastPrompt, code, (newCode) => {
                        if (creatorMode === 'single') {
                          setCurrentHtml(newCode);
                          currentHtmlRef.current = newCode;
                          updatePreview(newCode);
                        }
                      });
                    }}
                    disabled={visualFeedback.status === 'refining'}
                    style={{
                      background: 'var(--p)', color: '#fff', border: 'none', borderRadius: 'var(--r-sm)',
                      padding: '5px 12px', fontSize: '10.5px', fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit',
                      opacity: visualFeedback.status === 'refining' ? 0.6 : 1,
                    }}
                  >
                    {visualFeedback.status === 'refining' ? 'Refining...' : 'Auto-Fix Issues'}
                  </button>
                )}
                <button
                  onClick={() => { setVfPanelOpen(false); visualFeedback.reset(); }}
                  style={{
                    background: 'var(--s2)', color: 'var(--mu)', border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-sm)', padding: '5px 12px', fontSize: '10.5px', cursor: 'pointer', fontFamily: 'inherit',
                  }}
                >
                  Dismiss
                </button>
              </div>
            </div>
          ) : (
            <div style={{ padding: '12px', fontSize: '11px', color: 'var(--mu)' }}>
              No issues detected. The page looks good!
            </div>
          )}
        </div>
      )}

      {/* Component Gallery */}
      <ComponentGallery
        open={componentGalleryOpen}
        onClose={() => setComponentGalleryOpen(false)}
        onInsert={(comp) => {
          // Insert component HTML at the current context
          if (creatorMode === 'single') {
            const newHtml = currentHtml
              ? currentHtml.replace('</body>', `\n${comp.html}\n</body>`)
              : comp.html;
            setCurrentHtml(newHtml);
            currentHtmlRef.current = newHtml;
            updatePreview(newHtml);
          }
          setTimedStatus(`Inserted: ${comp.name}`, 1500);
        }}
      />

      {/* Deploy Dashboard */}
      <DeployDashboard open={deployDashboardOpen} onClose={() => setDeployDashboardOpen(false)} />
    </div>
  );
}
