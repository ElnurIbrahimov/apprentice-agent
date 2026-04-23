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
      { label: 'Law Firm', icon: '⚖️', desc: 'Practice areas, attorneys, consultations', prompt: 'Create a law firm website with hero, practice area cards (6), attorney profiles with headshots, case results counter, client testimonials, consultation booking form, and footer with office locations.' },
      { label: 'Consulting', icon: '📈', desc: 'Services, methodology, results', prompt: 'Create a consulting firm website with hero, services overview (6 cards), methodology/process steps, case studies with metrics, team carousel, trust badges (clients served, years), and contact CTA.' },
      { label: 'Dental Clinic', icon: '🦷', desc: 'Services, doctors, booking', prompt: 'Create a dental clinic website with hero, services grid (cleanings, implants, orthodontics, whitening, etc.), doctor profiles, before/after gallery, patient reviews, insurance accepted section, and appointment booking form.' },
      { label: 'Gym & Fitness', icon: '💪', desc: 'Classes, trainers, membership', prompt: 'Create a gym website with hero (bold action shot), class schedule grid, trainer profiles with specialties, membership tier cards (3), facilities gallery, transformation stories, and free trial CTA.' },
      { label: 'Hotel', icon: '🏨', desc: 'Rooms, amenities, booking', prompt: 'Create a luxury hotel website with full-screen hero, room category cards with prices, amenities grid with icons, photo gallery, guest reviews, location/map section, and booking date picker CTA.' },
      { label: 'Car Dealership', icon: '🚗', desc: 'Inventory, financing, trade-in', prompt: 'Create a car dealership website with hero, inventory search (make/model/year/price filters), featured vehicles grid (8 cards), financing calculator, trade-in value form, dealership info, and test drive booking.' },
      { label: 'Construction', icon: '🏗️', desc: 'Projects, services, estimates', prompt: 'Create a construction company website with hero, services (residential, commercial, renovation), project gallery with before/after slider, team section, safety certifications, free estimate form, and testimonials.' },
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
      { label: 'Photography', icon: '📷', desc: 'Masonry gallery, lightbox, contact', prompt: 'Create a photography portfolio with full-width hero image, masonry photo gallery with lightbox modal on click, category filter tabs (portraits, landscape, street, events), about section, and booking form.' },
      { label: 'Music Artist', icon: '🎵', desc: 'Bio, discography, tour dates', prompt: 'Create a music artist website with dark cinematic hero, latest release section with album art and streaming links, discography grid, tour dates table, music video embeds, merch store preview, and newsletter signup.' },
      { label: 'Freelancer', icon: '💻', desc: 'Services, rates, testimonials', prompt: 'Create a freelancer website with hero (name + title + CTA), services offered with pricing (hourly/project), project portfolio grid, client testimonials carousel, process steps, availability calendar placeholder, and contact form.' },
      { label: 'Author/Writer', icon: '📚', desc: 'Books, bio, blog, events', prompt: 'Create an author website with hero featuring latest book cover, books grid with buy links, author bio with photo, upcoming events/signings, blog feed, newsletter signup, and press kit download.' },
      { label: 'Podcast', icon: '🎙️', desc: 'Episodes, subscribe, guests', prompt: 'Create a podcast website with hero (podcast art + subscribe buttons for Apple/Spotify/Google), latest episodes list with play buttons, guest profiles, about the hosts, transcript previews, and sponsor section.' },
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
      { label: 'Signup Flow', icon: '✨', desc: 'Multi-step, progress, validation', prompt: 'Create a multi-step signup form with progress bar (4 steps: Account, Profile, Preferences, Confirm), animated transitions between steps, inline validation, password strength meter, and success confetti animation.' },
      { label: 'Settings Page', icon: '⚙️', desc: 'Sections, toggles, forms', prompt: 'Create an app settings page with sidebar categories (Profile, Notifications, Security, Billing, Integrations), toggle switches, input fields, avatar upload, connected accounts, danger zone with delete account. Dark theme.' },
      { label: 'Onboarding', icon: '👋', desc: 'Welcome slides, feature tour', prompt: 'Create a 5-step app onboarding flow with large illustrations (use CSS art), feature descriptions, dot pagination, skip button, animated transitions between slides, and "Get Started" final CTA.' },
      { label: 'Changelog', icon: '📋', desc: 'Version history, badges, filters', prompt: 'Create a changelog/release notes page with version badges (major/minor/patch colors), date headers, categorized changes (New, Improved, Fixed, Removed), search/filter bar, and "Subscribe to updates" CTA.' },
      { label: 'Documentation', icon: '📖', desc: 'Sidebar nav, code blocks, search', prompt: 'Create a documentation page with sidebar table of contents, search bar, breadcrumbs, markdown-style content with code blocks (syntax highlighted), copy buttons, info/warning callout boxes, and prev/next navigation.' },
      { label: 'API Reference', icon: '🔌', desc: 'Endpoints, params, examples', prompt: 'Create an API reference page with sidebar endpoint list, HTTP method badges (GET green, POST blue, PUT orange, DELETE red), parameter tables, request/response code examples, and "Try it" button placeholder.' },
      { label: 'Status Page', icon: '🟢', desc: 'Uptime, incidents, metrics', prompt: 'Create a system status page with overall status banner (All Systems Operational), service list with uptime bars (90 days), incident history timeline, subscribe to updates form, and uptime percentage badges.' },
    ]
  },
  {
    category: 'E-commerce',
    templates: [
      { label: 'Product Page', icon: '🛍️', desc: 'Gallery, details, reviews', prompt: 'Create a product detail page with image gallery (main + thumbnails), product title, price, color/size selectors, add-to-cart button, description tabs, reviews section, and related products.' },
      { label: 'Store Front', icon: '🏪', desc: 'Banner, categories, products', prompt: 'Create an e-commerce homepage with hero banner, category cards, featured products grid (8 items with image/name/price/rating), deals section with countdown timer, and newsletter signup.' },
      { label: 'Checkout', icon: '💳', desc: 'Cart, shipping, payment', prompt: 'Create a checkout page with order summary sidebar, shipping form, payment form with card input, express checkout buttons (Apple Pay, Google Pay), promo code input, and order total breakdown.' },
      { label: 'Product Compare', icon: '⚖️', desc: 'Side-by-side, specs, ratings', prompt: 'Create a product comparison page with 3 products side-by-side, spec rows (processor, RAM, storage, camera), star ratings, price comparison, pros/cons lists, and "Best For" badges.' },
      { label: 'Flash Sale', icon: '⚡', desc: 'Countdown, deals, urgency', prompt: 'Create a flash sale page with large countdown timer, deal cards (original price crossed out, sale price, % off badge, stock remaining bar), category tabs, and "sold out" overlay on expired items.' },
      { label: 'Digital Products', icon: '📦', desc: 'Downloads, licenses, previews', prompt: 'Create a digital product store for selling templates/fonts/icons. Product cards with preview thumbnails, format badges (PSD/AI/Figma), license selector (personal/commercial), instant download CTA, and bundle deals.' },
      { label: 'Subscription Box', icon: '📦', desc: 'Plans, what\'s included, reviews', prompt: 'Create a subscription box landing page with hero showing unboxed products, plan tiers (monthly/quarterly/annual), past boxes gallery, unboxing video placeholder, subscriber count, reviews, and gift option.' },
      { label: 'Food Delivery', icon: '🍕', desc: 'Menu, cart, delivery tracker', prompt: 'Create a food delivery app UI with restaurant header, menu categories (horizontal scroll), food items with photos/prices/add buttons, floating cart summary, delivery address input, and order tracking progress bar.' },
    ]
  },
  {
    category: 'Creative & Media',
    templates: [
      { label: 'Coming Soon', icon: '⏳', desc: 'Countdown, email signup', prompt: 'Create a coming soon page with animated countdown timer, email signup form, progress bar, social links, and a mesmerizing animated gradient background.' },
      { label: 'Event', icon: '🎪', desc: 'Speakers, schedule, tickets', prompt: 'Create an event/conference landing page with hero with date/location, speaker cards (6), schedule/agenda timeline, ticket tiers, venue map placeholder, sponsors grid, and FAQ.' },
      { label: 'Newsletter', icon: '📬', desc: 'Header, articles, CTA', prompt: 'Create an email newsletter template (HTML email compatible) with header logo, hero image, main article, 3 story cards, CTA button, social icons footer. 600px max-width, table-based layout.' },
      { label: 'Video Landing', icon: '🎬', desc: 'Hero video, chapters, subscribe', prompt: 'Create a video course landing page with hero video player (16:9 placeholder), course outline with chapter list, instructor bio, student count + rating, pricing with guarantee badge, and FAQ accordion.' },
      { label: 'NFT Gallery', icon: '🖼️', desc: 'Collection, bids, wallet', prompt: 'Create an NFT marketplace page with featured collection hero, NFT grid (12 cards with image, name, creator, price in ETH, bid button), filter sidebar (category, price range, status), and wallet connect button.' },
      { label: 'Magazine', icon: '📰', desc: 'Editorial layout, columns, hero', prompt: 'Create a digital magazine homepage with large editorial hero article, 2-column article grid, breaking news ticker, category tabs, trending sidebar, subscribe CTA, and author bylines with avatars.' },
      { label: 'Film/Movie', icon: '🎥', desc: 'Trailer, cast, reviews, showtimes', prompt: 'Create a movie promotional page with cinematic hero (title over backdrop), trailer embed, cast carousel with role names, critic scores (Rotten Tomatoes style), review quotes, showtime selector, and ticket CTA.' },
      { label: 'Art Exhibition', icon: '🎨', desc: 'Gallery, artist, dates, tickets', prompt: 'Create an art exhibition page with full-bleed hero artwork, exhibition details (dates, location, hours), artwork grid with titles and medium, artist statement, audio guide mention, and ticket booking.' },
    ]
  },
  {
    category: 'Education & Non-Profit',
    templates: [
      { label: 'Online Course', icon: '🎓', desc: 'Curriculum, instructor, enroll', prompt: 'Create an online course page with hero (title + instructor), course stats (duration, lessons, level), curriculum accordion, instructor bio, student reviews, certificate preview, pricing, and enroll CTA.' },
      { label: 'School/University', icon: '🏫', desc: 'Programs, campus, admissions', prompt: 'Create a university homepage with hero, program cards (8), campus life photo grid, upcoming events, news feed, admissions timeline, virtual tour CTA, and application deadline counter.' },
      { label: 'Non-Profit', icon: '🌍', desc: 'Mission, impact, donate', prompt: 'Create a non-profit website with hero (mission statement), impact stats (lives changed, donations, volunteers), programs section, stories of impact carousel, donation form with preset amounts ($25/$50/$100/custom), volunteer signup, and partner logos.' },
      { label: 'Church', icon: '⛪', desc: 'Services, sermons, community', prompt: 'Create a church website with hero (welcome message), service times, recent sermons list with audio/video, upcoming events, community groups, online giving button, location with map, and prayer request form.' },
      { label: 'Tutoring', icon: '📝', desc: 'Subjects, tutors, booking', prompt: 'Create a tutoring service website with hero, subject cards (Math, Science, English, SAT Prep, etc.), tutor profiles with ratings and hourly rates, how-it-works steps, free trial lesson CTA, and parent testimonials.' },
      { label: 'Conference', icon: '🎤', desc: 'Speakers, agenda, sponsors', prompt: 'Create a tech conference website with hero (name, date, location, ticket CTA), keynote speaker section, 2-day agenda with track tabs, sponsor tiers (platinum/gold/silver), venue info, early bird pricing, and live stream option.' },
    ]
  },
  {
    category: 'Technology',
    templates: [
      { label: 'Developer Portfolio', icon: '👨‍💻', desc: 'GitHub, projects, tech stack', prompt: 'Create a developer portfolio with terminal-style hero (typing animation), GitHub stats cards, project showcase (6 repos with stars/forks/language), tech stack icons grid, blog posts, open source contributions timeline, and contact form. Dark hacker theme.' },
      { label: 'AI/ML Product', icon: '🤖', desc: 'Demo, features, integrations', prompt: 'Create an AI product landing page with hero (animated neural network visualization in CSS), live demo section, feature comparison (vs competitors), integration logos, API code snippet, pricing, and enterprise CTA.' },
      { label: 'Open Source', icon: '🔓', desc: 'README, contributors, stars', prompt: 'Create an open source project page with hero (project name + description + GitHub badges), quick start code block, feature list, contributor avatars grid, star history chart placeholder, sponsor tiers, and "Star on GitHub" CTA.' },
      { label: 'CLI Tool', icon: '⌨️', desc: 'Install, commands, examples', prompt: 'Create a CLI tool documentation site with hero (terminal screenshot), one-line install command with copy button, command reference table, usage examples with syntax highlighting, comparison table, and GitHub link. Monospace dark theme.' },
      { label: 'Browser Extension', icon: '🧩', desc: 'Features, screenshots, install', prompt: 'Create a browser extension landing page with hero (browser mockup with extension), feature cards (6), screenshot carousel, browser compatibility badges (Chrome/Firefox/Edge/Safari), reviews, and "Add to Chrome" CTA button.' },
      { label: 'API Service', icon: '🔗', desc: 'Endpoints, pricing, docs', prompt: 'Create an API service landing page with hero, code example (curl request + JSON response), pricing tiers based on API calls, uptime guarantee badge, documentation preview, SDKs available (Python/Node/Go/Ruby icons), and API key signup form.' },
      { label: 'DevOps Dashboard', icon: '🖥️', desc: 'Pipelines, deploys, monitoring', prompt: 'Create a DevOps dashboard with sidebar, deployment pipeline visualization (build→test→staging→production stages), server metrics cards (CPU, RAM, Disk), recent deployments log, alert notifications, and uptime graphs. Dark theme.' },
      { label: 'Crypto/Web3', icon: '₿', desc: 'Wallet, tokens, swap', prompt: 'Create a DeFi/crypto dashboard with wallet connect button, token balances list, price charts (candlestick placeholder), swap interface (from/to tokens), transaction history, gas tracker, and portfolio allocation donut chart. Dark cyber theme.' },
    ]
  },
  {
    category: 'Social & Community',
    templates: [
      { label: 'Social Profile', icon: '👥', desc: 'Posts, followers, media grid', prompt: 'Create a social media profile page with cover photo, profile pic, bio, stats (posts/followers/following), tab bar (Posts/Media/Likes), post feed with like/comment/share buttons, and suggested users sidebar.' },
      { label: 'Forum', icon: '💬', desc: 'Categories, threads, users', prompt: 'Create a forum/community page with category cards (General, Help, Showcase, Off-Topic), latest threads list with avatars/replies/views, pinned announcements, search bar, user leaderboard sidebar, and new thread button.' },
      { label: 'Discord Server', icon: '🎮', desc: 'Channels, roles, invite', prompt: 'Create a Discord server landing page with hero (server name + member count), channel preview list, role cards with colors, rules section, featured community content, server stats, and "Join Server" CTA with copy-invite button.' },
      { label: 'Dating Profile', icon: '❤️', desc: 'Photos, bio, interests, match', prompt: 'Create a dating app profile page with photo carousel (5 photos), name/age/location, bio text, interest tags, prompts with answers ("My ideal weekend is..."), Spotify top artists, Instagram grid preview, and like/pass buttons.' },
      { label: 'Community Hub', icon: '🏘️', desc: 'Events, members, discussions', prompt: 'Create a community platform homepage with hero, upcoming events grid, member spotlight cards, discussion categories, resource library, community guidelines, and join/apply CTA. Warm welcoming design.' },
    ]
  },
  {
    category: 'Health & Wellness',
    templates: [
      { label: 'Medical Practice', icon: '🏥', desc: 'Doctors, services, appointments', prompt: 'Create a medical practice website with hero, services list (cardiology, dermatology, pediatrics, etc.), doctor profiles with credentials, patient portal login, insurance accepted logos, appointment booking form, and emergency contact info.' },
      { label: 'Fitness App', icon: '🏃', desc: 'Workouts, progress, nutrition', prompt: 'Create a fitness app landing page with hero (before/after transformation), workout plan preview, progress tracking charts, meal planning section, trainer video previews, app store badges, and free trial CTA.' },
      { label: 'Mental Health', icon: '🧘', desc: 'Therapy, resources, crisis line', prompt: 'Create a mental health support website with calming hero, therapy services (individual, couples, group), therapist profiles, resource library, self-assessment quiz CTA, crisis helpline banner, and appointment scheduler. Soft blue/green palette.' },
      { label: 'Spa & Wellness', icon: '🧖', desc: 'Treatments, packages, booking', prompt: 'Create a luxury spa website with full-bleed hero, treatment menu with prices, package deals (couples, full-day, weekend), gallery of facilities, client reviews, gift card purchase, and online booking. Elegant serif typography, earth tones.' },
      { label: 'Pharmacy', icon: '💊', desc: 'Products, prescriptions, delivery', prompt: 'Create an online pharmacy with hero, product categories (vitamins, personal care, prescriptions, baby care), featured products grid, prescription upload form, delivery info, health blog, and loyalty program section.' },
    ]
  },
  {
    category: 'Interactive & Fun',
    templates: [
      { label: 'Quiz/Survey', icon: '❓', desc: 'Questions, progress, results', prompt: 'Create an interactive quiz page with title, progress bar, question card with 4 answer options (highlight on select), next/back buttons, animated transitions between questions, timer, and results page with score, share buttons, and retry.' },
      { label: 'Calculator', icon: '🔢', desc: 'Inputs, formula, results', prompt: 'Create an interactive calculator app (like a mortgage or BMI calculator) with labeled inputs, sliders for ranges, real-time calculation display, results card with breakdown chart, comparison table, and share/print results buttons.' },
      { label: 'Recipe', icon: '🍳', desc: 'Ingredients, steps, nutrition', prompt: 'Create a recipe page with hero photo, recipe title/rating/time, ingredient checklist with servings adjuster, step-by-step instructions with photos, nutrition facts table, print recipe button, and related recipes carousel.' },
      { label: 'Timeline', icon: '📅', desc: 'Milestones, dates, progress', prompt: 'Create a vertical timeline page showing a company or project history. Each milestone has: date, title, description, icon. Alternating left/right layout, connecting line with dots, scroll-triggered fade-in animations.' },
      { label: 'Weather App', icon: '🌤️', desc: 'Current, forecast, location', prompt: 'Create a weather app UI with current weather card (temp, icon, condition, location), hourly forecast horizontal scroll, 7-day forecast list, weather details (humidity, wind, UV, pressure), sunrise/sunset times, and search location bar.' },
      { label: 'Music Player', icon: '🎶', desc: 'Now playing, playlist, controls', prompt: 'Create a music player UI with album art (large), song title/artist, progress bar with timestamps, playback controls (prev/play-pause/next, shuffle, repeat), volume slider, queue/playlist sidebar, and lyrics panel.' },
      { label: 'Chat UI', icon: '💬', desc: 'Messages, contacts, typing', prompt: 'Create a messaging app UI with contacts sidebar (avatars, last message, unread badge), chat area with message bubbles (sent/received, timestamps, read receipts), typing indicator, message input with emoji picker, and attachment button.' },
      { label: 'File Manager', icon: '📁', desc: 'Folders, files, upload', prompt: 'Create a file manager/cloud storage UI with breadcrumb navigation, grid/list view toggle, files and folders with icons, file details panel (size, modified, sharing), upload dropzone, search bar, and storage usage indicator.' },
    ]
  },
  {
    category: 'Landing Pages',
    templates: [
      { label: 'Waitlist', icon: '📋', desc: 'Teaser, signup, referral', prompt: 'Create a waitlist landing page with bold headline, product teaser (3 feature previews), email signup with referral counter ("You are #1,234 in line"), social proof ticker, and animated background particles.' },
      { label: 'Product Hunt', icon: '🏆', desc: 'Launch day, upvotes, demo', prompt: 'Create a Product Hunt style launch page with product hero, demo video embed, feature list with icons, founder story, upvote counter, press mentions, and early adopter pricing with countdown.' },
      { label: 'Newsletter Landing', icon: '✉️', desc: 'Value prop, past issues, subscribe', prompt: 'Create a newsletter landing page with bold headline, value proposition bullets, past issue previews (3 cards), subscriber count badge, testimonial quotes, email input with subscribe button, and "free forever" badge.' },
      { label: 'Lead Magnet', icon: '🧲', desc: 'Free resource, preview, download', prompt: 'Create a lead magnet landing page with hero (ebook/guide cover mockup), chapter preview list, what you\'ll learn bullets, author credentials, social proof (downloads count), email gate form, and bonus content mention.' },
      { label: 'Comparison', icon: '🔄', desc: 'Us vs them, feature table, switch', prompt: 'Create a comparison landing page ("Why switch from X to us") with hero, side-by-side feature comparison table with checkmarks/crosses, pricing comparison, migration guide steps, customer switch stories, and "Switch Now" CTA.' },
      { label: 'Black Friday', icon: '🏷️', desc: 'Deals, countdown, urgency', prompt: 'Create a Black Friday/Cyber Monday deals page with huge countdown timer, deals grid (original/sale price, % off badges), category filter, "Almost Gone" urgency indicators, early access email signup, and terms & conditions.' },
    ]
  },
  {
    category: 'Travel & Hospitality',
    templates: [
      { label: 'Travel Agency', icon: '🌴', desc: 'Destinations, packages, booking', prompt: 'Create a travel agency website with hero (destination carousel), featured packages grid (6 cards with photos, price from, duration, rating), destination explorer by continent, travel insurance section, testimonials with photos, "build your trip" CTA, and travel blog preview. Warm sunset palette.' },
      { label: 'Airbnb Clone', icon: '🏡', desc: 'Search, listings, filters', prompt: 'Create a short-term rental homepage (Airbnb-style) with hero image + search bar (destination/dates/guests), category pills (Beachfront/Cabin/Pool/Unique stays), listing cards grid (photo carousel, location, dates, price/night, rating), map preview toggle, filter modal trigger, and footer with host CTA.' },
      { label: 'Cruise Line', icon: '🛳️', desc: 'Itineraries, ships, deals', prompt: 'Create a cruise line website with full-screen hero (ship at sea), upcoming itineraries grid (ports, days, price from), ship fleet showcase, onboard experiences (dining/entertainment/spa), destination ports map, deals banner, and booking form with cabin class selector.' },
      { label: 'Tour Guide', icon: '🗺️', desc: 'Tours, guides, booking', prompt: 'Create a local tours website with hero video placeholder, tour cards grid (photo, duration, group size, price, language options, rating), filter by city/category/duration, meet-your-guide profiles, FAQ section, and instant booking with date picker and participant count.' },
      { label: 'Adventure Travel', icon: '🏔️', desc: 'Expeditions, difficulty, gear', prompt: 'Create an adventure travel outfitter site with bold cinematic hero (mountaineer/diver/trekker), expedition grid with difficulty levels (easy/moderate/expert), what is included checklist, required gear list, safety certifications, guide credentials, photo journal, and expedition signup.' },
      { label: 'Retreat Center', icon: '🧘', desc: 'Programs, schedule, lodging', prompt: 'Create a yoga/wellness retreat website with serene hero, upcoming retreats calendar, daily schedule sample, lodging photos, menu preview, facilitator bios, testimonials, pricing tiers, and booking form with room preference. Soft earthy palette.' },
      { label: 'Ski Resort', icon: '⛷️', desc: 'Lift status, lessons, lodging', prompt: 'Create a ski resort website with hero (snowy mountain), live conditions bar (snow depth/lifts open/temperature), trail map embed placeholder, lift ticket pricing, ski school packages, rental info, on-mountain dining, and lodging booking. Cool blue/white palette.' },
      { label: 'Bed & Breakfast', icon: '🛏️', desc: 'Rooms, breakfast, location', prompt: 'Create a cozy bed & breakfast website with warm hero, room gallery with tour, breakfast menu highlights, local attraction list with map, host bio, guest reviews, availability calendar, and book-direct CTA with best-price-guarantee badge. Homey traditional palette.' },
      { label: 'Airline', icon: '✈️', desc: 'Book flights, loyalty, status', prompt: 'Create an airline homepage with hero, flight search (from/to/dates/passengers/class) prominent, deals grid, loyalty program signup, check-in and flight status quick links, destinations map, seat preview, and mobile app download callouts.' },
      { label: 'Camping & RV Park', icon: '🏕️', desc: 'Sites, amenities, reservations', prompt: 'Create a campground website with outdoorsy hero, site types (tent/RV/cabin) with photos, amenities grid (showers/WiFi/laundry/fire pits), interactive site map, rates table, activities list, pet policy, and reservation form with date range. Earthy woodsy palette.' },
    ]
  },
  {
    category: 'Food & Beverage',
    templates: [
      { label: 'Coffee Shop', icon: '☕', desc: 'Menu, story, location', prompt: 'Create a coffee shop website with warm hero (steaming cup), story of the shop, menu with categories (espresso/drip/cold brew/pastries), bean origin map, roasting process, loyalty program signup, store hours & location, and order-ahead CTA. Warm brown/cream palette.' },
      { label: 'Bakery', icon: '🥐', desc: 'Products, custom orders, gallery', prompt: 'Create a bakery website with inviting hero, product categories (breads/pastries/cakes/cookies), custom cake order form with inspiration gallery, featured seasonal items, baking process photos, delivery zones, and daily-bake schedule. Warm bakery palette (cream, butter yellow, dusty pink).' },
      { label: 'Brewery', icon: '🍺', desc: 'Beers, taproom, events', prompt: 'Create a craft brewery website with moody industrial hero, beer lineup cards (name, style, ABV, IBU, description, rotating tap indicator), brewery tour booking, taproom hours, upcoming events calendar, merch store preview, and distributor/where-to-buy section.' },
      { label: 'Winery', icon: '🍷', desc: 'Wines, tastings, vineyard', prompt: 'Create a winery website with scenic vineyard hero, wine varietals grid (vintage/notes/price/stock), tasting room reservation calendar, vineyard history timeline, meet the winemaker, wine club tiers, shipping FAQ, and shop with age gate. Elegant burgundy/gold palette.' },
      { label: 'Food Truck', icon: '🚚', desc: 'Schedule, menu, follow', prompt: 'Create a food truck website with playful hero, today\'s location with map pin, this weeks schedule, full menu with prices, Instagram feed embed placeholder, catering inquiry form, and social follow badges. Bold street-food palette.' },
      { label: 'Catering', icon: '🥘', desc: 'Packages, gallery, quote', prompt: 'Create a catering company website with elegant hero (event setup), event packages (corporate/wedding/private), menu sample galleries per cuisine, minimum guest count info, past event gallery with client quotes, chef bio, and quote request form with event details. Refined neutral palette.' },
      { label: 'Meal Prep Service', icon: '🥗', desc: 'Plans, menu, delivery', prompt: 'Create a meal prep/delivery service website with fresh hero, weekly menu preview, subscription plan tiers (3/5/7 meals/week), dietary filters (keto/vegan/low-carb/paleo), nutritional info modals, delivery zones, how-it-works steps, and first-week-discount signup. Fresh green palette.' },
      { label: 'Ice Cream Shop', icon: '🍦', desc: 'Flavors, scoops, fun', prompt: 'Create a playful ice cream shop website with bright hero, flavor grid with descriptions (seasonal badges), custom-sundae builder UI (scoops/toppings/sauces), nearby locations, birthday party packages, loyalty card (10 scoops = free), and Instagram-worthy gallery. Pastel palette.' },
      { label: 'Farmers Market', icon: '🥕', desc: 'Vendors, schedule, map', prompt: 'Create a farmers market website with rustic hero, this-week vendor list (by category: produce/meat/dairy/baked goods), seasonal produce calendar, market hours and locations, community events, vendor application form, and "what\'s fresh this week" blog feed. Earthy organic palette.' },
      { label: 'Juice Bar', icon: '🥤', desc: 'Menu, benefits, cleanse', prompt: 'Create a juice/smoothie bar website with vibrant hero, menu grid organized by benefit (Energy/Detox/Beauty/Immunity), ingredient transparency callouts, 3-day cleanse packages, smoothie bowl photo gallery, loyalty program, and delivery/pickup toggle. Vibrant health palette.' },
    ]
  },
  {
    category: 'Home & Local Services',
    templates: [
      { label: 'Plumber', icon: '🔧', desc: 'Services, 24/7, estimate', prompt: 'Create a plumbing service website with trust-building hero (truck photo + licensed badge), services list (repair/install/emergency), 24/7 emergency banner with phone, service area map, upfront pricing transparency, customer reviews, and instant-quote form. Trustworthy blue palette.' },
      { label: 'Electrician', icon: '⚡', desc: 'Licensed, services, estimates', prompt: 'Create an electrician services website with hero (licensed badge + years in business), residential and commercial service categories, common projects gallery (panel upgrades, EV chargers, lighting), safety certifications, financing options, and online booking. Professional yellow/navy palette.' },
      { label: 'Cleaning Service', icon: '🧹', desc: 'Packages, booking, eco', prompt: 'Create a home cleaning service website with bright clean hero, packages (standard/deep/move-in-out), booking calendar with pricing by home size, eco-friendly product callout, background-checked staff emphasis, recurring discount, and instant-quote form.' },
      { label: 'Landscaping', icon: '🌳', desc: 'Design, maintenance, gallery', prompt: 'Create a landscaping company website with lush hero, services (design/install/maintenance/hardscape), before/after project gallery slider, seasonal package subscriptions, plant selection guide, sustainability practices, service area map, and consultation request form. Earthy green palette.' },
      { label: 'HVAC', icon: '❄️', desc: 'Repair, install, maintenance', prompt: 'Create an HVAC service website with hero (clean technician photo), heating/cooling service grid, maintenance plan tiers (bronze/silver/gold), energy-savings calculator widget placeholder, emergency service badge, brand partner logos (Carrier/Trane/Lennox), and appointment booking.' },
      { label: 'Pest Control', icon: '🐜', desc: 'Treatments, packages, urgency', prompt: 'Create a pest control website with clean hero (not gross — tech at work), pest type quick-diagnoses grid (ants/roaches/termites/rodents with "got this?" CTAs), inspection booking (free), service packages, before/after home photos, family-safe-products callout, and review wall.' },
      { label: 'Painter', icon: '🎨', desc: 'Interior, exterior, portfolio', prompt: 'Create a painting contractor website with hero (finished room photo), interior/exterior service split, portfolio gallery by room type, color consultation offer, prep-and-cleanup promise, 2-year workmanship warranty, licensed/insured/bonded badges, and free estimate form. Tasteful neutral palette.' },
      { label: 'Roofing', icon: '🏚️', desc: 'Inspections, replacement, storm', prompt: 'Create a roofing company website with confident hero (finished roof aerial photo), services (new roof, repair, storm damage, inspection), financing options, manufacturer-certified badges, before/after storm damage gallery, insurance claim assistance, and inspection request form.' },
      { label: 'Locksmith', icon: '🔑', desc: '24/7, automotive, residential', prompt: 'Create a locksmith service website with urgent hero (24/7 phone large), service categories (residential/commercial/automotive/emergency), service area map, upfront pricing (no hidden fees badge), typical response time, bonded-licensed-insured badges, and quick-quote form.' },
      { label: 'Interior Designer', icon: '🛋️', desc: 'Portfolio, process, packages', prompt: 'Create an interior design studio website with editorial hero (beautiful room), portfolio grid by style (modern/traditional/bohemian/minimalist), designers profile with philosophy, service packages (e-design/full service/consultation only), process steps timeline, press mentions, and discovery call CTA. Elegant neutral palette.' },
      { label: 'Handyman', icon: '🔨', desc: 'Jobs, rates, reviews', prompt: 'Create a handyman service website with approachable hero, common jobs grid with typical pricing (fans/furniture/drywall/painting/tile), hourly rate transparency, neighborhood-focus badge, quick booking form for small jobs, insured badge, and customer-since-when local roots emphasis.' },
      { label: 'Moving Company', icon: '📦', desc: 'Quote, packing, interstate', prompt: 'Create a moving company website with hero (truck + smiling team), service types (local/long-distance/packing/storage), instant online quote form (move date/size/distance), moving checklist resource, real customer reviews, insurance coverage explanation, and booking CTA.' },
    ]
  },
  {
    category: 'Pet & Animal',
    templates: [
      { label: 'Veterinary Clinic', icon: '🐾', desc: 'Services, vets, appointments', prompt: 'Create a vet clinic website with warm hero (vet + pet), services grid (wellness/surgery/dental/emergency), meet-the-vets bios with photos, new patient welcome packet download, appointment booking, emergency contact, pet portal login, and adoption events. Friendly trustworthy palette.' },
      { label: 'Dog Walker', icon: '🐕', desc: 'Packages, schedule, photos', prompt: 'Create a dog walking service website with joyful hero (dog running), service packages (30/45/60 min walks, dog park visits, solo vs group), daily photo updates mention, GPS walk-tracking feature, insurance & bonded, neighborhood map, and meet-and-greet booking. Playful warm palette.' },
      { label: 'Pet Groomer', icon: '✂️', desc: 'Services, photos, booking', prompt: 'Create a pet grooming website with before/after transformation hero, services grid (bath/haircut/nails/de-shed), pricing by pet size, appointment booking calendar, team groomers with specialties, photo gallery of happy pups, and first-time-client discount. Soft pastel palette.' },
      { label: 'Pet Store', icon: '🐠', desc: 'Supplies, food, fish/reptile', prompt: 'Create a pet supply store website with hero (cute pets), category grid (dog/cat/small animal/reptile/fish/bird), featured products carousel, brand logos, curbside pickup info, loyalty program, pet adoption partners, and subscription auto-delivery option.' },
      { label: 'Animal Shelter', icon: '🐶', desc: 'Adoptable, donate, volunteer', prompt: 'Create an animal shelter website with emotional hero ("adopt dont shop"), adoptable pets grid (photo/name/breed/age/bio), adoption process steps, donate button (preset amounts), volunteer signup, foster program info, success stories, and events calendar. Warm hopeful palette.' },
      { label: 'Horse Stable', icon: '🐴', desc: 'Lessons, boarding, trails', prompt: 'Create an equestrian/horse stable website with dramatic hero (rider + horse), services (lessons/boarding/training/trail rides), instructor bios, pricing and packages, facilities tour, show results, lesson booking form, and photo gallery. Classic English-country palette.' },
    ]
  },
  {
    category: 'Events & Entertainment',
    templates: [
      { label: 'Wedding Planner', icon: '💐', desc: 'Packages, portfolio, inquire', prompt: 'Create a wedding planner website with elegant hero (real-wedding photo), planning packages (full/partial/day-of), past weddings gallery by style, vendor network callouts, planning timeline resource, pricing FAQ, consult booking form, and press features. Romantic sophisticated palette.' },
      { label: 'DJ Service', icon: '🎧', desc: 'Events, playlists, booking', prompt: 'Create a wedding/event DJ website with cinematic hero (lit dance floor), event types (weddings/corporate/birthdays/schools), DJ bio with reel video, sample playlists by genre, equipment list with photos, pricing packages, availability checker, and booking form.' },
      { label: 'Photographer', icon: '📸', desc: 'Portfolio, packages, book', prompt: 'Create a photographer website (wedding/portrait/event) with image-first hero, portfolio grid (categories: weddings/portraits/events/brands), about the photographer with personality, packages with deliverables, booking process, FAQ, and inquiry form. Timeless editorial design.' },
      { label: 'Band/Musician', icon: '🎸', desc: 'Music, tour, merch', prompt: 'Create a band website with moody hero (band photo), latest release with streaming links (Spotify/Apple/YouTube/Bandcamp), upcoming tour dates table with ticket links, music videos gallery, merch shop preview, bio and discography, email list signup, and booking inquiries.' },
      { label: 'Comedy Club', icon: '🎤', desc: 'Lineup, tickets, reservations', prompt: 'Create a comedy club website with bold hero (stage spotlight), upcoming shows grid (headliner photo/name/date/time/ticket button), two-drink-minimum info, seating chart and reservation form, open-mic night schedule, past headliners wall, food menu, and newsletter. Dramatic theatrical palette.' },
      { label: 'Theater Company', icon: '🎭', desc: 'Season, tickets, education', prompt: 'Create a theater company website with artistic hero (current production photo), current season lineup, individual show pages preview, ticket subscriptions (3/5/7-show packages), education programs, donate/sponsor, casting auditions, and venue info. Dramatic classical palette.' },
      { label: 'Escape Room', icon: '🔓', desc: 'Rooms, difficulty, book', prompt: 'Create an escape room website with mysterious hero, room grid (title/theme/difficulty/duration/group size/photo), how-it-works steps, group party packages, leaderboard for fastest escapes, waiver form, gift cards, and room booking calendar. Dark mystery palette.' },
      { label: 'Dance Studio', icon: '💃', desc: 'Classes, schedule, teachers', prompt: 'Create a dance studio website with expressive hero (dancer mid-movement), class styles grid (ballet/hip-hop/contemporary/jazz/tap), weekly schedule table, instructor bios, try-a-free-class CTA, student performance photos, recital info, and registration form. Vibrant arts palette.' },
      { label: 'Music School', icon: '🎹', desc: 'Lessons, teachers, recitals', prompt: 'Create a music school website with warm hero (student + teacher at piano), instruments taught grid, teacher profiles with credentials, lesson packages (private/group/online), recital photos, try-a-lesson CTA, tuition rates, and registration. Classic educational palette.' },
      { label: 'Event Venue', icon: '🏛️', desc: 'Spaces, capacity, book', prompt: 'Create an event venue rental website with grand hero (empty space staged), space options (ballroom/garden/rooftop/intimate) each with capacity and photos, preferred vendor list, floor plans, pricing by day of week, virtual tour placeholder, and availability calendar. Elegant luxurious palette.' },
      { label: 'Summer Camp', icon: '🏕️', desc: 'Programs, dates, register', prompt: 'Create a summer camp website with joyful hero (kids at campfire), program tracks by age (day camp, overnight, specialty camps), session dates, daily schedule sample, counselor-in-training program, cost and financial aid, photos from past summers, and registration with early-bird pricing. Bright cheerful palette.' },
    ]
  },
  {
    category: 'Productivity SaaS',
    templates: [
      { label: 'Task Manager SaaS', icon: '✅', desc: 'Landing, features, pricing', prompt: 'Create a task-manager SaaS landing page (Todoist/Asana-style) with hero (app screenshot), problem/solution, 6 feature cards with illustrations, integrations logos, use-case sections (teams/students/freelancers), pricing tiers with 14-day trial, reviews, and CTA.' },
      { label: 'CRM SaaS', icon: '📇', desc: 'Sales pipeline, features', prompt: 'Create a CRM SaaS landing page (HubSpot/Pipedrive-style) with hero (pipeline view screenshot), key features (contact management, deals, email, reports), testimonial video placeholder, integration marketplace preview, pricing tiers, free tier callout, and demo request form.' },
      { label: 'Note-Taking App', icon: '📝', desc: 'Capture, organize, search', prompt: 'Create a note-taking app landing (Notion/Obsidian-style) with hero (app screenshot showing blocks), core concepts (blocks/databases/pages/links), killer features (3 showcases), template gallery preview, mobile app badges, pricing, and community mention. Minimal airy design.' },
      { label: 'Calendar/Scheduling', icon: '📅', desc: 'Calendly-style, integrations', prompt: 'Create a scheduling SaaS landing page (Calendly-style) with hero (booking page mockup), how-it-works 3 steps, features (round-robin, buffers, reminders), calendar integrations logos, use cases (sales, teams, solo), pricing, free version callout, and signup form.' },
      { label: 'Password Manager SaaS', icon: '🔐', desc: 'Security, sync, family', prompt: 'Create a password manager SaaS landing (1Password/Bitwarden-style) with hero (vault screenshot), security emphasis (zero-knowledge, encryption), device sync, autofill, family sharing, compromised password monitor, business plans, free trial, and security audit report link.' },
      { label: 'Team Chat SaaS', icon: '💬', desc: 'Channels, integrations, bots', prompt: 'Create a team chat SaaS landing page (Slack/Discord-style) with hero (app UI screenshot), features (channels/DMs/threads/huddles), integrations gallery, enterprise security, pricing, free tier with limits, testimonial quotes from team leads, and get-started CTA.' },
      { label: 'File Sharing SaaS', icon: '📂', desc: 'Upload, share, sync', prompt: 'Create a file-sharing/cloud-storage SaaS landing (Dropbox-style) with hero (folder view), features (file sharing, sync, version history, comments), security certifications, pricing per user, teams plan, native app badges, and 30-day free trial.' },
      { label: 'Form Builder SaaS', icon: '📋', desc: 'Templates, logic, analytics', prompt: 'Create a form-builder SaaS landing (Typeform-style) with hero (form preview), features (conditional logic, integrations, analytics), template gallery preview, use cases, pricing, free plan callout, customer logos, and create-first-form CTA.' },
      { label: 'Survey Tool SaaS', icon: '📊', desc: 'Templates, results, NPS', prompt: 'Create a survey tool SaaS landing with hero (survey in action), survey types (NPS/CSAT/market-research), advanced logic, beautiful results dashboards screenshot, integrations, HIPAA/GDPR badges, tier pricing, and start-free-survey CTA.' },
      { label: 'Project Management SaaS', icon: '📈', desc: 'Kanban, Gantt, time tracking', prompt: 'Create a project-management SaaS landing (Monday/ClickUp-style) with hero (Kanban board screenshot), views (list/board/calendar/Gantt/timeline), collaboration features, automations, templates, customer stories, pricing tiers, and 14-day trial signup.' },
      { label: 'Email Marketing SaaS', icon: '📧', desc: 'Campaigns, automation, list', prompt: 'Create an email marketing SaaS landing (Mailchimp/ConvertKit-style) with hero (campaign builder screenshot), features (drag-and-drop builder, automations, list growth, analytics), templates, deliverability emphasis, pricing by list size, free tier, and signup.' },
      { label: 'Analytics SaaS', icon: '📡', desc: 'Dashboards, events, integrations', prompt: 'Create an analytics SaaS landing (Mixpanel/Amplitude-style) with hero (dashboard screenshot), product features (events, funnels, retention, cohorts), tracking plan help, privacy-first callout, enterprise features, integration marketplace preview, pricing, and free tier with 1M events.' },
    ]
  },
  {
    category: 'Utility & Tools',
    templates: [
      { label: 'URL Shortener', icon: '🔗', desc: 'Shorten, analytics, custom', prompt: 'Create a URL shortener tool website with simple hero (one input + shorten button), features (custom aliases, QR codes, click analytics, expiration), free vs pro comparison, recent short links table placeholder, browser extension CTA, and API docs link.' },
      { label: 'File Converter', icon: '🔄', desc: 'Drag-drop, formats, batch', prompt: 'Create a file converter tool website with hero (drag-drop zone), format support (PDF/image/audio/video with conversions matrix), batch processing mention, privacy (files deleted after 24h), free file size limit with paid upgrade, recently converted list, and no-signup-required emphasis.' },
      { label: 'QR Code Generator', icon: '🔳', desc: 'Input, customize, download', prompt: 'Create a QR code generator tool website with hero (example QR), type tabs (URL/text/WiFi/vCard/email), customization (color/logo/shape), live preview, download as PNG/SVG, bulk mode for paid, scan statistics for dynamic QRs, and API pricing.' },
      { label: 'PDF Tools', icon: '📄', desc: 'Merge, split, compress, convert', prompt: 'Create a PDF utilities website (SmallPDF-style) with hero, tool grid (merge/split/compress/convert/rotate/edit/sign/unlock), drag-drop workflow, privacy guarantee (files deleted), free tier limits, pro unlimited, team plan, and get-started CTA.' },
      { label: 'Image Compressor', icon: '🖼️', desc: 'Shrink, batch, quality', prompt: 'Create an image compression tool website with hero (before/after file size comparison), drag-drop uploader, batch processing, quality slider preview, format support (JPG/PNG/WebP/AVIF), lossless option, API for developers, and 100% browser-side privacy messaging.' },
      { label: 'Color Palette Tool', icon: '🎨', desc: 'Generate, save, export', prompt: 'Create a color palette generator website with hero (live-generating palette on scroll), generate button, lock colors, save palette, explore curated palettes gallery, extract palette from image upload, export as CSS/Tailwind/PNG, and palettes-of-the-day feed.' },
      { label: 'Icon Library', icon: '✨', desc: 'Browse, download, license', prompt: 'Create an icon library website (Heroicons/Feather-style) with search bar hero, icon grid with live preview, category filter sidebar, stroke/solid/duotone variants, copy-SVG and download buttons, framework integrations (React/Vue/Svelte), and open-source license.' },
      { label: 'Font Pairing', icon: '🔤', desc: 'Combine, preview, Google Fonts', prompt: 'Create a font-pairing tool website with hero (sample pairing), pairing generator (randomize heading + body), Google Fonts integration note, user-curated pairings gallery, save-to-collection, copy CSS/@import, use-case suggestions (editorial/tech/luxury), and typography blog teaser.' },
      { label: 'Typing Test', icon: '⌨️', desc: 'WPM, races, history', prompt: 'Create a typing-speed-test website with big prompt text and input field front-and-center, WPM/accuracy/time displays, test duration selector (15/30/60/120s), custom text paste option, multiplayer race rooms, personal history chart, and global leaderboard. Minimalist monospace design.' },
      { label: 'Calculator Suite', icon: '🧮', desc: 'Mortgage, BMI, tip, etc.', prompt: 'Create a calculator hub website with hero (search calculators), category grid (financial/health/math/conversion), popular calculators with preview (mortgage/loan/BMI/tip/age/percentage), standalone calculator pages linked, recently used, and request-a-calculator form.' },
    ]
  },
  {
    category: 'Automotive & Transportation',
    templates: [
      { label: 'Auto Repair Shop', icon: '🔧', desc: 'Services, certs, appointments', prompt: 'Create an auto repair shop website with trust-heavy hero (clean shop + ASE certified logo), services (oil change/brakes/tires/diagnostics/alignment), transparent pricing, technician bios, warranty info, loaner-car availability, customer reviews, and appointment booking.' },
      { label: 'Car Rental', icon: '🚗', desc: 'Fleet, locations, book', prompt: 'Create a car rental company website with hero (sleek car + city), search bar (pickup/return location/dates), fleet grid by category (economy/SUV/luxury/electric), daily rate and features per car, loyalty program signup, locations map, and booking confirmation flow. Clean modern palette.' },
      { label: 'Bike Shop', icon: '🚲', desc: 'Sales, repair, community', prompt: 'Create a bike shop website with hero (action bike photo), inventory (road/mountain/hybrid/ebike/kids), repair services and turnaround time, group rides calendar, community events, size guide, financing, and test-ride booking. Energetic active palette.' },
      { label: 'Motorcycle Dealer', icon: '🏍️', desc: 'Inventory, service, gear', prompt: 'Create a motorcycle dealership website with bold hero (motorcycle + rider), new and used inventory grid with filters, service center booking, gear/parts shop preview, financing calculator, community rides, rider safety course, and dealership info. Dark bold palette.' },
      { label: 'Rideshare Driver Portal', icon: '🚖', desc: 'Earnings, tips, community', prompt: 'Create a driver-focused rideshare landing page with hero ("Earn on your schedule"), earnings calculator widget, requirements checklist, signup steps, city-by-city availability, driver perks (gas/insurance/tax help), testimonials, and referral program.' },
      { label: 'EV Charging', icon: '🔌', desc: 'Network, stations, app', prompt: 'Create an EV charging network website with hero (sleek charger), live station map with availability, how it works (download app, start charge, pay), network size stat, speed tiers (Level 2 / DC Fast / Ultra), business/fleet solutions, and app download badges. Clean electric-green palette.' },
    ]
  },
  {
    category: 'Kids & Family',
    templates: [
      { label: 'Daycare', icon: '🧸', desc: 'Programs, staff, enroll', prompt: 'Create a daycare/preschool website with warm hero (kids playing safely), programs by age group (infant/toddler/preschool/pre-K), daily schedule sample, meet the teachers, curriculum philosophy (Montessori/Reggio/play-based), tuition and hours, enrollment form, and parent testimonials. Cheerful primary palette.' },
      { label: 'After-School Program', icon: '🎒', desc: 'Activities, pickup, enroll', prompt: 'Create an after-school program website with fun hero (kids doing activities), program offerings (homework help/sports/arts/STEM), pickup schools serviced, daily schedule, staff bios, snacks provided, fees, enrollment form, and parent portal.' },
      { label: 'Pediatrician', icon: '👶', desc: 'Doctors, services, book', prompt: 'Create a pediatric clinic website with gentle hero (smiling family), services (well visits/sick visits/vaccines/lactation), pediatricians bios, new patient welcome, after-hours guidance, same-day sick appointments, insurance accepted, and patient portal.' },
      { label: 'Kids Party Planner', icon: '🎉', desc: 'Themes, packages, book', prompt: 'Create a childrens party planner website with colorful hero (decorated party), theme gallery (superhero/princess/unicorn/safari), package tiers, venue partner info (or at-home option), add-ons (balloon artist/magician/cake), booking form with date picker, and FAQs.' },
      { label: 'Family Photographer', icon: '📷', desc: 'Sessions, gallery, book', prompt: 'Create a family photographer website with warm hero (family photo), session types (newborn/maternity/family/senior), gallery by type, packages with deliverables, styling guide, booking calendar, client portal mention, and client testimonials. Warm editorial palette.' },
      { label: 'Kids Bookstore', icon: '📚', desc: 'Books, storytime, events', prompt: 'Create a childrens bookstore website with magical hero (kids reading), browse by age, staff pick shelves, storytime schedule, author-visit events, book club, gift cards, and online ordering with curbside. Whimsical warm palette.' },
    ]
  },
];

const DESIGN_DIRECTIVES = `DESIGN DIRECTIVES (read carefully — this is the quality bar)
- Do NOT default to: purple gradient hero, Plus Jakarta Sans / Inter / Poppins / Space Grotesk, generic glassmorphism cards, pastel tech-purple accents, gradient text on the H1, stacked feature-icon cards with identical rounded squares. These are AI-slop tells.
- Pick a distinctive typeface PAIR per brief — examples: Fraunces + Inter Tight (editorial), Instrument Serif + Geist (modern editorial), Playfair + Work Sans (classic luxury), Redaction + Mona Sans (brutalist editorial), IBM Plex Serif + IBM Plex Sans (technical), Syne + Archivo (bold display), DM Serif Display + DM Sans (elegant). Load via Google Fonts.
- Commit to ONE dominant color. Avoid evenly-distributed multi-color palettes. Warm beige + one terracotta, deep forest + one gold, off-black + one signal red — pick a register and hold it.
- Use MODERN CSS where it raises quality: container queries (@container), view transitions (view-transition-name), @layer cascade control, backdrop-filter, subgrid, :has() selectors, scroll-driven animations (animation-timeline: scroll()), color-mix(), oklch() colors.
- Respect the brief's vibe. If the brief says "minimal", do not add motion or gradients. If it says "editorial", lean into serif type and asymmetric layout. If it says "brutalist", drop rounded corners and use raw monospace.`;

const QUALITY_BAR = `QUALITY BAR
- Fully responsive — prefer container queries over media queries for component-level responsiveness
- Semantic HTML (main/nav/article/section/aside/footer), ARIA landmarks, visible focus states, prefers-reduced-motion respected
- Real domain-appropriate sample content — never "Lorem ipsum", never empty placeholder sections
- Motion is purposeful: page-load reveal, micro-interactions on hover, NOT everything pulsing and rotating
- Images via https://picsum.photos/<seed>/<w>/<h> or https://images.unsplash.com with meaningful alt text, OR high-quality inline SVG
- Icons via Lucide CDN (not emoji in primary UI)`;

const SYSTEM_PROMPT = `You are a senior web designer building a production-quality landing page in a single HTML file.

OUTPUT FORMAT
- Output ONLY complete HTML starting with <!DOCTYPE html> — no markdown fences, no prose
- All CSS in <style>, all JS in <script>. External libraries via CDN in <head>
- If user asks for modifications, return the COMPLETE updated HTML

ALLOWED LIBRARIES (use what the brief calls for, not everything)
- Google Fonts via <link> — always use a distinctive pair, never browser defaults
- Alpine.js (https://unpkg.com/alpinejs) for reactive behavior (menus, tabs, accordions) without a build step
- GSAP 3 (https://cdn.jsdelivr.net/npm/gsap@3) for polished motion — scroll triggers, timeline choreography
- Lucide icons (https://unpkg.com/lucide@latest) + lucide.createIcons() after DOM ready

${DESIGN_DIRECTIVES}

${QUALITY_BAR}`;

const getSystemPrompt = (tailwind: boolean) => tailwind
  ? `You are a senior web designer building a production-quality landing page in a single HTML file using Tailwind CSS.

OUTPUT FORMAT
- Output ONLY complete HTML starting with <!DOCTYPE html> — no markdown fences, no prose
- Include <script src="https://cdn.tailwindcss.com"></script> in <head>
- Configure custom theme inline via <script>tailwind.config = { theme: { extend: { fontFamily: {...}, colors: {...} } } }</script> BEFORE the CDN script loads (or via the play CDN's config attribute)
- If user asks for modifications, return the COMPLETE updated HTML

ALLOWED LIBRARIES
- Google Fonts via <link> — register a display + body pair, wire them into tailwind.config fontFamily
- Alpine.js for reactive behavior (menus, tabs, disclosure, carousels) without a build step
- GSAP 3 for polished motion when the brief warrants it
- Lucide icons, initialized with lucide.createIcons()

${DESIGN_DIRECTIVES}

TAILWIND-SPECIFIC
- Use arbitrary values [bg-[oklch(...)]] when a design requires it, not just preset palette
- Compose with @apply sparingly inside <style> only for tokens re-used 5+ times
- Prefer group-hover/peer/has-[...] modifiers over JS for simple interactions
- Container queries via @container + @[size]: variants (Tailwind 3.4+ syntax)

${QUALITY_BAR}`
  : SYSTEM_PROMPT;

/* ── Main Component ── */
interface WebCreatorProps {
  creatorMode?: 'web' | 'app' | 'game' | 'dashboard';
  customTemplates?: typeof TEMPLATE_CATEGORIES;
  customSystemPrompt?: string;
}

export function WebCreator({ creatorMode: _mode = 'web', customTemplates, customSystemPrompt }: WebCreatorProps = {}) {
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
  // ─── New features state ───
  const [publishUrl, setPublishUrl] = useState<string | null>(null);
  const [isPublishing, setIsPublishing] = useState(false);
  const [designReview, setDesignReview] = useState<string | null>(null);
  const [isReviewing, setIsReviewing] = useState(false);
  const [showBrandKit, setShowBrandKit] = useState(false);
  const [brandName, setBrandName] = useState(() => localStorage.getItem('aura-brand-name') || '');
  const [brandPrefs, setBrandPrefs] = useState(() => {
    try { return JSON.parse(localStorage.getItem('aura-brand-prefs') || '{}'); } catch { return {}; }
  });
  const [pages, setPages] = useState<{ name: string; html: string }[]>([]);
  const [activePageIndex, setActivePageIndex] = useState(0);
  const [showInspector, setShowInspector] = useState(false);
  const [inspectorData, setInspectorData] = useState<{ tag: string; text: string; styles: Record<string, string> } | null>(null);

  const modelMenuRef = useRef<HTMLDivElement>(null);
  const exportMenuRef = useRef<HTMLDivElement>(null);
  const codeEndRef = useRef<HTMLPreElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const chatScrollRef = useRef<HTMLDivElement>(null);
  const previewIframeRef = useRef<HTMLIFrameElement>(null);
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
      if (e.source !== previewIframeRef.current?.contentWindow) return;
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
    // Initialize pages if first generation
    if (pages.length === 0) {
      setPages([{ name: 'index.html', html }]);
      setActivePageIndex(0);
    }
  }, [pages.length]);

  // Listen for live text edits from the preview iframe
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      if (e.source !== previewIframeRef.current?.contentWindow) return;
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
    const basePrompt = capturedImageData ? IMAGE_SYSTEM_PROMPT : (customSystemPrompt || getSystemPrompt(useTailwind));
    const systemCtx = currentHtml
      ? `${basePrompt}${colorInstruction}${brandInstruction}\n\nCurrent page HTML:\n${currentHtml}`
      : `${basePrompt}${colorInstruction}${brandInstruction}`;

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

  // ─── ONE-CLICK PUBLISH ───
  const handlePublish = useCallback(async () => {
    if (!currentHtml || isPublishing) return;
    setIsPublishing(true);
    setPublishUrl(null);
    try {
      const res = await fetch('/api/share', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_name: 'AURA Creation',
          files: { 'index.html': currentHtml },
          entry_point: 'index.html',
          expires_days: 7,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setPublishUrl(data.url);
      } else {
        setPublishUrl(null);
        setDesignReview(`Publish failed: HTTP ${res.status}`);
      }
    } catch {
      setDesignReview('Publish failed: network error');
    }
    setIsPublishing(false);
  }, [currentHtml, isPublishing]);

  // ─── AI DESIGN REVIEW ───
  const handleDesignReview = useCallback(async () => {
    if (!currentHtml || isReviewing) return;
    setIsReviewing(true);
    setDesignReview(null);
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: `Review this HTML page for design quality issues. Check for: 1) Color contrast (WCAG AA), 2) Mobile responsiveness, 3) Typography hierarchy, 4) Spacing consistency, 5) Missing accessibility attributes. For each issue found, give a one-line description and a severity (critical/warning/info). Be concise — max 6 issues. Format each as: [severity] issue description\n\nHTML:\n${currentHtml.slice(0, 8000)}`,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setDesignReview(data.response || 'No issues found.');
      } else {
        setDesignReview(`Review failed: HTTP ${res.status}`);
      }
    } catch {
      setDesignReview('Review failed: network error');
    }
    setIsReviewing(false);
  }, [currentHtml, isReviewing]);

  // ─── BRAND KIT ───
  const saveBrandKit = useCallback(() => {
    if (!brandName.trim()) return;
    // Extract colors from current HTML
    const colorMatches = currentHtml.match(/#[0-9a-fA-F]{6}/g) || [];
    const uniqueColors = [...new Set(colorMatches)].slice(0, 6);
    const fontMatch = currentHtml.match(/font-family:\s*['"]?([^'";,]+)/);
    const prefs = {
      colors: uniqueColors,
      font: fontMatch?.[1]?.trim() || 'system-ui',
      savedAt: Date.now(),
    };
    setBrandPrefs(prefs);
    localStorage.setItem('aura-brand-name', brandName);
    localStorage.setItem('aura-brand-prefs', JSON.stringify(prefs));
    setShowBrandKit(false);
  }, [brandName, currentHtml]);

  const clearBrandKit = useCallback(() => {
    setBrandName('');
    setBrandPrefs({});
    localStorage.removeItem('aura-brand-name');
    localStorage.removeItem('aura-brand-prefs');
  }, []);

  // Inject brand into system prompt
  const brandInstruction = brandName && brandPrefs.colors?.length
    ? `\n\nUser's brand "${brandName}": use these colors: ${brandPrefs.colors.join(', ')}. Font: ${brandPrefs.font}.`
    : '';

  // ─── MULTI-PAGE ───
  const addPage = useCallback(() => {
    if (currentHtml) {
      // Save current page
      const updated = [...pages];
      updated[activePageIndex] = { name: pages[activePageIndex]?.name || 'index.html', html: currentHtml };
      // Add new blank page
      const newName = `page-${updated.length + 1}.html`;
      updated.push({ name: newName, html: '' });
      setPages(updated);
      setActivePageIndex(updated.length - 1);
      setCurrentHtml('');
      setShowTemplates(true);
    }
  }, [currentHtml, pages, activePageIndex]);

  const switchPage = useCallback((index: number) => {
    // Save current page first
    const updated = [...pages];
    updated[activePageIndex] = { name: pages[activePageIndex]?.name || 'index.html', html: currentHtml };
    setPages(updated);
    setActivePageIndex(index);
    setCurrentHtml(updated[index]?.html || '');
    if (!updated[index]?.html) setShowTemplates(true);
  }, [pages, activePageIndex, currentHtml]);

  // ─── ELEMENT INSPECTOR (from iframe messages) ───
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      if (e.data?.type === 'elementSelected' && showInspector) {
        setInspectorData({
          tag: e.data.tag || 'div',
          text: e.data.text || '',
          styles: {},
        });
      }
    };
    window.addEventListener('message', handler);
    return () => window.removeEventListener('message', handler);
  }, [showInspector]);

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
              {(customTemplates || TEMPLATE_CATEGORIES).map((cat) => {
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

          {/* ─── PUBLISH BUTTON ─── */}
          <button
            onClick={handlePublish}
            disabled={!currentHtml || isPublishing}
            className="flex items-center gap-1 px-2.5 py-1 text-[10px] rounded-lg font-medium disabled:opacity-30 transition-all active:scale-95"
            style={{ background: publishUrl ? 'rgba(52,211,153,0.2)' : 'rgba(124,58,237,0.2)', color: publishUrl ? '#34d399' : '#a78bfa' }}
            title="Publish to live URL"
          >
            {isPublishing ? '...' : publishUrl ? '✓ Live' : '🚀 Publish'}
          </button>
          {publishUrl && (
            <button
              onClick={() => { navigator.clipboard?.writeText(window.location.origin + publishUrl); }}
              className="text-[9px] px-1.5 py-0.5 rounded bg-green-500/10 text-green-400 hover:bg-green-500/20 transition-colors truncate max-w-[100px]"
              title={publishUrl}
            >
              📋 Copy URL
            </button>
          )}

          {/* ─── DESIGN REVIEW ─── */}
          <button
            onClick={handleDesignReview}
            disabled={!currentHtml || isReviewing}
            className="flex items-center gap-1 px-2 py-1 text-[10px] rounded-lg text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
            title="AI design critique"
          >
            {isReviewing ? '⏳' : '🔍'} Review
          </button>

          {/* ─── BRAND KIT ─── */}
          <button
            onClick={() => setShowBrandKit(b => !b)}
            className={`flex items-center gap-1 px-2 py-1 text-[10px] rounded-lg transition-colors ${brandName ? 'text-purple-400 bg-purple-500/10' : 'text-chat-text-secondary hover:text-chat-text'}`}
            title={brandName ? `Brand: ${brandName}` : 'Save brand kit'}
          >
            🎨 {brandName || 'Brand'}
          </button>

          {/* ─── INSPECTOR ─── */}
          <button
            onClick={() => setShowInspector(i => !i)}
            disabled={!currentHtml}
            className={`px-2 py-1 text-[10px] rounded-lg transition-colors disabled:opacity-30 ${showInspector ? 'bg-blue-500/20 text-blue-400' : 'text-chat-text-secondary hover:text-chat-text'}`}
            title="Element inspector"
          >
            🔧
          </button>

          {/* ─── PAGES ─── */}
          {pages.length > 1 && (
            <span className="text-[9px] text-chat-text-tertiary">
              p{activePageIndex + 1}/{pages.length}
            </span>
          )}
          <button
            onClick={addPage}
            disabled={!currentHtml}
            className="px-2 py-1 text-[10px] rounded-lg text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
            title="Add new page"
          >
            📄+
          </button>

          <div className="w-px h-4 bg-chat-border/30 mx-1" />

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

        {/* Design Review Panel */}
        {designReview && (
          <div className="px-3 py-2 border-b border-chat-border text-xs animate-fade-in" style={{ background: 'var(--surface-1)' }}>
            <div className="flex items-center justify-between mb-1.5">
              <span className="font-semibold text-chat-text">🔍 Design Review</span>
              <button onClick={() => setDesignReview(null)} className="text-chat-text-tertiary hover:text-chat-text text-xs">✕</button>
            </div>
            <div className="space-y-1 text-chat-text-secondary whitespace-pre-line leading-relaxed">
              {designReview.split('\n').filter(Boolean).map((line, i) => {
                const isCritical = line.toLowerCase().includes('[critical]');
                const isWarning = line.toLowerCase().includes('[warning]');
                return (
                  <div key={i} className={`flex items-start gap-1.5 ${isCritical ? 'text-red-400' : isWarning ? 'text-amber-400' : 'text-chat-text-secondary'}`}>
                    <span className="flex-shrink-0 mt-0.5">{isCritical ? '🔴' : isWarning ? '🟡' : '🔵'}</span>
                    <span>{line.replace(/\[(critical|warning|info)\]\s*/i, '')}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Brand Kit Modal */}
        {showBrandKit && (
          <div className="px-3 py-3 border-b border-chat-border animate-fade-in" style={{ background: 'var(--surface-1)' }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-chat-text">🎨 Brand Kit</span>
              <button onClick={() => setShowBrandKit(false)} className="text-chat-text-tertiary hover:text-chat-text text-xs">✕</button>
            </div>
            <input
              value={brandName}
              onChange={e => setBrandName(e.target.value)}
              placeholder="Brand name..."
              className="w-full px-2.5 py-1.5 rounded-lg text-xs bg-transparent border border-chat-border text-chat-text mb-2"
            />
            {brandPrefs.colors?.length > 0 && (
              <div className="flex items-center gap-1.5 mb-2">
                <span className="text-[10px] text-chat-text-secondary">Saved:</span>
                {brandPrefs.colors.map((c: string, i: number) => (
                  <span key={i} className="w-4 h-4 rounded-full border border-white/20" style={{ background: c }} title={c} />
                ))}
              </div>
            )}
            <div className="flex gap-2">
              <button
                onClick={saveBrandKit}
                disabled={!brandName.trim() || !currentHtml}
                className="px-3 py-1.5 text-[10px] rounded-lg font-medium disabled:opacity-30 transition-all"
                style={{ background: 'rgba(124,58,237,0.2)', color: '#a78bfa' }}
              >
                Save from Current Page
              </button>
              {brandName && (
                <button onClick={clearBrandKit} className="px-3 py-1.5 text-[10px] rounded-lg text-red-400 hover:bg-red-500/10 transition-colors">
                  Clear
                </button>
              )}
            </div>
          </div>
        )}

        {/* Pages bar */}
        {pages.length > 1 && (
          <div className="flex items-center gap-1 px-3 py-1.5 border-b border-chat-border overflow-x-auto scrollbar-hide" style={{ background: 'var(--surface-2)' }}>
            {pages.map((page, i) => (
              <button
                key={i}
                onClick={() => switchPage(i)}
                className={`px-2 py-1 text-[10px] rounded-lg transition-colors flex-shrink-0 ${i === activePageIndex ? 'bg-chat-accent text-white' : 'text-chat-text-secondary hover:text-chat-text'}`}
              >
                {page.name}
              </button>
            ))}
          </div>
        )}

        {/* Inspector sidebar */}
        {showInspector && inspectorData && (
          <div className="px-3 py-2 border-b border-chat-border animate-fade-in" style={{ background: 'var(--surface-1)' }}>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-semibold text-chat-text">🔧 Inspector: &lt;{inspectorData.tag}&gt;</span>
              <button onClick={() => { setShowInspector(false); setInspectorData(null); }} className="text-chat-text-tertiary hover:text-chat-text text-xs">✕</button>
            </div>
            {inspectorData.text && (
              <div className="text-[10px] text-chat-text-secondary mb-1.5 truncate">"{inspectorData.text}"</div>
            )}
            <div className="flex flex-wrap gap-1.5">
              <button
                onClick={() => setInput(`Change the ${inspectorData.tag} "${inspectorData.text}" to have a different color`)}
                className="px-2 py-1 text-[9px] rounded bg-blue-500/15 text-blue-400 hover:bg-blue-500/25 transition-colors"
              >
                🎨 Color
              </button>
              <button
                onClick={() => setInput(`Make the ${inspectorData.tag} "${inspectorData.text}" larger`)}
                className="px-2 py-1 text-[9px] rounded bg-green-500/15 text-green-400 hover:bg-green-500/25 transition-colors"
              >
                ↕ Size
              </button>
              <button
                onClick={() => setInput(`Add more padding/spacing to the ${inspectorData.tag} "${inspectorData.text}"`)}
                className="px-2 py-1 text-[9px] rounded bg-amber-500/15 text-amber-400 hover:bg-amber-500/25 transition-colors"
              >
                ⬜ Spacing
              </button>
              <button
                onClick={() => setInput(`Remove the ${inspectorData.tag} element that contains "${inspectorData.text}"`)}
                className="px-2 py-1 text-[9px] rounded bg-red-500/15 text-red-400 hover:bg-red-500/25 transition-colors"
              >
                🗑 Delete
              </button>
              <button
                onClick={() => setInput(`Duplicate the ${inspectorData.tag} "${inspectorData.text}" section`)}
                className="px-2 py-1 text-[9px] rounded bg-purple-500/15 text-purple-400 hover:bg-purple-500/25 transition-colors"
              >
                📋 Duplicate
              </button>
            </div>
          </div>
        )}

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
                    ref={previewIframeRef}
                    srcDoc={srcdoc}
                    sandbox="allow-scripts"
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
