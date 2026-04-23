/**
 * AppCreator — Build multi-page interactive web apps.
 *
 * Same chat-based creation as WebCreator but with app-specific system prompts
 * that generate routing, state management, and interactive components.
 * Reuses the WebCreator infrastructure via a mode prop.
 */

import { WebCreator } from './WebCreator';

const APP_TEMPLATES = [
  {
    category: 'Full Apps',
    templates: [
      { label: 'Todo App', icon: '✅', desc: 'CRUD, filters, local storage', prompt: 'Create a fully functional todo app with: add/edit/delete tasks, mark complete, filter (all/active/completed), drag-to-reorder, local storage persistence, task counter, clear completed button, and dark/light theme toggle. All in one HTML file with working JavaScript.' },
      { label: 'Kanban Board', icon: '📋', desc: 'Columns, drag-drop, cards', prompt: 'Create a Kanban board app with: 3 columns (To Do, In Progress, Done), draggable cards between columns, add/edit/delete cards, card labels/colors, search filter, and local storage. Include smooth drag animations.' },
      { label: 'Notes App', icon: '📝', desc: 'Rich text, folders, search', prompt: 'Create a notes app with: sidebar with folder list, note list, rich text editor (bold, italic, headings, lists, code), search across notes, create/delete/rename notes, auto-save to local storage, and word count.' },
      { label: 'Chat App UI', icon: '💬', desc: 'Contacts, messages, real-time', prompt: 'Create a WhatsApp-style chat app UI with: contacts sidebar with search, chat window with message bubbles (sent/received), typing indicator, message input with emoji picker, read receipts, online status dots, and unread badges. Add demo messages.' },
      { label: 'Calendar App', icon: '📅', desc: 'Month view, events, reminders', prompt: 'Create a calendar app with: month/week/day views, add/edit/delete events (title, time, color), event dots on dates, event list sidebar, navigate months, today button, and local storage persistence.' },
      { label: 'Expense Tracker', icon: '💰', desc: 'Transactions, charts, budgets', prompt: 'Create an expense tracker app with: add income/expense (amount, category, date, description), transaction list with filters, pie chart by category (CSS-drawn), monthly bar chart, balance card, budget limits per category, and CSV export button.' },
      { label: 'Pomodoro Timer', icon: '🍅', desc: 'Timer, sessions, stats', prompt: 'Create a Pomodoro timer app with: 25min work / 5min break cycle, large circular timer display, start/pause/reset buttons, session counter, task list to work on, notification sound, daily stats, and customizable durations.' },
      { label: 'Habit Tracker', icon: '📊', desc: 'Daily habits, streaks, heatmap', prompt: 'Create a habit tracker app with: add/remove habits, daily checkbox grid (week view), streak counter per habit, GitHub-style contribution heatmap, weekly completion percentage, and local storage. Motivational messages on streaks.' },
    ]
  },
  {
    category: 'Productivity',
    templates: [
      { label: 'Project Manager', icon: '📁', desc: 'Projects, tasks, timeline', prompt: 'Create a project management app with: project cards grid, task list per project with status/priority/assignee, Gantt chart timeline (CSS-drawn), progress bars, add/edit modals, and project overview dashboard with stats.' },
      { label: 'Bookmark Manager', icon: '🔖', desc: 'Collections, tags, search', prompt: 'Create a bookmark manager app with: add URL (auto-extracts title), organize in collections/folders, tag system, search and filter, grid/list view toggle, import/export JSON, and favicon display.' },
      { label: 'Flashcard App', icon: '🃏', desc: 'Decks, flip animation, spaced', prompt: 'Create a flashcard study app with: create decks, add cards (front/back), flip animation on click/tap, mark as know/don\'t know, spaced repetition scheduling, progress bar per deck, and shuffle mode.' },
      { label: 'Password Generator', icon: '🔑', desc: 'Customize, strength, copy', prompt: 'Create a password generator app with: length slider (8-64), toggle options (uppercase, lowercase, numbers, symbols), real-time password display, strength meter with color bar, copy button with animation, password history list, and pronounceable mode.' },
    ]
  },
  {
    category: 'Social & Content',
    templates: [
      { label: 'Social Feed', icon: '📱', desc: 'Posts, likes, comments, stories', prompt: 'Create an Instagram-style social feed with: stories carousel at top, post cards (image, caption, like/comment/share buttons, like count, comments preview), new post button, user profile link, and infinite scroll placeholder. Add demo content.' },
      { label: 'Forum/Reddit', icon: '🗣️', desc: 'Posts, votes, threads, sort', prompt: 'Create a Reddit-style forum with: post list with upvote/downvote arrows and counts, post titles with subreddit tags, sort tabs (Hot/New/Top), comment threads with nesting (3 levels), reply button, and create post form.' },
      { label: 'Markdown Editor', icon: '📄', desc: 'Live preview, toolbar, export', prompt: 'Create a Markdown editor with: split view (editor + live preview), formatting toolbar (headings, bold, italic, links, images, code, lists, tables), word/char count, export as HTML/MD, and fullscreen mode. Support GitHub-flavored markdown.' },
      { label: 'Poll Creator', icon: '📊', desc: 'Create polls, vote, results', prompt: 'Create a poll creator app with: create poll (question + options), share poll link, vote on polls, real-time bar chart results with percentages, total votes count, multiple poll management, and close poll option.' },
    ]
  },
  {
    category: 'Communication',
    templates: [
      { label: 'Email Client', icon: '✉️', desc: 'Inbox, compose, folders', prompt: 'Create an email client UI with: sidebar (inbox/sent/drafts/spam/trash folders with unread counts), email list (sender, subject, preview, time, star toggle, checkbox), email detail pane (from/to/subject/body, reply/forward buttons), compose modal (to/cc/bcc, subject, rich text body, send button), and search bar. Populate with 10 demo emails.' },
      { label: 'Video Chat', icon: '📹', desc: 'Grid view, controls, chat', prompt: 'Create a video chat UI (Zoom-like) with: participant grid (4-6 video placeholders with names and mute indicators), bottom toolbar (mute/unmute, video on/off, screen share, chat toggle, reactions, leave), side chat panel with messages, participant list, and meeting info header (title, duration timer, recording indicator).' },
      { label: 'Discord Clone', icon: '💬', desc: 'Servers, channels, voice', prompt: 'Create a Discord-style chat app with: server sidebar (server icons), channel list sidebar (#general, #random, voice channels), message area (user avatars, messages, timestamps, reactions), message input with emoji/gif/file attach, user list sidebar, and online/offline status indicators. Dark theme, demo data.' },
    ]
  },
  {
    category: 'Media & Content',
    templates: [
      { label: 'Music Player', icon: '🎵', desc: 'Now playing, queue, playlists', prompt: 'Create a music player app with: now playing view (large album art, song/artist, progress bar with seek, play/pause/skip/shuffle/repeat controls, volume slider), playlist sidebar (create/delete playlists), song queue, library view (grid of albums), and search. Dark Spotify-style theme. Add 8 demo songs.' },
      { label: 'Video Gallery', icon: '📺', desc: 'Grid, player, categories', prompt: 'Create a video streaming app UI with: hero banner (featured video), content rows with horizontal scroll (trending, recommended, recently added), video cards (thumbnail with duration badge, title, views, date), video player page (large player, title, description, related videos sidebar), and category filter tabs.' },
      { label: 'Photo Gallery', icon: '📷', desc: 'Grid, lightbox, albums', prompt: 'Create a photo gallery app with: masonry grid layout, click to open lightbox (full image, prev/next navigation, close), album sidebar (create albums, drag photos to organize), upload button with drag-drop zone, photo info panel (date, size, tags), and grid/list view toggle. Add 12 placeholder images.' },
      { label: 'Recipe Book', icon: '🍳', desc: 'Recipes, ingredients, timer', prompt: 'Create a recipe app with: recipe grid (food photos, name, prep time, difficulty badge), recipe detail (full photo, ingredients list with checkbox, step-by-step instructions with timer buttons per step, servings adjuster), add recipe form, search by ingredient, and meal planner calendar view. Warm cream/orange theme.' },
    ]
  },
  {
    category: 'Utility',
    templates: [
      { label: 'Weather App', icon: '🌤️', desc: 'Current, forecast, location', prompt: 'Create a weather app with: current conditions (city, temp large, condition icon, feels like, humidity, wind), hourly forecast horizontal scroll, 7-day forecast list, UV/pressure/visibility extras, location search bar, and dynamic gradient background that changes with weather (sunny=warm, rainy=cool, night=dark). Add demo data for 3 cities.' },
      { label: 'File Manager', icon: '📁', desc: 'Browse, upload, organize', prompt: 'Create a file manager app with: sidebar (folders tree, favorites, storage bar), breadcrumb navigation, file grid/list toggle view, file cards (icon by type, name, size, date), right-click context menu (rename/delete/move/copy), drag-drop to move, multi-select with checkboxes, upload button, and search. Light clean theme.' },
      { label: 'Calculator Pro', icon: '🔢', desc: 'Scientific, history, converter', prompt: 'Create a scientific calculator app with: standard mode (0-9, +/-/*/÷, =), scientific mode toggle (sin/cos/tan/log/ln/sqrt/pi/e/power/factorial), calculation history sidebar, unit converter (length/weight/temperature tabs), parentheses support, keyboard input, and dark theme with big LCD-style display.' },
      { label: 'Fitness Tracker', icon: '💪', desc: 'Workouts, stats, goals', prompt: 'Create a fitness tracker app with: dashboard (daily steps ring, calories, active minutes, distance), workout log (add: type, duration, calories), weekly bar chart, exercise library (filter by muscle group), personal records table, goal setting with progress bars, and streak calendar heatmap. Green health theme.' },
    ]
  },
  {
    category: 'Finance & Money',
    templates: [
      { label: 'Budget Planner', icon: '💰', desc: 'Income, categories, forecasts', prompt: 'Create a personal budget app with: monthly income/expense summary cards, category-wise spending with budget limits and progress bars, add transaction form with autocomplete categories, forecast-vs-actual comparison, savings goal trackers, recurring bill reminders list, and CSV export. Store everything in localStorage. Clean minimal theme.' },
      { label: 'Crypto Wallet UI', icon: '₿', desc: 'Balances, send/receive, history', prompt: 'Create a crypto wallet UI (no actual blockchain calls — mock data). Features: total portfolio value card, individual token balances list (ETH, BTC, USDC, etc. with icons), send/receive modal with address input and QR placeholder, transaction history table (in/out, amount, status, timestamp), price chart for selected token, and network selector dropdown. Dark cyber theme.' },
      { label: 'Tip Calculator', icon: '🧮', desc: 'Split bill, percentage, per person', prompt: 'Create a tip calculator app with: bill amount input (numeric keypad style), tip percentage selector (15/18/20/custom slider), number of people stepper, real-time per-person total display (large), total tip card, round-up toggle, and recent calculations history. Playful bright theme.' },
      { label: 'Loan Calculator', icon: '🏦', desc: 'Amortization, payments, comparison', prompt: 'Create a loan/mortgage calculator with: loan amount, interest rate, and term inputs, monthly payment large display, amortization schedule table (month/principal/interest/balance), principal-vs-interest pie chart (total over loan), payoff date card, extra payment scenarios comparison, and loan summary export.' },
      { label: 'Currency Converter', icon: '💱', desc: 'Rates, favorites, history', prompt: 'Create a currency converter app with: from/to currency dropdowns (40+ currencies with flags), amount input with live conversion, swap button, rate chart (last 30 days — use fake data), favorites/recent pairs list, and offline mode with cached rates in localStorage. Note: use static demo rates, no API.' },
      { label: 'Expense Splitter', icon: '👥', desc: 'Group trips, who owes whom', prompt: 'Create a group-expense splitter app (Splitwise-like). Features: create group with members, add expense with payer and split method (equal/by percentage/by shares), running balances per person, "who owes whom" settlement summary, expense log, and export as text summary. Local storage persistence.' },
      { label: 'Subscription Tracker', icon: '📇', desc: 'Services, renewals, total', prompt: 'Create a subscription tracker app with: subscription list (Netflix/Spotify/etc. with brand colors), next renewal date per item, monthly/yearly total cost cards, upcoming renewals calendar, category breakdown chart, cancel-candidate suggestions (unused for 30 days mock logic), and CSV export.' },
      { label: 'Invoice Generator', icon: '📄', desc: 'Client, items, PDF preview', prompt: 'Create an invoice generator app with: sender/client info fields, editable line items table (description/quantity/rate/amount auto-calc), tax and discount rows, invoice number auto-increment, preview pane styled like a real invoice, download as PDF (use window.print), and invoice history in localStorage.' },
      { label: 'Net Worth Tracker', icon: '📈', desc: 'Assets, liabilities, trends', prompt: 'Create a net worth tracking app with: add assets (cash/investments/property) and liabilities (loans/credit cards), auto-calculated net worth card, monthly history line chart, asset allocation donut, liability-to-asset ratio gauge, milestone tracker, and monthly snapshot saving.' },
      { label: 'Receipt Scanner Demo', icon: '🧾', desc: 'Upload, parse, categorize', prompt: 'Create a receipt scanner UI demo. Features: upload/drag-drop receipt image, simulated OCR extraction display (merchant, date, items, total — use fake parsed data), editable fields to correct, auto-category suggestion, save to expense list, and receipt thumbnail gallery view. Note: no real OCR, just demonstrates the UI.' },
    ]
  },
  {
    category: 'Developer Tools',
    templates: [
      { label: 'JSON Formatter', icon: '{ }', desc: 'Pretty-print, validate, diff', prompt: 'Create a JSON formatter/validator app with: large textarea for input, format button (pretty-print with 2-space indent), minify button, validate with error line highlighting, collapsible tree view of formatted output, copy button, and syntax highlighting (color keys, strings, numbers differently). Dev dark theme with monospace font.' },
      { label: 'Regex Tester', icon: '🔎', desc: 'Pattern, test string, matches', prompt: 'Create a regex tester app with: pattern input field with flag toggles (g/i/m/s/u), test string textarea, live match highlighting, match list sidebar with groups, substitution field with live replace preview, common regex cheat sheet panel, and save-patterns feature. Monospace dev theme.' },
      { label: 'Color Tool', icon: '🎨', desc: 'Picker, palette, contrast', prompt: 'Create a color toolkit app with: color picker (hue slider + saturation/brightness square), HEX/RGB/HSL display, palette generator (analogous/complementary/triadic modes, 5 colors each), WCAG contrast checker with pass/fail badges, copy button per format, saved palettes in localStorage, and gradient builder tab.' },
      { label: 'Base64 Encoder', icon: '🔐', desc: 'Text/file encode/decode', prompt: 'Create a base64 encoder/decoder app with: input textarea, encode/decode mode toggle, live output, file upload for binary-to-base64, data URL builder for images (paste data URL to see preview), copy button, and URL-safe base64 option. Minimal dev theme.' },
      { label: 'JWT Decoder', icon: '🔑', desc: 'Header, payload, verify', prompt: 'Create a JWT decoder app with: token textarea input, header and payload JSON viewers (formatted, syntax colored), expiration countdown if exp claim present, signature verification hint (client-side check with HS256 + secret input), and example token button. Pure JS, no backend.' },
      { label: 'Cron Builder', icon: '⏰', desc: 'Visual schedule, next runs', prompt: 'Create a cron expression builder app with: tab selectors for minute/hour/day/month/weekday (with preset buttons: every 5 min, hourly, daily, weekly), live cron expression display, human-readable translation ("Every day at 9:00 AM"), next 5 run times preview, and quick templates (backup, cleanup, report).' },
      { label: 'API Tester', icon: '📡', desc: 'Request builder, response view', prompt: 'Create an API tester app (Postman-like) with: HTTP method selector (GET/POST/PUT/DELETE), URL input, tabs for Headers/Params/Body (JSON editor), Send button, response panel (status code colored, time, size, headers view, pretty JSON body), and request history sidebar. Real fetch call with CORS warning banner.' },
      { label: 'UUID Generator', icon: '🆔', desc: 'v4, v1, bulk, copy', prompt: 'Create a UUID generator app with: generate button (creates v4 UUID), bulk mode (generate 10/50/100 at once), format selector (standard, no-hyphens, uppercase, URN), copy-all and copy-one buttons, history with timestamp, and nanoid alternative generator tab. Clean dev theme.' },
      { label: 'Markdown Preview', icon: '📝', desc: 'Editor, live render, export', prompt: 'Create a split-pane markdown editor with: left editor (monospace, line numbers), right preview (rendered HTML with GFM tables/task lists/code fences), sync scrolling, toolbar (bold/italic/heading/link/image/code), word count, export as HTML/MD download, and template presets (README, blog post, meeting notes).' },
      { label: 'SQL Playground', icon: '🗄️', desc: 'SQLite in browser, schema, results', prompt: 'Create a SQL playground app using sql.js from CDN. Features: schema tree sidebar, SQL editor with syntax highlighting, Run button, results table with pagination, sample database loader (northwind/chinook), saved queries list, export results as CSV, and error panel. Developer dark theme.' },
      { label: 'Diff Viewer', icon: '↔️', desc: 'Side-by-side, inline, unified', prompt: 'Create a text/code diff viewer app with: two large input textareas, view mode toggle (split/unified/inline), line-level diff with green/red highlighting, line numbers, ignore-whitespace toggle, copy-diff-as-patch button, and sample data button. Monospace dev theme.' },
    ]
  },
  {
    category: 'Creative Tools',
    templates: [
      { label: 'Drawing Pad', icon: '🖌️', desc: 'Canvas, brushes, layers', prompt: 'Create a drawing app using Canvas with: brush tool with size slider (1-50px) and color picker, eraser, fill bucket, shape tools (line/rect/circle), undo/redo (20 steps), multiple layers panel with visibility toggles, clear canvas, save as PNG download, and pressure-sensitive stroke width on supported devices.' },
      { label: 'Meme Maker', icon: '😂', desc: 'Templates, captions, download', prompt: 'Create a meme maker app with: template gallery (20 built-in emoji-based "memes" in CSS art), top and bottom caption inputs with live preview, font style selector (Impact/Comic Sans/Arial), text stroke toggle, add custom image upload option, download as PNG, and share to clipboard button.' },
      { label: 'Logo Maker', icon: '🏷️', desc: 'Shapes, text, colors, export', prompt: 'Create a simple logo maker app with: shape library (circle/square/triangle/star/hexagon/20 SVG icons), text input with font family and size, color picker for text and shape, layer ordering (bring forward/send back), simple effects (shadow/outline), transparent background toggle, and SVG/PNG export.' },
      { label: 'Collage Maker', icon: '🖼️', desc: 'Grid layouts, upload, arrange', prompt: 'Create a photo collage maker app with: 8 grid layout presets (2x2, 3x3, magazine-style), drag-and-drop image upload, per-slot image cropping with zoom, background color/gradient picker, border width slider, round-corner toggle, text overlay, and download as high-res PNG.' },
      { label: 'Mandala Maker', icon: '✨', desc: 'Symmetry painter, radial', prompt: 'Create a radial mandala drawing app with: canvas that mirrors strokes across 6/8/12/16 symmetry axes (toggle), brush size and color picker, opacity slider, background color, auto-rotation toggle (canvas spins while drawing), undo, clear, and save as PNG.' },
      { label: 'ASCII Art Generator', icon: '🔤', desc: 'Text-to-ASCII, image-to-ASCII', prompt: 'Create an ASCII art generator app with two modes. Text mode: figlet-style big text with 6 font styles. Image mode: upload image, convert to ASCII at configurable resolution (coarse/medium/fine), character set selector (@#*+=~:-., ), invert colors toggle, copy-to-clipboard button, and download as .txt.' },
      { label: 'Gradient Generator', icon: '🌈', desc: 'Linear, radial, CSS export', prompt: 'Create a CSS gradient generator with: type selector (linear/radial/conic), color stops (add/remove/drag to reorder), angle slider for linear, full-canvas live preview, CSS code output with copy button, Tailwind class equivalent, and preset gallery (30 curated gradients).' },
      { label: 'SVG Icon Lab', icon: '💎', desc: 'Compose, edit, export', prompt: 'Create an SVG icon editor app with: starter icon library (50 common icons), color/stroke-width controls, rotate/flip/scale transforms, combine two icons (union/subtract visual), export as SVG or PNG at 2x/4x, and copy as React/JSX snippet.' },
      { label: 'Pattern Maker', icon: '🔳', desc: 'Tileable patterns, SVG export', prompt: 'Create a repeating pattern generator with: base shape selector (dots/stripes/chevron/waves/hex/triangles/plaid), size slider, rotation, color pickers (primary/background), opacity, tile preview (repeated across canvas), and export as SVG or PNG tileable image for use as backgrounds.' },
    ]
  },
  {
    category: 'Lifestyle & Wellness',
    templates: [
      { label: 'Mood Journal', icon: '😊', desc: 'Daily check-in, trends, tags', prompt: 'Create a daily mood journal app with: today emoji mood selector (5 options), free-text reflection field, energy level slider, gratitude list (3 items), tags/activities multi-select, streak counter, 30-day mood chart (line with emoji markers), and localStorage persistence. Calming pastel theme.' },
      { label: 'Sleep Tracker', icon: '😴', desc: 'Bedtime, wake, quality', prompt: 'Create a sleep tracking app with: log entry form (bedtime/wake time/quality 1-10/dreams), auto-calculated duration, weekly bar chart of hours slept, ideal-sleep-time goal line, mood correlation chart, bedtime routine checklist, and smart-alarm time suggestion.' },
      { label: 'Water Intake', icon: '💧', desc: 'Goal, reminders, log', prompt: 'Create a water intake tracker app with: large central progress ring (current oz/ml vs daily goal), tap-to-add buttons (4oz/8oz/12oz/16oz/custom), hourly reminder setup, 7-day history bars, streak counter, hydration tips carousel, and unit toggle (oz/ml). Playful blue theme.' },
      { label: 'Gratitude Journal', icon: '🌻', desc: '3 things a day, calendar, export', prompt: 'Create a gratitude journal app with: daily "3 things Im grateful for" form, previous entries in reverse-chronological feed, calendar heatmap of days completed, monthly review showing most common words, share-as-image feature (generates beautiful quote card), and encryption-in-mind UI (privacy-first copy).' },
      { label: 'Meditation Timer', icon: '🧘', desc: 'Breath, bell, sessions', prompt: 'Create a meditation timer app with: large breath animation circle (breath-in/hold/out/hold with configurable timings — 4-7-8, box breathing presets), session length selector (5/10/20/30 min), interval bells (every 1/2/5 min), ambient sound options (rain/forest/bell — Web Audio API oscillators), session history log, and streak tracker.' },
      { label: 'Period Tracker', icon: '🩸', desc: 'Cycle, symptoms, predictions', prompt: 'Create a menstrual cycle tracker app with: calendar view showing period/fertile/ovulation days (color coded), log symptoms (cramps/mood/flow/notes) per day, cycle length statistics, next period prediction, symptom pattern chart, and private/password-protected option. Soft rose theme.' },
      { label: 'Goal Tracker', icon: '🎯', desc: 'SMART goals, milestones, streak', prompt: 'Create a goal-tracking app with: add SMART goal (specific/measurable/target date/why), milestone subtasks with checkboxes, progress % calculated from milestones, daily check-in, streak counter, timeline view of all goals, achievements archive, and motivational quote on completion.' },
      { label: 'Bucket List', icon: '🗺️', desc: 'Dreams, photos, locations', prompt: 'Create a bucket list app with: add item (title/category/location/target year/notes), grid of items with status (dreaming/planning/done), photo upload per item, category filters (travel/skill/experience/relationships), completion stats card, inspiration gallery, and share-list-as-image option.' },
      { label: 'Reading Log', icon: '📚', desc: 'Books, progress, goals', prompt: 'Create a personal reading tracker with: add book (title/author/ISBN/pages/genre/status), shelves (reading/to-read/finished), current reading with progress bar (% complete), annual reading goal tracker (books vs target), genre distribution donut, star ratings, notes/quotes per book, and Goodreads-style yearly recap.' },
      { label: 'Recipe Box', icon: '🍽️', desc: 'Saved recipes, meal plan, grocery', prompt: 'Create a personal recipe collection app with: add recipe (name/ingredients/steps/prep time/servings/photo URL/tags), filter/search recipes, weekly meal planner (drag recipes onto day slots), auto-generated grocery list from planned meals, print meal plan, and category filters (breakfast/lunch/dinner/snack).' },
    ]
  },
  {
    category: 'Education & Learning',
    templates: [
      { label: 'Flashcard Study', icon: '🗂️', desc: 'Decks, spaced repetition, stats', prompt: 'Create a flashcard study app with: create/edit decks, cards with front/back, flip animation on click, rate-your-answer buttons (again/hard/good/easy implementing SM-2 spaced repetition), review queue sorted by due date, streak counter, accuracy stats per deck, and import/export JSON.' },
      { label: 'Language Learning', icon: '🗣️', desc: 'Vocab, sentences, quiz', prompt: 'Create a language learning app (Duolingo-lite) with: vocabulary list per language (built-in: Spanish, French, German with 30 common words each), multiple-choice vocab quizzes, fill-in-the-blank sentence builder, daily streak tracker, XP points, leveling up, and hearts/lives system. Playful game-like design.' },
      { label: 'Typing Tutor', icon: '⌨️', desc: 'Lessons, WPM, accuracy', prompt: 'Create a typing tutor app with: beginner to advanced lessons (home row, top row, bottom row, numbers, full keyboard), live highlight of next key on virtual keyboard, WPM and accuracy tracking, lesson progress unlock system, 1/3/5 minute tests, mistakes analysis, and personal best history.' },
      { label: 'Math Practice', icon: '➗', desc: 'Problems, timer, levels', prompt: 'Create a math practice app with: operation selector (+/-/×/÷/mixed), difficulty levels (1-digit/2-digit/3-digit), timed mode (60s sprint) and untimed mode, on-screen number pad, instant correct/wrong feedback, score tracking, adaptive difficulty, and age-range presets.' },
      { label: 'Coding Exercises', icon: '👨‍💻', desc: 'Prompts, editor, tests', prompt: 'Create a coding practice app with: problem library (20 JS exercises: fizzbuzz, fibonacci, palindrome, etc. with difficulty tags), Monaco-style editor (simple textarea with monospace + indent), visible test cases, Run button (eval in sandbox — with warning), solution stats, hints reveal, and personal solve history.' },
      { label: 'Mind Map', icon: '🧠', desc: 'Nodes, connections, export', prompt: 'Create a mind-mapping app with: central topic node, click to add child nodes, drag-to-reposition, edit text inline, color per branch, connecting curves rendered with SVG, zoom/pan canvas, save/load as JSON, and export as PNG via html2canvas or similar approach.' },
      { label: 'Citation Generator', icon: '📖', desc: 'APA/MLA/Chicago, BibTeX', prompt: 'Create a citation generator app with: source-type selector (book/journal/website/video), input fields (author/title/year/publisher/URL), format selector (APA 7th/MLA 9th/Chicago/Harvard/BibTeX), live formatted citation output with copy button, saved citations list, and bibliography export.' },
      { label: 'Periodic Table', icon: '⚗️', desc: 'Elements, properties, filter', prompt: 'Create an interactive periodic table app with: full 118-element grid (color-coded by category: metal/non-metal/metalloid/noble gas), click element to open detail panel (atomic mass, electron config, discovery, uses, state at room temp), filter by category, compare two elements, and quiz mode.' },
      { label: 'World Map Quiz', icon: '🌍', desc: 'Countries, capitals, flags', prompt: 'Create a geography quiz app with: multiple quiz modes (flag-to-country, country-to-capital, capital-to-country, locate-on-map using SVG world map), difficulty by continent, score tracking, timer optional, progress bar, and review mistakes at end. Use emoji flags.' },
    ]
  },
];

const APP_SYSTEM_PROMPT = `You are a senior app developer building a production-quality single-page web application in a single HTML file.

OUTPUT FORMAT
- Output ONLY complete HTML starting with <!DOCTYPE html> — no markdown fences, no prose
- External libraries via CDN in <head>. All application code inline.
- If user asks for modifications, return the COMPLETE updated HTML

PICK A TRACK BASED ON APP COMPLEXITY

Track A — React (for anything with >3 components, shared state, routing, or forms with validation):
  <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <script type="text/babel" data-presets="env,react">
    // Components, hooks, context as normal JSX
    const { useState, useEffect, useMemo, useReducer, useRef, useCallback } = React;
    // ... your app ...
    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(<App />);
  </script>
  Write real components. Use hooks properly. Custom hooks for reusable logic. Error boundary for the tree. Keyboard shortcuts via useEffect + keydown listener.

Track B — Alpine.js (for smaller apps: one or two views, mostly view-state — dropdowns, tabs, toggles, simple forms):
  <script src="https://unpkg.com/alpinejs" defer></script>
  <script src="https://cdn.tailwindcss.com"></script>
  Use x-data for reactive state, x-show/x-if, x-for, x-model, Alpine.store() for cross-component state.

PERSISTENCE
- For small state (< 5MB — settings, preferences): localStorage
- For larger app data (notes, photos, large lists, offline-first): localforage (https://cdn.jsdelivr.net/npm/localforage@1) — IndexedDB with an async localStorage-compatible API
- Always handle load/save errors. Corrupt JSON shouldn't crash the app.

UX REQUIREMENTS (non-negotiable)
- Empty states that teach — when the list is empty, show an illustration + one clear CTA, not just blank space
- Loading skeletons for async work, not spinners
- Optimistic updates where the UI implies mutation (add-to-list appears instantly, rolls back on failure)
- Keyboard shortcuts for common actions (cmd/ctrl+K for search, / to focus input, ? to show help, Esc to close modals) — list them in a help modal
- Confirmation for destructive actions (use a confirm modal, not window.confirm)
- Toast/snackbar for background success/error, inline errors next to failing fields
- Animate additions/deletions (view-transition-name, or React's FLIP via GSAP, or Alpine transitions)
- Focus management: focus the input when a modal opens, restore focus on close
- Dark mode toggle respecting prefers-color-scheme on first load, with a manual override persisted to localStorage

DESIGN DIRECTIVES (quality bar — literal, not guidance)
- Do NOT default to: purple gradient accent, Plus Jakarta Sans / Inter / Poppins, glassmorphism everywhere, pastel tech palette, emoji-heavy UI. These are AI-slop tells.
- Pick a distinctive typeface pair — examples: Geist + Geist Mono (technical), IBM Plex Sans + IBM Plex Mono, Inter Tight + JetBrains Mono, DM Sans + DM Mono. Load via Google Fonts.
- One dominant color. Apps feel calmer with restraint than with rainbow status badges.
- Use modern CSS: container queries, :has(), subgrid, view-transition-name on list items, color-mix(), oklch() colors for accurate color math
- Respect the brief. "Minimal" means fewer borders, more whitespace, no drop-shadows.

QUALITY BAR
- Fully responsive (container queries preferred over breakpoints)
- Real keyboard support: Tab order, focus rings, arrow keys where lists imply it (up/down through items)
- Semantic HTML (button for buttons, not div), ARIA live regions for toasts, role="dialog" for modals with aria-modal
- Real domain-appropriate demo data so the app looks populated on first load (5-12 realistic items per list, not "Item 1/Item 2/Item 3")
- Performance: virtualize lists over 200 items, debounce search inputs, avoid layout thrashing on scroll
- Error handling at boundaries: try/catch around JSON.parse, handle fetch failures, guard against missing localStorage (private mode)`;

export function AppCreator() {
  return (
    <WebCreator
      creatorMode="app"
      customTemplates={APP_TEMPLATES}
      customSystemPrompt={APP_SYSTEM_PROMPT}
    />
  );
}
