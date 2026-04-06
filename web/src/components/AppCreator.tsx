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
];

const APP_SYSTEM_PROMPT = `You are an expert web application developer. Generate a complete, functional single-page web application with HTML, CSS, and JavaScript.

Rules:
- Output ONLY the complete HTML code starting with <!DOCTYPE html>
- Include ALL CSS in a <style> tag
- Include ALL JavaScript in a <script> tag
- The app MUST be fully functional with working JavaScript interactions
- Use localStorage for data persistence where appropriate
- Include proper event handlers, state management, and DOM manipulation
- Add smooth transitions and animations for interactions
- Make it fully responsive (mobile-first)
- Use CSS Grid/Flexbox for layouts
- Add proper error handling and input validation
- Include sample/demo data so the app looks populated
- NO frameworks, NO CDN dependencies (pure HTML/CSS/JS only)
- NO markdown fences, NO explanation text, ONLY the HTML document`;

export function AppCreator() {
  return (
    <WebCreator
      creatorMode="app"
      customTemplates={APP_TEMPLATES}
      customSystemPrompt={APP_SYSTEM_PROMPT}
    />
  );
}
