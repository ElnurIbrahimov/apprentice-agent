/**
 * DashboardCreator — Build data dashboards with AI.
 *
 * Generates interactive dashboards with charts, tables, and KPIs
 * using Chart.js CDN and vanilla JS.
 */

import { WebCreator } from './WebCreator';

const DASHBOARD_TEMPLATES = [
  {
    category: 'Analytics',
    templates: [
      { label: 'Marketing Dashboard', icon: '📈', desc: 'Traffic, campaigns, conversions', prompt: 'Create a marketing analytics dashboard with: KPI cards (visitors, leads, conversion rate, revenue), traffic sources pie chart, campaign performance bar chart, conversion funnel visualization, top pages table, and date range selector. Use Chart.js from CDN. Dark theme with purple accents.' },
      { label: 'Sales Dashboard', icon: '💵', desc: 'Revenue, pipeline, team', prompt: 'Create a sales dashboard with: revenue KPI card with trend arrow, monthly revenue line chart, sales pipeline (stage bars), top salespeople leaderboard, deals closing soon table, win rate gauge, and regional sales map placeholder. Use Chart.js. Professional blue theme.' },
      { label: 'Social Media', icon: '📱', desc: 'Followers, engagement, posts', prompt: 'Create a social media dashboard with: follower count cards per platform (Twitter, Instagram, YouTube, TikTok with brand colors), engagement rate chart, post performance table, best posting times heatmap grid, audience demographics donut chart, and content calendar view.' },
      { label: 'SEO Dashboard', icon: '🔍', desc: 'Rankings, traffic, keywords', prompt: 'Create an SEO dashboard with: organic traffic line chart, keyword rankings table (position, change, volume), top pages by traffic, backlink count card, domain authority gauge, Core Web Vitals scores (LCP, FID, CLS), and competitor comparison bar chart.' },
      { label: 'E-commerce', icon: '🛒', desc: 'Orders, revenue, products', prompt: 'Create an e-commerce dashboard with: today\'s orders/revenue/avg order cards, revenue line chart (30 days), top products table, order status breakdown (pending/shipped/delivered pie chart), customer map, inventory alerts, and recent orders feed.' },
    ]
  },
  {
    category: 'Operations',
    templates: [
      { label: 'DevOps Monitor', icon: '🖥️', desc: 'Servers, deploys, alerts', prompt: 'Create a DevOps monitoring dashboard with: server status cards (CPU, RAM, Disk with gauge charts), deployment history timeline, active alerts list with severity colors, response time chart, error rate graph, and uptime percentage badges. Dark terminal theme.' },
      { label: 'Project Tracker', icon: '📊', desc: 'Tasks, burndown, team load', prompt: 'Create a project tracking dashboard with: sprint burndown chart, task status columns (todo/in-progress/review/done counts), team workload bars per member, velocity trend chart, blocked items list, and upcoming deadlines calendar. Clean minimal theme.' },
      { label: 'Support Tickets', icon: '🎫', desc: 'Queue, SLA, satisfaction', prompt: 'Create a customer support dashboard with: open tickets count, average response time, SLA compliance percentage gauge, ticket volume line chart (by hour), category breakdown donut chart, agent performance table, CSAT score card, and escalated tickets list.' },
      { label: 'IoT Sensors', icon: '🌡️', desc: 'Real-time, gauges, alerts', prompt: 'Create an IoT sensor dashboard with: 6 sensor cards (temperature, humidity, pressure, air quality, noise, light) with current value, gauge visualization, and status indicator. Include time-series chart for selected sensor, alert thresholds, device map grid, and connection status badges.' },
      { label: 'Supply Chain', icon: '🚚', desc: 'Inventory, shipments, vendors', prompt: 'Create a supply chain dashboard with: inventory levels bar chart by warehouse, shipment tracking map placeholder, vendor performance radar chart, order fulfillment rate gauge, stock alerts table (low/critical), delivery timeline, and cost breakdown pie chart.' },
    ]
  },
  {
    category: 'Finance',
    templates: [
      { label: 'Investment Portfolio', icon: '💰', desc: 'Holdings, returns, allocation', prompt: 'Create an investment portfolio dashboard with: total value card with daily change, portfolio allocation donut chart, holdings table (symbol, shares, price, gain/loss, % change), performance line chart (1M/3M/1Y/All), sector exposure bar chart, and dividend income tracker.' },
      { label: 'Crypto Tracker', icon: '₿', desc: 'Prices, portfolio, charts', prompt: 'Create a crypto portfolio dashboard with: top 10 coins table (price, 24h change, market cap), portfolio balance card, holdings pie chart, price chart with candlestick placeholder, fear & greed index gauge, gas tracker, and watchlist with price alerts.' },
      { label: 'Budget Planner', icon: '🏦', desc: 'Income, expenses, savings', prompt: 'Create a personal budget dashboard with: monthly income vs expenses bar chart, spending by category donut chart, savings goal progress bars (3 goals), upcoming bills table, net worth trend line, cash flow waterfall chart, and budget vs actual comparison.' },
      { label: 'HR Analytics', icon: '👥', desc: 'Headcount, turnover, hiring', prompt: 'Create an HR analytics dashboard with: headcount by department bar chart, turnover rate trend line, open positions count, time-to-hire gauge, employee satisfaction score, diversity breakdown charts, hiring pipeline funnel, and upcoming reviews table.' },
    ]
  },
  {
    category: 'Specialized',
    templates: [
      { label: 'Fitness Stats', icon: '🏃', desc: 'Workouts, calories, progress', prompt: 'Create a fitness dashboard with: weekly activity ring (steps/calories/exercise), workout log table, weight trend line chart, personal records cards (bench, squat, deadlift), muscle group distribution radar chart, streak counter, and weekly goal progress bars.' },
      { label: 'Weather Station', icon: '🌤️', desc: 'Current, forecast, history', prompt: 'Create a weather dashboard with: current conditions card (temp, feels like, humidity, wind), 24-hour forecast chart, 7-day forecast cards, UV index gauge, air quality index, precipitation radar placeholder, sunrise/sunset times, and historical comparison chart.' },
      { label: 'Learning Progress', icon: '🎓', desc: 'Courses, skills, streaks', prompt: 'Create a learning dashboard with: courses in progress cards with completion %, skill radar chart (programming, design, data, etc.), daily study time bar chart, streak counter with heatmap, certifications earned grid, recommended courses, and XP/level system.' },
      { label: 'Content Creator', icon: '🎬', desc: 'Views, subscribers, revenue', prompt: 'Create a content creator dashboard with: subscriber count with growth trend, views line chart, top videos table, revenue card with RPM, audience retention curve, demographics donut chart, upload schedule calendar, and comment sentiment breakdown.' },
    ]
  },
  {
    category: 'Industry',
    templates: [
      { label: 'Real Estate', icon: '🏠', desc: 'Listings, revenue, occupancy', prompt: 'Create a real estate dashboard with: total portfolio value card, property listings table (address, type, rent, status, occupancy), monthly rental income line chart, occupancy rate gauge, maintenance requests queue, lease expiry timeline, revenue by property bar chart, and vacancy alerts. Professional blue/white theme.' },
      { label: 'Restaurant', icon: '🍽️', desc: 'Orders, revenue, menu stats', prompt: 'Create a restaurant analytics dashboard with: today\'s revenue/orders/avg ticket KPI cards, hourly orders bar chart, top selling items table with quantity, revenue breakdown by category pie chart, table turnover rate, customer ratings trend, staff performance leaderboard, and peak hours heatmap. Warm amber theme.' },
      { label: 'Fleet Tracking', icon: '🚚', desc: 'Vehicles, routes, efficiency', prompt: 'Create a fleet tracking dashboard with: active vehicles count card, vehicle status grid (moving/idle/stopped with color dots), fuel efficiency bar chart by vehicle, daily mileage line chart, delivery completion rate gauge, maintenance schedule table, driver performance scores, and cost-per-mile trend. Dark map-style theme.' },
      { label: 'Patient Health', icon: '🏥', desc: 'Vitals, appointments, records', prompt: 'Create a patient health dashboard with: patient info card (name, age, blood type), vital signs gauges (heart rate, blood pressure, temperature, SpO2), medication schedule table, appointment calendar, lab results trend charts (blood sugar, cholesterol), BMI calculator card, and recent visit notes timeline. Clean medical white/blue theme.' },
      { label: 'Energy Monitor', icon: '⚡', desc: 'Consumption, solar, costs', prompt: 'Create an energy monitoring dashboard with: current power consumption gauge (kW), daily usage area chart, solar panel generation vs consumption dual line chart, monthly cost bar chart, room-by-room breakdown donut chart, peak/off-peak usage comparison, carbon footprint card, and energy saving tips. Green eco theme.' },
      { label: 'Warehouse', icon: '📦', desc: 'Inventory, orders, shipping', prompt: 'Create a warehouse management dashboard with: total SKUs and inventory value cards, stock levels by category horizontal bar chart, incoming/outgoing shipments timeline, low stock alerts table (items below reorder point), order fulfillment rate gauge, storage utilization grid map, and daily pick/pack/ship volume chart.' },
    ]
  },
  {
    category: 'Research & Academia',
    templates: [
      { label: 'Lab Experiments', icon: '🧪', desc: 'Experiment pipeline, hypotheses, results', prompt: 'Create a research lab dashboard with: active experiments kanban (planning/running/analysis/writeup), hypothesis register with pre-registration status, results heatmap by experiment, reagent/equipment utilization bars, publication pipeline table, collaborator network graph placeholder, grant burn-rate line chart, and citation count trend. Academic clean theme.' },
      { label: 'Citation Manager', icon: '📚', desc: 'Papers, authors, impact factor', prompt: 'Create a citation/reference dashboard with: total papers card, citation count trend line, h-index gauge, top-cited papers table, co-author collaboration network placeholder, papers-by-year bar chart, journal impact factor distribution, and recent papers feed with abstract preview.' },
      { label: 'Grant Tracker', icon: '💼', desc: 'Applications, funding, deadlines', prompt: 'Create a research grant dashboard with: total funded amount card, pending applications count, grant pipeline (draft/submitted/review/awarded/rejected stages), upcoming deadlines Gantt chart, funding by agency donut chart, success rate gauge, and average award size trend line.' },
      { label: 'Clinical Trial', icon: '🏥', desc: 'Patients, enrollment, adverse events', prompt: 'Create a clinical trial operations dashboard with: enrollment progress bar vs target, active sites map placeholder, patient retention funnel, adverse events table with severity colors, dose-response chart, protocol deviation counter, database lock countdown, and phase status tracker.' },
      { label: 'Publication Metrics', icon: '📈', desc: 'Journals, altmetrics, Scopus', prompt: 'Create a publications metrics dashboard with: total publications and citations KPI cards, Altmetric score cards per recent paper, downloads vs citations scatter plot, social media mentions trend, journal impact factor comparison bars, open-access percentage gauge, and top-performing papers table.' },
    ]
  },
  {
    category: 'Education',
    templates: [
      { label: 'Student Performance', icon: '🎓', desc: 'Grades, trends, at-risk', prompt: 'Create a student performance dashboard with: class average KPI card, grade distribution histogram, subject-level bar chart, at-risk students table (below threshold), attendance heatmap (days vs students), homework completion rate gauge, test score trend line per student, and teacher performance comparison. Clean school theme.' },
      { label: 'Course Completion', icon: '📊', desc: 'Enrollment, progress, certificates', prompt: 'Create an online course platform dashboard with: active enrollments KPI, completion rate gauge, lessons-watched funnel, top courses by enrollment bar chart, drop-off heatmap per lesson, certificate issued counter, average time-to-complete, and instructor leaderboard.' },
      { label: 'Attendance Tracker', icon: '✅', desc: 'Daily roll, trends, alerts', prompt: 'Create an attendance dashboard with: daily attendance % card with trend arrow, attendance heatmap (students x days), chronic absenteeism alert table, excused vs unexcused donut, class-by-class comparison bar chart, weekly trend line, and parent-notification queue.' },
      { label: 'LMS Admin', icon: '🏫', desc: 'Users, content, engagement', prompt: 'Create a Learning Management System admin dashboard with: active users MAU/DAU cards, most-viewed content table, engagement heatmap (day of week x hour), quiz pass rates by course, storage usage gauge, top contributors list, support ticket queue, and content creation trend line.' },
      { label: 'University KPIs', icon: '🏛️', desc: 'Enrollment, retention, alumni', prompt: 'Create a university leadership dashboard with: total enrollment card, retention rate gauge per cohort, 4- and 6-year graduation rate trends, applications vs offers vs yields funnel, financial aid breakdown donut, diversity metrics stacked bars, alumni engagement score, and research funding total.' },
    ]
  },
  {
    category: 'Media & Creators',
    templates: [
      { label: 'Streamer Dashboard', icon: '🎥', desc: 'Stream uptime, chat, subs', prompt: 'Create a Twitch/YouTube streamer dashboard with: live/offline status pill, current viewer count card with trend, hours streamed this week bar, new subscribers counter, chat velocity line chart, top chatters leaderboard, bit/donation total, stream health (bitrate, FPS, dropped frames) gauges, and follower growth trend. Purple streamer theme.' },
      { label: 'Podcast Analytics', icon: '🎙️', desc: 'Downloads, retention, platforms', prompt: 'Create a podcast analytics dashboard with: total downloads and unique listeners cards, episode-by-episode download bar chart, listener retention curve (average minute-by-minute), platform breakdown donut (Apple/Spotify/Google), geographic map placeholder, listener growth line, sponsor revenue card, and top episodes table.' },
      { label: 'YouTube Studio', icon: '📹', desc: 'Views, watch time, subs', prompt: 'Create a YouTube creator dashboard with: total views (28d) card, watch time hours line chart, subscribers gained counter, top videos table with CTR and AVD, revenue per 1000 views (RPM) card, audience retention average, traffic source donut, and comment sentiment bar chart.' },
      { label: 'LiveOps Gaming', icon: '🎮', desc: 'DAU, ARPU, retention', prompt: 'Create a live game-ops dashboard with: DAU/MAU cards with stickiness ratio, revenue per daily active user trend, retention curve (D1/D7/D30 cohort heatmap), concurrent users line chart, top-spending players table, in-game currency sink/source ledger, matchmaking queue time gauge, and crash-free rate.' },
      { label: 'Newsroom', icon: '📰', desc: 'Stories, traffic, subscriptions', prompt: 'Create a newsroom analytics dashboard with: top stories by views table, pageviews vs unique visitors line, subscriber conversion funnel, section performance bar chart, average time-on-page per story, paywall bounce rate, social referral donut, and breaking-news spike detector timeline.' },
    ]
  },
  {
    category: 'Logistics & Transport',
    templates: [
      { label: 'Last-Mile Delivery', icon: '🚛', desc: 'Routes, drivers, on-time', prompt: 'Create a last-mile delivery dashboard with: deliveries today card, on-time rate gauge, active drivers map placeholder, route optimization savings counter, failed delivery reasons donut, avg stops-per-route bar chart, customer satisfaction trend, and fuel cost per delivery card.' },
      { label: 'Transit Operations', icon: '🚆', desc: 'Headway, ridership, delays', prompt: 'Create a public transit operations dashboard with: system-wide ridership KPI, on-time performance gauge per line, delay causes donut, station-by-station boarding heatmap, vehicle availability bar chart, maintenance backlog timeline, and incident feed with severity tags.' },
      { label: 'Airline Ops', icon: '✈️', desc: 'Flights, delays, load factor', prompt: 'Create an airline operations dashboard with: on-time departure rate gauge, flights scheduled vs operated card, average load factor trend, delay minutes by cause donut, aircraft utilization bar chart, fuel cost trend, crew rotation heatmap, and customer complaints table.' },
      { label: 'Shipping Tracker', icon: '📦', desc: 'Containers, ports, SLA', prompt: 'Create a shipping/freight dashboard with: containers in transit card, port congestion map placeholder, SLA compliance gauge, dwell-time histogram per port, demurrage fees trend, top lanes by volume bar chart, temperature-controlled alerts table, and weekly throughput area chart.' },
    ]
  },
  {
    category: 'Environmental & Civic',
    templates: [
      { label: 'Air Quality', icon: '🌫️', desc: 'AQI, PM2.5, sensor network', prompt: 'Create an air quality monitoring dashboard with: city-wide AQI card with color band, PM2.5/PM10/NO2/O3/CO small cards, 24-hour trend multi-line chart, sensor network map placeholder, station comparison bar chart, alert history timeline, and historical monthly comparison.' },
      { label: 'Carbon Footprint', icon: '🌱', desc: 'Emissions, scopes, reductions', prompt: 'Create a corporate carbon footprint dashboard with: total CO2e KPI card, Scope 1/2/3 stacked bar breakdown, emissions trend vs baseline line, reduction target progress bar, facility-level heatmap, offset purchases donut, supplier emissions table, and year-over-year comparison.' },
      { label: 'Smart City', icon: '🏙️', desc: '311 tickets, traffic, lights', prompt: 'Create a smart-city operations dashboard with: 311 service requests by category donut, traffic speed heatmap, streetlight outages map placeholder, waste-pickup completion gauge, crime incidents area chart, park utilization by neighborhood bar chart, and budget utilization per department table.' },
      { label: 'Election Results', icon: '🗳️', desc: 'Precincts, turnout, contests', prompt: 'Create an election results dashboard with: turnout percentage gauge, contest-by-contest results cards (candidate bars with vote %), precincts reporting progress bar, vote-by-type breakdown (early/mail/day-of) donut, historical turnout comparison, demographic split stacked bars, and live-updating timestamp.' },
      { label: 'Water Utility', icon: '💧', desc: 'Consumption, pressure, leaks', prompt: 'Create a water utility operations dashboard with: daily consumption card vs capacity, pressure by district heatmap, active leak alerts table, pump station status grid, reservoir level gauges, non-revenue water % card, maintenance ticket queue, and monthly billing trend.' },
    ]
  },
  {
    category: 'Manufacturing & IoT',
    templates: [
      { label: 'Factory OEE', icon: '🏭', desc: 'Availability, performance, quality', prompt: 'Create a factory OEE (Overall Equipment Effectiveness) dashboard with: plant-wide OEE gauge, availability/performance/quality breakdown, line-by-line comparison bar chart, downtime Pareto chart (causes), scrap rate trend line, shift performance heatmap, top 5 losses table, and production vs target area chart.' },
      { label: 'Quality Control', icon: '🔬', desc: 'Defects, SPC charts, yield', prompt: 'Create a quality control dashboard with: first-pass yield gauge, defect rate PPM card, SPC control chart (X-bar with UCL/LCL), defect Pareto chart, by-operator bar chart, customer returns trend, inspection completion %, and rework cost card.' },
      { label: 'Energy Plant', icon: '⚡', desc: 'Output, grid, emissions', prompt: 'Create a power plant operations dashboard with: real-time output MW gauge, day-ahead vs actual output line chart, fuel mix donut, emissions per MWh trend, turbine status grid, grid frequency line, reserves available bar chart, and forced outage rate KPI.' },
      { label: 'Robotics Fleet', icon: '🤖', desc: 'Uptime, tasks, faults', prompt: 'Create a robotic fleet dashboard with: fleet-wide uptime gauge, robots active map placeholder, task completion rate per robot bar chart, mean time between failures (MTBF) trend, fault category donut, battery level heatmap across fleet, maintenance schedule Gantt, and tasks per hour KPI.' },
      { label: 'Predictive Maintenance', icon: '🔧', desc: 'Anomalies, RUL, schedule', prompt: 'Create a predictive maintenance dashboard with: asset health score leaderboard (worst first), remaining useful life (RUL) distribution bar chart, anomaly detection timeline, vibration/temperature multi-line chart per asset, work orders queue, parts availability gauge, and cost savings from early detection card.' },
    ]
  },
];

const DASHBOARD_SYSTEM_PROMPT = `You are a senior data visualization designer building a production-quality dashboard in a single HTML file.

OUTPUT FORMAT
- Output ONLY complete HTML starting with <!DOCTYPE html> — no markdown fences, no prose
- External libraries via CDN in <head>
- If user asks for modifications, return the COMPLETE updated HTML

REQUIRED STACK
- Tailwind CDN for layout + components: <script src="https://cdn.tailwindcss.com"></script>
  Configure theme inline (colors, fonts) via tailwind.config
- Google Fonts — pick one pair that matches the vibe (e.g. Inter Tight + JetBrains Mono for technical, IBM Plex Serif + IBM Plex Sans for finance/editorial)
- Lucide icons (https://unpkg.com/lucide@latest) for KPI-card glyphs — never emoji in a serious dashboard

CHART LIBRARY — pick one based on brief complexity

- Chart.js v4 (simple): https://cdn.jsdelivr.net/npm/chart.js@4
  Good for: basic line/bar/pie/doughnut/radar, quick KPI sparklines. Configure with modern options (animation: 'easeOutQuart', plugins.legend.position).

- Apache ECharts 5 (richer): https://cdn.jsdelivr.net/npm/echarts@5
  Use for: drill-down interactions, brush select, timeline animations, treemap, sankey, calendar heatmaps, geo maps, stacked area with gradient fills, 3D (echarts-gl). Default to ECharts for anything analytics-heavy.

- date-fns (https://cdn.jsdelivr.net/npm/date-fns@3) for date range pickers if the dashboard has time controls

REQUIRED PATTERNS (this is the quality bar — don't skip)
- Skeleton loaders on initial render — subtle CSS shimmer on cards for 400-600ms before data "loads", then fade to real chart
- Sparklines inside every KPI card (tiny trend chart next to the big number)
- Working date-range picker that actually re-filters data (not decorative)
- Sort + filter on tables that actually re-render the rows
- Empty state when data is zero/filtered-out (not just blank)
- Loading, empty, error states for every chart
- Dark mode toggle via class="dark" on <html>, persisted to localStorage, respecting prefers-color-scheme on first load
- Tooltips on hover for every chart point (Chart.js: plugins.tooltip; ECharts: tooltip: { trigger: 'axis' })

DESIGN DIRECTIVES (quality bar — not optional)
- Pick ONE dominant accent color. Financial dashboards = serious palette (deep navy + gold), ops dashboards = calm (slate + single signal color), social dashboards = brand colors. Do NOT use rainbow status badges.
- Numbers are the hero. Large tabular-nums for KPI values (font-variant-numeric: tabular-nums).
- Whitespace. Don't cram 40 elements above the fold. A great dashboard has 4-6 cards on the first screen.
- Subtle dividers (border-slate-200/50 not solid gray-300), refined shadows (or none — flat with borders is more modern).
- Typography hierarchy: KPI number (36-48px), label (11-13px uppercase tracking-wide), chart axis (10-12px).
- Avoid AI-slop tells: pastel gradient backgrounds, three-rounded-square "feature" icons, emoji in KPI positions, "↑ 12%" as the only trend indicator (use sparkline too).

RESPONSIVENESS
- Fully responsive via container queries where possible (grid-cols-1 @[640px]:grid-cols-2 @[1024px]:grid-cols-4)
- Cards stack on mobile, charts remain readable (Chart.js: responsive: true, maintainAspectRatio: false)`;

export function DashboardCreator() {
  return (
    <WebCreator
      creatorMode="dashboard"
      customTemplates={DASHBOARD_TEMPLATES}
      customSystemPrompt={DASHBOARD_SYSTEM_PROMPT}
    />
  );
}
