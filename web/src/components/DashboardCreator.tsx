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
];

const DASHBOARD_SYSTEM_PROMPT = `You are an expert dashboard designer. Generate a complete, interactive dashboard page.

Rules:
- Output ONLY the complete HTML code starting with <!DOCTYPE html>
- Include Chart.js from CDN: <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
- Create responsive charts using Chart.js (line, bar, pie, doughnut, radar as needed)
- Include ALL CSS in a <style> tag — use CSS Grid for the dashboard layout
- ALL chart data should be realistic sample data (not empty)
- Add interactive elements: date range pickers, filter dropdowns, tab switches
- KPI cards should have large numbers with trend indicators (↑ green / ↓ red)
- Tables should have sortable headers (click to sort)
- Use a consistent color palette throughout
- Make fully responsive (stack cards on mobile)
- Add smooth loading animations on chart render
- Dark theme by default with proper contrast
- NO markdown fences, NO explanation text, ONLY the HTML document`;

export function DashboardCreator() {
  return (
    <WebCreator
      creatorMode="dashboard"
      customTemplates={DASHBOARD_TEMPLATES}
      customSystemPrompt={DASHBOARD_SYSTEM_PROMPT}
    />
  );
}
