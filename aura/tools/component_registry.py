"""Component Registry — fetch UI component templates on demand.

The AI calls get_component("pricing") and gets back production-ready
React + TypeScript + Tailwind code for that specific pattern.
Zero prompt overhead — components live here, not in system prompts.

Design tokens: zinc-950 bg, zinc-900 cards, purple accent, shadcn/ui patterns.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .tool_contract import ToolResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Component Registry — each entry has name, description, tags, and real code
# ---------------------------------------------------------------------------

COMPONENT_REGISTRY: Dict[str, Dict[str, Any]] = {

    # ------------------------------------------------------------------
    "hero": {
        "name": "Hero Section",
        "description": "Full-width hero with headline, subtitle, dual CTAs, and gradient accent",
        "tags": ["landing", "header", "marketing"],
        "code": '''import { ArrowRight, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Hero() {
  return (
    <section className="relative min-h-[80vh] flex items-center justify-center overflow-hidden bg-zinc-950 px-4">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-purple-900/20 via-zinc-950 to-zinc-950" />
      <div className="relative z-10 mx-auto max-w-4xl text-center">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-sm text-zinc-400 transition-all duration-200 hover:bg-white/10">
          <Sparkles className="w-4 h-4 text-purple-400" />
          <span>Now available in beta</span>
        </div>
        <h1 className="text-4xl font-bold tracking-tight text-zinc-50 sm:text-6xl lg:text-7xl">
          Build something{" "}
          <span className="bg-gradient-to-r from-purple-400 to-purple-600 bg-clip-text text-transparent">
            extraordinary
          </span>
        </h1>
        <p className="mx-auto mt-6 max-w-2xl text-lg text-zinc-400">
          Ship faster with production-ready components. Beautiful defaults,
          fully customizable, dark mode first.
        </p>
        <div className="mt-10 flex flex-col items-center gap-4 sm:flex-row sm:justify-center">
          <Button className="h-12 rounded-lg bg-purple-600 px-8 text-base font-semibold text-white transition-all duration-200 hover:bg-purple-700 active:scale-[0.98]">
            Get Started
            <ArrowRight className="ml-2 w-5 h-5" />
          </Button>
          <Button variant="ghost" className="h-12 rounded-lg border border-white/10 bg-white/5 px-8 text-base text-zinc-300 transition-all duration-200 hover:bg-white/10">
            View Demo
          </Button>
        </div>
      </div>
    </section>
  );
}''',
    },

    # ------------------------------------------------------------------
    "pricing": {
        "name": "Pricing Cards",
        "description": "Three-tier pricing grid with highlighted recommended plan",
        "tags": ["pricing", "cards", "marketing", "saas"],
        "code": '''import { Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const plans = [
  { name: "Starter", price: "$9", period: "/mo", features: ["5 projects", "10GB storage", "Email support", "Basic analytics"], highlight: false },
  { name: "Pro", price: "$29", period: "/mo", features: ["Unlimited projects", "100GB storage", "Priority support", "Advanced analytics", "Custom domains", "API access"], highlight: true },
  { name: "Enterprise", price: "$99", period: "/mo", features: ["Everything in Pro", "Unlimited storage", "24/7 phone support", "SSO & SAML", "Dedicated manager", "SLA guarantee"], highlight: false },
];

export function PricingCards() {
  return (
    <section className="bg-zinc-950 px-4 py-20">
      <div className="mx-auto max-w-6xl">
        <h2 className="text-center text-2xl font-bold text-zinc-50 sm:text-4xl">Simple, transparent pricing</h2>
        <p className="mx-auto mt-4 max-w-xl text-center text-zinc-400">No hidden fees. Cancel anytime.</p>
        <div className="mt-12 grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
          {plans.map((plan) => (
            <Card
              key={plan.name}
              className={`relative rounded-xl border transition-all duration-200 hover:-translate-y-0.5 ${
                plan.highlight
                  ? "border-purple-500 bg-zinc-900 shadow-lg shadow-purple-500/10"
                  : "border-white/10 bg-zinc-900"
              }`}
            >
              {plan.highlight && (
                <Badge className="absolute -top-3 left-1/2 -translate-x-1/2 bg-purple-600 text-white">Recommended</Badge>
              )}
              <CardHeader>
                <CardTitle className="text-lg font-semibold text-zinc-50">{plan.name}</CardTitle>
                <div className="mt-2">
                  <span className="text-4xl font-bold text-zinc-50">{plan.price}</span>
                  <span className="text-zinc-500">{plan.period}</span>
                </div>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  {plan.features.map((f) => (
                    <li key={f} className="flex items-center gap-3 text-sm text-zinc-300">
                      <Check className="w-4 h-4 shrink-0 text-purple-400" />
                      {f}
                    </li>
                  ))}
                </ul>
              </CardContent>
              <CardFooter>
                <Button
                  className={`w-full rounded-lg transition-all duration-200 active:scale-[0.98] ${
                    plan.highlight
                      ? "bg-purple-600 text-white hover:bg-purple-700"
                      : "bg-white/5 text-zinc-300 hover:bg-white/10"
                  }`}
                >
                  Get started
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}''',
    },

    # ------------------------------------------------------------------
    "stats": {
        "name": "Stats Dashboard",
        "description": "Four stat cards with icons, values, deltas, and trend indicators",
        "tags": ["dashboard", "analytics", "metrics"],
        "code": '''import { TrendingUp, TrendingDown, Users, DollarSign, ShoppingCart, Eye } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

const stats = [
  { label: "Total Revenue", value: "$45,231", delta: "+20.1%", up: true, icon: DollarSign },
  { label: "Active Users", value: "2,350", delta: "+15.3%", up: true, icon: Users },
  { label: "Orders", value: "1,247", delta: "-3.2%", up: false, icon: ShoppingCart },
  { label: "Page Views", value: "573K", delta: "+12.5%", up: true, icon: Eye },
];

export function StatsDashboard() {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {stats.map((s) => (
        <Card key={s.label} className="rounded-xl border border-white/10 bg-zinc-900 shadow-sm transition-all duration-200 hover:-translate-y-0.5">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <span className="text-sm text-zinc-400">{s.label}</span>
              <s.icon className="w-5 h-5 text-zinc-500" />
            </div>
            <div className="mt-3 text-2xl font-bold font-mono text-zinc-50">{s.value}</div>
            <div className="mt-1 flex items-center gap-1 text-sm">
              {s.up ? (
                <TrendingUp className="w-4 h-4 text-emerald-500" />
              ) : (
                <TrendingDown className="w-4 h-4 text-red-500" />
              )}
              <span className={s.up ? "text-emerald-500" : "text-red-500"}>{s.delta}</span>
              <span className="text-zinc-500">vs last month</span>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}''',
    },

    # ------------------------------------------------------------------
    "table": {
        "name": "Data Table",
        "description": "Sortable data table with status badges, avatars, and action menu",
        "tags": ["table", "data", "dashboard", "list"],
        "code": '''import { MoreHorizontal, ArrowUpDown } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const rows = [
  { id: 1, name: "Olivia Martin", email: "olivia@example.com", status: "active", amount: "$1,999" },
  { id: 2, name: "Jackson Lee", email: "jackson@example.com", status: "pending", amount: "$499" },
  { id: 3, name: "Isabella Nguyen", email: "isabella@example.com", status: "active", amount: "$2,499" },
  { id: 4, name: "William Kim", email: "william@example.com", status: "inactive", amount: "$149" },
];

const statusColors: Record<string, string> = {
  active: "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
  pending: "border-amber-500/30 bg-amber-500/10 text-amber-400",
  inactive: "border-zinc-500/30 bg-zinc-500/10 text-zinc-400",
};

export function DataTable() {
  return (
    <div className="rounded-xl border border-white/10 bg-zinc-900 shadow-sm overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/10 text-left">
            {["Name", "Status", "Amount", ""].map((h) => (
              <th key={h} className="px-6 py-3 font-medium text-zinc-400">
                {h && (
                  <button className="inline-flex items-center gap-1 transition-all duration-200 hover:text-zinc-50">
                    {h} <ArrowUpDown className="w-4 h-4" />
                  </button>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} className="border-b border-white/5 transition-all duration-200 hover:bg-white/5">
              <td className="px-6 py-4">
                <div className="font-medium text-zinc-50">{row.name}</div>
                <div className="text-zinc-500">{row.email}</div>
              </td>
              <td className="px-6 py-4">
                <Badge variant="outline" className={`rounded-full capitalize ${statusColors[row.status]}`}>
                  {row.status}
                </Badge>
              </td>
              <td className="px-6 py-4 font-mono text-zinc-50">{row.amount}</td>
              <td className="px-6 py-4 text-right">
                <Button variant="ghost" size="icon" className="h-8 w-8 text-zinc-400 transition-all duration-200 hover:text-zinc-50">
                  <MoreHorizontal className="w-4 h-4" />
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}''',
    },

    # ------------------------------------------------------------------
    "form": {
        "name": "Settings Form",
        "description": "Account settings form with grouped inputs, toggles, and save button",
        "tags": ["form", "settings", "input", "account"],
        "code": '''import { Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

export function SettingsForm() {
  return (
    <Card className="mx-auto max-w-2xl rounded-xl border border-white/10 bg-zinc-900">
      <CardHeader>
        <CardTitle className="text-2xl font-bold text-zinc-50">Account Settings</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="firstName" className="text-sm text-zinc-400">First name</Label>
            <Input id="firstName" placeholder="John" className="rounded-lg border-white/10 bg-black/20 text-zinc-50 placeholder:text-zinc-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="lastName" className="text-sm text-zinc-400">Last name</Label>
            <Input id="lastName" placeholder="Doe" className="rounded-lg border-white/10 bg-black/20 text-zinc-50 placeholder:text-zinc-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500" />
          </div>
        </div>
        <div className="space-y-2">
          <Label htmlFor="email" className="text-sm text-zinc-400">Email</Label>
          <Input id="email" type="email" placeholder="john@example.com" className="rounded-lg border-white/10 bg-black/20 text-zinc-50 placeholder:text-zinc-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500" />
        </div>
        <div className="space-y-4 rounded-lg border border-white/10 bg-white/5 p-4">
          <h3 className="text-lg font-semibold text-zinc-50">Notifications</h3>
          <div className="flex items-center justify-between">
            <Label htmlFor="emailNotif" className="text-sm text-zinc-300">Email notifications</Label>
            <Switch id="emailNotif" />
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="marketing" className="text-sm text-zinc-300">Marketing emails</Label>
            <Switch id="marketing" />
          </div>
        </div>
        <Button className="w-full rounded-lg bg-purple-600 text-white transition-all duration-200 hover:bg-purple-700 active:scale-[0.98]">
          <Save className="mr-2 w-5 h-5" /> Save Changes
        </Button>
      </CardContent>
    </Card>
  );
}''',
    },

    # ------------------------------------------------------------------
    "sidebar": {
        "name": "Sidebar Navigation",
        "description": "Collapsible sidebar with icon links, sections, and active state",
        "tags": ["navigation", "sidebar", "layout", "dashboard"],
        "code": '''import { Home, Settings, Users, BarChart3, FileText, ChevronLeft, LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";

const navItems = [
  { icon: Home, label: "Dashboard", href: "/", active: true },
  { icon: BarChart3, label: "Analytics", href: "/analytics", active: false },
  { icon: Users, label: "Customers", href: "/customers", active: false },
  { icon: FileText, label: "Documents", href: "/documents", active: false },
  { icon: Settings, label: "Settings", href: "/settings", active: false },
];

export function Sidebar({ collapsed = false }: { collapsed?: boolean }) {
  return (
    <aside className={`flex h-screen flex-col border-r border-white/10 bg-zinc-900 transition-all duration-200 ${collapsed ? "w-16" : "w-64"}`}>
      <div className="flex h-14 items-center justify-between border-b border-white/10 px-4">
        {!collapsed && <span className="text-lg font-bold text-zinc-50">Acme</span>}
        <Button variant="ghost" size="icon" className="h-8 w-8 text-zinc-400 transition-all duration-200 hover:text-zinc-50">
          <ChevronLeft className={`w-4 h-4 transition-transform ${collapsed ? "rotate-180" : ""}`} />
        </Button>
      </div>
      <nav className="flex-1 space-y-1 p-2">
        {navItems.map((item) => (
          <a
            key={item.label}
            href={item.href}
            className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all duration-200 ${
              item.active
                ? "bg-purple-600/10 text-purple-400 font-medium"
                : "text-zinc-400 hover:bg-white/5 hover:text-zinc-50"
            }`}
          >
            <item.icon className="w-5 h-5 shrink-0" />
            {!collapsed && <span>{item.label}</span>}
          </a>
        ))}
      </nav>
      <div className="border-t border-white/10 p-2">
        <a href="/logout" className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-zinc-400 transition-all duration-200 hover:bg-white/5 hover:text-red-400">
          <LogOut className="w-5 h-5 shrink-0" />
          {!collapsed && <span>Logout</span>}
        </a>
      </div>
    </aside>
  );
}''',
    },

    # ------------------------------------------------------------------
    "empty_state": {
        "name": "Empty State",
        "description": "Centered empty state with icon, message, and action button",
        "tags": ["empty", "placeholder", "onboarding"],
        "code": '''import { Inbox, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";

export function EmptyState({
  icon: Icon = Inbox,
  title = "No items yet",
  description = "Get started by creating your first item.",
  actionLabel = "Create New",
  onAction,
}: {
  icon?: React.ElementType;
  title?: string;
  description?: string;
  actionLabel?: string;
  onAction?: () => void;
}) {
  return (
    <div className="flex min-h-[400px] flex-col items-center justify-center rounded-xl border border-dashed border-white/10 bg-zinc-900/50 p-8 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-white/5">
        <Icon className="w-8 h-8 text-zinc-500" />
      </div>
      <h3 className="mt-4 text-lg font-semibold text-zinc-50">{title}</h3>
      <p className="mt-2 max-w-sm text-sm text-zinc-400">{description}</p>
      <Button
        onClick={onAction}
        className="mt-6 rounded-lg bg-purple-600 text-white transition-all duration-200 hover:bg-purple-700 active:scale-[0.98]"
      >
        <Plus className="mr-2 w-5 h-5" />
        {actionLabel}
      </Button>
    </div>
  );
}''',
    },

    # ------------------------------------------------------------------
    "navbar": {
        "name": "Navigation Bar",
        "description": "Sticky top navbar with logo, links, mobile menu, and user avatar",
        "tags": ["navigation", "header", "layout"],
        "code": '''import { Menu, X, Bell } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";

const links = [
  { label: "Dashboard", href: "/" },
  { label: "Projects", href: "/projects" },
  { label: "Team", href: "/team" },
  { label: "Settings", href: "/settings" },
];

export function Navbar() {
  const [open, setOpen] = useState(false);
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-zinc-950/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6">
        <a href="/" className="text-lg font-bold text-zinc-50">Acme</a>
        <nav className="hidden items-center gap-6 md:flex">
          {links.map((l) => (
            <a key={l.label} href={l.href} className="text-sm text-zinc-400 transition-all duration-200 hover:text-zinc-50">{l.label}</a>
          ))}
        </nav>
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" className="relative h-9 w-9 text-zinc-400 transition-all duration-200 hover:text-zinc-50">
            <Bell className="w-5 h-5" />
            <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-purple-500" />
          </Button>
          <div className="h-8 w-8 rounded-full bg-gradient-to-br from-purple-500 to-purple-700" />
          <Button variant="ghost" size="icon" className="h-9 w-9 text-zinc-400 md:hidden" onClick={() => setOpen(!open)}>
            {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </Button>
        </div>
      </div>
      {open && (
        <nav className="border-t border-white/10 bg-zinc-950 px-4 py-4 md:hidden">
          {links.map((l) => (
            <a key={l.label} href={l.href} className="block py-2 text-sm text-zinc-400 transition-all duration-200 hover:text-zinc-50">{l.label}</a>
          ))}
        </nav>
      )}
    </header>
  );
}''',
    },

    # ------------------------------------------------------------------
    "footer": {
        "name": "Footer",
        "description": "Multi-column footer with links, social icons, and copyright",
        "tags": ["footer", "layout", "marketing"],
        "code": '''import { Github, Twitter, Linkedin } from "lucide-react";

const columns = [
  { title: "Product", links: ["Features", "Pricing", "Changelog", "Docs"] },
  { title: "Company", links: ["About", "Blog", "Careers", "Contact"] },
  { title: "Legal", links: ["Privacy", "Terms", "License"] },
];

const socials = [
  { icon: Github, href: "#" },
  { icon: Twitter, href: "#" },
  { icon: Linkedin, href: "#" },
];

export function Footer() {
  return (
    <footer className="border-t border-white/10 bg-zinc-950 px-4 py-12">
      <div className="mx-auto grid max-w-7xl gap-8 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <span className="text-lg font-bold text-zinc-50">Acme</span>
          <p className="mt-3 text-sm text-zinc-400">Building the future, one component at a time.</p>
          <div className="mt-4 flex gap-3">
            {socials.map(({ icon: Icon, href }) => (
              <a key={href} href={href} className="flex h-9 w-9 items-center justify-center rounded-lg bg-white/5 text-zinc-400 transition-all duration-200 hover:bg-white/10 hover:text-zinc-50">
                <Icon className="w-4 h-4" />
              </a>
            ))}
          </div>
        </div>
        {columns.map((col) => (
          <div key={col.title}>
            <h4 className="text-sm font-semibold text-zinc-50">{col.title}</h4>
            <ul className="mt-3 space-y-2">
              {col.links.map((link) => (
                <li key={link}>
                  <a href="#" className="text-sm text-zinc-400 transition-all duration-200 hover:text-zinc-50">{link}</a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="mx-auto mt-12 max-w-7xl border-t border-white/10 pt-6 text-center text-sm text-zinc-500">
        &copy; {new Date().getFullYear()} Acme Inc. All rights reserved.
      </div>
    </footer>
  );
}''',
    },

    # ------------------------------------------------------------------
    "features": {
        "name": "Features Grid",
        "description": "Responsive grid of feature cards with icons and descriptions",
        "tags": ["features", "marketing", "landing", "grid"],
        "code": '''import { Zap, Shield, Globe, Layers, Cpu, Palette } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

const features = [
  { icon: Zap, title: "Lightning Fast", desc: "Optimized for speed with edge-first architecture and smart caching." },
  { icon: Shield, title: "Enterprise Security", desc: "SOC2 compliant with end-to-end encryption and role-based access." },
  { icon: Globe, title: "Global CDN", desc: "Deployed across 200+ edge locations for sub-50ms latency worldwide." },
  { icon: Layers, title: "Composable", desc: "Mix and match components. Works with any framework or design system." },
  { icon: Cpu, title: "AI-Powered", desc: "Built-in intelligence that learns your patterns and automates workflows." },
  { icon: Palette, title: "Fully Themeable", desc: "Design tokens, dark mode, and CSS variables out of the box." },
];

export function FeaturesGrid() {
  return (
    <section className="bg-zinc-950 px-4 py-20">
      <div className="mx-auto max-w-7xl">
        <h2 className="text-center text-2xl font-bold text-zinc-50 sm:text-4xl">Everything you need</h2>
        <p className="mx-auto mt-4 max-w-xl text-center text-zinc-400">Powerful features that scale with your team.</p>
        <div className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f) => (
            <Card key={f.title} className="rounded-xl border border-white/10 bg-zinc-900 transition-all duration-200 hover:-translate-y-0.5 hover:border-purple-500/30">
              <CardContent className="p-6">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-purple-600/10">
                  <f.icon className="w-5 h-5 text-purple-400" />
                </div>
                <h3 className="mt-4 text-lg font-semibold text-zinc-50">{f.title}</h3>
                <p className="mt-2 text-sm text-zinc-400">{f.desc}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}''',
    },

    # ------------------------------------------------------------------
    "testimonials": {
        "name": "Testimonials",
        "description": "Testimonial cards with quote, author avatar, name, and role",
        "tags": ["testimonials", "social-proof", "marketing"],
        "code": '''import { Quote } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

const testimonials = [
  { quote: "This completely transformed our workflow. We shipped 3x faster in the first month.", author: "Sarah Chen", role: "CTO at TechFlow", avatar: "SC" },
  { quote: "The best developer experience I've had. Everything just works out of the box.", author: "Marcus Johnson", role: "Lead Engineer at Scale", avatar: "MJ" },
  { quote: "Finally, a tool that understands what enterprise teams actually need.", author: "Priya Patel", role: "VP Engineering at Nexus", avatar: "PP" },
];

export function Testimonials() {
  return (
    <section className="bg-zinc-950 px-4 py-20">
      <div className="mx-auto max-w-7xl">
        <h2 className="text-center text-2xl font-bold text-zinc-50 sm:text-4xl">Loved by developers</h2>
        <div className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {testimonials.map((t) => (
            <Card key={t.author} className="rounded-xl border border-white/10 bg-zinc-900 transition-all duration-200 hover:-translate-y-0.5">
              <CardContent className="p-6">
                <Quote className="w-8 h-8 text-purple-500/30" />
                <p className="mt-4 text-sm leading-relaxed text-zinc-300">{t.quote}</p>
                <div className="mt-6 flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-purple-500 to-purple-700 text-sm font-bold text-white">
                    {t.avatar}
                  </div>
                  <div>
                    <div className="text-sm font-medium text-zinc-50">{t.author}</div>
                    <div className="text-sm text-zinc-500">{t.role}</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}''',
    },

    # ------------------------------------------------------------------
    "faq": {
        "name": "FAQ Accordion",
        "description": "Expandable FAQ section using shadcn Accordion component",
        "tags": ["faq", "accordion", "support", "marketing"],
        "code": '''import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

const faqs = [
  { q: "How do I get started?", a: "Sign up for a free account and follow our quickstart guide. You'll be up and running in under 5 minutes." },
  { q: "Can I cancel my subscription?", a: "Yes, you can cancel anytime from your account settings. No questions asked, no hidden fees." },
  { q: "Is there a free tier?", a: "Absolutely. Our Starter plan is free forever with generous limits for personal projects." },
  { q: "Do you offer enterprise pricing?", a: "Yes. Contact our sales team for custom pricing, SLAs, and dedicated support." },
  { q: "What frameworks are supported?", a: "We support React, Next.js, Vue, Svelte, and any framework that works with standard HTML/CSS." },
];

export function FAQ() {
  return (
    <section className="bg-zinc-950 px-4 py-20">
      <div className="mx-auto max-w-3xl">
        <h2 className="text-center text-2xl font-bold text-zinc-50 sm:text-4xl">Frequently asked questions</h2>
        <p className="mx-auto mt-4 max-w-xl text-center text-zinc-400">Can't find what you're looking for? Contact support.</p>
        <Accordion type="single" collapsible className="mt-12 space-y-2">
          {faqs.map((faq, i) => (
            <AccordionItem key={i} value={`item-${i}`} className="rounded-lg border border-white/10 bg-zinc-900 px-6">
              <AccordionTrigger className="text-sm font-medium text-zinc-50 hover:text-purple-400 transition-all duration-200">
                {faq.q}
              </AccordionTrigger>
              <AccordionContent className="text-sm text-zinc-400">
                {faq.a}
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </div>
    </section>
  );
}''',
    },

    # ------------------------------------------------------------------
    "cta": {
        "name": "Call to Action",
        "description": "Full-width CTA banner with gradient background and action button",
        "tags": ["cta", "marketing", "conversion"],
        "code": '''import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";

export function CTA() {
  return (
    <section className="bg-zinc-950 px-4 py-20">
      <div className="mx-auto max-w-4xl rounded-2xl border border-purple-500/20 bg-gradient-to-br from-purple-900/20 via-zinc-900 to-zinc-900 p-8 text-center shadow-lg shadow-purple-500/5 sm:p-12">
        <h2 className="text-2xl font-bold text-zinc-50 sm:text-4xl">Ready to get started?</h2>
        <p className="mx-auto mt-4 max-w-xl text-zinc-400">
          Join thousands of teams already building faster. Free to start, no credit card required.
        </p>
        <div className="mt-8 flex flex-col items-center gap-4 sm:flex-row sm:justify-center">
          <Button className="h-12 rounded-lg bg-purple-600 px-8 text-base font-semibold text-white transition-all duration-200 hover:bg-purple-700 active:scale-[0.98]">
            Start Building Free
            <ArrowRight className="ml-2 w-5 h-5" />
          </Button>
          <Button variant="ghost" className="h-12 rounded-lg border border-white/10 bg-white/5 px-8 text-base text-zinc-300 transition-all duration-200 hover:bg-white/10">
            Talk to Sales
          </Button>
        </div>
      </div>
    </section>
  );
}''',
    },

    # ------------------------------------------------------------------
    "auth": {
        "name": "Login/Signup Form",
        "description": "Auth card with email/password fields, social login, and toggle",
        "tags": ["auth", "login", "signup", "form"],
        "code": '''import { Github, Mail } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function AuthForm({ mode = "login" }: { mode?: "login" | "signup" }) {
  const isSignup = mode === "signup";
  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-950 px-4">
      <Card className="w-full max-w-md rounded-2xl border border-white/10 bg-zinc-900 shadow-lg">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl font-bold text-zinc-50">
            {isSignup ? "Create an account" : "Welcome back"}
          </CardTitle>
          <p className="mt-1 text-sm text-zinc-400">
            {isSignup ? "Enter your details to get started" : "Sign in to your account"}
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <Button variant="outline" className="rounded-lg border-white/10 bg-white/5 text-zinc-300 transition-all duration-200 hover:bg-white/10">
              <Github className="mr-2 w-4 h-4" /> GitHub
            </Button>
            <Button variant="outline" className="rounded-lg border-white/10 bg-white/5 text-zinc-300 transition-all duration-200 hover:bg-white/10">
              <Mail className="mr-2 w-4 h-4" /> Google
            </Button>
          </div>
          <div className="relative">
            <div className="absolute inset-0 flex items-center"><span className="w-full border-t border-white/10" /></div>
            <div className="relative flex justify-center text-xs"><span className="bg-zinc-900 px-2 text-zinc-500">or continue with</span></div>
          </div>
          {isSignup && (
            <div className="space-y-2">
              <Label htmlFor="name" className="text-sm text-zinc-400">Full name</Label>
              <Input id="name" placeholder="John Doe" className="rounded-lg border-white/10 bg-black/20 text-zinc-50 placeholder:text-zinc-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500" />
            </div>
          )}
          <div className="space-y-2">
            <Label htmlFor="email" className="text-sm text-zinc-400">Email</Label>
            <Input id="email" type="email" placeholder="john@example.com" className="rounded-lg border-white/10 bg-black/20 text-zinc-50 placeholder:text-zinc-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="password" className="text-sm text-zinc-400">Password</Label>
            <Input id="password" type="password" placeholder="••••••••" className="rounded-lg border-white/10 bg-black/20 text-zinc-50 placeholder:text-zinc-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500" />
          </div>
          <Button className="w-full rounded-lg bg-purple-600 text-white transition-all duration-200 hover:bg-purple-700 active:scale-[0.98]">
            {isSignup ? "Create Account" : "Sign In"}
          </Button>
          <p className="text-center text-sm text-zinc-500">
            {isSignup ? "Already have an account? " : "Don't have an account? "}
            <a href="#" className="text-purple-400 transition-all duration-200 hover:text-purple-300">
              {isSignup ? "Sign in" : "Sign up"}
            </a>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}''',
    },

    # ------------------------------------------------------------------
    "card_grid": {
        "name": "Card Grid Layout",
        "description": "Responsive grid of content cards with image, title, badge, and action",
        "tags": ["cards", "grid", "content", "blog"],
        "code": '''import { ArrowUpRight } from "lucide-react";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const items = [
  { title: "Getting Started Guide", desc: "Learn the fundamentals and ship your first project in minutes.", tag: "Tutorial", image: "bg-gradient-to-br from-purple-600 to-blue-600" },
  { title: "Advanced Patterns", desc: "Composition, server components, and performance optimization.", tag: "Advanced", image: "bg-gradient-to-br from-emerald-600 to-teal-600" },
  { title: "API Reference", desc: "Complete reference for all components, hooks, and utilities.", tag: "Docs", image: "bg-gradient-to-br from-amber-600 to-orange-600" },
  { title: "Deployment", desc: "Deploy to Vercel, AWS, or any platform with one command.", tag: "DevOps", image: "bg-gradient-to-br from-pink-600 to-rose-600" },
];

export function CardGrid() {
  return (
    <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
      {items.map((item) => (
        <Card key={item.title} className="group overflow-hidden rounded-xl border border-white/10 bg-zinc-900 transition-all duration-200 hover:-translate-y-0.5 hover:border-purple-500/30">
          <div className={`h-32 ${item.image}`} />
          <CardContent className="p-4">
            <Badge variant="outline" className="mb-2 border-white/10 text-zinc-400">{item.tag}</Badge>
            <h3 className="text-lg font-semibold text-zinc-50">{item.title}</h3>
            <p className="mt-1 text-sm text-zinc-400">{item.desc}</p>
          </CardContent>
          <CardFooter className="px-4 pb-4">
            <Button variant="ghost" className="h-8 px-0 text-sm text-purple-400 transition-all duration-200 hover:text-purple-300">
              Read more <ArrowUpRight className="ml-1 w-4 h-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
            </Button>
          </CardFooter>
        </Card>
      ))}
    </div>
  );
}''',
    },

    # ------------------------------------------------------------------
    "modal": {
        "name": "Modal/Dialog",
        "description": "Accessible dialog with header, body, footer actions using shadcn Dialog",
        "tags": ["modal", "dialog", "overlay"],
        "code": '''import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter,
  DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";

export function ConfirmDialog({
  title = "Are you sure?",
  description = "This action cannot be undone. This will permanently delete your data.",
  confirmLabel = "Delete",
  cancelLabel = "Cancel",
  variant = "destructive",
  onConfirm,
}: {
  title?: string;
  description?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: "destructive" | "default";
  onConfirm?: () => void;
}) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="destructive" className="rounded-lg">Open Dialog</Button>
      </DialogTrigger>
      <DialogContent className="rounded-2xl border border-white/10 bg-zinc-900 shadow-lg sm:max-w-md">
        <DialogHeader>
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-red-500/10">
            <AlertTriangle className="w-6 h-6 text-red-500" />
          </div>
          <DialogTitle className="text-center text-lg font-semibold text-zinc-50">{title}</DialogTitle>
          <DialogDescription className="text-center text-sm text-zinc-400">{description}</DialogDescription>
        </DialogHeader>
        <DialogFooter className="mt-4 flex gap-3 sm:justify-center">
          <Button variant="ghost" className="rounded-lg border border-white/10 bg-white/5 text-zinc-300 transition-all duration-200 hover:bg-white/10">
            {cancelLabel}
          </Button>
          <Button
            onClick={onConfirm}
            className={`rounded-lg transition-all duration-200 active:scale-[0.98] ${
              variant === "destructive" ? "bg-red-600 text-white hover:bg-red-700" : "bg-purple-600 text-white hover:bg-purple-700"
            }`}
          >
            {confirmLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}''',
    },

    # ------------------------------------------------------------------
    "tabs": {
        "name": "Tabbed Interface",
        "description": "Tab navigation with content panels using shadcn Tabs",
        "tags": ["tabs", "navigation", "content"],
        "code": '''import { Code, Eye, Settings } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";

export function TabbedInterface() {
  return (
    <Tabs defaultValue="preview" className="w-full">
      <TabsList className="inline-flex rounded-lg border border-white/10 bg-zinc-900 p-1">
        <TabsTrigger value="preview" className="flex items-center gap-2 rounded-md px-4 py-2 text-sm text-zinc-400 transition-all duration-200 data-[state=active]:bg-white/10 data-[state=active]:text-zinc-50">
          <Eye className="w-4 h-4" /> Preview
        </TabsTrigger>
        <TabsTrigger value="code" className="flex items-center gap-2 rounded-md px-4 py-2 text-sm text-zinc-400 transition-all duration-200 data-[state=active]:bg-white/10 data-[state=active]:text-zinc-50">
          <Code className="w-4 h-4" /> Code
        </TabsTrigger>
        <TabsTrigger value="settings" className="flex items-center gap-2 rounded-md px-4 py-2 text-sm text-zinc-400 transition-all duration-200 data-[state=active]:bg-white/10 data-[state=active]:text-zinc-50">
          <Settings className="w-4 h-4" /> Settings
        </TabsTrigger>
      </TabsList>
      <TabsContent value="preview" className="mt-4">
        <Card className="rounded-xl border border-white/10 bg-zinc-900">
          <CardContent className="p-6">
            <p className="text-sm text-zinc-400">Preview content goes here. Replace with your actual component preview.</p>
          </CardContent>
        </Card>
      </TabsContent>
      <TabsContent value="code" className="mt-4">
        <Card className="rounded-xl border border-white/10 bg-zinc-900">
          <CardContent className="p-6">
            <pre className="overflow-x-auto rounded-lg bg-black/20 p-4 font-mono text-sm text-zinc-300">
              {`<Button variant="primary">Click me</Button>`}
            </pre>
          </CardContent>
        </Card>
      </TabsContent>
      <TabsContent value="settings" className="mt-4">
        <Card className="rounded-xl border border-white/10 bg-zinc-900">
          <CardContent className="p-6">
            <p className="text-sm text-zinc-400">Configuration options for this component.</p>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  );
}''',
    },

    # ------------------------------------------------------------------
    "profile": {
        "name": "Profile/Account Page",
        "description": "User profile page with avatar, info card, and activity section",
        "tags": ["profile", "account", "user", "dashboard"],
        "code": '''import { MapPin, Calendar, LinkIcon, Edit2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export function ProfilePage() {
  return (
    <div className="mx-auto max-w-3xl space-y-6 px-4 py-8">
      <Card className="overflow-hidden rounded-xl border border-white/10 bg-zinc-900">
        <div className="h-32 bg-gradient-to-r from-purple-600 to-purple-800" />
        <CardContent className="relative px-6 pb-6">
          <div className="-mt-12 flex items-end justify-between">
            <div className="flex h-24 w-24 items-center justify-center rounded-full border-4 border-zinc-900 bg-gradient-to-br from-purple-500 to-purple-700 text-2xl font-bold text-white">
              JD
            </div>
            <Button variant="ghost" className="rounded-lg border border-white/10 bg-white/5 text-sm text-zinc-300 transition-all duration-200 hover:bg-white/10">
              <Edit2 className="mr-2 w-4 h-4" /> Edit Profile
            </Button>
          </div>
          <div className="mt-4">
            <h1 className="text-2xl font-bold text-zinc-50">John Doe</h1>
            <p className="mt-1 text-sm text-zinc-400">Full-stack developer building tools for the modern web.</p>
          </div>
          <div className="mt-4 flex flex-wrap gap-4 text-sm text-zinc-400">
            <span className="flex items-center gap-1"><MapPin className="w-4 h-4" /> San Francisco, CA</span>
            <span className="flex items-center gap-1"><Calendar className="w-4 h-4" /> Joined Mar 2024</span>
            <span className="flex items-center gap-1"><LinkIcon className="w-4 h-4" /> johndoe.dev</span>
          </div>
          <div className="mt-4 flex gap-2">
            <Badge variant="outline" className="border-purple-500/30 bg-purple-500/10 text-purple-400">React</Badge>
            <Badge variant="outline" className="border-purple-500/30 bg-purple-500/10 text-purple-400">TypeScript</Badge>
            <Badge variant="outline" className="border-purple-500/30 bg-purple-500/10 text-purple-400">Node.js</Badge>
          </div>
        </CardContent>
      </Card>
      <div className="grid gap-4 sm:grid-cols-3">
        {[{ label: "Projects", value: "24" }, { label: "Followers", value: "1.2K" }, { label: "Contributions", value: "847" }].map((s) => (
          <Card key={s.label} className="rounded-xl border border-white/10 bg-zinc-900 text-center">
            <CardContent className="p-4">
              <div className="text-2xl font-bold font-mono text-zinc-50">{s.value}</div>
              <div className="text-sm text-zinc-400">{s.label}</div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}''',
    },

    # ------------------------------------------------------------------
    "chat": {
        "name": "Chat Interface",
        "description": "Chat UI with message bubbles, input bar, and typing indicator",
        "tags": ["chat", "messaging", "realtime", "ai"],
        "code": '''import { Send, Paperclip, Bot, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const messages = [
  { role: "assistant", content: "Hello! How can I help you today?" },
  { role: "user", content: "I need help setting up my project." },
  { role: "assistant", content: "Sure! What framework are you using? I can walk you through the setup step by step." },
];

export function ChatInterface() {
  return (
    <div className="mx-auto flex h-[600px] max-w-2xl flex-col rounded-2xl border border-white/10 bg-zinc-900">
      <div className="flex items-center gap-3 border-b border-white/10 px-6 py-4">
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-purple-600/10">
          <Bot className="w-5 h-5 text-purple-400" />
        </div>
        <div>
          <div className="text-sm font-medium text-zinc-50">Aura Assistant</div>
          <div className="text-xs text-emerald-400">Online</div>
        </div>
      </div>
      <div className="flex-1 space-y-4 overflow-y-auto p-6">
        {messages.map((msg, i) => (
          <div key={i} className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
            <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
              msg.role === "user" ? "bg-purple-600" : "bg-white/5"
            }`}>
              {msg.role === "user" ? <User className="w-4 h-4 text-white" /> : <Bot className="w-4 h-4 text-purple-400" />}
            </div>
            <div className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm ${
              msg.role === "user"
                ? "bg-purple-600 text-white"
                : "bg-white/5 text-zinc-300"
            }`}>
              {msg.content}
            </div>
          </div>
        ))}
        <div className="flex gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-white/5">
            <Bot className="w-4 h-4 text-purple-400" />
          </div>
          <div className="flex items-center gap-1 rounded-2xl bg-white/5 px-4 py-3">
            <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-500 [animation-delay:-0.3s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-500 [animation-delay:-0.15s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-500" />
          </div>
        </div>
      </div>
      <div className="border-t border-white/10 p-4">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" className="h-10 w-10 shrink-0 text-zinc-400 transition-all duration-200 hover:text-zinc-50">
            <Paperclip className="w-5 h-5" />
          </Button>
          <Input placeholder="Type a message..." className="rounded-lg border-white/10 bg-black/20 text-zinc-50 placeholder:text-zinc-600 focus:border-purple-500 focus:ring-2 focus:ring-purple-500" />
          <Button size="icon" className="h-10 w-10 shrink-0 rounded-lg bg-purple-600 text-white transition-all duration-200 hover:bg-purple-700 active:scale-[0.98]">
            <Send className="w-5 h-5" />
          </Button>
        </div>
      </div>
    </div>
  );
}''',
    },

    # ------------------------------------------------------------------
    "kanban": {
        "name": "Kanban Board",
        "description": "Three-column kanban board with draggable task cards",
        "tags": ["kanban", "board", "tasks", "project-management"],
        "code": '''import { Plus, GripVertical, Clock, AlertCircle } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const columns = [
  {
    title: "To Do", count: 3, color: "text-zinc-400",
    tasks: [
      { title: "Design system audit", priority: "high", tag: "Design" },
      { title: "API rate limiting", priority: "medium", tag: "Backend" },
      { title: "Write unit tests", priority: "low", tag: "Testing" },
    ],
  },
  {
    title: "In Progress", count: 2, color: "text-purple-400",
    tasks: [
      { title: "User dashboard", priority: "high", tag: "Frontend" },
      { title: "Auth flow rework", priority: "medium", tag: "Full-stack" },
    ],
  },
  {
    title: "Done", count: 2, color: "text-emerald-400",
    tasks: [
      { title: "Landing page", priority: "low", tag: "Frontend" },
      { title: "CI/CD pipeline", priority: "medium", tag: "DevOps" },
    ],
  },
];

const priorityColors: Record<string, string> = {
  high: "text-red-400",
  medium: "text-amber-400",
  low: "text-zinc-500",
};

export function KanbanBoard() {
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      {columns.map((col) => (
        <div key={col.title} className="rounded-xl border border-white/10 bg-zinc-900/50 p-4">
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className={`text-sm font-semibold ${col.color}`}>{col.title}</span>
              <Badge variant="outline" className="border-white/10 text-xs text-zinc-500">{col.count}</Badge>
            </div>
            <Button variant="ghost" size="icon" className="h-7 w-7 text-zinc-500 transition-all duration-200 hover:text-zinc-50">
              <Plus className="w-4 h-4" />
            </Button>
          </div>
          <div className="space-y-3">
            {col.tasks.map((task) => (
              <Card key={task.title} className="cursor-grab rounded-lg border border-white/10 bg-zinc-900 transition-all duration-200 hover:border-purple-500/30 active:cursor-grabbing">
                <CardContent className="p-3">
                  <div className="flex items-start gap-2">
                    <GripVertical className="mt-0.5 w-4 h-4 shrink-0 text-zinc-600" />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-zinc-50">{task.title}</p>
                      <div className="mt-2 flex items-center gap-2">
                        <Badge variant="outline" className="border-white/10 text-xs text-zinc-400">{task.tag}</Badge>
                        <AlertCircle className={`w-3 h-3 ${priorityColors[task.priority]}`} />
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}''',
    },
}


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class ComponentRegistryTool:
    """Fetch production-ready UI component templates by name.

    Zero prompt overhead: the AI calls get_component() only when it needs
    a specific pattern, instead of embedding all component code in the
    system prompt.
    """

    name = "component_registry"
    description = "Fetch production-ready React + Tailwind + shadcn/ui component templates by name"

    def get_component(self, component_type: str) -> ToolResult:
        """Return the code for a component type.

        Args:
            component_type: Key from the registry (e.g. "hero", "pricing", "chat").

        Returns:
            ToolResult with the component code, or an error with suggestions.
        """
        key = component_type.lower().strip().replace("-", "_").replace(" ", "_")

        if key in COMPONENT_REGISTRY:
            entry = COMPONENT_REGISTRY[key]
            return ToolResult(
                success=True,
                result={
                    "name": entry["name"],
                    "component_type": key,
                    "code": entry["code"],
                    "tags": entry.get("tags", []),
                },
            )

        # Fuzzy match — check if the query appears in any key or tag
        suggestions = self.search_components(component_type)
        if suggestions:
            suggestion_names = ", ".join(s["key"] for s in suggestions[:5])
            return ToolResult(
                success=False,
                error=f"Unknown component '{component_type}'. Did you mean: {suggestion_names}?",
            )

        return ToolResult(
            success=False,
            error=f"Unknown component '{component_type}'. Use list_components() to see all {len(COMPONENT_REGISTRY)} available.",
        )

    def list_components(self) -> ToolResult:
        """List all available component types with descriptions."""
        items = [
            {"key": k, "name": v["name"], "description": v["description"], "tags": v.get("tags", [])}
            for k, v in COMPONENT_REGISTRY.items()
        ]
        return ToolResult(
            success=True,
            result={"components": items, "count": len(items)},
        )

    def search_components(self, query: str) -> list:
        """Search components by keyword across names, descriptions, and tags.

        Args:
            query: Free-text search query.

        Returns:
            List of matching component dicts sorted by relevance.
        """
        query_lower = query.lower().strip()
        results = []

        for key, entry in COMPONENT_REGISTRY.items():
            score = 0
            # Exact key match
            if query_lower == key:
                score += 100
            # Key contains query
            elif query_lower in key:
                score += 50
            # Name match
            if query_lower in entry["name"].lower():
                score += 40
            # Description match
            if query_lower in entry["description"].lower():
                score += 20
            # Tag match
            for tag in entry.get("tags", []):
                if query_lower in tag.lower():
                    score += 30
                    break

            if score > 0:
                results.append({
                    "key": key,
                    "name": entry["name"],
                    "description": entry["description"],
                    "tags": entry.get("tags", []),
                    "score": score,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    async def execute(self, action: str, **kwargs) -> ToolResult:
        """Main entry point for the tool contract pattern.

        Actions:
            get — fetch a component by type
            list — list all available components
            search — search components by keyword
        """
        action = action.lower().strip()

        if action == "get":
            component_type = kwargs.get("component_type") or kwargs.get("type") or kwargs.get("name", "")
            if not component_type:
                return ToolResult(success=False, error="Missing 'component_type' parameter")
            return self.get_component(component_type)

        elif action == "list":
            return self.list_components()

        elif action == "search":
            query = kwargs.get("query", "")
            if not query:
                return ToolResult(success=False, error="Missing 'query' parameter")
            results = self.search_components(query)
            return ToolResult(
                success=True,
                result={"results": results, "count": len(results)},
            )

        else:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Use: get, list, search",
            )
