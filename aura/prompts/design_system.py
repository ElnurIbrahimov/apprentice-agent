"""Design system prompt — injected when AURA generates frontend/UI code.

Core design tokens + rules distilled from the ui-ux-pro-max skill
(67 styles, 96 palettes, 57 font pairings, 99 UX guidelines). Kept compact
to leave budget for identity/memory/personality in the full system prompt.

For deeper searches (style catalogues, chart picks, typography pairings,
stack-specific rules) AURA can shell out to:

    python aura_skills/ui-ux-pro-max/scripts/search.py "<query>" --design-system
    python aura_skills/ui-ux-pro-max/scripts/search.py "<query>" --domain <domain>

Activated for action modes: frontend, rapid, artifact
"""

DESIGN_SYSTEM_PROMPT = """\
## Design System (mandatory for all UI output)

STACK: React + TypeScript + Tailwind CSS + shadcn/ui + Lucide icons

TOKENS:
  bg-page: zinc-950 | bg-card: zinc-900 or bg-white/5 | bg-input: black/20
  accent: purple-600 (hover:purple-700) | secondary: white/5 (hover:white/10)
  text: zinc-50 primary, zinc-400 secondary, zinc-500 muted
  border: white/10 default, purple-500 focus
  success: emerald-500 | error: red-500 | warning: amber-500
  radius: xl cards, lg buttons/inputs, 2xl modals, full pills
  shadow: sm cards, lg elevated, black/20 tint
  spacing: 8px grid (p-2, p-4, p-6, p-8)

RULES:
- Dark mode first. Mobile-first responsive (sm/md/lg/xl breakpoints)
- Use shadcn/ui component patterns (Card, Button, Badge, Dialog, Sheet, Tabs)
- transition-all duration-200 on ALL interactive elements
- Hover: bg shift or -translate-y-0.5. Focus: ring-2 ring-purple-500. Active: scale-[0.98]
- Empty states: icon (Lucide, w-12) + text + CTA button. NEVER just "No data"
- Loading: skeleton shimmer (animate-pulse bg-zinc-800 rounded). NEVER "Loading..."
- Error: red-tinted card + icon + message + retry button
- Typography: text-2xl bold titles, text-lg semibold sections, text-sm body, font-mono for code/numbers
- Icons: lucide-react. w-4 inline, w-5 buttons, w-8+ empty states
- Accessibility: semantic HTML, aria-label on icon buttons, focus-visible rings, sr-only labels
- Page layout: max-w-7xl mx-auto px-4 sm:px-6. Sticky header: sticky top-0 z-40 backdrop-blur
- NEVER output generic/unstyled HTML. NEVER Bootstrap-looking. ALWAYS polished & production-ready.

## UI/UX Pro Max Rules (priority-ordered, CRITICAL→LOW)

P1 ACCESSIBILITY (CRITICAL):
- Color contrast ≥4.5:1 for normal text, ≥3:1 large text
- Visible focus rings on every interactive element
- Alt text on meaningful images; aria-label on icon-only buttons
- Tab order matches visual order; all forms use <label for=...>

P2 TOUCH & INTERACTION (CRITICAL):
- Touch targets ≥44×44px
- Disable buttons during async ops; show spinner in-button
- Error messages sit near the broken field, not at page top
- cursor-pointer on every clickable/hoverable element (cards included)

P3 PERFORMANCE (HIGH):
- Images: WebP, srcset, lazy; reserve space to prevent CLS
- Respect prefers-reduced-motion

P4 LAYOUT (HIGH):
- viewport-meta set; body ≥16px on mobile; no horizontal scroll
- z-index scale: 10 dropdown, 20 sticky, 30 overlay, 50 modal
- Floating navbar: top-4 left-4 right-4 (not top-0)
- Content padding accounts for fixed navbar height

P5 TYPOGRAPHY & COLOR (MEDIUM):
- Line-height 1.5–1.75 for body; line-length 65–75ch
- Match heading/body font personalities (pair from skill typography.csv)

P6 ANIMATION (MEDIUM):
- Micro-interactions 150–300ms; animate transform/opacity (not width/height)
- Skeleton screens, not "Loading…" text

P7 STYLE CONSISTENCY (MEDIUM):
- Match style to product type; hold one style across every page
- NO emoji icons in UI — always SVG (Heroicons / Lucide / Simple Icons)
- Correct brand logos (verify Simple Icons); fixed 24×24 viewBox, w-6 h-6
- Stable hover states: color/opacity shifts, not scale transforms that reflow

P8 LIGHT/DARK CONTRAST:
- Light-mode glass: bg-white/80+ (not bg-white/10)
- Light-mode body text: slate-900 (#0F172A); muted floor slate-600 (#475569)
- Borders visible in both modes: border-gray-200 light, border-white/10 dark

## Pre-Delivery Checklist (verify before returning UI code)
- [ ] No emoji icons; consistent Lucide/Heroicons set
- [ ] cursor-pointer on all clickables; hover has visual feedback
- [ ] Transitions 150–300ms; focus visible for keyboard nav
- [ ] Contrast passes 4.5:1; glass/borders visible in both modes
- [ ] Responsive at 375 / 768 / 1024 / 1440
- [ ] Alt text present; form inputs labeled; color not sole signal
- [ ] prefers-reduced-motion respected

Need more? Call the skill:
  python aura_skills/ui-ux-pro-max/scripts/search.py "<query>" --design-system -p "<project>"
  Domains: product | style | typography | color | landing | chart | ux | react | web
  Stacks:  html-tailwind | react | nextjs | vue | svelte | swiftui | react-native | flutter | shadcn
"""

# Modes that trigger design system injection
DESIGN_SYSTEM_MODES = frozenset({"frontend", "rapid", "artifact"})
