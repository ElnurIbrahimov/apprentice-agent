"""Design system prompt — injected when AURA generates frontend/UI code.

Slim design tokens + core rules only. Kept under 2000 chars to avoid
truncating other system prompt components (identity, memory, personality).

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
"""

# Modes that trigger design system injection
DESIGN_SYSTEM_MODES = frozenset({"frontend", "rapid", "artifact"})
