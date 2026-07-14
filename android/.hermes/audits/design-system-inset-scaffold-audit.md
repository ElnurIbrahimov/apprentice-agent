# Aura Android — Design System & Scaffold Audit

**Branch:** `feat/tier-1-friction`  
**Inspected:** `2026-07-13`  
**Scope:** Theme, insets, root scaffold, bottom navigation, responsive proportion, all 12 screen composables, design tokens.

---

## 1. THEME HARDCODING — `AuraTokens.Dark` Used Everywhere

### Defect
`ChatScreen` and `AuraBottomBar` always reference `AuraTokens.Dark.*` directly instead of `MaterialTheme.colorScheme.*`. When the user selects **light mode** (or "system" with a light system theme), Material 3's color scheme switches to `LightColors` (line 43 of Theme.kt), but these composables bypass it entirely. Result: chat area and bottom bar stay dark while the rest of the app goes light.

### Confirmed Code

**NavGraph.kt — AuraBottomBar (lines 252, 272, 277–278):**
```kotlin
color = AuraTokens.Dark.surface1,
targetValue = if (selected) AuraTokens.Dark.surface3 else Color.Transparent,
val contentColor = if (selected) AuraTokens.Dark.textPrimary
                   else AuraTokens.Dark.textTertiary
```

**ChatScreen.kt — ChatHeader (lines 636, 655–656, 662):**
```kotlin
.background(AuraTokens.Dark.surface2)
color = if (modelMismatch) AuraTokens.Dark.glowOrange else AuraTokens.Dark.textPrimary
tint = AuraTokens.Dark.textTertiary
```

**ChatScreen.kt — ChatInputBar (lines 1030, 1040, 1057–1058, 1064, 1066, 1079, 1158–1159):**
```kotlin
.navigationBarsPadding()
.background(AuraTokens.Dark.surface2)
.background(AuraTokens.Dark.surface1)
.border(1.dp, AuraTokens.Dark.borderSubtle, ...)
color = AuraTokens.Dark.textPrimary,
cursorBrush = SolidColor(AuraTokens.Dark.accentPurple),
color = AuraTokens.Dark.textSecondary,
if (canSend) AuraTokens.Dark.sendReady else AuraTokens.Dark.surface2,
```

### Affected Files
| File | Lines with Hardcoded `AuraTokens.Dark` |
|------|----------------------------------------|
| `ui/nav/NavGraph.kt` | 252, 272, 277, 278 |
| `ui/screens/ChatScreen.kt` | 368, 380, 387, 625, 636, 647, 655, 656, 662, 673, 681, 691, 692, 723, 973, 984, 988, 1030, 1040, 1045, 1057, 1058, 1064, 1066, 1079, 1096, 1097, 1125, 1158, 1159, 1163, 1176 |

---

## 2. SCAFFOLD NESTING — 6 Screens Nest Scaffolds

### Defect
`NavGraph.kt` (line 116) owns a root `Scaffold` that provides `bottomBar` padding. Six screen composables embed their own `Scaffold` inside this root one. Material 3's `Scaffold` is designed to be a singleton per layout tree; nesting produces compounding `contentWindowInsets` padding, visual offsets, and top-bar / FAB overlaps.

### Nesting Tree
```
NavGraph.Scaffold (root — bottomBar, contentPadding)
├── HomeScreen           ✓
├── ChatScreen           ✓ (no Scaffold)
├── MemoryScreen         ✓
├── SettingsScreen       ✓
├── HandsScreen.Scaffold ❌ (line 103 — snackbarHost, FAB)
├── TasksScreen.Scaffold ❌ (line 83 — FAB)
├── RemindersScreen.Scaffold ❌ (line 64 — FAB)
├── ProfileScreen.Scaffold ❌ (line 32 — topBar)
├── IdentityEditorScreen.Scaffold ❌ (line 72 — topBar)
├── ProactiveHistoryScreen.Scaffold ❌ (line 90 — topBar)
└── (other screens)      ✓
```

### Confirmed Code
**HandsScreen.kt (lines 103–114):**
```kotlin
Scaffold(
    snackbarHost = { SnackbarHost(snackbar) },
    floatingActionButton = { ... }
) { padding ->
    Column(Modifier.fillMaxSize().padding(padding).padding(horizontal = 20.dp))
```

**TasksScreen.kt (lines 83–89):**
```kotlin
Scaffold(
    floatingActionButton = { ... }
) { padding ->
    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp).padding(padding),
```

**ProfileScreen.kt (lines 32–47):**
```kotlin
Scaffold(
    topBar = { TopAppBar(...) }
) { padding ->
    LazyColumn(modifier = Modifier.fillMaxSize().padding(padding).padding(horizontal = 16.dp))
```

This compounds: root Scaffold applies `bottomBar` padding via `Modifier.padding(padding)` on the `NavHost` (line 130), then the nested Scaffold re-reads content insets again.

### Affected Files
| File | Line | Nested Scaffold Purpose |
|------|------|------------------------|
| `ui/screens/HandsScreen.kt` | 103 | `snackbarHost` + FAB |
| `ui/screens/TasksScreen.kt` | 83 | FAB |
| `ui/screens/RemindersScreen.kt` | 64 | FAB |
| `ui/screens/ProfileScreen.kt` | 32 | `topBar` |
| `ui/screens/IdentityEditorScreen.kt` | 72 | `topBar` |
| `ui/screens/ProactiveHistoryScreen.kt` | 90 | `topBar` |

---

## 3. INSET DOUBLING — Overlapping `statusBarsPadding` / `navigationBarsPadding`

### Defect
The root `Scaffold` (NavGraph.kt:126) passes innerPadding to the `NavHost` via `Modifier.padding(padding)`. This already accounts for the bottom bar height. But individual screens add their own `statusBarsPadding()`, `navigationBarsPadding()`, and `imePadding()` on top, producing doubled inset values.

### Path A — Status Bar (top)
```
Root Scaffold padding (top inset)        — NavGraph line 130
  + ChatHeader .statusBarsPadding()      — ChatScreen line 625
  = 2× status bar height
```

### Path B — Navigation Bar (bottom)
```
Root Scaffold bottomBar inset             — NavGraph line 130
  + AuraBottomBar .navigationBarsPadding() — NavGraph line 256
  + ChatInputBar .navigationBarsPadding()  — ChatScreen line 1030
  = 2-3× navigation bar height
```

### Confirmed Code
**NavGraph.kt (lines 126, 130, 256):**
```kotlin
Scaffold(bottomBar = { AuraBottomBar(...) }) { padding ->    // line 116-126
    NavHost(modifier = Modifier.padding(padding)) { ... }      // line 130
// Inside AuraBottomBar:
    .navigationBarsPadding()                                   // line 256
```

**ChatScreen.kt (line 625):**
```kotlin
// Inside ChatHeader:
.statusBarsPadding()   // ON TOP of root Scaffold top content padding
```

**ChatScreen.kt (line 1030):**
```kotlin
// Inside ChatInputBar:
.navigationBarsPadding()   // ON TOP of root Scaffold bottom content padding
```

### Affected Files
| File | Line | Modifier | Conflict |
|------|------|----------|----------|
| `ui/nav/NavGraph.kt` | 130 | `Modifier.padding(padding)` | Root padding from Scaffold |
| `ui/nav/NavGraph.kt` | 256 | `.navigationBarsPadding()` | Doubles root inset at bottom |
| `ui/screens/ChatScreen.kt` | 625 | `.statusBarsPadding()` | Doubles root inset at top |
| `ui/screens/ChatScreen.kt` | 1030 | `.navigationBarsPadding()` | Triples root inset at bottom |
| `ui/screens/ChatScreen.kt` | 253 | `.imePadding()` | Acceptable (keyboard distinct) |
| `ui/screens/IdentityEditorScreen.kt` | 114 | `.imePadding()` | Acceptable |

---

## 4. BOTTOM BAR VISUALLY OVERSIZED

### Defect
The `AuraBottomBar` is ~76-90dp tall (depending on device navigation bar), far exceeding typical Material 3 bottom bars (~56dp before insets, ~72dp after).

### Dimension Breakdown
```
AuraBottomBar Surface (line 251)
├── navigationBarsPadding()               = 24–48dp (device-dependent)
├── padding(horizontal=16, vertical=12)   = +24dp vertical
│   └── Row (content)
│       └── padding(4.dp)                 = +8dp vertical
│           └── Item Box
│               └── padding(vertical=8)   = +16dp vertical (per item)
│                   ├── Icon (20.dp)
│                   ├── spacing (2.dp)
│                   └── Text (10sp ≈ 13dp)
└── Total (excl navBarInset)             = 12+4+8+20+2+13 = 59dp
    Total (incl navBarInset, ~32dp)      = ~91dp
```

Standard Material 3 `NavigationBar` with label placement uses ~80dp (including bottom nav bar inset). The Aura pill adds extra `12dp` outer padding + `4dp` inner Row padding, pushing it to ~59dp before insets vs Material 3's ~48-56dp before insets.

### Affected Files
| File | Lines |
|------|-------|
| `ui/nav/NavGraph.kt` | 251–316 |

---

## 5. RESPONSIVE PROPORTION — No Max-Width Architecture

### Defect
No screen applies `widthIn(max = ...)` or uses `BoxWithConstraints` to cap content width on tablets/landscape. On large screens (600dp+, tablets, desktop ChromeOS windows) every screen renders edge-to-edge with single-line chat messages spanning the full width.

### Current Width Strategy (all screens)
| Screen | Width Handling |
|--------|---------------|
| `NavGraph` root | None (no container) |
| `HomeScreen` | `padding(horizontal=20.dp)` only |
| `ChatScreen` | `.fillMaxSize()` only |
| `MemoryScreen` | `padding(horizontal=20.dp)` only |
| `SettingsScreen` | `padding(horizontal=20.dp)` only |
| `HandsScreen` | `padding(horizontal=20.dp)` only |
| `HistoryScreen` | `padding(horizontal=20.dp)` only |
| `TasksScreen` | `padding(16.dp)` only |
| `ToolsScreen` | `padding(horizontal=20.dp)` only |
| `RemindersScreen` | `padding(horizontal=16.dp)` only |
| `DiagnosticsScreen` | `padding(horizontal=20.dp)` only |
| `KnowledgeGraphScreen` | `padding(horizontal=20.dp)` only |
| `ProactiveHistoryScreen` | `padding(horizontal=16.dp)` only |
| `ProfileScreen` | `padding(horizontal=16.dp)` only |
| `IdentityEditorScreen` | None |
| `OnboardingScreen` | None |

**Missing:** No shared composable like `ResponsiveContainer` or `ContentWidthMax` that applies `widthIn(max = 600.dp).align(Alignment.CenterHorizontally)` on tablets.

---

## 6. TARGET ARCHITECTURE RECOMMENDATION

### 6.1 Theme Fix — Two-Phase Audit Migration

**Phase 1 — Convert `AuraTokens.Dark` call-sites to `MaterialTheme.colorScheme` equivalents:**

| Hardcoded Token | Replacement |
|----------------|-------------|
| `AuraTokens.Dark.surface1` | `MaterialTheme.colorScheme.surface` |
| `AuraTokens.Dark.surface2` | `MaterialTheme.colorScheme.surfaceVariant` |
| `AuraTokens.Dark.surface3` | `MaterialTheme.colorScheme.surfaceVariant` (or inverseOnSurface) |
| `AuraTokens.Dark.textPrimary` | `MaterialTheme.colorScheme.onSurface` |
| `AuraTokens.Dark.textSecondary` | `MaterialTheme.colorScheme.onSurfaceVariant` |
| `AuraTokens.Dark.textTertiary` | `MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)` |
| `AuraTokens.Dark.accentPurple` | `MaterialTheme.colorScheme.primary` |
| `AuraTokens.Dark.accentBlue` | `MaterialTheme.colorScheme.secondary` |
| `AuraTokens.Dark.accentGreen` | `MaterialTheme.colorScheme.tertiary` |
| `AuraTokens.Dark.borderSubtle` | `MaterialTheme.colorScheme.outlineVariant` |
| `AuraTokens.Dark.sendReady` | `MaterialTheme.colorScheme.primary` |
| `AuraTokens.Dark.aiError` | `MaterialTheme.colorScheme.errorContainer` |
| `AuraTokens.Dark.glowRed` | `MaterialTheme.colorScheme.error` |
| `AuraTokens.Dark.glowBlue` | `MaterialTheme.colorScheme.secondary` |
| `AuraTokens.Dark.glowOrange` | `MaterialTheme.colorScheme.tertiary` |

**Phase 2 — Immediate theme-dark tokens (`AuraTokens.Dark.*`) → `MaterialTheme.colorScheme.*` to fix Light Mode**

### 6.2 Scaffold Unification

```
Before:                               After:
NavGraph.Scaffold                     NavGraph.Scaffold (root only)
├── NavHost                           └── NavHost
│   ├── HandsScreen.Scaffold              ├── HandsScreen (no Scaffold)
│   ├── TasksScreen.Scaffold              ├── TasksScreen (no Scaffold)
│   ├── ProfileScreen.Scaffold            ├── ProfileScreen (no Scaffold)
│   └── ...5 more nested Scaffolds        └── ... (AuraScreenShell for all)
└── AuraBottomBar                     └── AuraBottomBar
```

**Pattern:**
1. `HandsScreen` → move `snackbarHost` to root Scaffold or use `AuraScreenShell` + `SnackbarHostState` passed via CompositionLocal
2. `TasksScreen` → move FAB to root Scaffold or use a local `Box` overlay
3. `RemindersScreen` → same pattern
4. `ProfileScreen` → inline `TopAppBar` into content Column (no Scaffold)
5. `IdentityEditorScreen` → same
6. `ProactiveHistoryScreen` → same

All secondary screens should use `AuraScreenShell` (already exists at `ui/components/AuraScreenShell.kt`) to get consistent title+subtitle rhythm without a nested Scaffold.

### 6.3 Inset Consolidation

```
Before:           Root Scaffold bottomBarPadding + AuraBottomBar .navigationBarsPadding() + ChatInputBar .navigationBarsPadding()
After:            Root Scaffold provides ALL insets. Screens use Modifier.windowInsetsPadding(WindowInsets.systemBars.only(WindowInsetsSides.Top)).
                  Bottom bar uses Modifier.windowInsetsBottomHeight(WindowInsets.navigationBars).
                  No screen applies .statusBarsPadding() or .navigationBarsPadding() directly.
```

**Rules:**
1. Root `Scaffold` is the **only** places that calls `Modifier.padding(padding)` for bottom bar + system bars
2. `AuraBottomBar` uses `.windowInsetsBottomHeight(WindowInsets.navigationBars)` — one call only
3. `ChatScreen` removes `.statusBarsPadding()` from `ChatHeader` (chat header doesn't need it — the NavHost padding covers it)
4. `ChatScreen` removes `.navigationBarsPadding()` from `ChatInputBar` (bottom bar padding from Scaffold already covers it)
5. `.imePadding()` stays on `ChatScreen` root Box — it's a keyboard-specific inset distinct from system bars

### 6.4 Bottom Bar Size Reduction

```
Before: Surface(12dp V) + Row(4dp V) + Box(8dp V × 2 = 16dp) + Icon(20dp) + Text(~13dp) = ~59dp + navBar
After:  Surface(8dp V) + Row(0dp V) + Box(6dp V × 2 = 12dp) + Icon(18dp) + Text(~11dp) = ~43dp + navBar
```

**Changes:**
- Outer Surface vertical padding: `12.dp` → `8.dp`
- Inner Row padding: `4.dp` → `0.dp`
- Item vertical padding: `8.dp` → `6.dp`
- Icon size: `20.dp` → `18.dp`
- Font size: `10.sp` → `9.sp`

### 6.5 Responsive Proportion

Introduce a shared `ResponsiveContainer` composable:

```kotlin
@Composable
fun ResponsiveContainer(
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit
) {
    Box(
        modifier = modifier
            .fillMaxSize()
            .widthIn(max = 600.dp)
            .align(Alignment.CenterHorizontally),
        contentAlignment = Alignment.TopCenter,
    ) {
        content()
    }
}
```

Placed as the root content wrapper in `NavGraph` after the root Scaffold padding is applied, or inside each screen for fine-grained control.

---

## 7. MIGRATION SEQUENCE

```
Commit 1: [refactor(theme)] Migrate ChatScreen AuraTokens.Dark → MaterialTheme.colorScheme
  - ChatScreen.kt: all ~40 hardcoded AuraTokens.Dark refs → MaterialTheme.colorScheme.* equivalents
  - Verify: light mode ChatScreen is readable, dark mode unchanged

Commit 2: [refactor(theme)] Migrate AuraBottomBar AuraTokens.Dark → MaterialTheme.colorScheme
  - NavGraph.kt: AuraBottomBar colors use MaterialTheme.colorScheme
  - Verify: bottom bar adapts to light/dark

Commit 3: [refactor(ui)] Unnest Scaffolds — Hands, Tasks, Reminders
  - Move snackbarHost/FABs to root Scaffold or CompositionLocal
  - HandsScreen.kt, TasksScreen.kt, RemindersScreen.kt: remove nested Scaffold

Commit 4: [refactor(ui)] Unnest Scaffolds — Profile, IdentityEditor, ProactiveHistory
  - Inline topAppBar into content Column
  - ProfileScreen.kt, IdentityEditorScreen.kt, ProactiveHistoryScreen.kt: remove Scaffold

Commit 5: [fix(ui)] Consolidate insets — remove doubled statusBarsPadding/navigationBarsPadding
  - ChatScreen.kt: remove .statusBarsPadding() from ChatHeader
  - ChatScreen.kt: remove .navigationBarsPadding() from ChatInputBar
  - NavGraph.kt: AuraBottomBar uses .windowInsetsBottomHeight() once
  - Verify: no visible gap changes on any device

Commit 6: [fix(ui)] Reduce bottom bar height to standard
  - NavGraph.kt: AuraBottomBar outer padding 12→8dp, inner padding 4→0dp, item padding 8→6dp, icon 20→18dp, text 10→9sp

Commit 7: [feat(ui)] Add ResponsiveContainer for tablet/landscape proportion
  - New composable in ui/components/ResponsiveContainer.kt
  - Apply to NavGraph content or each screen
  - Verify: on 600dp+ device content is capped at 600dp and centered
```

---

## 8. TEST / SCREENSHOT GATES

| Gate | What to Capture | Pass Criteria |
|------|----------------|---------------|
| **G1** | Light mode — ChatScreen + BottomBar | All surfaces are light-toned, no dark pill floating on a light screen |
| **G2** | Dark mode — ChatScreen + BottomBar | Surfaces match AuraTokens.Dark, unchanged |
| **G3** | System bar insets — pixel-level | Bottom of AuraBottomBar ends at navigation bar top, no gap/double padding |
| **G4** | Keyboard open — ChatScreen | Input bar sits above IME, no extra gap between input & bottom bar |
| **G5** | HandsScreen — snackbar visible | Snackbar appears above bottom bar, not behind it |
| **G6** | TasksScreen — FAB visible | FAB sits above bottom bar, not behind it |
| **G7** | Tablet landscape 800dp+ | Content max-width ~600dp, centered, not edge-to-edge |
| **G8** | Phone portrait 360dp | Content fills width as before, no regression |
| **G9** | ProfileScreen — back arrow + title | TopAppBar appears without double status-bar gap |

### Suggested Implementation

```kotlin
// ui/screenshot/ScreenshotGates.kt
@RunWith(ParameterizedTest::class)
class DesignSystemScreenshotTest {
    @Test fun lightMode_chat_bottomBar() { ... }
    @Test fun darkMode_chat_bottomBar() { ... }
    @Test fun bottomBar_inset_noDouble() { ... }
    @Test fun keyboard_open_inputPosition() { ... }
}
```

Use Paparazzi or Roborazzi for screenshot-comparison tests in CI.

---

## 9. SYSTEM MAP — All Affected Files

| File | Systemic Defect |
|------|----------------|
| `ui/theme/Theme.kt` | ✅ Clean — properly sets up Light/Dark |
| `ui/theme/AuraTokens.kt` | ✅ Clean — tokens defined correctly |
| `ui/nav/NavGraph.kt` | **THEME_HARDCODING, INSET_DOUBLING, BOTTOM_BAR_OVERSIZE** |
| `ui/screens/ChatScreen.kt` | **THEME_HARDCODING, INSET_DOUBLING** |
| `ui/screens/HandsScreen.kt` | **SCAFFOLD_NESTING** |
| `ui/screens/TasksScreen.kt` | **SCAFFOLD_NESTING** |
| `ui/screens/RemindersScreen.kt` | **SCAFFOLD_NESTING** |
| `ui/screens/ProfileScreen.kt` | **SCAFFOLD_NESTING** |
| `ui/screens/IdentityEditorScreen.kt` | **SCAFFOLD_NESTING** |
| `ui/screens/ProactiveHistoryScreen.kt` | **SCAFFOLD_NESTING** |
| `ui/screens/HomeScreen.kt` | ✅ Clean (no Scaffold, uses M3 colors) |
| `ui/screens/MemoryScreen.kt` | ✅ Clean |
| `ui/screens/SettingsScreen.kt` | ✅ Clean |
| `ui/screens/HistoryScreen.kt` | ✅ Clean |
| `ui/screens/ToolsScreen.kt` | ✅ Clean |
| `ui/screens/DiagnosticsScreen.kt` | ✅ Clean |
| `ui/screens/KnowledgeGraphScreen.kt` | ✅ Clean |
| `ui/screens/OnboardingScreen.kt` | ✅ Clean |
| `ui/components/AuraScreenShell.kt` | ✅ Clean — but underused (only MemoryScreen, HistoryScreen use it) |
| `MainActivity.kt` | ✅ `enableEdgeToEdge` is correct |

---

## 10. VERDICT

**Severity:** HIGH (blocking for light-mode users)  
**Scope:** 6 files need refactoring, 7 atomic commits  
**Risk:** Low — mechanical substitutions + removal of nested Scaffolds  
**Payoff:** Light mode works correctly, bottom bar is proportional, tablet layout is usable, insets are predictable
