---
name: dispatch
description: Distributed compute workspace. Calm, exact, light only.
colors:
  dispatch-green: "#197253"
  brandmark-green: "#217956"
  progress-green: "#438d6e"
  focus-green: "#58a78c"
  pressed-border: "#9ac4ae"
  pressed-ink: "#234d3a"
  soft-mint: "#eef6f2"
  canvas: "#f7f8fa"
  panel: "#ffffff"
  wash: "#fbfcfd"
  well: "#f2f5f7"
  ink: "#202b33"
  muted: "#66737c"
  muted-strong: "#56636b"
  hairline: "#e3e8ec"
  control-border: "#d8e0e5"
  warn-bg: "#fff8e8"
  warn-ink: "#865711"
  danger-ink: "#a43c32"
  danger-bg: "#fdf3f1"
  toast: "#23362e"
typography:
  display:
    fontFamily: "'Space Grotesk', 'DM Sans', system-ui, sans-serif"
    fontSize: "31px"
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: "-1.1px"
  display-compact:
    fontFamily: "'Space Grotesk', 'DM Sans', system-ui, sans-serif"
    fontSize: "20px"
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: "-0.7px"
  headline:
    fontFamily: "'DM Sans', system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 600
    lineHeight: 1.5
  title:
    fontFamily: "'DM Sans', system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 650
    lineHeight: 1.5
  body:
    fontFamily: "'DM Sans', system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "'DM Sans', system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 650
    letterSpacing: "1px"
  figure:
    fontFamily: "'Space Grotesk', 'DM Sans', system-ui, sans-serif"
    fontSize: "31px"
    fontWeight: 500
    letterSpacing: "-1px"
  mono:
    fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace"
    fontSize: "12px"
    lineHeight: 1.6
rounded:
  control: "6px"
  panel: "9px"
  bubble: "18px"
  pill: "999px"
spacing:
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "20px"
  xl: "28px"
  page: "36px"
components:
  button-primary:
    backgroundColor: "{colors.dispatch-green}"
    textColor: "{colors.panel}"
    rounded: "{rounded.control}"
    padding: "9px 14px"
  button-outline:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "9px 14px"
  button-pressed:
    backgroundColor: "{colors.soft-mint}"
    textColor: "{colors.pressed-ink}"
    rounded: "{rounded.control}"
  button-text:
    backgroundColor: "transparent"
    textColor: "{colors.muted}"
    padding: "6px"
  panel:
    backgroundColor: "{colors.panel}"
    rounded: "{rounded.panel}"
    padding: "18px 20px"
  input:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "10px 12px"
  pill:
    backgroundColor: "{colors.soft-mint}"
    textColor: "{colors.dispatch-green}"
    rounded: "5px"
    padding: "4px 8px"
  pill-neutral:
    backgroundColor: "#edf0f2"
    textColor: "{colors.muted-strong}"
    rounded: "5px"
    padding: "4px 8px"
---

# Design System: dispatch

## 1. Overview

**Creative North Star: "The Lab Notebook"**

dispatch looks like a careful engineer's notebook: white pages on a pale grey desk, hairline rules, one green pen. Nothing glows, nothing floats, nothing is decorated. Numbers are set in a slightly technical grotesque so they read as measurements; everything else is a quiet humanist sans. The interface earns trust by being plain and exact.

It is light only. There is no dark theme and none may be added. It rejects, by name, the dark neon AI dashboard (glowing gradients, glass cards, purple and cyan on black), the generic SaaS admin template (identical card grids, hero metrics), the cluttered research console where every panel competes, and anything toy-like or playful.

Layout is an app shell: a fixed 218px sidebar, a 68px top bar, and a content column padded 36px / 38px, capped at 1600px. Small surfaces (a desktop window, a dialog) keep the same panels and rhythm in a single column.

**Key Characteristics:**
- Light only: pale grey canvas, white panels, 1px hairlines.
- One accent, dispatch green, for primary actions, current selection, and healthy state.
- Two type families: Space Grotesk for display and figures, DM Sans for everything else.
- Flat. Depth comes from borders and tone, not shadows.
- 12px minimum text size everywhere.

## 2. Colors

A restrained palette: cool grey-blue neutrals and a single deep green.

### Primary
- **Dispatch Green** (`dispatch-green`): primary buttons, links, the active nav item's text, healthy status, chart "after" series. Never used as a large fill.
- **Brandmark Green** (`brandmark-green`): the logo tile only.
- **Soft Mint** (`soft-mint`): the tint behind selected or pressed things (active nav row, pressed toggle, user chat bubble, success callout). Paired with `pressed-border` and `pressed-ink`.
- **Progress Green** (`progress-green`) fills progress bars; **Focus Green** (`focus-green`) is the 3px focus ring.

### Neutral
- **Canvas** (`canvas`): the page. **Panel** (`panel`): every card, table, dialog. **Wash** (`wash`) and **Well** (`well`): toolbars, hover rows, code and log blocks.
- **Ink** (`ink`): text. **Muted** (`muted`): secondary text on white only. **Muted Strong** (`muted-strong`): secondary text on any tinted background.
- **Hairline** (`hairline`): panel borders and dividers. **Control Border** (`control-border`): inputs and outline buttons.

### Tertiary
- **Warn** (`warn-bg` + `warn-ink`) and **Danger** (`danger-bg` + `danger-ink`): state only, always with an icon or a word, never colour alone.
- **Toast** (`toast`): the one dark surface, a small transient notification.

### Named Rules
**The One Pen Rule.** Green is the only accent. If a screen needs a second hue to be understood, the layout is wrong.

**The Tint Contrast Rule.** `muted` fails 4.5:1 on tinted backgrounds. On anything that is not white, secondary text uses `muted-strong`.

## 3. Typography

**Display Font:** Space Grotesk (with DM Sans, system-ui)
**Body Font:** DM Sans (with system-ui, sans-serif)
**Label/Mono Font:** ui-monospace, SFMono-Regular, Consolas

**Character:** A technical grotesque for headings and measured numbers, a warm plain sans for reading. Tight negative tracking on large sizes; nothing is ever all-caps except the eyebrow label.

### Hierarchy
- **Display** (600, 31px, 1.25, -1.1px): the page title, one per view.
- **Display Compact** (600, 20px, -0.7px): the brand lockup and the one state line on small surfaces such as the desktop agent window.
- **Figure** (500, 31px, -1px, tabular): headline metrics and any number that is the point of the screen.
- **Headline** (600, 17px): panel titles.
- **Title** (650, 14px): sub-headings inside a panel.
- **Body** (400, 14px, 1.5): default text. Prose is capped near 70ch.
- **Label** (650, 12px, 1px tracking, uppercase, `muted`): the eyebrow above a title.
- **Mono** (12px, 1.6): ids, logs, code.

### Named Rules
**The Twelve Pixel Floor.** No text is smaller than 12px. This UI is demoed on projectors.

**The Measured Number Rule.** Numbers that carry a claim are set in Space Grotesk with tabular figures, and sit next to the condition they were measured under.

## 4. Elevation

Flat by default. Panels are white on a grey canvas with a 1px `hairline` border; that is the whole depth system. Shadows are reserved for things that genuinely float.

### Shadow Vocabulary
- **Button rest** (`box-shadow: 0 2px 3px #16392a10`): primary button only.
- **Floating composer** (`box-shadow: 0 4px 18px #1f2b3310`): the chat composer.
- **Toast** (`box-shadow: 0 8px 28px #19312725`): transient notification.
- **Focus ring** (`outline: 3px solid #58a78c; outline-offset: 3px`): every focusable control.

### Named Rules
**The Hairline Rule.** A panel is a 1px border, not a shadow. Panels are never nested inside panels; inside a panel, separate with a hairline or a tint.

## 5. Components

### Buttons
- **Shape:** softly squared (6px radius), 12px / 550 label, 9px 14px padding, icon gap 8px.
- **Primary:** dispatch green fill, white label, 1px green border. One per view.
- **Outline:** white fill, `control-border`. **Pressed / selected:** soft mint fill, `pressed-border`, `pressed-ink` label.
- **Text:** no border, `muted` label; `danger-ink` for destructive.
- **Hover / Focus:** `filter: brightness(0.97)` on hover, 150ms; the focus ring above. Disabled is 45% opacity.

### Chips
- **Status pill:** soft mint fill, green 12px label, 5px radius, optional 5px pulse dot. **Neutral:** `#edf0f2` fill, `muted-strong` label.
- **Count badge:** small mint tile with a number, right-aligned in nav rows and tabs.

### Cards / Containers
- **Corner Style:** 9px. **Background:** panel white. **Border:** 1px hairline. **Shadow:** none.
- **Internal Padding:** 18px 20px; toolbars 17px 19px with a wash background.
- **Callout:** a tinted block (well, soft mint, warn, danger) with a leading icon and a bold first line. No side stripes.
- **Ledger rows:** label left, value right in tabular figures, condition beneath in muted; rows separated by hairlines. Used instead of stat-tile grids.

### Inputs / Fields
- **Style:** white, 1px `#dce3e7` border, 6px radius, 10px 12px padding. Label above: 12px / 550.
- **Focus:** the focus ring. **Error:** `danger-ink` text under the field, with an icon.
- **Segmented toggle:** pill-shaped well with a white pressed segment and green label.

### Navigation
- **Sidebar:** brand lockup (green tile with arrow + "dispatch." in Space Grotesk, green full stop), a workspace row, an eyebrow label, then rows of icon + label + count. Active row: soft mint fill, green text.
- **Top bar:** breadcrumb left, connection state (dot + words) and account right.
- **Small surfaces:** the same brand lockup as a compact header with the connection state opposite it.

### Progress and state
- **Progress bar:** 4px track (`#e9eef0`), progress green fill, animated with `transform: scaleX`, never `width`.
- **Connection dot:** green live, amber reconnecting, always with a word beside it.
- **Waiting:** count elapsed time out loud in figures; a still screen reads as a hang.

## 6. Do's and Don'ts

### Do:
- **Do** stay light only: canvas `#f7f8fa`, panels `#ffffff`, hairlines `#e3e8ec`.
- **Do** use dispatch green `#197253` for the single primary action, selection, and healthy state, and nothing else.
- **Do** pair every state colour with an icon or a word; the UI must survive a washed-out projector.
- **Do** keep text at 12px or larger, and secondary text on tints at `#56636b`.
- **Do** set measured numbers in Space Grotesk with tabular figures beside their condition.
- **Do** honour `prefers-reduced-motion`; motion only conveys state (150 to 250ms, ease-out).

### Don't:
- **Don't** add a dark theme, `prefers-color-scheme: dark`, or `color-scheme: dark`. Ever.
- **Don't** build a dark neon AI dashboard: no glowing gradients, no glass cards, no purple or cyan on black.
- **Don't** build a generic SaaS admin template: no identical card grids, no hero-metric tiles.
- **Don't** build a cluttered research console: one long page where every panel competes.
- **Don't** make it toy-like: no emoji, no bouncy motion, no cartoon rounding.
- **Don't** use `border-left` or `border-right` wider than 1px as a coloured accent stripe.
- **Don't** nest a bordered box inside a bordered panel.
- **Don't** use gradient text, backdrop blur, or a second accent hue.
- **Don't** use em dashes in interface copy; use a colon, comma, or full stop.
