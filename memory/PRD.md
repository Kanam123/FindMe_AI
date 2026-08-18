# FINDME AI — Frontend Redesign (Flask/Jinja)

## Scope
Visual-only redesign of the existing Flask app in `/app/final_project`. Backend FROZEN — no Python, services, repositories, models, routes, or AI logic changed.

## Design System
- Dark-default foundation with light theme toggle (persisted in localStorage).
- Accent: blue (#3b82f6 / #2563eb). Fonts: Sora (display), Manrope (body), JetBrains Mono (labels).
- Lucide icons (CDN), soft shadows, consistent radii, subtle staggered animations, glass nav.

## Files Changed (frontend only)
- `static/css/app.css` — full design-system rewrite (was incomplete; many template classes had no styles).
- `static/javascript/engine.js` — theme toggle, mobile nav drawer, drag-drop upload + preview/remove, clear search, loading state, password visibility, people filter/sort, lucide init.
- `Templates/base.html` — fonts, icons, theme toggle, responsive nav, test IDs.
- `Templates/*.html` — index, search, result, profile, dashboard, ai_index, login, signup, people, about redesigned.

## Verified
- All 13 routes return 200; text search renders result cards; profile/dashboard/people/login visually confirmed (dark).
- Form field names, actions, and routes unchanged. No backend files modified (git status confirms).

## Notes
- 4 demo profiles were seeded via the existing `/api/people` endpoint (DB was empty) so results/dashboard render meaningfully. Deletable via the app; no backend/schema change.
- Light theme + mobile verified via CSS implementation (breakpoints 1024/860/560, hamburger drawer); screenshot tool only renders desktop/dark GET state.

## Backlog / Next
- P2: Client-side results sort wiring, skeleton loaders, empty-state illustrations.
