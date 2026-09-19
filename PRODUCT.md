# Product

## Register

product

## Users

Primary: hackathon judges watching a live demo on a laptop or projector, usually over the presenter's shoulder, with a few minutes of attention. They need to understand what the agent did and whether it worked within seconds, without reading documentation.

Secondary: the presenter driving the demo (an ML engineer on the team), who needs every control reachable without scrolling away from the conversation, and needs the screen to stay legible while a job runs for minutes.

## Product Purpose

dispatch is a distributed compute workspace: it tracks jobs across a fleet of workers and, in GPU Lab, lets you talk to a model running on a rented GPU while agents optimize training and inference. It exists to show, with measured evidence, that the agents make GPU work faster without changing the answers.

Success is a judge saying "I saw it get faster, and I believe the number" after one before/after run.

## Brand Personality

Calm, exact, honest. The voice of a careful engineer: plain statements, measured numbers, caveats stated rather than hidden. The chat is the product surface, in the lineage of ChatGPT and Claude: quiet, spacious, conversation first. Confidence comes from restraint and precision, not from decoration.

## Anti-references

- Dark neon AI dashboards: glowing gradients, glass cards, purple and cyan on black.
- Generic SaaS admin templates: identical card grids, hero metrics, stock panel layouts.
- Cluttered research consoles: one long page where every panel competes, like the original GPUShare page.
- Toy or playful interfaces: emoji, bouncy motion, cartoon rounding that undercuts the measurements.

## Design Principles

1. The conversation is the stage. Everything else is supporting evidence and stays out of the way until asked for.
2. One glance, one claim. At any moment the screen should make a single thing obvious: what is running, or what changed.
3. Measured, never estimated. Numbers carry their conditions; anything unmeasured is left blank. Caveats are part of the design, not fine print.
4. Progress is always visible. A long wait must never look like a hang, from across a room.
5. Belong to dispatch. GPU Lab is a room in the same house: same type, same green, same restraint.

## Accessibility & Inclusion

WCAG 2.1 AA as the baseline: 4.5:1 text contrast, visible focus, full keyboard operation of the chat and controls. Must stay legible on a washed-out projector, so state is never carried by color alone. Respect `prefers-reduced-motion`.
