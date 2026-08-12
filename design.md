# Craft, engineered · wild

## Mission
Create implementation-ready, token-driven UI guidance for Craft, engineered · wild that is optimized for consistency, accessibility, and fast delivery across e-commerce storefront.

## Brand
- Product/brand: Craft, engineered · wild
- URL: https://craft.wild.as/
- Audience: online shoppers and consumers
- Product surface: e-commerce storefront

## Style Foundations
- Visual style: clean, functional, implementation-oriented
- Main font style: `font.family.primary=Sneak`, `font.family.stack=Sneak, Helvetica Neue, Helvetica, Arial, sans-serif`, `font.size.base=16px`, `font.weight.base=400`, `font.lineHeight.base=24px`
- Typography scale: `font.size.xs=10px`, `font.size.sm=11px`, `font.size.md=13px`, `font.size.lg=13.5px`, `font.size.xl=14px`, `font.size.2xl=14.5px`, `font.size.3xl=15px`, `font.size.4xl=16px`
- Color palette: `color.text.primary=#0a0a0a`, `color.text.secondary=#2a2a2a`, `color.text.tertiary=#8b8b8b`, `color.text.inverse=#ffffff`, `color.surface.base=#000000`, `color.border.default=#e7e5e4`, `color.border.strong=rgba(10, 10, 10, 0.12) rgb(231, 229, 228) rgb(231, 229, 228)`
- Spacing scale: `space.1=2px`, `space.2=9px`, `space.3=14px`, `space.4=16px`, `space.5=17.39px`, `space.6=18px`, `space.7=19.87px`, `space.8=20px`
- Radius/shadow/motion tokens: `motion.duration.instant=200ms`, `motion.duration.fast=350ms`, `motion.duration.normal=400ms`, `motion.duration.slow=450ms`, `motion.duration.slower=550ms`, `motion.duration.step6=800ms`

## Accessibility
- Target: WCAG 2.2 AA
- Keyboard-first interactions required.
- Focus-visible rules required.
- Contrast constraints required.

## Writing Tone
Concise, confident, implementation-focused.

## Rules: Do
- Use semantic tokens, not raw hex values, in component guidance.
- Every component must define states for default, hover, focus-visible, active, disabled, loading, and error.
- Component behavior should specify responsive and edge-case handling.
- Interactive components must document keyboard, pointer, and touch behavior.
- Accessibility acceptance criteria must be testable in implementation.

## Rules: Don't
- Do not allow low-contrast text or hidden focus indicators.
- Do not introduce one-off spacing or typography exceptions.
- Do not use ambiguous labels or non-descriptive actions.
- Do not ship component guidance without explicit state rules.

## Guideline Authoring Workflow
1. Restate design intent in one sentence.
2. Define foundations and semantic tokens.
3. Define component anatomy, variants, interactions, and state behavior.
4. Add accessibility acceptance criteria with pass/fail checks.
5. Add anti-patterns, migration notes, and edge-case handling.
6. End with a QA checklist.

## Required Output Structure
- Context and goals.
- Design tokens and foundations.
- Component-level rules (anatomy, variants, states, responsive behavior).
- Accessibility requirements and testable acceptance criteria.
- Content and tone standards with examples.
- Anti-patterns and prohibited implementations.
- QA checklist.

## Component Rule Expectations
- Include keyboard, pointer, and touch behavior.
- Include spacing and typography token requirements.
- Include long-content, overflow, and empty-state handling.
- Include known page component density: buttons (18), links (17), lists (1).


## Quality Gates
- Every non-negotiable rule must use "must".
- Every recommendation should use "should".
- Every accessibility rule must be testable in implementation.
- Teams should prefer system consistency over local visual exceptions.
