"""React Bits-inspired frontend effects adapted for Streamlit component frames."""

from __future__ import annotations

from html import escape

import streamlit as st


def render_animated_hero(kicker: str, heading: str, body: str) -> None:
    """Render ShinyText and BorderGlow behavior without a separate React build.

    Streamlit does not expose its React component tree to Python, so the supplied
    React Bits behavior is translated into an isolated HTML component. The
    pointer geometry and border layers follow BorderGlow; the heading gradient
    follows ShinyText.
    """

    safe_kicker = escape(kicker)
    safe_heading = escape(heading)
    safe_body = escape(body)
    st.html(
        f"""
<style>
  .vita-react-bits {{
    display: block;
    width: 100%;
    font-family: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  }}
  .vita-react-bits,
  .vita-react-bits * {{ box-sizing: border-box; }}

  .vita-react-bits .border-glow-card {{
    --edge-proximity: 0;
    --cursor-angle: 110deg;
    --edge-sensitivity: 24;
    --color-sensitivity: 44;
    --border-radius: 24px;
    --glow-padding: 26px;
    --cone-spread: 24;
    --card-bg: #173f35;
    --fill-opacity: .34;
    --glow-color: hsl(72deg 68% 67% / 100%);
    --glow-color-60: hsl(72deg 68% 67% / 60%);
    --glow-color-50: hsl(72deg 68% 67% / 50%);
    --glow-color-40: hsl(72deg 68% 67% / 40%);
    --glow-color-30: hsl(72deg 68% 67% / 30%);
    --glow-color-20: hsl(72deg 68% 67% / 20%);
    --glow-color-10: hsl(72deg 68% 67% / 10%);
    --gradient-one: radial-gradient(at 80% 55%, #d5e76b 0, transparent 50%);
    --gradient-two: radial-gradient(at 69% 34%, #78c8a2 0, transparent 50%);
    --gradient-three: radial-gradient(at 8% 6%, #eff7b5 0, transparent 50%);
    --gradient-four: radial-gradient(at 41% 38%, #b2dbc1 0, transparent 50%);
    --gradient-five: radial-gradient(at 86% 85%, #91c5ad 0, transparent 50%);
    --gradient-six: radial-gradient(at 82% 18%, #f0dfa0 0, transparent 50%);
    --gradient-seven: radial-gradient(at 51% 4%, #c8e293 0, transparent 50%);
    --gradient-base: linear-gradient(#d5e76b 0 100%);
    position: relative;
    isolation: isolate;
    transform: translate3d(0, 0, .01px);
    display: grid;
    margin: 26px;
    border: 1px solid rgb(255 255 255 / 15%);
    border-radius: var(--border-radius);
    background: var(--card-bg);
    overflow: visible;
    box-shadow:
      rgba(23, 63, 53, .07) 0 2px 4px,
      rgba(23, 63, 53, .09) 0 8px 18px,
      rgba(23, 63, 53, .12) 0 22px 48px;
  }}

  .vita-react-bits .border-glow-card::before,
  .vita-react-bits .border-glow-card::after,
  .vita-react-bits .border-glow-card > .edge-light {{
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    transition: opacity .25s ease-out;
    z-index: -1;
  }}

  .vita-react-bits .border-glow-card:not(:hover):not(.sweep-active)::before,
  .vita-react-bits .border-glow-card:not(:hover):not(.sweep-active)::after,
  .vita-react-bits .border-glow-card:not(:hover):not(.sweep-active) > .edge-light {{
    opacity: 0;
    transition: opacity .75s ease-in-out;
  }}

  .vita-react-bits .border-glow-card::before {{
    border: 1px solid transparent;
    background:
      linear-gradient(var(--card-bg) 0 100%) padding-box,
      linear-gradient(rgb(255 255 255 / 0%) 0 100%) border-box,
      var(--gradient-one) border-box,
      var(--gradient-two) border-box,
      var(--gradient-three) border-box,
      var(--gradient-four) border-box,
      var(--gradient-five) border-box,
      var(--gradient-six) border-box,
      var(--gradient-seven) border-box,
      var(--gradient-base) border-box;
    opacity: calc((var(--edge-proximity) - var(--color-sensitivity)) / (100 - var(--color-sensitivity)));
    -webkit-mask-image: conic-gradient(
      from var(--cursor-angle) at center,
      #000 calc(var(--cone-spread) * 1%),
      transparent calc((var(--cone-spread) + 15) * 1%),
      transparent calc((100 - var(--cone-spread) - 15) * 1%),
      #000 calc((100 - var(--cone-spread)) * 1%)
    );
    mask-image: conic-gradient(
      from var(--cursor-angle) at center,
      #000 calc(var(--cone-spread) * 1%),
      transparent calc((var(--cone-spread) + 15) * 1%),
      transparent calc((100 - var(--cone-spread) - 15) * 1%),
      #000 calc((100 - var(--cone-spread)) * 1%)
    );
  }}

  .vita-react-bits .border-glow-card::after {{
    border: 1px solid transparent;
    background:
      var(--gradient-one) padding-box,
      var(--gradient-two) padding-box,
      var(--gradient-three) padding-box,
      var(--gradient-four) padding-box,
      var(--gradient-five) padding-box,
      var(--gradient-six) padding-box,
      var(--gradient-seven) padding-box,
      var(--gradient-base) padding-box;
    -webkit-mask-image:
      linear-gradient(to bottom, #000, #000),
      radial-gradient(ellipse at 50% 50%, #000 40%, transparent 65%),
      radial-gradient(ellipse at 66% 66%, #000 5%, transparent 40%),
      radial-gradient(ellipse at 33% 33%, #000 5%, transparent 40%),
      conic-gradient(from var(--cursor-angle) at center, transparent 5%, #000 15%, #000 85%, transparent 95%);
    mask-image:
      linear-gradient(to bottom, #000, #000),
      radial-gradient(ellipse at 50% 50%, #000 40%, transparent 65%),
      radial-gradient(ellipse at 66% 66%, #000 5%, transparent 40%),
      radial-gradient(ellipse at 33% 33%, #000 5%, transparent 40%),
      conic-gradient(from var(--cursor-angle) at center, transparent 5%, #000 15%, #000 85%, transparent 95%);
    -webkit-mask-composite: xor, source-over, source-over, source-over;
    mask-composite: subtract, add, add, add;
    opacity: calc(var(--fill-opacity) * (var(--edge-proximity) - var(--color-sensitivity)) / (100 - var(--color-sensitivity)));
    mix-blend-mode: soft-light;
  }}

  .vita-react-bits .edge-light {{
    inset: calc(var(--glow-padding) * -1) !important;
    pointer-events: none;
    z-index: 1 !important;
    -webkit-mask-image: conic-gradient(
      from var(--cursor-angle) at center, #000 2.5%, transparent 10%, transparent 90%, #000 97.5%
    );
    mask-image: conic-gradient(
      from var(--cursor-angle) at center, #000 2.5%, transparent 10%, transparent 90%, #000 97.5%
    );
    opacity: calc((var(--edge-proximity) - var(--edge-sensitivity)) / (100 - var(--edge-sensitivity)));
    mix-blend-mode: plus-lighter;
  }}

  .vita-react-bits .edge-light::before {{
    content: "";
    position: absolute;
    inset: var(--glow-padding);
    border-radius: inherit;
    box-shadow:
      inset 0 0 0 1px var(--glow-color),
      inset 0 0 3px var(--glow-color-50),
      inset 0 0 8px var(--glow-color-40),
      inset 0 0 20px 2px var(--glow-color-20),
      0 0 3px var(--glow-color-50),
      0 0 9px var(--glow-color-30),
      0 0 28px 2px var(--glow-color-10);
  }}

  .vita-react-bits .border-glow-inner {{
    position: relative;
    z-index: 1;
    min-height: 190px;
    padding: 31px 34px 29px;
    overflow: hidden;
    border-radius: calc(var(--border-radius) - 1px);
    background:
      radial-gradient(circle at 90% 15%, rgb(213 231 107 / 30%), transparent 12rem),
      linear-gradient(135deg, #173f35 0%, #2f6753 100%);
  }}

  .vita-react-bits .kicker {{
    margin: 0 0 8px;
    color: #d5e76b;
    font-size: 12px;
    line-height: 1.2;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
  }}

  .vita-react-bits .shiny-text {{
    display: inline-block;
    max-width: 100%;
    margin: 0 0 13px;
    color: #e8efe6;
    font-size: clamp(35px, 6vw, 68px);
    line-height: .96;
    font-weight: 800;
    letter-spacing: -.045em;
    background-image: linear-gradient(120deg, #e8efe6 0%, #e8efe6 35%, #fff 50%, #d5e76b 56%, #e8efe6 65%, #e8efe6 100%);
    background-size: 220% auto;
    background-position: 150% center;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shiny-sweep 4.4s linear .4s infinite;
  }}

  .vita-react-bits .description {{
    max-width: 760px;
    margin: 0;
    color: #e8efe6;
    font-size: clamp(15px, 2vw, 17px);
    line-height: 1.55;
  }}

  @keyframes shiny-sweep {{
    0% {{ background-position: 150% center; }}
    62% {{ background-position: -50% center; }}
    100% {{ background-position: -50% center; }}
  }}

  @media (max-width: 680px) {{
    .vita-react-bits .border-glow-card {{ margin: 18px 8px 20px; --glow-padding: 14px; }}
    .vita-react-bits .border-glow-inner {{ min-height: 184px; padding: 26px 23px; }}
    .vita-react-bits .shiny-text {{ font-size: 39px; }}
  }}

  @media (prefers-reduced-motion: reduce) {{
    .vita-react-bits .shiny-text {{
      animation: none;
      background-position: 50% center;
    }}
    .vita-react-bits .border-glow-card::before,
    .vita-react-bits .border-glow-card::after,
    .vita-react-bits .border-glow-card > .edge-light {{
      transition: none;
    }}
  }}
</style>
<div class="vita-react-bits">
  <section class="border-glow-card" id="vita-border-glow" aria-label="{safe_kicker}: {safe_heading}">
    <span class="edge-light" aria-hidden="true"></span>
    <div class="border-glow-inner">
      <p class="kicker">{safe_kicker}</p>
      <h1 class="shiny-text">{safe_heading}</h1>
      <p class="description">{safe_body}</p>
    </div>
  </section>
</div>
<script>
(() => {{
  const card = document.getElementById("vita-border-glow");
  if (!card) return;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function geometry(event) {{
    const rect = card.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    const dx = x - cx;
    const dy = y - cy;
    const kx = dx === 0 ? Infinity : cx / Math.abs(dx);
    const ky = dy === 0 ? Infinity : cy / Math.abs(dy);
    const edge = Math.min(Math.max(1 / Math.min(kx, ky), 0), 1);
    let angle = Math.atan2(dy, dx) * (180 / Math.PI) + 90;
    if (angle < 0) angle += 360;
    card.style.setProperty("--edge-proximity", (edge * 100).toFixed(3));
    card.style.setProperty("--cursor-angle", angle.toFixed(3) + "deg");
  }}

  card.addEventListener("pointermove", geometry, {{ passive: true }});

  if (!reduceMotion) {{
    card.classList.add("sweep-active");
    const start = performance.now();
    const duration = 3100;
    function intro(now) {{
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      const proximity = t < .7 ? Math.min(t * 165, 100) : Math.max((1 - t) * 333, 0);
      card.style.setProperty("--edge-proximity", proximity.toFixed(2));
      card.style.setProperty("--cursor-angle", (110 + 355 * eased).toFixed(2) + "deg");
      if (t < 1) requestAnimationFrame(intro);
      else card.classList.remove("sweep-active");
    }}
    requestAnimationFrame(intro);
  }}
}})();
</script>
        """,
        width="stretch",
        unsafe_allow_javascript=True,
    )
