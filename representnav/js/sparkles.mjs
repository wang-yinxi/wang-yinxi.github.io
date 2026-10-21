// Twinkling star dust behind the "Dream" chapter title. Pauses offscreen and for reduced motion.
export function mountSparkles(canvas, title) {
  if (!canvas || !title) return;
  const ctx = canvas.getContext('2d');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let width = 0, height = 0, ratio = 1, raf = 0, visible = false, stars = [], dust = [];
  const rand = (a, b) => a + Math.random() * (b - a);

  function layout() {
    const rect = canvas.getBoundingClientRect();
    ratio = Math.min(2, window.devicePixelRatio || 1);
    width = rect.width;
    height = rect.height;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const t = title.getBoundingClientRect();
    const box = { x: t.left - rect.left, y: t.top - rect.top, w: t.width, h: t.height };
    stars = Array.from({ length: 34 }, (_, i) => {
      // Most sparkles hug the title; a few drift further out.
      const near = i < 24;
      return {
        x: near ? rand(box.x - 40, box.x + box.w + 40) : rand(0, width),
        y: near ? rand(box.y - 50, box.y + box.h + 30) : rand(0, Math.min(height, box.y + box.h + 300)),
        r: near ? rand(4, 11) : rand(2.5, 6), phase: rand(0, Math.PI * 2), speed: rand(.6, 1.6),
        hue: Math.random() < .25 ? 'gold' : 'blue',
      };
    });
    dust = Array.from({ length: 140 }, () => ({ x: rand(0, width), y: rand(0, height), r: rand(.4, 1.6), vy: rand(-6, -2), vx: rand(-2, 2), a: rand(.15, .6) }));
  }

  function star(x, y, r, alpha, gold) {
    const core = gold ? '255,236,190' : '214,228,255';
    const g = ctx.createRadialGradient(x, y, 0, x, y, r * 2.4);
    g.addColorStop(0, `rgba(${core},${alpha * .55})`);
    g.addColorStop(1, `rgba(${core},0)`);
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(x, y, r * 2.4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = `rgba(255,255,255,${alpha})`;
    ctx.beginPath();
    ctx.moveTo(x, y - r * 1.6);
    ctx.quadraticCurveTo(x, y, x + r * 1.6, y);
    ctx.quadraticCurveTo(x, y, x, y + r * 1.6);
    ctx.quadraticCurveTo(x, y, x - r * 1.6, y);
    ctx.quadraticCurveTo(x, y, x, y - r * 1.6);
    ctx.fill();
  }

  let last = 0;
  function draw(now) {
    raf = 0;
    const dt = last ? Math.min(.05, (now - last) / 1000) : 0;
    last = now;
    const t = now / 1000;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    for (const d of dust) {
      d.y += d.vy * dt; d.x += d.vx * dt;
      if (d.y < -4) { d.y = height + 4; d.x = rand(0, width); }
      ctx.fillStyle = `rgba(190,210,255,${d.a})`;
      ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2); ctx.fill();
    }
    for (const s of stars) {
      const twinkle = Math.pow(Math.max(0, Math.sin(t * s.speed + s.phase)), 3);
      if (twinkle < .02) continue;
      star(s.x, s.y, s.r * (.6 + .4 * twinkle), twinkle, s.hue === 'gold');
    }
    if (visible && !reduced) raf = requestAnimationFrame(draw);
  }

  new ResizeObserver(() => { layout(); if (!raf) raf = requestAnimationFrame(draw); }).observe(canvas);
  new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting;
    if (visible && !raf) { last = 0; raf = requestAnimationFrame(draw); }
  }).observe(canvas);
  document.fonts?.ready.then(layout);
}
