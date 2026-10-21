/** Outcomes are relative to the strongest baseline in the same reported setting. */
export function compareResult(value, baseline, direction = 'up') {
  if (!Number.isFinite(value) || !Number.isFinite(baseline)) return 'unreported';
  const delta = value - baseline;
  if (Math.abs(delta) < 1e-9) return 'tie';
  return (direction === 'down' ? delta < 0 : delta > 0) ? 'win' : 'loss';
}

export function formatScore(value, precision = 1) {
  return Number.isFinite(value) ? value.toFixed(precision) : '—';
}

export function formatDelta(value, baseline, { unit = 'rate', precision = 1 } = {}) {
  if (!Number.isFinite(value) || !Number.isFinite(baseline)) return 'Unreported';
  const delta = Number((value - baseline).toFixed(precision));
  const sign = delta > 0 ? '+' : delta < 0 ? '−' : '';
  const suffix = unit === 'm' ? 'm' : unit === 'points' ? 'points' : 'pp';
  return `${sign}${Math.abs(delta).toFixed(precision)} ${suffix}`;
}

let mountCount = 0;

/** Insert the benchmark evidence before an existing ablation/results layout. */
export async function mountBenchmarks(container, { data, initialBenchmark = 'vlnverse' } = {}) {
  if (!container?.ownerDocument) throw new TypeError('mountBenchmarks requires a container element.');
  if (!data) {
    const response = await fetch(new URL('../data/benchmarks.json', import.meta.url));
    if (!response.ok) throw new Error(`Benchmark results could not be loaded (${response.status}).`);
    data = await response.json();
  }
  if (!data.benchmarks?.length) throw new Error('Benchmark results are missing.');
  const document = container.ownerDocument;
  const prefix = `benchmarks-${++mountCount}`;
  const make = (tag, className, text) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  };
  const section = make('section', 'benchmark-evidence');
  section.setAttribute('aria-labelledby', `${prefix}-title`);
  const header = make('div', 'benchmark-header');
  const title = make('h3', 'benchmark-title', data.title);
  title.id = `${prefix}-title`;
  header.append(title, make('p', 'benchmark-subtitle', data.subtitle));
  section.append(header);

  const tabs = make('div', 'benchmark-headlines');
  tabs.setAttribute('role', 'tablist');
  tabs.setAttribute('aria-label', 'Choose a benchmark to inspect all reported metrics');
  const panel = make('div', 'benchmark-panel');
  panel.id = `${prefix}-panel`;
  panel.setAttribute('role', 'tabpanel');
  panel.tabIndex = 0;
  const buttons = new Map();
  const listeners = [];
  let selectedId;
  let destroyed = false;

  function listen(target, event, handler) {
    target.addEventListener(event, handler);
    listeners.push(() => target.removeEventListener(event, handler));
  }

  for (const item of data.benchmarks) {
    const row = item.rows.find(entry => entry.id === item.headline);
    const outcome = compareResult(row.representNav, row.baseline.value, row.direction);
    const button = make('button', `benchmark-card is-${outcome}`);
    button.type = 'button';
    button.id = `${prefix}-tab-${item.id}`;
    button.dataset.benchmark = item.id;
    button.setAttribute('role', 'tab');
    button.setAttribute('aria-controls', panel.id);
    button.append(make('span', 'benchmark-name', item.name));
    button.append(make('span', 'benchmark-setting', item.headlineSetting));
    const score = make('span', 'benchmark-score', formatScore(row.representNav, row.precision));
    if (row.unit === 'rate') score.append(make('span', 'benchmark-unit', '%'));
    button.append(score);
    button.append(make('span', 'benchmark-metric', item.headlineMetric));
    button.append(make('span', `benchmark-delta is-${outcome}`,
      formatDelta(row.representNav, row.baseline.value, row)));
    button.append(make('span', 'benchmark-comparator',
      `vs ${formatScore(row.baseline.value, row.precision)} · ${row.baseline.name}`));
    listen(button, 'click', () => select(item.id));
    buttons.set(item.id, button);
    tabs.append(button);
  }

  function select(id) {
    if (destroyed) return;
    const item = data.benchmarks.find(entry => entry.id === id);
    if (!item) return;
    selectedId = id;
    for (const [key, button] of buttons) {
      const selected = key === id;
      button.setAttribute('aria-selected', String(selected));
      button.tabIndex = selected ? 0 : -1;
      button.classList.toggle('is-active', selected);
    }
    panel.setAttribute('aria-labelledby', buttons.get(id).id);
    panel.dataset.benchmark = id;
    const heading = make('div', 'benchmark-panel-header');
    heading.append(make('h4', 'benchmark-panel-title', item.title));
    heading.append(make('p', 'benchmark-panel-description',
      'Every RepresentNav metric, compared with the strongest reported baseline for that setting.'));

    const wrap = make('div', 'benchmark-table-wrap');
    wrap.tabIndex = 0;
    wrap.setAttribute('role', 'region');
    wrap.setAttribute('aria-label', `${item.title} comparison table`);
    const table = make('table', 'benchmark-table');
    table.append(make('caption', 'benchmark-table-caption', `${item.title} · Table 1`));
    const head = make('thead');
    const labels = make('tr');
    for (const label of ['Setting', 'Metric', 'RepresentNav', 'Best reported baseline', 'Baseline value']) {
      const th = make('th', '', label);
      th.scope = 'col';
      labels.append(th);
    }
    head.append(labels);
    table.append(head);
    const body = make('tbody');
    let previousSetting;
    for (const row of item.rows) {
      const outcome = compareResult(row.representNav, row.baseline.value, row.direction);
      const tr = make('tr', `benchmark-row is-${outcome}`);
      tr.dataset.metric = row.id;
      if (row.setting !== previousSetting) tr.classList.add('is-group-start');
      previousSetting = row.setting;
      const setting = make('th', 'benchmark-row-setting', row.setting);
      setting.scope = 'row';
      tr.append(setting);
      const arrow = row.direction === 'down' ? '↓' : '↑';
      tr.append(make('td', 'benchmark-row-metric', `${row.metric} ${arrow}${row.unit === 'm' ? ' (m)' : ''}`));
      const result = make('td', `benchmark-result is-${outcome}`);
      const score = make(outcome === 'win' || outcome === 'tie' ? 'strong' : 'span',
        'benchmark-result-value', formatScore(row.representNav, row.precision));
      result.append(score);
      const delta = make('span', `benchmark-row-delta is-${outcome}`,
        `${outcome === 'tie' ? 'Tie · ' : ''}${formatDelta(row.representNav, row.baseline.value, row)}`);
      const description = {
        win: 'Better than baseline', tie: 'Tied with baseline',
        loss: 'Below baseline performance', unreported: 'Unreported comparison',
      }[outcome];
      delta.setAttribute('aria-label', `${description}: ${delta.textContent}`);
      result.append(delta);
      tr.append(result);
      tr.append(make('td', 'benchmark-baseline-name', row.baseline.name || '—'));
      const baseline = make('td', 'benchmark-baseline-value');
      baseline.append(make(outcome === 'loss' ? 'strong' : 'span', '',
        formatScore(row.baseline.value, row.precision)));
      tr.append(baseline);
      body.append(tr);
    }
    table.append(body);
    wrap.append(table);
    const notes = make('ul', 'benchmark-notes');
    item.notes.forEach(note => notes.append(make('li', '', note)));
    panel.replaceChildren(heading, wrap, notes);
  }

  listen(tabs, 'keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const ids = [...buttons.keys()];
    const current = ids.indexOf(selectedId);
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? ids.length - 1 :
      (current + (event.key === 'ArrowRight' ? 1 : -1) + ids.length) % ids.length;
    select(ids[index]);
    buttons.get(ids[index]).focus();
  });

  section.append(tabs, panel, make('p', 'benchmark-metric-note', data.metricNote));
  const source = make('p', 'benchmark-source');
  const link = make(data.source.url ? 'a' : 'span', 'benchmark-source-link', data.source.label);
  if (data.source.url) {
    link.href = data.source.url;
    link.target = '_blank';
    link.rel = 'noopener';
  }
  source.append(link, document.createTextNode(`. ${data.source.note}`));
  section.append(source);
  const anchor = [...container.children].find(child => child.classList.contains('results-layout'));
  container.insertBefore(section, anchor || null);
  select(buttons.has(initialBenchmark) ? initialBenchmark : data.benchmarks[0].id);
  return {
    select,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      listeners.forEach(remove => remove());
      section.remove();
    },
  };
}
