/*
 * Screenshot harness for langgraph-lab.
 *
 * Drives the real app in a real browser and writes PNGs to screenshots/. No mockups: if a
 * screenshot shows a number, a model produced it on this machine.
 *
 *   node scripts/shoot.mjs --project p01 --url http://127.0.0.1:8111
 *
 * Each shot is taken in both colour schemes.
 */

import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, cur, i, arr) => {
    if (cur.startsWith('--')) acc.push([cur.slice(2), arr[i + 1]]);
    return acc;
  }, []),
);

const PROJECT = args.project ?? 'p01';
const OUT = args.out ?? 'screenshots';
const WIDTH = Number(args.width ?? 1440);
const DEFAULT_URL = {
  p01: 'http://127.0.0.1:8111',
  p02: 'http://127.0.0.1:8112',
  p03: 'http://127.0.0.1:8113',
  p04: 'http://127.0.0.1:8114',
  p05: 'http://127.0.0.1:8115',
};
const URL = args.url ?? DEFAULT_URL[PROJECT];
const RUN_TIMEOUT = 20 * 60 * 1000;

await mkdir(OUT, { recursive: true });

async function settle(page) {
  await page.waitForLoadState('networkidle').catch(() => {});
  await page.evaluate(() => document.fonts?.ready).catch(() => {});
  await page.waitForTimeout(250);
}

async function submit(page, resultSelector) {
  await Promise.all([
    page.waitForURL('**/run', { timeout: RUN_TIMEOUT }),
    page.click('button[type=submit]'),
  ]);
  await page.waitForSelector(resultSelector, { timeout: RUN_TIMEOUT }).catch(() => {});
}

const SCENARIOS = {
  p01: [
    ['1-form', null],
    [
      // The plateau: fixed_6 shown against fixed_3, the same shape as RESULTS.md.
      '2-score-trace',
      async (page) => {
        await page.selectOption('#task_id', 't1');
        await page.selectOption('#strategy', '__all__');
        await submit(page, '.trace');
      },
    ],
    [
      // t3: stuck at 0.00 for the whole loop, with the vague final draft visible.
      '3-stuck-at-zero',
      async (page) => {
        await page.selectOption('#task_id', 't3');
        await page.selectOption('#strategy', 'fixed_6');
        await submit(page, '.trace');
      },
    ],
  ],
  p02: [
    ['1-form', null],
    [
      // x2: HIGH confidence, wrong category. The headline finding on one screen.
      '2-confident-misroute',
      async (page) => {
        await page.selectOption('#ticket_id', 'x2');
        await page.selectOption('#strategy', '__all__');
        await submit(page, '.confband');
      },
    ],
    [
      // c2: a clean, unambiguous ticket the gate still escalates on MEDIUM confidence.
      '3-clean-ticket-under-confident',
      async (page) => {
        await page.selectOption('#ticket_id', 'c2');
        await page.selectOption('#strategy', '__all__');
        await submit(page, '.confband');
      },
    ],
  ],
  p03: [
    ['1-form', null],
    [
      // The headline: a failed branch, and a report that reads as complete regardless.
      '2-silent-failure',
      async (page) => {
        await page.selectOption('#brief_id', 'b1');
        await page.selectOption('#fail_aspect', 'financial');
        await page.selectOption('#flag_gaps', '');
        await submit(page, '.report');
      },
    ],
    [
      // The same failure, this time the synthesis is told to check and catches it.
      '3-flagged-catches-it',
      async (page) => {
        await page.selectOption('#brief_id', 'b2');
        await page.selectOption('#fail_aspect', 'technical');
        await page.selectOption('#flag_gaps', 'true');
        await submit(page, '.report');
      },
    ],
  ],
  p04: [
    ['1-form', null],
    [
      // The whole finding: naive_requeue's wasted final redraft, marked, beside checkpointed's
      // exact call count.
      '2-wasted-redraft',
      async (page) => {
        await page.selectOption('#scenario_id', 's3');
        await submit(page, '.steps');
      },
    ],
  ],
  p05: [
    ['1-form', null],
    [
      // Vegan case: full context safe, last-message-only recommending eggs.
      '2-handoff-comparison',
      async (page) => {
        await page.selectOption('#case_id', 'a4');
        await page.selectOption('#strategy', '__all__');
        await submit(page, '.resp');
      },
    ],
    [
      // The accessibility case: same pattern, different constraint.
      '3-accessibility-dropped',
      async (page) => {
        await page.selectOption('#case_id', 'a5');
        await page.selectOption('#strategy', '__all__');
        await submit(page, '.resp');
      },
    ],
  ],
};

const browser = await chromium.launch({ channel: 'chromium' });

try {
  const scenarios = SCENARIOS[PROJECT] ?? [];
  for (const scheme of ['light', 'dark']) {
    console.log(`${scheme}:`);
    for (const [name, steps] of scenarios) {
      const context = await browser.newContext({
        colorScheme: scheme,
        viewport: { width: WIDTH, height: 700 },
        deviceScaleFactor: 2,
      });
      const page = await context.newPage();
      page.on('pageerror', (e) => console.error(`  ! page error: ${e.message}`));
      await page.goto(URL, { waitUntil: 'domcontentloaded' });
      if (steps) await steps(page);
      await settle(page);
      const file = path.join(OUT, `${PROJECT}-${name}-${scheme}.png`);
      await page.screenshot({ path: file, fullPage: true });
      console.log(`  wrote ${file}`);
      await context.close();
    }
  }
} finally {
  await browser.close();
}

console.log('done');
