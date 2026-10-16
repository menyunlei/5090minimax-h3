import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const WD = process.cwd();
const LOG = path.join(WD, 'drive_log.ndjson');
const PIDF = path.join(WD, 'chrome.pid');
const PORT = 9223;
const BASE = 'http://127.0.0.1:8787';
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const T_START = Date.now();
const RIDE_CAP_MS = 240_000;
const TOTAL_CAP_MS = 700_000;

const log = (o) => {
  try { fs.appendFileSync(LOG, JSON.stringify({ t: new Date().toISOString(), ...o }) + '\n'); } catch (e) {}
};
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

const WPTS = [
  [700, 600], [690, 500], [685, 460], [676, 414],
  [630, 414], [580, 414], [530, 414], [480, 414], [450, 414], [420, 414], [395, 414], [380, 414],
];

async function main() {
  let baseState = null;
  try { baseState = await (await fetch(BASE + '/state')).json(); } catch (e) {}
  log({ ev: 'start', base: baseState && { sim_pos: baseState.sim_pos, sim_heading: baseState.sim_heading, queue_len: baseState.queue_len, busy: baseState.busy, pf: baseState.pf } });

  const prof = path.join(WD, '.chrome-prof');
  const chrome = spawn(CHROME, [
    '--headless=new', '--disable-gpu', '--mute-audio',
    '--remote-debugging-port=' + PORT, '--remote-allow-origins=*',
    '--no-first-run', '--no-default-browser-check',
    '--user-data-dir=' + prof, 'about:blank',
  ], { stdio: 'ignore' });
  chrome.unref();
  fs.writeFileSync(PIDF, String(chrome.pid));
  const killChrome = () => { try { spawn('taskkill', ['/PID', String(chrome.pid), '/T', '/F'], { stdio: 'ignore' }).unref(); } catch (e) {} };
  process.on('exit', killChrome);
  process.on('SIGINT', () => process.exit(2));
  process.on('SIGTERM', () => process.exit(3));

  let cdpUp = false;
  for (let i = 0; i < 60; i++) {
    try { const v = await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json(); if (v) { cdpUp = true; break; } } catch (e) {}
    await sleep(500);
  }
  if (!cdpUp) throw new Error('chrome CDP port never came up');

  let page = null;
  for (let i = 0; i < 10 && !page; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json();
      page = list.find(t => t.type === 'page');
    } catch (e) {}
    if (!page) await sleep(500);
  }
  if (!page || !page.webSocketDebuggerUrl) throw new Error('no cdp page target');
  log({ ev: 'tab', url: page.url, id: page.id });

  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((res, rej) => {
    ws.addEventListener('open', res, { once: true });
    ws.addEventListener('error', () => rej(new Error('ws open error')), { once: true });
  });
  let idc = 0; const pend = new Map();
  ws.addEventListener('message', ev => {
    try {
      const m = JSON.parse(ev.data);
      if (m.id && pend.has(m.id)) { const p = pend.get(m.id); pend.delete(m.id); p(m); }
    } catch (e) {}
  });
  const cmd = (method, params = {}) => new Promise((res, rej) => {
    const id = ++idc; pend.set(id, res);
    ws.send(JSON.stringify({ id, method, params }));
    setTimeout(() => { if (pend.has(id)) { pend.delete(id); rej(new Error('timeout ' + method)); } }, 8000);
  });
  const evl = async (expression) => {
    const r = await cmd('Runtime.evaluate', { expression, returnByValue: true });
    if (r.error) throw new Error(r.error.description || JSON.stringify(r.error));
    return r.result && r.result.result ? r.result.result.value : undefined;
  };
  await cmd('Runtime.enable');
  await cmd('Page.enable');
  await cmd('Page.navigate', { url: BASE + '/drive' });

  let booted = false;
  for (let i = 0; i < 60; i++) {
    try {
      const v = await evl("(document.readyState==='complete'||document.readyState==='interactive')&&(typeof tick==='function')&&(typeof px==='number')?1:0");
      if (v === 1) { booted = true; break; }
    } catch (e) {}
    await sleep(500);
  }
  if (!booted) throw new Error('page boot not detected');
  const boot = JSON.parse(await evl("JSON.stringify({px,py,hd,spd})"));
  log({ ev: 'booted', boot });

  const stKeys = { w: false, a: false, s: false, d: false };
  const setKey = async (k, v) => {
    if (stKeys[k] === v) return;
    stKeys[k] = v;
    try { await evl(`window.dispatchEvent(new KeyboardEvent('key${v ? 'down' : 'up'}',{key:'${k}'}))`); }
    catch (e) { log({ ev: 'key_err', k, err: String(e) }); }
  };

  let wi = 0, done = false, lastLogT = 0, lastCd = null;
  const rel = (want, hd) => ((want - hd + 540) % 360) - 180;
  const tRide0 = Date.now();
  while (!done && Date.now() - tRide0 < RIDE_CAP_MS) {
    let s = null;
    try { s = await evl("(typeof px==='number')?JSON.stringify({px,py,hd,spd,cd:committedDist,tl:traj.length,ks:Object.entries(keys).filter(([k,v])=>v&&k!=='autopilot').map(([k])=>k).join('')}):''"); } catch (e) {}
    if (!s) { await sleep(80); continue; }
    const p = JSON.parse(s);
    if (lastCd !== null && p.cd < lastCd - 50) {
      log({ ev: 'commit_local', cd_before: lastCd, at: [Math.round(p.px), Math.round(p.py)], hd: Math.round(p.hd) });
    }
    lastCd = p.cd;
    if (Date.now() - lastLogT > 500) {
      lastLogT = Date.now();
      log({ ev: 'ride', x: Math.round(p.px), y: Math.round(p.py), hd: Math.round(p.hd), spd: +p.spd.toFixed(2), cd: p.cd, wi, ks: p.ks });
    }
    while (wi < WPTS.length && Math.hypot(WPTS[wi][0] - p.px, WPTS[wi][1] - p.py) < 26) wi++;
    if (wi >= WPTS.length) {
      await setKey('w', false); await setKey('a', false); await setKey('d', false); await setKey('s', true);
      if (p.spd < 0.06) done = true;
      await sleep(60); continue;
    }
    let ti = wi;
    if (ti < WPTS.length - 1 && Math.hypot(WPTS[ti][0] - p.px, WPTS[ti][1] - p.py) < 45) ti++;
    const [wx, wy] = WPTS[ti];
    const want = (Math.atan2(wy - p.py, wx - p.px) * 180 / Math.PI + 360) % 360;
    const r = rel(want, p.hd);
    if (r > 7) { await setKey('d', true); await setKey('a', false); }
    else if (r < -7) { await setKey('a', true); await setKey('d', false); }
    else if (Math.abs(r) < 4) { await setKey('a', false); await setKey('d', false); }
    if (Math.abs(r) > 150 && p.spd > 2.2) {
      await setKey('w', false); await setKey('s', true);
    } else {
      await setKey('w', true); await setKey('s', false);
    }
    await sleep(50);
  }
  for (const k of ['w', 'a', 's', 'd']) await setKey(k, false);
  let endP = null;
  try { endP = JSON.parse(await evl("JSON.stringify({px,py,hd,spd})")); } catch (e) {}
  log({ ev: 'ride_end', done, endP, ms: Date.now() - tRide0 });

  let lastSim = null, lastQ = null, lastPfStatus = null, mTick = 0;
  while (Date.now() - T_START < TOTAL_CAP_MS) {
    try {
      const s = await (await fetch(BASE + '/state')).json();
      const sim = s.sim_pos;
      if (lastSim && (Math.abs(sim[0] - lastSim[0]) > 3 || Math.abs(sim[1] - lastSim[1]) > 3)) {
        log({ ev: 'leg_simpos', from: lastSim, to: sim, queue_len: s.queue_len, msg: String(s.last_msg).slice(0, 200) });
      }
      if (lastQ !== null && s.queue_len !== lastQ) {
        log({ ev: 'queue', from: lastQ, to: s.queue_len, msg: String(s.last_msg).slice(0, 200) });
      }
      if (s.pf && s.pf.status !== lastPfStatus) {
        log({ ev: 'pf', status: s.pf.status, msg: s.pf.msg, hits: s.pf.hits, misses: s.pf.misses, expired: s.pf.expired });
        lastPfStatus = s.pf.status;
      }
      lastSim = sim; lastQ = s.queue_len;
      log({ ev: 'state', sim, shd: Math.round(s.sim_heading), q: s.queue_len, busy: s.busy, gen: s.gen_pos, gen_no: s.gen_no, msg: String(s.last_msg).slice(0, 160), pf: s.pf });
    } catch (e) { log({ ev: 'state_err', err: String(e) }); }
    const mt = (Date.now() / 15000) | 0;
    if (mt !== mTick) {
      mTick = mt;
      try {
        const m = await (await fetch(BASE + '/mission.json')).json();
        log({ ev: 'mission', n: m.evals.length, tail: m.evals.slice(-3), queue_len: m.queue_len, gen_pos: m.gen_pos, busy: m.busy });
      } catch (e) { log({ ev: 'mission_err', err: String(e) }); }
    }
    await sleep(2000);
  }
  log({ ev: 'monitor_end' });
  try { ws.close(); } catch (e) {}
  killChrome();
}

main()
  .then(() => { log({ ev: 'exit' }); process.exit(0); })
  .catch(e => { log({ ev: 'fatal', err: String(e && e.stack ? e.stack : e) }); process.exit(1); });
