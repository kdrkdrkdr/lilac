console.log('[renderer] script loading...');

const $ = (id) => document.getElementById(id);
let pollTimer = null;

function log(msg) {
  console.log('[log]', msg);
  const el = $('log');
  if (el) el.textContent = msg;
}

/* ───────── titlebar ───────── */
$('win_min').onclick   = () => window.win.minimize();
$('win_close').onclick = () => window.win.close();

/* ───────── boot ───────── */
window.addEventListener('DOMContentLoaded', async () => {
  await refreshDevices();

  const targetPath = await window.lilac.defaultTarget();
  $('target').value = targetPath;
  if (targetPath) {
    log('loading engine…');
    const r = await window.lilac.init({ targetPath });
    if (r && r.ok) {
      log(`ready — sr=${r.sampleRate} hop=${r.hop}`);
      await refreshDevices();
      await syncSettings();
      $('start').disabled = false;
    } else {
      log(`init failed: ${r && r.error || 'unknown'}`);
    }
  }
});

window.addEventListener('error', (e) => {
  console.error('[renderer] window error:', e.message, e.filename, e.lineno);
});

/* ───────── devices ───────── */
async function refreshDevices() {
  try {
    const ds = await window.lilac.devices();
    const dIn = $('dev_in'), dOut = $('dev_out');
    dIn.innerHTML = ''; dOut.innerHTML = '';
    const dflt = (opt) => opt.is_default ? ' (default)' : '';
    let nIn = 0, nOut = 0;
    for (const d of ds) {
      const opt = document.createElement('option');
      opt.value = d.id;
      opt.textContent = d.name + dflt(d);
      if (d.is_input) { dIn.appendChild(opt);  nIn++; }
      else            { dOut.appendChild(opt); nOut++; }
      if (d.is_default) opt.selected = true;
    }
    log(`devices: ${nIn} input / ${nOut} output`);
  } catch (e) {
    console.error('[renderer] refreshDevices failed', e);
    log('devices: ERROR (see console)');
  }
}

/* ───────── settings ───────── */
async function syncSettings() {
  const s = await window.lilac.getSettings();
  if (!s) return;
  const db = Math.round(s.agcTargetDb);
  $('agc_db').value           = db;
  $('agc_db_val').textContent = db;
}

$('agc_db').oninput = (e) => {
  $('agc_db_val').textContent = e.target.value;
};
$('agc_db').onchange = async (e) => {
  await window.lilac.setAgcTargetDb({ db: parseFloat(e.target.value) });
};

/* ───────── target picker ───────── */
$('pick_t').onclick = async () => {
  const p = await window.lilac.pickFile({ filters: [{ name: 'WAV', extensions: ['wav'] }] });
  if (!p) return;
  $('target').value = p;
  const r = await window.lilac.setTarget({ targetPath: p });
  log(r && r.ok ? 'target updated' : 'target update failed');
};

/* ───────── start / stop ───────── */
function resetMeters() {
  $('bar_in').style.width   = '0%';
  $('bar_out').style.width  = '0%';
  $('vad_bar').style.width  = '0%';
  $('vad_val').textContent  = '0.00';
  $('stat_line').textContent = '—';
}

$('start').onclick = async () => {
  const inId  = parseInt($('dev_in').value, 10);
  const outId = parseInt($('dev_out').value, 10);
  const r = await window.lilac.start({ inputId: inId, outputId: outId });
  if (!r.ok) { log(`start: ${r.error}`); return; }
  $('start').disabled = true; $('stop').disabled = false;
  log('running');
  pollTimer = setInterval(updateStats, 80);
};

$('stop').onclick = async () => {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  await window.lilac.stop();
  $('start').disabled = false; $('stop').disabled = true;
  resetMeters();
  log('stopped');
};

/* ───────── stats ───────── */
let statTick = 0;
async function updateStats() {
  const s = await window.lilac.stats();
  if (!s) return;
  const pct = (v) => Math.min(100, Math.round(v * 400)) + '%';
  $('bar_in').style.width  = pct(s.input_rms);
  $('bar_out').style.width = pct(s.output_rms);
  const vad = s.last_vad ?? 0;
  $('vad_bar').style.width = Math.round(vad * 100) + '%';
  $('vad_val').textContent = vad.toFixed(2);
  $('stat_line').textContent =
    `recent=${s.recent_proc_ms.toFixed(1)}ms · avg=${s.avg_process_ms.toFixed(1)}ms · drops=${s.dropped_frames}`;
  if ((statTick++ % 20) === 0) {
    console.log(`[stats] in=${s.input_rms.toFixed(4)} out=${s.output_rms.toFixed(4)} vad=${vad.toFixed(2)} recent=${s.recent_proc_ms.toFixed(1)}ms`);
  }
}

/* ───────── github link ───────── */
$('repo_link').onclick = (e) => { e.preventDefault(); window.win.openGitHub(); };
