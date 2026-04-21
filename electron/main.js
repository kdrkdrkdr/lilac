const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs   = require('fs');
const { Lilac, enumDevices } = require('./lilac');

/* Portable layout: weights.bin and libvc.dll sit next to the .exe.
   In dev, weights are at cpp/weights.bin relative to the repo root. */
const EXE_DIR       = path.dirname(app.getPath('exe'));
const WEIGHTS_PATH  = app.isPackaged
  ? path.join(EXE_DIR, 'weights.bin')
  : path.join(__dirname, '..', 'cpp', 'weights.bin');
const DEFAULT_TARGET = app.isPackaged
  ? path.join(EXE_DIR, 'tsu_10s.wav')
  : path.join(__dirname, '..', 'samples', 'tsu_10s.wav');
const FIXED_K = 3;
const GITHUB_URL = 'https://github.com/kdrkdrkdr/lilac';

let win   = null;
let lilac = null;

/* Mono WAV reader (16-bit PCM or 32-bit float). Returns Float32Array at the
   file's own sample rate. Multi-channel input is averaged to mono. */
function readWavMono(filePath) {
  const buf = fs.readFileSync(filePath);
  if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE')
    throw new Error('not a WAV');
  let off = 12, fmt = null, dataOff = 0, dataLen = 0;
  while (off < buf.length) {
    const id = buf.toString('ascii', off, off + 4);
    const sz = buf.readUInt32LE(off + 4);
    if      (id === 'fmt ') { fmt = { format: buf.readUInt16LE(off+8), ch: buf.readUInt16LE(off+10), sr: buf.readUInt32LE(off+12), bits: buf.readUInt16LE(off+22) }; }
    else if (id === 'data') { dataOff = off + 8; dataLen = sz; break; }
    off += 8 + sz;
  }
  if (!fmt)   throw new Error('no fmt chunk');

  const frames = dataLen / (fmt.ch * fmt.bits / 8);
  const raw = new Float32Array(frames);
  if (fmt.format === 1 && fmt.bits === 16) {
    for (let i = 0; i < frames; i++) {
      let sum = 0;
      for (let c = 0; c < fmt.ch; c++) sum += buf.readInt16LE(dataOff + (i*fmt.ch + c)*2);
      raw[i] = (sum / fmt.ch) / 32768.0;
    }
  } else if (fmt.format === 3 && fmt.bits === 32) {
    for (let i = 0; i < frames; i++) {
      let sum = 0;
      for (let c = 0; c < fmt.ch; c++) sum += buf.readFloatLE(dataOff + (i*fmt.ch + c)*4);
      raw[i] = sum / fmt.ch;
    }
  } else {
    throw new Error(`unsupported WAV format ${fmt.format} bits=${fmt.bits}`);
  }
  return { samples: raw, sampleRate: fmt.sr };
}

/* Cheap linear-interp resampler for target-audio SE extraction. Target is
   only 2–10 s once per file and SE is a global low-dim feature, so the
   minor aliasing from skipping a lowpass is fine here. */
function resampleLinear(src, fromRate, toRate) {
  if (fromRate === toRate) return src;
  const ratio = fromRate / toRate;
  const outLen = Math.max(0, Math.floor(src.length / ratio));
  const out = new Float32Array(outLen);
  const nMinus1 = src.length - 1;
  for (let i = 0; i < outLen; i++) {
    const x = i * ratio;
    const i0 = Math.floor(x);
    const i1 = i0 + 1 > nMinus1 ? nMinus1 : i0 + 1;
    const frac = x - i0;
    out[i] = src[i0] * (1 - frac) + src[i1] * frac;
  }
  return out;
}

/* Load a WAV file at any rate and return it resampled to 22050 Hz mono. */
function loadTargetWav22050(filePath) {
  const { samples, sampleRate } = readWavMono(filePath);
  if (sampleRate === 22050) return samples;
  console.log(`[wav] resampling ${sampleRate} → 22050 (${samples.length} samples)`);
  return resampleLinear(samples, sampleRate, 22050);
}

function createWindow() {
  win = new BrowserWindow({
    width: 460, height: 600,
    resizable: false,
    maximizable: false,
    fullscreenable: false,
    frame: false,
    backgroundColor: '#1e1b24',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      devTools: false,
    },
  });
  win.setMenu(null);
  win.loadFile('renderer.html');

  /* Block F12 / Ctrl+Shift+I / Ctrl+R reload shortcuts. */
  win.webContents.on('before-input-event', (event, input) => {
    const k = (input.key || '').toLowerCase();
    if (k === 'f12') return event.preventDefault();
    if (input.control && input.shift && (k === 'i' || k === 'j' || k === 'c'))
      return event.preventDefault();
    if (input.control && (k === 'r' || k === 'u')) return event.preventDefault();
  });
}

// Surface uncaught native/JS errors rather than silently exiting.
process.on('uncaughtException', (err) => {
  console.error('[uncaughtException]', err);
  if (win) win.webContents.send('fatal', String(err && err.stack || err));
});
process.on('unhandledRejection', (err) => {
  console.error('[unhandledRejection]', err);
  if (win) win.webContents.send('fatal', String(err));
});

function wrap(name, fn) {
  return async (...args) => {
    try { return await fn(...args); }
    catch (e) {
      console.error(`[ipc:${name}]`, e);
      return { ok: false, error: String(e && e.message || e) };
    }
  };
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => {
  if (lilac) { try { lilac.destroy(); } catch {} lilac = null; }
  app.quit();
});

// ---- IPC handlers ----
ipcMain.handle('init', wrap('init', async (_evt, { targetPath }) => {
  if (lilac) { lilac.destroy(); lilac = null; }
  console.log('[init] loading target:', targetPath, 'weights:', WEIGHTS_PATH);
  const target = loadTargetWav22050(targetPath);
  lilac = new Lilac(WEIGHTS_PATH, target, FIXED_K);
  console.log('[init] engine ready');
  return { ok: true, sampleRate: lilac.sampleRate(), hop: lilac.hopSamples() };
}));

// ---- Window controls (custom title bar) ----
ipcMain.on('win:minimize', () => { if (win) win.minimize(); });
ipcMain.on('win:close',    () => { if (win) win.close(); });

ipcMain.handle('default_target', () => DEFAULT_TARGET);

ipcMain.handle('devices', wrap('devices', async () => {
  const ds = lilac ? lilac.listDevices() : enumDevices();
  console.log('[devices]', ds.length, 'found');
  return ds;
}));

ipcMain.handle('start', wrap('start', async (_evt, { inputId, outputId }) => {
  if (!lilac) return { ok: false, error: 'not initialized' };
  const rc = lilac.start(inputId, outputId);
  return rc === 0 ? { ok: true } : { ok: false, error: `start rc=${rc}` };
}));

ipcMain.handle('stop', wrap('stop', async () => {
  if (lilac) lilac.stop();
  return { ok: true };
}));

ipcMain.handle('set_target', wrap('set_target', async (_evt, { targetPath }) => {
  if (!lilac) return { ok: false };
  const target = loadTargetWav22050(targetPath);
  const rc = lilac.setTarget(target);
  lilac.resetSource();
  return { ok: rc === 0 };
}));

ipcMain.handle('stats', wrap('stats', async () => lilac ? lilac.getStats() : null));

ipcMain.handle('get_settings', wrap('get_settings', async () => {
  if (!lilac) return null;
  return { agcTargetDb: lilac.getAgcTargetDb() };
}));

ipcMain.handle('set_agc_target_db', wrap('set_agc_target_db', async (_e, { db }) => {
  if (!lilac) return { ok: false };
  lilac.setAgcTargetDb(db);
  return { ok: true };
}));

ipcMain.on('open_github', () => { shell.openExternal(GITHUB_URL); });

ipcMain.handle('pick_file', wrap('pick_file', async (_evt, { filters }) => {
  const res = await dialog.showOpenDialog(win, { properties: ['openFile'], filters });
  return res.canceled ? null : res.filePaths[0];
}));
