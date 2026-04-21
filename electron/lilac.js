// Koffi FFI bindings to libvc.dll
const koffi = require('koffi');
const path  = require('path');
const fs    = require('fs');

// Resolve DLL. When packaged, it sits next to the .exe (onefile layout).
// In dev, it's in ../cpp/ (built by mingw32-make).
function findDll(name) {
  const exeDir = path.dirname(process.execPath);
  const candidates = [
    path.join(exeDir, name),
    path.join(__dirname, '..', 'cpp', name),
    path.join(__dirname, name),
  ];
  for (const p of candidates) { if (fs.existsSync(p)) return p; }
  throw new Error(`cannot find ${name}`);
}

// libvc.dll has OpenBLAS + gfortran + rnnoise statically linked, so there
// are no sibling DLLs to pre-load.
const lib = koffi.load(findDll('libvc.dll'));

// Structs — must match vc.h exactly.
const LilacDevice = koffi.struct('LilacDevice', {
  id:         'int',
  is_input:   'int',
  is_default: 'int',
  name:       koffi.array('char', 256),
});

const LilacStats = koffi.struct('LilacStats', {
  input_rms:      'float',
  output_rms:     'float',
  dropped_frames: 'int',
  avg_process_ms: 'float',
  recent_proc_ms: 'float',
  is_running:     'int',
  last_vad:       'float',
});

// C-style function declarations.
const api = {
  create:       lib.func('void *lilac_create(const char *, float *, int, int)'),
  destroy:      lib.func('void lilac_destroy(void *)'),
  list_devices: lib.func('int lilac_list_devices(void *, _Out_ LilacDevice *, int)'),
  enum_devices: lib.func('int lilac_enum_devices(_Out_ LilacDevice *, int)'),
  start:        lib.func('int lilac_start(void *, int, int)'),
  stop:         lib.func('void lilac_stop(void *)'),
  sample_rate:  lib.func('int lilac_sample_rate(void *)'),
  hop_samples:  lib.func('int lilac_hop_samples(void *)'),
  set_target:   lib.func('int lilac_set_target(void *, float *, int)'),
  reset_source: lib.func('void lilac_reset_source(void *)'),
  get_stats:    lib.func('void lilac_get_stats(void *, _Out_ LilacStats *)'),
  set_agc_target_db: lib.func('void lilac_set_agc_target_db(void *, float)'),
  get_agc_target_db: lib.func('float lilac_get_agc_target_db(void *)'),
};

function readCName(charArr) {
  // charArr is a plain JS array of char codes (koffi-decoded).
  const bytes = Buffer.from(charArr);
  const nul = bytes.indexOf(0);
  return bytes.slice(0, nul === -1 ? bytes.length : nul).toString('utf8');
}

function enumDevicesStandalone() {
  const buf = Array.from({ length: 64 }, () => ({
    id: 0, is_input: 0, is_default: 0, name: new Array(256).fill(0),
  }));
  const n = api.enum_devices(buf, 64);
  return buf.slice(0, n).map(d => ({
    id: d.id, is_input: !!d.is_input, is_default: !!d.is_default,
    name: readCName(d.name),
  }));
}

class Lilac {
  constructor(weightsPath, targetWav, K = 3) {
    this.h = api.create(weightsPath, targetWav, targetWav.length, K);
    if (!this.h) throw new Error('lilac_create failed');
  }
  listDevices() {
    // Allocate 64-slot array; koffi fills in with _Out_ parameter.
    const buf = Array.from({ length: 64 }, () => ({
      id: 0, is_input: 0, is_default: 0, name: new Array(256).fill(0),
    }));
    const n = api.list_devices(this.h, buf, 64);
    return buf.slice(0, n).map(d => ({
      id:         d.id,
      is_input:   !!d.is_input,
      is_default: !!d.is_default,
      name:       readCName(d.name),
    }));
  }
  start(inputId = -1, outputId = -1) { return api.start(this.h, inputId, outputId); }
  stop()                             { api.stop(this.h); }
  sampleRate()                       { return api.sample_rate(this.h); }
  hopSamples()                       { return api.hop_samples(this.h); }
  setTarget(wav)                     { return api.set_target(this.h, wav, wav.length); }
  resetSource()                      { api.reset_source(this.h); }
  setAgcTargetDb(db)                 { api.set_agc_target_db(this.h, db); }
  getAgcTargetDb()                   { return api.get_agc_target_db(this.h); }
  getStats() {
    const s = {
      input_rms: 0, output_rms: 0, dropped_frames: 0,
      avg_process_ms: 0, recent_proc_ms: 0, is_running: 0, last_vad: 0,
    };
    api.get_stats(this.h, s);
    return { ...s, is_running: !!s.is_running };
  }
  destroy() { if (this.h) { api.destroy(this.h); this.h = null; } }
}

module.exports = { Lilac, enumDevices: enumDevicesStandalone };
