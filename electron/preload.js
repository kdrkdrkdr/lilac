const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('lilac', {
  init:            (args) => ipcRenderer.invoke('init', args),
  defaultTarget:   ()     => ipcRenderer.invoke('default_target'),
  devices:         ()     => ipcRenderer.invoke('devices'),
  start:           (args) => ipcRenderer.invoke('start', args),
  stop:            ()     => ipcRenderer.invoke('stop'),
  setTarget:       (args) => ipcRenderer.invoke('set_target', args),
  stats:           ()     => ipcRenderer.invoke('stats'),
  pickFile:        (args) => ipcRenderer.invoke('pick_file', args),
  getSettings:     ()     => ipcRenderer.invoke('get_settings'),
  setAgcTargetDb:  (args) => ipcRenderer.invoke('set_agc_target_db', args),
});

contextBridge.exposeInMainWorld('win', {
  minimize:  () => ipcRenderer.send('win:minimize'),
  close:     () => ipcRenderer.send('win:close'),
  openGitHub:() => ipcRenderer.send('open_github'),
});
