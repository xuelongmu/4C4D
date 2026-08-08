# Bundled Viser client

`index.html` is a production build of the Viser 1.0.30 web client with the
4C4D viewer's camera controls. It is served by `viewer_4c4d.py` so users do not
need Node.js or a patched Python environment.

The customized controls include:

- 0.4x default look sensitivity with a persistent slider;
- 1.0x default move sensitivity with a persistent slider;
- independent, persistent X/Y orbit inversion;
- independent, persistent X/Y move inversion;
- Space playback and the viewer's keyboard direction mappings;
- a cinematic sequencer with shot scrubbing, transport controls, key
  navigation, camera/lens tracks, an unlocked-by-default camera-follow toggle,
  and shot-synchronized dynamic playback;
- a persistent light/dark UI theme toggle.
- a wider, collapsible navigation-control palette that stays clear of the
  cinematic sequencer.

Viser is distributed under the Apache License 2.0; see `LICENSE.viser`.
