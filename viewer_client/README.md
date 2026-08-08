# Bundled Viser client

`index.html` is a production build of the Viser 1.0.30 web client with the
4C4D viewer's camera controls. It is served by `viewer_4c4d.py` so users do not
need Node.js or a patched Python environment.

The customized controls include:

- 0.4x default look sensitivity with a persistent slider;
- independent, persistent X/Y orbit inversion;
- independent, persistent vertical RMB-move inversion;
- Space playback and the viewer's keyboard direction mappings.

Viser is distributed under the Apache License 2.0; see `LICENSE.viser`.
