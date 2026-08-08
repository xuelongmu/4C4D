(() => {
  "use strict";

  const normalize = (value) => String(value || "").replace(/\s+/g, " ").trim();

  const icons = {
    start: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 4v12M15 4.5 7.5 10l7.5 5.5Z"/></svg>',
    previous: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m10.5 4.5-7 5.5 7 5.5Z"/><path d="m15.5 6.5 3.5 3.5-3.5 3.5L12 10Z"/></svg>',
    play: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m6 4.5 9 5.5-9 5.5Z"/></svg>',
    pause: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7 5v10M13 5v10"/></svg>',
    lock: '<svg viewBox="0 0 20 20" aria-hidden="true"><rect x="4.5" y="9" width="11" height="8" rx="1.5"/><path d="M7 9V6.5a3 3 0 0 1 6 0V9"/></svg>',
    unlock: '<svg viewBox="0 0 20 20" aria-hidden="true"><rect x="4.5" y="9" width="11" height="8" rx="1.5"/><path d="M13 9V6.5a3 3 0 0 0-5.7-1.3"/></svg>',
    next: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m9.5 4.5 7 5.5-7 5.5Z"/><path d="m4.5 6.5 3.5 3.5-3.5 3.5L1 10Z"/></svg>',
    end: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M16 4v12M5 4.5l7.5 5.5L5 15.5Z"/></svg>',
    addKey: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m6 5 5 5-5 5-5-5Z"/><path d="M15 6v8M11 10h8"/></svg>',
    deleteKey: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m14 5 5 5-5 5-5-5Z"/><path d="M1 10h6"/></svg>',
    sun: '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="3.25"/><path d="M10 1.5v2M10 16.5v2M1.5 10h2M16.5 10h2M4 4l1.4 1.4M14.6 14.6 16 16M16 4l-1.4 1.4M5.4 14.6 4 16"/></svg>',
    moon: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M16.5 12.3A7 7 0 0 1 7.7 3.5a7 7 0 1 0 8.8 8.8Z"/></svg>',
    collapse: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m5 7.5 5 5 5-5"/></svg>',
    expand: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m5 12.5 5-5 5 5"/></svg>',
  };

  function setIcon(button, name) {
    if (button) button.innerHTML = icons[name];
  }

  function labelElement(label) {
    return [...document.querySelectorAll("p, label")].find((element) => normalize(element.textContent) === label) || null;
  }

  function controlsForLabel(label) {
    const labelNode = labelElement(label);
    if (!labelNode) return [];
    let node = labelNode.parentElement;
    for (let depth = 0; node && depth < 6; depth += 1, node = node.parentElement) {
      const inputs = [...node.querySelectorAll("input, textarea")].filter((input) => input.type !== "hidden");
      if (inputs.length) return inputs;
    }
    return [];
  }

  function controlForLabel(label, preferredType) {
    const inputs = controlsForLabel(label);
    return (preferredType ? inputs.find((input) => input.type === preferredType) : inputs[0]) || null;
  }

  function buttonWithText(label) {
    return [...document.querySelectorAll("button")].find((button) => normalize(button.textContent) === label) || null;
  }

  function setNativeValue(input, value) {
    if (!input) return;
    const prototype = input instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
    if (setter) setter.call(input, String(value));
    else input.value = String(value);
    input.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: String(value) }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    input.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, key: "Enter" }));
    input.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true, key: "Enter" }));
    input.blur();
  }

  function numericValue(label, fallback = 0) {
    const parsed = Number(controlForLabel(label)?.value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function numericVectorValue(label, index, fallback = 0) {
    const parsed = Number(controlsForLabel(label)[index]?.value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function timecode(frame, fps) {
    const safeFps = Math.max(1, Math.round(fps));
    const safeFrame = Math.max(0, Math.round(frame));
    const frames = safeFrame % safeFps;
    const totalSeconds = Math.floor(safeFrame / safeFps);
    const seconds = totalSeconds % 60;
    const minutes = Math.floor(totalSeconds / 60) % 60;
    const hours = Math.floor(totalSeconds / 3600);
    return [hours, minutes, seconds, frames].map((part) => String(part).padStart(2, "0")).join(":");
  }

  function applyTheme(theme) {
    const resolved = theme === "light" ? "light" : "dark";
    document.documentElement.dataset.c4dTheme = resolved;
    document.documentElement.setAttribute("data-mantine-color-scheme", resolved);
    localStorage.setItem("4c4d-ui-theme", resolved);
    const themeButton = document.getElementById("c4d-seq-theme");
    if (themeButton) {
      setIcon(themeButton, resolved === "dark" ? "sun" : "moon");
      themeButton.setAttribute("aria-label", resolved === "dark" ? "Use light UI" : "Use dark UI");
      themeButton.title = resolved === "dark" ? "Use light UI" : "Use dark UI";
    }
  }

  function hideSyncControl() {
    const label = labelElement("Sequencer key data");
    const input = controlForLabel("Sequencer key data");
    if (!label || !input) return;
    let candidate = label.parentElement;
    for (let depth = 0; candidate && depth < 4; depth += 1) {
      if (candidate.querySelectorAll("p, label").length === 1 && candidate.querySelectorAll("input, textarea").length === 1) {
        candidate.style.display = "none";
        return;
      }
      candidate = candidate.parentElement;
    }
    label.style.display = "none";
    input.style.display = "none";
  }

  function enhanceMouseControls() {
    const panel = document.getElementById("4c4d-orbit-sensitivity-panel");
    if (!panel || panel.dataset.c4dEnhanced === "true") return;
    panel.dataset.c4dEnhanced = "true";
    panel.setAttribute("aria-label", "Viewport navigation controls");

    const header = document.createElement("div");
    header.className = "c4d-controls-header";
    const title = document.createElement("span");
    title.textContent = "Navigation";
    const toggle = makeButton("c4d-controls-toggle", "Collapse viewport controls", "⌄", "c4d-controls-toggle");
    header.append(title, toggle);
    panel.prepend(header);

    const setCollapsed = (collapsed) => {
      panel.classList.toggle("c4d-controls-collapsed", collapsed);
      document.documentElement.dataset.c4dControlsCollapsed = String(collapsed);
      toggle.textContent = collapsed ? "⌃" : "⌄";
      toggle.setAttribute("aria-label", collapsed ? "Expand viewport controls" : "Collapse viewport controls");
      toggle.title = collapsed ? "Expand viewport controls" : "Collapse viewport controls";
      localStorage.setItem("4c4d-viewport-controls-collapsed", String(collapsed));
    };
    setCollapsed(localStorage.getItem("4c4d-viewport-controls-collapsed") === "true");
    toggle.addEventListener("click", () => setCollapsed(!panel.classList.contains("c4d-controls-collapsed")));
  }

  function makeButton(id, label, text, extraClass = "") {
    const button = document.createElement("button");
    button.type = "button";
    button.id = id;
    button.className = `c4d-seq-button ${extraClass}`.trim();
    button.setAttribute("aria-label", label);
    button.title = label;
    button.textContent = text;
    return button;
  }

  function initialize() {
    if (document.getElementById("c4d-sequencer")) return;
    if (!controlForLabel("Shot frame") || !controlForLabel("Duration (frames)")) {
      window.setTimeout(initialize, 250);
      return;
    }

    const root = document.createElement("section");
    root.id = "c4d-sequencer";
    root.setAttribute("aria-label", "Cinematic sequencer");
    root.innerHTML = `
      <div class="c4d-seq-header">
        <span class="c4d-seq-title">Sequencer</span>
        <span class="c4d-seq-shot" id="c4d-seq-shot">shot_001</span>
        <button type="button" class="c4d-seq-button" id="c4d-seq-start" aria-label="Go to shot start" title="Go to shot start">${icons.start}</button>
        <button type="button" class="c4d-seq-button" id="c4d-seq-prev" aria-label="Previous keyframe" title="Previous keyframe">${icons.previous}</button>
        <button type="button" class="c4d-seq-button" id="c4d-seq-play" aria-label="Play shot timeline" title="Play shot timeline">${icons.play}</button>
        <button type="button" class="c4d-seq-button" id="c4d-seq-camera-lock" aria-label="Lock camera to keyed shot" title="Lock camera to keyed shot (L)">${icons.unlock}</button>
        <button type="button" class="c4d-seq-button" id="c4d-seq-next" aria-label="Next keyframe" title="Next keyframe">${icons.next}</button>
        <button type="button" class="c4d-seq-button" id="c4d-seq-end" aria-label="Go to shot end" title="Go to shot end">${icons.end}</button>
        <span class="c4d-seq-timecode" id="c4d-seq-timecode">00:00:00:00</span>
        <span class="c4d-seq-dynamic" id="c4d-seq-dynamic" title="Dynamic splat frame follows shot progress">4D 0</span>
        <button type="button" class="c4d-seq-button c4d-seq-key-action" id="c4d-seq-add" aria-label="Add or update keyframe" title="Add or update keyframe (K)">${icons.addKey}</button>
        <button type="button" class="c4d-seq-button" id="c4d-seq-delete" aria-label="Delete selected keyframe" title="Delete selected keyframe">${icons.deleteKey}</button>
        <span class="c4d-seq-spacer"></span>
        <button type="button" class="c4d-seq-button" id="c4d-seq-theme" aria-label="Toggle dark mode">${icons.moon}</button>
        <button type="button" class="c4d-seq-button" id="c4d-seq-collapse" aria-label="Collapse sequencer" title="Collapse sequencer">${icons.collapse}</button>
      </div>
      <div class="c4d-seq-main">
        <div class="c4d-seq-labels">
          <div class="c4d-seq-label-ruler" id="c4d-seq-range">0 — 119</div>
          <div class="c4d-seq-track-label" data-track="camera"><span class="c4d-seq-track-dot"></span><span class="c4d-seq-track-name">Camera transform</span><span class="c4d-seq-track-value" id="c4d-seq-camera-value">2 keys</span></div>
          <div class="c4d-seq-track-label" data-track="lens"><span class="c4d-seq-track-dot"></span><span class="c4d-seq-track-name">Lens / FOV</span><span class="c4d-seq-track-value" id="c4d-seq-lens-value">50 mm</span></div>
        </div>
        <div class="c4d-seq-lanes" id="c4d-seq-lanes">
          <div class="c4d-seq-ruler" id="c4d-seq-ruler"></div>
          <div class="c4d-seq-lane" data-track="camera" id="c4d-seq-camera-lane"></div>
          <div class="c4d-seq-lane" data-track="lens" id="c4d-seq-lens-lane"></div>
          <div class="c4d-seq-playhead" id="c4d-seq-playhead"></div>
        </div>
      </div>`;
    document.body.appendChild(root);

    const savedTheme = localStorage.getItem("4c4d-ui-theme");
    applyTheme(savedTheme || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
    const collapsed = localStorage.getItem("4c4d-sequencer-collapsed") === "true";
    root.classList.toggle("c4d-seq-collapsed", collapsed);
    setIcon(document.getElementById("c4d-seq-collapse"), collapsed ? "expand" : "collapse");

    let keys = [];
    let duration = 120;
    let currentFrame = 0;
    let dragging = false;
    let renderSignature = "";

    const setFrame = (frame) => {
      const clamped = Math.max(0, Math.min(duration - 1, Math.round(frame)));
      setNativeValue(controlForLabel("Shot frame"), clamped);
    };

    const nearestKey = (direction) => {
      const frames = keys.map((key) => Number(key.frame)).sort((a, b) => a - b);
      if (!frames.length) return direction < 0 ? 0 : duration - 1;
      if (direction < 0) return [...frames].reverse().find((frame) => frame < currentFrame) ?? frames[0];
      return frames.find((frame) => frame > currentFrame) ?? frames[frames.length - 1];
    };

    document.getElementById("c4d-seq-start").addEventListener("click", () => setFrame(0));
    document.getElementById("c4d-seq-end").addEventListener("click", () => setFrame(duration - 1));
    document.getElementById("c4d-seq-prev").addEventListener("click", () => setFrame(nearestKey(-1)));
    document.getElementById("c4d-seq-next").addEventListener("click", () => setFrame(nearestKey(1)));
    document.getElementById("c4d-seq-play").addEventListener("click", () => controlForLabel("Play shot timeline", "checkbox")?.click());
    document.getElementById("c4d-seq-camera-lock").addEventListener("click", () => controlForLabel("Lock camera to shot", "checkbox")?.click());
    document.getElementById("c4d-seq-add").addEventListener("click", () => buttonWithText("Add / update keyframe")?.click());
    document.getElementById("c4d-seq-delete").addEventListener("click", () => buttonWithText("Delete selected keyframe")?.click());
    document.getElementById("c4d-seq-theme").addEventListener("click", () => {
      applyTheme(document.documentElement.dataset.c4dTheme === "dark" ? "light" : "dark");
    });
    document.getElementById("c4d-seq-collapse").addEventListener("click", (event) => {
      const next = !root.classList.contains("c4d-seq-collapsed");
      root.classList.toggle("c4d-seq-collapsed", next);
      setIcon(event.currentTarget, next ? "expand" : "collapse");
      localStorage.setItem("4c4d-sequencer-collapsed", String(next));
    });

    const lanes = document.getElementById("c4d-seq-lanes");
    const frameFromPointer = (event) => {
      const rectangle = lanes.getBoundingClientRect();
      return ((event.clientX - rectangle.left) / Math.max(1, rectangle.width)) * (duration - 1);
    };
    lanes.addEventListener("pointerdown", (event) => {
      if (event.target.closest(".c4d-seq-key")) return;
      dragging = true;
      lanes.setPointerCapture(event.pointerId);
      setFrame(frameFromPointer(event));
    });
    lanes.addEventListener("pointermove", (event) => {
      if (dragging) setFrame(frameFromPointer(event));
    });
    lanes.addEventListener("pointerup", (event) => {
      dragging = false;
      if (lanes.hasPointerCapture(event.pointerId)) lanes.releasePointerCapture(event.pointerId);
    });

    let sequencerHovered = false;
    root.addEventListener("pointerenter", () => { sequencerHovered = true; });
    root.addEventListener("pointerleave", () => { sequencerHovered = false; });
    document.addEventListener("keydown", (event) => {
      if (!sequencerHovered && !root.contains(document.activeElement)) return;
      if (event.target instanceof Element && event.target.matches("input, textarea, select")) return;
      if (event.repeat) return;
      if (event.code === "Space") {
        event.preventDefault();
        event.stopPropagation();
        controlForLabel("Play shot timeline", "checkbox")?.click();
      } else if (event.key.toLowerCase() === "k") {
        event.preventDefault();
        buttonWithText("Add / update keyframe")?.click();
      } else if (event.key.toLowerCase() === "l") {
        event.preventDefault();
        controlForLabel("Lock camera to shot", "checkbox")?.click();
      }
    }, true);

    const renderTracks = () => {
      const ruler = document.getElementById("c4d-seq-ruler");
      ruler.replaceChildren();
      for (let index = 0; index <= 4; index += 1) {
        const frame = Math.round((duration - 1) * index / 4);
        const tick = document.createElement("div");
        tick.className = "c4d-seq-tick";
        tick.style.left = `${index * 25}%`;
        const label = document.createElement("span");
        label.textContent = String(frame);
        if (index === 4) label.style.transform = "translateX(-100%)";
        tick.appendChild(label);
        ruler.appendChild(tick);
      }
      const lanesByTrack = {
        camera: document.getElementById("c4d-seq-camera-lane"),
        lens: document.getElementById("c4d-seq-lens-lane"),
      };
      Object.values(lanesByTrack).forEach((lane) => lane.replaceChildren());
      keys.forEach((key) => {
        Object.entries(lanesByTrack).forEach(([track, lane]) => {
          const marker = document.createElement("button");
          marker.type = "button";
          marker.className = "c4d-seq-key";
          marker.dataset.frame = String(key.frame);
          marker.style.left = `${Math.max(0, Math.min(100, Number(key.frame) / Math.max(1, duration - 1) * 100))}%`;
          marker.setAttribute("aria-label", `${track} keyframe at frame ${key.frame}`);
          marker.title = track === "lens"
              ? `Shot ${key.frame} · ${key.focal_mm} mm · ${key.fov_degrees}°`
              : `Camera transform · frame ${key.frame}`;
          marker.addEventListener("click", (event) => {
            event.stopPropagation();
            setFrame(Number(key.frame));
          });
          lane.appendChild(marker);
        });
      });
    };

    const poll = () => {
      enhanceMouseControls();
      hideSyncControl();
      duration = Math.max(2, Math.round(numericValue("Duration (frames)", 120)));
      currentFrame = Math.max(0, Math.min(duration - 1, Math.round(numericValue("Shot frame", 0))));
      const fps = Math.max(1, numericValue("Output FPS", 24));
      const sceneFrame = numericValue("Frame", 0);
      const focal = numericVectorValue("Filmback W · H · Focal (mm)", 2, 0);
      const shotName = controlForLabel("Shot name")?.value || "Camera shot";
      const syncInput = controlForLabel("Sequencer key data");
      try {
        const payload = JSON.parse(syncInput?.value || "{}");
        keys = Array.isArray(payload.keys) ? payload.keys : [];
      } catch (_error) {
        keys = [];
      }
      const signature = JSON.stringify([duration, keys]);
      if (signature !== renderSignature) {
        renderSignature = signature;
        renderTracks();
      }
      document.getElementById("c4d-seq-shot").textContent = shotName;
      document.getElementById("c4d-seq-timecode").textContent = timecode(currentFrame, fps);
      document.getElementById("c4d-seq-range").textContent = `0 — ${duration - 1}`;
      document.getElementById("c4d-seq-camera-value").textContent = `${keys.length} key${keys.length === 1 ? "" : "s"}`;
      document.getElementById("c4d-seq-dynamic").textContent = `4D ${Math.round(sceneFrame)}`;
      document.getElementById("c4d-seq-lens-value").textContent = `${focal.toFixed(1)} mm`;
      document.getElementById("c4d-seq-playhead").style.left = `${currentFrame / Math.max(1, duration - 1) * 100}%`;
      root.querySelectorAll(".c4d-seq-key").forEach((marker) => {
        marker.classList.toggle("c4d-seq-selected", Number(marker.dataset.frame) === currentFrame);
      });
      const playing = Boolean(controlForLabel("Play shot timeline", "checkbox")?.checked);
      const playButton = document.getElementById("c4d-seq-play");
      playButton.classList.toggle("c4d-seq-active", playing);
      setIcon(playButton, playing ? "pause" : "play");
      const playAction = playing ? "Pause shot timeline" : "Play shot timeline";
      playButton.setAttribute("aria-label", playAction);
      playButton.title = playAction;
      const cameraLocked = Boolean(controlForLabel("Lock camera to shot", "checkbox")?.checked);
      const lockButton = document.getElementById("c4d-seq-camera-lock");
      lockButton.classList.toggle("c4d-seq-active", cameraLocked);
      setIcon(lockButton, cameraLocked ? "lock" : "unlock");
      const lockAction = cameraLocked ? "Unlock camera for free navigation" : "Lock camera to keyed shot";
      lockButton.setAttribute("aria-label", lockAction);
      lockButton.title = `${lockAction} (L)`;
      window.setTimeout(poll, 90);
    };
    poll();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, { once: true });
  else initialize();
})();
