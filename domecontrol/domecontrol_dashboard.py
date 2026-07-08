"""
Flask HTTP server with Server-Sent Events for live light-color and pitch updates.

Endpoints
---------
GET /              HTML dashboard — auto-updates via SSE.
GET /api/tally     JSON snapshot of current color tallies.
GET /api/pitch     JSON snapshot of current pitch reading.
GET /api/status    Server health and config info.
GET /events        SSE stream — pushes JSON on any tally or pitch change.
"""

import json
import queue
import threading
import time

from flask import Flask, Response, jsonify, render_template_string

from domecontrol.color_detector import Camera, detect_colors

# Current per-color counts  e.g. {"red": 2, "green": 1}
_tally: dict[str, int] = {}
_tally_lock = threading.Lock()

# Shared state — pitch  (set by main.py after PitchState is created)
# ---------------------------------------------------------------------------
_pitch_state = None  # Will be a PitchState instance once audio starts
_pitch_config = None  # Will be a PitchConfig instance

# SSE subscribers — each is a queue that receives JSON strings.
# ---------------------------------------------------------------------------
_subscribers: list[queue.Queue[str]] = []
_sub_lock = threading.Lock()

_server_start_time = time.monotonic()


def set_pitch_state(state, config=None) -> None:
    """Called from main.py to inject the shared PitchState."""
    global _pitch_state, _pitch_config
    _pitch_state = state
    _pitch_config = config

# SSE helpers
# ---------------------------------------------------------------------------

def _publish(data: dict) -> None:
    """Push *data* (as JSON) to every connected SSE client."""
    payload = json.dumps(data)
    with _sub_lock:
        dead: list[queue.Queue[str]] = []
        for q in _subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)

def publish_pitch(pitch_dict: dict) -> None:
    """Called from the audio loop when the detected note changes."""
    _publish({"pitch": pitch_dict})


def _subscribe() -> queue.Queue[str]:
    q: queue.Queue[str] = queue.Queue(maxsize=64)
    with _sub_lock:
        _subscribers.append(q)
    return q


def _unsubscribe(q: queue.Queue[str]) -> None:
    with _sub_lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass


# Camera loop (runs in a background thread)
# ---------------------------------------------------------------------------

def _colors_to_tally(colors: list[str]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for c in colors:
        tally[c] = tally.get(c, 0) + 1
    return tally


def camera_loop(device: int = 0, fps: float = 10.0) -> None:
    """
    Continuously capture frames, detect colors, and publish changes.

    Runs forever — intended to be started in a daemon thread.
    """
    cam = Camera(device)
    interval = 1.0 / fps
    global _tally

    try:
        while True:
            t0 = time.monotonic()
            frame = cam.read_frame()
            if frame is None:
                time.sleep(interval)
                continue

            colors = detect_colors(frame)
            new_tally = _colors_to_tally(colors)

            with _tally_lock:
                if new_tally != _tally:
                    _tally = new_tally
                    _publish({"tally": _tally})

            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        cam.release()


# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dome Camera — Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, -apple-system, sans-serif;
    background: #111; color: #eee;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 2rem;
  }
  h1 { margin-bottom: 0.5rem; font-size: 1.6rem; }
  #status {
    font-size: 0.85rem; color: #888; margin-bottom: 1.5rem;
  }
  #status.connected { color: #4a4; }
  #status.disconnected { color: #a44; }

  .section { margin-bottom: 2rem; width: 420px; }
  .section h2 { font-size: 1.15rem; margin-bottom: 0.6rem; color: #ccc; }

  /* --- Color tally table --- */
  table {
    border-collapse: collapse; width: 100%;
    background: #1a1a1a; border-radius: 8px; overflow: hidden;
  }
  th, td { padding: 0.7rem 1.2rem; text-align: left; }
  th { background: #222; font-weight: 600; }
  td.color-name { text-transform: capitalize; }
  .swatch {
    display: inline-block; width: 14px; height: 14px;
    border-radius: 50%; margin-right: 0.5rem;
    vertical-align: middle; border: 1px solid #555;
  }
  tr:nth-child(even) { background: #1f1f1f; }

  /* --- Pitch card --- */
  .pitch-card {
    background: #1a1a1a; border-radius: 8px; padding: 1.2rem 1.5rem;
    display: flex; align-items: center; gap: 1.5rem;
  }
  .pitch-note {
    font-size: 2.8rem; font-weight: 700; min-width: 100px;
    text-align: center; color: #6cf;
  }
  .pitch-note.silence { color: #555; font-size: 1.4rem; }
  .pitch-details { font-size: 0.9rem; color: #aaa; line-height: 1.7; }
  .pitch-details span { color: #ddd; }
  .conf-bar {
    display: inline-block; height: 8px; border-radius: 4px;
    background: #6cf; vertical-align: middle; transition: width 0.15s;
  }

  .empty-msg { color: #666; font-style: italic; margin-top: 0.5rem; }
  #updated { font-size: 0.75rem; color: #555; margin-top: 1rem; }
</style>
</head>
<body>
<h1>Dome Camera Dashboard</h1>
<div id="status" class="disconnected">Connecting&hellip;</div>

<!-- Pitch section -->
<div class="section" id="pitch-section">
  <h2>Detected Pitch</h2>
  <div class="pitch-card">
    <div class="pitch-note silence" id="pitch-note">--</div>
    <div class="pitch-details">
      Frequency: <span id="pitch-freq">--</span> Hz<br>
      Confidence: <span id="pitch-conf">0%</span>
        <span class="conf-bar" id="pitch-conf-bar" style="width:0"></span><br>
      Offset: <span id="pitch-cents">0</span> cents
    </div>
  </div>
</div>

<!-- Color tally section -->
<div class="section">
  <h2>Color Tally</h2>
  <table id="tally-table">
    <thead><tr><th>Color</th><th>Count</th></tr></thead>
    <tbody id="tally-body"></tbody>
  </table>
  <div class="empty-msg" id="empty">No lights detected</div>
</div>

<div id="updated"></div>

<script>
const SWATCH_COLORS = {
  red: '#e33', orange: '#e80', yellow: '#ee0', green: '#3b3',
  cyan: '#0cc', blue: '#36f', purple: '#93f', magenta: '#d3d',
  white: '#eee'
};

function renderTally(tally) {
  const tbody = document.getElementById('tally-body');
  const empty = document.getElementById('empty');
  tbody.innerHTML = '';

  const entries = Object.entries(tally).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) {
    empty.style.display = 'block';
  } else {
    empty.style.display = 'none';
    for (const [color, count] of entries) {
      const tr = document.createElement('tr');
      const swatchHex = SWATCH_COLORS[color] || '#888';
      tr.innerHTML =
        '<td class="color-name"><span class="swatch" style="background:' +
        swatchHex + '"></span>' + color + '</td><td>' + count + '</td>';
      tbody.appendChild(tr);
    }
  }
}

function renderPitch(p) {
  const noteEl = document.getElementById('pitch-note');
  const freqEl = document.getElementById('pitch-freq');
  const confEl = document.getElementById('pitch-conf');
  const confBar = document.getElementById('pitch-conf-bar');
  const centsEl = document.getElementById('pitch-cents');

  if (!p.note) {
    noteEl.textContent = '--';
    noteEl.className = 'pitch-note silence';
    freqEl.textContent = '--';
    confEl.textContent = '0%';
    confBar.style.width = '0';
    centsEl.textContent = '0';
  } else {
    noteEl.textContent = p.note;
    noteEl.className = 'pitch-note';
    freqEl.textContent = p.frequency ? p.frequency.toFixed(1) : '--';
    const pct = Math.round(p.confidence * 100);
    confEl.textContent = pct + '%';
    confBar.style.width = Math.min(pct, 100) + 'px';
    const sign = p.cents_offset > 0 ? '+' : '';
    centsEl.textContent = sign + p.cents_offset;
  }
}

function setUpdated() {
  document.getElementById('updated').textContent =
    'Last update: ' + new Date().toLocaleTimeString();
}

function connect() {
  const statusEl = document.getElementById('status');
  const es = new EventSource('/events');

  es.onopen = function() {
    statusEl.textContent = 'Connected';
    statusEl.className = 'connected';
  };

  es.onmessage = function(e) {
    const data = JSON.parse(e.data);
    if (data.tally !== undefined) renderTally(data.tally);
    if (data.pitch !== undefined) renderPitch(data.pitch);
    setUpdated();
  };

  es.onerror = function() {
    statusEl.textContent = 'Disconnected \\u2014 reconnecting\\u2026';
    statusEl.className = 'disconnected';
    es.close();
    setTimeout(connect, 2000);
  };
}

// Fetch initial state then open SSE stream.
Promise.all([
  fetch('/api/tally').then(r => r.json()).then(renderTally).catch(() => {}),
  fetch('/api/pitch').then(r => r.json()).then(renderPitch).catch(() => {}),
]).then(setUpdated);
connect();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/tally")
def api_tally():
    with _tally_lock:
        return jsonify(_tally)


@app.route("/api/pitch")
def api_pitch():
    """Current pitch reading as JSON."""
    if _pitch_state is None:
        return jsonify({"note": None, "frequency": None, "confidence": 0,
                        "cents_offset": 0, "timestamp": 0})
    return jsonify(_pitch_state.get().to_dict())


@app.route("/api/status")
def api_status():
    """Server health and config info."""
    info: dict = {
        "running": True,
        "uptime_seconds": round(time.monotonic() - _server_start_time, 1),
    }
    if _pitch_config is not None:
        info["audio_device"] = _pitch_config.device or "system default"
        info["sample_rate"] = _pitch_config.sample_rate
        info["update_rate_hz"] = _pitch_config.update_rate_hz
    return jsonify(info)


@app.route("/events")
def events():
    """SSE endpoint — streams tally and pitch updates as they happen."""
    q = _subscribe()

    # Send current state immediately so the client is in sync.
    initial_parts: dict = {}
    with _tally_lock:
        initial_parts["tally"] = _tally
    if _pitch_state is not None:
        initial_parts["pitch"] = _pitch_state.get().to_dict()
    initial = json.dumps(initial_parts)

    def stream():
        yield f"data: {initial}\n\n"
        try:
            while True:
                try:
                    payload = q.get(timeout=30)
                    yield f"data: {payload}\n\n"
                except queue.Empty:
                    # Send a keep-alive comment so proxies don't drop us.
                    yield ": keepalive\n\n"
        except GeneratorExit:
            _unsubscribe(q)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})
