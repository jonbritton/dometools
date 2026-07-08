#!/usr/bin/env python3
"""
Dome Camera — Colored Light Detection + Pitch Detection Server

Starts the camera capture loop and audio pitch detection loop in
background threads, then serves a live dashboard over HTTP.

Usage:
    python main.py [--device 0] [--host 0.0.0.0] [--port 5000] [--fps 10]
                   [--audio-device 0] [--config config.yaml] [--verbose]

Then open http://localhost:5000 in a browser.
"""

import argparse
import logging
import threading

from domecontrol.domecontrol_dashboard import app, camera_loop, set_pitch_state, publish_pitch

from domecontrol.pitch_server.audio import audio_loop
from domecontrol.pitch_server.config import PitchConfig, load_config
from domecontrol.pitch_server.state import PitchState


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Colored-light and pitch detection server",
    )
    # Camera args
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Camera capture rate in frames/sec (default: 10)")
    # HTTP args
    parser.add_argument("--host", default="0.0.0.0",
                        help="HTTP bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000,
                        help="HTTP port (default: 5000)")
    # Audio / pitch args
    parser.add_argument("--audio-device", default=None,
                        help="Audio device index or name (default: system default)")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml for pitch settings")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    # -- Logging ---------------------------------------------------------------
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # -- Pitch config ----------------------------------------------------------
    pitch_cfg = load_config(args.config)
    if args.audio_device is not None:
        # Accept integer device indices or string names.
        try:
            pitch_cfg.device = int(args.audio_device)
        except ValueError:
            pitch_cfg.device = args.audio_device

    # -- Shared pitch state ----------------------------------------------------
    pitch_state = PitchState()
    set_pitch_state(pitch_state, pitch_cfg)

    # -- Start camera loop -----------------------------------------------------
    cam_thread = threading.Thread(
        target=camera_loop,
        kwargs={"device": args.device, "fps": args.fps},
        daemon=True,
    )
    cam_thread.start()

    # -- Start audio loop ------------------------------------------------------
    audio_thread = threading.Thread(
        target=audio_loop,
        kwargs={
            "state": pitch_state,
            "config": pitch_cfg,
            "on_change": publish_pitch,
        },
        daemon=True,
    )
    audio_thread.start()

    # -- Start HTTP server -----------------------------------------------------
    print(f"Starting server on http://{args.host}:{args.port}")
    print("Endpoints:")
    print(f"  Dashboard  : http://{args.host}:{args.port}/")
    print(f"  Color tally: http://{args.host}:{args.port}/api/tally")
    print(f"  Pitch      : http://{args.host}:{args.port}/api/pitch")
    print(f"  Status     : http://{args.host}:{args.port}/api/status")
    print(f"  SSE stream : http://{args.host}:{args.port}/events")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
