#!/usr/bin/env python3
"""
scripts/check_carla_connection.py

Verifies that a CARLA simulator is reachable and that the installed Python API
is compatible with the running server. Exits with a non-zero code and a clear
human-readable message if anything is wrong.

Usage:
    python scripts/check_carla_connection.py [--host localhost] [--port 2000]
"""

import argparse
import os
import sys

# Make the workspace root importable when running this script directly.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _versions_compatible(client_version: str, server_version: str) -> bool:
    """Treat compatibility as matching major.minor (e.g. '0.9.15' vs '0.9.15')."""
    def parse(v):
        parts = v.split(".")
        try:
            return tuple(int(p) for p in parts[:2])
        except ValueError:
            return (0, 0)
    return parse(client_version) == parse(server_version)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CARLA simulator connectivity")
    parser.add_argument("--host", default="localhost", help="CARLA host (default: localhost)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port (default: 2000)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Connection timeout seconds")
    args = parser.parse_args()

    # 1) Is the carla python package installed at all?
    try:
        import carla  # noqa: F401
    except ImportError:
        print(
            "ERROR: The 'carla' Python module is not installed.\n"
            "Install the Python API matching your simulator version, e.g.:\n"
            "    pip install carla==0.9.15\n"
            "or append the CARLA .whl/.egg to PYTHONPATH.",
            file=sys.stderr,
        )
        return 2

    # 2) Can we reach the server and read its world/map?
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        world = client.get_world()
        carla_map = world.get_map()
    except Exception as e:
        print(
            f"ERROR: Could not connect to CARLA at {args.host}:{args.port}.\n"
            f"Reason: {e}\n"
            "Make sure the CARLA simulator is running and the host/port are correct.",
            file=sys.stderr,
        )
        return 3

    # 3) Are the client and server versions compatible?
    try:
        client_version = client.get_client_version()
        server_version = client.get_server_version()
    except Exception as e:
        # Some builds lack these accessors; treat as non-fatal.
        client_version = server_version = "unknown"
        print(f"WARNING: could not read client/server versions: {e}")

    map_name = carla_map.name
    spawn_points = carla_map.get_spawn_points()

    print("CARLA connection: OK")
    print(f"  Host / port         : {args.host}:{args.port}")
    print(f"  Client API version  : {client_version}")
    print(f"  Server version      : {server_version}")
    print(f"  Map                 : {map_name}")
    print(f"  Spawn points        : {len(spawn_points)}")

    if client_version != "unknown" and not _versions_compatible(client_version, server_version):
        print(
            "WARNING: client and server versions differ. Behavior may be unstable;\n"
            "         install the carla Python package matching the server version.",
            file=sys.stderr,
        )
        # Differing versions are a warning, not a hard failure.
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
