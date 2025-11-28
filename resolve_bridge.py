import os
import sys
import json
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# PATH JSON
# ---------------------------------------------------------------------------

def get_resolve_command_json_path() -> Path:
    appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
    return Path(appdata) / "GameplayFinder" / "resolve_command.json"


# ---------------------------------------------------------------------------
# LANCEMENT DU SCRIPT EXTERNE
# ---------------------------------------------------------------------------

def _launch_resolve_script():
    """
    Lance le script externe open_in_resolve_from_gameplayfinder.py
    (qui se trouve à côté de resolve_bridge.py)
    """
    script_path = Path(__file__).resolve().parent / "open_in_resolve_from_gameplayfinder.py"

    if not script_path.is_file():
        print(f"[resolve_bridge] Script introuvable : {script_path}")
        return

    python_exe = sys.executable or "python"

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        subprocess.Popen(
            [python_exe, str(script_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        print("[resolve_bridge] Script Resolve lancé.")
    except Exception as e:
        print(f"[resolve_bridge] ERREUR lancement script : {e!r}")


# ---------------------------------------------------------------------------
# ÉCRITURE DU JSON + LANCEMENT
# ---------------------------------------------------------------------------

def write_resolve_bridge_command(video_path: str, timestamp_sec: float) -> None:
    """
    Fonction appelée depuis Gameplay Finder.
    Écrit le JSON puis déclenche le script Resolve.
    """
    json_path = get_resolve_command_json_path()
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "video_path": str(video_path),
        "timestamp_sec": float(timestamp_sec),
    }

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[resolve_bridge] Commande écrite dans {json_path}")
    except Exception as e:
        print(f"[resolve_bridge] ERREUR écriture JSON : {e!r}")
        return

    # lancement immédiat
    _launch_resolve_script()
