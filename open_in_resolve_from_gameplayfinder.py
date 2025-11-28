#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
open_in_resolve_from_gameplayfinder.py

Script EXTERNE lancé par Gameplay Finder.
Il :
- lit le JSON resolve_command.json
- se connecte à DaVinci Resolve
- trouve ou importe un clip
- crée un segment local autour du timestamp demandé
- l'insère dans la timeline courante, à la playhead
"""

import os
import sys
import json


# ---------------------------------------------------------------------------
# LOG UTIL
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    try:
        print(msg)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# RESOLVE API
# ---------------------------------------------------------------------------

def ensure_resolve_api_on_path():
    programdata = os.environ.get("PROGRAMDATA")
    if not programdata:
        return
    modules_dir = os.path.join(
        programdata,
        "Blackmagic Design",
        "DaVinci Resolve",
        "Support",
        "Developer",
        "Scripting",
        "Modules",
    )
    if os.path.isdir(modules_dir) and modules_dir not in sys.path:
        sys.path.append(modules_dir)


def get_resolve_app():
    ensure_resolve_api_on_path()
    try:
        import DaVinciResolveScript  # type: ignore
    except Exception as e:
        log(f"[ERREUR] Import DaVinciResolveScript impossible : {e!r}")
        return None

    try:
        resolve = DaVinciResolveScript.scriptapp("Resolve")
    except Exception as e:
        log(f"[ERREUR] scriptapp('Resolve') a échoué : {e!r}")
        return None

    if not resolve:
        log("[ERREUR] Resolve renvoie None.")
        return None

    return resolve


# ---------------------------------------------------------------------------
# JSON COMMANDE
# ---------------------------------------------------------------------------

def get_json_path() -> str:
    appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(appdata, "GameplayFinder", "resolve_command.json")


def load_command():
    path = get_json_path()
    if not os.path.isfile(path):
        log(f"[ERREUR] JSON introuvable : {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log(f"[ERREUR] JSON illisible : {e!r}")
        return None

    video_path = data.get("video_path") or data.get("path")
    timestamp = data.get("timestamp_sec")

    if not video_path:
        log("[ERREUR] JSON invalide : video_path manquant")
        return None
    if timestamp is None:
        log("[ERREUR] JSON invalide : timestamp_sec manquant")
        return None

    try:
        timestamp = float(timestamp)
    except:
        log("[ERREUR] timestamp_sec n'est pas un float")
        return None

    return video_path, timestamp


# ---------------------------------------------------------------------------
# MEDIA POOL
# ---------------------------------------------------------------------------

def normalize(p): return os.path.normcase(os.path.normpath(p))


def find_clip_recursive(folder, target):
    target = normalize(target)
    try:
        clips = folder.GetClipList() or []
    except:
        clips = []

    for c in clips:
        try:
            props = c.GetClipProperty() or {}
            fp = props.get("File Path")
            if fp and normalize(fp) == target:
                return c
        except:
            pass

    try:
        subs = folder.GetSubFolderList() or []
    except:
        subs = []

    for s in subs:
        found = find_clip_recursive(s, target)
        if found:
            return found

    return None


def find_or_import_clip(media_pool, video_path: str):
    root = media_pool.GetRootFolder()
    if not root:
        log("[ERREUR] Pas de Media Pool root.")
        return None

    clip = find_clip_recursive(root, video_path)
    if clip:
        return clip

    log("[INFO] Importation du clip...")
    try:
        res = media_pool.ImportMedia([video_path])
    except Exception as e:
        log(f"[ERREUR] ImportMedia a échoué : {e!r}")
        return None

    if not res:
        return None

    return res[0]


# ---------------------------------------------------------------------------
# FPS, TIMECODE
# ---------------------------------------------------------------------------

def timecode_to_frame(tc: str, fps: float) -> int:
    h, m, s, f = [int(x) for x in tc.split(":")]
    return int(round((h * 3600 + m * 60 + s) * fps + f))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    resolve = get_resolve_app()
    if not resolve:
        return

    pm = resolve.GetProjectManager()
    project = pm.GetCurrentProject()
    if not project:
        log("[ERREUR] Aucun projet Resolve ouvert.")
        return

    media_pool = project.GetMediaPool()
    timeline = project.GetCurrentTimeline()

    if not media_pool or not timeline:
        log("[ERREUR] Media Pool ou Timeline introuvable.")
        return

    cmd = load_command()
    if not cmd:
        return

    video_path, timestamp = cmd

    clip = find_or_import_clip(media_pool, video_path)
    if not clip:
        log("[ERREUR] Clip introuvable/impossible à importer.")
        return

    props = clip.GetClipProperty() or {}
    try:
        clip_fps = float(props.get("FPS") or 60.0)
    except:
        clip_fps = 60.0

    frame_src = int(round(timestamp * clip_fps))
    pre = int(round(1.5 * clip_fps))
    post = int(round(1.5 * clip_fps))

    start_frame = max(0, frame_src - pre)
    end_frame = frame_src + post

    try:
        tl_fps = float(timeline.GetSetting("timelineFrameRate") or 25.0)
    except:
        tl_fps = 25.0

    try:
        tc_now = timeline.GetCurrentTimecode()
    except:
        tc_now = "00:00:00:00"

    record_frame = timecode_to_frame(tc_now, tl_fps)

    info = {
        "mediaPoolItem": clip,
        "startFrame": float(start_frame),
        "endFrame": float(end_frame),
        "recordFrame": float(record_frame),
        "trackIndex": 1,
    }

    try:
        media_pool.AppendToTimeline([info])
    except Exception as e:
        log(f"[ERREUR] AppendToTimeline a échoué : {e!r}")
        return

    log("[OK] Segment inséré dans la timeline courante.")


if __name__ == "__main__":
    main()
