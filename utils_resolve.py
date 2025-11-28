#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilitaires spécifiques à DaVinci Resolve.
Actuellement : détection de Resolve.exe en cours d'exécution (Windows).
"""

import subprocess


def is_resolve_running() -> bool:
    """
    Retourne True si Resolve.exe tourne déjà (Windows).
    On utilise 'tasklist' pour éviter d'avoir besoin de lib externe.
    """
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Resolve.exe"],
            capture_output=True,
            text=True,
            check=False,
        )
        return "Resolve.exe" in result.stdout
    except Exception:
        # En cas de doute on considère que Resolve n'est pas lancé.
        return False
