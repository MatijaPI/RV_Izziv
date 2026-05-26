"""
Centralno upravljanje poti za lokalni in strežniški način delovanja.

Način delovanja se določi z (padajoča prioriteta):
  1. Spremenljivka okolja RV_MODE=local|server
  2. Argument --mode local|server pri klicu main.py
  3. Samodejno zaznavanje (prisotnost /workspace ali /data)

Lokalni način:
  Vse poti so relativne glede na delovni imenik (kot prej).

Strežniški način (Docker):
  /workspace/  ← /media/FastDataMama/matijap/  (montiran z -v)
  /data/       ← /media/FastDataMama/data_rv_26/ (montiran z -v)
  /calib_photos/ ← /media/FastDataMama/zigab/calibration_photos/ (montiran z -v)
"""

import os
import sys

# ==============================================================================
# ZAZNAVANJE NAČINA DELOVANJA
# ==============================================================================

def _detect_mode():
    """
    Določi način delovanja. Prednost: env RV_MODE > samodejno zaznavanje.
    Vrne 'local' ali 'server'.
    """
    env = os.environ.get("RV_MODE", "").strip().lower()
    if env in ("local", "server"):
        return env

    # Samodejno zaznavanje: strežnik ima /workspace in /data montirani
    if os.path.isdir("/workspace") and os.path.isdir("/data"):
        return "server"

    return "local"


MODE = _detect_mode()


# ==============================================================================
# POTI – LOKALNI NAČIN
# ==============================================================================

def _local_paths():
    """Vrne slovar poti za lokalni način (relativno glede na delovni imenik)."""
    base = os.getcwd()
    return {
        "mode":               "local",
        "data_dir":           os.path.join(base, "data"),
        "output_dir":         os.path.join(base, "output"),
        "output_videos":      os.path.join(base, "output", "videos"),
        "output_graphs":      os.path.join(base, "output", "graphs"),
        "output_logs":        os.path.join(base, "output", "logs"),
        "model_dir":          os.path.join(base, "models"),
        "model_path":         os.path.join(base, "models", "hand_landmarker.task"),
        "calibration_conf":   os.path.join(base, "calibration", "conf"),
        "calibration_json":   os.path.join(base, "calibration", "calibration.json"),
        # Kalibracijske slike – lokalno v calibration/left|mid|right/
        "calib_photos": {
            "left":  os.path.join(base, "calibration", "left"),
            "mid":   os.path.join(base, "calibration", "mid"),
            "right": os.path.join(base, "calibration", "right"),
        },
    }


# ==============================================================================
# POTI – STREŽNIŠKI NAČIN (Docker)
# ==============================================================================

def _server_paths():
    """
    Vrne slovar poti za strežniški način.

    Pričakovani Docker mount-i:
      -v /media/FastDataMama/matijap/:/workspace
      -v /media/FastDataMama/data_rv_26/:/data
      -v /media/FastDataMama/zigab/calibration_photos/:/calib_photos
    """
    workspace   = "/workspace"
    data_root   = "/data"
    calib_root  = "/calib_photos"

    return {
        "mode":             "server",
        "data_dir":         os.path.join(data_root, "Data"),
        "output_dir":       os.path.join(workspace, "output"),
        "output_videos":    os.path.join(workspace, "output", "videos"),
        "output_graphs":    os.path.join(workspace, "output", "graphs"),
        "output_logs":      os.path.join(workspace, "output", "logs"),
        "model_dir":        os.path.join(workspace, "models"),
        "model_path":       os.path.join(workspace, "models", "hand_landmarker.task"),
        "calibration_conf": os.path.join(workspace, "calibration", "conf"),
        "calibration_json": os.path.join(workspace, "calibration", "calibration.json"),
        # Kalibracijske slike – strežnik ima ločeno mapo po kamerah
        "calib_photos": {
            "left":  os.path.join(calib_root, "cam_left_resized2"),
            "mid":   os.path.join(calib_root, "cam_mid_resized2"),
            "right": os.path.join(calib_root, "cam_right_resized2"),
        },
    }


# ==============================================================================
# AKTIVNE POTI
# ==============================================================================

def get_paths(mode_override=None):
    """
    Vrne slovar aktivnih poti.

    Args:
        mode_override: 'local', 'server' ali None (uporabi globalni MODE)

    Returns:
        dict s ključi: mode, data_dir, output_dir, output_videos, output_graphs,
                       output_logs, model_dir, model_path, calibration_conf,
                       calibration_json, calib_photos
    """
    m = (mode_override or MODE).strip().lower()
    if m == "server":
        return _server_paths()
    return _local_paths()


def print_config(paths):
    """Izpiše aktivno konfiguracijo poti."""
    print("  Način:         {}".format(paths["mode"].upper()))
    print("  Data:          {}".format(paths["data_dir"]))
    print("  Output:        {}".format(paths["output_dir"]))
    print("  Models:        {}".format(paths["model_dir"]))
    print("  Kalibracija:   {}".format(paths["calibration_conf"]))
    print("  Kal. slike:    left={left}  mid={mid}  right={right}".format(
        **paths["calib_photos"]))


def ensure_output_dirs(paths):
    """Ustvari vse izhodne mape, ki še ne obstajajo."""
    for key in ("output_videos", "output_graphs", "output_logs", "model_dir",
                "calibration_conf"):
        d = paths[key]
        if d:
            os.makedirs(d, exist_ok=True)