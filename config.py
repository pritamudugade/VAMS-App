
from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())


# Source
SOURCES_LIST = ["Image", "Video", "Webcam"]


# DL model config
DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
YOLOv8n = DETECTION_MODEL_DIR / "Object_detection.pt"
YOLOv8s = DETECTION_MODEL_DIR / "Activity_detector.pt"
YOLOv8m = DETECTION_MODEL_DIR / "Segmentation.pt"
YOLOv8l = DETECTION_MODEL_DIR / "Number_plate.pt"
YOLOv8x = DETECTION_MODEL_DIR / "Gun_detector.pt"

DETECTION_MODEL_LIST = [
    "Object_detection.pt",
    "Activity_detector.pt",
    "Segmentation.pt",
    "Number_plate.pt",
    "Gun_detector.pt"]


OBJECT_COUNTER = None
OBJECT_COUNTER1 = None