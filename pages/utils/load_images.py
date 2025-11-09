import os
import random
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import streamlit as st
import logging

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

# Get the project root directory
if "pages" in str(Path(__file__).parent):
    PROJECT_ROOT = Path(__file__).parent.parent.parent
else:
    PROJECT_ROOT = Path(__file__).parent

try:
    img_path_str = st.secrets["PATH_FOLDER_IMAGES"]
    DEFAULT_PATH = Path(img_path_str) if Path(img_path_str).is_absolute() else PROJECT_ROOT / img_path_str
    logger.info(f"Images folder from secrets: {DEFAULT_PATH}")
except KeyError:
    env_path = os.environ.get("PATH_FOLDER_IMAGES", "")
    if env_path:
        DEFAULT_PATH = Path(env_path)
    else:
        DEFAULT_PATH = PROJECT_ROOT / "pages" / "utils" / "images" / "1" / "DRIMDB_sample"


print(f"Images folder: {DEFAULT_PATH}")


def load_random_images(
    images_folder: Path | str = DEFAULT_PATH,
    num_images: int = 6
) -> List[Tuple[Image.Image, str, str]]:
    """
    Load random images from a folder with 'Good' and 'Bad' subfolders.
    
    Args:
        images_folder: Path to the main images folder
        num_images: Number of images to load
    
    Returns:
        List of tuples: [(image, label, filename), ...]
    """
    images_folder = Path(images_folder)
    
    all_images = _collect_images(images_folder)
    if not all_images:
        st.warning("No images found in the specified folder.")
        return []
    
    selected = random.sample(all_images, min(num_images, len(all_images)))
    return _load_selected_images(selected)


def _collect_images(base_folder: Path) -> List[Tuple[Path, str, str]]:
    """Collect image paths from Good and Bad subfolders."""
    all_images = []
    for label in ('Good', 'Bad'):
        folder = base_folder/label
        print(folder)
        if folder.exists():
            print("Folder found")
            images = [
                (folder / f, label, f)
                for f in os.listdir(folder)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            ]
            all_images.extend(images)
        else:
            print("Folder not found")
    return all_images   


def _load_selected_images(
    selected: List[Tuple[Path, str, str]]
) -> List[Tuple[Image.Image, str, str]]:
    """Load selected images, skipping any that fail."""
    loaded_images = []
    
    for img_path, label, filename in selected:
        try:
            img = Image.open(img_path)
            loaded_images.append((img, label, filename))
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
    
    return loaded_images


def display_images(images: List[Tuple[Image.Image, str, str]], cols: int = 3):
    """Display images in a Streamlit grid."""
    columns = st.columns(cols)
    
    for idx, (img, label, filename) in enumerate(images):
        with columns[idx % cols]:
            st.image(img, caption=f"{label}: {filename}", width='content')