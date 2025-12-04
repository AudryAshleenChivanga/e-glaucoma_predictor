"""
Image preprocessing utilities for retinal image analysis.
"""
import numpy as np
from PIL import Image
import io
from typing import Tuple, Optional
from config import MODEL_INPUT_SIZE, MODEL_CHANNELS


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load an image from bytes."""
    return Image.open(io.BytesIO(image_bytes))


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = MODEL_INPUT_SIZE,
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess a retinal image for model inference.
    
    Args:
        image: PIL Image object
        target_size: Target dimensions (height, width)
        normalize: Whether to normalize pixel values to [0, 1]
    
    Returns:
        Preprocessed numpy array ready for model input
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to target dimensions
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize pixel values
    if normalize:
        img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def augment_image(image: np.ndarray, augmentation_type: str = "basic") -> np.ndarray:
    """
    Apply data augmentation to an image.
    
    Args:
        image: Input image array
        augmentation_type: Type of augmentation to apply
    
    Returns:
        Augmented image array
    """
    augmented = image.copy()
    
    if augmentation_type == "flip_horizontal":
        augmented = np.fliplr(augmented)
    elif augmentation_type == "flip_vertical":
        augmented = np.flipud(augmented)
    elif augmentation_type == "rotate_90":
        augmented = np.rot90(augmented)
    elif augmentation_type == "brightness":
        factor = np.random.uniform(0.8, 1.2)
        augmented = np.clip(augmented * factor, 0, 1)
    
    return augmented


def validate_image(image_bytes: bytes) -> Tuple[bool, Optional[str]]:
    """
    Validate that the uploaded file is a valid image.
    
    Args:
        image_bytes: Raw image bytes
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        image = load_image_from_bytes(image_bytes)
        
        # Check image format
        if image.format not in ['JPEG', 'PNG', 'BMP', 'TIFF', None]:
            return False, "Unsupported image format. Please use JPEG, PNG, BMP, or TIFF."
        
        # Check image dimensions
        width, height = image.size
        if width < 50 or height < 50:
            return False, "Image is too small. Minimum dimensions are 50x50 pixels."
        
        if width > 10000 or height > 10000:
            return False, "Image is too large. Maximum dimensions are 10000x10000 pixels."
        
        return True, None
        
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def get_image_info(image_bytes: bytes) -> dict:
    """
    Get information about an uploaded image.
    
    Args:
        image_bytes: Raw image bytes
    
    Returns:
        Dictionary containing image information
    """
    image = load_image_from_bytes(image_bytes)
    
    return {
        "format": image.format,
        "mode": image.mode,
        "width": image.size[0],
        "height": image.size[1],
        "size_bytes": len(image_bytes)
    }

