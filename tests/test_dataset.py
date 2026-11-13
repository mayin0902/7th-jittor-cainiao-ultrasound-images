"""Exercise image loading with a minimal Jittor boundary; no model compilation."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def dataset_class():
    jittor = ModuleType("jittor")
    jittor.array = np.asarray
    dataset = ModuleType("jittor.dataset")

    class Dataset:
        def __init__(self, **kwargs):
            pass

    dataset.Dataset = Dataset
    source = Path(__file__).resolve().parents[1] / "data" / "dataset.py"
    spec = importlib.util.spec_from_file_location("ultrasound_dataset_under_test", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"jittor": jittor, "jittor.dataset": dataset}):
        spec.loader.exec_module(module)
    return module.CustomImageDataset


@pytest.fixture
def image_root(tmp_path):
    Image.fromarray(np.full((5, 7), 23, dtype=np.uint8)).save(tmp_path / "image.png")
    return tmp_path


def test_default_augmentations_load_image(dataset_class, image_root):
    dataset = dataset_class(image_root)
    image, name = dataset[0]
    assert image.shape == (3, 5, 7)
    assert name == "image.png"
    assert np.all(image == 23)


@pytest.mark.parametrize("mode", ["train", "val"])
def test_disabled_augmentation_still_returns_channels_first(dataset_class, image_root, mode):
    dataset = dataset_class(image_root, augmentations=(None, None), mode_type=mode,
                            metadata={"image_name": ["image.png"], "label": [2]})
    image, label = dataset[0]
    assert image.shape == (3, 5, 7)
    np.testing.assert_array_equal(label, [0, 0, 1, 0, 0, 0])


@pytest.mark.parametrize("mode", ["train", "val"])
def test_enabled_augmentation_remains_channels_first(dataset_class, image_root, mode):
    def transform(*, image):
        return {"image": image.astype(np.float32) / 255}

    dataset = dataset_class(image_root, augmentations=(transform, transform), mode_type=mode)
    image, name = dataset[0]
    assert image.shape == (3, 5, 7)
    assert image.dtype == np.float32
    np.testing.assert_allclose(image, 23 / 255)
    assert name == "image.png"
