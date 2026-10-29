import numpy as np
from scipy.ndimage import binary_dilation


class Vid:
    """Generator class that yields keyframe-aware temporally smoothed video frames."""

    def __init__(self, images, pad=0, sobel=2, keyframe=0.3, sensitivity=0.1):
        assert isinstance(
            images, (list, tuple)
        ), "Function only accepts list or tuple of image representations!"
        self.images = images
        if pad is None or pad == 0:
            self.pad = (None, None)
        elif isinstance(pad, int):
            self.pad = (pad, -pad)
        elif isinstance(pad, (list, tuple)):
            self.pad = (None if pad[0] == 0 else pad[0], None if pad[1] == 0 else -pad[1])
        else:
            raise ValueError("The value of 'pad' must be int or (int, int)")
        self.sobel = int(sobel)
        self.keyframe = keyframe
        self.sensitivity = sensitivity

    def __iter__(self):
        footprint = (
            np.ones((self.sobel * 6, self.sobel * 6), dtype=bool) if self.sobel else None
        )
        for i, image in enumerate(self.images):
            current_image = np.clip(
                np.copy(image[self.pad[0] : self.pad[1], :, :3]) / 255.0, 0.0, 1.0
            )
            if i == 0:
                last_image = np.copy(current_image)
                key_image = np.copy(current_image)
                yield last_image, True
            else:
                last_difference = np.max(
                    np.abs(current_image[:, :, :3] - last_image[:, :, :3]), axis=2
                )
                key_difference = np.max(
                    np.abs(current_image[:, :, :3] - key_image[:, :, :3]), axis=2
                )
                if (
                    np.mean(last_difference) < self.keyframe
                    or np.mean(key_difference) < self.keyframe
                ):
                    mask = key_difference > self.sensitivity
                    if footprint is not None:
                        mask = binary_dilation(mask, structure=footprint)
                    mask = np.expand_dims(mask, axis=-1)
                    last_image = current_image * mask + last_image * (1.0 - mask)
                    yield last_image, False
                else:
                    last_image = np.copy(current_image)
                    key_image = np.copy(current_image)
                    yield last_image, True
