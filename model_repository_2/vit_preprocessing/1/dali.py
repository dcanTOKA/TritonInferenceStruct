from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.triton import autoserialize

@autoserialize
@pipeline_def(batch_size=16, num_threads=4, device_id=0)
def dali_preprocess_pipeline():
    images = fn.external_source(device="cpu", name="INPUT_0", batch=True)
    original_sizes = fn.external_source(device="cpu", name="INPUT_1", batch=True)  # [height, width]

    images = fn.decoders.image(images, device='cpu', output_type=types.RGB)

    images = fn.slice(
        images,
        start=[0.0, 0.0],
        shape=original_sizes,
        axes=(0, 1),
        out_of_bounds_policy="pad"
    )

    images = fn.resize(
        images,
        resize_x=224,
        resize_y=224,
        mode="default"
    )

    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
    )

    return images
