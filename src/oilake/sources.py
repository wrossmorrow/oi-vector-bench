"""Upstream Open Images V7 metadata files.

All URLs are from https://storage.googleapis.com/openimages/web/download_v7.html.
Image pixels for the ~1.9M "dense annotation" subset live in the CVDF mirror,
s3://open-images-dataset/{train,validation,test}/{ImageID}.jpg (public, unsigned).
"""

GCS = "https://storage.googleapis.com/openimages"

# Every image (9,178,275 rows): URLs, author, license, title, rotation.
ALL_IMAGE_IDS = f"{GCS}/2018_04/image_ids_and_rotation.csv"

# The subset whose pixels are in the S3 mirror. Validation and test are mirrored
# in full; train only for the "boxable" subset.
MIRROR_IMAGE_LISTS = {
    "train": f"{GCS}/2018_04/train/train-images-boxable-with-rotation.csv",
    "validation": f"{GCS}/2018_04/validation/validation-images-with-rotation.csv",
    "test": f"{GCS}/2018_04/test/test-images-with-rotation.csv",
}

# Image-level labels over all ~9M images. Columns: ImageID,Source,LabelName,Confidence.
# Human labels: Confidence 1 = verified present, 0 = verified ABSENT (keep these).
HUMAN_LABELS = {
    "train": f"{GCS}/v7/oidv7-train-annotations-human-imagelabels.csv",
    "validation": f"{GCS}/v7/oidv7-val-annotations-human-imagelabels.csv",
    "test": f"{GCS}/v7/oidv7-test-annotations-human-imagelabels.csv",
}
MACHINE_LABELS = {
    "train": f"{GCS}/v7/oidv7-train-annotations-machine-imagelabels.csv",
    "validation": f"{GCS}/v7/oidv7-val-annotations-machine-imagelabels.csv",
    "test": f"{GCS}/v7/oidv7-test-annotations-machine-imagelabels.csv",
}

CLASS_DESCRIPTIONS = f"{GCS}/v7/oidv7-class-descriptions.csv"            # ~20.6k classes
BOXABLE_CLASS_DESCRIPTIONS = f"{GCS}/v7/oidv7-class-descriptions-boxable.csv"
TRAINABLE_CLASSES = f"{GCS}/v7/oidv7-classes-trainable.txt"
# Hierarchy exists only for the 600 boxable classes.
BOXABLE_HIERARCHY = f"{GCS}/2018_04/bbox_labels_600_hierarchy.json"

MIRROR_BUCKET = "open-images-dataset"
MIRROR_REGION = "us-east-1"
