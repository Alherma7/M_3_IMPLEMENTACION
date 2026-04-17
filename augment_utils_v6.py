# %% [code]
import albumentations as A
import random
import numpy as np


def extract_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            parts = [float(e) if '.' in e else int(e) for e in parts]
            data.append(parts)
    return data


def augment_data(image, bboxes, seed=42):

    coords = [b[:4] for b in bboxes]
    class_labels = [b[4] for b in bboxes]

    transform = A.Compose(
        [
            A.Affine(
                scale=(0.9, 1.1), 
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                p=1
            )
        ],
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'], 
            min_visibility=0.0,
            min_area=0.0
        ),
        seed=seed
    )
    
    transformed = transform(image=image, bboxes=coords, class_labels=class_labels)

    new_bboxes = []
    for i in range(len(transformed['bboxes'])):
        box = list(transformed['bboxes'][i])
        box.append(transformed['class_labels'][i])
        new_bboxes.append(box)
        
    return transformed["image"], new_bboxes


def write_bboxes_to_txt(bboxes, filepath):
    with open(filepath, 'w') as f:
        for x, y, w, h, cls in bboxes:
            f.write(f"{int(cls)} {x} {y} {w} {h}\n")