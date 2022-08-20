# Adds white padding to images to convert them to square images
import io
import os

import PIL
from PIL import Image, ImageOps

root = "path/to/dir"
for subdir, dirs, files in os.walk(root):
    for dir in dirs:
        writer = dir
        subroot = os.path.join(root, dir)
        for subdir, dirs, files in os.walk(subroot):
            for file in files:
                img_path = os.path.join(subroot, file)
                with open(img_path, 'rb') as f:
                    try:
                        img = Image.open(io.BytesIO(f.read()))
                    except PIL.UnidentifiedImageError:
                        # Check invalid images
                        print(img_path)

                img = ImageOps.pad(img, (256, 256), color='white')
                img.save(img_path)
