# Converts a folder of images to grayscale
import os
from PIL import Image, ImageOps

root = "path/to/dir"
for subdir, dirs, files in os.walk(root):
	for dir in dirs:
		writer = dir
		subroot = os.path.join(root, dir)
		for subdir, dirs, files in os.walk(subroot):
			for file in files:

				img_path = os.path.join(subroot, file)
				img = Image.open(img_path)
				gray_image = ImageOps.grayscale(img)
				gray_image.save(img_path)
