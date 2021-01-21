import imageio
import os
images = []

# for filename in os.listdir(modified_path):
filenames = []

for filename in os.listdir('./images'):
    if filename.startswith('img_'):
        filenames.append(filename)

sorted_files = [None] * len(filenames)

for filename in filenames:
    nr = int(filename[4:-4])
    sorted_files[nr-1] = filename

print(sorted_files)

images = []
for filename in sorted_files:
    images.append(imageio.imread(filename))
imageio.mimsave('coinrun.gif', images, fps=25)