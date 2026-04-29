from PIL import Image
import os

for root, dirs, files in os.walk('dataset'):
    for f in files:
        path = os.path.join(root, f)
        try:
            Image.open(path).verify()
        except Exception as e:
            print(f'Deleted: {path}')
            os.remove(path)

print("Done.")
