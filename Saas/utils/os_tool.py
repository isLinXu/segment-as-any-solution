
import os
def mkdir_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Directory created")
    else:
        print("Directory already exists")
