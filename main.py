import os
import shutil

source_folder = r'E:\dogs\Images\n02116738-African_hunting_dog'
dest = r'E:\dogs\test_images'
x = 0
for item in os.listdir(source_folder):
    if x < 80:
        if os.path.isfile(os.path.join(source_folder, item)):
            try:
                src = os.path.join(source_folder,item)
                des = os.path.join(dest, item)
                # print(src)
                shutil.copyfile(src, des)
                item = item.replace(".jpg", "")
                print(item)
                x += 1
            except:
                continue
    else:
        break


