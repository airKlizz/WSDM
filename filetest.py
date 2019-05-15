import os

_, dirs, _ os.walk("../Backup", topdown=False)

for folder in dirs:
    print(folder)
    for root, dirs, files in os.walk("../Backup/"+folder, topdown=False):
        for name in files:
            print(name)