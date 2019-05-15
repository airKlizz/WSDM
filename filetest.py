import os

for _, dirs, _ in os.walk("../Backup", topdown=False):
   for name in dirs:
      print(name)

print("DEDE")

for folder in dirs:
    print(folder)
    for root, dirs, files in os.walk("../Backup/"+folder, topdown=False):
        for name in files:
            print(name)