import os
import json

mapping = {}
base = "dataset/all"

for breed in os.listdir(base):
    for img in os.listdir(os.path.join(base, breed)):
        mapping[img] = breed

with open("mapping.json","w") as f:
    json.dump(mapping, f)

print("✅ mapping.json created")