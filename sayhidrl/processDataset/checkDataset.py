import json

with open("../data/coco_kvasirseg/train/annotations/train.json", "r") as f:
    data = json.load(f)

annos = data["annotations"]
counts = {}
for anno in annos:
    if anno["image_id"] not in counts:
        counts[anno["image_id"]] = 0
    counts[anno["image_id"]] += 1

max_id = 0
max_count = 0
for item in counts.items():
    if item[1] > max_count:
        max_count = item[1]
        max_id = item[0]

print(max_id, max_count)
