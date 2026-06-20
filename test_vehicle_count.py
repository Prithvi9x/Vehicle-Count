#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().system('pip install -q ultralytics opencv-python-headless matplotlib numpy')


# In[1]:


from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json


# In[2]:


#IMAGE_PATH = "test_image.png"
input_root = "FirstFrame_annotate-1/FirstFrame_annotate-1"
output_folder = "FirstFrame_annotate-1/output_images"
os.makedirs(output_folder, exist_ok=True)


# In[3]:


valid_extensions = (".jpg", ".jpeg", ".png")


# In[4]:


model = YOLO('yolov8l.pt')


# In[5]:


COCO_CLASSES = {
0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}


# In[6]:


TARGET_CLASSES = {
'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'
}


# In[7]:


def class_name(cid):
    return COCO_CLASSES.get(int(cid), str(cid))


# In[8]:


def process_image_and_count(image_path, lane_count=3):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Could not read {image_path}")
        return None, None

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    # Define lanes
    lane_boundaries = [(int(W*i/lane_count), int(W*(i+1)/lane_count)) for i in range(lane_count)]

    # Run YOLO
    results = model.predict(source=image_path, conf=0.35, iou=0.5, device="cpu", save=False, verbose=False)
    res = results[0]

    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    class_ids = res.boxes.cls.cpu().numpy().astype(int)

    # Filter detections
    filtered = []
    for (box, score, cid) in zip(boxes, scores, class_ids):
        name = class_name(cid)
        if name not in TARGET_CLASSES:
            continue
        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) < 300:  # area filter
            continue
        filtered.append({'box': box, 'score': float(score), 'class_id': int(cid), 'name': name})

    # Count per lane
    lane_counts = [dict() for _ in lane_boundaries]
    for det in filtered:
        x1, y1, x2, y2 = det['box']
        cx = (x1 + x2) / 2.0
        lane_idx = None
        for i, (lx1, lx2) in enumerate(lane_boundaries):
            if lx1 <= cx <= lx2:
                lane_idx = i
                break
        if lane_idx is None:
            lane_idx = 0 if cx < lane_boundaries[0][0] else len(lane_boundaries) - 1

        name = det['name']
        lane_counts[lane_idx][name] = lane_counts[lane_idx].get(name, 0) + 1

    # Draw results
    out = img_bgr.copy()
    for i, (x1, x2) in enumerate(lane_boundaries):
        cv2.rectangle(out, (x1, 0), (x2, H), (255, 255, 255), 2)
        cv2.putText(out, f"Lane {i+1}", (x1+10, 30+20*i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y2 = 50 + 20*i
        for k, v in lane_counts[i].items():
            cv2.putText(out, f"{k}:{v}", (x1+10, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y2 += 20

    for det in filtered:
        x1, y1, x2, y2 = map(int, det['box'])
        cls, score = det['name'], det['score']
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(out, f"{cls} {score:.2f}", (x1, max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return out, lane_counts


# In[9]:


for root, dirs, files in os.walk(input_root):
    for filename in files:
        if not filename.lower().endswith(valid_extensions):
            continue

        img_path = os.path.join(root, filename)
        relative_path = os.path.relpath(root, input_root)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        save_path = os.path.join(output_subfolder, f"output_{filename}")
        json_path = os.path.join(output_subfolder, f"output_{os.path.splitext(filename)[0]}.json") 

        if os.path.exists(save_path):
            print(f"Skipping already processed: {filename}")
            continue

        output_image, lane_counts = process_image_and_count(img_path)
        if output_image is None:
            continue

        cv2.imwrite(save_path, output_image)
        label_data = {
            "image": filename,
            "lane_counts": {},
            "total_per_lane": [],
            "overall_total": 0
        }

        overall_total = 0
        for i, lane in enumerate(lane_counts):
            lane_total = sum(lane.values())
            overall_total += lane_total
            label_data["lane_counts"][f"Lane_{i+1}"] = lane
            label_data["total_per_lane"].append(lane_total)

        label_data["overall_total"] = overall_total

        with open(json_path, "w") as jf:
            json.dump(label_data, jf, indent=4)

        print(f"{img_path} processed → saved to {save_path}")
        print(f"Counts saved to {json_path}")

print("Completed.")


# In[ ]:




