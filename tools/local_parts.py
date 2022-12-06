import os

def draw_point(img, point, bbox_size=10, color=(0, 0, 255)):
    img[point[1] - bbox_size // 2: point[1] + bbox_size // 2, point[0] - bbox_size // 2: point[0] + bbox_size // 2] = color

    return img

def in_bbox(loc, bbox):
    return loc[0] >= bbox[0] and loc[0] <= bbox[1] and loc[1] >= bbox[2] and loc[1] <= bbox[3]

data_root = 'datasets/CUB_200_2011'

img_txt = os.path.join(data_root, 'images.txt')
cls_txt = os.path.join(data_root, 'image_class_labels.txt')
bbox_txt = os.path.join(data_root, 'bounding_boxes.txt')
train_txt = os.path.join(data_root, 'train_test_split.txt')
part_cls_txt = os.path.join(data_root, 'parts', 'parts.txt')
part_loc_txt = os.path.join(data_root, 'parts', 'part_locs.txt')

id_to_path = {}
with open(img_txt, 'r') as f:
    img_lines = f.readlines()
for img_line in img_lines:
    img_id, img_path = int(img_line.split(' ')[0]), img_line.split(' ')[1][:-1]
    img_folder, img_name = img_path.split('/')[0], img_path.split('/')[1]
    id_to_path[img_id] = (img_folder, img_name)

id_to_bbox = {}
with open(bbox_txt, 'r') as f:
    bbox_lines = f.readlines()
for bbox_line in bbox_lines:
    cts = bbox_line.split(' ')
    img_id, bbox_x, bbox_y, bbox_width, bbox_height = int(cts[0]), int(cts[1].split('.')[0]), int(cts[2].split('.')[0]), int(cts[3].split('.')[0]), int(cts[4].split('.')[0])
    bbox_x2, bbox_y2 = bbox_x + bbox_width, bbox_y + bbox_height
    id_to_bbox[img_id] = (bbox_x, bbox_y, bbox_x2, bbox_y2)

# id_to_cls = {}
cls_to_id = {}
with open(cls_txt, 'r') as f:
    cls_lines = f.readlines()
for cls_line in cls_lines:
    img_id, cls_id = int(cls_line.split(' ')[0]), int(cls_line.split(' ')[1]) - 1   # 0 -> 199
    if cls_id not in cls_to_id.keys():
        cls_to_id[cls_id] = []
    cls_to_id[cls_id].append(img_id)

id_to_train = {}
with open(train_txt, 'r') as f:
    train_lines = f.readlines()
for train_line in train_lines:
    img_id, is_train = int(train_line.split(' ')[0]), int(train_line.split(' ')[1][:-1])
    id_to_train[img_id] = is_train

part_id_to_part = {}
with open(part_cls_txt, 'r') as f:
    part_cls_lines = f.readlines()
for part_cls_line in part_cls_lines:
    id_len = len(part_cls_line.split(' ')[0])
    part_id, part_name = part_cls_line[:id_len], part_cls_line[id_len + 1:]
    part_id_to_part[part_id] = part_name

id_to_part_loc = {}
with open(part_loc_txt, 'r') as f:
    part_loc_lines = f.readlines()
for part_loc_line in part_loc_lines:
    content = part_loc_line.split(' ')
    img_id, part_id, loc_x, loc_y, visible = int(content[0]), int(content[1]), int(float(content[2])), int(float(content[3])), int(content[4])
    if img_id not in id_to_part_loc.keys():
        id_to_part_loc[img_id] = []
    if visible == 1:
        id_to_part_loc[img_id].append([part_id, loc_x, loc_y])