import os
import cv2
import json
import torch
import numpy as np


def video_to_frames(video_path, output_folder, fps=30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / fps))

    frame_count = 0
    image_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            img_filename = os.path.join(output_folder, f"{image_count:05d}.jpg")
            cv2.imwrite(img_filename, frame)
            image_count += 1

        frame_count += 1

    cap.release()


def get_bounding_box_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    centers = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 10 and h >= 10:
            bounding_boxes.append((x, y, w, h))

            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y))

    # rows, cols = np.nonzero(mask)
    #
    # x1, y1 = int(np.min(cols)), int(np.min(rows))
    # x2, y2 = int(np.max(cols)), int(np.max(rows))

    return bounding_boxes, centers


def save_data(video_dir, ann_frame_idx, video_segments, frame_names, save_path):
    all_image_info = []

    for out_frame_idx in range(ann_frame_idx, len(frame_names)):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            h, w = out_mask.shape[-2:]
            mask_image = out_mask.reshape(h, w, 1)
            mask_image = mask_image.squeeze()

            binary_mask = mask_image.astype(np.uint8) * 255
            bounding_boxes, centers = get_bounding_box_from_mask(binary_mask)

            original_image = cv2.imread(os.path.join(video_dir, frame_names[out_frame_idx]))

            segmented_image = cv2.bitwise_and(original_image, original_image, mask=binary_mask)
            cv2.imwrite(os.path.join(save_path, frame_names[out_frame_idx]), segmented_image)

            image_info = {
                "index": str(out_frame_idx),
                "bounding_boxes": bounding_boxes,
                "centers": centers
            }
            all_image_info.append(image_info)

    with open(os.path.join(save_path, "images_info.json"), 'w') as json_file:
        json.dump(all_image_info, json_file)


def pipeline(predictor, inference_state, ann_frame_idx, ann_obj_id, points,
             labels, video_segments, frame_names, video_dir, task='person'):
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    save_path = './results/' + str(task)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_data(video_dir, ann_frame_idx, video_segments, frame_names, save_path)

    video_segments.clear()


def multi_frame_segment(video_dir, frame_names, points, labels, task=''):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    from sam2.build_sam import build_sam2_video_predictor

    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    inference_state = predictor.init_state(video_path=video_dir)

    predictor.reset_state(inference_state)

    video_segments = {}
    ann_frame_idx = 0  # the frame index we interact with

    pipeline(predictor, inference_state, ann_frame_idx, int(task), points,
             labels, video_segments, frame_names, video_dir, task)
