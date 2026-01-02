import os
import pickle
import torch
import numpy as np
import cv2
from typing import Optional, List
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from datetime import datetime

"""
This parser pipeline records you sprinting, extracts your pose landmarks for each frame, and enables you to label each frame
as a set position (1) or not (0). The pose landmarks are stored as a list[tuple[torch.tensor, int]] and pickled into a dataset 
directory.
"""

def downsample_video(
        cap: cv2.VideoCapture, output_path: str, target_fps: Optional[float] = 6.0
        ) -> None:
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    step = max(int(round(src_fps / target_fps)), 1) if target_fps else 1
    idx = 0

    while idx < 200:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            print("This is frame:", idx)
            out.write(frame)
        idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def initalize_landmarker() -> PoseLandmarker:
    base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, output_segmentation_masks=True
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    return detector


def lm_tensor_converter(results: NormalizedLandmarkList) -> torch.tensor:
    landmarks = results.pose_landmarks[0]
    np_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    torch_landmarks = torch.from_numpy(np_landmarks)
    return torch_landmarks


def normalize_landmarks(landmarks: np.ndarray, height: int, width: int) -> np.ndarray:
    landmarks[:, 0] *= width
    landmarks[:, 1] *= height
    return landmarks


def data_extractor(dir: str) -> list[tuple[torch.tensor, int]]:
    videos = sorted(os.listdir(dir))
    print(videos)

    for video in videos:

        video_path = os.path.join(dir, video)
        cap = cv2.VideoCapture(video_path)
        idx = 0

        dataset = []

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frames:", idx)
                break

            detector = initalize_landmarker()

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            name = video + " Frame " + str(idx)
            cv2.imshow(name, frame)
            key = cv2.waitKey(0)

            # 0 = not set position, 1 = set position
            label = 0
            if key == ord("0"):
                label = 0
            if key == ord("1"):
                label = 1
            if key == ord("2"):
                break

            landmarks = detector.detect(image)
            results = lm_tensor_converter(landmarks)

            labelled_points = (results, label)
            dataset.append(labelled_points)
            idx += 1
            
            print("---", video, "---")
            print("Frame: ", idx)
            print(labelled_points)
            print("\n")

        cap.release()
        cv2.destroyAllWindows()

    return dataset


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    TRAINING_DIR = "C:\\Data\\SprinterData\\TrainingVideos"
    DATASET_DIR = "C:\\Data\\SprinterData\\Datasets"
    os.makedirs(TRAINING_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    vid_filename = timestamp + ".mp4"
    dataset_filename = timestamp + ".pkl"
    output_path = os.path.join(TRAINING_DIR, vid_filename)
    output_pkl = os.path.join(DATASET_DIR, dataset_filename)

    downsample_video(cap, output_path)
    dataset = data_extractor(TRAINING_DIR)

    if dataset:
        with open(output_pkl, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Saved {len(dataset)} samples to {output_pkl}")
    else:
        print("\nNo samples were saved (no valid landmarks or images).")
