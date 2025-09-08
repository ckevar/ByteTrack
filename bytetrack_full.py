import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolox.tracker.byte_tracker import STrack
from sequenceloader import SequenceLoader
import argparse

import os
import cv2
import time
import torch

# Minimal args for BYTETracker
class Args(object):
    track_thresh = 0.5     # High-score threshold
    match_thresh = 0.8     # IoU threshold for matching
    track_buffer = 30      # Frames to keep lost tracks
    mot20 = False          # If using MOT20 dataset specifics
    def __init__(self, track_thresh, match_thresh, track_buffer, mot20):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.mot20 = mot20


def parse_args():

    parser = argparse.ArgumentParser(description="BYTETrack")

    # Tracker Related
    parser.add_argument("--load_detector", help="Path to detector (supports yolo only).",
                        default=None, required=True)
    parser.add_argument("--track_thresh", help="High-score threshold.",
                        default=0.5, type=float)
    parser.add_argument("--match_thresh", help="IoU threshold for matching.",
                        default=0.8, type=float)
    parser.add_argument("--track_buffer", help="Frames to keep lost tracks.",
                        default=30, type=int)
    parser.add_argument("--mot20", help="BYTETrackers handles mot20 differently",
                        default=False, type=bool)
    
    # Experiment Related
    parser.add_argument("--experiment_name", help="Results directory (not path)",
                        default=None, required=True)
    parser.add_argument("--data_dir", help="Dataset directory. Supported: MOT17, KITTI and WaymoV2-MOT",
                        default=None, required=True)

    parser.add_argument("--data_type", help="Format. Suported MOT17 and KITTI only.",
                        default="MOT17")

    parser.add_argument("--overwrite", help="If True, it processes the entire dataset from scratch. If False, it resumes from the cache file, if there exists a cache file.",
                        default=False)

    args = parser.parse_args()

    if ("MOT20" in args.data_dir or "mot20" in args.data_dir) and False == args.mot20:
        print("\n  [Warning]: You might want to enable --mot20 True, as it seems you are using MOT20. Otherwise ignore this message.\n")

    return args

def load_detector(model_filename):
    if "yolo" in model_filename:
        from ultralytics import YOLO
        return YOLO(model_filename)
    else:
        raise ValueError(f"Detect {detector_file} not supported.\n")


def unwrap_detections_ltrb_confs(detections):
    """
    confs = detections.boxes.conf.cpu().numpy()
    detections = detections.boxes.xyxy.cpu().numpy()
    """

    confs = detections.boxes.conf.unsqueeze(1)
    out = torch.cat([detections.boxes.xyxy, confs], dim=1)
    return out.cpu()
 
def store_results(output_file, results, data_type):
    if "MOT17" == data_type:
        save_format = "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
    elif "KITTI" == data_type:
        save_format = "%d %d pedestrian 0 0 -10 %.2f %.2f %.2f %.2f -10 -10 -10 -1000 -1000 -1000 -10"
        results = np.array(results)
        results[:, 4:6] += results[:, 2:4] # left-top-right-bottom: x1, y1, x2, y2
    else:
        raise ValueError(f"Dataset format --data_type={data_type} not supported.\n")

    fd = open(output_file, 'w')

    for row in results:
        print(save_format % (
            row[0], row[1], row[2], row[3], row[4], row[5]), file=fd)
    fd.close()

def run(sequence, detector, targs):
    print(f"Processing sequence {sequence['sequence_name']}")
    frame_rate = (sequence["update_ms"] / 1000) if sequence["update_ms"] else 30
    tracker = BYTETracker(args, frame_rate=frame_rate)
    results = []
    total_et = 0
    total_frame = 0
    frame_height, frame_width = sequence["image_size"]

    for frame_id in sequence["image_filenames"]:
        frame = cv2.imread(sequence["image_filenames"][frame_id], cv2.IMREAD_COLOR)
        start_time = time.time()

        detections = detector(frame, verbose=False)[0]
        detections = unwrap_detections_ltrb_confs(detections)
        online_targets = tracker.update(detections, 
                                        [frame_width, frame_height],
                                        [frame_width, frame_height])

        # Get tracking results
        for t in online_targets:
            tlwh = t.tlwh        # top-left x, y, w, h
            score = t.score
            results.append([
                frame_id, t.track_id, tlwh[0], tlwh[1], tlwh[2], tlwh[3], score])

        total_et += time.time() - start_time
        total_frame += 1
    
    print(f"Time elapsed: {total_et}, FPS: {total_frame/total_et}\n")
    
    # Store results.
    store_results(sequence["output_file"], results, sequence["data_type"])


if "__main__" == __name__:
    args = parse_args()

    sequences = SequenceLoader(args.data_dir,
                               args.data_type,
                               args.experiment_name,
                               args.overwrite)

    targs = Args(args.track_thresh, 
                 args.match_thresh, 
                 args.track_buffer, 
                 args.mot20)

    detector = load_detector(args.load_detector)

    for seq in sequences.next_sequence():
        run(seq, detector, targs)



