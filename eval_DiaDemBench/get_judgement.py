import os
import sys
import json
import argparse
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm

try:
    from utils.evaluator import dialogue_extraction, eval_one_sample
except ImportError as e:
    print(f"Import Error: {e}")

fout = None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model and save results.")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to ground truth file.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing videos.")
    parser.add_argument("--caption_path", type=str, required=True, help="Path to caption prediction file.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the evaluation results.")
    parser.add_argument("--num_process", type=int, required=True, help="Number of processes to use.")
    return parser.parse_args()


def load_ground_truth(gt_path):
    gt_dict = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            video_id = data["video_id"]
            if video_id not in gt_dict:
                gt_dict[video_id] = {
                    "gt_list": data["gt_dialogues"],
                    "has_off_screen_speaker": data["has_off_screen_speaker"],
                    "has_same_sex_speakers": data["has_same_sex_speakers"],
                    "only_opposite_sex_speakers": data["only_opposite_sex_speakers"],
                    "speaker_num": data["speaker_num"],
                    "people_num_in_screen": data["people_num_in_screen"],
                    "is_single_shot": data["is_single_shot"]
                }
            else:
                raise ValueError(f"Duplicate video_id: {video_id}")
    return gt_dict


def get_existing_video_ids(save_path):
    exist_ids = set()
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    exist_ids.add(data["video_id"])
                except json.JSONDecodeError:
                    continue
    return exist_ids


def load_captions(caption_path, gt_dict, exist_ids):
    with open(caption_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            video_id = data["video_id"]
            
            if video_id in exist_ids:
                continue
            
            if video_id in gt_dict:
                gt_dict[video_id]["caption"] = data["pred"]


def process_video(args_tuple):
    global fout
    video_id, anno, video_dir = args_tuple
    
    caption = anno["caption"]
    gt_list = anno["gt_list"]

    dt_list = None
    for _ in range(3):
        try:
            dt_list = dialogue_extraction(caption)
            if dt_list is not None:
                break
        except Exception:
            continue

    gt_num, dt_num, matched_results = None, None, None

    if dt_list is not None:
        for _ in range(3):
            try:
                gt_num, dt_num, matched_results = eval_one_sample(video_id, video_dir, dt_list, gt_list)
                if recall is not None:
                    break
            except Exception:
                continue

    res = {
        "video_id": video_id,
        "model_generation": caption,
        "dt_list": dt_list,
        "gt_list": gt_list,
        "gt_num": gt_num,
        "dt_num": dt_num,
        "matched_results": matched_results,
        "has_off_screen_speaker": anno["has_off_screen_speaker"],
        "has_same_sex_speakers": anno["has_same_sex_speakers"],
        "only_opposite_sex_speakers": anno["only_opposite_sex_speakers"],
        "speaker_num": anno["speaker_num"],
        "people_num_in_screen": anno["people_num_in_screen"],
        "is_single_shot": anno["is_single_shot"]
    }

    if fout is not None:
        fout.write(json.dumps(res, ensure_ascii=False) + '\n')
        fout.flush()
    
    return res


def main():
    global fout
    args = parse_args()
    
    print("Loading Ground Truth...")
    gt_dict = load_ground_truth(args.gt_path)
    
    print("Checking existing results...")
    exist_ids = get_existing_video_ids(args.save_path)
    
    print("Loading Predictions...")
    load_captions(args.caption_path, gt_dict, exist_ids)

    items_to_process = []
    for video_id, anno in gt_dict.items():
        if "caption" not in anno:
            continue
        items_to_process.append((video_id, anno, args.video_dir))

    print(f"Total videos to process: {len(items_to_process)}")

    fout = open(args.save_path, 'a', encoding='utf-8')

    try:
        with Pool(processes=args.num_process) as pool:
            list(tqdm(pool.imap(process_video, items_to_process), total=len(items_to_process)))
    finally:
        fout.close()
        fout = None


if __name__ == "__main__":
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass

    main()