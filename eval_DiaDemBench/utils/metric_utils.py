import os
import sys
import time
import json
import traceback
import multiprocessing
import random
import re
import regex
import numpy as np
from itertools import groupby
from opencc import OpenCC
from google import genai
from google.genai import types
from google.genai.types import (
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)

# ================= Configuration =================
CREDENTIALS_PATH = None # TODO
assert CREDENTIALS_PATH is not None
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH

LOCATION = "global"
MODEL_NAME = "gemini-2.5-flash"
SEED = 42

t2s = OpenCC('t2s')

# ================= Setup =================
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

set_seed(SEED)

try:
    with open(CREDENTIALS_PATH, 'r') as f:
        user_info = json.load(f)
    PROJECT_ID = user_info['project_id']
except Exception as e:
    print(f"Error loading credentials: {e}")
    sys.exit(1)

safety_settings = [
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.OFF),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.OFF),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.OFF),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.OFF)
]

GEN_CONFIG = types.GenerateContentConfig(
    temperature=0,
    top_p=0.001,
    thinking_config=types.ThinkingConfig(
        include_thoughts=False,
        thinking_budget=128
    ),
    safety_settings=safety_settings,
    seed=SEED,
)

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

PROMPT_TEMPLATE = """Given a video and several pairs of descriptive phrases about a certain subject, please help me determine whether the subjects in each pair refer to the same entity in the video.
For each pair of phrases, respond with 'Yes' or 'No', separated by a single space, without any extra characters. For example, if three pairs of phrases are provided, a valid response format would be: 'Yes No Yes'.

Descriptive phrases (each line contains a single pair):
{}
"""

# ================= Helper Functions =================
def generate_mm(filename, prompt):
    try:
        with open(filename, "rb") as f:
            video_bytes = f.read()
    except IOError as e:
        print(f"Error reading video file {filename}: {e}")
        return None, None

    video_file = types.Part(
        inline_data=types.Blob(
            data=video_bytes,
            mime_type='video/mp4'),
        video_metadata=types.VideoMetadata(fps=1)
    )
    contents = [video_file, prompt]

    answer = None
    thinking = None

    for i in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=GEN_CONFIG
            )

            ans_parts = []
            think_parts = []
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if not part.text:
                        continue
                    if part.thought:
                        think_parts.append(part.text)
                    else:
                        ans_parts.append(part.text)
            
            answer = ''.join(ans_parts) + '\n'
            thinking = ''.join(think_parts) + '\n'
            break
        except Exception:
            traceback.print_exc()
            time.sleep(3)

    if answer is None:
        return None, None
    return answer.strip(), thinking.strip()


def edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]


def calculate_similarity(str1, str2):
    str1 = ''.join(str1.split())
    str2 = ''.join(str2.split())
    dist = edit_distance(str1, str2)
    max_len = max(len(str1), len(str2))
    return 1.0 - dist / max_len


def flexible_longest_common_subsequence(dt_list, gt_list, thres=0.6, max_merge_window=6):
    m = len(dt_list)
    n = len(gt_list)

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    path = [[(0, 0)] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            score_skip_dt = dp[i - 1][j]
            score_skip_gt = dp[i][j - 1]
            
            best_score = max(score_skip_dt, score_skip_gt)
            best_step = (1, 0) if score_skip_dt >= score_skip_gt else (0, 1)

            gt_item = gt_list[j-1]
            gt_content = gt_item[1]
            
            dt_merge_content = ""
            current_speaker = None
            
            for k in range(1, min(max_merge_window, i) + 1):
                idx = i - k
                curr_dt_item = dt_list[idx]
                
                if k == 1:
                    current_speaker = curr_dt_item[0]
                    dt_merge_content = curr_dt_item[1]
                else:
                    if curr_dt_item[0] != current_speaker:
                        break
                    dt_merge_content = curr_dt_item[1] + " " + dt_merge_content

                sim = calculate_similarity(dt_merge_content, gt_content)
                
                if sim > thres:
                    candidate_score = dp[i - k][j - 1] + sim
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_step = (k, 1)

            dt_item = dt_list[i-1]
            dt_content = dt_item[1]
            gt_merge_content = ""
            current_gt_speaker = None
            
            for k in range(1, min(max_merge_window, j) + 1):
                if k == 1:
                    current_gt_speaker = gt_list[j-k][0]
                    gt_merge_content = gt_list[j-k][1]
                    continue 

                idx = j - k
                curr_gt_item = gt_list[idx]
                
                if curr_gt_item[0] != current_gt_speaker:
                    break
                gt_merge_content = curr_gt_item[1] + " " + gt_merge_content
                
                sim = calculate_similarity(dt_content, gt_merge_content)
                
                if sim > thres:
                    candidate_score = dp[i - 1][j - k] + sim
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_step = (1, k)

            dt_merge_content = ""
            current_dt_spk = None

            for k in range(1, min(max_merge_window, i) + 1):
                idx_dt = i - k
                curr_dt_item = dt_list[idx_dt]
                if k == 1:
                    current_dt_spk = curr_dt_item[0]
                    dt_merge_content = curr_dt_item[1]
                else:
                    if curr_dt_item[0] != current_dt_spk: break
                    dt_merge_content = curr_dt_item[1] + " " + dt_merge_content
                
                gt_merge_content = ""
                current_gt_spk = None
                
                for l in range(1, min(max_merge_window, j) + 1):
                    idx_gt = j - l
                    curr_gt_item = gt_list[idx_gt]
                    
                    if l == 1:
                        current_gt_spk = curr_gt_item[0]
                        gt_merge_content = curr_gt_item[1]
                    else:
                        if curr_gt_item[0] != current_gt_spk: break
                        gt_merge_content = curr_gt_item[1] + " " + gt_merge_content

                    if k > 1 and l > 1:
                        sim = calculate_similarity(dt_merge_content, gt_merge_content)
                        if sim > thres:
                            candidate_score = dp[i - k][j - l] + sim
                            if candidate_score > best_score:
                                best_score = candidate_score
                                best_step = (k, l)
            
            dp[i][j] = best_score
            path[i][j] = best_step

    matched_pairs = [] # ([dt_indices], [gt_indices], score)
    
    i, j = m, n
    while i > 0 and j > 0:
        step_i, step_j = path[i][j]
        
        if step_i == 0 and step_j == 0: 
            break
            
        prev_score = dp[i - step_i][j - step_j]
        curr_score = dp[i][j]
        
        if curr_score > prev_score + 1e-6:
            dt_indices = list(range(i - step_i, i))
            gt_indices = list(range(j - step_j, j))
            
            matched_pairs.append({
                'dt_indices': dt_indices,
                'gt_indices': gt_indices,
                'score': curr_score - prev_score
            })
        
        i -= step_i
        j -= step_j

    matched_pairs.reverse()
    return dp[m][n], matched_pairs

# ================= Data Cleaning =================
def contains_japanese(text):
    japanese_pattern = regex.compile(r'[\u3040-\u309f\u30a0-\u30ff]')
    return bool(japanese_pattern.search(text))


def traditional2simple(text: str) -> bool:
    return t2s.convert(text)


def clean_conversations(str_list):
    res = []
    for each in str_list:
        each = each.strip().replace("\"", "").replace("“", "").replace("”", "")
        if each and each[-1] in ["。", ".", "?", "？", "!", "！"]:
            each = each[:-1]

        if ":" not in each:
            continue

        index = each.find(":")
        sbj = each[:index].strip()
        content = each[index+1:].strip()
        
        content = re.sub(r"[^\w\s\u4e00-\u9fff]", "", content)
        content = content.lower()

        if not contains_japanese(content):
            content = traditional2simple(content)

        if len(sbj) > 100: sbj = sbj[:100]
        if len(content) > 1000: content = content[:1000]

        res.append((sbj, content))
    return res


def prepare_prompt(dt_list, gt_list, indices):
    pairs = ""
    for item in indices:
        dt_idx = item["dt_indices"][0]
        gt_idx = item["gt_indices"][0]
        pairs += f'"{dt_list[dt_idx][0]}" | "{gt_list[gt_idx][0]}"\n'
    return PROMPT_TEMPLATE.format(pairs)
