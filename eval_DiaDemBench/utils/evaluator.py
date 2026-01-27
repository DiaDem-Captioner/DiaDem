import os
import sys
import time
import json
import traceback
import random
import numpy as np
from google import genai
from google.genai import types
from google.genai.types import (
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)

# ================= Configuration =================
from .metric_utils import (
    clean_conversations,
    flexible_longest_common_subsequence,
    prepare_prompt,
    generate_mm
)

CREDENTIALS_PATH = None # TODO
assert CREDENTIALS_PATH is not None
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH

LOCATION = "global"
MODEL_NAME = "gemini-2.5-pro"
SEED = 42

# ================= Setup =================
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

set_seed(SEED)

# Load User Info
try:
    with open(CREDENTIALS_PATH, 'r') as f:
        user_info = json.load(f)
    PROJECT_ID = user_info['project_id']
except Exception as e:
    print(f"Error loading credentials: {e}")
    sys.exit(1)

# Gemini Client Config
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


# ================= Core Functions =================
def generate(prompt):
    contents = [prompt]
    caption = None
    thinking = None

    for i in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=GEN_CONFIG
            )

            ans_parts = []
            thinking_parts = []
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if not part.text:
                        continue
                    if part.thought:
                        thinking_parts.append(part.text)
                    else:
                        ans_parts.append(part.text)
            
            answer = ''.join(ans_parts) + '\n'
            thinking = ''.join(thinking_parts) + '\n'
            break
            
        except Exception:
            traceback.print_exc()
            time.sleep(3)

    if answer is None:
        return None, None
    return answer.strip(), thinking.strip()


def dialogue_extraction(video_description):
    prompt_template = """You are a highly skilled assistant specializing in extracting conversational dialogue from text.
Your task is to carefully analyze the given description of a video and accurately identify and extract all dialogue content within it.
Please directly output the dialogue in the following format without adding any other content. If no dialogue is present, state: "None."

Dialogue format:
Speaker A Description: "Dialogue from speaker A."
Speaker B Description: "Dialogue from speaker B."
Speaker A Description: "Further dialogue..."

Guidelines:
- The speaker descriptions (e.g., "person in red dress") must be derived from the video description.
- Ensure the descriptions are **accurate and distinguishable first**, and **concise** second.
- - The same speaker must always use exactly the same description, and different speakers must have different descriptions.
- Only include dialogue content; do not infer or add lines not explicitly present.

Video description:
{}
"""
    prompt = prompt_template.format(video_description)

    try:
        caption, thinking = generate(prompt)
        conversations = []
        if caption and len(caption) > 0:
            stripped_caption = caption.strip()
            if stripped_caption.startswith("None"):
                conversations = []
            else:
                conversations = stripped_caption.split("\n")
        return conversations
    except Exception:
        traceback.print_exc()
        return None


def eval_one_sample(video_id, video_dir, dt_list, gt_list):
    filename = os.path.join(video_dir, video_id + '.mp4')

    dt_list = clean_conversations(dt_list)
    gt_list = clean_conversations(gt_list)

    dt_num = len(dt_list)
    gt_num = len(gt_list)

    if dt_num == 0 or gt_num == 0:
        return gt_num, dt_num, []

    max_recall_num, indices = flexible_longest_common_subsequence(dt_list, gt_list)

    for item in indices:
        dt_num = dt_num - (len(item["dt_indices"]) - 1)
        gt_num = gt_num - (len(item["gt_indices"]) - 1)

    if len(indices) == 0:
        return gt_num, dt_num, []

    prompt = prepare_prompt(dt_list, gt_list, indices)
    answer, thinking = generate_mm(filename, prompt)

    if answer is None:
        print("answer is None")
        return None, None, None

    answer = answer.lower().replace("\n", " ").strip()
    answer_list = answer.split(" ")

    if len(answer_list) != len(indices):
        print("len(answer_list) != len(indices)")
        return None, None, None

    reference_correct_score = 0
    for judgement in answer_list:
        if judgement.startswith("yes"):
            reference_correct_score += 1

    matched_results = []
    for idx, item in enumerate(indices):
        dt_str = " ".join([dt_list[i][1] for i in item["dt_indices"]]).strip()
        gt_str = " ".join([gt_list[i][1] for i in item["gt_indices"]]).strip()
        
        sbj_judge = answer_list[idx]

        this_pair = {
            "dt": dt_str,
            "gt": gt_str,
            "edit_similarity_score": item["score"],
            "sbj_judge": sbj_judge,
        }
        matched_results.append(this_pair)

    return gt_num, dt_num, matched_results