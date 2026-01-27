import json
import sys
from collections import defaultdict

class MetricStats:
    def __init__(self):
        self.count = 0
        self.ref_rec_sum = 0.0
        self.ref_pre_sum = 0.0
        self.asr_rec_sum = 0.0
        self.asr_pre_sum = 0.0

    def add(self, ref_rec, ref_pre, asr_rec, asr_pre):
        self.count += 1
        self.ref_rec_sum += ref_rec
        self.ref_pre_sum += ref_pre
        self.asr_rec_sum += asr_rec
        self.asr_pre_sum += asr_pre

    def get_metrics_str(self):
        if self.count == 0:
            return "- / -"
        
        # Cal avg Recall and Precision
        avg_ref_rec = self.ref_rec_sum / self.count
        avg_ref_pre = self.ref_pre_sum / self.count
        avg_asr_rec = self.asr_rec_sum / self.count
        avg_asr_pre = self.asr_pre_sum / self.count

        # Cal F1
        denom_ref = avg_ref_rec + avg_ref_pre
        ref_f1 = (2 * avg_ref_rec * avg_ref_pre / denom_ref) if denom_ref > 0 else 0.0

        denom_asr = avg_asr_rec + avg_asr_pre
        asr_f1 = (2 * avg_asr_rec * avg_asr_pre / denom_asr) if denom_asr > 0 else 0.0

        return f"{ref_f1 * 100:.1f} / {asr_f1 * 100:.1f}"

# stats[shot_type][category][bucket] -> MetricStats
# shot_type: "singleshot", "multishot"
# category: "main", "speaker_num", "people_num", etc.
stats = {
    "singleshot": defaultdict(lambda: defaultdict(MetricStats)),
    "multishot": defaultdict(lambda: defaultdict(MetricStats))
}
global_overall = MetricStats()

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("matched_results") is None:
                continue

            if data.get("is_single_shot") == 1:
                shot_type = "singleshot"
            else:
                shot_type = "multishot"

            cur_ref_cnt = 0.0
            cur_asr_score = 0.0
            for matched_pairs in data["matched_results"]:
                cur_asr_score += matched_pairs["edit_similarity_score"]
                if matched_pairs["sbj_judge"].startswith("yes"):
                    cur_ref_cnt += 1

            gt_num = data["gt_num"]
            dt_num = data["dt_num"]

            cur_ref_rec = cur_ref_cnt / gt_num if gt_num > 0 else 1.0
            cur_ref_pre = cur_ref_cnt / dt_num if dt_num > 0 else 1.0
            cur_asr_rec = cur_asr_score / gt_num if gt_num > 0 else 1.0
            cur_asr_pre = cur_asr_score / dt_num if dt_num > 0 else 1.0

            global_overall.add(cur_ref_rec, cur_ref_pre, cur_asr_rec, cur_asr_pre)
            stats[shot_type]["overall"]["All"].add(cur_ref_rec, cur_ref_pre, cur_asr_rec, cur_asr_pre)

            spk_num = data["speaker_num"]
            if spk_num == 0: # 0 for Overlap
                spk_bucket = "Overlap"
            elif spk_num == 1:
                spk_bucket = "N=1"
            elif spk_num == 2:
                spk_bucket = "N=2"
            else:
                spk_bucket = "N>=3"
            stats[shot_type]["speaker_num"][spk_bucket].add(cur_ref_rec, cur_ref_pre, cur_asr_rec, cur_asr_pre)
            stats[shot_type]["speaker_num"]["All"].add(cur_ref_rec, cur_ref_pre, cur_asr_rec, cur_asr_pre)

            ppl_num = data["people_num_in_screen"]
            if ppl_num == 1:
                ppl_bucket = "N=1"
            elif ppl_num == 2:
                ppl_bucket = "N=2"
            else:
                ppl_bucket = "N>=3"
            stats[shot_type]["people_num"][ppl_bucket].add(cur_ref_rec, cur_ref_pre, cur_asr_rec, cur_asr_pre)
            stats[shot_type]["people_num"]["All"].add(cur_ref_rec, cur_ref_pre, cur_asr_rec, cur_asr_pre)

            bool_keys = ["has_off_screen_speaker", "has_same_sex_speakers", "only_opposite_sex_speakers"]
            for key in bool_keys:
                val = data.get(key)
                bucket_name = "Yes" if val == 1 else "No"
                stats[shot_type][key][bucket_name].add(cur_ref_rec, cur_ref_pre, cur_asr_rec, cur_asr_pre)
                stats[shot_type][key]["All"].add(cur_ref_rec, cur_ref_pre, cur_asr_rec, cur_asr_pre)


def print_formatted_table(title, category_key, buckets):
    col_width = 13 
    
    header_top = f"| {'Overall':^18} | "
    header_top += f"{'Single-shot (REF / ASR)':^{len(buckets)*col_width + (len(buckets)-1)*3}} | "
    header_top += f"{'Multi-shot (REF / ASR)':^{len(buckets)*col_width + (len(buckets)-1)*3}} | "
    
    header_bot = f"| {'(REF / ASR)':^18} |"
    
    row_sub = ""
    for _ in range(2): # Single and Multi
        row_sub += " " + " | ".join([f"{b:^{col_width}}" for b in buckets])
        row_sub += " |"
    
    header_bot += row_sub

    sep_line = "-" * len(header_bot)

    ss_data = []
    for b in buckets:
        ss_data.append(stats["singleshot"][category_key][b].get_metrics_str())
        
    ms_data = []
    for b in buckets:
        ms_data.append(stats["multishot"][category_key][b].get_metrics_str())
        
    overall_str = global_overall.get_metrics_str()

    data_row = f"| {overall_str:^18} |"
    for item in ss_data:
        data_row += f" {item:^{col_width}} |"
    for item in ms_data:
        data_row += f" {item:^{col_width}} |"

    output = []
    output.append("")
    output.append(f"=== {title} Table ===")
    output.append(sep_line)
    output.append(header_top)
    output.append(sep_line)
    output.append(header_bot)
    output.append(sep_line)
    output.append(data_row)
    output.append(sep_line)
    return "\n".join(output)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <result_jsonl_path>")
        sys.exit(1)

    result_path = sys.argv[1]
    log_path = result_path.replace(".jsonl", ".log")

    process_file(result_path)

    output_lines = []

    # 1. Speaker Num Table
    buckets_spk = ["All", "N=1", "N=2", "N>=3", "Overlap"]
    output_lines.append(print_formatted_table("Speaker Num", "speaker_num", buckets_spk))

    # 2. People Num Table
    buckets_ppl = ["All", "N=1", "N=2", "N>=3"]
    output_lines.append(print_formatted_table("People Num", "people_num", buckets_ppl))

    # 3. Has Off-screen Speaker Table
    buckets_bool = ["All", "Yes", "No"]
    output_lines.append(print_formatted_table("Off-screen Spk", "has_off_screen_speaker", buckets_bool))

    # 4. Same Sex Speakers Table
    output_lines.append(print_formatted_table("Have Same Sex Spk", "has_same_sex_speakers", buckets_bool))

    # 5. Only Opposite Sex Table
    output_lines.append(print_formatted_table("Only Opposite Sex Spk", "only_opposite_sex_speakers", buckets_bool))

    final_output = "\n".join(output_lines)
    
    print(final_output)
    
    with open(log_path, 'w', encoding='utf-8') as f_log:
        f_log.write(final_output)
        f_log.write("\n")