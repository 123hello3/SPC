import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import re
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams




NUM_PATTERN = r'(?<!\w)-?(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(?!\w)'
print('---------------  Running inference with vLLM (Batch Mode) ---------------')

# --- Configuration ---
MODEL_PATH = ""  
MAX_LINES = None         
BATCH_SIZE = 32         


def normalize_answer(ans: str):
    ans = str(ans)
    if not ans or not isinstance(ans, str):
        return ""
    original = ans.strip()
    if not original:
        return ""


    answer_tag_match = re.search(r'<answer>(.*?)</answer>', original, re.IGNORECASE | re.DOTALL)
    if answer_tag_match:
        content = answer_tag_match.group(1).strip()
        # num_match = re.search(r'(?<!\w)-?\d+(?:\.\d+)?(?!\w)', content)
        num_match = re.search(NUM_PATTERN, content)
        if num_match:
            return num_match.group(0)
        else:
            fallback = content.strip(' .,;!?')
            return fallback if fallback else ""


    boxed_match = re.search(r'\\boxed\{([^}]*)\}', original)
    if boxed_match:
        content = boxed_match.group(1).strip().replace('$', '').strip()
        # num_match = re.search(r'(?<!\w)-?\d+(?:\.\d+)?(?!\w)', content)
        num_match = re.search(NUM_PATTERN, content)
        if num_match:
            return num_match.group(0)
        elif content:
            return content


    tail = original[-200:] if len(original) > 200 else original
    # num_matches = list(re.finditer(r'(?<!\w)-?\d+(?:\.\d+)?(?!\w)', tail))
    num_matches = list(re.finditer(NUM_PATTERN, tail))
    if num_matches:
        return num_matches[-1].group(0)


    # all_num_matches = list(re.finditer(r'(?<!\w)-?\d+(?:\.\d+)?(?!\w)', original))
    all_num_matches = list(re.finditer(NUM_PATTERN, original))
    if all_num_matches:
        return all_num_matches[-1].group(0)

    tokens = original.split()
    for token in reversed(tokens):
        clean = token.strip('.,;!?":()[]{}\\$')
        if re.fullmatch(r'-?\d+(?:\.\d+)?', clean):
            return clean

    return ""


def extract_reference_answer(ref: str):
    ref = str(ref)
    lines = ref.strip().split('\n')
    for line in reversed(lines):
        if "####" in line:
            return line.split("####")[-1].strip()
    return lines[-1].strip() if lines else ""


# --- Load Model and Tokenizer via vLLM ---
print("Loading model and tokenizer with vLLM...")
llm = LLM(
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    dtype="auto",
    tensor_parallel_size=1,
    max_model_len=4096,
    gpu_memory_utilization=0.70
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("✅ LLM engine ready!")

INSTRUCTION = (
    """Please reason step by step, and put your final answer within \\boxed{}."""
)


def count_existing_results(output_file):
    if not os.path.exists(output_file):
        return 0
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except:
        return 0

def process_jsonl(input_file, benchmark):
    if "/home/UserLangYan/test_verl/model/" in MODEL_PATH:
        model_name = MODEL_PATH.replace("/home/UserLangYan/test_verl/model/", "")
    else:
        choice = input("模型是否为Qwen2.5-1.5B-Instruct（yes）如果是则输入yes，否则输入模型名称：").strip()
        if choice.lower() == "yes":
            model_name = "Qwen2.5-1.5B-Instruct"
        else:
            model_name = choice
    OUTPUT_FILE = f""
    CORRECT_FILE = OUTPUT_FILE.replace(".jsonl", "_correct.jsonl")
    INCORRECT_FILE = OUTPUT_FILE.replace(".jsonl", "_incorrect.jsonl")
       
    output_dir = os.path.dirname(OUTPUT_FILE)
    os.makedirs(output_dir, exist_ok=True)
    existing_count = count_existing_results(OUTPUT_FILE)
    if MAX_LINES is not None and existing_count >= MAX_LINES:
        print(f"Already processed {existing_count} lines (>= MAX_LINES={MAX_LINES}). Nothing to do.")
        return
    elif existing_count > 0:
        print(f"Resuming from line {existing_count + 1}...")

    # Load all lines
    all_lines = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if MAX_LINES is not None and i >= MAX_LINES:
                break
            all_lines.append(line.strip())

    lines_to_process = all_lines[existing_count:]
    total = len(lines_to_process)

    if total == 0:
        print("No new lines to process.")
        return

    # Open output files
    out_main = open(OUTPUT_FILE, 'a', encoding='utf-8')
    out_corr = open(CORRECT_FILE, 'a', encoding='utf-8')
    out_incorr = open(INCORRECT_FILE, 'a', encoding='utf-8')

    pbar = tqdm(total=total, desc="Processing", unit="q")

    # Prepare batched prompts
    batch_prompts = []
    batch_raw_data = []

    def _flush_batch(prompts, raw_datas):
        if not prompts:
            return
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=2048,
            stop_token_ids=[tokenizer.eos_token_id]
        )
        outputs = llm.generate(prompts, sampling_params,use_tqdm=False)

        for j, output in enumerate(outputs):
            data = raw_datas[j]
            question = data.get("question")
            reference_answer = data.get("answer", "")

            llm_response = output.outputs[0].text
            output_token_count = len(output.outputs[0].token_ids)

            model_ans_norm = normalize_answer(llm_response)
            ref_raw = extract_reference_answer(reference_answer)
            ref_ans_norm = normalize_answer(ref_raw)
            is_correct = (model_ans_norm == ref_ans_norm)

            result = {
                "question": question,
                "reference_answer": reference_answer,
                "llm_response": llm_response,
                "model_answer_normalized": model_ans_norm,
                "reference_answer_normalized": ref_ans_norm,
                "is_correct": is_correct,
                "output_token_count": output_token_count,
                "original_data": data
            }

            out_line = json.dumps(result, ensure_ascii=False) + "\n"
            out_main.write(out_line)
            if is_correct:
                out_corr.write(out_line)
            else:
                out_incorr.write(out_line)

        # Flush every batch (optional: reduce frequency if needed)
        out_main.flush()
        out_corr.flush()
        out_incorr.flush()

    # Process in batches
    for line_str in lines_to_process:
        try:
            data = json.loads(line_str)
            question = data.get("question")
            if not question:
                pbar.update(1)
                continue

            prompt = question + " " + INSTRUCTION
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_prompts.append(text)
            batch_raw_data.append(data)

            # If batch is full, process it
            if len(batch_prompts) >= BATCH_SIZE:
                _flush_batch(batch_prompts, batch_raw_data)
                batch_prompts.clear()
                batch_raw_data.clear()
                pbar.update(BATCH_SIZE)

        except Exception as e:
            print(f"\nError processing line: {e}")
            pbar.update(1)
            continue

    # Flush remaining
    if batch_prompts:
        _flush_batch(batch_prompts, batch_raw_data)
        pbar.update(len(batch_prompts))

    pbar.close()
    out_main.close()
    out_corr.close()
    out_incorr.close()

    print(f"\n✅ Done!")
    print(f"Main output: {OUTPUT_FILE}")
    print(f"Correct samples: {CORRECT_FILE}")
    print(f"Incorrect samples: {INCORRECT_FILE}")

    # ========== Generate Report ==========
    total = 0
    correct = 0
    total_tokens = 0
    count_with_tokens = 0

    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                total += 1
                if data.get("is_correct", False):
                    correct += 1
                tok = data.get("output_token_count")
                if isinstance(tok, (int, float)):
                    total_tokens += tok
                    count_with_tokens += 1

    report_path = OUTPUT_FILE.replace(".jsonl", "_report.txt")
    with open(report_path, 'w', encoding='utf-8') as rpt:
        rpt.write("=" * 50 + "\n")
        rpt.write(f"model: {MODEL_PATH}\n")
        rpt.write("📊 INFERENCE EVALUATION REPORT (vLLM - Batch Mode)\n")
        rpt.write("=" * 50 + "\n\n")

        rpt.write(f"Total samples: {total}\n")
        rpt.write(f"Correct:       {correct}\n")
        rpt.write(f"Incorrect:     {total - correct}\n")
        if total > 0:
            acc = correct / total
            rpt.write(f"Accuracy:      {acc:.2%}\n")
        rpt.write("\n")

        rpt.write("🔤 TOKEN USAGE STATISTICS\n")
        rpt.write("-" * 30 + "\n")
        rpt.write(f"Samples with token count: {count_with_tokens}\n")
        rpt.write(f"Total output tokens:      {total_tokens}\n")
        if count_with_tokens > 0:
            avg_tok = total_tokens / count_with_tokens
            rpt.write(f"Average tokens/sample:    {avg_tok:.2f}\n")
        else:
            rpt.write("Average tokens/sample:    N/A\n")
        rpt.write("\n✅ Report generated using vLLM batch inference.\n")

    print(f"📄 Full report saved to: {report_path}")


if __name__ == "__main__":
    for benchmark in ["gsm8k", "aime_2024", "aime_2025", "math_500", "svamp"]:
                    #   ,"amc23","Olympiad"]:
    # for benchmark in ["svamp"]:
        print("="*50)
        print(f"Processing benchmark: {benchmark}")
        print("="*50)
        if benchmark == "gsm8k":
            # input_file = "/home/UserLangYan/test_verl/infer/dataset/gsm8k/main/gsm8k_train_1.jsonl"
            input_file = "/home/gsm8k/main/test-00000-of-00001.jsonl"
        elif benchmark == "aime_2024":
            input_file = "/home/AIME_2024/aime_2024.jsonl"
        elif benchmark == "aime_2025":
            input_file = "/home/AIME_2025/aime2025.jsonl"
        elif benchmark == "math_500":
            input_file = "/home/MATH_500/math_500.jsonl"
        elif benchmark == "svamp":
            input_file = "/home/svamp/svamp.jsonl"
        elif benchmark == "amc23":
            input_file = "/home/amc23/test.jsonl"
        elif benchmark == "Olympiad":
            input_file = "/home/OlympiadBench/test.jsonl"

        process_jsonl(input_file, benchmark)