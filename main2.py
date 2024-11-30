print("Hello Nebula")

import logging
logging.basicConfig(level=logging.INFO)

import os
import sys
sys.path.extend([".", ".."])
from nebula import Nebula
import json
# from nebula.misc import fix_random_seed
# fix_random_seed(0)

TOKENIZER = "whitespace"
nebula = Nebula(
    vocab_size = 50000,
    seq_len = 512,
    tokenizer = TOKENIZER,
)

# 0. EMULATE IT: SKIP IF YOU HAVE JSON REPORT ALREADY
# PE = r"D:\spl3\speakeasy\data\codeblocks-20.03-setup.exe"
# report = nebula.dynamic_analysis_pe_file(PE)
# with open("reportCodeBlock.json", "w") as file:
#     json.dump(report, file, indent=4)

# Define the folder path
folder_path = r"D:\spl3\dataset\trainset\report_ransomware"

# List all files in the folder and filter for JSON files
json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

# Sort and select the first 10 files
first_10_files = json_files[:50]

# Process each file
for file_name in first_10_files:
    # Construct the full file path
    report_path = os.path.join(folder_path, file_name)

    # Preprocess and predict probability
    x_arr = nebula.preprocess(report_path)
    prob = nebula.predict_proba(x_arr)

    # Print the probability
    print(f"\n[!!!] Probability of {file_name} being malicious: {prob:.3f}")
