print("Hello Nebula")

import logging
logging.basicConfig(level=logging.INFO)

import sys
sys.path.extend([".", ".."])
from nebula import Nebula



# 0. EMULATE IT: SKIP IF YOU HAVE JSON REPORT ALREADY
PE = r"D:\spl3\speakeasy\data\Postman-win64-Setup.exe"
# PE = r"D:\spl3\speakeasy\data\Postman-win64-Setup.exe"
report = nebula.dynamic_analysis_pe_file(PE)
print("[!] Emulation Done")
# 1. PREPROCESS EMULATED JSON REPORT AS ARRAY
x_arr = nebula.preprocess(report)
print("[!] Normalization & Tokenization Done")
print(x_arr)

#TODO: model training
#TODO: prediction
#TODO: Report Generation
