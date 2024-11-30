import os
import subprocess
import json

# Directories for input and output files
data = 'D:\\spl3\\speakeasy\\data'
txt_reports = 'D:\\spl3\\speakeasy\\reports\\txt'
json_reports = 'D:\\spl3\\speakeasy\\reports\\json'

# Ensure output folders exist
if not os.path.exists(txt_reports):
    os.makedirs(txt_reports)
if not os.path.exists(json_reports):
    os.makedirs(json_reports)


# Function to parse necessary fields from plain text report and reformat JSON
def parse_speakeasy_output(report_path):
    # Load the .txt report JSON data
    with open(report_path, 'r') as file:
        data = json.load(file)

    # Extract the "sha256" field for naming the JSON report
    file_hash = data.get("sha256", "unknown_hash")

    # Extract and reformat entry points to match the desired output structure
    formatted_entry_points = [
        {
            "ep_type": entry["ep_type"],
            "start_addr": entry["start_addr"],
            "ep_args": entry["ep_args"],
            "apihash": entry["apihash"],
            "apis": [
                {
                    "pc": api.get("pc"),
                    "api_name": api.get("api_name"),
                    "args": api.get("args"),
                    "ret_val": api.get("ret_val")
                } for api in entry.get("apis", [])
            ],
            "ret_val": entry.get("ret_val"),
            "error": entry.get("error", {}),
            "dynamic_code_segments": entry.get("dynamic_code_segments", [])
        } for entry in data.get("entry_points", [])
    ]

    # Return the reformatted data and file hash
    return formatted_entry_points, file_hash


# Loop through all files in the folder
for filename in os.listdir(data):
    if filename.endswith(".exe"):
        print(f"started: {filename}")
        exe_path = os.path.join(data, filename)

        # Set report paths based on file name
        txt_report_name = f"{filename[:-4]}.txt"
        txt_report_path = os.path.join(txt_reports, txt_report_name)

        # Run Speakeasy emulator to generate plain text output
        command = ['python', '-m', 'speakeasy', '-t', exe_path, '-o', txt_report_path]
        try:
            subprocess.run(command, check=True)

            # Parse the text report into JSON format and get the file hash from the "sha256" field
            parsed_data, file_hash = parse_speakeasy_output(txt_report_path)
            print("Speakeasy processing complete.")

            # Save the parsed JSON report with the hash from the "sha256" field as filename
            json_report_path = os.path.join(json_reports, f"{file_hash}.json")
            with open(json_report_path, 'w') as f:
                json.dump(parsed_data, f, indent=4)

            print(f"Processed report generated: {json_report_path}")

        except subprocess.CalledProcessError as e:
            print(f"Error processing {filename}: {e}")
