import os
import subprocess

# Set the directory where your exe files are located
data = 'D:\\spl3\\speakeasy\\data'  # e.g., "D:\\s08\\speakeasy\\exefolder"
reports = 'D:\\spl3\\speakeasy\\reports'  # Optional, can save reports to a different folder

# Ensure output folder exists
if not os.path.exists(reports):
    os.makedirs(reports)

# Loop through all files in the folder
for filename in os.listdir(data):
    if filename.endswith(".exe"):
        exe_path = os.path.join(data, filename)
        report_name = f"{os.path.splitext(filename)[0]}.txt"
        report_path = os.path.join(reports, report_name)

        # Run the emulator and generate report
        command = ['python', '-m', 'speakeasy', '-t', exe_path, '-o', report_path]
        try:
            subprocess.run(command, check=True)
            print(f"Report generated: {report_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {filename}: {e}")
