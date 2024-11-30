import json

# Load the JSON data from the file
with open("reports/qbittorrent.txt", "r") as file:
    data = json.load(file)

# Extract and reformat entry points
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
    } for entry in data["entry_points"]
]

# Save the reformatted data to a new JSON file
with open("formatted_entry_points.json", "w") as output_file:
    json.dump(formatted_entry_points, output_file, indent=4)

print("Formatted data has been saved to 'formatted_entry_points.json'")
