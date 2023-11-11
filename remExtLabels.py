import os
import glob
import re

label_folder = "/home/shussain/Downloads/FireSmokeDetection/dataset/txt_labels"

label_files = glob.glob(os.path.join(label_folder, "*.txt"))

# Define a regular expression pattern to match valid YOLO labels with exactly 5 values
label_pattern = re.compile(r'^\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s*$', re.MULTILINE)


for label_file in label_files:
    with open(label_file, 'r') as f:
        content = f.read()

    valid_labels = label_pattern.findall(content)

    # Re-write the label file with only valid lines
    with open(label_file, 'w') as f:
        f.write('\n'.join(valid_labels))

    lines_removed = len(content.splitlines()) - len(valid_labels)
    print(f"Processed: {label_file}, Lines Removed: {lines_removed}")

