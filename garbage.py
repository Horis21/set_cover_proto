import os
import shutil

# Remove __pycache__
if os.path.exists("__pycache__"):
    shutil.rmtree("__pycache__")

# Remove log files
for file in os.listdir("."):
    if file.endswith(".log"):
        os.remove(file)

# Clear temp directory
shutil.rmtree("/tmp", ignore_errors=True)