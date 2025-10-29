import kagglehub

# Download latest version
path = kagglehub.dataset_download("palbha/cmapss-jet-engine-simulated-data")

print("Path to dataset files:", path)