import kagglehub

# Download latest version
path = kagglehub.dataset_download("orosas/marketing-mix-dataset")

print("Path to dataset files:", path)