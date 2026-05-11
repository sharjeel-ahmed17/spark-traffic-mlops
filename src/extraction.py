import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

# 1. Project folder set
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "data/raw")
os.makedirs(data_dir, exist_ok=True)
print("Using project folder:", base_dir)

# 2. Dataset download as pandas
print("\nDownloading Telco Churn dataset...")
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "fedesoriano/traffic-prediction-dataset",
    "traffic.csv"  
)
print("Downloaded! Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 records:")
print(df.head())

# 3. Save data to  project folder 
output_path = os.path.join(data_dir, "data.csv")
df.to_csv(output_path, index=False)
print("\nData saved at:", output_path)

# 4. Final check
print("\nFinal files in data/raw:")
print(os.listdir(data_dir))