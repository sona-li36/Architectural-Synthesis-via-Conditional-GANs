import os
from dotenv import load_dotenv

# Find the absolute path to the directory where this script lives, 
# then go one level up ('..') to grab the root .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
root_env_path = os.path.join(script_dir, '..', '.env')

# Load the root environment configuration safely
load_dotenv(dotenv_path=root_env_path)

# Now your relative path variables will work perfectly!
dataset_path = os.getenv("DATASET_PATH", "../data/dataset")
print(f"Executing script using dataset at: {dataset_path}")