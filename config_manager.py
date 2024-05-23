import yaml
import os
# Global variable to store configuration settings
CONFIG = None

# Function to load configuration settings from the YAML file
def load_config():
    global CONFIG
    file_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(file_path, 'r') as file:
        CONFIG = yaml.safe_load(file)
    return CONFIG

config = load_config()

# Access configuration settings throughout your application
def get_model_path():
    return CONFIG['paths']['model_path']

def get_candidate_repository_path():
     path=CONFIG['paths']['reference_image_repository']
     print(path)
     return path

def get_voting_repository_path():
    path= CONFIG['paths']['voting_repository']
    print(path)
    return path

# Example usage
if __name__ == "__main__":

    print("App Name:", get_model_path())
    print("Logging Level:", get_directory_path())
