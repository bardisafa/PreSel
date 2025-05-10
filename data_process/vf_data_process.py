import os
import json
from collections import defaultdict

def separate_instructions_by_task(instruction_data_path, metadata_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata and instruction data
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    with open(instruction_data_path, "r") as f:
        instruction_data = json.load(f)
    
    # Create a dictionary to store tasks based on task_name
    task_dict = defaultdict(list)
    
    # Map metadata by id for easy lookup
    metadata_map = {item['id']: item for item in metadata}
    
    # Loop through instruction data and categorize them by task_name
    for instruction in instruction_data:
        task_name = instruction.get('task_name')
        if task_name:
            task_dict[task_name].append(instruction)
    #save a list with all task_names
    task_names = list(task_dict.keys())
    with open(os.path.join(output_dir, "task_names.json"), "w") as f:
        json.dump(task_names, f, indent=4)
   
    # Print the number of unique tasks
    print(f"Total number of tasks: {len(task_dict)}")

    # # Save each task's instructions into separate JSON files
    # for task_name, task_samples in task_dict.items():
    #     task_file_path = os.path.join(output_dir, f"vf_{task_name}_data.json")
    #     with open(task_file_path, "w") as f:
    #         json.dump(task_samples, f, indent=4)
    #     print(f"Saved {len(task_samples)} samples for task '{task_name}' to '{task_file_path}'")

if __name__ == "__main__":
    # Define paths
    instruction_data_path = "/data/01/sylo/mnt_02/data/annotation_191-task_1k_add_idx.json"
    metadata_path = "/data/01/sylo/mnt_02/data/metadata.json"
    output_dir = "/data/01/sylo/mnt_02/data/vf_tasks/"
    
    # Call the function to separate instructions by task
    separate_instructions_by_task(instruction_data_path, metadata_path, output_dir)
