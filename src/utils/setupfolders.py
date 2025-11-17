import os
from pathlib import Path

def create_data_hierarchy(root_dir="src/ARRA_Data_Archive"):
    """
    Creates the recommended folder hierarchy for organizing diverse data.

    Args:
        root_dir (str): The name of the main parent directory.
    """

    # --- 1. Define the complete folder structure ---
    # Each string defines a full path relative to the root_dir
    folder_paths = [
        # 1. SOURCE DATA (Raw Vault)
        "01_Source_Data/01_WebScraping/SourceA_ProjectName/Raw_HTML",
        "01_Source_Data/01_WebScraping/SourceA_ProjectName/Metadata_Logs",
        "01_Source_Data/01_WebScraping/Extracted_Text_Initial",
        "01_Source_Data/02_Internal_Documents/Documents/PDFs",
        "01_Source_Data/02_Internal_Documents/Documents/Word_Excel",
        "01_Source_Data/02_Internal_Documents/Media/Audio_Raw",
        "01_Source_Data/02_Internal_Documents/Media/Images",
        "01_Source_Data/03_Structured_Imports/Database_Dumps",
        "01_Source_Data/03_Structured_Imports/CSVs_TSVs",

        # 2. PROCESSING (The Workshop)
        "02_Processing/01_Cleaning_Stage/WebScrape_To_Clean",
        "02_Processing/01_Cleaning_Stage/Docs_To_Normalize",
        "02_Processing/02_Transformation_Stage/Unstructured_Working",
        "02_Processing/02_Transformation_Stage/Structured_Working",
        "02_Processing/02_Transformation_Stage/Audio_Transcribed",
        "02_Processing/03_Code_Scripts/Cleaning_Scripts",
        "02_Processing/03_Code_Scripts/Analysis_Scripts",
        "02_Processing/03_Code_Scripts/Modeling_Scripts",
        "02_Processing/03_Code_Scripts/Dependencies",

        # 3. FINAL OUTPUT (The Showcase)
        "03_Final_Output/01_Curated_Datasets",
        "03_Final_Output/02_Analysis_Results/Reports",
        "03_Final_Output/02_Analysis_Results/Visualizations",
        "03_Final_Output/02_Analysis_Results/Interactive_Outputs",
        "03_Final_Output/03_Trained_Models",
    ]

    # --- 2. Create the directories ---
    print(f"Starting directory creation in: '{os.path.abspath(root_dir)}'")
    
    # Create the root directory first, using Path for robustness
    root_path = Path(root_dir)
    root_path.mkdir(exist_ok=True) 

    # Loop through the list and create each path
    for relative_path in folder_paths:
        full_path = root_path / relative_path
        
        # os.makedirs creates directories recursively (i.e., all parent folders too)
        # exist_ok=True prevents the script from failing if the directory already exists
        os.makedirs(full_path, exist_ok=True)
        print(f"-> Created: {full_path}")

    print("\n✅ Data hierarchy successfully created!")

# --- Execution ---
if __name__ == "__main__":
    # You can change 'My_Data_Project' to whatever you want the top folder to be called
    create_data_hierarchy(root_dir="My_Data_Project")