import os
from pathlib import Path

class DataHierarchyManager:
    """
    Manages the creation and retrieval of paths for a standardized data hierarchy.

    This class ensures a consistent three-stage flow: Source -> Processing -> Final Output.
    """

    def __init__(self, root_dir: str = "My_Data_Project"):
        """
        Initializes the manager, defines the structure, and creates the directories.

        Args:
            root_dir (str): The name of the main parent directory.
        """
        self.root_dir = Path(root_dir)
        # Dictionary mapping simple keys to their relative folder paths
        self.hierarchy_map = self._define_structure()
        self.create_hierarchy()
        print(f"\n✅ Data Hierarchy initialized under: {self.root_dir.resolve()}")


    def _define_structure(self) -> dict:
        """
        Defines the complete folder structure using simple keys.
        """
        return {
            # 1. SOURCE DATA (Keys for raw, untouched inputs)
            "ROOT_SOURCE": "01_Source_Data",
            "RAW_HTML": "01_Source_Data/01_WebScraping/SourceA_ProjectName/Raw_HTML",
            "METADATA_LOGS": "01_Source_Data/01_WebScraping/SourceA_ProjectName/Metadata_Logs",
            "INITIAL_EXTRACTED_TEXT": "01_Source_Data/01_WebScraping/Extracted_Text_Initial",
            "RAW_PDFS": "01_Source_Data/02_Internal_Documents/Documents/PDFs",
            "RAW_WORD_EXCEL": "01_Source_Data/02_Internal_Documents/Documents/Word_Excel",
            "RAW_AUDIO": "01_Source_Data/02_Internal_Documents/Media/Audio_Raw",
            "RAW_IMAGES": "01_Source_Data/02_Internal_Documents/Media/Images",
            "DATABASE_DUMPS": "01_Source_Data/03_Structured_Imports/Database_Dumps",
            "RAW_CSVS": "01_Source_Data/03_Structured_Imports/CSVs_TSVs",

            # 2. PROCESSING (Keys for intermediate, working data)
            "ROOT_PROCESSING": "02_Processing",
            "TO_CLEAN_WEB": "02_Processing/01_Cleaning_Stage/WebScrape_To_Clean",
            "TO_NORMALIZE_DOCS": "02_Processing/01_Cleaning_Stage/Docs_To_Normalize",
            "WORKING_UNSTRUCTURED": "02_Processing/02_Transformation_Stage/Unstructured_Working",
            "WORKING_STRUCTURED": "02_Processing/02_Transformation_Stage/Structured_Working",
            "AUDIO_TRANSCRIBED": "02_Processing/02_Transformation_Stage/Audio_Transcribed",
            "CLEANING_SCRIPTS": "02_Processing/03_Code_Scripts/Cleaning_Scripts",
            "ANALYSIS_SCRIPTS": "02_Processing/03_Code_Scripts/Analysis_Scripts",
            "MODELING_SCRIPTS": "02_Processing/03_Code_Scripts/Modeling_Scripts",
            "DEPENDENCIES": "02_Processing/03_Code_Scripts/Dependencies",

            # 3. FINAL OUTPUT (Keys for finalized results)
            "ROOT_OUTPUT": "03_Final_Output",
            "CURATED_DATASETS": "03_Final_Output/01_Curated_Datasets",
            "FINAL_REPORTS": "03_Final_Output/02_Analysis_Results/Reports",
            "FINAL_VISUALIZATIONS": "03_Final_Output/02_Analysis_Results/Visualizations",
            "INTERACTIVE_OUTPUTS": "03_Final_Output/02_Analysis_Results/Interactive_Outputs",
            "TRAINED_MODELS": "03_Final_Output/03_Trained_Models",
        }


    def create_hierarchy(self):
        """
        Creates all defined directories using the root path.
        """
        print(f"Creating data hierarchy in {self.root_dir}...")
        
        # Ensure the root directory exists
        self.root_dir.mkdir(exist_ok=True, parents=True)

        # Create all sub-directories
        for relative_path in self.hierarchy_map.values():
            full_path = self.root_dir / relative_path
            # parents=True creates intermediate directories; exist_ok=True prevents errors
            os.makedirs(full_path, exist_ok=True)


    def get_path(self, key_name: str) -> Path | None:
        """
        Retrieves the full absolute path for a given directory key.

        Args:
            key_name (str): The simple key name (e.g., 'RAW_HTML', 'FINAL_REPORTS').
                            Case insensitive lookup is performed.

        Returns:
            Path | None: The full absolute Path object, or None if the key is not found.
        """
        normalized_key = key_name.upper()

        if normalized_key in self.hierarchy_map:
            relative_path = self.hierarchy_map[normalized_key]
            # Join the root path with the relative path and resolve to absolute path
            return (self.root_dir / relative_path).resolve()
        else:
            print(f"⚠️ Error: Directory key '{key_name}' not found in hierarchy map.")
            print(f"Available keys: {', '.join(sorted(self.hierarchy_map.keys()))}")
            return None


# --- Example Usage ---
if __name__ == "__main__":
    
    # 1. Initialize the manager (this creates all folders)
    manager = DataHierarchyManager(root_dir="My_Global_Data_Project")

    # 2. Use the get_path method to find required directory locations
    
    # Example 1: Finding the folder for raw audio files
    audio_path = manager.get_path("RAW_AUDIO")
    if audio_path:
        print(f"\nPath for uploading audio: {audio_path}")
        # Use this path in your code to save a file:
        audio_file_path = audio_path / "meeting_transcript_001.mp3"
        print(f"Saving a file to: {audio_file_path}")
        # Note: We won't actually create the file here, just demonstrate the path.
        
    # Example 2: Finding the folder for analysis scripts
    scripts_path = manager.get_path("analysis_scripts")
    if scripts_path:
        print(f"\nPath for storing scripts: {scripts_path}")

    # Example 3: Trying a key that doesn't exist
    manager.get_path("NON_EXISTENT_FOLDER")