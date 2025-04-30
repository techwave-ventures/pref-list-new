from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import pandas as pd
import os # To check if file exists locally first (optional)

app = FastAPI()



origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specified origins
    allow_credentials=True, # Allows cookies/authorization headers
    allow_methods=["*"],    # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allows all headers
)



# --- Configuration ---
# Use environment variable or fallback for flexibility
CSV_URL = os.getenv("CUTOFF_CSV_URL", "https://drive.google.com/uc?export=download&id=1AkIPPpu1XGXhBleR-x1GFLpENISIGuFm")
LOCAL_CSV_PATH = "cutoff_data.csv" # Optional local cache path
MIN_PREFERENCES = 100 # Target minimum number of preferences

# Category-based range logic (percentile points above/below)
CATEGORY_RANGES = {
    "GENERAL": 10, # Using uppercase keys for consistency
    "OBC": 15,
    "EWS": 15,
    "VJ": 20,
    "NT": 20,
    "DT": 20,
    "SC": 25,
    "ST": 25,
    "TFWS": 5 # Example: TFWS might have a smaller range
    # Add other categories as needed
}
DEFAULT_RANGE = 15 # Fallback range if category not found

# --- Data Loading ---
def load_and_clean_data(url: str, local_path: str) -> pd.DataFrame:
    """Loads data from URL or local cache and performs cleaning."""
    df = None
    # Optional: Try loading from local cache first
    if os.path.exists(local_path):
        try:
            print(f"Loading data from local cache: {local_path}")
            df = pd.read_csv(local_path)
        except Exception as e:
            print(f"Error loading local cache {local_path}: {e}. Fetching from URL.")
            df = None # Ensure df is None if local load fails

    if df is None: # If not loaded locally or local load failed
        try:
            print(f"Fetching data from URL: {url}")
            df = pd.read_csv(url)
            # Optional: Save to local cache after fetching
            try:
                df.to_csv(local_path, index=False)
                print(f"Data cached locally to {local_path}")
            except Exception as e:
                print(f"Warning: Could not cache data locally: {e}")
        except Exception as e:
            print(f"FATAL: Error fetching data from URL {url}: {e}")
            # Raise an exception or handle appropriately if data loading fails
            raise HTTPException(status_code=503, detail="Could not load necessary college data.")

    # --- Data Cleaning ---
    print("Cleaning data...")
    # Ensure 'Cutoff' is numeric, coercing errors to NaN
    df['Cutoff'] = pd.to_numeric(df['Cutoff'], errors='coerce')
    # Drop rows where Cutoff could not be converted (became NaN)
    df.dropna(subset=['Cutoff'], inplace=True)
    # Clean 'Place' column: strip whitespace, convert to lowercase
    if 'Place' in df.columns:
        df['Place_clean'] = df['Place'].str.strip().str.lower()
    else:
        print("Warning: 'Place' column not found in the dataset.")
        df['Place_clean'] = None # Add column with None if it doesn't exist

    # Clean 'Branch' column
    if 'Branch' in df.columns:
         df['Branch'] = df['Branch'].str.strip() # Keep original case for matching? Or lower()? Let's keep original for now.
    else:
         print("Warning: 'Branch' column not found in the dataset.")
         df['Branch'] = None

    # Ensure essential columns exist
    required_cols = ['College Code', 'College Name', 'Choice Code', 'Branch', 'Cutoff', 'Place']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
         print(f"FATAL: Missing required columns in dataset: {missing_cols}")
         raise HTTPException(status_code=500, detail=f"Dataset is missing required columns: {missing_cols}")

    print("Data loaded and cleaned successfully.")
    return df

# Load data globally on startup
try:
    cutoff_df_global = load_and_clean_data(CSV_URL, LOCAL_CSV_PATH)
except HTTPException as e:
    # If loading fails critically on startup, maybe exit or log severely
    print(f"Application startup failed: {e.detail}")
    # Depending on deployment, might need to exit or let it fail to start
    cutoff_df_global = pd.DataFrame() # Assign empty dataframe to prevent further errors if needed


# --- Helper Function for Filtering ---
def filter_dataframe(df: pd.DataFrame, places: List[str], branches: List[str], category: str, percentile: float, use_place_filter: bool, lower_bound_cutoff: float) -> pd.DataFrame:
    """Applies filters to the dataframe."""
    # Get the range based on category, default if not found
    range_val = CATEGORY_RANGES.get(category.upper(), DEFAULT_RANGE)
    upper_bound_cutoff = min(percentile + range_val, 100) # Ensure upper bound doesn't exceed 100

    # Start with branch filter (always applied)
    branch_mask = df['Branch'].isin(branches)
    # Cutoff filter
    cutoff_mask = df['Cutoff'].between(lower_bound_cutoff, upper_bound_cutoff)

    # Combine masks
    combined_mask = branch_mask & cutoff_mask

    # Apply place filter conditionally
    if use_place_filter and 'Place_clean' in df.columns and df['Place_clean'].notna().all():
        places_lower = [p.strip().lower() for p in places]
        place_mask = df['Place_clean'].isin(places_lower)
        combined_mask &= place_mask
    elif use_place_filter:
         print("Warning: Place filter requested but 'Place_clean' column is missing or invalid. Skipping place filter.")


    return df[combined_mask]

# --- API Endpoint ---
@app.get("/preference-list")
def get_preference_list(
    places: List[str] = Query(..., description="List of preferred places/cities.", min_length=1),
    percentile: float = Query(..., ge=0, le=100, description="MHT-CET Percentile (0-100)."),
    branches: List[str] = Query(..., description="List of preferred branches.", min_length=1),
    category: str = Query("General", description="Admission category (e.g., General, OBC, SC, ST, EWS, TFWS).")
) -> Dict[str, Any]:
    """
    Generates a ranked preference list of colleges based on user inputs.
    It attempts to return at least 100 preferences by relaxing filters if necessary.
    """
    if cutoff_df_global.empty:
         raise HTTPException(status_code=503, detail="College data is currently unavailable. Please try again later.")

    df = cutoff_df_global.copy() # Work on a copy
    final_filter_level = "Initial"

    # --- Step 1: Initial Filter ---
    print("Step 1: Applying initial filters...")
    range_val = CATEGORY_RANGES.get(category.upper(), DEFAULT_RANGE)
    lower_bound = max(0, percentile - range_val) # Ensure lower bound is not negative
    filtered = filter_dataframe(df, places, branches, category, percentile, use_place_filter=True, lower_bound_cutoff=lower_bound)
    print(f"Step 1 Results: {len(filtered)} entries")

    # --- Step 2: Widen Cutoff Range (Lower Bound = 0) ---
    if len(filtered) < MIN_PREFERENCES:
        print(f"Step 2: Less than {MIN_PREFERENCES} results. Widening cutoff range [0, {percentile + range_val}]...")
        final_filter_level = "Widened Cutoff"
        lower_bound = 0 # Set lower bound to 0
        filtered = filter_dataframe(df, places, branches, category, percentile, use_place_filter=True, lower_bound_cutoff=lower_bound)
        print(f"Step 2 Results: {len(filtered)} entries")

    # --- Step 3: Remove Place Filter ---
    if len(filtered) < MIN_PREFERENCES:
        print(f"Step 3: Still less than {MIN_PREFERENCES} results. Removing place filter...")
        final_filter_level = "Removed Place Filter"
        lower_bound = 0 # Keep lower bound at 0
        filtered = filter_dataframe(df, places, branches, category, percentile, use_place_filter=False, lower_bound_cutoff=lower_bound)
        print(f"Step 3 Results: {len(filtered)} entries")


    # --- Final Processing (Sorting, Column Selection) ---
    if not filtered.empty:
        print("Sorting and finalizing results...")
        # Compute match score based on the *original* percentile for better ranking
        # Use .loc to avoid SettingWithCopyWarning
        filtered = filtered.copy() # Ensure we work on a copy before modification
        filtered.loc[:, 'MatchScore'] = abs(filtered['Cutoff'] - percentile)

        # Sort: Primarily by Cutoff (descending), secondarily by MatchScore (ascending - closer is better)
        filtered = filtered.sort_values(by=['Cutoff', 'MatchScore'], ascending=[False, True])

        # Select and reorder essential columns for the output
        # Include 'Place' in the final output
        selected_cols = ['College Code', 'College Name', 'Choice Code', 'Branch', 'Place', 'Cutoff']
        # Ensure all selected columns exist before selecting
        existing_selected_cols = [col for col in selected_cols if col in filtered.columns]
        filtered = filtered[existing_selected_cols]
    else:
         print("No colleges found matching any filter criteria.")


    return {
        "query": {
            "percentile": percentile,
            "category": category,
            "branches": branches,
            "places": places,
        },
        "filter_level_applied": final_filter_level, # Indicate how relaxed the filter was
        "range_value_used": range_val, # Show the +/- range used for the category
        "total_preferences_found": len(filtered),
        "unique_colleges_found": filtered['College Code'].nunique() if not filtered.empty else 0,
        "preferences": filtered.to_dict(orient='records') if not filtered.empty else []
    }

# Optional: Endpoint to get all raw data (useful for debugging)
@app.get("/all-data")
def get_all_data():
    if cutoff_df_global.empty:
         raise HTTPException(status_code=503, detail="College data is currently unavailable.")
    # Return a limited number of rows for preview if needed
    return cutoff_df_global.head(100).to_dict(orient='records')

# --- Health Check Endpoint ---
@app.get("/health")
def health_check():
    # Basic check: is the dataframe loaded?
    if cutoff_df_global is not None and not cutoff_df_global.empty:
        return {"status": "OK", "message": "Service is running and data is loaded."}
    else:
        # Use 503 Service Unavailable if data isn't ready
        raise HTTPException(status_code=503, detail="Service is running but college data is not loaded.")

# --- Run with Uvicorn (for local testing) ---
# if __name__ == "__main__":
#     import uvicorn
#     print("Starting FastAPI server with Uvicorn...")
#     uvicorn.run(app, host="0.0.0.0", port=8000)
#     # To run: uvicorn your_filename:app --reload
