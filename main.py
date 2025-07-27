from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import pandas as pd
import os
import asyncio # For scheduling
import httpx # For making HTTP requests to self
import time # For logging timestamps

app = FastAPI()

# --- Configuration for Hailing/Heartbeat ---
APP_BASE_URL = "https://pref-list-new-ztx6.onrender.com"
HAILING_ENDPOINT = "/hailing"
HEARTBEAT_INTERVAL_SECONDS = 14 * 60 # 14 minutes

origins = ["*"] # Consider restricting this in production

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Original Configuration ---
EXCEL_FILE_PATH = "final_cutoff.xlsx"
MIN_PREFERENCES = 25 # Set to 25 as requested

CATEGORY_RANGES = {
    "GENERAL": 10, "OBC": 15, "EWS": 15, "VJ": 20, "NT": 20,
    "DT": 20, "SC": 25, "ST": 25, "TFWS": 5
}
DEFAULT_RANGE = 15

# --- Data Loading ---
def load_and_clean_data_excel(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        print(f"FATAL: Excel file not found at path: {file_path}")
        raise HTTPException(status_code=500, detail="Excel file with cutoff data not found.")

    try:
        print(f"Loading data from Excel file: {file_path}")
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"FATAL: Error loading Excel file: {e}")
        raise HTTPException(status_code=500, detail="Failed to read Excel file.")

    # Cleaning
    df['Cutoff'] = pd.to_numeric(df['Cutoff'], errors='coerce')
    df.dropna(subset=['Cutoff'], inplace=True)

    if 'Place' in df.columns:
        df['Place_clean'] = df['Place'].astype(str).str.strip().str.lower()
    else:
        df['Place_clean'] = None

    if 'Branch' in df.columns:
        df['Branch'] = df['Branch'].astype(str).str.strip()
    else:
        df['Branch'] = None

    required_cols = ['College Code', 'College Name', 'Choice Code', 'Branch', 'Cutoff', 'Place']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"FATAL: Missing required columns in dataset: {missing_cols}")
        raise HTTPException(status_code=500, detail=f"Dataset is missing required columns: {missing_cols}")

    return df

try:
    cutoff_df_global = load_and_clean_data_excel(EXCEL_FILE_PATH)
except HTTPException as e:
    print(f"Application startup failed: {e.detail}")
    cutoff_df_global = pd.DataFrame()

# --- Helper Function for Filtering ---
# Added upper_bound_cutoff parameter
def filter_dataframe(df: pd.DataFrame, places: List[str], branches: List[str], category: str, percentile: float, use_place_filter: bool, lower_bound_cutoff: float, upper_bound_cutoff: float) -> pd.DataFrame:
    branch_mask = df['Branch'].isin(branches)
    cutoff_mask = df['Cutoff'].between(lower_bound_cutoff, upper_bound_cutoff)
    combined_mask = branch_mask & cutoff_mask

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
    if cutoff_df_global.empty:
        raise HTTPException(status_code=503, detail="College data is currently unavailable. Please try again later.")
    
    df = cutoff_df_global.copy()
    
    # Initial calculation of range and bounds
    range_val = CATEGORY_RANGES.get(category.upper(), DEFAULT_RANGE)
    current_lower_bound = max(0, percentile - range_val)
    upper_bound_cutoff = min(percentile + range_val, 100) # Upper bound remains percentile + range_val

    filtered = pd.DataFrame() # Initialize filtered DataFrame
    attempts = 0
    final_lower_bound_used = current_lower_bound # Track the final lower bound applied
    
    # Loop to expand lower bound if results are too few
    while len(filtered) < MIN_PREFERENCES and final_lower_bound_used >= 0:
        attempts += 1
        
        # Use the current_lower_bound for filtering in each iteration
        filtered = filter_dataframe(
            df, 
            places, 
            branches, 
            category, 
            percentile, 
            use_place_filter=True, 
            lower_bound_cutoff=final_lower_bound_used, # Use the current iteration's lower bound
            upper_bound_cutoff=upper_bound_cutoff # Pass upper bound
        )
        
        # If we have enough preferences, or lower_bound is already 0, break
        if len(filtered) >= MIN_PREFERENCES or final_lower_bound_used == 0:
            break
            
        # If not enough, reduce lower_bound by 10 for the next iteration
        final_lower_bound_used = max(0, final_lower_bound_used - 10)
        print(f"Attempt {attempts}: Found {len(filtered)} colleges. Reducing lower bound to {final_lower_bound_used:.2f} to find more.")

    if not filtered.empty:
        # Sorting and finalizing results...
        filtered = filtered.copy()
        filtered.loc[:, 'MatchScore'] = abs(filtered['Cutoff'] - percentile)
        filtered = filtered.sort_values(by=['Cutoff', 'MatchScore'], ascending=[False, True])
        selected_cols = ['College Code', 'College Name', 'Choice Code', 'Branch', 'Place', 'Cutoff']
        existing_selected_cols = [col for col in selected_cols if col in filtered.columns]
        filtered = filtered[existing_selected_cols]
    else:
        print(f"Even after expanding range, no colleges found matching criteria. Final lower bound: {final_lower_bound_used:.2f}")

    return {
        "query": {"percentile": percentile, "category": category, "branches": branches, "places": places,},
        "filter_level_applied": f"Expanded (final lower bound: {final_lower_bound_used:.2f})" if attempts > 1 else "Initial",
        "range_value_used": range_val,
        "final_lower_bound_cutoff": final_lower_bound_used, # Include final lower bound
        "total_preferences_found": len(filtered),
        "unique_colleges_found": filtered['College Code'].nunique() if not filtered.empty else 0,
        "preferences": filtered.to_dict(orient='records') if not filtered.empty else []
    }

@app.get("/all-data")
def get_all_data():
    if cutoff_df_global.empty:
        raise HTTPException(status_code=503, detail="College data is currently unavailable.")
    return cutoff_df_global.head(100).to_dict(orient='records')

@app.get("/health")
def health_check():
    if cutoff_df_global is not None and not cutoff_df_global.empty:
        return {"status": "OK", "message": "Service is running and data is loaded."}
    else:
        raise HTTPException(status_code=503, detail="Service is running but college data is not loaded.")

# --- NEW: Hailing (Keep-Alive) Endpoint ---
@app.get(HAILING_ENDPOINT)
async def hailing_route():
    """
    Simple endpoint to keep the server alive on free hosting tiers.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    # print(f"Hailing endpoint hit at {timestamp}")
    return {"status": "OK", "message": "Server is awake and responsive.", "timestamp": timestamp}

# --- NEW: Background Task for Heartbeat ---
async def heartbeat_task():
    """
    Periodically calls the hailing endpoint to keep the server alive.
    """
    # Wait a bit for the server to fully start before the first heartbeat
    await asyncio.sleep(10) 
    
    full_hailing_url = f"{APP_BASE_URL}{HAILING_ENDPOINT}"
    print(f"Heartbeat task started. Will ping {full_hailing_url} every {HEARTBEAT_INTERVAL_SECONDS} seconds.")
    
    async with httpx.AsyncClient() as client:
        while True:
            try:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
                # print(f"Sending heartbeat to {full_hailing_url} at {timestamp}...")
                response = await client.get(full_hailing_url, timeout=10.0) # Added timeout
                response.raise_for_status() # Raise an exception for bad status codes
                print(f"Heartbeat success: Status {response.status_code} - {response.json().get('message', 'OK')}")
            except httpx.RequestError as exc:
                print(f"Heartbeat Error: An error occurred while requesting {exc.request.url!r} - {exc}")
            except httpx.HTTPStatusError as exc:
                print(f"Heartbeat Error: Status {exc.response.status_code} while requesting {exc.request.url!r} - Response: {exc.response.text}")
            except Exception as e:
                print(f"Heartbeat Error: An unexpected error occurred: {e}")
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

# --- NEW: FastAPI Startup Event to run the heartbeat task ---
@app.on_event("startup")
async def startup_event():
    """
    Actions to perform on application startup.
    """
    print("Application startup: Initializing background tasks...")
    # Create a background task for the heartbeat
    # This task will run in the background without blocking the main application
    asyncio.create_task(heartbeat_task())
    print("Heartbeat task scheduled.")

# --- Run with Uvicorn (for local testing) ---
# if __name__ == "__main__":
#     import uvicorn
#     print("Starting FastAPI server with Uvicorn for local development...")
#     # Ensure the APP_BASE_URL is correct for local testing if you rely on it for the heartbeat
#     # For local, if uvicorn runs on 8000, APP_BASE_URL should be http://localhost:8000
#     uvicorn.run("your_filename:app", host="0.0.0.0", port=8000, reload=True)
#     # Replace "your_filename" with the actual name of your Python file (e.g., main.py -> "main:app")
