from fastapi import FastAPI, Query
from typing import List
import pandas as pd

app = FastAPI()

# Load and clean the dataset once at startup
cutoff_df = pd.read_csv("https://drive.google.com/uc?export=download&id=1AkIPPpu1XGXhBleR-x1GFLpENISIGuFm")
cutoff_df['Cutoff'] = pd.to_numeric(cutoff_df['Cutoff'], errors='coerce')
cutoff_df['Place_clean'] = cutoff_df['Place'].str.strip().str.lower()

# Category-based range logic
CATEGORY_RANGES = {
    "General": 10,
    "OBC": 15,
    "EWS": 15,
    "VJ": 20,
    "NT": 20,
    "DT": 20,
    "SC": 25,
    "ST": 25
}

@app.get("/preference-list")
def get_preference_list(
    places: List[str] = Query(...),
    percentile: float = Query(..., ge=0, le=100),
    branches: List[str] = Query(...),
    category: str = Query("General")
):
    df = cutoff_df.copy()
    range_val = CATEGORY_RANGES.get(category.upper(), 15)

    filtered = df[
        (df['Place_clean'].isin([p.lower() for p in places])) &
        (df['Branch'].isin(branches)) &
        (df['Cutoff'].between(percentile - range_val, percentile + range_val))
    ]

    # Compute match score and sort
    filtered.loc[:, 'MatchScore'] = abs(filtered['Cutoff'] - percentile)
    filtered = filtered.sort_values(by=['Cutoff', 'MatchScore'], ascending=[False, True])
    filtered = filtered.drop(columns=['MatchScore', 'Place_clean'])

    # Select only essential columns
    selected_cols = ['College Code', 'College Name', 'Choice Code', 'Branch', 'Cutoff']
    if all(col in filtered.columns for col in selected_cols):
        filtered = filtered[selected_cols]

    return {
        "category": category,
        "range_applied": f"±{range_val} percentile",
        "total_preferences": len(filtered),
        "unique_colleges": filtered['College Code'].nunique() if 'College Code' in filtered.columns else filtered['College Name'].nunique(),
        "preferences": filtered.to_dict(orient='records')
    }

@app.get("/all")
def get_all_entries():
    return cutoff_df.to_dict(orient='records')
