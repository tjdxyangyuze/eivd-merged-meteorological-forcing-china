import pandas as pd
import numpy as np

def stratified_proportional_sampling(input_csv, output_csv, total_samples=50, random_seed=42):
    """
    Stratified Proportional Random Sampling for Validation Stations.
    1. Pre-screen for high data completeness (> 14,800 valid days).
    2. Stratify by Region (110°E), Elevation (3 bands), and Climate Zone (3 types).
    3. Allocate sample size proportionally to stratum density.
    4. Apply computer-generated random sampling within each stratum.
    """
    
    # Load the station inventory dataset
    # Required columns: 'Station_ID', 'Lon', 'Lat', 'Elevation', 'Climate_Zone', 'Valid_Days'
    df = pd.read_csv(input_csv)
    
    print(f"Total stations in original network: {len(df)}")
    
    # ==========================================
    # Step 1: Pre-screen for Record Completeness
    # ==========================================
    # Retain only stations with data availability > 14,800 days (1980-2020)
    high_quality_pool = df[df['Valid_Days'] > 14800].copy()
    print(f"Stations after completeness screening (>14,800 days): {len(high_quality_pool)}")
    
    if len(high_quality_pool) < total_samples:
        raise ValueError("Not enough high-quality stations to fulfill the 50-station requirement.")

    # ==========================================
    # Step 2: Three-Dimensional Stratification
    # ==========================================
    # 2.1 Spatial Stratification (110°E split)
    high_quality_pool['Region'] = high_quality_pool['Lon'].apply(
        lambda x: 'East' if x >= 110.0 else 'West'
    )
    
    # 2.2 Elevation Stratification (3 bands)
    def get_elev_band(elev):
        if elev < 500: return '<500m'
        elif 500 <= elev <= 3000: return '500-3000m'
        else: return '>3000m'
    high_quality_pool['Elev_Band'] = high_quality_pool['Elevation'].apply(get_elev_band)
    
    # 2.3 Create the composite Stratum key
    high_quality_pool['Stratum'] = (
        high_quality_pool['Region'] + '_' + 
        high_quality_pool['Elev_Band'] + '_' + 
        high_quality_pool['Climate_Zone']
    )
    
    # ==========================================
    # Step 3: Proportional Allocation & Random Sampling
    # ==========================================
    sampled_dfs = []
    
    # Calculate the proportion of each stratum
    stratum_counts = high_quality_pool['Stratum'].value_counts()
    total_pool_size = len(high_quality_pool)
    
    # Use a fixed random seed to ensure reproducibility (as noted in the response)
    np.random.seed(random_seed)
    
    for stratum, count in stratum_counts.items():
        # Calculate how many stations to select from this stratum
        proportion = count / total_pool_size
        n_samples_for_stratum = int(round(proportion * total_samples))
        
        # Get the sub-pool for this specific stratum
        stratum_df = high_quality_pool[high_quality_pool['Stratum'] == stratum]
        
        # Ensure we don't request more samples than available in the stratum
        n_samples_for_stratum = min(n_samples_for_stratum, len(stratum_df))
        
        # Apply computer-generated simple random sampling
        sampled = stratum_df.sample(n=n_samples_for_stratum, random_state=random_seed)
        sampled_dfs.append(sampled)
    
    # Combine the sampled stations from all strata
    final_sample = pd.concat(sampled_dfs)
    
    # ==========================================
    # Step 4: Adjust for rounding discrepancies
    # ==========================================
    # Due to rounding, the total might be slightly more or less than 50
    if len(final_sample) > total_samples:
        final_sample = final_sample.sample(n=total_samples, random_state=random_seed)
    elif len(final_sample) < total_samples:
        # If under 50, randomly select the remaining from the unselected high-quality pool
        remaining_pool = high_quality_pool.drop(final_sample.index)
        needed = total_samples - len(final_sample)
        if len(remaining_pool) >= needed:
            extra_sample = remaining_pool.sample(n=needed, random_state=random_seed)
            final_sample = pd.concat([final_sample, extra_sample])
            
    # Sort and save
    final_sample = final_sample.sort_values(by='Station_ID').reset_index(drop=True)
    final_sample.to_csv(output_csv, index=False)
    
    print(f"\nSuccessfully selected {len(final_sample)} stations.")
    print("\nDistribution of the final 50 stations:")
    print(final_sample['Region'].value_counts())
    print(final_sample['Elev_Band'].value_counts())
    print(final_sample['Climate_Zone'].value_counts())
    
    return final_sample

# ==========================================
# Run the script
# ==========================================
if __name__ == "__main__":
    # Replace these filenames with your actual data files
    INPUT_DATA = 'station_inventory.csv' 
    OUTPUT_DATA = 'selected_50_validation_stations.csv'
    
    # Execute the sampling
    try:
        selected_stations = stratified_proportional_sampling(
            input_csv=INPUT_DATA, 
            output_csv=OUTPUT_DATA, 
            total_samples=50, 
            random_seed=42  # Fixed random seed for reproducibility
        )
        print("\nSampling completed. Check", OUTPUT_DATA)
    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_DATA}'. Please ensure the input file exists.")
