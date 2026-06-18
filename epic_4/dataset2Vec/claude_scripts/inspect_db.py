import sqlite3
import pandas as pd
import os
import json

db_path = "store/optuna.db"
parquet_path = "store/leaderboards.parquet"

print("Checking SQLite (optuna.db):")
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count studies (dataset-model pairs completed)
    cursor.execute("SELECT COUNT(*) FROM studies;")
    num_studies = cursor.fetchone()[0]
    print(f"=> Number of completed studies (dataset-model pairs): {num_studies}")
    
    # Count total trials run across all studies
    cursor.execute("SELECT COUNT(*) FROM trials;")
    num_trials = cursor.fetchone()[0]
    print(f"=> Number of total trials run: {num_trials}")
    
    # Get distinct model names and dataset IDs from study names (format: {dataset_id}__{model_name})
    cursor.execute("SELECT study_name, study_id FROM studies;")
    studies_info = cursor.fetchall()
    unique_models_optuna = set()
    dataset_counts_optuna = {}
    
    # We will map study_id to dataset_id
    study_to_dataset = {}
    for sname, sid in studies_info:
        if "__" in sname:
            dataset_id, model_name = sname.split("__", 1)
            unique_models_optuna.add(model_name)
            dataset_counts_optuna[dataset_id] = dataset_counts_optuna.get(dataset_id, 0) + 1
            study_to_dataset[sid] = dataset_id

    # Print trials table columns to debug column name
    cursor.execute("PRAGMA table_info(trials);")
    columns = cursor.fetchall()
    print("Trials table columns:", [col[1] for col in columns])
    
    # We will look for a time column or datetime column
    # Usually it is 'datetime_start' or 'datetime_complete' or 'datetime_finished' in Optuna schema versions.
    # Let's inspect what's available.
    time_col = None
    for col in columns:
        if 'datetime' in col[1] or 'time' in col[1]:
            time_col = col[1]
            break
            
    if time_col:
        cursor.execute(f"SELECT study_id, {time_col} FROM trials WHERE {time_col} IS NOT NULL;")
        trials_finished = cursor.fetchall()
        dataset_latest_trial = {}
        for sid, finished_time in trials_finished:
            dataset_id = study_to_dataset.get(sid)
            if dataset_id:
                # finished_time could be float or string or int depending on optuna version
                if dataset_id not in dataset_latest_trial or finished_time > dataset_latest_trial[dataset_id]:
                    dataset_latest_trial[dataset_id] = finished_time
    # Query trial count per study state
    cursor.execute("""
        SELECT s.study_name, COUNT(t.trial_id) 
        FROM studies s 
        LEFT JOIN trials t ON s.study_id = t.study_id AND t.state = 'COMPLETE'
        GROUP BY s.study_id;
    """)
    study_trial_counts = cursor.fetchall()
    
    dataset_to_study_counts = {}
    for sname, count in study_trial_counts:
        if "__" in sname:
            dataset_id, model_name = sname.split("__", 1)
            dataset_to_study_counts.setdefault(dataset_id, []).append((model_name, count))
            
    print("\n=> Trial completion progress per dataset:")
    for d_id, studies in sorted(dataset_to_study_counts.items()):
        total_complete = sum(count for _, count in studies)
        n_studies = len(studies)
        print(f"   - {d_id}: {n_studies} studies in DB, total {total_complete} COMPLETE trials (avg {total_complete/n_studies:.1f} per study)")
        # Show detail of incomplete studies (less than 50 trials)
        incomplete = [(m, c) for m, c in studies if c < 50]
        if incomplete:
            print(f"     * Incomplete studies ({len(incomplete)}): {incomplete[:5]} ...")
        else:
            print("     * All studies fully completed (50/50 trials).")
            
    conn.close()
else:
    print(f"=> {db_path} does not exist.")

print("\nChecking Parquet (leaderboards.parquet):")
if os.path.exists(parquet_path):
    df = pd.read_parquet(parquet_path)
    print(f"=> Number of dataset records: {len(df)}")
    print(f"=> Datasets in leaderboards: {df['dataset_id'].tolist()}")
    
    unique_models_leaderboard = set()
    for idx, row in df.iterrows():
        lead = row['leaderboard']
        # Decodes if encoded as JSON string or raw
        if isinstance(lead, str):
            lead = json.loads(lead)
        elif isinstance(lead, bytes):
            lead = json.loads(lead.decode('utf-8'))
            
        for entry in lead:
            if isinstance(entry, str):
                entry = json.loads(entry)
            unique_models_leaderboard.add(entry.get('model_name'))
            
    print(f"=> Unique models stored in leaderboards ({len(unique_models_leaderboard)}): {sorted(list(unique_models_leaderboard))}")
else:
    print(f"=> {parquet_path} does not exist.")
