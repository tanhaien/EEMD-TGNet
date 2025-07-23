#!/usr/bin/env python3
"""
Process Real GEFCom2014 Solar Dataset
"""
import os
import pandas as pd
import numpy as np

def process_gefcom2014_solar():
    """Process the real GEFCom2014 solar dataset"""
    
    base_path = "/home/anhpt95te/Downloads/UAI paper/data/GEFCom2014 Data/Solar"
    output_path = "/home/anhpt95te/Downloads/UAI paper/data"
    
    if not os.path.exists(base_path):
        print(f"GEFCom2014 Solar data not found at {base_path}")
        return None
    
    print("Processing Real GEFCom2014 Solar Dataset...")
    
    # Read instructions
    instructions_file = os.path.join(base_path, "Instructions.txt")
    if os.path.exists(instructions_file):
        with open(instructions_file, 'r') as f:
            instructions = f.read()
        print("Dataset Instructions:")
        print("=" * 50)
        print(instructions[:500] + "..." if len(instructions) > 500 else instructions)
        print("=" * 50)
    
    # Process multiple tasks - let's use Task 1 as primary dataset
    task_dirs = [d for d in os.listdir(base_path) if d.startswith('Task ') and d != 'Task 15']
    print(f"Found {len(task_dirs)} tasks")
    
    all_data = []
    
    for task_dir in sorted(task_dirs)[:3]:  # Process first 3 tasks for demonstration
        task_path = os.path.join(base_path, task_dir)
        print(f"\nProcessing {task_dir}...")
        
        # Load training data
        train_files = [f for f in os.listdir(task_path) if f.startswith('train')]
        predictor_files = [f for f in os.listdir(task_path) if f.startswith('predictors')]
        
        if train_files and predictor_files:
            train_file = os.path.join(task_path, train_files[0])
            predictor_file = os.path.join(task_path, predictor_files[0])
            
            try:
                # Read training data
                train_df = pd.read_csv(train_file)
                predictors_df = pd.read_csv(predictor_file)
                
                print(f"  Train data shape: {train_df.shape}")
                print(f"  Predictors shape: {predictors_df.shape}")
                print(f"  Train columns: {list(train_df.columns)}")
                print(f"  Predictor columns: {list(predictors_df.columns)}")
                
                # Merge training and predictor data if they have common columns
                if 'ZONEID' in train_df.columns and 'ZONEID' in predictors_df.columns:
                    if 'TIMESTAMP' in train_df.columns and 'TIMESTAMP' in predictors_df.columns:
                        merged_df = pd.merge(train_df, predictors_df, 
                                           on=['ZONEID', 'TIMESTAMP'], how='inner')
                        merged_df['task'] = task_dir
                        all_data.append(merged_df)
                        print(f"  Merged data shape: {merged_df.shape}")
                
            except Exception as e:
                print(f"  Error processing {task_dir}: {e}")
    
    if all_data:
        # Combine all tasks
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nCombined dataset shape: {combined_df.shape}")
        print(f"Columns: {list(combined_df.columns)}")
        
        # Clean and prepare data
        if 'TIMESTAMP' in combined_df.columns:
            combined_df['datetime'] = pd.to_datetime(combined_df['TIMESTAMP'])
            combined_df = combined_df.sort_values(['task', 'ZONEID', 'datetime'])
        
        # Save processed dataset
        output_file = os.path.join(output_path, "gefcom2014_solar_real.csv")
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Processed GEFCom2014 solar data saved to: {output_file}")
        
        # Generate summary
        print(f"\nDataset Summary:")
        print(f"- Total records: {len(combined_df):,}")
        print(f"- Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
        print(f"- Zones: {combined_df['ZONEID'].nunique()}")
        print(f"- Tasks: {combined_df['task'].nunique()}")
        
        # Show sample data
        print(f"\nSample data:")
        print(combined_df.head())
        
        return output_file
    
    else:
        print("No data could be processed from GEFCom2014 solar dataset")
        return None

if __name__ == "__main__":
    process_gefcom2014_solar()