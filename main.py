import pandas as pd
import os
from src.data_loader import load_data
from src.preprocessing import clean_data
from src.features import build_features
from src.visualization import visualization

def run_pipeline():
    input_path = 'data/raw/train.csv'
    output_path = 'data/processed/train_processed.csv'
    
    os.makedirs('data/processed', exist_ok=True)

    print("🚀 Pipeline Execution Started...")
    raw_df = load_data(input_path)
    
    if raw_df is not None:
        clean_df = clean_data(raw_df)
        
        final_df = build_features(clean_df)
        
        visualization(final_df)
        
        final_df.to_csv(output_path, index=False)
        print(f"💾 Processed data saved to: {output_path}")
        print("🎉 Pipeline finished successfully!")

if __name__ == "__main__":
    run_pipeline()