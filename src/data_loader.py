import pandas as pd
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ فایل پیدا نشد: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ داده‌ها با موفقیت بارگذاری شدند. ابعاد: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ خطا در خواندن فایل: {e}")
        return None