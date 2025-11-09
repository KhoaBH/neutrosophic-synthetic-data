import pandas as pd
import numpy as np
import time
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import warnings

def run_gan_benchmark(
    input_csv="diabetes_cleaned.csv",
    output_sizes=[100000, 1000000, 5000000, 10000000],
    n_repetitions=1
):
    """
    Run end-to-end benchmark for CTGAN,
    measuring Train time and Sample time separately.
    """
    total_start = time.time()
    results = []
    
    print("="*70)
    print("MODE: BENCHMARK (GAN - CTGAN)")
    print("="*70 + "\n")

    print(f"[INFO] Loading data from {input_csv}...")
    t0 = time.time()
    try:
        data = pd.read_csv(input_csv)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        print(f"[INFO] Loaded: {len(data)} rows ({time.time() - t0:.2f}s)")
    except Exception as e:
        print(f"[ERROR] Could not read file: {e}")
        return

    for r in range(1, n_repetitions + 1):
        print("\n" + "="*70)
        print(f"RUN {r}/{n_repetitions}")
        print("="*70)
        
        print("  Training CTGAN...")
        t_train_start = time.time()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            synthesizer = CTGANSynthesizer(metadata)
            synthesizer.fit(data)
            
        t_train_end = time.time()
        time_train = t_train_end - t_train_start
        print(f"    ✓ Train complete: {time_train:.2f}s")

        for size in output_sizes:
            print(f"    Testing sampling size: {size:,}")
            t_sample_start = time.time()
            
            try:
                synthetic_data = synthesizer.sample(num_rows=size)
                t_sample_end = time.time()
                time_sample = t_sample_end - t_sample_start
                print(f"      ✓ Sample complete: {time_sample:.2f}s")
                
            except Exception as e:
                print(f"      ✓ Sample: FAILED (Error occurred: {e})")
                time_sample = np.nan
            
            results.append({
                "run": r,
                "size": size,
                "train_time": time_train,
                "sample_time": time_sample
            })

    print("\n" + "="*70)
    print("BENCHMARK RESULTS (CTGAN)")
    print("="*70)
    
    df_results = pd.DataFrame(results)
    
    df_summary = df_results.groupby('size').agg(
        train_mean=('train_time', 'mean'),
        train_std=('train_time', 'std'),
        sample_mean=('sample_time', 'mean'),
        sample_std=('sample_time', 'std')
    ).reset_index()

    df_summary['total_mean'] = df_summary['train_mean'] + df_summary['sample_mean']
    df_summary['total_std'] = np.sqrt(df_summary['train_std']**2 + df_summary['sample_std']**2)

    print("--- Runtime Statistics (seconds) ---")
    presentation_table = pd.DataFrame()
    presentation_table['Sample Size'] = df_summary['size'].apply(lambda x: f"{x:,.0f}")
    
    presentation_table['Train Time (s)'] = df_summary.apply(
        lambda row: f"{row['train_mean']:.2f} ± {row['train_std']:.2f}" if pd.notna(row['train_mean']) else "N/A",
        axis=1
    )
    
    presentation_table['Sample Time (s)'] = df_summary.apply(
        lambda row: f"{row['sample_mean']:.2f} ± {row['sample_std']:.2f}" if pd.notna(row['sample_mean']) else "N/A (OOM)",
        axis=1
    )
    
    presentation_table['Total Time (s)'] = df_summary.apply(
        lambda row: f"{row['total_mean']:.2f} ± {row['total_std']:.2f}" if pd.notna(row['total_mean']) else "N/A (OOM)",
        axis=1
    )

    print(presentation_table.to_string(index=False))

    total_time = time.time() - total_start
    print(f"\nBENCHMARK COMPLETE (Total time: {total_time:.2f}s)")
    
    return df_results

if __name__ == "__main__":
    run_gan_benchmark()