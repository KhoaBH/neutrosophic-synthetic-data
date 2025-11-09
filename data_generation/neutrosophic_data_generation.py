import pandas as pd
import numpy as np
import time
from scipy.stats import (
    norm, gamma, lognorm, expon, weibull_min, 
    beta, uniform, chi2
)
from copulas.multivariate import GaussianMultivariate
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, LongType
import warnings
import os
import copy
#
class DistributionFitter:
    DISTRIBUTIONS = {
        'normal': {
            'scipy': norm,
            'params': lambda data: {'loc': np.mean(data), 'scale': max(np.std(data), 1e-6)},
            'generate': lambda params, size: np.random.normal(params['loc'], params['scale'], size),
            'has_ppf': True
        },
        'lognormal': {
            'scipy': lognorm,
            'params': lambda data: {
                's': np.std(np.log(data + 1e-10)),
                'scale': np.exp(np.mean(np.log(data + 1e-10)))
            },
            'generate': lambda params, size: np.random.lognormal(
                np.log(params['scale']), params['s'], size
            ),
            'has_ppf': True
        },
        'gamma': {
            'scipy': gamma,
            'params': lambda data: {
                'shape': max((np.mean(data) / max(np.std(data), 1e-6)) ** 2, 0.1),
                'scale': max(np.var(data) / max(np.mean(data), 1e-6), 0.1)
            },
            'generate': lambda params, size: np.random.gamma(params['shape'], params['scale'], size),
            'has_ppf': True
        },
        'exponential': {
            'scipy': expon,
            'params': lambda data: {'scale': np.mean(data)},
            'generate': lambda params, size: np.random.exponential(params['scale'], size),
            'has_ppf': True
        },
        'weibull': {
            'scipy': weibull_min,
            'params': lambda data: { 'c': 1.5, 'scale': np.mean(data) },
            'generate': lambda params, size: params['scale'] * np.random.weibull(params['c'], size),
            'has_ppf': True
        },
        'beta': {
            'scipy': beta,
            'params': lambda data: {
                'a': 2.0, 'b': 2.0,
                'loc': np.min(data),
                'scale': np.max(data) - np.min(data)
            },
            'generate': lambda params, size: (
                np.random.beta(params['a'], params['b'], size) * params['scale'] + params['loc']
            ),
            'has_ppf': True
        },
        'uniform': {
            'scipy': uniform,
            'params': lambda data: {'loc': np.min(data), 'scale': np.max(data) - np.min(data)},
            'generate': lambda params, size: np.random.uniform(
                params['loc'], params['loc'] + params['scale'], size
            ),
            'has_ppf': True
        },
        'chi2': {
            'scipy': chi2,
            'params': lambda data: {'df': max(int(np.mean(data)), 1)},
            'generate': lambda params, size: np.random.chisquare(params['df'], size),
            'has_ppf': True
        }
    }
    
    @staticmethod
    def smart_distribution_selection(stats, data_sample):
        data_sample = data_sample[data_sample > 0]
        if len(data_sample) < 10:
            return {'name': 'normal', 'params': {'loc': stats['mean_nonzero'], 'scale': stats['std_nonzero']}}
        
        mean = np.mean(data_sample)
        std = np.std(data_sample)
        skew = stats['skewness']
        kurt = stats['kurtosis']
        data_min = np.min(data_sample)
        data_max = np.max(data_sample)
        
        if data_min >= 0 and data_max <= 1:
            return {
                'name': 'beta',
                'params': {'a': 2.0, 'b': 2.0, 'loc': data_min, 'scale': data_max - data_min}
            }
        
        if data_min >= 0:
            if skew > 2:
                return {'name': 'exponential', 'params': {'scale': mean}}
            elif skew > 0.5:
                if kurt > 3:
                    return {
                        'name': 'lognormal',
                        'params': {'s': max(std / mean, 0.1), 'scale': mean}
                    }
                else:
                    return {
                        'name': 'gamma',
                        'params': {
                            'shape': max((mean / std) ** 2, 0.1),
                            'scale': max(std ** 2 / mean, 0.1)
                        }
                    }
        
        if abs(skew) < 0.5 and abs(kurt - 3) < 1:
            return {'name': 'normal', 'params': {'loc': mean, 'scale': std}}
        
        if kurt < 1.5:
            return {'name': 'uniform', 'params': {'loc': data_min, 'scale': data_max - data_min}}
        
        if kurt > 5:
            return {'name': 'chi2', 'params': {'df': max(int(mean), 1)}}
        
        return {'name': 'normal', 'params': {'loc': mean, 'scale': std}}

def fuzz_distribution_params(dist_name, dist_params, IN, safe_mode=True):
    if safe_mode:
        global_fuzz = np.random.uniform(1 - IN, 1 + IN)
        fuzzed = {}
        for param_name, param_value in dist_params.items():
            if param_name == 'df': # Integer params
                fuzzed[param_name] = max(int(param_value * global_fuzz), 1)
            elif param_name in ['scale', 'shape', 'c', 's', 'a', 'b']:
                fuzzed[param_name] = max(param_value * global_fuzz, 1e-6)
            else: # loc, etc
                fuzzed[param_name] = param_value * global_fuzz
    else:
        fuzzed = {}
        for param_name, param_value in dist_params.items():
            fuzz_factor = np.random.uniform(1 - IN, 1 + IN)
            if param_name == 'df':
                fuzzed[param_name] = max(int(param_value * fuzz_factor), 1)
            elif param_name in ['scale', 'shape', 'c', 's', 'a', 'b']:
                fuzzed[param_name] = max(param_value * fuzz_factor, 1e-6)
            else:
                fuzzed[param_name] = param_value * fuzz_factor
    
    return fuzzed

def generate_from_distribution_optimized(dist_name, dist_params, percentiles, stats):
    dist_info = DistributionFitter.DISTRIBUTIONS[dist_name]
    
    if dist_info['has_ppf']:
        try:
            scipy_dist = dist_info['scipy']
            
            # Convert params for scipy
            if dist_name == 'lognormal':
                synthetic = scipy_dist.ppf(percentiles, s=dist_params['s'], scale=dist_params['scale'])
            elif dist_name == 'gamma':
                synthetic = scipy_dist.ppf(percentiles, a=dist_params['shape'], scale=dist_params['scale'])
            elif dist_name == 'weibull':
                synthetic = scipy_dist.ppf(percentiles, c=dist_params['c'], scale=dist_params['scale'])
            elif dist_name == 'beta':
                synthetic = scipy_dist.ppf(
                    percentiles, 
                    a=dist_params['a'], 
                    b=dist_params['b'],
                    loc=dist_params['loc'], 
                    scale=dist_params['scale']
                )
            elif dist_name == 'exponential':
                synthetic = scipy_dist.ppf(percentiles, scale=dist_params['scale'])
            elif dist_name == 'uniform':
                synthetic = scipy_dist.ppf(percentiles, loc=dist_params['loc'], scale=dist_params['scale'])
            elif dist_name == 'chi2':
                synthetic = scipy_dist.ppf(percentiles, df=dist_params['df'])
            elif dist_name == 'normal':
                synthetic = scipy_dist.ppf(percentiles, loc=dist_params['loc'], scale=dist_params['scale'])
            else:
                raise NotImplementedError
            
            # Handle NaN/Inf
            synthetic = np.nan_to_num(synthetic, nan=stats['mean_nonzero'], 
                                    posinf=stats['max'], neginf=stats['min'])
            
            # Clip to a reasonable range (avoid extreme outliers)
            synthetic = np.clip(synthetic, stats['min'] * 0.9, stats['max'] * 1.1)
            
            return synthetic
            
        except Exception as e:
            print(f"[WARN] ppf failed for {dist_name}: {e}, fallback to generate")
    
    # FALLBACK: Old way (oversample + sort) - only used if ppf fails
    oversample_factor = 2
    large_sample = dist_info['generate'](dist_params, len(percentiles) * oversample_factor)
    sorted_synth = np.sort(large_sample)
    indices = (percentiles * (len(sorted_synth) - 1)).astype(int)
    return sorted_synth[indices]

def inject_zeros_optimized(data, zero_ratio):
    """
    Add zeros to data - use boolean mask instead of random.choice
    
    Args:
        data: Data array
        zero_ratio: Proportion of zeros (0.0 - 1.0)
    
    Returns:
        Array with zeros injected
    """
    if zero_ratio <= 0.01:
        return data
    
    zero_mask = np.random.rand(len(data)) < zero_ratio
    data[zero_mask] = 0
    
    return data

def handle_binary_column_safe(df_sample, col_name, output_size):
    """
    Handle binary/categorical column safely.
    
    Args:
        df_sample: Pandas DataFrame sample
        col_name: Column name
        output_size: Number of values to generate
    
    Returns:
        Array of synthetic values
    """
    probs = df_sample[col_name].value_counts(normalize=True)
    
    # Laplace smoothing - avoid crash when only 1 class is present
    epsilon = 0.01
    probs = (probs + epsilon) / (probs.sum() + epsilon * len(probs))
    
    return np.random.choice(probs.index, size=output_size, p=probs.values)

def compute_column_stats_spark_optimized(spark_df, binary_threshold=0.05):
    start = time.time()
    print("Computing statistics...")
    
    col_stats = {}
    total_rows = spark_df.count()
    
    numeric_cols = [
        c.name for c in spark_df.schema.fields 
        if c.dataType.typeName() in ["integer", "long", "double", "float"]
    ]
    
    if not numeric_cols:
        print("[WARN] No numeric columns found.")
        return {}, []

    stat_exprs = []
    for col_name in numeric_cols:
        c = col(col_name)
        stat_exprs.extend([
            F.mean(c).alias(f"{col_name}_mean"),
            F.stddev(c).alias(f"{col_name}_std"),
            F.skewness(c).alias(f"{col_name}_skewness"),
            F.kurtosis(c).alias(f"{col_name}_kurtosis"),
            F.min(c).alias(f"{col_name}_min"),
            F.max(c).alias(f"{col_name}_max")
        ])
    stats_row = spark_df.agg(*stat_exprs).collect()[0].asDict()

    unique_exprs = [
        F.countDistinct(col_name).alias(f"{col_name}_unique") 
        for col_name in numeric_cols
    ]
    unique_counts_row = spark_df.agg(*unique_exprs).collect()[0].asDict()

    df_pandas = spark_df.toPandas()
    
    for col_name in numeric_cols:
        unique_count = unique_counts_row.get(f"{col_name}_unique", 0)
        unique_ratio = (unique_count / total_rows) if total_rows > 0 else 0
        is_binary = unique_count <= 10
        
        mean_val = stats_row.get(f"{col_name}_mean", 0.0) or 0.0
        std_val = stats_row.get(f"{col_name}_std", 0.0) or 0.0
        
        nonzero_data = df_pandas[col_name].values[df_pandas[col_name].values > 0]
        if len(nonzero_data) > 0:
            mean_nonzero = np.mean(nonzero_data)
            std_nonzero = np.std(nonzero_data)
        else:
            mean_nonzero = mean_val
            std_nonzero = std_val

        col_stats[col_name] = {
            "mean": mean_val,
            "std": std_val,
            "mean_nonzero": mean_nonzero, 
            "std_nonzero": std_nonzero,  
            "skewness": stats_row.get(f"{col_name}_skewness", 0.0) or 0.0,
            "kurtosis": stats_row.get(f"{col_name}_kurtosis", 0.0) or 0.0,
            "min": stats_row.get(f"{col_name}_min"),
            "max": stats_row.get(f"{col_name}_max"),
            "is_binary": is_binary,
            "unique_count": unique_count,
            "zero_ratio": np.sum(df_pandas[col_name].values == 0) / len(df_pandas[col_name])
        }

    print(f"[INFO] Statistics computed in: {time.time() - start:.2f}s")
    return col_stats, numeric_cols

# ================================== #
# REFACTORED: Sequential Generation
# ================================== #
def generate_adaptive_synthetic_sequential(
    df_pandas, 
    distribution_map, 
    numeric_cols,
    IN, 
    output_size, 
    copula_sample_limit=100000, 
    auto_select=True
):
    start = time.time()
    df_pandas_sample = df_pandas.sample(n=min(copula_sample_limit, len(df_pandas)))

    if auto_select:
        for col_name in numeric_cols:
            if not distribution_map[col_name]['is_binary']:
                data_sample = df_pandas_sample[col_name].values
                data_sample = data_sample[data_sample > 0]
                
                if len(data_sample) < 10:
                    distribution_map[col_name]['distribution'] = 'normal'
                    distribution_map[col_name]['dist_params'] = {
                        'loc': distribution_map[col_name]['mean_nonzero'],
                        'scale': distribution_map[col_name]['std_nonzero']
                    }
                else:
                    dist_info = DistributionFitter.smart_distribution_selection(
                        distribution_map[col_name], 
                        data_sample
                    )
                    distribution_map[col_name]['distribution'] = dist_info['name']
                    distribution_map[col_name]['dist_params'] = dist_info['params']

    cols_to_fit = [col for col in numeric_cols if not distribution_map[col]['is_binary']]
    copula_model = None
    if cols_to_fit:
        copula_model = GaussianMultivariate()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            copula_model.fit(df_pandas_sample[cols_to_fit])
    
    uniform_df = pd.DataFrame() 
    learned_mean = np.zeros(len(cols_to_fit))
    learned_cov = np.eye(len(cols_to_fit))
    
    if copula_model:
        learned_cov = copula_model.correlation
        gaussian_samples = np.random.multivariate_normal(learned_mean, learned_cov, output_size)
        uniform_samples = norm.cdf(gaussian_samples)
        uniform_df = pd.DataFrame(uniform_samples, columns=cols_to_fit)
    
    synthetic_data = pd.DataFrame()

    for col_name in numeric_cols:
        stats = distribution_map[col_name]
        
        if stats["is_binary"]:
            synthetic_data[col_name] = handle_binary_column_safe(
                df_pandas_sample, col_name, output_size
            )
            continue 
        
        if col_name in uniform_df.columns:
            percentile_adj = np.clip(uniform_df[col_name].values, 0.001, 0.999)
        else:
            percentile_adj = np.random.uniform(0.001, 0.999, size=output_size)
        
        if auto_select and 'distribution' in stats:
            dist_name = stats['distribution']
            dist_params = stats['dist_params']
            fuzzed_params = fuzz_distribution_params(dist_name, dist_params, IN, safe_mode=True)
            
            adjusted_col = generate_from_distribution_optimized(
                dist_name, fuzzed_params, percentile_adj, stats
            )
        else:
            # Fallback: normal distribution
            IN = IN 
            mu = stats['mean_nonzero'] * np.random.uniform(1 - IN, 1 + IN)
            sigma = max(stats['std_nonzero'], 1e-6) * np.random.uniform(1 - IN, 1 + IN)
            adjusted_col = norm.ppf(percentile_adj, loc=mu, scale=sigma)
            adjusted_col = np.nan_to_num(adjusted_col, nan=mu)
        
        adjusted_col = inject_zeros_optimized(adjusted_col.copy(), stats['zero_ratio'])

        synthetic_data[col_name] = adjusted_col
    
    print(f"[Sequential] Completed in {time.time() - start:.2f}s")
    return synthetic_data

# ================================== #
# REFACTORED: Spark Generation
# ================================== #
def generate_adaptive_synthetic_with_spark_optimized(
    spark,
    distribution_map, 
    numeric_cols,
    IN, 
    output_size, 
    copula_model,
    cols_to_fit,
    num_partitions=200
):
    start = time.time()
    
    print("  [SPARK] Broadcasting variables...")
    dist_map_bc = spark.sparkContext.broadcast(distribution_map)
    cols_to_fit_bc = spark.sparkContext.broadcast(cols_to_fit)
    numeric_cols_bc = spark.sparkContext.broadcast(numeric_cols)
    
    learned_mean = np.zeros(len(cols_to_fit))
    learned_cov = np.eye(len(cols_to_fit))
    if copula_model:
        learned_cov = copula_model.correlation
        
    learned_mean_bc = spark.sparkContext.broadcast(learned_mean)
    learned_cov_bc = spark.sparkContext.broadcast(learned_cov)
    in_bc = spark.sparkContext.broadcast(IN)

    schema = StructType([])
    for col_name in numeric_cols:
        schema.add(StructField(col_name, DoubleType(), True))
    
    print(f"  [SPARK] Creating base DataFrame with {output_size} rows in {num_partitions} partitions...")
    base_df = spark.range(0, output_size, numPartitions=num_partitions).toDF("id")

    def generate_partition(iterator):
        dist_map = dist_map_bc.value
        cols_to_fit_p = cols_to_fit_bc.value
        numeric_cols_p = numeric_cols_bc.value
        learned_mean_p = learned_mean_bc.value
        learned_cov_p = learned_cov_bc.value
        in_p = in_bc.value

        for pdf in iterator:
            part_size = len(pdf)
            if part_size == 0:
                continue
            
            uniform_df = pd.DataFrame() 
            if cols_to_fit_p:
                gaussian_samples = np.random.multivariate_normal(learned_mean_p, learned_cov_p, part_size)
                uniform_samples = norm.cdf(gaussian_samples)
                uniform_df = pd.DataFrame(uniform_samples, columns=cols_to_fit_p)

            synthetic_data = pd.DataFrame()
            
            for col_name in numeric_cols_p:
                stats = dist_map[col_name]
                
                if stats["is_binary"]:
                    probs_series = pd.Series(stats["binary_probs"])
                    epsilon = 0.01
                    probs_series = (probs_series + epsilon) / (probs_series.sum() + epsilon * len(probs_series))
                    synthetic_col = np.random.choice(
                        probs_series.index, size=part_size, p=probs_series.values
                    )
                    synthetic_data[col_name] = synthetic_col
                    continue
                
                if col_name in uniform_df.columns:
                    percentile_adj = np.clip(uniform_df[col_name].values, 1e-6, 1.0 - 1e-6)
                else:
                    percentile_adj = np.random.uniform(0.001, 0.999, size=part_size)
                
                if 'distribution' in stats:
                    dist_name = stats['distribution']
                    dist_params = stats['dist_params']
                    fuzzed_params = fuzz_distribution_params(dist_name, dist_params, in_p, safe_mode=True)
                    adjusted_col = generate_from_distribution_optimized(
                        dist_name, fuzzed_params, percentile_adj, stats
                    )
                else:
                    mu = stats['mean_nonzero'] * np.random.uniform(1 - in_p, 1 + in_p)
                    sigma = max(stats['std_nonzero'], 1e-6) * np.random.uniform(1 - in_p, 1 + in_p)
                    adjusted_col = norm.ppf(percentile_adj, loc=mu, scale=sigma)
                    adjusted_col = np.nan_to_num(adjusted_col, nan=mu)
                adjusted_col = inject_zeros_optimized(adjusted_col.copy(), stats['zero_ratio'])

                synthetic_data[col_name] = adjusted_col

            yield synthetic_data
            
    synthetic_df = base_df.mapInPandas(generate_partition, schema=schema)
    synthetic_df = synthetic_df.cache()
    count = synthetic_df.count()
    print(f"  [SPARK] Generated {count} rows. Time: {time.time() - start:.2f}s")
    
    return synthetic_df

# ================================================== #
# Main functions (UNCHANGED)
# ================================================== #
def run_benchmark_mode(
    input_csv="diabetes_cleaned.csv",
    output_sizes=[100000, 1000000, 5000000, 10000000, 50000000],
    n_repetitions=3,
    in_value=0.1,
    num_partitions=200
):
    total_start = time.time()
    results = []
    
    SEQ_THRESHOLD = 10_000_000 

    spark = SparkSession.builder \
        .appName("SyntheticDataBenchmark") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print(f"[INFO] Loading data from {input_csv}...")
    t0 = time.time()
    try:
        spark_df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_csv)
        spark_df = spark_df.cache()
        df_pandas_full = spark_df.toPandas() 
        print(f"[INFO] Loaded: {spark_df.count()} rows ({time.time() - t0:.2f}s)")
    except Exception as e:
        print(f"[ERROR] Could not read file: {e}")
        spark.stop()
        return

    print("[INFO] Fitting models...")
    distribution_map_original, numeric_cols = compute_column_stats_spark_optimized(
        spark_df, binary_threshold=0.05
    )
    df_pandas_sample = df_pandas_full.sample(n=min(100000, len(df_pandas_full)))   
    map_for_spark = copy.deepcopy(distribution_map_original)
    cols_to_fit = [col for col in numeric_cols if not map_for_spark[col]['is_binary']]
    for col_name in cols_to_fit:
        data_sample = df_pandas_sample[col_name].values
        data_sample = data_sample[data_sample > 0]
        if len(data_sample) < 10:
             map_for_spark[col_name]['distribution'] = 'normal'
             map_for_spark[col_name]['dist_params'] = {
                 'loc':  map_for_spark[col_name]['mean_nonzero'],
                 'scale':  map_for_spark[col_name]['std_nonzero']
             }
        else:
            dist_info = DistributionFitter.smart_distribution_selection(
                map_for_spark[col_name], data_sample
            )
            map_for_spark[col_name]['distribution'] = dist_info['name']
            map_for_spark[col_name]['dist_params'] = dist_info['params']
    for col_name in numeric_cols:
        if map_for_spark[col_name]["is_binary"]:
             map_for_spark[col_name]["binary_probs"] = \
                 df_pandas_sample[col_name].value_counts(normalize=True).to_dict()
    copula_model = None
    if cols_to_fit:
        copula_model = GaussianMultivariate()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            copula_model.fit(df_pandas_sample[cols_to_fit])
    print("[INFO] Models fitted.")
    
    for r in range(1, n_repetitions + 1):
        print("\n" + "="*70)
        print(f"RUN {r}/{n_repetitions}")
        print("="*70)
        
        for size in output_sizes:
            print(f"  Testing size: {size:,} (Run {r})")
            np.random.seed(42 + r)
            t_start_spark = time.time()
            synthetic_df = generate_adaptive_synthetic_with_spark_optimized(
                spark, map_for_spark, numeric_cols, in_value, 
                size, copula_model, cols_to_fit, num_partitions
            )
            synthetic_df.unpersist()
            t_end_spark = time.time()
            time_spark = t_end_spark - t_start_spark
            print(f"    ✓ Spark: {time_spark:.2f}s")

            time_seq = np.nan 
            
            if size <= SEQ_THRESHOLD:
                try:
                    np.random.seed(42 + r)
                    t_start_seq = time.time()
                    generate_adaptive_synthetic_sequential(
                        df_pandas_full, 
                        copy.deepcopy(distribution_map_original),
                        numeric_cols, in_value, size, 100000, auto_select=True
                    )
                    t_end_seq = time.time()
                    time_seq = t_end_seq - t_start_seq
                    print(f"    ✓ Sequential: {time_seq:.2f}s")
                except Exception as e:
                    print(f"    ✓ Sequential: FAILED (Error occurred: {e})")
            else:
                print(f"    ✓ Sequential: SKIPPED (Size > {SEQ_THRESHOLD:,})")
            
            
            results.append({
                "run": r, "size": size,
                "spark_time": time_spark, "seq_time": time_seq
            })

    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    
    df_results = pd.DataFrame(results)
    df_summary = df_results.groupby('size').agg({
        'spark_time': ['mean', 'std'],
        'seq_time': ['mean', 'std']
    })
    
    print(df_summary)
    
    print("\n--- Speedup (Sequential / Spark) ---")
    df_summary['speedup'] = df_summary[('seq_time', 'mean')] / df_summary[('spark_time', 'mean')]
    print(df_summary['speedup'])

    total_time = time.time() - total_start
    print(f"\nBENCHMARK COMPLETE (Total time: {total_time:.2f}s)")
    
    spark.stop()
    return df_results

def run_generation_mode(
    input_csv="diabetes_cleaned.csv",
    output_size=1000000,
    in_levels=[0.0, 0.05, 0.1, 0.2, 0.3],
    output_dir="synthetic_data",
    use_spark=True,
    num_partitions=200
):
    """Generate multiple synthetic data files with different IN levels"""
    total_start = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Will save files to directory: {output_dir}")
    
    if use_spark:
        spark = SparkSession.builder \
            .appName("SyntheticDataGeneration") \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
    
    print(f"[INFO] Loading data from {input_csv}...")
    t0 = time.time()
    try:
        if use_spark:
            spark_df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_csv)
            spark_df = spark_df.cache()
            df_pandas_full = spark_df.toPandas()
        else:
            df_pandas_full = pd.read_csv(input_csv)
            spark_df = None
        print(f"[INFO] Loaded: {len(df_pandas_full)} rows ({time.time() - t0:.2f}s)")
    except Exception as e:
        print(f"[ERROR] Could not read file: {e}")
        if use_spark:
            spark.stop()
        return

    print("[INFO] Fitting models...")
    if use_spark:
        distribution_map_original, numeric_cols = compute_column_stats_spark_optimized(
            spark_df, binary_threshold=0.05
        )
    else:
        numeric_cols = df_pandas_full.select_dtypes(include=[np.number]).columns.tolist()
        distribution_map_original = {}
        for col in numeric_cols:
            nonzero_data = df_pandas_full[df_pandas_full[col] > 0][col]
            distribution_map_original[col] = {
                'mean': df_pandas_full[col].mean(),
                'std': df_pandas_full[col].std(),
                'mean_nonzero': nonzero_data.mean() if len(nonzero_data) > 0 else df_pandas_full[col].mean(),
                'std_nonzero': nonzero_data.std() if len(nonzero_data) > 0 else df_pandas_full[col].std(),
                'skewness': df_pandas_full[col].skew(),
                'kurtosis': df_pandas_full[col].kurtosis(),
                'min': df_pandas_full[col].min(),
                'max': df_pandas_full[col].max(),
                'is_binary': df_pandas_full[col].nunique() <= 10,
                'unique_count': df_pandas_full[col].nunique(),
                'zero_ratio': (df_pandas_full[col] == 0).sum() / len(df_pandas_full)
            }
    
    df_pandas_sample = df_pandas_full.sample(n=min(100000, len(df_pandas_full)))
    
    if use_spark:
        map_for_generation = copy.deepcopy(distribution_map_original)
        cols_to_fit = [col for col in numeric_cols if not map_for_generation[col]['is_binary']]

        for col_name in cols_to_fit:
            data_sample = df_pandas_sample[col_name].values
            data_sample = data_sample[data_sample > 0]
            if len(data_sample) < 10:
                map_for_generation[col_name]['distribution'] = 'normal'
                map_for_generation[col_name]['dist_params'] = {
                    'loc': map_for_generation[col_name]['mean_nonzero'],
                    'scale': map_for_generation[col_name]['std_nonzero']
                }
            else:
                dist_info = DistributionFitter.smart_distribution_selection(
                    map_for_generation[col_name], data_sample
                )
                map_for_generation[col_name]['distribution'] = dist_info['name']
                map_for_generation[col_name]['dist_params'] = dist_info['params']

        for col_name in numeric_cols:
            if map_for_generation[col_name]["is_binary"]:
                map_for_generation[col_name]["binary_probs"] = \
                    df_pandas_sample[col_name].value_counts(normalize=True).to_dict()

        copula_model = None
        if cols_to_fit:
            copula_model = GaussianMultivariate()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                copula_model.fit(df_pandas_sample[cols_to_fit])
        print("[INFO] Models fitted.")
    
    print("\n" + "="*70)
    print(f"STARTING TO GENERATE {len(in_levels)} FILES WITH DIFFERENT IN LEVELS")
    print("="*70)
    
    generation_times = []
    
    for idx, in_value in enumerate(in_levels, 1):
        print(f"\n[{idx}/{len(in_levels)}] Generating file with IN = {in_value}")
        np.random.seed(42)
        
        t_start = time.time()
        
        if use_spark:
            synthetic_df = generate_adaptive_synthetic_with_spark_optimized(
                spark, map_for_generation, numeric_cols, in_value,
                output_size, copula_model, cols_to_fit, num_partitions
            )
            
            output_filename = f"synthetic_IN_{in_value:.3f}_size_{output_size}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"  Saving file: {output_filename}...")
            synthetic_df.toPandas().to_csv(output_path, index=False)
            synthetic_df.unpersist()
            
        else:
            synthetic_df = generate_adaptive_synthetic_sequential(
                df_pandas_full,
                copy.deepcopy(distribution_map_original),
                numeric_cols,
                in_value,
                output_size,
                100000,
                auto_select=True
            )
            
            output_filename = f"synthetic_IN_{in_value:.3f}_size_{output_size}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"  Saving file: {output_filename}...")
            synthetic_df.to_csv(output_path, index=False)
        
        t_end = time.time()
        elapsed = t_end - t_start
        generation_times.append({
            'IN': in_value,
            'time': elapsed,
            'filename': output_filename
        })
    print("\n" + "="*70)
    print("FILE GENERATION SUMMARY")
    print("="*70)
    
    df_generation = pd.DataFrame(generation_times)
    print(df_generation.to_string(index=False))
    
    total_time = time.time() - total_start
    print(f"\nGENERATED {len(in_levels)} FILES")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average per file: {total_time/len(in_levels):.2f}s")
    print(f"  All files saved in: {output_dir}/")
    
    if use_spark:
        spark.stop()
    
    return df_generation



def main():  
    MODE = "generate"
    
    if MODE == "benchmark":
        print("\n" + "="*70)
        print("MODE: BENCHMARK (Comparing Spark vs Sequential)")
        print("="*70 + "\n")
        
        results = run_benchmark_mode(
            input_csv="winequality-white-scaled.csv",
            output_sizes=[100000, 1000000, 5000000, 10000000, 50000000],
            n_repetitions=3,
            in_value=0.1,
            num_partitions=200
        )
        
    elif MODE == "generate":
        print("\n" + "="*70)
        print("MODE: GENERATION (Creating synthetic data files)")
        print("="*70 + "\n")
        
        results = run_generation_mode(
            input_csv="winequality-white-scaled.csv",
            output_size=4898,
            in_levels=[0.00, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3],
            output_dir="wine_synthetic_data",
            use_spark=True,
            num_partitions=200
        )
        
    else:
        print(f"[ERROR] Invalid MODE: '{MODE}'")
        print("        Only 'benchmark' or 'generate' are accepted")
        return
    
    print("\n" + "="*70)
    print("PROGRAM COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()