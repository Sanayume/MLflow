
import scipy.io
import numpy as np
import os
import gc # For garbage collection
from datetime import date, timedelta
from scipy.ndimage import uniform_filter

# Define constants
DATA_DIR = '/sandbox/datasets/rainfalldata/'
MASK_DIR = '/sandbox/datasets/mask/'
OUTPUT_DIR = '/sandbox/outputs/feature_engineering/'
PRODUCTS = ['CMORPH', 'CHIRPS', 'GSMAP', 'IMERG', 'PERSIANN', 'SM2RAIN']
START_YEAR = 2016
END_YEAR = 2020
RAIN_THRESHOLD = 0.1 # mm/d for defining raining products
LOW_INTENSITY_THRESHOLD = 0.5 # for weak signal features

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Mask
print("Loading mask...")
mask_data = scipy.io.loadmat(os.path.join(MASK_DIR, 'changjiang_mask.mat'))
changjiang_mask = mask_data['data'].astype(bool) # Ensure boolean type
print(f"Mask shape: {changjiang_mask.shape}")
del mask_data
gc.collect()

# Load CHM data (reference product)
print("Loading CHM data...")
chm_data_path = os.path.join(DATA_DIR, f'CHM_{START_YEAR}_{END_YEAR}.mat')
chm_mat = scipy.io.loadmat(chm_data_path)
chm_raw = chm_mat['data'] # Assuming 'data' is the key
print(f"CHM raw data shape: {chm_raw.shape}")

# Apply mask to CHM
chm_masked = chm_raw.copy()
chm_masked[:, ~changjiang_mask] = np.nan # Apply mask to all time steps
np.save(os.path.join(OUTPUT_DIR, '长江CHM_原始值.npy'), chm_masked)
print("Saved 长江CHM_原始值.npy")
del chm_raw, chm_mat, chm_masked
gc.collect()

# 2. Load all product data and store in a dictionary
product_data = {}
for product in PRODUCTS:
    print(f"Loading {product} data...")
    file_path = os.path.join(DATA_DIR, f'{product}_{START_YEAR}_{END_YEAR}.mat')
    mat_data = scipy.io.loadmat(file_path)
    data_array = mat_data['data']
    # Apply mask
    data_array[:, ~changjiang_mask] = np.nan
    product_data[product] = data_array
    print(f"{product} data shape: {data_array.shape}")
    del mat_data, data_array
    gc.collect()

# Get time dimension and spatial shape
num_time_steps = product_data[PRODUCTS[0]].shape[0]
spatial_shape = product_data[PRODUCTS[0]].shape[1:]
print(f"Number of time steps: {num_time_steps}")
print(f"Spatial shape: {spatial_shape}")

# Feature Engineering
print("\nStarting Feature Engineering...")

# 1. 基础信息特征 (Raw Values)
print("1. Calculating Raw Values...")
for product, data_array in product_data.items():
    feature_name = f'长江{product}_原始值'
    np.save(os.path.join(OUTPUT_DIR, f'{feature_name}.npy'), data_array)
    print(f"Saved {feature_name}.npy")
gc.collect()

# Prepare for multi-product features: stack all products
# This will be (num_time_steps, num_products, spatial_shape[0], spatial_shape[1])
all_products_stacked = np.stack([product_data[p] for p in PRODUCTS], axis=1)
print(f"Stacked all products shape: {all_products_stacked.shape}")
gc.collect()

# 2. 多产品协同特征 (Multi-Product Synergy Features)
print("2. Calculating Multi-Product Synergy Features...")

# Mean across products
mean_all_products = np.nanmean(all_products_stacked, axis=1)
np.save(os.path.join(OUTPUT_DIR, '长江多产品均值.npy'), mean_all_products)
print("Saved 长江多产品均值.npy")

# Standard Deviation across products
std_all_products = np.nanstd(all_products_stacked, axis=1)
np.save(os.path.join(OUTPUT_DIR, '长江多产品标准差.npy'), std_all_products)
print("Saved 长江多产品标准差.npy")

# Median across products
median_all_products = np.nanmedian(all_products_stacked, axis=1)
np.save(os.path.join(OUTPUT_DIR, '长江多产品中位数.npy'), median_all_products)
print("Saved 长江多产品中位数.npy")

# Max across products
max_all_products = np.nanmax(all_products_stacked, axis=1)
np.save(os.path.join(OUTPUT_DIR, '长江多产品最大值.npy'), max_all_products)
print("Saved 长江多产品最大值.npy")

# Min across products
min_all_products = np.nanmin(all_products_stacked, axis=1)
np.save(os.path.join(OUTPUT_DIR, '长江多产品最小值.npy'), min_all_products)
print("Saved 长江多产品最小值.npy")

# Range across products
range_all_products = max_all_products - min_all_products
np.save(os.path.join(OUTPUT_DIR, '长江多产品极差.npy'), range_all_products)
print("Saved 长江多产品极差.npy")

# Rain Product Count (rainfall > 0.1 mm/d)
count_raining_products = np.sum(all_products_stacked > RAIN_THRESHOLD, axis=1)
np.save(os.path.join(OUTPUT_DIR, '长江指示降雨产品数量.npy'), count_raining_products)
print("Saved 长江指示降雨产品数量.npy")

# Coefficient of Variation (CV)
cv_all_products = np.divide(std_all_products, mean_all_products,
                            out=np.full_like(std_all_products, np.nan),
                            where=mean_all_products != 0)
np.save(os.path.join(OUTPUT_DIR, '长江多产品变异系数.npy'), cv_all_products)
print("Saved 长江多产品变异系数.npy")
gc.collect()

# Now, we can delete the stacked array to free memory
del all_products_stacked
gc.collect()

# 3. 时序动态捕捉特征 (Temporal Dynamic Features)
print("3. Calculating Temporal Dynamic Features...")

# Need to generate dates for cyclical features
start_date = date(START_YEAR, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(num_time_steps)]

# Cyclical Features
day_of_year = np.array([d.timetuple().tm_yday for d in dates])
sin_day_of_year = np.sin(2 * np.pi * day_of_year / 366.0)
cos_day_of_year = np.cos(2 * np.pi * day_of_year / 366.0)

# Broadcast to spatial dimensions
sin_day_of_year_spatial = np.tile(sin_day_of_year[:, np.newaxis, np.newaxis], (1, spatial_shape[0], spatial_shape[1]))
cos_day_of_year_spatial = np.tile(cos_day_of_year[:, np.newaxis, np.newaxis], (1, spatial_shape[0], spatial_shape[1]))

np.save(os.path.join(OUTPUT_DIR, '长江年内日周期_sin.npy'), sin_day_of_year_spatial)
np.save(os.path.join(OUTPUT_DIR, '长江年内日周期_cos.npy'), cos_day_of_year_spatial)
print("Saved 长江年内日周期_sin.npy and 长江年内日周期_cos.npy")
del sin_day_of_year_spatial, cos_day_of_year_spatial
gc.collect()

# Seasonal Dummies (using month)
months = np.array([d.month for d in dates])
season_spring = ((months >= 3) & (months <= 5)).astype(int)
season_summer = ((months >= 6) & (months <= 8)).astype(int)
season_autumn = ((months >= 9) & (months <= 11)).astype(int)
season_winter = ((months == 12) | (months == 1) | (months == 2)).astype(int)

season_spring_spatial = np.tile(season_spring[:, np.newaxis, np.newaxis], (1, spatial_shape[0], spatial_shape[1]))
season_summer_spatial = np.tile(season_summer[:, np.newaxis, np.newaxis], (1, spatial_shape[0], spatial_shape[1]))
season_autumn_spatial = np.tile(season_autumn[:, np.newaxis, np.newaxis], (1, spatial_shape[0], spatial_shape[1]))
season_winter_spatial = np.tile(season_winter[:, np.newaxis, np.newaxis], (1, spatial_shape[0], spatial_shape[1]))

np.save(os.path.join(OUTPUT_DIR, '长江季节_春.npy'), season_spring_spatial)
np.save(os.path.join(OUTPUT_DIR, '长江季节_夏.npy'), season_summer_spatial)
np.save(os.path.join(OUTPUT_DIR, '长江季节_秋.npy'), season_autumn_spatial)
np.save(os.path.join(OUTPUT_DIR, '长江季节_冬.npy'), season_winter_spatial)
print("Saved seasonal dummy features.")
del season_spring_spatial, season_summer_spatial, season_autumn_spatial, season_winter_spatial
gc.collect()

# Lag Features (1, 2, 3 days)
lag_days = [1, 2, 3]
features_to_lag = {
    '多产品均值': mean_all_products,
    '多产品标准差': std_all_products,
    '指示降雨产品数量': count_raining_products
}
# Add raw products to features_to_lag
for p in PRODUCTS:
    features_to_lag[f'{p}_原始值'] = product_data[p]

for feature_name_prefix, data_array in features_to_lag.items():
    for lag in lag_days:
        lagged_data = np.full_like(data_array, np.nan)
        if lag < num_time_steps:
            lagged_data[lag:] = data_array[:-lag]
        np.save(os.path.join(OUTPUT_DIR, f'长江{feature_name_prefix}_滞后{lag}天.npy'), lagged_data)
        print(f"Saved 长江{feature_name_prefix}_滞后{lag}天.npy")
    gc.collect()

# Lagged differences (e.g., lag1 - lag2 of mean_all_products)
mean_lag1 = np.full_like(mean_all_products, np.nan)
if 1 < num_time_steps:
    mean_lag1[1:] = mean_all_products[:-1]
mean_lag2 = np.full_like(mean_all_products, np.nan)
if 2 < num_time_steps:
    mean_lag2[2:] = mean_all_products[:-2]

lag1_lag2_mean_diff = mean_lag1 - mean_lag2
np.save(os.path.join(OUTPUT_DIR, '长江多产品均值_滞后1减滞后2差分.npy'), lag1_lag2_mean_diff)
print("Saved 长江多产品均值_滞后1减滞后2差分.npy")
del mean_lag1, mean_lag2, lag1_lag2_mean_diff
gc.collect()

# Difference Features (current - lag1)
for feature_name_prefix, data_array in features_to_lag.items():
    diff_data = np.full_like(data_array, np.nan)
    if 1 < num_time_steps:
        diff_data[1:] = data_array[1:] - data_array[:-1]
    np.save(os.path.join(OUTPUT_DIR, f'长江{feature_name_prefix}_当前减滞后1差分.npy'), diff_data)
    print(f"Saved 长江{feature_name_prefix}_当前减滞后1差分.npy")
gc.collect()

# Sliding Window Statistics (3, 7, 15 days)
window_sizes = [3, 7, 15]
features_for_window = {
    '多产品均值': mean_all_products,
    'CMORPH_原始值': product_data['CMORPH'],
    'GSMAP_原始值': product_data['GSMAP'],
    'PERSIANN_原始值': product_data['PERSIANN']
}

for feature_name_suffix, data_array in features_for_window.items():
    print(f"Calculating sliding window features for {feature_name_suffix}...")
    for window in window_sizes:
        # Mean
        sw_mean = np.full_like(data_array, np.nan)
        for t in range(window - 1, num_time_steps):
            sw_mean[t] = np.nanmean(data_array[t - window + 1 : t + 1], axis=0)
        np.save(os.path.join(OUTPUT_DIR, f'长江{feature_name_suffix}_滑动{window}天均值.npy'), sw_mean)
        print(f"Saved 长江{feature_name_suffix}_滑动{window}天均值.npy")

        # Standard Deviation
        sw_std = np.full_like(data_array, np.nan)
        for t in range(window - 1, num_time_steps):
            sw_std[t] = np.nanstd(data_array[t - window + 1 : t + 1], axis=0)
        np.save(os.path.join(OUTPUT_DIR, f'长江{feature_name_suffix}_滑动{window}天标准差.npy'), sw_std)
        print(f"Saved 长江{feature_name_suffix}_滑动{window}天标准差.npy")

        # Max
        sw_max = np.full_like(data_array, np.nan)
        for t in range(window - 1, num_time_steps):
            sw_max[t] = np.nanmax(data_array[t - window + 1 : t + 1], axis=0)
        np.save(os.path.join(OUTPUT_DIR, f'长江{feature_name_suffix}_滑动{window}天最大值.npy'), sw_max)
        print(f"Saved 长江{feature_name_suffix}_滑动{window}天最大值.npy")

        # Sum
        sw_sum = np.full_like(data_array, np.nan)
        for t in range(window - 1, num_time_steps):
            sw_sum[t] = np.nansum(data_array[t - window + 1 : t + 1], axis=0)
        np.save(os.path.join(OUTPUT_DIR, f'长江{feature_name_suffix}_滑动{window}天总和.npy'), sw_sum)
        print(f"Saved 长江{feature_name_suffix}_滑动{window}天总和.npy")

        # Range
        sw_range = np.full_like(data_array, np.nan)
        for t in range(window - 1, num_time_steps):
            sw_range[t] = np.nanmax(data_array[t - window + 1 : t + 1], axis=0) - np.nanmin(data_array[t - window + 1 : t + 1], axis=0)
        np.save(os.path.join(OUTPUT_DIR, f'长江{feature_name_suffix}_滑动{window}天极差.npy'), sw_range)
        print(f"Saved 长江{feature_name_suffix}_滑动{window}天极差.npy")
    gc.collect()

# 4. 空间关联特征 (Spatial Correlation Features)
print("4. Calculating Spatial Correlation Features (optimized for NaN handling)...")

# Helper for NaN-aware mean/std using uniform_filter trick
def nan_filter_2d_mean_std(arr_3d, size, func):
    output = np.full_like(arr_3d, np.nan)
    arr_zeros_nan = np.nan_to_num(arr_3d, nan=0.0)
    arr_mask = (~np.isnan(arr_3d)).astype(float) # 1 for non-NaN, 0 for NaN

    for t in range(arr_3d.shape[0]):
        sum_filtered = uniform_filter(arr_zeros_nan[t], size=size, mode='constant', cval=0.0)
        count_filtered = uniform_filter(arr_mask[t], size=size, mode='constant', cval=0.0)
        if func == np.nanmean:
            output[t] = np.divide(sum_filtered, count_filtered, out=np.full_like(sum_filtered, np.nan), where=count_filtered != 0)
        elif func == np.nanstd:
            sum_sq_filtered = uniform_filter(np.nan_to_num(arr_3d[t]**2, nan=0.0), size=size, mode='constant', cval=0.0)
            mean_sq = np.divide(sum_sq_filtered, count_filtered, out=np.full_like(sum_sq_filtered, np.nan), where=count_filtered != 0)
            mean_val = np.divide(sum_filtered, count_filtered, out=np.full_like(sum_filtered, np.nan), where=count_filtered != 0)
            variance = mean_sq - mean_val**2
            variance[variance < 0] = 0 # Prevent negative variance due to floating point
            output[t] = np.sqrt(variance)
    return output

# Helper for NaN-aware max/min using explicit loops (more robust for these ops)
def nan_max_filter_spatial(arr_3d, size):
    output = np.full_like(arr_3d, np.nan)
    pad_width = size // 2
    arr_padded = np.pad(arr_3d, ((0,0), (pad_width, pad_width), (pad_width, pad_width)), mode='constant', constant_values=np.nan)
    for t in range(arr_3d.shape[0]):
        for i in range(arr_3d.shape[1]):
            for j in range(arr_3d.shape[2]):
                window = arr_padded[t, i:i+size, j:j+size]
                output[t, i, j] = np.nanmax(window)
    return output

neighborhood_sizes = [3, 5] # 3x3, 5x5
features_for_spatial = {
    '多产品均值': mean_all_products,
    'GSMAP_原始值': product_data['GSMAP'],
    'PERSIANN_原始值': product_data['PERSIANN']
}

for feature_name_suffix, data_array in features_for_spatial.items():
    print(f"Calculating neighborhood features for {feature_name_suffix}...")
    for size in neighborhood_sizes:
        # Mean
        neighbor_mean = nan_filter_2d_mean_std(data_array, size, np.nanmean)
        np.save(os.path.join(OUTPUT_DIR, f'长江{feature_name_suffix}_邻域{size}x{size}均值.npy'), neighbor_mean)
        print(f"Saved 长江{feature_name_suffix}_邻域{size}x{size}均值.npy")

        # Standard Deviation
        neighbor_std = nan_filter_2d_mean_std(data_array, size, np.nanstd)
        np.save(os.path.join(OUTPUT_DIR, f'长江{feature_name_suffix}_邻域{size}x{size}标准差.npy'), neighbor_std)
        print(f"Saved 长江{feature_name_suffix}_邻域{size}x{size}标准差.npy")

        # Max
        neighbor_max = nan_max_filter_spatial(data_array, size) # Using custom nan_max_filter_spatial
        np.save(os.path.join(OUTPUT_DIR, f'长江{feature_name_suffix}_邻域{size}x{size}最大值.npy'), neighbor_max)
        print(f"Saved 长江{feature_name_suffix}_邻域{size}x{size}最大值.npy")
        del neighbor_max
        gc.collect()

        # Center minus neighbor mean
        center_minus_neighbor_mean = data_array - neighbor_mean
        np.save(os.path.join(OUTPUT_DIR, f'长江{feature_name_suffix}_中心减邻域{size}x{size}均值差.npy'), center_minus_neighbor_mean)
        print(f"Saved 长江{feature_name_suffix}_中心减邻域{size}x{size}均值差.npy")
        del center_minus_neighbor_mean
        gc.collect()

    del neighbor_mean, neighbor_std # Clean up intermediate neighborhood features
    gc.collect()

# Spatial Gradient Features
for feature_name_suffix, data_array in features_for_spatial.items():
    print(f"Calculating spatial gradient features for {feature_name_suffix}...")
    gradient_magnitude = np.full_like(data_array, np.nan)
    gradient_direction = np.full_like(data_array, np.nan)

    for t in range(num_time_steps):
        slice_2d = data_array[t, :, :]
        # Replace NaN with 0 for gradient calculation, then mask results
        slice_2d_filled = np.nan_to_num(slice_2d, nan=0.0)
        dy, dx = np.gradient(slice_2d_filled)

        magnitude = np.sqrt(dx**2 + dy**2)
        direction = np.arctan2(dy, dx)

        # Apply mask back
        magnitude[np.isnan(slice_2d)] = np.nan
        direction[np.isnan(slice_2d)] = np.nan

        gradient_magnitude[t, :, :] = magnitude
        gradient_direction[t, :, :] = direction

    np.save(os.path.join(OUTPUT_DIR, f'长江{feature_name_suffix}_空间梯度幅度.npy'), gradient_magnitude)
    np.save(os.path.join(OUTPUT_DIR, f'长江{feature_name_suffix}_空间梯度方向.npy'), gradient_direction)
    print(f"Saved 长江{feature_name_suffix}_空间梯度幅度.npy and 长江{feature_name_suffix}_空间梯度方向.npy")
    del gradient_magnitude, gradient_direction
    gc.collect()

# 5. 弱信号增强 / 模糊性特征 (Weak Signal Enhancement / Fuzziness Features)
print("5. Calculating Weak Signal Enhancement / Fuzziness Features...")

distance_to_threshold_0_1mm = np.abs(mean_all_products - RAIN_THRESHOLD)
np.save(os.path.join(OUTPUT_DIR, '长江距0_1mm阈值距离.npy'), distance_to_threshold_0_1mm)
print("Saved 长江距0_1mm阈值距离.npy")
del distance_to_threshold_0_1mm
gc.collect()

std_all_products_if_mean_low = np.full_like(std_all_products, np.nan)
low_mean_mask = mean_all_products < LOW_INTENSITY_THRESHOLD
std_all_products_if_mean_low[low_mean_mask] = std_all_products[low_mean_mask]
np.save(os.path.join(OUTPUT_DIR, '长江低强度降雨下产品标准差.npy'), std_all_products_if_mean_low)
print("Saved 长江低强度降雨下产品标准差.npy")
gc.collect()

# Fraction of products in 0 to 0.5mm range
all_products_stacked_reloaded_for_fraction = np.stack([product_data[p] for p in PRODUCTS], axis=1) # Re-stack
fraction_products_in_low_range = np.sum((all_products_stacked_reloaded_for_fraction > 0) & (all_products_stacked_reloaded_for_fraction <= LOW_INTENSITY_THRESHOLD), axis=1) / len(PRODUCTS)
np.save(os.path.join(OUTPUT_DIR, '长江低强度区间产品比例.npy'), fraction_products_in_low_range)
print("Saved 长江低强度区间产品比例.npy")
del all_products_stacked_reloaded_for_fraction, fraction_products_in_low_range
gc.collect()

cv_if_mean_low = np.full_like(cv_all_products, np.nan)
cv_if_mean_low[low_mean_mask] = cv_all_products[low_mean_mask]
np.save(os.path.join(OUTPUT_DIR, '长江低强度条件变异系数.npy'), cv_if_mean_low)
print("Saved 长江低强度条件变异系数.npy")
del low_mean_mask, cv_if_mean_low
gc.collect()

# 6. 高阶交互特征 (High-Order Interaction Features)
print("6. Calculating High-Order Interaction Features...")

# Reload cyclical features for interaction if needed
sin_day_of_year_spatial = np.load(os.path.join(OUTPUT_DIR, '长江年内日周期_sin.npy'))

product_std_times_sin_day = std_all_products * sin_day_of_year_spatial
np.save(os.path.join(OUTPUT_DIR, '长江产品标准差乘年内日周期sin.npy'), product_std_times_sin_day)
print("Saved 长江产品标准差乘年内日周期sin.npy")
del product_std_times_sin_day, sin_day_of_year_spatial
gc.collect()

# Reload std_all_products_if_mean_low and cv_all_products for interaction if they were deleted
std_all_products_if_mean_low_reloaded = np.load(os.path.join(OUTPUT_DIR, '长江低强度降雨下产品标准差.npy'))
cv_all_products_reloaded = np.load(os.path.join(OUTPUT_DIR, '长江多产品变异系数.npy'))

low_intensity_std_times_cv = std_all_products_if_mean_low_reloaded * cv_all_products_reloaded
np.save(os.path.join(OUTPUT_DIR, '长江低强度标准差乘变异系数.npy'), low_intensity_std_times_cv)
print("Saved 长江低强度标准差乘变异系数.npy")
del low_intensity_std_times_cv, std_all_products_if_mean_low_reloaded, cv_all_products_reloaded
gc.collect()

# count_raining_products and std_all_products should still be in memory from earlier.
# If not, reload. For safety, let's reload them.
count_raining_products_reloaded = np.load(os.path.join(OUTPUT_DIR, '长江指示降雨产品数量.npy'))
std_all_products_reloaded = np.load(os.path.join(OUTPUT_DIR, '长江多产品标准差.npy'))

rain_count_std_interaction = count_raining_products_reloaded * std_all_products_reloaded
np.save(os.path.join(OUTPUT_DIR, '长江降雨产品数量乘产品标准差.npy'), rain_count_std_interaction)
print("Saved 长江降雨产品数量乘产品标准差.npy")
del rain_count_std_interaction, count_raining_products_reloaded, std_all_products_reloaded
gc.collect()

# Final cleanup of all loaded data in product_data and other main variables
for product in PRODUCTS:
    if product in product_data:
        del product_data[product]
del mean_all_products, std_all_products, median_all_products, max_all_products, min_all_products, range_all_products, count_raining_products, cv_all_products
gc.collect()

print("\nFeature Engineering Complete! All features saved as .npy files.")
