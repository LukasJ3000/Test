#!/usr/bin/env python
"""
Script 2.5

This script recursively processes CSV files from a Source Folder (with multiple sub-folders)
and saves the FFT analysis plots in a Target Folder with the same sub-folder structure.
It accepts CSV files where the delimiter is ';' and the decimal separator can be either '.' or ','.
Additionally, it collects output variables from each CSV file into a single Excel file named
"output_variables.xlsx" stored in the TARGET_FOLDER.

CSV file format:
  - Row 1: Column names
  - Row 2: Units
  - Row 3: Empty
  - Data starts from Row 4

Pre-processing steps:
  1. Outlier removal on the computed resistance (replace outliers with the median).
  2. Smoothing the resistance signal using a moving average.
  3. Applying a Hann window to the resistance signal before FFT.

Additional modifications:
  - All data is converted to float64.
  - Infinite values are replaced with 9999999999.9.
  - If a solitary infinity (9999999999.9) occurs in any column (i.e. surrounded by normal values), that row is dropped.
  - In the resistance column, any NaN or negative values are replaced with 0.
  - For Gaussian fitting, the peak frequency is used as the fixed Gaussian mean.

Output variables (per CSV):
  - Peak Power (FFT amplitude at the peak)
  - Peak Frequency
  - Gaussian Standard Deviation

These, along with the CSV file name and any error messages, are stored in a single Excel file ("output_variables.xlsx")
with the following columns:
    1. CSV File Name
    2. Peak Power
    3. Peak Frequency
    4. Gaussian Standard Deviation
    5. Errors

For files where an error occurs, the variable columns are left empty and the error message is recorded.
If a CSV file is already processed (its plot exists), it is marked with "Already processed".
"""

import os
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import get_window


def calculate_resistance(voltage, current):
    """
    Calculate resistance from voltage and current arrays.
    If the result is 2D, return the first column.
    """
    voltage = np.asarray(voltage)
    current = np.asarray(current)
    res = np.where(current != 0, voltage / current, np.nan)
    return res if res.ndim == 1 else res[:, 0]


def calculate_Fs(timely):
    """
    Calculate sampling frequency for FFT power spectrum.
    Raises a ValueError if the average time interval is NaN or zero.
    """
    time_diff = np.diff(timely.ravel())
    avg_time_interval = np.mean(time_diff)
    if np.isnan(avg_time_interval) or avg_time_interval == 0:
        raise ValueError("Invalid time interval: check the time column data for missing or zero values.")
    return 1 / avg_time_interval


def fftit(df_array, Fs, lbFreq, ubFreq):
    """Conduct Fast-Fourier Transform on a 1D numpy vector."""
    df_array = df_array.ravel()
    # Detrend by subtracting the mean
    meanLess = df_array - np.mean(df_array)
    Y = np.fft.fft(meanLess)
    L = len(df_array)
    P2 = np.abs(Y / L)
    P1 = P2[:L // 2 + 1]
    P1[1:-1] *= 2  # One-sided FFT
    f = np.linspace(0, Fs / 2, len(P1))
    valid = (f >= lbFreq) & (f < ubFreq)
    P1 = P1[valid]
    f = f[valid]
    maxP1 = np.max(P1)
    FundFreq = f[P1 == maxP1] if np.size(P1) > 0 else None
    return P1, f, FundFreq


def gaussian(x, amp, mean, stddev):
    """Gaussian function for curve fitting."""
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def gaussian_fixed_mean(x, amp, stddev, mean_fixed):
    """Gaussian function with fixed mean."""
    return amp * np.exp(-((x - mean_fixed) ** 2) / (2 * stddev ** 2))


def normalize_array(arr):
    """Normalize a NumPy array to the range [0, 1]."""
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(arr)


def moving_average(data, window_size):
    """Compute the moving average of a 1D NumPy array,
    keeping the same size by padding with the median value."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if window_size > len(data):
        raise ValueError("window_size cannot be larger than the data length")

    # Compute median for padding
    median_val = np.median(data)

    # Asymmetric padding for even window sizes to maintain equal length
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left

    # Pad with the median value
    padded = np.pad(data, (pad_left, pad_right), mode='constant', constant_values=median_val)

    # Perform convolution with 'valid' to get same-size output
    ma = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')

    return ma


def replace_outliers_with_median(data, m=3.0):
    """
    Replace outliers in a 1D NumPy array with the median.
    Outliers are defined as points deviating more than m times the median absolute deviation.
    """
    data = np.array(data, copy=True)
    median_val = np.median(data)
    deviation = np.abs(data - median_val)
    mad = np.median(deviation) if np.median(deviation) != 0 else 1.0
    threshold = m * mad
    outlier_mask = deviation > threshold
    data[outlier_mask] = median_val
    return data


def smooth_data(data, window_size=5):
    """Smooth a 1D NumPy array using a simple moving average filter."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def drop_solitary_infinity_rows(df, inf_value=9999999999.9):
    """
    For each column in the DataFrame, drop any row that contains a solitary infinity value (inf_value),
    meaning that the value equals inf_value and both the previous and next rows are not inf_value.
    """
    rows_to_drop = set()
    n = len(df)
    for col in df.columns:
        values = df[col].values
        if n < 3:
            continue
        for i in range(1, n - 1):
            if (values[i] == inf_value and
                    values[i - 1] != inf_value and
                    values[i + 1] != inf_value):
                rows_to_drop.add(i)
    if rows_to_drop:
        df = df.drop(list(rows_to_drop)).reset_index(drop=True)
    return df


def convert_decimal(x):
    """
    Converter function for pd.read_csv.
    Replaces comma with dot if needed and converts the string to float.
    """
    try:
        if isinstance(x, str):
            return float(x.replace(',', '.'))
        return float(x)
    except:
        return np.nan


def run_spucker_detectors(df: pd.DataFrame, columns: list, threshold: float) -> int:
    # Instantiate the detectors
    spucker_detector = SpuckerCounter(threshold=threshold)

    data_values = df.iloc[:, 1]

    # Execute the detection algorithms
    count = spucker_detector.count(data_values)
    print("Spucker:", count)

    return count


def run_tsad_detectors(df: pd.DataFrame, columns: list, result_csv_path: str) -> pd.DataFrame:
    """
    Executes the KMeans and IsolationForest anomaly detectors on the selected columns of the dataframe.
    The anomaly scores are saved to a CSV file and also returned as a DataFrame.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        columns (list): List of columns to use for detection.
        result_csv_path (str): File path to save the results CSV.

    Returns:
        pd.DataFrame: DataFrame containing anomaly scores from both detectors.
                    Columns: 'kmeans_anomaly_score' and 'isolation_forest_anomaly_score'.
    """

    tsad_csv_path = f"{result_csv_path}tsad.csv"

    # Instantiate the detectors
    tsad_detector = AutoTSADAnomalyDetector()

    temp_csv_path = "temp_univariate_data.csv"
    save_univariate_time_series(df, columns[0], temp_csv_path)

    # Execute the detection algorithms
    anomaly_scores_tsad = tsad_detector.detect(temp_csv_path)

    tsad_df = pd.DataFrame({'tsad_anomaly_score': pd.Series(anomaly_scores_tsad)})
    tsad_df.to_csv(tsad_csv_path, index=False)

    return tsad_df


def run_dwt_detector(df: pd.DataFrame, columns: list, result_csv_path: str) -> pd.DataFrame:
    """
    Executes the KMeans and IsolationForest anomaly detectors on the selected columns of the dataframe.
    The anomaly scores are saved to a CSV file and also returned as a DataFrame.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        columns (list): List of columns to use for detection.
        result_csv_path (str): File path to save the results CSV.

    Returns:
        pd.DataFrame: DataFrame containing anomaly scores from both detectors.
                    Columns: 'kmeans_anomaly_score' and 'isolation_forest_anomaly_score'.
    """

    dwt_csv_path = f"{result_csv_path}dwt.csv"

    # Instantiate the detectors
    dwt_detector = DWTMLEADAnomalyDetector()

    # Convert the selected columns to a numpy array for processing
    data_values = df[columns].values

    # Execute the detection algorithms
    anomaly_scores_dwt = dwt_detector.detect(data_values)

    dwt_df = pd.DataFrame({'dwt_anomaly_score': pd.Series(anomaly_scores_dwt)})
    dwt_df.to_csv(dwt_csv_path, index=False)

    return dwt_df

def _local_peak_window(f, y, peak_idx, frac=0.5, max_bins=300):
    """
    Build a contiguous window around the peak where y >= frac*peak.
    Ensures at least ~a dozen bins when possible, and caps at max_bins.
    """
    peak = y[peak_idx]
    if peak <= 0 or np.isnan(peak):
        return slice(peak_idx, peak_idx+1)

    thresh = frac * peak
    left = peak_idx
    while left > 0 and y[left-1] >= thresh:
        left -= 1
    right = peak_idx
    n = len(y)
    while right < n-1 and y[right+1] >= thresh:
        right += 1

    # pad a bit beyond half-max to give the fit some tails
    pad = min(10, peak_idx-left, right-peak_idx)
    left = max(0, left - pad)
    right = min(n-1, right + pad)

    # ensure not overly large
    if right - left + 1 > max_bins:
        half = max_bins // 2
        left = max(0, peak_idx - half)
        right = min(n-1, left + max_bins - 1)
    return slice(left, right+1)

def _fwhm_sigma(f, y):
    """
    Estimate FWHM in Hz using linear interpolation around half max,
    then convert to sigma via FWHM = 2*sqrt(2*ln2)*sigma.
    Returns (sigma_est, fwhm_est) or (None, None) if not possible.
    """
    if len(f) < 3:
        return None, None
    peak_idx = int(np.nanargmax(y))
    ymax = y[peak_idx]
    if not np.isfinite(ymax) or ymax <= 0:
        return None, None
    half = 0.5 * ymax

    # search left crossing
    i = peak_idx
    while i > 0 and y[i] > half:
        i -= 1
    if i == 0 and y[i] > half:
        return None, None
    # linear interp left
    x1, y1 = f[i], y[i]
    x2, y2 = f[i+1], y[i+1]
    if y2 == y1:
        return None, None
    f_left = x1 + (half - y1) * (x2 - x1) / (y2 - y1)

    # search right crossing
    j = peak_idx
    n = len(y)
    while j < n-1 and y[j] > half:
        j += 1
    if j == n-1 and y[j] > half:
        return None, None
    # linear interp right
    x1, y1 = f[j-1], y[j-1]
    x2, y2 = f[j], y[j]
    if y2 == y1:
        return None, None
    f_right = x1 + (half - y1) * (x2 - x1) / (y2 - y1)

    fwhm = float(max(f_right - f_left, 0.0))
    if fwhm <= 0 or not np.isfinite(fwhm):
        return None, None
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return sigma, fwhm

def stable_gaussian_stddev(fft_bins, norm_power_values, peak_idx, peak_frequency):
    """
    Returns a stable estimate of sigma (stddev in Hz) for a peak:
    1) primary: FWHM-derived sigma (no fitting, robust)
    2) optional refinement: bounded curve_fit in a local window
    """
    f = np.asarray(fft_bins)
    y = np.asarray(norm_power_values)
    # Guard rails
    mask = np.isfinite(f) & np.isfinite(y)
    f, y = f[mask], y[mask]
    if len(f) < 5:
        return np.nan

    # Build a local window around the peak (half-max region)
    win = _local_peak_window(f, y, peak_idx, frac=0.5, max_bins=300)
    fx, yx = f[win], y[win]

    # FWHM -> sigma (robust to baseline & neighbors)
    sigma0, fwhm0 = _fwhm_sigma(fx, yx)
    if sigma0 is None or not np.isfinite(sigma0):
        return np.nan

    # If you just want robustness, stop here:
    # return float(sigma0)

    # Otherwise, refine with bounded fit using sigma0 as init
    def g(x, amp, stddev):
        return gaussian_fixed_mean(x, amp, stddev, mean_fixed=peak_frequency)

    amp0 = float(np.nanmax(yx))
    # Reasonable bounds: amp >= 0, stddev in [bin_width, span/2]
    uniq = np.unique(fx)
    bin_step = np.median(np.diff(uniq)) if len(uniq) > 1 else 1.0
    span = float(np.nanmax(fx) - np.nanmin(fx)) if len(fx) else 1.0
    lower = [0.0, max(bin_step, 1e-6)]
    upper = [np.inf, max(5*bin_step, span/2 if span > 0 else np.inf)]
    p0 = [amp0, float(np.clip(sigma0, lower[1], upper[1]))]

    try:
        popt, _ = curve_fit(g, fx, yx, p0=p0, bounds=(lower, upper), maxfev=20000)
        return float(popt[1])
    except Exception:
        # Fall back to FWHM estimate if fitting struggles
        return float(sigma0)

# -----------------------------
# Configuration: Set Source and Target Folders
# -----------------------------
SOURCE_FOLDER = r"F:\Cogni2\output"  # <-- Update this path
TARGET_FOLDER = r"F:\Cogni2\measure"  # <-- Update this path

# SOURCE_FOLDER = r"C:\Users\Marcian\Downloads\Beschichtung 2"    # <-- Update this path
# TARGET_FOLDER = r"C:\Users\Marcian\Downloads\Beschichtung 2 Results"    # <-- Update this path

# Create the target folder if it doesn't exist.
if not os.path.isdir(TARGET_FOLDER):
    os.makedirs(TARGET_FOLDER)

# Global list to accumulate output variables for each CSV file.
output_variables_list = []


# -----------------------------
# Function to process a single CSV file
# -----------------------------
def process_csv_file(source_csv_path, target_csv_dir, target_data_path, target_plot_path):
    """
    Process a single CSV file:
      - Reads header rows and data (using converters to handle both '.' and ',' as decimals).
      - Converts all data to float64.
      - Replaces any infinite values with 9999999999.9.
      - Drops rows where a solitary infinity occurs in any column.
      - Converts time from ms to s if needed.
      - Calculates resistance.
      - Replaces NaN or negative resistance values with 0.
      - Applies pre-processing on the resistance signal:
           1. Outlier removal (replace outliers with the median)
           2. Smoothing (moving average)
           3. Windowing (Hann window)
      - Calculates sampling frequency and performs FFT analysis.
      - Performs Gaussian fitting using a fixed mean equal to the peak frequency.
      - Saves an FFT analysis plot in the corresponding target directory.
      - Returns output variables (Peak Power, Peak Frequency, Gaussian Standard Deviation)
        or records an error message.
    """
    csv_filename = os.path.basename(source_csv_path)
    try:

        # Read CSV data starting from Row 4.
        df = pd.read_csv(source_csv_path, sep=';')

        resistance_col = df.columns[1]
        time_col = df.columns[0]

        resistance = df[resistance_col].values
        # --- Pre-processing on Resistance Signal ---
        resistance = replace_outliers_with_median(resistance, m=3.0)
        #print("SMOPOTH")
        resistance = smooth_data(resistance, window_size=5)
        window = get_window('hann', len(resistance))
        windowed_resistance = resistance * window
        # Update resistance column.
        resistance_col = "Resistance_(Ohms)"
        df[resistance_col] = resistance

        threshold = 0.6

        print("Running Analysis on", csv_filename)
        # Run the detectors and store results
        # dwt = run_dwt_detector(df, [resistance_col], target_csv_dir)
        #counter = run_spucker_detectors(df, [resistance_col], 0.6)
        #df_kmeans, score = run_anomaly_detectors(df, [resistance_col], target_data_path)
        # Create and save a plot for one set of anomaly scores (for example, using KMeans scores)
        # You could call this function separately for each set of results if needed.
        # create_and_save_anomaly_plot(df, df_kmeans["kmeans_anomaly_score"].values, 0.6, target_plot_path)

        # avg_tsad = tsad["kmeans_anomaly_score"].mean()
        # avg_tsad_anomaly = tsad.loc[
        #    tsad["tsad_anomaly_score"] > threshold, "tsad_anomaly_score"].mean()

        # Compute averages for IsolationForest
        #avg_kmeans = df_kmeans["kmeans_anomaly_score"].mean()
        #avg_kmeans_anomaly = df_kmeans.loc[df_kmeans["kmeans_anomaly_score"] > threshold, "kmeans_anomaly_score"].mean()

        # avg_isolation = df_isolation["isolation_forest_anomaly_score"].mean()
        # avg_isolation_anomaly = df_isolation.loc[
        #    df_isolation["isolation_forest_anomaly_score"] > threshold, "isolation_forest_anomaly_score"].mean()

        # Prepare output entry (note: no Gaussian Mean column now).
        # Calculate sampling frequency using time values.

        Fs = calculate_Fs(df[time_col].values)
        print("Sampling Rate", Fs)
        # Perform FFT analysis on the pre-processed (windowed) resistance signal.
        power_values, fft_bins, _ = fftit(windowed_resistance, Fs, 100, 3000)#3000

        norm_power_values = normalize_array(power_values)
        window_size = 5
        norm_power_values = moving_average(norm_power_values, window_size)

        peak_index = np.argmax(norm_power_values)
        if peak_index <= 0:
            raise ValueError("No valid peak found in FFT analysis.")
        peak_frequency = fft_bins[peak_index]
        peak_power = power_values[peak_index]

        plt.figure(figsize=(10, 6))

        plt.plot(fft_bins, norm_power_values, color=IOTcolors['grey1'], linewidth=2, label='FFT')
        plot_filename = os.path.splitext(csv_filename)[0] + "_FFT_Analysis2.png"
        target_plot_path = os.path.join(target_csv_dir, plot_filename)
        plt.savefig(target_plot_path)
        plt.close()

        print("Peak power", peak_power, " Index:", peak_index, " Freq", peak_frequency, "Len", )

        # For Gaussian fitting, use the peak frequency as the fixed mean.
        # Define a Gaussian function with fixed mean.
        def gaussian_fixed(x, amp, stddev):
            return gaussian_fixed_mean(x, amp, stddev, peak_frequency)

        gauss_stddev = 1

        # Initial guess: amplitude and stddev.
        initial_guess = [np.max(norm_power_values), 1]
        print("INITIAL STD", gauss_stddev)
        popt, _ = curve_fit(gaussian_fixed, fft_bins[(fft_bins >= (peak_frequency - peak_frequency * 0.8)) &
                                                     (fft_bins <= (peak_frequency + peak_frequency * 0.8))],
                            norm_power_values[(fft_bins >= (peak_frequency - peak_frequency * 0.8)) &
                                              (fft_bins <= (peak_frequency + peak_frequency * 0.8))],
                            p0=initial_guess, maxfev=10000)


        # 1) Build a data-driven fit window around the peak
        #win = _local_peak_window(fft_bins, norm_power_values, peak_index, frac=0.5, max_bins=300)
        #fx = fft_bins[win]
        #yx = norm_power_values[win]

        # 2) Get an initial sigma from FWHM (robust, no fit needed)
        #sigma0, fwhm0 = _fwhm_sigma(fx, yx)
        #if sigma0 is None or not np.isfinite(sigma0):
            # Fallback: small multiple of bin step
        #    uniq = np.unique(fx)
        #    bin_step = np.median(np.diff(uniq)) if len(uniq) > 1 else 1.0
        #    sigma0 = 3.0 * bin_step

        # 3) Prepare bounded curve fit with reasonable limits
        #def g(x, amp, stddev):
        #    return gaussian_fixed_mean(x, amp, stddev, mean_fixed=peak_frequency)

        #amp0 = float(np.nanmax(yx))
        #uniq = np.unique(fx)
        #bin_step = np.median(np.diff(uniq)) if len(uniq) > 1 else 1.0
        #span = float(np.nanmax(fx) - np.nanmin(fx)) if len(fx) else 1.0

        #lower = [0.0, max(bin_step, 1e-6)]
        #upper = [np.inf, max(5 * bin_step, span / 2 if span > 0 else np.inf)]
        #p0 = [amp0, float(np.clip(sigma0, lower[1], upper[1]))]

        # 4) (Optional) weight by an estimated noise level near the window edges
        #    Here we use a simple constant sigma from the lower quantiles.
        #noise_level = np.quantile(yx, 0.1)
        #sigma_weights = None
        #if np.isfinite(noise_level) and noise_level > 0:
        #    sigma_weights = np.full_like(yx, noise_level)

        # Perform the fit
        #popt, _ = curve_fit(g, fx, yx, p0=p0, bounds=(lower, upper), sigma=sigma_weights, maxfev=20000)

        gauss_stddev = popt[1]
        print("STDDEV", gauss_stddev)
        # Here, we set Gaussian mean to the peak frequency.
        gauss_mean = peak_frequency

        # Plot FFT analysis and Gaussian fit.
        plt.figure(figsize=(10, 6))

        # Increase line thickness using linewidth (lw)
        plt.plot(fft_bins, norm_power_values, color=IOTcolors['grey1'], linewidth=2, label='FFT')

        x_fit = np.linspace(0, peak_frequency + 1000, 500)
        y_fit = gaussian_fixed(x_fit, *popt)
        plt.plot(x_fit, y_fit, color=IOTcolors['red'], linewidth=3, label='Gaussian Fit')  # Thicker line

        plt.axvline(x=gauss_mean, color=IOTcolors['blue'], linestyle='--', linewidth=2,
                    label='Fixed Mean (Peak Frequency)')
        plt.axvline(x=gauss_mean + gauss_stddev, color=IOTcolors['gold'], linestyle=':', linewidth=2,
                    label='Mean + Std Dev')
        plt.axvline(x=gauss_mean - gauss_stddev, color=IOTcolors['gold'], linestyle=':', linewidth=2,
                    label='Mean - Std Dev')

        plt.title(f'FFT Analysis for {csv_filename}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Normalized Power')
        plt.legend()
        plt.grid()

        plt.show()

        # Save the plot.
        plot_filename = os.path.splitext(csv_filename)[0] + "_FFT_Analysis.png"
        target_plot_path = os.path.join(target_csv_dir, plot_filename)
        plt.savefig(target_plot_path)
        plt.close()
        print(f"Processed and saved plot for {csv_filename}")

        # Prepare output entry (note: no Gaussian Mean column now).
        output_entry = {
            "CSV File Name": csv_filename,
            "Peak Power": peak_power,
            "Peak Frequency": peak_frequency,
            "Gaussian Standard Deviation": gauss_stddev,
            "Number of KMeans Anomalies": score,
            "Average KMeans": avg_kmeans,
            "AVG KMeans + Threshold": avg_kmeans_anomaly,
            "Spucker": counter,
            "Errors": ""
        }
        output_variables_list.append(output_entry)
    except Exception as e:
        error_msg = f"{csv_filename}: {str(e)}"
        print(f"Error processing {csv_filename}: {error_msg}")
        output_entry = {
            "CSV File Name": csv_filename,
            "Peak Power": "",
            "Peak Frequency": "",
            "Gaussian Standard Deviation": "",
            "Number of KMeans Anomalies": "",
            "Average KMeans": "",
            "AVG KMeans + Threshold": "",
            "Spucker": "",
            "Errors": error_msg
        }
        output_variables_list.append(output_entry)


# -----------------------------
# Function to recursively process folders and CSV files
# -----------------------------
def process_folders(source_folder, target_folder):
    """
    Recursively processes all CSV files in the source folder and saves the results
    in the target folder with the same sub-folder structure.
    """
    for root, dirs, files in os.walk(source_folder):
        rel_path = os.path.relpath(root, source_folder)
        target_dir = os.path.join(target_folder, rel_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for file in files:
            if file != 'TS_PL_Test_1.csv':
                continue
            if file.lower().endswith('_1.csv'):
                source_csv_path = os.path.join(root, file)
                last_folder = os.path.basename(os.path.dirname(source_csv_path))
                if "_10s" not in last_folder:
                    continue
                # If the plot already exists, mark as already processed.
                plot_filename = os.path.splitext(file)[0] + "_FFT_Analysis.png"
                data_filename = os.path.splitext(file)[0] + "_FFT_Analysis_"
                target_plot_path = os.path.join(target_dir, plot_filename)
                target_data_path = os.path.join(target_dir, data_filename)
                process_csv_file(source_csv_path, target_dir, target_data_path, target_plot_path)


# Define custom colors using RGB tuples normalized to [0, 1]
IOTcolors = {
    'black': (0 / 255, 0 / 255, 0 / 255),
    'blue': (0 / 255, 82 / 255, 156 / 255),
    'gold': (246 / 255, 168 / 255, 0 / 255),
    'red': (204 / 255, 0 / 255, 0 / 255),
    'grey5': (242 / 255, 242 / 255, 242 / 255),
    'grey4': (217 / 255, 217 / 255, 217 / 255),
    'grey3': (191 / 255, 191 / 255, 191 / 255),
    'grey2': (166 / 255, 166 / 255, 166 / 255),
    'grey1': (127 / 255, 127 / 255, 127 / 255)
}


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == '__main__':
    process_folders(SOURCE_FOLDER, TARGET_FOLDER)
    # Save output variables to a single Excel file.
    output_df = pd.DataFrame(output_variables_list,
                             columns=[
                                 "CSV File Name", "Peak Power", "Peak Frequency",
                                 "Gaussian Standard Deviation", "Number of KMeans Anomalies", "Average KMeans",
                                 "AVG KMeans + Threshold", "Spucker", "Errors"
                             ])
    output_excel_path = os.path.join(TARGET_FOLDER, "output_variables.xlsx")
    output_excel_path_csv = os.path.join(TARGET_FOLDER, "output_variables.csv")
    output_df.to_excel(output_excel_path, index=False)
    output_df.to_csv(output_excel_path_csv, index=False)
    print("Processing complete.")
    print(f"Output variables saved to {output_excel_path}.")