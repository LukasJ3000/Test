import csv
import os
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helper Functions
# -----------------------------

def drop_solitary_infinity_rows(df, inf_value=9999999999.9): #Ergibt eigentlich keinen sinn, weil inf werte in der CSV als merkwüridge sonderzeichen auftauchen bzw. Nan werden #inf_value definieren oder die funktion rausschmeißen
    """
    For each column in the DataFrame, drop any row that contains a solitary infinity value (inf_value),
    i.e. the value equals inf_value and both the previous and next rows are not inf_value.
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
    Converter for pd.read_csv: if x is a string, replace comma with dot, then convert to float.
    """
    try:
        if isinstance(x, str):
            return float(x.replace(',', '.'))
        return float(x)
    except:
        return np.nan


def calculate_Fs(time_vector):
    """
    Calculate sampling frequency (Fs) from a 1D time vector (in seconds).
    Fs = 1 / (average time difference)
    """
    time_diff = np.diff(time_vector)
    avg_interval = np.mean(time_diff)
    if avg_interval <= 0 or np.isnan(avg_interval):
        raise ValueError("Time interval is non-positive or NaN. Check your time data.")
    return 1.0 / avg_interval


def merge3(vec1, vec2, ovrlpSlope): # hier gibt es irgendwie merge1, merge2 und merge3?
    """
    Merge two vectors by overlapping them (merging vec2 into vec1) while ensuring
    that the slope (direction) at the merging point is the same in both vectors.

    Parameters:
        vec1: array-like
            The first vector.
        vec2: array-like
            The second vector.
        ovrlpSlope: int
            The number of elements used to define the slope.

    Returns:
        vecMerged: np.ndarray
            The merged vector.
    """
    # Ensure inputs are numpy arrays
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    # Determine scaling factor sigN based on the magnitude of vec1's maximum
    max_val_scaled = abs(np.max(vec1)) * 100
    if max_val_scaled > 100:
        sigN = 100
    elif max_val_scaled > 10:  # implicitly < 100
        sigN = 1000
    elif max_val_scaled > 1:  # implicitly < 10
        sigN = 10000
    else:
        sigN = 100000

    #Scaling für resistance überprüfen!

    # Compute rounded/truncated versions of the vectors.
    # If ovrlpSlope is zero, take the whole vector; otherwise, exclude the last ovrlpSlope elements.
    if ovrlpSlope > 0:
        rvec1 = np.floor(vec1[:-ovrlpSlope] * sigN)
        rvec2 = np.floor(vec2[:-ovrlpSlope] * sigN)
    else:
        rvec1 = np.floor(vec1 * sigN)
        rvec2 = np.floor(vec2 * sigN)

    # Only check intersections in the last part of vec1 (as large as rvec2)
    if len(rvec2) > 0:
        rvec1 = rvec1[-len(rvec2):]

    # Calculate index offset to map indices in rvec1 back to indices in vec1
    vec1IdxOffset = len(vec1) - len(rvec1) - ovrlpSlope

    # Find indices in rvec2 whose values also occur in rvec1.
    # This is analogous to Matlab's [~,~,rib] = intersect(rvec1, rvec2)
    rib = np.where(np.isin(rvec2, rvec1))[0].tolist()

    maxIdx1 = None
    minIdx2 = None
    flag = True
    while flag:
        if not rib:
            # No intersection found: return vec1 unmerged.
            vecMerged = vec1
            flag = False
        else:
            # Get the smallest index in rvec2 where there is an intersection.
            minIdx2 = min(rib)
            common_val = rvec2[minIdx2]
            # Find the last occurrence in rvec1 of this common value.
            indices = np.where(rvec1 == common_val)[0]
            if len(indices) == 0:
                # In theory this should not happen, but if it does, remove the index and continue.
                rib.remove(minIdx2)
                continue
            rmaxIdx1 = indices[-1]
            # Check if merging would yield additional datapoints.
            if rmaxIdx1 + len(rvec2) - minIdx2 > len(rvec1):
                # Map index from rvec1 back to the original vec1.
                maxIdx1 = rmaxIdx1 + vec1IdxOffset

                # Define slices for slope checking in vec1 and vec2.
                # Ensure we do not exceed array bounds.
                end1 = maxIdx1 + ovrlpSlope if maxIdx1 + ovrlpSlope < len(vec1) else len(vec1)
                end2 = minIdx2 + ovrlpSlope if minIdx2 + ovrlpSlope < len(vec2) else len(vec2)

                # Compute the slope direction: mean of the next ovrlpSlope points minus the current value.
                vec1Up = (np.mean(vec1[maxIdx1:end1]) - vec1[maxIdx1]) > 0 if end1 - maxIdx1 > 0 else False
                vec2Up = (np.mean(vec2[minIdx2:end2]) - vec2[minIdx2]) > 0 if end2 - minIdx2 > 0 else False

                if vec1Up == vec2Up:
                    # Merge the two vectors.
                    # In Matlab: [vec1(1:maxIdx1); vec2((minIdx2+1):end)]
                    # In Python, note that slicing is 0-indexed and end-exclusive.
                    vecMerged = np.concatenate((vec1[:maxIdx1 + 1], vec2[minIdx2 + 1:]))
                    flag = False
                else:
                    # Slope directions differ: remove the current candidate and try the next.
                    rib.remove(minIdx2)
            else:
                # The condition is not met; remove the candidate index and try again.
                rib.remove(minIdx2)

    return vecMerged, maxIdx1, minIdx2

def merge1(vec1, vec2, ovrlpSlope):
    """
    Merge two vectors by overlapping them.
    Merges vec2 into vec1, checking that the overlapping portions
    have the same slope (direction) over a window of length ovrlpSlope.

    Parameters:
        vec1, vec2 : array-like
            Input vectors.
        ovrlpSlope : int
            Number of elements used to define the slope in the overlapping region.

    Returns:
        vecMerged : numpy.ndarray
            The merged vector.
    """
    # Convert inputs to numpy arrays
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    # Compute scaling factor sigN based on abs(max(vec1))*100
    max_val_scaled = np.abs(np.max(vec1)) * 100
    if max_val_scaled > 100:
        sigN = 100
    elif max_val_scaled < 100 and max_val_scaled > 10:
        sigN = 1000
    elif max_val_scaled < 10 and max_val_scaled > 1:
        sigN = 10000
    else:
        sigN = 100000

    # Trim each vector by excluding the last ovrlpSlope elements,
    # then scale and floor them.
    rvec1 = np.floor(vec1[:-ovrlpSlope] * sigN) if ovrlpSlope > 0 else np.floor(vec1 * sigN)
    rvec2 = np.floor(vec2[:-ovrlpSlope] * sigN) if ovrlpSlope > 0 else np.floor(vec2 * sigN)

    # Use only the last part of rvec1, with the same length as rvec2.
    len_rvec2 = len(rvec2)
    rvec1 = rvec1[-len_rvec2:]

    # Compute offset to map indices from rvec1 back to full vec1.
    vec1IdxOffset = len(vec1) - len(rvec1) - ovrlpSlope

    # Find indices in rvec2 that also occur in rvec1.
    rib = np.nonzero(np.isin(rvec2, rvec1))[0].tolist()

    # Initialize outputs.
    vecMerged = None
    maxIdx1 = None
    minIdx2 = None
    flag = True

    while flag:
        if len(rib) == 0:
            # No intersection found: return vec1 unchanged.
            vecMerged = vec1.copy()
            flag = False
        else:
            # Choose the smallest index candidate from rvec2.
            minIdx2_candidate = min(rib)
            # Find the last occurrence of rvec2[minIdx2_candidate] in rvec1.
            indices = np.where(rvec1 == rvec2[minIdx2_candidate])[0]
            if len(indices) == 0:
                rib.remove(minIdx2_candidate)
                continue
            rmaxIdx1 = indices[-1]  # 0-indexed position in rvec1

            # Ensure that there are enough data points after the merging point.
            if (rmaxIdx1 + 1 + (len(rvec2) - minIdx2_candidate)) > len(rvec1):
                # Calculate corresponding index in the full vec1.
                maxIdx1 = rmaxIdx1 + vec1IdxOffset
                minIdx2 = minIdx2_candidate
                # Adjust for Python slicing (end index is non-inclusive).
                maxIdx1_py = maxIdx1 + 1

                # Compute local slopes for the next ovrlpSlope+1 elements.
                vec1_slice = vec1[maxIdx1: maxIdx1 + ovrlpSlope + 1]
                vec2_slice = vec2[minIdx2: minIdx2 + ovrlpSlope + 1]

                slope1 = (np.mean(vec1_slice) - vec1[maxIdx1]) if len(vec1_slice) > 0 else 0
                slope2 = (np.mean(vec2_slice) - vec2[minIdx2]) if len(vec2_slice) > 0 else 0

                vec1Up = slope1 > 0
                vec2Up = slope2 > 0

                if vec1Up == vec2Up:
                    # Merge the vectors.
                    vecMerged = np.concatenate((vec1[:maxIdx1_py], vec2[minIdx2 + 1:]))
                    flag = False
                else:
                    # Remove the current candidate and continue searching.
                    rib.remove(minIdx2_candidate)
            else:
                rib.remove(minIdx2_candidate)

    return vecMerged, maxIdx1, minIdx2

def merge2(vec1, vec2, ovrlpSlope=15000):
    """
    Merge two vectors by overlapping them, merging vec2 into vec1.

    Parameters:
    vec1 (numpy array): First vector
    vec2 (numpy array): Second vector
    ovrlpSlope (int): Number of elements to define the slope

    Returns:
    numpy array: Merged vector
    """
    # Determine significant figure scaling
    max_val = abs(np.max(vec1)) * 100

    if max_val > 100:
        sigN = 100
    elif 10 < max_val <= 100:
        sigN = 1000
    elif 1 < max_val <= 10:
        sigN = 10000
    else:
        sigN = 100000

    # Round vectors and truncate by ovrlpSlope
    rvec1 = np.floor(vec1[:-ovrlpSlope] * sigN)
    rvec2 = np.floor(vec2[:-ovrlpSlope] * sigN)

    # Consider only the last part of vec1, same length as rvec2
    rvec1 = rvec1[-len(rvec2):]
    vec1IdxOffset = len(vec1) - len(rvec1) - ovrlpSlope  # Offset for index alignment

    # Find intersection indices
    common, idx1, rib = np.intersect1d(rvec1, rvec2, return_indices=True)

    while True:
        if len(rib) == 0:
            return vec1, None, None  # Return vec1 if no intersections exist

        minIdx2 = rib.min()
        rminEntr2 = rvec2[minIdx2]
        common_val = rvec2[minIdx2]
        # Find the last occurrence in rvec1 of this common value.
        indices = np.where(rvec1 == common_val)[0]
        if len(indices) == 0:
            return vec1, None, None
        rmaxIdx1 = indices[-1]      # Corresponding index in rvec1
        maxIdx1 = rmaxIdx1 + vec1IdxOffset

        # Check slope consistency in the overlap region
        vec1Up = np.mean(vec1[maxIdx1:maxIdx1 + ovrlpSlope]) - vec1[maxIdx1] > 0
        vec2Up = np.mean(vec2[minIdx2:minIdx2 + ovrlpSlope]) - vec2[minIdx2] > 0

        if vec1Up == vec2Up:
            return np.concatenate((vec1[:maxIdx1 + 1], vec2[minIdx2 + 1:])), maxIdx1, minIdx2
        else:
            rib = np.delete(rib, np.argmin(rib))  # Remove smallest index and retry

def calculate_resistance(voltage, current):
    """
    Calculate resistance from voltage and current arrays.
    If the result is 2D, return the first column.
    """
    voltage = np.asarray(voltage)
    current = np.asarray(current)
    if np.all(voltage <= 0):
        voltage = np.abs(voltage)
    if np.all(current <= 0):
        current = np.abs(current)
    res = np.divide(voltage, current, out=np.zeros_like(voltage), where=(current > 0))

    return res if res.ndim == 1 else res[:, 0]


def get_order_from_filename(file):
    """
    Extracts the order number from the filename.
    Assumes the filename is in the format: ..._<order>.csv, e.g., TS_PL_123_1.csv.
    Returns the integer order. If conversion fails, returns 0.
    """
    base = os.path.splitext(os.path.basename(file))[0]
    parts = base.split('_')
    try:
        return int(parts[-1])
    except:
        return 0

def movmedian(x, win): #convolve vs rolling... besser wir haben dann nur eine funktion
    """
    MATLAB smoothdata(...,'movmedian', win) equivalent for 1D arrays.
    Uses pandas rolling median with center=True.
    """
    x = np.asarray(x, dtype=float)
    # pandas handles NaNs gracefully; backfill edges to mimic MATLAB's centered window
    s = pd.Series(x)
    y = s.rolling(window=win, center=True, min_periods=1).median().to_numpy()
    return y

def first_intersect_info(vec1, vec2, sigN=100.0): #Warum wird das gebraucht? Ist das nicht in den merge 1, 2, 3 schon drin?
    """
    Replicates the MATLAB 'intersect' rounding logic:
      rvec1 = floor(vec1*sigN); rvec2 = floor(vec2*sigN);
      [~,~,rib] = intersect(rvec1, rvec2); minIdx2 = min(rib); minEntr2 = rvec2(minIdx2)
    Returns (minIdx2, minEntr2, rvec1, rvec2).
    """
    rvec1 = np.floor(np.asarray(vec1) * sigN)
    rvec2 = np.floor(np.asarray(vec2) * sigN)

    # intersect that returns indices into rvec2 (rib in MATLAB is indices in the second array)
    common, idx1, idx2 = np.intersect1d(rvec1, rvec2, return_indices=True)
    if len(idx2) == 0:
        return None, None, rvec1, rvec2
    minIdx2 = int(idx2.min())
    minEntr2 = rvec2[minIdx2]
    return minIdx2, minEntr2, rvec1, rvec2

def merge_experiment_csvs(source_path, destination_file):
    """
    Merges all CSV files for a particular experiment into one large CSV file.

    Process:
      1. Recursively find all CSV files under source_path.
      2. Sort files based on the order number extracted from their filenames.
      3. For each CSV:
           - Read the CSV file.
           - Identify time, voltage, and current columns.
           - Compute resistance using calculate_resistance() and add it as new attribute "Resistance (Ohms)".
           - Drop all attributes except the first (time) and "Resistance (Ohms)".
           - Plot the resistance time-series for this CSV file.
      4. Iteratively merge the CSV files based on overlapping segments in the "Resistance (Ohms)" attribute:
           - Shift the new CSV's time column by adding the last time of the merged data so that time is continuous.
           - Use merge2 on the "Resistance (Ohms)" attribute arrays.
           - Print the row numbers where the merge occurred.
      5. After each merge, display a debug plot of the merged resistance time-series.
      6. Save the final merged DataFrame to destination_file.
    """
    # Gather all CSV file paths.
    csv_files = []
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.lower().endswith('.csv') and "merged" not in file:
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print("No CSV files found in source path.")
        return

    # Sort files based on the order number (last part of the filename).
    csv_files.sort(key=get_order_from_filename)

    data_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, sep=";")
            # Identify time, voltage, and current columns.
            time_col = df.columns[0]
            voltage_col = next((col for col in df.columns
                                if "volt" in col.lower() or "spannung" in col.lower() or "voltage" in col.lower()), None)
            current_col = next((col for col in df.columns
                                if "current" in col.lower() or "ampere" in col.lower() or "amperage" in col.lower() or "strom" in col.lower()), None)
            if time_col is None or voltage_col is None or current_col is None:
                print(f"Time, voltage or current column not found in {file}. Skipping.")
                continue

            # Compute resistance and add as a new attribute.
            df["Resistance (Ohms)"] = calculate_resistance(df[voltage_col], df[current_col])
            # Drop all attributes except time and "Resistance (Ohms)"
            df = df[[time_col, "Resistance (Ohms)"]]

            start_time = df[time_col].iloc[0]
            data_list.append((file, start_time, df))

            if DEBUG:
                print(f"Read {file} | Order: {get_order_from_filename(file)} | Start time: {start_time:.6f} s | Rows: {len(df)}")
                plt.figure()
                plt.plot(df[time_col], df["Resistance (Ohms)"], label=os.path.basename(file))
                plt.xlabel("Time (s)")
                plt.ylabel("Resistance (Ohms)")
                plt.title(f"Resistance Time Series - {os.path.basename(file)}")
                plt.legend()
                plt.show()
        except Exception as ex:
            print(f"Error reading {file}: {ex}")

    # Sort the list based on the order extracted from filename.
    data_list.sort(key=lambda x: get_order_from_filename(x[0]))

    # Start with the first CSV.
    merged_df = data_list[0][2]
    time_col = merged_df.columns[0]  # time column remains the first
    res_col = "Resistance (Ohms)"
    merged_time = merged_df[time_col].values
    merged_res = merged_df[res_col].values

    if DEBUG:
        plt.figure()
        plt.plot(merged_time, merged_res, label=os.path.basename(data_list[0][0]))
        plt.xlabel("Time (s)")
        plt.ylabel("Resistance (Ohms)")
        plt.title("Initial Resistance Time Series")
        plt.legend()
        plt.show()

    # Iteratively merge subsequent CSV files.
    for i in range(1, len(data_list)):
        file, start_time, df_next = data_list[i]
        # Shift df_next's time column so that time is continuous.
        shift = merged_time[-1]
        df_next[time_col] = df_next[time_col] + shift
        next_time = df_next[time_col].values
        next_res = df_next[res_col].values

        if DEBUG:
            print(f"Merging file {file} | Order: {get_order_from_filename(file)} | Original start: {start_time:.6f} s, shifted start: {df_next[time_col].iloc[0]:.6f} s")

        # Calculate sampling frequency from the merged time array.
        Fs = calculate_Fs(merged_time)  # Assumes calculate_Fs() is defined elsewhere.
        if DEBUG:
            print(f"Calculated Fs: {Fs:.2f} Hz")

        merged_res_s = movmedian(merged_res, win=50)
        next_res_s = movmedian(next_res, win=50)

        # Merge based on the "Resistance (Ohms)" attribute.
        # Assumes merge2() is defined to merge two arrays given their sampling frequency.
        merged_res_new, maxIdx1, minIdx2 = merge2(merged_res_s, next_res_s)


        if maxIdx1 is None or minIdx2 is None:
            print("No valid overlap found. Appending file without merge.")
            merged_df = pd.concat([merged_df, df_next], ignore_index=True)
            if DEBUG:
                print("Merge row indices: (None, None)")
        else:
            if DEBUG:
                print(f"Merging at merged_df row {maxIdx1} and file {file} row {minIdx2}")
            #new_part = df_next.iloc[minIdx2 + 1:].reset_index(drop=True)
            #merged_df = pd.concat([merged_df.iloc[:maxIdx1], new_part], ignore_index=True)
            merged_time_new = np.concatenate((merged_time[:maxIdx1 + 1], next_time[minIdx2 + 1:]))
            dt = 1.0 / Fs
            dt = 0.000002
            merged_time_new = merged_time[0] + np.arange(len(merged_res_new)) * dt
            # Create a new DataFrame with the merged values.
            merged_df = pd.DataFrame({time_col: merged_time_new, res_col: merged_res_new})

        merged_time = merged_df[time_col].values
        merged_res = merged_df[res_col].values

        if DEBUG:
            plt.figure()
            plt.plot(merged_time, merged_res, label=f"Merged Resistance after {i + 1} files")
            plt.xlabel("Time (s)")
            plt.ylabel("Resistance (Ohms)")
            plt.title("Merged Resistance Time Series")
            plt.legend()
            plt.show()

    merged_df.to_csv(destination_file, index=False, sep=';', float_format='%.8f')
    print(f"Merged CSV saved to {destination_file}")

###############

def fix_header_name(header_token):
    """
    Adjust header attribute names based on keywords.
    - 'zeit' becomes 'Time'
    - any token containing 'volt' or 'spannung' becomes 'Voltage'
    - any token containing 'current', 'ampere', 'amperage', or 'strom' becomes 'Current'
    Otherwise, returns the stripped token.
    """
    token = header_token.strip()
    lower_token = token.lower()
    if "zeit" in lower_token:
        return "Time"
    elif "volt" in lower_token or "spannung" in lower_token:
        return "Voltage"
    elif any(word in lower_token for word in ["current", "ampere", "amperage", "strom"]):
        return "Current"
    else:
        return token


def merge_headers(header_line1, header_line2):
    """
    Merge two header lines and apply attribute renaming.

    Example:
      header_line1: "Zeit,Pyrometer1,Pyrometer2,Current,Acoustic Noise,Voltage"
      header_line2: "(s),(°C),(°C),(A),(mV),(V)"

    The function will rename 'Zeit' to 'Time', replace any attribute containing
    'volt' or 'spannung' with 'Voltage', and any attribute containing
    'current|ampere|amperage|strom' with 'Current'.

    Returns a merged header like:
      "Time (s),Pyrometer1 (°C),Pyrometer2 (°C),Current (A),Acoustic Noise (mV),Voltage (V)"
    """
    delimiter = ',' if ',' in header_line1 else ';'
    tokens1 = header_line1.split(delimiter)
    tokens2 = header_line2.split(delimiter)

    merged = []
    for t1, t2 in zip(tokens1, tokens2):
        fixed_t1 = fix_header_name(t1)
        unit = t2.strip()
        if unit:
            merged.append(f"{fixed_t1} {unit}")
        else:
            merged.append(fixed_t1)
    return ';'.join(merged)


def fix_decimal_format_line(line, time_index=None):
    """
    Fix a numeric row where numbers are split into two tokens by commas.

    The function first replaces any occurrence of the infinity symbol ("∞")
    with "9999999999,9". Then, it splits the row by commas, and processes each
    pair of tokens (integer and decimal parts). If a pair corresponds to the time
    attribute (indicated by time_index), the number is divided by 1000 to convert
    from milliseconds to seconds. The final values are joined with semicolons.

    Example:
      Input: "0,00000400,60,94189000,73,08021000,154,54780000,∞"
      (if time_index==0 and the time column is in ms, then the first value is converted)

      After replacement, the tokens for the time column (first two tokens) are merged,
      converted, and formatted. Other columns are simply merged with a dot.
    """
    # Replace infinity symbol with the large number representation.
    # line = line.replace("∞", "9999999999,9")
    line = line.replace("∞", "0,0")
    if ";" in line:
        return line.replace(",", ".")
    tokens = line.split(',')
    if len(tokens) % 2 != 0:
        raise ValueError("Numeric row does not contain an even number of comma-separated tokens.")

    fixed_numbers = []
    num_columns = len(tokens) // 2
    for col in range(num_columns):
        i = col * 2
        integer_part = tokens[i].strip()
        decimal_part = tokens[i + 1].strip()
        full_number_str = f"{integer_part}.{decimal_part}"

        if time_index is not None and col == time_index:
            # Convert the time value from ms to s.
            original_val = float(full_number_str)
            converted = original_val / 1000
            # Preserve the number of decimals of the original decimal part.
            dp = len(decimal_part)
            new_value_str = f"{converted:.{dp}f}"
            fixed_numbers.append(new_value_str)
        else:
            if float(full_number_str) < 0:
                fixed_numbers.append("0.0")
            else:
                fixed_numbers.append(full_number_str)

    return ';'.join(fixed_numbers)


def fix_file(input_filename, output_filename):
    """
    Reads the dirty file, merges the first two header rows, skips any empty rows,
    fixes the numeric rows by converting the split numbers (comma-separated)
    into proper numbers (with dot as decimal separator and semicolon as field separator),
    and writes the result to a new file.

    The header is merged as: "Zeit (s),Pyrometer1 (°C),..." (comma separated),
    while the numeric rows are fixed to have values like "0.00000400;60.94189000;..."
    """

    # Read
    with open(input_filename, 'r', encoding='utf-8') as infile:
        lines = [line.rstrip('\n') for line in infile if line.strip()]

    # Merge the first two header lines with renaming.
    merged_header = merge_headers(lines[0], lines[1])

    # Check for the time attribute unit in the merged header.
    header_tokens = merged_header.split(';')
    time_index = None
    for i, token in enumerate(header_tokens):
        token_lower = token.lower()
        # If the token starts with "time" and contains "ms", we assume it's in milliseconds.
        if token_lower.startswith("time") and "ms" in token_lower:
            time_index = i
            # Update the header to reflect the conversion (now in seconds).
            header_tokens[i] = "Time (s)"
            break
    merged_header = ';'.join(header_tokens)

    # Process the remaining lines (data rows).
    fixed_data_lines = []
    for line in lines[2:]:
        if line.strip():
            fixed_line = fix_decimal_format_line(line, time_index=time_index)
            fixed_data_lines.append(fixed_line)

    # Write back
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write(merged_header + "\n")
        for data_line in fixed_data_lines:
            outfile.write(data_line + "\n")

def clean(files, path):
    for file in files:
        print("Starting" , file)
        last_folder = os.path.basename(os.path.dirname(file))
        file_name = os.path.basename(file)
        output_filename = os.path.join(path, last_folder, file_name)
        if os.path.exists(output_filename):
            continue
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        fix_file(file, output_filename)

def merge(files, path):
    completed_folders = []
    merged_files = []
    for file in files:
        last_folder = os.path.basename(os.path.dirname(file))
        if last_folder in completed_folders:
            continue
        completed_folders.append(last_folder)
        file_name, file_ext = os.path.splitext(os.path.basename(file))
        file_name = file_name[:-2]
        output_folder = os.path.join(path, last_folder)
        merged_filename = os.path.join(output_folder, f"{file_name}_merged{file_ext}")
        merged_files.append(merged_filename)
        if os.path.exists(merged_filename):
           continue

        merge_experiment_csvs(output_folder, merged_filename)
    return merged_files


def split_file(file, output_folder, base_file_name):
    df = pd.read_csv(file, sep=";")
    time_column = df.columns[0]

    #df[time_column] = pd.to_numeric(df[time_column], errors='coerce')

    min_time = df[time_column].min()
    max_time = df[time_column].max()
    print(min_time, max_time)
    i = 1
    while min_time < max_time:
        end_time = min_time + 10

        if max_time - min_time < 10:
            end_time = max_time

        segment = df[(df[time_column] >= min_time) & (df[time_column] < end_time)]
        #print(segment)
        if not segment.empty:
            output_file = os.path.join(output_folder, f"{base_file_name}_{i}.csv")
            segment.to_csv(output_file, index=False, sep=';', float_format='%.8f')
            print(f"Saved segment {i} to {output_file}")

        min_time = end_time
        i += 1


def split(files):
    for file in files:
        last_folder = os.path.basename(os.path.dirname(file))  # Get the last folder name
        file_name, file_ext = os.path.splitext(os.path.basename(file))  # Extract file name and extension

        # Determine the parent directory of the current folder
        parent_dir = os.path.dirname(os.path.dirname(file))

        # Create the new folder with "_10s" appended
        output_folder = os.path.join(parent_dir, f"{last_folder}_10s")
        if os.path.exists(output_folder):
            continue

        os.makedirs(output_folder, exist_ok=True)  # Ensure the new folder is created

        split_file(file, output_folder, file_name.replace("_merged", ""))

def get_all_files(path, min_name=""):
    """
    Recursively get all CSV files under a directory, skipping:
      - files alphabetically lower than 'min_name'
      - files containing 'merged' in the name (case-insensitive)

    Args:
        path (str): Base directory path
        min_name (str): Lowest filename (case-insensitive) to include alphabetically

    Returns:
        list[str]: Full paths of matching files
    """
    file_list = []
    min_name_lower = min_name.lower()

    for root, _, filenames in os.walk(path):
        for file in filenames:
            file_lower = file.lower()

            # Skip if file name (excluding extension) is alphabetically before min_name
            if file_lower < min_name_lower:
                continue

            # Skip if not a CSV or contains 'merged'
            if not file_lower.endswith('.csv') or "merged" in file_lower:
                continue

            file_list.append(os.path.join(root, file))

    return file_list

def preprocessing(input_path, output_path):
    files = get_all_files(input_path)
    print("Starting Cleaning the Data!")
    clean(files, output_path)
    print("Starting Merging the Data!")
    merged_files = merge(files, output_path)
    print("Spliting Merged Files!")
    split(merged_files)

# -----------------------------
# Main Execution
# -----------------------------
SOURCE_PATH = os.path.join(os.path.dirname(__file__), "input") #"TS_PL_118"
DESTINATION_FILE = os.path.join(os.path.dirname(__file__), "output")
DEBUG = False # Will Show Merge Pictures for verification if True

if __name__ == "__main__":
    preprocessing(SOURCE_PATH, DESTINATION_FILE)
