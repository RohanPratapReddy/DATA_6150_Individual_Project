import requests
import json
import re
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def fetch_acndata(api_key, site, start_time=None, end_time=None, timeseries=False):
    """
    Fetch JSON data from the ACN-Data API for the specified site and time range.

    Parameters:
    - api_key (str): API key for authentication.
    - site (str): Site ID (e.g., "caltech", "jpl", "office001").
    - start_time (datetime): Start time as a datetime object.
    - end_time (datetime): End time as a datetime object.
    - timeseries (bool): Whether to fetch time series data (default: False).

    Returns:
    - dict: Parsed JSON data from the API.
    """
    # Validate site
    if site not in {"caltech", "jpl", "office001"}:
        raise ValueError("Invalid site. Valid options are 'caltech', 'jpl', 'office001'.")

    # Base URL
    base_url = f"https://ev.caltech.edu/api/v1/sessions/{site}"
    if timeseries:
        base_url += "/ts/"

    # Create the "where" clause for filtering
    where_clause = (
        f'connectionTime >= "{start_time.strftime("%a, %d %b %Y %H:%M:%S GMT")}" '
        f'and connectionTime <= "{end_time.strftime("%a, %d %b %Y %H:%M:%S GMT")}"'
    )

    # Query parameters
    params = {
        "where": where_clause,
        "sort": "connectionTime",
        "pretty": "true",  # Make the response JSON human-readable
        "max_results": 100  # Adjust as needed
    }

    # Authorization headers
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        if response.content:
            print(f"Response content: {response.content.decode()}")
        return None

def json_load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def repair_json(json_string):
    """
    Repairs common issues in JSON strings using regex.
    """
    # Remove trailing commas (inside objects or arrays)
    json_string = re.sub(r",\s*([\]}])", r"\1", json_string)

    # Ensure proper closing of JSON structure
    open_braces = json_string.count("{")
    close_braces = json_string.count("}")
    open_brackets = json_string.count("[")
    close_brackets = json_string.count("]")

    # Add missing braces or brackets
    if open_braces > close_braces:
        json_string += "}" * (open_braces - close_braces)
    if open_brackets > close_brackets:
        json_string += "]" * (open_brackets - close_brackets)

    # Replace single quotes with double quotes for proper JSON format
    json_string = re.sub(r"(?<!\\)'", r'"', json_string)

    # Escape unescaped quotes inside strings
    json_string = re.sub(r'(?<=:)\s*"([^"]*?)"(?!:)', r'"\1"', json_string)

    # Ensure keys in JSON are properly quoted
    json_string = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1 "\2":', json_string)

    return json_string

def repair_and_parse_json(file_path):
    """
    Attempts to repair a malformed JSON file and parse it.
    """
    with open(file_path, 'r') as file:
        raw_data = file.read()

    # Repair the JSON string
    repaired_data = repair_json(raw_data)

    try:
        # Attempt to parse the repaired JSON
        parsed_data = json.loads(repaired_data)
        print("JSON parsed successfully!")
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"Failed to parse repaired JSON: {e}")
        with open("repaired_file.json", "w") as repaired_file:
            repaired_file.write(repaired_data)
        print("Repaired JSON saved to 'repaired_file.json' for manual inspection.")
        return None

def json_extract_acndata_with_user_inputs(data):
    main_items = data['_items']
    expanded_data = []

    for item in main_items:
        base_data = {key: value for key, value in item.items() if key != 'userInputs'}
        user_inputs = item.get('userInputs', [])
        if user_inputs:
            # Duplicate base_data for each user input
            for input_entry in user_inputs:
                combined_data = {**base_data, **input_entry}  # Merge dictionaries
                expanded_data.append(combined_data)
        else:
            # Include rows without user inputs as is
            expanded_data.append(base_data)
    return expanded_data

def json_extract_acndata_without_user_inputs(data):
    """
    Extract the main items and include only specific fields.
    """
    # Define the desired fields
    desired_fields = {
        "_id", "stationID", "spaceID", "siteID", "clusterID",
        "connectionTime", "disconnectTime", "doneChargingTime", "kWhDelivered"
    }

    main_items = data["_items"]
    filtered_data = []

    for item in main_items:
        # Filter fields to include only the desired ones
        filtered_item = {key: value for key, value in item.items() if key in desired_fields}
        filtered_data.append(filtered_item)

    return filtered_data

def process_charging_data(df):
    # Fill empty 'doneChargingTime' values with 'disconnectTime' values
    df['doneChargingTime'].fillna(df['disconnectTime'])


    # Calculate new columns
    df['chargingTime'] = (df['doneChargingTime'] - df['connectionTime']).dt.total_seconds()  # in seconds
    df['standbyTime'] = (df['disconnectTime'] - df['doneChargingTime']).dt.total_seconds()  # in seconds
    df['sessionDuration'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds()  # in seconds

    # Avoid division by zero in power calculations
    df['chargingTimeHours'] = (df['chargingTime'] / 3600).replace(0, pd.NA)  # in hours
    df['sessionDurationHours'] = (df['sessionDuration'] / 3600).replace(0, pd.NA)  # in hours

    # Calculate power columns
    df['powerkW'] = df['kWhDelivered'] / df['chargingTimeHours']
    df['minpowerkW'] = df['kWhDelivered'] / df['sessionDurationHours']

    # Convert time columns to HH:MM:SS format
    df['chargingTime'] = pd.to_timedelta(df['chargingTime'], unit='s').apply(lambda x: str(x).split('.')[0])
    df['standbyTime'] = pd.to_timedelta(df['standbyTime'], unit='s').apply(lambda x: str(x).split('.')[0])
    df['sessionDuration'] = pd.to_timedelta(df['sessionDuration'], unit='s').apply(lambda x: str(x).split('.')[0])

    # Return the modified DataFrame
    return df



def getdata_as_df(file_path, user_input=False):
    try:
        data = json_load_data(file_path)
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        print("Error Occured at file: ",file_path)
        print("Attempting to repair and parse the JSON file...")
        data = repair_and_parse_json(file_path)

    if data and isinstance(data, dict) and "_items" in data:
        if user_input:
            df = pd.DataFrame(json_extract_acndata_with_user_inputs(data))
            return df
        else:
            df = pd.DataFrame(json_extract_acndata_without_user_inputs(data))
            return df
    else:
        print("Invalid or unsupported JSON format.")
        return None

def getdf(file_path, user_input=False):
    """
    Converts columns in the DataFrame to the specified data types:
    - '_id': Convert hexadecimal to a numerical string
    - 'clusterID': Convert to string
    - 'connectionTime', 'disconnectTime', 'doneChargingTime': Convert to datetime
    - 'siteID', 'spaceID', 'stationID': Convert to string

    Parameters:
    - file_path: Path to input file.
    - user_input: Boolean flag to include user input data.

    Returns:
    - pd.DataFrame: Processed DataFrame, or None if the data is invalid.
    """
    try:
        df = getdata_as_df(file_path, user_input)
        if df is not None:
            # Convert '_id' from hexadecimal to numerical string
            if '_id' in df.columns:
                df['_id'] = df['_id'].astype(str)
                df['_id'] = df['_id'].apply(lambda x: str(int(x, 16)))

            # Convert specific columns to string
            for col in ['clusterID', 'siteID', 'spaceID', 'stationID']:
                if col in df.columns:
                    df[col] = df[col].astype(str)

            # Convert time columns to datetime
            for col in ['connectionTime', 'disconnectTime', 'doneChargingTime']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

            # Process charging data
            df = process_charging_data(df)

            # Reorder and filter columns
            expected_columns = [
                '_id', 'clusterID', 'siteID', 'stationID', 'spaceID',
                'connectionTime', 'disconnectTime', 'doneChargingTime',
                'chargingTime', 'standbyTime', 'sessionDuration',
                'kWhDelivered', 'powerkW', 'minpowerkW'
            ]
            df = df[[col for col in expected_columns if col in df.columns]]
            return df
        else:
            print(f"Warning: No valid data in {file_path}")
            return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def clean_acndata(df):
    """
    Cleans the ACN-Data DataFrame by handling NaT values in time columns,
    filling missing values in doneChargingTime with disconnectTime, ensuring
    no negative values in time-related or numeric columns, and calculating
    missing values in powerkW. All time deltas are formatted as HH:MM:SS.

    Parameters:
        df (pd.DataFrame): The DataFrame containing ACN-Data.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Define columns for processing
    time_columns = ['connectionTime', 'disconnectTime', 'doneChargingTime']
    numeric_columns = ['kWhDelivered', 'powerkW', 'minpowerkW']
    time_delta_columns = ['chargingTime', 'standbyTime', 'sessionDuration']

    # Convert time columns to datetime
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Fill missing doneChargingTime with disconnectTime
    if 'doneChargingTime' in df.columns and 'disconnectTime' in df.columns:
        df['doneChargingTime'].fillna(df['disconnectTime'], inplace=True)

    # Recalculate and format time delta columns
    if 'connectionTime' in df.columns and 'doneChargingTime' in df.columns:
        df['chargingTime'] = (df['doneChargingTime'] - df['connectionTime']).dt.total_seconds()
    if 'doneChargingTime' in df.columns and 'disconnectTime' in df.columns:
        df['standbyTime'] = (df['disconnectTime'] - df['doneChargingTime']).dt.total_seconds()
    if 'connectionTime' in df.columns and 'disconnectTime' in df.columns:
        df['sessionDuration'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds()

    # Convert negative values in time delta columns to positive and format as HH:MM:SS
    for col in time_delta_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: abs(x) if pd.notna(x) else pd.NA)
            df[col] = df[col].apply(lambda x: pd.Timedelta(seconds=x).components if pd.notna(x) else pd.NA)
            df[col] = df[col].apply(lambda x: f"{int(x.hours):02}:{int(x.minutes):02}:{int(x.seconds):02}" if pd.notna(x) else pd.NA)

    # Convert negative values in other numeric columns to positive
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: abs(x) if pd.notna(x) else pd.NA)

    # Calculate missing powerkW values
    if 'powerkW' in df.columns and 'kWhDelivered' in df.columns and 'chargingTime' in df.columns:
        df['chargingTime'] = df['chargingTime'].apply(lambda x: pd.to_timedelta(f"0 days {x}").total_seconds() if pd.notna(x) else pd.NA)
        df['powerkW'].fillna(
            df.apply(lambda row: row['kWhDelivered'] / (row['chargingTime'] / 3600)
                     if pd.notna(row['kWhDelivered']) and pd.notna(row['chargingTime']) and row['chargingTime'] > 0 else pd.NA, axis=1),
            inplace=True
        )

    # Handle any remaining NaN or problematic values in powerkW
    if 'powerkW' in df.columns:
        df['powerkW'] = df['powerkW'].apply(lambda x: 0 if pd.isna(x) else x)
    df['chargingTime'] = df['chargingTime'].apply(lambda x: abs(x) if pd.notna(x) else pd.NA)
    df['chargingTime'] = df['chargingTime'].apply(lambda x: pd.Timedelta(seconds=x).components if pd.notna(x) else pd.NA)
    df['chargingTime'] = df['chargingTime'].apply(
        lambda x: f"{int(x.hours):02}:{int(x.minutes):02}:{int(x.seconds):02}" if pd.notna(x) else pd.NA)

    return df





def calculate_charging_time_statistics(data):
    # Convert 'chargingTime' to a timedelta type
    data['chargingTime'] = pd.to_timedelta(data['chargingTime'])

    # Calculate the average charging time across all sessions
    average_charging_time = data['chargingTime'].mean()

    # Calculate the average charging time for each station
    station_avg_charging_time = data.groupby('stationID')['chargingTime'].mean()

    # Create a DataFrame for visualization
    station_avg_charging_time_df = station_avg_charging_time.reset_index()

    # Plot the average charging time per station grouped by siteID
    for site_id, site_data in data.groupby('siteID'):
        site_station_avg = site_data.groupby('stationID')['chargingTime'].mean().reset_index()
        site_station_avg['chargingTime'] = site_station_avg['chargingTime'].dt.total_seconds() / 60  # Convert to minutes

        # Sort data in descending order
        site_station_avg = site_station_avg.sort_values(by='chargingTime', ascending=False)

        # Normalize the charging times for color mapping
        norm = plt.Normalize(site_station_avg['chargingTime'].min(), site_station_avg['chargingTime'].max())
        cmap = cm.RdYlGn

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            site_station_avg['stationID'],
            site_station_avg['chargingTime'],
            color=cmap(norm(site_station_avg['chargingTime']))
        )

        # Add a colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, aspect=50)
        cbar.set_label('Average Charging Time (minutes)')

        ax.set_xlabel('Station ID')
        ax.set_ylabel('Average Charging Time (minutes)')
        ax.set_title(f'Average Charging Time Per Station for Site {site_id}')
        ax.set_xticks(range(len(site_station_avg['stationID'])))
        ax.set_xticklabels(site_station_avg['stationID'], rotation=90)
        plt.tight_layout()
        plt.show()

    return average_charging_time, station_avg_charging_time





































