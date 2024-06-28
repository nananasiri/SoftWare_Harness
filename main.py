import os
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import csv
# from IPython.display import display, Markdown

# Fetch all directories matching the pattern
# dirs = glob('*/*/Fidget_Ball')

# Use a more flexible pattern to match both 'Fidget_Ball' and 'Fidget Ball'
dirs = []
for pattern in ['*/*/Fidget_Ball', '*/*/Fidget Ball']:
    dirs.extend(glob(pattern))
print(dirs)

# Check if directories are found
if not dirs:
    print("No directories found matching the patterns.")
    exit()

# Initialize variables
analyses = {
    'Analysis 1': 'moderate_intense_fidgeter',
    'Analysis 2': 'infrequent_frequent_fidgeter',
    'Analysis 3': 'light_heavy_fidgeter',
    'Analysis 4': 'min_max_intensity_fidgeter',
    'Analysis 5': 'fidgeting_nonfidgeting',
    'Analysis 6': 'inactive_active_fidgeter'
}

# Ask the user which analysis to analyze


print("Select a feature exctractor you wish to employ:")
for analysis in analyses.values():
    print(f"- {analysis}")

selected_analysis = input("Please enter the name of the analysis you want to analyze: ")

# Ensure the selected analysis is valid
if selected_analysis not in analyses.values():
    print(f"{selected_analysis} is not a valid analysis.")
    exit()

# Ask the user to enter the file patterns they want to analyze
event_patterns_input = input("Enter the event file patterns ending with .csv (comma-separated, e.g., *event3_3.csv): ")
patterns = event_patterns_input.split(',')

# Initialize variables
sensor_sums = {i: 0 for i in range(6)}
sensor_counts = {i: 0 for i in range(6)}
deleted_files = []
threshold = 7
min_val_on_bar = 0

# Dictionary to hold cumulative counts for the selected analysis
folder_counts = {}
file_sums = []


def read_csv_file(file_path):
    with open(file_path, 'r') as file:
        # Use csv.Sniffer to detect the delimiter
        try:
            dialect = csv.Sniffer().sniff(file.read(2048))  # Read a larger sample size if files are large or varied
            file.seek(0)  # Reset file read position
            df = pd.read_csv(file, delimiter=dialect.delimiter)
        except csv.Error:
            # Fallback if Sniffer fails, you can customize this as needed
            print(f"Could not determine delimiter automatically for {file_path}.")
            return None
    return df


# Process the specified files
for pattern in patterns:
    pattern_files_found = False
    for dir_path in dirs:
        files_path = glob(os.path.join(dir_path, pattern))
        # Commenting out the line as requested
        # if not files_path:
        #     print(f"No files found for pattern: {pattern} in {dir_path}")
        #     continue

        pattern_files_found = True
        for file_path in files_path:
            folder_name = os.path.basename(os.path.dirname(dir_path))

            if folder_name not in folder_counts:
                folder_counts[folder_name] = {
                    'sum': {i: 0 for i in range(6)},
                    'count': {i: 0 for i in range(6)},
                    'total_sum': 0,
                    'total_count': 0,
                    'line_count_above_threshold': 0,
                    'line_count_below_threshold': 0,
                    'sensor_values_above_threshold': [],
                    'fidget_sequence': [],
                    'analysis_name': selected_analysis  # Assign the analysis name
                }

            try:
                # Read the CSV file
                df = read_csv_file(file_path)
                print(df.shape)
                #                 # After reading the CSV file
                #                 df = pd.read_csv(file_path, sep=' ', header=None)
                #                 print(df.shape)

                #                 # Check if the DataFrame has at least 6 columns (index 5)
                if df.shape[1] < 6:
                    print(f"File {file_path} does not contain enough columns.")
                    continue  # Skip to the next file




            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

            # Process each row
            for idx, row in df.iterrows():
                above_threshold = any(row[i] > threshold for i in range(5, 11))
                if above_threshold:
                    folder_counts[folder_name]['fidget_sequence'].append('x')
                    folder_counts[folder_name]['line_count_above_threshold'] += 1
                else:
                    folder_counts[folder_name]['fidget_sequence'].append('o')
                    folder_counts[folder_name]['line_count_below_threshold'] += 1

            # Process each sensor column (columns 5 to 10)
            for i in range(5, 11):
                try:
                    if pd.api.types.is_numeric_dtype(df.iloc[:, i]):
                        # Filter values greater than the threshold
                        sensor_data_above = df[df.iloc[:, i] > threshold]
                        sensor_count_above = len(sensor_data_above)
                        sensor_sum = sensor_data_above.iloc[:, i].sum()

                        # Update the sum and count for the sensor
                        folder_counts[folder_name]['sum'][i - 5] += sensor_sum
                        folder_counts[folder_name]['count'][i - 5] += sensor_count_above
                        folder_counts[folder_name]['total_sum'] += sensor_sum
                        folder_counts[folder_name]['total_count'] += sensor_count_above

                        # Append sensor values above threshold to list
                        folder_counts[folder_name]['sensor_values_above_threshold'].extend(
                            sensor_data_above.iloc[:, i].values)
                    else:
                        pass
                except Exception as e:
                    print(f"Error processing sensor data in file {file_path}: {e}")
                    continue

            # Record the sum for the current file
            file_sums.append({'File': file_path, 'Sum': round(sensor_sum, 2)})

    if not pattern_files_found:
        print(f"No files found for pattern: {pattern}")

# Ensure there is data for the selected analysis
if not folder_counts:
    print(f"No data found for {selected_analysis}")
    exit()

print(f"Data collected for analysis: {selected_analysis}")
print(f"Number of folders with data: {len(folder_counts)}")

# Create OUTPUT directory if it doesn't exist
output_dir = 'OUTPUT'
os.makedirs(output_dir, exist_ok=True)

# Define output file path based on the analysis name
output_file_path = os.path.join(output_dir, f'{selected_analysis}.csv')


# Function definitions for each analysis

def moderate_intense_fidgeter(selected_data, output_file_path):
    # Calculate the average values for the folder and sort by overall average
    folder_averages = []
    zero_average_folders = []
    for folder, data in selected_data.items():
        overall_average = round(data['total_sum'] / data['total_count'], 2) if data['total_count'] > 0 else 0
        if overall_average == 0:
            zero_average_folders.append(folder)
        folder_averages.append((folder, overall_average))

    # Sort folders by overall average in ascending order
    folder_averages.sort(key=lambda x: x[1])

    # Prepare data for CSV output
    output_data = []
    for folder, overall_average in folder_averages:
        if overall_average > 0:
            folder_data = {'Folder': folder, 'Overall_Average': round(overall_average, 2)}
            for sensor, total_sum in selected_data[folder]['sum'].items():
                count = selected_data[folder]['count'][sensor]
                average = round(total_sum / count, 2) if count > 0 else 0
                folder_data[f'Sensor_{sensor}_Average'] = round(average, 2)
                folder_data[f'Sensor_{sensor}_Sum'] = round(total_sum, 2)
                folder_data[f'Sensor_{sensor}_Count'] = count
            output_data.append(folder_data)

    # Convert the data to a DataFrame and save as CSV
    df_output = pd.DataFrame(output_data)
    df_output.to_csv(output_file_path, index=False)

    print("#" * 10, "Average of sensor values above threshold (magnitudes/Count) {Moderate to Intense Fidgeter}",
          "#" * 10)

    # Print the sorted average values for the folder
    print("Subjects sorted by overall average value:")
    for folder, overall_average in folder_averages:
        if overall_average > 0:
            print(f"Participant: {folder}")
            for sensor, total_sum in selected_data[folder]['sum'].items():
                count = selected_data[folder]['count'][sensor]
                average = round(total_sum / count, 2) if count > 0 else 0
            print(f"  Overall Average for {folder}: {round(overall_average, 2)}")

    # Plotting the bar with folder positions
    fig, ax = plt.subplots(figsize=(10, 2))

    # Create a colorful bar
    cmap = plt.get_cmap('rainbow')
    norm = plt.Normalize(threshold, 450)
    color_bar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    color_bar.set_array([])

    # Draw the color bar
    cb = plt.colorbar(color_bar, orientation='horizontal', ax=ax, pad=0.4)
    cb.set_label('Overall Average')

    # Plot folder positions
    for folder, overall_average in folder_averages:
        if overall_average > 0:
            ax.plot(overall_average, 0.5, 'o', label=folder)

    # Set limits and labels
    ax.set_xlim((min_val_on_bar), 450)
    ax.set_yticks([])
    ax.set_xlabel('Overall Average Value')

    # Add threshold label
    ax.text(min_val_on_bar, 0.6, f'Threshold = {threshold}', color='black', ha='center')

    # Place the color bar below the x-axis
    cb.ax.set_position([0.125, -0.15, 0.775, 0.03])

    # Adjust the legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.title('Moderate <- (Identifying Moderate and Intense Fidgeting Based on Overall Average Value) -> Intense')
    plt.show()

    # Print the folders with zero average
    print("Folders with zero average value:")
    for folder in zero_average_folders:
        print(f"  {folder}")


def infrequent_frequent_fidgeter(selected_data, output_file_path):
    # Calculate zero_average_folders
    zero_average_folders = [folder for folder, data in selected_data.items() if data['total_count'] == 0]

    # Additional Plot: Frequency of Lines Above Threshold
    fig, ax = plt.subplots(figsize=(10, 5))

    print("#" * 20, "Counting Lines above threshold -> Infrequent, Frequent", "#" * 20)

    # Calculate the frequency categories
    frequencies = []
    for folder, data in selected_data.items():
        if folder not in zero_average_folders:  # Exclude folders with zero average
            frequency_category = 'Frequent' if data['line_count_above_threshold'] > threshold else 'Infrequent'
            frequencies.append((folder, data['line_count_above_threshold'], frequency_category))

    # Sort by line count
    frequencies.sort(key=lambda x: x[1], reverse=True)

    # Extract data for plotting
    folders, line_counts, categories = zip(*frequencies) if frequencies else ([], [], [])

    # Prepare data for CSV output without the 'Category' column
    output_data = [{'Folder': folder, 'Line_Count': line_count} for folder, line_count, category in frequencies]

    # Convert the data to a DataFrame and save as CSV
    df_output = pd.DataFrame(output_data)
    df_output.to_csv(output_file_path, index=False)

    # Debugging output
    print("Frequencies (Folder, Line Count, Category):")
    for item in frequencies:
        print(item)

    # Plot the frequency data
    colors = ['green' if category == 'Frequent' else 'red' for category in categories]
    if folders:
        ax.barh(folders, line_counts, color=colors)
    ax.set_xlabel('Line Count Above Threshold')
    ax.set_ylabel('Folders')
    ax.set_title('Frequency of Lines Above Threshold (Infrequent vs. Frequent)')

    # Add a legend
    frequent_patch = plt.Line2D([0], [0], color='green', lw=4, label='Frequent')
    # infrequent_patch = plt.Line2D([0], [0], color='red', lw=4, label='Infrequent')
    # ax.legend(handles=[frequent_patch, infrequent_patch])
    ax.legend(handles=[frequent_patch])

    plt.show()


def light_heavy_fidgeter(selected_data, output_file_path):
    # Calculate zero_average_folders
    zero_average_folders = [folder for folder, data in selected_data.items() if data['total_count'] == 0]

    # New Task: Sum up the magnitudes (sensor values) above threshold: Light to Heavy User for Fidgeting
    fig, ax = plt.subplots(figsize=(10, 5))

    print("#" * 20, "Sum of Magnitudes above threshold -> Light, Medium, Heavy Users", "#" * 20)

    # Calculate the sum of magnitudes for each folder
    magnitude_sums = []
    for folder, data in selected_data.items():
        if folder not in zero_average_folders:  # Exclude folders with zero average
            total_sum = data['total_sum']
            magnitude_sums.append((folder, round(total_sum, 2)))

    # Sort by total sum
    magnitude_sums.sort(key=lambda x: x[1], reverse=True)

    # Extract data for plotting
    folders, total_sums = zip(*magnitude_sums) if magnitude_sums else ([], [])

    # Define categories based on the sum of magnitudes
    categories = []
    for total_sum in total_sums:
        if total_sum <= 1000:
            categories.append('Light')
        elif total_sum <= 3000:
            categories.append('Medium')
        else:
            categories.append('Heavy')

    # Plot the magnitude sums data
    colors = ['blue' if category == 'Light' else 'orange' if category == 'Medium' else 'red' for category in categories]
    if folders:
        ax.barh(folders, total_sums, color=colors)
    ax.set_xlabel('Total Sum of Magnitudes Above Threshold')
    ax.set_ylabel('Folders')
    ax.set_title('Sum of Magnitudes Above Threshold (Light to Heavy Users)')

    # Add a legend
    light_patch = plt.Line2D([0], [0], color='blue', lw=4, label='Light')
    medium_patch = plt.Line2D([0], [0], color='orange', lw=4, label='Medium')
    heavy_patch = plt.Line2D([0], [0], color='red', lw=4, label='Heavy')
    ax.legend(handles=[light_patch, medium_patch, heavy_patch])

    # Prepare data for CSV output without the 'Category' column
    output_data = [{'Folder': folder, 'Total_Sum': total_sum} for folder, total_sum in zip(folders, total_sums)]

    # Convert the data to a DataFrame and save as CSV
    df_output = pd.DataFrame(output_data)
    df_output.to_csv(output_file_path, index=False)

    plt.show()


def min_max_intensity_fidgeter(selected_data, output_file_path):
    # Calculate zero_average_folders
    zero_average_folders = [folder for folder, data in selected_data.items() if data['total_count'] == 0]

    # New Task: Maximum and Minimum Intensity
    print("#" * 20, "Maximum and Minimum Intensity", "#" * 20)

    # Calculate the top 10% and bottom 10% average values for each folder
    intensity_data = []
    for folder, data in selected_data.items():
        if folder not in zero_average_folders:
            sensor_values = data['sensor_values_above_threshold']
            if sensor_values:
                sensor_values = sorted(sensor_values)
                top_10_percent = sensor_values[int(0.9 * len(sensor_values)):]
                bottom_10_percent = sensor_values[:int(0.1 * len(sensor_values))]

                top_10_avg = np.mean(top_10_percent) if top_10_percent else 0
                bottom_10_avg = np.mean(bottom_10_percent) if bottom_10_percent else 0

                intensity_data.append({
                    'Folder': folder,
                    'Top_10_Avg': round(top_10_avg, 2),
                    'Bottom_10_Avg': round(bottom_10_avg, 2)
                })

    # Convert the data to a DataFrame and save as CSV
    df_output = pd.DataFrame(intensity_data)
    df_output.to_csv(output_file_path, index=False)

    # Print the intensity data
    for item in intensity_data:
        print(f"Participant: {item['Folder']}")
        print(f"  Top 10% Average: {item['Top_10_Avg']:.2f}")
        print(f"  Bottom 10% Average: {item['Bottom_10_Avg']:.2f}")


def fidgeting_nonfidgeting(selected_data, output_file_path):
    # Calculate zero_average_folders
    zero_average_folders = [folder for folder, data in selected_data.items() if data['total_count'] == 0]

    # New Task: Count the lines below threshold and count the lines above threshold -> Compute Quantity of Fidgeting vs Not Fidgeting
    print("#" * 20, "Count of Fidgeting vs Not Fidgeting", "#" * 20)

    fidgeting_data = []
    # Calculate the counts for each folder
    for folder, data in selected_data.items():
        fidgeting_data.append({
            'Folder': folder,
            'Lines_Above_Threshold': data['line_count_above_threshold'],
            'Lines_Below_Threshold': data['line_count_below_threshold']
        })

    # Convert the data to a DataFrame and save as CSV
    df_output = pd.DataFrame(fidgeting_data)
    df_output.to_csv(output_file_path, index=False)

    # Print the fidgeting data
    for item in fidgeting_data:
        print(f"Participant: {item['Folder']}")
        print(f"  Lines Above Threshold (Fidgeting): {item['Lines_Above_Threshold']}")
        print(f"  Lines Below Threshold (Not Fidgeting): {item['Lines_Below_Threshold']}")


def inactive_active_fidgeter(selected_data, output_file_path):
    # Calculate zero_average_folders
    zero_average_folders = [folder for folder, data in selected_data.items() if data['total_count'] == 0]

    # New Task: Inactive to Active Fidgeter - Fidgeting Sequence Analysis
    print("#" * 20, "Inactive to Active Fidgeter - Fidgeting Sequence Analysis", "#" * 20)

    def calculate_sequence_lengths(sequence, char):
        return [len(list(group)) for key, group in itertools.groupby(sequence) if key == char]

    fidgeting_sequences = []
    for folder, data in selected_data.items():
        if folder not in zero_average_folders:
            fidget_sequence = data['fidget_sequence']
            x_lengths = calculate_sequence_lengths(fidget_sequence, 'x')
            o_lengths = calculate_sequence_lengths(fidget_sequence, 'o')

            avg_x_length = np.mean(x_lengths) if x_lengths else 0
            avg_o_length = np.mean(o_lengths) if o_lengths else 0

            fidgeting_sequences.append((folder, round(avg_x_length, 2), round(avg_o_length, 2)))

    # Sort by average length of x (fidgeting) sequences
    fidgeting_sequences.sort(key=lambda x: x[1], reverse=True)

    # Prepare data for CSV output without the 'Category' column
    sequence_data = [{'Folder': folder, 'Avg_Fidget_Length': avg_x_length, 'Avg_NonFidget_Length': avg_o_length}
                     for folder, avg_x_length, avg_o_length in fidgeting_sequences]

    # Convert the data to a DataFrame and save as CSV
    df_output = pd.DataFrame(sequence_data)
    df_output.to_csv(output_file_path, index=False)

    # Print the fidgeting sequence analysis
    print("Fidgeting Sequence Analysis (Inactive to Active Fidgeter):")
    for item in sequence_data:
        print(
            f"Participant: {item['Folder']}, Average Length of x (Fidgeting): {item['Avg_Fidget_Length']:.2f}, Average Length of o (Not Fidgeting): {item['Avg_NonFidget_Length']:.2f}")

    # Plot the fidgeting sequence analysis
    folders, avg_x_lengths, avg_o_lengths = zip(*fidgeting_sequences) if fidgeting_sequences else ([], [], [])

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.35
    index = np.arange(len(folders))

    bars1 = ax.bar(index, avg_x_lengths, bar_width, label='Average Length of (Fidgeting)', color='blue')
    bars2 = ax.bar(index + bar_width, avg_o_lengths, bar_width, label='Average Length of (Not Fidgeting)',
                   color='orange')

    ax.set_xlabel('Participant')
    ax.set_ylabel('Average Length')
    ax.set_title('Fidgeting Sequence Analysis (Inactive to Active Fidgeter)')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(folders, rotation=90)
    ax.legend()

    plt.show()


# Execute the selected analysis function
if selected_analysis == analyses['Analysis 1']:
    moderate_intense_fidgeter(folder_counts, output_file_path)
elif selected_analysis == analyses['Analysis 2']:
    infrequent_frequent_fidgeter(folder_counts, output_file_path)
elif selected_analysis == analyses['Analysis 3']:
    light_heavy_fidgeter(folder_counts, output_file_path)
elif selected_analysis == analyses['Analysis 4']:
    min_max_intensity_fidgeter(folder_counts, output_file_path)
elif selected_analysis == analyses['Analysis 5']:
    fidgeting_nonfidgeting(folder_counts, output_file_path)
elif selected_analysis == analyses['Analysis 6']:
    inactive_active_fidgeter(folder_counts, output_file_path)

print(f"Analysis results have been saved to {output_file_path}")
# *event3_2.csv,*event3_3.csv,*event3_4.csv,*event3_5.csv