import pandas as pd

def eclipse_results_to_df(filepath_no_ext):
    """
    Converts Eclipse simulation results into a pandas DataFrame.
    
    Parameters:
    - filepath_no_ext (str): Path to the Eclipse output file without the file extension.

    Returns:
    - pd.DataFrame: DataFrame containing well-specific simulation results.
    """
    
    # Read the list of keywords from the FSMSPEC file
    keyword_list = read_keyword(filepath_no_ext + ".FSMSPEC", 'KEYWORD')
    
    # Read the list of well names from the FSMSPEC file
    wellname_list_result = read_keyword(filepath_no_ext + ".FSMSPEC", 'WGNAMES')
    
    # Create column names by combining well names and keywords
    columns_list = [wellname_list_result[i] + keyword_list[i] for i in range(len(keyword_list))]
    
    # Read the simulation results (parameter values) from the FUNSMRY file
    final_result_list = read_params(filepath_no_ext + ".FUNSMRY")
    
    # Convert the results into a DataFrame with well names + keywords as column names
    return pd.DataFrame(final_result_list, columns=columns_list)


def read_keyword(filepath, keyword):
    """
    Reads and extracts values associated with a specific keyword from a text file.
    
    Parameters:
    - filepath (str): Path to the file where the keyword is located.
    - keyword (str): The keyword to search for in the file.

    Returns:
    - list: A list of values associated with the specified keyword.
    """
    
    # Open the file and read all lines
    with open(filepath, 'r') as f:
        f_lines = f.readlines()
    
    curr_num = 0    # Current count of values read
    max_num = 1     # Maximum number of values to read
    reached_keyword = False  # Flag to indicate if the keyword has been found
    result_list = []  # List to store the extracted values

    # Loop through all lines in the file
    for i in range(len(f_lines)):
        line = f_lines[i]

        # If the keyword is found in the current line, extract info
        if keyword in line:
            reached_keyword = True
            max_num = int(line.split()[-2])  # Get the number of values to extract
            type = line.split()[-1][1:-1]    # Determine the data type (CHAR)
        
        # Stop if the required number of values has been read
        if curr_num >= max_num:
            break
        # If the keyword has been found, begin extracting values
        elif reached_keyword and keyword not in line:
            if type == 'CHAR':
                # Read 7 values in each line (11 characters per value, with spaces removed)
                for j in range(7):
                    result = line[j*11+2:(j+1)*11-1].replace(" ", "")
                    if result != '':  # Append non-empty results to the list
                        result_list.append(result.replace(":+:+:+:+",""))  # Remove unwanted symbols
                        curr_num += 1  # Increment the count of values read

    return result_list


def read_params(filepath):
    """
    Reads and extracts numerical parameter data from an Eclipse summary file.
    
    Parameters:
    - filepath (str): Path to the file containing parameter data.

    Returns:
    - list: A list of lists, where each sublist contains the parameter values for a specific timestep.
    """
    
    # Open the file and read all lines
    with open(filepath, 'r') as f:
        f_lines = f.readlines()
    
    curr_num = 0    # Current count of values read
    max_num = 1     # Maximum number of values to read
    reached_keyword = False  # Flag to indicate if 'PARAMS' keyword is found
    result_list = []  # Temporary list to store current timestep's values
    final_result_list = []  # List to store all timesteps' values

    # Loop through all lines in the file
    for i in range(len(f_lines)):
        line = f_lines[i]

        # If 'PARAMS' is found, initialize data extraction for a new set of values
        if 'PARAMS' in line:
            curr_num = 0
            reached_keyword = True
            max_num = int(line.split()[-2])  # Get the number of values to extract
            type = line.split()[-1][1:-1]    # Determine the data type (REAL)
        
        # Stop extraction once all required values for the current set are read
        if curr_num >= max_num:
            reached_keyword = False
        # If within the correct section and reading 'REAL' type data, extract numbers
        elif reached_keyword and 'PARAMS' not in line:
            if type == 'REAL':
                splitted_line = line.split()  # Split the line into individual values
                for result in splitted_line:
                    result_list.append(float(result))  # Convert string values to float
                    curr_num += 1  # Increment the count of values read
            
            # If all values for the current set are read, append to final result list
            if curr_num >= max_num:
                final_result_list.append(result_list)
                result_list = []  # Reset for the next set of values

    return final_result_list
