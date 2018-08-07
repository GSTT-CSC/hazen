# mr_coil_analysis
#
# This script analyses the results of the mr_coil_qa script
#
# The script expects to find a csv file that contains data produced by the mr_coil_qa
# script.
#
#
# Neil Heraghty
# neil.heraghty@nhs.net
#
# First version 12/07/2018

import numpy as np
import os
from scipy import signal
from pydicom.filereader import dcmread
import math
import csv
import pandas as pd
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import string


# First, read in the csv file as a pandas dataframe
location = r"E:\Neil's Data\Neil's Work\Medical Physics\Elective\Python\hazen\coil_snr_results_Combined_v2.csv"
df = pd.read_csv(location)

# Sort by Coil Name, Element ID and Slice Location
df_sort = df.sort_values(by=['Coil Name', 'Element ID', 'Slice Location']).reset_index()

# Count the unique coil/element/location combinations
df_org = df.groupby(['Coil Name', 'Element ID', 'Slice Location']).size().reset_index().rename(columns={0:'count'})
combo_count = df_org['count']
combo_coil = df_org['Coil Name']
combo_element = df_org['Element ID']
combo_loc = df_org['Slice Location']

i=0
# Start analysis by down-selecting to a single coil/element/location
for entry in range(len(df_org)):
    j = combo_count[entry]

    # Create a new dataframe using just the combo rows
    df_combo = df_sort[i:(i+j)]

    # Create a unique name
    combo_name = '_Coil_' + combo_coil[entry] + '_Element_' + combo_element[entry] + '_Loc_' + str(combo_loc[entry])
    # Remove illegal filename characters
    valid_chars = "-_.()%s%s" % (string.ascii_letters, string.digits)
    combo_name = ''.join(c for c in combo_name if c in valid_chars)

    # Convert date column to Timestamp format
    df_combo.loc[:, 'Date'] = pd.to_datetime(df_combo.loc[:, 'Date'], format='%Y%m%d')

    # Create plot
    plt_max = max(df_combo['Coil Element SNR'])
    temp_plt = df_combo.plot(x='Date', y='Coil Element SNR', style='.', title=combo_name, legend=False)
    temp_plt.set_ylabel('Coil Element SNR')
    temp_plt.set_ylim(0,(2*plt_max))

    # Save plot to results folder
    parent_path = os.getcwd()
    save_path = parent_path + '\\' + 'plots'
    os.chdir(save_path)
    plot_name = str(combo_name) + '.png'
    plt.savefig(plot_name)
    os.chdir(parent_path)

    # Increment to next unique group
    i = i+j
