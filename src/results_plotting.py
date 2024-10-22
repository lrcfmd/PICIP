import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np

#data reformat
phase_fields=["MgBOF"]#,"MgAlCu","LiAlBO"]
for field in phase_fields:
    #df=pd.read_csv('../simulation/data/testing/'+field+'_'+str(f)+'.csv')
    df=pd.read_csv('~/phd/PICIP/simulation/data/testing/beta_test.csv')
    for i in df.columns:
        print(i)
    stds=df['Experimental std'].unique()
    pes=df['Predicted error'].unique()
    us=df['Unknown'].unique()
    x=len(pes)*len(stds)*len(us)
    repeats=len(df)/x
    print(df.head())
    print(f'{pes=}, {stds=}, {us=}, {repeats=}')
    #df['Predicted error'].astype('string').hist(bins=100)
    #plt.show()
    #df=df.iloc[300*len(df['Unknown'].unique())]
    #df=df[df['Prior size']==1]
    df_new=pd.DataFrame()
    df_new['Batch size']=df['Batch size']
    df_new['End type']=df['End type']
    df_new['Experimental std']=df['Experimental std']
    df_new['Predicted error']=df['Predicted error']
    df_new['Unknown']=df['Unknown']
    df_new['Initial distance']=df['d(closest,u) for batch0']
    df_new['Score 0']=df['purity for batch0']
    df_new['Score 1']=df['purity for batch1']
    df_new['Score 2']=df['purity for batch2']
    df_new['Score 3']=df['purity for batch3']
    df_new['Score 4']=df['purity for batch4']
    df_new['Score 5']=df['purity for batch5']
    df_new['Score 6']=df['purity for batch6']
    df_new['Score 7']=df['purity for batch7']
    df_new['Score 8']=df['purity for batch8']
    df_new['Score 9']=df['purity for batch9']
    df_new['Score 10']=df['purity for batch10']
    df_new['Score 11']=df['purity for batch11']
    df_new['Score 12']=df['purity for batch12']
    df_new['Score 13']=df['purity for batch13']
    df_new['Score 14']=df['purity for batch14']
    df_new['Score 15']=df['purity for batch15']
    df=df.replace('Mg 6 B 2 O 6 F 6','Mg$_3$B(OF)$_3$')
    df=df.replace('Mg 20 B 12 O 36 F 4','Mg$_5$B$_3$O$_9$F')
    #df_new.loc[(df_new['Score 14']!=df_new['Score 5'])&
    #       (df_new['End type']=='Nulled'),'End type']='Nulled late'
    #print(df_new['End type'].unique())
    #df_new.to_csv(
    #    '../simulation/data/testing/refined/'+field+"_"+str(f)+".csv",
    #    index=False)
    df_new.to_csv(
        '~/phd/PICIP/simulation/data/testing/beta_refined.csv',
        index=False)
'''

# Define the batch size to consider
selected_batch_size = 1

# Define phase fields

#f=0
#field="LiAlBO"
#areas=[0.010855905568422534, 0.06023519825812226, 0.05504769425952735,
#       0.05573254640131034, 0.04648471959693422, 0.029810377908400874]

#field="MgBOF"
#f=3
#areas=[0.21633578830363878, 0.06871842687292054]

field="MgAlCu"
f=0
areas=[0.41849353831792396, 0.5033310039089045, 0.15173949382547855]

#selected_stds=[0.1]
#selected_pes=[0.01,0.02,0.04,0.06,0.08,1]
selected_stds=[0.1,0.05,0.02]
selected_pes=[0.01,0.02,0.04]
selected_bs=[1]

# Read and preprocess data
df = pd.read_csv(f'../simulation/data/testing/refined/{field}_{f}.csv')
df=df[df['Batch size'].isin(selected_bs)]

# This code block is used to plot the score vs number of samples
'''
'''
for unknown_value,area in zip(df['Unknown'].unique(),areas):
    print(unknown_value)
    plt.figure(figsize=(10, 6))

    # Filter the dataframe to remove fails removing failures is imperative
    #for all operations except when calculating failure rate
    dfa=df[df['End type']!="Nulled"]
    dfa.set_index(['Experimental std', 'Predicted error', 'Unknown'], inplace=True)
    # Filter the DataFrame for the current 'Unknown' value
    df_filtered = dfa.loc[(selected_stds, selected_pes, unknown_value), :]

    # Group by 'Experimental std' and 'Predicted error'
    for (std, pe), df_group in df_filtered.groupby(level=['Experimental std', 'Predicted error']):
        print(len(df_group))
        medians = []
        means=[]
        lowers = []
        uppers = []
        samples = range(0, 16)
    
        for sample_number in samples:
            sample_col = f'Score {sample_number}'
            if sample_col in df_group.columns:
                scores = df_group[sample_col]
                means.append(np.mean(scores)*100)
                medians.append(np.median(scores)*100)
                lowers.append(np.percentile(scores, 16)*100)
                uppers.append(np.percentile(scores, 84)*100)
            else:
                print('fail')
                medians.append(np.nan)
                lowers.append(np.nan)
                uppers.append(np.nan)

        medians = np.array(medians)
        lowers = np.array(lowers)
        uppers = np.array(uppers)
        
        plt.errorbar(samples, medians, yerr=[medians - lowers, uppers - medians],
                     label=f'Std={std}, PE={pe}', capsize=5)

    plt.xlabel('Sample number')
    plt.ylabel('Percentage purity of unknown phase /%')
    plt.title(f'Median Score vs Number of samples for Unknown={unknown_value}'
              +f' (Area={area:.3f})')
    plt.legend(title='Parameter Choices')
    plt.grid(True)
    name=("../simulation/new_graphs/"+field+"/"+unknown_value+"_score_vs_n")
    name=re.sub(r'-?\d+\.\d+|-?\d+',
           lambda match: str(round(float(match.group()))), name)
    name=name.replace(" ","")
    print(name)
    plt.savefig(name)
    plt.show()
    '''

#bar chart showing expected score vs initial distance
'''
import itertools
# Iterate over unique values of 'Unknown'
nth_sample=3
for unknown_value,area in zip(df['Unknown'].unique(),areas):
    plt.figure(figsize=(12, 8))
    bar_width = 0.1
    positions = np.arange(10)

    # Filter the dataframe to remove fails
    dfa=df[df['End type']!="Nulled"]
    dfa.set_index(['Experimental std', 'Predicted error', 'Unknown'], inplace=True)

    # Generate all combinations of stds and pes
    combinations = list(itertools.product(selected_stds, selected_pes))

    for idx, (std, pe) in enumerate(combinations):
        # Filter the DataFrame for the current 'Unknown' value and selected std and pe values
        df_filtered = dfa.xs((std, pe, unknown_value), level=['Experimental std', 'Predicted error', 'Unknown'], drop_level=False)

        df_filtered = df_filtered.reset_index()  # Reset index to access 'Initial distance'
    
        # Define bin edges to ensure bins start at 0
        bin_edges = np.linspace(0, df_filtered['Initial distance'].max(), 11)
        
        # Bin the data into 10 bins based on 'Initial distance'
        df_filtered['Distance Bin'] = pd.cut(df_filtered['Initial distance'], bins=bin_edges)
        df_filtered['Distance Bin'] = df_filtered['Distance Bin'].astype('category')  # Ensure it's categorical
    
        # Calculate the expected score after 5 samples for each bin
        expected_scores = df_filtered.groupby(
            'Distance Bin',observed=False)['Score '+str(nth_sample)].mean()
        
        # Count the number of data points in each bin
        bin_counts = df_filtered['Distance Bin'].value_counts(sort=False)
    
        # Plot the bar chart for the current combination
        bars = plt.bar(positions + idx * bar_width, expected_scores, bar_width, label=f'Std={std}, PE={pe}')
        
        # Annotate each bar with the number of data points
        for bar, count in zip(bars, bin_counts):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{count}', va='bottom') # va='bottom' aligns the text just above the bar

    # **Change for X-axis Labels**:
    bin_labels = [f'{interval.left:.2f}-{interval.right:.2f}' for interval in df_filtered['Distance Bin'].cat.categories]
    plt.xlabel('Distance to unknown phase from random initial start')
    plt.ylabel('Mean \"maximum purity\" of unknown phase after 3 samples')
    plt.title(f'Expected Score after three samples for unknown={unknown_value}'
              +f' (Area={area:.3f})')
    plt.xticks(positions + bar_width * (len(combinations) / 2), bin_labels, rotation=45)
    plt.legend(title='Parameter choices')
    plt.grid(True)
    plt.tight_layout()
    name=("../simulation/new_graphs/"+field+"/"+unknown_value+"_score_vs_d")
    name=re.sub(r'-?\d+\.\d+|-?\d+',
           lambda match: str(round(float(match.group()))), name)
    name=name.replace(" ","")
    print(name)
    plt.savefig(name)
    plt.show()
    '''

#This block of code is for plotting a bar chart of the failure rate for each 
#unkown
'''
# Create an empty DataFrame to store the results
results = pd.DataFrame(columns=[
    'Phase Field', 'σ', 'Predicted error', 'Unknown', 'Failure Rate'
])


# Filter the DataFrame to include only selected std and pe values
df = df[df['Experimental std'].isin(selected_stds) & df['Predicted error'].isin(selected_pes)]

# Set the multi-index using 'Experimental std', 'Predicted error', and 'Unknown'
df.set_index(['Experimental std', 'Predicted error', 'Unknown'], inplace=True)

# Group by the multi-index and iterate over the groups
for (std, pe, u), df_group in df.groupby(level=['Experimental std', 'Predicted error', 'Unknown']):
    # Calculate failure rate for each group as the proportion of 'Nulled' entries
    failure_rate = (df_group['End type'] == 'Nulled').mean()*100

    # Create a DataFrame for the new row
    new_row = pd.DataFrame({
        'Phase Field': [field],
        'σ': [std],
        'Predicted error': [pe],
        'Unknown': [u],
        'Failure Rate': [failure_rate]
    })

    # Append the new row to the results DataFrame using pd.concat
    results = pd.concat([results, new_row], ignore_index=True)

# Display the table
print(results)

# Plotting the failure rates
for unknown_value,area in zip(results['Unknown'].unique(),areas):
    plt.figure(figsize=(12, 8))
    bar_width = 0.1
    filtered_results = results[results['Unknown'] == unknown_value]
    positions = np.arange(len(filtered_results))

    bars = plt.bar(positions, filtered_results['Failure Rate'], bar_width, label=f'Unknown={unknown_value}')

    # Annotate each bar with the failure rate
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.2f}', va='bottom')  # va='bottom' aligns the text just above the bar

    # **Change for X-axis Labels**:
    bin_labels = [f'{row["σ"]}, {row["Predicted error"]}' for idx, row in filtered_results.iterrows()]
    plt.xlabel('Parameter choice (Experimental error, Predicted error)')
    plt.ylabel('Percentage failure rate /%')
    plt.title(f'Failure Rate for Unknown={unknown_value} (Area={area:.3f})')
    plt.xticks(positions, bin_labels, rotation=45)
    #plt.legend(title='Combinations')
    plt.grid(True)
    name=("../simulation/new_graphs/"+field+"/"+unknown_value+"_failure_rate")
    name=re.sub(r'-?\d+\.\d+|-?\d+',
           lambda match: str(round(float(match.group()))), name)
    name=name.replace(" ","")
    plt.savefig(name)
    plt.show()
    '''

#Comparison codes

# Define phase fields and corresponding areas
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
areas_dict = {
    "MgBOF": [0.2163, 0.0687],
    "MgAlCu": [0.4185, 0.5033, 0.1517],
    "LiAlBO": [0.0109, 0.0602, 0.0550,0.0557, 0.0465, 0.0298],
}

# File location template
file_template = '../simulation/data/compare_b=1/{}.csv'
#compare area vs optimal predicted error
'''

# Experimental std and predicted error values to compare
exp_std_value = 0.1
pe_values = [0.01, 0.04]

# Create an empty DataFrame to store the results
comparison_results = pd.DataFrame(columns=[
    'Phase Field', 'Unknown', 'Area', 'Score Difference'
])

# Iterate over phase fields
for phase_field in phase_fields:
    # Read the CSV file for the current phase field
    df = pd.read_csv(file_template.format(phase_field))

    # Filter the DataFrame to include only selected batch size and experimental std value
    df = df[(df['Batch size'] == 1) & (df['Experimental std'] == exp_std_value)]

    # Extract unique unknowns
    unknowns = df['Unknown'].unique()

    # Ensure the areas are in the same order as the unknowns
    areas = areas_dict[phase_field]

    # Calculate score differences for each unknown
    for idx, unknown in enumerate(unknowns):
        # Filter data for the current unknown
        df_unknown = df[df['Unknown'] == unknown]

        # Calculate scores for each predicted error value
        score_pe_01 = df_unknown[df_unknown['Predicted error'] == pe_values[0]]['Score 3'].mean()
        score_pe_04 = df_unknown[df_unknown['Predicted error'] == pe_values[1]]['Score 3'].mean()

        # Calculate the score difference
        score_difference = score_pe_01 - score_pe_04

        # Append the results to the DataFrame
        new_row = pd.DataFrame({
            'Phase Field': [phase_field],
            'Unknown': [unknown],
            'Area': [areas[idx]],
            'Score Difference': [score_difference]
        })
        comparison_results = pd.concat([comparison_results, new_row], ignore_index=True)

# Display the comparison results
print(comparison_results)

# Plotting the relationship between area and score difference
plt.figure(figsize=(12, 8))
for phase_field in phase_fields:
    subset = comparison_results[comparison_results['Phase Field'] == phase_field]
    plt.scatter(subset['Area'], subset['Score Difference'], label=phase_field)

plt.xlabel('Area')
plt.ylabel('Score Difference after 3 samples (pe=0.01 - pe=0.04)')
plt.title('Score difference between predicted errors after 3 samples vs area'+
         f' (std={exp_std_value})')
plt.legend(title='Phase Field')
plt.grid(True)
plt.show()
'''
# Experimental std and predicted error value to compare
'''
exp_std_value = 0.1
pe_value = 0.01

# Create an empty DataFrame to store the results
comparison_results = pd.DataFrame(columns=[
    'Phase Field', 'Unknown', 'Area', 'Failure Rate'
])

# Iterate over phase fields
for phase_field in phase_fields:
    # Read the CSV file for the current phase field
    df = pd.read_csv(file_template.format(phase_field))

    # Filter the DataFrame to include only selected batch size, experimental std value, and predicted error value
    df = df[(df['Batch size'] == 1) & (df['Experimental std'] == exp_std_value) & (df['Predicted error'] == pe_value)]

    # Extract unique unknowns
    unknowns = df['Unknown'].unique()

    # Ensure the areas are in the same order as the unknowns
    areas = areas_dict[phase_field]

    # Calculate failure rates for each unknown
    for idx, unknown in enumerate(unknowns):
        # Filter data for the current unknown
        df_unknown = df[df['Unknown'] == unknown]

        # Calculate the failure rate as the proportion of 'Nulled' entries
        failure_rate = 100*(df_unknown['End type'] == 'Nulled').mean()

        # Append the results to the DataFrame
        new_row = pd.DataFrame({
            'Phase Field': [phase_field],
            'Unknown': [unknown],
            'Area': [areas[idx]],
            'Failure Rate': [failure_rate]
        })
        comparison_results = pd.concat([comparison_results, new_row], ignore_index=True)

# Display the comparison results
print(comparison_results)

# Plotting the relationship between area and failure rate
plt.figure(figsize=(12, 8))
for phase_field in phase_fields:
    subset = comparison_results[comparison_results['Phase Field'] == phase_field]
    plt.scatter(subset['Area'], subset['Failure Rate'], label=phase_field)

plt.xlabel('Area')
plt.ylabel('Failure Rate /%')
plt.title(f'Relationship between Area and Failure Rate (std={exp_std_value}, pe={pe_value})')
plt.legend(title='Phase Field')
plt.grid(True)
plt.show()
'''
#failure rate vs initial distance
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define phase fields and corresponding areas
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
areas_dict = {
    "MgBOF": [0.2163, 0.0687],
    "MgAlCu": [0.4185, 0.5033, 0.1517],
    "LiAlBO": [0.0109, 0.0602, 0.0550, 0.0557, 0.0465, 0.0298],
}

# File location template
file_template = '../simulation/data/compare_b=1/{}.csv'

# Experimental std and predicted error values to compare
exp_std_value = 0.1
pe_value = 0.01

# Specify the number of bins
num_bins = 6

# Iterate over phase fields
for phase_field in phase_fields:
    # Read the CSV file for the current phase field
    df = pd.read_csv(file_template.format(phase_field))

    # Filter the DataFrame to include only selected batch size, experimental std value, and predicted error value
    df = df[(df['Batch size'] == 1) & (df['Experimental std'] == exp_std_value) & (df['Predicted error'] == pe_value)]

    # Extract unique unknowns
    unknowns = df['Unknown'].unique()

    # Ensure the areas are in the same order as the unknowns
    areas = areas_dict[phase_field]

    # Calculate failure rates for each unknown
    for idx, unknown in enumerate(unknowns):
        # Filter data for the current unknown
        df_unknown = df[df['Unknown'] == unknown].copy()

        # Calculate failures based on 'End type'
        df_unknown.loc[:, 'Failure'] = df_unknown['End type'] == 'Nulled'

        # Bin the initial distance
        bins = np.linspace(0, df_unknown['Initial distance'].max(), num_bins + 1)
        df_unknown.loc[:, 'Distance Bin'] = pd.cut(df_unknown['Initial distance'], bins, right=False)

        # Count the number of failures in each bin
        failure_counts = 100*df_unknown.groupby('Distance Bin')['Failure'].mean()

        # Plot the results
        plt.figure(figsize=(12, 8))
        failure_counts.plot(kind='bar', color='red', alpha=0.7)
        plt.xlabel('Initial Distance Bins')
        plt.ylabel('Failure rate /%')
        plt.title(f'Failure rate vs. Initial Distance for {unknown}\n(Phase Field: {phase_field}, std={exp_std_value}, pe={pe_value})')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        '''
#all failure rates
'''
import matplotlib.pyplot as plt
import pandas as pd

# Define phase fields and corresponding areas
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
areas_dict = {
    "MgBOF": [0.2163, 0.0687],
    "MgAlCu": [0.4185, 0.5033, 0.1517],
    "LiAlBO": [0.0109, 0.0602, 0.0550, 0.0557, 0.0465, 0.0298],
}

# File location template
file_template = '../simulation/data/compare_b=1/{}.csv'

# Experimental std and predicted error values to compare
exp_std_value = 0.1
pe_value = 0.01

# Create a DataFrame to store the failure rates
failure_rates = pd.DataFrame(columns=['Phase Field', 'Unknown', 'Failure Rate'])

# Iterate over phase fields
for phase_field in phase_fields:
    # Read the CSV file for the current phase field
    df = pd.read_csv(file_template.format(phase_field))

    # Filter the DataFrame to include only selected batch size, experimental std value, and predicted error value
    df = df[(df['Batch size'] == 1) & (df['Experimental std'] == exp_std_value) & (df['Predicted error'] == pe_value)]

    # Extract unique unknowns
    unknowns = df['Unknown'].unique()

    # Calculate failure rates for each unknown
    for unknown in unknowns:
        # Filter data for the current unknown
        df_unknown = df[df['Unknown'] == unknown].copy()

        # Calculate the failure rate as the proportion of 'Nulled' entries
        failure_rate = 100*(df_unknown['End type'] == 'Nulled').mean()

        # Append the results to the DataFrame
        new_row = pd.DataFrame({
            'Phase Field': [phase_field],
            'Unknown': [unknown],
            'Failure Rate': [failure_rate]
        })
        failure_rates = pd.concat([failure_rates, new_row], ignore_index=True)

# Display the failure rates
print(failure_rates)

# Plotting the failure rates
plt.figure(figsize=(12, 8))
for phase_field in phase_fields:
    subset = failure_rates[failure_rates['Phase Field'] == phase_field]
    plt.bar(subset['Unknown'], subset['Failure Rate'], label=phase_field)

plt.xlabel('Unknown')
plt.ylabel('Failure Rate /%')
plt.title(f'Failure Rate for Each Unknown (std={exp_std_value}, pe={pe_value})')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Phase Field')
plt.grid(True)
plt.tight_layout()
plt.show()
'''
#Failure rate vs area
'''
import matplotlib.pyplot as plt
import pandas as pd

# Define phase fields and corresponding areas
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
areas_dict = {
    "MgBOF": [0.2163, 0.0687],
    "MgAlCu": [0.4185, 0.5033, 0.1517],
    "LiAlBO": [0.0109, 0.0602, 0.0550, 0.0557, 0.0465, 0.0298],
}

# File location template
file_template = '../simulation/data/compare_b=1/{}.csv'

# Experimental std and predicted error values to compare
exp_std_value = 0.1
pe_value = 0.01

# Create a DataFrame to store the failure rates
failure_rates = pd.DataFrame(columns=['Phase Field', 'Unknown', 'Area', 'Failure Rate'])

# Iterate over phase fields
for phase_field in phase_fields:
    # Read the CSV file for the current phase field
    df = pd.read_csv(file_template.format(phase_field))

    # Filter the DataFrame to include only selected batch size, experimental std value, and predicted error value
    df = df[(df['Batch size'] == 1) & (df['Experimental std'] == exp_std_value) & (df['Predicted error'] == pe_value)]

    # Extract unique unknowns
    unknowns = df['Unknown'].unique()

    # Ensure the areas are in the same order as the unknowns
    areas = areas_dict[phase_field]

    # Calculate failure rates for each unknown
    for idx, unknown in enumerate(unknowns):
        # Filter data for the current unknown
        df_unknown = df[df['Unknown'] == unknown].copy()

        # Calculate the failure rate as the proportion of 'Nulled' entries
        failure_rate = 100*(df_unknown['End type'] == 'Nulled').mean()

        # Append the results to the DataFrame
        new_row = pd.DataFrame({
            'Phase Field': [phase_field],
            'Unknown': [unknown],
            'Area': [areas[idx]],
            'Failure Rate': [failure_rate]
        })
        failure_rates = pd.concat([failure_rates, new_row], ignore_index=True)

# Display the failure rates
print(failure_rates)

# Plotting the failure rates vs area
plt.figure(figsize=(12, 8))
for phase_field in phase_fields:
    subset = failure_rates[failure_rates['Phase Field'] == phase_field]
    plt.scatter(subset['Area'], subset['Failure Rate'], label=phase_field)

plt.xlabel('Area')
plt.ylabel('Failure Rate /%')
plt.title(f'Failure Rate vs Area (std={exp_std_value}, pe={pe_value})')
plt.legend(title='Phase Field')
plt.grid(True)
plt.show()
'''

#graph to show best failure rate
'''
import matplotlib.pyplot as plt
import pandas as pd

# Define phase fields and corresponding areas
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
areas_dict = {
    "MgBOF": [0.2163, 0.0687],
    "MgAlCu": [0.4185, 0.5033, 0.1517],
    "LiAlBO": [0.0109, 0.0602, 0.0550, 0.0557, 0.0465, 0.0298],
}

# File location template
file_template = '../simulation/data/compare_b=1/{}.csv'

# Experimental std and predicted error values to compare
exp_std_values = [0.02, 0.05, 0.1]
pe_values = [0.01, 0.02, 0.04]

# Create a DataFrame to store the failure rates
failure_rates = pd.DataFrame(columns=['Phase Field', 'Unknown', 'Experimental std', 'Predicted error', 'Failure Rate'])

# Iterate over phase fields
for phase_field in phase_fields:
    # Read the CSV file for the current phase field
    df = pd.read_csv(file_template.format(phase_field))

    # Iterate over the combinations of experimental std and predicted error
    for exp_std_value in exp_std_values:
        for pe_value in pe_values:
            # Filter the DataFrame to include only selected batch size, experimental std value, and predicted error value
            df_filtered = df[(df['Batch size'] == 1) & (df['Experimental std'] == exp_std_value) & (df['Predicted error'] == pe_value)]

            # Debug print
            print(f"Processing Phase Field: {phase_field}, STD: {exp_std_value}, PE: {pe_value}, Entries: {len(df_filtered)}")

            # Extract unique unknowns
            unknowns = df_filtered['Unknown'].unique()

            # Calculate failure rates for each unknown
            for unknown in unknowns:
                # Filter data for the current unknown
                df_unknown = df_filtered[df_filtered['Unknown'] == unknown].copy()

                # Calculate the failure rate as the proportion of 'Nulled' entries
                failure_rate = 100*(df_unknown['End type'] == 'Nulled').mean()

                # Append the results to the DataFrame
                new_row = pd.DataFrame({
                    'Phase Field': [phase_field],
                    'Unknown': [unknown],
                    'Experimental std': [exp_std_value],
                    'Predicted error': [pe_value],
                    'Failure Rate': [failure_rate]
                })
                failure_rates = pd.concat([failure_rates, new_row], ignore_index=True)

# Debug print
print(failure_rates)

# Compute the average failure rate for each combination of experimental std and predicted error
average_failure_rates = failure_rates.groupby(['Experimental std', 'Predicted error'])['Failure Rate'].mean().reset_index()

# Ensure the order of the experimental std and predicted error values
average_failure_rates['Experimental std'] = pd.Categorical(average_failure_rates['Experimental std'], categories=exp_std_values, ordered=True)
average_failure_rates['Predicted error'] = pd.Categorical(average_failure_rates['Predicted error'], categories=pe_values, ordered=True)
average_failure_rates = average_failure_rates.sort_values(['Experimental std', 'Predicted error'])

# Plotting the average failure rates
fig, ax = plt.subplots(figsize=(12, 8))

# Define the positions and width for the bars
bar_width = 0.2
positions = range(len(exp_std_values))

# Plot bars for each predicted error value
for i, pe_value in enumerate(pe_values):
    pe_subset = average_failure_rates[average_failure_rates['Predicted error'] == pe_value]
    ax.bar([p + bar_width*i for p in positions], pe_subset['Failure Rate'], bar_width, label=f'PE={pe_value}')

# Set the x-ticks and labels
ax.set_xticks([p + bar_width for p in positions])
ax.set_xticklabels([f'STD={val}' for val in exp_std_values])

ax.set_xlabel('Experimental std')
ax.set_ylabel('Average Failure Rate /%')
ax.set_title('Average Failure Rate (all unknowns) vs Experimental std and Predicted error')
ax.legend(title='Predicted error')
ax.grid(True)

plt.show()
'''
#graph to gauge best pe for the different std
'''
import matplotlib.pyplot as plt
import pandas as pd

# Define phase fields and corresponding areas
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
areas_dict = {
    "MgBOF": [0.2163, 0.0687],
    "MgAlCu": [0.4185, 0.5033, 0.1517],
    "LiAlBO": [0.0109, 0.0602, 0.0550, 0.0557, 0.0465, 0.0298],
}

# File location template
file_template = '../simulation/data/compare_b=1/{}.csv'

# Experimental std and predicted error values to compare
exp_std_values = [0.02, 0.05, 0.1]
pe_values = [0.01, 0.02, 0.04]

# Create a DataFrame to store the scores
scores = pd.DataFrame(columns=['Phase Field', 'Unknown', 'Experimental std', 'Predicted error', 'Score After 3 Batches'])

# Iterate over phase fields
for phase_field in phase_fields:
    # Read the CSV file for the current phase field
    df = pd.read_csv(file_template.format(phase_field))

    # Iterate over the combinations of experimental std and predicted error
    for exp_std_value in exp_std_values:
        for pe_value in pe_values:
            # Filter the DataFrame to include only selected batch size, experimental std value, and predicted error value
            df_filtered = df[(df['Batch size'] == 1) & (df['Experimental std'] == exp_std_value) & (df['Predicted error'] == pe_value)]

            # Debug print
            print(f"Processing Phase Field: {phase_field}, STD: {exp_std_value}, PE: {pe_value}, Entries: {len(df_filtered)}")

            # Extract unique unknowns
            unknowns = df_filtered['Unknown'].unique()

            # Calculate scores for each unknown
            for unknown in unknowns:
                # Filter data for the current unknown
                df_unknown = df_filtered[df_filtered['Unknown'] == unknown].copy()

                # Calculate the average score after 3 batches
                score_after_3_batches = df_unknown['Score 3'].mean()

                # Append the results to the DataFrame
                new_row = pd.DataFrame({
                    'Phase Field': [phase_field],
                    'Unknown': [unknown],
                    'Experimental std': [exp_std_value],
                    'Predicted error': [pe_value],
                    'Score After 3 Batches': [score_after_3_batches]
                })
                scores = pd.concat([scores, new_row], ignore_index=True)

# Debug print
print(scores)

# Compute the average score after 3 batches for each combination of experimental std and predicted error
average_scores = scores.groupby(['Experimental std', 'Predicted error'])['Score After 3 Batches'].mean().reset_index()

# Ensure the order of the experimental std and predicted error values
average_scores['Experimental std'] = pd.Categorical(average_scores['Experimental std'], categories=exp_std_values, ordered=True)
average_scores['Predicted error'] = pd.Categorical(average_scores['Predicted error'], categories=pe_values, ordered=True)
average_scores = average_scores.sort_values(['Experimental std', 'Predicted error'])

# Plotting the average scores
fig, ax = plt.subplots(figsize=(12, 8))

# Define the positions and width for the bars
bar_width = 0.2
positions = range(len(exp_std_values))

# Plot bars for each predicted error value
for i, pe_value in enumerate(pe_values):
    pe_subset = average_scores[average_scores['Predicted error'] == pe_value]
    ax.bar([p + bar_width*i for p in positions], pe_subset['Score After 3 Batches'], bar_width, label=f'PE={pe_value}')

# Set the x-ticks and labels
ax.set_xticks([p + bar_width for p in positions])
ax.set_xticklabels([f'STD={val}' for val in exp_std_values])

ax.set_xlabel('Experimental std')
ax.set_ylabel('Average Score After 3 Batches')
ax.set_title('Average Score After 3 Batches vs Experimental std and Predicted error')
ax.legend(title='Predicted error')
ax.grid(True)

plt.show()
'''
#score vs sample, all unkowns
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define phase fields
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
phase_fields = ["MgBOF"]

# File location template
file_template = '../simulation/data/compare_b=1/{}.csv'

# Experimental std and predicted error values to compare
exp_std_value = 0.1
pe_value = 0.04

# Iterate over phase fields
plt.figure(figsize=(12, 8))
for phase_field in phase_fields:
    # Read the CSV file for the current phase field
    df = pd.read_csv(file_template.format(phase_field))

    # Filter the DataFrame to include only selected batch size, experimental std value, and predicted error value
    df_filtered = df[(df['Batch size'] == 1) & (df['Experimental std'] == exp_std_value) & (df['Predicted error'] == pe_value)]

    # Extract unique unknowns
    unknowns = df_filtered['Unknown'].unique()

    # Plotting

    # Calculate and plot median score vs number of samples with error bars for each unknown
    for unknown in unknowns:
        df_unknown = df_filtered[df_filtered['Unknown'] == unknown]

        medians = []
        lower_bounds = []
        upper_bounds = []

        for sample_num in range(0, 16):
            col_name = f'Score {sample_num}'
            if col_name in df_unknown.columns:
                median = df_unknown[col_name].median()
                lower_bound = np.percentile(df_unknown[col_name], 16)
                upper_bound = np.percentile(df_unknown[col_name], 84)

                medians.append(median)
                lower_bounds.append(median - lower_bound)
                upper_bounds.append(upper_bound - median)

        plt.errorbar(range(0, len(medians)), medians, yerr=[lower_bounds, upper_bounds], label=unknown, capsize=5)

    plt.xlabel('Number of Samples')
    plt.ylabel('Median Score')
    plt.title(f'Median Score vs Number of Samples (std={exp_std_value}, pe={pe_value})')
    plt.legend(title='Unknown')
    plt.grid(True)
plt.show()
'''
#score vs sample, average for each phase field (+check on excluding fail)
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define phase fields
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]

# File location template
file_template = '../simulation/data/compare_b=1/{}.csv'

# Experimental std and predicted error values to compare
exp_std_value = 0.05
pe_value = 0.02

# Plotting
plt.figure(figsize=(12, 8))

# Iterate over phase fields
for phase_field in phase_fields:
    # Read the CSV file for the current phase field
    df = pd.read_csv(file_template.format(phase_field))

    # Filter the DataFrame to include only selected batch size, experimental std value, and predicted error value
    df_filtered = df[(df['Batch size'] == 1) & (df['Experimental std'] == exp_std_value) & (df['Predicted error'] == pe_value)]

    # Calculate the average score for each sample number across all unknowns
    averages = []
    lower_bounds = []
    upper_bounds = []

    for sample_num in range(1, 16):
        col_name = f'Score {sample_num}'
        if col_name in df_filtered.columns:
            median = df_filtered[col_name].median()
            lower_bound = np.percentile(df_filtered[col_name], 16)
            upper_bound = np.percentile(df_filtered[col_name], 84)

            lower_bounds.append(median - lower_bound)
            upper_bounds.append(upper_bound - median)
            averages.append(median)

    plt.errorbar(
        range(0, len(averages)), averages, yerr=[lower_bounds, upper_bounds],
        label=phase_field+'+f', capsize=5)
# Iterate over phase fields
for phase_field in phase_fields:
    # Read the CSV file for the current phase field
    df = pd.read_csv(file_template.format(phase_field))

    # Filter the DataFrame to include only selected batch size, experimental std value, and predicted error value
    df_filtered = df[(df['Batch size'] == 1) 
                     & (df['Experimental std'] == exp_std_value)
                     & (df['Predicted error'] == pe_value)
                     & (df['End type']!='Nulled')]

    # Calculate the average score for each sample number across all unknowns
    averages = []
    lower_bounds = []
    upper_bounds = []

    for sample_num in range(1, 16):
        col_name = f'Score {sample_num}'
        if col_name in df_filtered.columns:
            median = df_filtered[col_name].median()
            lower_bound = np.percentile(df_filtered[col_name], 16)
            upper_bound = np.percentile(df_filtered[col_name], 84)

            lower_bounds.append(median - lower_bound)
            upper_bounds.append(upper_bound - median)
            averages.append(median)

    plt.errorbar(
        range(0, len(averages)), averages, yerr=[lower_bounds, upper_bounds],
        label=phase_field, capsize=10)

plt.xlabel('Number of Samples')
plt.ylabel('Average Score')
plt.title(f'Average Score vs Number of Samples (std={exp_std_value}, pe={pe_value})')
plt.legend(title='Phase Field')
plt.grid(True)
plt.show()
'''
'''
import matplotlib.pyplot as plt
import pandas as pd

# Define phase fields
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]

# File templates
data_file_template = '../simulation/data/compare_b=1/{}.csv'
angle_error_file_template = '../simulation/error_testing/phase_field_{}.csv'

# Experimental std, predicted error, and batch size to filter
exp_std_value = 0.1
pe_value = 0.02
batch_size_value = 1

# Create a DataFrame to store the results
results = pd.DataFrame(columns=['Phase Field', 'Unknown', 'Angle Error', 'Average Score After 3 Batches'])

# Iterate over phase fields
for phase_field in phase_fields:
    # Read the CSV file for the current phase field
    df = pd.read_csv(data_file_template.format(phase_field))

    # Read the angle error CSV file for the current phase field
    angle_errors = pd.read_csv(angle_error_file_template.format(phase_field))

    # Filter the DataFrame to include only selected batch size, experimental std value, and predicted error value
    df_filtered = df[(df['Batch size'] == batch_size_value)
                     & (df['Experimental std'] == exp_std_value)
                     & (df['Predicted error'] == pe_value)
                     & (df['End type']!="Nulled")]

    # Extract unique unknowns
    unknowns = df_filtered['Unknown'].unique()

    # Calculate the average score after 3 batches for each unknown
    for unknown in unknowns:
        # Filter data for the current unknown
        df_unknown = df_filtered[df_filtered['Unknown'] == unknown].copy()

        # Calculate the average score after 3 batches
        average_score_after_3_batches = df_unknown['Score 10'].median()

        # Get the angle error for the current unknown
        angle_error = angle_errors[
            angle_errors['Unknown'] ==unknown]['max_angular_error'].values[0]

        # Append the results to the DataFrame
        new_row = pd.DataFrame({
            'Phase Field': [phase_field],
            'Unknown': [unknown],
            'Angle Error': [angle_error],
            'Average Score After 3 Batches': [average_score_after_3_batches]
        })
        results = pd.concat([results, new_row], ignore_index=True)

# Display the results
print(results)

# Plotting the results
plt.figure(figsize=(12, 8))
for phase_field in phase_fields:
    subset = results[results['Phase Field'] == phase_field]
    plt.scatter(subset['Angle Error'], subset['Average Score After 3 Batches'], label=phase_field)

plt.xlabel('Average angular error for triangle with maximal average angular error')
plt.ylabel('Median Score After 10 Batches')
plt.title(f'Median Score After 10 Batches vs Max average angular error (std={exp_std_value}, pe={pe_value}, batch size={batch_size_value})')
plt.legend(title='Phase Field')
plt.grid(True)
plt.show()
'''


#df alterations


#LiAlBO
'''
field="LiAlBO"
df=pd.read_csv('../simulation/data/compare_batches/refined_'+field+'_0.csv')
dfb=pd.read_csv('../simulation/data/compare_batches/refined_'+field+'_35.csv')
df=pd.concat((df,dfb),ignore_index=True)
df=df.replace('Li 2 Al B 5 O 10','Li$_2$AlB$_5$O$_{10}$')
df=df.replace('Li Al B 2 O 5','LiAlB$_2$O$_5$')
df=df.replace('Li 3 Al B 2 O 6','Li$_3$AlB$_2$O$_6$')
df=df.replace('Li Al 7 B 4 O 17','LiAl$_7$B$_4$O$_{17}$')
df=df.replace('Li 2.46 Al 0.18 B O 3','Li$_{2.46}$Al$_{0.18}$BO$_3$')
df=df.replace('Li 2 Al B O 4','Li$_2$AlBO$_4$')
stds=df['Experimental std'].unique()
pes=df['Predicted error'].unique()
bs=df['Batch size'].unique()
us=df['Unknown'].unique()
x=len(pes)*len(stds)*len(us)*len(bs)
repeats=len(df)/x
print(f'{pes=}, {stds=}, {bs=}, {us=}, {repeats=}')
for i in df.columns:
    print(i)
print(df.head())
df.to_csv("../simulation/data/compare_batches/"+field+'.csv',index=False)
df=pd.read_csv('../simulation/data/compare_batches/'+field+'.csv')
stds=df['Experimental std'].unique()
pes=df['Predicted error'].unique()
bs=df['Batch size'].unique()
us=df['Unknown'].unique()
x=len(pes)*len(stds)*len(us)*len(bs)
repeats=len(df)/x
print(f'{pes=}, {stds=}, {bs=}, {us=}, {repeats=}')
for i in df.columns:
    print(i)
print(df.head())
'''
#dfb=dfb[dfb['Predicted error'].isin([0.01,0.02,0.04])]
#dfb.to_csv("../simulation/data/compare_b=1/"+field+'.csv')
#f=3
#dfb=pd.read_csv('../simulation/data/compare_b=1/'+field+'.csv')
#df=pd.concat((df,dfb),ignore_index=True)
#df.to_csv("../simulation/data/compare_batches/"+field+'.csv')
#print(df.head)
#df=pd.read_csv('../simulation/data/compare_batches/'+field+'.csv')
#substitute unknowns
#field="MgAlCu"
'''
df=pd.read_csv('../simulation/data/compare_batches/'+field+'.csv')
stds=df['Experimental std'].unique()
pes=df['Predicted error'].unique()
bs=df['Batch size'].unique()
us=df['Unknown'].unique()
x=len(pes)*len(stds)*len(us)*len(bs)
repeats=len(df)/x
print(df.head())
print(f'{pes=}, {stds=}, {bs=}, {us=}, {repeats=}')
print(df['Unknown'].unique())
df=df.replace("Mg 6 Al 4 Cu 8","Mg$_3$(AlCu$_2$)$_2$")
df=df.replace("Mg 2 Al 4 Cu 2","MgAl$_2$Cu")
df=df.replace("Mg 6 Al 15 Cu 18","Mg$_2$Al$_5$Cu$_6$")
print(df['Unknown'].unique())
df.to_csv("../simulation/data/compare_batches/"+field+'.csv',index=False)
df=pd.read_csv('../simulation/data/compare_batches/'+field+'.csv')
stds=df['Experimental std'].unique()
pes=df['Predicted error'].unique()
bs=df['Batch size'].unique()
us=df['Unknown'].unique()
x=len(pes)*len(stds)*len(us)*len(bs)
repeats=len(df)/x
print(df.head())
print(f'{pes=}, {stds=}, {bs=}, {us=}, {repeats=}')
'''
#df=df.replace('Mg 6 B 2 O 6 F 6','Mg$_3$B(OF)$_3$')
#df=df.replace('Mg 20 B 12 O 36 F 4','Mg$_5$B$_3$O$_9$F')
#df=df.replace('Mg 20 B 12 O 36 F 4','Mg$_5$B$_3$O$_9$F')
#['Li 2 Al B 5 O 10' 'Li Al B 2 O 5' 'Li 3 Al B 2 O 6' 'Li 2 Al B O 4'
# 'Li Al 7 B 4 O 17' 'Li 2.46 Al 0.18 B O 3']
#df=df.replace('Li 2 Al B 5 O 10','Li$_2$AlB$_5$O$_{10}$')
#df=df.replace('Li Al B 2 O 5','LiAlB$_2$O$_5$')
#df=df.replace('Li 3 Al B 2 O 6','Li$_3$AlB$_2$O$_6$')
#df=df.replace('Li Al 7 B 4 O 17','LiAl$_7$B$_4$O$_{17}$')
#df=df.replace('Li 2.46 Al 0.18 B O 3','Li$_{2.46}$Al$_{0.18}$BO$_3$')
#df=df.replace('Li 2 Al B O 4','Li$_2$AlBO$_4$')
#print(df['Unknown'].unique())
#df.to_csv("../simulation/data/compare_batches/"+field+'.csv',index=False)
#df=pd.read_csv('../simulation/data/compare_batches/'+field+'.csv')
#stds=df['Experimental std'].unique()
#pes=df['Predicted error'].unique()
#bs=df['Batch size'].unique()
#us=df['Unknown'].unique()
#x=len(pes)*len(stds)*len(us)*len(bs)
#repeats=len(df)/x
#constant pe graph
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os

# Define phase field and experimental stds in increasing order
phase_field = "MgBOF"
experimental_stds = [0.02, 0.05, 0.1]

# File location template
data_file_template = '../simulation/data/compare_batches/{}.csv'
predicted_error = 0.02

# Directory to save the plots
save_dir_template = '../simulation/unknown_all_figs/{}'

# Batch sizes to consider
batch_sizes = [1, 3, 5]

# Read the CSV file for the current phase field
df = pd.read_csv(data_file_template.format(phase_field))

# Filter the DataFrame to exclude 'Nulled' end types and to include only the selected predicted error value
df_filtered = df[(df['Predicted error'] == predicted_error) & (df['End type'] != 'Nulled')]

# Extract unique unknowns
unknowns = df_filtered['Unknown'].unique()

# Function to calculate median and 68% error bounds
def calculate_statistics(df, batch_size, samples):
    medians = []
    lower_bounds = []
    upper_bounds = []
    
    for sample_num in range(0, samples+1):
        if sample_num % batch_size == 0:
            batch_index = sample_num // batch_size
            col_name = f'Score {batch_index}'
            if col_name in df.columns:
                median = df[col_name].median() * 100
                lower_bound = np.percentile(df[col_name], 16) * 100
                upper_bound = np.percentile(df[col_name], 84) * 100
                
                medians.append(median)
                lower_bounds.append(median - lower_bound)
                upper_bounds.append(upper_bound - median)
            else:
                medians.append(np.nan)
                lower_bounds.append(np.nan)
                upper_bounds.append(np.nan)
        else:
            medians.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
    
    return medians, lower_bounds, upper_bounds

# Plotting
samples = 15

for unknown in unknowns:
    plt.figure(figsize=(12, 8))
    
    for batch_size in batch_sizes:
        for exp_std_value in experimental_stds:
            df_unknown = df_filtered[(df_filtered['Unknown'] == unknown) & 
                                     (df_filtered['Experimental std'] == exp_std_value) & 
                                     (df_filtered['Batch size'] == batch_size)]
            medians, lower_bounds, upper_bounds = calculate_statistics(df_unknown, batch_size, samples)
            
            valid_indices = [i for i in range(samples+1) if i % batch_size == 0]
            valid_medians = [medians[i] for i in valid_indices]
            valid_lower_bounds = [lower_bounds[i] for i in valid_indices]
            valid_upper_bounds = [upper_bounds[i] for i in valid_indices]
            valid_samples = [i for i in valid_indices]
            
            plt.errorbar(valid_samples, valid_medians, yerr=[valid_lower_bounds, valid_upper_bounds], 
                         label=f'$\\sigma_E={exp_std_value}$, Batch size={batch_size}', capsize=5)
    
    plt.xlabel('Number of Samples')
    plt.ylabel('Percentage purity of unknown phase /%')
    plt.title(f'Percentage purity of {unknown} vs total number of samples for $\\sigma_P={predicted_error}$')
    plt.legend(title='Parameter choices')
    plt.grid(True)
    plt.tight_layout()

    # Create directory if it doesn't exist
    clean_unknown = re.sub(r'[\$\_]', '', unknown)
    save_dir = save_dir_template.format(clean_unknown)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(f'{save_dir}/pe=2.png', dpi=400)
    plt.show()
    plt.close()
    '''

#plot score vs sample number for all parameters each unknown
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os
import matplotlib.cm as cm
from matplotlib.colors import to_rgba

width_in_inches = 7
aspect_ratio = 3/4
height_in_inches = 4
plt.rcParams['figure.figsize'] = (width_in_inches, height_in_inches)
plt.rcParams['font.size'] = 8

# Define phase field and experimental stds in increasing order
phase_field = "LiAlBO"
experimental_stds = [0.02, 0.05, 0.1]

# File location template
data_file_template = '/home/danny/phd/composition_exploration/exploration_algorithm/simulation/data/compare_batches/{}.csv'
predicted_errors = [0.01, 0.02, 0.04]

# Directory to save the plots
save_dir_template = '../simulation/unknown_all_figs/{}'

# Batch sizes to consider
batch_sizes = [1,3,5]

# Read the CSV file for the current phase field
df = pd.read_csv(data_file_template.format(phase_field))
stds=df['Experimental std'].unique()
pes=df['Predicted error'].unique()
bs=df['Batch size'].unique()
us=df['Unknown'].unique()
x=len(pes)*len(stds)*len(us)*len(bs)
repeats=len(df)/x
print(df.head())
print(f'{pes=}, {stds=}, {bs=}, {us=}, {repeats=}')

# Filter the DataFrame to exclude 'Nulled' end types
df_filtered = df[df['End type'] != 'Nulled']

# Extract unique unknowns
unknowns = df_filtered['Unknown'].unique()

# Function to calculate median and 68% error bounds
def calculate_statistics(df, batch_size, samples):
    medians = []
    lower_bounds = []
    upper_bounds = []
    
    for sample_num in range(0, samples+1):
        if sample_num % batch_size == 0:
            batch_index = sample_num // batch_size
            col_name = f'Score {batch_index}'
            if col_name in df.columns:
                median = df[col_name].median() * 100
                lower_bound = np.percentile(df[col_name], 16) * 100
                upper_bound = np.percentile(df[col_name], 84) * 100
                
                medians.append(median)
                lower_bounds.append(median - lower_bound)
                upper_bounds.append(upper_bound - median)
            else:
                medians.append(np.nan)
                lower_bounds.append(np.nan)
                upper_bounds.append(np.nan)
        else:
            medians.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
    
    return medians, lower_bounds, upper_bounds

# Define distinct colors for each predicted error
base_colors = {
    0.01: 'red',
    0.02: 'blue',
    0.04: 'green'
}

# Define shades for each batch size
shades = {
    1: 1.0,
    3: 0.7,
    5: 0.4
}
colours={0.01:{1:'lightcoral',3:'orangered',5:'firebrick'},
         0.02:{1:'palegreen',3:'lime',5:'olivedrab'},
         0.04:{1:'deepskyblue',3:'blue',5:'blueviolet'}}

# Plotting
samples = 15

for unknown in unknowns:
    for exp_std_value in experimental_stds:
        plt.figure()
        
        for predicted_error in predicted_errors:
            base_color = to_rgba(base_colors[predicted_error])
            for batch_size in batch_sizes:
                df_unknown = df_filtered[(df_filtered['Unknown'] == unknown) & 
                                         (df_filtered['Experimental std'] == exp_std_value) & 
                                         (df_filtered['Batch size'] == batch_size) & 
                                         (df_filtered['Predicted error'] == predicted_error)]
                medians, lower_bounds, upper_bounds = calculate_statistics(df_unknown, batch_size, samples)
                
                valid_indices = [i for i in range(samples+1) if i % batch_size == 0]
                valid_medians = [medians[i] for i in valid_indices]
                valid_lower_bounds = [lower_bounds[i] for i in valid_indices]
                valid_upper_bounds = [upper_bounds[i] for i in valid_indices]
                valid_samples = [i for i in valid_indices]
                
                # Adjust the base color by the shade
                c = colours[predicted_error][batch_size]
                
                # Plot the medians and error bars
                plt.errorbar(
                    valid_samples, valid_medians,
                    yerr=[valid_lower_bounds, valid_upper_bounds],
                    label=f'$\\sigma_P={predicted_error}$, Batch size={batch_size}',
                    capsize=10, elinewidth=0, markeredgewidth=3, color=c,
                    linewidth=1.5)
        
        plt.xlabel('Number of Samples')
        plt.ylabel('Purity Score /%')
        #plt.title(f'Purity Score for {unknown} for $\\sigma_E={exp_std_value}$')
        plt.legend(title='Parameter choices')
        plt.grid(True)

        # Ensure the layout is tight
        #plt.tight_layout()
        ax=plt.gca()
        ax.set_ylim([15,105])
        
        # Create directory if it doesn't exist
        clean_unknown = re.sub(r'[\$\_\.\{\}]', '', unknown)
        save_dir = save_dir_template.format(clean_unknown)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the plot
        plt.savefig(
            f'{save_dir}/{clean_unknown}_std_{exp_std_value}.png', dpi=600)
        
        # Show the plot
        #plt.show()
        '''
#plot failure rate for all combinations each unknown
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os

width_in_inches = 7
height_in_inches = 4
plt.rcParams['figure.figsize'] = (width_in_inches, height_in_inches)
plt.rcParams['font.size'] = 8

# Define phase field and experimental stds in increasing order
phase_field = "MgAlCu"
experimental_stds = [0.02, 0.05, 0.1]

# File location template
data_file_template = '/home/danny/phd/composition_exploration/exploration_algorithm/simulation/data/compare_batches/{}.csv'
predicted_errors = [0.01, 0.02, 0.04]

# Directory to save the plots
save_dir_template = '../simulation/unknown_all_figs/{}'

# Batch sizes to consider
batch_sizes = [1,3,5]

# Read the CSV file for the current phase field
df = pd.read_csv(data_file_template.format(phase_field))

# Extract unique unknowns
unknowns = df['Unknown'].unique()

# Function to calculate failure rates
def calculate_failure_rate(df, batch_size, predicted_error, experimental_std):
    df_filtered = df[(df['Batch size'] == batch_size) &
                     (df['Predicted error'] == predicted_error) &
                     (df['Experimental std'] == experimental_std)]
    failure_rate = 100*df_filtered['End type'].value_counts(normalize=True).get('Nulled', 0)
    return failure_rate

# Plotting
for unknown in unknowns:
    failure_rates = []
    labels = []

    for batch_size in batch_sizes:
        for predicted_error in predicted_errors:
            for exp_std_value in experimental_stds:
                failure_rate = calculate_failure_rate(df[df['Unknown'] == unknown], batch_size, predicted_error, exp_std_value)
                failure_rates.append(failure_rate)
                labels.append(f'$\sigma_P$={predicted_error}, BS={batch_size}')

    # Create the bar chart
    x = np.arange(len(labels) // len(experimental_stds))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()

    for i, exp_std_value in enumerate(experimental_stds):
        rates = failure_rates[i::len(experimental_stds)]
        ax.bar(x + i * width, rates, width, label=f'$\\sigma_E={exp_std_value}$')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('PICIP Error ($\sigma_P$) and Batch Size (BS)')
    ax.set_ylabel('Failure Rate /%')
    #ax.set_title(f'Failure Rate for {unknown}')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels[::len(experimental_stds)], rotation=45, ha='right')
    ax.set_ylim([0,70])
    ax.legend(title='Experimental std')

    # Create directory if it doesn't exist
    clean_unknown = re.sub(r'[\$\_\.\{\}]', '', unknown)
    save_dir = save_dir_template.format(clean_unknown)
    os.makedirs(save_dir, exist_ok=True)
    plt.gcf().subplots_adjust(bottom=0.25)  # Adjust the bottom space

    # Show the plot
    plt.savefig(
        f'{save_dir}/{clean_unknown}_failure_rate.png',dpi=600)
    #plt.show()
    '''
#Key note one, predicted error vs experimental std averaged
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os

width_in_inches = 16.51 / 2.54 *2
aspect_ratio = 2.5/4
height_in_inches = width_in_inches * aspect_ratio
plt.rcParams['figure.figsize'] = (width_in_inches, height_in_inches)
plt.rcParams['font.size'] = 14

# Define phase fields and corresponding areas
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
experimental_stds = [0.02, 0.05, 0.1]
predicted_errors = [0.01, 0.02, 0.04]

# File location template
data_file_template = '../simulation/data/compare_batches/{}.csv'

# Batch sizes to consider
batch_size = 1

# Initialize an empty list to store the data
all_data = []

# Read and combine the data from all phase fields
for phase_field in phase_fields:
    df = pd.read_csv(data_file_template.format(phase_field))
    df = df[df['End type'] != 'Nulled']
    df = df[df['Batch size'] == batch_size]
    all_data.append(df)

# Concatenate all data into a single DataFrame
combined_df = pd.concat(all_data)

# Function to calculate median and 68% error bounds
def calculate_statistics(df, samples):
    medians = []
    lower_bounds = []
    upper_bounds = []

    for sample_num in range(0, samples+1):
        col_name = f'Score {sample_num}'
        if col_name in df.columns:
            median = df[col_name].median() * 100
            lower_bound = np.percentile(df[col_name], 16) * 100
            upper_bound = np.percentile(df[col_name], 84) * 100

            medians.append(median)
            lower_bounds.append(median - lower_bound)
            upper_bounds.append(upper_bound - median)
        else:
            medians.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)

    return medians, lower_bounds, upper_bounds

# Plotting
samples = 15

colours={0.02:{0.01:'lightcoral',0.02:'orangered',0.04:'firebrick'},
         0.05:{0.01:'palegreen',0.02:'lime',0.04:'olivedrab'},
         0.1:{0.01:'deepskyblue',0.02:'blue',0.04:'blueviolet'}}

for exp_std_value in experimental_stds:
    for predicted_error in predicted_errors:
        df_filtered = combined_df[(combined_df['Predicted error'] == predicted_error) &
                                  (combined_df['Experimental std'] == exp_std_value)]
        medians, lower_bounds, upper_bounds = calculate_statistics(df_filtered, samples)

        valid_indices = [i for i in range(samples+1)]
        valid_medians = [medians[i] for i in valid_indices]
        valid_lower_bounds = [lower_bounds[i] for i in valid_indices]
        valid_upper_bounds = [upper_bounds[i] for i in valid_indices]
        valid_samples = [i for i in valid_indices]

        # Plot the medians and error bars
        c=colours[exp_std_value][predicted_error]
        plt.errorbar(
            valid_samples, valid_medians,
            yerr=[valid_lower_bounds, valid_upper_bounds],
            label=f'$\\sigma_P={predicted_error}$, $\\sigma_E={exp_std_value}$',
            capsize=10, elinewidth=0, markeredgewidth=3,c=c)

plt.xlabel('Number of Samples')
plt.ylabel('Purity Score /%')
#plt.title('Purity Score vs number of samples, averaged over all unknowns')
plt.legend(title='Parameter choices')
plt.grid(True)

# Ensure the layout is tight
#plt.tight_layout()

# Show the plot

# Set the text size to 10pt

#plt.savefig(
#    '../simulation/comparison graphs/uavg_score_vs_n.png',
#    dpi=600,bbox_inches='tight')
plt.show()
'''
# dive, unklnowns where PE affects it the most and the least
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os

# Define phase fields and corresponding areas
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
experimental_std_initial = 0.01
experimental_std_plot = 0.1
predicted_errors = [0.01, 0.04]
batch_size = 1

# File location template
data_file_template = '../simulation/data/compare_batches/{}.csv'

# Initialize an empty list to store the data
all_data = []

# Read and combine the data from all phase fields
for phase_field in phase_fields:
    df = pd.read_csv(data_file_template.format(phase_field))
    all_data.append(df)

# Concatenate all data into a single DataFrame
combined_df = pd.concat(all_data)

# Extract unique unknowns
unknowns = combined_df['Unknown'].unique()

# Function to calculate the difference in median score after 4 samples
def calculate_score_difference(df, predicted_errors, samples=4):
    scores = {}
    for pe in predicted_errors:
        df_pe = df[df['Predicted error'] == pe]
        col_name = f'Score {samples}'
        if col_name in df_pe.columns:
            scores[pe] = df_pe[col_name].median() * 100
        else:
            scores[pe] = np.nan
    return abs(scores[predicted_errors[0]] - scores[predicted_errors[1]])

# Calculate the score differences for each unknown
score_differences = []
for unknown in unknowns:
    df_unknown = combined_df[(combined_df['Unknown'] == unknown) & 
                             (combined_df['Experimental std'] == experimental_std_initial) &
                             (df['Batch size'] == batch_size)]
    score_diff = calculate_score_difference(df_unknown, predicted_errors)
    score_differences.append((unknown, score_diff))

# Sort the unknowns by score difference
score_differences.sort(key=lambda x: x[1])

# Get the unknowns with the most and least difference
least_diff_unknown = score_differences[0][0]
most_diff_unknown = score_differences[-1][0]

# Function to calculate median and 68% error bounds
def calculate_statistics(df, samples):
    medians = []
    lower_bounds = []
    upper_bounds = []
    
    for sample_num in range(0, samples+1):
        col_name = f'Score {sample_num}'
        if col_name in df.columns:
            median = df[col_name].median() * 100
            lower_bound = np.percentile(df[col_name], 16) * 100
            upper_bound = np.percentile(df[col_name], 84) * 100
            
            medians.append(median)
            lower_bounds.append(median - lower_bound)
            upper_bounds.append(upper_bound - median)
        else:
            medians.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
    
    return medians, lower_bounds, upper_bounds

# Function to plot the score vs number of samples with error bars
def plot_score_vs_samples(df, unknown, predicted_errors, experimental_std, batch_size, samples=15):
    plt.figure(figsize=(12, 8))
    for pe in predicted_errors:
        df_filtered = df[(df['Unknown'] == unknown) & 
                         (df['Predicted error'] == pe) & 
                         (df['Experimental std'] == experimental_std) & 
                         (df['Batch size'] == batch_size)]
        medians, lower_bounds, upper_bounds = calculate_statistics(df_filtered, samples)
        
        valid_indices = [i for i in range(samples+1)]
        valid_medians = [medians[i] for i in valid_indices]
        valid_lower_bounds = [lower_bounds[i] for i in valid_indices]
        valid_upper_bounds = [upper_bounds[i] for i in valid_indices]
        valid_samples = [i for i in valid_indices]
        
        # Plot the medians and error bars
        plt.errorbar(
            valid_samples, valid_medians,
            yerr=[valid_lower_bounds, valid_upper_bounds],
            label=f'{unknown}, PE={pe}',
            capsize=10, elinewidth=1, markeredgewidth=1)
    
    plt.xlabel('Number of Samples')
    plt.ylabel('Percentage purity of unknown phase /%')
    plt.title(f'Percentage purity vs total number of samples for $\\sigma_E={experimental_std}$')
    plt.legend(title='Parameter choices')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot for the unknown with the least difference and the unknown with the most difference on the same axis
plt.figure(figsize=(12, 8))

for unknown in [least_diff_unknown, most_diff_unknown]:
    for pe in predicted_errors:
        df_filtered = combined_df[(combined_df['Unknown'] == unknown) & 
                                  (combined_df['Predicted error'] == pe) & 
                                  (combined_df['Experimental std'] == experimental_std_plot) & 
                                  (combined_df['Batch size'] == batch_size)]
        medians, lower_bounds, upper_bounds = calculate_statistics(df_filtered, samples=15)
        
        valid_indices = [i for i in range(16)]
        valid_medians = [medians[i] for i in valid_indices]
        valid_lower_bounds = [lower_bounds[i] for i in valid_indices]
        valid_upper_bounds = [upper_bounds[i] for i in valid_indices]
        valid_samples = [i for i in valid_indices]
        
        plt.errorbar(
            valid_samples, valid_medians,
            yerr=[valid_lower_bounds, valid_upper_bounds],
            label=f'{unknown}, PE={pe}',
            capsize=10, elinewidth=1, markeredgewidth=1)

plt.xlabel('Number of Samples')
plt.ylabel('Percentage purity of unknown phase /%')
plt.title(f'Percentage purity vs total number of samples for $\\sigma_E={experimental_std_plot}$')
plt.legend(title='Parameter choices')
plt.grid(True)
plt.tight_layout()
plt.show()
'''

# max abs difference in score between predicted errors
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os

# Define phase fields and corresponding areas
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
experimental_std = 0.1
predicted_errors = [0.01, 0.02]
batch_size = 1

# File location template
data_file_template = '../simulation/data/compare_batches/{}.csv'

# Initialize an empty list to store the data
all_data = []

# Read and combine the data from all phase fields
for phase_field in phase_fields:
    df = pd.read_csv(data_file_template.format(phase_field))
    df = df[df['Batch size'] == batch_size]
    all_data.append(df)

# Concatenate all data into a single DataFrame
combined_df = pd.concat(all_data)

# Extract unique unknowns
unknowns = combined_df['Unknown'].unique()

# Function to calculate the absolute difference in score between two PEs for all unknowns
def calculate_absolute_difference(df, pe1, pe2, samples,unknowns):
    differences = []
    mindifferences = []
    for sample_num in range(0, samples+1):
        col_name = f'Score {sample_num}'
        if col_name in df.columns:
            diffs = []
            for unknown in unknowns:
                df_pe1 = df[(df['Predicted error'] == pe1) & (df['Unknown'] == unknown)]
                df_pe2 = df[(df['Predicted error'] == pe2) & (df['Unknown'] == unknown)]
                if col_name in df_pe1.columns and col_name in df_pe2.columns:
                    score_pe1 = df_pe1[col_name].median() * 100
                    score_pe2 = df_pe2[col_name].median() * 100
                    diff = score_pe1 - score_pe2
                    diffs.append(diff)
            differences.append(max(diffs))
            mindifferences.append(min(diffs))
        else:
            differences.append(np.nan)
            mindifferences.append(np.nan)
    return differences,mindifferences

# Plotting
samples = 15

plt.figure(figsize=(12, 8))

max_diffs = calculate_absolute_difference(
    combined_df[combined_df['Experimental std'] == experimental_std],
    predicted_errors[0],predicted_errors[1],samples,unknowns)[0]

plt.plot(range(samples + 1), max_diffs, label='Max Absolute Difference (1,2)')
max_diffs = calculate_absolute_difference(
    combined_df[combined_df['Experimental std'] == experimental_std],
    0.01,0.04,samples,unknowns)[0]

plt.plot(range(samples + 1), max_diffs, label='Max Absolute Difference (1,4)')
max_diffs = calculate_absolute_difference(
    combined_df[combined_df['Experimental std'] == experimental_std],
    0.02,0.04,samples,unknowns)[0]

plt.plot(range(samples + 1), max_diffs, label='Max Absolute Difference (2,4)')
max_diffs = calculate_absolute_difference(
    combined_df[combined_df['Experimental std'] == experimental_std],
    predicted_errors[0],predicted_errors[1],samples,unknowns)[1]

plt.plot(range(samples + 1), max_diffs, label='Min Absolute Difference (1,2)')
max_diffs = calculate_absolute_difference(
    combined_df[combined_df['Experimental std'] == experimental_std],
    0.01,0.04,samples,unknowns)[1]

plt.plot(range(samples + 1), max_diffs, label='Min Absolute Difference (1,4)')
max_diffs = calculate_absolute_difference(
    combined_df[combined_df['Experimental std'] == experimental_std],
    0.02,0.04,samples,unknowns)[1]

plt.plot(
    range(samples + 1), max_diffs, label='Min Absolute Difference (2,4)')

plt.xlabel('Number of Samples')
plt.ylabel('Maximum Absolute Difference in Percentage Purity /%')
plt.title(f'Maximum Absolute Difference in Score after N Samples for $\\sigma_E={experimental_std}$')
plt.legend(title='Predicted Error')
plt.grid(True)
plt.tight_layout()
plt.show()
'''
#-> impies use batch number 6
#show distribution for n'th batch
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os

# Define phase fields and corresponding areas
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
experimental_std = 0.1
predicted_errors = [0.01, 0.04]
batch_size = 1

# File location template
data_file_template = '../simulation/data/compare_batches/{}.csv'

# Initialize an empty list to store the data
all_data = []

# Read and combine the data from all phase fields
for phase_field in phase_fields:
    df = pd.read_csv(data_file_template.format(phase_field))
    df = df[df['End type'] != 'Nulled']  # Remove failures
    df = df[df['Batch size'] == batch_size]
    all_data.append(df)

# Concatenate all data into a single DataFrame
combined_df = pd.concat(all_data)

# Extract unique unknowns
unknowns = combined_df['Unknown'].unique()

# Function to calculate the difference in score between two PEs for all unknowns
def calculate_difference(df, pe1, pe2, sample_num):
    differences = []
    for unknown in unknowns:
        df_pe1 = df[(df['Predicted error'] == pe1) & (df['Unknown'] == unknown)]
        df_pe2 = df[(df['Predicted error'] == pe2) & (df['Unknown'] == unknown)]
        col_name = f'Score {sample_num}'
        if col_name in df_pe1.columns and col_name in df_pe2.columns:
            score_pe1 = df_pe1[col_name].median() * 100
            score_pe2 = df_pe2[col_name].median() * 100
            diff = score_pe1 - score_pe2
            differences.append((unknown, diff))
    return differences

# Calculate differences after 6 samples
sample_num = 6
differences = calculate_difference(combined_df[combined_df['Experimental std'] == experimental_std], predicted_errors[0], predicted_errors[1], sample_num)

# Extract unknowns and differences
unknowns, diff_values = zip(*differences)

# Plotting
plt.figure(figsize=(12, 8))
plt.bar(unknowns, diff_values, color='blue')

plt.xlabel('Unknowns')
plt.ylabel('Difference in Percentage Purity /%')
plt.title(f'Difference in Score after {sample_num} Samples between PE=0.01 and PE=0.04 for $\\sigma_E={experimental_std}$')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()
'''
#extreme compare 
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os

width_in_inches = 16.51 / 2.54 *2
aspect_ratio = 2.5/4
height_in_inches = width_in_inches * aspect_ratio
plt.rcParams['figure.figsize'] = (width_in_inches, height_in_inches)
plt.rcParams['font.size'] = 14

# Define phase fields
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
experimental_std = 0.1
predicted_errors = [0.01, 0.04]
batch_size = 1

# File location template
data_file_template = '../simulation/data/compare_batches/{}.csv'

# Initialize an empty list to store the data
all_data = []

# Read and combine the data from all phase fields
for phase_field in phase_fields:
    df = pd.read_csv(data_file_template.format(phase_field))
    df = df[df['End type'] != 'Nulled']  # Remove failures
    df = df[df['Batch size'] == batch_size]
    all_data.append(df)

# Concatenate all data into a single DataFrame
combined_df = pd.concat(all_data)

# Function to calculate median and 68% error bounds
def calculate_statistics(df, samples):
    medians = []
    lower_bounds = []
    upper_bounds = []

    for sample_num in range(0, samples+1):
        col_name = f'Score {sample_num}'
        if col_name in df.columns:
            median = df[col_name].median() * 100
            print(col_name)
            lower_bound = np.percentile(df[col_name], 16) * 100
            upper_bound = np.percentile(df[col_name], 84) * 100

            medians.append(median)
            lower_bounds.append(median - lower_bound)
            upper_bounds.append(upper_bound - median)
        else:
            medians.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)

    return medians, lower_bounds, upper_bounds

# Define the unknowns and colors
unknowns_to_plot = {
    "MgAl$_2$Cu": ("palegreen", "lime"),
    "Li$_2$AlB$_5$O$_{10}$": ("lightcoral", "orangered")
}

# Plotting
samples = 15

for unknown, (light_color, dark_color) in unknowns_to_plot.items():
    for pe, color in zip(predicted_errors, [light_color, dark_color]):
        df_filtered = combined_df[(combined_df['Unknown'] == unknown) &
                                  (combined_df['Predicted error'] == pe) &
                                  (combined_df['Experimental std'] == experimental_std)]
        medians, lower_bounds, upper_bounds = calculate_statistics(df_filtered, samples)

        valid_indices = [i for i in range(samples+1)]
        valid_medians = [medians[i] for i in valid_indices]
        valid_lower_bounds = [lower_bounds[i] for i in valid_indices]
        valid_upper_bounds = [upper_bounds[i] for i in valid_indices]
        valid_samples = [i for i in valid_indices]

        # Plot the medians and error bars
        plt.errorbar(
            valid_samples, valid_medians,
            yerr=[valid_lower_bounds, valid_upper_bounds],
            label=f'{unknown}, $\sigma_P$={pe}',
            color=color,
            capsize=10, elinewidth=0, markeredgewidth=3)

plt.xlabel('Number of Samples')
plt.ylabel('Purity Score /%')
#plt.title('How $\sigma_P$ affects PICIP\'s performance for MgAl$_2$Cu,'
#          +' Li$_2$AlB$_5$O$_{10}$, $\sigma_E=0.1$')
plt.legend(title='Parameter choices')
plt.grid(True)
plt.savefig(
    '../simulation/comparison graphs/extreme_score_vs_n.png',
    dpi=600,bbox_inches='tight')
plt.show()
'''
#code to add are unknown to phase field
'''
#f=0
#field="LiAlBO"
areas=[0.010855905568422534, 0.06023519825812226, 0.05504769425952735,
       0.05573254640131034, 0.04648471959693422, 0.029810377908400874]

phase_field="LiAlBO"
#f=3
#areas=[0.21633578830363878, 0.06871842687292054]

#phase_field="MgAlCu"
#areas=[0.41849353831792396, 0.5033310039089045, 0.15173949382547855]
data_file_template = '../simulation/data/compare_batches/{}.csv'

# Read and combine the data from all phase fields
df = pd.read_csv(data_file_template.format(phase_field))
area_mapping={}
for u,a in zip(df['Unknown'].unique(),areas):
    area_mapping[u]=a
print(area_mapping)
df['Area']=df['Unknown'].map(area_mapping)
for u,a in zip(df['Unknown'].unique(),areas):
    print(u)
    print(df[df['Unknown']==u]['Area'])
df.to_csv(data_file_template.format(phase_field),index=False)
'''
#predicted error vs area
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os

# Define phase fields
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
experimental_std = 0.1
predicted_errors = [0.01, 0.02, 0.04]
batch_size = 1

# File location template
data_file_template = '../simulation/data/compare_batches/{}.csv'

# Initialize an empty list to store the data
all_data = []

# Read and combine the data from all phase fields
for phase_field in phase_fields:
    df = pd.read_csv(data_file_template.format(phase_field))
    df = df[df['End type'] != 'Nulled']  # Remove failures
    df = df[df['Batch size'] == batch_size]
    all_data.append(df)

# Concatenate all data into a single DataFrame
combined_df = pd.concat(all_data)

# Ensure 'Area' column exists in the DataFrame
if 'Area' not in combined_df.columns:
    raise ValueError("The dataframe must contain an 'Area' column.")

# Calculate the median score after 4 samples for each unknown and predicted error
sample_num = 4
results = []

for pe in predicted_errors:
    for unknown in combined_df['Unknown'].unique():
        df_filtered = combined_df[(combined_df['Unknown'] == unknown) & 
                                  (combined_df['Predicted error'] == pe) & 
                                  (combined_df['Experimental std'] == experimental_std)]
        col_name = f'Score {sample_num}'
        if col_name in df_filtered.columns:
            median_score = df_filtered[col_name].median() * 100
            area = df_filtered['Area'].unique()[0]
            results.append({'Unknown': unknown, 'Predicted Error': pe, 'Median Score': median_score, 'Area': area})

results_df = pd.DataFrame(results)

# Plot median score after 4 samples vs area for each predicted error
plt.figure(figsize=(12, 8))
for pe in predicted_errors:
    subset = results_df[results_df['Predicted Error'] == pe]
    plt.scatter(subset['Area'], subset['Median Score'], label=f'PE={pe}')

plt.xlabel('Area')
plt.ylabel('Median Percentage Purity after 4 Samples /%')
plt.title(f'Median Percentage Purity after 4 Samples vs Area for $\\sigma_E={experimental_std}$')
plt.legend(title='Predicted Error')
plt.grid(True)
plt.tight_layout()
plt.show()

# Determine the best predicted error for each unknown
best_scores = results_df.loc[results_df.groupby('Unknown')['Median Score'].idxmax()]

# Plot the best predicted error as a function of area
plt.figure(figsize=(12, 8))great thankyou. now i want to see a bar chart for each unknown showing the failure rate. each barchart should have an x axis giving the diffent combinations of predicted error and batch size, and then for each section three bars of different colours for the different experimental std's. remember to not fileter the dataframe by end type initially!
for unknown in best_scores['Unknown'].unique():
    subset = best_scores[best_scores['Unknown'] == unknown]
    plt.scatter(subset['Area'], subset['Median Score'], label=f'{unknown}, Best PE={subset["Predicted Error"].values[0]}')

plt.xlabel('Area')
plt.ylabel('Best Median Percentage Purity after 4 Samples /%')
plt.title(f'Best Predicted Error as a Function of Area for $\\sigma_E={experimental_std}$')
plt.legend(title='Unknowns and Best Predicted Error')
plt.grid(True)
plt.tight_layout()
plt.show()
'''
#delta area
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os

width_in_inches = 16.51 / 2.54 *2
aspect_ratio = 2.5/4
height_in_inches = width_in_inches * aspect_ratio
plt.rcParams['figure.figsize'] = (width_in_inches, height_in_inches)
plt.rcParams['font.size'] = 14
# Define phase fields
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
experimental_std = 0.1
predicted_errors = [0.01, 0.04]
batch_size = 1

# File location template
data_file_template = '../simulation/data/compare_batches/{}.csv'

# Initialize an empty list to store the data
all_data = []
phase_field_mapping = {}

# Read and combine the data from all phase fields
for phase_field in phase_fields:
    df = pd.read_csv(data_file_template.format(phase_field))
    df = df[df['End type'] != 'Nulled']  # Remove failures
    df = df[df['Batch size'] == batch_size]
    df['Phase Field'] = phase_field  # Add phase field column
    all_data.append(df)
    phase_field_mapping.update({unknown: phase_field for unknown in df['Unknown'].unique()})

# Concatenate all data into a single DataFrame
combined_df = pd.concat(all_data)

# Ensure 'Area' column exists in the DataFrame
if 'Area' not in combined_df.columns:
    raise ValueError("The dataframe must contain an 'Area' column.")
# Calculate the median score after 4 samples for each unknown and predicted error
sample_num = 6
results = []

for unknown in combined_df['Unknown'].unique():
    df_unknown = combined_df[(combined_df['Unknown'] == unknown) & 
                             (combined_df['Experimental std'] == experimental_std)]
    col_name = f'Score {sample_num}'
    if col_name in df_unknown.columns:
        median_score_pe_01 = df_unknown[df_unknown['Predicted error'] == 0.01][col_name].median() * 100
        median_score_pe_04 = df_unknown[df_unknown['Predicted error'] == 0.04][col_name].median() * 100
        score_diff = median_score_pe_04 - median_score_pe_01
        area = df_unknown['Area'].unique()[0]
        phase_field = phase_field_mapping[unknown]
        results.append({'Unknown': unknown, 'Score Difference': score_diff, 'Area': area, 'Phase Field': phase_field})

results_df = pd.DataFrame(results)

# Define colors for each phase field
colors = {'MgBOF': 'blue', 'MgAlCu': 'green', 'LiAlBO': 'red'}

# Plot the score difference after 4 samples vs area for each unknown
for phase_field in phase_fields:
    subset = results_df[results_df['Phase Field'] == phase_field]
    plt.scatter(subset['Area'], subset['Score Difference'], color=colors[phase_field], label=phase_field)

plt.xlabel('Area of region containing unknown phase')
plt.ylabel('Score$_6$ when $\sigma_p=0.04$ - Score$_6$ when $\sigma_p$=0.01')
#plt.title(f'Effect of $\sigma_p$ on median percentage purity after 6 samples'
#          + ' (score), in relation to area of region containing unknown phase')
plt.legend(title='Phase Field')
plt.grid(True)
plt.savefig(
    '../simulation/comparison graphs/delta_score_vs_area.png',
    dpi=600,bbox_inches='tight')
plt.show()
'''
#average failure
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

width_in_inches = 16.51 / 2.54 *2
aspect_ratio = 2.5/4
height_in_inches = width_in_inches * aspect_ratio
plt.rcParams['figure.figsize'] = (width_in_inches, height_in_inches)
plt.rcParams['font.size'] = 14
# Define phase fields and parameters
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
experimental_stds = [0.02, 0.05, 0.1]
predicted_errors = [0.01, 0.02, 0.04]
batch_size = 1

# File location template
data_file_template = '../simulation/data/compare_batches/{}.csv'

# Initialize an empty list to store the data
all_data = []

# Read and combine the data from all phase fields
for phase_field in phase_fields:
    df = pd.read_csv(data_file_template.format(phase_field))
    df = df[df['Batch size'] == batch_size]
    all_data.append(df)

# Concatenate all data into a single DataFrame
combined_df = pd.concat(all_data)

# Calculate average failure rates for each combination of experimental std and predicted error
failure_rates = []

for exp_std in experimental_stds:
    for pe in predicted_errors:
        df_filtered = combined_df[(combined_df['Experimental std'] == exp_std) &
                                  (combined_df['Predicted error'] == pe)]
        failure_rate = (df_filtered['End type'] == 'Nulled').mean() * 100  # Convert to percentage
        failure_rates.append({'Experimental std': exp_std, 'Predicted error': pe, 'Failure Rate': failure_rate})

failure_rates_df = pd.DataFrame(failure_rates)

# Plotting
fig, ax = plt.subplots()

# Define the bar width and positions
bar_width = 0.2
bar_positions = np.arange(len(experimental_stds))

# Colors for the predicted errors
colors = ['blue', 'green', 'red']

# Plot each predicted error
for i, pe in enumerate(predicted_errors):
    pe_data = failure_rates_df[failure_rates_df['Predicted error'] == pe]
    ax.bar(bar_positions + i * bar_width, pe_data['Failure Rate'], bar_width, color=colors[i],
           label=f'$\\sigma_P={pe}$')

# Set the x-axis labels and title
ax.set_xticks(bar_positions + bar_width)
ax.set_xticklabels([f'$\\sigma_E={std}$' for std in experimental_stds])
ax.set_xlabel(r'Experimental Error ($\sigma_E$)')
ax.set_ylabel('Average Failure Rate /%')
#ax.set_title('Average Failure Rate across All Unknowns')
ax.legend(title=r'Predicted Error ($\sigma_P$)')
ax.grid(True)
plt.savefig(
    '../simulation/comparison graphs/avg_failure.png',
    dpi=600,bbox_inches='tight')
plt.show()
'''


#failure compare for edge unknowns
'''
import matplotlib.pyplot as plt
import pandas as pd
impplt.savefig(
    '../simulation/comparison graphs/batch_score_vs_n.png',
    dpi=600,bbox_inches='tight')
ort numpy as np
import re
import os

# Define the chosen unknowns
chosen_unknowns = ["MgAl$_2$Cu","Li$_2$AlB$_5$O$_{10}$"]

# Define phase fields and parameters
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
experimental_std = 0.1
predicted_errors = [0.01, 0.02, 0.04]
batch_size = 1

# File location template
data_file_template = '../simulation/data/compare_batches/{}.csv'

# Initialize an empty list to store the data
all_data = []

# Read and combine the data from all phase fields
for phase_field in phase_fields:
    df = pd.read_csv(data_file_template.format(phase_field))
    df = df[df['Batch size'] == batch_size]
    all_data.append(df)

# Concatenate all data into a single DataFrame
combined_df = pd.concat(all_data)

# Filter the DataFrame to include only the chosen unknowns and the specified experimental std
df_filtered = combined_df[(combined_df['Unknown'].isin(chosen_unknowns)) &
                          (combined_df['Experimental std'] == experimental_std)]

# Calculate failure rates for the chosen unknowns
failure_rates = []

for unknown in chosen_unknowns:
    for pe in predicted_errors:
        df_pe = df_filtered[(df_filtered['Unknown'] == unknown) & 
                            (df_filtered['Predicted error'] == pe)]
        failure_rate = (df_pe['End type'] == 'Nulled').mean()
        failure_rates.append({'Unknown': unknown, 'Predicted error': pe, 'Failure Rate': failure_rate})

failure_rates_df = pd.DataFrame(failure_rates)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Define the bar width and positions
bar_width = 0.2
bar_positions = np.arange(len(predicted_errors))

# Colors for the unknowns
colors = {"MgAl$_2$Cu": 'blue', "Li$_2$AlB$_5$O$_{10}$": 'red'}

# Plot each unknown
for i, unknown in enumerate(chosen_unknowns):
    unknown_data = failure_rates_df[failure_rates_df['Unknown'] == unknown]
    ax.bar(bar_positions + i * bar_width, unknown_data['Failure Rate'], bar_width, color=colors[unknown], label=unknown)

# Set the x-axis labels and title
ax.set_xticks(bar_positions + bar_width / 2)
ax.set_xticklabels([f'PE={pe}' for pe in predicted_errors])
ax.set_xlabel('Predicted Error')
ax.set_ylabel('Failure Rate')
ax.set_title(f'Failure Rate Comparison for Unknowns (σ_E={experimental_std})')
ax.legend(title='Unknowns')
ax.grid(True)

plt.tight_layout()
plt.show()
'''
#failure rate vs area 
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Define phase fields and parameters
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
experimental_std = 0.1
batch_size = 1

# File location template
data_file_template = '../simulation/data/compare_batches/{}.csv'

# Initialize an empty list to store the data
all_data = []
phase_field_mapping = {}

# Read and combine the data from all phase fields
for phase_field in phase_fields:
    df = pd.read_csv(data_file_template.format(phase_field))
    df = df[df['Batch size'] == batch_size]
    df['Phase Field'] = phase_field  # Add phase field column
    print(df.head())
    all_data.append(df)
    phase_field_mapping.update({unknown: phase_field for unknown in df['Unknown'].unique()})

# Concatenate all data into a single DataFrame
combined_df = pd.concat(all_data)

# Ensure 'Area' column exists in the DataFrame
if 'Area' not in combined_df.columns:
    raise ValueError("The dataframe must contain an 'Area' column.")

# Calculate failure rates for each unknown
failure_rates = []

for unknown in combined_df['Unknown'].unique():
    df_unknown = combined_df[(combined_df['Unknown'] == unknown) &
                             (combined_df['Experimental std'] == experimental_std)]
    failure_rate = (df_unknown['End type'] == 'Nulled').mean()
    area = df_unknown['Area'].unique()[0]
    phase_field = phase_field_mapping[unknown]
    failure_rates.append({'Unknown': unknown, 'Failure Rate': failure_rate, 'Area': area, 'Phase Field': phase_field})

failure_rates_df = pd.DataFrame(failure_rates)

# Define colors for each phase field
colors = {'MgBOF': 'blue', 'MgAlCu': 'green', 'LiAlBO': 'red'}

# Plotting
plt.figure(figsize=(12, 8))

for phase_field in phase_fields:
    subset = failure_rates_df[failure_rates_df['Phase Field'] == phase_field]
    plt.scatter(subset['Area'], subset['Failure Rate'], color=colors[phase_field], label=phase_field)

plt.xlabel('Area')
plt.ylabel('Failure Rate')
plt.title('Failure Rate vs Area for Each Unknown')
plt.legend(title='Phase Field')
plt.grid(True)
plt.tight_layout()
plt.show()
'''
#batch sizes
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#setup for paper
width_in_inches = 16.51 / 2.54 *2
aspect_ratio = 2.5/4
height_in_inches = width_in_inches * aspect_ratio
plt.rcParams['figure.figsize'] = (width_in_inches, height_in_inches)
plt.rcParams['font.size'] = 14

# Define phase fields and parameters
phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
predicted_error = 0.02
experimental_std = 0.05
batch_sizes = [1, 3, 5]
max_samples = 15

# File location template
data_file_template = '../simulation/data/compare_batches/{}.csv'

# Initialize an empty list to store the data
all_data = []

# Read and combine the data from all phase fields
for phase_field in phase_fields:
    df = pd.read_csv(data_file_template.format(phase_field))
    all_data.append(df)

# Concatenate all data into a single DataFrame
combined_df = pd.concat(all_data)

# Filter the DataFrame to include only the specified predicted error and experimental std
df_filtered = combined_df[(combined_df['Predicted error'] == predicted_error) &
                          (combined_df['Experimental std'] == experimental_std) &
                          (combined_df['End type'] != 'Nulled')]

# Plotting
plt.figure(figsize=(12, 8))

colors={0.02:{1:'lightcoral',3:'orangered',5:'firebrick'},
         0.1:{1:'deepskyblue',3:'blue',5:'blueviolet'}}
colors={1:'red',3:'blue',5:'green'}

#for exp_error in [0.02,0.1]:
#    df_f=df_filtered[df_filtered['Experimental std']==exp_error]
for batch_size in batch_sizes:
    sample_numbers = [0] + list(range(batch_size, max_samples + 1, batch_size))
    medians = []
    lower_bounds = []
    upper_bounds = []
    
    for sample_num in sample_numbers:
        if sample_num == 0:
            col_name = 'Score 0'
        else:
            batch_index = sample_num // batch_size
            col_name = f'Score {batch_index}'
        
        if col_name in df_filtered.columns:
            scores = df_filtered[df_filtered['Batch size'] == batch_size][col_name] * 100
            median_score = scores.median()
            lower_bounds.append(np.percentile(scores, 16))
            upper_bounds.append(np.percentile(scores, 84))
            medians.append(median_score)
        else:
            medians.append(None)
            lower_bounds.append(None)
            upper_bounds.append(None)

    c=colors[batch_size]
    plt.errorbar(sample_numbers, medians, yerr=[np.array(medians) - np.array(lower_bounds), 
                                                np.array(upper_bounds) - np.array(medians)], 
                 label=f'Batch size={batch_size}', marker='o',capsize=10, elinewidth=0, markeredgewidth=3,
                 c=c)

plt.xlabel('Number of Samples')
plt.ylabel('Purity Score /%')
#plt.title(f'Percentage Purity vs Number of Samples for $\\sigma_E={experimental_std}$, $\\sigma_P={predicted_error}$')
plt.legend(title='Parameter choices')
plt.grid(True)
#plt.tight_layout()
plt.savefig(
    '../simulation/comparison graphs/batch_score_vs_n.png',
    dpi=600,bbox_inches='tight')
plt.show()
'''
#compare old vs new
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os

width_in_inches = 16.51 / 2.54 *2
aspect_ratio = 2.5/4
height_in_inches = width_in_inches * aspect_ratio
plt.rcParams['figure.figsize'] = (width_in_inches, height_in_inches)
plt.rcParams['font.size'] = 14

# Define phase fields and corresponding areas
#phase_fields = ["MgBOF", "MgAlCu", "LiAlBO"]
experimental_stds = [0.02, 0.05, 0.1]
predicted_errors = [0.01, 0.02, 0.04]
predicted_errors_b = [0.02, 0.05, 0.1]

# File location template
data_file_template = '~/phd/composition_exploration/exploration_algorithm/simulation/data/compare_batches/{}.csv'

# Batch sizes to consider
batch_size = 1

# Initialize an empty list to store the data
all_data = []

dfa = pd.read_csv(data_file_template.format('MgBOF'))
dfb=pd.read_csv('../simulation/data/testing/beta_refined.csv')
dfa = dfa[dfa['End type'] != 'Nulled']
dfa= dfa[dfa['Batch size'] == batch_size]
dfb = dfb[dfb['End type'] != 'Nulled']
dfb = dfb[dfb['Batch size'] == batch_size]
for i in dfb.columns:
    print(i)

# Function to calculate median and 68% error bounds
def calculate_statistics(df, samples):
    medians = []
    lower_bounds = []
    upper_bounds = []

    for sample_num in range(0, samples+1):
        col_name = f'Score {sample_num}'
        if col_name in df.columns:
            median = df[col_name].median() * 100
            lower_bound = np.percentile(df[col_name], 16) * 100
            upper_bound = np.percentile(df[col_name], 84) * 100

            medians.append(median)
            lower_bounds.append(median - lower_bound)
            upper_bounds.append(upper_bound - median)
        else:
            medians.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)

    return medians, lower_bounds, upper_bounds

# Plotting
samples = 15

colours={0.02:{0.01:'lightcoral',0.02:'orangered',0.04:'firebrick'},
         0.05:{0.01:'palegreen',0.02:'lime',0.04:'olivedrab'},
         0.1:{0.01:'deepskyblue',0.02:'blue',0.04:'blueviolet'}}

for exp_std_value in experimental_stds:
    for predicted_error in predicted_errors:
        dfa_filtered = dfa[(dfa['Predicted error'] == predicted_error) &
                                  (dfa['Experimental std'] == exp_std_value)]
        medians, lower_bounds, upper_bounds = calculate_statistics(dfa_filtered, samples)

        valid_indices = [i for i in range(samples+1)]
        valid_medians = [medians[i] for i in valid_indices]
        valid_lower_bounds = [lower_bounds[i] for i in valid_indices]
        valid_upper_bounds = [upper_bounds[i] for i in valid_indices]
        valid_samples = [i for i in valid_indices]

        # Plot the medians and error bars
        c=colours[exp_std_value][predicted_error]
        plt.errorbar(
            valid_samples, valid_medians,
            yerr=[valid_lower_bounds, valid_upper_bounds],
            label=f'$\\sigma_P={predicted_error}$',
            capsize=10, elinewidth=0, markeredgewidth=3)


    plt.xlabel('Number of Samples')
    plt.ylabel('Purity Score /%')
    #plt.title('Purity Score vs number of samples, averaged over all unknowns')
    plt.legend(title='Parameter choices')
    plt.grid(True)

    for predicted_error in predicted_errors_b:
        dfb_filtered = dfb[(dfb['Predicted error'] == predicted_error) &
                                  (dfb['Experimental std'] == exp_std_value)]
        medians, lower_bounds, upper_bounds = calculate_statistics(dfb_filtered, samples)

        valid_indices = [i for i in range(samples+1)]
        valid_medians = [medians[i] for i in valid_indices]
        valid_lower_bounds = [lower_bounds[i] for i in valid_indices]
        valid_upper_bounds = [upper_bounds[i] for i in valid_indices]
        valid_samples = [i for i in valid_indices]

        # Plot the medians and error bars
        #c=colours[exp_std_value][predicted_error]
        plt.errorbar(
            valid_samples, valid_medians,
            yerr=[valid_lower_bounds, valid_upper_bounds],
            label=f'$\\sigma_P={predicted_error}$, beta',
            capsize=10, elinewidth=0, markeredgewidth=3)

    plt.xlabel('Number of Samples')
    plt.ylabel('Purity Score /%')
    #plt.title('Purity Score vs number of samples, averaged over all unknowns')
    plt.legend(title='Parameter choices')
    plt.grid(True)

    # Ensure the layout is tight
    #plt.tight_layout()

    # Show the plot

    # Set the text size to 10pt

    #plt.savefig(
    #    '../simulation/comparison graphs/uavg_score_vs_n.png',
    #    dpi=600,bbox_inches='tight')
    plt.show()


