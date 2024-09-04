#cd C:\\Users\james\OneDrive\Desktop\FYP
def main():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from st_aggrid import AgGrid, GridOptionsBuilder

    st.title("Descriptive Analysis Dashboard")
    st.sidebar.title("Settings")

    # Upload data
    injury_df = pd.read_csv('Injury_History.csv')
    players_df = pd.read_csv('players.csv')
    standings_df = pd.read_csv('TeamStandings.csv')
    players2_df = pd.read_csv('players2021-2022.csv')    

    # Preprocessing: injury_df
    injury_df['Date'] = pd.to_datetime(injury_df['Date'], format='%d/%m/%Y')
    injury_df = injury_df.sort_values('Date')
    injury_df['Injury Type'] = injury_df['Notes'].str.extract(r'(\b[\w\s]+\b)')[0]  # Extract first word before space

    #Preprocessing: players_df
    players_df.drop(['Height', 'College'], axis = 1, inplace = True)
    players_df['Position'] = players_df['Position'].replace({'PG': 'G', 'SG': 'G', 'PF': 'F', 'SF': 'F'})

    ## Scenario 3 first: If have salary but no stats, fill stats with 0
    cols_stats = ['Points', 'Rebounds', 'Assists']
    players_df.loc[players_df['Salary'].notnull(), cols_stats] = players_df.loc[players_df['Salary'].notnull(), cols_stats].fillna(0)

    ## Scenario 2: If all NaN (new player), fill all with 0
    ### This checks if all the specified columns are NaN, and if so, fills them with 0
    players_df.loc[players_df[cols_stats + ['Salary']].isnull().all(axis=1), cols_stats + ['Salary']] = 0

    ## Scenario 1: If have points/rebounds/assists, fill salary by position average
    ### Temporarily fill NaN Salaries with 0 for those without any stats to avoid affecting the mean calculation
    temp_salary_filled = players_df['Salary'].fillna(0)
    players_df.loc[:, 'Salary'] = players_df.groupby('Position')['Salary'].transform(lambda x: x.fillna(x.mean()))

    ### Correct any 0 values filled temporarily for salary with the actual mean of the position
    players_df.loc[players_df['Salary'] == 0, 'Salary'] = players_df.groupby('Position')['Salary'].transform('mean')
    
    # Preprocessing: standings_df
    standings_df[['Wins', 'Losses']] = standings_df['Overall'].str.split('-', expand=True)
    standings_df['Wins'] = pd.to_numeric(standings_df['Wins'])
    standings_df['Losses'] = pd.to_numeric(standings_df['Losses'])

    # Preprocessing: players2_df
    columns_with_nas = ['FG%', '3P%', '2P%', 'eFG%', 'FT%']
    for col in columns_with_nas:
        # Create a new column indicating missingness (1 if missing, 0 if not)
        players2_df[col + '_missing'] = players2_df[col].isnull().astype(int)
        
        # Fill missing values in the original column with 0
        players2_df[col] = players2_df[col].fillna(0)


    players2_df['Pos'] = players2_df['Pos'].replace({'PG': 'G', 'SG': 'G', 'PF': 'F', 'SF': 'F', 'SG-PG': 'G',  'SG-SF':'G', 'SF-SG':'F', 'PF-SF':'F', 'C-PF': 'C', 'SG-PG-SF':'G', 'PG-SG':'G'})
    
    players2_df.rename(columns={'Pos': 'Position', 'Player': 'Name', 'Tm':'Team'}, inplace=True)

    #categorical data: pos, tm 
    #do it when it's predictive modelling's turn
    #players2_df = pd.get_dummies(players2_df, columns=['Position', 'Team'], drop_first=False)  # Convert categorical columns to one-hot encoded
    #players2_df

    #remove: player-additional
    players2_df.drop(['Player-additional'], axis=1, inplace=True)

    #normalize: age, rk, g, mp
    min_max_scaler = MinMaxScaler()
    players2_df[['Age', 'Rk', 'G', 'MP']] = min_max_scaler.fit_transform(players2_df[['Age', 'Rk', 'G', 'MP']])

    #skewed: age, gs, mo, fg, fga, ft, fta, ft%, orb, drb, trb, ast, stl, blk, tov, pf, pts
    # For columns that are strictly positive and right-skewed
    # Note: Box-Cox requires positive values, so ensure there are no zero values
    skewed_columns = ['GS', 'FG', 'FGA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

    for col in skewed_columns:
        # Adding 1 for log transformation to handle zero values
        players2_df[col] = np.log1p(players2_df[col])
    
    def update_tot_to_latter(group):
        # If 'TOT' is present in the group
        if 'TOT' in group['Team'].values:
            # Get the last non-'TOT' team entry for the player
            latter_team = group.loc[group['Team'] != 'TOT', 'Team'].values[-1] if not group.loc[group['Team'] != 'TOT'].empty else None
            # Replace 'TOT' with the latter team
            if latter_team:
                group['Team'] = group['Team'].replace('TOT', latter_team)
        return group

    # Apply the function to your DataFrame grouped by 'Player' and 'Age'
    players2_df_updated = players2_df.groupby(['Name', 'Age'], group_keys=False).apply(update_tot_to_latter)

    players2_df_updated = players2_df_updated.drop_duplicates(subset=['Name', 'Age'], keep='last')

    # Proceed with your aggregation logic
    players2_agg = players2_df_updated.groupby(['Name', 'Age']).agg({
        'G': 'mean', 'GS': 'mean', 'MP': 'mean', 'FG': 'mean', 'FGA': 'mean',
        'FG%': 'mean', '3P': 'mean', '3PA': 'mean', '3P%': 'mean',
        '2P': 'mean', '2PA': 'mean', '2P%': 'mean', 'eFG%': 'mean',
        'FT': 'mean', 'FTA': 'mean', 'FT%': 'mean', 'ORB': 'mean',
        'DRB': 'mean', 'TRB': 'mean', 'AST': 'mean', 'STL': 'mean',
        'BLK': 'mean', 'TOV': 'mean', 'PF': 'mean', 'PTS': 'mean'
    }).reset_index()


    # Add 'Team' and 'Position' back to the aggregated dataframe
    players2_agg = players2_agg.merge(players2_df_updated[['Name', 'Age', 'Team', 'Position']].drop_duplicates(), on=['Name', 'Age'], how='left')

    # Verify no duplicates are introduced
    assert players2_agg.duplicated(subset=['Name', 'Age']).sum() == 0, "Duplicates found after merging Team and Position"

    # Merging aggregated stats with players_df and standings_df as described previously
    combined_df = pd.merge(players2_agg, players_df[['Name', 'Weight', 'Salary']], on='Name', how='left')
    combined_df = pd.merge(combined_df, standings_df[['Team', 'Overall']].drop_duplicates(), on='Team', how='left')

    cols_stats = ['PTS', 'TRB', 'AST']

    # Scenario 2: If all NaN (new player), fill all with 0
    # This checks if all the specified columns are NaN, and if so, fills them with 0
    combined_df.loc[combined_df[cols_stats + ['Salary']].isnull().all(axis=1), cols_stats + ['Salary']] = 0

    # Scenario 1: If have points/rebounds/assists, fill salary by position average
    # Temporarily fill NaN Salaries with 0 for those without any stats to avoid affecting the mean calculation
    temp_salary_filled = combined_df['Salary'].fillna(0)
    combined_df.loc[:, 'Salary'] = combined_df.groupby('Position')['Salary'].transform(lambda x: x.fillna(x.mean()))

    # Correct any 0 values filled temporarily for salary with the actual mean of the position
    combined_df.loc[combined_df['Salary'] == 0, 'Salary'] = combined_df.groupby('Position')['Salary'].transform('mean')

    # Calculate the average weight for each position
    position_avg_weight = combined_df.groupby('Position')['Weight'].mean()

    # Function to apply position average weight where weight is missing
    def fill_weight(row):
        if pd.isnull(row['Weight']):
            return position_avg_weight[row['Position']]
        else:
            return row['Weight']

    # Apply the function across the dataframe
    combined_df['Weight'] = combined_df.apply(fill_weight, axis=1)

    #combined_df = combined_df.drop('Overall', axis =1)
    for injury_part in ['ankle', 'hamstring', 'shoulder', 'elbow', 'knee', 'calf', 'finger', 'back', 'fracture', 'hip', 'toe']:
        injury_df[f'{injury_part}_injury'] = injury_df['Notes'].str.contains(injury_part, case=False, na=False).astype(int)

    injury_parts = ['ankle', 'hamstring', 'shoulder', 'elbow', 'knee', 'calf', 'finger', 'back', 'fracture', 'hip', 'toe']
    agg_injury_data = injury_df.groupby('Name')[[f'{part}_injury' for part in injury_parts]].sum().reset_index()

    def label_severity(note):
        if '(DTD)' in note:
            return 'Day-to-Day'
        elif 'placed on IL' in note:
            return 'Injured List'
        elif 'out for season' in note or 'out indefinitely' in note:
            return 'Out for Season/Indefinitely'
        else:
            return 'Unknown'
        
    def label_severity_numeric(note):
        if 'DTD' in note:
            return 1  # Day-to-Day
        elif 'placed on IL' in note:
            return 2  # Injured List
        elif 'out for season' in note or 'out indefinitely' in note:
            return 3  # Out for Season/Indefinitely
        else:
            return 0  # Unknown

    injury_df['Injury Severity'] = injury_df['Notes'].apply(label_severity)
    injury_df['Injury Severity Numeric'] = injury_df['Notes'].apply(label_severity_numeric)

    severity_pivot = injury_df.pivot_table(index='Name', columns='Injury Severity', aggfunc='size', fill_value=0)
    severity_pivot.columns = [column.replace(" ", "_") for column in severity_pivot.columns]  # Optional: Make column names code-friendly
    severity_pivot.reset_index(inplace=True)

    final_df = pd.merge(combined_df, agg_injury_data, on='Name', how='left')
    final_df = pd.merge(final_df, severity_pivot, on='Name', how='left')

    final_df_descriptive = final_df.copy()
    final_df_descriptive = final_df_descriptive.fillna(0)
    final_df_predictive = final_df.copy()

    # Calculating position averages for each injury type
    position_averages = final_df_predictive.groupby('Position')[[
        'ankle_injury', 'hamstring_injury', 'shoulder_injury', 'knee_injury',
        'calf_injury', 'finger_injury', 'back_injury', 'fracture_injury',
        'hip_injury', 'toe_injury', 'Day-to-Day', 'Injured_List',
        'Out_for_Season/Indefinitely', 'Unknown'
    ]].mean()

    # Filling missing values with position averages
    for col in position_averages.columns:
        final_df_predictive[col] = final_df_predictive.groupby('Position')[col].transform(lambda x: x.fillna(x.mean()))

    final_df_predictive = pd.get_dummies(final_df_predictive, columns = ['Team', 'Position'])
    
    #Additional Preprocessing on the final dataframe
    injury_parts = ['ankle_injury', 'hamstring_injury', 'shoulder_injury', 'knee_injury', 'calf_injury',
                    'finger_injury', 'back_injury', 'fracture_injury', 'hip_injury', 'toe_injury', 'elbow_injury']

    final_df_descriptive['Total Injuries'] = final_df_descriptive[injury_parts].sum(axis=1)
    injury_df['Year'] = injury_df['Date'].dt.year
    injury_df['Month'] = injury_df['Date'].dt.month
    injury_types = ['ankle_injury', 'hamstring_injury', 'shoulder_injury', 'knee_injury', 'calf_injury', 'finger_injury', 'back_injury', 'fracture_injury', 'hip_injury', 'toe_injury', 'elbow_injury']
    age_bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    age_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
    final_df_descriptive['Age_Group'] = pd.cut(final_df_descriptive['Age'], bins=age_bins, labels=age_labels, right=False)
    weight_bins = [150, 200, 250, 300]
    final_df_descriptive['Weight_Group'] = pd.cut(final_df_descriptive['Weight'], bins=weight_bins)
    team_mapping = {'DEN': 'Denver Nuggets', 'PHI': 'Philadelphia 76ers', 
                                             'PHO': 'Phoenix Suns', 'BOS': 'Boston Celtics',
                                             'OKC': 'Oklahoma City Thunder', 'DET': 'Detroit Pistons',
                                             'MIL': 'Milwaukee Bucks', 'SAC': 'Sacramento Kings',
                                             'ORL': 'Orlando Magic', 'IND': 'Indiana Pacers',
                                             'NYK': 'New York Knicks', 'CHI': 'Chicago Bulls',
                                             'NOP': 'New Orleans Pelicans', 'HOU': 'Houston Rockets',
                                             'LAC': 'Los Angeles Clippers', 'BRK': 'Brooklyn Nets',
                                             'GSW': 'Golden State Warriors', 'POR': 'Portland Trail Blazers',
                                             'LAL': 'Los Angeles Lakers', 'MIN': 'Minnesota Timberwolves',
                                             'WAS': 'Washington Wizards', 'SAS': 'San Antonio Spurs',
                                             'TOR': 'Toronto Raptors', 'CHO': 'Charlotte Hornets',
                                             'MIA': 'Miami Heat', 'ATL': 'Atlanta Hawks',
                                             'DAL': 'Dallas Mavericks', 'MEM': 'Memphis Grizzlies',
                                             'UTA': 'Utah Jazz', 'CLE': 'Cleveland Cavaliers'}  # Complete mapping provided in your environment

    def plot_average_injuries_by_age_and_position(df):
        age_bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
        age_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
        df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
        injury_by_age_pos = df.groupby(['Position', 'Age_Group'])['Total Injuries'].mean().reset_index()
        fig = px.bar(injury_by_age_pos, x='Position', y='Total Injuries', color='Age_Group', barmode='group',
                    labels={'Total Injuries': 'Average Total Injuries'}, title='Average Total Injuries by Position and Age Group')
        st.plotly_chart(fig)

    def plot_playoff_injury_comparison(df_descriptive, df_standings):
        df_descriptive['Team'] = df_descriptive['Team'].replace(team_mapping)
        df_standings['Made_Playoffs'] = df_standings['Rk'].apply(lambda x: 1 if x <= 20 else 0)
        
        # Merge the descriptive data with standings to include playoff information
        df_playoffs = pd.merge(df_descriptive, df_standings[['Team', 'Made_Playoffs']], on='Team', how='left')
        
        # Standard color palette
        color_palette = px.colors.qualitative.Plotly
        
        # Aggregate data for plotting average injuries
        injuries = df_playoffs.groupby('Made_Playoffs')['Total Injuries'].mean().reset_index()
        fig = px.bar(injuries, x='Made_Playoffs', y='Total Injuries', text='Total Injuries',
                    labels={'Made_Playoffs': 'Made Playoffs (1 = Yes, 0 = No)', 'Total Injuries': 'Average Number of Injuries'},
                    title='Average Injuries: Playoff Teams vs. Non-Playoff Teams',
                    color_discrete_sequence=[color_palette[0]])  # Use first color in palette
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)

        # Additional Metrics
        performance_metrics = ['PTS', 'AST', 'TRB']
        for metric in performance_metrics:
            fig = px.bar(df_playoffs, x='Made_Playoffs', y=metric, color='Made_Playoffs',
                        labels={'Made_Playoffs': 'Made Playoffs', metric: f'Average {metric}'},
                        title=f'Average {metric}: Playoff Teams vs. Non-Playoff Teams',
                        color_discrete_map={0: color_palette[1], 1: color_palette[2]})  # Map colors to playoff status
            st.plotly_chart(fig)

        # Injury Types
        for injury in injury_types:
            injury_data = df_playoffs.groupby('Made_Playoffs')[injury].sum().reset_index()
            fig = px.bar(injury_data, x='Made_Playoffs', y=injury, color='Made_Playoffs',
                        labels={'Made_Playoffs': 'Made Playoffs', injury: 'Total Injuries'},
                        title=f'Total {injury.replace("_", " ").title()}: Playoff Teams vs. Non-Playoff Teams',
                        color_discrete_map={0: color_palette[1], 1: color_palette[2]})  # Consistent color mapping
            st.plotly_chart(fig)

        # Correlation metrics
        correlation_data = df_playoffs[['Made_Playoffs', 'Total Injuries', 'PTS', 'AST', 'TRB']].corr()
        st.write('Correlation Matrix:', correlation_data)

    def plot_monthly_trends(df):
        monthly_data = df.groupby(['Year', 'Month']).size().reset_index(name='Count')
        fig = px.line(monthly_data, x='Month', y='Count', color='Year', title='Monthly Trends in Injuries',
                    labels={'Count': 'Number of Injuries', 'Month': 'Month of the Year'})
        st.plotly_chart(fig)

    def plot_yearly_trends(df):
        yearly_data = df.groupby('Year').size().reset_index(name='Count')
        fig = px.bar(yearly_data, x='Year', y='Count', title='Yearly Trends in Injuries')
        st.plotly_chart(fig)

    def plot_injured_vs_noninjured_stats(df_descriptive):
        df_descriptive['Injury_Status'] = df_descriptive['Total Injuries'].apply(lambda x: 'Injured' if x > 0 else 'Non-Injured')
        metrics_to_visualize = ['Age', 'G', 'PTS', 'TRB', 'AST']

        # Creating a single figure for all metrics
        fig = make_subplots(rows=len(metrics_to_visualize), cols=1, subplot_titles=[f'Distribution of {metric} for Injured vs. Non-Injured Players' for metric in metrics_to_visualize])

        for i, metric in enumerate(metrics_to_visualize, start=1):
            for status in df_descriptive['Injury_Status'].unique():
                filtered_df = df_descriptive[df_descriptive['Injury_Status'] == status]
                fig.add_trace(go.Box(y=filtered_df[metric], name=status), row=i, col=1)

        fig.update_layout(height=300 * len(metrics_to_visualize), width=700, boxmode='group')
        st.plotly_chart(fig)

    def plot_injury_by_age_group(df):
        injury_by_age_group = df.groupby('Age_Group')['Total Injuries'].sum().reset_index()
        fig = px.bar(injury_by_age_group, x='Age_Group', y='Total Injuries', title='Injuries by Age Group', color='Total Injuries')
        st.plotly_chart(fig)

    def plot_injury_by_weight_group(df_descriptive):
            fig, ax = plt.subplots()
            injury_by_weight_group = df_descriptive.groupby('Weight_Group')[injury_types].sum()
            injury_by_weight_group.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title('Injury Counts by Weight Group')
            st.pyplot(fig)

    def plot_top_injured_players(df):
        top_players = df.groupby('Name')['Total Injuries'].sum().nlargest(20)
        fig = px.bar(top_players, text='value',
                    labels={'value': 'Frequency of Injuries', 'Name': 'Player'},
                    title='Top 20 Players by Frequency of Injuries')
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    def plot_injuries_by_body_part_positions(df):
        injury_by_position = df.groupby('Position')[injury_types].sum()
        fig = px.bar(injury_by_position, barmode='group',
                    labels={'value': 'Sum of Injuries', 'variable': 'Injury Type'},
                    title='Sum of Injuries by Body Part Across Positions')
        fig.update_layout(xaxis_title='Position', yaxis_title='Sum of Injuries',
                        xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, use_container_width=True)

    def plot_total_injuries_by_body_part(df):
        injuries_totals = df[injury_types].sum().sort_values(ascending=False)
        fig = px.bar(injuries_totals, labels={'index': 'Injury Type', 'value': 'Total Injuries'},
                    title='Total Injuries by Body Part')
        fig.update_traces(texttemplate='%{value}', textposition='outside')
        fig.update_layout(xaxis_title='Injury Type', yaxis_title='Total Injuries')
        st.plotly_chart(fig, use_container_width=True)


    def plot_age_distribution_by_injury_type(df):
        fig = go.Figure()
        for injury in injury_types:
            filtered_df = df[df[injury] > 0]
            fig.add_trace(go.Histogram(x=filtered_df['Age'], name=injury))

        fig.update_layout(barmode='overlay', title='Age Distribution by Injury Type', xaxis_title='Age', yaxis_title='Frequency')
        fig.update_traces(opacity=0.6)
        st.plotly_chart(fig)

    def plot_team_injuries(df_descriptive, df_standings):
        # First, replace team abbreviations with full names in both descriptive and standings dataframes
        team_mapping = {
            'DEN': 'Denver Nuggets', 'PHI': 'Philadelphia 76ers',
            'PHO': 'Phoenix Suns', 'BOS': 'Boston Celtics',
            'OKC': 'Oklahoma City Thunder', 'DET': 'Detroit Pistons',
            'MIL': 'Milwaukee Bucks', 'SAC': 'Sacramento Kings',
            'ORL': 'Orlando Magic', 'IND': 'Indiana Pacers',
            'NYK': 'New York Knicks', 'CHI': 'Chicago Bulls',
            'NOP': 'New Orleans Pelicans', 'HOU': 'Houston Rockets',
            'LAC': 'Los Angeles Clippers', 'BRK': 'Brooklyn Nets',
            'GSW': 'Golden State Warriors', 'POR': 'Portland Trail Blazers',
            'LAL': 'Los Angeles Lakers', 'MIN': 'Minnesota Timberwolves',
            'WAS': 'Washington Wizards', 'SAS': 'San Antonio Spurs',
            'TOR': 'Toronto Raptors', 'CHO': 'Charlotte Hornets',
            'MIA': 'Miami Heat', 'ATL': 'Atlanta Hawks',
            'DAL': 'Dallas Mavericks', 'MEM': 'Memphis Grizzlies',
            'UTA': 'Utah Jazz', 'CLE': 'Cleveland Cavaliers'
        }
        df_descriptive['Team'] = df_descriptive['Team'].replace(team_mapping)
        df_standings['Team'] = df_standings['Team'].replace(team_mapping)

        # Determine playoff status based on the rankings
        df_standings['Made_Playoffs'] = df_standings['Rk'].apply(lambda x: 1 if x <= 20 else 0)
        playoff_teams = df_standings[df_standings['Made_Playoffs'] == 1]['Team']

        # Group injuries by team
        team_injuries = df_descriptive.groupby('Team')['Total Injuries'].sum().reset_index()

        # Filter to include only playoff teams
        team_injuries = team_injuries[team_injuries['Team'].isin(playoff_teams)].sort_values(by='Total Injuries', ascending=False)

        if not team_injuries.empty:
            fig = px.bar(team_injuries, x='Team', y='Total Injuries', title='Injuries in Playoff Teams', labels={'Total Injuries': 'Total Number of Injuries', 'Team': 'Team'})
            fig.update_layout(xaxis_title='Team', yaxis_title='Total Number of Injuries', xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig)
        else:
            st.write("No playoff teams found in the filtered data.")

    def display_interactive_table(df):
        gb = GridOptionsBuilder.from_dataframe(df)

        # Configure global settings for all columns
        gb.configure_pagination(paginationAutoPageSize=True)  # Enable auto pagination
        gb.configure_side_bar()  # Enable sidebar for filtering
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
        gb.configure_selection('single')  # Enable single row selection
        grid_options = gb.build()

        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            enable_enterprise_modules=True,
            update_mode='MODEL_CHANGED',
            fit_columns_on_grid_load=True
        )
        return grid_response

     # Define available visualizations in a dictionary to control the display of filters
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Yearly Trends", "Monthly Trends",
         "Sum of Injuries by Body Part Across Positions",
        "Total Injuries by Body Part", "Top 20 Players by Frequency of Injuries",
        "Average Injuries by Age and Position", "Playoff Injury Comparison",
        "Team Injuries Analysis",
        "Injured vs Non-Injured Stats", "Injury by Age Group", "Injury by Weight Group",
        "Age Distribution by Injury Type"]
    )


    # Main content area
    # Only show year and month filters for time-based analyses
    if analysis_type in ["Yearly Trends", "Monthly Trends"]:
        year_options = ['All Years'] + sorted(injury_df['Year'].unique().tolist())
        selected_year = st.sidebar.selectbox('Select Year', year_options)
        
        if selected_year != 'All Years':
            filtered_df = injury_df[injury_df['Year'] == selected_year]
        else:
            filtered_df = injury_df.copy()

        if analysis_type == "Monthly Trends":
            month_options = ['All Months'] + sorted(filtered_df['Month'].unique().tolist())
            selected_month = st.sidebar.selectbox('Select Month', month_options)
            if selected_month != 'All Months':
                filtered_df = filtered_df[filtered_df['Month'] == selected_month]

        # Plotting functions are called based on the filtered data
        if analysis_type == "Yearly Trends":
            plot_yearly_trends(filtered_df)
        elif analysis_type == "Monthly Trends":
            plot_monthly_trends(filtered_df)

        if st.checkbox('Show Raw Data'):
            st.write(injury_df) 
        if st.button('Show Interactive Table'):
            response = display_interactive_table(injury_df)
            if response:
                selected = response['selected_rows']
                st.write('Selected Rows:', selected)
    elif analysis_type == "Average Injuries by Age and Position":
        plot_average_injuries_by_age_and_position(final_df_descriptive)
        if st.checkbox('Show Raw Data'):
            st.write(final_df_descriptive) 
        if st.button('Show Interactive Table'):
            response = display_interactive_table(final_df_descriptive)
            if response:
                selected = response['selected_rows']
                st.write('Selected Rows:', selected)
    elif analysis_type == 'Sum of Injuries by Body Part Across Positions':
        plot_injuries_by_body_part_positions(final_df_descriptive)
        if st.checkbox('Show Raw Data'):
            st.write(final_df_descriptive)
        if st.button('Show Interactive Table'):
            response = display_interactive_table(final_df_descriptive)
            if response:
                selected = response['selected_rows']
                st.write('Selected Rows:', selected) 
    elif analysis_type == "Total Injuries by Body Part":
        plot_total_injuries_by_body_part(final_df_descriptive)
        if st.checkbox('Show Raw Data'):
            st.write(final_df_descriptive) 
        if st.button('Show Interactive Table'):
            response = display_interactive_table(final_df_descriptive)
            if response:
                selected = response['selected_rows']
                st.write('Selected Rows:', selected)
    elif analysis_type == "Top 20 Players by Frequency of Injuries":
        plot_top_injured_players(final_df_descriptive)
        if st.checkbox('Show Raw Data'):
            st.write(final_df_descriptive)
        if st.button('Show Interactive Table'):
            response = display_interactive_table(final_df_descriptive)
            if response:
                selected = response['selected_rows']
                st.write('Selected Rows:', selected) 
    elif analysis_type == "Playoff Injury Comparison":
        plot_playoff_injury_comparison(final_df_descriptive, standings_df)
        if st.checkbox('Show Raw Data'):
            st.write(final_df_descriptive)
            st.write(standings_df)
        if st.button('Show Interactive Table - Descriptive'):
            response = display_interactive_table(final_df_descriptive)
        if st.button('Show Interactive Table - Standings'):
            response = display_interactive_table(standings_df)
            if response:
                selected = response['selected_rows']
                st.write('Selected Rows:', selected)
    elif analysis_type == "Team Injuries Analysis":
        plot_team_injuries(final_df_descriptive, standings_df)
        if st.checkbox('Show Raw Data'):
                    st.write(final_df_descriptive)
                    st.write(standings_df)
        if st.button('Show Interactive Table - Descriptive'):
            response = display_interactive_table(final_df_descriptive)
        if st.button('Show Interactive Table - Standings'):
            response = display_interactive_table(standings_df)
            if response:
                selected = response['selected_rows']
                st.write('Selected Rows:', selected)            
    elif analysis_type == "Injured vs Non-Injured Stats":
        plot_injured_vs_noninjured_stats(final_df_descriptive)
        if st.checkbox('Show Raw Data'):
            st.write(final_df_descriptive) 
        if st.button('Show Interactive Table'):
            response = display_interactive_table(final_df_descriptive)
            if response:
                selected = response['selected_rows']
                st.write('Selected Rows:', selected) 
    elif analysis_type == "Injury by Age Group":
        plot_injury_by_age_group(final_df_descriptive)
        if st.checkbox('Show Raw Data'):
            st.write(final_df_descriptive) 
        if st.button('Show Interactive Table'):
            response = display_interactive_table(final_df_descriptive)
            if response:
                selected = response['selected_rows']
                st.write('Selected Rows:', selected) 
    elif analysis_type == "Injury by Weight Group":
        plot_injury_by_weight_group(final_df_descriptive)
        if st.checkbox('Show Raw Data'):
            st.write(final_df_descriptive) 
        if st.button('Show Interactive Table'):
            response = display_interactive_table(final_df_descriptive)
            if response:
                selected = response['selected_rows']
                st.write('Selected Rows:', selected) 
    elif analysis_type == "Age Distribution by Injury Type":
        plot_age_distribution_by_injury_type(final_df_descriptive)
        if st.checkbox('Show Raw Data'):
            st.write(final_df_descriptive) 
        if st.button('Show Interactive Table'):
            response = display_interactive_table(final_df_descriptive)
            if response:
                selected = response['selected_rows']
                st.write('Selected Rows:', selected) 
    # Optional: display the raw data 
    # HOW ABOUT THE OTHER DATA FRAMES??? DIVE IN VIA THIS POV?
    
if __name__ == "__main__":
    main()
