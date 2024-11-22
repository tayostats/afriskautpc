import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from mplsoccer import PyPizza
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# Load the dataset
df = pd.read_csv("general.csv")  # Update with the actual path to your data

# Define Role-Based Metrics
role_metrics = {
    'goalkeeper': ['percentile_rate_saves', 'percentile_rate_conceded', 'percentile_rate_claims'],
    'right back': ['percentile_success_rate_passes', 'percentile_success_rate_tackles', 'percentile_rate_int_value',
                   'percentile_rate_block_value', 'percentile_success_rate_ground_duels', 'percentile_success_rate_aerial_duels',
                   'percentile_rate_clearance_value', 'percentile_success_rate_crosses'],
    'left back': ['percentile_success_rate_passes', 'percentile_success_rate_tackles', 'percentile_rate_int_value',
                  'percentile_rate_block_value', 'percentile_success_rate_ground_duels', 'percentile_success_rate_aerial_duels',
                  'percentile_rate_clearance_value', 'percentile_success_rate_crosses'],
    'center back': ['percentile_success_rate_passes', 'percentile_success_rate_tackles', 'percentile_rate_int_value', 
                    'percentile_rate_block_value', 'percentile_success_rate_ground_duels', 'percentile_success_rate_aerial_duels',
                    'percentile_rate_clearance_value'],
    'defensive midfielder': ['percentile_pass_value', 'percentile_progressive', 'percentile_key_pass_value', 
                             'percentile_dribble_value', 'percentile_tackle_value', 'percentile_ground_duels', 
                             'percentile_aerial_duels'],
    'central midfielder': ['percentile_pass_value', 'percentile_progressive', 'percentile_key_pass_value', 
                           'percentile_dribble_value', 'percentile_tackle_value', 'percentile_ground_duels', 
                           'percentile_aerial_duels', 'percentile_assists'],
    'attacking midfielder': ['percentile_pass_value', 'percentile_progressive', 'percentile_key_pass_value', 
                             'percentile_dribble_value', 'percentile_ground_duels', 'percentile_assists', 'percentile_goals'],
    'right winger': ['percentile_dribbles', 'percentile_key_pass_value', 'percentile_shots', 'percentile_goals', 
                     'percentile_assists', 'percentile_crosses', 'percentile_ground_duels', 'percentile_aerial_duels'],
    'left winger': ['percentile_dribbles', 'percentile_key_pass_value', 'percentile_shots', 'percentile_goals', 
                    'percentile_assists', 'percentile_crosses', 'percentile_ground_duels', 'percentile_aerial_duels'],
    'supporting striker': ['percentile_dribbles', 'percentile_key_pass_value', 'percentile_shots', 'percentile_goals', 
                           'percentile_assists', 'percentile_crosses', 'percentile_ground_duels', 'percentile_aerial_duels'],
    'center forward': ['percentile_dribbles', 'percentile_key_pass_value', 'percentile_shots', 'percentile_goals', 
                       'percentile_assists', 'percentile_crosses', 'percentile_ground_duels', 'percentile_aerial_duels'],
}
role_metrics2 = {
    'goalkeeper': ['rate_saves', 'rate_conceded', 'rate_claims', 'success_rate_passes','attempted_passes','general_metric'],
    'right back': ['success_rate_passes', 'attempted_passes','success_rate_tackles','attempted_tackles', 'rate_int_value',
                   'rate_block_value', 'success_rate_ground_duels','attempted_ground_duels', 'success_rate_aerial_duels',
                   'attempted_aerial_duels','rate_clearance_value', 'success_rate_crosses','attempted_crosses', 'success_rate_dribbles', 'attempted_dribbles', 
          'cumulative_assist','cumulative_is_goal','rate_progressive_carries','general_metric'],
    'left back': ['success_rate_passes', 'attempted_passes','success_rate_tackles','attempted_tackles', 'rate_int_value',
                   'rate_block_value', 'success_rate_ground_duels','attempted_ground_duels', 'success_rate_aerial_duels',
                   'attempted_aerial_duels','rate_clearance_value', 'success_rate_crosses','attempted_crosses', 'success_rate_dribbles', 'attempted_dribbles',
          'cumulative_assist','cumulative_is_goal','rate_progressive_carries','general_metric'],
    'center back': ['success_rate_passes','attempted_passes', 'success_rate_tackles','attempted_tackles', 'rate_int_value', 
                    'rate_block_value', 'success_rate_ground_duels', 'attempted_ground_duels', 'success_rate_aerial_duels',
                    'attempted_aerial_duels','rate_clearance_value', 'rate_progressive_carries','success_rate_dribbles', 'attempted_dribbles',
          'cumulative_assist','cumulative_is_goal','general_metric'],
    'defensive midfielder': ['success_rate_passes', 'attempted_passes','success_rate_dribbles','attempted_dribbles', 'success_rate_tackles',
          'attempted_tackles','rate_assist',
          'cumulative_assist','cumulative_is_goal', 'rate_key_pass_value', 'success_rate_ground_duels', 'attempted_ground_duels',
          'success_rate_aerial_duels','attempted_aerial_duels', 'rate_progressive', 'rate_progressive_carries','general_metric'],
    'central midfielder': ['success_rate_passes','attempted_passes', 'success_rate_dribbles', 'attempted_dribbles','success_rate_tackles','rate_progressive_carries','attempted_tackles',
          'rate_assist',
          'cumulative_assist','cumulative_is_goal', 'rate_key_pass_value', 'success_rate_ground_duels', 'attempted_ground_duels',
          'success_rate_aerial_duels', 'attempted_aerial_duels','rate_progressive', 'general_metric'],
    'attacking midfielder': ['success_rate_passes','attempted_passes', 'success_rate_dribbles','attempted_dribbles', 'success_rate_tackles','attempted_tackles',
          'rate_assist',
          'cumulative_assist','cumulative_is_goal', 'rate_key_pass_value', 'success_rate_ground_duels','attempted_ground_duels', 
          'success_rate_aerial_duels', 'attempted_aerial_duels','rate_progressive','rate_progressive_carries', 'general_metric'],
    'right winger': ['success_rate_dribbles', 'attempted_dribbles','success_rate_shots', 'attempted_shots', 
          'rate_assists','cumulative_assists','rate_goals', 'cumulative_is_goal', 'success_rate_crosses', 'attempted_crosses','success_rate_ground_duels', 'attempted_ground_duels',
          'success_rate_aerial_duels', 'attempted_aerial_duels','rate_key_pass_value', 'rate_progressive_carries', 'rate_dribbles', 'general_metric'],
    'left winger': ['success_rate_dribbles', 'attempted_dribbles','success_rate_shots', 'attempted_shots', 
          'rate_assists','cumulative_assists','rate_goals', 'cumulative_is_goal', 'success_rate_crosses', 'attempted_crosses','success_rate_ground_duels', 'attempted_ground_duels',
          'success_rate_aerial_duels', 'attempted_aerial_duels','rate_key_pass_value', 'rate_progressive_carries', 'rate_dribbles', 'general_metric'],
    'supporting striker': ['success_rate_dribbles', 'attempted_dribbles','success_rate_shots', 'attempted_shots', 
          'rate_assists','cumulative_assists','rate_goals', 'cumulative_is_goal', 'success_rate_crosses', 'attempted_crosses','success_rate_ground_duels', 'attempted_ground_duels',
          'success_rate_aerial_duels', 'attempted_aerial_duels','rate_key_pass_value','rate_progressive_carries',  'rate_dribbles', 'general_metric'],
    'center forward': ['success_rate_dribbles', 'attempted_dribbles','success_rate_shots', 'attempted_shots', 
          'rate_assists','cumulative_assists','rate_goals', 'cumulative_is_goal', 'success_rate_crosses', 'attempted_crosses','success_rate_ground_duels', 'attempted_ground_duels',
          'success_rate_aerial_duels', 'attempted_aerial_duels','rate_key_pass_value', 'rate_progressive_carries', 'rate_dribbles', 'general_metric'],
}
parameter_display_names2 = {
            'success_rate_dribbles': "Dribble Success Rate (%)",
            'rate_key_pass_value': "Key Passes per 90",
            'success_rate_shots': "Shot Accuracy (%)",
            'rate_goals': "Goals per 90",
            'rate_dribbles': "Completed Dribbles per 90",
            'rate_assists': "Assists per 90",
            'rate_assist': "Assists per 90",
            'success_rate_crosses': "Cross Completion Rate (%)",
            'success_rate_ground_duels': "Ground Duel Success Rate (%)",
            'success_rate_aerial_duels': "Aerial Duel Success Rate (%)",
            'rate_progressive': "Progressive Passes Per 90",
            'tackle_value': "Tackle Success Rate (%)",
            'success_rate_passes': "Pass Completion Rate (%)",
            'success_rate_tackles': "Tackle Success Rate (%)",
            'rate_int_value': "Interceptions Per 90",
            'rate_block_value': "Blocks Per 90",
            'success_rate_ground_duels': "Ground Duel Success Rate (%)",
            'success_rate_aerial_duels': "Aerial Duel Success Rate (%)",
            'rate_clearance_value': "Clearances Per 90",
            'rate_saves': "Saves Per 90",
            'rate_conceded': "Goals Conceded Per 90",
            'rate_claims': "Claims Per 90",
            'attempted_passes': 'Attempted Passes',
            'attempted_crosses': 'Attempted Crosses',
            'attempted_tackles': 'Attempted Tackles',
            'attempted_ground_duels': 'Attempted Ground Duels',
            'attempted_aerial_duels': 'Attempted Aerial Duels',
            'attempted_dribbles': 'Attempted Dribbles',
            'attempted_shots': 'Attempted Shots',
            'general_metric': "Aggregate Weighted Score",
            'cumulative_is_goal':'Goals',
            'cumulative_assist': 'Assists',
            'cumulative_assists': 'Assists',
            'rate_progressive_carries': 'Progressive Carries Per 90'
        }
parameter_display_names3 = {'player_name': 'Player Name',
                           'matches_played': 'Matches Played',
                           'team_name': 'Team Name',
                           'player_role': 'Player Role',
                           'country_y': 'Region',
                           'level_y': 'Level'}

# Core columns to display for each player
core_columns = ['player_name', 'matches_played', 'team_name', 'country_y', 'level_y', 'player_role']

# Streamlit Multi-Page Setup
st.set_page_config(page_title="Afriskaut Internal Scouting Tool", layout="wide")
st.sidebar.image("Afriskaut Logo White.png", use_column_width=True)
# Add navigation between the two sections
page = st.sidebar.selectbox("Select page", ["Metric Table", "Player Similarity & Percentile Rankings", "Interactive Plots"])

# DataFrame Demo Page
if page == "Metric Table":
    st.image("Afriskaut Logo White.png", width=100)
    
    # Handle missing player roles
    df['player_role'] = df['player_role'].fillna('Goalkeeper')
    
    # Define position categories for organizational purposes
    position_categories = {
        'Defender': ['Center Back', 'Left Back', 'Right Back'],
        'Midfielder': ['Defensive Midfielder', 'Central Midfielder', 'Attacking Midfielder'],
        'Forward': ['Left Winger', 'Right Winger', 'Supporting Striker', 'Center Forward'],
        'Goalkeeper': ['Goalkeeper']
    }
    
    # Define forward roles
    forward_roles = {'Left Winger', 'Right Winger', 'Supporting Striker', 'Center Forward'}
    
    # Function to filter players by role
    def filter_by_selected_role(df, selected_role):
        primary_role_df = df[df['player_role'].str.split(',').str[0].str.strip() == selected_role]
        if selected_role in forward_roles:
            combined_roles_df = df[
                df['player_role'].apply(lambda x: any(role.strip() == selected_role for role in x.split(','))) &
                df['player_role'].apply(lambda x: set(r.strip() for r in x.split(',')) <= forward_roles)
            ]
            return pd.concat([primary_role_df, combined_roles_df]).drop_duplicates()
        return primary_role_df
    
    # Search by player name (remains untouched)
    player_name_search = st.text_input("Search Player by Name (Optional)", "", help="Enter a player name to search.")
    
    if player_name_search:
        search_df = df[df['player_name'].str.contains(player_name_search, case=False, na=False)]
        if not search_df.empty:
            player_role = search_df.iloc[0]['player_role'].split(',')[0].strip().lower()
            role_columns = role_metrics2.get(player_role, [])
            display_columns = core_columns + role_columns
            search_df = search_df[display_columns]
            for col in search_df.columns:
                if col.startswith("success_"):
                    search_df[col] *= 100
            search_df = search_df.rename(columns={**parameter_display_names2, **parameter_display_names3})
            st.write(f"## Search Results for '{player_name_search}'")
            st.dataframe(search_df)
        else:
            st.warning(f"No player found with the name '{player_name_search}'.")
    else:
        # Filters: Country and Level
        st.write("### Filters")
        
        # Country Filter
        countries = sorted(df['country_y'].dropna().unique())
        selected_country = st.selectbox("Select Country", ['All'] + countries, help="Filter by country.")
        
        # Apply country filter
        if selected_country != 'All':
            df = df[df['country_y'] == selected_country]
        
        # Level Filter
        levels = sorted(df['level_y'].dropna().unique())
        levels = ['Unclassified'] + [level for level in levels if level != '-']
        selected_level = st.selectbox("Select Level", ['All'] + levels, help="Filter by player level.")
        
        # Apply level filter
        if selected_level != 'All':
            if selected_level == 'Unclassified':
                df = df[df['level_y'] == '-']
            else:
                df = df[df['level_y'] == selected_level]
        
        # Position and Role Filters
        selected_position_category = st.selectbox("Select Position Category", list(position_categories.keys()), help="Choose a position category.")
        available_roles = position_categories[selected_position_category]
        selected_role = st.selectbox("Select Specific Player Role", available_roles, help="Choose a player role to view relevant metrics.")
        
        # Matches Played Filter
        matches_played = st.slider("Matches Played", min_value=1, max_value=7, value=3, help="Filter players by the number of matches played.")
        df = df[df['matches_played'] >= matches_played]
        
        # Apply role filtering
        df = filter_by_selected_role(df, selected_role)
        
        # Select columns based on role-specific metrics
        role_columns = role_metrics2.get(selected_role.lower(), [])
        display_columns = core_columns + role_columns
        df = df[display_columns]
        
        # Normalize metrics that start with "success_"
        for col in df.columns:
            if col.startswith("success_"):
                df[col] *= 100
        
        # Rename columns for display
        df = df.rename(columns={**parameter_display_names2, **parameter_display_names3})
        
        # Sort by 'Aggregate Weighted Score' if it exists
        if 'Aggregate Weighted Score' in df.columns:
            df = df.sort_values(by="Aggregate Weighted Score", ascending=False)
        
        # Number of Rows to Display
        num_rows = st.slider("Number of Rows to Display", min_value=1, max_value=100, value=5, help="Select the number of rows to display.")
        df = df.head(num_rows).round(2)
        
        st.write(f"## Displaying Top {num_rows} {selected_role}s")
        st.dataframe(df)


# Pizza Chart Demo Page
elif page == "Player Similarity & Percentile Rankings":
    st.image("Afriskaut Logo White.png", width=100)
    #st.title("Player Similarity & Percentile Rankings")

    # Sidebar - Player selection for pizza chart
    player_display_names = []
    seen_names = set()
    
    for index, row in df.iterrows():
        player_name = row['player_name']
        team_id = row['team_id']
        if player_name in seen_names:
            display_name = f"{player_name} (Team ID: {team_id})"
        else:
            display_name = player_name
            seen_names.add(player_name)
        player_display_names.append(display_name)
    
    # Player selection dropdown
    selected_display_name = st.selectbox("Select a Player", player_display_names)
    selected_player_name = selected_display_name.split(" (Team ID:")[0]
    selected_team_id = None
    
    if "Team ID:" in selected_display_name:
        selected_team_id = selected_display_name.split(" (Team ID: ")[1][:-1]
    
    # Retrieve the selected player data
    if selected_team_id:
        player_data = df[(df['player_name'] == selected_player_name) & (df['team_id'] == selected_team_id)]
    else:
        player_data = df[df['player_name'] == selected_player_name].iloc[0:1]
    
    if player_data.empty:
        st.error("Player not found in the dataset.")
    else:
        # Get player role and metrics
        player_roles = player_data['player_role'].values[0] if 'player_role' in player_data.columns else None
        matches_played = player_data['matches_played'].values[0] if 'matches_played' in player_data.columns else None
        team_name = player_data['team_name'].values[0]
    
        # Check and split roles
        if player_roles and isinstance(player_roles, str):
            player_roles_list = player_roles.split(",")
        else:
            player_roles_list = []
    
        primary_role = player_roles_list[0].strip().lower() if player_roles_list else 'goalkeeper'
        metrics = role_metrics.get(primary_role, role_metrics['goalkeeper'])
    
        player_s = player_data[metrics].round()
        player_values = [0 if v is None or np.isnan(v) else v for v in player_s.values.flatten().tolist()]
    
        st.write(f"### {selected_player_name} ({team_name}) - {primary_role.capitalize()}")
        st.write(f'Percentile Rankings vs All {primary_role + "s"}')
        if matches_played:
            st.write(f'Matches Played: {matches_played}')

        # Plotting pizza chart
        parameter_display_names = {
            'percentile_dribbles': "Dribble Success Rate",
            'percentile_key_pass_value': "Key Passes per 90",
            'percentile_shots': "Shot Accuracy",
            'percentile_goals': "Goals per 90",
            'percentile_assists': "Assists per 90",
            'percentile_crosses': "Cross Completion Rate",
            'percentile_success_rate_crosses': "Cross Completion Rate",
            'percentile_ground_duels': "Ground Duel Success Rate",
            'percentile_aerial_duels': "Aerial Duel Success Rate",
            'percentile_pass_value': "Pass Success Rate",
            'percentile_progressive': "Progressive Passes Per 90",
            'percentile_dribble_value': "Dribble Success Rate",
            'percentile_tackle_value': "Tackle Success Rate",
            'percentile_success_rate_passes': "Pass Completion Rate",
            'percentile_success_rate_tackles': "Tackle Success Rate",
            'percentile_rate_int_value': "Interceptions Per 90",
            'percentile_rate_block_value': "Blocks Per 90",
            'percentile_success_rate_ground_duels': "Ground Duel Success Rate",
            'percentile_success_rate_aerial_duels': "Aerial Duel Success Rate",
            'percentile_rate_clearance_value': "Clearances Per 90",
            'percentile_rate_saves': "Saves Per 90",
            'percentile_rate_conceded': "Goals Conceded Per 90",
            'percentile_rate_claims': "Claims Per 90"
        }

        baker = PyPizza(
        params=[parameter_display_names[m] for m in metrics],
        straight_line_color="#ffffff", last_circle_color="#ffffff",
        last_circle_lw=2, other_circle_lw=1, other_circle_ls="-."
    )

    fig, ax = baker.make_pizza(
        player_values, figsize=(8, 8), param_location=110,
        kwargs_slices=dict(facecolor="cornflowerblue", edgecolor="#ffffff", zorder=2, linewidth=1),
        kwargs_params=dict(color="#ffffff", fontsize=12, va="center"),
        kwargs_values=dict(color="#ffffff", fontsize=12,
                           bbox=dict(edgecolor="#ffffff", facecolor="cornflowerblue",
                                     boxstyle="round,pad=0.2", lw=1))
    )
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    st.pyplot(fig, ax)

    # Similar players
    df[metrics] = df[metrics].fillna(0)
    role_filtered_df = df[(df['player_role'].str.contains(primary_role, case=False, na=False)) | df['player_role'].isna()]

    if role_filtered_df.empty:
        st.write(f"No players found with the role '{primary_role}'. Please check the role name and try again.")
    else:
        player_data = role_filtered_df[role_filtered_df['player_name'] == selected_player_name]

        if player_data.empty:
            st.write(f"No data found for '{selected_player_name}' in the '{primary_role}' role.")
        else:
            scaler = StandardScaler()
            scaled_metrics = scaler.fit_transform(role_filtered_df[metrics])

            model = NearestNeighbors(n_neighbors=6, algorithm='auto')
            model.fit(scaled_metrics)

            player_index = role_filtered_df[role_filtered_df['player_name'] == selected_player_name].index[0]
            scaled_index = role_filtered_df.index.get_loc(player_index)

            distances, indices = model.kneighbors([scaled_metrics[scaled_index]])
            similar_players = role_filtered_df.iloc[indices[0]].iloc[1:]  # Exclude the queried player

            if not similar_players.empty:
                st.write("### Similar Players")
                for _, row in similar_players.iterrows():
                    st.write(f" - {row['player_name']} ({row['team_name']})")
            else:
                st.write("No similar players found.")


 
elif page == "Interactive Plots":
    

# Sample data structure for demonstration (replace this with actual data)
# Assuming `player_data` is a DataFrame containing columns like "Player Name", "Team Name", "Position", and various metrics in `role_metrics2`
# Also assuming `role_metrics2` dictionary contains metric names and descriptions

    # Define role_metrics2 - example structure
    
   

    # Metric descriptions dictionary
    metric_descriptions = {
        'rate_saves': 'Percentage of shots saved by the goalkeeper.',
        'rate_conceded': 'Percentage of shots conceded by the goalkeeper.',
        'rate_claims': 'Percentage of aerial claims made by the goalkeeper.',
        'general_metric': 'A general performance score based on multiple metrics.',
        
        'success_rate_passes': 'Percentage of successful passes made.',
        'attempted_passes': 'Number of passes attempted.',
        'success_rate_tackles': 'Percentage of successful tackles made.',
        'attempted_tackles': 'Number of tackles attempted.',
        'rate_int_value': 'Percentage of successful interceptions.',
        'rate_block_value': 'Percentage of successful block attempts.',
        'success_rate_ground_duels': 'Percentage of successful ground duels.',
        'attempted_ground_duels': 'Number of ground duels attempted.',
        'success_rate_aerial_duels': 'Percentage of successful aerial duels.',
        'attempted_aerial_duels': 'Number of aerial duels attempted.',
        'rate_clearance_value': 'Percentage of successful clearances.',
        'success_rate_crosses': 'Percentage of successful crosses.',
        'attempted_crosses': 'Number of crosses attempted.',
        
        'success_rate_dribbles': 'Percentage of successful dribbles.',
        'attempted_dribbles': 'Number of dribbles attempted.',
        'success_rate_shots': 'Percentage of successful shots on target.',
        'attempted_shots': 'Number of shots attempted.',
        'rate_assists': 'Percentage of assists made.',
        'cumulative_assists': 'Total number of assists made.',
        'rate_goals': 'Percentage of goals scored per attempt.',
        'cumulative_is_goal': 'Total number of goals scored.',
        'rate_key_pass_value': 'Percentage of key passes made.',
        'rate_dribbles': 'Percentage of dribbles made compared to total attempts.',
        'rate_progressive': 'Percentage of successful progressive actions.',
    }
    
    
    # Start Streamlit app




# Function to adjust success metrics (multiply by 100)
    # Define dictionaries for plot descriptions and quadrant meanings for all possible metric combinations

# Plot Descriptions
    plot_descriptions = {
    # Goalkeeper
    ('rate_saves', 'rate_conceded'): "This plot compares a player's **Save Rate** with their **Conceded Rate**. A high save rate and low conceded rate is indicative of a strong goalkeeper performance.",
    ('rate_conceded', 'rate_saves'): "This plot compares a player's **Save Rate** with their **Conceded Rate**. A high save rate and low conceded rate is indicative of a strong goalkeeper performance.",
    ('rate_saves', 'rate_claims'): "This plot compares a goalkeeper's **Save Rate** with their **Claims Rate**. A high save rate and claims rate indicates dominance in aerial situations and shot-stopping.",
    ('rate_claims', 'rate_saves'): "This plot compares a goalkeeper's **Save Rate** with their **Claims Rate**. A high save rate and claims rate indicates dominance in aerial situations and shot-stopping.",
    ('rate_conceded', 'rate_claims'): "This compares a goalkeeper's **Conceded Rate** with their **Claims Rate**. A high claims rate and low conceded rate means the goalkeeper is effective both in dealing with crosses and in preventing goals.",
    ('rate_claims', 'rate_conceded'): "This compares a goalkeeper's **Conceded Rate** with their **Claims Rate**. A high claims rate and low conceded rate means the goalkeeper is effective both in dealing with crosses and in preventing goals.",
    
    # Right Back
    ('success_rate_passes', 'success_rate_tackles'): "This plot compares a player's **Pass Success Rate** with their **Tackle Success Rate**. A player with high values in both is effective at both distributing the ball and defending.",
    ('success_rate_passes', 'attempted_passes'): "This plot compares a player's **Pass Success Rate** with their **Passes Attempted**. Players with high pass attempts and success are key playmakers.",
    ('success_rate_tackles', 'attempted_tackles'): "This plot compares a player's **Tackle Success Rate** with their **Tackles Attempted**. Players with high tackle attempts and success are strong defensively.",
    ('rate_int_value', 'rate_block_value'): "This plot compares a defender's **Interceptions Rate** with their **Block Rate**. Players excelling in both are strong at intercepting passes and blocking shots.",
    ('success_rate_ground_duels', 'attempted_ground_duels'): "This plot compares a player's **Ground Duel Success Rate** with their **Ground Duels Attempted**. A high success rate and attempts shows a dominant player in ground duels.",
    ('success_rate_aerial_duels', 'attempted_aerial_duels'): "This plot compares a player's **Aerial Duel Success Rate** with their **Aerial Duels Attempted**. A high success rate and attempts shows dominance in aerial duels.",
    ('rate_clearance_value', 'success_rate_crosses'): "This plot compares a player's **Clearance Rate** with their **Cross Success Rate**. Players excelling in both are likely to be strong defensively and in crossing situations.",
    
    # Left Back
    ('success_rate_passes', 'success_rate_tackles'): "This plot compares a player's **Pass Success Rate** with their **Tackle Success Rate**. A player with high values in both is effective at both distributing the ball and defending.",
    ('success_rate_passes', 'attempted_passes'): "This plot compares a player's **Pass Success Rate** with their **Passes Attempted**. This plot shows key playmakers who distribute the ball effectively.",
    ('success_rate_tackles', 'attempted_tackles'): "This plot compares a player's **Tackle Success Rate** with their **Tackles Attempted**. This plot shows players who are strong defensively.",
    ('rate_int_value', 'rate_block_value'): "This plot compares a defender's **Interceptions Rate** with their **Block Rate**. Players excelling in both are strong at intercepting passes and blocking shots.",
    ('success_rate_ground_duels', 'attempted_ground_duels'): "This plot compares a player's **Ground Duel Success Rate** with their **Ground Duels Attempted**. Shows a dominant player in ground duels.",
    ('success_rate_aerial_duels', 'attempted_aerial_duels'): "This plot compares a player's **Aerial Duel Success Rate** with their **Aerial Duels Attempted**. Shows a dominant player in aerial duels.",
    ('rate_clearance_value', 'success_rate_crosses'): "This plot compares a player's **Clearance Rate** with their **Cross Success Rate**. Dominant in defending and crossing situations.",
    
    # Center Back
    ('success_rate_passes', 'success_rate_tackles'): "This plot compares a player's **Pass Success Rate** with their **Tackle Success Rate**. A player with high values in both is effective at both distributing the ball and defending.",
    ('success_rate_passes', 'attempted_passes'): "This plot compares a player's **Pass Success Rate** with their **Passes Attempted**. Shows accurate passers of the ball.",
    ('success_rate_tackles', 'attempted_tackles'): "This plot compares a player's **Tackle Success Rate** with their **Tackles Attempted**. Shows players players attempting high volumes of tackles and succeeding at them.",
    ('rate_int_value', 'rate_block_value'): "This plot compares a defender's **Interceptions Rate** with their **Block Rate**. Well rounded defenders are ones who excel in both areas.",
    ('success_rate_ground_duels', 'attempted_ground_duels'): "This plot compares a player's **Ground Duel Success Rate** with their **Ground Duels Attempted**. Dominates in defensive duels.",
    ('success_rate_aerial_duels', 'attempted_aerial_duels'): "This plot compares a player's **Aerial Duel Success Rate** with their **Aerial Duels Attempted**. Effective in the air.",
    ('rate_clearance_value', 'success_rate_crosses'): "This plot compares a player's **Clearance Rate** with their **Cross Success Rate**. Strong at both defending and delivering crosses.",
    
    # Defensive Midfielder
    ('success_rate_passes', 'success_rate_dribbles'): "This plot compares a player's **Pass Success Rate** with their **Dribble Success Rate**. Effective at both distributing the ball and evading opposition.",
    ('rate_assist', 'rate_key_pass_value'): "This plot compares a player's **Assist Rate** with their **Key Pass Success Rate**. A player contributing to offensive play with passing and assists.",
    ('success_rate_ground_duels', 'attempted_ground_duels'): "This plot compares a player's **Ground Duel Success Rate** with their **Ground Duels Attempted**.",
    ('success_rate_aerial_duels', 'attempted_aerial_duels'): "This plot compares a player's **Aerial Duel Success Rate** with their **Aerial Duels Attempted**.",
    
    # Central Midfielder
    ('success_rate_passes', 'success_rate_dribbles'): "This plot compares a player's **Pass Success Rate** with their **Dribble Success Rate**. Effective both in distribution and evading opposition.",
    ('rate_assist', 'rate_key_pass_value'): "This plot compares a player's **Assist Rate** with their **Key Pass Success Rate**. This plot shows creative midfielders with high key passes per 90 and assists.",
    ('success_rate_tackles', 'attempted_tackles'): "This plot compares a player's **Tackle Success Rate** with their **Tackles Attempted**.",
    ('success_rate_ground_duels', 'attempted_ground_duels'): "This plot compares a player's **Ground Duel Success Rate** with their **Ground Duels Attempted**..",
    ('success_rate_aerial_duels', 'attempted_aerial_duels'): "This plot compares a player's **Aerial Duel Success Rate** with their **Aerial Duels Attempted**.",
    
    # Attacking Midfielder
    ('success_rate_passes', 'rate_assists'): "This plot compares a player's **Pass Success Rate** with their **Assist Rate**. Shows highly creative midfielders excelling in passing and setting up goals.",
    ('success_rate_dribbles', 'rate_assists'): "This plot compares a player's **Dribble Success Rate** with their **Assist Rate**. Shows player who beats defenders and creates goal-scoring opportunities.",
    ('success_rate_tackles', 'attempted_tackles'): "This plot compares a player's **Tackle Success Rate** with their **Tackles Attempted**.",
    ('rate_key_pass_value', 'success_rate_crosses'): "This plot compares a player's **Key Pass Success Rate** with their **Cross Success Rate**. Shows players good at creating goal-scoring chances through both passes and crosses.",
    
    # Right Winger
    ('success_rate_dribbles', 'rate_assists'): "This plot compares a player's **Dribble Success Rate** with their **Assist Rate**. Shows creative wingers who excels at dribbling and creating goals.",
    ('success_rate_shots', 'rate_goals'): "This plot compares a player's **Shot Success Rate** with their **Goal Success Rate**. Shows clinical attackers with high shot conversion and goal-scoring abilities.",
    ('success_rate_crosses', 'attempted_crosses'): "This plot compares a player's **Cross Success Rate** with their **Cross Attempts**. Shows key players who attempt a high number of crosses, and convert.",
    
    # Left Winger
    ('success_rate_dribbles', 'rate_assists'): "This plot compares a player's **Dribble Success Rate** with their **Assist Rate**. Shows creative wingers who excel at dribbling and creating goals.",
    ('success_rate_shots', 'rate_goals'): "This plot compares a player's **Shot Success Rate** with their **Goal Success Rate**. Shows clinical attackers with high shot conversion and goal-scoring abilities.",
    ('success_rate_crosses', 'attempted_crosses'): "This plot compares a player's **Cross Success Rate** with their **Cross Attempts**. Shows key players who attempt a high number of crosses, and convert.",
    
    # Supporting Striker
    ('success_rate_shots', 'rate_goals'): "This plot compares a player's **Shot Success Rate** with their **Goal Success Rate**. Shows efficient and clinical attackers.",
    ('rate_assists', 'success_rate_crosses'): "This plot compares a player's **Assist Rate** with their **Cross Success Rate**. Shows key players who attempt a high number of crosses, and convert efficiently.",
    
    # Center Forward
    ('success_rate_shots', 'rate_goals'): "This plot compares a player's **Shot Success Rate** with their **Goal Success Rate**. Shows highly efficient forwards with excellent goal-scoring ability.",
    ('rate_assists', 'success_rate_crosses'): "This plot compares a player's **Assist Rate** with their **Cross Success Rate**.",
    ('success_rate_dribbles', 'attempted_dribbles'): "This plot compares a player's **Dribble Success Rate** with their **Dribbles Attempted**. A high success rate with a high number of attempted dribbles indicates a player who not only tries to dribble frequently but also does so effectively. Conversely, a player with a low success rate and a high number of attempts might struggle to beat defenders, while a player with few attempts and a high success rate is selective and efficient in their dribbles.",
        #
    ('success_rate_shots', 'attempted_shots'): "This plot compares a player's **Shot Accuracy** with their **Shots Attempted**.",
    
    ('cumulative_is_goal', 'attempted_shots'): "This plot compares a player's **Goals Scored** with their **Shots Attempted**. A clinical striker will have a high number of goals per shot attempted.",
    ('rate_assists', 'cumulative_assists'): "This plot compares a player's **Assist Rate** with their **Assists Recorded**. Shows players who regularly provides assists.",
    ('cumulative_is_goal', 'cumulative_assists'): "This plot compares a player's **Goals Scored** with their **Assists**. It highlights a player's overall offensive contribution, showing how many goals they score versus how many they create for others. A player with a high number of both goals and assists is typically a key offensive force, capable of both finishing opportunities and setting up teammates for success.",
    ('rate_goals', 'rate_assists'): "This plot compares a player's **Goals Per 90** with their **Assists Per 90**. It offers insight into a player's efficiency in both scoring and assisting. A player with a high goal rate and assist rate is typically a highly effective attacking player, contributing both as a scorer and a creator. Conversely, a player with a low rate in either or both indicates areas that may need improvement for overall offensive effectiveness.",
    ('success_rate_ground_duels', 'success_rate_aerial_duels'): "This plot compares a player's Ground Duel Success Rate with their Aerial Duel Success Rate. A player with high success in both ground and aerial duels is versatile and effective in various types of physical contests, excelling in both on-the-ground and aerial situations. High ground duel success with lower aerial success might suggest a player who is dominant on the ground but less effective in the air, whereas the reverse indicates a strong aerial presence but potential ground duel limitations. This plot is useful for assessing players' all-around ability to win physical duels across different scenarios.",
    ('success_rate_dribbles', 'rate_assist'): "This plot compares a player's **Dribble Success Rate** with their **Assist Rate**. A player with high dribble success and a high assist rate is highly effective at breaking down defenses and creating goal-scoring opportunities for teammates. High dribble success with lower assists may indicate a player skilled at advancing play individually, while lower dribble success with higher assists could suggest a player who distributes effectively after receiving the ball."
        


        
}

    
    # Quadrant Interpretations
    quadrant_meanings = {
    # Goalkeeper
    ('rate_saves', 'rate_conceded'): {
        "Top-Right": "High in both saves and low conceded rate. A top-performing goalkeeper.",
        "Top-Left": "Low saves but low conceded rate. Could indicate a goalkeeper with limited work but still manages to keep the score down.",
        "Bottom-Right": "High saves but high conceded rate. Indicates a goalkeeper who faces many shots but struggles to stop them.",
        "Bottom-Left": "Low in both areas. The goalkeeper is ineffective at stopping shots and letting in too many goals."
    },
    ('rate_saves', 'rate_claims'): {
        "Top-Right": "High saves and high claims rate. Dominant goalkeeper in aerial and shot-stopping situations.",
        "Top-Left": "High saves but low claims. Good at saving shots but struggles with crosses or aerial duels.",
        "Bottom-Right": "Low saves but high claims. May be a goalkeeper who is good at dealing with crosses but not very good at stopping shots.",
        "Bottom-Left": "Low in both. Needs improvement both in terms of shot-stopping and handling aerial threats."
    },
    
    # Right Back
    ('success_rate_passes', 'success_rate_tackles'): {
        "Top-Right": "High in both passing and tackling. A well-rounded player who contributes offensively and defensively.",
        "Top-Left": "High in tackles but low in passing success. Likely a player with a defensive role who is not as effective in possession.",
        "Bottom-Right": "High in passing but low in tackling. Likely an offensive player who contributes more in attacking moves and less in defensive duels.",
        "Bottom-Left": "Low in both. Needs improvement in both passing and tackling."
    },
    ('success_rate_passes', 'attempted_passes'): {
        "Top-Right": "High pass success with a high volume of passes. A key playmaker who distributes the ball effectively.",
        "Top-Left": "Low success but high attempts. A player who tries to pass often but struggles with accuracy.",
        "Bottom-Right": "High success but low attempts. A player who is efficient with fewer passes, indicating selective passing.",
        "Bottom-Left": "Low success and low attempts. A player who might not contribute much in terms of passing."
    },
    
    # Left Back
    ('success_rate_passes', 'success_rate_tackles'): {
        "Top-Right": "High in both passing and tackling. A player who is effective in possession and defensively solid.",
        "Top-Left": "High tackles but low pass success. A strong defender, but not as effective in possession.",
        "Bottom-Right": "High pass success but low tackles. Likely more offensive, contributing with passes but not as strong defensively.",
        "Bottom-Left": "Low in both. Needs improvement both in possession and in defensive work."
    },
    ('success_rate_passes', 'attempted_passes'): {
        "Top-Right": "High pass success with a high volume of passes. A key contributor to possession and offensive build-up.",
        "Top-Left": "Low pass success but high attempts. A player who tries to distribute often but lacks accuracy.",
        "Bottom-Right": "High pass success but low attempts. A selective passer who contributes effectively in key moments.",
        "Bottom-Left": "Low success and low attempts. A player who does not play a significant role in passing."
    },
    ('success_rate_tackles', 'attempted_tackles'): {
        "Top-Right": "High tackle success and high tackles attempted. A highly defensive player who wins many duels.",
        "Top-Left": "High tackle success but low attempts. Effective in defensive situations but does not need to make many tackles.",
        "Bottom-Right": "Low tackle success but high attempts. A player who tries to defend but does not win many duels.",
        "Bottom-Left": "Low tackle success and low attempts. A player who struggles defensively."
    },
    ('rate_int_value', 'rate_block_value'): {
        "Top-Right": "High interceptions and blocks. A player who is excellent at cutting out passes and stopping shots.",
        "Top-Left": "High interceptions but low blocks. Strong at reading the game and intercepting passes, but not as effective in shot-blocking.",
        "Bottom-Right": "Low interceptions but high blocks. A player who struggles to intercept passes but is good at blocking shots.",
        "Bottom-Left": "Low in both. Needs improvement both in interceptions and shot-blocking."
    },
    ('success_rate_ground_duels', 'attempted_ground_duels'): {
        "Top-Right": "High success and high attempts in ground duels. A player who dominates in defensive duels.",
        "Top-Left": "High success but low attempts. A player who is selective and effective in ground duels.",
        "Bottom-Right": "Low success but high attempts. A player who tries to engage in ground duels but doesn't win many.",
        "Bottom-Left": "Low in both. Struggles in ground duels both in terms of attempts and success."
    },
    ('success_rate_aerial_duels', 'attempted_aerial_duels'): {
        "Top-Right": "High success and high attempts in aerial duels. A dominant player in the air.",
        "Top-Left": "High success but low attempts. A player who is highly effective in the air but does not challenge aerially often.",
        "Bottom-Right": "Low success but high attempts. A player who frequently contests aerial duels but is not successful.",
        "Bottom-Left": "Low in both. Needs to improve both in aerial duels and the frequency of aerial challenges."
    },
    ('rate_clearance_value', 'success_rate_crosses'): {
        "Top-Right": "High clearances and accurate crosses. A player who is both defensively solid and effective in delivering crosses.",
        "Top-Left": "High clearances but inaccurate crosses. Strong defensively but not as effective in crossing situations.",
        "Bottom-Right": "Low clearances but accurate crosses. A player who is more offensive, contributing with crosses but not as strong defensively.",
        "Bottom-Left": "Low in both. Needs improvement both in clearing the ball and crossing."
    },
    
    # Center Back
    ('success_rate_passes', 'success_rate_tackles'): {
        "Top-Right": "High pass success and high tackle success. A player who is a defensive stalwart and capable of distributing the ball well.",
        "Top-Left": "High tackles but low pass success. A defensive specialist who struggles to distribute the ball accurately.",
        "Bottom-Right": "High pass success but low tackle success. A center back who is good at passing but lacks effectiveness in defense.",
        "Bottom-Left": "Low in both. A player who needs improvement in both passing and defensive contributions."
    },
    ('success_rate_passes', 'attempted_passes'): {
        "Top-Right": "High pass success with many attempted passes. A key distributor from the back.",
        "Top-Left": "Low pass success but many attempts. A player who tries to pass a lot but struggles with accuracy.",
        "Bottom-Right": "High success but low attempts. Selective and efficient in passing, but not involved in building possession frequently.",
        "Bottom-Left": "Low success and low attempts. A player with minimal contribution to passing."
    },
    ('success_rate_tackles', 'attempted_tackles'): {
        "Top-Right": "High success and high attempts in tackles. A player who consistently wins defensive duels.",
        "Top-Left": "High success but low attempts. Strong in tackles but does not face many situations requiring intervention.",
        "Bottom-Right": "Low success but high attempts. A player who tries to tackle often but does not win many duels.",
        "Bottom-Left": "Low in both. Struggles defensively both in terms of success and frequency of tackles."
    },
    ('rate_int_value', 'rate_block_value'): {
        "Top-Right": "High interceptions and high blocks. A center back who is highly effective in breaking up play and defending shots.",
        "Top-Left": "High interceptions but low blocks. Effective at intercepting passes but not as dominant in shot-blocking.",
        "Bottom-Right": "Low interceptions but high blocks. A player who struggles to intercept but excels at preventing shots.",
        "Bottom-Left": "Low in both. Needs improvement in both interceptions and blocking shots."
    },
    ('success_rate_ground_duels', 'attempted_ground_duels'): {
        "Top-Right": "High success and high attempts. A central defender who is dominant in ground duels.",
        "Top-Left": "High success but low attempts. A player who is highly effective but does not engage often in ground duels.",
        "Bottom-Right": "Low success but high attempts. A player who engages in many ground duels but does not win many.",
        "Bottom-Left": "Low in both. Needs to improve both ground duel success and frequency."
    },
    ('success_rate_aerial_duels', 'attempted_aerial_duels'): {
        "Top-Right": "High success and high attempts. Dominates in aerial duels.",
        "Top-Left": "High success but low attempts. Strong in aerial duels but does not contest many.",
        "Bottom-Right": "Low success but high attempts. Frequently contests aerial duels but is not effective.",
        "Bottom-Left": "Low in both. Needs improvement both in terms of aerial duels contested and won."
    },
    ('rate_clearance_value', 'success_rate_crosses'): {
        "Top-Right": "High clearance and high crossing success. A player who excels both in defending and creating crosses.",
        "Top-Left": "High clearance but low crossing success. Dominates defensively but struggles with crossing.",
        "Bottom-Right": "Low clearance but high crossing success. More offensive in nature but not effective defensively.",
        "Bottom-Left": "Low in both. Needs improvement in both clearing the ball and delivering accurate crosses."
    },
    
    # Defensive Midfielder
    ('success_rate_passes', 'success_rate_dribbles'): {
        "Top-Right": "High pass success and high dribble success. A highly effective ball carrier and distributor.",
        "Top-Left": "High passes but low dribbles. Strong at distribution but not a key dribbler.",
        "Bottom-Right": "Low passes but high dribbles. A player who is more involved in dribbling than passing.",
        "Bottom-Left": "Low in both. Needs to improve in both distribution and dribbling."
    },
    ('success_rate_tackles', 'attempted_tackles'): {
        "Top-Right": "High tackle success and high tackles attempted. A strong defensive midfielder who dominates duels.",
        "Top-Left": "High tackle success but low attempts. Selectively effective defensively.",
        "Bottom-Right": "Low tackle success but high attempts. Tries to tackle but does not win many duels.",
        "Bottom-Left": "Low in both. Needs improvement in defensive work."
    },
    ('rate_assist', 'rate_key_pass_value'): {
        "Top-Right": "High assists and key passes. A creative player who excels in creating opportunities.",
        "Top-Left": "High assists but low key passes. Strong in creating goals but not as effective in playmaking.",
        "Bottom-Right": "Low assists but high key passes. Creates opportunities but doesn't finish them.",
        "Bottom-Left": "Low in both. Needs improvement in both playmaking and assisting."
    },
    ('success_rate_ground_duels', 'attempted_ground_duels'): {
        "Top-Right": "Strong defensive presence in both ground duels won and frequency.",
        "Top-Left": "Strong ground duel success but not as frequent in attempting duels.",
        "Bottom-Right": "High duel attempts but not as successful at winning duels."
    },
    # Central Midfielder
    ('success_rate_passes', 'success_rate_dribbles'): {
        "Top-Right": "High pass and dribble success. A well-rounded midfielder who can distribute and carry the ball effectively.",
        "Top-Left": "High passes but low dribbles. Primarily a distributor, with less emphasis on carrying the ball.",
        "Bottom-Right": "Low passes but high dribbles. A more dynamic player who carries the ball but struggles with passing.",
        "Bottom-Left": "Low in both. Needs to improve both in passing and dribbling."
    },
    ('rate_assist', 'rate_key_pass_value'): {
        "Top-Right": "High assists and high key passes. A creative playmaker who regularly sets up goal-scoring opportunities.",
        "Top-Left": "High assists but low key passes. Effective at creating assists but not involved in key plays or build-up.",
        "Bottom-Right": "Low assists but high key passes. A player who creates chances but does not often finish them.",
        "Bottom-Left": "Low in both. Needs improvement both in creating assists and key passes."
    },
    ('success_rate_ground_duels', 'attempted_ground_duels'): {
        "Top-Right": "High success and high attempts in ground duels. A player who competes effectively in defensive duels.",
        "Top-Left": "High success but low attempts. A player who wins ground duels when needed but does not frequently engage.",
        "Bottom-Right": "Low success but high attempts. A player who tries hard in ground duels but is not consistently successful.",
        "Bottom-Left": "Low in both. Needs improvement in both duel frequency and success."
    },
    ('success_rate_aerial_duels', 'attempted_aerial_duels'): {
        "Top-Right": "High success and high attempts in aerial duels. A dominant presence in the air.",
        "Top-Left": "High success but low attempts. Strong in aerial duels but not frequently challenged in the air.",
        "Bottom-Right": "Low success but high attempts. Frequently contests aerial duels but does not win many.",
        "Bottom-Left": "Low in both. Needs improvement in both aerial duels contested and success."
    },
    
    # Attacking Midfielder
    ('success_rate_dribbles', 'rate_assists'): {
        "Top-Right": "High dribble success and high assist rate. A creative and dangerous player who can both beat defenders and create goal-scoring opportunities.",
        "Top-Left": "High dribble success but low assists. A player who can beat defenders but struggles to set up goals.",
        "Bottom-Right": "Low dribble success but high assists. A player who is not great at dribbling but excels in creating opportunities for teammates.",
        "Bottom-Left": "Low in both. Needs improvement both in dribbling and creating assists."
    },
    ('success_rate_dribbles', 'success_rate_shots'): {
        "Top-Right": "High dribble success and high shot success. A dynamic attacking player who can beat defenders and finish chances effectively.",
        "Top-Left": "High dribble success but low shot success. A player who can beat defenders but struggles to convert chances into goals.",
        "Bottom-Right": "Low dribble success but high shot success. A player who may not dribble well but is clinical in front of goal.",
        "Bottom-Left": "Low in both. Needs to work on both dribbling and finishing."
    },
    ('rate_key_pass_value', 'rate_assists'): {
        "Top-Right": "High key passes and high assists. A highly creative playmaker who regularly provides goal-scoring opportunities.",
        "Top-Left": "High key passes but low assists. A player who creates chances but struggles to register assists.",
        "Bottom-Right": "Low key passes but high assists. A player who might not create many opportunities but is effective when they do.",
        "Bottom-Left": "Low in both. Needs improvement in both creating key passes and assisting."
    },
    
    # Right Winger
    ('success_rate_dribbles', 'attempted_dribbles'): {
        "Top-Right": "High dribble success and high dribble attempts. A player who is a consistent threat in one-on-one situations.",
        "Top-Left": "High dribble success but low attempts. A player who is very effective when dribbling but doesn't do it often.",
        "Bottom-Right": "Low dribble success but high attempts. A player who tries to dribble often but struggles to beat defenders.",
        "Bottom-Left": "Low in both. Needs to work on both dribbling success and frequency."
    },
    ('success_rate_shots', 'attempted_shots'): {
        "Top-Right": "High shot success and high attempts. A player who is both clinical in front of goal and regularly gets chances.",
        "Top-Left": "High shot success but low attempts. A player who is selective and accurate when shooting but doesn't get many chances.",
        "Bottom-Right": "Low shot success but high attempts. A player who takes many shots but struggles with conversion.",
        "Bottom-Left": "Low in both. Needs to improve both shot accuracy and frequency."
    },
    ('rate_assists', 'rate_goals'): {
        "Top-Right": "High assists and high goals. A player who is both a goal scorer and playmaker, contributing significantly in the attack.",
        "Top-Left": "High assists but low goals. A player who is great at creating opportunities but not as effective at scoring.",
        "Bottom-Right": "Low assists but high goals. A goal-scorer who might not contribute much in the build-up play.",
        "Bottom-Left": "Low in both. Needs improvement in both goal-scoring and playmaking."
    },
    
    # Left Winger
    ('success_rate_dribbles', 'attempted_dribbles'): {
        "Top-Right": "High dribble success and high attempts. A player who is constantly looking to beat defenders.",
        "Top-Left": "High dribble success but low attempts. Effective dribbler but not frequently involved in dribbling situations.",
        "Bottom-Right": "Low dribble success but high attempts. Tries to dribble often but is not successful.",
        "Bottom-Left": "Low in both. Needs to improve both dribbling success and frequency."
    },
    ('success_rate_shots', 'attempted_shots'): {
        "Top-Right": "High shot success and high attempts. A clinical forward who regularly gets chances and finishes well.",
        "Top-Left": "High shot success but low attempts. A player who takes fewer shots but is very efficient.",
        "Bottom-Right": "Low shot success but high attempts. A player who shoots often but struggles with accuracy.",
        "Bottom-Left": "Low in both. Needs improvement in both shot accuracy and frequency."
    },
    ('rate_assists', 'rate_goals'): {
        "Top-Right": "High goals and high assists. A well-rounded attacking player contributing to both scoring and creating chances.",
        "Top-Left": "High assists but low goals. A player who creates many opportunities but doesn't score often.",
        "Bottom-Right": "Low assists but high goals. A player who is a prolific goal-scorer but doesn't contribute as much to build-up play.",
        "Bottom-Left": "Low in both. Needs improvement both in terms of creating and scoring goals."
    },
    
    # Supporting Striker
    ('success_rate_dribbles', 'attempted_dribbles'): {
        "Top-Right": "High dribble success and high attempts. A skillful player who regularly takes on defenders.",
        "Top-Left": "High dribble success but low attempts. A player who is effective in dribbles but is selective about when to take them.",
        "Bottom-Right": "Low dribble success but high attempts. A player who attempts to dribble frequently but isn't always successful.",
        "Bottom-Left": "Low in both. Needs improvement in both dribbling success and frequency."
    },
    ('success_rate_shots', 'attempted_shots'): {
        "Top-Right": "High shot success and high attempts. A clinical forward who consistently takes and converts chances.",
        "Top-Left": "High shot success but low attempts. A player who is efficient with fewer chances.",
        "Bottom-Right": "Low shot success but high attempts. A player who shoots often but struggles to score.",
        "Bottom-Left": "Low in both. Needs improvement both in shooting and shot attempts."
    },
    ('rate_assists', 'rate_goals'): {
        "Top-Right": "High assists and high goals. A well-rounded attacker who both scores and creates for teammates.",
        "Top-Left": "High assists but low goals. A player who excels in creating chances but not as much in scoring.",
        "Bottom-Right": "Low assists but high goals. A player who scores frequently but doesn't contribute as much to the team's playmaking.",
        "Bottom-Left": "Low in both. Needs improvement both in scoring and assisting."
    },
    # Center Forward
    ('success_rate_dribbles', 'rate_assists'): {
        "Top-Right": "High dribble success and high assist rate. A dynamic forward who can both beat defenders and provide key passes.",
        "Top-Left": "High dribble success but low assist rate. A player who is effective at dribbling but doesn't create many assists.",
        "Bottom-Right": "Low dribble success but high assist rate. A player who struggles to beat defenders but is good at setting up teammates.",
        "Bottom-Left": "Low in both. Needs improvement in dribbling and assisting."
    },
    ('success_rate_shots', 'rate_goals'): {
        "Top-Right": "High shot success and high goals. A clinical striker who frequently finds the back of the net and converts chances efficiently.",
        "Top-Left": "High shot success but low goals. A player who is very accurate but doesn't get many opportunities to score.",
        "Bottom-Right": "Low shot success but high goals. A forward who takes many shots but has a strong conversion rate.",
        "Bottom-Left": "Low in both. Needs improvement in both shot conversion and goal-scoring frequency."
    },
    ('rate_assists', 'rate_goals'): {
        "Top-Right": "High goals and high assists. A prolific attacker who contributes significantly both as a goal-scorer and a playmaker.",
        "Top-Left": "High assists but low goals. A creative forward who excels in setting up teammates but doesn't score often.",
        "Bottom-Right": "Low assists but high goals. A goal-scoring forward who isn't heavily involved in creating opportunities for others.",
        "Bottom-Left": "Low in both. Needs improvement in both scoring and creating chances."
    },
    ('success_rate_passes', 'success_rate_tackles'): {
        "Top-Right": "High pass success and high tackle success. A well-rounded player who contributes both offensively and defensively.",
        "Top-Left": "High pass success but low tackle success. Primarily offensive, but not as strong defensively.",
        "Bottom-Right": "Low pass success but high tackle success. A player who excels defensively but struggles with distribution.",
        "Bottom-Left": "Low in both. Needs to improve both in passing and tackling."
    },

    # For Other Metrics (General Metrics and Position Combinations)
    ('rate_progressive', 'success_rate_aerial_duels'): {
        "Top-Right": "High in progressive play and aerial duels. A player who moves the ball forward and also wins aerial challenges.",
        "Top-Left": "High progressive play but low aerial duels success. A player who advances the ball but struggles with aerial duels.",
        "Bottom-Right": "Low in progressive play but high in aerial duels. A player who may not progress the ball much but excels in aerial challenges.",
        "Bottom-Left": "Low in both. Needs improvement in both advancing the ball and winning aerial duels."
    },
    ('attempted_crosses', 'rate_block_value'): {
        "Top-Right": "High cross attempts and high block value. A player who crosses frequently and defends well by blocking shots.",
        "Top-Left": "High cross attempts but low block value. A player who gets in crosses often but isn't as effective defensively.",
        "Bottom-Right": "Low cross attempts but high block value. A player who doesn't cross much but is strong defensively.",
        "Bottom-Left": "Low in both. Needs improvement in both crossing and defending."
    },
    ('success_rate_passes', 'rate_int_value'): {
        "Top-Right": "High pass success and high interception value. A player who distributes the ball well and breaks up opposition attacks effectively.",
        "Top-Left": "High pass success but low interception value. A player who is good in possession but not as involved in breaking up play.",
        "Bottom-Right": "Low pass success but high interception value. A player who might not pass well but has a strong defensive presence.",
        "Bottom-Left": "Low in both. Needs improvement in both passing and interceptions."
    },
    ('success_rate_tackles', 'rate_clearance_value'): {
        "Top-Right": "High tackle success and high clearance value. A player who excels in both defensive challenges and clearing the ball from danger.",
        "Top-Left": "High tackle success but low clearance value. Strong in tackles but not as effective in clearing the ball.",
        "Bottom-Right": "Low tackle success but high clearance value. A player who might struggle with tackles but clears the ball well.",
        "Bottom-Left": "Low in both. Needs improvement in both tackling and clearing the ball."
    },
}

    
    # Function to display the plot description and quadrant meaning dynamically
    def display_plot_info(selected_metrics):
        st.subheader("Plot Description:")
        # Create both possible orderings of the selected metrics
        plot_key = tuple(selected_metrics)
        plot_key_reversed = tuple(reversed(selected_metrics))
        
        # Check if either ordering exists in plot_descriptions
        if plot_key in plot_descriptions:
            st.write(plot_descriptions[plot_key])
        elif plot_key_reversed in plot_descriptions:
            st.write(plot_descriptions[plot_key_reversed])
        else:
            st.write("No description available for this metric combination.")

        
       # st.subheader("Quadrant Interpretation:")
        #if plot_key in quadrant_meanings:
         #   for quadrant, interpretation in quadrant_meanings[plot_key].items():
          #      st.write(f"- **{quadrant}**: {interpretation}")
    
    
    
    # Assuming df, role_metrics2, and parameter_display_names2 are already defined
    # Function to adjust metrics
    def adjust_success_metrics(df, metric_column):
        if metric_column.startswith('success_'):
            df[metric_column] = df[metric_column] * 100
        return df
    
    # Assuming df is the DataFrame and other relevant structures are defined
    def scatter_plot_page(df):
        st.title("Interactive Scatter Plot for Player Metrics")
        st.write("This page allows you to compare player metrics visually. Select a position and two metrics to plot against each other, and use the slider to control the number of players displayed.")
        
        # Assign 'goalkeeper' role to players with NaN player roles
        df['player_role'] = df['player_role'].fillna('goalkeeper')
        
        # Handle multiple roles
        df['player_roles'] = df['player_role'].str.split(',').apply(lambda x: [role.strip().lower() for role in x])
        
        # Display dropdown for position selection
        roles_list = set(role for roles in df['player_roles'] for role in roles)
        selected_position = st.selectbox("Select Role", options=list(roles_list))
        
        # Filter data for selected position
        filtered_data = df[df['player_roles'].apply(lambda x: selected_position in x)]
        
        # Metric selection
        metric1_display_names = {parameter_display_names2[m]: m for m in role_metrics2[selected_position]}
        metric1 = st.selectbox("Select Metric for X-axis", options=list(metric1_display_names.keys()))
        
        metric2_display_names = {parameter_display_names2[m]: m for m in role_metrics2[selected_position] if m != metric1_display_names[metric1]}
        metric2 = st.selectbox("Select Metric for Y-axis", options=list(metric2_display_names.keys()))
        
        # Display plot descriptions and quadrant interpretation
        display_plot_info([metric1_display_names[metric1], metric2_display_names[metric2]])
        
        # Adjust metrics with the success_ prefix (multiply by 100)
        filtered_data = adjust_success_metrics(filtered_data, metric1_display_names[metric1])
        filtered_data = adjust_success_metrics(filtered_data, metric2_display_names[metric2])
        
        # Slider for number of players
        num_players = st.slider("Select the number of players to display", min_value=20, max_value=500, value=100)
        data_for_plot = filtered_data.head(num_players)
        
        # Create scatter plot
        fig = px.scatter(
            data_for_plot,
            x=metric1_display_names[metric1],
            y=metric2_display_names[metric2],
            hover_name="player_name",
            hover_data={"team_name": True,  "matches_played": True},
            text=None
        )
        
        fig.update_traces(marker=dict(size=10), textposition='top center', hoverinfo='text+name')
        fig.update_layout(
            title=f"{parameter_display_names2[metric1_display_names[metric1]]} vs {parameter_display_names2[metric2_display_names[metric2]]}",
            xaxis_title=parameter_display_names2[metric1_display_names[metric1]],
            yaxis_title=parameter_display_names2[metric2_display_names[metric2]]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Assuming df is your DataFrame and role_metrics2 is already defined, run the Streamlit app
    if __name__ == "__main__":
        scatter_plot_page(df)
