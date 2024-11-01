import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from mplsoccer import PyPizza
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset (update the path if necessary)
df = pd.read_csv("general.csv")  # Replace with the actual path to your data

# Define Role-Based Metrics
role_metrics = {
    'center forward': ['percentile_dribbles', 'percentile_key_pass_value', 'percentile_shots', 'percentile_goals', 
                       'percentile_assists', 'percentile_crosses', 'percentile_ground_duels', 'percentile_aerial_duels'],
    'supporting striker': ['percentile_dribbles', 'percentile_key_pass_value', 'percentile_shots', 'percentile_goals', 
                           'percentile_assists', 'percentile_crosses', 'percentile_ground_duels', 'percentile_aerial_duels'],
    'left winger': ['percentile_dribbles', 'percentile_key_pass_value', 'percentile_shots', 'percentile_goals', 
                    'percentile_assists', 'percentile_crosses', 'percentile_ground_duels', 'percentile_aerial_duels'], 
    'right winger': ['percentile_dribbles', 'percentile_key_pass_value', 'percentile_shots', 'percentile_goals', 
                     'percentile_assists', 'percentile_crosses', 'percentile_ground_duels', 'percentile_aerial_duels'],
    'attacking midfielder': ['percentile_pass_value', 'percentile_progressive', 'percentile_key_pass_value', 
                             'percentile_dribble_value', 'percentile_ground_duels', 'percentile_assists', 'percentile_goals'],
    'central midfielder': ['percentile_pass_value', 'percentile_progressive', 'percentile_key_pass_value', 
                           'percentile_dribble_value', 'percentile_tackle_value', 'percentile_ground_duels', 
                           'percentile_aerial_duels', 'percentile_assists'],
    'defensive midfielder': ['percentile_pass_value', 'percentile_progressive', 'percentile_key_pass_value', 
                             'percentile_dribble_value', 'percentile_tackle_value', 'percentile_ground_duels', 
                             'percentile_aerial_duels'],
    'center back': ['percentile_success_rate_passes', 'percentile_success_rate_tackles', 'percentile_rate_int_value', 
                    'percentile_rate_block_value', 'percentile_success_rate_ground_duels', 'percentile_success_rate_aerial_duels',
                    'percentile_rate_clearance_value'],
    'left back':  ['percentile_success_rate_passes', 'percentile_success_rate_tackles', 'percentile_rate_int_value',
                   'percentile_rate_block_value', 'percentile_success_rate_ground_duels', 'percentile_success_rate_aerial_duels',
                   'percentile_rate_clearance_value', 'percentile_success_rate_crosses'],
    'right back': ['percentile_success_rate_passes', 'percentile_success_rate_tackles', 'percentile_rate_int_value',
                   'percentile_rate_block_value', 'percentile_success_rate_ground_duels', 'percentile_success_rate_aerial_duels',
                   'percentile_rate_clearance_value', 'percentile_success_rate_crosses'],
    'goalkeeper': ['percentile_rate_saves', 'percentile_rate_conceded', 'percentile_rate_claims']
}

# Define a dictionary for display names on the pizza chart
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

# Sidebar - Player selection
st.sidebar.title("Player Search and Filter")
player_name = st.sidebar.selectbox("Select a Player", df['player_name'].unique())

# Get player data
player_data = df[df['player_name'] == player_name]

# Check if player_data is not empty
if player_data.empty:
    st.error("Player not found in the dataset.")
else:
    # Get player role and metrics
    player_roles = player_data['player_role'].values[0] if 'player_role' in player_data.columns else None
    matches_played = player_data['matches_played'].values[0] if 'matches_played' in player_data.columns else None
    team_name = player_data['team_name'].values[0]

    if player_roles and isinstance(player_roles, str):
        player_roles_list = player_roles.split(",")
    else:
        player_roles_list = []
    
    # Use the first role for metrics selection if available
    primary_role = player_roles_list[0].strip().lower() if player_roles_list else 'goalkeeper'
    if primary_role in role_metrics:
        metrics = role_metrics[primary_role]
    else:
        metrics = role_metrics['goalkeeper']
        primary_role = 'goalkeeper'

    player_s = player_data[metrics].round()
    player_values = [0 if v is None or np.isnan(v) else v for v in player_s.values.flatten().tolist()]

    st.write(f"### {player_name} ({team_name}) - {primary_role.capitalize()}")
    st.write(f'Percentile Rankings vs All {primary_role + "s"}')
    if matches_played:
        st.write(f'Matches Played: {matches_played}')

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

    # Similar player recommendation section
    # Similar player recommendation section
    st.write(f"### Similar Players to {player_name}")
    
    # Filter dataset by role, filling NaN with 0 to avoid missing values issue
    # Ensure metrics are properly loaded without NaN values
    # Ensure metrics are properly loaded without NaN values
    # Ensure metrics are properly loaded without NaN values
    # Ensure metrics are properly loaded without NaN values
    # Ensure metrics are properly loaded without NaN values
    df[metrics] = df[metrics].fillna(0)
    
    # Filter players by the primary role or those without any roles
    role_filtered_df = df[(df['player_role'].str.contains(primary_role, case=False, na=False)) | df['player_role'].isna()]
    
    # Check if any players match the role or have no assigned role
    if role_filtered_df.empty:
        st.write(f"No players found with the role '{primary_role}'. Please check the role name and try again.")
    else:
        # Make sure the player exists in the filtered DataFrame
        player_data = role_filtered_df[role_filtered_df['player_name'] == player_name]
        
        if player_data.empty:
            st.write(f"No data found for '{player_name}' in the '{primary_role}' role.")
        else:
            # Initialize variable for similar players
            similar_players = None
    
            # Collect players without roles if the player has no assigned role
            if pd.isna(player_data['player_role'].values[0]):
                # Filter only players with no roles for comparison
                no_role_players = role_filtered_df[role_filtered_df['player_role'].isna()]
                if no_role_players.empty:
                    similar_players = pd.DataFrame()  # No similar players found
                else:
                    # Standardize metrics for players without roles
                    scaler = StandardScaler()
                    scaled_metrics = scaler.fit_transform(no_role_players[metrics])
                    
                    # Train NearestNeighbors model on no role players
                    model = NearestNeighbors(n_neighbors=6, algorithm='auto')
                    model.fit(scaled_metrics)
                    
                    # Find the index of the player in the no role dataset
                    player_index = no_role_players[no_role_players['player_name'] == player_name].index[0]
                    scaled_index = no_role_players.index.get_loc(player_index)
                    
                    # Find similar players
                    distances, indices = model.kneighbors([scaled_metrics[scaled_index]])
                    similar_players = no_role_players.iloc[indices[0]].iloc[1:]  # Exclude the queried player
            
            else:
                # Standardize metrics only for role-filtered players
                scaler = StandardScaler()
                scaled_metrics = scaler.fit_transform(role_filtered_df[metrics])
                
                # Train NearestNeighbors model
                model = NearestNeighbors(n_neighbors=6, algorithm='auto')
                model.fit(scaled_metrics)
                
                # Find the index of the player in the filtered dataset
                player_index = role_filtered_df[role_filtered_df['player_name'] == player_name].index[0]
                scaled_index = role_filtered_df.index.get_loc(player_index)
                
                # Find similar players
                distances, indices = model.kneighbors([scaled_metrics[scaled_index]])
                similar_players = role_filtered_df.iloc[indices[0]].iloc[1:]  # Exclude the queried player
    
            # Display only the heading and similar players if any
            #st.write("### Similar Players")
            if similar_players is not None and not similar_players.empty:
                for _, row in similar_players.iterrows():
                    st.write(f" - {row['player_name']} ({row['team_name']})")
            else:
                st.write("No similar players found.")
