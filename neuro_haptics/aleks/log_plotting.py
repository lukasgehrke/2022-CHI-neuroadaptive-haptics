import matplotlib.pyplot as plt

# Get the first 5 standard colors from the default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:5]

# Create a map that maps integers to these colors
color_map = {i: color for i, color in enumerate(colors)}

color_map

def plot_q_learning(df, fig, ax):
    initial_q_value = 0
    
    df_plot_q_values = df.pivot(index='t', columns='action', values='new_Q_value')
    
    # Draw dots if the action was chosen
    # Need the NaNs here
    line_colors = [color_map[col] for col in df_plot_q_values.columns]
    df_plot_q_values.plot(ax=ax, marker='o', markersize=8, zorder=3, color=line_colors)

    
    # Draw lines connecting the dots, representing current Q values
    df_plot_q_values = df_plot_q_values.interpolate()
    df_plot_q_values = df_plot_q_values.fillna(initial_q_value)    

    for col in range(5):
        if col not in df_plot_q_values.columns:
            df_plot_q_values[col] = initial_q_value

    df_plot_q_values.plot(ax=ax, linewidth=3, color=line_colors)




    df_plot = df

    df_plot.plot('t', [
        # 'reward', 
        'reward_adjusted', 
        'alpha', 'epsilon'], color=['lightblue', 'lightgreen', 'lightpink'],
        ax=ax, zorder=0,
        linewidth=1)

    # lines = ax.get_lines()
    # lines[0].set_linewidth(2)  # Make the reward line thicker
    # lines[1].set_color('lightblue')  # Make the reward_adjusted line light blue
    # lines[0].set_linestyle('--')  # Make the reward line dashed

    # Draw a vertical light gray line at x=25
    ax.axvline(x=25, color='lightgray', linestyle='--', zorder=0, linewidth=1)

    # Make the plot bigger
    fig = plt.gcf()
    fig.set_size_inches(12, 8)


    # Exclude line labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:5] + handles[10:], labels[:5] + labels[10:], fontsize='small')