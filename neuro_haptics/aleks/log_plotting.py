from rl.ucbq_agent import UCBQAgent

import matplotlib.pyplot as plt

# Get the first 5 standard colors from the default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:5]

# Create a map that maps integers to these colors
color_map = {i: color for i, color in enumerate(colors)}

def plot_q_values(df_plot_q_values, ax, initial_q_value=0):
    line_colors = [color_map[col] for col in df_plot_q_values.columns]
    df_plot_q_values.plot(ax=ax, marker='o', 
                        #   markersize=8, zorder=3, 
                          color=line_colors)
    
    df_plot_q_values = df_plot_q_values.interpolate().fillna(initial_q_value)
    for col in range(len(df_plot_q_values.columns)):
        if col not in df_plot_q_values.columns:
            df_plot_q_values[col] = initial_q_value
    df_plot_q_values.plot(ax=ax, 
                        #   linewidth=3, 
                          color=line_colors)

def plot_additional_metrics(df, ax):
    df.plot('t', ['reward_adjusted', 'alpha', 'epsilon'],
            # Re-enable when plotting not_separate
            #  color=['lightblue', 'lightgreen', 'lightpink'], 
             ax=ax, zorder=0, linewidth=1)

def customize_plot(ax):
    ax.axvline(x=25, color='lightgray', linestyle='--', 
            #    zorder=0, linewidth=1
               )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:5] + handles[10:], labels[:5] + labels[10:], fontsize='small')

def plot_q_learning(df, ax):
    df_plot_q_values = df.pivot(index='t', columns='action', values='new_Q_value')
    plot_q_values(df_plot_q_values, ax)
    plot_additional_metrics(df, ax)
    customize_plot(ax)

def plot_q_learning_separate(df, ax):
    initial_q_value = UCBQAgent().start_q_value
    ax1, ax2 = ax
    
    df_plot_q_values = df.pivot(index='t', columns='action', values='new_Q_value')
    plot_q_values(df_plot_q_values, ax1, initial_q_value)
    plot_additional_metrics(df, ax2)
    
    # Pre-emptive exploration
    # ax1.axvline(x=25, color='lightgray', linestyle='--', zorder=0)
    # ax2.axvline(x=25, color='lightgray', linestyle='--', zorder=0)
       
    handles1, labels1 = ax1.get_legend_handles_labels()
    num_actions = 4
    ax1.legend(handles1[:num_actions], labels1[:num_actions], fontsize='small')
