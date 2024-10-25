# 2022-CHI-neuroadaptive-haptics

# Launch dev environment

1. Open command prompt (Win + R)
2. Run `code "D:\Lukas\2022-CHI-neuroadaptive-haptics" "D:\Lukas\2022-CHI-neuroadaptive-haptics\run_experiment_dev.ipynb"`
3. Make sure you're on the correct branch `aleks-lab` (check in the bottom left corner)
4. Press `Run All` in the notebook that opens
5. In the VSCode that opens the unity project, make sure you're on the correct branch `aleks-nah`

# Experiment setup steps

1. Use VIVE Pro Eye 3
2. Switch on 4 lighthouses (don’t have to be blue, but need to show up in SteamVR)
    1. Make sure to connect the lighthousese before the trackers
3. Switch on steam
4. Switch on right tracker
    1. Check that it is paired to Steam VR (sometimes it might forget the trackers)
        1. If that happens pair manually
            1. Right click on the first (every time the first) controller,
            2. pair a diffeerent device
            3. select tracker
            4. Hold the button until iit starts flashing
5. Switch on the “callibration” tracker
6. Switch on right Nova1 glove
    1. Callibrate in SenseCom, before starting Unity
7. Start Unity project
    1. branch `aleks-no-buttons`
    2. If the hand is all weird,close unity and calibrate again, and open ujnity agaim
8. Run`../2022-CHI-neuroadaptive-haptics/neuro_haptics/aleks/run_experiment_lsl.py`
    1. Make sure to Ctrl+C after you’re done with the script. If you restart the Unity project while the AI script is still running, Unity will throw an error
9. Run `../2022-CHI-neuroadaptive-haptics/neuro_haptics/aleks/log_plot_live.ipynb` for a fancy live plot of the logs