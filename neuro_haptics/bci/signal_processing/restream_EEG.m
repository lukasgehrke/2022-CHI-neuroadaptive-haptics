
%% config
eeglab; 

path = '/Users/lukasgehrke/Documents/publications/2022-CHI-neuroadaptive-haptics/'; % repository path

%% load data and parse events

EEG = pop_loadxdf(fullfile(path, 'example_data', 'EMS1.xdf'), ...
    'streamtype', 'EEG', 'exclude_markerstreams', {});
allEventsLats = [EEG.event.latency];

%% instantiate the library
disp('Loading library...');
lib = lsl_loadlib(); % needs the lsl library in path

% make a new stream outlet
disp('Creating a new streaminfo...');
info = lsl_streaminfo(lib,'BrainVision RDA','EEG',64,250,'cf_float32','sdfwerr32432');

disp('Opening an outlet...');
outlet = lsl_outlet(info);

% make a new stream outlet
info_marker = lsl_streaminfo(lib,'fastReach_restream','Markers',1,0,'cf_string','sdfwerr32432');
disp('Opening an outlet...');
outlet_marker = lsl_outlet(info_marker);

%% send data into the outlet, sample by sample
disp('Now transmitting data...');

i = 1;
last_event = '';
while true
    data = double(EEG.data(:,i));
    outlet.push_sample(data);
    pause(0.004);

    i = i + 1;
    if i > size(EEG.data,2)
        i = 1;
    end

    current_ev_ix = max(find(i>allEventsLats));
    if ~isempty(current_ev_ix)
        event = EEG.event(current_ev_ix).type;

        if ~strcmp(event, last_event)
            disp(event);
            outlet_marker.push_sample({event});
            last_event = event;
        end
    end

end




