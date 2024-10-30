% Main Processing Script to pipeline process NAH data
% 
% Uses BeMoBIL Pipeline to parse XDF file and preprocess EEG, EMG and
% Classifier Output data

%% config
current_sys = "c060";
eeglab_ver(current_sys);

%% load configuration
nah_bemobil_config;

% set to 1 if all files should be recomputed and overwritten
force_recompute = 1;
subjects = 1; % [1:7]; % 1

%% preprocess

for subject = subjects
    nah_import(bemobil_config, subject);
    % nah_preprocess_EEG;
end

%% export features

for subject = subjects

    disp(subject);

    %% parse data

    out_path = [bemobil_config.study_folder filesep ...
        bemobil_config.single_subject_analysis_folder filesep 'sub-' num2str(subject) filesep];
    if ~exist(out_path, 'dir')
        mkdir(out_path);
    end

    EEG = pop_loadset([bemobil_config.study_folder filesep ...
        bemobil_config.raw_EEGLAB_data_folder filesep ...
        'sub-' num2str(subject) filesep 'sub-' num2str(subject) '_' ...
        bemobil_config.merged_filename]); % eye components rejected
    
    EYE = pop_loadset([bemobil_config.study_folder filesep ...
        bemobil_config.raw_EEGLAB_data_folder filesep ...
        'sub-' num2str(subject) filesep 'sub-' num2str(subject) '_' ...
        bemobil_config.merged_physio_filename]); % eye components rejected

    events = {EEG.event.type}';
    all_lats = {EEG.event.latency}';

    %% dmatrix questionnaire answers and grab events

    % grab events, add grab latency to event matrix
    grabs = find(contains(events, 'What:grab'));
    grab_events = events(grabs);
    [C,IA,IC] = unique(grab_events);
    ixs = grabs(sort(IA)); % IA is first index of uniques    
    grab_events = events(ixs);
    lats = all_lats(ixs);
    
    grab_events = nah_parse_events(grab_events);
    event_table = struct2table(grab_events);
    event_table.latency = lats;

    % append questionnaire results
    quest = find(contains(events, 'What:end'));
    quest_events = events(quest);
    quest_events = nah_parse_events(quest_events);
    quest_table = struct2table(quest_events);
    quest_table = quest_table(:, {'answerID'});

    % placement accuracy
    pa_ixs = find(contains(events, 'What:placement'));
    pa_ixs = pa_ixs(sort(IA)); % IA is first index of uniques
    place_events = events(pa_ixs);
    place_events = nah_parse_events(place_events);
    pa_table = struct2table(place_events);
    pa_table = pa_table(:, {'AccuracyCm'});

    %% Find bad epochs

    EEG.event = EEG.event(ixs);
    [EEG.event.type] = deal('grab');
    EEG = pop_epoch( EEG, {  'grab'  }, [-1  2], 'epochinfo', 'yes');

    EYE.event = EEG.event;
    EYE = pop_epoch( EYE, {  'grab'  }, [-1  2], 'epochinfo', 'yes');

    % clean
    [~, rmepochs] = pop_autorej(EEG, 'nogui', 'on');

    % add column bad_epoch to event_table and set to 1 for rmepoch indices
    event_table.bad_epoch = zeros(height(event_table), 1);
    event_table.bad_epoch(rmepochs) = 1;

    %% Features

    % Fixations: find first fixation on target object after grab event
    fix_events = find(contains(events, 'focus:in;object: PlacementPos'));
    fix_lats = all_lats(fix_events);

    for i = 1:size(event_table,1)
        grab = event_table.latency{i};
        first_fix_after_grab_ix = min((find((cell2mat(fix_lats) - grab > 0) == 1)));
        fix_delay(i) = (fix_lats{first_fix_after_grab_ix} - grab) / EEG.srate;
    end
    fix_delay = fix_delay';
    fix_delay = table(fix_delay);

    behavior = [event_table quest_table pa_table fix_delay];
    writetable(behavior, strcat(out_path, filesep, sprintf('behavior_s%d.csv', subject)), 'Delimiter', ';');
    clear fix_delay

    %% ERP:
    
    erp = EEG.data(:,250:500,:);
    save(strcat(out_path, filesep, 'erp', '.mat'), 'erp');

    % Gaze velocity
    gaze = EYE.data(:,250:500,:);
    gaze_direction_chans = find(contains({EYE.chanlocs.labels}, 'GazeDirection'));
    validity_channel = find(contains({EYE.chanlocs.labels}', 'DataValidity'));

    for i = 1:size(EYE.epoch,2)
        tmp = diff(gaze(gaze_direction_chans,:,i),1,2);
        gaze_velocity(i,:) = sqrt(sum(tmp.^2));
        
%         invalid_samples = gaze(validity_channel,1:end-1,i) == 1;
%         gaze_velocity(i,invalid_samples) = nan;
    end

    save(strcat(strcat(out_path, filesep), 'gaze_velocity', '.mat'), 'gaze_velocity');
    clear gaze_velocity

    %% EEG: channel ERSP
    
    % elecs = [13, 65];
    % 
    % % newtimef settings
    % fft_options = struct();
    % fft_options.cycles = [3 0.5];
    % fft_options.padratio = 2;
    % fft_options.freqrange = [3 100];
    % fft_options.freqscale = 'linear';
    % fft_options.n_freqs = 60;
    % fft_options.timesout = 200;
    % fft_options.alpha = NaN;
    % fft_options.powbase = NaN;
    % 
    % for elec = elecs
    %     ersp_in = squeeze(EEG.data(elec,:,:));
    % 
    %     [~,~,~,times,freqs,~,~,tfdata] = newtimef(ersp_in,...
    %         EEG.pnts,...
    %         [EEG.xmin EEG.xmax]*1000,...
    %         EEG.srate,...
    %         'cycles',fft_options.cycles,...
    %         'freqs',fft_options.freqrange,...
    %         'freqscale',fft_options.freqscale,...
    %         'padratio',fft_options.padratio,...
    %         'baseline',[NaN],... % no baseline, since that is only a subtraction of the freq values, we do it manually
    %         'nfreqs',fft_options.n_freqs,...
    %         'timesout',fft_options.timesout,...
    %         'plotersp','off',...
    %         'plotitc','off',...
    %         'verbose','off');
    % 
    %     ersp = abs(tfdata).^2; %discard phase (complex valued)
    % 
    %     % clean
    %     ersp = ersp(:,times>-300,:);
    %     base = squeeze(mean(ersp(:,times<-100,:),2));
    % 
    %     % dummy check
    %     tmp_ersp = mean(ersp,3);
    %     tmp_ersp = tmp_ersp ./ mean(base,2); % divisive baseline
    %     figure; imagesc(10.*log10(tmp_ersp), [-2 2]); axis xy; colorbar; %cbar([-1 1]);
    % 
    %     save(strcat(out_path, filesep, 'times', '.mat'), 'times');
    %     save(strcat(out_path, filesep, 'freqs', '.mat'), 'freqs');
    %     save(strcat(out_path, filesep, EEG.chanlocs(elec).labels, '_ersp', '.mat'), 'ersp');
    % end

end
