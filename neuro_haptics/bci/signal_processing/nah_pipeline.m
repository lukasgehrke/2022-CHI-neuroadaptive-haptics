% Main Processing Script to pipeline process NAH data
% 
% Uses BeMoBIL Pipeline to parse XDF file and preprocess EEG, EMG and
% Classifier Output data

%% config
current_sys = "mac";
eeglab_ver(current_sys);

%% load configuration
nah_bemobil_config;

% set to 1 if all files should be recomputed and overwritten
force_recompute = 1;
subjects = 1 % [1:7]; % 1

%% preprocess

for subject = subjects
    nah_import(bemobil_config, subject);
    % nah_preprocess_EEG;
end

%% export features

for subject = subjects

    disp(subject);

    %% parse data

    EEG = pop_loadset([bemobil_config.study_folder filesep ...
        bemobil_config.raw_EEGLAB_data_folder filesep ...
        'sub-' num2str(subject) filesep 'sub-' num2str(subject) '_' ...
        bemobil_config.merged_filename]); % eye components rejected
    out_path = [bemobil_config.study_folder filesep ...
        bemobil_config.single_subject_analysis_folder filesep 'sub-' num2str(subject) filesep];
    if ~exist(out_path)
        mkdir(out_path);
    end

    %% dmatrix questionnaire answers and grab events

    events = {EEG.event.type}';
    lats = {EEG.event.latency}';

    % grab events, add grab latency to event matrix
    grabs = find(contains(events, 'What:grab'));
    grab_events = events(grabs);
    [C,IA,IC] = unique(grab_events);
    ixs = grabs(sort(IA)); % IA is first index of uniques    
    grab_events = events(ixs);
    lats = lats(ixs);
    
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

    % clean
    [~, rmepochs] = pop_autorej(EEG, 'nogui', 'on');

    % add column bad_epoch to event_table and set to 1 for rmepoch indices
    event_table.bad_epoch = zeros(height(event_table), 1);
    event_table.bad_epoch(rmepochs) = 1;

    behavior = [event_table quest_table pa_table];
    writetable(behavior, strcat(out_path, filesep, sprintf('behavior_s%d.csv', subject)), 'Delimiter', ';');

    %% Features

    % ERP:
    erp = EEG.data(:,250:500,:);
    save(strcat(out_path, filesep, 'erp', '.mat'), 'erp');

    % Gaze:
    % - find first fixation on target object after grab event
    % - gaze velocity?


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