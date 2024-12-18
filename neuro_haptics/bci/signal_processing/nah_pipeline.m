% Main Processing Script to pipeline process NAH data
% 
% Uses BeMoBIL Pipeline to parse XDF file and preprocess EEG, EMG and
% Classifier Output data

function nah_pipeline(subjects)

    %% config
    current_sys = "c060"; %"mac"% 
    eeglab_ver(current_sys);
    
    %% load configuration
    nah_bemobil_config;
    
    % set to 1 if all files should be recomputed and overwritten
    force_recompute = 1;
    
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
            bemobil_config.merged_filename]);
        
        EYE = pop_loadset([bemobil_config.study_folder filesep ...
            bemobil_config.raw_EEGLAB_data_folder filesep ...
            'sub-' num2str(subject) filesep 'sub-' num2str(subject) '_' ...
            bemobil_config.merged_physio_filename]);
    
        Motion = pop_loadset([bemobil_config.study_folder filesep ...
            bemobil_config.raw_EEGLAB_data_folder filesep ...
            'sub-' num2str(subject) filesep 'sub-' num2str(subject) '_' ...
            bemobil_config.merged_motion_filename]);
    
        event_of_int = 'What:grab';
    
        %% dmatrix questionnaire answers and grab events
    
        events = EEG.event;
        types = {events.type}';
        grabs = find(contains(types, event_of_int));
        grab_events = events(grabs);
        grab_events = nah_parse_events(grab_events);
        event_table = struct2table(grab_events);
        [~, unique_idx] = unique(event_table.Number, 'stable');
        event_table = event_table(unique_idx, :);
    
        % append questionnaire results
        quest = find(contains(types, 'What:rating'));
        quest_events = events(quest);
        quest_events = nah_parse_events(quest_events);
        quest_table = struct2table(quest_events);
        quest_table = quest_table(:, {'answerID'});
    
        % placement accuracy
        pa_ixs = find(contains(types, 'What:placement'));
        place_events = events(pa_ixs);
        place_events = nah_parse_events(place_events);
        pa_table = struct2table(place_events);
        [~, unique_idx] = unique(pa_table.Number, 'stable');
        pa_table = pa_table(unique_idx, :);
        pa_table = pa_table(:, {'AccuracyCm'});
    
        %% Epoching and finding bad epochs
    
        epoch_tw = [-1 2];
    
        EEG.event = table2struct(event_table);
        [EEG.event.type] = deal('event');
        % copy over events
        EYE.event = EEG.event;
        Motion.event = EEG.event;
    
        EEG = pop_epoch( EEG, {  'event'  }, epoch_tw, 'epochinfo', 'yes');
    
        % clean epochs and remove 5 percent of bad epochs from data
        [~, rmepochs] = pop_autorej(EEG, 'nogui', 'on');
        rmepochs = rmepochs(1:.1*size(EEG.data,3));
    
        % add column bad_epoch to event_table and set to 1 for rmepoch indices
        event_table.bad_epoch = zeros(height(event_table), 1);
        event_table.bad_epoch(rmepochs) = 1;
    
        %% Features: Events
    
        % Feature: First intersection of gaze with target location
        fix_events = find(contains(types, 'focus:in;object: PlacementPos'));
        fix_lats = [events(fix_events).latency];
    
        for i = 1:size(event_table,1)
            grab = event_table.latency(i);
            first_fix_after_grab_ix = min((find((fix_lats - grab > 0) == 1)));
            tmp_fix_delay = (fix_lats(first_fix_after_grab_ix) - grab) / EEG.srate;

            if isempty(tmp_fix_delay)
                fix_delay(i) = 0;
            elseif tmp_fix_delay > 1
                fix_delay(i) = 0;
            else
                fix_delay(i) = tmp_fix_delay;
            end
        end

        fix_delay = fix_delay';
        fix_delay = table(fix_delay);
    
        behavior = [event_table quest_table pa_table fix_delay];
        writetable(behavior, strcat(out_path, filesep, sprintf('behavior_s%d.csv', subject)), 'Delimiter', ';');
        clear fix_delay
    
        %% Features: Time Series
    
        % Feature: ERP
        erp = EEG.data;
        save(strcat(out_path, filesep, 'erp', '.mat'), 'erp');
    
        % Feature: Hand Motion
        hand_cart_chans = find(contains({Motion.chanlocs.labels}, 'NAH_rb_handRight_cart'));
    
        Motion = pop_epoch( Motion, {  'event'  }, epoch_tw, 'epochinfo', 'yes');
        hand_motion = Motion.data(hand_cart_chans,:,:);
        save(strcat(out_path, filesep, 'hand_motion', '.mat'), 'hand_motion');
    
        % Feature: Gaze for fixation detection?
        eye_cart_chans = find(contains({EYE.chanlocs.labels}, 'GazeDirection'));
        validity_chan = find(contains({EYE.chanlocs.labels}', 'DataValidity'));
    
        EYE = pop_epoch( EYE, {  'event'  }, epoch_tw, 'epochinfo', 'yes');
        gaze = EYE.data([eye_cart_chans, validity_chan],:,:);
        save(strcat(out_path, filesep, 'gaze', '.mat'), 'gaze');
    
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
end
