function event = nah_parse_events(events)

    clear event

    % split each entry in events by the colon and keep the second part
    types = {events.type}';
    types = cellfun(@(x) strsplit(x, ';'), types, 'UniformOutput', false);
    
    % loop through each cell in events and make a struct whose fields are taken from the key:value pair
    for i = 1:length(events)
        for j = 1:length(types{i})
            keyval = strsplit(types{i}{j}, ':');

            % if keyval{2} is a number, convert it to a number
            if ~isnan(str2double(keyval{2}))
                keyval{2} = str2double(keyval{2});
            end
            event(i).(keyval{1}) = keyval{2};
        end

        % copy other event fields
        event(i).latency = events(i).latency;
        event(i).urevent = events(i).urevent;
        event(i).duration = events(i).duration;

    end
end
