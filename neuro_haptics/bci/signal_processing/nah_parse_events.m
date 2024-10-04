function event = nah_parse_events(events)

    clear event

    % split each entry in events by the colon and keep the second part
    events = cellfun(@(x) strsplit(x, ';'), events, 'UniformOutput', false);
    
    % loop through each cell in events and make a struct whose fields are taken from the key:value pair
    for i = 1:length(events)
        for j = 1:length(events{i})
            keyval = strsplit(events{i}{j}, ':');
            event(i).(keyval{1}) = keyval{2};
        end
    end
end
