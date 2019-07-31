% http://www.fieldtriptoolbox.org/example/making_your_own_trialfun_for_conditional_trial_definition/
% http://www.fieldtriptoolbox.org/reference/ft_read_event/
function [trl, event] = trialfun_seeg(cfg);
    hdr = ft_read_header(cfg.dataset);
    fid = fopen(cfg.event_file);
    raw_event = textscan(fid, '%s%s%d%d', 'delimiter', ',', 'HeaderLines', 1);
    fclose(fid);

    % extract event
    event = struct('type', '', 'value', 0, 'sample', 0);

    for i = 1:numel(raw_event{1})
        event(i).type = raw_event{2}(i);
        event(i).value = raw_event{3}(i);
        event(i).sample = raw_event{4}(i);
    end

    if cfg.trialdef.eventtype == '?'
        value = [event.value]';
        sample = [event.sample]';
    else
        value = [event(strcmp(cfg.trialdef.eventtype, [event.type]')==1).value]';
        sample = [event(strcmp(cfg.trialdef.eventtype, [event.type]')==1).sample]';
    end

    % extract trial
    trl = zeros(numel(sample), 3);
    prestim = cfg.trialdef.prestim * hdr.Fs; % e.g., 1 sec before trigger
    poststim = cfg.trialdef.poststim * hdr.Fs; % e.g., 2 sec after trigger
    offset = -hdr.nSamplesPre; % number of samples prior to the trigger

    for i = 1:numel(sample)

        if isempty(cfg.trialdef.eventvalue)
            trlbegin = sample(i) - prestim;
            trlend = sample(i) + poststim;
            newtrl = [trlbegin trlend offset];
            trl(i, 1:(length(newtrl))) = [trlbegin trlend offset]; % store in the trl matrix
        else

            if ismember(value(i), cfg.trialdef.eventvalue)
                trlbegin = sample(i) - prestim;
                trlend = sample(i) + poststim;
                newtrl = [trlbegin trlend offset];
                trl(i, 1:(length(newtrl))) = [trlbegin trlend offset]; % store in the trl matrix
            end

        end

    end

end
