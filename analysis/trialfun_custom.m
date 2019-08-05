% http://www.fieldtriptoolbox.org/example/making_your_own_trialfun_for_conditional_trial_definition/
% http://www.fieldtriptoolbox.org/reference/ft_read_event/
% http://www.fieldtriptoolbox.org/faq/what_is_the_relation_between_events_such_as_triggers_and_trials/
function [trl, event] = trialfun_custom(cfg)
    fid = fopen(cfg.event_file);
    raw_event = textscan(fid, '%s%s%s%s', 'delimiter', ',', 'HeaderLines', 1);
    fclose(fid);

    event = struct('type', raw_event{2}, 'value', raw_event{3}, 'sample', raw_event{4});
    sel = true(size(event));

    if isfield(cfg.trialdef, 'eventtype')
        sel = sel & strcmp({event.type}', cfg.trialdef.eventtype);
    end

    if isfield(cfg.trialdef, 'eventvalue')
        sel = sel & strcmp({event.value}', cfg.trialdef.eventvalue);
    end

    hdr = ft_read_header(cfg.dataset);
    pretrig = -round(cfg.trialdef.pretrig * hdr.Fs); % e.g., 0.2 sec before trigger
    posttrig = round(cfg.trialdef.posttrig * hdr.Fs); % e.g., 1 sec after trigger

    e_sample = [str2num(char(event(sel).sample))];
    e_type = {event(sel).type}';
    e_value = [str2num(char(event(sel).value))];

    experiment = loadjson(cfg.experiment_file);

    e_type_unique = fieldnames(experiment);
    e_type_mask = zeros(size(e_type));

    for i = 1:numel(e_type_unique)
        e_type_mask(find(strcmp(e_type_unique(i), e_type))) = i;
    end

    begin_sample = e_sample(:) + pretrig;
    end_sample = e_sample(:) + posttrig;
    % http://www.fieldtriptoolbox.org/reference/ft_definetrial/
    offset = zeros(size(begin_sample)) + pretrig; % number of samples prior to the trigger

    % http://www.fieldtriptoolbox.org/faq/is_it_possible_to_keep_track_of_trial-specific_information_in_my_fieldtrip_analysis_pipeline/
    trl = table2array(table(begin_sample, end_sample, offset, e_type_mask, e_value));

end
