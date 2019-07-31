clear subject;
clear cfg;
clear data;

% build subject data format
subject.id = '024';
subject.dir = fullfile('/home/bearsuny/Projects/seeg/data', ['subject_', subject.id]);
subject.seeg_file = [subject.id, '_LAT.edf'];
subject.event_file = [subject.id, '_event.csv'];

% set trial function
cfg = [];
cfg.dataset = fullfile(subject.dir, subject.seeg_file);
cfg.event_file = fullfile(subject.dir, subject.event_file);
cfg.trialfun = 'trialfun_seeg';

% set event
cfg.trialdef.eventtype = 'block';
cfg.trialdef.eventvalue = [];
cfg.trialdef.prestim = 0.2;
cfg.trialdef.poststim = 1;
cfg = ft_definetrial(cfg);

data = ft_preprocessing(cfg);


