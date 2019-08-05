clc;
clear;

addpath('../tool/jsonlab-1.5');

% global config
clear g_cfg;
g_cfg.data_root = '../data';
g_cfg.subject_list = {'024', '015'};
g_cfg.seeg_file_name = 'LAT.edf';
g_cfg.event_file_name = 'event.csv';
g_cfg.electrode_file_name = 'electrode.json';
g_cfg.experiment_file_name = 'experiment.json';
g_cfg.bad_channel = {'MARK', 'EMG', 'EKG'};

disp(g_cfg);

for i = 1:numel(g_cfg.subject_list)

    clear cfg;
    cfg.subject.id = g_cfg.subject_list{i};
    cfg.subject.root = fullfile(g_cfg.data_root, cfg.subject.id);
    cfg.dataset = fullfile(cfg.subject.root, g_cfg.seeg_file_name);
    cfg.event_file = fullfile(cfg.subject.root, g_cfg.event_file_name);
    cfg.electrode_file = fullfile(cfg.subject.root, g_cfg.electrode_file_name);
    cfg.experiment_file = fullfile(cfg.subject.root, g_cfg.experiment_file_name);

%     cfg.trialdef.eventtype = 'condition'; % e.g., 'condtion'
%     cfg.trialdef.eventvalue = '12'; % e.g., '10'
    cfg.trialdef.pretrig = 200/1000;
    cfg.trialdef.posttrig = 640/1000;
    cfg.trialfun = 'trialfun_custom';

    cfg.bad_channel = g_cfg.bad_channel;

    subject_proc(cfg);
    break;
end

clear;
