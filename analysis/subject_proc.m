function [] = subject_proc(cfg)
    disp(cfg);

    % save eventtype label for later usage
    experiment = loadjson(cfg.experiment_file);
    e_type_unique = fieldnames(experiment);
    save_root = cfg.subject.root;
    bad_channel = cfg.bad_channel;

    % extract trial
    % http://www.fieldtriptoolbox.org/reference/ft_preprocessing/
    electrode = loadjson(cfg.electrode_file);
    cfg.electrode = electrode;
    electrode_name = fieldnames(cfg.electrode);

    cfg.channel = {};

    % exclude bad channel defined in electrode json file.
    for i = 1:numel(electrode_name)

        if ~ismember(electrode_name(i), cfg.bad_channel)
            electrode_item_list = cfg.electrode.(electrode_name{i});

            for j = 1:numel(electrode_item_list)
                cfg.channel = [cfg.channel; electrode_item_list(j)];
            end

        end

    end

    cfg = ft_definetrial(cfg);
    data = ft_preprocessing(cfg);

    % reject artifact
    % http://www.fieldtriptoolbox.org/tutorial/visual_artifact_rejection/
    %     cfg = [];
    %     cfg.method = 'trial';
    %     cfg.alim = 3e2;
    %     data = ft_rejectvisual(cfg, data);

    % detrend and baseline correct
    % http://www.fieldtriptoolbox.org/workshop/natmeg/preprocessing/
    % http://www.fieldtriptoolbox.org/reference/ft_preprocessing/
    cfg = [];
    cfg.preproc.detrend = 'yes';
    cfg.preproc.baselinewindow = [-0.1 0];
    data = ft_preprocessing(cfg, data);

    % groupby brain area and average and save
    for i = 1:numel(e_type_unique)
        experiment_item_list = experiment.(e_type_unique{i});

        for j = 1:numel(experiment_item_list)
            cfg = [];
            cfg.trials = find(data.trialinfo(:, 1) == i & data.trialinfo(:, 2) == experiment_item_list(j));

            if isempty(cfg.trials)
                continue;
            end

            file_path = fullfile(save_root, 'avg', e_type_unique{i}, num2str(experiment_item_list(j)));
            mkdir(file_path);

            for k = 1:numel(electrode_name)
                cfg.channel = {};
                electrode_item_list = electrode.(electrode_name{k});

                disp([i, j, electrode_name{k}]);

                flag = false;

                for y = 1:numel(bad_channel)

                    if strcmp(electrode_name{k}, bad_channel{y})
                        flag = true;
                    end

                end

                if flag
                    continue;
                end

                for z = 1:numel(electrode_item_list)
                    cfg.channel = [cfg.channel; electrode_item_list(z)];
                end

                % http://www.fieldtriptoolbox.org/reference/ft_timelockanalysis/
                cfg.outputfile = fullfile(file_path, [electrode_name{k}, '.mat']);
                data_item = ft_timelockanalysis(cfg, data);
            end

        end

    end

end
