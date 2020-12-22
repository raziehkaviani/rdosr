%% plot the training curves: training loss and accuracy

%% setting of plot (this is the only part needs to care about)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_epochs = 20000;  % number of epochs to plot

results_path = {'results/cifar10_20_s0.001'};
names = {'rdosr-cifar10'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% get log data
log_file = 'history.log';
loss_F = zeros(num_epochs, 1);
accu_F = zeros(num_epochs, 1);
loss_EDC = zeros(num_epochs, 1);
loss_euc = zeros(num_epochs, 1);
for i = 1:length(results_path)
    data = load_data(fullfile(results_path{i}, log_file));
    data = data(1:num_epochs, :);
    loss_EDC(:, i) = data(:, 1);
    loss_euc(:, i) = data(:, 2);
end


%% plot the training loss and testing accuracy curves
close all
figure('name', 'Training Curves', 'numbertitle', 'off', ...
    'units', 'normalized', 'position', [.1 .2 .6, .4])
subplot(121)
plot(loss_EDC, 'linewidth', 2)
grid on
xlabel('Training Epoch')
ylabel('Testing EDC Loss')
set(gca, 'fontsize', 10, 'fontweight', 'bold')

subplot(122)
plot(loss_euc, 'linewidth', 2)
grid on
xlabel('Training Epoch')
ylabel('Testing euc Loss')
set(gca, 'fontsize', 10, 'fontweight', 'bold')


%% function for reading log
function data = load_data(file_dir)
    fid = fopen(file_dir);
    line = fgetl(fid);
    data = [];
    try
        while ischar(line)
            line = fgetl(fid);
            str = strsplit(line, ',');
            if length(str) ~= 3
                continue;
            end
            data = [data ; [str2double(str{2}) str2double(str{3})]];
        end
        fclose(fid);
    catch
        fclose(fid);
    end
end
