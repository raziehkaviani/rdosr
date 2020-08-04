%% plot the training curves: training loss and accuracy

%% setting of plot (this is the only part needs to care about)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_epochs = 15000;  % number of epochs to plot

results_path = {'results/PaviaU_10_s0.001'};
names = {'rdosr-PaviaU'};

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
    loss_F(:, i) = data(:, 1);
    accu_F(:, i) = data(:, 2);
    loss_EDC(:, i) = data(:, 3);
    loss_EDC(:, i) = data(:, 4);
end


%% plot the training loss and testing accuracy curves
close all
figure('name', 'Training Curves', 'numbertitle', 'off', ...
    'units', 'normalized', 'position', [.1 .2 .6, .4])
subplot(141)
plot(loss_F, 'linewidth', 2)
grid on
xlabel('Training Epoch')
ylabel('Testing F Loss')
set(gca, 'fontsize', 10, 'fontweight', 'bold')

subplot(142)
plot(accu_F, 'linewidth', 2)
grid on
xlabel('Training Epoch')
ylabel('Testing F Accuracy')
set(gca, 'fontsize', 10, 'fontweight', 'bold')

subplot(143)
plot(loss_F, 'linewidth', 2)
grid on
xlabel('Training Epoch')
ylabel('Testing EDC Loss')
set(gca, 'fontsize', 10, 'fontweight', 'bold')

subplot(144)
plot(loss_F, 'linewidth', 2)
grid on
xlabel('Training Epoch')
ylabel('Testing euc Loss')
set(gca, 'fontsize', 10, 'fontweight', 'bold')
legend(names, 'location', 'northeast', 'interpreter', 'latex', ...
    'fontsize', 16)


%% function for reading log
function data = load_data(file_dir)
    fid = fopen(file_dir);
    line = fgetl(fid);
    data = [];
    try
        while ischar(line)
            line = fgetl(fid);
            str = strsplit(line, ',');
            if length(str) ~= 5
                continue;
            end
            data = [data ; [str2double(str{2}) str2double(str{3}) ...
                str2double(str{4}) str2double(str{5})]];
        end
        fclose(fid);
    catch
        fclose(fid);
    end
end
