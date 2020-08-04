clear all;
%% plot the histogram of reconstruction errors

%% setting of plot (this is the only part needs to care about)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

results_path = 'results/PaviaU_10_s0.001';
names = 'rdosr-PaviaU';

view_error_histogram=1;
step = 0.1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% get log data
output_file = 'outputs.mat';
load(fullfile(results_path, output_file));

kwn = recons_error(y~=-1);
unk = recons_error(y==-1);


%% plot the histogram for known and unknown sets
close all;
figure('name', 'Histograms', 'numbertitle', 'off', ...
    'units', 'normalized', 'position', [.1 .2 .6, .4])
x = (min(recons_error)-0.05):step:(max(recons_error)+0.05);
if(view_error_histogram)
    histogram(unk,x,'FaceColor','r','LineStyle','--','EdgeAlpha',0.1,'Normalization','probability');
    hold on;
    histogram(kwn,x,'FaceColor','g','EdgeAlpha',0.1,'Normalization','probability');
    legend('Unknown','Known');
    xlim([0 max(unk)]);
    pause(1)
end
xlabel('Reconstruction Error')
ylabel('P(Reconstruction Error)')
set(gca, 'fontsize', 10, 'fontweight', 'bold')


%% plot ROC curve
label_act = ones(1,size(recons_error,2));
label_act(1,size(kwn,2)+1:end) = -1;
label_pred = recons_error;
[X,Y,T,AUC] = perfcurve(label_act, label_pred, -1);

figure('name', 'ROC Curve', 'numbertitle', 'off', ...
    'units', 'normalized')
plot(X,Y, 'linewidth', 2);
xlabel('False Positive Rate')
ylabel('True Positive Rate')
set(gca, 'fontsize', 10, 'fontweight', 'bold')
legend(names, 'location', 'northeast', 'interpreter', 'latex', ...
    'fontsize', 16)
