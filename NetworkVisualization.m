% Have RegionTransformMatrix.xlsx in working directory, import transform
% matrix
RegionTransformMatrix =table2array(readtable('RegionTransformMatrix.xlsx'));
RegionTransformMatrix(isnan(RegionTransformMatrix))=0;


% Have new_Ta_Shun_2.h5 file in working directory
Spikes = h5read('new_Ta_Shun_2.h5','/spike_state/data');
UIDS = h5read('new_Ta_Shun_2.h5','/spike_state/uids');
TransposeSpikes = Spikes';
TransposeUIDS = UIDS';
NewUIDS = regexprep(TransposeUIDS, '\W', '');
Combined1=array2table(TransposeSpikes,'VariableNames',NewUIDS);
% Use sort_nat here (a function found on an online matlab repo to help sort
% the columuns in a way that takes numbers into account.
Sorted=Combined1(:,sort_nat(Combined1.Properties.VariableNames));
SortedSpiking=table2array(Sorted);
FinalSpikes = double(SortedSpiking(:,1:end-3));

%Transform from neuron class x time to brain region x time:
RegionXTime = FinalSpikes * RegionTransformMatrix;

% Create Heat Map
colormap(hot)
imagesc(RegionXTime)
colorbar
title({'Localized Neuronal Firing'; 'Activity in EB Subregion'})
xlabel('EB Sub-Regions')
xticks([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18])
xticklabels({'R8', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'L8'})
ylabel('Time -- [Seconds]')
yticklabels({'1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0'})

