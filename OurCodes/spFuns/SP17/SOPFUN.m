function  [PPSal1] = SOPFUN(adjcMatrix,bdIds,FeaDist,PPSal,param)
% ���¼������ֵ���ر��� bndCon 
% 2016.11.26 15:04pm
% COPYRIGHT BY XIAOFEI ZHOU,SHANGHAI UNIVERSITY
% 
    [clipVal, geoSigma, neiSigma] = EstimateDynamicParas(adjcMatrix, FeaDist);
    
    
    [~, ~, bgWeight] = MyEstimateBgProb(FeaDist, adjcMatrix, bdIds, clipVal, geoSigma);
    
    %post-processing for cleaner fg cue
    removeLowVals = param.removeLowVals;
    if removeLowVals
       thresh = graythresh(PPSal);  %automatic threshold
       PPSal(PPSal < thresh) = 0;
    end
    
    PPSal1 = MySaliencyOptimization(adjcMatrix, bdIds, FeaDist, neiSigma, bgWeight, PPSal);

    clear adjcMatrix FeaDist PPSal
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 ������bdConSigma, �� bgWeight�ļ��㷽ʽ
function [bgProb, bdCon, bgWeight] = ...
    MyEstimateBgProb(colDistM, adjcMatrix, bdIds, clipVal, geoSigma)
% Estimate background probability using boundary connectivity

bdCon = BoundaryConnectivity(adjcMatrix, colDistM, bdIds, clipVal, geoSigma, true);

% bdConSigma = 1; %sigma for converting bdCon value to background probability
% fgProb = exp(-bdCon.^2 / (2 * bdConSigma * bdConSigma)); %Estimate bg probability
meanbdCon = mean(bdCon.^2);
if meanbdCon==0
    meanbdCon = meanbdCon + eps;
end
fgProb = exp(-2*(bdCon.^2) ./ (meanbdCon));
bgProb = 1 - fgProb;

bgWeight = bgProb;
% Give a very large weight for very confident bg sps can get slightly
% better saliency maps, you can turn it off.

% fixHighBdConSP = true;
% highThresh = 3;
% if fixHighBdConSP
%     bgWeight(bdCon > highThresh) = 1000;
% end
clear colDistM adjcMatrix bdIds clipVal geoSigma
end

% 2 ����Wn�ļ��㷽��������ȫ�ֹ�һ���ķ�ʽ����ָ��
function optwCtr = MySaliencyOptimization(adjcMatrix, bdIds, colDistM, neiSigma, bgWeight, fgWeight)
% Solve the least-square problem in Equa(9) in our paper


adjcMatrix_nn = LinkNNAndBoundary(adjcMatrix, bdIds);
colDistM(adjcMatrix_nn == 0) = Inf;
Wn = Dist2WeightMatrix(colDistM, neiSigma);      %smoothness term

% allDists = colDistM(adjcMatrix_nn > 0);
% maxVal = max(allDists);
% minVal = min(allDists);
% theta = 10;
% colDistM(adjcMatrix_nn == 0) = Inf;
% colDistM = (colDistM - minVal) / (maxVal - minVal + eps);
% Wn = exp(-colDistM * theta);


mu = 0.1;                                                   %small coefficients for regularization term
W = Wn + adjcMatrix * mu;                                   %add regularization term
D = diag(sum(W));

bgLambda = 5;   %global weight for background term, bgLambda > 1 means we rely more on bg cue than fg cue.
E_bg = diag(bgWeight * bgLambda);       %background term
E_fg = diag(fgWeight);          %foreground term

spNum = length(bgWeight);
optwCtr =(D - W + E_bg + E_fg) \ (E_fg * ones(spNum, 1));

clear adjcMatrix bdIds colDistM neiSigma bgWeight fgWeight
end
