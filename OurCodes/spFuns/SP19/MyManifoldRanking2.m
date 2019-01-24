function [stage2] = MyManifoldRanking2(adjcMatrix, PPSal, bdIds, colDistM)

alpha=0.99;
theta=10;
spNum = size(adjcMatrix, 1);

%% Construct Super-Pixel Graph
adjcMatrix_nn = LinkNNAndBoundary2(adjcMatrix, bdIds); 
% This super-pixels linking method is from the author's code, but is 
% slightly different from that in our Saliency Optimization

W = SetSmoothnessMatrix(colDistM, adjcMatrix_nn, theta);
% The smoothness setting is also different from that in Saliency
% Optimization, where exp(-d^2/(2*sigma^2)) is used
D = diag(sum(W));
optAff =(D-alpha*W)\eye(spNum);
optAff(1:spNum+1:end) = 0;  %set diagonal elements to be zero

% %% Stage 2
% th=mean(stage1);
% stage2=optAff*(stage1 >= th);

stage2 = optAff*PPSal;

clear adjcMatrix PPSal bdIds colDistM
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function W = SetSmoothnessMatrix(colDistM, adjcMatrix_nn, theta)
allDists = colDistM(adjcMatrix_nn > 0);
maxVal = max(allDists);
minVal = min(allDists);

colDistM(adjcMatrix_nn == 0) = Inf;
colDistM = (colDistM - minVal) / (maxVal - minVal + eps);
W = exp(-colDistM * theta);
end

function adjcMatrix = LinkNNAndBoundary2(adjcMatrix, bdIds)
%link boundary SPs
% adjcMatrix(bdIds, bdIds) = 1;

% %link neighbor's neighbor
% adjcMatrix = (adjcMatrix * adjcMatrix + adjcMatrix) > 0;
% adjcMatrix = double(adjcMatrix);

spNum = size(adjcMatrix, 1);
adjcMatrix(1:spNum+1:end) = 0;  %diagnal elements set to be zero
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 3 另一种计算相似性的方法，局部归一化 2016.11.24
% function W = SetSmoothnessMatrix1(colDistM, adjcMatrix_nn, theta)
%     spNum  = size(colDistM,1);
%     adjcMatrix_nn(adjcMatrix_nn==2) = 1;
%     adjcMatrix_nn(1:spNum+1:end) = 0;
%     colDistM = full(colDistM);
%     adjcMatrix_nn = full(adjcMatrix_nn);% 1/0,对角线为0
%    
%     colDistM1      = colDistM.*adjcMatrix_nn;
%     meanFeaDist = sum(colDistM1,2)./(sum(adjcMatrix_nn,2)+eps);
%     meanFeaDist   = repmat(meanFeaDist,[1,size(colDistM1,2)]);
%     colDistM1(adjcMatrix_nn==0) = inf;
%     
%     W  = exp(-2*colDistM1./(meanFeaDist));
%     W(isnan(W)) = 0;
%     clear colDistM1 meanFeaDist
% 
% % allDists = colDistM(adjcMatrix_nn > 0);
% % maxVal = max(allDists);
% % minVal = min(allDists);
% % 
% % colDistM(adjcMatrix_nn == 0) = Inf;
% % colDistM = (colDistM - minVal) / (maxVal - minVal + eps);
% % W = exp(-colDistM * theta);
% clear colDistM adjcMatrix_nn  theta
% end