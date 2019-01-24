function SALS = temporalPropagationNew3(CURINFOR,PREINFOR,TPSAL1,betas)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 于前后相邻帧构造图模型，并执行GMR
% 在初步时域传播的基础上进行GMR传播
% 2016.11.17 13:42PM
% copyright by xiaofei zhou
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPSCALENUM = length(CURINFOR.fea);
SALS = cell(SPSCALENUM,1);
no_dims = 0.99;
imSal_pre0 = PREINFOR.imsal;
[height,width]  = size(PREINFOR.imsal);
% normDist = sqrt((height.^2 + width.^2));

for ss=1:SPSCALENUM
%% 1. initialization &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    beta = betas{ss,1};
    indexs  = beta(:,2);
    weights = beta(:,1);
    weights = weights/sum(weights);
    
    % 多尺度信息
    tmpSPcur = CURINFOR.spinfor{ss,1};
    tmpSPpre = PREINFOR.spinfor{ss,1};
    regionCenter = tmpSPcur.region_center;
    
    % 特征
    tmpFea_pre = PREINFOR.fea{ss,1};
    tmpFea_cur = CURINFOR.fea{ss,1};
    regionFea_pre = [tmpFea_pre.colorHist_rgb,tmpFea_pre.colorHist_lab,tmpFea_pre.colorHist_hsv,...
               tmpFea_pre.lbp_top_Hist,tmpFea_pre.regionCov,tmpFea_pre.LM_textureHist,tmpFea_pre.flowHist];
    regionFea_cur = [tmpFea_cur.colorHist_rgb,tmpFea_cur.colorHist_lab,tmpFea_cur.colorHist_hsv,...
               tmpFea_cur.lbp_top_Hist,tmpFea_cur.regionCov,tmpFea_cur.LM_textureHist,tmpFea_cur.flowHist];
    regionFea = [regionFea_pre;regionFea_cur];    
    [feas,regionFea_mapping] = pca(regionFea,no_dims);
%     d1_pre  = [tmpFea_pre.colorHist_rgb]; d1_cur  = [tmpFea_cur.colorHist_rgb];
%     d2_pre  = [tmpFea_pre.colorHist_lab]; d2_cur  = [tmpFea_cur.colorHist_lab];
%     d3_pre  = [tmpFea_pre.colorHist_hsv]; d3_cur  = [tmpFea_cur.colorHist_hsv];
%     d4_pre  = [tmpFea_pre.lbp_top_Hist];  d4_cur  = [tmpFea_cur.lbp_top_Hist];
%     d5_pre  = [tmpFea_pre.regionCov];     d5_cur  = [tmpFea_cur.regionCov];
%     d6_pre  = [tmpFea_pre.LM_textureHist];d6_cur  = [tmpFea_cur.LM_textureHist];
%     d7_pre  = [tmpFea_pre.flowHist];      d7_cur = [tmpFea_cur.flowHist];
    clear tmpFea_pre tmpFea_cur
   
    % 映射信息
    tmp_MAPSET = tmpSPcur.mapsets;% correSets correSets_dist

    % preparation
    tmpsalpre = normalizeSal(PREINFOR.spsal{ss,1});% 前一帧ss 尺度下各区域的显著性值(此处很重要！！！)
%     tmpsalcur0 = normalizeSal(TPSAL1{ss,1});% 单尺度下当前帧的各区域显著性值
    tmpsalcur0 = zeros(tmpSPcur.spNum,1);
%% 2 构造图模型连接前后帧 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
init_SALS = [tmpsalpre;tmpsalcur0];
nodeNum = tmpSPpre.spNum + tmpSPcur.spNum;
AdjcMatrix_nn = zeros(nodeNum);

% 2.1 PRE ----------------------------------------------------------------------
adjcMatrix_pre = tmpSPpre.adjcMatrix; 
bdIds_pre      = tmpSPpre.bdIds;
adjcMatrix_pre_nn = LinkNNAndBoundary2(adjcMatrix_pre, bdIds_pre);

% 2.2 CUR ----------------------------------------------------------------------  
adjcMatrix_cur = tmpSPcur.adjcMatrix; 
bdIds_cur      = tmpSPcur.bdIds;
adjcMatrix_cur_nn = LinkNNAndBoundary2(adjcMatrix_cur, bdIds_cur);

% 2.3 intersection  ----------------------------------------------------------
% 当前帧与前一帧的映射关系
adjMatric_pre_cur = zeros(tmpSPcur.spNum,tmpSPpre.spNum);
for kk=1:tmpSPcur.spNum
    cur2preNeightbor = tmp_MAPSET{kk,1}.correSets;
    adjMatric_pre_cur(kk,cur2preNeightbor) = 1;
end
% % 二者边界互联
% for pp=1:length(bdIds_cur)
%     for cc=1:length(bdIds_pre)
%         adjMatric_pre_cur(bdIds_cur(pp),bdIds_pre(cc)) = 1;
%     end
% end

% inter-layer graph
AdjcMatrix_nn(1:tmpSPpre.spNum,1:tmpSPpre.spNum)         = adjcMatrix_pre_nn;
AdjcMatrix_nn(tmpSPpre.spNum+1:end,tmpSPpre.spNum+1:end) = adjcMatrix_cur_nn;
AdjcMatrix_nn(1:tmpSPpre.spNum,tmpSPpre.spNum+1:end)     = adjMatric_pre_cur';
AdjcMatrix_nn(tmpSPpre.spNum+1:end,1:tmpSPpre.spNum)     = adjMatric_pre_cur;
AdjcMatrix_nn(1:nodeNum+1:end)                           = 0; 

clear adjcMatrix_pre bdIds_pre adjcMatrix_cur bdIds_cur
clear adjcMatrix_pre_nn adjcMatrix_cur_nn adjMatric_pre_cur
%% 3 integration & propagation &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
if 1
alpha=0.99;
theta=10;
tmpsalcur = zeros(nodeNum,7);
% www       = zeros(nodeNum,nodeNum,7);
% diffs     = [];
%  for ii=1:7
%      dcur = eval(['d' num2str(ii) '_cur']);
%      dpre = eval(['d' num2str(ii) '_pre']);
%      fea = [dpre;dcur];% nodeNum*feaDim
     colDistM = GetDistanceMatrix(feas);
     
     W = SetSmoothnessMatrix(colDistM, AdjcMatrix_nn, theta);
     % The smoothness setting is also different from that in Saliency
     % Optimization, where exp(-d^2/(2*sigma^2)) is used
     
     D = diag(sum(W));
     optAff =(D-alpha*W)\eye(nodeNum);
     optAff(1:nodeNum+1:end) = 0;%set diagonal elements to be zero
     for iter=1:1
     tmpsal = optAff*init_SALS;
%      tmpsalcur(:,ii) = tmpsal;
     [fea_Img_cur, ~] = CreateImageFromSPs(tmpsal(tmpSPpre.spNum+1:end), tmpSPcur.pixelList, height, width, true);
     [fea_Img_pre, ~] = CreateImageFromSPs(tmpsal(1:tmpSPpre.spNum), tmpSPpre.pixelList, height, width, true);
     figure,
     subplot(1,2,1),imshow(fea_Img_cur,[]),title('cur')
     subplot(1,2,2),imshow(fea_Img_pre,[]),title(['pre'])
     tmpsalcur0 = normalizeSal(tmpsal(tmpSPpre.spNum+1:end));
     init_SALS = [tmpsalpre;tmpsalcur0];
     end
%      www(:,:,ii) = optAff;
%      
%      % 计算同imsal_pre的diff
%      tmpdiff       = fea_Img_cur - imSal_pre0;
%      tmpdiff       = sum(tmpdiff(:).*tmpdiff(:))/length(tmpdiff(:));% 平均平方误差
%      diffs         = [diffs,tmpdiff];
%      
%      clear W D optAff colDistM dcur dpre fea
%  end
 
%     diffs = exp(-2*diffs./(mean(diffs)+eps));
%     WS    = diffs./(sum(diffs)+eps);
%    [valueDiff,indexDiff] = min(diffs);
%    indexWorse = indexDiff(end);% 最差特征图的编号！！！
end

if 0
colDistM = GetDistanceMatrix(feas);
cur_pre_affinity = colDistM(tmpSPpre.spNum+1:end,1:tmpSPpre.spNum);% curNum*preNum
theta = 2/mean(cur_pre_affinity(:));
WW = exp(-theta*cur_pre_affinity);
tmpsalcur0 = WW*tmpsalpre;
end



%% 4 integration & save &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
tmpsalcur1 =tmpsalcur0;
%    tmpsalcur1 = tmpsal(tmpSPpre.spNum+1:end);
   %tmpsalcur1 = tmpsalcur(tmpSPpre.spNum+1:end,:);clear tmpsalcur
%    tmpsalcur1(:,indexWorse) = 0;% 最差显著性图置零！！！
%    tmpsalcur1 = tmpsalcur1.*repmat(WS,[size(tmpsalcur1,1),1]);%weights'
%    tmpsalcur1 = sum(tmpsalcur1,2);
   tmpsalcur1 = normalizeSal(tmpsalcur1);% 归一化
    
    [PP_Img, ~] = CreateImageFromSPs(tmpsalcur1, tmpSPcur.pixelList, height, width, true);
    figure,imshow(PP_Img,[]),title('PP result')

    [rcenter_PP,ccenter_PP] = computeObjectCenter(PP_Img);
    regionDist_PP           = computeRegion2CenterDist(regionCenter,[rcenter_PP,ccenter_PP],[height,width]);
    PP_compactness          = computeCompactness(tmpsalcur1,regionDist_PP);
    SALS{ss,1}.SalValue     = tmpsalcur1;clear tmpsalcur1
    SALS{ss,1}.compactness  = 1/(PP_compactness);clear PP_compactness
    SALS{ss,1}.PP_Img       = PP_Img; clear PP_Img
%     SALS{ss,1}.WS           = WS;
    clear  tmp_MAPSET tmpFullresult tmpsalpre tmpsalcur1
    clear d1_pre d2_pre d3_pre d4_pre d5_pre d6_pre d7_pre 
    clear d1_cur d2_cur d3_cur d4_cur d5_cur d6_cur d7_cur
end

clear CURINFOR PREINFOR beta


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function adjcMatrix = LinkNNAndBoundary2(adjcMatrix, bdIds)
%link boundary SPs
adjcMatrix(bdIds, bdIds) = 1;


%link neighbor's neighbor
adjcMatrix = (adjcMatrix * adjcMatrix + adjcMatrix) > 0;
adjcMatrix = double(adjcMatrix);

spNum = size(adjcMatrix, 1);
adjcMatrix(1:spNum+1:end) = 0;  %diagnal elements set to be zero
end

function W = SetSmoothnessMatrix(colDistM, adjcMatrix_nn, theta)
allDists = colDistM(adjcMatrix_nn > 0);
maxVal = max(allDists);
minVal = min(allDists);

colDistM(adjcMatrix_nn == 0) = Inf;
colDistM = (colDistM - minVal) / (maxVal - minVal + eps);
W = exp(-colDistM * theta);
end