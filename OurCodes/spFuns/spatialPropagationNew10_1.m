function [TPSPSAL,TPSPSALRegionSal] = spatialPropagationNew10_1(CURINFOR,IMSAL_TPSAL1,param,cur_image,GPsign)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 在时域传播的基础上进行空域传播
% CURINFOR
% fea/ORLabels/spinfor(mapsets，region_center_prediction)
% 
% spinfor{ss,1}.adjcMatrix;
% spinfor{ss,1}.colDistM 
% spinfor{ss,1}.clipVal 
% spinfor{ss,1}.idxcurrImage 
% spinfor{ss,1}.adjmat
% spinfor{ss,1}.pixelList 
% spinfor{ss,1}.area 
% spinfor{ss,1}.spNum 
% spinfor{ss,1}.bdIds 
% spinfor{ss,1}.posDistM 
% spinfor{ss,1}.region_center
% 
% FEA{ss,1}.colorHist_rgb 
% FEA{ss,1}.colorHist_lab 
% FEA{ss,1}.colorHist_hsv 
% FEA{ss,1}.lbpHist   
% FEA{ss,1}.hogHist  
% FEA{ss,1}.regionCov   
% FEA{ss,1}.geoDist    
% FEA{ss,1}.flowHist  
%
% TPSAL(全尺寸)
% 各尺度下各区域的显著性值
% 
% V1: 2016.10.14 20:01PM
% 仿照CVPR2016 GRAB思想进行传播优化
% 
% V2:2016.10.18 15:45PM
% 结合最新的 iterative propagation 进行修改
% 目前是背景传播+优化+前景传播；采用常规的图结构；无迭代
% 
% V3: 2016.10.19 16:18PM
% 引入迭代机制
% 
% V4： 2016.10.30 14：59PM
% object-biased + regression-based-{GMR --> SO --> GMR}
% 迭代的传播更新策略
% 采用常规图结构/最近邻
% 
% V5： 2016.11.02 21：03PM
% 新的空域传播方式
% 1) CURINFOR.fea  利用全部的特征，串接起来
%     fea{ss,1}.colorHist_rgb 
%     fea{ss,1}.colorHist_lab 
%     fea{ss,1}.colorHist_hsv 
%     fea{ss,1}.lbpHist     
%     fea{ss,1}.lbp_top_Hist 
%     fea{ss,1}.hogHist
%     fea{ss,1}.regionCov   
%     fea{ss,1}.geoDist    
%     fea{ss,1}.flowHist   
% 2) PCA           降维
% 
% [DB.P.colorHist_rgb_mappedA,DB.P.colorHist_rgb_mapping] = pca(D0.P.colorHist_rgb,no_dims);
% 3) 构建传播阵
% 
% V6: 2016.11.09 9:32am
% LOCAL-->GLOBAL + ITERATION
% 区域权重 * 全局权重， 局部到全局的传播
% copyright by xiaofei zhou,shanghai university,shanghai,china
% zxforchid@163.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n this is spatial propagation process, wait a minute .........')
no_dims = param.no_dims;
bgRatio = param.bgRatio;
sp_iternum = param.sp_iternum;

[r,c] = size(IMSAL_TPSAL1);
% iterSal = IMSAL_TPSAL1;
ss = 1;% 仅仅一个尺度

%% A: 开讯迭代 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
for iter = 1:sp_iternum
    fprintf('\n the %d iteration ......',iter)
    tmpFEA       = CURINFOR.fea{ss,1};
    tmpSPinfor   = CURINFOR.spinfor{ss,1};% 单尺度下的分割结果 
    pixelList    = tmpSPinfor.pixelList;
    regionCenter = tmpSPinfor.region_center;
    
    if iter==1
        iterSal  = computeRegionSal(IMSAL_TPSAL1,pixelList);
        clear IMSAL_TPSAL1
    end
   %% 1 根据融合后的时域传播图像，计算物体重心 &&&&&&&&&&&&&&&&&&&&&&&&
   fprintf('\n obtain object-center ...\n')
   [iterSal_Img, ~]  = CreateImageFromSPs(iterSal, pixelList, r, c, true);
   [rcenter,ccenter] = computeObjectCenter(iterSal_Img);% x-->row, y-->col  
   clear iterSal_Img
   
    regionSal  = iterSal;
%     regionSal  = computeRegionSal(iterSal,pixelList);% 各尺度下的区域显著性值
    regionDist = computeRegion2CenterDist(regionCenter,[rcenter,ccenter],[r,c]);
    init_compactness = sum(regionSal.*regionDist);
    clear regionSal regionDist
    
    %% 2 形成大的特征矩阵 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    fprintf('\n compute features ...\n')
    regionFea = [tmpFEA.colorHist_rgb,tmpFEA.colorHist_lab,tmpFEA.colorHist_hsv,...
               tmpFEA.lbp_top_Hist,tmpFEA.regionCov,tmpFEA.LM_textureHist,tmpFEA.flowHist];
%     regionFea = [tmpFEA.colorHist_rgb,tmpFEA.colorHist_lab,tmpFEA.colorHist_hsv,tmpFEA.LM_texture,...
%         tmpFEA.lbp_top_Hist,tmpFEA.hogHist,tmpFEA.regionCov,tmpFEA.LM_textureHist,tmpFEA.geoDist,tmpFEA.flowHist];

    % regionFea_mappedA 为最终的区域特征
    [regionFea_mappedA,regionFea_mapping] = pca(regionFea,no_dims);
%     ZZ         = repmat(sqrt(sum(regionFea_mappedA.*regionFea_mappedA)),[tmpSPinfor.spNum,1]);% 特征全局归一化 2016.10.28 9:32AM
%     ZZ(ZZ==0)  = eps;
%     regionFea_mappedA  = regionFea_mappedA./ZZ;
%     FeaDist    = GetDistanceMatrix(regionFea_mappedA);    
    
    clear regionFea_mapping ZZ regionFea 
    
    %% 3 local propagation &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    fprintf('\n local propagation ...\n')
    [LP_sal,LP_compactness] = ...
        localPropagation(regionFea_mappedA,iterSal,init_compactness,tmpSPinfor,[r,c]);
    
    %% 4 global propagation &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    fprintf('\n global propagation ...\n')
    GP_sal = globalPropagation(LP_sal,LP_compactness,regionFea_mappedA,tmpSPinfor,[r,c]);    

    %% 5 save & clear
    iterSal = GP_sal;
    clear LP_sal GP_sal regionFea_mappedA regionFea_mapping 
end

%% B: 分配最终结果 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n assigenment the last result ...')
[iterSal_img, ~]  = CreateImageFromSPs(iterSal, tmpSPinfor.pixelList, r, c, true);
switch GPsign
    case 'YES'
         iterSal_img = graphCut_Refine(cur_image,iterSal_img); 
         TPSPSAL     = iterSal_img;
    case 'NO'
         TPSPSAL     = iterSal_img;  
end
TPSPSAL = normalizeSal(guidedfilter(TPSPSAL,TPSPSAL,5,0.1));

TPSPSALRegionSal = cell(length(CURINFOR.fea),1);% 各尺度下的结果
for ss=1:length(CURINFOR.fea)
    tmpSPinfor   = CURINFOR.spinfor{ss,1};
    TPSPSALRegionSal{ss,1} = ...
        computeRegionSal(TPSPSAL,tmpSPinfor.pixelList);% 各尺度下的区域显著性值
    clear tmpSPinfor
end

clear CURINFOR IMSAL_TPSAL1 param cur_image GPsign

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 子函数区域  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 计算物体重心
function [xcenter,ycenter] = computeObjectCenter(refImage)
[r,c] = size(refImage);
row = 1:r;
row = row';
col = 1:c;
XX = repmat(row,1,c).*refImage;
YY = repmat(col,r,1).*refImage;
xcenter = sum(XX(:))/sum(refImage(:));
ycenter = sum(YY(:))/sum(refImage(:));
clear refImage
end

% 2 根据初始融合结果，计算各尺度下的显著性值
function regionSal = computeRegionSal(refImage,pixelList)
regionSal = zeros(length(pixelList),1);

for i=1:length(pixelList)
    regionSal(i,1) = mean(refImage(pixelList{i,1}));
end
regionSal = normalizeSal(regionSal);

clear refImage pixelList
end

% 3 计算各区域距离中心的距离 
function dist = computeRegion2CenterDist(regionCenter,objectCenter,imageSize)
sigmaRatio = 0.25;
rcenter = objectCenter(1);
ccenter = objectCenter(2); 
r = imageSize(1);
c = imageSize(2); 
dist = zeros(size(regionCenter,1),1);
sigma=[r*sigmaRatio c*sigmaRatio];

for i=1:length(regionCenter)
    tmpRegionCenter = regionCenter(i,:);
    xx = tmpRegionCenter(1);
    yy = tmpRegionCenter(2);
    dist(i,1) = exp(-(xx-rcenter)^2/(2*sigma(1)^2)-(yy-ccenter)^2/(2*sigma(2)^2));
end
clear regionCenter objectCenter imageSize
end

% 6. 全局传播(去除空间距离，因为特征中包含了位置信息) &&&&&&&&&&&&&&&&&&&&&&&&
% 去除自身 2016.11.09  13:35PM
function result_sal = ...
    globalPropagation(LP_sal,LP_compactness,regionFea,tmpSPinfor,imgsize)
r = imgsize(1);
c = imgsize(2);
spaSigma = 0.25;

% propagate ------------------
%    kdNum = size(tmpfea,1);
    knn=round(size(regionFea,1)*1/15);
    kdtree = vl_kdtreebuild(regionFea');% 输入 feaDim*sampleNum
    [indexs, distance] = vl_kdtreequery(kdtree,regionFea',regionFea', 'NumNeighbors', knn) ;
    distance1 = distance(2:end,:);% 舍弃第一行，自身尔；(knn-1)*sampleNum
    indexs1 = indexs(2:end,:);

%     meanDist = (repmat(mean(distance1),[(knn-1),1])+eps);
%     alpha = 2./meanDist;
    alpha = 1/mean(distance1(:));
    dist = exp(-alpha*distance1);
%     posWeight = Dist2WeightMatrix(tmpSPinfor.posDistM, spaSigma);
%     [spNum,~]=size(posWeight);
%      posWeight(1:spNum+1:end) = 0;
%      cor_posWeight=zeros(knn-1,spNum);
%      for k=1:spNum
%          cor_posWeight(:,k)=posWeight(indexs1(:,k),k);
%      end    
%      dist=dist.*cor_posWeight;
     WIJ = dist./(repmat(sum(dist),[(knn-1),1])+eps);
%     dist = distance1./(repmat(sum(distance1),[(knn-1),1])+eps);
%     result = sum(LP_sal(indexs1).*dist);
    GP_sal = sum(LP_sal(indexs1).*WIJ);
    GP_sal = normalizeSal(GP_sal);
    GP_sal = GP_sal';
    
    CC = 0.6*normalizeSal(1./max(dist))+0.2;
    CC = CC';
    
% fusion -----------------------------
    [GP_Img, ~]  = CreateImageFromSPs(GP_sal, tmpSPinfor.pixelList, r, c, true);
    [rcenter_GP,ccenter_GP] = computeObjectCenter(GP_Img);
    regionCenter = tmpSPinfor.region_center;
    regionDist_GP = ...
        computeRegion2CenterDist(regionCenter,[rcenter_GP,ccenter_GP],[r,c]);
    GP_compactness = sum(GP_sal.*regionDist_GP);
    wGP   = GP_compactness/(GP_compactness + LP_compactness);
    wLP   = LP_compactness/(GP_compactness + LP_compactness);

% RESULT ------------------------------
    WS = [wLP*CC,wGP*(1-CC)];
    WS = WS./repmat((sum(WS,2)+eps),[1,2]);
    result_sal = normalizeSal(WS(:,1).*LP_sal + WS(:,2).*GP_sal);
    
%     result_sal = normalizeSal(wLP*CC.*LP_sal + wGP*(1-CC).*GP_sal);
    
    clear GP_Img GP_sal regionDist_GP GP_compactness
    
clear meanDist alpha WIJ
clear indexs distance indexs1 distance1
clear LP_sal regionFea kdtree
end

% 7 局部传播 2016.11.09  13:42PM &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
function [result_sal,result_compactness] = ...
    localPropagation(regionFea,regionSal,init_compactness,tmpSPinfor,imgsize)
% initial ---------------------
adjcMatrix = tmpSPinfor.adjcMatrix;
spNum = size(adjcMatrix,1);
r = imgsize(1);
c = imgsize(2);
a = 0.6;b=0.2;
    adjcMatrix1 = adjcMatrix;
    adjcMatrix1(adjcMatrix1==2) = 1;
    adjcMatrix1(1:spNum+1:end) = 0;
    adjmat = full(adjcMatrix1); % 仅仅是邻域   
    clear adjcMatrix1 adjcMatrix

% propagate -------------------
    LP = zeros(spNum,1);
    CC = zeros(spNum,1);
    for ii=1:spNum
        tmpAdj = adjmat(ii,:);
        adjIndex = find(tmpAdj==1);
        
        tmpFea = regionFea(ii,:);
        tmpFea_adj = regionFea(adjIndex,:);
        feadiff = repmat(tmpFea,[length(adjIndex),1]) - tmpFea_adj;
        feadiff = sqrt(sum(feadiff.*feadiff,2));% size(adjsetfea,1)*1
        alpha_fea = 2/(mean(feadiff(:))+eps);
        feadiff = exp(-alpha_fea*feadiff);
        
        SAL_adj = regionSal(adjIndex,:);
        wij = feadiff/(sum(feadiff(:))+eps);
        LP(ii,1) = sum(wij.*SAL_adj);
        
        [maxValue,maxIndex] = max(feadiff);
        maxValue(maxValue==eps)=eps;
        CC(ii,1) = 1/maxValue;
        
        clear SAL_adj wij tmpFea_adj feadiff
    end
    LP  = normalizeSal(LP);
    CC = a*normalizeSal(CC) + b;% 区域权重
    
    
    % fusion -----------------------------
    [LP_Img, ~]  = CreateImageFromSPs(LP, tmpSPinfor.pixelList, r, c, true);
    [rcenter_LP,ccenter_LP] = computeObjectCenter(LP_Img);
    regionCenter = tmpSPinfor.region_center;
    regionDist_LP = ...
        computeRegion2CenterDist(regionCenter,[rcenter_LP,ccenter_LP],[r,c]);
    compactness_LP = sum(LP.*regionDist_LP);
    wlp = compactness_LP/(compactness_LP+init_compactness);
    winit = init_compactness/(compactness_LP+init_compactness);

    % RESULT ------------------------------
%     result_sal = normalizeSal(wlp*LP + winit*regionSal);
    WS = [winit*CC,wlp*(1-CC)];
    WS = WS./repmat((sum(WS,2)+eps),[1,2]);
    result_sal = normalizeSal(WS(:,1).*regionSal + WS(:,2).*LP);
%     result_sal = normalizeSal(winit*CC.*regionSal + wlp*(1-CC).*LP);
    
    [result_Img, ~]  = CreateImageFromSPs(result_sal, tmpSPinfor.pixelList, r, c, true);
    [rcenter_result,ccenter_result] = computeObjectCenter(result_Img);
    result_regionDist = ...
        computeRegion2CenterDist(regionCenter,[rcenter_result,ccenter_result],[r,c]);
    result_compactness = sum(result_sal.*result_regionDist);
    
    clear result_Img result_regionDist
    clear adjcMatrix regionFea regionSal init_compactness tmpSPinfor imgsize

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 4 关联阵
% function W = SetSmoothnessMatrix(colDistM, adjcMatrix_nn, theta)
% allDists = colDistM(adjcMatrix_nn > 0);
% maxVal = max(allDists);
% minVal = min(allDists);
% 
% colDistM(adjcMatrix_nn == 0) = Inf;
% colDistM = (colDistM - minVal) / (maxVal - minVal + eps);
% W = exp(-colDistM * theta);
% end
% 
% % 5 2-hop & bb
% function adjcMatrix = LinkNNAndBoundary2(adjcMatrix, bdIds)
% %link boundary SPs
% adjcMatrix(bdIds, bdIds) = 1;
% 
% %link neighbor's neighbor
% adjcMatrix = (adjcMatrix * adjcMatrix + adjcMatrix) > 0;
% adjcMatrix = double(adjcMatrix);
% 
% spNum = size(adjcMatrix, 1);
% adjcMatrix(1:spNum+1:end) = 0;  %diagnal elements set to be zero
% end
