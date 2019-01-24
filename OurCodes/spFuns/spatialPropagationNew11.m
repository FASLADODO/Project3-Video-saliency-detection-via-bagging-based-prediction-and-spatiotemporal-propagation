% function [TPSPSAL,TPSPSALRegionSal] = spatialPropagationNew10(CURINFOR,IMSAL_TPSAL1,param,cur_image,GPsign)
function [TPSPSAL_Img,TPSPSAL_RegionSal] = spatialPropagationNew11(CURINFOR,IMSAL_TPSAL1,TPSAL1,param,cur_image,GPsign)
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
% 
% V7: 2016.11.14 16:27PM
% 仅做以此传播，无需迭代
% TPSAL1 于时域传播后的各尺度下的区域显著性图上做传播
% 
% V8: 2016.11.15 16:43 PM
% GMR（局部）+ SOP（全局）+ GMR（局部），局部/全局在于W构成之不同
% 
% copyright by xiaofei zhou,shanghai university,shanghai,china
% zxforchid@163.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n this is spatial propagation process, wait a minute .........')
no_dims = param.no_dims;
bgRatio = param.bgRatio;
fgRatio = param.fgRatio;
sp_iternum = param.sp_iternum;
alpha=0.99;
theta=10;

[r,c] = size(IMSAL_TPSAL1);
% iterSal = IMSAL_TPSAL1;
ss = 1;% 仅仅一个尺度

%% A: 开讯迭代 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
iterSal = TPSAL1{1,1};
iterSal_Img = IMSAL_TPSAL1;
for iter = 1:sp_iternum
    fprintf('\n the %d iteration ......',iter)
    tmpFEA       = CURINFOR.fea{ss,1};
    tmpSPinfor   = CURINFOR.spinfor{ss,1};% 单尺度下的分割结果 
    regionCenter = tmpSPinfor.region_center;
    adjcMatrix   = tmpSPinfor.adjcMatrix;
    bdIds        = tmpSPinfor.bdIds;
    spNum        = tmpSPinfor.spNum;
    
   %% 1 根据融合后的时域传播图像，计算物体重心 &&&&&&&&&&&&&&&&&&&&&&&&
    fprintf('\n obtain object-center ...\n')
    [rcenter,ccenter] = computeObjectCenter(iterSal_Img);% x-->row, y-->col  
    regionSal  = iterSal;
    regionDist = computeRegion2CenterDist(regionCenter,[rcenter,ccenter],[r,c]);
    init_compactness = computeCompactness(regionSal,regionDist);
    init_compactness = 1/(init_compactness);
    clear regionSal regionDist
    %% 1.1 计算位置信息 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    %% 2 形成大的特征矩阵 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    fprintf('\n compute features ...\n')
    regionFea = [tmpFEA.colorHist_rgb,tmpFEA.colorHist_lab,tmpFEA.colorHist_hsv,...
               tmpFEA.lbp_top_Hist,tmpFEA.regionCov,tmpFEA.LM_textureHist,tmpFEA.flowHist];
%     regionFea = [tmpFEA.colorHist_rgb,tmpFEA.colorHist_lab,tmpFEA.colorHist_hsv];
           
    % regionFea_mappedA 为最终的区域特征
%     regionFea_mappedA = regionFea;
    [regionFea_mappedA,regionFea_mapping] = pca(regionFea,no_dims);
%     ZZ         = repmat(sqrt(sum(regionFea_mappedA.*regionFea_mappedA)),[tmpSPinfor.spNum,1]);% 特征全局归一化 2016.10.28 9:32AM
%     ZZ(ZZ==0)  = eps;
%     regionFea_mappedA  = regionFea_mappedA./ZZ;
    FeaDist = GetDistanceMatrix(regionFea_mappedA);    
    
    clear regionFea_mapping ZZ regionFea 
    
    %% 3 contruct affinity matrix &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    fprintf('\n contruct affinity matrix ...\n')
    % 3.1 LOCAL ---------------------------------------------------------
    adjcMatrix_local = LinkNNAndBoundary2(adjcMatrix, bdIds);
%     FeaDist_location = GetDistanceMatrix(tmpFEA.flowHist);
%     W_location       = SetSmoothnessMatrix(FeaDist_location, adjcMatrix_local, theta);

    W_local          = SetSmoothnessMatrix(FeaDist, adjcMatrix_local, theta);
%     W_local          = W_location.*W_local;
    D_local          = diag(sum(W_local));
    optAff_local     = (D_local-alpha*W_local)\eye(spNum);
    optAff_local(1:spNum+1:end) = 0;% 关联阵对角线元素置零！！！
    
    % 3.2 GLOBAL --------------------------------------------------------
    knn=round(size(regionFea_mappedA,1)*1/15);mu=0.1;
    kdtree = vl_kdtreebuild(regionFea_mappedA');% 输入 feaDim*sampleNum
    [indexs, distance] = vl_kdtreequery(kdtree,regionFea_mappedA',regionFea_mappedA', 'NumNeighbors', knn) ;
    indexs1 = indexs(2:end,:);
    adjcMatrix_global = zeros(spNum,spNum);
    for i=1:spNum  
        tmp_nozeros = indexs1(:,i);
        adjcMatrix_global(i,tmp_nozeros) = 1;
        adjcMatrix_global(tmp_nozeros,i) = 1;
    end
    adjcMatrix_global(bdIds, bdIds) = 1;
    W_global  = SetSmoothnessMatrix(FeaDist, adjcMatrix_global, theta);
    W_global  = W_global + adjcMatrix_global*mu;  %add regularization term
    D_global  = diag(sum(W_global));

    
    %% 4 GMR-LOCAL &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
     fprintf('\n GMR-Local ...\n')
     % 将确定性的背景样本置为0 
     BGNUM = round(bgRatio*spNum);% 确定性背景样本的比例！！！
     tmp_iterSal = iterSal;
%      thresh1 = graythresh(tmp_iterSal);  %automatic threshold
%      tmp_iterSal(tmp_iterSal < 0.05) = 0;
%      tmp_iterSal(tmp_iterSal > 1.5*thresh1) = 1;

%      [valueBG,indexBG] = sort(tmp_iterSal);
%      BG_index = indexBG(1:BGNUM);
%      tmp_iterSal(BG_index) = 0;
     
     GMR_Local1 = optAff_local*tmp_iterSal;
     GMR_Local1 = normalizeSal(GMR_Local1);
     clear tmp_iterSal thresh1
     
     if 1
    %% 5 SOP-GLOBAL &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
     fprintf('\n SOP-Global ...\n')
     fgWeight = GMR_Local1;
     bgWeight = 1 - GMR_Local1;
     
     tmp_bgWeight = bgWeight;
     thresh2_1 = graythresh(tmp_bgWeight);  %automatic threshold
%      tmp_bgWeight(tmp_bgWeight < 0.05)          = 0;
     tmp_bgWeight(tmp_bgWeight > 1.5*thresh2_1) = 1000;
%      [valueSOP,indexSOP] = sort(tmp_bgWeight,'descend');
%      BG_index = indexSOP(1:BGNUM);
%      tmp_bgWeight(BG_index) = 1000;
     
     tmp_fgWeight = fgWeight;
%      thresh2_2 = graythresh(fgWeight);  %automatic threshold
     tmp_fgWeight(tmp_fgWeight < 0.05) = 0;
%      tmp_fgWeight(tmp_fgWeight > 1.5*thresh2_2) = 1;
     clear thresh2_1 thresh2_2
     
     bgLambda = 5;
     E_bg = diag(tmp_bgWeight * bgLambda);
     E_fg = diag(tmp_fgWeight);
     SOP_Global =(D_global - W_global + E_bg + E_fg) \ (E_fg * ones(spNum, 1));
     SOP_Global = normalizeSal(SOP_Global);
     clear GMR_Local1 E_bg E_fg tmp_fgWeight fgWeight bgWeight
     end
     
    %% 6 GMR-LOCAL &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
     fprintf('\n GMR-Local ...\n')
     % 6.1 将确定性的前景样本置为1 ---------------------------------
     FGNUM = round(fgRatio*spNum);% 确定性前景样本的比例！！！
%      SOP_Global = GMR_Local1;
     tmp_SOP_Global = SOP_Global;
%      thresh3 = graythresh(tmp_SOP_Global);  %automatic threshold
%      tmp_SOP_Global(tmp_SOP_Global < 0.05)        = 0;
%      tmp_SOP_Global(tmp_SOP_Global > 1.5*thresh3) = 1;

%      [valueFG,indexFG] = sort(tmp_SOP_Global,'descend');
%      FG_index = indexFG(1:FGNUM);
%      tmp_SOP_Global(FG_index) = 1;
     clear thresh3
     
     % 6.2 使用mid-level feature构建关联阵 -------------------------
    FeaDistNew      = GetDistanceMatrix(tmp_SOP_Global); 
    W_localNew      = SetSmoothnessMatrix(FeaDistNew, adjcMatrix_local, theta);
%     W_localNew      = W_location.*W_localNew;
    D_localNew      = diag(sum(W_localNew));
    optAff_localNew = (D_localNew-alpha*W_localNew)\eye(spNum);
    optAff_localNew(1:spNum+1:end) = 0;% 关联阵对角线元素置零！！！
    
    % 6.3 传播 -----------------------------------------------------
     GMR_Local2 = optAff_localNew*tmp_SOP_Global;
     GMR_Local2 = normalizeSal(GMR_Local2);
     clear SOP_Global tmp_SOP_Global
     
    %% 7 fusion &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
     fprintf('\n integration initial and propagation ...\n')
    [PP_Img, ~]  = CreateImageFromSPs(GMR_Local2, tmpSPinfor.pixelList, r, c, true);
    [rcenter_PP,ccenter_PP] = computeObjectCenter(PP_Img);
    regionDist_PP = ...
        computeRegion2CenterDist(regionCenter,[rcenter_PP,ccenter_PP],[r,c]);
    PP_compactness = computeCompactness(GMR_Local2,regionDist_PP);
    PP_compactness = 1/(PP_compactness);
    
    winit = init_compactness/(init_compactness + PP_compactness);
    wpp   = PP_compactness  /(init_compactness + PP_compactness);
    
    tmp_iterSal = normalizeSal(winit*iterSal + wpp*GMR_Local2);
    [tmp_iterSal_Img, ~]  = CreateImageFromSPs(tmp_iterSal, tmpSPinfor.pixelList, r, c, true);
    clear PP_Img rcenter_PP ccenter_PP PP_compactness winit wpp
    
    %% 5 save & clear
    iterSal     = tmp_iterSal;
    iterSal_Img = tmp_iterSal_Img;
    
    clear tmp_iterSal tmp_iterSal_Img 
    clear LP_sal GP_sal GP_Img regionFea_mappedA regionFea_mapping 
end

%% B: 分配最终结果 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n assigenment the last result ...')
% [iterSal_Img, ~]  = CreateImageFromSPs(iterSal, tmpSPinfor.pixelList, r, c, true);
switch GPsign
    case 'YES'
         iterSal_Img     = graphCut_Refine(cur_image,iterSal_Img); 
         TPSPSAL_Img     = iterSal_Img;
    case 'NO'
         TPSPSAL_Img     = iterSal_Img;  
end
TPSPSAL_Img = normalizeSal(guidedfilter(TPSPSAL_Img,TPSPSAL_Img,5,0.1));

TPSPSAL_RegionSal = cell(length(CURINFOR.fea),1);% 各尺度下的结果
for ss=1:length(CURINFOR.fea)
    tmpSPinfor   = CURINFOR.spinfor{ss,1};
    TPSPSAL_RegionSal{ss,1} = ...
        computeRegionSal(TPSPSAL_Img,tmpSPinfor.pixelList);% 各尺度下的区域显著性值
    clear tmpSPinfor
end

clear CURINFOR IMSAL_TPSAL1 param cur_image GPsign

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 子函数区域  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 2-hop & bb &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% 调整了下顺序，先2-hop， 然后再四周边界连接 2016.11.15 18:58PM
% 再反过来，先连边界，再 2-hop;同 zhuwangjaing保持一致！！！ 2016.11.16 10:31AM
function adjcMatrix = LinkNNAndBoundary2(adjcMatrix, bdIds)
%link boundary SPs
adjcMatrix(bdIds, bdIds) = 1;

%link neighbor's neighbor
adjcMatrix = (adjcMatrix * adjcMatrix + adjcMatrix) > 0;
adjcMatrix = double(adjcMatrix);

spNum = size(adjcMatrix, 1);
adjcMatrix(1:spNum+1:end) = 0;  %diagnal elements set to be zero
clear bdIds
end

% 2 关联阵 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
function W = SetSmoothnessMatrix(colDistM, adjcMatrix_nn, theta)
allDists = colDistM(adjcMatrix_nn > 0);
maxVal = max(allDists);
minVal = min(allDists);

colDistM(adjcMatrix_nn == 0) = Inf;
colDistM = (colDistM - minVal) / (maxVal - minVal + eps);% 距离归一化
W = exp(-colDistM * theta);
clear colDistM adjcMatrix_nn theta
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 6. 全局传播(去除空间距离，因为特征中包含了位置信息) &&&&&&&&&&&&&&&&&&&&&&&&
% % 去除自身 2016.11.09  13:35PM
% % compactness的计算采用 yuming fang的做法， spatial variance 2016.11.15
% % 
% function [result_sal,result_Img] = ...
%     globalPropagation(regionFea,LP_sal,LP_compactness,tmpSPinfor,imgsize)
% r = imgsize(1);
% c = imgsize(2);
% spaSigma = 0.25;
% 
% % 6.1 propagate -----------------------------------------------------------
% %    kdNum = size(tmpfea,1);
%     knn=round(size(regionFea,1)*1/15);
%     kdtree = vl_kdtreebuild(regionFea');% 输入 feaDim*sampleNum
%     [indexs, distance] = vl_kdtreequery(kdtree,regionFea',regionFea', 'NumNeighbors', knn) ;
%     distance1 = distance(2:end,:);% 舍弃第一行，自身尔；(knn-1)*sampleNum
%     indexs1 = indexs(2:end,:);
%     
%     alpha = 1/mean(distance1(:));
%     dist = exp(-alpha*distance1);
%     WIJ = dist./(repmat(sum(dist),[(knn-1),1])+eps);
%     GP_sal = sum(LP_sal(indexs1).*WIJ);
%     GP_sal = normalizeSal(GP_sal);
%     GP_sal = GP_sal';
% 
% % 6.2 fusion --------------------------------------------------------------
%     [GP_Img, ~]  = CreateImageFromSPs(GP_sal, tmpSPinfor.pixelList, r, c, true);
%     [rcenter_GP,ccenter_GP] = computeObjectCenter(GP_Img);
%     regionCenter = tmpSPinfor.region_center;
%     regionDist_GP = ...
%         computeRegion2CenterDist(regionCenter,[rcenter_GP,ccenter_GP],[r,c]);
%     GP_compactness = computeCompactness(GP_sal,regionDist_GP);
%     GP_compactness = 1/(GP_compactness+eps);
%     wGP   = GP_compactness/(GP_compactness + LP_compactness);
%     wLP   = LP_compactness/(GP_compactness + LP_compactness);
% 
% % 6.3 RESULT --------------------------------------------------------------
% result_sal = normalizeSal(wGP*GP_sal + wLP*LP_sal);
% %     result_sal = ...
% %         normalizeSal(GP_compactness*GP_sal + LP_compactness*LP_sal + ...
% %         0.5*(GP_compactness + LP_compactness)*(LP_sal.*GP_sal));
% [result_Img, ~]  = CreateImageFromSPs(result_sal, tmpSPinfor.pixelList, r, c, true);  
%     clear GP_Img GP_sal regionDist_GP GP_compactness
%     
% clear meanDist alpha WIJ
% clear indexs distance indexs1 distance1
% clear LP_sal regionFea kdtree
% end
% 
% % 7 局部传播 2016.11.09  13:42PM &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% % compactness的计算采用 yuming fang的做法， spatial variance 2016.11.15
% % 
% function [result_sal,result_compactness] = ...
%     localPropagation(regionFea,regionSal,init_compactness,tmpSPinfor,imgsize)
% % 7.1 initial -------------------------------------------------------------
% adjcMatrix = tmpSPinfor.adjcMatrix;
% spNum = size(adjcMatrix,1);
% r = imgsize(1);
% c = imgsize(2);
% 
%     adjcMatrix1 = adjcMatrix;
%     adjcMatrix1(adjcMatrix1==2) = 1;
%     adjcMatrix1(1:spNum+1:end) = 0;
%     adjmat = full(adjcMatrix1); % 仅仅是邻域   
%     clear adjcMatrix1 adjcMatrix
% 
% % 7.2 propagate -----------------------------------------------------------
%     LP_Sal = zeros(spNum,1);
%     for ii=1:spNum
%         tmpAdj = adjmat(ii,:);
%         adjIndex = find(tmpAdj==1);
%         
%         tmpFea = regionFea(ii,:);
%         tmpFea_adj = regionFea(adjIndex,:);
%         feadiff = repmat(tmpFea,[length(adjIndex),1]) - tmpFea_adj;
%         feadiff = sqrt(sum(feadiff.*feadiff,2));% size(adjsetfea,1)*1
%         alpha_fea = 2/(mean(feadiff(:))+eps);
%         feadiff = exp(-alpha_fea*feadiff);
%         
%         SAL_adj = regionSal(adjIndex,:);
%         wij = feadiff/(sum(feadiff(:))+eps);
%         LP_Sal(ii,1) = sum(wij.*SAL_adj);
%         
%         clear SAL_adj wij tmpFea_adj feadiff
%     end
%     LP_Sal  = normalizeSal(LP_Sal);
% % 7.3 fusion & compute the compactness of result sal ----------------------
%     [LP_Img, ~]  = CreateImageFromSPs(LP_Sal, tmpSPinfor.pixelList, r, c, true);
%     [rcenter_LP,ccenter_LP] = computeObjectCenter(LP_Img);
%     regionCenter = tmpSPinfor.region_center;
%     regionDist_LP = ...
%         computeRegion2CenterDist(regionCenter,[rcenter_LP,ccenter_LP],[r,c]);
%     LP_compactness = computeCompactness(LP_Sal,regionDist_LP);
%     LP_compactness = 1/(LP_compactness+eps);% note!!!
%     wlp = LP_compactness/(LP_compactness+init_compactness);
%     winit = init_compactness/(LP_compactness+init_compactness);
%     result_sal = normalizeSal(wlp*LP_Sal + winit*regionSal);
%     
%     [result_Img, ~]  = CreateImageFromSPs(result_sal, tmpSPinfor.pixelList, r, c, true);
%     [rcenter_result,ccenter_result] = computeObjectCenter(result_Img);
%     result_regionDist = ...
%         computeRegion2CenterDist(regionCenter,[rcenter_result,ccenter_result],[r,c]);
%     result_compactness = computeCompactness(result_sal,result_regionDist);
%     result_compactness = 1/(result_compactness+eps);
%     
%     clear result_Img result_regionDist
%     clear adjcMatrix regionFea regionSal init_compactness tmpSPinfor imgsize
% 
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
