% function [TPSPSAL,TPSPSALRegionSal] = spatialPropagationNew10(CURINFOR,IMSAL_TPSAL1,param,cur_image,GPsign)
% function [TPSPSAL_Img,TPSPSAL_RegionSal] = spatialPropagationNew10(CURINFOR,IMSAL_TPSAL1,TPSAL1,param,cur_image,GPsign)
function [TPSPSAL_Img,TPSPSAL_RegionSal,TPSPSAL_compactness] = ...
    spatialPropagationNew10(CURINFOR,IMSAL_TPSAL1,TPSAL1,...
                            param,cur_image,GPsign,saveInfor,model)
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
% V8: 2016.11.18 7:51AM
% 利用model的mapping进行PCA
% 返回最终结果的的compactness值
% copyright by xiaofei zhou,shanghai university,shanghai,china
% zxforchid@163.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n this is spatial propagation process, wait a minute .........')
% no_dims = param.no_dims;
% bgRatio = param.bgRatio;
sp_iternum = param.sp_iternum;
mapping      = model{1,1}.mapping;

    
[r,c] = size(IMSAL_TPSAL1);
% iterSal = IMSAL_TPSAL1;
% ss = 1;% 仅仅一个尺度
% imwrite(IMSAL1,[saliencyMapPATH,salNAME,'_IMSAL_BOOSTSALS1.png']) 
%% A: 开讯迭代 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
iterSal_Img = 0;
for iter=1:sp_iternum
    fprintf('\n the %d iteration ......',iter)
for ss = 1:length(TPSAL1)
    iterSal = TPSAL1{ss,1}.SalValue;
    
    fprintf('\n the %d scale ......',ss)
    tmpFEA       = CURINFOR.fea{ss,1};
    tmpSPinfor   = CURINFOR.spinfor{ss,1};% 单尺度下的分割结果 
    pixelList    = tmpSPinfor.pixelList;
    regionCenter = tmpSPinfor.region_center;
    spNum        = tmpSPinfor.spNum;
    
   %% 1 根据融合后的时域传播图像，计算物体重心 &&&&&&&&&&&&&&&&&&&&&&&&
   fprintf('\n obtain object-center ...\n')
%    [rcenter,ccenter] = computeObjectCenter(iterSal_Img);% x-->row, y-->col  
%     regionDist = computeRegion2CenterDist(regionCenter,[rcenter,ccenter],[r,c]);
%     init_compactness = computeCompactness(iterSal,regionDist);
%     init_compactness = 1/(init_compactness+eps);
    init_compactness = TPSAL1{ss,1}.compactness;
    
    %% 2 形成大的特征矩阵 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    fprintf('\n compute features ...\n')
    regionFea = [tmpFEA.colorHist_rgb,tmpFEA.colorHist_lab,tmpFEA.colorHist_hsv,...
               tmpFEA.lbp_top_Hist,tmpFEA.regionCov,tmpFEA.LM_textureHist,tmpFEA.flowHist];
%     regionFea1 = bsxfun(@minus, regionFea, mapping.mean);
%     regionFea_mappedA = regionFea1*mapping.M;
    
    % regionFea_mappedA 为最终的区域特征
    [regionFea_mappedA,regionFea_mapping] = pca(regionFea,param.no_dims);
%     ZZ         = repmat(sqrt(sum(regionFea_mappedA.*regionFea_mappedA)),[tmpSPinfor.spNum,1]);% 特征全局归一化 2016.10.28 9:32AM
%     ZZ(ZZ==0)  = eps;
%     regionFea_mappedA  = regionFea_mappedA./ZZ;
%     FeaDist    = GetDistanceMatrix(regionFea_mappedA);    
    
    clear regionFea_mapping ZZ regionFea regionFea1
    
    if 1
    %% 3 local propagation &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    fprintf('\n local propagation ...\n')
     tmp_iterSal = iterSal;
%      threshLocal = graythresh(tmp_iterSal);  %automatic threshold
%      tmp_iterSal(tmp_iterSal < 0.05) = 0;
%      tmp_iterSal(tmp_iterSal > 1.5*threshLocal) = 1;
     
%      [valueBG,indexBG] = sort(tmp_iterSal);
%      BG_index = indexBG(1:BGNUM);
%      tmp_iterSal(BG_index) = 0;
     
    [LP_sal,LP_Img,LP_compactness] = ...
        localPropagation(regionFea_mappedA,tmp_iterSal,init_compactness,tmpSPinfor,[r,c]);
    clear tmp_iterSal valueBG indexBG BG_index  threshLocal
    imwrite(uint8(255*LP_Img),[saveInfor.saliencyMap,saveInfor.frame_name,'_LP_',num2str(iter),'_',num2str(ss),'.png']) 
    clear LP_Img
    end
    %% 4 global propagation &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if 1
    fprintf('\n global propagation ...\n')
     tmp_LP_sal = LP_sal;
%      threshGlobal = graythresh(tmp_LP_sal);  %automatic threshold
%      tmp_LP_sal(tmp_LP_sal < 0.05) = 0;
%      tmp_LP_sal(tmp_LP_sal > 1.5*threshGlobal) = 1;
     
%      [valueBG,indexBG] = sort(tmp_LP_sal);
%      BG_index = indexBG(1:BGNUM);
%      tmp_LP_sal(BG_index) = 0;
      
    [GP_sal,GP_Img,GP_compactness] = ...
        globalPropagation(regionFea_mappedA,tmp_LP_sal,LP_compactness,tmpSPinfor,[r,c]);    
     clear tmp_LP_sal valueBG indexBG BG_index threshGlobal
     imwrite(uint8(255*GP_Img),[saveInfor.saliencyMap,saveInfor.frame_name,'_LGP_',num2str(iter),'_',num2str(ss),'.png']) 
%      clear GP_Img
    end
     %% 4.1 MIDE LEVEL &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
     if 0
     tmp_GP_sal = GP_sal;
%      tmp_GP_sal = iterSal;
%      thresh3 = graythresh(tmp_GP_sal);  %automatic threshold
%      tmp_GP_sal(tmp_GP_sal < 0.05) = 0;
%      tmp_GP_sal(tmp_GP_sal > 1.5*thresh3) = 1; 
     
     [GP_sal,GP_Img,GP_compactness] = ...
        localPropagation(tmp_GP_sal,tmp_GP_sal,GP_compactness,tmpSPinfor,[r,c]);
     imwrite(uint8(255*GP_Img),[saveInfor.saliencyMap,saveInfor.frame_name,'_LGLP_',num2str(iter),'_',num2str(ss),'.png']) 
     end
    %% 5 save & clear &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    TPSAL1{ss,1}.SalValue = GP_sal;
    TPSAL1{ss,1}.PP_Img   = GP_Img;
    TPSAL1{ss,1}.compactness = GP_compactness;
%     iterSal = GP_sal;
%     iterSal_Img = GP_Img;
    iterSal_Img = iterSal_Img + GP_Img;
    
    clear LP_sal GP_sal GP_Img regionFea_mappedA regionFea_mapping 
end
%% 6 多尺度下的融合 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
iterSal_Img = normalizeSal(iterSal_Img);

end
%% B: 分配最终结果 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n assigenment the last result ...')
% [iterSal_Img, ~]  = CreateImageFromSPs(iterSal, tmpSPinfor.pixelList, r, c, true);
switch GPsign
    case 'YES'
         iterSal_Img = graphCut_Refine(cur_image,iterSal_Img); 
         TPSPSAL_Img     = iterSal_Img;
    case 'NO'
         TPSPSAL_Img     = iterSal_Img;  
end
% TPSPSAL_Img = normalizeSal(guidedfilter(TPSPSAL_Img,TPSPSAL_Img,5,0.1));
TPSPSAL_Img = normalizeSal(TPSPSAL_Img);   
[height,width] = size(TPSPSAL_Img);
[rcenter_sal,ccenter_sal] = computeObjectCenter(TPSPSAL_Img);


TPSPSAL_RegionSal = cell(length(CURINFOR.fea),1);% 各尺度下的结果
TPSPSAL_compactness = cell(length(CURINFOR.fea),1);% 各尺度下的结果
for ss=1:length(CURINFOR.fea)
    tmpSPinfor   = CURINFOR.spinfor{ss,1};
    pixelList    = tmpSPinfor.pixelList;
    regionCenter = tmpSPinfor.region_center;
    regionDist_sal = computeRegion2CenterDist(regionCenter,[rcenter_sal,ccenter_sal],[height,width]);
    
    TPSPSAL_RegionSal{ss,1} = ...
        computeRegionSal(TPSPSAL_Img,tmpSPinfor.pixelList);% 各尺度下的区域显著性值   
    
    Sal_compactness = computeCompactness(TPSPSAL_RegionSal{ss,1},regionDist_sal);
    TPSPSAL_compactness{ss,1} = 1/(Sal_compactness);clear Sal_compactness
    
    clear tmpSPinfor
end

clear CURINFOR IMSAL_TPSAL1 param cur_image GPsign

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 子函数区域  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2 根据初始融合结果，计算各尺度下的显著性值
function regionSal = computeRegionSal(refImage,pixelList)
regionSal = zeros(length(pixelList),1);

for i=1:length(pixelList)
    regionSal(i,1) = mean(refImage(pixelList{i,1}));
end
regionSal = normalizeSal(regionSal);

clear refImage pixelList
end

% 6. 全局传播(去除空间距离，因为特征中包含了位置信息) &&&&&&&&&&&&&&&&&&&&&&&&
% 去除自身 2016.11.09  13:35PM
% compactness的计算采用 yuming fang的做法， spatial variance 2016.11.15
% 
function [result_sal,result_Img,result_compactness] = ...
    globalPropagation(regionFea,LP_sal,LP_compactness,tmpSPinfor,imgsize)
r = imgsize(1);
c = imgsize(2);
spaSigma = 0.25;

% 6.1 propagate -----------------------------------------------------------
%    kdNum = size(tmpfea,1);
    knn=round(size(regionFea,1)*1/15);
    kdtree = vl_kdtreebuild(regionFea');% 输入 feaDim*sampleNum
    [indexs, distance] = vl_kdtreequery(kdtree,regionFea',regionFea', 'NumNeighbors', knn) ;
    distance1 = distance(2:end,:);% 舍弃第一行，自身尔；(knn-1)*sampleNum
    indexs1 = indexs(2:end,:);
    
    alpha = 1/mean(distance1(:));
    dist = exp(-alpha*distance1);
%     dist(dist>0.6) = 1;
    WIJ = dist./(repmat(sum(dist),[(knn-1),1])+eps);
    GP_sal = sum(LP_sal(indexs1).*WIJ);
    GP_sal = normalizeSal(GP_sal);
    GP_sal = GP_sal';

% 6.2 fusion --------------------------------------------------------------
    [GP_Img, ~]  = CreateImageFromSPs(GP_sal, tmpSPinfor.pixelList, r, c, true);
    [rcenter_GP,ccenter_GP] = computeObjectCenter(GP_Img);
    regionCenter = tmpSPinfor.region_center;
    regionDist_GP = ...
        computeRegion2CenterDist(regionCenter,[rcenter_GP,ccenter_GP],[r,c]);
    GP_compactness = computeCompactness(GP_sal,regionDist_GP);
    GP_compactness = 1/(GP_compactness);
    wGP   = GP_compactness/(GP_compactness + LP_compactness);
    wLP   = LP_compactness/(GP_compactness + LP_compactness);

% 6.3 RESULT --------------------------------------------------------------
result_sal = normalizeSal(wGP*GP_sal + wLP*LP_sal);
%     result_sal = ...
%         normalizeSal(GP_compactness*GP_sal + LP_compactness*LP_sal + ...
%         0.5*(GP_compactness + LP_compactness)*(LP_sal.*GP_sal));
[result_Img, ~]  = CreateImageFromSPs(result_sal, tmpSPinfor.pixelList, r, c, true);  
    [rcenter_result,ccenter_result] = computeObjectCenter(result_Img);
    result_regionDist = ...
        computeRegion2CenterDist(regionCenter,[rcenter_result,ccenter_result],[r,c]);
    result_compactness = computeCompactness(result_sal,result_regionDist);
    result_compactness = 1/(result_compactness);
    
    clear GP_Img GP_sal regionDist_GP GP_compactness
    
clear meanDist alpha WIJ
clear indexs distance indexs1 distance1
clear LP_sal regionFea kdtree
end

% 7 局部传播 2016.11.09  13:42PM &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% compactness的计算采用 yuming fang的做法， spatial variance 2016.11.15
% 
function [result_sal,result_Img,result_compactness] = ...
    localPropagation(regionFea,regionSal,init_compactness,tmpSPinfor,imgsize)
% 7.1 initial -------------------------------------------------------------
adjcMatrix = tmpSPinfor.adjcMatrix;
spNum = size(adjcMatrix,1);
r = imgsize(1);
c = imgsize(2);

    adjcMatrix1 = adjcMatrix;
    adjcMatrix1(adjcMatrix1==2) = 1;
    adjcMatrix1(1:spNum+1:end) = 0;
    adjmat = full(adjcMatrix1); % 仅仅是邻域   
    clear adjcMatrix1 adjcMatrix

% 7.2 propagate -----------------------------------------------------------
    LP_Sal = zeros(spNum,1);
    for ii=1:spNum
        tmpAdj = adjmat(ii,:);
        adjIndex = find(tmpAdj==1);
        
        tmpFea = regionFea(ii,:);
        tmpFea_adj = regionFea(adjIndex,:);
        feadiff = repmat(tmpFea,[length(adjIndex),1]) - tmpFea_adj;
        feadiff = sqrt(sum(feadiff.*feadiff,2));% size(adjsetfea,1)*1
        alpha_fea = 2/(mean(feadiff(:))+eps);
        feadiff = exp(-alpha_fea*feadiff);
%         feadiff(feadiff>0.6) = 1;
        
        SAL_adj = regionSal(adjIndex,:);
        wij = feadiff/(sum(feadiff(:))+eps);
        
        LP_Sal(ii,1) = sum(wij.*SAL_adj);
        
        clear SAL_adj wij tmpFea_adj feadiff
    end
    LP_Sal  = normalizeSal(LP_Sal);
% 7.3 fusion & compute the compactness of result sal ----------------------
    [LP_Img, ~]  = CreateImageFromSPs(LP_Sal, tmpSPinfor.pixelList, r, c, true);
    [rcenter_LP,ccenter_LP] = computeObjectCenter(LP_Img);
    regionCenter = tmpSPinfor.region_center;
    regionDist_LP = ...
        computeRegion2CenterDist(regionCenter,[rcenter_LP,ccenter_LP],[r,c]);
    LP_compactness = computeCompactness(LP_Sal,regionDist_LP);
    LP_compactness = 1/(LP_compactness);% note!!!
    wlp = LP_compactness/(LP_compactness+init_compactness);
    winit = init_compactness/(LP_compactness+init_compactness);
    result_sal = normalizeSal(wlp*LP_Sal + winit*regionSal);
    
    [result_Img, ~]  = CreateImageFromSPs(result_sal, tmpSPinfor.pixelList, r, c, true);
    [rcenter_result,ccenter_result] = computeObjectCenter(result_Img);
    result_regionDist = ...
        computeRegion2CenterDist(regionCenter,[rcenter_result,ccenter_result],[r,c]);
    result_compactness = computeCompactness(result_sal,result_regionDist);
    result_compactness = 1/(result_compactness);
    
    clear result_regionDist
    clear adjcMatrix regionFea regionSal init_compactness tmpSPinfor imgsize

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
