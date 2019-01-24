function [TPSPSAL_Img,TPSPSAL_RegionSal] = spatialPropagationNew8_2(TPSAL,CURINFOR,image,flow)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMR+SOP+(GMR/optimal)
% 2016.11.21 21:49PM
% copyright by xiaofei zhou,shanghai university,shanghai,china
% zxforchid@163.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. 提取特征 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% im_R im_G im_B im_L im_A im_B1 im_H im_S im_V Magn Ori im_Y im_X
FEA = prepaFea(image,flow);
[height,width,dims] = size(image);

%% 2. begin %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPSCALENUM = length(CURINFOR.fea);
SALS = cell(SPSCALENUM,1);
alpha=0.99;
theta=10;
iterNum = 1;
TPSPSAL_Img = 0;
TPSPSAL_RegionSal = cell(SPSCALENUM,1);
SIGN_GMR = 1;% optimal 部分的GMR
for ss=1:SPSCALENUM
    fprintf('\n scale num %d .........................................',ss)
    fprintf('\n initialization ............\n')
%% 1 initial &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    tmpSPinfor = CURINFOR.spinfor{ss,1};% 单尺度下的分割结果 
    spNum      = tmpSPinfor.spNum;
    adjcMatrix = tmpSPinfor.adjcMatrix;
    bdIds      = tmpSPinfor.bdIds;
 
    regionSal  = TPSAL{ss,1}.SalValue;% 各区域的初始显著性值
    
%     tmpFEA       = CURINFOR.fea{ss,1};
%     regionFea = [tmpFEA.colorHist_rgb,tmpFEA.colorHist_lab,tmpFEA.colorHist_hsv , ...
%                  tmpFEA.lbp_top_Hist, tmpFEA.regionCov    ,tmpFEA.LM_textureHist, ...
%                  tmpFEA.flowHist];
             
    regionFea  = computeRegionFea(FEA,tmpSPinfor);% 各区域的特征
    ZZ         = repmat(sqrt(sum(regionFea.*regionFea)),[spNum,1]);% 特征全局归一化 2016.10.28 9:32AM
    ZZ(ZZ==0)  = eps;
    regionFea  = regionFea./ZZ;
%     [regionFea,regionFea_mapping] = pca(regionFea,0.995);
    
    FeaDist    = GetDistanceMatrix(regionFea);
    
    %% 1.1 构建局部+全局传播阵 （localW + globalW） &&&&&&&&&&&&&&&&&
    % localW
    adjcMatrix_local = LinkNNAndBoundary2(adjcMatrix, bdIds); 
    W_local          = SetSmoothnessMatrix(FeaDist, adjcMatrix_local, theta);
    D_local          = diag(sum(W_local));
    optAff_local     = (D_local-alpha*W_local)\eye(spNum);
    optAff_local(1:spNum+1:end) = 0;
    
%     % globalW
%     knn=round(size(regionFea,1)*1/4);
%     kdtree = vl_kdtreebuild(regionFea');% 输入 feaDim*sampleNum
%     [indexs, distance] = vl_kdtreequery(kdtree,regionFea',regionFea', 'NumNeighbors', knn) ;
%     indexs1 = indexs(2:end,:);% 每一列为一个区域对应的KNN邻域，令这些邻域为1，其他为0
%     adjcMatrix_global = zeros(spNum,spNum);
%     for i=1:spNum  
%         tmp_nozeros = indexs1(:,i);
%         adjcMatrix_global(i,tmp_nozeros) = 1;
%         adjcMatrix_global(tmp_nozeros,i) = 1;
%     end
%     W_global          = SetSmoothnessMatrix(FeaDist, adjcMatrix_global, theta);
%     D_global          = diag(sum(W_global));
%     optAff_global     = (D_global-alpha*W_global)\eye(spNum);
%     optAff_global(1:spNum+1:end) = 0;
% %     [~, ~, neiSigma_global] = EstimateDynamicParas(adjcMatrix_global, FeaDist);
    
%% 2 iterative spatial propagation &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% regression-based propagation 
PPSal = regionSal;
fprintf('\n propagation ............\n')
for iter=1:iterNum
%      fprintf('\n iter time %d ......',iter)
     % 2.1 GMR ------- --------------------------------------------------
     GMR1 = optAff_local*PPSal;
     GMR1 = normalizeSal(GMR1); 
        
     % 2.2 SOP ----------------------------------------------------------
     bgWeight = 1-GMR1;
     fgWeight = GMR1;
     SOP = ...
         SaliencyOptimizationNew(adjcMatrix_local, theta, FeaDist, bgWeight, fgWeight);         
     SOP = normalizeSal(SOP);      
     clear bgWeight fgWeight
     
     % 2.3 GMR（optimal）--------------------------------------------------
     if SIGN_GMR==1
     GMR2 = optAff_local*SOP;
     GMR2 = normalizeSal(GMR2); 
     else
         GMR2=SOP;
     end
     
     
     % 赋值 ---------------------------------------------------------------
     PPSal = GMR2;
     clear GMR1 GMR2 SOP 
end

%% 3 integration with original sal &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n integration ............\n')
% 3.1 由 regionSal--->pixelSal ------
regionCenter = tmpSPinfor.region_center;
[PP_Img, ~]  = CreateImageFromSPs(PPSal, tmpSPinfor.pixelList, height, width, true);
[rcenter_PP,ccenter_PP] = computeObjectCenter(PP_Img);
regionDist_PP = ...
     computeRegion2CenterDist(regionCenter,[rcenter_PP,ccenter_PP],[height,width]);
PP_compactness = computeCompactness(PPSal,regionDist_PP);
PP_compactness = 1/(PP_compactness);
clear PP_Img rcenter_PP ccenter_PP regionDist_PP

% 3.2 integration PPSal & TPSAL -----
tpCompactness = TPSAL{ss,1}.compactness;
tpSal         = TPSAL{ss,1}.SalValue;
wtp = tpCompactness/(tpCompactness+PP_compactness);
wpp = PP_compactness/(tpCompactness+PP_compactness);
clear PP_compactness tpCompactness 

tmp_TPSPSAL_sal       = normalizeSal(wpp*PPSal + wtp*tpSal);
[tmp_TPSPSAL_Img, ~]  = CreateImageFromSPs(tmp_TPSPSAL_sal, tmpSPinfor.pixelList, height,width, true);

tmp_TPSPSAL_Img       = graphCut_Refine(image,tmp_TPSPSAL_Img); 
tmp_TPSPSAL_Img       = normalizeSal(guidedfilter(tmp_TPSPSAL_Img,tmp_TPSPSAL_Img,5,0.1));

TPSPSAL_Img = TPSPSAL_Img + tmp_TPSPSAL_Img;% 像素级的显著性图
TPSPSAL_RegionSal{ss,1} = ...
        computeRegionSal(tmp_TPSPSAL_Img,tmpSPinfor.pixelList);% 各尺度下的区域显著性值
    
clear tmp_TPSPSAL_Img tmp_TPSPSAL_sal
clear PPSal FeaDist regionFea   

end
TPSPSAL_Img = normalizeSal(TPSPSAL_Img);

clear TPSAL CURINFOR image flow


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. 提取特征用于传播（像素级的特征）
% 改为只有7维的特征向量 2016.10.16 22:46PM
% 改为10维特征 RGB/LAB/MO/XY 2016.10.26 14：12PM
% 改为8维特征  RGB/LAB/XY     2016.10.26 22:54PM
function FEA = prepaFea(image,flow)
image = double(image);
[height,width,dims] = size(image);

% apperance
im_R = image(:,:,1);
im_G = image(:,:,2);
im_B = image(:,:,3);

[im_L, im_A, im_B1] = ...
    rgb2lab_dong(double(im_R(:)),double(im_G(:)),double(im_B(:)));
im_L=reshape(im_L,size(im_R));
im_A=reshape(im_A,size(im_R));
im_B1=reshape(im_B1,size(im_R));
        
% imgHSV=colorspace('HSV<-',uint8(image));      
% im_H=imgHSV(:,:,1);
% im_S=imgHSV(:,:,2);
% im_V=imgHSV(:,:,3);

% motion
curFlow = double(flow);
Magn    = sqrt(curFlow(:,:,1).^2+curFlow(:,:,2).^2);    
Ori     = atan2(-curFlow(:,:,1),curFlow(:,:,2));
clear flow

% location x,y
im_Y = repmat([1:height]',[1,width]);
im_X = repmat([1:width], [height,1]);

%% 2 preparation
FEA = zeros(height,width,10);
FEA(:,:,1) = im_R;FEA(:,:,2) = im_G;FEA(:,:,3) = im_B;
FEA(:,:,4) = im_L;FEA(:,:,5) = im_A;FEA(:,:,6) = im_B1;
FEA(:,:,7) = Magn;FEA(:,:,8) = Ori; 
FEA(:,:,9) = im_X;FEA(:,:,10) = im_Y;

% FEA = zeros(height,width,13);
% FEA(:,:,1) = im_R;FEA(:,:,2) = im_G;FEA(:,:,3) = im_B;
% FEA(:,:,4) = im_L;FEA(:,:,5) = im_A;FEA(:,:,6) = im_B1;
% FEA(:,:,7) = im_H;FEA(:,:,8) = im_S;FEA(:,:,9) = im_V;
% FEA(:,:,10) = Magn;FEA(:,:,11) = Ori;
% FEA(:,:,12) = im_Y;
% FEA(:,:,13) = im_X;
clear im_R im_G im_B im_L im_A im_B1 im_H im_S im_V Magn Ori im_Y im_X
clear image 
end

% 2. 各区域的特征值 (区域级的特征)
% 改为只有7维的特征向量 2016.10.16 22:46PM
% 改为10维特征 RGB/LAB/MO/XY 2016.10.26 14:12PM
% 改为8维特征 RGB/LAB/XY     2016.10.26 22:54PM
function regionFea = computeRegionFea(FEA,tmpSPinfor)
% im_R = FEA(:,:,1);im_G = FEA(:,:,2);im_B = FEA(:,:,3);
% im_L = FEA(:,:,4);im_A = FEA(:,:,5);im_B1 = FEA(:,:,6);
% im_H = FEA(:,:,7);im_S = FEA(:,:,8);im_V = FEA(:,:,9);
% Magn = FEA(:,:,10);Ori = FEA(:,:,11);
% im_Y = FEA(:,:,12);
% im_X = FEA(:,:,13);
im_R = FEA(:,:,1);im_G = FEA(:,:,2);im_B = FEA(:,:,3);
im_L = FEA(:,:,4);im_A = FEA(:,:,5);im_B1 = FEA(:,:,6);
Magn = FEA(:,:,7);Ori  = FEA(:,:,8);
im_X = FEA(:,:,9);im_Y  = FEA(:,:,10);

[height,width] = size(im_L);
regionFea = zeros(tmpSPinfor.spNum,size(FEA,3));clear FEA
for sp=1:tmpSPinfor.spNum
    pixelList = tmpSPinfor.pixelList{sp,1};
    tmpfea = [mean(im_R(pixelList)),mean(im_G(pixelList)),mean(im_B(pixelList)), ...
              mean(im_L(pixelList)),mean(im_A(pixelList)),mean(im_B1(pixelList)), ... 
              mean(Magn(pixelList)),mean(Ori(pixelList)), ...
              mean(im_X(pixelList))/width,mean(im_Y(pixelList))/height];
              
        
%     tmpfea = [mean(im_R(pixelList)),mean(im_G(pixelList)),mean(im_B(pixelList)), ...
%             mean(im_L(pixelList)),mean(im_A(pixelList)),mean(im_B1(pixelList)), ...
%             mean(im_H(pixelList)),mean(im_S(pixelList)),mean(im_V(pixelList)), ...
%             mean(Magn(pixelList)),mean(Ori(pixelList)),mean(im_Y(pixelList))/height, ...
%             mean(im_X(pixelList))/width];
    regionFea(sp,:) = tmpfea;

    clear tmpfea pixelList
    
end

clear FEA tmpSPinfor

end

% 3 关联阵
function W = SetSmoothnessMatrix(colDistM, adjcMatrix_nn, theta)
allDists = colDistM(adjcMatrix_nn > 0);
maxVal = max(allDists);
minVal = min(allDists);

colDistM(adjcMatrix_nn == 0) = Inf;
colDistM = (colDistM - minVal) / (maxVal - minVal + eps);
W = exp(-colDistM * theta);
end

% 4 2-hop & bb
function adjcMatrix = LinkNNAndBoundary2(adjcMatrix, bdIds)
%link boundary SPs
adjcMatrix(bdIds, bdIds) = 1;

%link neighbor's neighbor
adjcMatrix = (adjcMatrix * adjcMatrix + adjcMatrix) > 0;
adjcMatrix = double(adjcMatrix);

spNum = size(adjcMatrix, 1);
adjcMatrix(1:spNum+1:end) = 0;  %diagnal elements set to be zero
end

% 5 根据初始融合结果，计算各尺度下的显著性值
function regionSal = computeRegionSal(refImage,pixelList)
regionSal = zeros(length(pixelList),1);

for i=1:length(pixelList)
    regionSal(i,1) = mean(refImage(pixelList{i,1}));
end
regionSal = normalizeSal(regionSal);

clear refImage pixelList
end
