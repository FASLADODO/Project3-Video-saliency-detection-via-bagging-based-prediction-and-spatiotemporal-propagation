function SALS = spatialPropagationNew8(TPSAL,CURINFOR,image,flow,beta,IMSAL_TPSAL1)
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
for ss=1:SPSCALENUM
    fprintf('\n scale num %d ............',ss)
%% 1 initial &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    tmpSPinfor = CURINFOR.spinfor{ss,1};% 单尺度下的分割结果 
    spNum      = tmpSPinfor.spNum;
    adjcMatrix = tmpSPinfor.adjcMatrix;
    bdIds      = tmpSPinfor.bdIds;
 
    regionSal  = TPSAL{ss,1};% 各区域的初始显著性值
    regionFea  = computeRegionFea(FEA,tmpSPinfor);% 各区域的特征
    ZZ         = repmat(sqrt(sum(regionFea.*regionFea)),[spNum,1]);% 特征全局归一化 2016.10.28 9:32AM
    ZZ(ZZ==0)  = eps;
    regionFea  = regionFea./ZZ;
    FeaDist    = GetDistanceMatrix(regionFea);
    
    %% 1.1 构建局部+全局传播阵 （localW + globalW） &&&&&&&&&&&&&&&&&
    % localW
    adjcMatrix_local = LinkNNAndBoundary2(adjcMatrix, bdIds); 
    W_local          = SetSmoothnessMatrix(FeaDist, adjcMatrix_local, theta);
    D_local          = diag(sum(W_local));
    optAff_local     = (D_local-alpha*W_local)\eye(spNum);
    optAff_local(1:spNum+1:end) = 0;
%     [~, ~, neiSigma_local] = EstimateDynamicParas(adjcMatrix_local, FeaDist);
    
    % globalW
    knn=round(size(regionFea,1)*1/4);
    kdtree = vl_kdtreebuild(regionFea');% 输入 feaDim*sampleNum
    [indexs, distance] = vl_kdtreequery(kdtree,regionFea',regionFea', 'NumNeighbors', knn) ;
    indexs1 = indexs(2:end,:);% 每一列为一个区域对应的KNN邻域，令这些邻域为1，其他为0
    adjcMatrix_global = zeros(spNum,spNum);
    for i=1:spNum  
        tmp_nozeros = indexs1(:,i);
        adjcMatrix_global(i,tmp_nozeros) = 1;
        adjcMatrix_global(tmp_nozeros,i) = 1;
    end
    W_global          = SetSmoothnessMatrix(FeaDist, adjcMatrix_global, theta);
    D_global          = diag(sum(W_global));
    optAff_global     = (D_global-alpha*W_global)\eye(spNum);
    optAff_global(1:spNum+1:end) = 0;
%     [~, ~, neiSigma_global] = EstimateDynamicParas(adjcMatrix_global, FeaDist);
    
%% 2 iterative spatial propagation &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% regression-based propagation 
endSal = regionSal;
for iter=1:iterNum
     fprintf('\n iter time %d ......',iter)
     % 2.1.1 局部传播 --------------------------------------------------
     LP = optAff_local*endSal;
     LP = normalizeSal(LP); 
        
     % 2.1.2 局部优化 -----------------------------------------------------
     bgWeight = 1-LP;
     fgWeight = LP;
     LOP = ...
         SaliencyOptimizationNew(adjcMatrix_local, theta, FeaDist, bgWeight, fgWeight);         
     LOP = normalizeSal(LOP);      
     clear bgWeight fgWeight
     
     % 2.2.1 全局传播 -----------------------------------------------------
     GP = optAff_local*LOP;
     GP = normalizeSal(GP); 
%         
%      % 2.2.2 全局优化 -----------------------------------------------------
%      bgWeight = 1-GP;
%      fgWeight = GP;
%      GOP = ...
%          SaliencyOptimizationNew(adjcMatrix_global, theta, FeaDist, bgWeight, fgWeight);
%      GOP = normalizeSal(GOP);
%      clear bgWeight fgWeight
     
     GOP = GP;
     % 赋值 ---------------------------------------------------------------
     endSal = GOP;
     clear LP LOP GP GOP
end
    
%% 3 SAVE --------------------------------------------------------------
SALS{ss,1}.GPSAL = endSal;

clear endSal FeaDist regionFea   

end

clear FEA


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

% 5 由区域级显著性到像素级显著性 2016.10.23 19:33PM 
function result = createImgFromSP(regionSal, pixelList, height, width)
regionSal = normalizeSal(regionSal);
result = zeros(height,width);

for i=1:length(pixelList)
    result(pixelList{i}) = regionSal(i);
end

clear regionSal pixelList height width
end