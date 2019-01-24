function SALS = spatialPropagationNew8(TPSAL,CURINFOR,image,flow,beta,IMSAL_TPSAL1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��ʱ�򴫲��Ļ����Ͻ��п��򴫲�
% CURINFOR
% fea/ORLabels/spinfor(mapsets��region_center_prediction)
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
% TPSAL(ȫ�ߴ�)
% ���߶��¸������������ֵ
% 
% V1: 2016.10.14 20:01PM
% ����CVPR2016 GRAB˼����д����Ż�
% 
% V2:2016.10.18 15:45PM
% ������µ� iterative propagation �����޸�
% Ŀǰ�Ǳ�������+�Ż�+ǰ�����������ó����ͼ�ṹ���޵���
% 
% V3: 2016.10.19 16:18PM
% �����������
% 
% V4�� 2016.10.30 14��59PM
% object-biased + regression-based-{GMR --> SO --> GMR}
% �����Ĵ������²���
% ���ó���ͼ�ṹ/�����
% 
% copyright by xiaofei zhou,shanghai university,shanghai,china
% zxforchid@163.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. ��ȡ���� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    tmpSPinfor = CURINFOR.spinfor{ss,1};% ���߶��µķָ��� 
    spNum      = tmpSPinfor.spNum;
    adjcMatrix = tmpSPinfor.adjcMatrix;
    bdIds      = tmpSPinfor.bdIds;
 
    regionSal  = TPSAL{ss,1};% ������ĳ�ʼ������ֵ
    regionFea  = computeRegionFea(FEA,tmpSPinfor);% �����������
    ZZ         = repmat(sqrt(sum(regionFea.*regionFea)),[spNum,1]);% ����ȫ�ֹ�һ�� 2016.10.28 9:32AM
    ZZ(ZZ==0)  = eps;
    regionFea  = regionFea./ZZ;
    FeaDist    = GetDistanceMatrix(regionFea);
    
    %% 1.1 �����ֲ�+ȫ�ִ����� ��localW + globalW�� &&&&&&&&&&&&&&&&&
    % localW
    adjcMatrix_local = LinkNNAndBoundary2(adjcMatrix, bdIds); 
    W_local          = SetSmoothnessMatrix(FeaDist, adjcMatrix_local, theta);
    D_local          = diag(sum(W_local));
    optAff_local     = (D_local-alpha*W_local)\eye(spNum);
    optAff_local(1:spNum+1:end) = 0;
%     [~, ~, neiSigma_local] = EstimateDynamicParas(adjcMatrix_local, FeaDist);
    
    % globalW
    knn=round(size(regionFea,1)*1/4);
    kdtree = vl_kdtreebuild(regionFea');% ���� feaDim*sampleNum
    [indexs, distance] = vl_kdtreequery(kdtree,regionFea',regionFea', 'NumNeighbors', knn) ;
    indexs1 = indexs(2:end,:);% ÿһ��Ϊһ�������Ӧ��KNN��������Щ����Ϊ1������Ϊ0
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
     % 2.1.1 �ֲ����� --------------------------------------------------
     LP = optAff_local*endSal;
     LP = normalizeSal(LP); 
        
     % 2.1.2 �ֲ��Ż� -----------------------------------------------------
     bgWeight = 1-LP;
     fgWeight = LP;
     LOP = ...
         SaliencyOptimizationNew(adjcMatrix_local, theta, FeaDist, bgWeight, fgWeight);         
     LOP = normalizeSal(LOP);      
     clear bgWeight fgWeight
     
     % 2.2.1 ȫ�ִ��� -----------------------------------------------------
     GP = optAff_local*LOP;
     GP = normalizeSal(GP); 
%         
%      % 2.2.2 ȫ���Ż� -----------------------------------------------------
%      bgWeight = 1-GP;
%      fgWeight = GP;
%      GOP = ...
%          SaliencyOptimizationNew(adjcMatrix_global, theta, FeaDist, bgWeight, fgWeight);
%      GOP = normalizeSal(GOP);
%      clear bgWeight fgWeight
     
     GOP = GP;
     % ��ֵ ---------------------------------------------------------------
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
% 1. ��ȡ�������ڴ��������ؼ���������
% ��Ϊֻ��7ά���������� 2016.10.16 22:46PM
% ��Ϊ10ά���� RGB/LAB/MO/XY 2016.10.26 14��12PM
% ��Ϊ8ά����  RGB/LAB/XY     2016.10.26 22:54PM
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

% 2. �����������ֵ (���򼶵�����)
% ��Ϊֻ��7ά���������� 2016.10.16 22:46PM
% ��Ϊ10ά���� RGB/LAB/MO/XY 2016.10.26 14:12PM
% ��Ϊ8ά���� RGB/LAB/XY     2016.10.26 22:54PM
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

% 3 ������
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

% 5 �����������Ե����ؼ������� 2016.10.23 19:33PM 
function result = createImgFromSP(regionSal, pixelList, height, width)
regionSal = normalizeSal(regionSal);
result = zeros(height,width);

for i=1:length(pixelList)
    result(pixelList{i}) = regionSal(i);
end

clear regionSal pixelList height width
end