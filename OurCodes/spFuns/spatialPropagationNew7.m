function SALS = spatialPropagationNew7(TPSAL,CURINFOR,image,flow,beta,IMSAL_TPSAL1)
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
% V4: 2016.10.24 22:21PM
% ��beta����߶�Ӧ����������������
% 
% copyright by xiaofei zhou,shanghai university,shanghai,china
% zxforchid@163.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1 ��ȡ���� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
indexs  = beta(:,2);
weights = beta(:,1);
[maxvalue,maxindex] = max(weights);
ID = indexs(maxindex(end));

FEA = CURINFOR.fea;
[height,width,dims] = size(image);

%% 2. begin %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPSCALENUM = length(CURINFOR.fea);
SALS = cell(SPSCALENUM,1);
alpha=0.99;
theta=10;
% AL = 0.75;AH=1.25; % ѡȡȷ�����������ڴ�����ǰ�����ڱ�����������������ǰ������
iterNum = 20;
% for iter=1:20
for ss=1:SPSCALENUM
    fprintf('\n scale num %d ............',ss)
%% 1 initial &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    tmpSPinfor = CURINFOR.spinfor{ss,1};% ���߶��µķָ��� 
    regionSal  = TPSAL{ss,1};% ������ĳ�ʼ������ֵ
    tmp_regionFea  = FEA{ss,1};
    d1 = [tmp_regionFea.colorHist_rgb];
    d2 = [tmp_regionFea.colorHist_lab];
    d3 = [tmp_regionFea.colorHist_hsv];
    d4 = [tmp_regionFea.lbpHist];      
    d5 = [tmp_regionFea.hogHist];      
    d6 = [tmp_regionFea.regionCov];    
    d7 = [tmp_regionFea.geoDist];      
    d8 = [tmp_regionFea.flowHist];     
    
    spNum      = tmpSPinfor.spNum;
    
    regionFea = eval(['d' num2str(ID)]);
    adjcMatrix = tmpSPinfor.adjcMatrix;
    bdIds      = tmpSPinfor.bdIds;
    FeaDist    = GetDistanceMatrix(regionFea);
    pixelnums = [];
    for ii=1:spNum
        tmppixelnum = tmpSPinfor.pixelList{ii,1}; 
        pixelnums = [pixelnums;length(tmppixelnum)];
        clear tmppixelnum
    end
    
    adjcMatrix_nn = LinkNNAndBoundary2(adjcMatrix, bdIds); % adjcMatrix_w���ڴ�����
    W = SetSmoothnessMatrix(FeaDist, adjcMatrix_nn, theta);
    D = diag(sum(W));
    optAff =(D-alpha*W)\eye(spNum);
    optAff(1:spNum+1:end) = 0;
    
%     [~, ~, neiSigma] = EstimateDynamicParas(adjcMatrix, FeaDist);    
    [clipVal, geoSigma, neiSigma] = EstimateDynamicParas(adjcMatrix, FeaDist);
%     [bgProb, bdCon, bgWeight] = EstimateBgProb(FeaDist, adjcMatrix, bdIds, clipVal, geoSigma);
    
%% 2 ��ʼ���� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&    
    for iter=1:iterNum
        fprintf('\n iter time %d ......',iter)
        if iter==1 % ��һ�ε���ʱ��������OR�������ֵ  
          % 3.2 �Ż�
%           fgWeight = regionSal> meanSal0;
%           bgWeight = regionSal< meanSal0;
          fgWeight = regionSal;
          bgWeight = 1 - regionSal;
          TPR = SaliencyOptimization(adjcMatrix, bdIds, FeaDist, neiSigma, bgWeight, fgWeight);
          TPR = normalizeSal(TPR);
          
          % 3.4 FPP ǰ������
          Pixel_TPR = createImgFromSP(TPR, tmpSPinfor.pixelList, height, width);
          threshold = graythresh(Pixel_TPR);
          TPR_GT = TPR>threshold;
%           meanSal_TPR = sum(pixelnums.*TPR)/sum(pixelnums);
%           TPR_GT = (TPR>meanSal_TPR);% ���ÿɿ���ǰ��
          FPP = optAff*TPR_GT;
          FPP = normalizeSal(FPP);% ���򼶵�������
        else
%           meanSal0 = sum(pixelnums.*GPSAL)/sum(pixelnums);
%           fgWeight = GPSAL> meanSal0;
%           bgWeight = GPSAL< meanSal0;
          fgWeight = GPSAL;% ǰ������
          bgWeight = 1 - GPSAL;% ��������

          TPR = SaliencyOptimization(adjcMatrix, bdIds, FeaDist, neiSigma, bgWeight, fgWeight);
          TPR = normalizeSal(TPR);
          
          Pixel_TPR = createImgFromSP(TPR, tmpSPinfor.pixelList, height, width);
          threshold = graythresh(Pixel_TPR);
          TPR_GT = TPR>threshold;
%           meanSal_TPR = sum(pixelnums.*TPR)/sum(pixelnums);
%           TPR_GT = (TPR>meanSal_TPR);% ���ÿɿ���ǰ��
          FPP = optAff*TPR_GT;
          FPP = normalizeSal(FPP);% ���򼶵�������
        end
        GPSAL = FPP;
    end
    
%% 3 SAVE --------------------------------------------------------------
    SALS{ss,1}.GPSAL = GPSAL;
    
    clear SPP GPSAL
end

% end


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. ��ȡ�������ڴ��������ؼ���������
% ��Ϊֻ��7ά���������� 2016.10.16 22:46PM
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
        
imgHSV=colorspace('HSV<-',uint8(image));      
im_H=imgHSV(:,:,1);
im_S=imgHSV(:,:,2);
im_V=imgHSV(:,:,3);

% motion
curFlow = double(flow);
Magn    = sqrt(curFlow(:,:,1).^2+curFlow(:,:,2).^2);    
Ori     = atan2(-curFlow(:,:,1),curFlow(:,:,2));
clear flow

% location x,y
im_Y = repmat([1:height]',[1,width]);
im_X = repmat([1:width], [height,1]);

%% 2 preparation
FEA = zeros(height,width,7);
FEA(:,:,1) = im_L;FEA(:,:,2) = im_A;FEA(:,:,3) = im_B1;
FEA(:,:,4) = Magn;FEA(:,:,5) = Ori;FEA(:,:,6) = im_Y;
FEA(:,:,7) = im_X;

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
function regionFea = computeRegionFea(FEA,tmpSPinfor)
% im_R = FEA(:,:,1);im_G = FEA(:,:,2);im_B = FEA(:,:,3);
% im_L = FEA(:,:,4);im_A = FEA(:,:,5);im_B1 = FEA(:,:,6);
% im_H = FEA(:,:,7);im_S = FEA(:,:,8);im_V = FEA(:,:,9);
% Magn = FEA(:,:,10);Ori = FEA(:,:,11);
% im_Y = FEA(:,:,12);
% im_X = FEA(:,:,13);
im_L = FEA(:,:,1);im_A = FEA(:,:,2);im_B1 = FEA(:,:,3);
Magn = FEA(:,:,4);Ori  = FEA(:,:,5);im_Y  = FEA(:,:,6);
im_X = FEA(:,:,7);

[height,width] = size(im_L);
regionFea = zeros(tmpSPinfor.spNum,size(FEA,3));clear FEA
for sp=1:tmpSPinfor.spNum
    pixelList = tmpSPinfor.pixelList{sp,1};
    tmpfea = [mean(im_L(pixelList)),mean(im_A(pixelList)),mean(im_B1(pixelList)), ...
            mean(Magn(pixelList)),mean(Ori(pixelList)),mean(im_Y(pixelList))/height, ...
            mean(im_X(pixelList))/width];
        
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