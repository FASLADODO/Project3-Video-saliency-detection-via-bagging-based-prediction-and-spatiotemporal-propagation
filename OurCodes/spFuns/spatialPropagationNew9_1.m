function [TPSPSAL,TPSPSALRegionSal] = spatialPropagationNew9_1(CURINFOR,IMSAL_TPSAL1,param,cur_image,imsal_pre,GPsign)
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
% V5�� 2016.11.02 21��03PM
% �µĿ��򴫲���ʽ
% 1) CURINFOR.fea  ����ȫ������������������
%     fea{ss,1}.colorHist_rgb 
%     fea{ss,1}.colorHist_lab 
%     fea{ss,1}.colorHist_hsv 
%     fea{ss,1}.lbpHist     
%     fea{ss,1}.lbp_top_Hist 
%     fea{ss,1}.hogHist
%     fea{ss,1}.regionCov   
%     fea{ss,1}.geoDist    
%     fea{ss,1}.flowHist   
% 2) PCA           ��ά
% 
% [DB.P.colorHist_rgb_mappedA,DB.P.colorHist_rgb_mapping] = pca(D0.P.colorHist_rgb,no_dims);
% 3) ����������
% 
% copyright by xiaofei zhou,shanghai university,shanghai,china
% zxforchid@163.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n this is spatial propagation process, wait a minute .........')
no_dims = param.no_dims;
[r,c] = size(IMSAL_TPSAL1);
alpha=0.99;
theta=10;
iterSal = IMSAL_TPSAL1;
iternum = 10;
%% A: ��Ѷ���� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
for iter = 1:iternum
    fprintf('\n the %d iteration ......',iter)
%% �����ںϺ��ʱ�򴫲�ͼ�񣬼����������� &&&&&&&&&&&&&&&&&&&&&&&&&&&&
[rcenter,ccenter] = computeObjectCenter(iterSal);% x-->row, y-->col

%% ��������߶ȴ��� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n propagation ...\n')
LPRegionSals = 0;
propagate.salValue    = cell(length(CURINFOR.fea),1);
propagate.regionDist    = cell(length(CURINFOR.fea),1);
initialSal.salValue   = cell(length(CURINFOR.fea),1);
initialSal.regionDist = cell(length(CURINFOR.fea),1);
for ss=1:length(CURINFOR.fea)
    tmpFEA       = CURINFOR.fea{ss,1};
    tmpSPinfor   = CURINFOR.spinfor{ss,1};% ���߶��µķָ��� 
    spNum        = tmpSPinfor.spNum;
    adjcMatrix   = tmpSPinfor.adjcMatrix;
    bdIds        = tmpSPinfor.bdIds;
    pixelList    = tmpSPinfor.pixelList;
    regionCenter = tmpSPinfor.region_center;
    
    %% 0 ���߶��µĶ�Ӧ������ͼ�� object-biased prior &&&&&&&&&&&&&&&
    regionSal  = computeRegionSal(IMSAL_TPSAL1,pixelList);% ���߶��µ�����������ֵ
    regionDist = computeRegion2CenterDist(regionCenter,[rcenter,ccenter],[r,c]);
    initialSal.salValue{ss,1}   = regionSal;
    initialSal.regionDist{ss,1} = regionDist;
    
    %% 1 ���߶����γɴ���������� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    regionFea = [tmpFEA.colorHist_rgb,tmpFEA.colorHist_lab,tmpFEA.colorHist_hsv,tmpFEA.lbpHist,...
        tmpFEA.lbp_top_Hist,tmpFEA.hogHist,tmpFEA.regionCov,tmpFEA.geoDist,tmpFEA.flowHist];

    %% 2 PCA ѹ�� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    % regionFea_mappedA Ϊ���յ���������
    [regionFea_mappedA,regionFea_mapping] = pca(regionFea,no_dims);
    ZZ         = repmat(sqrt(sum(regionFea_mappedA.*regionFea_mappedA)),[spNum,1]);% ����ȫ�ֹ�һ�� 2016.10.28 9:32AM
    ZZ(ZZ==0)  = eps;
    regionFea_mappedA  = regionFea_mappedA./ZZ;
    FeaDist    = GetDistanceMatrix(regionFea_mappedA);    
    
    clear regionFea_mappedA regionFea_mapping ZZ regionFea 
    %% 3 ���������ϵ������������ &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    % 3.1 ��ȡsinkPoints
    deltas = ones(spNum,1);
    IFNUM = round(0.3*spNum);% һ��ͼ���Լ��1/5������Ϊ��������
    SCB = regionDist.*regionSal;
    SCB = normalizeSal(SCB);
    [value,index] = sort(SCB);
    IF_index = index(1:IFNUM);
    deltas(IF_index,1) = 0;
%     [SCB_Img, ~] = CreateImageFromSPs(SCB, pixelList, r, c, true);
%     threshold = graythresh(SCB_Img);
%     deltas = double(SCB > threshold);
    IF = diag(deltas);
    
    % 3.2 ����������  localW
    adjcMatrix_local = LinkNNAndBoundary2(adjcMatrix, bdIds); 
    W_local          = SetSmoothnessMatrix(FeaDist, adjcMatrix_local, theta);
    D_local          = diag(sum(W_local));
    optAff_local     = (D_local-alpha*W_local)\eye(spNum);
%     optAff_local     = (D_local-alpha*W_local*IF)\eye(spNum);
    optAff_local(1:spNum+1:end) = 0;% �Խ�������
    
    clear FeaDist deltas W_local D_local adjcMatrix_local IF SCB SCB_Img  
    %% 4 ���� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
     LP = optAff_local*regionSal;
     LP = normalizeSal(LP); 
     [LP_Img, ~]  = CreateImageFromSPs(LP, pixelList, r, c, true);
     LPRegionSals = LPRegionSals + LP_Img;
     propagate.salValue{ss,1}    = LP;
     clear LP optAff_local regionSal
end

%% �ںϳ�ʼ�봫�����������ͼ &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n fusion ...')
LPRegionSals = normalizeSal(LPRegionSals);
[rcenter_LP,ccenter_LP] = computeObjectCenter(LPRegionSals);
tmpTPSPSAL = 0;% �ںϺ���߶����ؼ���������ؼ�������ͼ
for ss=1:length(CURINFOR.fea)
    tmpSPinfor   = CURINFOR.spinfor{ss,1};
    regionCenter = tmpSPinfor.region_center;
    
    % compactness_diff
    propagate.regionDist{ss,1} = ...
        computeRegion2CenterDist(regionCenter,[rcenter_LP,ccenter_LP],[r,c]);
    compactness_LP   = sum(propagate.regionDist{ss,1}.*propagate.salValue{ss,1});
    compactness_init = sum(initialSal.regionDist{ss,1}.*initialSal.salValue{ss,1});  
     w1_1 = compactness_init/(compactness_LP+compactness_init);
     w2_1 = compactness_LP  /(compactness_LP+compactness_init);
     
     % imsal_pre_diff
    [init_image, ~]  = CreateImageFromSPs(initialSal.salValue{ss,1}, tmpSPinfor.pixelList, r, c, true);
    [propa_image, ~] = CreateImageFromSPs(propagate.salValue{ss,1}, tmpSPinfor.pixelList, r, c, true); 
    tmp_init_diff  = init_image - imsal_pre;
    tmp_propa_diff = propa_image - imsal_pre;
    tmp_init_diff  = sum(tmp_init_diff(:).*tmp_init_diff(:))/length(tmp_init_diff(:));
    tmp_propa_diff = sum(tmp_propa_diff(:).*tmp_propa_diff(:))/length(tmp_propa_diff(:));
     w1_0 = tmp_init_diff /(tmp_propa_diff+tmp_init_diff);
     w2_0 = tmp_propa_diff/(tmp_propa_diff+tmp_init_diff);
    
     % fused_diff
     w1 = w1_1*w1_0;
     w2 = w2_1*w2_0;
     w1 = w1/(w1+w2);
     w2 = w2/(w1+w2);
     
     fusedmap = ...
         normalizeSal(w1*initialSal.salValue{ss,1} + w2*propagate.salValue{ss,1});
%      TPSPSALRegionSal{ss,1} = fusedmap;
     [fusedmap_img, ~]  = CreateImageFromSPs(fusedmap, tmpSPinfor.pixelList, r, c, true);
     tmpTPSPSAL = tmpTPSPSAL + fusedmap_img;
     
     clear tmpSPinfor tmpSPinfor fusedmap fusedmap_img
end

tmpTPSPSAL = normalizeSal(tmpTPSPSAL);
iterSal = tmpTPSPSAL;
clear TPSPSAL
end

%% B: �������ս�� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n assigenment the last result ...')
switch GPsign
    case 'YES'
         iterSal = graphCut_Refine(cur_image,iterSal); 
         TPSPSAL = iterSal;
    case 'NO'
         TPSPSAL = iterSal;  
end
TPSPSAL = normalizeSal(guidedfilter(TPSPSAL,TPSPSAL,5,0.1));

TPSPSALRegionSal = cell(length(CURINFOR.fea),1);% ���߶��µĽ��
for ss=1:length(CURINFOR.fea)
    tmpSPinfor   = CURINFOR.spinfor{ss,1};
    TPSPSALRegionSal{ss,1} = ...
        computeRegionSal(TPSPSAL,tmpSPinfor.pixelList);% ���߶��µ�����������ֵ
    clear tmpSPinfor
end



end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% �Ӻ�������  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 ������������
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

% 2 ���ݳ�ʼ�ںϽ����������߶��µ�������ֵ
function regionSal = computeRegionSal(refImage,pixelList)
regionSal = zeros(length(pixelList),1);

for i=1:length(pixelList)
    regionSal(i,1) = mean(refImage(pixelList{i,1}));
end
regionSal = normalizeSal(regionSal);

clear refImage pixelList
end

% 3 ���������������ĵľ��� 
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


% 4 ������
function W = SetSmoothnessMatrix(colDistM, adjcMatrix_nn, theta)
allDists = colDistM(adjcMatrix_nn > 0);
maxVal = max(allDists);
minVal = min(allDists);

colDistM(adjcMatrix_nn == 0) = Inf;
colDistM = (colDistM - minVal) / (maxVal - minVal + eps);
W = exp(-colDistM * theta);
end

% 5 2-hop & bb
function adjcMatrix = LinkNNAndBoundary2(adjcMatrix, bdIds)
%link boundary SPs
adjcMatrix(bdIds, bdIds) = 1;

%link neighbor's neighbor
adjcMatrix = (adjcMatrix * adjcMatrix + adjcMatrix) > 0;
adjcMatrix = double(adjcMatrix);

spNum = size(adjcMatrix, 1);
adjcMatrix(1:spNum+1:end) = 0;  %diagnal elements set to be zero
end

