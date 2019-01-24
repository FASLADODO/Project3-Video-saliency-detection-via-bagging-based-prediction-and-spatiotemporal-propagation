function [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun13_1(TPSAL,CURINFOR,image,flow,param)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ������ԭʼ�� saliency optimization �Ŀ��򴫲��㷨
% feaDist����ɫ������ĺͣ�bgWeight�ɴ˵ó�
% 
% 2016.11.24 20:40AM
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FEA = prepaFea(image,flow);
[height,width,dims] = size(image);
SPSCALENUM = length(CURINFOR.fea);
SALS = cell(SPSCALENUM,1);
alpha=0.99;
theta=10;
iterNum = 1;
TPSPSAL_Img = 0;
TPSPSAL_RegionSal = cell(SPSCALENUM,1);
SIGN_GMR = 0;
for ss=1:SPSCALENUM
    fprintf('\n scale num %d .........................................',ss)
    fprintf('\n initialization ............\n')
%% 1 initial &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    tmpSPinfor = CURINFOR.spinfor{ss,1};% ���߶��µķָ��� 
    spNum      = tmpSPinfor.spNum;
    adjcMatrix = tmpSPinfor.adjcMatrix;
    bdIds      = tmpSPinfor.bdIds;
 
    regionSal  = TPSAL{ss,1}.SalValue;% ������ĳ�ʼ������ֵ
    regionFea0  = computeRegionFea(image,flow,tmpSPinfor);% ����������� L,a,b,man,ori
    
%     tmpFEA       = CURINFOR.fea{ss,1};
%     regionFea = [tmpFEA.colorHist_rgb,tmpFEA.colorHist_lab,tmpFEA.colorHist_hsv , ...
%                  tmpFEA.lbp_top_Hist, tmpFEA.regionCov    ,tmpFEA.LM_textureHist, ...
%                  tmpFEA.flowHist];
             
%     ZZ         = repmat(sqrt(sum(regionFea.*regionFea)),[spNum,1]);% ����ȫ�ֹ�һ�� 2016.10.28 9:32AM
%     ZZ(ZZ==0)  = eps;
%     regionFea  = regionFea./ZZ;
%     [regionFea,regionFea_mapping] = pca(regionFea,0.995);

%     regionFea  = regionFea0(:,1:3);clear regionFea0
%     FeaDist    = GetDistanceMatrix(regionFea);

    FeaDist = feaDistIntegrate(regionFea0);
%% 2 iterative spatial propagation &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% regression-based propagation 
PPSal = regionSal;
PPSal = normalizeSal(PPSal); 
fprintf('\n propagation ............\n')
for iter=1:iterNum
    [clipVal, geoSigma, neiSigma] = EstimateDynamicParas(adjcMatrix, FeaDist);
    [~, ~, bgWeight] = EstimateBgProb(FeaDist, adjcMatrix, bdIds, clipVal, geoSigma);
    
    %post-processing for cleaner fg cue
    removeLowVals = param.removeLowVals;
    if removeLowVals
       thresh = graythresh(PPSal);  %automatic threshold
       PPSal(PPSal < thresh) = 0;
    end
    
    PPSal = SaliencyOptimization(adjcMatrix, bdIds, FeaDist, neiSigma, bgWeight, PPSal);
    PPSal = normalizeSal(PPSal);  

end

%% 3 integration with original sal &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n integration ............\n')
% 3.1 �� regionSal--->pixelSal ------
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

wpp = 1;wtp=1;
tmp_TPSPSAL_sal       = normalizeSal(wpp*PPSal + wtp*tpSal);
[tmp_TPSPSAL_Img, ~]  = CreateImageFromSPs(tmp_TPSPSAL_sal, tmpSPinfor.pixelList, height,width, true);

tmp_TPSPSAL_Img       = graphCut_Refine(image,tmp_TPSPSAL_Img); 
tmp_TPSPSAL_Img       = normalizeSal(guidedfilter(tmp_TPSPSAL_Img,tmp_TPSPSAL_Img,6,0.1));

TPSPSAL_Img = TPSPSAL_Img + tmp_TPSPSAL_Img;% ���ؼ���������ͼ
TPSPSAL_RegionSal{ss,1} = ...
        computeRegionSal(tmp_TPSPSAL_Img,tmpSPinfor.pixelList);% ���߶��µ�����������ֵ
    
clear tmp_TPSPSAL_Img tmp_TPSPSAL_sal
clear PPSal FeaDist regionFea   

end
TPSPSAL_Img = normalizeSal(TPSPSAL_Img);

clear TPSAL CURINFOR image flow


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. �����������ֵ (���򼶵�����) &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% ����Lab & Man/Ori, 2016.11.24
function regionFea = computeRegionFea(image,flow,tmpSPinfor)
meanRgbCol = GetMeanColor(image, tmpSPinfor.pixelList);
meanLabCol = colorspace('Lab<-', double(meanRgbCol)/255);
clear image

curFlow = double(flow);
Magn    = sqrt(curFlow(:,:,1).^2+curFlow(:,:,2).^2);    
Ori     = atan2(-curFlow(:,:,1),curFlow(:,:,2));
meanMagn = GetMeanColor(Magn, tmpSPinfor.pixelList);
meanOri  = GetMeanColor(Ori, tmpSPinfor.pixelList);
clear Ori Magn flow
clear  tmpSPinfor

% [height,width] = size(im_L);
% regionFea = zeros(tmpSPinfor.spNum,5);
regionFea = [meanLabCol,meanMagn,meanOri];

clear meanLabCol meanMagn meanOri


end


% 2 ���ݳ�ʼ�ںϽ����������߶��µ�������ֵ *********************************
function regionSal = computeRegionSal(refImage,pixelList)
regionSal = zeros(length(pixelList),1);

for i=1:length(pixelList)
    regionSal(i,1) = mean(refImage(pixelList{i,1}));
end
regionSal = normalizeSal(regionSal);

clear refImage pixelList
end


% % 3 �������ֿ�������룬Ȼ������  *****************************************
% % ������Ҫ�� Lab & Flow(man/ori)  2016.11.24
% function FeaDist = feaDistIntegrate(regionFea)
% % [regionFea,mapping] = scaleForSVM_corrected1(regionFea,0,1);% ��������ÿһά��һ����0~1֮��,����Թ�ϵδ��
% % lamda1 = 0.55;lamda2 = 0.35;lamda3 = 0.15;
% % FeaColor = regionFea(:,1:3);
% % FeaOri   = regionFea(:,5);
% % FeaMan   = regionFea(:,4);
% % 
% % FeaDistColor = GetDistanceMatrix(FeaColor);
% % FeaDistOri   = GetDistanceMatrix(FeaOri);
% % FeaDistMan   = GetDistanceMatrix(FeaMan);
% % 
% % FeaDist      = lamda1*FeaDistColor + lamda2*FeaDistOri + lamda3*FeaDistMan;
% regionFea(:,1) = normalizeSal(regionFea(:,1));
% regionFea(:,2) = normalizeSal(regionFea(:,2));
% regionFea(:,3) = normalizeSal(regionFea(:,3));
% regionFea(:,4) = normalizeSal(regionFea(:,4));
% regionFea(:,5) = normalizeSal(regionFea(:,5));
% 
% FeaDist = zeros(size(regionFea,1),size(regionFea,1));
% for i=1:size(regionFea,1)
%     tmpI = regionFea(i,:);
%     for j=1:size(regionFea,1)
%         tmpJ = regionFea(j,:);
%         tmpD = sum((tmpI - tmpJ).*(tmpI - tmpJ));
%         FeaDist(i,j) = sqrt(tmpD);
%         
%         clear tmpJ tmpD
%     end
%     clear tmpI
% end
% 
% clear regionFea
% 
% end


