function [ TPSPSAL_Img,TPSPSAL_RegionSal ] = spfun19_6( TPSAL,CURINFOR,image,flow,param )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMR  前景传播
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
    tmpSPinfor = CURINFOR.spinfor{ss,1};% 单尺度下的分割结果 
    spNum      = tmpSPinfor.spNum;
    adjcMatrix = tmpSPinfor.adjcMatrix;
    bdIds      = tmpSPinfor.bdIds;
 
    regionSal  = TPSAL{ss,1}.SalValue;% 各区域的初始显著性值
%     PP_Img     = TPSAL1{ss,1}.PP_Img;
%     threshold = graythresh(PP_Img);
%     regionFea0  = computeRegionFea(image,flow,tmpSPinfor);% 各区域的特征 L,a,b,man,ori
     tmpFEA    = CURINFOR.fea{ss,1};
     regionFea = [tmpFEA.regionFea];    
     
%      feaIndex = model{1,1}.feaIndex;
%      regionFea(:,feaIndex) = [];

   % Flow + color
%     FeaDist = feaDistIntegrate(regionFea);clear regionFea0
%% 2 iterative spatial propagation &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% regression-based propagation 
PPSal = regionSal;
PPSal = normalizeSal(PPSal); 
fprintf('\n propagation ............\n')
for iter=1:iterNum
    thresh = graythresh(PPSal); 
    PPSal0 = PPSal > thresh;
%     indexFg = find(PPSal1 == 1);
%     bdIds1 = abortFGpoints(indexFg,bdIds);
    PPSal0 = double(PPSal0);
    
    % color
    FeaDist1 = feaDistIntegrate(regionFea(:,4:6));
    [PPSal1] = MyManifoldRanking(adjcMatrix, PPSal, bdIds, FeaDist1);
    PPSal1 = normalizeSal(PPSal1);  
    
    % motion
    FeaDist2 = feaDistIntegrate(regionFea(:,8:9));
    [PPSal2] = MyManifoldRanking(adjcMatrix, PPSal, bdIds, FeaDist2);
    PPSal2 = normalizeSal(PPSal2);
    
    PPSal = normalizeSal(PPSal1 + PPSal2);
    
    
%     % 2.1 GMR -------------------------------------------------------------
%     [PPSal] = MyManifoldRanking(adjcMatrix, PPSal1, bdIds, FeaDist);
%      PPSal  = normalizeSal(PPSal); 
    clear FeaDist1 PPSal1 FeaDist2 PPSal2 
    clear thresh PPSal0 indexFg bdIds1 
    
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

wpp = 1;wtp=1;
tmp_TPSPSAL_sal       = normalizeSal(wpp*PPSal + wtp*tpSal);
[tmp_TPSPSAL_Img, ~]  = CreateImageFromSPs(tmp_TPSPSAL_sal, tmpSPinfor.pixelList, height,width, true);

tmp_TPSPSAL_Img       = graphCut_Refine(image,tmp_TPSPSAL_Img); 
tmp_TPSPSAL_Img       = normalizeSal(guidedfilter(tmp_TPSPSAL_Img,tmp_TPSPSAL_Img,6,0.1));

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
% 1. 各区域的特征值 (区域级的特征) &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% 保留Lab & Man/Ori, 2016.11.24
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


% 2 根据初始融合结果，计算各尺度下的显著性值 *********************************
function regionSal = computeRegionSal(refImage,pixelList)
regionSal = zeros(length(pixelList),1);

for i=1:length(pixelList)
    regionSal(i,1) = mean(refImage(pixelList{i,1}));
end
regionSal = normalizeSal(regionSal);

clear refImage pixelList
end

%3 Bid剔除前景点
function result = abortFGpoints(indexFg,bdIds)
indexFG_IN_BGs = [];
for ii=1:length(indexFg)
    tmpFG = indexFg(ii);% 前景点于所有区域的标号
    indexFG_IN_BG = find(bdIds == tmpFG);
    indexFG_IN_BGs = [indexFG_IN_BGs;indexFG_IN_BG];
end
bdIds(indexFG_IN_BGs) = [];
result = bdIds;
end
