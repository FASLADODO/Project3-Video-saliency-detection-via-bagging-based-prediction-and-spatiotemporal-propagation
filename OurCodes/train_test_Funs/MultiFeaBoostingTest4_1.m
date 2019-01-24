% function [result,betas,initSalImg] = MultiFeaBoostingTest3_1(ORFEA, PRE_INFOR, model, param, spinfor,curImage,MVF_Foward_fn_f)
% function [result,betas,initSalImg] = MultiFeaBoostingTest3_1(ORFEA, imSal_pre0, model, param, spinfor)
function [result,initSalImg] = MultiFeaBoostingTest4_1(ORFEA, imSal_pre0, model, param, spinfor)
% 引入随机抽样（于较多样本上）
% 2016.12.13 
% 
SP_SCALE_NUM = length(ORFEA.selfFea);
result = cell(SP_SCALE_NUM,1);
% betas = cell(SP_SCALE_NUM,1);
[height,width,dims]  = size(imSal_pre0);

%% BEGIN &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
initSalImg = 0;% 初始预测结果，像素级的
for ss=1:SP_SCALE_NUM %单尺度下的预测 
    %% 1 获取OR各区域对应的像素个数，用于获取标签 -----------------
    tmpSP           = spinfor{ss,1};
    tmpPixellist    = tmpSP.pixelList;
    regionCenter    = tmpSP.region_center;
    
    % original
    selfFea         = ORFEA.selfFea{ss,1};
    
    
   %% 2 集成特征 -------------------------------------------------
   if param.numMultiContext % 有 multi-context
       multiContextFea = ORFEA.multiContextFea{ss,1};
       sp_sal_data = [selfFea.regionFea,multiContextFea.regionFea];
   else
       sp_sal_data = [selfFea.regionFea];   
   end
   clear selfFea multiContextFea
   
   %% 3 testing &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
%    param.predictN = length(model);
   SALS = zeros(tmpSP.spNum,param.predictN);
   for kk=1:param.predictN
    segment_saliency_regressor = model{kk,1}.dic;
    scalemap                   = model{kk,1}.scalemap;
    [sp_sal_data_mappedA] = scaleForSVM_corrected2(sp_sal_data,scalemap.MIN,scalemap.MAX,0,1);% 归一化
    
    sp_sal_prob = regRF_predict( sp_sal_data_mappedA, segment_saliency_regressor );
    tmpSal = normalizeSal(sp_sal_prob);
    SALS(:,kk) = tmpSal;
    clear sp_sal_prob sp_sal_data_mappedA segment_saliency_regressor scalemap
    clear tmpSal
     % salquality
   end
% % compute correDist ------------------------------------
% DISTS = computeCorreDist(SALS);
% [valueDISTS,indexDISTS] = sort(DISTS,'ascend');
% TOPNUMS = round((param.predictN)*2/3);
% indexDISTS1 = indexDISTS(1:TOPNUMS);
% valueDISTS1 = valueDISTS(1:TOPNUMS);
% meanDist = mean(valueDISTS1(:));
% if meanDist==0
%     meanDist = meanDist + eps;
% end
% valueDISTS1 = exp(-2*valueDISTS1/meanDist);% 距离的反比例
% normFactor = sum(valueDISTS1(:));
% if normFactor==0
%     normFactor = normFactor + eps;
% end
% valueDISTS1 = valueDISTS1/normFactor;
% PPSal = SALS(:,indexDISTS1) .* repmat(valueDISTS1',[tmpSP.spNum,1]);
% PPSal = sum(PPSal,2);
% SalValue = normalizeSal(PPSal);
% clear DISTS valueDISTS indexDISTS indexDISTS1 valueDISTS1 PPSal SALS
SalValue = sum(SALS,2);
SalValue = normalizeSal(SalValue);
clear SALS
%% integration ---------------------------------------------------
   [SalValue_Img, ~] = CreateImageFromSPs(SalValue, tmpPixellist, height, width, true);
   [rcenter_sal,ccenter_sal] = computeObjectCenter(SalValue_Img);
   regionDist_sal = computeRegion2CenterDist(regionCenter,[rcenter_sal,ccenter_sal],[height,width]);
   Sal_compactness = computeCompactness(SalValue,regionDist_sal);
   
   
   OG_Label = zeros(size(SalValue,1),1);
  
   result{ss,1}.OG_Label = OG_Label;
   result{ss,1}.SalValue = SalValue;
   result{ss,1}.PP_Img   = SalValue_Img; % 单尺度下的对应的像素级显著性图
   result{ss,1}.compactness = 1/(Sal_compactness);
   initSalImg  = initSalImg + SalValue_Img;
   
   clear SalValue LABELS WW beta tmpPixellist SalValue_Img indexWorse Sal_compactness
end

initSalImg = normalizeSal(initSalImg);% 多尺度平均意义下的像素级显著性图

clear ORFEA imSal_pre0 model param spinfor
end
%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% 
% % 2 根据初始融合结果，计算各尺度下的显著性值 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% function regionSal = computeRegionSal(refImage,pixelList)
% regionSal = zeros(length(pixelList),1);
% 
% for i=1:length(pixelList)
%     regionSal(i,1) = mean(refImage(pixelList{i,1}));
% end
% regionSal = normalizeSal(regionSal);
% 
% clear refImage pixelList
% end
% 
% % 6 GT 之光流映射
% % 返回preGT于下一帧的映射图result(二值的)
% function result = preGT_flow_mapping(fpre_GT,MVF)
% %% 1 self mapping 获取物体的x,y坐标
% objIndex = find(fpre_GT(:)==1);
% [height,width] = size(fpre_GT);
% [sy,sx] = ind2sub([height,width],objIndex);
% 
% %% 2 optical flow map(由前一帧的 GT 经过 光流 进行映射)
% MVF = double(MVF);
% XX = MVF(:,:,1);
% YY = MVF(:,:,2);% 计算object的光流偏移
% avgFlow(1) = mean(mean(XX(objIndex)));
% avgFlow(2) = mean(mean(YY(objIndex)));
% sxNew = round(sx + avgFlow(1));
% syNew = round(sy + avgFlow(2));
% 
% % remove flow ouside of image
% tmp = (sxNew>=3 & sxNew<=width-3) & (syNew>=3 & syNew<=height-3);
% sxNew = sxNew(tmp); syNew = syNew(tmp);
%  
% % construct boundingBox
% result = zeros(height,width);
% for ii=1:length(sxNew)
%     result(syNew(ii),sxNew(ii)) = 1;
% end
% 
% clear fpre_GT MVF_Foward_fn_f
% 
% end