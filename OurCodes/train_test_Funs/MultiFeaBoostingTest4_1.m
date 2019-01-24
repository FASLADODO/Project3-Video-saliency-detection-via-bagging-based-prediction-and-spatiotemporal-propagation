% function [result,betas,initSalImg] = MultiFeaBoostingTest3_1(ORFEA, PRE_INFOR, model, param, spinfor,curImage,MVF_Foward_fn_f)
% function [result,betas,initSalImg] = MultiFeaBoostingTest3_1(ORFEA, imSal_pre0, model, param, spinfor)
function [result,initSalImg] = MultiFeaBoostingTest4_1(ORFEA, imSal_pre0, model, param, spinfor)
% ��������������ڽ϶������ϣ�
% 2016.12.13 
% 
SP_SCALE_NUM = length(ORFEA.selfFea);
result = cell(SP_SCALE_NUM,1);
% betas = cell(SP_SCALE_NUM,1);
[height,width,dims]  = size(imSal_pre0);

%% BEGIN &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
initSalImg = 0;% ��ʼԤ���������ؼ���
for ss=1:SP_SCALE_NUM %���߶��µ�Ԥ�� 
    %% 1 ��ȡOR�������Ӧ�����ظ��������ڻ�ȡ��ǩ -----------------
    tmpSP           = spinfor{ss,1};
    tmpPixellist    = tmpSP.pixelList;
    regionCenter    = tmpSP.region_center;
    
    % original
    selfFea         = ORFEA.selfFea{ss,1};
    
    
   %% 2 �������� -------------------------------------------------
   if param.numMultiContext % �� multi-context
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
    [sp_sal_data_mappedA] = scaleForSVM_corrected2(sp_sal_data,scalemap.MIN,scalemap.MAX,0,1);% ��һ��
    
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
% valueDISTS1 = exp(-2*valueDISTS1/meanDist);% ����ķ�����
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
   result{ss,1}.PP_Img   = SalValue_Img; % ���߶��µĶ�Ӧ�����ؼ�������ͼ
   result{ss,1}.compactness = 1/(Sal_compactness);
   initSalImg  = initSalImg + SalValue_Img;
   
   clear SalValue LABELS WW beta tmpPixellist SalValue_Img indexWorse Sal_compactness
end

initSalImg = normalizeSal(initSalImg);% ��߶�ƽ�������µ����ؼ�������ͼ

clear ORFEA imSal_pre0 model param spinfor
end
%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% 
% % 2 ���ݳ�ʼ�ںϽ����������߶��µ�������ֵ &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
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
% % 6 GT ֮����ӳ��
% % ����preGT����һ֡��ӳ��ͼresult(��ֵ��)
% function result = preGT_flow_mapping(fpre_GT,MVF)
% %% 1 self mapping ��ȡ�����x,y����
% objIndex = find(fpre_GT(:)==1);
% [height,width] = size(fpre_GT);
% [sy,sx] = ind2sub([height,width],objIndex);
% 
% %% 2 optical flow map(��ǰһ֡�� GT ���� ���� ����ӳ��)
% MVF = double(MVF);
% XX = MVF(:,:,1);
% YY = MVF(:,:,2);% ����object�Ĺ���ƫ��
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