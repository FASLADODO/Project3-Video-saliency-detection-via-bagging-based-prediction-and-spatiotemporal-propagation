function [TPSAL1,IMSAL_TPSAL1] =  ...
    integrate_Boost_TP_SAL(SAL1,SAL2,spinforCur,height,width)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 用于融合 BoostSal & TPSal
% 2016.11.18 20:47PM
% Sal1 & Sal2 显著性图
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMSAL_TPSAL1 = 0;% 时域传播后的各尺度平均后的像素级显著性图
TPSAL1 = cell(length(SAL1),1); % 新的时域传播后的各尺度下的区域显著性值
for ss=1:length(SAL1)
    
    % 1 initial ***********************************************************
    tmpSPinfor   = spinforCur{ss,1};
    
    tmp_SAL1_RegionSal   = SAL1{ss,1}.SalValue;
    tmp_SAL1_Compactness = SAL1{ss,1}.compactness;
 
    tmp_SAL2_RegionSal   = SAL2{ss,1}.SalValue;
    tmp_SAL2_Compactness = SAL2{ss,1}.compactness;
    
    % 2 compute compactness ***********************************************
    wSAL1    = tmp_SAL1_Compactness/(tmp_SAL1_Compactness + tmp_SAL2_Compactness);
    wSAL2    = tmp_SAL2_Compactness/(tmp_SAL1_Compactness + tmp_SAL2_Compactness);    
    if wSAL1<0.2
        wSAL1 = 0;
%         tmp_SAL1_Compactness = 0;
    end
    
    if wSAL2<0.2
        wSAL2 = 0;
%         tmp_SAL2_Compactness = 0;
    end

    wSAL1 = 1;wSAL2 = 1;
    % 3 integration *******************************************************
    TPSAL_regional = normalizeSal(wSAL1*tmp_SAL1_RegionSal + wSAL2*tmp_SAL2_RegionSal);
    TPSAL1{ss,1}.SalValue = TPSAL_regional;
    
    % 4 compute result compactness ****************************************
    [tmp_IMSAL_TPSAL, ~] =  ...
        CreateImageFromSPs(TPSAL_regional, tmpSPinfor.pixelList, height, width, true);
    TPSAL1{ss,1}.PP_Img  = tmp_IMSAL_TPSAL;
    IMSAL_TPSAL1 = IMSAL_TPSAL1 + tmp_IMSAL_TPSAL;
    
    regionCenter    = tmpSPinfor.region_center;
    [rcenter_sal,ccenter_sal] = computeObjectCenter(tmp_IMSAL_TPSAL);
    regionDist_sal = computeRegion2CenterDist(regionCenter,[rcenter_sal,ccenter_sal],[height,width]);
    tmp_Sal_compactness = computeCompactness(TPSAL_regional,regionDist_sal);
    TPSAL1{ss,1}.compactness = 1/tmp_Sal_compactness; 
    clear tmp_Sal_compactness regionCenter rcenter_sal ccenter_sal regionDist_sal 
    
    clear tmpSPinfor tmp_SAL1_RegionSal tmp_SAL1_Compactness tmp_SAL2_RegionSal tmp_SAL2_Compactness
    clear TPSAL_regional tmp_IMSAL_TPSAL
end
IMSAL_TPSAL1 = normalizeSal(IMSAL_TPSAL1);


clear SAL1 SAL2 spinforCur height width
end