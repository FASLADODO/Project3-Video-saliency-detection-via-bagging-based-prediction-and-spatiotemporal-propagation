function [TPSAL1,IMSAL_TPSAL1] =  ...
    integrate_Boost_TP_SAL1(SAL1,SAL2,SAL3,spinforCur,height,width)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% �����ں� BoostSal & TPSal
% 2016.11.18 20:47PM
% SAL1/SAL2/SAL3 ��ͬ��������ͼ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMSAL_TPSAL1 = 0;% ʱ�򴫲���ĸ��߶�ƽ��������ؼ�������ͼ
TPSAL1 = cell(length(SAL1),1); % �µ�ʱ�򴫲���ĸ��߶��µ�����������ֵ
for ss=1:length(SAL1)
    
    % 1 initial ***********************************************************
    tmpSPinfor   = spinforCur{ss,1};
    
    tmp_SAL1_RegionSal   = SAL1{ss,1}.SalValue;
    tmp_SAL1_Compactness = SAL1{ss,1}.compactness;
 
    tmp_SAL2_RegionSal   = SAL2{ss,1}.SalValue;
    tmp_SAL2_Compactness = SAL2{ss,1}.compactness;
    
    tmp_SAL3_RegionSal   = SAL3{ss,1}.SalValue;
    tmp_SAL3_Compactness = SAL3{ss,1}.compactness;
    
    % 2 compute compactness ***********************************************
    wSAL1 = tmp_SAL1_Compactness/(tmp_SAL1_Compactness + tmp_SAL2_Compactness + tmp_SAL3_Compactness);
    wSAL2 = tmp_SAL2_Compactness/(tmp_SAL1_Compactness + tmp_SAL2_Compactness + tmp_SAL3_Compactness);
    wSAL3 = tmp_SAL3_Compactness/(tmp_SAL1_Compactness + tmp_SAL2_Compactness + tmp_SAL3_Compactness);   
    if wSAL1<0.2
        wSAL1 = 0;
    end
    if wSAL2<0.2
        wSAL2 = 0;
    end
    if wSAL3<0.2
        wSAL3 = 0;
    end
    wSAL1 = 1;wSAL2=1;wSAL3=1;
    % 3 integration *******************************************************
    TPSAL_regional = normalizeSal(wSAL1*tmp_SAL1_RegionSal + wSAL2*tmp_SAL2_RegionSal + wSAL3*tmp_SAL3_RegionSal);
    TPSAL1{ss,1}.SalValue = TPSAL_regional;
    
    % 4 compute result compactness ****************************************
    [tmp_IMSAL_TPSAL, ~] =  ...
        CreateImageFromSPs(TPSAL_regional, tmpSPinfor.pixelList, height, width, true);
    TPSAL1{ss,1}.PP_Img  = tmp_IMSAL_TPSAL;
    IMSAL_TPSAL1 = IMSAL_TPSAL1 +tmp_IMSAL_TPSAL;
    
    regionCenter    = tmpSPinfor.region_center;
    [rcenter_sal,ccenter_sal] = computeObjectCenter(tmp_IMSAL_TPSAL);
    regionDist_sal = computeRegion2CenterDist(regionCenter,[rcenter_sal,ccenter_sal],[height,width]);
    tmp_Sal_compactness = computeCompactness(TPSAL_regional,regionDist_sal);
    TPSAL1{ss,1}.compactness = 1/tmp_Sal_compactness; 
    clear tmp_Sal_compactness regionCenter rcenter_sal ccenter_sal regionDist_sal 
    
    clear tmp_IMSAL_TPSAL tmp_SAL1_RegionSal tmp_SAL2_RegionSal tmp_SAL3_RegionSal
    clear tmp_SAL1_Compactness tmp_SAL2_Compactness tmp_SAL3_Compactness
end
IMSAL_TPSAL1 = normalizeSal(IMSAL_TPSAL1);


clear SAL1 SAL2 SAL3 spinforCur height width
end