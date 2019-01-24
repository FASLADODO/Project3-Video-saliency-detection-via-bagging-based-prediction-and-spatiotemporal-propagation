function SALS = spatialPropagationNew2(TPSAL,CURINFOR,image,flow,beta,IMSAL_TPSAL1)
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
% copyright by xiaofei zhou,shanghai university,shanghai,china
% zxforchid@163.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPSCALENUM = length(CURINFOR.fea);
SALS = cell(SPSCALENUM,1);
betaindexs  = beta(:,2);
betavalues = beta(:,1);
[value,index] = max(betavalues);
FEA_ID = betaindexs(index(end));% ��ʹ�����������
alpha=0.99;
theta=10;
AL = 0.75;AH=1.25; % ѡȡȷ�����������ڴ�����ǰ�����ڱ�����������������ǰ������
for ss=1:SPSCALENUM
    %% 1 initial -----------------------------------------------------------
    tmpSPinfor = CURINFOR.spinfor{ss,1};% ���߶��µķָ��� 
    regionSal = TPSAL{ss,1};
    tmpORlabel = CURINFOR.ORLabels{ss,1};
    tmpFea_cur = CURINFOR.fea{ss,1};
    d1_cur = [tmpFea_cur.colorHist_rgb];
    d2_cur = [tmpFea_cur.colorHist_lab];
    d3_cur = [tmpFea_cur.colorHist_hsv];
    d4_cur = [tmpFea_cur.lbpHist];
    d5_cur = [tmpFea_cur.hogHist];
    d6_cur = [tmpFea_cur.regionCov];
    d7_cur = [tmpFea_cur.geoDist];
    d8_cur = [tmpFea_cur.flowHist];
    
    spNum = tmpSPinfor.spNum;
    
    
    %% 2 constrcut new Graph -----------------------------------------------
     adjcMatrix = tmpSPinfor.adjcMatrix;
     
    % 2.1 obtain bdIds
     ISORlabel        = tmpORlabel(:,1);
     ISBorderlabel    = tmpORlabel(:,2);
     [index_out_OR,~] = find(ISORlabel==0);% OR������
     [index_in_OR,~]  = find(ISORlabel==1);% OR��������
     adjcMatrix(index_out_OR,:) = [];
     adjcMatrix(:,index_out_OR) = [];   
     index_bdIds = [];% border �� OR �е����
     for dd=1:length(index_in_OR)% Ѱ��border,�����¹��������� 1,1
         tmpID = index_in_OR(dd);
         if ISBorderlabel(tmpID)==1
             index_bdIds = [index_bdIds;dd];
         end
     end
     bdIds = index_bdIds;
    
    % 2.3 ����������
    dall = eval(['d' num2str(FEA_ID) '_cur']);
    d_in_OR = dall(index_in_OR,:);
    FeaDist = GetDistanceMatrix(d_in_OR);

    %% 3 �������Ż� ---------------------------------------------------------
    % 3.1 TPGT
    regionSal_in_OR = regionSal(index_in_OR);
    regionSal_in_OR = normalizeSal(regionSal_in_OR);
    pixelnums = [];tmpregionSal = [];
    for pp=1:length(index_in_OR)
        tmppixelnum = tmpSPinfor.pixelList{index_in_OR(pp),1}; 
        tmpregionSal = [tmpregionSal;mean(mean(IMSAL_TPSAL1(tmppixelnum)))];
        pixelnums = [pixelnums;length(tmppixelnum)];
    end
    meanSal = sum(pixelnums.*regionSal_in_OR)/sum(pixelnums);
    
    % 3.3 TPR  �Ż� saliency optimization
    fgWeight = 1.25*tmpregionSal;
    bgWeight = 0.75*tmpregionSal;
%     fgWeight = 1.25*regionSal_in_OR;
%     bgWeight = 0.75*(1 - regionSal_in_OR);
%     fgWeight = TPP;
%     bgWeight = TPP0; 
    [clipVal, geoSigma, neiSigma] = EstimateDynamicParas(adjcMatrix, FeaDist);
    TPR = SaliencyOptimization(adjcMatrix, bdIds, FeaDist, neiSigma, bgWeight, fgWeight);
    TPR = normalizeSal(TPR);
    
    % 3.4 SPP ǰ������
    alpha=0.99;
    theta=10;
    spNum0 = size(adjcMatrix, 1);
    adjcMatrix_nn = LinkNNAndBoundary2(adjcMatrix, bdIds); 
    W = SetSmoothnessMatrix(FeaDist, adjcMatrix_nn, theta);
    D = diag(sum(W));
    optAff =(D-alpha*W)\eye(spNum0);
    optAff(1:spNum0+1:end) = 0;  %set diagonal elements to be zero
    meanSal_TPR = sum(pixelnums.*TPR)/sum(pixelnums);
    TPR_GT = TPR >= AH*meanSal_TPR;
%     SPP = optAff*TPR_GT;
    TPGTL = regionSal_in_OR <= AL*meanSal;
    TPGTH = regionSal_in_OR > AH*meanSal;
    SPPH = optAff*TPGTH;
    SPPL = optAff*TPGTL;
    SPPH = normalizeSal(SPPH);
    SPPL = normalizeSal(SPPL);
    SPP = SPPH+SPPL;
    SPP = normalizeSal(SPP);
    SPP = SPPH;
    %% 4 OR--->ALL ---------------------------------------------------------
    GPSAL = zeros(spNum,1);
    gg1 = 1;
    for gg=1:spNum
        if ismember(gg,index_out_OR)
            GPSAL(gg,1) = 0;
        else
            GPSAL(gg,1) = SPP(gg1,1);
            gg1 = gg1 + 1;
        end
    end
    
    %% 5 SAVE --------------------------------------------------------------
    SALS{ss,1}.GPSAL = GPSAL;
    
    clear SPP GPSAL
end



end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 ������
function W = SetSmoothnessMatrix(colDistM, adjcMatrix_nn, theta)
allDists = colDistM(adjcMatrix_nn > 0);
maxVal = max(allDists);
minVal = min(allDists);

colDistM(adjcMatrix_nn == 0) = Inf;
colDistM = (colDistM - minVal) / (maxVal - minVal + eps);
W = exp(-colDistM * theta);
end

% 2 2-hop & bb
function adjcMatrix = LinkNNAndBoundary2(adjcMatrix, bdIds)
%link boundary SPs
adjcMatrix(bdIds, bdIds) = 1;

%link neighbor's neighbor
adjcMatrix = (adjcMatrix * adjcMatrix + adjcMatrix) > 0;
adjcMatrix = double(adjcMatrix);

spNum = size(adjcMatrix, 1);
adjcMatrix(1:spNum+1:end) = 0;  %diagnal elements set to be zero
end