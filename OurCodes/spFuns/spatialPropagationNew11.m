% function [TPSPSAL,TPSPSALRegionSal] = spatialPropagationNew10(CURINFOR,IMSAL_TPSAL1,param,cur_image,GPsign)
function [TPSPSAL_Img,TPSPSAL_RegionSal] = spatialPropagationNew11(CURINFOR,IMSAL_TPSAL1,TPSAL1,param,cur_image,GPsign)
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
% V6: 2016.11.09 9:32am
% LOCAL-->GLOBAL + ITERATION
% 
% V7: 2016.11.14 16:27PM
% �����Դ˴������������
% TPSAL1 ��ʱ�򴫲���ĸ��߶��µ�����������ͼ��������
% 
% V8: 2016.11.15 16:43 PM
% GMR���ֲ���+ SOP��ȫ�֣�+ GMR���ֲ������ֲ�/ȫ������W����֮��ͬ
% 
% copyright by xiaofei zhou,shanghai university,shanghai,china
% zxforchid@163.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n this is spatial propagation process, wait a minute .........')
no_dims = param.no_dims;
bgRatio = param.bgRatio;
fgRatio = param.fgRatio;
sp_iternum = param.sp_iternum;
alpha=0.99;
theta=10;

[r,c] = size(IMSAL_TPSAL1);
% iterSal = IMSAL_TPSAL1;
ss = 1;% ����һ���߶�

%% A: ��Ѷ���� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
iterSal = TPSAL1{1,1};
iterSal_Img = IMSAL_TPSAL1;
for iter = 1:sp_iternum
    fprintf('\n the %d iteration ......',iter)
    tmpFEA       = CURINFOR.fea{ss,1};
    tmpSPinfor   = CURINFOR.spinfor{ss,1};% ���߶��µķָ��� 
    regionCenter = tmpSPinfor.region_center;
    adjcMatrix   = tmpSPinfor.adjcMatrix;
    bdIds        = tmpSPinfor.bdIds;
    spNum        = tmpSPinfor.spNum;
    
   %% 1 �����ںϺ��ʱ�򴫲�ͼ�񣬼����������� &&&&&&&&&&&&&&&&&&&&&&&&
    fprintf('\n obtain object-center ...\n')
    [rcenter,ccenter] = computeObjectCenter(iterSal_Img);% x-->row, y-->col  
    regionSal  = iterSal;
    regionDist = computeRegion2CenterDist(regionCenter,[rcenter,ccenter],[r,c]);
    init_compactness = computeCompactness(regionSal,regionDist);
    init_compactness = 1/(init_compactness);
    clear regionSal regionDist
    %% 1.1 ����λ����Ϣ &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    %% 2 �γɴ���������� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    fprintf('\n compute features ...\n')
    regionFea = [tmpFEA.colorHist_rgb,tmpFEA.colorHist_lab,tmpFEA.colorHist_hsv,...
               tmpFEA.lbp_top_Hist,tmpFEA.regionCov,tmpFEA.LM_textureHist,tmpFEA.flowHist];
%     regionFea = [tmpFEA.colorHist_rgb,tmpFEA.colorHist_lab,tmpFEA.colorHist_hsv];
           
    % regionFea_mappedA Ϊ���յ���������
%     regionFea_mappedA = regionFea;
    [regionFea_mappedA,regionFea_mapping] = pca(regionFea,no_dims);
%     ZZ         = repmat(sqrt(sum(regionFea_mappedA.*regionFea_mappedA)),[tmpSPinfor.spNum,1]);% ����ȫ�ֹ�һ�� 2016.10.28 9:32AM
%     ZZ(ZZ==0)  = eps;
%     regionFea_mappedA  = regionFea_mappedA./ZZ;
    FeaDist = GetDistanceMatrix(regionFea_mappedA);    
    
    clear regionFea_mapping ZZ regionFea 
    
    %% 3 contruct affinity matrix &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    fprintf('\n contruct affinity matrix ...\n')
    % 3.1 LOCAL ---------------------------------------------------------
    adjcMatrix_local = LinkNNAndBoundary2(adjcMatrix, bdIds);
%     FeaDist_location = GetDistanceMatrix(tmpFEA.flowHist);
%     W_location       = SetSmoothnessMatrix(FeaDist_location, adjcMatrix_local, theta);

    W_local          = SetSmoothnessMatrix(FeaDist, adjcMatrix_local, theta);
%     W_local          = W_location.*W_local;
    D_local          = diag(sum(W_local));
    optAff_local     = (D_local-alpha*W_local)\eye(spNum);
    optAff_local(1:spNum+1:end) = 0;% ������Խ���Ԫ�����㣡����
    
    % 3.2 GLOBAL --------------------------------------------------------
    knn=round(size(regionFea_mappedA,1)*1/15);mu=0.1;
    kdtree = vl_kdtreebuild(regionFea_mappedA');% ���� feaDim*sampleNum
    [indexs, distance] = vl_kdtreequery(kdtree,regionFea_mappedA',regionFea_mappedA', 'NumNeighbors', knn) ;
    indexs1 = indexs(2:end,:);
    adjcMatrix_global = zeros(spNum,spNum);
    for i=1:spNum  
        tmp_nozeros = indexs1(:,i);
        adjcMatrix_global(i,tmp_nozeros) = 1;
        adjcMatrix_global(tmp_nozeros,i) = 1;
    end
    adjcMatrix_global(bdIds, bdIds) = 1;
    W_global  = SetSmoothnessMatrix(FeaDist, adjcMatrix_global, theta);
    W_global  = W_global + adjcMatrix_global*mu;  %add regularization term
    D_global  = diag(sum(W_global));

    
    %% 4 GMR-LOCAL &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
     fprintf('\n GMR-Local ...\n')
     % ��ȷ���Եı���������Ϊ0 
     BGNUM = round(bgRatio*spNum);% ȷ���Ա��������ı���������
     tmp_iterSal = iterSal;
%      thresh1 = graythresh(tmp_iterSal);  %automatic threshold
%      tmp_iterSal(tmp_iterSal < 0.05) = 0;
%      tmp_iterSal(tmp_iterSal > 1.5*thresh1) = 1;

%      [valueBG,indexBG] = sort(tmp_iterSal);
%      BG_index = indexBG(1:BGNUM);
%      tmp_iterSal(BG_index) = 0;
     
     GMR_Local1 = optAff_local*tmp_iterSal;
     GMR_Local1 = normalizeSal(GMR_Local1);
     clear tmp_iterSal thresh1
     
     if 1
    %% 5 SOP-GLOBAL &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
     fprintf('\n SOP-Global ...\n')
     fgWeight = GMR_Local1;
     bgWeight = 1 - GMR_Local1;
     
     tmp_bgWeight = bgWeight;
     thresh2_1 = graythresh(tmp_bgWeight);  %automatic threshold
%      tmp_bgWeight(tmp_bgWeight < 0.05)          = 0;
     tmp_bgWeight(tmp_bgWeight > 1.5*thresh2_1) = 1000;
%      [valueSOP,indexSOP] = sort(tmp_bgWeight,'descend');
%      BG_index = indexSOP(1:BGNUM);
%      tmp_bgWeight(BG_index) = 1000;
     
     tmp_fgWeight = fgWeight;
%      thresh2_2 = graythresh(fgWeight);  %automatic threshold
     tmp_fgWeight(tmp_fgWeight < 0.05) = 0;
%      tmp_fgWeight(tmp_fgWeight > 1.5*thresh2_2) = 1;
     clear thresh2_1 thresh2_2
     
     bgLambda = 5;
     E_bg = diag(tmp_bgWeight * bgLambda);
     E_fg = diag(tmp_fgWeight);
     SOP_Global =(D_global - W_global + E_bg + E_fg) \ (E_fg * ones(spNum, 1));
     SOP_Global = normalizeSal(SOP_Global);
     clear GMR_Local1 E_bg E_fg tmp_fgWeight fgWeight bgWeight
     end
     
    %% 6 GMR-LOCAL &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
     fprintf('\n GMR-Local ...\n')
     % 6.1 ��ȷ���Ե�ǰ��������Ϊ1 ---------------------------------
     FGNUM = round(fgRatio*spNum);% ȷ����ǰ�������ı���������
%      SOP_Global = GMR_Local1;
     tmp_SOP_Global = SOP_Global;
%      thresh3 = graythresh(tmp_SOP_Global);  %automatic threshold
%      tmp_SOP_Global(tmp_SOP_Global < 0.05)        = 0;
%      tmp_SOP_Global(tmp_SOP_Global > 1.5*thresh3) = 1;

%      [valueFG,indexFG] = sort(tmp_SOP_Global,'descend');
%      FG_index = indexFG(1:FGNUM);
%      tmp_SOP_Global(FG_index) = 1;
     clear thresh3
     
     % 6.2 ʹ��mid-level feature���������� -------------------------
    FeaDistNew      = GetDistanceMatrix(tmp_SOP_Global); 
    W_localNew      = SetSmoothnessMatrix(FeaDistNew, adjcMatrix_local, theta);
%     W_localNew      = W_location.*W_localNew;
    D_localNew      = diag(sum(W_localNew));
    optAff_localNew = (D_localNew-alpha*W_localNew)\eye(spNum);
    optAff_localNew(1:spNum+1:end) = 0;% ������Խ���Ԫ�����㣡����
    
    % 6.3 ���� -----------------------------------------------------
     GMR_Local2 = optAff_localNew*tmp_SOP_Global;
     GMR_Local2 = normalizeSal(GMR_Local2);
     clear SOP_Global tmp_SOP_Global
     
    %% 7 fusion &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
     fprintf('\n integration initial and propagation ...\n')
    [PP_Img, ~]  = CreateImageFromSPs(GMR_Local2, tmpSPinfor.pixelList, r, c, true);
    [rcenter_PP,ccenter_PP] = computeObjectCenter(PP_Img);
    regionDist_PP = ...
        computeRegion2CenterDist(regionCenter,[rcenter_PP,ccenter_PP],[r,c]);
    PP_compactness = computeCompactness(GMR_Local2,regionDist_PP);
    PP_compactness = 1/(PP_compactness);
    
    winit = init_compactness/(init_compactness + PP_compactness);
    wpp   = PP_compactness  /(init_compactness + PP_compactness);
    
    tmp_iterSal = normalizeSal(winit*iterSal + wpp*GMR_Local2);
    [tmp_iterSal_Img, ~]  = CreateImageFromSPs(tmp_iterSal, tmpSPinfor.pixelList, r, c, true);
    clear PP_Img rcenter_PP ccenter_PP PP_compactness winit wpp
    
    %% 5 save & clear
    iterSal     = tmp_iterSal;
    iterSal_Img = tmp_iterSal_Img;
    
    clear tmp_iterSal tmp_iterSal_Img 
    clear LP_sal GP_sal GP_Img regionFea_mappedA regionFea_mapping 
end

%% B: �������ս�� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n assigenment the last result ...')
% [iterSal_Img, ~]  = CreateImageFromSPs(iterSal, tmpSPinfor.pixelList, r, c, true);
switch GPsign
    case 'YES'
         iterSal_Img     = graphCut_Refine(cur_image,iterSal_Img); 
         TPSPSAL_Img     = iterSal_Img;
    case 'NO'
         TPSPSAL_Img     = iterSal_Img;  
end
TPSPSAL_Img = normalizeSal(guidedfilter(TPSPSAL_Img,TPSPSAL_Img,5,0.1));

TPSPSAL_RegionSal = cell(length(CURINFOR.fea),1);% ���߶��µĽ��
for ss=1:length(CURINFOR.fea)
    tmpSPinfor   = CURINFOR.spinfor{ss,1};
    TPSPSAL_RegionSal{ss,1} = ...
        computeRegionSal(TPSPSAL_Img,tmpSPinfor.pixelList);% ���߶��µ�����������ֵ
    clear tmpSPinfor
end

clear CURINFOR IMSAL_TPSAL1 param cur_image GPsign

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% �Ӻ�������  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 2-hop & bb &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% ��������˳����2-hop�� Ȼ�������ܱ߽����� 2016.11.15 18:58PM
% �ٷ������������߽磬�� 2-hop;ͬ zhuwangjaing����һ�£����� 2016.11.16 10:31AM
function adjcMatrix = LinkNNAndBoundary2(adjcMatrix, bdIds)
%link boundary SPs
adjcMatrix(bdIds, bdIds) = 1;

%link neighbor's neighbor
adjcMatrix = (adjcMatrix * adjcMatrix + adjcMatrix) > 0;
adjcMatrix = double(adjcMatrix);

spNum = size(adjcMatrix, 1);
adjcMatrix(1:spNum+1:end) = 0;  %diagnal elements set to be zero
clear bdIds
end

% 2 ������ &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
function W = SetSmoothnessMatrix(colDistM, adjcMatrix_nn, theta)
allDists = colDistM(adjcMatrix_nn > 0);
maxVal = max(allDists);
minVal = min(allDists);

colDistM(adjcMatrix_nn == 0) = Inf;
colDistM = (colDistM - minVal) / (maxVal - minVal + eps);% �����һ��
W = exp(-colDistM * theta);
clear colDistM adjcMatrix_nn theta
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 6. ȫ�ִ���(ȥ���ռ���룬��Ϊ�����а�����λ����Ϣ) &&&&&&&&&&&&&&&&&&&&&&&&
% % ȥ������ 2016.11.09  13:35PM
% % compactness�ļ������ yuming fang�������� spatial variance 2016.11.15
% % 
% function [result_sal,result_Img] = ...
%     globalPropagation(regionFea,LP_sal,LP_compactness,tmpSPinfor,imgsize)
% r = imgsize(1);
% c = imgsize(2);
% spaSigma = 0.25;
% 
% % 6.1 propagate -----------------------------------------------------------
% %    kdNum = size(tmpfea,1);
%     knn=round(size(regionFea,1)*1/15);
%     kdtree = vl_kdtreebuild(regionFea');% ���� feaDim*sampleNum
%     [indexs, distance] = vl_kdtreequery(kdtree,regionFea',regionFea', 'NumNeighbors', knn) ;
%     distance1 = distance(2:end,:);% ������һ�У��������(knn-1)*sampleNum
%     indexs1 = indexs(2:end,:);
%     
%     alpha = 1/mean(distance1(:));
%     dist = exp(-alpha*distance1);
%     WIJ = dist./(repmat(sum(dist),[(knn-1),1])+eps);
%     GP_sal = sum(LP_sal(indexs1).*WIJ);
%     GP_sal = normalizeSal(GP_sal);
%     GP_sal = GP_sal';
% 
% % 6.2 fusion --------------------------------------------------------------
%     [GP_Img, ~]  = CreateImageFromSPs(GP_sal, tmpSPinfor.pixelList, r, c, true);
%     [rcenter_GP,ccenter_GP] = computeObjectCenter(GP_Img);
%     regionCenter = tmpSPinfor.region_center;
%     regionDist_GP = ...
%         computeRegion2CenterDist(regionCenter,[rcenter_GP,ccenter_GP],[r,c]);
%     GP_compactness = computeCompactness(GP_sal,regionDist_GP);
%     GP_compactness = 1/(GP_compactness+eps);
%     wGP   = GP_compactness/(GP_compactness + LP_compactness);
%     wLP   = LP_compactness/(GP_compactness + LP_compactness);
% 
% % 6.3 RESULT --------------------------------------------------------------
% result_sal = normalizeSal(wGP*GP_sal + wLP*LP_sal);
% %     result_sal = ...
% %         normalizeSal(GP_compactness*GP_sal + LP_compactness*LP_sal + ...
% %         0.5*(GP_compactness + LP_compactness)*(LP_sal.*GP_sal));
% [result_Img, ~]  = CreateImageFromSPs(result_sal, tmpSPinfor.pixelList, r, c, true);  
%     clear GP_Img GP_sal regionDist_GP GP_compactness
%     
% clear meanDist alpha WIJ
% clear indexs distance indexs1 distance1
% clear LP_sal regionFea kdtree
% end
% 
% % 7 �ֲ����� 2016.11.09  13:42PM &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% % compactness�ļ������ yuming fang�������� spatial variance 2016.11.15
% % 
% function [result_sal,result_compactness] = ...
%     localPropagation(regionFea,regionSal,init_compactness,tmpSPinfor,imgsize)
% % 7.1 initial -------------------------------------------------------------
% adjcMatrix = tmpSPinfor.adjcMatrix;
% spNum = size(adjcMatrix,1);
% r = imgsize(1);
% c = imgsize(2);
% 
%     adjcMatrix1 = adjcMatrix;
%     adjcMatrix1(adjcMatrix1==2) = 1;
%     adjcMatrix1(1:spNum+1:end) = 0;
%     adjmat = full(adjcMatrix1); % ����������   
%     clear adjcMatrix1 adjcMatrix
% 
% % 7.2 propagate -----------------------------------------------------------
%     LP_Sal = zeros(spNum,1);
%     for ii=1:spNum
%         tmpAdj = adjmat(ii,:);
%         adjIndex = find(tmpAdj==1);
%         
%         tmpFea = regionFea(ii,:);
%         tmpFea_adj = regionFea(adjIndex,:);
%         feadiff = repmat(tmpFea,[length(adjIndex),1]) - tmpFea_adj;
%         feadiff = sqrt(sum(feadiff.*feadiff,2));% size(adjsetfea,1)*1
%         alpha_fea = 2/(mean(feadiff(:))+eps);
%         feadiff = exp(-alpha_fea*feadiff);
%         
%         SAL_adj = regionSal(adjIndex,:);
%         wij = feadiff/(sum(feadiff(:))+eps);
%         LP_Sal(ii,1) = sum(wij.*SAL_adj);
%         
%         clear SAL_adj wij tmpFea_adj feadiff
%     end
%     LP_Sal  = normalizeSal(LP_Sal);
% % 7.3 fusion & compute the compactness of result sal ----------------------
%     [LP_Img, ~]  = CreateImageFromSPs(LP_Sal, tmpSPinfor.pixelList, r, c, true);
%     [rcenter_LP,ccenter_LP] = computeObjectCenter(LP_Img);
%     regionCenter = tmpSPinfor.region_center;
%     regionDist_LP = ...
%         computeRegion2CenterDist(regionCenter,[rcenter_LP,ccenter_LP],[r,c]);
%     LP_compactness = computeCompactness(LP_Sal,regionDist_LP);
%     LP_compactness = 1/(LP_compactness+eps);% note!!!
%     wlp = LP_compactness/(LP_compactness+init_compactness);
%     winit = init_compactness/(LP_compactness+init_compactness);
%     result_sal = normalizeSal(wlp*LP_Sal + winit*regionSal);
%     
%     [result_Img, ~]  = CreateImageFromSPs(result_sal, tmpSPinfor.pixelList, r, c, true);
%     [rcenter_result,ccenter_result] = computeObjectCenter(result_Img);
%     result_regionDist = ...
%         computeRegion2CenterDist(regionCenter,[rcenter_result,ccenter_result],[r,c]);
%     result_compactness = computeCompactness(result_sal,result_regionDist);
%     result_compactness = 1/(result_compactness+eps);
%     
%     clear result_Img result_regionDist
%     clear adjcMatrix regionFea regionSal init_compactness tmpSPinfor imgsize
% 
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
