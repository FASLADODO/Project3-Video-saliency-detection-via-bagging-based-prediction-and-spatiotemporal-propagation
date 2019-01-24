function SALS = temporalPropagationNew2(CURINFOR,PREINFOR,FullResultCurSal,betas,ratio)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ����ǰ���õ��ĵ�ǰ֡���߶���ÿһ�������mapsets��������һ֡����ƥ������
% ���ݸ���������Ȩ�أ�����Ӧ�Ĺ���Ȩ��
% mapsets OR_CUR--->OR_PRE
% CURINFOR
% fea/ORlabels/spinfor(mapsets(correSets/correSets_dist)��region_center_prediction)
%
% PREINFOR
% spsal/fea/ORlabels/spinfor
% 
% ���� fea
%     fea{ss,1}.colorHist_rgb
%     fea{ss,1}.colorHist_lab
%     fea{ss,1}.colorHist_hsv
%     fea{ss,1}.lbpHist  
%     fea{ss,1}.hogHist  
%     fea{ss,1}.regionCov  
%     fea{ss,1}.geoDist    
%     fea{ss,1}.flowHist   
%     
% FullResultCurSal{ss,1}.FullValue ���������������ֵ
% FullResultCurSal{ss,1}.FullLabel ��������ı�ǩ��object/background��
% beta ������Ȩ�� & ��Ӧ���
% SALS ʱ�򴫲���ĸ������������ͼ
% 
% V2: 2016.08.18
% V3: 2016.08.26 15:15PM
% ��temporalPropagation�����Ͻ��е��޸�,ȫ�ߴ����
% NOTE:����ȷ����ǰ֡OR�е�������ǰһ֡�е����ƥ��������λ��OR��
%      OR��������������ֵ��������������ʱ�򴫲�
% 
% V4��2016.08.31 14��57PM
% NOTE: ���������õĻ�ֵ tmpsalpre = PREINFOR.spsal{ss,1}; ����Ҫ������
%       Ϊǰһ֡���߶��¸������������ֵ ������
% 
% V5: 2016.10.09 16:49PM
% ������֡���������ص��������������� area_similar 0~1
% �� pos_diff ��Ϊ pos_similar 0~1
% �� fea_diff ��Ϊ fea_similar 0~1
%
% V6: 2016.10.31 20:49PM
% betas Ϊ���߶��¾���һ��beta��������Ҫע�⣡����
% 
% V7: 2016.11.02 11:05AM
% ����LBP-TOP��������Ϊ���������������˳��;��9������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [height,width] = size(CURINFOR.spinfor{1,1}.idxcurrImage);
SPSCALENUM = length(CURINFOR.fea);
SALS = cell(SPSCALENUM,1);

% % ������������Ȩ�ؼ����
% indexs  = beta(:,2);
% weights = beta(:,1);
% weights = weights/sum(weights);

%% 
[height,width]  = size(PREINFOR.imsal);
normDist = sqrt((height.^2 + width.^2));
for ss=1:SPSCALENUM
    %% 1. initialization ----------------------------------------------------
    % ������������Ȩ�ؼ����
    beta = betas{ss,1};
    indexs  = beta(:,2);
    weights = beta(:,1);
    weights = weights/sum(weights);
    
    % ��߶���Ϣ
    tmpSPcur = CURINFOR.spinfor{ss,1};
    regionCenter = tmpSPcur.region_center;
%     tmpSPpre = PREINFOR.spinfor{ss,1};

    % ����
    tmpFea_pre = PREINFOR.fea{ss,1};
    tmpFea_cur = CURINFOR.fea{ss,1};
    d1_pre  = [tmpFea_pre.colorHist_rgb]; d1_cur  = [tmpFea_cur.colorHist_rgb];
    d2_pre  = [tmpFea_pre.colorHist_lab]; d2_cur  = [tmpFea_cur.colorHist_lab];
    d3_pre  = [tmpFea_pre.colorHist_hsv]; d3_cur  = [tmpFea_cur.colorHist_hsv];
%     d4_pre  = [tmpFea_pre.LM_texture];    d4_cur  = [tmpFea_cur.LM_texture];
%     d4_pre = [tmpFea_pre.lbpHist];      d4_cur = [tmpFea_cur.lbpHist];
    d4_pre  = [tmpFea_pre.lbp_top_Hist];  d4_cur  = [tmpFea_cur.lbp_top_Hist];
%     d6_pre  = [tmpFea_pre.hogHist];       d6_cur  = [tmpFea_cur.hogHist];
    d5_pre  = [tmpFea_pre.regionCov];     d5_cur  = [tmpFea_cur.regionCov];
    d6_pre  = [tmpFea_pre.LM_textureHist];d6_cur  = [tmpFea_cur.LM_textureHist];
%     d9_pre  = [tmpFea_pre.geoDist];       d9_cur  = [tmpFea_cur.geoDist];
    d7_pre = [tmpFea_pre.flowHist];       d7_cur = [tmpFea_cur.flowHist];
    clear tmpFea_pre tmpFea_cur
   
    % ӳ����Ϣ
    tmp_MAPSET = tmpSPcur.mapsets;% correSets correSets_dist

    % ȫ�ߴ��boost result
%     tmpFullresult = normalizeSal(FullResultCurSal{ss,1});
    tmpFullresult = FullResultCurSal{ss,1};% revised in 2016.10.24 22:11PM
    
    % preparation
    tmpsalpre = normalizeSal(PREINFOR.spsal{ss,1});% ǰһ֡ss �߶��¸������������ֵ(�˴�����Ҫ������)
    tmpsalcur = zeros(tmpSPcur.spNum,length(weights));% ���߶��µ�ǰ֡�ĸ�����������ֵ
    
    %% 2. begin -------------------------------------------------------------
    for sp=1:tmpSPcur.spNum
        SIGNS = tmp_MAPSET{sp,1}.correSets;
        if isempty(SIGNS)% ������λ��OR�⣬����������
        tmpsalcur(sp,:) = repmat(tmpFullresult.SalValue(sp,1),[1,length(weights)]);
        else
        % 2.1 ��ؼ�����Ϣ����š����롢������
        tmp_correSets         = tmp_MAPSET{sp,1}.correSets;
        tmp_correSets_dist    = tmp_MAPSET{sp,1}.correSets_dist;% ����λ��
        tmp_correSets_overlap = tmp_MAPSET{sp,1}.correSets_overlap;
        tmp_sals              = tmpsalpre(tmp_correSets,1)';
        
        tmp_correSets_dist = tmp_correSets_dist./normDist;% ���öԽ��߾�����й�һ�� 0~1
        tmp_correSets_dist(isnan(tmp_correSets_dist)) = 0;
        alpha_pos = 2/(mean(tmp_correSets_dist(:))+eps);
        pos_diff = exp(-alpha_pos*tmp_correSets_dist);
        
        % 2.2 ��ؼ�����������������
        % d1_pre_corre: sampleCorre*feadim      d1_cur_sp: 1*feadim
        d1_pre_corre = d1_pre(tmp_correSets,:);   d1_cur_sp = d1_cur(sp,:);
        d2_pre_corre = d2_pre(tmp_correSets,:);   d2_cur_sp = d2_cur(sp,:);
        d3_pre_corre = d3_pre(tmp_correSets,:);   d3_cur_sp = d3_cur(sp,:);
        d4_pre_corre = d4_pre(tmp_correSets,:);   d4_cur_sp = d4_cur(sp,:);
        d5_pre_corre = d5_pre(tmp_correSets,:);   d5_cur_sp = d5_cur(sp,:);
        d6_pre_corre = d6_pre(tmp_correSets,:);   d6_cur_sp = d6_cur(sp,:);
        d7_pre_corre = d7_pre(tmp_correSets,:);   d7_cur_sp = d7_cur(sp,:);
%         d8_pre_corre = d8_pre(tmp_correSets,:);   d8_cur_sp = d8_cur(sp,:);
%         d9_pre_corre = d9_pre(tmp_correSets,:);   d9_cur_sp = d9_cur(sp,:);
%         d10_pre_corre = d10_pre(tmp_correSets,:); d10_cur_sp = d10_cur(sp,:);
        
        
        % 2.3 ����beta֮index��weigh,����Ȩ������ֵ
        delta = 0;
        for ii=1:length(indexs)
            tmpindex = indexs(ii,1);
            tmpweights = weights(ii,1);
            ds = eval(['d' num2str(tmpindex) '_cur_sp']);
            dc = eval(['d' num2str(tmpindex) '_pre_corre']);
            
            % revised in 2016.08.30 14:43PM -------------------------
            alld = [ds;dc];
            alld(isnan(alld)) = 0;
            
            ds = alld(1,:);
            dc = alld(2:end,:);
            
            fea_diff = repmat(ds,[size(dc,1),1]) - dc;
            fea_diff = sqrt(sum(fea_diff.*fea_diff,2));
            alpha = 2/(mean(fea_diff(:))+eps);
            fea_diff = exp(-alpha*fea_diff);
                        
            diff = fea_diff' .* pos_diff;
            if sum(diff)==0
                tmpDiffSal = (diff*tmp_sals')/(sum(diff)+eps);
            else
                tmpDiffSal = (diff*tmp_sals')/sum(diff);
            end
%             delta = delta + tmpweights * tmpDiffSal;
            tmpsalcur(sp,ii) = tmpDiffSal;
            clear tmpindex tmpweights ds dc fea_diff diff
        end
%          tmpsalcur(sp,1) = ratio * delta + tmpFullresult(sp,1);
%         tmpsalcur(sp,1) = delta + ratio*tmpFullresult(sp,1);
        
        % 2.4 clear
        clear tmp_correSets tmp_correSets_dist tmp_sals pos_diff
        clear d1_pre_corre d2_pre_corre d3_pre_corre d4_pre_corre 
        clear d5_pre_corre d6_pre_corre d7_pre_corre d8_pre_corre d9_pre_corre d10_pre_corre
        clear d1_cur_sp d2_cur_sp d3_cur_sp d4_cur_sp
        clear d5_cur_sp d6_cur_sp d7_cur_sp d8_cur_sp d9_cur_sp d10_cur_sp
        end
    end
    
    %% fusion  ---------------------------------------------------------------------
    tmpsalcur = tmpsalcur.*repmat(weights',[size(tmpsalcur,1),1]);%weights'
    tmpsalcur = sum(tmpsalcur,2);
    tmpsalcur = normalizeSal(tmpsalcur);% ��һ��
    
    % ����ӳ�䴫���Ľ����compactness 2016.11.15 13:29PM
    [PP_Img, ~] = CreateImageFromSPs(tmpsalcur, tmpSPcur.pixelList, height, width, true);
    [rcenter_PP,ccenter_PP] = computeObjectCenter(PP_Img);
    regionDist_PP = computeRegion2CenterDist(regionCenter,[rcenter_PP,ccenter_PP],[height,width]);
    PP_compactness = computeCompactness(tmpsalcur,regionDist_PP);
    SALS{ss,1}.PP_Img = PP_Img; clear PP_Img
    SALS{ss,1}.SalValue = tmpsalcur; clear tmpsalcur rcenter_PP ccenter_PP
    SALS{ss,1}.compactness = 1/(PP_compactness);clear PP_compactness regionDist_PP
    
    clear tmpSPcur tmp_MAPSET tmpFullresult tmpsalpre tmpsalcur
    clear d1_pre d2_pre d3_pre d4_pre d5_pre d6_pre d7_pre d8_pre d9_pre d10_pre
    clear d1_cur d2_cur d3_cur d4_cur d5_cur d6_cur d7_cur d8_cur d9_cur d10_cur
end

clear CURINFOR PREINFOR FullResultCurSal beta
end
