% function [SALS] = temporalPropagationNew1(CURINFOR,PREINFOR,FullResultCur,betas)
% function [SALS] = temporalPropagationNew4(CURINFOR,PREINFOR,FullResultCur,model)
function [SALS,IMG] = temporalPropagationNew4_1_1(CURINFOR,PREINFOR,model)
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
%     fea{ss,1}.lbp_top_Hist
%     fea{ss,1}.hogHist  
%     fea{ss,1}.regionCov  
%     fea{ss,1}.geoDist    
%     fea{ss,1}.flowHist   
%     
% FullResultCur{ss,1}.FullValue ���������������ֵ
% FullResultCur{ss,1}.FullLabel ��������ı�ǩ��object/background��
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
%
% V8�� 2016.11.14 15:49PM
% SALS �������߶��¶�Ӧ������������ֵ����Ӧ��compactnesֵ
% �����Ǵ����Ľ���� ������������
% 
% V9: 2016.11.17 23:50PM
% �������ɭ�ֺ󣬲���ѹ�����������д���
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [height,width] = size(CURINFOR.spinfor{1,1}.idxcurrImage);
SPSCALENUM = length(CURINFOR.fea);
SALS = cell(SPSCALENUM,1);

% imSal_pre0 = PREINFOR.imsal;
[height,width]  = size(PREINFOR.imsal);
normDist = sqrt((height.^2 + width.^2));
ffNum = 2;
%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
IMG = 0;% ��߶��µ��ںϽ��
for ss=1:SPSCALENUM
%% 1. initialization &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&    
    % ��߶���Ϣ
    tmpSPcur = CURINFOR.spinfor{ss,1};
    tmpFea_pre = PREINFOR.fea{ss,1};
    data_pre_mappedA = [tmpFea_pre.regionFea];
    [data_pre_mappedA,scalemap] = scaleForSVM_corrected1(data_pre_mappedA,0,1);
    
    tmpFea_cur = CURINFOR.fea{ss,1};
    data_cur_mappedA = [tmpFea_cur.regionFea];
    [data_cur_mappedA,scalemap] = scaleForSVM_corrected1(data_cur_mappedA,0,1);
    
   
    % ӳ����Ϣ
    tmp_MAPSET = tmpSPcur.mapsets;% correSets correSets_dist

    % preparation
    tmpsalpre = normalizeSal(PREINFOR.spsal{ss,1});% ǰһ֡ss �߶��¸������������ֵ(�˴�����Ҫ������)
%     tmpsalcur = zeros(tmpSPcur.spNum,length(weights));% ���߶��µ�ǰ֡�ĸ�����������ֵ
    tmpsalcur = zeros(tmpSPcur.spNum,ffNum);
    regionCenter = tmpSPcur.region_center;
%% 2. ��ʼ���� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    for sp=1:tmpSPcur.spNum
        SIGNS = tmp_MAPSET{sp,1}.correSets;
        if isempty(SIGNS)% ������λ��OR�⣨����ؼ���������������
        tmpsalcur(sp,:) = 0;
        else
        % 2.1 ��ؼ�����Ϣ����š����롢������ **********************************************
        tmp_correSets         = tmp_MAPSET{sp,1}.correSets;% ȫ�ߴ�״̬�µ���ؼ�
        tmp_correSets_dist    = tmp_MAPSET{sp,1}.correSets_dist;% ����λ��
        tmp_correSets_overlap = tmp_MAPSET{sp,1}.correSets_overlap;
        tmp_sals              = tmpsalpre(tmp_correSets,1)';
        
        tmp_correSets_dist = tmp_correSets_dist./normDist;% ���öԽ��߾�����й�һ�� 0~1
        tmp_correSets_dist(isnan(tmp_correSets_dist)) = 0;
        alpha_pos = 2/(mean(tmp_correSets_dist(:))+eps);
        pos_diff = exp(-alpha_pos*tmp_correSets_dist);

        % 2.2 ��ؼ�����������������  ********************************************************
        for ff=1:ffNum
            if ff==1 % mean Lab
            ds = data_cur_mappedA(sp,4:6);
            dc = data_pre_mappedA(tmp_correSets,4:6);
            end
            if ff==2     % Flow(mang/ori)
            ds = data_cur_mappedA(sp,8:9);
            dc = data_pre_mappedA(tmp_correSets,8:9);
            end
            
            % revised in 2016.08.30 14:43PM -------------------------
            alld = [ds;dc];% sampleNum * feaDim
            alld(isnan(alld)) = 0;
                      
            ds = alld(1,:);
            dc = alld(2:end,:);
            
            fea_diff = repmat(ds,[size(dc,1),1]) - dc;
            fea_diff = sqrt(sum(fea_diff.*fea_diff,2));
            alpha_fea = 2/(mean(fea_diff(:))+eps);
            fea_diff = exp(-alpha_fea*fea_diff);                   
            diff = fea_diff' .* pos_diff;

            if sum(diff)==0
                tmpDiffSal = (diff*tmp_sals')/(sum(diff)+eps);
            else
                tmpDiffSal = (diff*tmp_sals')/sum(diff);
            end
%             delta = delta + tmpweights * tmpDiffSal;
            tmpsalcur(sp,ff) = tmpDiffSal;
            clear tmpindex tmpweights ds dc fea_diff diff
        end
        % 2.4 clear ********************************************************************************************
        clear tmp_correSets tmp_correSets_dist tmp_sals pos_diff

        end
    end
    
% %% 3. ����������ں�Ȩ�أ�preDist * compactness�� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
%     tmpsalcur = normalizeSal(tmpsalcur);% ��һ��
%     [PP_Img, ~] = CreateImageFromSPs(tmpsalcur, tmpSPcur.pixelList, height, width, true);
%     [rcenter_PP,ccenter_PP] = computeObjectCenter(PP_Img);
%     regionDist_PP = computeRegion2CenterDist(regionCenter,[rcenter_PP,ccenter_PP],[height,width]);
%     PP_compactness = computeCompactness(tmpsalcur,regionDist_PP);
    
%     IMG = IMG + PP_Img;
    tmpsalcur = sum(tmpsalcur,2);
    SALS{ss,1}.SalValue    = tmpsalcur;         clear tmpsalcur % ���߶��µ�����������ֵ
    SALS{ss,1}.compactness = 10000;
    SALS{ss,1}.PP_Img      = 10000;
%     SALS{ss,1}.compactness = 1/(PP_compactness);clear PP_compactness % ���߶��¶�Ӧ��compactnessֵ
%     SALS{ss,1}.PP_Img      = PP_Img;            clear PP_Img % ���߶��¶�Ӧ�����ؼ�������ͼ
    clear  tmp_MAPSET tmpFullresult tmpsalpre 
    
end

IMG = normalizeSal(IMG);% ���߶��µ�ƽ����ͽ��

clear CURINFOR PREINFOR FullResultCur model
end