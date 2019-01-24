% function [SALS] = temporalPropagationNew1(CURINFOR,PREINFOR,FullResultCur,betas)
% function [SALS] = temporalPropagationNew4(CURINFOR,PREINFOR,FullResultCur,model)
function [SALS,IMG] = temporalPropagationNew4_1(CURINFOR,PREINFOR,param)
% 4--->3 传播, 4:pre  3:cur, PRE--->CUR 传播
% 无model,这里采用 pre信息校正 cur信息 （PCA）
% 
SPSCALENUM = length(CURINFOR.fea);
SALS = cell(SPSCALENUM,1);

% imSal_pre0 = PREINFOR.imsal;
[height,width]  = size(PREINFOR.imsal);
normDist = sqrt((height.^2 + width.^2));
ffNum = 2;
%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
IMG = 0;% 多尺度下的融合结果
for ss=1:SPSCALENUM
%% 1. initialization &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&    
    % 多尺度信息
    tmpSPcur = CURINFOR.spinfor2{ss,1};% 临时增加的一个spinfor 4-->3 
    tmpFea_pre = PREINFOR.fea{ss,1};
    data_pre_mappedA = [tmpFea_pre.regionFea];
    [data_pre_mappedA,scalemap] = scaleForSVM_corrected1(data_pre_mappedA,0,1);
    
    tmpFea_cur = CURINFOR.fea{ss,1};
    data_cur_mappedA = [tmpFea_cur.regionFea];
    [data_cur_mappedA,scalemap] = scaleForSVM_corrected1(data_cur_mappedA,0,1);
   
    % 映射信息
    tmp_MAPSET = tmpSPcur.mapsets;% correSets correSets_dist

    % preparation
    tmpsalpre = normalizeSal(PREINFOR.spsal{ss,1});% 前一帧ss 尺度下各区域的显著性值(此处很重要！！！)
%     tmpsalcur = zeros(tmpSPcur.spNum,length(weights));% 单尺度下当前帧的各区域显著性值
    tmpsalcur = zeros(tmpSPcur.spNum,ffNum);
    regionCenter = tmpSPcur.region_center;
%% 2. 开始传播 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    for sp=1:tmpSPcur.spNum
        SIGNS = tmp_MAPSET{sp,1}.correSets;
        if isempty(SIGNS)% 该区域位于OR外（无相关集），等于其自身
        tmpsalcur(sp,:) = 0;
        else
        % 2.1 相关集的信息：标号、距离、显著性 **********************************************
        tmp_correSets         = tmp_MAPSET{sp,1}.correSets;% 全尺寸状态下的相关集
        tmp_correSets_dist    = tmp_MAPSET{sp,1}.correSets_dist;% 区域位置
        tmp_correSets_overlap = tmp_MAPSET{sp,1}.correSets_overlap;
        tmp_sals              = tmpsalpre(tmp_correSets,1)';
        
        tmp_correSets_dist = tmp_correSets_dist./normDist;% 利用对角线距离进行归一化 0~1
        tmp_correSets_dist(isnan(tmp_correSets_dist)) = 0;
        alpha_pos = 2/(mean(tmp_correSets_dist(:))+eps);
        pos_diff = exp(-alpha_pos*tmp_correSets_dist);

        % 2.2 相关集特征，构建传播阵  ********************************************************
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
    
%% 3. 计算各特征融合权重（preDist * compactness） &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    tmpsalcur = sum(tmpsalcur,2);
    tmpsalcur = normalizeSal(tmpsalcur);% 归一化
    [PP_Img, ~] = CreateImageFromSPs(tmpsalcur, tmpSPcur.pixelList, height, width, true);
    [rcenter_PP,ccenter_PP] = computeObjectCenter(PP_Img);
    regionDist_PP = computeRegion2CenterDist(regionCenter,[rcenter_PP,ccenter_PP],[height,width]);
    PP_compactness = computeCompactness(tmpsalcur,regionDist_PP);
    
    IMG = IMG + PP_Img;
    
    SALS{ss,1}.SalValue    = tmpsalcur;         clear tmpsalcur % 单尺度下的区域显著性值
    SALS{ss,1}.compactness = 1/(PP_compactness);clear PP_compactness % 单尺度下对应的compactness值
    SALS{ss,1}.PP_Img      = PP_Img;            clear PP_Img % 单尺度下对应的像素级显著性图
    clear  tmp_MAPSET tmpFullresult tmpsalpre 
    
end

IMG = normalizeSal(IMG);% 各尺度下的平均求和结果

clear CURINFOR PREINFOR FullResultCur model
end