% function [spinforCur] = findTemporalAdjNew2(spinforPre, spinforCur,MVF_Backward,curORLabels,preORLabels)
function [spinforCur1] = findTemporalAdjNew2(spinforPre, spinforCur, MVF_Backward)
% [spinforPre,spinforCur] = findTemporalAdjNew1(spinforPre, spinforCur,MVF_Backward, curORLabels,preORLabels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 寻找当前帧与前一帧的相邻区域(全尺寸搜寻)
% 多尺度分割包含的信息:
% spinfor{ss,1}.adjcMatrix = adjcMatrix;
% spinfor{ss,1}.colDistM = colDistM;
% spinfor{ss,1}.clipVal = clipVal;
% spinfor{ss,1}.idxcurrImage = idxcurrImage;
% spinfor{ss,1}.adjmat = adjmat;
% spinfor{ss,1}.pixelList =pixelList;
% spinfor{ss,1}.area = area;
% spinfor{ss,1}.spNum = spNum;
% spinfor{ss,1}.bdIds = bdIds;
% spinfor{ss,1}.posDistM = posDistM;
% spinfor{ss,1}.region_center = region_center;
% 
% MVF:实现了OR_CUR--->OR_PRE区域的映射
% ORLabels: 各区域同OR的隶属关系 
% NOTE:用于确保当前帧OR中的区域于前一帧中的最佳匹配区域亦位于OR中
% V1：2016.08.26 14:07PM
% V2: 2016.10.09 15:15PM
% 增加计算两帧各区域间的重叠度 结果中增加 correSets_overlap
%
% V3： 2016.10.10 19：19PM
% 重叠度与距离相结合寻找最佳匹配区域，在利用其邻域构建相关集 findCorreSet
% 
% V4： 2016.10.24 21：59PM
% 全尺寸寻找相关集，不分OR内外
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


SPSCALENUM = length(spinforPre);% SLIC的尺度数目
spinforCur1 = spinforCur;
for ss=1:SPSCALENUM
    tmpSP_Pre = spinforPre{ss,1};
    tmpSP_Cur = spinforCur{ss,1};
%     tmp_curORLabels = curORLabels{ss,1};
%     tmp_preORLabels = preORLabels{ss,1};
    
    region_center_pre = tmpSP_Pre.region_center;
    region_center_cur = tmpSP_Cur.region_center;

    %0. 计算各尺度下的重叠度 revised in 2016.10.09 15:15PM (curNum*preNum)
    cur_pre_overlap = ...
        computeOverlap(tmpSP_Pre.idxcurrImage,tmpSP_Cur.idxcurrImage,MVF_Backward);
    
    %1. 获取各区域的MVF spNum*2
    mvf_sp_cur = computeMVFSP(tmpSP_Cur.spNum,tmpSP_Cur.idxcurrImage,tmpSP_Cur.area,MVF_Backward);
    
    %2. 投影映射 cur--->pre (后向前投影，用于寻找当前帧的任一区域于前一帧的相关集)
    region_center_prediction = region_center_cur + mvf_sp_cur;
    
    %3. 利用欧式距离，寻找最匹配超像素区域(OR_CUR--->OR_PRE) 
%     map_set_cur = ...
%         findCorreSet(region_center_prediction,region_center_pre, tmp_curORLabels,tmp_preORLabels);
    % 注意，这里是 tmpSP_Pre.adjmat 邻接1 不邻接0  对角线为0 20161014
    % 全尺寸寻找相关集，不分OR内外 2016.10.24 21:58PM
%     map_set_cur = findCorreSet(region_center_prediction,region_center_pre,tmp_curORLabels,tmp_preORLabels, tmpSP_Pre.adjmat,cur_pre_overlap);
    map_set_cur = findCorreSet(region_center_prediction,region_center_pre,tmpSP_Pre.adjmat,cur_pre_overlap);
    spinforCur1{ss,1}.mapsets = map_set_cur;% correSets correSets_dist
    spinforCur1{ss,1}.region_center_prediction = region_center_prediction;% 保存投影预测的位置
    
    clear region_center_pre region_center_cur map_set_cur
    clear region_center_prediction mvf_sp_cur
    clear tmpSP_Pre tmpSP_Cur
end

clear MVF_Backward spinforPre 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%1. 获取各区域的MVF spNum*2 ------------------------------------------------
function mvf_sp = computeMVFSP(region_num,labels,num_cluster,mvf)
[height,width] = size(labels);
    mvf_sp = zeros(region_num, 2); %Average MV of each SuperPixel, y/x
    for i = 1 : height
        for j = 1 : width
            tmp_label = labels(i, j);
            mvf_sp(tmp_label, 1) = mvf_sp(tmp_label, 1) + mvf(i, j, 2); %y  
            mvf_sp(tmp_label, 2) = mvf_sp(tmp_label, 2) + mvf(i, j, 1); %x
        end
    end
    for i = 1 : region_num
        mvf_sp(i, :) = mvf_sp(i, :) / num_cluster(i);
    end
    
    clear region_num labels num_cluster mvf
end

% 2. 寻找相关集(OR_CUR--->OR_PRE) ------------------------------------------
% 全尺寸搜寻，不一定在OR中 2016.08.2614:43PM
% 返回结果引入最匹配SP的标号及其邻域标号，和对应的位置空间距离
% 加入 cur_pre_overlap （前后帧投影重叠度） 2016.10.09 16：10PM
% 重叠度与距离相结合寻找最佳匹配区域，在利用其邻域构建相关集 2016.10.10 18:18PM
% 当前帧的OR外区域无相关集 2016.10.24 10:28AM
% 舍弃OR 2016.11.18
function result = ...
    findCorreSet(region_center_prediction,region_center_pre, adjmat_pre, cur_pre_overlap)
% function result = ...
%     findCorreSet(region_center_prediction,region_center_pre,tmp_curORLabels,tmp_preORLabels, adjmat_pre, cur_pre_overlap)
% findCorreSet(region_center_prediction,region_center_pre, tmp_curORLabels,tmp_preORLabels)
region_num = size(region_center_prediction,1);
region_num_pre = size(region_center_pre,1);
result = cell(region_num,1);
% cur_ISORlabels = tmp_curORLabels(:,1);
% cur_out_OR_indexs = find(cur_ISORlabels==0);
% pre_ISORlabels = tmp_preORLabels(:,1);
% pre_out_OR_indexs = find(pre_ISORlabels==0);
for i = 1 : region_num
    
    % 当前帧之第i区域是否在OR区域，在则无相关集（2016.10.24 21:51PM）
%     iSIGN = ismember(i,cur_out_OR_indexs);% OR外1  OR内0
%     if iSIGN==1 % 位于OR外,置为空
%         result{i,1}.correSets_dist = [];       
%         result{i,1}.correSets      = [];
%     else
        
    % 计算距离： i 与前一帧各区域的距离(位置空间欧氏距离)
    tmp_dist       = zeros(region_num_pre, 2);
    tmp_dist(:, 1) = region_center_prediction(i, 1) - region_center_pre(:, 1);
    tmp_dist(:, 2) = region_center_prediction(i, 2) - region_center_pre(:, 2);
    tmp_dist2      = sqrt(tmp_dist(:, 1) .^ 2 + tmp_dist(:, 2) .^ 2);

    % revised in 2016.10.09 16:15PM 基于前后帧重叠度的相关集寻找
    tmpoverlap = cur_pre_overlap(i,:);
    correSetsIndex1 = find(tmpoverlap>0);% 全尺寸状态下的相关集
%     correSetSign = ismember(correSetsIndex, pre_out_OR_indexs);
%     correSetsIndex1 = correSetsIndex(correSetSign==0);
    
    % revised in 2016.10.13 20:23PM;该变量有可能为空
    if ~isempty(correSetsIndex1) % 距离与重叠度相结合
    correSets = correSetsIndex1; 
    correSets_dist = tmp_dist2(correSets);
    correSets_overlap = tmpoverlap(correSets);
    clear correSetsIndex correSetSign correSetsIndex1 
    
    % added in 2016.10.10 10:20AM 
    % 重叠度结合距离寻找最匹配区域，然后利用其邻域构建相关集
    DIST = (1./correSets_dist).*correSets_overlap';clear correSets_dist
    [maxvalue,maxindex] = max(DIST);
    ID = correSets(maxindex(end));clear correSets % 于前一帧最匹配的区域ID编号
    correSetsLabel = adjmat_pre(ID,:);
    correSetsIndex1 = find(correSetsLabel==1);% 邻域区域
%     correSetSign = ismember(correSetsIndex, pre_out_OR_indexs);
%     correSetsIndex1 = correSetsIndex(correSetSign==0);
    correSets = [ID,correSetsIndex1];  
    correSets_dist = tmp_dist2(correSets);
    
    else % 仅是距离
    % 全部置为空 2016.10.14  10:07AM；
    % 即当前帧于前一帧的投影位于OR外，即背景区域
    correSets         = []; 
    correSets_overlap = [];
    correSets_dist    = [];
    end
    
    result{i,1}.correSets_dist    = correSets_dist';       
    result{i,1}.correSets_overlap = correSets_overlap;
    result{i,1}.correSets         = correSets;
    
    clear tmp_dist tmp_dist2 tmp_region_idx tmp_region_idx1
    clear correSetsLabel correSetsIndex correSets correSets_dist
%     end
end

clear region_center_prediction region_center_pre 
clear tmp_curORLabels tmp_preORLabels adjmat_pre

end

