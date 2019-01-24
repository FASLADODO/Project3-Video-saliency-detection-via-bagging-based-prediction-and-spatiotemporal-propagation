function [spinforPre,spinforCur] = findTemporalAdj(spinforPre, spinforCur,MVF_Backward, aidIndexsCur,aidIndexsPre)
% 寻找当前帧与前一帧的相邻区域
% 2016.08.01 15:10PM
% xiaofei zhou
%
% spinforPre{ss,1}.idxcurrImage = idxcurrImage;
% spinforPre{ss,1}.adjmat = adjmat;
% spinforPre{ss,1}.pixelList =pixelList;
% spinforPre{ss,1}.area = area;
% spinforPre{ss,1}.spNum = spNum;
% spinforPre{ss,1}.bdIds = bdIds;
% spinforPre{ss,1}.posDistM = posDistM;
% 
% changged：mapset仅保留OR区域点 20160801 19:01PM
% aidIndexs OR外部区域编号
% 实现了OR_CUR--->OR_PRE区域的映射
% 


SPSCALENUM = length(spinforPre);% SLIC的尺度数目

for ss=1:SPSCALENUM
    tmpSP_Pre = spinforPre{ss,1};
    tmpSP_Cur = spinforCur{ss,1};
    tmpSP_aidIndexscur = aidIndexsCur{ss,1};
    tmpSP_aidIndexspre = aidIndexsPre{ss,1};
    
    region_center_pre = tmpSP_Pre.region_center;
    region_center_cur = tmpSP_Cur.region_center;

    % 获取各区域的MVF spNum*2
    mvf_sp_cur = computeMVFSP(tmpSP_Cur.spNum,tmpSP_Cur.idxcurrImage,tmpSP_Cur.area,MVF_Backward);
    
    % 投影映射 cur--->pre (后向前投影，用于寻找当前帧的任一区域于前一帧的相关集)
    region_center_prediction = region_center_cur + mvf_sp_cur;
    
    % 寻找相关集(OR_CUR--->OR_PRE)
    map_set_cur = ...
        findCorreSet(region_center_prediction,region_center_pre, tmpSP_aidIndexscur,tmpSP_aidIndexspre);
    spinforCur{ss,1}.mapsets = map_set_cur;
    spinforCur{ss,1}.region_center_prediction = region_center_prediction;% 保存投影预测的位置
    
    clear region_center_pre region_center_cur map_set_cur
    clear region_center_prediction mvf_sp_cur
    clear tmpSP_Pre tmpSP_Cur
end

clear MVF_Backward

end

function result = findCorreSet(region_center_prediction,region_center_pre, tmpSP_aidIndexscur,tmpSP_aidIndexspre)
region_num = size(region_center_prediction,1);
region_num_pre = size(region_center_pre,1);
result = zeros(region_num,1);% 返回的是最相关的区域的编号（于前一帧中的编号）
for i = 1 : region_num
    % 当前帧之第i区域是否在OR区域
    iSIGN = ismember(i,tmpSP_aidIndexscur);
    if iSIGN % 位于OR外
        result(i) = 0;
    else
    
    % 计算距离 i 与前一帧各区域的距离
    tmp_dist       = zeros(region_num_pre, 2);
    tmp_dist(:, 1) = region_center_prediction(i, 1) - region_center_pre(:, 1);
    tmp_dist(:, 2) = region_center_prediction(i, 2) - region_center_pre(:, 2);
    tmp_dist2      = sqrt(tmp_dist(:, 1) .^ 2 + tmp_dist(:, 2) .^ 2);
%     [tmp_min_dist, tmp_region_idx] = min(tmp_dist2);
    
%     % 检测最匹配区域（pre帧中）是否在OR区域中
%     idxSIGN = ismember(tmp_region_idx,tmpSP_aidIndexspre);  
%     tmp_region_idx1 = tmp_region_idx(idxSIGN~=1);% 位于OR内的区域样本距离
    
    % --- revised in 2016.08.18 14:35PM -----------------------------------
    % --- 找出位于OR中的最佳匹配区域 
    [sortValue, sortIndex] = sort(tmp_dist2);% 升序排列
    for jj=1:length(sortIndex)
        tmp_region_idx = sortIndex(jj);
        idxSIGN        = ismember(tmp_region_idx,tmpSP_aidIndexspre);
        if idxSIGN~=1 % 位于OR内
            tmp_region_idx1 = tmp_region_idx;% pre中的OR最佳匹配区域
%             jj=length(sortIndex); % 跳出循环
            break;
        end
    end
    % ---------------------------------------------------------------------
    
    result(i) = tmp_region_idx1(1);       
    end       
   
end

clear region_center_prediction region_center_pre tmpSP_aidIndexscur tmpSP_aidIndexspre

end
function mvf_sp = computeMVFSP(region_num,labels,num_cluster,mvf)
[height,width] = size(labels);
    mvf_sp = zeros(region_num, 2); %Average MV of each SuperPixel, y/x
    for i = 1 : height
        for j = 1 : width
            tmp_label = labels(i, j);
            mvf_sp(tmp_label, 1) = mvf_sp(tmp_label, 1) + mvf(i, j, 2); %y   %%%%在每个区域内累加运动幅值，为什么不再每个区域的每个bin内？？？
            mvf_sp(tmp_label, 2) = mvf_sp(tmp_label, 2) + mvf(i, j, 1); %x
        end
    end
    for i = 1 : region_num
        mvf_sp(i, :) = mvf_sp(i, :) / num_cluster(i);
    end
    
    clear region_num labels num_cluster mvf
end