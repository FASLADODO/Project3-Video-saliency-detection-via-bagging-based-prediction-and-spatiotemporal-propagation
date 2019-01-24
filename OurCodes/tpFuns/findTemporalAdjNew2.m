% function [spinforCur] = findTemporalAdjNew2(spinforPre, spinforCur,MVF_Backward,curORLabels,preORLabels)
function [spinforCur1] = findTemporalAdjNew2(spinforPre, spinforCur, MVF_Backward)
% [spinforPre,spinforCur] = findTemporalAdjNew1(spinforPre, spinforCur,MVF_Backward, curORLabels,preORLabels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ѱ�ҵ�ǰ֡��ǰһ֡����������(ȫ�ߴ���Ѱ)
% ��߶ȷָ��������Ϣ:
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
% MVF:ʵ����OR_CUR--->OR_PRE�����ӳ��
% ORLabels: ������ͬOR��������ϵ 
% NOTE:����ȷ����ǰ֡OR�е�������ǰһ֡�е����ƥ��������λ��OR��
% V1��2016.08.26 14:07PM
% V2: 2016.10.09 15:15PM
% ���Ӽ�����֡���������ص��� ��������� correSets_overlap
%
% V3�� 2016.10.10 19��19PM
% �ص������������Ѱ�����ƥ�����������������򹹽���ؼ� findCorreSet
% 
% V4�� 2016.10.24 21��59PM
% ȫ�ߴ�Ѱ����ؼ�������OR����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


SPSCALENUM = length(spinforPre);% SLIC�ĳ߶���Ŀ
spinforCur1 = spinforCur;
for ss=1:SPSCALENUM
    tmpSP_Pre = spinforPre{ss,1};
    tmpSP_Cur = spinforCur{ss,1};
%     tmp_curORLabels = curORLabels{ss,1};
%     tmp_preORLabels = preORLabels{ss,1};
    
    region_center_pre = tmpSP_Pre.region_center;
    region_center_cur = tmpSP_Cur.region_center;

    %0. ������߶��µ��ص��� revised in 2016.10.09 15:15PM (curNum*preNum)
    cur_pre_overlap = ...
        computeOverlap(tmpSP_Pre.idxcurrImage,tmpSP_Cur.idxcurrImage,MVF_Backward);
    
    %1. ��ȡ�������MVF spNum*2
    mvf_sp_cur = computeMVFSP(tmpSP_Cur.spNum,tmpSP_Cur.idxcurrImage,tmpSP_Cur.area,MVF_Backward);
    
    %2. ͶӰӳ�� cur--->pre (����ǰͶӰ������Ѱ�ҵ�ǰ֡����һ������ǰһ֡����ؼ�)
    region_center_prediction = region_center_cur + mvf_sp_cur;
    
    %3. ����ŷʽ���룬Ѱ����ƥ�䳬��������(OR_CUR--->OR_PRE) 
%     map_set_cur = ...
%         findCorreSet(region_center_prediction,region_center_pre, tmp_curORLabels,tmp_preORLabels);
    % ע�⣬������ tmpSP_Pre.adjmat �ڽ�1 ���ڽ�0  �Խ���Ϊ0 20161014
    % ȫ�ߴ�Ѱ����ؼ�������OR���� 2016.10.24 21:58PM
%     map_set_cur = findCorreSet(region_center_prediction,region_center_pre,tmp_curORLabels,tmp_preORLabels, tmpSP_Pre.adjmat,cur_pre_overlap);
    map_set_cur = findCorreSet(region_center_prediction,region_center_pre,tmpSP_Pre.adjmat,cur_pre_overlap);
    spinforCur1{ss,1}.mapsets = map_set_cur;% correSets correSets_dist
    spinforCur1{ss,1}.region_center_prediction = region_center_prediction;% ����ͶӰԤ���λ��
    
    clear region_center_pre region_center_cur map_set_cur
    clear region_center_prediction mvf_sp_cur
    clear tmpSP_Pre tmpSP_Cur
end

clear MVF_Backward spinforPre 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%1. ��ȡ�������MVF spNum*2 ------------------------------------------------
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

% 2. Ѱ����ؼ�(OR_CUR--->OR_PRE) ------------------------------------------
% ȫ�ߴ���Ѱ����һ����OR�� 2016.08.2614:43PM
% ���ؽ��������ƥ��SP�ı�ż��������ţ��Ͷ�Ӧ��λ�ÿռ����
% ���� cur_pre_overlap ��ǰ��֡ͶӰ�ص��ȣ� 2016.10.09 16��10PM
% �ص������������Ѱ�����ƥ�����������������򹹽���ؼ� 2016.10.10 18:18PM
% ��ǰ֡��OR����������ؼ� 2016.10.24 10:28AM
% ����OR 2016.11.18
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
    
    % ��ǰ֮֡��i�����Ƿ���OR������������ؼ���2016.10.24 21:51PM��
%     iSIGN = ismember(i,cur_out_OR_indexs);% OR��1  OR��0
%     if iSIGN==1 % λ��OR��,��Ϊ��
%         result{i,1}.correSets_dist = [];       
%         result{i,1}.correSets      = [];
%     else
        
    % ������룺 i ��ǰһ֡������ľ���(λ�ÿռ�ŷ�Ͼ���)
    tmp_dist       = zeros(region_num_pre, 2);
    tmp_dist(:, 1) = region_center_prediction(i, 1) - region_center_pre(:, 1);
    tmp_dist(:, 2) = region_center_prediction(i, 2) - region_center_pre(:, 2);
    tmp_dist2      = sqrt(tmp_dist(:, 1) .^ 2 + tmp_dist(:, 2) .^ 2);

    % revised in 2016.10.09 16:15PM ����ǰ��֡�ص��ȵ���ؼ�Ѱ��
    tmpoverlap = cur_pre_overlap(i,:);
    correSetsIndex1 = find(tmpoverlap>0);% ȫ�ߴ�״̬�µ���ؼ�
%     correSetSign = ismember(correSetsIndex, pre_out_OR_indexs);
%     correSetsIndex1 = correSetsIndex(correSetSign==0);
    
    % revised in 2016.10.13 20:23PM;�ñ����п���Ϊ��
    if ~isempty(correSetsIndex1) % �������ص�������
    correSets = correSetsIndex1; 
    correSets_dist = tmp_dist2(correSets);
    correSets_overlap = tmpoverlap(correSets);
    clear correSetsIndex correSetSign correSetsIndex1 
    
    % added in 2016.10.10 10:20AM 
    % �ص��Ƚ�Ͼ���Ѱ����ƥ������Ȼ�����������򹹽���ؼ�
    DIST = (1./correSets_dist).*correSets_overlap';clear correSets_dist
    [maxvalue,maxindex] = max(DIST);
    ID = correSets(maxindex(end));clear correSets % ��ǰһ֡��ƥ�������ID���
    correSetsLabel = adjmat_pre(ID,:);
    correSetsIndex1 = find(correSetsLabel==1);% ��������
%     correSetSign = ismember(correSetsIndex, pre_out_OR_indexs);
%     correSetsIndex1 = correSetsIndex(correSetSign==0);
    correSets = [ID,correSetsIndex1];  
    correSets_dist = tmp_dist2(correSets);
    
    else % ���Ǿ���
    % ȫ����Ϊ�� 2016.10.14  10:07AM��
    % ����ǰ֡��ǰһ֡��ͶӰλ��OR�⣬����������
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

