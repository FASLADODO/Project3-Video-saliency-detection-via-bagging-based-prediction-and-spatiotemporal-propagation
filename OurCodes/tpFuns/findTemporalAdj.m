function [spinforPre,spinforCur] = findTemporalAdj(spinforPre, spinforCur,MVF_Backward, aidIndexsCur,aidIndexsPre)
% Ѱ�ҵ�ǰ֡��ǰһ֡����������
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
% changged��mapset������OR����� 20160801 19:01PM
% aidIndexs OR�ⲿ������
% ʵ����OR_CUR--->OR_PRE�����ӳ��
% 


SPSCALENUM = length(spinforPre);% SLIC�ĳ߶���Ŀ

for ss=1:SPSCALENUM
    tmpSP_Pre = spinforPre{ss,1};
    tmpSP_Cur = spinforCur{ss,1};
    tmpSP_aidIndexscur = aidIndexsCur{ss,1};
    tmpSP_aidIndexspre = aidIndexsPre{ss,1};
    
    region_center_pre = tmpSP_Pre.region_center;
    region_center_cur = tmpSP_Cur.region_center;

    % ��ȡ�������MVF spNum*2
    mvf_sp_cur = computeMVFSP(tmpSP_Cur.spNum,tmpSP_Cur.idxcurrImage,tmpSP_Cur.area,MVF_Backward);
    
    % ͶӰӳ�� cur--->pre (����ǰͶӰ������Ѱ�ҵ�ǰ֡����һ������ǰһ֡����ؼ�)
    region_center_prediction = region_center_cur + mvf_sp_cur;
    
    % Ѱ����ؼ�(OR_CUR--->OR_PRE)
    map_set_cur = ...
        findCorreSet(region_center_prediction,region_center_pre, tmpSP_aidIndexscur,tmpSP_aidIndexspre);
    spinforCur{ss,1}.mapsets = map_set_cur;
    spinforCur{ss,1}.region_center_prediction = region_center_prediction;% ����ͶӰԤ���λ��
    
    clear region_center_pre region_center_cur map_set_cur
    clear region_center_prediction mvf_sp_cur
    clear tmpSP_Pre tmpSP_Cur
end

clear MVF_Backward

end

function result = findCorreSet(region_center_prediction,region_center_pre, tmpSP_aidIndexscur,tmpSP_aidIndexspre)
region_num = size(region_center_prediction,1);
region_num_pre = size(region_center_pre,1);
result = zeros(region_num,1);% ���ص�������ص�����ı�ţ���ǰһ֡�еı�ţ�
for i = 1 : region_num
    % ��ǰ֮֡��i�����Ƿ���OR����
    iSIGN = ismember(i,tmpSP_aidIndexscur);
    if iSIGN % λ��OR��
        result(i) = 0;
    else
    
    % ������� i ��ǰһ֡������ľ���
    tmp_dist       = zeros(region_num_pre, 2);
    tmp_dist(:, 1) = region_center_prediction(i, 1) - region_center_pre(:, 1);
    tmp_dist(:, 2) = region_center_prediction(i, 2) - region_center_pre(:, 2);
    tmp_dist2      = sqrt(tmp_dist(:, 1) .^ 2 + tmp_dist(:, 2) .^ 2);
%     [tmp_min_dist, tmp_region_idx] = min(tmp_dist2);
    
%     % �����ƥ������pre֡�У��Ƿ���OR������
%     idxSIGN = ismember(tmp_region_idx,tmpSP_aidIndexspre);  
%     tmp_region_idx1 = tmp_region_idx(idxSIGN~=1);% λ��OR�ڵ�������������
    
    % --- revised in 2016.08.18 14:35PM -----------------------------------
    % --- �ҳ�λ��OR�е����ƥ������ 
    [sortValue, sortIndex] = sort(tmp_dist2);% ��������
    for jj=1:length(sortIndex)
        tmp_region_idx = sortIndex(jj);
        idxSIGN        = ismember(tmp_region_idx,tmpSP_aidIndexspre);
        if idxSIGN~=1 % λ��OR��
            tmp_region_idx1 = tmp_region_idx;% pre�е�OR���ƥ������
%             jj=length(sortIndex); % ����ѭ��
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
            mvf_sp(tmp_label, 1) = mvf_sp(tmp_label, 1) + mvf(i, j, 2); %y   %%%%��ÿ���������ۼ��˶���ֵ��Ϊʲô����ÿ�������ÿ��bin�ڣ�����
            mvf_sp(tmp_label, 2) = mvf_sp(tmp_label, 2) + mvf(i, j, 1); %x
        end
    end
    for i = 1 : region_num
        mvf_sp(i, :) = mvf_sp(i, :) / num_cluster(i);
    end
    
    clear region_num labels num_cluster mvf
end