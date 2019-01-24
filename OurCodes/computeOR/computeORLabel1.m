% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% ����OR�����ǩ�� OR���⣬�߽磬ǰ���� 
%
% V1��2016.10.09 8:41AM
% ���Լ��ٸ�����������������������ƽ�⣡����
% 
% V2��2016.10.20 10:46AM
% ���ڸ���ʱ������ѡ���������������ٵ㵫Ҫ�߱������б���
% �޳�OR=1 OBJ=0�У�OBJ���򲿷�
%
% copyright by xiaofei zhou,IVPLab, shanghai university, shanghai,china
% zxforchid@163.com
% www.ivp.shu.edu.cn
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
function ORLabels = computeORLabel1(LCEND, objIndex, spinfor,param)
% lend = [x1,y1,x2,y2];
% objectIndex GT��ǩ
% 
ORTHS = param.ORTHS;
% OR_th = ORTHS(1);
% OR_BORDER_th = ORTHS(2);
% OR_OB_th = ORTHS(3);
OR_OB_th_L = ORTHS(4);
OR_OB_th_H = ORTHS(5);

SPSCALENUM = length(spinfor);
ORLabels = cell(SPSCALENUM,1);

[height,width,dims] = size(spinfor{1,1}.idxcurrImage);
ORI = zeros(height,width);
ORI(LCEND(2):LCEND(4),LCEND(1):LCEND(3))=1;
ORIndex = find(ORI(:)==1);

for ss=1:SPSCALENUM % ÿ���߶���
    tmpSP = spinfor{ss,1};
    
    LABEL = [];ISOR=[];ISOBJ=[];ISBORDER = [];
    for sp=1:tmpSP.spNum % ������
        TMP = find(tmpSP.idxcurrImage==sp);
        
        %% 1. �����ж��Ƿ���OR������ 1/0
        indSP_OR = ismember(TMP, ORIndex);
        ratio_OR = sum(indSP_OR)/length(indSP_OR);
        if 1
        % revised in 2016.08.29 8:07AM ------------------------------------
        if ratio_OR==0
            ISOR = [ISOR;0];
            ISBORDER = [ISBORDER;0];
        else
            ISOR = [ISOR;1];% ����OR����
            if ratio_OR>0 && ratio_OR<1
                ISBORDER = [ISBORDER;1];% �߽糬���������п��ܱ߽����OR������object����
            end
            if ratio_OR==1
                ISBORDER = [ISBORDER;0];
            end
        end
        % -----------------------------------------------------------------
        end
             
        %% 2. ���ж��Ƿ���Object�� 1/0
        if isempty(objIndex)
        ISOBJ = [ISOBJ;100];  
        else
        indSP_GT = ismember(TMP,objIndex);
        ratio_GT = sum(indSP_GT)/length(indSP_GT);
        % revised in 2016.10.12 20:25PM
%         if ratio_GT<OR_OB_th_L  % �޽������߽���С��0.2�������ڵ���0.8��������
        if ratio_GT == 0 % ����ʱ��ѡȡȷ���Եĸ����� 2016.10.20
            ISOBJ = [ISOBJ;0];
        end
        if ratio_GT>=OR_OB_th_H  % �н����Ҵ��ڵ���0.8��ǰ��
            ISOBJ = [ISOBJ;1];
        end
%         if ratio_GT>=OR_OB_th_L && ratio_GT<OR_OB_th_H %�н���������0.2~0.8֮��
        if ratio_GT>0 && ratio_GT<OR_OB_th_H
            ISOBJ = [ISOBJ;50];
        end
               
        end
            
    end
    %% 3 ��ÿ���߶��£����޳�������������OBJECT�ڽӵ�����
    adjcMatrix = tmpSP.adjcMatrix;
    indexOBJS = ISOR.*ISOBJ;
    index_in_OBJ = find(indexOBJS==1);
    ADJS = [];
    for ii=1:length(index_in_OBJ)
        tmpOBJ = index_in_OBJ(ii);
        tmpADJS = adjcMatrix(tmpOBJ);
        tmpADJS(tmpADJS==0) = [];
        ADJS = [ADJS,tmpADJS];
    end
    ADJS = unique(ADJS);%���յ�OBJ����index
     
%     index_out_OBJ = [];
    for ii=1:length(ISOR)
        if ISOR(ii)==1 && ISOBJ(ii)==0% ����1�����ܺ���OBJ����
            SIGN = ismember(ii,ADJS);
            if SIGN==1 % ��������������obj������,����������Ϊ75
                ISOBJ(ii) = 75;
%                index_out_OBJ = [index_out_OBJ;ii];
            end
        end
    end
    
    %% 4 SAVE
    
    LABEL = [ISOR,ISBORDER,ISOBJ];
    ORLabels{ss,1} = LABEL;
    clear tmpSP
end

clear LCEND objIndex spinfor param

end
%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
