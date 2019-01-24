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
% V3�� 2016.10.28 10��50AM
% ���߶Ⱦ�ֵȡ����
% 
% copyright by xiaofei zhou,IVPLab, shanghai university, shanghai,china
% zxforchid@163.com
% www.ivp.shu.edu.cn
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
function ORLabels = computeORLabel2(spinfor,REGION_SALS)
SPSCALENUM = length(spinfor);
ORLabels = cell(SPSCALENUM,1);
[height,width,dims] = size(spinfor{1,1}.idxcurrImage);

for ss=1:SPSCALENUM % ÿ���߶���
    tmpSP = spinfor{ss,1};
    tmpSal = REGION_SALS{ss,1};% ���߶��µ�ͼ��������
    
    LABEL = [];ISOR=[];ISOBJ=[];ISBORDER = [];
    
    meancut=1.5.*mean(tmpSal(:));% �ߵ���ֵ����ѡ������
    thresh=0.05;
    
    for sp=1:tmpSP.spNum % ������
        ISOR = [ISOR;1];
        ISBORDER = [ISBORDER;0];

        meantemp=mean(tmpSal(tmpSP.pixelList(sp)));
        
        %% 2. ���ж��Ƿ���Object�� 1/0
        if isempty(objIndex)
        ISOBJ = [ISOBJ;100];  
        else
        if meantemp>=thresh && meantemp<meancut 
            ISOBJ = [ISOBJ;50];
        end
        if meantemp>=meancut  
            ISOBJ = [ISOBJ;1];
        end
        if meantemp<thresh
            ISOBJ = [ISOBJ;0];
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
