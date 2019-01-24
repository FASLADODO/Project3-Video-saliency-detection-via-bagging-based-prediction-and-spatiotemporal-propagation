% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% 计算OR区域标签： OR内外，边界，前背景 
%
% V1：2016.10.09 8:41AM
% 尝试减少负样本个数，以求正负样本平衡！！！
% 
% V2：2016.10.20 10:46AM
% 用于更新时样本的选择，正负样本可以少点但要具备区分判别性
% 剔除OR=1 OBJ=0中，OBJ邻域部分
%
% copyright by xiaofei zhou,IVPLab, shanghai university, shanghai,china
% zxforchid@163.com
% www.ivp.shu.edu.cn
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
function ORLabels = computeORLabel1(LCEND, objIndex, spinfor,param)
% lend = [x1,y1,x2,y2];
% objectIndex GT标签
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

for ss=1:SPSCALENUM % 每个尺度下
    tmpSP = spinfor{ss,1};
    
    LABEL = [];ISOR=[];ISOBJ=[];ISBORDER = [];
    for sp=1:tmpSP.spNum % 各区域
        TMP = find(tmpSP.idxcurrImage==sp);
        
        %% 1. 首先判断是否在OR区域内 1/0
        indSP_OR = ismember(TMP, ORIndex);
        ratio_OR = sum(indSP_OR)/length(indSP_OR);
        if 1
        % revised in 2016.08.29 8:07AM ------------------------------------
        if ratio_OR==0
            ISOR = [ISOR;0];
            ISBORDER = [ISBORDER;0];
        else
            ISOR = [ISOR;1];% 属于OR区域
            if ratio_OR>0 && ratio_OR<1
                ISBORDER = [ISBORDER;1];% 边界超像素区域（有可能边界既与OR交又与object交）
            end
            if ratio_OR==1
                ISBORDER = [ISBORDER;0];
            end
        end
        % -----------------------------------------------------------------
        end
             
        %% 2. 再判断是否在Object中 1/0
        if isempty(objIndex)
        ISOBJ = [ISOBJ;100];  
        else
        indSP_GT = ismember(TMP,objIndex);
        ratio_GT = sum(indSP_GT)/length(indSP_GT);
        % revised in 2016.10.12 20:25PM
%         if ratio_GT<OR_OB_th_L  % 无交集或者交集小于0.2（即大于等于0.8），背景
        if ratio_GT == 0 % 更新时，选取确定性的负样本 2016.10.20
            ISOBJ = [ISOBJ;0];
        end
        if ratio_GT>=OR_OB_th_H  % 有交集且大于等于0.8，前景
            ISOBJ = [ISOBJ;1];
        end
%         if ratio_GT>=OR_OB_th_L && ratio_GT<OR_OB_th_H %有交集，介于0.2~0.8之间
        if ratio_GT>0 && ratio_GT<OR_OB_th_H
            ISOBJ = [ISOBJ;50];
        end
               
        end
            
    end
    %% 3 于每个尺度下，再剔除背景样本中与OBJECT邻接的区域
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
    ADJS = unique(ADJS);%最终的OBJ邻域index
     
%     index_out_OBJ = [];
    for ii=1:length(ISOR)
        if ISOR(ii)==1 && ISOBJ(ii)==0% 背景1（可能含有OBJ邻域）
            SIGN = ismember(ii,ADJS);
            if SIGN==1 % 背景区域且属于obj的邻域,将此区域置为75
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
