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
% V3： 2016.10.28 10：50AM
% 单尺度均值取样本
% 
% copyright by xiaofei zhou,IVPLab, shanghai university, shanghai,china
% zxforchid@163.com
% www.ivp.shu.edu.cn
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
function ORLabels = computeORLabel2(spinfor,REGION_SALS)
SPSCALENUM = length(spinfor);
ORLabels = cell(SPSCALENUM,1);
[height,width,dims] = size(spinfor{1,1}.idxcurrImage);

for ss=1:SPSCALENUM % 每个尺度下
    tmpSP = spinfor{ss,1};
    tmpSal = REGION_SALS{ss,1};% 单尺度下的图像显著性
    
    LABEL = [];ISOR=[];ISOBJ=[];ISBORDER = [];
    
    meancut=1.5.*mean(tmpSal(:));% 高低阈值用于选择样本
    thresh=0.05;
    
    for sp=1:tmpSP.spNum % 各区域
        ISOR = [ISOR;1];
        ISBORDER = [ISBORDER;0];

        meantemp=mean(tmpSal(tmpSP.pixelList(sp)));
        
        %% 2. 再判断是否在Object中 1/0
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
