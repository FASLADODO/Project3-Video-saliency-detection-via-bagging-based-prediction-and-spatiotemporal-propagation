function UPDATA_DIC = updateDIC(fpre_gt,CURINFOR,param,PRE_DIC)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 根据外观模型的结果，自适应的更新字典
% 衡量的是前后帧之间的相对变化
% 
% fpre_gt/fcur_gt 分别表示前一帧与当前帧的伪GT
%
% CURINFOR 当前帧预测得到的一些结果
% spsal/psal/imsal/imgt/fea/out_OR/spinfor
%
% param
% THL 高低阈值
%
% PRE_DIC
% 初始字典信息，构建字典（D0，DB）
% D0.P D0.N      %144+3
% PRE_DIC.D0
% DB.P           DB.N          
% DB.eigvalP     DB.eigvalN   
% DB.meanP       DB.meanN 
% DB.numsampleP  DB.numsampleN 
% PRE_DIC.DB  
%
% V1: 2016.08.01  9:07AM
% V2: 2016.08.18 19:23PM
% 
% copyright by xiaofei zhou
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 获取阈值
th = param.THL(1);
tl = param.THL(2);
ee = param.ee;
fcur_gt = CURINFOR.imgt;

%% 1. 获取当前帧的OR内的正负样本，做更新字典用
gtinfor = getGTINFOR(fcur_gt,ee);
LCEND = gtinfor.LCEND;% LCEND 物体的大致区域 [X1,Y1,X2,Y2]
spinfor = CURINFOR.spinfor;
[height,width,dim] = size(fcur_gt);
ORI = zeros(height,width);
ORI(LCEND(2):LCEND(4),LCEND(1):LCEND(3))=1;
ORIndex = find(ORI(:)==1);
SPSCALENUM = length(CURINFOR.out_OR);

%% 2. 得到各尺度下，各区域的标签属性
% 位于OR内外（1/0）， 位于OR边界与否（1/0）, 属于Object与否（1/-1）
ORTHS = param.ORTHS;
OR_th = ORTHS(1);
OR_BORDER_th = ORTHS(2);
OR_OB_th = ORTHS(3);
cur_or_infor.SPLabels = cell(SPSCALENUM,1);
for ss=1:SPSCALENUM % 每个尺度下
    tmpSP = spinfor{ss,1};
    
    LABEL = [];LABEL_OR=[];LABEL_GT=[];OR_BORDER = [];
    for sp=1:tmpSP.spNum % 各区域
        TMP = find(tmpSP.idxcurrImage==sp);
        
        % 1. 首先判断是否在OR区域内 1/0
        indSP_OR = ismember(TMP, ORIndex);
        ratio_OR = sum(indSP_OR)/length(indSP_OR);
        if  ratio_OR< OR_th % 位于OR外部
            LABEL_OR = [LABEL_OR;0];
            OR_BORDER = [OR_BORDER;0];
        else
            LABEL_OR = [LABEL_OR;1];
            % 2. 判定OR区域边界超像素 <1 OR边界超像素
            if ratio_OR<OR_BORDER_th
                OR_BORDER = [OR_BORDER;1];% 边界超像素区域
            else
                OR_BORDER = [OR_BORDER;0];
            end        
        end
        
        %3. 再判断是否在Object中 1/0
        indSP_GT = ismember(TMP,gtinfor.objIndex);
        ratio_GT = sum(indSP_GT)/length(indSP_GT);
        if  ratio_GT < OR_OB_th
            LABEL_GT = [LABEL_GT;0];% NEGTIVE
        else
            LABEL_GT = [LABEL_GT;1]; % POSITIVE
        end;
        
    end
    LABEL = [LABEL_OR,OR_BORDER,LABEL_GT];
    cur_or_infor.SPLabels{ss,1} = LABEL;
    clear tmpSP
end
clear  LABEL LABEL_OR OR_BORDER LABEL_GT

% 各尺度下组织搜集特征,构成原始字典元素D0
D0_P = [];D0_N = [];
for ss=1:SPSCALENUM % 每个尺度下
%     tmpSP = spinfor{ss,1};% 单尺度分割信息
    tmpSPLabel = cur_or_infor.SPLabels{ss,1};% 单尺度超像素区域标签信息
    ORSIGN = tmpSPLabel(:,1);
    OB_BG_SIGN = tmpSPLabel(:,3);
    
    % 此亦是各尺度下的OR中正负样本的编号
    % ORSIGN与OB_BG_SIGN： (1,1) object, (1,0) background, 得到 indexs
    PS_SIGN = find((ORSIGN.*OB_BG_SIGN)==1);% (1,1)
    NS_SIGN = (ORSIGN==1).* (OB_BG_SIGN==0);% (1,0)
    NS_SIGN = find(NS_SIGN==1);
    
    % 组织特征:所有尺度下的所有原始特征（OR中的区域样本）
    tmp_CURINFOR_fea = CURINFOR.fea{ss,1};
    D0_P = [D0_P,(tmp_CURINFOR_fea(PS_SIGN,:))']; % 列为样本
    D0_N = [D0_N,(tmp_CURINFOR_fea(NS_SIGN,:))'];
 
    clear tmpSP tmpMFea tmpLabel tmp_CURINFOR_fea

end
D0.P = D0_P;
D0.N = D0_N;
cur_or_infor.D0 = D0;% 当前帧的正负样本组成的原始特征字典

clear D0 D0_P D0_N


%% 3. 变化率：相对于前一帧的变化率，自适应更新
RATIO = 1 - sum(sum(fpre_gt.*fcur_gt))/sum(sum(fpre_gt));% y_(t-1)  y_t
beta = param.beta;
betaP = beta(1);
betaN = beta(2);
% 自适应更新(三种条件)
if RATIO<tl % 不更
    fprintf('\nNONUPDATe')
    UPDATA_DIC = PRE_DIC;
end

if RATIO>=th % 全更
    fprintf('\nFULLUPDATe')
    % 利用cur_or_infor.D0 进行PCA得到DB
    DB_infor = D02DB(cur_or_infor.D0);
% DB_infor.DB.P = basisP;
% DB_infor.DB.N = basisN;
% DB_infor.DB.eigvalP = eigvalP;
% DB_infor.DB.eigvalN = eigvalN;
% DB_infor.DB.meanP = meanP;
% DB_infor.DB.meanN = meanN;
% DB_infor.DB.numsampleP = numsampleP;
% DB_infor.DB.numsampleN = numsampleN;

%     DB.P = DB_infor.DBP; % revised in 2016.08.18 15:19PM
%     DB.N = DB_infor.DBN;
    UPDATA_DIC.DB = DB_infor.DB; 
    UPDATA_DIC.D0 = cur_or_infor.D0;
    clear DB_infor DB cur_or_infor
end

if RATIO>=tl && RATIO<th % 部分更
    fprintf('\nPartialUPDATe')
%     % 利用cur_or_infor.D0 结合 PRE_DIC，使用RPCA进行样本更新
%     D00.P = [PRE_DIC.D0.P,cur_or_infor.D0.P];
%     D00.N = [PRE_DIC.D0.N,cur_or_infor.D0.N];
%     DB_infor = D02DB(D00);

%     % 确定保留的原子个数
%     PNUM = size(PRE_DIC.D0.P,2);
%     NNUM = size(PRE_DIC.D0.N,2);   
    
% PRE_DIC
% 初始字典信息，构建字典（D0，DB）
% D0.P D0.N      %144+3
% PRE_DIC.D0
% DB.P           DB.N          
% DB.eigvalP     DB.eigvalN   
% DB.meanP       DB.meanN 
% DB.numsampleP  DB.numsampleN 
% PRE_DIC.DB  

%     % revised in 20160807 16:01PM 
%     % positive samples
%     [basisP, eigvalP, meanP, numsampleP] = ...
%         sklmNew(cur_or_infor.D0.P, PRE_DIC.DB.P, PRE_DIC.DB.eigvalP, PRE_DIC.DB.meanP, PRE_DIC.DB.numsampleP);    
%     
%     % negative samples
%     [basisN, eigvalN, meanN, numsampleN] = ...
%         sklmNew(cur_or_infor.D0.N, PRE_DIC.DB.N, PRE_DIC.DB.eigvalN, PRE_DIC.DB.meanN, PRE_DIC.DB.numsampleN);    
%     
%     
%     DBinfor.P          = basisP(:,round(beta*PNUM));
%     DBinfor.N          = basisN(:,round(beta*NNUM));
%     DBinfor.eigvalP    = eigvalP;
%     DBinfor.eigvalN    = eigvalN;
%     DBinfor.meanP      = meanP;
%     DBinfor.meanN      = meanN;
%     DBinfor.numsampleP = numsampleP;
%     DBinfor.numsampleN = numsampleN;
%     DB = DBinfor;
    
% --- revised in 2016.08.19 15:06PM ---------------------------------------
% 将所有的样本集中起来，直接使用PCA（本质同IPCA一致）
D0.P = [PRE_DIC.D0.P, cur_or_infor.D0.P];% 147*sampleNum
D0.N = [PRE_DIC.D0.N, cur_or_infor.D0.N];
muP = mean(D0.P,2);
muN = mean(D0.N,2);
[basisP, eigvalP, meanP, numsampleP] = sklmNew(D0.P, [], [], muP);  
[basisN, eigvalN, meanN, numsampleN] = sklmNew(D0.N, [], [], muN); 
PNUM = size(basisP,2);
NNUM = size(basisN,2);   

DB.P = basisP(:,1:round(betaP*PNUM));
DB.N = basisN(:,1:round(betaN*NNUM));
DB.eigvalP = eigvalP;
DB.eigvalN = eigvalN;
DB.meanP = meanP;
DB.meanN = meanN;
DB.numsampleP = numsampleP;
DB.numsampleN = numsampleN;

% -------------------------------------------------------------------------    

%     DB.P = DB_infor.DBP(:,round(beta*PNUM));
%     DB.N = DB_infor.DBN(:,round(beta*NNUM));
    UPDATA_DIC.DB = DB; 
    UPDATA_DIC.D0 = D0;
    clear D00 DB_infor DB cur_or_infor D0
end

clear fpre_gt fcur_gt CURINFOR THl PRE_DIC
end

function gtinfor = getGTINFOR(fcur_gt,ee)
% 确定operation region: LCEND(x1,y1,x2,y2), lefttop & rightbottom
% 20160802  12:37PM
% 
[height,width] = size(fcur_gt);
showimg = zeros(height,width,3);

objIndex = find(fcur_gt(:)==1);
[sy,sx] = ind2sub([height,width],objIndex);
[boxself,lcSelf] = boundingboxNew(fcur_gt,showimg);

% --- revised in 2016.08.21 9:29AM ----------------------------------------
% 去除单物体情形 
if size(lcSelf,1)>1
lc1 = min(lcSelf(:,1:2));
lc2 = max(lcSelf(:,3:4));
LC = [lc1,lc2];    
else
LC = lcSelf;
end

% lc1 = min(lcSelf(:,1:2));
% lc2 = max(lcSelf(:,3:4));
% LC = [lc1,lc2];
% -------------------------------------------------------------------------
dw = LC(3) - LC(1)+1;
dh = LC(4) - LC(2)+1;

centerY = LC(2) + round(dh/2);
centerX = LC(1) + round(dw/2);

ew = ee*round(dw/2);
eh = ee*round(dh/2);

x1New = centerX - ew;
y1New = centerY - eh;
if x1New<1
    x1New = 2;  
end
if y1New<1
    y1New = 2;
end

x2New = centerX + ew;
y2New = centerY + eh;
if x2New>width
    x2New = width-2;  
end
if y2New>height
    y2New = height-2;
end
LC1 = [x1New,y1New,x2New,y2New];
% [BoxEnd,LCEND]=draw_rect(fpre_Image,LC1(1),LC1(2),LC1(3),LC1(4));
[BoxEnd,LCEND]=draw_rect(showimg,LC1(1),LC1(2),LC1(3),LC1(4));
clear BoxEnd boxself

gtinfor.objIndex = objIndex;% 物体像素为1，背景为0
gtinfor.lcSelf = lcSelf;% 物体的boundingbox的位置坐标
gtinfor.sy = sy; % 物体的像素的位置坐标
gtinfor.sx = sx;
gtinfor.LCEND = LCEND;% 最终物体的boundingbox的左上角、右下角坐标

clear objIdex lcSelf LCEND showimg
end

function [result,lc]=draw_rect(RGB_img,x1,y1,x2,y2)


rgb = [255 0 0];                                 % 杈规棰滆壊
                          
x1=floor(x1);
x2=floor(x2);
y1=floor(y1);
y2=floor(y2);

% if x1<3||x2<3||y1<3||y2<3
%     x1=3;
%     x2=3;
%     y1=3;
%     y2=3;
% end

if x1<3
    x1=3;
end
if x2<3
    x2=3;
end
if y1<3
    y1=3;
end
if y2<3
   y2=3;
end
 

result = RGB_img;
if size(result,3) == 3
    for k=1:3
          %鐢昏竟妗嗛『搴忎负锛氫笂鍙充笅宸︾殑鍘熷垯
%             result(x1,y1:y1+x2,k)=rgb(1,k);
%             result(x1:x1+y2,y1+x2,k) = rgb(1,k);
%             result(x1+y2,y1:y1+x2,k) = rgb(1,k);  
%             result(x1:x1+y2,y1,k) = rgb(1,k);  
          
            result(y1-1:y1+1,x1:x2,k)=rgb(1,k);
            result(y1:y2,x2-1:x2+1,k) = rgb(1,k);
            result(y2-1:y2+1,x1:x2,k) = rgb(1,k);  
            result(y1:y2,x1-1:x1+1,k) = rgb(1,k);          
       
    end
end
    


lc = [x1,y1,x2,y2];


end