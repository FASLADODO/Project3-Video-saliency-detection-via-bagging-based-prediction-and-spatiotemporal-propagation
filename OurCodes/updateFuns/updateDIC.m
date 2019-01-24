function UPDATA_DIC = updateDIC(fpre_gt,CURINFOR,param,PRE_DIC)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% �������ģ�͵Ľ��������Ӧ�ĸ����ֵ�
% ��������ǰ��֮֡�����Ա仯
% 
% fpre_gt/fcur_gt �ֱ��ʾǰһ֡�뵱ǰ֡��αGT
%
% CURINFOR ��ǰ֡Ԥ��õ���һЩ���
% spsal/psal/imsal/imgt/fea/out_OR/spinfor
%
% param
% THL �ߵ���ֵ
%
% PRE_DIC
% ��ʼ�ֵ���Ϣ�������ֵ䣨D0��DB��
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
% ��ȡ��ֵ
th = param.THL(1);
tl = param.THL(2);
ee = param.ee;
fcur_gt = CURINFOR.imgt;

%% 1. ��ȡ��ǰ֡��OR�ڵ������������������ֵ���
gtinfor = getGTINFOR(fcur_gt,ee);
LCEND = gtinfor.LCEND;% LCEND ����Ĵ������� [X1,Y1,X2,Y2]
spinfor = CURINFOR.spinfor;
[height,width,dim] = size(fcur_gt);
ORI = zeros(height,width);
ORI(LCEND(2):LCEND(4),LCEND(1):LCEND(3))=1;
ORIndex = find(ORI(:)==1);
SPSCALENUM = length(CURINFOR.out_OR);

%% 2. �õ����߶��£�������ı�ǩ����
% λ��OR���⣨1/0���� λ��OR�߽����1/0��, ����Object���1/-1��
ORTHS = param.ORTHS;
OR_th = ORTHS(1);
OR_BORDER_th = ORTHS(2);
OR_OB_th = ORTHS(3);
cur_or_infor.SPLabels = cell(SPSCALENUM,1);
for ss=1:SPSCALENUM % ÿ���߶���
    tmpSP = spinfor{ss,1};
    
    LABEL = [];LABEL_OR=[];LABEL_GT=[];OR_BORDER = [];
    for sp=1:tmpSP.spNum % ������
        TMP = find(tmpSP.idxcurrImage==sp);
        
        % 1. �����ж��Ƿ���OR������ 1/0
        indSP_OR = ismember(TMP, ORIndex);
        ratio_OR = sum(indSP_OR)/length(indSP_OR);
        if  ratio_OR< OR_th % λ��OR�ⲿ
            LABEL_OR = [LABEL_OR;0];
            OR_BORDER = [OR_BORDER;0];
        else
            LABEL_OR = [LABEL_OR;1];
            % 2. �ж�OR����߽糬���� <1 OR�߽糬����
            if ratio_OR<OR_BORDER_th
                OR_BORDER = [OR_BORDER;1];% �߽糬��������
            else
                OR_BORDER = [OR_BORDER;0];
            end        
        end
        
        %3. ���ж��Ƿ���Object�� 1/0
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

% ���߶�����֯�Ѽ�����,����ԭʼ�ֵ�Ԫ��D0
D0_P = [];D0_N = [];
for ss=1:SPSCALENUM % ÿ���߶���
%     tmpSP = spinfor{ss,1};% ���߶ȷָ���Ϣ
    tmpSPLabel = cur_or_infor.SPLabels{ss,1};% ���߶ȳ����������ǩ��Ϣ
    ORSIGN = tmpSPLabel(:,1);
    OB_BG_SIGN = tmpSPLabel(:,3);
    
    % �����Ǹ��߶��µ�OR�����������ı��
    % ORSIGN��OB_BG_SIGN�� (1,1) object, (1,0) background, �õ� indexs
    PS_SIGN = find((ORSIGN.*OB_BG_SIGN)==1);% (1,1)
    NS_SIGN = (ORSIGN==1).* (OB_BG_SIGN==0);% (1,0)
    NS_SIGN = find(NS_SIGN==1);
    
    % ��֯����:���г߶��µ�����ԭʼ������OR�е�����������
    tmp_CURINFOR_fea = CURINFOR.fea{ss,1};
    D0_P = [D0_P,(tmp_CURINFOR_fea(PS_SIGN,:))']; % ��Ϊ����
    D0_N = [D0_N,(tmp_CURINFOR_fea(NS_SIGN,:))'];
 
    clear tmpSP tmpMFea tmpLabel tmp_CURINFOR_fea

end
D0.P = D0_P;
D0.N = D0_N;
cur_or_infor.D0 = D0;% ��ǰ֡������������ɵ�ԭʼ�����ֵ�

clear D0 D0_P D0_N


%% 3. �仯�ʣ������ǰһ֡�ı仯�ʣ�����Ӧ����
RATIO = 1 - sum(sum(fpre_gt.*fcur_gt))/sum(sum(fpre_gt));% y_(t-1)  y_t
beta = param.beta;
betaP = beta(1);
betaN = beta(2);
% ����Ӧ����(��������)
if RATIO<tl % ����
    fprintf('\nNONUPDATe')
    UPDATA_DIC = PRE_DIC;
end

if RATIO>=th % ȫ��
    fprintf('\nFULLUPDATe')
    % ����cur_or_infor.D0 ����PCA�õ�DB
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

if RATIO>=tl && RATIO<th % ���ָ�
    fprintf('\nPartialUPDATe')
%     % ����cur_or_infor.D0 ��� PRE_DIC��ʹ��RPCA������������
%     D00.P = [PRE_DIC.D0.P,cur_or_infor.D0.P];
%     D00.N = [PRE_DIC.D0.N,cur_or_infor.D0.N];
%     DB_infor = D02DB(D00);

%     % ȷ��������ԭ�Ӹ���
%     PNUM = size(PRE_DIC.D0.P,2);
%     NNUM = size(PRE_DIC.D0.N,2);   
    
% PRE_DIC
% ��ʼ�ֵ���Ϣ�������ֵ䣨D0��DB��
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
% �����е���������������ֱ��ʹ��PCA������ͬIPCAһ�£�
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
% ȷ��operation region: LCEND(x1,y1,x2,y2), lefttop & rightbottom
% 20160802  12:37PM
% 
[height,width] = size(fcur_gt);
showimg = zeros(height,width,3);

objIndex = find(fcur_gt(:)==1);
[sy,sx] = ind2sub([height,width],objIndex);
[boxself,lcSelf] = boundingboxNew(fcur_gt,showimg);

% --- revised in 2016.08.21 9:29AM ----------------------------------------
% ȥ������������ 
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

gtinfor.objIndex = objIndex;% ��������Ϊ1������Ϊ0
gtinfor.lcSelf = lcSelf;% �����boundingbox��λ������
gtinfor.sy = sy; % ��������ص�λ������
gtinfor.sx = sx;
gtinfor.LCEND = LCEND;% ���������boundingbox�����Ͻǡ����½�����

clear objIdex lcSelf LCEND showimg
end

function [result,lc]=draw_rect(RGB_img,x1,y1,x2,y2)


rgb = [255 0 0];                                 % 边框颜色
                          
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
          %画边框顺序为：上右下左的原则
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