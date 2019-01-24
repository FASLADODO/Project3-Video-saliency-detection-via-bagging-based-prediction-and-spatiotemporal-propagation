function UPDATA_DIC = updateDIC3_1(CURINFOR,param,fpre_GT,MVF_Foward_fn_f)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��ǰһ֡��GT���Ƶ�ǰ֡��objIndex����ȡ
% copyright by xiaofei zhou
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1.initialization &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
ee = param.ee;
fcur_gt = CURINFOR.imgt;

mapping_pre     = preGT_flow_mapping(fpre_GT,    MVF_Foward_fn_f);
    intersec = fpre_GT.*mapping_pre;
    union    = fpre_GT+mapping_pre;
    union(union==2) = 1;
    if sum(union(:))==0
        iu_ratio = sum(intersec(:))/(sum(union(:))+eps);
    else
        iu_ratio = sum(intersec(:))/sum(union(:));
    end
    
    if iu_ratio>=param.flow_ratio(2) %flow�ǿɿ���  pre
        newGT0 = intersec;
    else
        newGT0 = fpre_GT;
    end

newGT = newGT0.*fcur_gt;

gtinfor = getGTINFOR(newGT,ee);% �ú����ܾ������� LCEND, ������Ĵ��·�Χ
SPSCALENUM = length(CURINFOR.ORLabels);

% ������ͬOR�Ĺ�ϵ: objIndex�ܹؼ�
ORLabels = computeORLabel(gtinfor.LCEND, gtinfor.objIndex, CURINFOR.spinfor,param);% sampleNum*3/cell

% ԭʼ�ֵ�
CURD0 = computeD0(SPSCALENUM,CURINFOR.spinfor,ORLabels,CURINFOR.fea);


%% 2. �仯�ʣ������ǰһ֡�ı仯�ʣ�����Ӧ����
fpre_GT = double(fpre_GT>=0.5);
newGT   = double(newGT>=0.5);
RATIO   = 1 - sum(sum(fpre_GT.*newGT))/sum(sum(fpre_GT));% y_(t-1)  y_t

%% 2.1 ��������������ֵ ---------------------
spSal = computeGTinfor(fcur_gt,CURINFOR.spinfor);

% %% 2.2 Boosting ѵ��  ----------------------
% [UPDATA_DIC.D0, UPDATA_DIC.beta, UPDATA_DIC.model, tmodel] = ...
%     MultiFeaBoostingTrain(UPDATA_DIC.DB,UPDATA_DIC.D0,ORLabels,spSal,param);

%% 2.2 ��ȡDB������D0 ----------------------
    fprintf('\nFULLUPDATe = %d',RATIO)
    % ����cur_or_infor.D0 ����PCA�õ�DB
    DB = D02DBNew(CURD0,param);
    UPDATA_DIC.DB = DB; 
    UPDATA_DIC.D0 = CURD0;
    clear CURD0 DB 
%     [UPDATA_DIC.D0, UPDATA_DIC.beta, UPDATA_DIC.model, tmodel] = ...
%        MultiFeaBoostingTrain(UPDATA_DIC.DB,UPDATA_DIC.D0,ORLabels,spSal,param);
   
   [UPDATA_DIC.D0, UPDATA_DIC.beta, UPDATA_DIC.model, tmodel] = ...
       MultiFeaBoostingTrainNew0(UPDATA_DIC.DB,UPDATA_DIC.D0,ORLabels,spSal,param);

%% clear
clear fpre_GT fcur_gt CURINFOR THl PRE_DIC
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%0 ������������ pre & cur ��D0���кϲ�
function result = mergePreCurD0(CURD0,PRED0)
    result.P.colorHist_rgb = [CURD0.P.colorHist_rgb;PRED0.P.colorHist_rgb];
    result.P.colorHist_lab = [CURD0.P.colorHist_lab;PRED0.P.colorHist_lab]; 
    result.P.colorHist_hsv = [CURD0.P.colorHist_hsv;PRED0.P.colorHist_hsv];
    result.P.lbpHist       = [CURD0.P.lbpHist;      PRED0.P.lbpHist];    
    result.P.hogHist       = [CURD0.P.hogHist;      PRED0.P.hogHist];  
    result.P.regionCov     = [CURD0.P.regionCov;    PRED0.P.regionCov];    
    result.P.geoDist       = [CURD0.P.geoDist;      PRED0.P.geoDist];     
    result.P.flowHist      = [CURD0.P.flowHist;     PRED0.P.flowHist];      
    
    result.N.colorHist_rgb = [CURD0.N.colorHist_rgb;PRED0.N.colorHist_rgb];
    result.N.colorHist_lab = [CURD0.N.colorHist_lab;PRED0.N.colorHist_lab]; 
    result.N.colorHist_hsv = [CURD0.N.colorHist_hsv;PRED0.N.colorHist_hsv];
    result.N.lbpHist       = [CURD0.N.lbpHist;      PRED0.N.lbpHist];    
    result.N.hogHist       = [CURD0.N.hogHist;      PRED0.N.hogHist];  
    result.N.regionCov     = [CURD0.N.regionCov;    PRED0.N.regionCov];    
    result.N.geoDist       = [CURD0.N.geoDist;      PRED0.N.geoDist];     
    result.N.flowHist      = [CURD0.N.flowHist;     PRED0.N.flowHist]; 
    
    clear CURD0 PRED0
end

%1 ����D0(����cell�¼��н��в���)
function D0 = computeD0(ScaleNums,spinfor,ORLabels,fea)
% fea ȫ�ߴ磬 sampleNum = tmpSP.spNum
    D0.P = struct;D0.N = struct;% sampleNum*feaDim
    DP_colorHist_rgb = []; DN_colorHist_rgb = [];
    DP_colorHist_lab = []; DN_colorHist_lab = [];
    DP_colorHist_hsv = []; DN_colorHist_hsv = [];
    DP_lbpHist       = []; DN_lbpHist       = [];
    DP_hogHist       = []; DN_hogHist       = [];
    DP_regionCov     = []; DN_regionCov     = [];
    DP_geoDist       = []; DN_geoDist       = [];
    DP_flowHist      = []; DN_flowHist      = [];
    
    for ss=1:ScaleNums
        tmpSP = spinfor{ss,1};
        tmpORlabel = ORLabels{ss,1};% spNum*3
        ISORlabel = tmpORlabel(:,1);
%         index_out_OR = find(ISORlabel~=1);
        ISOBJlabel = tmpORlabel(:,3);
        PNlabel = ISORlabel.*ISOBJlabel;% (1,1) P 
%         PNlabel(index_out_OR,:) = [];
        indexP = find(PNlabel==1);
%         indexN = find(PNlabel==0);

        % revised in 2016.08.30 22:32PM  ----------------------------------
        % ȫ�ߴ�״̬�� ����ͳ�Ƹ������ı��
        indexN = [];% (1,0) N
        for sp=1:tmpSP.spNum
            if ISORlabel(sp)==1 && ISOBJlabel(sp)==0
                indexN = [indexN;sp];
            end
        end
        % -----------------------------------------------------------------
        
        DP_colorHist_rgb = [DP_colorHist_rgb;fea{ss,1}.colorHist_rgb(indexP,:)];
        DP_colorHist_lab = [DP_colorHist_lab;fea{ss,1}.colorHist_lab(indexP,:)];
        DP_colorHist_hsv = [DP_colorHist_hsv;fea{ss,1}.colorHist_hsv(indexP,:)];  
        DP_lbpHist       = [DP_lbpHist;      fea{ss,1}.lbpHist(indexP,:)];
        DP_hogHist       = [DP_hogHist;      fea{ss,1}.hogHist(indexP,:)];
        DP_regionCov     = [DP_regionCov;    fea{ss,1}.regionCov(indexP,:)];
        DP_geoDist       = [DP_geoDist;      fea{ss,1}.geoDist(indexP,:)];
        DP_flowHist      = [DP_flowHist;     fea{ss,1}.flowHist(indexP,:)];
        
        DN_colorHist_rgb = [DN_colorHist_rgb;fea{ss,1}.colorHist_rgb(indexN,:)];
        DN_colorHist_lab = [DN_colorHist_lab;fea{ss,1}.colorHist_lab(indexN,:)];
        DN_colorHist_hsv = [DN_colorHist_hsv;fea{ss,1}.colorHist_hsv(indexN,:)];  
        DN_lbpHist       = [DN_lbpHist;      fea{ss,1}.lbpHist(indexN,:)];
        DN_hogHist       = [DN_hogHist;      fea{ss,1}.hogHist(indexN,:)];
        DN_regionCov     = [DN_regionCov;    fea{ss,1}.regionCov(indexN,:)];
        DN_geoDist       = [DN_geoDist;      fea{ss,1}.geoDist(indexN,:)];
        DN_flowHist      = [DN_flowHist;     fea{ss,1}.flowHist(indexN,:)];    
        
    end
    D0.P.colorHist_rgb = DP_colorHist_rgb; 
    D0.P.colorHist_lab = DP_colorHist_lab; 
    D0.P.colorHist_hsv = DP_colorHist_hsv; 
    D0.P.lbpHist       = DP_lbpHist;
    D0.P.hogHist       = DP_hogHist;
    D0.P.regionCov     = DP_regionCov;
    D0.P.geoDist       = DP_geoDist;
    D0.P.flowHist      = DP_flowHist;
    
    D0.N.colorHist_rgb = DN_colorHist_rgb; 
    D0.N.colorHist_lab = DN_colorHist_lab; 
    D0.N.colorHist_hsv = DN_colorHist_hsv; 
    D0.N.lbpHist       = DN_lbpHist;
    D0.N.hogHist       = DN_hogHist;
    D0.N.regionCov     = DN_regionCov;
    D0.N.geoDist       = DN_geoDist;
    D0.N.flowHist      = DN_flowHist;

clear ScaleNums spinfor ORLabels fea
clear DP_colorHist_rgb DP_colorHist_lab DP_colorHist_hsv DP_lbpHist DP_hogHist DP_regionCov DP_geoDist DP_flowHist
clear DN_colorHist_rgb DN_colorHist_lab DN_colorHist_hsv DN_lbpHist DN_hogHist DN_regionCov DN_geoDist DN_flowHist

end


% %2 ����OR�����ǩ�� OR���⣬�߽磬ǰ���� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% function ORLabels = computeORLabel(LCEND, objIndex, spinfor,param)
% % lend = [x1,y1,x2,y2];
% % objectIndex GT��ǩ
% % 
% ORTHS = param.ORTHS;
% OR_th = ORTHS(1);
% OR_BORDER_th = ORTHS(2);
% OR_OB_th = ORTHS(3);
% 
% SPSCALENUM = length(spinfor);
% ORLabels = cell(SPSCALENUM,1);
% 
% [height,width,dims] = size(spinfor{1,1}.idxcurrImage);
% ORI = zeros(height,width);
% ORI(LCEND(2):LCEND(4),LCEND(1):LCEND(3))=1;
% ORIndex = find(ORI(:)==1);
% 
% for ss=1:SPSCALENUM % ÿ���߶���
%     tmpSP = spinfor{ss,1};
%     
%     LABEL = [];ISOR=[];ISOBJ=[];ISBORDER = [];
%     for sp=1:tmpSP.spNum % ������
%         TMP = find(tmpSP.idxcurrImage==sp);
%         
%         % 1. �����ж��Ƿ���OR������ 1/0
%         indSP_OR = ismember(TMP, ORIndex);
%         ratio_OR = sum(indSP_OR)/length(indSP_OR);
%         
%         % revised in 2016.08.29 8:07AM ------------------------------------
%         if ratio_OR==0
%             ISOR = [ISOR;0];
%             ISBORDER = [ISBORDER;0];
%         else
%             ISOR = [ISOR;1];% ����OR����
%             
%             if ratio_OR>0 && ratio_OR<1
%                 ISBORDER = [ISBORDER;1];% �߽糬��������
%             end
%             
%             if ratio_OR==1
%                 ISBORDER = [ISBORDER;0];
%             end
%         end
%         % -----------------------------------------------------------------
% %         if  ratio_OR< OR_th % λ��OR�ⲿ
% %             ISOR = [ISOR;0];
% %             ISBORDER = [ISBORDER;0];
% %         else
% %             ISOR = [ISOR;1];
% %             % 2. �ж�OR����߽糬���� <1 OR�߽糬����
% %             if ratio_OR<OR_BORDER_th
% %                 ISBORDER = [ISBORDER;1];% �߽糬��������
% %             else
% %                 ISBORDER = [ISBORDER;0];
% %             end        
% %         end
%         
%         %3. ���ж��Ƿ���Object�� 1/0
%         if isempty(objIndex)
%         ISOBJ = [ISOBJ;100];  
%         else
%         indSP_GT = ismember(TMP,objIndex);
%         ratio_GT = sum(indSP_GT)/length(indSP_GT);
%         if  ratio_GT < OR_OB_th
%             ISOBJ = [ISOBJ;0];% NEGTIVE
%         else
%             ISOBJ = [ISOBJ;1]; % POSITIVE
%         end;            
%         end
% 
%         
%     end
%     LABEL = [ISOR,ISBORDER,ISOBJ];
%     ORLabels{ss,1} = LABEL;
%     clear tmpSP
% end
% 
% clear LCEND objIndex spinfor param
% 
% end


% 3 ��ȡLCEND & OBJECTINDEX
function gtinfor = getGTINFOR(fcur_gt,ee)
% ȷ��operation region: LCEND(x1,y1,x2,y2), lefttop & rightbottom
% 20160802  12:37PM
% 
[height,width] = size(fcur_gt);
showimgs = zeros(height,width,3);
showimgs(:,:,1) = fcur_gt;
showimgs(:,:,2) = fcur_gt;
showimgs(:,:,3) = fcur_gt;
objIndex = find(fcur_gt(:)==1);
[sy,sx] = ind2sub([height,width],objIndex);
[boxself,lcSelf] = boundingboxNew1(fcur_gt,showimgs);

% --- revised in 2016.08.21 9:29AM ----------------------------------------
% ȥ������������ 
if size(lcSelf,1)>1
lc1 = min(lcSelf(:,1:2));
lc2 = max(lcSelf(:,3:4));
LC = [lc1,lc2];    
else
LC = lcSelf;
end
% -------------------------------------------------------------------------
dw = LC(3) - LC(1)+1;
dh = LC(4) - LC(2)+1;

centerY = LC(2) + round(dh/2);
centerX = LC(1) + round(dw/2);

extend_length_h = round(ee*dh/2);eh = round(dh/2) + extend_length_h;
extend_length_w = round(ee*dw/2);ew = round(dw/2) + extend_length_w;

x1New = centerX - ew;
y1New = centerY - eh;
if x1New<3
    x1New = 3;  
end
if y1New<3
    y1New = 3;
end

x2New = centerX + ew;
y2New = centerY + eh;
if x2New>width
    x2New = width-3;  
end
if y2New>height
    y2New = height-3;
end
LC1 = [x1New,y1New,x2New,y2New];
[BoxEnd,LCEND]=draw_rect(showimgs,LC1(1),LC1(2),LC1(3),LC1(4));

figure,
subplot(1,2,1),imshow(boxself,[])
subplot(1,2,2),imshow(BoxEnd,[])

clear BoxEnd boxself
gtinfor.objIndex = objIndex;% ��������Ϊ1������Ϊ0
gtinfor.lcSelf = lcSelf;% �����boundingbox��λ������
gtinfor.sy = sy; % ��������ص�λ������
gtinfor.sx = sx;
gtinfor.LCEND = LCEND;% ���������boundingbox�����Ͻǡ����½�����

clear objIdex lcSelf LCEND showimg
end

% 4 ��ȡLCEND,����gtinfor��
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

% 5 ������GT�õ��ĸ�����sal��label,����boostingѵ��
function spSal = computeGTinfor(imGT,spinfor)
% result = computeGTinfor(imGT,spinfor,objth)
imGT = double(imGT>=0.5);
% GTinfor.spSal = cell(length(spinfor),1);
% GTinfor.spLabel = cell(length(spinfor),1);
spSal = cell(length(spinfor),1);
for ss=1:length(spinfor)
    tmpSP = spinfor{ss,1};
    tmpSPsal = zeros(tmpSP.spNum,1);
%     tmpSPlabel = zeros(tmpSP.spNum,1);
    for sp=1:tmpSP.spNum
        tmpSPsal(sp,1) = mean(imGT(tmpSP.pixelList{sp,1}));     
%         if tmpSPsal(sp,1)>=objth
%             tmpSPlabel(sp,1) = 1;% ǰ��
%         else
%             tmpSPlabel(sp,1) = 0;% ����
%         end
    end
    spSal{ss,1} = tmpSPsal;
%     GTinfor.spSal{ss,1} = tmpSPsal;
%     GTinfor.spLabel{ss,1} = tmpSPlabel;
    clear tmpSPsal tmpSP tmpSPlabel
end

clear imGT spinfor objth
end

% 6 GT ֮����ӳ��
% ����preGT����һ֡��ӳ��ͼresult(��ֵ��)
function result = preGT_flow_mapping(fpre_GT,MVF_Foward_fn_f)
%% 1 self mapping ��ȡ�����x,y����
objIndex = find(fpre_GT(:)==1);
[height,width] = size(fpre_GT);
[sy,sx] = ind2sub([height,width],objIndex);

%% 2 optical flow map(��ǰһ֡�� GT ���� ���� ����ӳ��)
MVF_Foward_fn_f = double(MVF_Foward_fn_f);
XX = MVF_Foward_fn_f(:,:,1);
YY = MVF_Foward_fn_f(:,:,2);% ����object�Ĺ���ƫ��
avgFlow(1) = mean(mean(XX(objIndex)));
avgFlow(2) = mean(mean(YY(objIndex)));
sxNew = round(sx + avgFlow(1));
syNew = round(sy + avgFlow(2));

% remove flow ouside of image
tmp = (sxNew>=3 & sxNew<=width-3) & (syNew>=3 & syNew<=height-3);
sxNew = sxNew(tmp); syNew = syNew(tmp);
 
% construct boundingBox
result = zeros(height,width);
for ii=1:length(sxNew)
    result(syNew(ii),sxNew(ii)) = 1;
end

clear fpre_GT MVF_Foward_fn_f

end