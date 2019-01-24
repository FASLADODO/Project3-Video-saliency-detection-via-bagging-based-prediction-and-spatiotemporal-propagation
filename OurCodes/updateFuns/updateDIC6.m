function UPDATA_DIC = updateDIC6(CURINFOR,param,fpre_GT,MVF_Foward_fn_f)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 由前一帧的GT限制当前帧的objIndex的求取
% copyright by xiaofei zhou
% V1: 2016.10.09 19:12pm
% 统一前后物体位置信息估算 
% V3： 2016.10.13 14：08PM
% 根据前述的背景字典，于此处进行微调适应
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1.initialization &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
ee = param.ee;
fcur_gt = CURINFOR.imgt;

%% 2 在原有OR的基础上，确定物体的位置 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% revised in 2016.10.09 22:14PM
mapping_pre = preGT_flow_mapping(fpre_GT,    MVF_Foward_fn_f);
intersec = fpre_GT.*mapping_pre;
union    = fpre_GT+mapping_pre;
union(union==2) = 1;
% newGT0 = intersec;
newGT0 = union;% 先并后交 2016.10.23 19:09PM
newGT = newGT0.*fcur_gt;


objIndex = find(newGT(:)==1);% 当下确定的object位置信息
SPSCALENUM = length(CURINFOR.ORLabels);
LCEND = CURINFOR.LCEND;% 已有OR范围

% gtinfor = getGTINFOR(newGT,ee);% 该函数能决定仅是 LCEND, 即物体的大致范围
% 各区域同OR的关系: objIndex很关键
%  <L 0; >=H 1; L~H 50 old version
% 剔除原OR背景中，作为OBJ邻域的部分（与object稍有交集） 2016.10.20 14:08PM
ORLabels = computeORLabel1(LCEND, objIndex, CURINFOR.spinfor,param);% sampleNum*3/cell

% 原始字典
CURD0 = computeD0(SPSCALENUM,CURINFOR.spinfor,ORLabels,CURINFOR.fea);


%% 3. 更新
% 变化率：相对于前一帧的变化率，自适应更新
% fpre_GT = double(fpre_GT>=0.5);
% newGT   = double(newGT>=0.5);
% RATIO   = 1 - sum(sum(fpre_GT.*newGT))/sum(sum(fpre_GT));% y_(t-1)  y_t

%% 3.1 计算区域显著性值 (区域均值作为显著性值)---------------------
gt2spSal = computeGTinfor(fcur_gt,CURINFOR.spinfor);

% %% 2.2 Boosting 训练  ----------------------
% [UPDATA_DIC.D0, UPDATA_DIC.beta, UPDATA_DIC.model, tmodel] = ...
%     MultiFeaBoostingTrain(UPDATA_DIC.DB,UPDATA_DIC.D0,ORLabels,spSal,param);

%% 3.2 获取DB，更新D0 ----------------------
%     fprintf('\nFULLUPDATe = %d',RATIO)
    % 利用cur_or_infor.D0 进行PCA得到DB
%     DB = D02DBNew(CURD0,param);
    DB = [];
    UPDATA_DIC.DB = DB; 
    UPDATA_DIC.D0 = CURD0;
    clear CURD0 DB 
    
%     [UPDATA_DIC.D0, UPDATA_DIC.beta, UPDATA_DIC.model, tmodel] = ...
%        MultiFeaBoostingTrain(UPDATA_DIC.DB,UPDATA_DIC.D0,ORLabels,spSal,param);
   
%    [UPDATA_DIC.D0, UPDATA_DIC.beta, UPDATA_DIC.model, tmodel] = ...
%        MultiFeaBoostingTrainNew0(UPDATA_DIC.DB,UPDATA_DIC.D0,ORLabels,spSal,param);
   
   % revised in 2016.10.13 14:06PM  (新的训练学习方式) --- 
   [UPDATA_DIC.D0, UPDATA_DIC.beta, UPDATA_DIC.model, tmodel] = ...
       MultiFeaBoostingTrainNew2(UPDATA_DIC.DB,UPDATA_DIC.D0,ORLabels,gt2spSal,param,CURINFOR.spinfor);
%% clear
clear fpre_GT fcur_gt CURINFOR THl PRE_DIC
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%1 计算D0(所有cell下集中进行测试)
% 利用确定性样本进行训练（正样本 object; 负样本 border） 2016.10.09 19:48PM
% 2016.10.13 18:59PM 选择正负样本 （1,1）（1,0）
function D0 = computeD0(ScaleNums,spinfor,ORLabels,fea)
% fea 全尺寸， sampleNum = tmpSP.spNum， 包含OR内外区域的所有特征
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
        indexP = find(PNlabel==1);% P于所有样本中的index标记

        
        % revised in 2016.10.13 19:03PM ------
        % 构建背景字典元素 OR=1 OBJECT=0, N于所有样本中的index标记
        if 1 
        indexN = [];
        for dd=1:length(ISORlabel)
            if ISORlabel(dd)==1 && ISOBJlabel(dd)==0
                indexN = [indexN;dd];
            end
        end 
        end
        
        if 0
        % revised in 2016.08.30 22:32PM  ----------------------------------
        % revised in 2016.10.09 19:59PM 寻找确定性负样本 border
        % 全尺寸状态， 需间接统计负样本的标号  ISBORDERlabel==1
        indexN = [];% (1,0) N
        ISBORDERlabel = tmpORlabel(:,2);
        for sp=1:tmpSP.spNum
            if ISORlabel(sp)==1 && ISOBJlabel(sp)==0 && ISBORDERlabel(sp)==1
                indexN = [indexN;sp];
            end
        end
        % -----------------------------------------------------------------
        end
        
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

% 5 计算由GT得到的各区域sal与label,用于boosting训练
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
%             tmpSPlabel(sp,1) = 1;% 前景
%         else
%             tmpSPlabel(sp,1) = 0;% 背景
%         end
    end
    spSal{ss,1} = tmpSPsal;
%     GTinfor.spSal{ss,1} = tmpSPsal;
%     GTinfor.spLabel{ss,1} = tmpSPlabel;
    clear tmpSPsal tmpSP tmpSPlabel
end

clear imGT spinfor objth
end

% 6 GT 之光流映射
% 返回preGT于下一帧的映射图result(二值的)
function result = preGT_flow_mapping(fpre_GT,MVF_Foward_fn_f)
%% 1 self mapping 获取物体的x,y坐标
objIndex = find(fpre_GT(:)==1);
[height,width] = size(fpre_GT);
[sy,sx] = ind2sub([height,width],objIndex);

%% 2 optical flow map(由前一帧的 GT 经过 光流 进行映射)
MVF_Foward_fn_f = double(MVF_Foward_fn_f);
XX = MVF_Foward_fn_f(:,:,1);
YY = MVF_Foward_fn_f(:,:,2);% 计算object的光流偏移
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

% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% % 3 获取LCEND & OBJECTINDEX
% function gtinfor = getGTINFOR(fcur_gt,ee)
% % 确定operation region: LCEND(x1,y1,x2,y2), lefttop & rightbottom
% % 20160802  12:37PM
% % 
% [height,width] = size(fcur_gt);
% showimgs = zeros(height,width,3);
% showimgs(:,:,1) = fcur_gt;
% showimgs(:,:,2) = fcur_gt;
% showimgs(:,:,3) = fcur_gt;
% objIndex = find(fcur_gt(:)==1);
% [sy,sx] = ind2sub([height,width],objIndex);
% [boxself,lcSelf] = boundingboxNew1(fcur_gt,showimgs);
% 
% % --- revised in 2016.08.21 9:29AM ----------------------------------------
% % 去除单物体情形 
% if size(lcSelf,1)>1
% lc1 = min(lcSelf(:,1:2));
% lc2 = max(lcSelf(:,3:4));
% LC = [lc1,lc2];    
% else
% LC = lcSelf;
% end
% % -------------------------------------------------------------------------
% dw = LC(3) - LC(1)+1;
% dh = LC(4) - LC(2)+1;
% 
% centerY = LC(2) + round(dh/2);
% centerX = LC(1) + round(dw/2);
% 
% extend_length_h = round(ee*dh/2);eh = round(dh/2) + extend_length_h;
% extend_length_w = round(ee*dw/2);ew = round(dw/2) + extend_length_w;
% 
% x1New = centerX - ew;
% y1New = centerY - eh;
% if x1New<3
%     x1New = 3;  
% end
% if y1New<3
%     y1New = 3;
% end
% 
% x2New = centerX + ew;
% y2New = centerY + eh;
% if x2New>width
%     x2New = width-3;  
% end
% if y2New>height
%     y2New = height-3;
% end
% LC1 = [x1New,y1New,x2New,y2New];
% [BoxEnd,LCEND]=draw_rect(showimgs,LC1(1),LC1(2),LC1(3),LC1(4));
% 
% figure,
% subplot(1,2,1),imshow(boxself,[])
% subplot(1,2,2),imshow(BoxEnd,[])
% 
% clear BoxEnd boxself
% gtinfor.objIndex = objIndex;% 物体像素为1，背景为0
% gtinfor.lcSelf = lcSelf;% 物体的boundingbox的位置坐标
% gtinfor.sy = sy; % 物体的像素的位置坐标
% gtinfor.sx = sx;
% gtinfor.LCEND = LCEND;% 最终物体的boundingbox的左上角、右下角坐标
% 
% clear objIdex lcSelf LCEND showimg
% end



% % 4 获取LCEND,用于gtinfor中
% function [result,lc]=draw_rect(RGB_img,x1,y1,x2,y2)
% 
% 
% rgb = [255 0 0];                                 % 杈规棰滆壊
%                           
% x1=floor(x1);
% x2=floor(x2);
% y1=floor(y1);
% y2=floor(y2);
% 
% % if x1<3||x2<3||y1<3||y2<3
% %     x1=3;
% %     x2=3;
% %     y1=3;
% %     y2=3;
% % end
% 
% if x1<3
%     x1=3;
% end
% if x2<3
%     x2=3;
% end
% if y1<3
%     y1=3;
% end
% if y2<3
%    y2=3;
% end
%  
% 
% result = RGB_img;
% if size(result,3) == 3
%     for k=1:3
%           %鐢昏竟妗嗛『搴忎负锛氫笂鍙充笅宸︾殑鍘熷垯
% %             result(x1,y1:y1+x2,k)=rgb(1,k);
% %             result(x1:x1+y2,y1+x2,k) = rgb(1,k);
% %             result(x1+y2,y1:y1+x2,k) = rgb(1,k);  
% %             result(x1:x1+y2,y1,k) = rgb(1,k);  
%           
%             result(y1-1:y1+1,x1:x2,k)=rgb(1,k);
%             result(y1:y2,x2-1:x2+1,k) = rgb(1,k);
%             result(y2-1:y2+1,x1:x2,k) = rgb(1,k);  
%             result(y1:y2,x1-1:x1+1,k) = rgb(1,k);          
%        
%     end
% end
%     
% 
% 
% lc = [x1,y1,x2,y2];
% 
% 
% end
