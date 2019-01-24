function [result,DIC] = initDic2_2(fpre_Image,fpre_GT,param,flow)
% function [result,DIC] = initDic2_2(fpre_Image,fcur_Image,fnext_Image,fcur_GT,param,flow)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V10: 2016.10.28 7:14AM
% 去除OR区域，直接于全尺寸图像进行操作
% 
% V11: 2016.11.2 9:18AM
% 引入LBP-TOP
% 加入 fpre_pre_Image fpre_Image fcur_Image 分别指代第一帧，第二帧及第三帧 
% 
% result.objectnum 
% result.objectarea 
% result.objectcenter 
% ------------------------------------------------
% Copyright by xiaofei zhou, IVPLab, shanghai univeristy,shanghai, china
% http://www.ivp.shu.edu.cn
% email: zxforchid@163.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
spnumbers = param.spnumbers;
% ee = param.ee;
[height,width,dims] = size(fpre_Image);
pre_image           = fpre_Image;

%% Ⅰ 根据GT 确定object的大致位置，即 operation region (OR):
%%  LCEND(x1,y1,x2,y2), lefttop & rightbottom
objIndex = find(fpre_GT(:)==1);

%% Ⅱ 特征提取，图像的全部区域(LOW & MID LEVEL Feature)
% 2.1 超像素分割，获取多尺度信息 
spinfor = multiscaleSLIC(fpre_Image,spnumbers);
    
% 2.2 提取特征（self + variance）利用确定性样本作为训练样本
ORFEA = featureExtractNew2_1(pre_image,spinfor,flow, param,objIndex);

%% Ⅲ PCA构建双字典
% DB = D02DBNew(ORFEA.D0,param);
DB = [];

%% Ⅳ 输出第一帧的区域显著性图（结合GT，实现输出区域均值）
spSal = computeGTinfor(fpre_GT,spinfor);

%% Ⅴ BOOSTING框架（训练）
[tmodel] = MultiFeaBoostingTrainNew4(ORFEA.D0,param);

%% Ⅵ 输出第一帧各尺度下各区域的特征 2016.10.24 21:36PM
FEA = computeFullFea(param,spinfor,ORFEA);

%% Ⅶ 输出字典及相关信息
% 初始字典信息，构建字典（D0，DB）
% DIC.D0 = D0;
DIC.DB = DB;
% DIC.beta = beta;
DIC.model = tmodel;

clear  DB D0 beta model tmodel

% PRE_information
result.fea      = FEA;            clear FEA     % 各尺度下各区域的特征（全尺寸），即是selFea 2016.10.24 9:49AM
result.spinfor  = spinfor;        clear spinfor % 多尺度分割信息
result.ORLabels = ORFEA.ORLabels; clear ORFEA   % 各尺度下各区域同OR的关系
result.imsal    = fpre_GT;
result.imgt     = fpre_GT;        clear fpre_GT
result.spsal    = spSal;          clear spSal   % 各尺度下各区域显著性值（GT区域均值）

clear fpre_Image fpre_GT param flow
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 计算由GT得到的各区域sal与label,用于boosting训练
% 以区域均值作为区域显著性值
function spSal = computeGTinfor(imGT,spinfor)
imGT = double(imGT>=0.5);
spSal = cell(length(spinfor),1);
for ss=1:length(spinfor)
    tmpSP = spinfor{ss,1};
    tmpSPsal = zeros(tmpSP.spNum,1);
%     tmpSPlabel = zeros(tmpSP.spNum,1);
    for sp=1:tmpSP.spNum
        tmpSPsal(sp,1) = mean(imGT(tmpSP.pixelList{sp,1}));     
    end
    tmpSPsal = normalizeSal(tmpSPsal);% 归一化
    spSal{ss,1} = tmpSPsal;
    clear tmpSPsal tmpSP tmpSPlabel
end

clear imGT spinfor objth
end


% 2 计算全尺寸下的各区域的特征（OR外置零）SELF + MULTICONTRAST ----------------
% 计算的是所有尺度下的所有区域的特征 
% 全尺寸输出特征 不分OR内外，各区域均有特征对应  2016.10.24 21:34PM
% 去除一些特征： LBP/GEODESIC/MULTI-CONTEXT
% 加入 LM_texture & LM_textureHist 2016.11.05 9:09AM
% 保留 multi-context 2016.11.05 13:42PM
% 多保留Geodesic特征 2016.11.06 20:57PM
% 2016.11.12 13:54PM
% 去除HOG/GEO/LM_textureture 四种特征
function FEA = computeFullFea(param,spinfor,ORFEA)
% ORFEA.selfFea
%      selfFea{ss,1}.colorHist_rgb 
%      selfFea{ss,1}.colorHist_lab 
%      selfFea{ss,1}.colorHist_hsv 
%      selfFea{ss,1}.lbpHist 
%      selfFea{ss,1}.lbp_top_Hist
%      selfFea{ss,1}.hogHist    
%      selfFea{ss,1}.regionCov   
%      selfFea{ss,1}.geoDist    
%      selfFea{ss,1}.flowHist  
% ORFEA.multiContextFea
% FEA
% 2016.08.24 20:39PM
% 
FEA = cell(length(param.spnumbers),1);
for ss=1:length(param.spnumbers)
    tmpSP = spinfor{ss,1};
%     tmpORlabel = ORFEA.ORLabels{ss,1};
%     ISORlabel = tmpORlabel(:,1);
%     Indexs_out_OR = find(ISORlabel==0);% OR外区域标号
    
    tmpselfFea         = ORFEA.selfFea{ss,1};
    
    if param.numMultiContext
        tmpmultiContextFea = ORFEA.multiContextFea{ss,1};
            numMultiContext = 3;
    else
            numMultiContext = 0;
    end
%     numMultiContext = 0;% multicontext 2016.11.05 13:40PM    
%     % selfFea + multicontrast
%     colorHist_rgb  = zeros(tmpSP.spNum,size(tmpselfFea.colorHist_rgb,2)+numMultiContext);
%     colorHist_lab  = zeros(tmpSP.spNum,size(tmpselfFea.colorHist_lab,2)+numMultiContext);
%     colorHist_hsv  = zeros(tmpSP.spNum,size(tmpselfFea.colorHist_hsv,2)+numMultiContext);
% %     LM_texture     = zeros(tmpSP.spNum,size(tmpselfFea.LM_texture,2)+numMultiContext);
%     LM_textureHist = zeros(tmpSP.spNum,size(tmpselfFea.LM_textureHist,2)+numMultiContext);
% %     lbpHist       = zeros(tmpSP.spNum,size(tmpselfFea.lbpHist,2)+numMultiContext);
%     lbp_top_Hist   = zeros(tmpSP.spNum,size(tmpselfFea.lbp_top_Hist,2)+numMultiContext);
% %     hogHist        = zeros(tmpSP.spNum,size(tmpselfFea.hogHist,2)+numMultiContext);
%     regionCov      = zeros(tmpSP.spNum,size(tmpselfFea.regionCov,2)+numMultiContext);
% %     geoDist        = zeros(tmpSP.spNum,size(tmpselfFea.geoDist,2)+numMultiContext);
%     flowHist       = zeros(tmpSP.spNum,size(tmpselfFea.flowHist,2)+numMultiContext);
    regionFea      = zeros(tmpSP.spNum,size(tmpselfFea.regionFea,2)+numMultiContext);
%     nn=1;
    for sp=1:tmpSP.spNum       
%             if 0 % 无 multi-context
%             colorHist_rgb(sp,:) = [tmpselfFea.colorHist_rgb(sp,:)];
%             colorHist_lab(sp,:) = [tmpselfFea.colorHist_lab(sp,:)];
%             colorHist_hsv(sp,:) = [tmpselfFea.colorHist_hsv(sp,:)];
%             LM_texture(sp,:)    = [tmpselfFea.LM_texture(sp,:)];
%             LM_textureHist(sp,:)= [tmpselfFea.LM_textureHist(sp,:)];
%             lbp_top_Hist(sp,:)  = [tmpselfFea.lbp_top_Hist(sp,:)];
%             hogHist(sp,:)       = [tmpselfFea.hogHist(sp,:)];
%             regionCov(sp,:)     = [tmpselfFea.regionCov(sp,:)];
%             flowHist(sp,:)      = [tmpselfFea.flowHist(sp,:)];
%             end
            
            if param.numMultiContext % 有 multi-context
%             colorHist_rgb(sp,:)  = [tmpselfFea.colorHist_rgb(sp,:), tmpmultiContextFea.colorHist_rgb(sp,:)];
%             colorHist_lab(sp,:)  = [tmpselfFea.colorHist_lab(sp,:), tmpmultiContextFea.colorHist_lab(sp,:)];
%             colorHist_hsv(sp,:)  = [tmpselfFea.colorHist_hsv(sp,:), tmpmultiContextFea.colorHist_hsv(sp,:)];
%             LM_textureHist(sp,:) = [tmpselfFea.LM_textureHist(sp,:),tmpmultiContextFea.LM_textureHist(sp,:)];
%             lbp_top_Hist(sp,:)   = [tmpselfFea.lbp_top_Hist(sp,:),  tmpmultiContextFea.lbp_top_Hist(sp,:)];
%             regionCov(sp,:)      = [tmpselfFea.regionCov(sp,:),     tmpmultiContextFea.regionCov(sp,:)];
%             flowHist(sp,:)       = [tmpselfFea.flowHist(sp,:),      tmpmultiContextFea.flowHist(sp,:)];
            regionFea(sp,:)      = [tmpselfFea.regionFea(sp,:),     tmpmultiContextFea.regionFea(sp,:)];
            else
%             colorHist_rgb(sp,:)  = [tmpselfFea.colorHist_rgb(sp,:)];
%             colorHist_lab(sp,:)  = [tmpselfFea.colorHist_lab(sp,:)];
%             colorHist_hsv(sp,:)  = [tmpselfFea.colorHist_hsv(sp,:)];
%             LM_textureHist(sp,:) = [tmpselfFea.LM_textureHist(sp,:)];
%             lbp_top_Hist(sp,:)   = [tmpselfFea.lbp_top_Hist(sp,:)];
%             regionCov(sp,:)      = [tmpselfFea.regionCov(sp,:)];
%             flowHist(sp,:)       = [tmpselfFea.flowHist(sp,:)];
            regionFea(sp,:)      = [tmpselfFea.regionFea(sp,:)];
            end
       
    end
%     FEA{ss,1}.colorHist_rgb  = colorHist_rgb;
%     FEA{ss,1}.colorHist_lab  = colorHist_lab;
%     FEA{ss,1}.colorHist_hsv  = colorHist_hsv;
% %     FEA{ss,1}.LM_texture     = LM_texture;
%     FEA{ss,1}.LM_textureHist = LM_textureHist;
%     FEA{ss,1}.lbp_top_Hist   = lbp_top_Hist;
% %     FEA{ss,1}.hogHist        = hogHist;
%     FEA{ss,1}.regionCov      = regionCov;
% %     FEA{ss,1}.geoDist        = geoDist;
%     FEA{ss,1}.flowHist       = flowHist;
    FEA{ss,1}.regionFea       = regionFea;
    
    clear colorHist_rgb colorHist_lab colorHist_hsv  lbpHist lbp_top_Hist
    clear LM_texture LM_textureHist hogHist regionCov geoDist flowHist 
end
clear param spinfor ORFEA
end

