function [result,betas] = MultiFeaBoostingTest3(ORFEA, imSal_pre0, model, param, spinfor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 多特征Boostings算法，内核是adaboost 与 SRC
% ORFEA.selfFea
%  selfFea{ss,1}.colorHist_rgb 
%  selfFea{ss,1}.colorHist_lab 
%  selfFea{ss,1}.colorHist_hsv 
%  selfFea{ss,1}.lbpHist     
%  selfFea{ss,1}.lbp_top_Hist
%  selfFea{ss,1}.hogHist    
%  selfFea{ss,1}.regionCov   
%  selfFea{ss,1}.geoDist    
%  selfFea{ss,1}.flowHist  
% ORFEA.multiContextFea 亦如此
% ORFEA.ORLabels
% 
% beta 分类器权重 & 对应标号
% model
% model{t,1}.dic      dic.p      dic.n
% model{t,1}.mapping  mapping.p  mapping.n
% 
% 其中 mapping
% mapping.mean
% mapping.M
% mapping.lambda
% 
% V1: 2016.08.24 21：27PM
% 基于权重的众数投票 及 BOOSTING 得到标签及显著性值
%
% V2：2016.08.30 10:43AM
% 于每个单尺度下进行的预测（OR区域）
%
% V3: 2016.08.30 19:59PM
% 使用PCA基向量做字典，
% 分类器函数：weakClassifierNew0
% 之前是weakClassifier
%
% V4: 2016.10.13 9:01AM
% 使用背景字典进行预测
% 加入 spinfor ，测试时获取标签
% 
% V5: 2016.10.24 9:56AM
% 去除多context特征
% 
% V6: 2016.10.28 8:23AM
% 所有超像素区域均参与测试，index_in_OR为所有超像素区域
%
% V7: 2016.10.31 12:57PM
% 新的融合方式,同时返回WS，即各弱分类器的权重值
% imSal_pre0 前一帧的
% betas 所有尺度下的各组b弱分类器权重beta
% 
% V8: 2016.11.02 11:05AM
% 增加LBP-TOP特征，作为第五个特征，后续顺延;共9种特征
%
% 2016.11.12 13:54PM
% 去除HOG/GEO/LM_textureture 四种特征
% 同时舍弃最差那一个 featureMap!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SP_SCALE_NUM = length(ORFEA.selfFea);
nfeature = 7;
ntype=1;
result = cell(SP_SCALE_NUM,1);
betas = cell(SP_SCALE_NUM,1);
[height,width,dims]  = size(imSal_pre0);
FeaNames = {'rgb','lab','hsv','lbp-top','regonCov','LM-texturehist','flow'};
%% BEGIN &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
for ss=1:SP_SCALE_NUM %单尺度下的预测 
    %% 1 获取OR各区域对应的像素个数，用于获取标签 --------------------
    numPixel = [];
    tmpORlabel      = ORFEA.ORLabels{ss,1};
    ISORlabel       = tmpORlabel(:,1);
    tmpSP           = spinfor{ss,1};
    tmpPixellist    = tmpSP.pixelList;
    
    index_in_OR     = find(ISORlabel==1);
    for pp=1:length(index_in_OR)
        tmpPP = tmpPixellist{index_in_OR(pp)};
        numPixel = [numPixel;length(tmpPP)];
    end

    % original
    selfFea         = ORFEA.selfFea{ss,1};
    multiContextFea = ORFEA.multiContextFea{ss,1};
%    multiContextFea = [];% 去除multi-context/geodesic/lbp，2016.11.05 9:22AM
   %% 2 此时的 index_in_OR 为所有超像素区域 ， 2016.10.28 8:22AM ----
   % 单尺度下的OR区域特征 sampleNum*Feadim ----------------------------
   if 0 % 无 multi-context 
   d1=[selfFea.colorHist_rgb(index_in_OR,:)];
   d2=[selfFea.colorHist_lab(index_in_OR,:)];
   d3=[selfFea.colorHist_hsv(index_in_OR,:)];
   d4=[selfFea.LM_texture(index_in_OR,:)];
   d5=[selfFea.lbp_top_Hist(index_in_OR,:)];
   d6=[selfFea.hogHist(index_in_OR,:)];
   d7=[selfFea.regionCov(index_in_OR,:)];
   d8=[selfFea.LM_textureHist(index_in_OR,:)];
   d9=[selfFea.flowHist(index_in_OR,:)];
   end
   
   if 1 % 有 multi-context
   d1 = [selfFea.colorHist_rgb(index_in_OR,:), multiContextFea.colorHist_rgb(index_in_OR,:)];
   d2 = [selfFea.colorHist_lab(index_in_OR,:), multiContextFea.colorHist_lab(index_in_OR,:)];
   d3 = [selfFea.colorHist_hsv(index_in_OR,:), multiContextFea.colorHist_hsv(index_in_OR,:)];
%    d4 = [selfFea.LM_texture(index_in_OR,:),    multiContextFea.LM_texture(index_in_OR,:)];
%    d4=[selfFea.lbpHist(index_in_OR,:),      multiContextFea.lbpHist(index_in_OR,:)];
   d4 = [selfFea.lbp_top_Hist(index_in_OR,:),  multiContextFea.lbp_top_Hist(index_in_OR,:)];
%    d6 = [selfFea.hogHist(index_in_OR,:),       multiContextFea.hogHist(index_in_OR,:)];
   d5 = [selfFea.regionCov(index_in_OR,:),     multiContextFea.regionCov(index_in_OR,:)];
   d6 = [selfFea.LM_textureHist(index_in_OR,:),multiContextFea.LM_textureHist(index_in_OR,:)];
%    d9 = [selfFea.geoDist(index_in_OR,:),       multiContextFea.geoDist(index_in_OR,:)];
   d7= [selfFea.flowHist(index_in_OR,:),      multiContextFea.flowHist(index_in_OR,:)];
   end
   clear selfFea multiContextFea
   
   n = size(d1,1);% 样本数目
   SalValue  = [];% spNum * sampleNum
   LABELS = [];% sampleNum * classifierNum
   beta = zeros(nfeature,2);% 各尺度下对应的一组权重值
   diffs = [];
   
   %% 3 多特征Boosting框架 -----------------------------------------
   for j = 1:nfeature
      dic = model{j,1}.dic;

        switch (floor((j-1)/ntype))
            case 0; 
                [dec,pred_l] = ...
                       weakClassifierNew1_0(dic,d1',param, numPixel);            
            case 1;
                [dec,pred_l] = ...
                       weakClassifierNew1_0(dic,d2',param, numPixel);         
            case 2;
                [dec,pred_l] = ...
                       weakClassifierNew1_0(dic,d3',param, numPixel);   
            case 3;
                [dec,pred_l] = ...
                       weakClassifierNew1_0(dic,d4',param, numPixel);   
            case 4;
                [dec,pred_l] = ...
                       weakClassifierNew1_0(dic,d5',param, numPixel);   
            case 5;
                [dec,pred_l] = ...
                       weakClassifierNew1_0(dic,d6',param, numPixel);   
            case 6;
                [dec,pred_l] = ...
                       weakClassifierNew1_0(dic,d7',param, numPixel);   
%             case 7;
%                 [dec,pred_l] = ...
%                        weakClassifierNew1_0(dic,d8',param, numPixel);   
%                                
%             case 8;
%                 [dec,pred_l] = ...
%                        weakClassifierNew1_0(dic,d9',param, numPixel); 
% 
%             case 9;
%                 [dec,pred_l] = ...
%                        weakClassifierNew1_0(dic,d10',param, numPixel); 
        end
        
        dec = normalizeSal(dec);
        [m,n] = size(dec);
        if m==1
            dec = dec';
        end
%         SalValue = SalValue + beta(j,1) * dec;
        SalValue = [SalValue,dec];
        pred_l(pred_l==0) = -1; 
        LABELS = [LABELS,pred_l'];
        
       [tmpFeamap, ~] = CreateImageFromSPs(dec, tmpPixellist, height, width, true);
%        figure,imshow(tmpFeamap,[]),title(FeaNames{1,j})% testing 2016.11.12 14:45PM
        tmpdiff       = tmpFeamap - imSal_pre0;
        tmpdiff       = sum(tmpdiff(:).*tmpdiff(:))/length(tmpdiff(:));% 平均平方误差
        diffs         = [diffs,tmpdiff];
        
        clear dec pred_l tmpdiff tmpFeamap
   end

   %% 4 计算预测值同前一帧的差异，用于获得各特征图之权重
   [valueDiff,indexDiff] = max(diffs);
   indexWorse = indexDiff(end);% 最差特征图的编号！！！
   diffs = exp(-2*diffs./(mean(diffs)+eps));
   WS = diffs;
%    WS   = diffs./(sum(diffs)+eps);
   beta(:,1) = WS';
   beta(:,2) = [1:nfeature]';
   
   betas{ss,1} = beta;
   SalValue(:,indexWorse) = 0;% 最差显著性图置零！！！
   SalValue = SalValue.*repmat(WS,[size(SalValue,1),1]);
   SalValue = sum(SalValue,2);
   SalValue = normalizeSal(SalValue);
   
   % 在初步融合的结果的基础上，引入前帧的显著性图，进行进一步的优化 2016.11.12
   enhanceRatio      = param.enhhanceRatio;
   [SalValue_Img, ~] = CreateImageFromSPs(SalValue, tmpPixellist, height, width, true);
   SalValue_Img      = SalValue_Img + exp(enhanceRatio * imSal_pre0);
   SalValue_Img      = normalizeSal(SalValue_Img);
   SalValue_Img      = guidedfilter(SalValue_Img,SalValue_Img,6,0.1);
   SalValue_Img      = normalizeSal(SalValue_Img);
%    figure,imshow(SalValue_Img,[]),title('init')
   SalValue          = computeRegionSal(SalValue_Img,tmpPixellist);
   % ---------------------------------------------------------------------
   
   LABELS(:,indexWorse) = 0;% 最差显著性图标签置零！！！
   OG_Label1 = sum(repmat(WS,[size(LABELS,1),1]).*LABELS,2);% >=0 1 , <0 0
   OG_Label = OG_Label1>=0;
   OG_Label = double(OG_Label);
  
   result{ss,1}.OG_Label = OG_Label;
   result{ss,1}.SalValue = SalValue;
   
   clear SalValue LABELS WW beta tmpPixellist SalValue_Img indexWorse
end

clear ORFEA imSal_pre0 model param spinfor
end


% 2 根据初始融合结果，计算各尺度下的显著性值 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
function regionSal = computeRegionSal(refImage,pixelList)
regionSal = zeros(length(pixelList),1);

for i=1:length(pixelList)
    regionSal(i,1) = mean(refImage(pixelList{i,1}));
end
regionSal = normalizeSal(regionSal);

clear refImage pixelList
end
