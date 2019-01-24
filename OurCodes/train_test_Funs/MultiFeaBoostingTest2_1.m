function result = MultiFeaBoostingTest2_1(ORFEA, beta, model, param, spinfor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 多特征Boostings算法，内核是adaboost 与 SRC
% ORFEA.selfFea
%  selfFea{ss,1}.colorHist_rgb 
%  selfFea{ss,1}.colorHist_lab 
%  selfFea{ss,1}.colorHist_hsv 
%  selfFea{ss,1}.lbpHist     
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SP_SCALE_NUM = length(ORFEA.selfFea);
nfeature = 8;
ntype=1;
result = cell(SP_SCALE_NUM,1);
% beta(:,1) = beta(:,1)/sum(beta(:,1));% 对各权重进行归一化
for ss=1:SP_SCALE_NUM %单尺度下的预测 
    % 获取OR各区域对应的像素个数，用于获取标签
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
  
   % 此时的 index_in_OR 为所有超像素区域 ， 2016.10.28 8:22AM
   % 单尺度下的OR区域特征 sampleNum*Feadim ----------------------------
   d1=[selfFea.colorHist_rgb(index_in_OR,:),multiContextFea.colorHist_rgb(index_in_OR,:)];
   d2=[selfFea.colorHist_lab(index_in_OR,:),multiContextFea.colorHist_lab(index_in_OR,:)];
   d3=[selfFea.colorHist_hsv(index_in_OR,:),multiContextFea.colorHist_hsv(index_in_OR,:)];
   d4=[selfFea.lbpHist(index_in_OR,:),      multiContextFea.lbpHist(index_in_OR,:)];
   d5=[selfFea.hogHist(index_in_OR,:),      multiContextFea.hogHist(index_in_OR,:)];
   d6=[selfFea.regionCov(index_in_OR,:),    multiContextFea.regionCov(index_in_OR,:)];
   d7=[selfFea.geoDist(index_in_OR,:),      multiContextFea.geoDist(index_in_OR,:)];
   d8=[selfFea.flowHist(index_in_OR,:),     multiContextFea.flowHist(index_in_OR,:)];
    
   clear selfFea multiContextFea
   
   n = size(d1,1);% 样本数目
   SalValue  = zeros(1,n);% 1*sampleNum
   LABELS = [];% sampleNum * classifierNum
   % 多特征Boosting框架 -----------------------------------------
   for j = 1:size(beta,1)
      idx = beta(j,2);
      dic = model{j,1}.dic;
%       dic_p = model{j,1}.dic.p;
%       dic_n = model{j,1}.dic.n;
%       mapping_p = model{j,1}.mapping.p;
%       mapping_n = model{j,1}.mapping.n;

        switch (floor((idx-1)/ntype))
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
            case 7;
                [dec,pred_l] = ...
                       weakClassifierNew1_0(dic,d8',param, numPixel);   
        end
        
%         dec = normalizeSal(dec);% revised in 2016.08.30 10:46AM
        SalValue = SalValue + beta(j,1) * dec;
        pred_l(pred_l==0) = -1; 
        LABELS = [LABELS,pred_l'];
        
        clear dec pred_l
   end
   
   % 强化 & 归一化 & 标签与显著性值
   SalValue = normalizeSal(SalValue);
%    SalValue = normal_enhanced(SalValue);
   WW = beta(:,1)';
   OG_Label1 = sum(repmat(WW,[size(LABELS,1),1]).*LABELS,2);% >=0 1 , <0 0
   OG_Label = OG_Label1>=0;
   OG_Label = double(OG_Label);
   
%    result{ss,1}.SalValue = SalValue;
   result{ss,1}.OG_Label = OG_Label;
   result{ss,1}.SalValue = SalValue;
%    result{ss,1}.SalValue = normalizeSal(OG_Label);
   clear SalValue LABELS WW 
end

clear ORFEA beta model param
end



