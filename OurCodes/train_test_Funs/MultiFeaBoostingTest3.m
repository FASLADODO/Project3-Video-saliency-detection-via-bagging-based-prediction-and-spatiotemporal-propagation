function [result,betas] = MultiFeaBoostingTest3(ORFEA, imSal_pre0, model, param, spinfor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ������Boostings�㷨���ں���adaboost �� SRC
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
% ORFEA.multiContextFea �����
% ORFEA.ORLabels
% 
% beta ������Ȩ�� & ��Ӧ���
% model
% model{t,1}.dic      dic.p      dic.n
% model{t,1}.mapping  mapping.p  mapping.n
% 
% ���� mapping
% mapping.mean
% mapping.M
% mapping.lambda
% 
% V1: 2016.08.24 21��27PM
% ����Ȩ�ص�����ͶƱ �� BOOSTING �õ���ǩ��������ֵ
%
% V2��2016.08.30 10:43AM
% ��ÿ�����߶��½��е�Ԥ�⣨OR����
%
% V3: 2016.08.30 19:59PM
% ʹ��PCA���������ֵ䣬
% ������������weakClassifierNew0
% ֮ǰ��weakClassifier
%
% V4: 2016.10.13 9:01AM
% ʹ�ñ����ֵ����Ԥ��
% ���� spinfor ������ʱ��ȡ��ǩ
% 
% V5: 2016.10.24 9:56AM
% ȥ����context����
% 
% V6: 2016.10.28 8:23AM
% ���г����������������ԣ�index_in_ORΪ���г���������
%
% V7: 2016.10.31 12:57PM
% �µ��ںϷ�ʽ,ͬʱ����WS����������������Ȩ��ֵ
% imSal_pre0 ǰһ֡��
% betas ���г߶��µĸ���b��������Ȩ��beta
% 
% V8: 2016.11.02 11:05AM
% ����LBP-TOP��������Ϊ���������������˳��;��9������
%
% 2016.11.12 13:54PM
% ȥ��HOG/GEO/LM_textureture ��������
% ͬʱ���������һ�� featureMap!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SP_SCALE_NUM = length(ORFEA.selfFea);
nfeature = 7;
ntype=1;
result = cell(SP_SCALE_NUM,1);
betas = cell(SP_SCALE_NUM,1);
[height,width,dims]  = size(imSal_pre0);
FeaNames = {'rgb','lab','hsv','lbp-top','regonCov','LM-texturehist','flow'};
%% BEGIN &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
for ss=1:SP_SCALE_NUM %���߶��µ�Ԥ�� 
    %% 1 ��ȡOR�������Ӧ�����ظ��������ڻ�ȡ��ǩ --------------------
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
%    multiContextFea = [];% ȥ��multi-context/geodesic/lbp��2016.11.05 9:22AM
   %% 2 ��ʱ�� index_in_OR Ϊ���г��������� �� 2016.10.28 8:22AM ----
   % ���߶��µ�OR�������� sampleNum*Feadim ----------------------------
   if 0 % �� multi-context 
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
   
   if 1 % �� multi-context
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
   
   n = size(d1,1);% ������Ŀ
   SalValue  = [];% spNum * sampleNum
   LABELS = [];% sampleNum * classifierNum
   beta = zeros(nfeature,2);% ���߶��¶�Ӧ��һ��Ȩ��ֵ
   diffs = [];
   
   %% 3 ������Boosting��� -----------------------------------------
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
        tmpdiff       = sum(tmpdiff(:).*tmpdiff(:))/length(tmpdiff(:));% ƽ��ƽ�����
        diffs         = [diffs,tmpdiff];
        
        clear dec pred_l tmpdiff tmpFeamap
   end

   %% 4 ����Ԥ��ֵͬǰһ֡�Ĳ��죬���ڻ�ø�����ͼ֮Ȩ��
   [valueDiff,indexDiff] = max(diffs);
   indexWorse = indexDiff(end);% �������ͼ�ı�ţ�����
   diffs = exp(-2*diffs./(mean(diffs)+eps));
   WS = diffs;
%    WS   = diffs./(sum(diffs)+eps);
   beta(:,1) = WS';
   beta(:,2) = [1:nfeature]';
   
   betas{ss,1} = beta;
   SalValue(:,indexWorse) = 0;% ���������ͼ���㣡����
   SalValue = SalValue.*repmat(WS,[size(SalValue,1),1]);
   SalValue = sum(SalValue,2);
   SalValue = normalizeSal(SalValue);
   
   % �ڳ����ںϵĽ���Ļ����ϣ�����ǰ֡��������ͼ�����н�һ�����Ż� 2016.11.12
   enhanceRatio      = param.enhhanceRatio;
   [SalValue_Img, ~] = CreateImageFromSPs(SalValue, tmpPixellist, height, width, true);
   SalValue_Img      = SalValue_Img + exp(enhanceRatio * imSal_pre0);
   SalValue_Img      = normalizeSal(SalValue_Img);
   SalValue_Img      = guidedfilter(SalValue_Img,SalValue_Img,6,0.1);
   SalValue_Img      = normalizeSal(SalValue_Img);
%    figure,imshow(SalValue_Img,[]),title('init')
   SalValue          = computeRegionSal(SalValue_Img,tmpPixellist);
   % ---------------------------------------------------------------------
   
   LABELS(:,indexWorse) = 0;% ���������ͼ��ǩ���㣡����
   OG_Label1 = sum(repmat(WS,[size(LABELS,1),1]).*LABELS,2);% >=0 1 , <0 0
   OG_Label = OG_Label1>=0;
   OG_Label = double(OG_Label);
  
   result{ss,1}.OG_Label = OG_Label;
   result{ss,1}.SalValue = SalValue;
   
   clear SalValue LABELS WW beta tmpPixellist SalValue_Img indexWorse
end

clear ORFEA imSal_pre0 model param spinfor
end


% 2 ���ݳ�ʼ�ںϽ����������߶��µ�������ֵ &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
function regionSal = computeRegionSal(refImage,pixelList)
regionSal = zeros(length(pixelList),1);

for i=1:length(pixelList)
    regionSal(i,1) = mean(refImage(pixelList{i,1}));
end
regionSal = normalizeSal(regionSal);

clear refImage pixelList
end
