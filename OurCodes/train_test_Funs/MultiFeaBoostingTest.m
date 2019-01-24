function result = MultiFeaBoostingTest(ORFEA, beta, model, param)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ������Boostings�㷨���ں���adaboost �� SRC
% ORFEA.selfFea
%  selfFea{ss,1}.colorHist_rgb 
%  selfFea{ss,1}.colorHist_lab 
%  selfFea{ss,1}.colorHist_hsv 
%  selfFea{ss,1}.lbpHist     
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
% V2��2016.08.30 10:43AM
% ��ÿ�����߶��½��е�Ԥ�⣨OR����
% V3: 2016.08.30 19:59PM
% ʹ��PCA���������ֵ䣬
% ������������weakClassifierNew0
% ֮ǰ��weakClassifier
% V4: 2016.10.13 9:01AM
% ʹ�ñ����ֵ����Ԥ��
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SP_SCALE_NUM = length(ORFEA.selfFea);
nfeature = 8;
ntype=1;
result = cell(SP_SCALE_NUM,1);
% beta(:,1) = beta(:,1)/sum(beta(:,1));% �Ը�Ȩ�ؽ��й�һ��
for ss=1:SP_SCALE_NUM
    selfFea         = ORFEA.selfFea{ss,1};
    multiContextFea = ORFEA.multiContextFea{ss,1};
    
   % ���߶��µ�OR�������� sampleNum*Feadim ----------------------------
   d1=[selfFea.colorHist_rgb,multiContextFea.colorHist_rgb];
   d2=[selfFea.colorHist_lab,multiContextFea.colorHist_lab];
   d3=[selfFea.colorHist_hsv,multiContextFea.colorHist_hsv];
   d4=[selfFea.lbpHist,      multiContextFea.lbpHist];
   d5=[selfFea.hogHist,      multiContextFea.hogHist];
   d6=[selfFea.regionCov,    multiContextFea.regionCov];
   d7=[selfFea.geoDist,      multiContextFea.geoDist];
   d8=[selfFea.flowHist,     multiContextFea.flowHist];
    
   clear selfFea multiContextFea
   
   n = size(d1,1);% ������Ŀ
   SalValue  = zeros(1,n);% 1*sampleNum
   LABELS = [];% sampleNum * classifierNum
   % ������Boosting��� -----------------------------------------
   for j = 1:size(beta,1)
      idx = beta(j,2);
      dic_p = model{j,1}.dic.p;
      dic_n = model{j,1}.dic.n;
      mapping_p = model{j,1}.mapping.p;
      mapping_n = model{j,1}.mapping.n;

        switch (floor((idx-1)/ntype))
            case 0; 
                [dec,pred_l] = ...
                       weakClassifierNew0(dic_p',dic_n',mapping_p,mapping_n,d1',param);              
            case 1;
                [dec,pred_l] = ...
                       weakClassifierNew0(dic_p',dic_n',mapping_p,mapping_n,d2',param);      
            case 2;
                [dec,pred_l] = ...
                       weakClassifierNew0(dic_p',dic_n',mapping_p,mapping_n,d3',param);
            case 3;
                [dec,pred_l] = ...
                       weakClassifierNew0(dic_p',dic_n',mapping_p,mapping_n,d4',param);
            case 4;
                [dec,pred_l] = ...
                       weakClassifierNew0(dic_p',dic_n',mapping_p,mapping_n,d5',param);
            case 5;
                [dec,pred_l] = ...
                       weakClassifierNew0(dic_p',dic_n',mapping_p,mapping_n,d6',param);
            case 6;
                [dec,pred_l] = ...
                       weakClassifierNew0(dic_p',dic_n',mapping_p,mapping_n,d7',param);
            case 7;
                [dec,pred_l] = ...
                       weakClassifierNew0(dic_p',dic_n',mapping_p,mapping_n,d8',param);
        end
        
%         dec = normalizeSal(dec);% revised in 2016.08.30 10:46AM
        SalValue = SalValue + beta(j,1) * dec;
        pred_l(pred_l==0) = -1; 
        LABELS = [LABELS,pred_l'];
        
        clear dec pred_l
   end
   
   % ǿ�� & ��һ�� & ��ǩ��������ֵ
   SalValue = normalizeSal(SalValue);
%    SalValue = normal_enhanced(SalValue);
   WW = beta(:,1)';
%    WW = (WW - min(WW(:))) /(max(WW(:)) - min(WW(:)) + eps);
%    if sum(WW) == 0
%        WW = WW/(sum(WW)+eps);
%    else
%        WW = WW/(sum(WW));
%    end
   OG_Label1 = sum(repmat(WW,[size(LABELS,1),1]).*LABELS,2);% >=0 1 , <0 0
   OG_Label = OG_Label1>=0;
   OG_Label = double(OG_Label);
   
%    result{ss,1}.SalValue = SalValue;
   result{ss,1}.OG_Label = OG_Label;
   result{ss,1}.SalValue = normalizeSal(OG_Label);
   clear SalValue LABELS WW 
end

clear ORFEA beta model param
end



