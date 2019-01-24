function [result,DIC] = initDic2_2(fpre_Image,fpre_GT,param,flow)
% function [result,DIC] = initDic2_2(fpre_Image,fcur_Image,fnext_Image,fcur_GT,param,flow)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V10: 2016.10.28 7:14AM
% ȥ��OR����ֱ����ȫ�ߴ�ͼ����в���
% 
% V11: 2016.11.2 9:18AM
% ����LBP-TOP
% ���� fpre_pre_Image fpre_Image fcur_Image �ֱ�ָ����һ֡���ڶ�֡������֡ 
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

%% �� ����GT ȷ��object�Ĵ���λ�ã��� operation region (OR):
%%  LCEND(x1,y1,x2,y2), lefttop & rightbottom
objIndex = find(fpre_GT(:)==1);

%% �� ������ȡ��ͼ���ȫ������(LOW & MID LEVEL Feature)
% 2.1 �����طָ��ȡ��߶���Ϣ 
spinfor = multiscaleSLIC(fpre_Image,spnumbers);
    
% 2.2 ��ȡ������self + variance������ȷ����������Ϊѵ������
ORFEA = featureExtractNew2_1(pre_image,spinfor,flow, param,objIndex);

%% �� PCA����˫�ֵ�
% DB = D02DBNew(ORFEA.D0,param);
DB = [];

%% �� �����һ֡������������ͼ�����GT��ʵ����������ֵ��
spSal = computeGTinfor(fpre_GT,spinfor);

%% �� BOOSTING��ܣ�ѵ����
[tmodel] = MultiFeaBoostingTrainNew4(ORFEA.D0,param);

%% �� �����һ֡���߶��¸���������� 2016.10.24 21:36PM
FEA = computeFullFea(param,spinfor,ORFEA);

%% �� ����ֵ估�����Ϣ
% ��ʼ�ֵ���Ϣ�������ֵ䣨D0��DB��
% DIC.D0 = D0;
DIC.DB = DB;
% DIC.beta = beta;
DIC.model = tmodel;

clear  DB D0 beta model tmodel

% PRE_information
result.fea      = FEA;            clear FEA     % ���߶��¸������������ȫ�ߴ磩������selFea 2016.10.24 9:49AM
result.spinfor  = spinfor;        clear spinfor % ��߶ȷָ���Ϣ
result.ORLabels = ORFEA.ORLabels; clear ORFEA   % ���߶��¸�����ͬOR�Ĺ�ϵ
result.imsal    = fpre_GT;
result.imgt     = fpre_GT;        clear fpre_GT
result.spsal    = spSal;          clear spSal   % ���߶��¸�����������ֵ��GT�����ֵ��

clear fpre_Image fpre_GT param flow
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 ������GT�õ��ĸ�����sal��label,����boostingѵ��
% �������ֵ��Ϊ����������ֵ
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
    tmpSPsal = normalizeSal(tmpSPsal);% ��һ��
    spSal{ss,1} = tmpSPsal;
    clear tmpSPsal tmpSP tmpSPlabel
end

clear imGT spinfor objth
end


% 2 ����ȫ�ߴ��µĸ������������OR�����㣩SELF + MULTICONTRAST ----------------
% ����������г߶��µ�������������� 
% ȫ�ߴ�������� ����OR���⣬���������������Ӧ  2016.10.24 21:34PM
% ȥ��һЩ������ LBP/GEODESIC/MULTI-CONTEXT
% ���� LM_texture & LM_textureHist 2016.11.05 9:09AM
% ���� multi-context 2016.11.05 13:42PM
% �ౣ��Geodesic���� 2016.11.06 20:57PM
% 2016.11.12 13:54PM
% ȥ��HOG/GEO/LM_textureture ��������
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
%     Indexs_out_OR = find(ISORlabel==0);% OR��������
    
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
%             if 0 % �� multi-context
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
            
            if param.numMultiContext % �� multi-context
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

