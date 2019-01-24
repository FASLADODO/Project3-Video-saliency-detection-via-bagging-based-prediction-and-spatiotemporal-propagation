% function UPDATA_DIC = updateDIC7(CURINFOR,param,fpre_GT,MVF_Foward_fn_f)
function UPDATA_DIC = updateDIC7(CURINFOR,param)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��ǰһ֡��GT���Ƶ�ǰ֡��objIndex����ȡ
% copyright by xiaofei zhou
% V1: 2016.10.09 19:12pm
% ͳһǰ������λ����Ϣ���� 
% V3�� 2016.10.13 14��08PM
% ����ǰ���ı����ֵ䣬�ڴ˴�����΢����Ӧ
% V4�� 2016.10.28 9:44AM
% ֱ�Ӳ��õ�ǰ֡Ԥ����֮��ֵͼ�����б�������֮ѡ��
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%REGION_SALS = CURINFOR.spsal;% ���߶�֮����������ͼ
fcur_gt = CURINFOR.imgt;
newGT = fcur_gt;
% union = fpre_GT + fcur_gt;
% union(union==2) = 1;
% newGT = union;

objIndex = find(newGT(:)==1);
% LCEND = CURINFOR.LCEND;
% ORLabels = computeORLabel(LCEND, objIndex, CURINFOR.spinfor,param);
ORLabels = computeORLabel(objIndex, CURINFOR.spinfor,param);

% ORLabels = computeORLabelNew(CURINFOR,param);


SPSCALENUM = length(CURINFOR.spinfor);
CURD0 = computeD0(SPSCALENUM,CURINFOR.spinfor,ORLabels,CURINFOR.fea);



%% 3.1 ��������������ֵ (�����ֵ��Ϊ������ֵ)---------------------
% gt2spSal = computeGTinfor(fcur_gt,CURINFOR.spinfor);

%% 3.2 ��ȡDB������D0 ----------------------
    DB = [];
    UPDATA_DIC.DB = DB; 
    UPDATA_DIC.D0 = CURD0;
    clear CURD0 DB 
    
%   % revised in 2016.10.13 14:06PM  (�µ�ѵ��ѧϰ��ʽ) --- 
%    [UPDATA_DIC.D0, UPDATA_DIC.beta, UPDATA_DIC.model, tmodel] = ...
%        MultiFeaBoostingTrainNew2(UPDATA_DIC.DB,UPDATA_DIC.D0,ORLabels,gt2spSal,param,CURINFOR.spinfor);
% [UPDATA_DIC.model] = MultiFeaBoostingTrainNew3(UPDATA_DIC.D0,param);% 2016.10.31 12:55PM
[UPDATA_DIC.model] = MultiFeaBoostingTrainNew4(UPDATA_DIC.D0,param);% 2016.10.31 12:55PM
%% clear
clear fpre_GT fcur_gt CURINFOR THl PRE_DIC
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%1 ����D0(����cell�¼��н��в���)
% ����ȷ������������ѵ���������� object; ������ border�� 2016.10.09 19:48PM
% 2016.10.13 18:59PM ѡ���������� ��1,1����1,0��
% ��ʱ�� ISORLabel ȫΪ 1
% 2016.11.02 14:09PM �����µ�����LBP-TOP���ܼƹ�9������
% ����Geodesic ������ �ϼƹ�10�������� 2016.11.06 21��12PM
% ȥ��OR���� 2016.11.18 8:03AM
function D0 = computeD0(ScaleNums,spinfor,ORLabels,fea)
% fea ȫ�ߴ磬 sampleNum = tmpSP.spNum�� ����OR�����������������
    D0.P = struct;D0.N = struct;% sampleNum*feaDim
%     DP_colorHist_rgb = []; DN_colorHist_rgb = [];
%     DP_colorHist_lab = []; DN_colorHist_lab = [];
%     DP_colorHist_hsv = []; DN_colorHist_hsv = [];
%     DP_LM_textureHist= []; DN_LM_textureHist= [];
%     DP_lbptop_Hist   = []; DN_lbptop_Hist   = [];
%     DP_regionCov     = []; DN_regionCov     = [];
%     DP_flowHist      = []; DN_flowHist      = [];
    DP_regionFea     = []; DN_regionFea     = [];
    
    for ss=1:ScaleNums
        tmpSP = spinfor{ss,1};
%         tmpORlabel = ORLabels{ss,1};% spNum*3
%         ISORlabel = tmpORlabel(:,1);
% %         index_out_OR = find(ISORlabel~=1);
%         ISOBJlabel = tmpORlabel(:,3);
%         PNlabel = ISORlabel.*ISOBJlabel;% (1,1) P 
%         indexP = find(PNlabel==1);% P�����������е�index���
% 
%         
%         % revised in 2016.10.13 19:03PM ------
%         % ���������ֵ�Ԫ�� OR=1 OBJECT=0, N�����������е�index���
%         if 1 
%         indexN = [];
%         for dd=1:length(ISORlabel)
%             if ISORlabel(dd)==1 && ISOBJlabel(dd)==0
%                 indexN = [indexN;dd];
%             end
%         end 
%         end
        ISOBJECT = ORLabels{ss,1};% 1/0/50/100
        indexP = find(ISOBJECT==1);% ȷ��OR��������Щ�� object 
        indexN = find(ISOBJECT==0);

        
%         DP_colorHist_rgb = [DP_colorHist_rgb; fea{ss,1}.colorHist_rgb(indexP,:)];
%         DP_colorHist_lab = [DP_colorHist_lab; fea{ss,1}.colorHist_lab(indexP,:)];
%         DP_colorHist_hsv = [DP_colorHist_hsv; fea{ss,1}.colorHist_hsv(indexP,:)];  
% %         DP_LM_texture    = [DP_LM_texture;    fea{ss,1}.LM_texture(indexP,:)]; 
%         DP_LM_textureHist= [DP_LM_textureHist;fea{ss,1}.LM_textureHist(indexP,:)]; 
% %         DP_lbpHist       = [DP_lbpHist;      fea{ss,1}.lbpHist(indexP,:)];
%         DP_lbptop_Hist   = [DP_lbptop_Hist;   fea{ss,1}.lbp_top_Hist(indexP,:)];
% %         DP_hogHist       = [DP_hogHist;       fea{ss,1}.hogHist(indexP,:)];
%         DP_regionCov     = [DP_regionCov;     fea{ss,1}.regionCov(indexP,:)];
% %         DP_geoDist       = [DP_geoDist;       fea{ss,1}.geoDist(indexP,:)];
%         DP_flowHist      = [DP_flowHist;      fea{ss,1}.flowHist(indexP,:)];
        DP_regionFea     = [DP_regionFea;     fea{ss,1}.regionFea(indexP,:)];
        
%         DN_colorHist_rgb = [DN_colorHist_rgb; fea{ss,1}.colorHist_rgb(indexN,:)];
%         DN_colorHist_lab = [DN_colorHist_lab; fea{ss,1}.colorHist_lab(indexN,:)];
%         DN_colorHist_hsv = [DN_colorHist_hsv; fea{ss,1}.colorHist_hsv(indexN,:)]; 
% %         DN_LM_texture    = [DN_LM_texture;    fea{ss,1}.LM_texture(indexN,:)]; 
%         DN_LM_textureHist= [DN_LM_textureHist;fea{ss,1}.LM_textureHist(indexN,:)]; 
% %         DN_lbpHist       = [DN_lbpHist;      fea{ss,1}.lbpHist(indexN,:)];
%         DN_lbptop_Hist   = [DN_lbptop_Hist;   fea{ss,1}.lbp_top_Hist(indexN,:)];
% %         DN_hogHist       = [DN_hogHist;       fea{ss,1}.hogHist(indexN,:)];
%         DN_regionCov     = [DN_regionCov;     fea{ss,1}.regionCov(indexN,:)];
% %         DN_geoDist       = [DN_geoDist;       fea{ss,1}.geoDist(indexN,:)];
%         DN_flowHist      = [DN_flowHist;      fea{ss,1}.flowHist(indexN,:)];    
        DN_regionFea     = [DN_regionFea;     fea{ss,1}.regionFea(indexN,:)];
        
    end
%     D0.P.colorHist_rgb  = DP_colorHist_rgb; 
%     D0.P.colorHist_lab  = DP_colorHist_lab; 
%     D0.P.colorHist_hsv  = DP_colorHist_hsv; 
% %     D0.P.LM_texture     = DP_LM_texture;
%     D0.P.LM_textureHist = DP_LM_textureHist;
% %     D0.P.lbpHist       = DP_lbpHist;
%     D0.P.lbp_top_Hist   = DP_lbptop_Hist;
% %     D0.P.hogHist        = DP_hogHist;
%     D0.P.regionCov      = DP_regionCov;
% %     D0.P.geoDist        = DP_geoDist;
%     D0.P.flowHist       = DP_flowHist;
    D0.P.regionFea      = DP_regionFea;
    
%     D0.N.colorHist_rgb  = DN_colorHist_rgb; 
%     D0.N.colorHist_lab  = DN_colorHist_lab; 
%     D0.N.colorHist_hsv  = DN_colorHist_hsv; 
% %     D0.N.LM_texture     = DN_LM_texture;
%     D0.N.LM_textureHist = DN_LM_textureHist;
% %     D0.N.lbpHist       = DN_lbpHist;
%     D0.N.lbp_top_Hist   = DN_lbptop_Hist;
% %     D0.N.hogHist        = DN_hogHist;
%     D0.N.regionCov      = DN_regionCov;
% %     D0.N.geoDist        = DN_geoDist;
%     D0.N.flowHist       = DN_flowHist;
    D0.N.regionFea      = DN_regionFea;
    
clear ScaleNums spinfor ORLabels fea DN_LM_texture DN_LM_textureHist
clear DP_colorHist_rgb DP_colorHist_lab DP_colorHist_hsv DP_lbpHist DP_hogHist DP_regionCov DP_geoDist DP_flowHist DP_lbptop_Hist
clear DN_colorHist_rgb DN_colorHist_lab DN_colorHist_hsv DN_lbpHist DN_hogHist DN_regionCov DN_geoDist DN_flowHist DN_lbptop_Hist 

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
