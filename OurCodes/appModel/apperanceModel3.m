function [CURINFOR_2,imwriteInfor] = ...
    apperanceModel3( fcur_Image , fnext_Image, ...
                     PRE_INFOR  , PRE_DIC    , ...
                     OPTICALFLOW, index_f_cur, param       ,saveInfor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V1�� 2016.11.18 16��26PM
% ���� double TP ����
% MVF_Foward_f_fn, MVF_Foward_f_fp, ...
% Copyright by xiaofei zhou, IVPLab, shanghai univeristy,shanghai, china
% http://www.ivp.shu.edu.cn
% email: zxforchid@163.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
%%                          1 �ɵ�2֡��֪��BOOST_SAL & TPSAL_pre                       &&
%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n the processing based on the frame %d *************************************',index_f_cur)
load ([OPTICALFLOW,'opf_',num2str(index_f_cur),'.mat']) 
[TPSAL_1_2,Boost_SALS_1_2,IMSAL_BOOST_SALS_1_2,CURINFOR_2,spinforCur0] = ...
              step1fun(fcur_Image     ,  ...
                       PRE_INFOR      , PRE_DIC        , ...
                       MVF_Foward_f_fp, MVF_Foward_f_fn, ...
                       param);

clear MVF_Foward_fnnn_f MVF_Foward_fnn_f MVF_Foward_fn_f MVF_Foward_f_fn 
clear MVF_Foward_f_fnn MVF_Foward_f_fnnn MVF_Foward_f_fnnnn MVF_Foward_f_fp MVF_Foward_f_fpp

%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
%%          2 �ɵ�3֡��֪��BOOST_SAL & TPSAL & SPSAL �����Ľ�һ������ӳ��                &&
%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n the processing based on the frame %d *************************************',(index_f_cur+1))
load ([OPTICALFLOW,'opf_',num2str(index_f_cur+1),'.mat']) 
[CURINFOR_3] = ...
              step2fun(fnext_Image     , ...
                       PRE_INFOR      , PRE_DIC         ,  ...
                       MVF_Foward_f_fp, MVF_Foward_f_fnn, ...
                       param          , saveInfor);
clear MVF_Foward_fnnn_f MVF_Foward_fnn_f MVF_Foward_fn_f MVF_Foward_f_fn 
clear MVF_Foward_f_fnn MVF_Foward_f_fnnn MVF_Foward_f_fnnnn MVF_Foward_f_fp MVF_Foward_f_fpp

%% V ����ʱ���򴫲� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n backward temporal propagation ...............................')
load ([OPTICALFLOW,'opf_',num2str(index_f_cur),'.mat']) 
%  3--->2ͶӰ��Ѱ����ؼ�(��2��ÿһ������ͬ3���ڵ���ϵ)          
%                                            3             2             2--->3
[spinforCur2] = findTemporalAdjNew2(CURINFOR_3.spinfor, spinforCur0, MVF_Foward_f_fp);
CURINFOR_2.spinfor2 = spinforCur2;% ���� 2 ͶӰ�� 3 ����ؼ�
clear spinforCur2 spinforCur0

%��3��2����                                             2          3
[TPSAL_3_2,TPIMG_3_2] = temporalPropagationNew4_1(CURINFOR_2,CURINFOR_3,param);
CURINFOR_2.spinforCur2 = [];

clear CURINFOR_3 
clear MVF_Foward_fnnn_f MVF_Foward_fnn_f MVF_Foward_fn_f MVF_Foward_f_fn 
clear MVF_Foward_f_fnn MVF_Foward_f_fnnn MVF_Foward_f_fnnnn MVF_Foward_f_fp MVF_Foward_f_fpp

%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
%%                             3 ��Ϣ�ںϲ�����һ���Ŀ��򴫲�                            &&
%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n the integration process and spatial refinement ****************************************\n')
fprintf('\n integration .................................................')
% ���� BOOSTSAL & TPSAL �� integration, 2016.11.22
[height,width,dims] = size(fcur_Image);
cur_image = double(fcur_Image);
[TPSAL1,IMSAL_TPSAL1] =  ...
    integrate_Boost_TP_SAL1(Boost_SALS_1_2,TPSAL_1_2,TPSAL_3_2,CURINFOR_2.spinfor,height,width);
clear Boost_SALS_1_2 TPSAL_1_2 TPSAL_3_2 

% 3.1 ���򴫲�   
fprintf('\n spatial propagation .........................................')
[TPSPSAL_Img,TPSPSAL_RegionSal] = SP19FUN(TPSAL1,CURINFOR_2,cur_image,[],param);
% clear MVF_Foward_f_fp

%4. ��ȡ��ֵ��GT_CUR��OSTU/MEAN��
fprintf('\n obtain GT ...................................................')
threshold = graythresh(TPSPSAL_Img);
IMGT = im2bw(TPSPSAL_Img,threshold);

% 5 imwriteInfor д���ļ��е�ͼ��
fprintf('\n save imwrite information ....................................')
imwriteInfor = struct;
% 5.1. ֱ�ӵķ�����
imwriteInfor.IMSAL_BOOST_SALS1 = IMSAL_BOOST_SALS_1_2;

% 5.2. ��ʱ�򴫲��Ľ��
imwriteInfor.IMSAL_TPSAL1 = IMSAL_TPSAL1;

% 5.3. ʱ���򴫲���Ľ��
imwriteInfor.IMSAL_SPSAL1 = TPSPSAL_Img;

%% save information
CURINFOR_2.spsal    = TPSPSAL_RegionSal;     % ���߶��¸�����Ľ�����˴���ʱ��Ľ�� + ����Ľ���������� 2016.10.20 9:08AM
CURINFOR_2.imsal    = TPSPSAL_Img;      % 2016.10.19 22:46PM  
CURINFOR_2.imgt     = IMGT;            % ��ֵͼ������computeGTinfor���� 2016.10.23 18:48PM

%% clear 
clear fpre_Image fcur_Image fnext_Image fnext_next_Image 
clear PRE_INFOR PRE_DIC 
clear OPTICALFLOW index_f_cur param saveInfor
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% �ӳ��� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0 �õ���2֡�� BOOSTResult & TPSAL1 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
function [TPSAL1,BoostSAL,IMSAL_BOOST_SALS1,CURINFOR,spinforCur0] = ...
              step1fun(fcur_Image     ,  ...
                       PRE_INFOR      , PRE_DIC        , ...
                       MVF_Foward_f_fp, MVF_Foward_f_fn,  ...
                       param)
%% ��������ȡ����ȡ��ǰ֡������ &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n initial + SLIC + extractFea .................................')
cur_image  = double(fcur_Image);

%2. SLIC
spinforCur = multiscaleSLIC(fcur_Image,param.spnumbers);

%3. ������ȡ ��OR�����������
ORFEA =  ...
    featureExtractNew2_1(cur_image,spinforCur,MVF_Foward_f_fp,param);
CURINFOR.ORLabels =  ORFEA.ORLabels;
% CURINFOR.regionFea = ORFEA.regionFea;

%% �� BOOSTING���(���ԣ�OR�����Ԥ��) &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n prediction ..................................................')
[BoostSAL,IMSAL_BOOST_SALS1] = ...
    MultiFeaBoostingTest4_1(ORFEA, PRE_INFOR.imsal, PRE_DIC.model, param, spinforCur);

%% �� �����ǰ֡���߶��¸������������OR���ȫ��Ϊ0����Ϊȫ�ߴ�״̬��&&&&&  
fprintf('\n obtain full feature .........................................')
FEA = computeFullFea(param,spinforCur,ORFEA);
CURINFOR.fea = FEA;

%% �� ʱ���򴫲� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n temporal propagation ........................................')
%1. Ϊ��ǰ֡ÿ��������ǰһ֡��Ѱ����ƥ������ȫ�ߴ磩  &&&&&&&&&&&&&&&&&&&&&&
spinforCur0 = spinforCur;
[spinforCur] = findTemporalAdjNew2(PRE_INFOR.spinfor, spinforCur,MVF_Foward_f_fn);% 2--->1 f_fn
CURINFOR.spinfor = spinforCur;
 
%2. ���� (��ؼ���OR���� + beta) ����ʱ�򴫲� (ȫ�ߴ�),�����ж�߶��ں� &&&&&&
[TPSAL1,IMSAL_TPSAL1] = temporalPropagationNew4(CURINFOR,PRE_INFOR,PRE_DIC.model);

clear fpre_Image fcur_Image fnext_Image 
clear PRE_INFOR PRE_DIC MVF_Foward_f_fp MVF_Foward_f_fn param
end

% 00 �õڶ�֡���ֵ估imsal����predict & ���� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% 1 Ԥ��&���� 3  1--->3
% 3 ��cur_image
function  [CURINFOR] = ...
              step2fun(fcur_Image     ,  ...
                       PRE_PRE_INFOR  , PRE_PRE_DIC     ,  ...
                       MVF_Foward_f_fp, MVF_Foward_f_fnn, ...
                       param          ,saveInfor)

%% ��������ȡ����ȡ��ǰ֡������ &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n initial + SLIC + extractFea ..................................')
[height,width,dims] = size(fcur_Image);
cur_image  = fcur_Image;

%2. SLIC
spinforCur = multiscaleSLIC(fcur_Image,param.spnumbers);

%3. ������ȡ ��OR�����������
ORFEA =  ...
    featureExtractNew2_1(cur_image,spinforCur,MVF_Foward_f_fp,param);% ������4--->5  f_fp ������
CURINFOR.ORLabels =  ORFEA.ORLabels;
% CURINFOR.regionFea = ORFEA.regionFea;

%% �� BOOSTING���(���ԣ�OR�����Ԥ��) &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n prediction ..................................................')
[BoostSAL,IMSAL_BOOST_SALS1] = ...
    MultiFeaBoostingTest4_1(ORFEA, PRE_PRE_INFOR.imsal, PRE_PRE_DIC.model, param, spinforCur);

%% �� �����ǰ֡���߶��¸������������OR���ȫ��Ϊ0����Ϊȫ�ߴ�״̬��&&&&&  
fprintf('\n obtain full feature .........................................')
FEA = computeFullFea(param,spinforCur,ORFEA);
CURINFOR.fea = FEA;
clear FEA

%% �� ʱ���򴫲� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n temporal & spatial propagation ..............................')
%1. Ϊ��ǰ֡ÿ��������ǰһ֡��Ѱ����ƥ������ȫ�ߴ磩  &&&&&&&&&&&&&&&&&&&&&&
%    3--->1 ����ؼ�                          1               3          3--->1
[spinforCur] = findTemporalAdjNew2(PRE_PRE_INFOR.spinfor, spinforCur, MVF_Foward_f_fnn);
CURINFOR.spinfor = spinforCur;
 
%2. ���� (��ؼ���OR���� + beta) ����ʱ�򴫲� (ȫ�ߴ�),�����ж�߶��ں� &&&&&&
%    1 ʱ�򴫲� 3                          3           1            1    
[TPSAL,TPIMG] = temporalPropagationNew4(CURINFOR,PRE_PRE_INFOR,PRE_PRE_DIC.model);
[TPSAL1,IMSAL_TPSAL1] =  ...
    integrate_Boost_TP_SAL(TPSAL,BoostSAL,spinforCur,height,width);
clear TPIMG TPSAL spinforCur

% 3. ���򴫲�   &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% [TPSPSAL_Img,TPSPSAL_RegionSal,~] =  ...
%     spatialPropagationNew10(CURINFOR,IMSAL_TPSAL1,TPSAL1,param, cur_image,param.gpSign,saveInfor,PRE_PRE_DIC.model);
% [TPSPSAL_Img,TPSPSAL_RegionSal] = spatialPropagationNew8_2(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp);
% [TPSPSAL_Img,TPSPSAL_RegionSal] = spatialPropagationNew12_0(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
[TPSPSAL_Img,TPSPSAL_RegionSal] = SP19FUN(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
clear TPSAL1

% 4 save information������� 4 ֡�������Ϣ�� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&     
CURINFOR.spsal    = TPSPSAL_RegionSal;     
% CURINFOR.imsal    = normalizeSal(TPSPSAL_Img+IMSAL_TPSAL1); % ʱ�����ٽ��һ�Σ�����  
CURINFOR.imsal    = TPSPSAL_Img;

clear ORFEA TPSPSAL_RegionSal TPSPSAL_Img IMSAL_TPSAL1

clear fpre_Image fcur_Image fnext_Image 
clear PRE_PRE_INFOR PRE_PRE_DIC 
clear MVF_Foward_f_fp MVF_Foward_f_fnn 
clear param saveInfor

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%1. ����ȫ�ߴ��µĸ������������OR�����㣩SELF + MULTICONTRAST ----------------
% ȫ�ߴ�������� ����OR���⣬���������������Ӧ  2016.10.24 21:34PM
% ȥ��һЩ������ LBP/GEODESIC/MULTI-CONTEXT
% ���� LM_texture & LM_textureHist 2016.11.05 9:09AM
% ���� multi-context 2016.11.05 13:42PM
% �ౣ��Geodesic���� 2016.11.06 20:57PM
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

