function [CURINFOR_3,imwriteInfor] = ...
    apperanceModel3_0(fcur_Image,fnext_Image,f_next_next_Image,...
                      PRE_INFOR,PRE_PRE_INFOR,PRE_DIC,PRE_PRE_DIC,...
                      OPTICALFLOW, index_f_cur,param,saveInfor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V2 2016.12.10 
% 引入 double TP 机制
% 这里以 1,2,3,4,5 为例
% 其中3: fcur_Image, 1,2 分别为pre_pre,pre;4,5分别为next,next_next 
%
% Copyright by xiaofei zhou, IVPLab, shanghai univeristy,shanghai, china
% http://www.ivp.shu.edu.cn
% email: zxforchid@163.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
%%                          1 由第 3 帧推知的BOOST_SAL & TPSAL_pre                     &&
%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n the processing based on the frame %d *************************************',index_f_cur)
load ([OPTICALFLOW,'opf_',num2str(index_f_cur),'.mat']) % 载入第3帧光流

[TPSAL_12_3,BoostSAL_12_3,IMSAL_BOOST_SALS_12_3,CURINFOR_3,spinforCur_3_0] = ...
                         step1fun( fcur_Image     , ...
                                   PRE_INFOR      , PRE_PRE_INFOR  , PRE_DIC         , PRE_PRE_DIC, ...
                                   MVF_Foward_f_fp, MVF_Foward_f_fn, MVF_Foward_f_fnn, ...
                                   param);
clear MVF_Foward_fnnn_f MVF_Foward_fnn_f MVF_Foward_fn_f MVF_Foward_f_fn 
clear MVF_Foward_f_fnn MVF_Foward_f_fnnn MVF_Foward_f_fnnnn MVF_Foward_f_fp MVF_Foward_f_fpp
                               

%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
%%          2 由第4帧推知的BOOST_SAL & TPSAL & SPSAL 及做的进一步后向映射                &&
%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n the processing based on the frame %d *************************************',(index_f_cur+1))
load ([OPTICALFLOW,'opf_',num2str(index_f_cur+1),'.mat']) % 载入第4帧光流
% 此时 cur(3)-->pre, next(4)-->cur, next_next(5)-->next
[CURINFOR_4] = step2fun(fnext_Image     ,  ...
                        PRE_INFOR       , PRE_PRE_INFOR   , PRE_DIC            ,PRE_PRE_DIC, ...
                        MVF_Foward_f_fp , MVF_Foward_f_fnn, MVF_Foward_f_fnnn  , ...
                        param           , saveInfor);
clear MVF_Foward_fnnn_f MVF_Foward_fnn_f MVF_Foward_fn_f MVF_Foward_f_fn 
clear MVF_Foward_f_fnn MVF_Foward_f_fnnn MVF_Foward_f_fnnnn MVF_Foward_f_fp MVF_Foward_f_fpp

% 4-->3 时域反向传播 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n backward temporal propagation ...............................')
[TPSAL_4_3,TPIMG_4_3,CURINFOR_3] =  ...
                                 tpsp_on_nexts(OPTICALFLOW   ,index_f_cur,index_f_cur+1, param , ...
                                               spinforCur_3_0,CURINFOR_3 ,CURINFOR_4);
% load ([OPTICALFLOW,'opf_',num2str(index_f_cur),'.mat']) % 载入第3帧光流
% % 3向4映射，对于3中的每一区域，于4中寻找相关集
% %         3--->4 的mapset                    4               3          3--->4
% [spinforCur2] = findTemporalAdjNew2(CURINFOR_4.spinfor, spinforCur_3_0, MVF_Foward_f_fp);
% CURINFOR_3.spinfor2 = spinforCur2;
% clear spinforCur2 
% 
% %         4--->3 的传播                                3        4
% [TPSAL_4_3,TPIMG_4_3] = temporalPropagationNew4_1(CURINFOR_3,CURINFOR_4,param);
% CURINFOR_3.spinforCur2 = [];
% 
% clear CURINFOR_4 
% clear MVF_Foward_fnnn_f MVF_Foward_fnn_f MVF_Foward_fn_f MVF_Foward_f_fn 
% clear MVF_Foward_f_fnn MVF_Foward_f_fnnn MVF_Foward_f_fnnnn MVF_Foward_f_fp MVF_Foward_f_fpp

%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
%%          3 由第5帧推知的BOOST_SAL & TPSAL & SPSAL 及做的进一步后向映射                &&
%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n the processing based on the frame %d *************************************',(index_f_cur+2))
load ([OPTICALFLOW,'opf_',num2str(index_f_cur+2),'.mat']) % 载入第5帧光流
% 此时 next_next(5)-->cur
[CURINFOR_5] = step2fun(f_next_next_Image     ,  ...
                        PRE_INFOR       , PRE_PRE_INFOR   , PRE_DIC            ,PRE_PRE_DIC, ...
                        MVF_Foward_f_fp , MVF_Foward_f_fnnn, MVF_Foward_f_fnnnn  , ...
                        param           , saveInfor);
clear MVF_Foward_fnnn_f MVF_Foward_fnn_f MVF_Foward_fn_f MVF_Foward_f_fn 
clear MVF_Foward_f_fnn MVF_Foward_f_fnnn MVF_Foward_f_fnnnn MVF_Foward_f_fp MVF_Foward_f_fpp

% 5-->3 时域反向传播 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n backward temporal propagation ...............................')
[TPSAL_5_3,TPIMG_5_3,CURINFOR_3] =  ...
                                 tpsp_on_nexts(OPTICALFLOW   ,index_f_cur,index_f_cur+2, param , ...
                                               spinforCur_3_0,CURINFOR_3 ,CURINFOR_5);

%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
%%                             4 信息融合并做进一步的空域传播                            &&
%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n the integration process and spatial refinement ****************************************\n')
fprintf('\n integration .................................................')
[height,width,dims] = size(fcur_Image);
cur_image = double(fcur_Image);
[TPSAL1,IMSAL_TPSAL1] =  ...
    integrate_Boost_TP_SAL2(BoostSAL_12_3,TPSAL_12_3,TPSAL_4_3,TPSAL_5_3,CURINFOR_3.spinfor,height,width);
clear Boost_SALS_2_3 TPSAL_2_3 TPSAL_4_3 TPSAL_5_3

% 3.1 空域传播   
fprintf('\n spatial propagation .........................................')
t5 = clock;
[TPSPSAL_Img,TPSPSAL_RegionSal] = SP19FUN(TPSAL1,CURINFOR_3,cur_image,[],param);
t6 = clock;
deltat_spatiopropagation = etime(t6,t5);

clear TPSAL1 fpre_Image fcur_Image fnext_Image f_next_next_Image cur_image

%4. 获取二值化GT_CUR（OSTU/MEAN）
fprintf('\n obtain GT ...................................................')
threshold = graythresh(TPSPSAL_Img);
IMGT = im2bw(TPSPSAL_Img,threshold);

% 5 imwriteInfor 写入文件中的图像
fprintf('\n save imwrite information ....................................')
imwriteInfor = struct;

% 5.1. 直接的分类结果
imwriteInfor.IMSAL_BOOST_SALS1 = IMSAL_BOOST_SALS_12_3;

% 5.2. 仅时域传播的结果
imwriteInfor.IMSAL_TPSAL1 = IMSAL_TPSAL1;

% 5.3. 时空域传播后的结果
imwriteInfor.IMSAL_SPSAL1 = TPSPSAL_Img;

%% save information
CURINFOR_3.spsal    = TPSPSAL_RegionSal;  
CURINFOR_3.imsal    = TPSPSAL_Img;        
CURINFOR_3.imgt     = IMGT;            
%% clear 
clear FEA ORFEA SPSALS spinforCur IMSAL_SPSAL1 IMGT TPSPSAL_RegionSal TPSPSAL_Img
clear IMSAL_BOOST_SALS0 IMSAL_BOOST_SALS1 IMSAL_TPSAL0 IMSAL_TPSAL1 IMSAL_SPSAL0
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 子程序 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0 计算第四帧的 BoostSal & TPSal &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% eg, 1/2帧预测第 3 帧 
% 此时的 cur_Image 是第三帧
function [TPSAL1,Boost_SAL,IMSAL_BOOST_SALS,CURINFOR,spinforCur0] = ...
              step1fun(fcur_Image     ,  ...
                       PRE_INFOR      , PRE_PRE_INFOR  , PRE_DIC         , PRE_PRE_DIC, ...
                       MVF_Foward_f_fp, MVF_Foward_f_fn, MVF_Foward_f_fnn, ...
                       param)
%% Ⅰ特征提取：提取当前帧的特征 
fprintf('\n initial + SLIC + extractFea .................................')
% pre_image  = double(fpre_Image);
cur_image  = double(fcur_Image);
% next_image = double(fnext_Image);

%2. SLIC
spinforCur = multiscaleSLIC(fcur_Image,param.spnumbers);
CURINFOR.spinfor = spinforCur;
spinforCur0 = spinforCur;

%3. 特征提取
ORFEA = ...
    featureExtractNew2_1(cur_image,spinforCur,MVF_Foward_f_fp,param);
CURINFOR.ORLabels =  ORFEA.ORLabels;
% CURINFOR.regionFea = ORFEA.regionFea;

%% Ⅱ BOOSTING框架(测试；OR区域的预测) 
fprintf('\n prediction ..................................................')
t5 = clock;
[Boost_SAL,IMSAL_BOOST_SALS] = ...
    predict_bypre(ORFEA,PRE_DIC,PRE_PRE_DIC,param,spinforCur,PRE_INFOR);
t6 = clock;
deltat_predictiontest = etime(t6,t5);

%% Ⅲ 输出当前帧各尺度下各区域的特征（全尺寸特征输出， 2016.10.24 21:48PM）
fprintf('\n obtain full feature .........................................')
FEA = computeFullFea(param,spinforCur,ORFEA);
CURINFOR.fea = FEA;
clear FEA

%% Ⅳ 时空域传播 
fprintf('\n temporal and spatial propagation ............................')
%2. 利用 (相关集（OR集） + beta) 进行时域传播 (全尺寸),并进行多尺度融合 
% 这里要注意无 BoostResult!!! 2016.11.22
[TPSAL1,IMSAL_TPSAL1] =  ...
    temporalPP_bypre1(PRE_DIC,PRE_PRE_DIC,PRE_INFOR,PRE_PRE_INFOR,...
                      MVF_Foward_f_fn,MVF_Foward_f_fnn,...
                      CURINFOR,spinforCur);

clear fpre_Image fcur_Image fnext_Image PRE_INFOR 
clear PRE_DIC PRE_PRE_DIC MVF_Foward_f_fp param
end

% 00 计算第五帧的信息 TPSPSAL &&&&&&&&&&&& &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% 利用1,2帧的字典进行预测 第 4/5 帧，传播，分别对应 PRE_PRE/ PRE_PRE_PRE
% 4 f--->fnn/f--->fnnn 时域传播之用
% 5 f--->fnnn/f--->fnnnn 时域传播之用
% 
function [CURINFOR] = ...
              step2fun(fcur_Image     , ...
                       PRE_PRE_INFOR  , PRE_PRE_PRE_INFOR, PRE_PRE_DIC      , PRE_PRE_PRE_DIC, ...
                       MVF_Foward_f_fp, MVF_Foward_f_fnn , MVF_Foward_f_fnnn, ...
                       param          , saveInfor)
%% Ⅰ特征提取：提取当前帧的特征 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n initial + SLIC + extractFea .................................')
% pre_image  = double(fpre_Image);
cur_image  = double(fcur_Image);
% next_image = double(fnext_Image);

%2. SLIC
spinforCur = multiscaleSLIC(fcur_Image,param.spnumbers);
CURINFOR.spinfor = spinforCur;

%3. 特征提取
ORFEA = ...
    featureExtractNew2_1(cur_image,spinforCur,MVF_Foward_f_fp,param);
CURINFOR.ORLabels =  ORFEA.ORLabels;
% CURINFOR.regionFea = ORFEA.regionFea;

%% Ⅱ BOOSTING框架(测试；OR区域的预测)  &&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n prediction ..................................................')
[Boost_SAL,IMSAL_BOOST_SALS] = ...
    predict_bypre(ORFEA,PRE_PRE_DIC,PRE_PRE_PRE_DIC,param,spinforCur,PRE_PRE_INFOR);
clear IMSAL_BOOST_SALS

%% Ⅲ 输出当前帧各尺度下各区域的特征（全尺寸特征输出， 2016.10.24 21:48PM）
fprintf('\n obtain full feature .........................................')
FEA = computeFullFea(param,spinforCur,ORFEA);
CURINFOR.fea = FEA;
clear   FEA

%% Ⅳ 时空域传播 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
fprintf('\n temporal and spatial propagation ............................')
%2. 利用 (相关集（OR集） + beta) 进行时域传播 (全尺寸),并进行多尺度融合 
[TPSAL1,IMSAL_TPSAL1] =  ...
    temporalPP_bypre(PRE_PRE_DIC,PRE_PRE_PRE_DIC,PRE_PRE_INFOR,PRE_PRE_PRE_INFOR,...
                     MVF_Foward_f_fnn,MVF_Foward_f_fnnn,...
                     CURINFOR,spinforCur,Boost_SAL);

% 3. 空域传播 
[TPSPSAL_Img,TPSPSAL_RegionSal] = SP19FUN(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
                             
CURINFOR.spsal    = TPSPSAL_RegionSal;       
CURINFOR.imsal    = TPSPSAL_Img;

clear ORFEA TPSPSAL_Img IMSAL_TPSAL1 TPSPSAL_RegionSal
clear MVF_Foward_f_fp MVF_Foward_f_fnn MVF_Foward_f_fnnn
clear fpre_Image fcur_Image fnext_Image 
clear PRE_PRE_INFOR PRE_PRE_PRE_INFOR PRE_PRE_DIC PRE_PRE_PRE_DIC
end

% 000 
% 利用4/5的信息做反向传播
% 下面是以4为例进行说明！！！
% 需要载入当前帧的光流信息
function [TPSAL_4_3,TPIMG_4_3,CURINFOR_3] =  ...
                                 tpsp_on_nexts(OPTICALFLOW   ,index_f_cur,ID, param , ...
                                               spinforCur_3_0,CURINFOR_3 ,CURINFOR_4)

%% 4-->3 时域反向传播 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% fprintf('\n backward temporal propagation ...............................')
load ([OPTICALFLOW,'opf_',num2str(index_f_cur),'.mat']) % 载入第3帧光流
if ID==index_f_cur+1 % 第4帧
% 3向4映射，对于3中的每一区域，于4中寻找相关集
%         3--->4 的mapset                    4               3          3--->4
[spinforCur2] = findTemporalAdjNew2(CURINFOR_4.spinfor, spinforCur_3_0, MVF_Foward_f_fp);
CURINFOR_3.spinfor2 = spinforCur2;
end
if ID==index_f_cur+2 % 第5帧
[spinforCur2] = findTemporalAdjNew2(CURINFOR_4.spinfor, spinforCur_3_0, MVF_Foward_f_fpp);
CURINFOR_3.spinfor2 = spinforCur2;   
end
clear spinforCur2 spinforCur_3_0

%         4--->3 的传播                                3        4
[TPSAL_4_3,TPIMG_4_3] = temporalPropagationNew4_1(CURINFOR_3,CURINFOR_4,param);
CURINFOR_3.spinfor2 = [];

clear CURINFOR_4 
clear MVF_Foward_fnnn_f MVF_Foward_fnn_f MVF_Foward_fn_f MVF_Foward_f_fn 
clear MVF_Foward_f_fnn MVF_Foward_f_fnnn MVF_Foward_f_fnnnn MVF_Foward_f_fp MVF_Foward_f_fpp

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. 计算全尺寸下的各区域的特征（OR外置零）SELF + MULTICONTRAST &&&&&&&&&&&&&%
% 全尺寸输出特征 不分OR内外，各区域均有特征对应  2016.10.24 21:34PM
% 去除一些特征： LBP/GEODESIC/MULTI-CONTEXT
% 加入 LM_texture & LM_textureHist 2016.11.05 9:09AM
% 保留 multi-context 2016.11.05 13:42PM
% 多保留Geodesic特征 2016.11.06 20:57PM
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

% 2. 利用 pre & pre_pre 计算预测当前帧 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&%
% 提取了多context特征 2016.10.24 10:12AM
% 权重改为 0.5/0.5
% 以各尺度相加求平均的结果作为像素级显著性图！！！
% [BoostResult,IMSAL_BOOST_SALS1] = MultiFeaBoostingTest4_1(ORFEA, PRE_INFOR.imsal, PRE_DIC.model, param, spinforCur);
function [BoostResult_cur,IMSAL_BOOST_SALS1] = ...
                           predict_bypre(ORFEA,PRE_DIC,   PRE_PRE_DIC,...
                                         param,spinforCur,PREINFOR)
%--------------------------------------------------------------------------
% prediction
[BoostResult_pre,IMSAL_BOOST_SALS_pre]         = ...
    MultiFeaBoostingTest4_1(ORFEA, PREINFOR.imsal, PRE_DIC.model,     param, spinforCur);% 2016.10.31 12:21PM
[BoostResult_pre_pre,IMSAL_BOOST_SALS_pre_pre] = ...
    MultiFeaBoostingTest4_1(ORFEA, PREINFOR.imsal, PRE_PRE_DIC.model, param, spinforCur);% 2016.10.31 12:21PM
% BoostResult{ss,1}.OG_Label 
% BoostResult{ss,1}.SalValue

% IMSAL_BOOST_SALS1 = normalizeSal(IMSAL_BOOST_SALS_pre + IMSAL_BOOST_SALS_pre_pre);
 IMSAL_BOOST_SALS1 = 0;
clear IMSAL_BOOST_SALS_pre_pre IMSAL_BOOST_SALS_pre

aa=0.5;
% fusion
BoostResult_cur = cell(length(BoostResult_pre_pre),1);
for ss=1:length(BoostResult_pre_pre)
    BoostResult_cur{ss,1}.SalValue = ...
        normalizeSal(aa*(BoostResult_pre{ss,1}.SalValue) + (1-aa)*(BoostResult_pre_pre{ss,1}.SalValue));   
    BoostResult_cur{ss,1}.OG_Label = ...
        normalizeSal(aa*(BoostResult_pre{ss,1}.OG_Label) + (1-aa)*(BoostResult_pre_pre{ss,1}.OG_Label));
    BoostResult_cur{ss,1}.PP_Img   = ...
        normalizeSal(aa*(BoostResult_pre{ss,1}.PP_Img) + (1-aa)*(BoostResult_pre_pre{ss,1}.PP_Img));
    IMSAL_BOOST_SALS1 = IMSAL_BOOST_SALS1 + BoostResult_cur{ss,1}.PP_Img;
    
    [height,width,dims] = size(BoostResult_cur{ss,1}.PP_Img);
    [rcenter_BoostResult,ccenter_BoostResult] = computeObjectCenter(BoostResult_cur{ss,1}.PP_Img);
    
    tmpSPinfor   = spinforCur{ss,1};
    regionCenter = tmpSPinfor.region_center;
    regionDist_BoostResult = ...
        computeRegion2CenterDist(regionCenter,[rcenter_BoostResult,ccenter_BoostResult],[height,width]);
    BoostResult_compactness = computeCompactness(BoostResult_cur{ss,1}.SalValue,regionDist_BoostResult);
    BoostResult_compactness = 1/(BoostResult_compactness);    
    BoostResult_cur{ss,1}.compactness = BoostResult_compactness;
    clear tmpSPinfor regionCenter regionDist_BoostResult BoostResult_compactness
end
IMSAL_BOOST_SALS1 = normalizeSal(IMSAL_BOOST_SALS1);

clear ORFEA param spinforCur cur_image

end


% 3. 利用 pre & pre_pre 进行时域传播 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&%
% findTemporalAdjNew2 全尺寸状态下的相关集 2016.10.24 10:50AM
% temporalPropagationNew2 赋值区域显著性值时无归一化 2016.10.24 22:15PM
% 基于compactness的融合准则  2016.11.15 16:02PM
% 去除OR限制，2016.11.18 8:16AM
% TPSAL1  SalValue/compactness/PP_Img 2016.11.18
% 先相加，再做归一化，最后在同BoostResult相加； 2016.11.22
function [TPSAL1,IMSAL_TPSAL1] =  ...
    temporalPP_bypre(PRE_DIC,PRE_PRE_DIC,PRE_INFOR,PRE_PRE_INFOR,...
                     MVF_Foward_f_fn,MVF_Foward_f_fnn,...
                     CURINFOR,spinforCur,BoostResult)
%--------------------------------------------------------------------------
% RATIOS = RATIOS./(sum(RATIOS)+eps);
[height,width,dims] = size(MVF_Foward_f_fn);
% 1 pre -------------------------------------------------------------------
[spinforCur_pre] = findTemporalAdjNew2(PRE_INFOR.spinfor, spinforCur,MVF_Foward_f_fn);
CURINFOR.spinfor = spinforCur_pre;
[TPSAL_pre,TPIMG_pre] = temporalPropagationNew4_1_1(CURINFOR,PRE_INFOR,PRE_DIC.model);

% 2 pre_pre
% % TPSAL_pre 不需要归一化， 两次映射叠加 2016.10.24 22:14PM(舍弃) ----------
[spinforCur_pre_pre] = findTemporalAdjNew2(PRE_PRE_INFOR.spinfor, spinforCur,MVF_Foward_f_fnn);
 CURINFOR.spinfor = spinforCur_pre_pre;
[TPSAL_pre_pre,TPIMG_pre_pre] = temporalPropagationNew4_1_1(CURINFOR,PRE_PRE_INFOR,PRE_PRE_DIC.model);


% 3 integrate pre & pre_pre
IMSAL_TPSAL1 = 0;% 时域传播后的各尺度平均后的像素级显著性图
TPSAL1 = cell(length(BoostResult),1); % 新的时域传播后的各尺度下的区域显著性值
for ss=1:length(BoostResult)
    tmpSPinfor  = spinforCur{ss,1};
    regionCenter = tmpSPinfor.region_center;
    
    % 3.1   PRE & PRE_PRE 先 integrate -------------------
    tmp_Pre_Sal =  ...
        (TPSAL_pre{ss,1}.SalValue + TPSAL_pre_pre{ss,1}.SalValue);% 先相加，再归一化
    tmp_Pre_Sal = normalizeSal(tmp_Pre_Sal);
    [PP_Img, ~] = CreateImageFromSPs(tmp_Pre_Sal, tmpSPinfor.pixelList, height, width, true);
    [rcenter_PP,ccenter_PP] = computeObjectCenter(PP_Img);
    regionDist_PP = computeRegion2CenterDist(regionCenter,[rcenter_PP,ccenter_PP],[height,width]);
    pre_compactness = computeCompactness(tmp_Pre_Sal,regionDist_PP);
    pre_compactness = 1/pre_compactness;
    clear PP_Img rcenter_PP ccenter_PP regionDist_PP
    
    % 3.2 与 BoostPredict integrate ---------------------
    tmp_Boost_RegionSal   = BoostResult{ss,1}.SalValue;
    tmp_Boost_Compactness = BoostResult{ss,1}.compactness; 
    
    wboost = tmp_Boost_Compactness  /(tmp_Boost_Compactness + pre_compactness);
    wpre   = pre_compactness        /(tmp_Boost_Compactness + pre_compactness);
    if wboost<0.2
        wboost = 0;
    end
    
    if wpre<0.2
        wpre=0;
    end
    wboost = 1;wpre = 1;
    TPSAL_regional = normalizeSal(wboost*tmp_Boost_RegionSal + wpre*tmp_Pre_Sal);
    TPSAL1{ss,1}.SalValue = TPSAL_regional;
    
    % 3.3 integrate 后， 计算结果的 PP_Img & compactness -----------------
    [tmp_IMSAL_TPSAL, ~] = CreateImageFromSPs(TPSAL_regional, tmpSPinfor.pixelList, height, width, true);
    TPSAL1{ss,1}.PP_Img  = tmp_IMSAL_TPSAL;
    IMSAL_TPSAL1 = IMSAL_TPSAL1 +tmp_IMSAL_TPSAL;
    
    [rcenter_sal,ccenter_sal] = computeObjectCenter(tmp_IMSAL_TPSAL);
    regionDist_sal = computeRegion2CenterDist(regionCenter,[rcenter_sal,ccenter_sal],[height,width]);
    tmp_Sal_compactness = computeCompactness(TPSAL_regional,regionDist_sal);
    TPSAL1{ss,1}.compactness = 1/tmp_Sal_compactness; 
    clear PP_Img rcenter_PP ccenter_PP regionDist_PP  tmp_Sal_compactness
    
    % 3.3 clear some variables ------------------------------------------
    clear tmp_Boost_RegionSal tmp_Boost_Compactness tmp_Pre_Sal pre_compactness
    clear TPSAL_regional tmp_IMSAL_TPSAL 
end


IMSAL_TPSAL1 = normalizeSal(IMSAL_TPSAL1);

clear PRE_DIC PRE_PRE_DIC PRE_INFOR PRE_PRE_INFOR MVF_Foward_f_fn MVF_Foward_f_fnn CURINFOR spinforCur BOOST_SALS
end

% 用于 step1fun  仅是时域传播的 integration,未加入BoostResult,
% 这里要注意同temporalPP_bypre的区别！！！
% 先相加，再做归一化； 2016.11.22  
function [TPSAL1,IMSAL_TPSAL1] =  ...
    temporalPP_bypre1(PRE_DIC,PRE_PRE_DIC,PRE_INFOR,PRE_PRE_INFOR,...
                     MVF_Foward_f_fn,MVF_Foward_f_fnn,...
                     CURINFOR,spinforCur)
%--------------------------------------------------------------------------
% RATIOS = RATIOS./(sum(RATIOS)+eps);
[height,width,dims] = size(MVF_Foward_f_fn);
% 1 pre -------------------------------------------------------------------
t5 = clock;
[spinforCur_pre] = findTemporalAdjNew2(PRE_INFOR.spinfor, spinforCur,MVF_Foward_f_fn);
CURINFOR.spinfor = spinforCur_pre;
% [TPSAL_pre,TPIMG_pre] = temporalPropagationNew4(CURINFOR,PRE_INFOR,PRE_DIC.model);
[TPSAL_pre,TPIMG_pre] = temporalPropagationNew4_1_1(CURINFOR,PRE_INFOR,PRE_DIC.model);
t6 = clock;
deltat_temporalpropagation = etime(t6,t5);


% 2 pre_pre
% % TPSAL_pre 不需要归一化， 两次映射叠加 2016.10.24 22:14PM(舍弃) ----------
[spinforCur_pre_pre] = findTemporalAdjNew2(PRE_PRE_INFOR.spinfor, spinforCur,MVF_Foward_f_fnn);
 CURINFOR.spinfor = spinforCur_pre_pre;
% [TPSAL_pre_pre,TPIMG_pre_pre] = temporalPropagationNew4(CURINFOR,PRE_PRE_INFOR,PRE_PRE_DIC.model);
[TPSAL_pre_pre,TPIMG_pre_pre] = temporalPropagationNew4_1_1(CURINFOR,PRE_PRE_INFOR,PRE_PRE_DIC.model);


% 3 integrate pre & pre_pre
IMSAL_TPSAL1 = 0;% 时域传播后的各尺度平均后的像素级显著性图
TPSAL1 = cell(length(TPSAL_pre),1); % 新的时域传播后的各尺度下的区域显著性值
for ss=1:length(TPSAL_pre)
    tmpSPinfor  = spinforCur{ss,1};
    regionCenter = tmpSPinfor.region_center;
    
    % 3.1   PRE & PRE_PRE 先 integrate -------------------
    tmp_Pre_Sal =  ...
        (TPSAL_pre{ss,1}.SalValue + TPSAL_pre_pre{ss,1}.SalValue);% 先相加，再归一化
    tmp_Pre_Sal = normalizeSal(tmp_Pre_Sal);
    [PP_Img, ~] = CreateImageFromSPs(tmp_Pre_Sal, tmpSPinfor.pixelList, height, width, true);
    [rcenter_PP,ccenter_PP] = computeObjectCenter(PP_Img);
    regionDist_PP = computeRegion2CenterDist(regionCenter,[rcenter_PP,ccenter_PP],[height,width]);
    pre_compactness = computeCompactness(tmp_Pre_Sal,regionDist_PP);
    pre_compactness = 1/pre_compactness;

    TPSAL1{ss,1}.SalValue = tmp_Pre_Sal;
    TPSAL1{ss,1}.PP_Img  = PP_Img;
    TPSAL1{ss,1}.compactness = pre_compactness; 
    IMSAL_TPSAL1 = IMSAL_TPSAL1 + PP_Img;
    
    clear PP_Img rcenter_PP ccenter_PP regionDist_PP  
    
    % 3.3 clear some variables ------------------------------------------
    clear tmp_Boost_RegionSal tmp_Boost_Compactness tmp_Pre_Sal pre_compactness
    clear TPSAL_regional tmp_IMSAL_TPSAL 
end


IMSAL_TPSAL1 = normalizeSal(IMSAL_TPSAL1);

clear PRE_DIC PRE_PRE_DIC PRE_INFOR PRE_PRE_INFOR MVF_Foward_f_fn MVF_Foward_f_fnn CURINFOR spinforCur BOOST_SALS
end



