function [tmodel] = MultiFeaBoostingTrainNew3_parallel(D0,param)
% [D0, beta, model, tmodel] = MultiFeaBoostingTrainNew3(DB,D0,ORLabels,spSal,param, spinfor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 训练多特征Boost框架，得到各弱分类器权重
% DB.P.colorHist_rgb_mappedA,DB.P.colorHist_rgb_mapping
% DB.P.colorHist_lab_mappedA,DB.P.colorHist_lab_mapping
% DB.P.colorHist_hsv_mappedA,DB.P.colorHist_hsv_mapping`
% DB.P.lbpHist_mappedA,      DB.P.lbpHist_mapping
% DB.P.hogHist_mappedA,      DB.P.hogHist_mapping
% DB.P.regionCov_mappedA,    DB.P.regionCov_mapping
% DB.P.geoDist_mappedA,      DB.P.geoDist_mapping
% DB.P.flowHist_mappedA,     DB.P.flowHist_mapping
% 
% DB.N类似
% 其中 mapping
% mapping.mean
% mapping.M
% mapping.lambda
% 
% ORFEA.D0  sampleNum*FeaDims
% D0.P D0.N
%     D0.P.colorHist_rgb
%     D0.P.colorHist_lab
%     D0.P.colorHist_hsv
%     D0.P.lbpHist    
%     D0.P.lbp_top_Hist  
%     D0.P.hogHist     
%     D0.P.regionCov    
%     D0.P.geoDist      
%     D0.P.flowHist    
% 
% ORFEA.ORLabels
% GTinfor
% spSal 各区域的显著性值： 4cell spnum
% 
% V1: 2016.08.24 23:00PM
% V2: 2016.08.30 10:12AM
% 多尺度下，正负样本集中一起构成训练集，进行训练，
% 故得出的显著性值 tdec不能进行统一归一化
% 
% V3: 2016.08.30 19:46PM
% 于MultiFeaBoostingTrain基础上进行修改：使用PCA的基向量做为字典
% 舍弃D0输出
% 
% V4: 2016.10.12 10:17AM
% 实现基于学习型背景字典的adaboost算法
% 加入 spinfor ，训练时获取标签
% OR区域中的训练（OR中提取正负样本）
%
% V5： 2016.10.29 10：13AM
% indexP + indexN ~= spNum !!!
%
% V6：2016.10.31 12：30PM
% 去除Adaboost框架，仅是学习字典
% 
% V7: 2016.11.02 11:05AM
% 增加LBP-TOP特征，作为第五个特征，后续顺延
% 
% V8: 2016.11.05 9:10AM
%     D0.P.colorHist_rgb 
%     D0.P.colorHist_lab 
%     D0.P.colorHist_hsv  
%     D0.P.LM_texture  
%     D0.P.LM_textureHist
%     D0.P.lbp_top_Hist  
%     D0.P.hogHist     
%     D0.P.regionCov     
%     D0.P.flowHist   
% V9: 2016.11.09 16:02PM
% 并行
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nfeature = 10;
% 初始字典亦是训练集（仅仅是背景样本）
% 去除了LBP/GEODSIC,增加了 LM
% % init_dic1 = D0.N.colorHist_rgb;
% % init_dic2 = D0.N.colorHist_lab;
% % init_dic3 = D0.N.colorHist_hsv;
% % init_dic4 = D0.N.LM_texture;
% % % init_dic4 = D0.N.lbpHist;
% % init_dic5 = D0.N.lbp_top_Hist;
% % init_dic6 = D0.N.hogHist;
% % init_dic7 = D0.N.regionCov;
% % init_dic8 = D0.N.LM_textureHist;
% % init_dic9 = D0.N.geoDist;
% % init_dic10 = D0.N.flowHist;

%% 训练各弱分类器
tmodel = cell(nfeature,1);
parfor i = 1:nfeature
    fprintf('\n the %d feature ...',i)
    
%     init_dic = eval(['init_dic' num2str(i)]);% 载入初始化字典
%     dic_learn_ornot = param.SR.dic_learn_ornot;
%     switch dic_learn_ornot
%          case 'YES'
%                DIC = DLFun0(init_dic',param);
%          case 'NO'
%                DIC = init_dic';
%     end

% %     final_dic = DLFun0(init_dic',param);% 仅是进行字典学习
    DIC = LearnFun(i,D0,param);
    tmodel{i,1}.dic       = DIC;

%     clear d l final_dic dec_v pred_l sal
%     
%     eval(['clear',' ','init_dic' num2str(i)])
end


end

function DIC = LearnFun(i,D0,param)
init_dic1 = D0.N.colorHist_rgb;
init_dic2 = D0.N.colorHist_lab;
init_dic3 = D0.N.colorHist_hsv;
init_dic4 = D0.N.LM_texture;
% init_dic4 = D0.N.lbpHist;
init_dic5 = D0.N.lbp_top_Hist;
init_dic6 = D0.N.hogHist;
init_dic7 = D0.N.regionCov;
init_dic8 = D0.N.LM_textureHist;
init_dic9 = D0.N.geoDist;
init_dic10 = D0.N.flowHist;

    init_dic = eval(['init_dic' num2str(i)]);% 载入初始化字典
    dic_learn_ornot = param.SR.dic_learn_ornot;
    switch dic_learn_ornot
         case 'YES'
               DIC = DLFun0(init_dic',param);
         case 'NO'
               DIC = init_dic';
    end
    clear d l final_dic dec_v pred_l sal
    eval(['clear',' ','init_dic' num2str(i)])
end