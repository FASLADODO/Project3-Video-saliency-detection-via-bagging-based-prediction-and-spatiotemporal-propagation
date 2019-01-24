function [tmodel] = MultiFeaBoostingTrainNew3_parallel(D0,param)
% [D0, beta, model, tmodel] = MultiFeaBoostingTrainNew3(DB,D0,ORLabels,spSal,param, spinfor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ѵ��������Boost��ܣ��õ�����������Ȩ��
% DB.P.colorHist_rgb_mappedA,DB.P.colorHist_rgb_mapping
% DB.P.colorHist_lab_mappedA,DB.P.colorHist_lab_mapping
% DB.P.colorHist_hsv_mappedA,DB.P.colorHist_hsv_mapping`
% DB.P.lbpHist_mappedA,      DB.P.lbpHist_mapping
% DB.P.hogHist_mappedA,      DB.P.hogHist_mapping
% DB.P.regionCov_mappedA,    DB.P.regionCov_mapping
% DB.P.geoDist_mappedA,      DB.P.geoDist_mapping
% DB.P.flowHist_mappedA,     DB.P.flowHist_mapping
% 
% DB.N����
% ���� mapping
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
% spSal �������������ֵ�� 4cell spnum
% 
% V1: 2016.08.24 23:00PM
% V2: 2016.08.30 10:12AM
% ��߶��£�������������һ�𹹳�ѵ����������ѵ����
% �ʵó���������ֵ tdec���ܽ���ͳһ��һ��
% 
% V3: 2016.08.30 19:46PM
% ��MultiFeaBoostingTrain�����Ͻ����޸ģ�ʹ��PCA�Ļ�������Ϊ�ֵ�
% ����D0���
% 
% V4: 2016.10.12 10:17AM
% ʵ�ֻ���ѧϰ�ͱ����ֵ��adaboost�㷨
% ���� spinfor ��ѵ��ʱ��ȡ��ǩ
% OR�����е�ѵ����OR����ȡ����������
%
% V5�� 2016.10.29 10��13AM
% indexP + indexN ~= spNum !!!
%
% V6��2016.10.31 12��30PM
% ȥ��Adaboost��ܣ�����ѧϰ�ֵ�
% 
% V7: 2016.11.02 11:05AM
% ����LBP-TOP��������Ϊ���������������˳��
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
% ����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nfeature = 10;
% ��ʼ�ֵ�����ѵ�����������Ǳ���������
% ȥ����LBP/GEODSIC,������ LM
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

%% ѵ������������
tmodel = cell(nfeature,1);
parfor i = 1:nfeature
    fprintf('\n the %d feature ...',i)
    
%     init_dic = eval(['init_dic' num2str(i)]);% �����ʼ���ֵ�
%     dic_learn_ornot = param.SR.dic_learn_ornot;
%     switch dic_learn_ornot
%          case 'YES'
%                DIC = DLFun0(init_dic',param);
%          case 'NO'
%                DIC = init_dic';
%     end

% %     final_dic = DLFun0(init_dic',param);% ���ǽ����ֵ�ѧϰ
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

    init_dic = eval(['init_dic' num2str(i)]);% �����ʼ���ֵ�
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