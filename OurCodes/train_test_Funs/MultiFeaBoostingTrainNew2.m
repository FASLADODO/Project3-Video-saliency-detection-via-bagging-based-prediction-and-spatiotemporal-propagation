function [D0, beta, model, tmodel] = MultiFeaBoostingTrainNew2(DB,D0,ORLabels,spSal,param, spinfor)
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
%     D0.P.hogHist    
%     D0.P.regionCov   
%     D0.P.geoDist  
%     D0.P.flowHist   
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% features & labels
% revised in 2016.08.30 19:51PM
% revised again 2016.10.12 10:29AM
% �ۻ�OR�и�����������ֵ
SPSP = [];SPSN = [];
numSPP = [];numSPN = [];
spScaleLabelP = [];spScaleLabelN = [];
for ss=1:length(ORLabels)
    tmpORlabel = ORLabels{ss,1};
    tmpSPsal = spSal{ss,1};% ��������
    tmpSP = spinfor{ss,1};
    tmpPixellist = tmpSP.pixelList;
    
    ISORlabel = tmpORlabel(:,1);
    ISOBJlabel = tmpORlabel(:,3);
    % indexP indexN��������������ͼ���index���
    PNlabel = ISORlabel.*ISOBJlabel;
    index_P = find(PNlabel==1);% OR��ǰ�����: OR=1 OBJ=1
    index_N = [];% OR=1 OBJ=0 
    for ii=1:size(tmpORlabel,1)
       if ISORlabel(ii)==1 && ISOBJlabel(ii)==0 % OR�б������:OR=1 OBJ=0
           index_N = [index_N;ii];
       end
    end
    
    tmpPSAL = tmpSPsal(index_P,:);% ��ѡ����������Ӧ��������ֵ
    tmpNSAL = tmpSPsal(index_N,:);
    SPSP = [SPSP;tmpPSAL];
    SPSN = [SPSN;tmpNSAL];
    
    
    % ��ȡ�������ӦԪ����Ŀ
    for pp=1:length(index_P)
        tmpPP = tmpPixellist{index_P(pp)};
        numSPP = [numSPP;length(tmpPP)];
    end

    for nn=1:length(index_N)
        tmpNN = tmpPixellist{index_N(nn)};
        numSPN = [numSPN;length(tmpNN)];
    end
    
    % ��ȡ����������Ӧ�ĳ߶ȱ��
    spScaleLabelP = [spScaleLabelP;ss*ones(length(index_P),1)];
    spScaleLabelN = [spScaleLabelN;ss*ones(length(index_N),1)];
    
    
    clear index_P index_N tmpORlabel tmpSPsal ISORlabel 
    clear tmpPSAL tmpNSAL ISOBJlabel PNlabel
end
% LABELP = ones(size(SPSP,1),1);
% LABELN = zeros(size(SPSN,1),1);
% LABEL = [LABELP;LABELN];
SAL = [SPSP;SPSN];
numPixel = [numSPP;numSPN];
spScaleLabel = [spScaleLabelP;spScaleLabelN];

D0.SALP = SPSP;
D0.SALN = SPSN;
D0.numPixel = numPixel;
% D0.LABELP = LABELP;
% D0.LABELN = LABELN;

clear SPSP SPSN LABELN LABELP

% ѵ������Ϣ ---------------------------------------------------
nfeature = 8;
% revised in 2016.08.28 18:39PM ................................  
% ��������  sampleNum* Feadim (ԭʼ����)
d1=[D0.P.colorHist_rgb;D0.N.colorHist_rgb];
d2=[D0.P.colorHist_lab;D0.N.colorHist_lab];
d3=[D0.P.colorHist_hsv;D0.N.colorHist_hsv];
d4=[D0.P.lbpHist;      D0.N.lbpHist];
d5=[D0.P.hogHist;      D0.N.hogHist];
d6=[D0.P.regionCov;    D0.N.regionCov];
d7=[D0.P.geoDist;      D0.N.geoDist];
d8=[D0.P.flowHist;     D0.N.flowHist];

datanum = [size(d1,1),size(d2,1),size(d3,1),size(d4,1), ...
    size(d5,1),size(d6,1),size(d7,1),size(d8,1)];

% �����ǩ
LABEL = [ones(size(D0.P.colorHist_rgb,1),1);zeros(size(D0.N.colorHist_rgb,1),1)];
l1=LABEL;l2=l1;l3=l1;l4=l1;l5=l1;l6=l1;l7=l1;l8=l1;

% ����������ֵ
% SAL = LABEL;
s1=SAL;s2=s1;s3=s1;s4=s1;s5=s1;s6=s1;s7=s1;s8=s1;

% ��ʼ�ֵ�����ѵ�����������Ǳ���������
init_dic1 = D0.N.colorHist_rgb;
init_dic2 = D0.N.colorHist_lab;
init_dic3 = D0.N.colorHist_hsv;
init_dic4 = D0.N.lbpHist;
init_dic5 = D0.N.hogHist;
init_dic6 = D0.N.regionCov;
init_dic7 = D0.N.geoDist;
init_dic8 = D0.N.flowHist;

%% ѵ������������
tmodel = cell(nfeature,1);
tlabel = cell(nfeature,1);
tdec   = cell(nfeature,1);
for i = 1:nfeature
    disp(i)
    d = eval(['d' num2str(i)]);
    l = eval(['l' num2str(i)]);
    sal = eval(['s' num2str(i)]);
    
%     dic_p = eval(['dic' num2str(i) '_p_mappedA']);
%     dic_n = eval(['dic' num2str(i) '_n_mappedA']);
%     mapping_p = eval(['dic' num2str(i) '_p_mapping']);
%     mapping_n = eval(['dic' num2str(i) '_n_mapping']);
    
%     [dec_v,pred_l] = ...
%         weakClassifierNew0(dic_p',dic_n',mapping_p,mapping_n,d',param);
%     tmodel{i,1}.dic.p     = dic_p;
%     tmodel{i,1}.dic.n     = dic_n;
%     tmodel{i,1}.mapping.p = mapping_p;
%     tmodel{i,1}.mapping.n = mapping_n;
    
    init_dic = eval(['init_dic' num2str(i)]);% �����ʼ���ֵ�
    [~,dec_v,pred_l,final_dic] = ...
        weakClassifierNew1_1(init_dic',d',sal, param, numPixel,spScaleLabel);% 1_1 �µ��ֵ�ѧϰ��ʽ 2016.10.28 8:07AM
    tmodel{i,1}.dic       = final_dic;
    tlabel{i}             = pred_l;% 1*samplenum
    tdec{i}               = dec_v;% 1*samplenum

    clear d l final_dic dec_v pred_l sal
    
    eval(['clear',' ','d' num2str(i)])
    eval(['clear',' ','init_dic' num2str(i)])
%     eval(['clear',' ','dic' num2str(i) '_p_mappedA'])
%     eval(['clear',' ','dic' num2str(i) '_n_mappedA'])
%     eval(['clear',' ','dic' num2str(i) '_p_mapping'])
%     eval(['clear',' ','dic' num2str(i) '_n_mapping'])
end


%% ���������������ϵ��
%1. initial ---------------------------------
iter = nfeature;
n_weaker = nfeature;ntype =1;
D = cell(nfeature,1);

%2. sample weights --------------------------
for j = 1:nfeature
    D{j} = ones(datanum(j),1) / datanum(j);
end

%3. adaboost: iteration ---------------------
tbeta = []; % ��������Ȩ��ϵ��
beta = []; % ��ǿ�����ķ�����Ȩ��
tt = [];% ��ǿ�����ķ��������
for t = 1:iter
    
    % 3.1 ��������ʼ�������Ȩ�� -------------------
    for j = 1:n_weaker
        if sum(j==tt) ~= 0 
            tbeta = [tbeta; -inf];
            continue; 
        end
        fi = floor((j-1)/ntype)+1;
        l = eval(['l' num2str(fi)]);       
        sal = eval(['s' num2str(fi)]);

%         if ~isempty(tdec{j})
%             y_dec = D{fi} .* abs(tdec{j}');  
%         else
%             y_dec = D{fi};
%         end
        % �����ʼ�������Ȩ��ϵ��
        % optional 1 �ع�Ƕ�
%         b = 0.5 * log(sum(y_dec(tlabel{j}'==l))/(sum(y_dec(tlabel{j}'~=l))+eps));

        % optional 2���Է����ǩ�Ƕȼ��㣩
        error_weaker = double(tlabel{j}'~=l);% ����Ϊ1
%         error_weaker = sum(D{fi} .* error_weaker)/(length(l)+eps);% ��������Ȩ��D
        error_weaker = sum(D{fi} .* error_weaker);
        if error_weaker==0
            b = 0.5 * log((1-error_weaker)/(error_weaker+eps));   
        else
            b = 0.5 * log((1-error_weaker)/(error_weaker)); 
        end
        tbeta = [tbeta; b];
    end
    [var, idx] = max(tbeta);% ����������С
    idx1 = find(tbeta == var);
    idx = idx1(end);
    if var<0 break; end % ���ƴ�����С�� 0.5
    beta = [beta; var];
    model{t,1}.dic = tmodel{idx,1}.dic;
%     model{t,1}.mapping = tmodel{idx,1}.mapping; % revised in 2016.10.13 8:11AM
    tt = [tt; idx];
    
    % 3.2 ��С��ȷ����������Ȩ�� ------------------------
    fi = floor((idx-1)/ntype)+1;
    l = eval(['l' num2str(fi)]);
    l(l==0) = -1;
%     pred_l = tlabel{j}'; % �˴�����jʼ�յ���8
    pred_l = (tlabel{fi})';% revised in 2016.10.10 19:46PM
    pred_l(pred_l==0) = -1;
    D{fi} = D{fi}.*exp(- var * (pred_l .* l));% ���ּ�С��ͬ�ţ����������
    
    % old version, above is newing
%     if ~isempty(tdec{idx})
%         D{fi} = D{fi} .* exp(-var * tdec{idx}' .* l);
%     else
%         D{fi} = D{fi} .* exp(-var .* l);
%     end
    D{fi} = D{fi} / sum(D{fi});
    
    % 3.3 ��� & ���� ----------------------------------
    tbeta = [];
    t = t + 1;
    clear l pred_l var idx
end
beta = [beta tt];% ������Ȩ�� & ��Ӧ���


clear ORLabels spSal param

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     colorHist_rgb
%     colorHist_lab
%     colorHist_hsv 
%     lbpHist     
%     hogHist     
%     regionCov
%     geoDist   
%     flowHist  



% if 0
% % �ֵ���Ϣ ----------------------------------------------------
% dic1_p_mappedA = DB.P.colorHist_rgb_mappedA;dic1_n_mappedA = DB.N.colorHist_rgb_mappedA;
% dic1_p_mapping = DB.P.colorHist_rgb_mapping;dic1_n_mapping = DB.N.colorHist_rgb_mapping;
% 
% dic2_p_mappedA = DB.P.colorHist_lab_mappedA;dic2_n_mappedA = DB.N.colorHist_lab_mappedA;
% dic2_p_mapping = DB.P.colorHist_lab_mapping;dic2_n_mapping = DB.N.colorHist_lab_mapping;
% 
% dic3_p_mappedA = DB.P.colorHist_hsv_mappedA;dic3_n_mappedA = DB.N.colorHist_hsv_mappedA;
% dic3_p_mapping = DB.P.colorHist_hsv_mapping;dic3_n_mapping = DB.N.colorHist_hsv_mapping;
% 
% dic4_p_mappedA = DB.P.lbpHist_mappedA;      dic4_n_mappedA = DB.N.lbpHist_mappedA;
% dic4_p_mapping = DB.P.lbpHist_mapping;      dic4_n_mapping = DB.N.lbpHist_mapping;
% 
% dic5_p_mappedA = DB.P.hogHist_mappedA;      dic5_n_mappedA = DB.N.hogHist_mappedA;
% dic5_p_mapping = DB.P.hogHist_mapping;      dic5_n_mapping = DB.N.hogHist_mapping;
% 
% dic6_p_mappedA = DB.P.regionCov_mappedA;    dic6_n_mappedA = DB.N.regionCov_mappedA;
% dic6_p_mapping = DB.P.regionCov_mapping;    dic6_n_mapping = DB.N.regionCov_mapping;
% 
% dic7_p_mappedA = DB.P.geoDist_mappedA;      dic7_n_mappedA = DB.N.geoDist_mappedA;
% dic7_p_mapping = DB.P.geoDist_mapping;      dic7_n_mapping = DB.N.geoDist_mapping;
% 
% dic8_p_mappedA = DB.P.flowHist_mappedA;     dic8_n_mappedA = DB.N.flowHist_mappedA;
% dic8_p_mapping = DB.P.flowHist_mapping;     dic8_n_mapping = DB.N.flowHist_mapping;
% end
