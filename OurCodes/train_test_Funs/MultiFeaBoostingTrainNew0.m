function [D0, beta, model, tmodel] = MultiFeaBoostingTrainNew0(DB,D0,ORLabels,spSal,param)
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
%     D0.P.hogHist    
%     D0.P.regionCov   
%     D0.P.geoDist  
%     D0.P.flowHist   
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% features & labels
% revised in 2016.08.30 19:51PM
% % 累积OR中各区域显著性值
% SPSP = [];SPSN = [];
% for ss=1:length(ORLabels)
%     tmpORlabel = ORLabels{ss,1};
%     tmpSPsal = spSal{ss,1};
%     
%     ISORlabel = tmpORlabel(:,1);
%     ISOBJlabel = tmpORlabel(:,3);
%     
%     PNlabel = ISORlabel.*ISOBJlabel;
%     index_P = find(PNlabel==1);%(1,1)， OR中前景标号
%     index_N = [];
%     for ii=1:size(tmpORlabel,1)
%        if ISORlabel(ii)==1 && ISOBJlabel(ii)==0 % OR中背景标号 (1,0)
%            index_N = [index_N;ii];
%        end
%     end
%     
%     tmpPSAL = tmpSPsal(index_P,:);
%     tmpNSAL = tmpSPsal(index_N,:);
%     
%     SPSP = [SPSP;tmpPSAL];
%     SPSN = [SPSN;tmpNSAL];
%     
%     clear index_P index_N tmpORlabel tmpSPsal ISORlabel 
%     clear tmpPSAL tmpNSAL ISOBJlabel PNlabel
% end
% LABELP = ones(size(SPSP,1),1);
% LABELN = zeros(size(SPSN,1),1);
% 
% LABEL = [LABELP;LABELN];
% SAL = [SPSP;SPSN];

% D0.SALP = SPSP;
% D0.SALN = SPSN;
% D0.LABELP = LABELP;
% D0.LABELN = LABELN;

% clear SPSP SPSN LABELN LABELP

% 训练集信息 ---------------------------------------------------
nfeature = 8;
% revised in 2016.08.28 18:39PM ................................  
% 区域特征  sampleNum* Feadim (原始特征)
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

% 区域标签
LABEL = [ones(size(D0.P.colorHist_rgb,1),1);zeros(size(D0.N.colorHist_rgb,1),1)];
l1=LABEL;l2=l1;l3=l1;l4=l1;l5=l1;l6=l1;l7=l1;l8=l1;

% 区域显著性值
SAL = LABEL;
s1=SAL;s2=s1;s3=s1;s4=s1;s5=s1;s6=s1;s7=s1;s8=s1;

% 字典信息 ----------------------------------------------------
dic1_p_mappedA = DB.P.colorHist_rgb_mappedA;dic1_n_mappedA = DB.N.colorHist_rgb_mappedA;
dic1_p_mapping = DB.P.colorHist_rgb_mapping;dic1_n_mapping = DB.N.colorHist_rgb_mapping;

dic2_p_mappedA = DB.P.colorHist_lab_mappedA;dic2_n_mappedA = DB.N.colorHist_lab_mappedA;
dic2_p_mapping = DB.P.colorHist_lab_mapping;dic2_n_mapping = DB.N.colorHist_lab_mapping;

dic3_p_mappedA = DB.P.colorHist_hsv_mappedA;dic3_n_mappedA = DB.N.colorHist_hsv_mappedA;
dic3_p_mapping = DB.P.colorHist_hsv_mapping;dic3_n_mapping = DB.N.colorHist_hsv_mapping;

dic4_p_mappedA = DB.P.lbpHist_mappedA;      dic4_n_mappedA = DB.N.lbpHist_mappedA;
dic4_p_mapping = DB.P.lbpHist_mapping;      dic4_n_mapping = DB.N.lbpHist_mapping;

dic5_p_mappedA = DB.P.hogHist_mappedA;      dic5_n_mappedA = DB.N.hogHist_mappedA;
dic5_p_mapping = DB.P.hogHist_mapping;      dic5_n_mapping = DB.N.hogHist_mapping;

dic6_p_mappedA = DB.P.regionCov_mappedA;    dic6_n_mappedA = DB.N.regionCov_mappedA;
dic6_p_mapping = DB.P.regionCov_mapping;    dic6_n_mapping = DB.N.regionCov_mapping;

dic7_p_mappedA = DB.P.geoDist_mappedA;      dic7_n_mappedA = DB.N.geoDist_mappedA;
dic7_p_mapping = DB.P.geoDist_mapping;      dic7_n_mapping = DB.N.geoDist_mapping;

dic8_p_mappedA = DB.P.flowHist_mappedA;     dic8_n_mappedA = DB.N.flowHist_mappedA;
dic8_p_mapping = DB.P.flowHist_mapping;     dic8_n_mapping = DB.N.flowHist_mapping;

%% 训练各弱分类器
tmodel = cell(nfeature,1);
tlabel = cell(nfeature,1);
tdec   = cell(nfeature,1);
for i = 1:nfeature
    d = eval(['d' num2str(i)]);
    l = eval(['l' num2str(i)]);
    
    dic_p = eval(['dic' num2str(i) '_p_mappedA']);
    dic_n = eval(['dic' num2str(i) '_n_mappedA']);
    mapping_p = eval(['dic' num2str(i) '_p_mapping']);
    mapping_n = eval(['dic' num2str(i) '_n_mapping']);
    
%     s = ['-t 0'];
%    m = svmtrain(l, d, s);
%    [pred_l, acc, dec_v] = svmpredict(l, d, m);
    [dec_v,pred_l] = ...
        weakClassifierNew0(dic_p',dic_n',mapping_p,mapping_n,d',param);
    tmodel{i,1}.dic.p     = dic_p;
    tmodel{i,1}.dic.n     = dic_n;
    tmodel{i,1}.mapping.p = mapping_p;
    tmodel{i,1}.mapping.n = mapping_n;
    tlabel{i}             = pred_l;% 1*samplenum
    tdec{i}               = dec_v;% 1*samplenum

    clear d l dic_p dic_n mapping_p mapping_n dec_v pred_l
    
    eval(['clear',' ','d' num2str(i)])
    eval(['clear',' ','dic' num2str(i) '_p_mappedA'])
    eval(['clear',' ','dic' num2str(i) '_n_mappedA'])
    eval(['clear',' ','dic' num2str(i) '_p_mapping'])
    eval(['clear',' ','dic' num2str(i) '_n_mapping'])
end


%% 求解弱分类器连接系数
%1. initial ---------------------------------
iter = nfeature;
n_weaker = nfeature;ntype =1;
D = cell(nfeature,1);

%2. sample weights --------------------------
for j = 1:nfeature
    D{j} = ones(datanum(j),1) / datanum(j);
end

%3. adaboost: iteration ---------------------
tbeta = []; % 弱分类器权重系数
beta = []; % 由强至弱的分类器权重
tt = [];% 由强至弱的分类器标号
for t = 1:iter
    
    % 3.1 计算错误率及分类器权重 -------------------
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
        % 错误率及分类器权重系数
        % optional 1 回归角度
%         b = 0.5 * log(sum(y_dec(tlabel{j}'==l))/(sum(y_dec(tlabel{j}'~=l))+eps));

        % optional 2（以分类标签角度计算）
        error_weaker = double(tlabel{j}'~=l);% 不等为1
%         error_weaker = sum(D{fi} .* error_weaker)/(length(l)+eps);% 引入样本权重D
        error_weaker = sum(D{fi} .* error_weaker);
        if error_weaker==0
            b = 0.5 * log((1-error_weaker)/(error_weaker+eps));   
        else
            b = 0.5 * log((1-error_weaker)/(error_weaker)); 
        end
        tbeta = [tbeta; b];
    end
    [var, idx] = max(tbeta);% 即错误率最小
    idx1 = find(tbeta == var);
    idx = idx1(end);
    if var<0 break; end % 限制错误率小于 0.5
    beta = [beta; var];
    model{t,1}.dic = tmodel{idx,1}.dic;
    model{t,1}.mapping = tmodel{idx,1}.mapping;
    tt = [tt; idx];
    
    % 3.2 缩小正确分类样本的权重 ------------------------
    fi = floor((idx-1)/ntype)+1;
    l = eval(['l' num2str(fi)]);
    l(l==0) = -1;
%     pred_l = tlabel{j}'; % 此处错误，j始终等于8
    pred_l = (tlabel{fi})';% revised in 2016.10.10 19:46PM
    pred_l(pred_l==0) = -1;
    D{fi} = D{fi}.*exp(- var * (pred_l .* l));% 正分减小，错分增大
    
    % old version, above is newing
%     if ~isempty(tdec{idx})
%         D{fi} = D{fi} .* exp(-var * tdec{idx}' .* l);
%     else
%         D{fi} = D{fi} .* exp(-var .* l);
%     end
    D{fi} = D{fi} / sum(D{fi});
    
    % 3.3 清空 & 计数 ----------------------------------
    tbeta = [];
    t = t + 1;
    clear l pred_l var idx
end
beta = [beta tt];% 分类器权重 & 对应标号


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