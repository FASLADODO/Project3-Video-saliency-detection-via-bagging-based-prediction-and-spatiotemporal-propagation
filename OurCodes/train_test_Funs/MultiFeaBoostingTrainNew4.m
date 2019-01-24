function [tmodel] = MultiFeaBoostingTrainNew4(D0,param)
% 引入随机抽样（于较多样本上）
% 2016.12.13 
% 
%% initial &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
trainDataP = [D0.P.regionFea];
trainDataN = [D0.N.regionFea];

trn_sal_data =[trainDataP;trainDataN];
% % trn_sal_data = [trn_sal_data0(:,4:6),trn_sal_data0(:,8:9)];% Lab+Flow,2016.12.13
% % clear trn_sal_data0
trn_sal_lab = [ones(size(trainDataP,1),1);zeros(size(trainDataN,1),1)];

opt.importance = 0;
opt.do_trace = 1;
num_tree = param.num_tree;% 200;
mtry = floor(sqrt(size(trn_sal_data,2)));% 特征维数

%% 多次随机抽样，获取平衡样本  &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% if size(trainDataP,1) > size(trainDataN,1)
%     param.predictN = round(size(trainDataP,1)/size(trainDataN,1));
% else
%     param.predictN = round(size(trainDataN,1)/size(trainDataP,1));
% end
clear trainDataN trainDataP
% iterNum = 2*(param.predictN);
% 随机抽样iterm次（人为设定），每次抽取2/3
iterNum = param.predictN;
tmodel = cell(iterNum,1);
for kk=1:iterNum
% 平衡数据，并于此种条件下归一化！！！
[trn_sal_data_mappedA0, trn_sal_lab0] = balanceDataNew(trn_sal_data, trn_sal_lab);
[trn_sal_data_mappedA0,scalemap] = scaleForSVM_corrected1(trn_sal_data_mappedA0,0,1);

% 训练
model = regRF_train( trn_sal_data_mappedA0, trn_sal_lab0, num_tree, mtry, opt );
segment_saliency_regressor = compressRegModel(model);
tmodel{kk,1}.dic           = segment_saliency_regressor;
tmodel{kk,1}.scalemap      = scalemap;

clear trn_sal_data_mappedA0 trn_sal_lab0 segment_saliency_regressor scalemap
end 
clear trn_sal_data_mappedA trn_sal_lab
% clear trn_sal_data_mappedA0 trn_sal_lab0 segment_saliency_regressor scalemap
% % 获取重要的特征
% feaMse = model.importance(:,2);
% [values,indexs] = sort(feaMse);
% feaDim = size(trn_sal_data_mappedA,2);
% lengths = round(0.8*feaDim);
% indexs1 = index((lengths+1):end,:);
% tmodel{1,1}.feaIndex = indexs1;%要舍弃的编号


clear trn_sal_data trn_sal_lab 

clear D0 param

end
