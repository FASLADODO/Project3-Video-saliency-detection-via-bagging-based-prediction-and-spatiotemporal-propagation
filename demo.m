%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright by xiaofei zhou, IVPLab, shanghai univeristy,shanghai, china
% http://www.ivp.shu.edu.cn/Default.aspx
% email: zxforchid@163.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;close all;clc

allRootPath = ['.\datas\'];
DatasetNames = {'DAVIS'};

%% parameterize %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.spnumbers    = [400];% 100:50:250;%
param.ORTHS        = [0,0.8];% OR�����ǩ��ֵ�� OR BORDER OBJ
param.beta         = [0.5,0.5];
param.no_dims      = 0.99;% PCA��������ϵ��
param.bgRatio      = 0.3;% ����������ռ����
param.fgRatio      = 0.1;% ����������ռ����
param.sp_iternum   = 1;% ���򴫲���������
param.enhhanceRatio= 1.25;
param.num_tree = 200;
param.knnNums  = 15;% spNum/20
param.removeLowVals = false;
param.numMultiContext = false;
param.bdIds  = 0.75;
param.predictN = 15;

% param.flow_ratio = [0.7,0.8];



for number=[1]%
    param.sp19ID  = num2str(number);% 0:3
%% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% ��ƵID  -----------------------------------------------------------------
for ds=1:length(DatasetNames)
fprintf( ['DatasetName: ',DatasetNames{1,ds},'\n']);
root_videoDataSet=[allRootPath,'IMG\',DatasetNames{1,ds},'\'];
videoNames = dir(root_videoDataSet);
videoNames = videoNames(3:end);
DatasetName = DatasetNames{1,ds};

% ��ƵID   ------------------------------------------------------------------
for vn=1:length(videoNames)
fprintf( ['VideoName: ',videoNames(vn).name,'\n']);

% ��������Ƶ·��
video_data=[root_videoDataSet,videoNames(vn).name,'\'];
Groundtruth_path=[video_data,'ground-truth'];

% �������·��
saliencyMap=['.\Results20180409\',DatasetName,'\',videoNames(vn).name,'\'];
if( ~exist( saliencyMap, 'dir' ) )
    mkdir( saliencyMap );
end
OPTICALFLOW=['.\data\opticalFlow\',DatasetName,'\',videoNames(vn).name,'\'];% ̨ʽ��

%% ֡LOOP *&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% �ӵ���֡��ʼԤ�⣬�ʼ�ɵڶ�֡��ʼѵ��������
computePerFrame(video_data,Groundtruth_path,saliencyMap,OPTICALFLOW,param)
% ֡LOOP ------------------------------------------------------------------


end

end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end




