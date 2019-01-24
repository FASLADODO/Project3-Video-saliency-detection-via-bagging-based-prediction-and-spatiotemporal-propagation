function computePerFrame(video_data,Groundtruth_path,saliencyMap,OPTICALFLOW,param)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��ȡ��Ƶ֡��Ϣ
[frames,GT,frame_names] = readAllFrames_Li(video_data,Groundtruth_path);

% �ӵ� 2 ֡��ʼԤ��
for f = 2:length(frames)-2
    fprintf('\n processing frame %d &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&',f)
    %% initialize &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    fprintf('\n initialize Dic using the second frame ##############################################################################################\n')
    if f==2 %  �ӵڶ�֡��ʼ������
        % NOTE: ��ȡflow_1_2��������ȡ��1֡�Ĺ���������ѵ����ʼ�ֵ�
        load ([OPTICALFLOW,'opf_',num2str(f-1),'.mat'])
        clear MVF_Foward_fn_f MVF_Foward_f_fn
        fpre_GT               = GT{1,f-1};
        fpre_GT               = double(fpre_GT>=0.5);
        %                                  pre_image    pre_gt
       [PRE_INFOR, PRE_DIC]  = initDic2_2(frames{1,f-1},fpre_GT,param,MVF_Foward_f_fp);
        PRE_PRE_DIC   = [];
        PRE_PRE_INFOR = [];
        clear MVF_Foward_f_fp
        
    else % ���� double-frameģ�ͣ�����Ԥ���봫����pre & pre_pre��
        PRE_PRE_DIC   = PRE_DIC;          clear PRE_DIC
        PRE_DIC       = UPDATA_DIC;       clear UPDATA_DIC
        PRE_PRE_INFOR = PRE_INFOR;        clear PRE_INFOR
        PRE_INFOR     = CURRENTINFOR;
%         fpre_GT       = CURRENTINFOR.imgt;clear CURRENTINFOR
    end
 
    %% ��OBJ_LC��Ԥ������������ֵ ��apperanceModel�� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&   
    saveInfor.saliencyMap = saliencyMap;
    saveInfor.frame_name  = frame_names{1,f}(1:end-4);

    fprintf('\n compute the saliency value  #########################################################################################################\n')
    % t-1, t, t+1 3֡����
    if  f==2 || f==(length(frames)-2)
     [CURRENTINFOR,imwriteInfor] = ...
          apperanceModel3( frames{1,f}  , frames{1,f+1}, ...
                           PRE_INFOR    , PRE_DIC    , ...
                           OPTICALFLOW  , f          , param         ,saveInfor);
    end
    % t-2, t-1, t, t+1, t+2  5֡����
    if f>=3 && f<=(length(frames)-3)
    [CURRENTINFOR,imwriteInfor] = ...
          apperanceModel3_0(frames{1,f}  , frames{1,f+1},frames{1,f+2} ,...
                            PRE_INFOR    , PRE_PRE_INFOR, PRE_DIC      ,PRE_PRE_DIC  ,...
                            OPTICALFLOW  , f            , param        ,saveInfor);
    end

    % ������׶ε�������ͼ������Ա�۲�Ƚ�
    saveSALSNew(imwriteInfor, saliencyMap,frame_names{1,f}(1:end-4));
    
    %% updata �����ֵ� ��updataModel��&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    fprintf('\n update information #####################################################################################################################\n')
    UPDATA_DIC = updateDIC7(CURRENTINFOR,param);
%     UPDATA_DIC = updateDIC7(CURRENTINFOR,param,fpre_GT,MVF_Foward_fn_f);
 end

end