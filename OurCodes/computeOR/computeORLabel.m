% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% ����OR�����ǩ�� OR���⣬�߽磬ǰ���� 
% ���Լ��ٸ�����������������������ƽ�⣡����
%
% 2016.10.09 8:41AM
% copyright by xiaofei zhou,IVPLab, shanghai university, shanghai,china
% zxforchid@163.com
% www.ivp.shu.edu.cn
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% function ORLabels = computeORLabel(LCEND, objIndex, spinfor,param)
% ����ORlabel
function ORLabels = computeORLabel(objIndex, spinfor,param)
% lend = [x1,y1,x2,y2];
% objectIndex GT��ǩ
% 
ORTHS = param.ORTHS;
% OR_th = ORTHS(1);
% OR_BORDER_th = ORTHS(2);
% OR_OB_th = ORTHS(3);
OR_OB_th_L = ORTHS(1);
OR_OB_th_H = ORTHS(2);

SPSCALENUM = length(spinfor);
ORLabels = cell(SPSCALENUM,1);

[height,width,dims] = size(spinfor{1,1}.idxcurrImage);
% ORI = zeros(height,width);
% ORI(LCEND(2):LCEND(4),LCEND(1):LCEND(3))=1;
% ORIndex = find(ORI(:)==1);

for ss=1:SPSCALENUM % ÿ���߶���
    tmpSP = spinfor{ss,1};
    
    LABEL = [];ISOR=[];ISOBJ=[];ISBORDER = [];
    for sp=1:tmpSP.spNum % ������
        TMP = find(tmpSP.idxcurrImage==sp);
        
%         % 1. �����ж��Ƿ���OR������ 1/0
%         indSP_OR = ismember(TMP, ORIndex);
%         ratio_OR = sum(indSP_OR)/length(indSP_OR);

       
%         ISOR = [ISOR;1];% ȫ�ߴ�״̬����λ��OR�У�������Border�� 2016.11.07
%         ISBORDER = [ISBORDER;0];

%         if 0
%         % revised in 2016.08.29 8:07AM ------------------------------------
%         if ratio_OR==0
%             ISOR = [ISOR;0];
%             ISBORDER = [ISBORDER;0];
%         else
%             ISOR = [ISOR;1];% ����OR����
%             
%             if ratio_OR>0 && ratio_OR<1
%                 ISBORDER = [ISBORDER;1];% �߽糬���������п��ܱ߽����OR������object����
%             end
%             
%             if ratio_OR==1
%                 ISBORDER = [ISBORDER;0];
%             end
%         end
%         % -----------------------------------------------------------------
%         end
%              
        %3. ���ж��Ƿ���Object�� 1/0
        if isempty(objIndex)
        ISOBJ = [ISOBJ;100];  
        else
        indSP_GT = ismember(TMP,objIndex);
        ratio_GT = sum(indSP_GT)/length(indSP_GT);
        % revised in 2016.10.12 20:25PM
        if ratio_GT<=OR_OB_th_L  % �޽������߽���С�ڵ���0.2 or 0�������ڵ���0.8��������
            ISOBJ = [ISOBJ;0];
        end
        if ratio_GT>=OR_OB_th_H  % �н����Ҵ��ڵ���0.8��ǰ��
            ISOBJ = [ISOBJ;1];
        end
        if ratio_GT>OR_OB_th_L && ratio_GT<OR_OB_th_H %�н���������0.2~0.8֮��
            ISOBJ = [ISOBJ;50];
        end
        
%         if ratio_GT==0 % revised in 2016.10.12;�н�������object
%             ISOBJ = [ISOBJ;0];
%         else
%             ISOBJ = [ISOBJ;1];
%         end
        
        
%         if  ratio_GT < OR_OB_th 
%             ISOBJ = [ISOBJ;0];% NEGTIVE
%         else
%             ISOBJ = [ISOBJ;1]; % POSITIVE
%         end;            
        end

        
    end
%     LABEL = [ISOR,ISBORDER,ISOBJ];
    ORLabels{ss,1} = ISOBJ;
    clear tmpSP
end

clear LCEND objIndex spinfor param

end
%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% %1 ����OR�����ǩ�� OR���⣬�߽磬ǰ���� &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% function ORLabels = computeORLabel(LCEND, objIndex, spinfor,param)
% % lend = [x1,y1,x2,y2];
% % objectIndex GT��ǩ
% % 
% ORTHS = param.ORTHS;
% OR_th = ORTHS(1);
% OR_BORDER_th = ORTHS(2);
% OR_OB_th = ORTHS(3);% default 0.8
% 
% SPSCALENUM = length(spinfor);
% ORLabels = cell(SPSCALENUM,1);
% 
% [height,width,dims] = size(spinfor{1,1}.idxcurrImage);
% ORI = zeros(height,width);
% ORI(LCEND(2):LCEND(4),LCEND(1):LCEND(3))=1;
% ORIndex = find(ORI(:)==1);
% 
% for ss=1:SPSCALENUM % ÿ���߶���
%     tmpSP = spinfor{ss,1};
%     
%     LABEL = [];ISOR=[];ISOBJ=[];ISBORDER = [];
%     for sp=1:tmpSP.spNum % ������
%         TMP = find(tmpSP.idxcurrImage==sp);
%         
%         
%         % 1. �����ж��Ƿ���OR������ 1/0
%         indSP_OR = ismember(TMP, ORIndex);
%         ratio_OR = sum(indSP_OR)/length(indSP_OR);% accuracy score
%         
%         % revised in 2016.08.29 8:07AM ------------------------------------
%         if ratio_OR==0
%             ISOR = [ISOR;0];
%             ISBORDER = [ISBORDER;0];
%         else
%             ISOR = [ISOR;1];% ����OR����
%             
%             if ratio_OR>0 && ratio_OR<1
%                 ISBORDER = [ISBORDER;1];% �߽糬��������
%             end
%             
%             if ratio_OR==1
%                 ISBORDER = [ISBORDER;0];
%             end
%         end
%         % -----------------------------------------------------------------
% %         if  ratio_OR< OR_th % λ��OR�ⲿ
% %             ISOR = [ISOR;0];
% %             ISBORDER = [ISBORDER;0];
% %         else
% %             ISOR = [ISOR;1];
% %             % 2. �ж�OR����߽糬���� <1 OR�߽糬����
% %             if ratio_OR<OR_BORDER_th
% %                 ISBORDER = [ISBORDER;1];% �߽糬��������
% %             else
% %                 ISBORDER = [ISBORDER;0];
% %             end        
% %         end
%         
%         %3. ���ж��Ƿ���Object�� 1/0
%         if isempty(objIndex)
%         ISOBJ = [ISOBJ;100];  
%         else
%         indSP_GT = ismember(TMP,objIndex);
%         ratio_GT = sum(indSP_GT)/length(indSP_GT);% accuracy score
%         
%         if  ratio_GT < OR_OB_th
%             ISOBJ = [ISOBJ;0];% NEGTIVE
%         else
%             ISOBJ = [ISOBJ;1]; % POSITIVE
%         end;            
%         end
% 
%         
%     end
%     LABEL = [ISOR,ISBORDER,ISOBJ];
%     ORLabels{ss,1} = LABEL;
%     clear tmpSP
% end
% 
% clear LCEND objIndex spinfor param
% 
% end