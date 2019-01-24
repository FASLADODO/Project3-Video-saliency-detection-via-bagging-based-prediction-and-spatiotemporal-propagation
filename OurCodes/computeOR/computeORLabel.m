% &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% 计算OR区域标签： OR内外，边界，前背景 
% 尝试减少负样本个数，以求正负样本平衡！！！
%
% 2016.10.09 8:41AM
% copyright by xiaofei zhou,IVPLab, shanghai university, shanghai,china
% zxforchid@163.com
% www.ivp.shu.edu.cn
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% function ORLabels = computeORLabel(LCEND, objIndex, spinfor,param)
% 舍弃ORlabel
function ORLabels = computeORLabel(objIndex, spinfor,param)
% lend = [x1,y1,x2,y2];
% objectIndex GT标签
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

for ss=1:SPSCALENUM % 每个尺度下
    tmpSP = spinfor{ss,1};
    
    LABEL = [];ISOR=[];ISOBJ=[];ISBORDER = [];
    for sp=1:tmpSP.spNum % 各区域
        TMP = find(tmpSP.idxcurrImage==sp);
        
%         % 1. 首先判断是否在OR区域内 1/0
%         indSP_OR = ismember(TMP, ORIndex);
%         ratio_OR = sum(indSP_OR)/length(indSP_OR);

       
%         ISOR = [ISOR;1];% 全尺寸状态，均位于OR中，均不是Border， 2016.11.07
%         ISBORDER = [ISBORDER;0];

%         if 0
%         % revised in 2016.08.29 8:07AM ------------------------------------
%         if ratio_OR==0
%             ISOR = [ISOR;0];
%             ISBORDER = [ISBORDER;0];
%         else
%             ISOR = [ISOR;1];% 属于OR区域
%             
%             if ratio_OR>0 && ratio_OR<1
%                 ISBORDER = [ISBORDER;1];% 边界超像素区域（有可能边界既与OR交又与object交）
%             end
%             
%             if ratio_OR==1
%                 ISBORDER = [ISBORDER;0];
%             end
%         end
%         % -----------------------------------------------------------------
%         end
%              
        %3. 再判断是否在Object中 1/0
        if isempty(objIndex)
        ISOBJ = [ISOBJ;100];  
        else
        indSP_GT = ismember(TMP,objIndex);
        ratio_GT = sum(indSP_GT)/length(indSP_GT);
        % revised in 2016.10.12 20:25PM
        if ratio_GT<=OR_OB_th_L  % 无交集或者交集小于等于0.2 or 0（即大于等于0.8），背景
            ISOBJ = [ISOBJ;0];
        end
        if ratio_GT>=OR_OB_th_H  % 有交集且大于等于0.8，前景
            ISOBJ = [ISOBJ;1];
        end
        if ratio_GT>OR_OB_th_L && ratio_GT<OR_OB_th_H %有交集，介于0.2~0.8之间
            ISOBJ = [ISOBJ;50];
        end
        
%         if ratio_GT==0 % revised in 2016.10.12;有交叠即是object
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
% %1 计算OR区域标签： OR内外，边界，前背景 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
% function ORLabels = computeORLabel(LCEND, objIndex, spinfor,param)
% % lend = [x1,y1,x2,y2];
% % objectIndex GT标签
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
% for ss=1:SPSCALENUM % 每个尺度下
%     tmpSP = spinfor{ss,1};
%     
%     LABEL = [];ISOR=[];ISOBJ=[];ISBORDER = [];
%     for sp=1:tmpSP.spNum % 各区域
%         TMP = find(tmpSP.idxcurrImage==sp);
%         
%         
%         % 1. 首先判断是否在OR区域内 1/0
%         indSP_OR = ismember(TMP, ORIndex);
%         ratio_OR = sum(indSP_OR)/length(indSP_OR);% accuracy score
%         
%         % revised in 2016.08.29 8:07AM ------------------------------------
%         if ratio_OR==0
%             ISOR = [ISOR;0];
%             ISBORDER = [ISBORDER;0];
%         else
%             ISOR = [ISOR;1];% 属于OR区域
%             
%             if ratio_OR>0 && ratio_OR<1
%                 ISBORDER = [ISBORDER;1];% 边界超像素区域
%             end
%             
%             if ratio_OR==1
%                 ISBORDER = [ISBORDER;0];
%             end
%         end
%         % -----------------------------------------------------------------
% %         if  ratio_OR< OR_th % 位于OR外部
% %             ISOR = [ISOR;0];
% %             ISBORDER = [ISBORDER;0];
% %         else
% %             ISOR = [ISOR;1];
% %             % 2. 判定OR区域边界超像素 <1 OR边界超像素
% %             if ratio_OR<OR_BORDER_th
% %                 ISBORDER = [ISBORDER;1];% 边界超像素区域
% %             else
% %                 ISBORDER = [ISBORDER;0];
% %             end        
% %         end
%         
%         %3. 再判断是否在Object中 1/0
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