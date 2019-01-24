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
% function ORLabels = computeORLabel(objIndex, spinfor,param)
% ������GT����ȡ������������������ͼ��ȡ���� 2016.11.18 13:09PM
function ORLabels = computeORLabelNew(CURINFOR,param)
% lend = [x1,y1,x2,y2];
% objectIndex GT��ǩ
% 
ORTHS = param.ORTHS1;
OR_OB_th_L = ORTHS(1);
OR_OB_th_H = ORTHS(2);

spinfor   = CURINFOR.spinfor;
RegionSal = CURINFOR.spsal;

SPSCALENUM = length(spinfor);
ORLabels = cell(SPSCALENUM,1);

imsal_cur   = CURINFOR.imsal;
 
threshLocal = graythresh(imsal_cur);
meancut     = OR_OB_th_H*threshLocal;% 0.9 or 1  
thresh      = OR_OB_th_L*threshLocal;% 0.15 or 0.2

% meancut   = OR_OB_th_H*mean(imsal_cur(:));
% thresh    = OR_OB_th_L;% 0.05

% if meancut<=thresh % ��С�������� 2016.11.18, eg.birdfall2
% %     thresh      = meancut;clear meancut
%     threshLocal = graythresh(imsal_cur);
%     meancut     = 0.9*threshLocal;  
%     thresh      = 0.15*threshLocal;
%     if meancut<=thresh % ������ֵ����С�ڵ���ֵ
%         thresh = 0.5*thresh;
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ss=1:SPSCALENUM % ÿ���߶���
    tmpSP = spinfor{ss,1};
    tmp_regionSal = RegionSal{ss,1};
    ISOBJ=[];
    for i=1:tmpSP.spNum 
        meantemp=tmp_regionSal(i,1);
%         if meantemp>maxi
%             maxi=meantemp;
%             maxind=i;
%         end
        if meantemp>= meancut % object 
            ISOBJ=[ISOBJ;1];
        end
        if meantemp<= thresh  % background
            ISOBJ=[ISOBJ;0];
        end 
        if meantemp<meancut && meantemp>thresh % ģ������
            ISOBJ=[ISOBJ;50];
        end
        clear meantemp
    end
    ORLabels{ss,1} = ISOBJ;
    clear ISOBJ
end
       

clear CURINFOR param
end
%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
