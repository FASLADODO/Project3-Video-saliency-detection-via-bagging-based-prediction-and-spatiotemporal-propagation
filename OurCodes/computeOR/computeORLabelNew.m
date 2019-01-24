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
% function ORLabels = computeORLabel(objIndex, spinfor,param)
% 舍弃以GT来获取样本，这里以显著性图获取样本 2016.11.18 13:09PM
function ORLabels = computeORLabelNew(CURINFOR,param)
% lend = [x1,y1,x2,y2];
% objectIndex GT标签
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

% if meancut<=thresh % 极小物体情形 2016.11.18, eg.birdfall2
% %     thresh      = meancut;clear meancut
%     threshLocal = graythresh(imsal_cur);
%     meancut     = 0.9*threshLocal;  
%     thresh      = 0.15*threshLocal;
%     if meancut<=thresh % 若高阈值还是小于低阈值
%         thresh = 0.5*thresh;
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ss=1:SPSCALENUM % 每个尺度下
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
        if meantemp<meancut && meantemp>thresh % 模糊区域
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
