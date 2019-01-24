function [TPSPSAL_Img,TPSPSAL_RegionSal] = SP14FUN(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param)
switch param.sp14ID
    case '0'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun14_0(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '1'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun14_1(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '2'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun14_2(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '3'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun14_3(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
end

clear TPSAL1 CURINFOR cur_image MVF_Foward_f_fp param
end