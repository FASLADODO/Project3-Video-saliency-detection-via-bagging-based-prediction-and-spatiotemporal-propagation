function [TPSPSAL_Img,TPSPSAL_RegionSal] = SP18FUN(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param)
switch param.sp18ID
    case '0'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun18_0(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '1'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun18_1(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '2'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun18_2(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '3'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun18_3(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '4'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun18_4(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
end

clear TPSAL1 CURINFOR cur_image MVF_Foward_f_fp param
end