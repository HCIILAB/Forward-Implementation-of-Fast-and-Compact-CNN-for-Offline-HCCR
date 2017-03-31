/*
 * Copyright © DLVC 2017.
 *
 * Paper: Building Fast and Compact Convolutional Neural Networks for Offline Handwritten Chinese Character Recognition (arXiv:1702.07975 [cs.CV])
 * Authors: Xuefeng Xiao, Lianwen Jin, Yafeng Yang, Weixin Yang, Jun Sun, Tianhai Chang
 * Email: xiaoxuefengchina@gmail.com
 */ 
#ifndef SAVE_BIT_HPP
#define SAVE_BIT_HPP

#include <stdio.h>


typedef unsigned char byte;
/***********************************************************************/ 
/*   Function: read one bit from buffer                                */ 
/*   pBuffer[in]: the input buffer                                     */ 
/*   nStart[in]: the start place                                       */ 
/*   nEnd[out]: save the end place                                     */ 
/*   retByte[out]: save the read value                                 */ 
/*   return: void                                                      */ 
/***********************************************************************/ 
void ReadOneBit( byte* pBuffer, int nStart, /* out */int& nEnd, /* out */ byte& retByte );  
 
/***********************************************************************/ 
/*   Function: read a piece of data from input buffer                  */ 
/*   pBuffer[in]: the input buffer                                     */ 
/*   nStart[in]: the start place                                       */ 
/*   btLength[in]: the length of read data                             */ 
/*   nEnd[out]: save the end place                                     */ 
/*   retData[out]: save the read value                                 */ 
/*   return: void                                                      */ 
/***********************************************************************/ 
 
template<typename T>  
void  ReadDataFromBuffer( byte* pBuffer, int nStart, byte btLength, /* out */int& nEnd, /* out */ T& retData )  
{  
    retData = 0;  
    if ( btLength > sizeof(T)*8 )  {
        printf("data size is error \n");
        return ;  
    }
      
    byte btData;  
    T tData;  
    while ( btLength-- )  
    {  
        ReadOneBit(pBuffer, nStart, nStart, btData);  
        tData = btData << btLength;  
        retData |= tData;  
    }  
      
    nEnd = nStart;  
}  
 
/***********************************************************************/ 
/*   Function: read a string from input buffer                         */ 
/*   pBuffer[in]: the input buffer                                     */ 
/*   nStart[in]: the start place                                       */ 
/*   btLength[in]: the length of read string                           */ 
/*   nEnd[out]: save the end place                                     */ 
/*   retData[out]: save the read string                                */ 
/*   return: void                                                      */ 
/***********************************************************************/ 
void ReadStringFromBuffer( byte* pBuffer, int nStart, int nCount, /* out */int& nEnd, /* out */char* pRetData );  


/***********************************************************************/ 
/*   Function: write one bit to a buffer                               */ 
/*   pBuffer[in]: the stored buffer                                    */ 
/*   btData[in]: the value need to be written                          */ 
/*   nStart[in]: the start place                                       */
/*   nEnd[out]: save the end place                                     */ 
/*   return: void                                                      */ 
/***********************************************************************/  
void WriteOneBit( byte* pBuffer, byte btData, int nStart,  /* out */int& nEnd );  

/***********************************************************************/ 
/*   Function: write a piece of data to a buffer                       */ 
/*   pBuffer[in]: the stored buffer                                    */ 
/*   tData[in]: the data nee to be written                             */ 
/*   nStart[in]: the start place                                       */ 
/*   btLength[in]: the length of written data                          */ 
/*   nEnd[out]: save the end place                                     */ 
/*   return: void                                                      */ 
/***********************************************************************/ 
template<typename T>  
void  WriteDataToBuffer( byte* pBuffer, T tData, int nStart, byte btLength, /* out */int& nEnd )  
{  
/* //Big endian mode  
    byte btDataLength = sizeof(T);  
    if ( btLength > sizeof(T)*8 )  
        return;  
      
    int nDataStart = 0; 
    while ( btLength-- )  
    {  
        byte bitData;  
        ReadOneBit((byte*)&tData, nDataStart, nDataStart, bitData);  
        WriteOneBit(pBuffer, bitData, nStart, nStart);  
    }  
      
    nEnd = nStart;  
*/ 
 
    //Small endian mode       
    if ( btLength > sizeof(T)*8 )  
        return;   
  
    byte* ptData = (byte*)&tData;    
  
    int nSet = btLength / 8;  
    int nRin = btLength % 8;        
  
    byte bitData;  
    byte byteData;  
    int nTempEnd;   
 
    byteData = ptData[nSet];  
    while ( nRin-- )  
    {  
        ReadOneBit(&byteData, 7-nRin, nTempEnd, bitData);  
        WriteOneBit(pBuffer, bitData, nStart, nStart);  
    }  
 
  
    while ( nSet )  
    {  
        byteData = ptData[--nSet];  
 
        int i=0;  
        while ( i!=8 )  
        {  
            ReadOneBit(&byteData, i++, nTempEnd, bitData);  
            WriteOneBit(pBuffer, bitData, nStart, nStart);  
        }  
    }  
    nEnd = nStart;   
}   
 
/***********************************************************************/ 
/*   Function: write a string to a buffer                              */ 
/*   pBuffer[in]: the stored buffer                                    */ 
/*   pchar[in]: the string needed to be WtriteStringToBuffer           */
/*   nStart[in]: the start place                                       */ 
/*   nCount[in]: the length of written string                          */ 
/*   nEnd[out]: save the end place                                     */ 
/*   return: void                                                      */ 
/***********************************************************************/
void WtriteStringToBuffer( byte* pBuffer, char* pchar, int nStart,  int nCount, /* out */int& nEnd  );  


#endif

