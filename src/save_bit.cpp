/*
 * Copyright © DLVC 2017.
 *
 * Paper: Building Fast and Compact Convolutional Neural Networks for Offline Handwritten Chinese Character Recognition (arXiv:1702.07975 [cs.CV])
 * Authors: Xuefeng Xiao, Lianwen Jin, Yafeng Yang, Weixin Yang, Jun Sun, Tianhai Chang
 * Email: xiaoxuefengchina@gmail.com
 */
#include "save_bit.hpp"


void ReadOneBit( byte* pBuffer, int nStart, /* out */int& nEnd, /* out */ byte& retByte )  
{  
    byte btData = pBuffer[nStart/8];
    btData = btData << nStart%8;  
    retByte = btData >> 7;  
    nEnd = nStart+1;  
}  
 
void ReadStringFromBuffer( byte* pBuffer, int nStart, int nCount, /* out */int& nEnd, /* out */char* pRetData )  
{  
    for ( int nIndex=0; nIndex<nCount; nIndex++ )  
    {  
        ReadDataFromBuffer(pBuffer, nStart, 8, nStart, pRetData[nIndex]);  
    }  
    nEnd = nStart;  
}
 
 
void WriteOneBit( byte* pBuffer, byte btData, int nStart,  /* out */int& nEnd )  
{  
    int nSet = nStart / 8;  
    byte c = pBuffer[nSet];  
    switch ( btData )  
    {  
    case 1:  
        c |= ( 1 << (7- nStart % 8) );  
        break;  
    case 0:  
        c &= ( ~(1 << (7- nStart % 8) ) );  
        break;  
    default:  
        return;  
    }  
    pBuffer [nSet] = c;  
    nEnd = nStart +1;  
}  
 
void WtriteStringToBuffer( byte* pBuffer, char* pchar, int nStart,  int nCount, /* out */int& nEnd  )  
{  
    for ( int nIndex=0; nIndex<nCount; nIndex++ )  
    {  
        WriteDataToBuffer(pBuffer, pchar[nIndex], nStart, 8, nStart);  
    }  
    nEnd = nStart;  
} 