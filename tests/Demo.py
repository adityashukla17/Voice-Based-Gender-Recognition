import copy
from scipy.io import wavfile
def demo(name):
    fs, samp= wavfile.read('/Users/kakadiadhwani/Desktop/MINIPROJECT/DataSet/'+name)
    lent=len(samp)
    sample_3=copy.copy(samp)
    c=0
    zeropos=[];
    for i in range(1,lent-1) :
        if samp[i]<0 and samp[i+1]>0 :
            c=c+1
        
    c1=c
    x1=2*lent/c1
    y=fs/x1
    for i in range(1,(lent-6)):
        sample_3[i+3]=(samp[i]+samp[i+1]+samp[i+2]+samp[i+3]+samp[i+4]+samp[i+5]+samp[i+6])/7
    c=0;
    for j in range(1,(lent-4)):
        if sample_3[j]<0 and sample_3[j+1]>0:
            c=c+1
            zeropos.append(j)
    x1=2*lent/c
    y=fs/x1
    with open ('/Users/kakadiadhwani/Desktop/MINIPROJECT/Result/output.txt','w') as f:
        if y<200:
            f.write('FILE NAME :'+name+'\n')
            f.write("Result:MALE"+'\n')
        else:
            f.write('FILE NAME : '+name+'\n')
            f.write("Result:FEMALE"+'\n')
    print("Output is in Result folder")

