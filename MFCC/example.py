from python_speech_features import mfcc
import scipy.io.wavfile as wav
with open ('/Users/kakadiadhwani/Desktop/MINIPROJECT/Result/mfcc.txt','w') as f:
    print("Calculate MFCC values for each audio file in DataSet...Please Wait..!")
    for i in range(1,601):
        if(i<10):
            name="arctic_a000"+str(i)+".wav"
        elif(i<100):
            name="arctic_a00"+str(i)+".wav"
        elif(i<=300):
            name="arctic_a0"+str(i)+".wav"
        elif(i>300):
             if(i<310):
                 j=i-300
                 name="arctic_b000"+str(j)+".wav"
             elif(i<400):
                 j=i-300
                 name="arctic_b00"+str(j)+".wav"
             elif(i<=600):
                j=i-300
                name="arctic_b0"+str(j)+".wav"
        print("Calculating MFCC values for "+name+"Audio file...")
        (rate,sig) = wav.read('/Users/kakadiadhwani/Desktop/MINIPROJECT/DataSet/'+name)
        mfcc_feat = mfcc(sig,rate)
        mf= mfcc(sig,rate)
        for i in range(1,14):
            sum=0
            for j in range(1,100):
                sum=sum+mfcc_feat[j:j+1,:i]
            mf[1:2,:i]=sum/100
        f.write(str(mf[1:2,:])+'\n')
print("Finish")
