#!/usr/bin/env python
# coding: utf-8

# In[52]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import sympy as sym
import sys
ROUND_CONST = 100000

#국내모델/해외모델, 제조사, 유종, 차종 선제적 선택

#국내모델, 해외모델

#제조사(국내모델) : 현대, 기아, 쌍용, 르노삼성, 한국지엠
#차종(국내모델) : 경차, 소형차, 중형차, 대형차, SUV
#유종(국내모델) : 휘발유, 경유, LPG

#제조사(해외모델) : BMW, 포르쉐, 벤츠, 아우디, 크라이슬러
#차종(해외모델) : 경차, 소형차, 중형차, 대형차, SUV
#유종(해외모델) : 휘발유, 경유

#인공지능 모델 특징량으로 최고 출력, 최고 토크 제외
#인공지능 모델 특징량으로 배기량, 공차 중량, CO2 배출량, 복합연비 4가지 요소 선정


print("[자동차 연비 예측 프로그램을 시작합니다.]")
s = input("모델의 종류(국내모델, 해외모델)를 입력해주세요. :")

if s == "국내모델":
    s = input("제조사(현대, 기아, 쌍용, 르노삼성, 한국지엠)를 입력해주세요. :")
    if s == "현대":
        s = input("차종(중형차, 대형차, SUV)을 입력해주세요. :")
        if s == "중형차":
            s = input("유종(휘발유, 경유, LPG)을 입력해주세요. :")
            if s == "휘발유":
                filename = "현대_중형_휘발유.csv"
                filesize = 125
            elif s == "경유":
                filename = "현대_중형_경유.csv"
                filesize = 56
            elif s == "LPG":
                filename = "현대_중형_LPG.csv"
                filesize = 13
            else:
                print("잘못된 값을 입력하셨습니다.")
                sys.exit()
        elif s == "대형차": #최적화 실패
            s = input("유종(휘발유, 경유)을 입력해주세요. :")
            if s == "휘발유":
                filename = "현대_대형_휘발유.csv"
                filesize = 114
            elif s == "경유":
                filename = "현대_대형_경유.csv"
                filesize = 67
            else:
                print("잘못된 값을 입력하셨습니다.")
                sys.exit()
        elif s == "SUV":
            s = input("유종(휘발유, 경유)을 입력해주세요. :")
            if s == "휘발유":
                filename = "현대_SUV_휘발유.csv"
                filesize = 80
            elif s == "경유":
                filename = "현대_SUV_경유.csv"
                filesize = 95
            else:
                print("잘못된 값을 입력하셨습니다.")
                sys.exit()
        
    elif s == "기아":
        s = input("차종(경차, 중형차, 대형차, SUV)을 입력해주세요. :")
        if s == "경차":
            s = input("유종(휘발유)을 입력해주세요. :")
            if s == "휘발유":
                filename = "기아_경차_휘발유.csv"
                filesize = 17
            else:
                print("잘못된 값을 입력하셨습니다.")
                sys.exit()
        elif s == "중형차":
            s = input("유종(휘발유, 경유)을 입력해주세요. :")
            if s == "휘발유":
                filename = "기아_중형_휘발유.csv"
                filesize = 58
            elif s == "경유":
                filename = "기아_중형_경유.csv"
                filesize = 31
            else:
                print("잘못된 값을 입력하셨습니다.")
                sys.exit()
        elif s == "대형차":
            s = input("유종(휘발유, 경유)을 입력해주세요. :")
            if s == "휘발유":
                filename = "기아_대형_휘발유.csv"
                filesize = 65
            elif s == "경유":
                filename = "기아_대형_경유.csv"
                filesize = 60
            else:
                print("잘못된 값을 입력하셨습니다.")
                sys.exit()
        elif s == "SUV":
            s = input("유종(휘발유, 경유)을 입력해주세요. :")
            if s == "휘발유":
                filename = "기아_SUV_휘발유.csv"
                filesize = 43
            elif s == "경유":
                filename = "기아_SUV_경유.csv"
                filesize = 62
            else:
                print("잘못된 값을 입력하셨습니다.")
                sys.exit()
                
    elif s == "쌍용":
        s = input("차종(중형차)을 입력해주세요. :")
        if s == "중형차":
            s = input("유종(휘발유, 경유)을 입력해주세요. :")
            if s == "휘발유":
                filename = "쌍용_중형_휘발유.csv"
                filesize = 14
            elif s == "경유":
                filename = "쌍용_중형_경유.csv"
                filesize = 14
            else:
                print("잘못된 값을 입력하셨습니다.")
                sys.exit()
        
    elif s == "르노삼성":
        s = input("차종(중형차)을 입력해주세요. :")
        if s == "중형차":
            s = input("유종(휘발유)을 입력해주세요. :")
            if s == "휘발유":
                filename = "르노삼성_중형_휘발유.csv"
                filesize = 12
            else:
                print("잘못된 값을 입력하셨습니다.")
                sys.exit()
                
    elif s == "한국지엠":
        s = input("차종(중형차)을 입력해주세요. :")
        if s == "중형차":
            s = input("유종(휘발유)을 입력해주세요. :")
            if s == "휘발유":
                filename = "한국지엠_중형_휘발유.csv"
                filesize = 17
            else:
                print("잘못된 값을 입력하셨습니다.")
                sys.exit()
    else:
        print("잘못된 값을 입력하셨습니다.")
        sys.exit()
        
elif select=="해외모델":
    sys.exit()
else:
    print("잘못된 값을 입력하셨습니다.")
    sys.exit()
    
dv = input("예측하고자 하는 모델의 배기량(cc)을 입력해주세요 : ")
weight = input("예측하고자 하는 모델의 중량(kg)을 입력해주세요 : ")
co = input("예측하고자 하는 모델의 Co2 배출량(g/km)을 입력해주세요 : ")

f = pd.read_csv(filename,encoding='cp949')
x = []
y = []
z = []
w = []

for i in range(filesize):
    x.append(int(f.iloc[i,0])) #배기량
    y.append(int(f.iloc[i,1])) #공차 중량
    z.append(int(f.iloc[i,2])) #CO2 배출량
    w.append(float(f.iloc[i,3])) #복합연비
    
a = sym.Symbol('a')
b = sym.Symbol('b')
c = sym.Symbol('c')
d = sym.Symbol('d')

residual = 0;
for i in range(filesize):
    residual += (x[i]*a + y[i]*b + z[i]*c + d - w[i])**2

residual_a = sym.diff(residual,a)
residual_b = sym.diff(residual,b)
residual_c = sym.diff(residual,c)
residual_d = sym.diff(residual,d)

residual_constant_a = residual_a.subs([(a,0),(b,0),(c,0),(d,0)]) #방정식 A의 상수 = -8.2
residual_constant_b = residual_b.subs([(a,0),(b,0),(c,0),(d,0)]) #방정식 B의 상수 = -53283.6
residual_constant_c = residual_c.subs([(a,0),(b,0),(c,0),(d,0)]) #방정식 C의 상수 = -3591.6
residual_constant_d = residual_d.subs([(a,0),(b,0),(c,0),(d,0)]) #방정식 D의 상수 = -8.2

residual_coefficient_da_a = residual_a.subs([(a,1),(b,0),(c,0),(d,0)]) - residual_constant_a
residual_coefficient_da_b = residual_a.subs([(a,0),(b,1),(c,0),(d,0)]) - residual_constant_a
residual_coefficient_da_c = residual_a.subs([(a,0),(b,0),(c,1),(d,0)]) - residual_constant_a
residual_coefficient_da_d = residual_a.subs([(a,0),(b,0),(c,0),(d,1)]) - residual_constant_a

residual_coefficient_db_a = residual_b.subs([(a,1),(b,0),(c,0),(d,0)]) - residual_constant_b
residual_coefficient_db_b = residual_b.subs([(a,0),(b,1),(c,0),(d,0)]) - residual_constant_b
residual_coefficient_db_c = residual_b.subs([(a,0),(b,0),(c,1),(d,0)]) - residual_constant_b
residual_coefficient_db_d = residual_b.subs([(a,0),(b,0),(c,0),(d,1)]) - residual_constant_b

residual_coefficient_dc_a = residual_a.subs([(a,1),(b,0),(c,0),(d,0)]) - residual_constant_c
residual_coefficient_dc_b = residual_a.subs([(a,0),(b,1),(c,0),(d,0)]) - residual_constant_c
residual_coefficient_dc_c = residual_a.subs([(a,0),(b,0),(c,1),(d,0)]) - residual_constant_c
residual_coefficient_dc_d = residual_a.subs([(a,0),(b,0),(c,0),(d,1)]) - residual_constant_c

residual_coefficient_dd_a = residual_a.subs([(a,1),(b,0),(c,0),(d,0)]) - residual_constant_d
residual_coefficient_dd_b = residual_a.subs([(a,0),(b,1),(c,0),(d,0)]) - residual_constant_d
residual_coefficient_dd_c = residual_a.subs([(a,0),(b,0),(c,1),(d,0)]) - residual_constant_d
residual_coefficient_dd_d = residual_a.subs([(a,0),(b,0),(c,0),(d,1)]) - residual_constant_d

matrix_a =[[int(residual_coefficient_da_a),int(residual_coefficient_da_b),int(residual_coefficient_da_c),int(residual_coefficient_da_d)],[int(residual_coefficient_db_a),int(residual_coefficient_db_b),int(residual_coefficient_db_c),int(residual_coefficient_db_d)],[int(residual_coefficient_dc_a),int(residual_coefficient_dc_b),int(residual_coefficient_dc_c),int(residual_coefficient_dc_d)],[int(residual_coefficient_dd_a),int(residual_coefficient_dd_b),int(residual_coefficient_dd_c),int(residual_coefficient_dd_d)]]
matrix_b = [int(-residual_constant_a),int(-residual_constant_b),int(-residual_constant_c),int(-residual_constant_d)]

print("\n[오차의 총합에 대한 미분방정식을 출력합니다.]")
print(residual_a)
print(residual_b)
print(residual_c)
print(residual_d)

k = [-0.5, -0.5, -0.5, 0.5] 
mu = 0.000000001

print("\n[경사하강법을 이용한 최적화를 100,000회 진행합니다.]")
for i in range(ROUND_CONST):
    sigma_a = 0
    sigma_b = 0
    sigma_c = 0
    sigma_d = 0
    
    for j in range(filesize):
        sigma_a += 2 * x[j] * (k[0]*x[j]+k[1]*y[j]+k[2]*z[j]+k[3]-w[j])
        sigma_b += 2 * y[j] * (k[0]*x[j]+k[1]*y[j]+k[2]*z[j]+k[3]-w[j])
        sigma_c += 2 * z[j] * (k[0]*x[j]+k[1]*y[j]+k[2]*z[j]+k[3]-w[j])
        sigma_d += 2 * w[j] * (k[0]*x[j]+k[1]*y[j]+k[2]*z[j]+k[3]-w[j])
        
    new_a = k[0] - mu * sigma_a
    new_b = k[1] - mu * sigma_b
    new_c = k[2] - mu * sigma_c
    new_d = k[3] - mu * sigma_d
        
    k[0] = new_a
    k[1] = new_b
    k[2] = new_c
    k[3] = new_d

inv_A = np.linalg.inv(matrix_a)
result = np.dot(inv_A,matrix_b)
#linear_model = float(result[0])*a + float(result[1])*b + float(result[2])*c + float(result[3])
linear_model =  float(k[0])*a + float(k[1])*b + float(k[2])*c + float(k[3])

print("인공지능 모델의 a 계수 : ",k[0])
print("인공지능 모델의 b 계수 : ",k[1])
print("인공지능 모델의 c 계수 : ",k[2])
print("인공지능 모델의 d 계수 : ",k[3])

prediction = linear_model.subs([(a,dv),(b,weight),(c,co)])

print("\n[입력하신 데이터를 기반으로 한 예측 결과를 출력합니다.]")
print("[예측한 모델의 평균연비는 ",prediction,"㎞/ℓ입니다.]")


# In[ ]:




