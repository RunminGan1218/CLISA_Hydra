# FACED-ablation

## Results

#### data: FACED_new

global_att=true

![ablation-r1](ablation-r1.png)



activ = 'relu'

![ablation-r2](ablation-r2.png)




has_att = False, ext_fea.mode='de'

![ablation-r3](ablation-r3.png)



has_att = False, ext_fea.mode='me'

![ablation-r4](ablation-r4.png)



# FACED-variations of convolution design

| 类型         | 类别 | 值         |
| ------------ | ---- | ---------- |
| 一维空域卷积 | 9    | 52.25/8.00 |
| 无空洞设计   | 9    | 53.58/8.95 |
| 无时域卷积   | 9    | 52.20/8.11 |
| 二维卷积     | 9    | 55.72/8.56 |
| 一维空域卷积 | 2    | 72.98/4.17 |
| 无空洞设计   | 2    | 71.91/5.33 |
| 无时域卷积   | 2    | 73.99/4.65 |
| 二维卷积     | 2    | 72.24/4.02 |



一维空域卷积，9分类；*mslen=1

![ablation-r5](ablation-r5.png)



无空洞设计，9分类；*dilation=[1,1,1,1]

![ablation-r16](ablation-r16.png)



无时域卷积，9分类

![ablation-r7](ablation-r7.png)



二维卷积，9分类

![ablation-r11](ablation-r11.png)



一维空域卷积，2分类

![ablation-r19](ablation-r19.png)



无空洞设计，2分类

![ablation-r20](ablation-r20.png)



无时域卷积，2分类

![ablation-r21](ablation-r21.png)



二维卷积，2分类

![ablation-r22](ablation-r22.png)
