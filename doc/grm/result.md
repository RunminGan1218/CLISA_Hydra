# FACED


## Results  
### def_data

#### seg_att
lr=0.0005
seg_att=1
![alt text](image-9.png)
seg_att=2
![alt text](image-20.png)
seg_att=3
![alt text](image-21.png)
seg_att=4
![alt text](image-22.png)

seg_att=5
![alt text](image-11.png)

seg_att=7
![alt text](image-23.png)

seg_att=10
![alt text](image-7.png)

seg_att=15
![alt text](image-2.png)


seg_att=20
![alt text](image-10.png)

seg_att=25
![alt text](image-8.png)

seg_att=30
![alt text](image-13.png)

#### mslen
mslen=1
![alt text](image-12.png)

mslen=2
![alt text](image-16.png)
mslen=3
![alt text](image-3.png)
mslen=4
![alt text](image-18.png)
mslen=5
![alt text](image-14.png)
mslen=6
![alt text](image-19.png)

mslen=7
![alt text](image-15.png)

#### wd
0.00015 0.001
![alt text](image-31.png)
0.00015 0.0022  pretrain 100/30
![alt text](image-32.png)
![alt text](image-73.png)
dif seed = 2469  不同seed之间很稳定
![alt text](image-75.png)
0.00015 0.005  pretrain 100/30
![alt text](image-33.png)
![alt text](image-45.png)

0.00015 0.011
![alt text](image-34.png)
0.00015 0.025
![alt text](image-35.png)
0.00015 0.056
![alt text](image-36.png)
0.00015 0.125
![alt text](image-37.png)

0.0015 0.001
![alt text](image-26.png)
0.0015 0.0022
![alt text](image-25.png)
0.0015 0.005
![alt text](image-24.png)
0.0015 0.011
![alt text](image-27.png)
0.0015 0.025
![alt text](image-28.png)
0.0015 0.056
![alt text](image-29.png)
0.0015 0.125
![alt text](image-30.png)

0.015 0.001
![alt text](image-44.png)
0.015 0.0022
![alt text](image-38.png)
0.015 0.005
![alt text](image-39.png)
0.015 0.011
![alt text](image-40.png)
0.015 0.025
![alt text](image-41.png)
0.015 0.056
![alt text](image-42.png)
0.015 0.125
![alt text](image-43.png)

lr=0.00005
![alt text](image-1.png)

std_model 10 5
![alt text](image-6.png)

#### ablation
no dilation
15 3
![alt text](image-67.png)
![alt text](image-66.png)
1 6
![alt text](image-74.png)

translayer 
std model 15 3 cross channel no position embedding  4.15 MB
![alt text](image-80.png)
std model 10.06 MB
![alt text](image-82.png)
1layer 7.04 MB
![alt text](image-88.png)
1layer time
![alt text](image-94.png)

direct DE
![alt text](image-83.png)

#### best model
1 3
![alt text](image-56.png)

![alt text](image-57.png)

![alt text](image-58.png)

![alt text](image-59.png)

![alt text](image-60.png)

1 6
![alt text](image-51.png)

![alt text](image-52.png)

![alt text](image-53.png)

![alt text](image-54.png)

![alt text](image-55.png)
### def_c2_data

std_model
0.00015 0.001
![alt text](image-50.png)
0.00015 0.0022
![alt text](image-46.png)
0.00015 0.005
![alt text](image-47.png)
0.00015 0.0075
![alt text](image-48.png)
0.00015 0.011
![alt text](image-49.png)

direct DE
![alt text](image-85.png)

translayer
channel:
![alt text](image-86.png)
time:
![alt text](image-87.png)
1layer time
![alt text](image-95.png)

### def_lessICA_data
15 3
![alt text](image-65.png)

![alt text](image-64.png)

![alt text](image-63.png)

![alt text](image-62.png)

![alt text](image-61.png)

1 6
![alt text](image-68.png)
![alt text](image-69.png)
![alt text](image-70.png)
![alt text](image-71.png)
![alt text](image-72.png)

### def_lessICA_c2
15 3
![alt text](image-81.png)


### cus_data
std_model
![alt text](image-3.png)

seg_att=20
![seg_att=20](image.png)

std_model 22 11
![alt text](image-4.png)

### old_data
std_model except seg_att=3 
![alt text](image-5.png)
std_model*2
![alt text](image-93.png)
### old_data_c2
std_model*2
![alt text](image-92.png)




# SEED
## raw data raw code
EPOCH_CHOOSE 1 3 15 [1 3 6 12]
![alt text](image-78.png)
EPOCH_CHOOSE 1 5 24 [1 5 10 19]
![alt text](image-77.png)

EPOCH_CHOOSE 0 3 15 [1 3 6 12]
![alt text](image-79.png)
EPOCH_CHOOSE 0 5 24 [1 5 10 19]
![alt text](image-76.png)

# SEEDV
###
direct DE
![alt text](image-84.png)



# confusion mat
## std model
FACED9
![alt text](image-89.png)
SEEDV 王

SEED 甘

## Direct DE
FACED9
![alt text](image-90.png)
SEEDV
![alt text](image-91.png)
SEED 甘


## no att
FACED9 王


SEEDV 王


SEED 王

