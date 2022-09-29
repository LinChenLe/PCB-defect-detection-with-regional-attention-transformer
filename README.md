# PCB-defect-detection-with-regional-attention-transformer
this is a PCB defect detection thesis and code using transformer method and optimize it, do not requires set the default box, only need each pixel position, so this method can be made easily to use for any PCB style.
ours thesis in the (not public now), but unfortunately ours only have Chinese version.
# Thanks
this thesis used data is form DeepPCB: https://github.com/tangsanli5201/DeepPCB and TDD-Net https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB (Augmentation dataset), ours very thanks for both thesis public dataset, make we to use.

# 資料集
	1. DeepPCB資料集請至原論文github「https://github.com/tangsanli5201/DeepPCB」下載。
	2. TDD-Net資料集請至github「https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB」下載「資料增量後的數據集(Augmentation Dataset)」。

# RAT在DeepPCB資料集
	1. 將「make_mask.py」放入「Regional_Attention_Transformer」，並將程式內的「dir_path」指定為「"./DeepPCB-master/PCBData/"」以及「file_path」指定為「dir_path+"trainval.txt"」，執行完畢後將「file_path」指定為「dir_path+"test.txt"」(此程式會建立每個瑕疵的位置遮罩(mask)以及類別，以方便後續使用)。
##### 主幹架構在DeepPCB資料集預訓練
