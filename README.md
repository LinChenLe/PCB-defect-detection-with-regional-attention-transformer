# PCB-defect-detection-with-regional-attention-transformer
this is a PCB defect detection thesis and code using transformer method and optimize it, do not requires set the default box, only need each pixel position, so this method can be made easily to use for any PCB style.
ours thesis in the (not public now), but unfortunately ours only have Chinese version.
# Thanks
this thesis used data is form DeepPCB: https://github.com/tangsanli5201/DeepPCB and TDD-Net https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB (Augmentation dataset), ours very thanks for both thesis public dataset, make we to use.

# 資料集
1. DeepPCB資料集請至原論文github https://github.com/tangsanli5201/DeepPCB 下載。
2. TDD-Net資料集請至github https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB 下載「資料增量後的數據集(Augmentation Dataset)。

# RAT在DeepPCB資料集
## DeepPCB資料前處理
1. 確認「Answer/DeepPCB/」資料夾中是否含有資料夾「gt」、「submit」、「change.py」、「meanAveragePrecision.py」。
2. 將「make_mask.py」放入「Regional-Attention-Transformer」，並將程式內的「dir_path」指定為「"./DeepPCB-master/PCBData/"」以及「file_path」指定為「dir_path+"trainval.txt"」，執行完畢後將「file_path」指定為「dir_path+"test.txt"」(此程式會建立每個瑕疵的位置遮罩(mask)以及類別，以方便後續使用)。
3. 將「DeepPCB-master/evaluation/gt.zip」複製到「Regional-Attention-Transformer/Answer/DeepPCB/」中，並解壓縮，再將資料夾所有檔案在終端機使用「chmod 777 ./gt/ -R」，修改權限，再執行「change.py」檔案，修改所有答案的點符號，使其能夠對應於本論文之程式。
## 主幹架構在DeepPCB資料集預訓練
1. 至「/Program/Train&Val_DeepPCB/Backbone_for_DeepPCB/」資料夾中將所有「.py」檔案複製至「Regional-Attention-Transformer」資料夾中，確認「config.py」中的「data_foloder」為「"./DeepPCB-master/PCBData/"」。
2. 根據需求可更改其餘「.py」檔中的參數。
3. 執行「main.py」。
4.★若需要從儲存點開始訓練，請將儲存點的權重檔案放置「/ckpt/」中，並在「config.py」中設定參數「load_back_bone_ckpt」為「"ckpt"+"權重檔名稱"」，得以從儲存點開始訓練。★
## RAT整體訓練在DeepPCB資料集
1. 至「/Program/Train&Val_DeepPCB/RAT_for_DeepPCB/」資料夾中將所有「.py」檔案複製至「Regional-Attention-Transformer」資料夾中，確認「config.py」中的「data_foloder」為「"./DeepPCB-master/PCBData/"」。
	2. 確認在「/ckpt/」中含有在DeepPCB中預訓練過的權重檔案，並確認「config.py」中的「load_backbone_ckpt」參數為「"ckpt"+"預訓練權重檔名"」。
	3.根據需求可更改其餘「.py」檔中的參數
	4. 執行「main.py」。
	5. 訓練過程中所儲存的權重紀錄點會在「/ckpt/」中。
	★若需要從儲存點開始訓練，請將儲存點的權重檔案放置「/ckpt/」中，並在「config.py」中設定參數「load_ckpt」為「"ckpt"+"權重檔名稱"」，得以從儲存點開始訓練。★
  ## RAT消融實驗在DeepPCB資料集
  1. 至「/Program/Train&Val_DeepPCB/選擇要去除Anchor還是RegionalAttention/」資料夾中將所有「.py」檔案複製至「Regional_Attention_Transformer」資料夾中，確認「config.py」中的「data_foloder」為「"./DeepPCB-master/PCBData/"」。
	2.與整體訓練步驟2、3、4、5一樣。
  # RAT在TDD-Net資料集
  ## TDD-Net資料前處理
  1. 將下載好的資料透過TDD-Net的github「https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB 」「Train」中的步驟「1、以及2、」建立拆分後的訓練集以及驗證集。
	2. 將拆分好的資料集放入「/TDD-Net_dataset/original_data/」並執行「convert2txt.py」檔案，將二進制檔轉為文字檔。
  ## 主幹架構在TDD-Net資料集預訓練
  1. 將「/Program/Train&Val_TDD-Net/Backbone_for_TDD-Net/」中的.py檔案複製到「Regional_Attention_Transformer」資料夾中，，確認「config.py」中的「data_foloder」為「"./TDD_dataset/convert_data/"」。
  2. 根據需求可更改其餘「.py」檔中的參數。
	3. 執行「main.py」。
	4. 訓練過程中所儲存的權重紀錄點會在「/ckpt/」中。
	★若需要從儲存點開始訓練，請將儲存點的權重檔案放置「/ckpt/」中，並在「config.py」中設定參數「load_back_bone_ckpt」為「"ckpt"+"權重檔名稱"」，得以從儲存點開始訓練。★
   ## RAT整體訓練在TDD-Net資料集
   1. 至「/Program/Train&Val_TDD-Net/RAT_for_TDD-Net/」資料夾中將所有「.py」檔案複製至「Regional-Attention-Transformer」資料夾中，確認「config.py」中的「data_foloder」為「"./TDD_dataset/convert_data/"」。
	2. 確認在「/ckpt/」中含有在TDD-Net中預訓練過的權重檔案，並確認「config.py」中的「load_backbone_ckpt」參數為「"ckpt"+"預訓練權重檔名"」。
	3.根據需求可更改其餘「.py」檔中的參數
	4. 執行「main.py」。
	5. 訓練過程中所儲存的權重紀錄點會在「/ckpt/」中。
	★若需要從儲存點開始訓練，請將儲存點的權重檔案放置「/ckpt/」中，並在「config.py」中設定參數「load_ckpt」為「"ckpt"+"權重檔名稱"」，得以從儲存點開始訓練。★
  # RAT消融實驗在TDD-Net資料集
  1. 至「/Program/Train&Val_TDD-Net/選擇要去除Anchor還是RegionalAttention/」資料夾中將所有「.py」檔案複製至「Regional_Attention_Transformer」資料夾中，確認「config.py」中的「data_foloder」為「"./TDD_dataset/convert_data/
	2.與整體訓練步驟2、3、4、5一樣。

