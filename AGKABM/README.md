在当前目录下，运行 `python3 -u ./model/train.py`，会用 `./data` 中的数据训出一个模型存放到 ./pth 中，训练集/验证集比例 7:3

运行 `python3 -u ./model/test.py ${model_name.pth}`，会用 `./test_data` 中的数据来进行测试
