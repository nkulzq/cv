# Quick Start

1. 在py文件同级创建test文件夹

2. 在test文件夹中放入测试数据，数据格式要求与训练数据相同，均为一个文件夹下八张图片，且图片命名方式也必须相同

3. 运行classify.py，将输出success或fail，输出顺序为文件夹名称的字母序

# Issues

1. OOM：最好使用RTX4090及以上显卡，将会占用10-15G显存，程序默认在单卡运行，可以指定你想要使用的显卡

   ```
   os.environ["CUDA_VISIBLE_DEVICES"] = '0'
   ```
 
