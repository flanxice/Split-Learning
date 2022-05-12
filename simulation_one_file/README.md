## Simulation one file

```shell
1、enter processing.py
set the device numbers
```

```shell
2、python3 processing.py
(run processing.py 
to download datasets and create the split datas and labels)
```

```shell
3、enter Main.py
set learning rate, num_device, num_iterations, step, split_point,
train_batch_size, test_batch_size, IFTestSplit, IFAccumulate 
```

```shell
4、python3 Main.py
(run Main.py 
to get test_loss, test_acc, train_loss, and time..)
```



##### notes

不通过分割截面数据保存。
