## Simulation multi files

```shell
1、enter CONTROL.py
set ramdom seed, IFConvergence, IFTestSplit ,and gpu distribution
```

```shell
2、enter Server_Center.py
set learning_rate, num_device, num_iterations, step, split_points,
train_batch_size, test_batch_size..
```

```shell
3、python3 Server_Center.py
run Server_Center.py and will dowload loss, acc, time and so on.
```



##### notes:

通过保存中间传输数据来模拟数据传输。服务器端也是多线程。