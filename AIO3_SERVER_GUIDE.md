# DACG-IR 的 AIO3-v1 服务器运行方法

下面命令不需要设置环境变量。只需根据服务器修改绝对路径。

假设：

```text
代码：/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/all-in-one-model/DACG-IR-code
数据：/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/AIO3
输出：/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1
manifest：/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1/manifests
协议文档：/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/AIO3_TRAINING_EVALUATION_PROTOCOL.md
```

## 1. 验证数据

```bash
cd /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/all-in-one-model/DACG-IR-code

python -m aio3_runner.verify_data \
  --manifest-dir /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1/manifests
```

## 2. 100-step smoke

```bash
python -m aio3_runner.train \
  --manifest-dir /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1/manifests \
  --data-root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/AIO3 \
  --output-root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1 \
  --protocol-document /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/AIO3_TRAINING_EVALUATION_PROTOCOL.md \
  --run-kind smoke \
  --wandb-entity c14150591-sjtu
```

如果显存不足，只增加：

```text
--micro-batch-size 4
```

有效 batch 仍为12，不改变协议。

## 3. 暂停/恢复 smoke 验收

首次运行在第50步暂停：

```bash
python -m aio3_runner.train \
  --manifest-dir /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1/manifests \
  --data-root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/AIO3 \
  --output-root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1 \
  --protocol-document /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/AIO3_TRAINING_EVALUATION_PROTOCOL.md \
  --run-kind smoke \
  --pause-at-step 50 \
  --wandb-entity c14150591-sjtu
```

记录终端输出或输出目录中的 run 路径，然后恢复：

```bash
python -m aio3_runner.train \
  --resume /绝对路径/到该run/checkpoints/latest.pth \
  --data-root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/AIO3
```

## 4. 5000-step pilot

```bash
python -m aio3_runner.train \
  --manifest-dir /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1/manifests \
  --data-root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/AIO3 \
  --output-root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1 \
  --protocol-document /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/AIO3_TRAINING_EVALUATION_PROTOCOL.md \
  --run-kind pilot \
  --wandb-entity c14150591-sjtu
```

## 5. 200000-step formal

正式训练前必须提交代码，保证 Git 工作区干净。formal 必须从随机初始化开始，不能从
pilot 权重继续：

```bash
python -m aio3_runner.train \
  --manifest-dir /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1/manifests \
  --data-root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/AIO3 \
  --output-root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/outputs/AIO3/aio3-v1 \
  --protocol-document /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/AIO3_TRAINING_EVALUATION_PROTOCOL.md \
  --run-kind formal \
  --seed 3407 \
  --wandb-entity c14150591-sjtu
```

最终权重：

```text
<RUN_DIR>/checkpoints/best_macro_psnr.pth
```

断点恢复权重：

```text
<RUN_DIR>/checkpoints/latest.pth
```

## 6. 正式测试

```bash
python -m aio3_runner.evaluate \
  --checkpoint /绝对路径/到formal-run/checkpoints/best_macro_psnr.pth \
  --num-workers 4
```

数据根目录、manifest、W&B和输出目录会从 formal run 的冻结配置中读取。结果写入：

```text
<RUN_DIR>/test/
```
