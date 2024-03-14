import urllib.request
import time

# 下载地址列表
download_list = [
    "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz",
    "http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz",
    "http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz",
    "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz",
    "http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz",
    "http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz",
    "http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz",
    "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz",
    "http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz",
    "http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz",
    "http://download.tensorflow.org/models/ens3_adv_inception_v3_2017_08_18.tar.gz",
    "http://download.tensorflow.org/models/ens4_adv_inception_v3_2017_08_18.tar.gz",
    "http://download.tensorflow.org/models/adv_inception_resnet_v2_2017_12_18.tar.gz",
    "http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz"
]

# 遍历下载地址列表
for url in download_list:
    # 提取文件名
    filename = url.split("/")[-1]

    # 下载文件
    print(f"[{time.asctime(time.localtime())}] Downloading", filename)
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"[{time.asctime(time.localtime())}] Download completed:", filename)
    except Exception as e:
        print(f"[{time.asctime(time.localtime())}] Download failed:", filename)
        print(f"[{time.asctime(time.localtime())}] Error:", str(e))