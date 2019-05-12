# 環境構築

## Light GBMのインストール
### cmake
```
sudo apt install cmake
```

このサイト[https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#linux]を参照して、
```
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
mkdir build ; cd build
cmake ..
make -j4
```

もしCXXコンパイラのエラーが出たら、以下を実行後に再度`cmake ..`を実行
```
sudo apt-get update && sudo apt-get install build-essential
```


## FabIOのインストール
サイト[https://github.com/silx-kit/fabio/releases/tag/v0.9.0]からlatestバージョンをダウンロードして展開
fabioのディレクトリに移動して、以下を実行
```
sudo pip install -r ci/requirements_travis.txt --trusted-host www.silx.org
python setup.py build
python setup.py test
sudo pip install .
```